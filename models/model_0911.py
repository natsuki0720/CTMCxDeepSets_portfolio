
import os
from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class varSets_Datasets(Dataset):
    def __init__(self, states, del_t, outputs, transform=None):
        self.states = states
        self.del_t = del_t
        self.outputs = outputs
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        delta_t = self.del_t[idx]
        target = self.outputs[idx]

        if self.transform:
            state, delta_t = self.transform(state, delta_t)

        state = torch.as_tensor(state, dtype=torch.long)
        delta_t = torch.as_tensor(delta_t, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        length = torch.tensor(state.shape[1], dtype=torch.long)
        return state, delta_t, target, length

def collate_fn(batch):
    state_batch = [item[0] for item in batch]
    delta_t_batch = [item[1] for item in batch]
    target_batch = torch.stack([item[2] for item in batch])
    lengths = torch.tensor([s.shape[1] for s in state_batch], dtype=torch.long)
    max_length = int(lengths.max().item()) if lengths.numel() > 0 else 0

    state_padded = []
    for s in state_batch:
        L = s.shape[1] if s.dim() == 2 else 0
        if s.dim() != 2:
            s = torch.zeros((2, 0), dtype=torch.long)
            L = 0
        pad_size = max(0, max_length - L)
        if pad_size > 0:
            s = F.pad(s, (0, pad_size), mode="constant", value=0)
        state_padded.append(s)
    state_padded = torch.stack(state_padded, dim=0)

    delta_t_padded = []
    for dt in delta_t_batch:
        L = dt.shape[0] if dt.dim() == 1 else 0
        if dt.dim() != 1:
            dt = torch.zeros((0,), dtype=torch.float32)
            L = 0
        pad_size = max(0, max_length - L)
        if pad_size > 0:
            dt = F.pad(dt, (0, pad_size), mode="constant", value=0.0)
        delta_t_padded.append(dt)
    delta_t_padded = torch.stack(delta_t_padded, dim=0)

    return state_padded, delta_t_padded, target_batch, lengths

class DeepSets_varSets_forDiagnel(nn.Module):
    def __init__(
        self,
        num_categories: int = 4,
        embedding_dim: int = 16,
        token_hidden1: int = 256,
        token_hidden2: int = 256,
        output_hidden1: int = 128,
        output_hidden2: int = 64,
        dropout: float = 0.2,
        input_is_one_based: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device if device is not None else set_device()
        self.input_is_one_based = input_is_one_based

        self.embedding = nn.Embedding(
            num_embeddings=num_categories + (1 if input_is_one_based else 1),
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        in_feat = embedding_dim * 2 + 1

        self.fc1 = nn.Linear(in_feat, token_hidden1)
        self.ln1 = nn.LayerNorm(token_hidden1)
        self.fc2 = nn.Linear(token_hidden1, token_hidden2)
        self.ln2 = nn.LayerNorm(token_hidden2)
        self.drop = nn.Dropout(dropout)

        self.att_fc = nn.Linear(token_hidden2, token_hidden2)
        self.att_score = nn.Linear(token_hidden2, 1)

        self.out_fc1 = nn.Linear(token_hidden2, output_hidden1)
        self.out_ln1 = nn.LayerNorm(output_hidden1)
        self.out_fc2 = nn.Linear(output_hidden1, output_hidden2)
        self.out_ln2 = nn.LayerNorm(output_hidden2)
        self.out_fc3 = nn.Linear(output_hidden2, 3)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def _prepare_indices(self, idx: torch.Tensor) -> torch.Tensor:
        if self.input_is_one_based:
            idx = torch.where(idx > 0, idx, torch.zeros_like(idx))
            return idx
        else:
            return torch.where(idx >= 0, idx + 1, torch.zeros_like(idx))

    def forward(self, state: torch.Tensor, delta_t: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        device = state.device
        B, _, L = state.shape

        pre = state[:, 0, :]
        post = state[:, 1, :]

        pre_idx = self._prepare_indices(pre)
        post_idx = self._prepare_indices(post)

        pre_emb = self.embedding(pre_idx)
        post_emb = self.embedding(post_idx)
        dt = delta_t.unsqueeze(-1)

        x = torch.cat([pre_emb, post_emb, dt], dim=-1)

        x = self.drop(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop(F.gelu(self.ln2(self.fc2(x))))

        arange_L = torch.arange(L, device=device).unsqueeze(0)
        key_padding_mask = arange_L >= lengths.unsqueeze(1)

        attn_input = torch.tanh(self.att_fc(x))
        attn_score = self.att_score(attn_input).squeeze(-1)
        attn_score = attn_score.masked_fill(key_padding_mask, float('-inf'))
        attn_weights = F.softmax(attn_score, dim=1)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        h = self.drop(F.gelu(self.out_ln1(self.out_fc1(pooled))))
        h = self.drop(F.gelu(self.out_ln2(self.out_fc2(h))))
        out = F.softplus(self.out_fc3(h))
        return out
