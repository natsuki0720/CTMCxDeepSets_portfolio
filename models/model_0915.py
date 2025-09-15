
import os
from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# -----------------------------
# Device helper (unchanged)
# -----------------------------
def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# -----------------------------
# Dataset class (unchanged API)
# -----------------------------
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

# -----------------------------
# Collate function (unchanged name & signature)
# -----------------------------
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

# -----------------------------
# PMA pooling (Permutation-invariant)
# -----------------------------
class _PMA(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, k: int = 1):
        super().__init__()
        self.k = k
        self.seed = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # x: [B, L, d_model]
        B = x.size(0)
        Q = self.seed.unsqueeze(0).expand(B, self.k, -1)  # [B, k, d]
        out, _ = self.mha(Q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.ln(out)
        return out[:, 0, :] if self.k == 1 else out.mean(dim=1)

class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, p: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(F.gelu(self.fc1(x)))
        h = self.drop(self.fc2(h))
        return self.ln(x + h)

# -----------------------------
# Upgraded model (same class name & __init__ signature)
# -----------------------------
class DeepSets_varSets_forDiagnel(nn.Module):
    """
    Drop-in compatible but stronger model:
    - Strict permutation invariance (no positional encodings)
    - Transformer encoder over elements (permutation-equivariant)
    - PMA pooling (k=1) to get a set-level vector
    - Positive 3-d output via softplus
    Public API is compatible with the original file.
    """
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
        # --- new optional knobs with safe defaults ---
        d_model: Optional[int] = None,     # if None -> max(128, embedding_dim*4)
        nhead: Optional[int] = None,       # if None -> 4 (or 8 if d_model>=192)
        num_layers: int = 3,               # default moderate depth
        dim_feedforward: Optional[int] = None,  # if None -> 4*d_model
    ):
        super().__init__()
        self.device = device if device is not None else set_device()
        self.input_is_one_based = input_is_one_based

        # --- Embedding for discrete states (0 = padding reserved) ---
        # The original design used a single embedding table for categories.
        # We keep it but use it twice (pre/post) and concatenate, just like before.
        self.embedding = nn.Embedding(
            num_embeddings=(num_categories + 1),  # 0..num_categories, 0 is PAD
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # Token feature before encoder: concat(pre_emb, post_emb, dt)
        token_in_dim = embedding_dim * 2 + 1

        # Project token features up to d_model for the Transformer
        if d_model is None:
            d_model = max(128, embedding_dim * 4)
        self.proj = nn.Sequential(
            nn.Linear(token_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        if nhead is None:
            nhead = 8 if d_model >= 192 else 4
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Transformer encoder (no positional encodings -> permutation-equivariant)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # PMA pooling to get set-level vector (permutation-invariant)
        self.pma = _PMA(d_model=d_model, nhead=nhead, dropout=dropout, k=1)

        # Output head (use original hidden sizes for compatibility spirit)
        self.pre_head = _ResidualMLP(d_model, hidden=max(output_hidden1, d_model), p=dropout)
        self.out_fc1 = nn.Linear(d_model, output_hidden1)
        self.out_ln1 = nn.LayerNorm(output_hidden1)
        self.out_fc2 = nn.Linear(output_hidden1, output_hidden2)
        self.out_ln2 = nn.LayerNorm(output_hidden2)
        self.out_fc3 = nn.Linear(output_hidden2, 3)

        # Init similar to original
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
        # Match original behavior:
        # one-based indices: positive stay, non-positive -> 0 (PAD)
        # zero-based or -1-for-pad: shift by +1 for non-negative, else 0
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

        pre_emb = self.embedding(pre_idx)     # [B, L, E]
        post_emb = self.embedding(post_idx)   # [B, L, E]
        dt = delta_t.unsqueeze(-1)            # [B, L, 1]

        # token features -> projection to d_model
        x = torch.cat([pre_emb, post_emb, dt], dim=-1)  # [B, L, 2E+1]
        x = self.proj(x)                                 # [B, L, d_model]

        # padding mask from lengths
        arange_L = torch.arange(L, device=device).unsqueeze(0)
        key_padding_mask = arange_L >= lengths.unsqueeze(1)  # [B, L], True=PAD

        # encode then pool (strictly permutation-invariant overall)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, d_model]
        pooled = self.pma(h, key_padding_mask=key_padding_mask)     # [B, d_model]

        # output head (retain original spirit & softplus)
        z = self.pre_head(pooled)
        h = F.gelu(self.out_ln1(self.out_fc1(z)))
        h = F.gelu(self.out_ln2(self.out_fc2(h)))
        out = F.softplus(self.out_fc3(h))
        return out
