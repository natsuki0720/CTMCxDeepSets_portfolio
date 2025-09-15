import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class varSets_Datasets(Dataset):
    def __init__(self, states, del_t, outputs, transform = None):
        self.states = states # リスト: 各要素は (2, num_samples_i)
        self.del_t = del_t # リスト: 各要素は (num_samples_i,)
        self.outputs = outputs # リスト: 各要素は (3,)
        self.transform = transform
        self.num_sumples = len(states)
    
    def __len__(self):
        return  self.num_sumples
    
    def __getitem__(self, idx):
        state = self.states[idx]
        delta_t = self.del_t[idx]
        target = self.outputs[idx]

        if self.transform:
            state, delta_t = self.transform(state, delta_t)
        
        state = torch.tensor(state, dtype=torch.long) # (2, num_samples_i)
        delta_t = torch.tensor(delta_t, dtype=torch.float32) # (num_samples_i,)
        target = torch.tensor(target, dtype=torch.float32) # (3,)

        length = torch.tensor(state.shape[1])  # ここ追加 (num_samples_i)
        
        return state, delta_t, target, length

# あとでやる
def collate_fn(batch):
    state_batch = [item[0] for item in batch]     # 各 state は (2, num_samples_i)
    delta_t_batch = [item[1] for item in batch]   # 各 delta_t は (num_samples_i,)
    target_batch = torch.stack([item[2] for item in batch])  # (batch_size, 3)
    
    # 各サンプルの長さを取得
    lengths = lengths = torch.tensor([s.shape[1] if len(s.shape) > 1 else 0 for s in state_batch]) # (batch_size,)
    
    # 最大長を取得
    max_length = lengths.max() if len(lengths) > 0 else 0
    
    # state のパディング
    state_padded = []
    for s in state_batch:
        if len(s.shape) < 2:  # shape[1]がない場合
            s = s.unsqueeze(0)  # 必要なら2次元に変換
        
        pad_size = max_length - s.shape[1]
        
        # パディング
        s_padded = torch.nn.functional.pad(s, (0, pad_size), mode='constant', value=0)
        state_padded.append(s_padded)
    state_padded = torch.stack(state_padded)  # (batch_size, 2, max_length)
    
    # delta_t のパディング
    delta_t_padded = []
    for dt in delta_t_batch:
        pad_size = max_length - dt.shape[0]
        dt_padded = torch.nn.functional.pad(dt, (0, pad_size), mode='constant', value=0)
        delta_t_padded.append(dt_padded)
    delta_t_padded = torch.stack(delta_t_padded)  # (batch_size, max_length)
    
    return state_padded, delta_t_padded, target_batch, lengths

class DeepSets_varSets_forDiagnel(nn.Module):
    def __init__(self, num_categories=4, embedding_dim=2, device=None):
        super(DeepSets_varSets_forDiagnel, self).__init__()
        self.device = device if device else set_device()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        # 特徴抽出層（遷移ごと）
        self.fc1 = nn.Linear(embedding_dim * 2 + 1, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=0.3)

        # Attention pooling 層（masked mean pooling の代替）
        self.att_fc = nn.Linear(64, 64)
        self.att_score = nn.Linear(64, 1)

        # 出力層
        self.fc3 = nn.Linear(64, 64)
        self.dropout3 = nn.Dropout(p=0.3)

        self.fc4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(p=0.1)

        self.fc5 = nn.Linear(32, 3)

    def forward(self, state, delta_t, lengths):
        batch_size = state.size(0)
        max_length = state.size(2)

        pre = state[:, 0, :]
        post = state[:, 1, :]

        pre_embedded = self.embedding(pre.long() - 1)
        post_embedded = self.embedding(post.long() - 1)

        delta_t = delta_t.unsqueeze(2)

        x = torch.cat([pre_embedded, post_embedded, delta_t], dim=2)  # (B, L, 2D+1)
        x = x.view(-1, x.size(2))  # (B×L, F)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))  # (B×L, 64)

        x = x.view(batch_size, max_length, -1)  # (B, L, 64)

        # マスク作成
        mask = torch.arange(max_length, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2).float()  # (B, L, 1)

        # Attention pooling
        attn_input = torch.tanh(self.att_fc(x))               # (B, L, 64)
        attn_score = self.att_score(attn_input).squeeze(2)    # (B, L)
        attn_score = attn_score.masked_fill(mask.squeeze(2) == 0, float('-inf'))  # (B, L)
        attn_weights = F.softmax(attn_score, dim=1)           # (B, L)
        x_pooled = torch.sum(x * attn_weights.unsqueeze(2), dim=1)  # (B, 64)

        # 出力層
        x = self.dropout3(F.relu(self.fc3(x_pooled)))
        x = self.dropout4(F.relu(self.fc4(x)))
        output = F.softplus(self.fc5(x))  # (B, 3)

        return output
def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
        