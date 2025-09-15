
import math
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import nn

def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# -----------------------------
# Collate (pad to max length)
# -----------------------------
def collate_set_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    batch: list of tuples (state, delta_t, lengths_optional)
        state: LongTensor [2, L]  (two integer features per element: s1, s2)
        delta_t: FloatTensor [L]
        lengths_optional: LongTensor [1] or None

    Returns:
        state_padded: LongTensor [B, 2, Lmax]
        delta_t_padded: FloatTensor [B, Lmax]
        lengths: LongTensor [B]
    """
    B = len(batch)
    lengths = []
    states = []
    deltas = []
    for (s, dt, l) in batch:
        if s is None or s.dim() != 2 or s.size(0) != 2:
            s = torch.zeros( (2, 0), dtype=torch.long )
        if dt is None or dt.dim() != 1:
            dt = torch.zeros( (0,), dtype=torch.float32 )
        L = s.size(1)
        if l is None:
            lengths.append(L)
        else:
            lengths.append(int(l.item()) if l.numel()==1 else int(l[0].item()))
        states.append(s)
        deltas.append(dt)

    Lmax = max([int(x) for x in lengths]) if lengths else 0

    state_pad = []
    for s in states:
        pad = Lmax - s.size(1)
        if pad > 0:
            s = F.pad(s, (0, pad), value=0)  # pad on the right (width dimension)
        state_pad.append(s)
    state_padded = torch.stack(state_pad, dim=0) if B>0 else torch.zeros((0,2,0),dtype=torch.long)

    delta_pad = []
    for dt in deltas:
        pad = Lmax - dt.size(0)
        if pad > 0:
            dt = F.pad(dt, (0, pad), value=0.0)
        delta_pad.append(dt)
    delta_t_padded = torch.stack(delta_pad, dim=0) if B>0 else torch.zeros((0,0),dtype=torch.float32)

    lengths = torch.tensor(lengths, dtype=torch.long)
    return state_padded, delta_t_padded, lengths

# -----------------------------
# Positional (index) encoding for sets (order-agnostic optional)
# We'll add a small sinusoidal encoding by element index; doesn't enforce order.
# -----------------------------
def sinusoidal_position_encoding(L: int, d: int, device=None) -> torch.Tensor:
    pe = torch.zeros(L, d, device=device)
    position = torch.arange(0, L, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class ResidualMLP(nn.Module):
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

class SetRegressorLarge(nn.Module):
    """
    Bigger model for set-to-parameters regression.
    Inputs:
        state: Long [B, 2, L]  (s1, s2)
        delta_t: Float [B, L]
        lengths: Long [B]
    Output:
        Tuple(pred, None) where pred is Float [B, 3] with softplus activation (positive)
    """
    def __init__(self,
                 n_state1: int = 128,
                 n_state2: int = 128,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dt_mlp_hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        # Embeddings for two integer states
        self.emb_s1 = nn.Embedding(n_state1, d_model // 2, padding_idx=0)
        self.emb_s2 = nn.Embedding(n_state2, d_model // 2, padding_idx=0)

        # Project scalar delta_t to d_model
        self.dt_mlp = nn.Sequential(
            nn.Linear(1, dt_mlp_hidden),
            nn.GELU(),
            nn.Linear(dt_mlp_hidden, d_model),
        )

        # fuse [state_emb_concat, dt_emb] -> d_model via gated MLP
        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fuse_ln = nn.LayerNorm(d_model)

        # Learnable [CLS] token
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder (bigger)
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

        # Output head (deeper, residual)
        self.pre_head = ResidualMLP(d_model, hidden=512, p=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, state: torch.Tensor, delta_t: torch.Tensor, lengths: torch.Tensor):
        """
        state: [B, 2, L] (long)
        delta_t: [B, L] (float)
        lengths: [B] (long)
        """
        device = state.device

        B, two, L = state.shape
        assert two == 2, f"expected state with shape [B, 2, L], got {state.shape}"

        s1 = state[:, 0, :]  # [B, L]
        s2 = state[:, 1, :]  # [B, L]

        # Embedding lookup (padding_idx=0 must be reserved)
        e1 = self.emb_s1(s1.clamp_min_(0))       # [B, L, d/2]
        e2 = self.emb_s2(s2.clamp_min_(0))       # [B, L, d/2]
        state_emb = torch.cat([e1, e2], dim=-1)  # [B, L, d_model]

        dt_emb = self.dt_mlp(delta_t.unsqueeze(-1))  # [B, L, d_model]

        x = torch.cat([state_emb, dt_emb], dim=-1)   # [B, L, 2*d_model]
        x = self.fuse_ln(self.fuse(x))               # [B, L, d_model]

        # Add a learnable CLS token for pooling
        cls_tok = self.cls.expand(B, -1, -1)         # [B, 1, d_model]
        x = torch.cat([cls_tok, x], dim=1)           # [B, 1+L, d_model]

        # Build key padding mask (True for PAD positions)
        # We have a CLS at position 0 (never masked), elements 1..L correspond to set items.
        arange_L = torch.arange(L, device=device).unsqueeze(0)          # [1, L]
        mask_items = arange_L >= lengths.unsqueeze(1)                   # [B, L]
        key_padding_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), mask_items], dim=1)  # [B, 1+L]

        # Optional: add sinusoidal encoding to elements (skip CLS)
        pe = sinusoidal_position_encoding(L, x.size(-1), device=device) if L>0 else torch.zeros((0, x.size(-1)), device=device)
        if L > 0:
            x[:, 1:, :] = x[:, 1:, :] + pe.unsqueeze(0)

        # Encode
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)      # [B, 1+L, d_model]

        # Take CLS representation
        cls_h = h[:, 0, :]                                              # [B, d_model]

        # Head
        z = self.pre_head(cls_h)
        out = self.head(z)
        out = F.softplus(out)  # positive parameters

        # Return a tuple so that existing code using model(...)[0] won't break
        return out, None

# Small factory for convenience
def build_large_model(n_state1: int = 128, n_state2: int = 128) -> SetRegressorLarge:
    return SetRegressorLarge(n_state1=n_state1, n_state2=n_state2)
