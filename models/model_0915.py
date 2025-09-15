
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from torch import nn

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
            s = F.pad(s, (0, pad), value=0)  # pad on the right
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

class PMA(nn.Module):
    """
    Pooling by Multihead Attention (k=1). Permutation-invariant pooling.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, k: int = 1):
        super().__init__()
        self.k = k
        self.seed = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        B = x.size(0)
        Q = self.seed.unsqueeze(0).expand(B, self.k, -1)  # [B, k, d]
        out, _ = self.mha(Q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.ln(out)
        return out[:, 0, :] if self.k == 1 else out.mean(dim=1)

class SetRegressorInvariant(nn.Module):
    """
    Permutation-invariant set regressor with a configurable Transformer encoder and PMA pooling.
    Inputs:
        state: Long [B, 2, L]  (s1, s2) with 0 as padding
        delta_t: Float [B, L]
        lengths: Long [B]
    Output:
        Tuple(pred, None) where pred is Float [B, 3] with softplus activation (positive)
    """
    def __init__(self,
                 n_state1: int = 128,
                 n_state2: int = 128,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 512,
                 dt_mlp_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        self.emb_s1 = nn.Embedding(n_state1, d_model // 2, padding_idx=0)
        self.emb_s2 = nn.Embedding(n_state2, d_model // 2, padding_idx=0)

        self.dt_mlp = nn.Sequential(
            nn.Linear(1, dt_mlp_hidden),
            nn.GELU(),
            nn.Linear(dt_mlp_hidden, d_model),
        )

        self.fuse = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fuse_ln = nn.LayerNorm(d_model)

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

        self.pma = PMA(d_model=d_model, nhead=nhead, dropout=dropout, k=1)

        self.pre_head = ResidualMLP(d_model, hidden=max(128, d_model), p=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(256, d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(256, d_model), max(128, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(128, d_model // 2), 3),
        )

    def forward(self, state: torch.Tensor, delta_t: torch.Tensor, lengths: torch.Tensor):
        device = state.device
        B, two, L = state.shape
        assert two == 2, f"expected state with shape [B, 2, L], got {state.shape}"

        s1 = state[:, 0, :]
        s2 = state[:, 1, :]

        e1 = self.emb_s1(s1.clamp_min_(0))
        e2 = self.emb_s2(s2.clamp_min_(0))
        state_emb = torch.cat([e1, e2], dim=-1)

        dt_emb = self.dt_mlp(delta_t.unsqueeze(-1))

        x = torch.cat([state_emb, dt_emb], dim=-1)
        x = self.fuse_ln(self.fuse(x))

        arange_L = torch.arange(L, device=device).unsqueeze(0)
        mask_items = arange_L >= lengths.unsqueeze(1)  # [B, L]

        h = self.encoder(x, src_key_padding_mask=mask_items)
        pooled = self.pma(h, key_padding_mask=mask_items)

        z = self.pre_head(pooled)
        out = self.head(z)
        out = F.softplus(out)
        return out, None

SIZE_PRESETS: Dict[str, Dict] = {
    # ~5-6M params (depending on vocab sizes)
    "tiny": dict(d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dt_mlp_hidden=64),
    # ~9-12M params
    "small": dict(d_model=160, nhead=4, num_layers=4, dim_feedforward=640, dt_mlp_hidden=80),
    # ~14-18M params
    "base": dict(d_model=192, nhead=6, num_layers=4, dim_feedforward=768, dt_mlp_hidden=96),
    # ~22-30M params
    "large": dict(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dt_mlp_hidden=128),
}

def build_set_model(size: str = "small", n_state1: int = 128, n_state2: int = 128) -> SetRegressorInvariant:
    if size not in SIZE_PRESETS:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(SIZE_PRESETS.keys())}")
    cfg = SIZE_PRESETS[size]
    return SetRegressorInvariant(n_state1=n_state1, n_state2=n_state2, **cfg)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
