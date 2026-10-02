import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
# Standard embedding components (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding,
                                   padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding    = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout            = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout         = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# PatchMLP embedding (fixed for CTG / long sequences)
# ─────────────────────────────────────────────────────────────────────────────

class EmbLayer(nn.Module):
    """
    Single-scale patch embedding for one input channel group.

    Splits the sequence into overlapping patches of length `patch_len`,
    projects each patch independently via a Linear layer, then mean-pools
    over all patches → one d_model vector per input channel.

    Output shape: (B, V, d_model)  where V = number of input channels.

    Previous bug: used `self.d_model = d_model // patch_num`, which rounds to 0
    for long sequences (e.g. seq_len=7200 → patch_num ≈ 300 → 0).
    Fix: project each patch directly to d_model, then mean-pool — no division needed.
    """
    def __init__(self, patch_len: int, patch_step: int, d_model: int):
        super().__init__()
        self.patch_len  = patch_len
        self.patch_step = patch_step
        # Each patch (length patch_len) → d_model features
        self.ff = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T, C) or (B, C, T); normalise to (B, C, T)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.shape}")
        if x.shape[-1] < x.shape[-2]:          # (B, T, C) → (B, C, T)
            x = x.permute(0, 2, 1)
        B, V, L = x.shape

        # Patchify: (B, V, num_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        # Project each patch: (B, V, num_patches, d_model)
        x = self.ff(x)
        # Mean-pool over patches: (B, V, d_model)
        return x.mean(dim=2)


class Emb(nn.Module):
    """
    Multi-scale patch embedding for PatchMLP.

    Four EmbLayers with different patch sizes capture patterns at different
    temporal resolutions. Their outputs are concatenated along the feature dim.

    Output shape: (B, V, d_model)  — each of the V channels gets a d_model
    representation built from 4 temporal scales.

    Default patch_len for CTG at 4 Hz (4 pts = 1 s):
        1200 →  5 min  (baseline trend)
         600 →  2.5 min (full contraction response)
         300 →  75 s   (deceleration + recovery)
         150 →  37.5 s (single acceleration/deceleration)

    Previous bug: patch_step was set equal to patch_len (non-overlapping), then
    halved — creating overlapping patches, but EmbLayer used `d_model // patch_num`
    which rounds to 0 for long sequences. Fixed by using mean-pooling instead.
    """
    def __init__(
        self,
        seq_len:    int,
        d_model:    int,
        patch_len: list = None,
    ):
        super().__init__()
        if patch_len is None:
            patch_len = [1200, 600, 300, 150]   # CTG defaults at 4 Hz

        assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
        d_scale   = d_model // 4               # each scale contributes d_model//4 features
        patch_step = [p // 2 for p in patch_len]  # 50% overlap

        self.EmbLayer_1 = EmbLayer(patch_len[0], patch_step[0], d_scale)
        self.EmbLayer_2 = EmbLayer(patch_len[1], patch_step[1], d_scale)
        self.EmbLayer_3 = EmbLayer(patch_len[2], patch_step[2], d_scale)
        self.EmbLayer_4 = EmbLayer(patch_len[3], patch_step[3], d_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each EmbLayer: (B, V, d_model//4)
        s_x1 = self.EmbLayer_1(x)
        s_x2 = self.EmbLayer_2(x)
        s_x3 = self.EmbLayer_3(x)
        s_x4 = self.EmbLayer_4(x)
        # Concatenate along feature dim: (B, V, d_model)
        return torch.cat([s_x1, s_x2, s_x3, s_x4], dim=-1)