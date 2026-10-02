"""
model/PatchMLP.py
Multi-scale patch MLP for CTG classification.

Architecture:
  Emb: each input channel (FHR, TOCO) is independently patchified at 4 scales,
       each patch projected to d_model//4 via a linear + mean-pool over patches,
       then concatenated → (B, 2, d_model).
  Encoder: MLP-Mixer style — ff1 mixes features (d_model dim),
           ff2 mixes across the 2 channel tokens (enc_in=2 dim).
  Classifier: mean over 2 channel tokens → Linear → logit.

Patch size guide (at 4 Hz, 4 pts = 1 s):
    150  →  37.5 s  (single deceleration)
    300  →  75   s  (deceleration + recovery)
    600  → 150   s  (full contraction response)
   1200  →   5 min  (baseline trend)
"""

import torch
import torch.nn as nn
from layers.Embed import Emb


class Encoder(nn.Module):
    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feature mixing: operates on d_model dimension
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Token mixing: operates on enc_in=2 dimension (FHR ↔ TOCO)
        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # x: (B, 2, d_model)
        y_0 = self.norm1(self.ff1(x) + x)          # feature mix + residual

        y_1 = y_0.permute(0, 2, 1)                  # (B, d_model, 2)
        y_1 = self.ff2(y_1)                          # token mix across 2 channels ✓
        y_1 = y_1.permute(0, 2, 1)                  # (B, 2, d_model)

        return self.norm2(y_1 * y_0 + x)            # gated residual


class Model(nn.Module):
    """
    Args (read from args):
        args.in_channels  — number of input channels (default 2: FHR + TOCO)
        args.seq_len      — input window length in time steps (default 7200)
        args.d_model      — hidden dimension; must be divisible by 4 (default 64)
        args.e_layers     — number of Encoder blocks (default 2)
        args.use_norm     — instance-normalize each channel over time (default 1)
        args.num_classes  — 2 for binary (outputs (B,1)), >2 for multi-class

    Note: the token-mixing MLP always uses in_channels as its dimension.
    enc_in in the YAML is ignored by this model — remove it to avoid confusion.
    """
    def __init__(self, configs):
        super().__init__()
        self.in_channels = getattr(configs, 'in_channels', 2)
        self.use_norm    = getattr(configs, 'use_norm', 1)
        d_model          = getattr(configs, 'd_model', 64)
        e_layers         = getattr(configs, 'e_layers', 2)

        self.emb = Emb(configs.seq_len, d_model)

        self.encoder_layers = nn.ModuleList([
            Encoder(d_model, self.in_channels)   # token dim always = in_channels
            for _ in range(e_layers)
        ])

        self.classifier = nn.Linear(d_model, 1 if configs.num_classes == 2 else configs.num_classes)

    def forward(self, x):
        # data_provider returns (B, T, C); normalise to (B, T, C) if not already
        if x.shape[-1] != self.in_channels:
            x = x.permute(0, 2, 1)             # → (B, T, C)
        x = torch.nan_to_num(x, nan=0.0)       # CTG signal-dropout protection

        if self.use_norm:
            # Instance-normalise each channel over time (dim=1 for (B,T,C))
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

        x = self.emb(x)                         # (B, 2, d_model)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)                       # (B, d_model) — pool over 2 channel tokens
        return self.classifier(x)               # (B, 1) or (B, num_classes)