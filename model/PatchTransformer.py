"""
model/PatchTransformer.py
PatchTST-inspired transformer for time-series classification.
Reference: Nie et al., 2023 (https://arxiv.org/abs/2211.14730)

Splits the signal into non-overlapping patches, embeds each patch, and
processes with a Transformer encoder. Classification via a [CLS] token.

Patch size guide (at 4 Hz):
    30  →  7.5 s   | 240 patches
    60  → 15   s   | 120 patches  ← default (≈ deceleration length)
   120  → 30   s   |  60 patches
   240  → 60   s   |  30 patches
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Args (read from args):
        args.seq_len      — input time series length T
        args.in_channels  — number of input channels (default 2: FHR + TOCO)
        args.patch_size   — patch length in time steps (default 60)
        args.d_model      — transformer hidden dimension (default 64)
        args.num_heads    — number of attention heads (default 4, must divide d_model)
        args.e_layers     — number of transformer encoder layers (default 2)
        args.dropout      — dropout probability (default 0.1)
        args.num_classes  — 2 for binary (outputs (B,1)), >2 for multi-class
    """
    def __init__(self, args):
        super().__init__()
        input_length = args.seq_len
        in_channels  = getattr(args, 'in_channels', 2)
        patch_size   = getattr(args, 'patch_size',  60)
        d_model      = getattr(args, 'd_model',     64)
        num_heads    = getattr(args, 'num_heads',   4)
        num_layers   = getattr(args, 'e_layers',    2)
        dropout      = getattr(args, 'dropout',     0.1)
        out_dim      = 1 if args.num_classes == 2 else args.num_classes

        self.patch_size  = patch_size
        self.num_patches = input_length // patch_size  # truncate if not divisible
        self.in_channels = in_channels

        # Project flattened patch (C * patch_size) → d_model
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_channels * patch_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token and positional embeddings
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token,     std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise to (B, C, T) — data_provider returns (B, T, C)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        # Replace NaN (CTG signal dropouts) with 0
        x = torch.nan_to_num(x, nan=0.0)
        B, C, T = x.shape

        # Truncate to nearest multiple of patch_size
        T_use = self.num_patches * self.patch_size
        x = x[..., :T_use]

        # Patchify → (B, num_patches, C * patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size)            # (B, C, P, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)  # (B, P, C*patch_size)

        # Embed and add positional encoding
        x = self.patch_embedding(x)                                  # (B, P, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                               # (B, P+1, d_model)
        x = self.dropout(x + self.pos_embedding)

        # Transformer + classify via CLS token
        x = self.transformer(x)                                      # (B, P+1, d_model)
        return self.classifier(self.norm(x[:, 0]))                   # (B, out_dim)