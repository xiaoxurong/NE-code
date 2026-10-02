"""
model/InceptionTime.py
InceptionTime: Fawaz et al., 2020 (https://arxiv.org/abs/1909.04939)
Multi-scale 1D CNN with residual connections for time-series classification.
Does not need seq_len — uses AdaptiveAvgPool1d for the final pooling step.

CTG-specific kernel size guide (at 4 Hz):
    Original paper defaults (9, 19, 39) are designed for UCR datasets and
    correspond to 2s / 5s / 10s — far too short to capture CTG patterns.

    Clinically meaningful windows:
        Short-term variability:  10–20s →  41–81 points
        Variable deceleration:   30–60s → 121–241 points
        Late deceleration:       60–90s → 241–361 points

    Recommended defaults for CTG (set via args.kernel_sizes):
        [41, 121, 241]  →  ~10s / ~30s / ~60s
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _InceptionBlock(nn.Module):
    """Single Inception module: bottleneck → 3 parallel scales + maxpool branch."""
    def __init__(
        self,
        in_channels:     int,
        nb_filters:      int = 32,
        bottleneck_size: int = 32,
        kernel_sizes:    List[int] = (41, 121, 241),
        dropout:         float = 0.0,
    ):
        super().__init__()
        self.use_bottleneck = in_channels > bottleneck_size
        bn_ch = bottleneck_size if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, 1, bias=False)

        # Three parallel convolutions — kernel sizes must be odd for symmetric padding
        k0, k1, k2 = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.conv_small  = nn.Conv1d(bn_ch, nb_filters, k0, padding=k0 // 2, bias=False)
        self.conv_medium = nn.Conv1d(bn_ch, nb_filters, k1, padding=k1 // 2, bias=False)
        self.conv_large  = nn.Conv1d(bn_ch, nb_filters, k2, padding=k2 // 2, bias=False)

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_channels, nb_filters, 1, bias=False)

        self.bn      = nn.BatchNorm1d(nb_filters * 4)
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x) if self.use_bottleneck else x
        branches = [
            self.conv_small(z),
            self.conv_medium(z),
            self.conv_large(z),
            self.mp_conv(self.maxpool(x)),
        ]
        return self.dropout(self.act(self.bn(torch.cat(branches, dim=1))))


class _ResidualGroup(nn.Module):
    """Three InceptionBlocks with a residual shortcut around the group."""
    def __init__(
        self,
        in_channels:  int,
        nb_filters:   int = 32,
        kernel_sizes: List[int] = (41, 121, 241),
        dropout:      float = 0.0,
    ):
        super().__init__()
        nb_ch_out = nb_filters * 4
        self.block1 = _InceptionBlock(in_channels, nb_filters, kernel_sizes=kernel_sizes, dropout=dropout)
        self.block2 = _InceptionBlock(nb_ch_out,   nb_filters, kernel_sizes=kernel_sizes, dropout=dropout)
        self.block3 = _InceptionBlock(nb_ch_out,   nb_filters, kernel_sizes=kernel_sizes, dropout=dropout)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, nb_ch_out, 1, bias=False),
            nn.BatchNorm1d(nb_ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block3(self.block2(self.block1(x))) + self.shortcut(x))


class Model(nn.Module):
    """
    Args (read from args):
        args.in_channels  — input channels (default 2: FHR + TOCO)
        args.d_model      — nb_filters per inception branch (output = d_model*4 per block)
        args.e_layers     — number of residual groups (each = 3 inception blocks)
        args.kernel_sizes — list of 3 kernel sizes in *original* time steps (default [41, 121, 241])
                            At 4 Hz: 41≈10s, 121≈30s, 241≈60s.
                            These are automatically rescaled to post-stem time steps.
        args.stem_stride  — temporal downsampling factor applied before inception blocks (default 4).
                            stride=4: T 7200 → 1800, making large-kernel convs ~16× cheaper.
                            Clinical meaning of kernel_sizes is preserved — specify them in
                            original time steps regardless of stride.
        args.num_classes  — 2 for binary (outputs (B,1)), >2 for multi-class
    """
    def __init__(self, args):
        super().__init__()
        in_channels  = getattr(args, 'in_channels',  2)
        nb_filters   = getattr(args, 'd_model',      32)
        num_groups   = getattr(args, 'e_layers',     2)
        kernel_sizes = getattr(args, 'kernel_sizes', [41, 121, 241])
        stem_stride  = getattr(args, 'stem_stride',  4)
        out_dim      = 1 if args.num_classes == 2 else args.num_classes

        self.in_channels = in_channels

        # ── Downsampling stem ────────────────────────────────────────────────
        # Reduces T by stem_stride before the expensive inception blocks.
        # A learned strided conv (rather than avg/max pooling) lets the network
        # decide which temporal features to preserve during downsampling.
        stem_k = stem_stride * 2 + 1    # e.g. stride=4 → k=9; always odd
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, nb_filters, stem_k,
                      stride=stem_stride, padding=stem_k // 2, bias=False),
            nn.BatchNorm1d(nb_filters),
            nn.ReLU(),
        )

        # ── Rescale kernel sizes to post-stem resolution ─────────────────────
        # kernel_sizes are specified in original time steps; divide by stride,
        # clamp to ≥3, and enforce odd values for symmetric padding.
        scaled_ks = [max(3, k // stem_stride) for k in kernel_sizes]
        scaled_ks = [k if k % 2 == 1 else k + 1 for k in scaled_ks]

        dropout   = getattr(args, 'dropout', 0.0)

        nb_ch_out = nb_filters * 4
        groups, ch = [], nb_filters   # stem already outputs nb_filters channels
        for _ in range(num_groups):
            groups.append(_ResidualGroup(ch, nb_filters, kernel_sizes=scaled_ks, dropout=dropout))
            ch = nb_ch_out

        self.network    = nn.Sequential(*groups)
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(nb_ch_out, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise to (B, C, T) — data_provider returns (B, T, C)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        x = torch.nan_to_num(x, nan=0.0)
        x = self.stem(x)              # (B, nb_filters, T/stem_stride)
        x = self.network(x)           # (B, nb_ch_out, T/stem_stride)
        x = self.gap(x).squeeze(-1)   # (B, nb_ch_out)
        return self.classifier(x)     # (B, out_dim)