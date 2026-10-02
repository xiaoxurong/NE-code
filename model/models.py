"""
model.py — CTG classification models
=====================================
Existing models revised (bug fixes + dropout regularisation).
New models: InceptionTime, PatchTransformer, ROCKET, EnsembleModel.
"""

import math
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Existing models — revised
# ─────────────────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """
    Bug fixed: _get_flattened_size previously referenced self.pool / self.conv1 /
    self.conv2 which do not exist (the Sequential is self.features). Fixed.
    Dropout added for regularisation.
    """
    def __init__(self, input_size: int, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.flattened_size = self._get_flattened_size(input_size, in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def _get_flattened_size(self, input_size: int, in_channels: int) -> int:
        with torch.no_grad():
            x = torch.randn(1, in_channels, input_size)
            return self.features(x).numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class EnhancedCNN(nn.Module):
    """4-layer CNN. Added Dropout."""
    def __init__(self, input_size: int, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32,  64,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(64,  128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2, 2),
        )
        self.flattened_size = self._get_flattened_size(input_size, in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def _get_flattened_size(self, input_size: int, in_channels: int) -> int:
        with torch.no_grad():
            return self.features(torch.randn(1, in_channels, input_size)).numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class CNN1D(nn.Module):
    """Single-channel 2-layer CNN."""
    def __init__(self, input_size: int, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
        )
        self.flattened_size = self._get_flattened_size(input_size, in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def _get_flattened_size(self, input_size: int, in_channels: int) -> int:
        with torch.no_grad():
            return self.features(torch.randn(1, in_channels, input_size)).numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class EnhancedCNN1D(nn.Module):
    """Single-channel 3-layer CNN."""
    def __init__(self, input_size: int, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32,  kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32,  64,  kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(64,  128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2, 2),
        )
        self.flattened_size = self._get_flattened_size(input_size, in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def _get_flattened_size(self, input_size: int, in_channels: int) -> int:
        with torch.no_grad():
            return self.features(torch.randn(1, in_channels, input_size)).numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class CNN_LSTM(nn.Module):
    """
    Bug fixed: CNN output was fed to LSTM without transposing, making the channel
    dimension the sequence dimension. Fixed to transpose (B, C, T') → (B, T', C)
    so the LSTM models temporal dependencies across CNN feature maps.
    _get_flattened_size now returns the channel dim (LSTM input_size), not the
    time dim.
    """
    def __init__(
        self,
        input_size: int,
        in_channels: int = 1,
        num_classes: int = 2,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        kernel_size = 5
        padding = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size, padding=padding), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64,       kernel_size, padding=padding), nn.ReLU(), nn.MaxPool1d(2, 2),
        )

        # input_size for LSTM = number of CNN output channels (64)
        lstm_input_size = self._get_cnn_out_channels(input_size, in_channels)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def _get_cnn_out_channels(self, input_size: int, in_channels: int) -> int:
        with torch.no_grad():
            out = self.features(torch.randn(1, in_channels, input_size))
            return out.shape[1]  # channel dim → LSTM input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, C, T')
        x = x.transpose(1, 2)         # (B, T', C) — time is the sequence dim for LSTM
        lstm_out, _ = self.lstm(x)    # (B, T', hidden)
        x = lstm_out[:, -1, :]        # (B, hidden) — last time step
        return self.classifier(x)


class TimeSeriesTransformer(nn.Module):
    """
    Bug fixed: previously flattened d_model * seq_len into a huge linear layer
    (e.g. 64 * 7200 = 460800 weights) which is both slow and heavily overfit-prone
    on small datasets. Replaced with global average pooling over time, which is
    standard practice and parameter-free.
    Debug print() removed.
    Note: for long CTG sequences (>1000 steps) prefer PatchTransformer — the O(T²)
    attention here is expensive.
    """
    def __init__(
        self,
        input_size: int,         # number of input channels / features per time step
        num_heads: int = 4,
        num_layers: int = 2,
        d_model: int = 64,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.permute(0, 2, 1)          # (B, T, C)
        x = self.embedding(x)            # (B, T, d_model)
        x = self.transformer(x)          # (B, T, d_model)
        x = self.norm(x.mean(dim=1))     # global avg pool over T → (B, d_model)
        return self.classifier(x)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear.
    math import moved to top of file.
    Initialisation: lora_A from N(0,1)*scaling, lora_B zeroed so the adapter
    starts as an identity (no perturbation) — this is the canonical LoRA init.
    """
    def __init__(self, original_layer: nn.Linear, r: int = 4, scaling_factor: float = 0.01):
        super().__init__()
        self.original_layer = original_layer
        out_f, in_f = original_layer.out_features, original_layer.in_features

        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))  # zero init → no perturbation at start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.data *= scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + F.linear(x, self.lora_B @ self.lora_A)


# ─────────────────────────────────────────────────────────────────────────────
# New model 1 — InceptionTime
# ─────────────────────────────────────────────────────────────────────────────

class _InceptionBlock(nn.Module):
    """
    Single Inception module: bottleneck → three parallel conv scales + maxpool branch.
    Output channels = nb_filters * 4.
    """
    def __init__(self, in_channels: int, nb_filters: int = 32, bottleneck_size: int = 32):
        super().__init__()
        # Bottleneck projection (skip for very small in_channels)
        self.use_bottleneck = in_channels > bottleneck_size
        bn_ch = bottleneck_size if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, 1, bias=False)

        self.conv_small  = nn.Conv1d(bn_ch, nb_filters, kernel_size=9,  padding=4,  bias=False)
        self.conv_medium = nn.Conv1d(bn_ch, nb_filters, kernel_size=19, padding=9,  bias=False)
        self.conv_large  = nn.Conv1d(bn_ch, nb_filters, kernel_size=39, padding=19, bias=False)

        self.maxpool     = nn.MaxPool1d(3, stride=1, padding=1)
        self.mp_conv     = nn.Conv1d(in_channels, nb_filters, 1, bias=False)

        self.bn  = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x) if self.use_bottleneck else x
        branches = [
            self.conv_small(z),
            self.conv_medium(z),
            self.conv_large(z),
            self.mp_conv(self.maxpool(x)),
        ]
        return self.act(self.bn(torch.cat(branches, dim=1)))


class _ResidualInceptionGroup(nn.Module):
    """Three InceptionBlocks with a residual shortcut around the group."""
    def __init__(self, in_channels: int, nb_filters: int = 32, num_blocks: int = 3):
        super().__init__()
        nb_ch_out = nb_filters * 4
        blocks = []
        ch = in_channels
        for _ in range(num_blocks):
            blocks.append(_InceptionBlock(ch, nb_filters))
            ch = nb_ch_out
        self.blocks = nn.Sequential(*blocks)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, nb_ch_out, 1, bias=False),
            nn.BatchNorm1d(nb_ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.blocks(x) + self.shortcut(x))


class InceptionTime(nn.Module):
    """
    InceptionTime: Fawaz et al., 2020 (https://arxiv.org/abs/1909.04939).
    Multi-scale 1D CNN with residual connections. Does NOT need input_length
    (uses AdaptiveAvgPool1d for the global pooling step).

    Args:
        in_channels:  number of input channels (2 for FHR + TOCO).
        nb_filters:   filters per inception branch (output = nb_filters * 4 per block).
        num_groups:   number of residual groups (each = 3 inception blocks). Default 2.
        num_classes:  number of output classes.
    """
    def __init__(
        self,
        in_channels: int = 2,
        nb_filters: int = 32,
        num_groups: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()
        nb_ch_out = nb_filters * 4
        groups = []
        ch = in_channels
        for _ in range(num_groups):
            groups.append(_ResidualInceptionGroup(ch, nb_filters))
            ch = nb_ch_out
        self.network    = nn.Sequential(*groups)
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(nb_ch_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)              # (B, nb_ch_out, T)
        x = self.gap(x).squeeze(-1)      # (B, nb_ch_out)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# New model 2 — PatchTransformer (PatchTST-style)
# ─────────────────────────────────────────────────────────────────────────────

class PatchTransformer(nn.Module):
    """
    PatchTST-inspired transformer for time-series classification.
    Reference: Nie et al., 2023 (https://arxiv.org/abs/2211.14730).

    Splits the input signal into non-overlapping patches, linearly projects each
    patch, prepends a learnable [CLS] token, and processes with a Transformer
    encoder. Classification is done from the CLS token.

    Why patches?
      - Reduces sequence length from T to T/patch_size → O(T²/P²) attention cost.
      - Each patch is a local receptive field (natural for CTG patterns like
        decelerations which span several seconds).
      - Less overfitting than full-sequence transformers on small datasets.

    Args:
        input_length:  length of the input time series T.
        in_channels:   number of input channels (2 for FHR + TOCO).
        patch_size:    number of time steps per patch. Choose so that one patch
                       covers a clinically meaningful window (e.g. patch_size=60
                       at 4 Hz = 15 s per patch).
        d_model:       transformer hidden dimension.
        num_heads:     attention heads (must divide d_model).
        num_layers:    transformer encoder layers.
        dropout:       dropout probability.
        num_classes:   output classes.
    """
    def __init__(
        self,
        input_length: int,
        in_channels: int = 2,
        patch_size: int = 60,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = input_length // patch_size  # truncate if not divisible

        # Project flattened patch → d_model
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
            norm_first=True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape

        # Truncate to nearest multiple of patch_size
        T_use = self.num_patches * self.patch_size
        x = x[..., :T_use]                                           # (B, C, T_use)

        # Patchify → (B, num_patches, C * patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size)            # (B, C, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)  # (B, num_patches, C*patch_size)

        # Embed + positional encoding
        x = self.patch_embedding(x)                                  # (B, num_patches, d_model)
        cls = self.cls_token.expand(B, -1, -1)                       # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                               # (B, num_patches+1, d_model)
        x = self.dropout(x + self.pos_embedding)

        # Transformer
        x = self.transformer(x)                                      # (B, num_patches+1, d_model)

        # Classify via CLS token
        return self.classifier(self.norm(x[:, 0]))                   # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# New model 3 — ROCKET
# ─────────────────────────────────────────────────────────────────────────────

class ROCKET(nn.Module):
    """
    ROCKET: RandOm Convolutional KErnel Transform (Dempster et al., 2020).
    https://arxiv.org/abs/1910.13051

    All convolutional weights are FIXED (random, not trained). Only the linear
    classifier head is trained — making this extremely fast and surprisingly
    competitive on small datasets.

    Use this as a diagnostic baseline first:
      - If ROCKET ≈ your CNNs → your CNNs are not learning beyond random projections.
      - If ROCKET >> your CNNs → you likely need better regularisation / less capacity.

    Kernels:
      - Lengths sampled from {7, 9, 11}.
      - Weights: N(0,1), mean-subtracted per kernel.
      - Biases: U(-1, 1).
      - Dilations: powers of 2 up to floor(log2((T-1)/(length-1))).
      - Paddings: 0 or (length-1)*dilation//2 with equal probability.

    Features per kernel: PPV (proportion of positive activations) + global max.
    Total feature dim: 2 * num_kernels.
    Only the linear head (2*num_kernels → num_classes) is trained.

    Args:
        input_length: length of input time series T.
        in_channels:  number of input channels.
        num_kernels:  number of random kernels (10 000 is the paper default).
        num_classes:  number of output classes.
        seed:         random seed for reproducibility.
    """
    def __init__(
        self,
        input_length: int,
        in_channels: int = 2,
        num_kernels: int = 10_000,
        num_classes: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)

        candidate_lengths = np.array([7, 9, 11])
        lengths = rng.choice(candidate_lengths, num_kernels)

        # Max dilation exponent for each kernel
        max_exponents = np.floor(np.log2((input_length - 1) / (lengths - 1))).astype(int)
        max_exponents = np.maximum(max_exponents, 0)
        dilation_exps = np.array([rng.integers(0, e + 1) for e in max_exponents])
        dilations     = 2 ** dilation_exps

        # Padding: 0 or half-padding
        half_paddings = ((lengths - 1) * dilations) // 2
        paddings      = np.where(rng.random(num_kernels) < 0.5, half_paddings, 0)

        # Group kernels by (length, dilation, padding) for batched Conv1d
        groups: dict = defaultdict(list)
        for i in range(num_kernels):
            groups[(int(lengths[i]), int(dilations[i]), int(paddings[i]))].append(i)

        self.conv_layers: nn.ModuleList = nn.ModuleList()
        for (length, dilation, padding), indices in groups.items():
            n = len(indices)
            w = rng.standard_normal((n, in_channels, length)).astype(np.float32)
            w -= w.mean(axis=2, keepdims=True)   # mean-centre each kernel
            b = rng.uniform(-1, 1, n).astype(np.float32)

            conv = nn.Conv1d(in_channels, n, length, dilation=dilation, padding=padding, bias=True)
            conv.weight = nn.Parameter(torch.from_numpy(w), requires_grad=False)
            conv.bias   = nn.Parameter(torch.from_numpy(b), requires_grad=False)
            self.conv_layers.append(conv)

        # Freeze everything except the linear head
        for p in self.parameters():
            p.requires_grad_(False)

        self.classifier = nn.Linear(2 * num_kernels, num_classes)
        self.classifier.weight.requires_grad_(True)
        self.classifier.bias.requires_grad_(True)

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract PPV and max features from all kernel groups."""
        ppv_parts, max_parts = [], []
        for conv in self.conv_layers:
            out = conv(x)                                    # (B, n_group, T')
            ppv_parts.append((out > 0).float().mean(dim=2)) # (B, n_group)
            max_parts.append(out.amax(dim=2))               # (B, n_group)
        ppv = torch.cat(ppv_parts, dim=1)                   # (B, num_kernels)
        maxv = torch.cat(max_parts, dim=1)                  # (B, num_kernels)
        return torch.cat([ppv, maxv], dim=1)                 # (B, 2*num_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._extract_features(x))


# ─────────────────────────────────────────────────────────────────────────────
# New model 4 — EnsembleModel
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleModel(nn.Module):
    """
    Soft-voting ensemble: averages the softmax probabilities from each member.
    All member models must accept the same input and return (B, num_classes) logits.

    Usage:
        ensemble = EnsembleModel([cnn, cnn_lstm, patch_transformer])
        logits   = ensemble(x)   # averaged probabilities, then log for loss if needed

    Training tip: train each member independently, then combine. Fine-tuning the
    ensemble jointly is also possible but risks the members collapsing to the same
    solution.
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average softmax probabilities across members
        probs = torch.stack(
            [F.softmax(m(x), dim=-1) for m in self.models], dim=0
        )                       # (num_models, B, num_classes)
        return probs.mean(dim=0)  # (B, num_classes) — these ARE probabilities, not logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)