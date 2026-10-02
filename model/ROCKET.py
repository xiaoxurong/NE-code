"""
model/ROCKET.py
RandOm Convolutional KErnel Transform (Dempster et al., 2020).
https://arxiv.org/abs/1910.13051

All convolutional weights are FIXED (not trained).
Only the linear head is trained — making this extremely fast.

Run this first as a diagnostic baseline before comparing deep models.
"""

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Args (read from args):
        args.seq_len      — input time series length T
        args.in_channels  — number of input channels (default 2: FHR + TOCO)
        args.num_kernels  — number of random kernels (default 10000)
        args.num_classes  — 2 for binary (outputs (B,1)), >2 for multi-class
        args.seed         — random seed for kernel generation
    """
    def __init__(self, args):
        super().__init__()
        input_length = args.seq_len
        in_channels  = getattr(args, 'in_channels',  2)
        num_kernels  = getattr(args, 'num_kernels',  10_000)
        seed         = getattr(args, 'seed',         42)

        # Binary → single logit output (BCEWithLogitsLoss + squeeze in exp_classification)
        out_dim = 1 if args.num_classes == 2 else args.num_classes
        self.in_channels = in_channels

        rng = np.random.default_rng(seed)

        candidate_lengths = np.array([7, 9, 11])
        lengths = rng.choice(candidate_lengths, num_kernels)

        max_exponents = np.floor(np.log2((input_length - 1) / (lengths - 1))).astype(int)
        max_exponents = np.maximum(max_exponents, 0)
        dilation_exps = np.array([rng.integers(0, e + 1) for e in max_exponents])
        dilations     = 2 ** dilation_exps

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
            w -= w.mean(axis=2, keepdims=True)
            b = rng.uniform(-1, 1, n).astype(np.float32)

            conv = nn.Conv1d(in_channels, n, length,
                             dilation=dilation, padding=padding, bias=True)
            conv.weight = nn.Parameter(torch.from_numpy(w), requires_grad=False)
            conv.bias   = nn.Parameter(torch.from_numpy(b), requires_grad=False)
            self.conv_layers.append(conv)

        # Freeze conv weights; only the linear head trains
        for p in self.parameters():
            p.requires_grad_(False)

        self.classifier = nn.Linear(2 * num_kernels, out_dim)
        self.classifier.weight.requires_grad_(True)
        self.classifier.bias.requires_grad_(True)

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        ppv_parts, max_parts = [], []
        for conv in self.conv_layers:
            out = conv(x)
            ppv_parts.append((out > 0).float().mean(dim=2))
            max_parts.append(out.amax(dim=2))
        ppv  = torch.cat(ppv_parts, dim=1)
        maxv = torch.cat(max_parts, dim=1)
        return torch.cat([ppv, maxv], dim=1)   # (B, 2 * num_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise to (B, C, T) — data_provider returns (B, T, C)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        # Replace NaN (CTG signal dropouts) with 0 before convolution
        x = torch.nan_to_num(x, nan=0.0)
        return self.classifier(self._extract_features(x))  # (B, out_dim)