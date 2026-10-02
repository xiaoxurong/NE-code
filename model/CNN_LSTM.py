"""
model/CNN_LSTM.py
Two-layer 1D CNN followed by an LSTM for CTG classification.

CNN extracts local features → LSTM models temporal dependencies
across the CNN feature maps.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Args (read from args):
        args.seq_len      — input length in time steps (default 7200)
        args.in_channels  — input channels (default 2: FHR + TOCO)
        args.num_classes  — 2 for binary (outputs (B,1)), >2 for multi-class
        args.lstm_hidden  — LSTM hidden size (default 64)
        args.lstm_layers  — LSTM layers (default 1)
        args.dropout      — dropout probability (default 0.3)
    """
    def __init__(self, args):
        super().__init__()
        in_channels  = getattr(args, 'in_channels',  2)
        seq_len      = getattr(args, 'seq_len',      7200)
        lstm_hidden  = getattr(args, 'lstm_hidden',  64)
        lstm_layers  = getattr(args, 'lstm_layers',  1)
        dropout      = getattr(args, 'dropout',      0.3)
        out_dim      = 1 if args.num_classes == 2 else args.num_classes

        self.in_channels = in_channels

        kernel_size = 5
        padding     = kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size, padding=padding), nn.ReLU(), nn.MaxPool1d(2, 2),
            nn.Conv1d(32,          64, kernel_size, padding=padding), nn.ReLU(), nn.MaxPool1d(2, 2),
        )

        # Derive LSTM input size (= CNN output channels = 64) via a dry run
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            lstm_input_size = self.features(dummy).shape[1]   # channel dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),   # (B, 1) for binary — matches BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # data_provider returns (B, T, C); normalise to (B, C, T) for Conv1d
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        x = torch.nan_to_num(x, nan=0.0)

        x = self.features(x)           # (B, 64, T')
        x = x.transpose(1, 2)          # (B, T', 64) — time is sequence dim for LSTM
        lstm_out, _ = self.lstm(x)     # (B, T', lstm_hidden)
        x = lstm_out[:, -1, :]         # (B, lstm_hidden) — last time step
        return self.classifier(x)      # (B, 1) or (B, num_classes)