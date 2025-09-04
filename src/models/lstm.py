from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, x):
        # x: [batch, seq, features]
        output, (hn, cn) = self.lstm(x)
        last_hidden = output[:, -1, :]
        logits = self.head(last_hidden)
        return logits



