import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=32, num_layers=1, output_dim=1, dropout=0.2
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1 else 0.0
            ),  # dropout applied between layers if stacked LSTM
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # take last time step output
        out = self.dropout(out)
        out = self.fc(out)
        return out
