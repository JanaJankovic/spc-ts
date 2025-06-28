import torch
import torch.nn as nn
import torch.nn.functional as F

# === Main Model ===
class BasePredictor(nn.Module):
    def __init__(self, temporal_dim, static_dim, rnn_hidden=64, mlp_hidden=32, fusion_hidden=64, output_dim=1, use_lstm=True):
        super().__init__()
        self.rnn_type = 'LSTM' if use_lstm else 'RNN'
        rnn_cls = nn.LSTM if use_lstm else nn.RNN

        self.rnn = rnn_cls(input_size=temporal_dim, hidden_size=rnn_hidden, batch_first=True)
        self.dropout_rnn = nn.Dropout(0.2)

        self.mlp_static = nn.Sequential(
            nn.Linear(static_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(rnn_hidden + mlp_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, output_dim)
        )

    def forward(self, x_temporal, x_static):
        # x_temporal: [B, T, F], x_static: [B, static_dim]
        rnn_out, _ = self.rnn(x_temporal)
        rnn_feat = self.dropout_rnn(rnn_out[:, -1, :])  # Last timestep

        static_feat = self.mlp_static(x_static)
        combined = torch.cat([rnn_feat, static_feat], dim=1)
        return self.fusion(combined)  # [B, output_dim]


# === Residual CNN ===
class ResidualCNN(nn.Module):
    def __init__(self, temporal_dim, cnn_channels=16, kernel_size=3, output_dim=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=temporal_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=output_dim, kernel_size=1)
        )

    def forward(self, x_temporal):
        # x_temporal: [B, T, F]
        x = x_temporal.permute(0, 2, 1)  # [B, F, T]
        x = self.cnn(x)
        x = x.mean(dim=2)  # Global average pooling over time
        return x.squeeze(-1)  # [B] or [B, output_dim]
