import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePredictor(nn.Module):
    def __init__(
        self,
        temporal_dim,  # Number of input features per timestep
        static_dim=0,  # Set to 0 if no static features
        rnn_hidden=64,
        mlp_hidden=32,
        fusion_hidden=64,
        output_dim=1,  # Forecasting horizon
        use_lstm=True,
    ):
        super().__init__()

        rnn_cls = nn.LSTM if use_lstm else nn.RNN
        self.rnn = rnn_cls(
            input_size=temporal_dim, hidden_size=rnn_hidden, batch_first=True
        )
        self.dropout_rnn = nn.Dropout(0.2)

        self.use_mlp = static_dim > 0
        if self.use_mlp:
            self.mlp_static = nn.Sequential(
                nn.Linear(static_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.ReLU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(rnn_hidden + mlp_hidden, fusion_hidden),
                nn.ReLU(),
                nn.Linear(fusion_hidden, output_dim),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(rnn_hidden, fusion_hidden),
                nn.ReLU(),
                nn.Linear(fusion_hidden, output_dim),
            )

    def forward(self, x_temporal, x_static=None):
        # x_temporal: [B, T, F]
        rnn_out, _ = self.rnn(x_temporal)  # [B, T, hidden]
        rnn_feat = self.dropout_rnn(rnn_out[:, -1, :])  # [B, hidden]

        if self.use_mlp and x_static is not None:
            static_feat = self.mlp_static(x_static)  # [B, mlp_hidden]
            combined = torch.cat([rnn_feat, static_feat], dim=1)
        else:
            combined = rnn_feat

        return self.fusion(combined)  # [B, output_dim]


# === Residual CNN ===
class ResidualCNN(nn.Module):
    def __init__(self, temporal_dim, cnn_channels=16, kernel_size=3, output_dim=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=temporal_dim,
                out_channels=cnn_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=output_dim, kernel_size=1),
        )

    def forward(self, x_temporal):
        # x_temporal: [B, T, F]
        x = x_temporal.permute(0, 2, 1)  # [B, F, T]
        x = self.cnn(x)
        x = x.mean(dim=2)  # Global average pooling over time
        return x.squeeze(-1)  # [B] or [B, output_dim]
