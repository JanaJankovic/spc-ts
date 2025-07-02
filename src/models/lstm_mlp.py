import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_MLP(nn.Module):
    def __init__(
        self,
        temporal_dim,
        static_dim=0,
        rnn_hidden=64,
        mlp_hidden=32,
        fusion_hidden=64,
        output_dim=1,
        use_lstm=True,
        dropout_rnn=0.2,
        dropout_fusion=0.2,
    ):
        super().__init__()

        rnn_cls = nn.LSTM if use_lstm else nn.RNN
        self.rnn = rnn_cls(
            input_size=temporal_dim, hidden_size=rnn_hidden, batch_first=True
        )
        self.dropout_rnn = nn.Dropout(dropout_rnn)

        self.use_mlp = static_dim > 0
        if self.use_mlp:
            self.mlp_static = nn.Sequential(
                nn.Linear(static_dim, mlp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_fusion),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.ReLU(),
            )

            self.fusion = nn.Sequential(
                nn.Linear(rnn_hidden + mlp_hidden, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_fusion),
                nn.Linear(fusion_hidden, output_dim),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(rnn_hidden, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_fusion),
                nn.Linear(fusion_hidden, output_dim),
            )

    def forward(self, x_temporal, x_static=None):
        rnn_out, _ = self.rnn(x_temporal)
        rnn_feat = self.dropout_rnn(rnn_out[:, -1, :])

        if self.use_mlp and x_static is not None:
            static_feat = self.mlp_static(x_static)
            combined = torch.cat([rnn_feat, static_feat], dim=1)
        else:
            combined = rnn_feat

        return self.fusion(combined)
