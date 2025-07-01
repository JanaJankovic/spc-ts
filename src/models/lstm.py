import torch.nn as nn
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_shape,
        output_size=1,
        lstm_hidden_size=20,
        lstm_layers=3,
        dense_size=20,
        dropout_lstm=0.25,
        dropout_fc=0.25,
    ):
        super(SimpleLSTM, self).__init__()
        time_steps, num_features = input_shape

        self.lstm_stack = nn.ModuleList()
        for i in range(lstm_layers):
            in_size = num_features if i == 0 else lstm_hidden_size
            self.lstm_stack.append(
                nn.LSTM(
                    input_size=in_size, hidden_size=lstm_hidden_size, batch_first=True
                )
            )
        self.dropout1 = nn.Dropout(dropout_lstm)
        self.fc1 = nn.Linear(lstm_hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size, output_size)
        self.dropout2 = nn.Dropout(dropout_fc)

    def forward(self, x):
        # x: (batch, time, features)
        for lstm in self.lstm_stack:
            x, _ = lstm(x)
        x = x[:, -1, :]  # Use output of last timestep
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
