import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    def __init__(self, input_shape, output_size=1):
        super(CNNLSTMModel, self).__init__()
        time_steps, num_features = input_shape

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=48, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2) if time_steps >= 4 else nn.Identity()
        time_steps = time_steps // 2 if time_steps >= 4 else time_steps

        self.conv2 = nn.Conv1d(in_channels=48, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2) if time_steps >= 4 else nn.Identity()
        time_steps = time_steps // 2 if time_steps >= 4 else time_steps

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2) if time_steps >= 4 else nn.Identity()
        time_steps = time_steps // 2 if time_steps >= 4 else time_steps

        self.dropout1 = nn.Dropout(0.25)

        # LSTM expects input of shape (batch_size, time_steps, features)
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=20, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=20, hidden_size=20, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=20, hidden_size=20, batch_first=True, bidirectional=False)

        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, output_size)

    def forward(self, x):
        # x: (batch, time, features)
        x = x.permute(0, 2, 1)  # to (batch, features, time) for Conv1d

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # back to (batch, time, features) for LSTM

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)  # output shape: (batch, time, 20)

        x = x[:, -1, :]  # take last timestep
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
