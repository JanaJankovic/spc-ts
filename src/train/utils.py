import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
JSON_PATH = os.path.join(SRC_DIR, "hpo.json")


def get_parameters(model_name):
    with open(JSON_PATH, "r") as f:
        space = json.load(f)

    if model_name not in space:
        raise ValueError(f"❌ Model '{model_name}' not found in HPO search space JSON.")

    s = space[model_name]
    p = {}  # parameters

    if model_name == "base_residual":
        p["rnn_hidden"] = random.choice(s["neurons"])
        p["mlp_hidden"] = random.choice(s["neurons"])
        p["fusion_hidden"] = random.choice(s["neurons"])
        p["dropout_rnn"] = random.choice(s["dropout"])
        p["dropout_fc"] = random.choice(s["dropout"])
        p["dropout_cnn"] = random.choice(s["dropout"])
        p["cnn_channels"] = random.choices(s["cnn_channel"], k=random.choice(s["cnn_size"]))
        p["learning_rate"] = random.choice(s["learning_rate"])
        p["kernel_size"] = random.choice(s["kernel_size"])
        p["optimizer"] = random.choice(s["optimizer"])
        p["batch_size"] = random.choice(s["batch_size"])


    elif model_name == "cnn_lstm":
        p["cnn_channels"] = random.choices(s["cnn_channel"], k=random.choice(s["cnn_size"]))
        p["kernel_size"] = random.choice(s["kernel_size"])
        p["lstm_hidden_size"] = random.choice(s["neurons"])
        p["lstm_layers"] = random.choice(s["lstm_layers"])
        p["dense_size"] = random.choice(s["neurons"])
        p["dropout_conv"] = random.choice(s["dropout"])
        p["dropout_fc"] = random.choice(s["dropout"])
        p["learning_rate"] = random.choice(s["learning_rate"])
        p["use_maxpool"] = random.choice(s["use_maxpool"])
        p["optimizer"] = random.choice(s["optimizer"])
        p["batch_size"] = random.choice(s["batch_size"])



    elif model_name == "di_rnn":
        p["hidden_size"] = random.choice(s["neurons"])
        p["bp_hidden_size"] = random.choice(s["neurons"])
        p["dropout_rnn"] = random.choice(s["dropout"])
        p["lr_rnn"] = random.choice(s["learning_rate"])
        p["lr_bpnn"] = random.choice(s["learning_rate"])
        p["optimizer"] = random.choice(s["optimizer"])
        p["batch_size"] = random.choice(s["batch_size"])



    elif model_name == "cnn_di_rnn":
        p["hidden_size"] = random.choice(s["neurons"])
        p["bp_hidden_size"] = random.choice(s["neurons"])
        p["dropout_rnn"] = random.choice(s["dropout"])
        p["lr_rnn"] = random.choice(s["learning_rate"])
        p["lr_bpnn"] = random.choice(s["learning_rate"])
        p["cnn_channels"] = random.choice(s["cnn_channel"])
        p["kernel_size"] = random.choice(s["kernel_size"])
        p["optimizer"] = random.choice(s["optimizer"])
        p["batch_size"] = random.choice(s["batch_size"])


    else:
        raise ValueError(f"⚠️ No sampling logic defined for model '{model_name}'.")

    return p


def calculate_aunl(losses, val_losses):
    # Ensure inputs are numpy arrays
    losses = np.array(losses)
    val_losses = np.array(val_losses)

    # Number of points
    n = len(losses)

    if n <= 1:
        return 1, 1

    # Min-Max scaling of losses and val_losses
    losses_scaled = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
    val_losses_scaled = (val_losses - np.min(val_losses)) / (
        np.max(val_losses) - np.min(val_losses)
    )

    # Calculate AUNL using trapezoidal rule
    h = 1 / (n - 1)  # Uniform step size (normalized over the range [a, b])
    aunl = np.sum((losses_scaled[:-1] + losses_scaled[1:]) / 2) * h
    aunl_val = np.sum((val_losses_scaled[:-1] + val_losses_scaled[1:]) / 2) * h

    return aunl, aunl_val


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        se = (y_true - y_pred) ** 2
        return torch.sqrt(torch.mean(se) + self.eps)