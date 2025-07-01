import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from scipy.stats import spearmanr, ConstantInputWarning
import warnings

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
        p["residual_layers"] = random.choice(s["residual_layers"])
        p["hidden_dim"] = random.choice(s["neurons"])
        p["learning_rate"] = random.choice(s["learning_rate"])
        p["kernel_size"] = random.choice(s["kernel_size"])
        p["optimizer"] = random.choice(s["optimizer"])
        p["batch_size"] = random.choice(s["batch_size"])
        p["use_lstm"] = random.choice(s["use_lstm"])

    elif model_name == "lstm":
        p["lstm_hidden_size"] = random.choice(s["neurons"])
        p["lstm_layers"] = random.choice(s["lstm_layers"])
        p["dense_size"] = random.choice(s["neurons"])
        p["dropout_lstm"] = random.choice(s["dropout"])
        p["dropout_fc"] = random.choice(s["dropout"])
        p["learning_rate"] = random.choice(s["learning_rate"])
        p["optimizer"] = random.choice(s["optimizer"])
        p["batch_size"] = random.choice(s["batch_size"])

    elif model_name == "cnn_lstm":
        p["cnn_channels"] = random.choices(s["cnn_channel"], k=random.choice(s["cnn_size"]))
        p["kernel_size"] = random.choice(s["kernel_size"])
        p["lstm_hidden_size"] = random.choice(s["neurons"])
        p["lstm_layers"] = random.choice(s["lstm_layers"])
        p["dense_size"] = random.choice(s["neurons"])
        p["dropout_cnn"] = random.choice(s["dropout"])
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

    n = len(losses)
    if n <= 1:
        return 1.0, 1.0

    # AUNL for training loss
    if np.all(losses == losses[0]):
        aunl = 1.0
    else:
        losses_min, losses_max = np.min(losses), np.max(losses)
        losses_scaled = (losses - losses_min) / (losses_max - losses_min)
        h = 1 / (n - 1)
        aunl = np.sum((losses_scaled[:-1] + losses_scaled[1:]) / 2) * h

    # AUNL for validation loss
    if np.all(val_losses == val_losses[0]):
        aunl_val = 1.0
    else:
        val_min, val_max = np.min(val_losses), np.max(val_losses)
        val_losses_scaled = (val_losses - val_min) / (val_max - val_min)
        h = 1 / (n - 1)
        aunl_val = np.sum((val_losses_scaled[:-1] + val_losses_scaled[1:]) / 2) * h

    return aunl, aunl_val


# === Metric Calculation ===
def calculate_metrics(scaler, y_true, y_pred, elapsed_time, type="test", scale=True):
    if scale:
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    mda = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0

    return {
        "model": None,  # to be filled later
        "model_component": None,
        "epoch": None,
        "type": type,
        "inference": elapsed_time,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "MDA": mda,
        "Spearman": spearman_corr,
    }

def drop_extra_targets(dataloader):
    # Unpack the (X, y, y_true) tuples into X and y only
    dataset = dataloader.dataset
    two_tensor_list = [ (x, y) for (x, y, _) in dataset ]
    
    # Stack all data along axis 0 to create new tensors
    X = torch.stack([x for (x, y) in two_tensor_list])
    y = torch.stack([y for (x, y) in two_tensor_list])

    return DataLoader(
        dataset=TensorDataset(X, y),
        batch_size=dataloader.batch_size,
        shuffle=False,
        drop_last=False,
    )

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        se = (y_true - y_pred) ** 2
        return torch.sqrt(torch.mean(se) + self.eps)