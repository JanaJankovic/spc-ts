# logs loss and val loss
# log metrics based on the metrics functionfrom sklearn.metrics import (
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from scipy.stats import spearmanr
import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import time
from datetime import datetime
from scipy.stats import spearmanr, ConstantInputWarning
import warnings


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        se = (y_true - y_pred) ** 2
        return torch.sqrt(torch.mean(se) + self.eps)


def calculate_metrics(y_true, y_pred, elapsed_time, type="test"):
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

    # Handle constant input warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0  # or np.nan if you prefer

    return {
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


def log_training_loss(log_path, epoch, train_loss, val_loss, start_time, end_time):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    start_dt = datetime.fromtimestamp(start_time)
    end_dt = datetime.fromtimestamp(end_time)

    entry = {
        "start_epoch_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_epoch_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    pd.DataFrame([entry]).to_csv(
        log_path, mode="a", index=False, header=not os.path.exists(log_path)
    )


def log_evaluation_metrics(
    log_path, epoch, y_true, y_pred, scaler, eval_type, elapsed_time, metric_fn
):
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)
    metrics = metric_fn(y_true, y_pred, elapsed_time, type=eval_type)
    metrics["epoch"] = epoch + 1
    metrics["type"] = eval_type

    column_order = [
        "epoch",
        "type",
        "inference",
        "MAE",
        "MSE",
        "RMSE",
        "MAPE",
        "R2",
        "MDA",
        "Spearman",
    ]
    metrics_ordered = {key: metrics[key] for key in column_order}

    pd.DataFrame([metrics_ordered]).to_csv(
        log_path, mode="a", index=False, header=not os.path.exists(log_path)
    )
