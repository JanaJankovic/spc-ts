import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from scipy.stats import spearmanr, ConstantInputWarning
import warnings

# === File Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOSS_LOG = os.path.join(LOG_DIR, "loss.csv")
METRIC_LOG = os.path.join(LOG_DIR, "metrics.csv")
TRIAL_LOG = os.path.join(LOG_DIR, "trial_info.csv")
PRED_DATA = os.path.join(LOG_DIR, "eval_data.csv")


# === Metric Calculation ===
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


# === Log Functions ===
def log_trial_info(model_name: str, model_type: str, trial: int, params: dict, data_config: dict):
    entry = {
        "model_name": model_name,
        "model_type": model_type,
        "trial": trial,
        "data_config": json.dumps(data_config, sort_keys=True),
        "params": json.dumps(params, sort_keys=True),
    }

    df = pd.DataFrame([entry], columns=["model_name", "model_type", "trial", "data_config", "params"])
    df.to_csv(TRIAL_LOG, mode="a", index=False, header=not os.path.exists(TRIAL_LOG))


def log_training_loss(epoch, train_loss, val_loss, start_time, end_time, model_name, model_component="main"):
    start_dt = datetime.fromtimestamp(start_time)
    end_dt = datetime.fromtimestamp(end_time)

    entry = {
        "model": model_name,
        "component": model_component,
        "epoch": epoch + 1,
        "start_epoch_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_epoch_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    df = pd.DataFrame([entry], columns=[
        "model",
        "component",
        "epoch",
        "start_epoch_time",
        "end_epoch_time",
        "train_loss",
        "val_loss",
    ])
    df.to_csv(LOSS_LOG, mode="a", index=False, header=not os.path.exists(LOSS_LOG))


def log_evaluation_metrics(epoch, y_true, y_pred, scaler, eval_type, elapsed_time, model_name, model_component="main"):
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)

    metrics = calculate_metrics(y_true, y_pred, elapsed_time, type=eval_type)
    metrics.update({
        "epoch": epoch + 1,
        "model": model_name,
        "model_component": model_component,
    })

    df = pd.DataFrame([metrics], columns=[
        "model",
        "model_component",
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
    ])
    df.to_csv(METRIC_LOG, mode="a", index=False, header=not os.path.exists(METRIC_LOG))


def log_eval_data(model_name, scaler, y_true, y_pred, component="main"):
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)

    entry = {
        "model_name": model_name,
        "component": component,
        "y_true": json.dumps(y_true.tolist(), sort_keys=True),
        "y_pred": json.dumps(y_pred.tolist(), sort_keys=True),
    }

    df = pd.DataFrame([entry], columns=["model_name", "component", "y_true", "y_pred"])
    df.to_csv(PRED_DATA, mode="a", index=False, header=not os.path.exists(PRED_DATA))


def create_logs_files():
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOSS_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "model", "component", "epoch", "start_epoch_time", "end_epoch_time", "train_loss", "val_loss"
        ])

    with open(METRIC_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "model", "model_component", "epoch", "type", "inference", "MAE", "MSE", "RMSE", "MAPE", "R2", "MDA", "Spearman"
        ])

    with open(TRIAL_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["model_name", "model_type", "trial", "params", "data"])

    with open(PRED_DATA, "w", newline="") as f:
        csv.writer(f).writerow(["model_name", "component", "y_true", "y_pred"])
