import src.train.main as train_pipeline
import src.models.model as model_handler
from src.train.utils import get_parameters
import src.logs.utils as log
from src.train.dispatcher import get_training_pipeline
from src.models.model import get_model_fn, get_model_component_names
from src.data.pipeline import cnn_lstm_pipeline
from src.train.pipelines.standard import evaluate_model
from src.train.utils import calculate_metrics
import os
import subprocess
import torch.nn as nn
import pandas as pd
import torch
import json
import os
import time
import re

EPOCHS = 50
TRIALS = 25

DATA_DIR = "data/processed"


def extract_model_type(model_name):
    match = re.search(r"_t\d+_(.+)\.pt$", model_name)
    if match:
        return match.group(1)
    return None


def get_params_from_trial_info(
    csv_path, model_name, params_column="params", name_column="model_name"
):
    df = pd.read_csv(csv_path)
    row = df[df[name_column] == model_name]
    if row.empty:
        raise ValueError(f"Model name '{model_name}' not found in {csv_path}")
    params_str = row.iloc[0][params_column]
    try:
        params_dict = json.loads(params_str)
    except Exception as e:
        print(f"Failed to parse params for {model_name}: {params_str}")
        raise e
    return params_dict


def models_per_file(data_config):
    file_names = [
        f
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f)) and f.startswith("mm")
    ]

    for file in file_names:
        data_config["load_path"] = os.path.join(DATA_DIR, file)

        # train_pipeline.train_model(
        #     "linear_regression",
        #     model_fn=lambda config, p: model_handler.get_linear_regression(config, p),
        #     data_config=data_config,
        #     param_sampler=lambda: get_parameters("linear_regression"),
        #     trials=TRIALS,
        #     epochs=EPOCHS,
        #     early_stopping=False,
        # )

        train_pipeline.train_model(
            "cnn_lstm",
            model_fn=lambda config, p: model_handler.get_cnn_lstm(config, p),
            data_config=data_config,
            param_sampler=lambda: get_parameters("cnn_lstm"),
            trials=TRIALS,
            epochs=EPOCHS,
            early_stopping=False,
        )


def models_once(data_config):
    data_config["load_path"] = DATA_DIR

    train_pipeline.train_model(
        "universal",
        model_fn=lambda config, p: model_handler.get_universal(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("cnn_lstm"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=False,
    )

    train_pipeline.train_model(
        "universal",
        model_fn=lambda config, p: model_handler.get_universal(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("cnn_lstm"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=True,
    )


def evaluate_model_afer_training(model, model_name, data_config, params):
    file_names = [
        f
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f)) and f.startswith("mm")
    ]

    for file_name in file_names:
        data_config["load_path"] = os.path.join(DATA_DIR, file_name)
        scaler, data, _ = cnn_lstm_pipeline(
            data_config["load_path"],
            data_config["lookback"],
            data_config["horizon"],
            params["batch_size"],
            data_config["time_col"],
            data_config["target_col"],
            data_config["freq"],
            data_config["use_calendar"],
            data_config["use_weather"],
        )

        _, _, test_loader = data

        start = time.time()
        _, y_pred, y_true = evaluate_model(model, test_loader, nn.L1Loss(), "cuda")
        end = time.time()

        metrics = calculate_metrics(
            scaler, y_true, y_pred, end - start, "test_per_dataset"
        )
        log.log_evaluation_metrics(EPOCHS, metrics, model_name, file_name)
        log.log_eval_data(model_name, scaler, y_true, y_pred, file_name)


def evaluate_per_file():
    df_trials = pd.read_csv("logs/trial_info.csv")
    df_universal = df_trials[
        df_trials["model_name"].str.contains("universal", case=False, na=False)
    ]
    model_names = df_universal["model_name"].values

    for model_name in model_names:
        data_config = get_params_from_trial_info(
            "logs/trial_info.csv", model_name, "data", "model_name"
        )
        params = get_params_from_trial_info(
            "logs/trial_info.csv", model_name, "params", "model_name"
        )

        model = torch.load(f"models/{model_name}")
        evaluate_model_afer_training(model, model_name, data_config, params)


def tl_model(model, model_name, data_config, params):
    file_names = [
        f
        for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f)) and f.startswith("mm")
    ]
    components = ["lstm", "dense_lstm"]

    for component in components:
        for file_name in file_names:
            f = os.path.basename(file_name)
            name_target = os.path.splitext(f)[0]

            tl_model_name = f"{model_name[:-3]}_t_{name_target}.pt"
            pipeline, _ = get_training_pipeline("transfer_learning")
            data_config["transfer_learning"] = f"Universal to target({name_target})"
            data_config["load_path"] = os.path.join(DATA_DIR, file_name)
            data_config["transfer_learning_component"] = component
            model_type = extract_model_type(model_name)
            model_fn = get_model_fn("cnn_lstm")
            c_name = get_model_component_names(model, model_type, component)

            pipeline(
                model=model,
                model_type="cnn_lstm",
                model_name=tl_model_name,
                model_component=name_target,
                model_fn=model_fn,
                param_names_to_tune=c_name,
                data_config=data_config,
                params=params,
                epochs=EPOCHS,
            )


def tl_per_file():
    df_trials = pd.read_csv("logs/trial_info.csv")
    # Filter: contains 'universal', does NOT contain 'mm'
    df_universal = df_trials[
        df_trials["model_name"].str.contains("universal", case=False, na=False)
        & ~df_trials["model_name"].str.contains("mm", case=False, na=False)
    ]
    model_names = df_universal["model_name"].values
    for model_name in model_names:
        data_config = get_params_from_trial_info(
            "logs/trial_info.csv", model_name, "data", "model_name"
        )
        params = get_params_from_trial_info(
            "logs/trial_info.csv", model_name, "params", "model_name"
        )
        model = torch.load(f"models/{model_name}")

        tl_model(model, model_name, data_config, params)


def git_commit_and_push(message="Automated commit after successful run"):
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ Git commit & push completed.")
    except subprocess.CalledProcessError as e:
        print("❌ Git operation failed:", e)


if __name__ == "__main__":
    # log.create_logs_files()

    horizons = [1, 3, 7]

    data_config = {
        "load_path": "",
        "lookback": 14,
        "horizon": 1,
        "target_col": "load",
        "time_col": "datetime",
        "freq": "1d",
        "device": "cuda",
        "use_calendar": False,
        "use_weather": False,
    }

    for horizon in horizons:
        data_config["horizon"] = horizon
        # Comparison models
        models_per_file(data_config)

        # CNN-LSTM train & eval
        # models_once(data_config)

    # evaluate_per_file()

    # # # TL per file
    tl_per_file()

    # git_commit_and_push("add: final results")
