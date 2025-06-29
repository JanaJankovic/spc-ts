import os
import json
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
JSON_PATH = os.path.join(SRC_DIR, "hpo.json")


def get_parameters(model_name):
    with open(JSON_PATH, "r") as f:
        space = json.load(f)

    if model_name not in space:
        raise ValueError(f"❌ Model '{model_name}' not found in HPO search space JSON.")

    model_space = space[model_name]
    parameters = {}

    # Sample standard hyperparameters
    for key in model_space:
        if key not in ["cnn_size", "cnn_channel", "neurons"]:
            parameters[key] = random.choice(model_space[key])

    # Handle CNN channels
    if model_name == "base_residual" and "cnn_channel" in model_space:
        cnn_size = random.choice(model_space.get("cnn_size", [1]))
        parameters["cnn_channels"] = random.choices(
            model_space["cnn_channel"], k=cnn_size
        )

    elif model_name in ["cnn_di_rnn", "cnn_lstm"] and "cnn_channel" in model_space:
        # Ensure cnn_channels is always a list
        parameters["cnn_channels"] = [random.choice(model_space["cnn_channel"])]

    # Handle neurons and per-model assignments
    if "neurons" in model_space:
        neuron_val = random.choice(model_space["neurons"])

        if model_name == "base_residual":
            parameters["rnn_hidden"] = neuron_val
            parameters["mlp_hidden"] = neuron_val
            parameters["fusion_hidden"] = neuron_val
            parameters["dropout_rnn"] = random.choice(model_space["dropout"])
            parameters["dropout_fc"] = random.choice(model_space["dropout"])
            parameters["dropout_cnn"] = random.choice(model_space["dropout"])

        elif model_name == "di_rnn":
            parameters["hidden_size"] = neuron_val
            parameters["bp_hidden_size"] = neuron_val
            parameters["dropout_rnn"] = random.choice(model_space["dropout"])

        elif model_name == "cnn_di_rnn":
            parameters["hidden_size"] = neuron_val
            parameters["bp_hidden_size"] = neuron_val
            parameters["dropout_rnn"] = random.choice(model_space["dropout"])
            parameters["dropout_cnn"] = random.choice(model_space["dropout"])

        elif model_name == "cnn_lstm":
            parameters["lstm_hidden_size"] = neuron_val
            parameters["dense_size"] = neuron_val
            parameters["dropout_cnn"] = random.choice(model_space["dropout"])
            parameters["dropout_fc"] = random.choice(model_space["dropout"])

    return parameters


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
