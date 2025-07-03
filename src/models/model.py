from src.models.cnn_di_rnn import CNN_DIRNN
from src.models.di_rnn import DIRNN
from src.models.lstm_mlp import LSTM_MLP
from src.models.cnn_lstm import CNNLSTMModel
from src.models.lstm import SimpleLSTM
from src.models.residual_lstm import ResidualLSTM
from src.models.linear import LinearRegression
import src.data.pipeline as data_pipeline
import torch.nn as nn
import torch
import numpy as np


def get_optimizer(name, model_params, lr=1e-3, **kwargs):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model_params, lr=lr, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, **kwargs)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model_params, lr=lr, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, **kwargs)
    elif name == "adagrad":
        return torch.optim.Adagrad(model_params, lr=lr, **kwargs)
    elif name == "nadam":
        return torch.optim.NAdam(model_params, lr=lr, **kwargs)
    else:
        raise ValueError(f"❌ Unknown optimizer: '{name}'")


def get_lstm(data_config, parameters):
    batch_size = parameters["batch_size"]

    scaler, data, input_shape = data_pipeline.lstm_pipeline(
        data_config["load_path"],
        data_config["lookback"],
        data_config["horizon"],
        batch_size,
        data_config["time_col"],
        data_config["target_col"],
        data_config["freq"],
    )

    model = SimpleLSTM(
        input_shape=input_shape,
        output_size=data_config["horizon"],
        lstm_hidden_size=parameters["lstm_hidden_size"],
        lstm_layers=parameters["lstm_layers"],
        dense_size=parameters["dense_size"],
        dropout_lstm=parameters["dropout_lstm"],
        dropout_fc=parameters["dropout_fc"],
    )
    model.to(data_config["device"])

    optimizer = get_optimizer(
        parameters["optimizer"], model.parameters(), parameters["learning_rate"]
    )
    criterion = nn.L1Loss()

    return scaler, data, model, optimizer, criterion


def get_cnn_lstm(data_config, parameters):
    batch_size = parameters["batch_size"]

    scaler, data, input_shape = data_pipeline.cnn_lstm_pipeline(
        data_config["load_path"],
        data_config["lookback"],
        data_config["horizon"],
        batch_size,
        data_config["time_col"],
        data_config["target_col"],
        data_config["freq"],
        data_config["use_calendar"],
        data_config["use_weather"],
    )

    model = CNNLSTMModel(
        input_shape=input_shape,
        output_size=data_config["horizon"],
        conv_channels=parameters["cnn_channels"],
        kernel_size=parameters["kernel_size"],
        use_maxpool=parameters["use_maxpool"],
        lstm_hidden_size=parameters["lstm_hidden_size"],
        lstm_layers=parameters["lstm_layers"],
        dense_size=parameters["dense_size"],
        dropout_conv=parameters["dropout_cnn"],
        dropout_fc=parameters["dropout_fc"],
    )
    model.to(data_config["device"])

    optimizer = get_optimizer(
        parameters["optimizer"], model.parameters(), parameters["learning_rate"]
    )
    criterion = nn.L1Loss()

    return scaler, data, model, optimizer, criterion


def get_lstm_mlp(data_config, parameters):
    batch_size = parameters["batch_size"]

    scaler, data, input_shape = data_pipeline.base_residual_pipeline(
        load_path=data_config["load_path"],
        lookback=data_config["lookback"],
        horizon=data_config["horizon"],
        batch=batch_size,
        time_col=data_config["time_col"],
        target_col=data_config["target_col"],
        freq=data_config["freq"],
        use_calendar=data_config["use_calendar"],
        use_weather=data_config["use_weather"],
    )

    lstm_mlp = LSTM_MLP(
        temporal_dim=input_shape[-1],
        static_dim=0,
        rnn_hidden=parameters["rnn_hidden"],
        mlp_hidden=parameters["mlp_hidden"],
        fusion_hidden=parameters["fusion_hidden"],
        output_dim=data_config["horizon"],
        use_lstm=parameters["use_lstm"],
        dropout_rnn=parameters["dropout_rnn"],
        dropout_fusion=parameters["dropout_fc"],
    )

    lstm_mlp.to(data_config["device"])

    optimizer = get_optimizer(
        parameters["optimizer"], lstm_mlp.parameters(), parameters["learning_rate"]
    )

    criterion = nn.L1Loss()

    return (
        scaler,
        data,
        lstm_mlp,
        optimizer,
        criterion,
    )


def get_residual(input_shape, data_config, parameters):
    residual_model = ResidualLSTM(
        input_dim=input_shape,
        hidden_dim=parameters["hidden_dim"],
        num_layers=parameters["residual_layers"],
        output_dim=data_config["horizon"],
        dropout=parameters["dropout_cnn"],
    )
    residual_model.to(data_config["device"])

    optimizer = get_optimizer(
        parameters["optimizer"],
        residual_model.parameters(),
        parameters["learning_rate"],
    )
    return residual_model, optimizer


def get_di_rnn(data_config, parameters):
    scaler, data = data_pipeline.di_rnn_pipeline(
        load_path=data_config["load_path"],
        m=data_config["m"],
        n=data_config["n"],
        horizon=data_config["horizon"],
        batch_size=parameters["batch_size"],
        target_col=data_config["target_col"],
    )

    model = DIRNN(
        seq_input_size=1,
        per_input_size=1,
        hidden_size=parameters["hidden_size"],
        bp_hidden_size=parameters["bp_hidden_size"],
        dropout=parameters["dropout_rnn"],
        horizon=data_config["horizon"],
    )
    model.to(data_config["device"])

    criterion = nn.L1Loss()

    return scaler, data, model, criterion


def get_cnn_di_rnn(data_config, parameters):
    scaler, data = data_pipeline.di_rnn_pipeline(
        load_path=data_config["load_path"],
        m=data_config["m"],
        n=data_config["n"],
        horizon=data_config["horizon"],
        batch_size=parameters["batch_size"],
        target_col=data_config["target_col"],
    )

    train_loader = data[0]
    x_seq, x_per, _ = next(iter(train_loader))  # Get one batch
    seq_input_size = x_seq.shape[-1]
    per_input_size = x_per.shape[-1]

    model = CNN_DIRNN(
        seq_input_size=seq_input_size,
        per_input_size=per_input_size,
        hidden_size=parameters["hidden_size"],
        bp_hidden_size=parameters["bp_hidden_size"],
        dropout=parameters["dropout_rnn"],
        horizon=data_config["horizon"],
        cnn_out_channels=parameters["cnn_channels"],
        kernel_size=parameters["kernel_size"],
    )
    model.to(data_config["device"])

    criterion = nn.L1Loss()

    return scaler, data, model, criterion


def get_universal(data_config, parameters):
    batch_size = parameters["batch_size"]

    scalers, data, input_shape = data_pipeline.uni_model_pipeline(
        data_config["load_path"],
        data_config["lookback"],
        data_config["horizon"],
        batch_size,
        data_config["time_col"],
        data_config["target_col"],
        data_config["freq"],
    )
    print(input_shape)

    model = CNNLSTMModel(
        input_shape=input_shape,
        output_size=data_config["horizon"],
        conv_channels=parameters["cnn_channels"],
        kernel_size=parameters["kernel_size"],
        use_maxpool=parameters["use_maxpool"],
        lstm_hidden_size=parameters["lstm_hidden_size"],
        lstm_layers=parameters["lstm_layers"],
        dense_size=parameters["dense_size"],
        dropout_conv=parameters["dropout_cnn"],
        dropout_fc=parameters["dropout_fc"],
    )
    model.to(data_config["device"])

    optimizer = get_optimizer(
        parameters["optimizer"], model.parameters(), parameters["learning_rate"]
    )
    criterion = nn.L1Loss()

    return scalers, data, model, optimizer, criterion


def get_linear_regression(data_config, parameters):
    batch_size = parameters["batch_size"]

    scaler, data, input_shape = data_pipeline.lstm_pipeline(
        data_config["load_path"],
        data_config["lookback"],
        data_config["horizon"],
        batch_size,
        data_config["time_col"],
        data_config["target_col"],
        data_config["freq"],
    )

    model = LinearRegression(
        input_dim=np.prod(input_shape), output_dim=data_config["horizon"]
    )
    model.to(data_config["device"])

    optimizer = get_optimizer(
        parameters["optimizer"], model.parameters(), parameters["learning_rate"]
    )
    criterion = nn.L1Loss()

    return scaler, data, model, optimizer, criterion


def find_submodules_by_type(model, target_type, prefix=""):
    matches = []
    for name, module in model.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(module, target_type):
            matches.append(path)
        # If it's a container, search deeper
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for i, submod in enumerate(module):
                subpath = f"{path}.{i}"
                if isinstance(submod, target_type):
                    matches.append(subpath)
                # You could recurse deeper here if you allow nested containers
    return matches


def get_submodule_by_path(model, path):
    """
    Resolves 'convs.0' to model.convs[0] etc.
    """
    parts = path.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def get_model_component_names(model, model_type, component):
    if component == "dense":
        # Look for any nn.Linear not inside CNN/ModuleList, or standardize as 'fc2'
        # Or simply return ['fc2'] if standard.
        return ["fc2"]
    elif component == "cnn":
        # Find all Conv1d layers dynamically
        return find_submodules_by_type(model, nn.Conv1d)
    elif component == "dense_cnn":
        # Both
        return find_submodules_by_type(model, nn.Conv1d) + ["fc2"]
    else:
        raise ValueError(f"Unknown component '{component}'")


def get_model_fn(model_type):
    if model_type == "cnn_lstm":
        return get_cnn_lstm
    elif model_type == "lstm":
        return get_lstm
    elif model_type == "di_rnn":
        return get_di_rnn
    elif model_type == "cnn_di_rnn":
        return get_cnn_di_rnn
    elif model_type == "base_residual":
        return get_lstm_mlp
    elif model_type == "universal":
        return get_universal
    elif model_type == "linear_regression":
        return get_linear_regression
    else:
        raise ValueError(f"❌ Unknown model type: '{model_type}'")
