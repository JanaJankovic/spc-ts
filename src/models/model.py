from src.models.cnn_di_rnn import CNN_DIRNN
from src.models.di_rnn import DIRNN
from src.models.base_res import BasePredictor, ResidualRNN
from src.models.cnn_lstm import CNNLSTMModel
from src.models.lstm import SimpleLSTM
import src.data.pipeline as data_pipeline
import torch.nn as nn
import torch


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


def get_base_residual(data_config, parameters):
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

    base_model = BasePredictor(
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

    residual_model = ResidualRNN(
        input_dim=input_shape[-1],
        hidden_dim=parameters["hidden_dim"],
        num_layers=parameters["residual_layers"],
        output_dim=data_config["horizon"],
        dropout=parameters["dropout_cnn"],
    )
    base_model.to(data_config["device"])
    residual_model.to(data_config["device"])

    base_optimizer = get_optimizer(
        parameters["optimizer"], base_model.parameters(), parameters["learning_rate"]
    )
    residual_optimizer = get_optimizer(
        parameters["optimizer"],
        residual_model.parameters(),
        parameters["learning_rate"],
    )
    criterion = nn.L1Loss()

    return (
        scaler,
        data,
        (base_model, residual_model),
        (base_optimizer, residual_optimizer),
        criterion,
    )


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
        return get_base_residual
    elif model_type == "universal":
        return get_universal
    else:
        raise ValueError(f"❌ Unknown model type: '{model_type}'")


def get_model_component_names(model_type, component):
    if model_type == "lstm":
        if component in ["dense", "dense_cnn"]:
            return ["fc2"]
        else:
            raise ValueError(
                f"Unknown component '{component}' for model_type '{model_type}'"
            )
    elif model_type == "di_rnn":
        if component in ["dense", "dense_cnn"]:
            return ["bpnn.fc2"]
        else:
            raise ValueError(
                f"Unknown component '{component}' for model_type '{model_type}'"
            )
    elif model_type == "base_residual":
        if component in ["dense", "dense_cnn"]:
            return ["fc"]
        else:
            raise ValueError(
                f"Unknown component '{component}' for model_type '{model_type}'"
            )
    elif model_type == "cnn_lstm":
        if component == "dense":
            return ["fc2"]
        elif component == "cnn":
            return [f"convs.{i}" for i in range(3)]
        elif component == "dense_cnn":
            return [f"convs.{i}" for i in range(3)] + ["fc2"]
        else:
            raise ValueError(
                f"Unknown component '{component}' for model_type '{model_type}'"
            )
    elif model_type == "cnn_di_rnn":
        if component == "dense":
            return ["bpnn.fc2"]
        elif component == "cnn":
            return ["cnn_seq", "cnn_per"]
        elif component == "dense_cnn":
            return ["cnn_seq", "cnn_per", "bpnn.fc2"]
        else:
            raise ValueError(
                f"Unknown component '{component}' for model_type '{model_type}'"
            )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
