from src.models.cnn_di_rnn import CNN_DIRNN
from src.models.di_rnn import DIRNN
from src.models.base_res import BasePredictor, ResidualCNN
from src.models.cnn_lstm import CNNLSTMModel
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


def get_cnn_lstm(data_config, parameters):
    batch_size = parameters["batch_size"]

    scalers, data, input_shape = data_pipeline.cnn_lstm_pipeline(
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

    optimizer = get_optimizer(
        parameters["optimizer"], model.parameters(), parameters["learning_rate"]
    )
    criterion = nn.L1Loss()

    return scalers, data, model, optimizer, criterion


def get_base_residual(data_config, parameters):
    batch_size = parameters["batch_size"]

    scalers, data, input_shape = data_pipeline.base_residual_pipeline(
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

    residual_model = ResidualCNN(
        temporal_dim=input_shape[-1],
        cnn_channels=parameters["cnn_channels"],
        kernel_size=parameters["kernel_size"],
        output_dim=data_config["horizon"],
        dropout=parameters["dropout_cnn"],
    )

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
        scalers,
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

    optimizers = {
        "s_rnn": get_optimizer(parameters["optimizer"], model.s_rnn.parameters(), parameters["lr_rnn"]),
        "p_rnn": get_optimizer(parameters["optimizer"], model.p_rnn.parameters(), parameters["lr_rnn"]),
        "bpnn": get_optimizer(parameters["optimizer"], model.bpnn.parameters(), parameters["lr_bpnn"]),
    }

    criterion = nn.L1Loss()

    return scaler, data, model, optimizers, criterion


def get_cnn_di_rnn(data_config, parameters):
    scaler, data = data_pipeline.di_rnn_pipeline(
        load_path=data_config["load_path"],
        m=data_config["m"],
        n=data_config["n"],
        horizon=data_config["horizon"],
        target_col=data_config["target_col"],
    )

    x_seq, x_per, _ = data[0]  # Get one batch of training data
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

    optimizers = {
        "s_rnn": get_optimizer(parameters["optimizer"], model.s_rnn.parameters(), parameters["lr_rnn"]),
        "p_rnn": get_optimizer(parameters["optimizer"], model.p_rnn.parameters(), parameters["lr_rnn"]),
        "bpnn": get_optimizer(parameters["optimizer"], model.bpnn.parameters(), parameters["lr_bpnn"]),
    }

    criterion = nn.L1Loss()

    return scaler, data, model, optimizers, criterion
