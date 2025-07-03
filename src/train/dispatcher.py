from src.train.globals import GlobalTracker
from src.train.pipelines.standard import standard_train_pipeline
from src.train.pipelines.modular import train_dirnn_pipeline
from src.train.pipelines.residual import train_residual_pipeline
from src.train.pipelines.transfer import transfer_learning_pipeline
from src.train.pipelines.universal import universal_train_pipeline

# Store best AUNL and best metric (use None or inf as default)
DI_RNN_tracker = {
    "s_rnn": {"aunl": float("inf"), "metric": float("inf")},
    "p_rnn": {"aunl": float("inf"), "metric": float("inf")},
    "bpnn": {"aunl": float("inf"), "metric": float("inf")},
}

CNN_LSTM_tracker = {
    "cnn_lstm": {"aunl": float("inf"), "metric": float("inf")},
}

UNIVERSAL_tracker = {
    "universal": {"aunl": float("inf"), "metric": float("inf")},
}

LSTM_tracker = {
    "lstm": {"aunl": float("inf"), "metric": float("inf")},
}

BASE_RESIDUAL_tracker = {
    "base": {"aunl": float("inf"), "metric": float("inf")},
    "residual": {"aunl": float("inf"), "metric": float("inf")},
}


def get_training_pipeline(model_type):
    if model_type == "lstm":
        return standard_train_pipeline, GlobalTracker(LSTM_tracker)
    if model_type == "cnn_lstm":
        return standard_train_pipeline, GlobalTracker(CNN_LSTM_tracker)
    elif model_type == "di_rnn" or model_type == "cnn_di_rnn":
        return train_dirnn_pipeline, GlobalTracker(DI_RNN_tracker)
    elif model_type == "base_residual":
        return train_residual_pipeline, GlobalTracker(BASE_RESIDUAL_tracker)
    elif model_type == "universal":
        return universal_train_pipeline, GlobalTracker(UNIVERSAL_tracker)
    elif model_type == "transfer_learning":
        return transfer_learning_pipeline, None
    elif model_type == "linear_regression":
        return standard_train_pipeline, None
