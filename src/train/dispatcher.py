from src.train.globals import GlobalTracker
from src.train.pipelines.standard import standard_train_pipeline
from src.train.pipelines.modular import train_dirnn_pipeline


DI_RNN_tracker = {
    's_rnn': float('inf'),
    'p_rnn': float('inf'),
    'bpnn': float('inf'),
}

CNN_LSTM_tracker = {
    'cnn_lstm': float('inf'),
}

BASE_RESIDUAL_tracker = {
    'base': float('inf'),
    'residual': float('inf'),
}


def get_training_pipeline(model_type):
    if model_type == "cnn_lstm":
        return standard_train_pipeline, GlobalTracker(CNN_LSTM_tracker)
    elif model_type == "di_rnn" or model_type == "cnn_di_rnn":
        return train_dirnn_pipeline, GlobalTracker(DI_RNN_tracker)
