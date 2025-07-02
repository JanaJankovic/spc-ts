import src.train.main as train_pipeline
import src.models.model as model_handler
from src.train.utils import get_parameters
import src.logs.utils as log
import torch
import gc
import os

EPOCHS = 50
TRIALS = 25

DATA_DIR = "data/processed"


def lstm(data_config):
    train_pipeline.train_model(
        'lstm',
        model_fn=lambda config, p: model_handler.get_lstm(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("lstm"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=False
    )

    train_pipeline.train_model(
        'lstm',
        model_fn=lambda config, p: model_handler.get_lstm(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("lstm"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=True
    )


def cnn_lstm(data_config):
    train_pipeline.train_model(
        'cnn_lstm',
        model_fn=lambda config, p: model_handler.get_cnn_lstm(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("cnn_lstm"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=False
    )

    train_pipeline.train_model(
        'cnn_lstm',
        model_fn=lambda config, p: model_handler.get_cnn_lstm(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("cnn_lstm"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=True
    )



def base_residual(data_config):
    train_pipeline.train_model(
        'base_residual',
        model_fn=lambda config, p: model_handler.get_base_residual(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("base_residual"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=False
    )

    train_pipeline.train_model(
        'base_residual',
        model_fn=lambda config, p: model_handler.get_base_residual(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("base_residual"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=True
    )



def di_rnn(data_config):
    train_pipeline.train_model(
        'di_rnn',
        model_fn=lambda config, p: model_handler.get_di_rnn(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("di_rnn"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=False
    )

    train_pipeline.train_model(
        'di_rnn',
        model_fn=lambda config, p: model_handler.get_di_rnn(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("di_rnn"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=True
    )  


def cnn_di_rnn(data_config):
    train_pipeline.train_model(
        'cnn_di_rnn',
        model_fn=lambda config, p: model_handler.get_cnn_di_rnn(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("cnn_di_rnn"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=False
    )

    train_pipeline.train_model(
        'cnn_di_rnn',
        model_fn=lambda config, p: model_handler.get_cnn_di_rnn(config, p),
        data_config=data_config,
        param_sampler=lambda: get_parameters("cnn_di_rnn"),
        trials=TRIALS,
        epochs=EPOCHS,
        early_stopping=True
    )


if __name__ == "__main__":
    log.create_logs_files() 

    data_config = {
        'load_path': '',
        'm': 336,
        'n': 14,
        'lookback': 336,
        'horizon': 1,
        'target_col': 'load',
        'time_col': 'datetime',
        'freq': '1h',
        'use_calendar': True,
        'use_weather': True, 
        'device' : 'cuda'
    }

    mm_files = [f for f in os.listdir(DATA_DIR) if f.startswith("mm") and f.endswith(".csv")]
    
    data_config['freq'] = '1d'
    data_config['lookback'] = 14

    for file in mm_files:
        data_config['load_path'] = f'data/processed/{file}'

        lstm(data_config)
        torch.cuda.empty_cache(); gc.collect()

        cnn_lstm(data_config)
        torch.cuda.empty_cache(); gc.collect()

        base_residual(data_config)
        torch.cuda.empty_cache(); gc.collect()


    # 1 hour
    # for file in mm_files:
    #     data_config['load_path'] = f'data/processed/{file}'

    #     cnn_lstm(data_config)
    #     torch.cuda.empty_cache(); gc.collect()

    #     base_residual(data_config)
    #     torch.cuda.empty_cache(); gc.collect()

        # di_rnn(data_config)
        # torch.cuda.empty_cache(); gc.collect()

        # cnn_di_rnn(data_config)
        # torch.cuda.empty_cache(); gc.collect()

