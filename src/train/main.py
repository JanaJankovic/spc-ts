from src.train.dispatcher import get_training_pipeline
from src.logs.utils import log_trial_info
from datetime import datetime
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")


def train_model(
    model_type,
    model_fn,
    data_config,
    param_sampler,
    trials=1,
    epochs=50,
    early_stopping=False,
    device='cpu'
):
    os.makedirs(MODELS, exist_ok=True)

    pipeline, tracker = get_training_pipeline(model_type)
    tracker = tracker if early_stopping else None

    for trial in range(trials):
        print(f"\n🔁 Trial {trial+1}/{trials}")
        params = param_sampler()

        date_str = datetime.now().strftime("%d%m%Y")
        model_name = f"{date_str}_t{trial+1}_{model_type}.pt"

        log_trial_info(model_name, model_type, trial, params)

        model = pipeline(
            model_name=model_name,
            model_type=model_type,
            model_fn=model_fn,
            data_config=data_config,
            params=params,
            device=device,
            epochs=epochs,
            tracker=tracker
        )

        torch.save(model, os.path.join(MODELS, model_name))
        print(f"💾 Saved model from last epoch as {model_name}")

    print(f"\n🏁 All trials complete.")
    return model
