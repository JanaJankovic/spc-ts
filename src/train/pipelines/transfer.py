import torch
from tqdm import tqdm
import os
from src.train.globals import GLOBAL_PATIENCE, MIN_EPOCHS
import src.logs.utils as log
import time
from src.models.model import get_optimizer, get_submodule_by_path
from src.train.pipelines.residual import compute_residual_dataset
from src.train.utils import calculate_metrics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")

LEARNING_RATE = 0.0001


def freeze_all_except(model, component_names):
    # Freeze all params
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze only the specified submodules
    for component_name in component_names:
        submodule = get_submodule_by_path(model, component_name)
        for param in submodule.parameters():
            param.requires_grad = True


def transfer_learning_pipeline(
    model,
    model_type,
    model_name,
    model_component,
    param_names_to_tune,
    model_fn,
    data_config,
    params,
    epochs,
):
    scaler, (train_loader, val_loader, test_loader), _, optimizer, criterion = model_fn(
        data_config, params
    )
    # Step 1: Freeze everything except desired parameters
    device = data_config["device"]
    freeze_all_except(model, param_names_to_tune)
    model = model.to(device)

    optimizer = get_optimizer(params["optimizer"], model.parameters(), LEARNING_RATE)

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_pred, train_y = [], []

        str_time = time.time()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_pred.append(output.detach().cpu())
            train_y.append(y.detach().cpu())

        train_loss /= len(train_loader)
        etr = time.time()

        y_train_pred_flat = torch.cat(train_pred, dim=0).reshape(-1, 1)
        y_train_true_flat = torch.cat(train_y, dim=0).reshape(-1, 1)

        # Validation
        svl = time.time()
        model.eval()
        val_loss = 0

        val_pred, val_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

                val_pred.append(output.detach().cpu())
                val_y.append(y.detach().cpu())

        y_val_pred_flat = torch.cat(val_pred, dim=0).reshape(-1, 1)
        y_val_true_flat = torch.cat(val_y, dim=0).reshape(-1, 1)

        val_loss /= len(val_loader)
        evl = time.time()

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model, os.path.join(MODELS, model_name))
            print("✅ Model improved and was saved.")
        else:
            patience += 1
            print(f"⏳ No improvement. Patience: {patience}/{GLOBAL_PATIENCE}")
            if epoch > MIN_EPOCHS:
                if patience >= GLOBAL_PATIENCE:
                    print("🛑 Early stopping triggered.")
                    break

        log.log_training_loss(
            epoch, train_loss, val_loss, str_time, evl, model_name, model_component
        )

        log.log_evaluation_metrics(
            epoch,
            calculate_metrics(
                scaler,
                y_train_true_flat.numpy(),
                y_train_pred_flat.numpy(),
                etr - str_time,
                "train",
            ),
            model_name,
            model_component,
        )

        log.log_evaluation_metrics(
            epoch,
            calculate_metrics(
                scaler,
                y_val_true_flat.numpy(),
                y_val_pred_flat.numpy(),
                evl - svl,
                "val",
            ),
            model_name,
            model_component,
        )

    # Load best model
    model = torch.load(os.path.join(MODELS, model_name))

    # Final test
    model.eval()
    test_loss = 0
    test_pred, test_y = [], []

    sts = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            test_loss += loss.item()
            test_pred.append(output.cpu())
            test_y.append(y.cpu())

    test_loss /= len(test_loader)
    ets = time.time()
    print(f"🧪 Final Test Loss: {test_loss:.4f}")

    y_test_pred_flat = torch.cat(test_pred, dim=0).reshape(-1, 1)
    y_test_true_flat = torch.cat(test_y, dim=0).reshape(-1, 1)

    log.log_evaluation_metrics(
        epoch,
        calculate_metrics(
            scaler,
            y_test_true_flat.numpy(),
            y_test_pred_flat.numpy(),
            ets - sts,
            "test",
        ),
        model_name,
        model_component,
    )

    # LOG data
    log.log_eval_data(
        model_name, scaler, y_test_true_flat.numpy(), y_test_pred_flat.numpy(), "tl"
    )

    return model
