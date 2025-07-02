import torch
from tqdm import tqdm
import os
from src.train.globals import GLOBAL_PATIENCE, MIN_EPOCHS
import src.logs.utils as log
import time
from src.models.model import get_optimizer
from src.train.pipelines.residual import compute_residual_dataset
from src.train.utils import calculate_metrics

LEARNING_RATE = 0.0001


def freeze_all_except(model, component_names):
    # Freeze all parameters once
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze each requested component
    for component_name in component_names:
        for param in getattr(model, component_name).parameters():
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
    if model_type == "base_residual":
        base_model, res_model = model
        scaler, data_loaders, _, optimizer, criterion = model_fn(data_config, params)
        device = data_config["device"]

        residual_tl_pipeline(
            base_model,
            res_model,
            model_name,
            model_component,
            scaler,
            data_loaders,
            optimizer,
            criterion,
            epochs,
            device,
            freeze_base=True,
        )
    else:
        standard_tl_pipeline(
            model,
            model_name,
            model_component,
            param_names_to_tune,
            model_fn,
            data_config,
            params,
            epochs,
        )


def standard_tl_pipeline(
    model,
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

        val_loss /= len(val_loader)
        evl = time.time()

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model, model_name)
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
                torch.cat(train_y),
                torch.cat(train_pred),
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
                torch.cat(val_y),
                torch.cat(val_pred),
                evl - svl,
                "val",
            ),
            model_name,
            model_component,
        )

    # Load best model
    model = torch.load(model_name)

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

    log.log_evaluation_metrics(
        epoch,
        calculate_metrics(
            scaler,
            torch.cat(test_y),
            torch.cat(test_pred),
            ets - sts,
            "test",
        ),
        model_name,
        model_component,
    )

    # LOG data
    log.log_eval_data(model_name, scaler, torch.cat(test_y), torch.cat(test_pred), "tl")

    return model


def residual_tl_pipeline(
    base_model,
    res_model,
    model_name,
    model_component,
    scaler,
    data_loaders,
    optimizer,
    criterion,
    epochs,
    device,
    freeze_base=True,
):
    train_loader, val_loader, test_loader = data_loaders

    # Optionally freeze base model (standard TL)
    if freeze_base:
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(epochs):
        res_model.train()
        train_loss = 0
        train_pred, train_y = [], []

        str_time = time.time()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_base = base_model(x)
            residual_target = y - y_base

            optimizer.zero_grad()
            y_residual_pred = res_model(x, y_base)
            loss = criterion(y_residual_pred, residual_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Save predictions (for logging, not used for training)
            y_pred_final = y_base + y_residual_pred
            train_pred.append(y_pred_final.detach().cpu())
            train_y.append(y.detach().cpu())

        train_loss /= len(train_loader)
        etr = time.time()

        # Validation
        svl = time.time()
        res_model.eval()
        val_loss = 0
        val_pred, val_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_base = base_model(x)
                residual_target = y - y_base
                y_residual_pred = res_model(x, y_base)
                loss = criterion(y_residual_pred, residual_target)
                val_loss += loss.item()
                y_pred_final = y_base + y_residual_pred
                val_pred.append(y_pred_final.cpu())
                val_y.append(y.cpu())

        val_loss /= len(val_loader)
        evl = time.time()

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        )

        # Early stopping, logging, etc. as usual (add your own patience logic)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(res_model, model_name)
            print("✅ Model improved and was saved.")
        else:
            patience += 1
            print(f"⏳ No improvement. Patience: {patience}")
            # add patience logic if you want

        log.log_training_loss(
            epoch, train_loss, val_loss, str_time, evl, model_name, model_component
        )

        # Logging metrics (adapt if your logging API is different)
        log.log_evaluation_metrics(
            epoch,
            calculate_metrics(
                scaler,
                torch.cat(train_y),
                torch.cat(train_pred),
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
                torch.cat(val_y),
                torch.cat(val_pred),
                evl - svl,
                "val",
            ),
            model_name,
            model_component,
        )

    # Load best model
    res_model = torch.load(model_name)

    # Final test
    res_model.eval()
    test_loss = 0
    test_pred, test_y = [], []

    sts = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_base = base_model(x)
            y_residual_pred = res_model(x, y_base)
            y_pred_final = y_base + y_residual_pred
            loss = criterion(y_pred_final, y)
            test_loss += loss.item()
            test_pred.append(y_pred_final.cpu())
            test_y.append(y.cpu())

    test_loss /= len(test_loader)
    ets = time.time()
    print(f"🧪 Final Test Loss: {test_loss:.4f}")

    log.log_evaluation_metrics(
        epoch,
        calculate_metrics(
            scaler,
            torch.cat(test_y),
            torch.cat(test_pred),
            ets - sts,
            "test",
        ),
        model_name,
        model_component,
    )

    log.log_eval_data(
        model_name, scaler, torch.cat(test_y), torch.cat(test_pred), "residual_tl"
    )

    return res_model
