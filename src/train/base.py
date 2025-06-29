import torch
import os
from datetime import datetime
import time
import src.logs.utils as log
from src.train.utils import calculate_aunl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")

GLOBAL_BEST_AUNL = float("inf")
GLOBAL_PATIENCE = 5


def forward_batch(model, batch, device):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x_seq, x_per, y = batch
        x_seq, x_per, y = x_seq.to(device), x_per.to(device), y.to(device)
        y_pred = model(x_seq, x_per)
    else:
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

    return y_pred, y


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for i, batch in enumerate(train_loader):
        preds, targets = forward_batch(model, batch, device)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

        print(
            f"🟦 [Train] Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}",
            end="\r",
        )

    print()  # For newline after loop
    return (
        total_loss / len(train_loader.dataset),
        torch.cat(all_preds),
        torch.cat(all_targets),
    )


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            preds, targets = forward_batch(model, batch, device)
            loss = criterion(preds, targets)

            total_loss += loss.item() * targets.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            print(
                f"🟨 [Eval ] Batch {i + 1}/{len(val_loader)} - Loss: {loss.item():.4f}",
                end="\r",
            )

    print()  # Newline
    return (
        total_loss / len(val_loader.dataset),
        torch.cat(all_preds),
        torch.cat(all_targets),
    )


def train_trial(
    model_name, model_fn, data_config, params, epochs, device, early_stopping=False
):
    global GLOBAL_BEST_AUNL, GLOBAL_PATIENCE

    scaler, (train_loader, val_loader, _), model, optimizer, criterion = model_fn(
        data_config, params
    )
    target_scaler = scaler["target"] if isinstance(scaler, dict) else scaler
    model.to(device)

    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_preds, val_targets = evaluate_model(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        end_time = time.time()

        print(
            f"📈 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # === Logging ===
        log.log_training_loss(
            epoch, train_loss, val_loss, start_time, end_time, model_name
        )

        log.log_evaluation_metrics(
            epoch,
            train_targets.numpy(),
            train_preds.numpy(),
            target_scaler,
            "train",
            end_time - start_time,
            model_name,
        )

        log.log_evaluation_metrics(
            log.METRIC_LOG,
            epoch,
            val_targets.numpy(),
            val_preds.numpy(),
            target_scaler,
            "val",
            end_time - start_time,
            log.evaluate_metrics,
            model_name,
        )

        # === Early Stopping based on AUNL ===
        if early_stopping:
            _, aunl_val = calculate_aunl(train_losses, val_losses)

            if aunl_val < GLOBAL_BEST_AUNL:
                GLOBAL_BEST_AUNL = aunl_val
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"⚠️ AUNL {aunl_val:.4f} > best {GLOBAL_BEST_AUNL:.4f} ({patience_counter}/{GLOBAL_PATIENCE})"
                )

                if patience_counter >= GLOBAL_PATIENCE:
                    print(
                        f"🛑 Early Stopping: No AUNL improvement after {GLOBAL_PATIENCE} epochs."
                    )
                    break

    return model


def train_model(
    model_type,
    model_fn,
    data_config,
    param_sampler,
    trials=1,
    epochs=50,
    device=None,
):
    os.makedirs(MODELS, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    for trial in range(trials):
        print(f"\n🔁 Trial {trial+1}/{trials}")
        params = param_sampler()

        # Save model name early
        date_str = datetime.now().strftime("%d%m%Y")
        model_name = f"{date_str}_t{trial+1}_{model_type}.pt"

        # Log trial info
        log.log_trial_info(model_name, model_type, trial, params)

        model = train_trial(model_name, model_fn, data_config, params, epochs, device)

        torch.save(model, os.path.join(MODELS, model_name))
        print(f"💾 Saved model from last epoch as {model_name}")

    print(f"\n🏁 All trials complete.")
    return model
