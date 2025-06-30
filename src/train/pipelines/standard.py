import torch
import os
import time
import src.logs.utils as log
from src.train.utils import calculate_aunl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")

GLOBAL_PATIENCE = 10

def forward_batch(model, batch, device):
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


def standard_train_pipeline(
    model_name, model_type, model_component, model_fn, data_config, params, epochs, device, tracker=None,
):
    early_stopping = True if tracker else False
    global GLOBAL_PATIENCE

    scaler, (train_loader, val_loader, test_loader), model, optimizer, criterion = model_fn(
        data_config, params
    )
    model = model if not isinstance(model, tuple) else model[0]
    optimizer = optimizer if not isinstance(optimizer, tuple) else optimizer[0]

    target_scaler = scaler["target"] if isinstance(scaler, dict) else scaler
    model.to(device)

    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(epochs):
        
        str = time.time()
        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        etr = time.time()

        svl = time.time()
        val_loss, val_preds, val_targets = evaluate_model(
            model, val_loader, criterion, device
        )
        evl = time.time()


        train_losses.append(train_loss)
        val_losses.append(val_loss)


        print(
            f"📈 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # === Logging ===
        log.log_training_loss(
            epoch, 
            train_loss, 
            val_loss, 
            str, 
            evl, 
            model_name, 
            model_component
        )

        log.log_evaluation_metrics(
            epoch,
            train_targets.numpy(),
            train_preds.numpy(),
            target_scaler,
            "train",
            etr - str,
            model_name,
            model_component
        )

        log.log_evaluation_metrics(
            epoch,
            val_targets.numpy(),
            val_preds.numpy(),
            target_scaler,
            "val",
            evl - svl,
            model_name,
            model_component
        )

        
        # === Early Stopping based on AUNL ===
        if early_stopping:
            _, aunl_val = calculate_aunl(train_losses, val_losses)
            updated = tracker.update(model_type, aunl_val)

            if updated:
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"⚠️ AUNL {aunl_val:.4f} > best {tracker.get_score(model_type):.4f} ({patience_counter}/{GLOBAL_PATIENCE})"
                )

                if patience_counter >= GLOBAL_PATIENCE:
                    print(
                        f"🛑 Early Stopping: No AUNL improvement after {GLOBAL_PATIENCE} epochs."
                    )
                    break

    sts = time.time()
    _, test_preds, test_targets = evaluate_model(
        model, test_loader, criterion, device
    )
    ets = time.time()
    
    log.log_evaluation_metrics(
        epoch,
        test_targets.numpy(),
        test_preds.numpy(),
        target_scaler,
        "test",
        ets - sts,
        model_name,
        model_component
    )

    log.log_eval_data(
        model_name, 
        target_scaler, 
        test_targets.numpy(), 
        test_preds.numpy(),
        model_component, 
    )
    
    if model_component != "base":
        torch.save(model, os.path.join(MODELS, model_name))
        print(f"💾 Saved model from last epoch as {model_name}")

    return model
