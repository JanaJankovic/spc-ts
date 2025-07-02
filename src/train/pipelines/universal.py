import torch
import os
import time
import numpy as np
from src.train.utils import calculate_avg_metrics, calculate_aunl
from src.train.globals import TRACKING_METRIC, MIN_EPOCHS, GlobalTracker
import src.logs.utils as log


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")


def forward_batch(model, batch, device):
    x, y, consumer_id = batch
    x, y = x.to(device), y.to(device)
    y_pred = model(x)  # <-- just x!
    return y_pred, y, consumer_id


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_targets, all_ids = [], [], []
    for batch in train_loader:
        preds, targets, consumer_ids = forward_batch(model, batch, device)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_ids.append(consumer_ids.detach().cpu())
    out_ids = torch.cat(all_ids)
    return (
        total_loss / len(train_loader.dataset),
        torch.cat(all_preds),
        torch.cat(all_targets),
        out_ids,
    )


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            preds, targets, consumer_ids = forward_batch(model, batch, device)
            loss = criterion(preds, targets)
            total_loss += loss.item() * targets.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.append(consumer_ids.cpu())
    out_ids = torch.cat(all_ids)
    return (
        total_loss / len(loader.dataset),
        torch.cat(all_preds),
        torch.cat(all_targets),
        out_ids,
    )


def universal_train_pipeline(
    model_name,
    model_type,
    model_component,
    model_fn,
    data_config,
    params,
    epochs,
    tracker: GlobalTracker = None,
    universal_model=None,
):
    device = data_config["device"]
    early_stopping = tracker is not None

    scalers, (train_loader, val_loader, test_loader), model, optimizer, criterion = (
        model_fn(data_config, params)
    )

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        strt = time.time()
        train_loss, train_preds, train_targets, train_ids = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        etr = time.time()

        svl = time.time()
        val_loss, val_preds, val_targets, val_ids = evaluate_model(
            model, val_loader, criterion, device
        )
        evl = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"📈 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        log.log_training_loss(
            epoch,
            train_losses[-1],
            val_losses[-1],
            strt,
            evl,
            model_name,
            model_component,
        )

        # --- Logging averaged metrics ---
        log.log_evaluation_metrics(
            epoch,
            calculate_avg_metrics(
                train_targets, train_preds, train_ids, scalers, etr - strt, "train"
            ),
            model_name,
        )
        val_metrics = calculate_avg_metrics(
            val_targets, val_preds, val_ids, scalers, evl - svl, "val"
        )
        log.log_evaluation_metrics(
            epoch,
            val_metrics,
            model_name,
        )

        # === Early stopping ===
        if early_stopping:
            metric = val_metrics[TRACKING_METRIC]
            _, aunl = calculate_aunl(train_losses, val_losses)

            if epoch > MIN_EPOCHS:
                if aunl > tracker.get_score(model_type, "aunl"):
                    print(
                        f"⚠️ No improvement in AUNL ({aunl}/{tracker.get_score(model_type, 'aunl')})"
                    )
                    print("🛑 Early stopping.")
                    break

            if metric < tracker.get_score(model_type, "metric"):
                tracker.update_aunl(model_type, aunl)
                tracker.update_metric(model_type, metric)

    # --- Final test evaluation ---
    sts = time.time()
    _, test_preds, test_targets, test_ids = evaluate_model(
        model, test_loader, criterion, device
    )
    ets = time.time()
    log.log_evaluation_metrics(
        epoch,
        calculate_avg_metrics(
            test_targets, test_preds, test_ids, scalers, ets - sts, "test"
        ),
        model_name,
    )

    torch.save(model, os.path.join(MODELS, model_name))
    return model
