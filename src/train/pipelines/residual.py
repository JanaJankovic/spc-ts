import torch
from torch.utils.data import DataLoader, TensorDataset
import time
from src.logs.utils import log_training_loss, log_evaluation_metrics, log_eval_data
from src.train.utils import RMSELoss
from src.train.pipelines.standard import standard_train_pipeline
import os
from src.train.utils import calculate_aunl
from src.train.globals import GLOBAL_PATIENCE, MIN_EPOCHS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")

def compute_residual_dataset(model, data_loader, scaler, device):
    model.eval()
    all_inputs, all_residuals = [], []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred_inv = scaler["target"].inverse_transform(y_pred.cpu().numpy())
            y_true_inv = scaler["target"].inverse_transform(y.cpu().numpy())
            residuals = y_true_inv - y_pred_inv
            all_inputs.append(X.cpu())
            all_residuals.append(torch.tensor(residuals, dtype=torch.float32))

    return TensorDataset(torch.cat(all_inputs), torch.cat(all_residuals))


def train_residual_model(res_model, residual_dataset, optimizer, batch_size, epochs, device, model_name, val_loader, scaler, tracker, model_type="residual"):
    train_loader = DataLoader(residual_dataset, batch_size=batch_size)
    criterion = RMSELoss()
    early_stopping = True if tracker else False
    train_losses, val_losses = [], []
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()
        total_train_loss = 0

        # === Train ===
        y_true_train, y_pred_train = [], []
        start_train = time.time()
        res_model.train()
        for X, r in train_loader:
            X, r = X.to(device), r.to(device)
            optimizer.zero_grad()
            pred = res_model(X)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Save for logging
            y_pred_train.append(pred.detach().cpu())
            y_true_train.append(r.detach().cpu())

        end_train = time.time()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        y_pred_train = torch.cat(y_pred_train).numpy()
        y_true_train = torch.cat(y_true_train).numpy()

        # === Val ===
        res_model.eval()
        total_val_loss = 0
        y_true_val, y_pred_val = [], []
        start_val = time.time()
        with torch.no_grad():
            for i, (X_val, y_val) in enumerate(val_loader):
                X_val, y_val = X_val.to(device), y_val.to(device)

                base_pred = res_model.base_model(X_val)
                residual_pred = res_model(X_val)

                # Shape safety
                if residual_pred.dim() == 1:
                    residual_pred = residual_pred.unsqueeze(1)
                if base_pred.dim() == 1:
                    base_pred = base_pred.unsqueeze(1)

                combined = base_pred + residual_pred
                combined = combined.view(combined.size(0), -1)
                y_val = y_val.view(y_val.size(0), -1)

                y_pred_val.append(combined.cpu())
                y_true_val.append(y_val.cpu())

                val_loss = criterion(combined, y_val)
                total_val_loss += val_loss.item()

        end_val = time.time()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        y_pred_val = torch.cat(y_pred_val).numpy()
        y_true_val = torch.cat(y_true_val).numpy()
        end_time = time.time()

        print(f"🔁 Residual Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        log_training_loss(epoch, avg_train_loss, avg_val_loss, start_time, end_time, model_name, model_component="residual")

        log_evaluation_metrics(
            epoch,
            y_true_train.reshape(-1, 1),
            y_pred_train.reshape(-1, 1),
            scaler["target"],
            "train",
            end_train - start_train,
            model_name,
            model_component="residual",
        )

        log_evaluation_metrics(
            epoch,
            y_true_val.reshape(-1, 1),
            y_pred_val.reshape(-1, 1),
            scaler["target"],
            "val",
            end_val - start_val,
            model_name,
            model_component="residual",
        )


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
                    if epoch > MIN_EPOCHS:
                        print(
                        f"🛑 Early Stopping: No AUNL improvement after {GLOBAL_PATIENCE} epochs."
                        )
                        print(f"⏹️ Stopping training at epoch {epoch+1}.")
                        break


def test_evaluation(base_model, res_model, test_loader, scaler, model_name, epoch, device):
    base_model.eval()
    res_model.eval()
    all_preds, all_targets = [], []

    start_time = time.time()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            base_pred = base_model(X)
            res_pred = res_model(X)

            # Ensure both predictions are 2D: [batch_size, horizon]
            if base_pred.dim() == 1:
                base_pred = base_pred.unsqueeze(1)
            if res_pred.dim() == 1:
                res_pred = res_pred.unsqueeze(1)

            combined = base_pred + res_pred

            # Ensure target is also 2D
            if y.dim() == 1:
                y = y.unsqueeze(1)

            # Sanity check: all should have same shape
            if combined.shape != y.shape:
                print(f"Shape mismatch: combined={combined.shape}, y={y.shape}")
                continue  # skip bad batch

            all_preds.append(combined.cpu())
            all_targets.append(y.cpu())

    end_time = time.time()

    try:
        y_true = torch.cat(all_targets, dim=0).numpy().reshape(-1, 1)
        y_pred = torch.cat(all_preds, dim=0).numpy().reshape(-1, 1)
    except RuntimeError as e:
        print(f"[ERROR] Failed to concatenate predictions: {e}")
        print(f"Sample shapes:")
        for i, (yp, yt) in enumerate(zip(all_preds, all_targets)):
            print(f"  Batch {i}: pred={yp.shape}, target={yt.shape}")
        raise

    log_evaluation_metrics(epoch, y_true, y_pred, scaler["target"], "test", end_time - start_time, model_name, model_component="residual")
    log_eval_data(model_name, scaler["target"], y_true, y_pred, component="residual")


def train_residual_pipeline(model_name, model_type, model_fn, data_config, params, epochs, tracker, model_component="main"):
    print(f"🧠 Initializing base and residual models: {model_type}")
    device = data_config['device']
    
    scaler, (train_loader, val_loader, test_loader), (base_model, residual_model), (_, residual_optimizer), _ = model_fn(
        data_config, params
    )

    print("🚀 Training base model...")
    base_model = standard_train_pipeline(
        model_name=model_name,
        model_type='base',
        model_component="base",
        model_fn=model_fn,
        data_config=data_config,
        params=params,
        epochs=epochs,
        tracker=tracker
    )

    print("📉 Computing residuals from base predictions...")
    residual_dataset = compute_residual_dataset(base_model, train_loader, scaler, device)

    print("🧠 Training residual model...")
    residual_model.base_model = base_model  # ensure base model is accessible if needed
    train_residual_model(residual_model, residual_dataset, residual_optimizer, params['batch_size'], epochs, device, model_name, val_loader, scaler, tracker)

    print("🧪 Final test evaluation (base + residual):")
    test_evaluation(base_model, residual_model, test_loader, scaler, model_name, epoch=epochs - 1, device=device)

    torch.save(base_model, os.path.join(MODELS, f"base_{model_name}"))
    torch.save(residual_model, os.path.join(MODELS, f"res_{model_name}"))
    print(f"💾 Saved models from last epoch.")

    return base_model, residual_model
