import torch
from torch.utils.data import DataLoader, TensorDataset
import time
from src.logs.utils import log_training_loss, log_evaluation_metrics, log_eval_data
from src.train.utils import RMSELoss
from src.train.pipelines.standard import standard_train_pipeline
import os
from src.train.utils import calculate_aunl
from src.train.globals import GLOBAL_PATIENCE, MIN_EPOCHS
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")

def compute_residual_dataset(model, data_loader, device):
    model.eval()
    all_inputs, all_residuals = [], []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                X, _, y_scaled = batch  # (X, y_smooth, y_true)
            else:
                X, y_scaled = batch     # (X, y_smooth)

            X, y_scaled = X.to(device), y_scaled.to(device)
            y_pred = model(X)
            residuals = y_scaled - y_pred

            all_inputs.append(X.cpu())
            all_residuals.append(residuals.cpu())
    
    return TensorDataset(torch.cat(all_inputs), torch.cat(all_residuals))


def train_residual_model(res_model, residual_dataset, optimizer, batch_size, epochs, device, model_name, val_loader, scaler, tracker):
    train_loader = DataLoader(residual_dataset, batch_size=batch_size)
    criterion = RMSELoss()
    patience_counter = 0
    early_stopping = tracker is not None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        res_model.train()
        y_true_train, y_pred_train = [], []
        train_loss = 0

        start_train = time.time()
        for X, r in train_loader:
            X, r = X.to(device), r.to(device)
            optimizer.zero_grad()
            pred = res_model(X)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_pred_train.append(pred.detach().cpu())
            y_true_train.append(r.detach().cpu())

        train_losses.append(train_loss / len(train_loader))
        end_train = time.time()

        # === Validation ===
        res_model.eval()
        y_pred_val, y_true_val = [], []
        val_loss = 0
        start_val = time.time()

        with torch.no_grad():
            for X, _, y_true in val_loader:
                X, y_true = X.to(device), y_true.to(device)

                base = res_model.base_model(X)
                res = res_model(X)
                combined = base + res

                print("Batch output shape:", combined.shape, "Target shape:", y_true.shape)

                val_loss += criterion(combined, y_true).item()
                y_pred_val.append(combined.cpu())
                y_true_val.append(y_true.cpu())

        val_losses.append(val_loss / len(val_loader))
        end_val = time.time()

        def tensors_to_numpy(tensor_list):
            # Convert list of tensors to one numpy array (N, output_dim)
            return np.concatenate([t.numpy() for t in tensor_list], axis=0)

        # Convert training predictions and targets
        y_true_train_np = tensors_to_numpy(y_true_train)
        y_pred_train_np = tensors_to_numpy(y_pred_train)

        # Convert validation predictions and targets
        y_true_val_np = tensors_to_numpy(y_true_val)
        y_pred_val_np = tensors_to_numpy(y_pred_val)

        # Then pass these numpy arrays instead of lists:
        log_evaluation_metrics(epoch, y_true_train_np, y_pred_train_np, scaler, "train", end_train - start_train, model_name, "residual")
        log_evaluation_metrics(epoch, y_true_val_np, y_pred_val_np, scaler, "val", end_val - start_val, model_name, "residual")



        # === Early stopping ===
        if early_stopping:
            _, aunl = calculate_aunl(train_losses, val_losses)
            if tracker.update("residual", aunl):
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⚠️ No improvement in AUNL ({patience_counter}/{GLOBAL_PATIENCE})")
                if patience_counter >= GLOBAL_PATIENCE and epoch > MIN_EPOCHS:
                    print("🛑 Early stopping.")
                    break



def test_evaluation(base_model, res_model, test_loader, scaler, model_name, epoch, device):
    base_model.eval()
    res_model.eval()
    y_pred, y_true = [], []

    start_test = time.time()
    with torch.no_grad():
        for X, _, y in test_loader:
            X = X.to(device)
            combined = base_model(X) + res_model(X)
            y_pred.append(combined.cpu())
            y_true.append(y.cpu())
    end_test = time.time()

    y_pred = torch.cat(y_pred).numpy().reshape(-1, 1)
    y_true = torch.cat(y_true).numpy().reshape(-1, 1)
    
    log_evaluation_metrics(epoch, y_true, y_pred, scaler, "test", end_test-start_test, model_name, "residual")
    log_eval_data(model_name, scaler, y_true, y_pred, component="residual")



def train_residual_pipeline(model_name, model_type, model_component, model_fn, data_config, params, epochs, tracker=None):
    print("⚙️ Setting up models...")
    device = data_config["device"]
    scaler, (train_loader, val_loader, test_loader), (base_model, residual_model), (_, optimizer), _ = model_fn(data_config, params)

    print("🚀 Training base model...")
    base_model = standard_train_pipeline(
        model_name=model_name,
        model_type="base",
        model_component="base",
        model_fn=model_fn,
        data_config=data_config,
        params=params,
        epochs=epochs,
        tracker=tracker,
    )

    print("📉 Computing residual dataset...")
    residual_dataset = compute_residual_dataset(base_model, train_loader, device)

    print("🧠 Training residual model...")
    residual_model.base_model = base_model
    train_residual_model(residual_model, residual_dataset, optimizer, params["batch_size"], epochs, device, model_name, val_loader, scaler, tracker)

    print("🧪 Testing combined model...")
    test_evaluation(base_model, residual_model, test_loader, scaler, model_name, epoch=epochs - 1, device=device)

    torch.save(base_model, os.path.join(MODELS, f"base_{model_name}"))
    torch.save(residual_model, os.path.join(MODELS, f"res_{model_name}"))
    print("💾 Models saved.")

    return base_model, residual_model

