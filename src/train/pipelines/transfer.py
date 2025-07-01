import torch
from tqdm import tqdm
import os
from src.train.globals import GLOBAL_PATIENCE, MIN_EPOCHS
import src.logs.utils as log
import time

def freeze_all_except(model, parameter_names_to_tune):
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(target) for target in parameter_names_to_tune)

def run_transfer_learning(
    model_name, model_type, model_component, param_names_to_tune, model_fn, data_config, params, epochs
):
    scaler, (train_loader, val_loader, test_loader), model, optimizer, criterion = model_fn(
        data_config, params
    )
    # Step 1: Freeze everything except desired parameters
    device = data_config['device']
    freeze_all_except(model, param_names_to_tune)
    model = model.to(device)

    # Step 3: Training loop
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_pred, train_y = [], []
        
        str = time.time()
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

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

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
            torch.cat(train_y),
            torch.cat(train_pred),
            scaler,
            "train",
            etr - str,
            model_name,
            model_component
        )

        log.log_evaluation_metrics(
            epoch,
            torch.cat(val_y),
            torch.cat(val_pred),
            scaler,
            "val",
            evl - svl,
            model_name,
            model_component
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
            test_loss += loss.item()
            test_pred.append(output.cpu())
            test_y.append(y.cpu())

    test_loss /= len(test_loader)
    ets = time.time()
    print(f"🧪 Final Test Loss: {test_loss:.4f}")


    log.log_evaluation_metrics(
        epoch,
        torch.cat(test_y),
        torch.cat(test_pred),
        scaler,
        "test",
        ets - sts,
        model_name,
        model_component
    )

    return model
