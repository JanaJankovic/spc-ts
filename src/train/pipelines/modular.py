import torch
import time
from src.logs.utils import log_training_loss, log_evaluation_metrics, log_eval_data
from src.train.utils import calculate_aunl
from src.train.utils import RMSELoss
import os
from src.train.globals import GLOBAL_PATIENCE, MIN_EPOCHS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")



class ComponentTrainer:
    def __init__(self, model, model_name, model_type, component_name, evaluation_component,
                 scaler, optimizer, patience=5, min_epochs=10, tracker=None):
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.component_name = component_name
        self.evaluation_component = evaluation_component
        self.scaler = scaler
        self.optimizer = optimizer
        self.patience = patience
        self.min_epochs = min_epochs
        self.tracker = tracker
        self.early_stopping = True if tracker else False

        self.criterion = RMSELoss()
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0

        self._freeze_all_except()

    def _freeze_all_except(self):
        for param in self.model.parameters():
            param.requires_grad = False
        getattr(self.model, self.component_name).requires_grad_(True)


    def _run_forward(self, x_seq, x_per):
        if self.model_type == 'di_rnn':
            if self.component_name == 's_rnn':
                return self.model.s_rnn(x_seq)
            elif self.component_name == 'p_rnn':
                return self.model.p_rnn(x_per)
            else:
                return self.model(x_seq, x_per)
        elif self.model_type == 'cnn_di_rnn':
            if self.component_name == 's_rnn':
                # apply CNN preprocessing before passing to s_rnn
                x_seq = x_seq.permute(0, 2, 1)
                x_seq = torch.relu(self.model.cnn_seq(x_seq))
                x_seq = x_seq.permute(0, 2, 1)
                return self.model.s_rnn(x_seq)
            
            elif self.component_name == 'p_rnn':
                x_per = x_per.permute(0, 2, 1)
                x_per = torch.relu(self.model.cnn_per(x_per))
                x_per = x_per.permute(0, 2, 1)
                return self.model.p_rnn(x_per)
            
            else:
                return self.model(x_seq, x_per)

    def train_one_epoch(self, x_seq, x_per, y_true):
        self.model.train()
        self.optimizer.zero_grad()

        pred = self._run_forward(x_seq, x_per)
        loss = self.criterion(pred, y_true)
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred.detach().cpu(), y_true.detach().cpu()

    def evaluate(self, val_data):
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            x_seq_val, x_per_val, y_val = val_data
            preds = self._run_forward(x_seq_val, x_per_val)
            loss = self.criterion(preds, y_val)

            all_preds.append(preds.cpu())
            all_targets.append(y_val.cpu())

        return loss.item(), torch.cat(all_preds), torch.cat(all_targets)

    def train(self, x_seq, x_per, y_true, val_data, epochs):
        patience_counter = 0
        print(f"🧪 Early stopping for [{self.component_name}]: {'✅ Enabled' if self.early_stopping else '❌ Disabled'}")

        for epoch in range(epochs):
            # === TRAIN ===
            train_start = time.time()
            train_loss, y_pred_tr, y_true_tr = self.train_one_epoch(x_seq, x_per, y_true)
            train_end = time.time()

            # === VALIDATE ===
            val_start = time.time()
            val_loss, y_pred_val, y_true_val = self.evaluate(val_data)
            val_end = time.time()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"\r📘 [{self.component_name}] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", end="\r")

            # Log training + validation loss with proper durations
            log_training_loss(
                epoch, train_loss, val_loss,
                train_start, train_end,  # correct train duration
                self.model_name, self.component_name
            )

            if self.component_name == self.evaluation_component:
            # === Optional: log metrics for both sets ===
                log_evaluation_metrics(epoch, y_true_tr, y_pred_tr, self.scaler, "train", train_end - train_start, self.model_name)
                log_evaluation_metrics(epoch, y_true_val, y_pred_val, self.scaler, "val", val_end - val_start, self.model_name)

            # === AUNL-based early stopping ===
            if self.early_stopping:
                _, aunl_val = calculate_aunl(self.train_losses, self.val_losses)
                updated = self.tracker.update(self.component_name, aunl_val)

                if updated:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(
                        f"⚠️ AUNL {aunl_val:.4f} > best {self.tracker.get_score(self.component_name):.4f} "
                        f"({patience_counter}/{GLOBAL_PATIENCE})"
                    )
                    if patience_counter >= GLOBAL_PATIENCE:
                        if epoch > MIN_EPOCHS:
                            print(
                                f"🛑 Early Stopping: No AUNL improvement after {GLOBAL_PATIENCE} epochs."
                            )
                            print(f"⏹️ Stopping training at epoch {epoch+1}.")
                            break

        
        if self.component_name == self.evaluation_component:
            test_start = time.time()
            _, y_pred_test, y_true_test = self.evaluate(val_data)
            test_end = time.time()

            log_evaluation_metrics(epoch, y_true_test, y_pred_test, self.scaler, "test", test_end - test_start, self.model_name)
            log_eval_data(self.model_name, self.scaler, y_true_test, y_pred_test)



def train_dirnn_pipeline(model_name, model_type, model_fn, data_config, params, epochs, tracker=None,  model_component="main"):
    scaler, (train_data, val_data, _), model, optimizers, _ = model_fn(data_config, params)

    device = data_config['device']

    X_seq_train, X_per_train, y_train = [torch.tensor(x, dtype=torch.float32).to(device) for x in train_data]
    X_seq_val, X_per_val, y_val = [torch.tensor(x, dtype=torch.float32).to(device) for x in val_data]
    val_tensors = (X_seq_val, X_per_val, y_val)

    for component in ['s_rnn', 'p_rnn', 'bpnn']:
        trainer = ComponentTrainer(
            model=model,
            model_name=model_name,
            model_type=model_type,
            component_name=component,
            evaluation_component='bpnn',
            scaler=scaler,
            optimizer=optimizers[component],  # ✅ pass correct optimizer
            tracker=tracker
        )
        trainer.train(X_seq_train, X_per_train, y_train, val_tensors, epochs)

    torch.save(model, os.path.join(MODELS, model_name))
    print(f"💾 Saved model from last epoch as {model_name}")

    return model
