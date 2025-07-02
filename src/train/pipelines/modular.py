import torch
import time
from src.logs.utils import log_training_loss, log_evaluation_metrics, log_eval_data
from src.train.utils import calculate_aunl, calculate_metrics
from src.train.utils import RMSELoss
import os
from src.train.globals import TRACKING_METRIC, MIN_EPOCHS
from src.models.model import get_optimizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODELS = os.path.join(PROJECT_ROOT, "models")


class ComponentTrainer:
    def __init__(
        self,
        model,
        model_name,
        model_type,
        component_name,
        evaluation_component,
        scaler,
        params,
        device,
        patience=5,
        min_epochs=10,
        tracker=None,
    ):
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.component_name = component_name
        self.evaluation_component = evaluation_component
        self.scaler = scaler
        self.optimizer_name = params["optimizer"]
        self.lr = params["lr_bpnn"] if component_name == "bpnn" else params["lr_rnn"]
        self.patience = patience
        self.min_epochs = min_epochs
        self.tracker = tracker
        self.device = device
        self.early_stopping = True if tracker else False

        self.criterion = RMSELoss()
        self.train_losses = []
        self.val_losses = []
        self.patience_counter = 0

        self._freeze_all_except()

    def _freeze_all_except(self):
        for param in self.model.parameters():
            param.requires_grad = False
        # Recursively unfreeze all parameters in the selected component
        for param in getattr(self.model, self.component_name).parameters():
            param.requires_grad = True

    def _run_forward(self, x_seq, x_per):
        if self.model_type == "di_rnn":
            if self.component_name == "s_rnn":
                return self.model.s_rnn(x_seq)
            elif self.component_name == "p_rnn":
                return self.model.p_rnn(x_per)
            else:
                return self.model(x_seq, x_per)
        elif self.model_type == "cnn_di_rnn":
            if self.component_name == "s_rnn":
                # apply CNN preprocessing before passing to s_rnn
                x_seq = x_seq.permute(0, 2, 1)
                x_seq = torch.relu(self.model.cnn_seq(x_seq))
                x_seq = x_seq.permute(0, 2, 1)
                return self.model.s_rnn(x_seq)

            elif self.component_name == "p_rnn":
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

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_seq_val, x_per_val, y_val in data_loader:
                x_seq_val = x_seq_val.to(self.device)
                x_per_val = x_per_val.to(self.device)
                y_val = y_val.to(self.device)

                preds = self._run_forward(x_seq_val, x_per_val)
                loss = self.criterion(preds, y_val)

                batch_size = y_val.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_preds.append(preds.cpu())
                all_targets.append(y_val.cpu())

        avg_loss = total_loss / total_samples
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        return avg_loss, all_preds, all_targets

    def train(self, train_loader, val_loader, test_loader, epochs):
        self.optimizer = get_optimizer(
            self.optimizer_name,
            filter(
                lambda p: p.requires_grad,
                getattr(self.model, self.component_name).parameters(),
            ),
            self.lr,
        )

        patience_counter = 0
        print(
            f"🧪 Early stopping for [{self.component_name}]: {'✅ Enabled' if self.early_stopping else '❌ Disabled'}"
        )

        for epoch in range(epochs):
            # === TRAIN ===
            train_start = time.time()
            self.model.train()

            train_loss_sum = 0.0
            y_pred_train_list = []
            y_true_train_list = []

            for x_seq, x_per, y_true in train_loader:
                x_seq = x_seq.to(self.device)
                x_per = x_per.to(self.device)
                y_true = y_true.to(self.device)

                train_loss, y_pred_batch, y_true_batch = self.train_one_epoch(
                    x_seq, x_per, y_true
                )
                train_loss_sum += train_loss * y_true.size(
                    0
                )  # weighted sum for average loss

                y_pred_train_list.append(y_pred_batch)
                y_true_train_list.append(y_true_batch)

            avg_train_loss = train_loss_sum / len(train_loader.dataset)
            y_pred_tr = torch.cat(y_pred_train_list)
            y_true_tr = torch.cat(y_true_train_list)
            train_end = time.time()

            # === VALIDATE ===
            val_start = time.time()
            val_loss, y_pred_val, y_true_val = self.evaluate(val_loader)
            val_end = time.time()

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)

            print(
                f"\r📘 [{self.component_name}] Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}",
                end="\r",
            )

            # Log training + validation loss with proper durations
            log_training_loss(
                epoch,
                avg_train_loss,
                val_loss,
                train_start,
                train_end,
                self.model_name,
                self.component_name,
            )

            if self.component_name == self.evaluation_component:
                # Log evaluation metrics on train and val sets
                log_evaluation_metrics(
                    epoch,
                    calculate_metrics(
                        self.scaler,
                        y_true_tr,
                        y_pred_tr,
                        train_end - train_start,
                        "train",
                    ),
                    self.model_name,
                )
                log_evaluation_metrics(
                    epoch,
                    calculate_metrics(
                        self.scaler, y_true_val, y_pred_val, val_end - val_start, "val"
                    ),
                    self.model_name,
                )

            if self.early_stopping:
                metrics = calculate_metrics(self.scaler, y_true_val, y_pred_val, 0, "")
                metric = metrics[TRACKING_METRIC]
                _, aunl = calculate_aunl(self.train_losses, self.val_losses)

                if epoch > MIN_EPOCHS:
                    if aunl > self.tracker.get_score(self.component_name, "aunl"):
                        print(
                            f"⚠️ No improvement in AUNL ({aunl}/{self.tracker.get_score(self.component_name, 'aunl')})"
                        )
                        print("🛑 Early stopping.")
                        break

                if metric < self.tracker.get_score(self.component_name, "metric"):
                    self.tracker.update_aunl(self.component_name, aunl)
                    self.tracker.update_metric(self.component_name, metric)

        # Test evaluation after training
        if self.component_name == self.evaluation_component:
            test_start = time.time()
            _, y_pred_test, y_true_test = self.evaluate(test_loader)
            test_end = time.time()

            log_evaluation_metrics(
                epoch,
                calculate_metrics(
                    self.scaler, y_true_test, y_pred_test, test_end - test_start, "test"
                ),
                self.model_name,
            )
            log_eval_data(self.model_name, self.scaler, y_true_test, y_pred_test)


def train_dirnn_pipeline(
    model_name,
    model_type,
    model_fn,
    data_config,
    params,
    epochs,
    tracker=None,
    model_component="main",
    universal_model=None,
):
    scaler, (train_loader, val_loader, test_loader), model, criterion = model_fn(
        data_config, params
    )

    for component in ["s_rnn", "p_rnn", "bpnn"]:
        trainer = ComponentTrainer(
            model=model,
            model_name=model_name,
            model_type=model_type,
            component_name=component,
            evaluation_component="bpnn",
            scaler=scaler,
            tracker=tracker,
            device=data_config["device"],
            params=params,
        )
        trainer.train(train_loader, val_loader, test_loader, epochs)

    torch.save(model, os.path.join(MODELS, model_name))
    print(f"💾 Saved model from last epoch as {model_name}")

    return model
