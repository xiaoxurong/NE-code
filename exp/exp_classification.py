import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import binary_classification_metrics, multiclass_classification_metrics, find_optimal_threshold
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al., 2017).
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    α  — up-weights the minority (positive) class.
         Rule of thumb: α ≈ 1 - (pos_count / total).
    γ  — focusing exponent. γ=0 → standard BCE. γ=2 is the paper default.
         Higher γ pushes harder on the hard examples.

    Args (from args):
        args.focal_alpha  (float, default 0.75)
        args.focal_gamma  (float, default 2.0)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: both (B,) — same shape as BCEWithLogitsLoss expects
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t     = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        wd = getattr(self.args, 'weight_decay', 0.0)
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd)

    def _select_criterion(self, train_labels=None):
        """
        Binary: focal loss if use_focal, otherwise BCE-with-logits. With pos_weight='auto'
        the positive class is weighted by n_negative / n_positive of THIS fold's training
        labels, so the weighting follows the prevalence of whatever label is being used.
        A number sets the weight directly; 'none' (default) leaves BCE unweighted.
        """
        if self.args.num_classes == 2:
            if getattr(self.args, 'use_focal', False):
                alpha = getattr(self.args, 'focal_alpha', 0.75)
                gamma = getattr(self.args, 'focal_gamma', 2.0)
                print(f"Using FocalLoss (alpha={alpha}, gamma={gamma})")
                return FocalLoss(alpha=alpha, gamma=gamma)
            pw = str(getattr(self.args, 'pos_weight', 'none')).lower()
            if pw == 'auto':
                if train_labels is None:
                    return nn.BCEWithLogitsLoss()  # loss value only (validation/test reporting)
                y = np.asarray(train_labels).reshape(-1)
                n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
                weight = n_neg / max(n_pos, 1)
                print(f"Using BCEWithLogitsLoss with pos_weight={weight:.3f} "
                      f"({n_neg} negative / {n_pos} positive training windows)")
                return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=self.device))
            if pw not in ('none', ''):
                return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw), device=self.device))
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _early_stop_value(self, val_loss, val_metric):
        """Quantity early stopping minimises: validation loss, or minus validation AUROC."""
        if getattr(self.args, 'early_stop_metric', 'loss') == 'auroc':
            auroc = val_metric.get('auroc', np.nan)
            return -auroc if np.isfinite(auroc) else np.inf
        return val_loss

    @staticmethod
    def _spread_note(probs) -> str:
        sd = float(np.std(probs))
        return f" | pred sd {sd:.4f}" + (" (predictions nearly constant)" if sd < 1e-3 else "")

    def forward_classification(self, batch_x):
        outputs = self.model(batch_x)
        if outputs.dim() == 3:
            outputs = outputs.mean(dim=1)
        return outputs

    # ─────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, setting: str):
        train_data, train_loader = self._get_data('train')
        val_data,   val_loader   = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer      = self._select_optimizer()
        criterion      = self._select_criterion(train_data.y.numpy())
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.forward_classification(batch_x)

                if self.args.num_classes == 2:
                    loss = criterion(logits.squeeze(-1), batch_y.float())
                else:
                    loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_loss, val_metric, val_probs, _ = self.validate(val_loader, criterion)

            print(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss {np.mean(train_losses):.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"AUROC {val_metric['auroc']:.4f} | "
                f"F1 {val_metric['f1']:.4f}"
                + self._spread_note(val_probs)
            )

            early_stopping(self._early_stop_value(val_loss, val_metric), self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        best_ckpt = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        _, _, val_probs, val_trues = self.validate(val_loader, criterion)
        self.optimal_threshold = find_optimal_threshold(val_trues, val_probs)
        print(f"Optimal threshold (val): {self.optimal_threshold:.3f}")
        return self.model

    # ─────────────────────────────────────────────────────────────────────────
    # Test
    # ─────────────────────────────────────────────────────────────────────────

    def test(self, setting: str) -> dict:
        test_data, test_loader = self._get_data('test')

        ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        else:
            print(f"Warning: checkpoint not found at {ckpt_path}, using current weights.")

        criterion = self._select_criterion()
        threshold = getattr(self, 'optimal_threshold', 0.5)   # ← use val-tuned threshold
        test_loss, test_metric, test_probs, test_trues = self.validate(
            test_loader, criterion, threshold=threshold)
        # kept so run.py can save out-of-fold predictions (order matches test_idx)
        self.test_probs, self.test_trues = test_probs, test_trues

        print(
            f"[Test]  Loss {test_loss:.4f} | Threshold {threshold:.3f} | "
            + " | ".join(f"{k.upper()} {v:.4f}" for k, v in test_metric.items())
        )
        return test_metric

    # ─────────────────────────────────────────────────────────────────────────
    # Validate  (shared by the train loop and test)
    # ─────────────────────────────────────────────────────────────────────────

    def validate(self, loader, criterion, threshold=0.5):
        self.model.eval()
        losses, preds, trues = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.forward_classification(batch_x)

                if self.args.num_classes == 2:
                    logits = logits.squeeze(-1)
                    loss   = criterion(logits, batch_y.float())
                    prob   = torch.sigmoid(logits).cpu().numpy()
                else:
                    loss = criterion(logits, batch_y)
                    prob = torch.softmax(logits, dim=1).cpu().numpy()

                losses.append(loss.item())
                preds.append(prob)
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        if np.isnan(preds).any() or np.isinf(preds).any():
            print(f"WARNING: predictions contain NaN/Inf!  "
                  f"min={np.nanmin(preds):.4f}  max={np.nanmax(preds):.4f}")

        if self.args.num_classes == 2:
            metric = binary_classification_metrics(trues, preds, threshold=threshold)
        else:
            metric = multiclass_classification_metrics(trues, preds, average="macro")

        return np.mean(losses), metric, preds, trues