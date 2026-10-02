import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import binary_classification_metrics, multiclass_classification_metrics, find_optimal_threshold
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss (Lin et al., 2017).
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    α  — up-weights the minority (positive) class.
         Rule of thumb: α ≈ 1 - (pos_count / total). For 39/290 ≈ 0.85.
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

    def _select_criterion(self):
        if self.args.num_classes == 2:
            if getattr(self.args, 'use_focal', False):
                alpha = getattr(self.args, 'focal_alpha', 0.75)
                gamma = getattr(self.args, 'focal_gamma', 2.0)
                print(f"Using FocalLoss (alpha={alpha}, gamma={gamma})")
                return FocalLoss(alpha=alpha, gamma=gamma)
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def forward_classification(self, batch_x):
        outputs = self.model(batch_x)
        if outputs.dim() == 3:
            outputs = outputs.mean(dim=1)
        return outputs

    # ─────────────────────────────────────────────────────────────────────────
    # ROCKET: precompute fixed features once, then only train the linear head
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _precompute_rocket_features(self, loader) -> tuple:
        """
        Extract frozen ROCKET features for an entire split in one pass.
        Returns (features, labels) as CPU tensors — only needs to run once.
        """
        self.model.eval()
        all_feats, all_labels = [], []
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(self.device)
            # Normalise layout and handle NaN — mirrors ROCKET.forward pre-processing
            if batch_x.shape[1] != self.model.in_channels:
                batch_x = batch_x.permute(0, 2, 1)
            batch_x = torch.nan_to_num(batch_x, nan=0.0)
            feats = self.model._extract_features(batch_x)   # (B, 2*num_kernels)
            all_feats.append(feats.cpu())
            all_labels.append(batch_y.cpu())
        return torch.cat(all_feats), torch.cat(all_labels)

    def _train_rocket(self, setting: str):
        """
        ROCKET-specific training path.
        Features are precomputed once; only the linear head is trained each epoch.
        This reduces per-epoch cost from O(kernels × T × N) to O(features × N).
        """
        train_data, train_loader = self._get_data('train')
        val_data,   val_loader   = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # ── Precompute (the expensive step — runs only once) ─────────────────
        print("Precomputing ROCKET features (runs once)...")
        train_feats, train_labels = self._precompute_rocket_features(train_loader)
        val_feats,   val_labels   = self._precompute_rocket_features(val_loader)
        print(f"  Train features: {train_feats.shape} | "
              f"Val features: {val_feats.shape}")

        # ── Only optimise the linear classifier head ─────────────────────────
        optimizer     = optim.Adam(self.model.classifier.parameters(),
                                   lr=self.args.learning_rate)
        criterion     = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Mini-batch loader over precomputed features
        feat_loader = DataLoader(
            TensorDataset(train_feats, train_labels),
            batch_size=self.args.batch_size, shuffle=True,
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []

            for feats, labels in feat_loader:
                feats  = feats.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model.classifier(feats)       # (B, 1)
                if self.args.num_classes == 2:
                    loss = criterion(logits.squeeze(-1), labels.float())
                else:
                    loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validate on precomputed val features ─────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model.classifier(val_feats.to(self.device))
                if self.args.num_classes == 2:
                    val_loss  = criterion(val_logits.squeeze(-1),
                                          val_labels.float().to(self.device)).item()
                    val_probs = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()
                else:
                    val_loss  = criterion(val_logits,
                                          val_labels.to(self.device)).item()
                    val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()

            if np.isnan(val_probs).any() or np.isinf(val_probs).any():
                print("WARNING: val predictions contain NaN/Inf!")

            val_trues = val_labels.numpy()
            if self.args.num_classes == 2:
                val_metric = binary_classification_metrics(val_trues, val_probs, threshold=0.5)
            else:
                val_metric = multiclass_classification_metrics(val_trues, val_probs, average="macro")

            print(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss {np.mean(train_losses):.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"AUROC {val_metric['auroc']:.4f} | "
                f"F1 {val_metric['f1']:.4f}"
            )

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        best_ckpt = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        return self.model

    # ─────────────────────────────────────────────────────────────────────────
    # Train  (dispatches to _train_rocket for ROCKET)
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, setting: str):
        if self.args.model == 'ROCKET':
            return self._train_rocket(setting)

        train_data, train_loader = self._get_data('train')
        val_data,   val_loader   = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer      = self._select_optimizer()
        criterion      = self._select_criterion()
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

            val_loss, val_metric, _, _ = self.validate(val_loader, criterion)

            print(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss {np.mean(train_losses):.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"AUROC {val_metric['auroc']:.4f} | "
                f"F1 {val_metric['f1']:.4f}"
            )

            early_stopping(val_loss, self.model, path)
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
        test_loss, test_metric, _, _ = self.validate(test_loader, criterion, threshold=threshold)

        print(
            f"[Test]  Loss {test_loss:.4f} | Threshold {threshold:.3f} | "
            + " | ".join(f"{k.upper()} {v:.4f}" for k, v in test_metric.items())
        )
        return test_metric

    # ─────────────────────────────────────────────────────────────────────────
    # Validate  (shared by train loop, test, and ROCKET's test path)
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