# -*- coding: utf-8 -*-
"""
Run Using this 

python GRU_D/gru_d.py --data-dir Data_group_7 --window-hours 3

"""

from __future__ import annotations

import os
import sys
import copy
import random
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve
)

from data_loader_grud import DataLoader
from unified_preprocessor_grud import UnifiedPreprocessor


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class FoldResult:
    fold: int
    auroc: float
    auprc: float
    f1: float
    sensitivity: float
    specificity: float
    accuracy: float
    brier: float

Number = Union[int, float]

def class_counts(y_train: np.ndarray) -> Tuple[float, float]:
    pos = max(float((y_train == 1).sum()), 1.0)
    neg = max(float((y_train == 0).sum()), 1.0)
    return pos, neg


class BinaryFocalLoss(nn.Module):
    """
    Weighted binary focal loss.
    alpha_pos and alpha_neg control class weighting.
    gamma controls focus on hard examples.
    """
    def __init__(
        self,
        alpha_pos: float = 0.75,
        alpha_neg: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = float(alpha_neg)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1.0 - targets)
        loss = alpha_t * ((1.0 - pt).clamp(min=1e-8) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def resolve_loss_config(params: Dict, y_train: np.ndarray) -> Dict:
    """
    Three strategies:
    1) standard_bce   -> BCEWithLogitsLoss(pos_weight=1.0)
    2) weighted_bce   -> BCEWithLogitsLoss(pos_weight=neg/pos)
    3) weighted_focal -> Focal loss with class-balanced alpha + gamma
    """
    loss_name = str(params.get("loss_name", "weighted_bce"))
    pos, neg = class_counts(y_train)
    imbalance_pos_weight = neg / pos

    if loss_name == "standard_bce":
        return {
            "loss_name": "standard_bce",
            "pos_weight": 1.0,
        }

    if loss_name == "weighted_bce":
        return {
            "loss_name": "weighted_bce",
            "pos_weight": imbalance_pos_weight,
        }

    if loss_name == "weighted_focal":
        # alpha_pos = neg / (neg + pos)
        # alpha_neg = pos / (neg + pos)
        return {
            "loss_name": "weighted_focal",
            "alpha_pos": 0.65,
            "alpha_neg": 0.35,
            "focal_gamma": float(params.get("focal_gamma", 2.0)),
        }

    raise ValueError(f"Unknown loss_name: {loss_name}")


def build_criterion(loss_config: Dict, device: torch.device) -> nn.Module:
    loss_name = str(loss_config["loss_name"])

    if loss_name in {"standard_bce", "weighted_bce"}:
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [float(loss_config["pos_weight"])],
                dtype=torch.float32,
                device=device,
            )
        )

    if loss_name == "weighted_focal":
        return BinaryFocalLoss(
            alpha_pos=float(loss_config["alpha_pos"]),
            alpha_neg=float(loss_config["alpha_neg"]),
            gamma=float(loss_config["focal_gamma"]),
        )

    raise ValueError(f"Unknown loss_name in criterion builder: {loss_name}")


def describe_loss(loss_config: Dict) -> str:
    loss_name = str(loss_config["loss_name"])

    if loss_name == "standard_bce":
        return "standard_bce(pos_weight=1.0000)"

    if loss_name == "weighted_bce":
        return f"weighted_bce(pos_weight={float(loss_config['pos_weight']):.4f})"

    return (
        "weighted_focal("
        f"alpha_pos={float(loss_config['alpha_pos']):.4f}, "
        f"alpha_neg={float(loss_config['alpha_neg']):.4f}, "
        f"gamma={float(loss_config['focal_gamma']):.2f})"
    )

class GRUDDataset(Dataset):
    def __init__(
        self,
        x_seq: np.ndarray,
        x_mask: np.ndarray,
        x_delta: np.ndarray,
        x_static: np.ndarray,
        y: np.ndarray,
    ):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.x_mask = torch.tensor(x_mask, dtype=torch.float32)
        self.x_delta = torch.tensor(x_delta, dtype=torch.float32)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_seq.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.x_seq[idx],
            self.x_mask[idx],
            self.x_delta[idx],
            self.x_static[idx],
            self.y[idx],
        )


class GRUDModel(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim

        self.gamma_x = nn.Linear(input_dim, input_dim)
        self.gamma_h = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim + static_dim, 1)
        self.register_buffer(
            'x_mean', torch.zeros(input_dim,dtype=torch.float32)
        )
        # self.input_dropout = nn.Dropout(dropout)
        # self.recurrent_dropout = nn.Dropout(dropout * 0.5)
        # self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor,
        x_delta: torch.Tensor,
        x_static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, F_dim = x_seq.shape
        device = x_seq.device

        h = torch.zeros(B, self.hidden_dim, device=device)
        x_last = torch.zeros(B, F_dim, device=device)

        x_mean = self.x_mean

        for t in range(T):
            x_t = x_seq[:, t, :]
            m_t = x_mask[:, t, :]
            d_t = x_delta[:, t, :]

            x_t_no_nan = torch.nan_to_num(x_t, nan=0.0)

            gamma_x_t = torch.exp(-torch.relu(self.gamma_x(d_t)))
            gamma_h_t = torch.exp(-torch.relu(self.gamma_h(d_t)))

            h = gamma_h_t * h
            # h = self.recurrent_dropout(h)

            x_t_hat = m_t * x_t_no_nan + (1 - m_t) * (
                gamma_x_t * x_last + (1 - gamma_x_t) * x_mean
            )

            x_last = m_t * x_t_no_nan + (1 - m_t) * x_last

            gru_input = torch.cat([x_t_hat, m_t], dim=1)
            # gru_input = self.input_dropout(gru_input)
            h = self.gru_cell(gru_input, h)


        if x_static is not None:
            out = torch.cat([h, x_static], dim=1)
        else:
            out = h

        out = self.dropout(out)
        logits = self.fc(out).squeeze(1)
        return logits

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_feature_mean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    c = x.shape[2]
    means = np.zeros(c, dtype=np.float32)
    for j in range(c):
        obs = x[:, :, j][m[:, :, j] == 1]
        means[j] = float(obs.mean()) if obs.size > 0 else 0.0
    return means


def make_torch_loader(
    batch,
    batch_size: int,
    shuffle: bool,
):
    dataset = GRUDDataset(
        x_seq=batch.X,
        x_mask=batch.mask,
        x_delta=batch.delta,
        x_static=batch.X_static,
        y=batch.y,
    )
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(acc),
        "brier": float(brier),
    }


def best_youden_j(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(j_scores[best_idx]), float(thresholds[best_idx])

def summarize_results(results: List[FoldResult]) -> Dict[str, float]:
    keys = ["auroc", "auprc", "f1", "sensitivity", "specificity", "accuracy", "brier"]
    summary = {}
    for k in keys:
        vals = np.array([getattr(r, k) for r in results], dtype=float)
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return summary

def print_summary(title: str, summary: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


def print_test_metrics(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(
        f"TEST AUROC={metrics['auroc']:.4f} | "
        f"AUPRC={metrics['auprc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"TEST Brier={metrics['brier']:.4f} | "
        f"Sens={metrics['sensitivity']:.4f} | "
        f"Spec={metrics['specificity']:.4f} | "
        f"Acc={metrics['accuracy']:.4f}"
    )

def plot_loss_history(
    history: Dict[str, List[float]],
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def generate_all_outputs(y_true, y_prob, metrics, save_dir, window_hours):
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. PR Curve ──────────────────────────────
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.calibration import calibration_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='orange',
             label=f'GRU-D (AUPRC={auprc:.3f})')
    plt.axhline(y=baseline, color='k', linestyle='--',
                label=f'Random (={baseline:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (GRU-D, {window_hours}h Window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'pr_curves.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("PR curve saved.")

    # ── 2. Calibration Curve ──────────────────────
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10
    )

    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, fraction_pos, 's-',
             color='green', label='GRU-D')
    plt.plot([0,1],[0,1],'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve (GRU-D, {window_hours}h Window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'calibration_curve.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Calibration curve saved.")

    # ── 3. Results Summary ────────────────────────
    summary = f"""
    GRU-D Results Summary ({window_hours}h Window)
    ==================================
    Model:        GRU-D (Gated Recurrent Unit with Decay)
    Window:       {window_hours} hours
    Threshold:    {metrics['threshold']:.4f}

    Cross-Validation Results (10-fold):
      CV_AUROC:   {metrics['cv_auroc']:.4f}
      CV_AUPRC:   {metrics['cv_auprc']:.4f}
      CV_F1:      {metrics['cv_f1']:.4f}
      CV_Brier:   {metrics['cv_brier']:.4f}

    Test Set Results:
      TEST_AUROC:  {metrics['test_auroc']:.4f}
      TEST_AUPRC:  {metrics['test_auprc']:.4f}
      TEST_F1:     {metrics['test_f1']:.4f}
      TEST_Brier:  {metrics['test_brier']:.4f}
      Sensitivity: {metrics['sensitivity']:.4f}
      Specificity: {metrics['specificity']:.4f}
      Accuracy:    {metrics['accuracy']:.4f}

    Overfitting Analysis:
      CV->Test AUROC gap: {metrics['test_auroc'] - metrics['cv_auroc']:+.4f}
      CV->Test AUPRC gap: {metrics['test_auprc'] - metrics['cv_auprc']:+.4f}

    Best Hyperparameters (Optuna):
      hidden_dim:   {metrics['hidden_dim']}
      dropout:      {metrics['dropout']}
      lr:           {metrics['lr']:.6f}
      weight_decay: {metrics['weight_decay']:.5f}
      batch_size:   {metrics['batch_size']}
      loss:         weighted_bce (pos_weight≈6.04)
    """
    with open(os.path.join(save_dir, 'results_summary.txt'), 'w') as f:
        f.write(summary)
    print("Results summary saved.")
    print(summary)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: TorchDataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_static: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs_all = []
    ys_all = []
    total_loss = 0.0
    total_n = 0

    for x_seq, x_mask, x_delta, x_static, y in loader:
        x_seq = x_seq.to(device)
        x_mask = x_mask.to(device)
        x_delta = x_delta.to(device)
        x_static = x_static.to(device)
        y = y.to(device)

        logits = model(x_seq, x_mask, x_delta, x_static if use_static else None)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.append(probs)
        ys_all.append(y.detach().cpu().numpy())

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    y_true = np.concatenate(ys_all, axis=0)
    y_prob = np.concatenate(probs_all, axis=0)
    avg_loss = total_loss / max(total_n, 1)
    return y_true, y_prob, avg_loss

@torch.no_grad()
def predict_proba(model: nn.Module, loader: TorchDataLoader, device: torch.device, use_static: bool) -> np.ndarray:
    model.eval()
    probs_all = []

    for x_seq, x_mask, x_delta, x_static, _ in loader:
        x_seq = x_seq.to(device)
        x_mask = x_mask.to(device)
        x_delta = x_delta.to(device)
        x_static = x_static.to(device)

        logits = model(x_seq, x_mask, x_delta, x_static if use_static else None)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.append(probs)

    return np.concatenate(probs_all, axis=0)


def train_one_epoch(
    model: nn.Module,
    loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_static: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0

    for x_seq, x_mask, x_delta, x_static, y in loader:
        x_seq = x_seq.to(device)
        x_mask = x_mask.to(device)
        x_delta = x_delta.to(device)
        x_static = x_static.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_seq, x_mask, x_delta, x_static if use_static else None)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)

def fit_grud_model(
    train_batch,
    val_batch,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
    use_static: bool,
    seed: int,
    criterion: nn.Module,
    trial: Optional[optuna.Trial] = None,
    return_history: bool = False,
):

    set_seed(seed)

    input_dim = train_batch.X.shape[2]
    static_dim = train_batch.X_static.shape[1] if use_static else 0

    model = GRUDModel(
        input_dim=input_dim,
        static_dim=static_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)

    x_mean_np = compute_feature_mean(train_batch.X, train_batch.mask)
    model.x_mean.copy_(torch.tensor(x_mean_np, dtype=torch.float32))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5
    # )


    train_loader = make_torch_loader(train_batch, batch_size=batch_size, shuffle=True)
    val_loader = make_torch_loader(val_batch, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_auprc = -np.inf
    best_epoch = 0

    best_val_loss = np.inf
    no_improve_loss = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_auprc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_static)
        y_val_true, val_prob, val_loss = evaluate_model(model, val_loader, device, criterion, use_static)
        val_auprc = average_precision_score(y_val_true.astype(int), val_prob)
        # scheduler.step(val_auprc)
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_auprc"].append(float(val_auprc))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auprc={val_auprc:.4f}"
        )
        if val_auprc > best_val_auprc + 1e-6:
          best_val_auprc = float(val_auprc)
          best_epoch = epoch
          best_state = copy.deepcopy(model.state_dict())

        if trial is not None:
          trial.report(float(val_auprc), step=epoch)
          if trial.should_prune():
            raise optuna.TrialPruned()

        if epoch - best_epoch >= patience:
          break

        if val_loss < best_val_loss - 1e-4:
          best_val_loss = val_auprc
          no_improve_loss = 0
        else:
          no_improve_loss += 1
        if no_improve_loss >= 5:
          print(f"Early stop by val_loss at epoch {epoch}")
          break

    if best_state is not None:
      model.load_state_dict(best_state)

    if return_history:
      return model, history
    return model

def tune_threshold_on_holdout(
    arrays,
    trainval_idx: np.ndarray,
    args,
    device: torch.device,
    params: Dict,
    holdout_size: float = 0.15,
) -> float:
    """
    Train on (trainval - holdout), predict on holdout,
    then find Youden J threshold on that holdout only.
    """
    y_trainval = np.array(arrays.Mortality, dtype=float)[trainval_idx].astype(int)

    # Split off a small holdout JUST for threshold tuning
    tune_tr_idx, tune_ho_idx = train_test_split(
        trainval_idx,
        test_size=holdout_size,
        stratify=y_trainval,
        random_state=args.seed + 777,  # different seed to avoid overlap
    )

    train_arrays = UnifiedPreprocessor.subset_arrays(arrays, tune_tr_idx)
    ho_arrays    = UnifiedPreprocessor.subset_arrays(arrays, tune_ho_idx)

    prep = UnifiedPreprocessor(
        data_dir=args.data_dir,
        window_hours=args.window_hours,
        use_forward_fill=False,
    )
    prep.fit(train_arrays, branch="grud")
    train_batch = prep.transform(train_arrays)
    ho_batch    = prep.transform(ho_arrays)

    y_train = train_batch.y.astype(np.float32)
    loss_config = resolve_loss_config(params, y_train)
    criterion   = build_criterion(loss_config, device)

    model = fit_grud_model(
        train_batch=train_batch,
        val_batch=ho_batch,
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]),
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        use_static=bool(args.use_static),
        seed=args.seed + 777,
        criterion=criterion,
    )

    ho_loader = make_torch_loader(ho_batch, int(params["batch_size"]), shuffle=False)
    ho_prob   = predict_proba(model, ho_loader, device, bool(args.use_static))
    ho_true   = ho_batch.y.astype(int)

    _, threshold = best_youden_j(ho_true, ho_prob)
    return float(threshold)


def run_cv_grud(
    arrays,
    trainval_idx: np.ndarray,
    args,
    device: torch.device,
    params: Dict,
    threshold: float,
    return_oof: bool = False,
):
    y_trainval = np.array(arrays.Mortality, dtype=float)[trainval_idx].astype(int)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_results: List[FoldResult] = []
    oof_prob = np.zeros(len(trainval_idx), dtype=np.float32)
    oof_true = y_trainval.copy()

    for fold, (tr_pos, va_pos) in enumerate(skf.split(trainval_idx, y_trainval), start=1):
        print(f"\n--- Fold {fold}/{args.n_splits} ---")

        train_idx = trainval_idx[tr_pos]
        val_idx = trainval_idx[va_pos]

        train_arrays = UnifiedPreprocessor.subset_arrays(arrays, train_idx)
        val_arrays = UnifiedPreprocessor.subset_arrays(arrays, val_idx)

        prep = UnifiedPreprocessor(
            data_dir=args.data_dir,
            window_hours=args.window_hours,
            use_forward_fill=False,
        )
        prep.fit(train_arrays, branch="grud")

        train_batch = prep.transform(train_arrays)
        val_batch = prep.transform(val_arrays)

        y_train = train_batch.y.astype(np.float32)
        loss_config = resolve_loss_config(params, y_train)
        criterion = build_criterion(loss_config, device)
        print(f"Loss: {describe_loss(loss_config)}")

        model = fit_grud_model(
            train_batch=train_batch,
            val_batch=val_batch,
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            batch_size=int(params["batch_size"]),
            epochs=args.epochs,
            patience=args.patience,
            device=device,
            use_static=bool(args.use_static),
            seed=args.seed + fold,
            criterion=criterion,
        )

        val_loader = make_torch_loader(
            val_batch,
            batch_size=int(params["batch_size"]),
            shuffle=False,
        )
        val_score = predict_proba(model, val_loader, device, bool(args.use_static))
        metrics = compute_metrics(val_batch.y.astype(int), val_score, threshold=threshold)

        fold_result = FoldResult(
            fold=fold,
            auroc=metrics["auroc"],
            auprc=metrics["auprc"],
            f1=metrics["f1"],
            sensitivity=metrics["sensitivity"],
            specificity=metrics["specificity"],
            accuracy=metrics["accuracy"],
            brier=metrics["brier"],
        )
        fold_results.append(fold_result)

        print(
            f"AUROC={fold_result.auroc:.4f} | "
            f"AUPRC={fold_result.auprc:.4f} | "
            f"F1={fold_result.f1:.4f} | "
            f"Brier={fold_result.brier:.4f} | "
            f"Sens={fold_result.sensitivity:.4f} | "
            f"Spec={fold_result.specificity:.4f} | "
            f"Acc={fold_result.accuracy:.4f}"
        )

        if return_oof:
            oof_prob[va_pos] = val_score

    if return_oof:
        return fold_results, oof_true, oof_prob
    return fold_results

def fit_final_and_test(
    arrays,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    device: torch.device,
    params: Dict,
    threshold: float,
    plot_curve: bool = False,
    phase_name: str = "final_fit",
):

    inner_train_idx, inner_val_idx = train_test_split(
        trainval_idx,
        test_size=max(args.test_size, 0.1),
        stratify=np.array(arrays.Mortality, dtype=float)[trainval_idx].astype(int),
        random_state=args.seed,
    )

    inner_train_arrays = UnifiedPreprocessor.subset_arrays(arrays, inner_train_idx)
    inner_val_arrays = UnifiedPreprocessor.subset_arrays(arrays, inner_val_idx)
    test_arrays = UnifiedPreprocessor.subset_arrays(arrays, test_idx)

    final_prep = UnifiedPreprocessor(
        data_dir=args.data_dir,
        window_hours=args.window_hours,
        use_forward_fill=False,
    )
    final_prep.fit(inner_train_arrays, branch="grud")

    inner_train_batch = final_prep.transform(inner_train_arrays)
    inner_val_batch = final_prep.transform(inner_val_arrays)
    test_batch = final_prep.transform(test_arrays)

    y_inner_train = inner_train_batch.y.astype(np.float32)
    loss_config = resolve_loss_config(params, y_inner_train)
    criterion = build_criterion(loss_config, device)
    print(f"Final fit loss: {describe_loss(loss_config)}")

    if plot_curve:
        final_model, history = fit_grud_model(
            train_batch=inner_train_batch,
            val_batch=inner_val_batch,
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            batch_size=int(params["batch_size"]),
            epochs=args.epochs,
            patience=args.patience,
            device=device,
            use_static=bool(args.use_static),
            seed=args.seed + 999,
            criterion=criterion,
            return_history=True,
        )

        plot_loss_history(
            history=history,
            title=phase_name,
            save_path=f"{phase_name}_loss.png",
            show=bool(args.plot_show),
        )
    else:
        final_model = fit_grud_model(
            train_batch=inner_train_batch,
            val_batch=inner_val_batch,
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            batch_size=int(params["batch_size"]),
            epochs=args.epochs,
            patience=args.patience,
            device=device,
            use_static=bool(args.use_static),
            seed=args.seed + 999,
            criterion=criterion,
            return_history=False,
        )

    test_loader = make_torch_loader(
        test_batch,
        batch_size=int(params["batch_size"]),
        shuffle=False,
    )
    test_score = predict_proba(final_model, test_loader, device, bool(args.use_static))
    test_metrics = compute_metrics(test_batch.y.astype(int), test_score, threshold=threshold)

    return test_metrics, test_score, test_batch.y.astype(int)


def objective(trial: optuna.Trial, arrays, trainval_idx: np.ndarray, args, device: torch.device) -> float:
    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.4, step=0.1),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "loss_name": "weighted_bce",
        "focal_gamma": 2.0,
    }



    if params["loss_name"] == "weighted_focal":
        params["focal_gamma"] = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0, 4.0])
    else:
        params["focal_gamma"] = args.focal_gamma

    y_trainval = np.array(arrays.Mortality, dtype=float)[trainval_idx].astype(int)
    opt_tr_idx, opt_va_idx = train_test_split(
        trainval_idx,
        test_size=args.optuna_val_size,
        stratify=y_trainval,
        random_state=trial.number,
    )

    train_arrays = UnifiedPreprocessor.subset_arrays(arrays, opt_tr_idx)
    val_arrays = UnifiedPreprocessor.subset_arrays(arrays, opt_va_idx)

    prep = UnifiedPreprocessor(
        data_dir=args.data_dir,
        window_hours=args.window_hours,
        use_forward_fill=False,
    )
    prep.fit(train_arrays, branch="grud")

    train_batch = prep.transform(train_arrays)
    val_batch = prep.transform(val_arrays)

    y_train = train_batch.y.astype(np.float32)
    loss_config = resolve_loss_config(params, y_train)
    criterion = build_criterion(loss_config, device)

    model = fit_grud_model(
        train_batch=train_batch,
        val_batch=val_batch,
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]),
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        use_static=bool(args.use_static),
        seed=args.seed,
        criterion=criterion,
        trial=trial,
        return_history=False,
    )

    val_loader = make_torch_loader(val_batch, batch_size=int(params["batch_size"]), shuffle=False)
    y_val_true, y_val_prob, _ = evaluate_model(model, val_loader, device, criterion, bool(args.use_static))

    auprc = float(average_precision_score(y_val_true, y_val_prob))
    auroc = float(roc_auc_score(y_val_true, y_val_prob))

    trial.set_user_attr("loss_desc", describe_loss(loss_config))
    trial.set_user_attr("auprc", auprc)
    trial.set_user_attr("auroc", auroc)

    return auprc

def main():
    parser = argparse.ArgumentParser(description="10% untouched test + n-fold CV for GRU-D")
    parser.add_argument("--data-dir", type=str, default=os.path.join(REPO_ROOT, "Data_group_7"), help="Path to Data_group_7 directory")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--window-hours", type=int, choices=[1, 3, 6, 12, 24], default=3)
    parser.add_argument("--use-static", type=int, choices=[0, 1], default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--loss-name",
        type=str,
        default="weighted_bce",
        choices=["standard_bce", "weighted_bce", "weighted_focal"],
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--do-optuna", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--optuna-val-size", type=float, default=0.2)
    parser.add_argument("--study-name", type=str, default="grud_optuna_3way_loss_switch")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--plot-final-curves", type=int, choices=[0, 1], default=1)
    parser.add_argument("--plot-show", type=int, choices=[0, 1], default=0)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(REPO_ROOT, "GRU_D", "Results"))
    args = parser.parse_args()


    set_seed(args.seed)
    device = get_device(args.device)
    use_static = bool(args.use_static)

    print(f"Using device: {device}")
    print(f"Data Directory: {args.data_dir}")
    print("=" * 72)
    print(
        f"GRU-D CV RUNNER | holdout_test={args.test_size:.0%} | cv={args.n_splits}-fold | "
        f"window={args.window_hours}h | use_static={args.use_static} | device={device}"
    )
    print("=" * 72)

    loader = DataLoader(data_dir=args.data_dir)
    arrays = loader.load()

    y_all = np.array(arrays.Mortality, dtype=float)
    keep = ~np.isnan(y_all)
    idx_all = np.where(keep)[0]
    y_valid = y_all[keep].astype(int)

    trainval_idx, test_idx = train_test_split(
        idx_all,
        test_size=args.test_size,
        stratify=y_valid,
        random_state=args.seed,
    )

    print(f"\nUntouched test size: {len(test_idx)}")
    print(f"Train+Val size:      {len(trainval_idx)}")

    initial_threshold = float(args.threshold)

    window_output_dir = os.path.join(
        args.plot_dir,
        f"final_grud_result_{args.window_hours}h"
    )
    os.makedirs(window_output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Phase A: baseline CV
    # ----------------------------------------------------------
    baseline_params = {
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "loss_name": args.loss_name,
        "focal_gamma": args.focal_gamma,
    }

    print("\n" + "=" * 72)
    print("PHASE A | BASELINE CV ON 90% WITH WEIGHTED BCE")
    print("=" * 72)

    baseline_fold_results, baseline_oof_true, baseline_oof_prob = run_cv_grud(
        arrays=arrays,
        trainval_idx=trainval_idx,
        args=args,
        device=device,
        params=baseline_params,
        threshold=initial_threshold,
        return_oof=True,
    )
    baseline_summary = summarize_results(baseline_fold_results)
    print_summary("BASELINE CV SUMMARY", baseline_summary)

    # ----------------------------------------------------------
    # Phase B: threshold from SEPARATE holdout (not OOF)
    # ----------------------------------------------------------
    print("\nPHASE B | THRESHOLD FROM SEPARATE HOLDOUT")
    baseline_threshold = tune_threshold_on_holdout(
        arrays, trainval_idx, args, device, baseline_params
    )
    print(f"Baseline holdout threshold = {baseline_threshold:.6f}")


    # ----------------------------------------------------------
    # Phase C: baseline final refit -> test
    # ----------------------------------------------------------
    baseline_test_metrics, _, _ = fit_final_and_test(
        arrays=arrays,
        trainval_idx=trainval_idx,
        test_idx=test_idx,
        args=args,
        device=device,
        params=baseline_params,
        threshold=baseline_threshold,
        plot_curve=bool(args.plot_final_curves),
        phase_name=os.path.join(
            window_output_dir, "phaseC_baseline_final_refit"
        ),
    )


    print_test_metrics(
        "PHASE C | BASELINE FINAL REFIT ON 90% -> TEST ON 10%",
        baseline_test_metrics,
    )

    # ----------------------------------------------------------
    # Phase D: Optuna on 90% pool
    # ----------------------------------------------------------
    if bool(args.do_optuna):
        print("\n" + "=" * 72)
        print("PHASE D | OPTUNA ON 90% POOL (OBJECTIVE = AUPRC ONLY)")
        print("         loss_name ∈ {standard_bce, weighted_bce, weighted_focal}")
        print("=" * 72)

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=False,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )

        study.optimize(
            lambda trial: objective(trial, arrays, trainval_idx, args, device),
            n_trials=args.n_trials,
        )

        tuned_params = dict(study.best_params)
        if "focal_gamma" not in tuned_params:
            tuned_params["focal_gamma"] = args.focal_gamma
        if "loss_name" not in tuned_params:
            tuned_params["loss_name"] = "weighted_bce"


        print("\nBest Optuna params:", tuned_params)
        print("Best Optuna AUPRC:", study.best_trial.value)
        print("Best trial attrs:", study.best_trial.user_attrs)
    else:
        tuned_params = {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "loss_name": args.loss_name,
            "focal_gamma": args.focal_gamma,
        }
        print("\nOptuna disabled. Using manually chosen params.")

    # ----------------------------------------------------------
    # Phase E: threshold from separate holdout
    # ----------------------------------------------------------
    print("\nPHASE E | THRESHOLD FROM SEPARATE HOLDOUT")
    tuned_threshold = tune_threshold_on_holdout(
        arrays, trainval_idx, args, device, tuned_params
    )
    print(f"Tuned holdout threshold = {tuned_threshold:.6f}")


    # ----------------------------------------------------------
    # Phase F: tuned CV with tuned threshold
    # ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("PHASE F | TUNED CV ON 90% WITH BEST PARAMS + OOF YOUDEN THRESHOLD")
    print("=" * 72)

    tuned_fold_results = run_cv_grud(
        arrays=arrays,
        trainval_idx=trainval_idx,
        args=args,
        device=device,
        params=tuned_params,
        threshold=tuned_threshold,
        return_oof=False,
    )
    tuned_summary = summarize_results(tuned_fold_results)
    print_summary("TUNED CV SUMMARY", tuned_summary)

    # ----------------------------------------------------------
    # Phase G: tuned final refit -> test
    # ----------------------------------------------------------
    tuned_test_metrics, test_score, test_true = fit_final_and_test(
        arrays=arrays,
        trainval_idx=trainval_idx,
        test_idx=test_idx,
        args=args,
        device=device,
        params=tuned_params,
        threshold=tuned_threshold,
        plot_curve=bool(args.plot_final_curves),
        phase_name=os.path.join(
            window_output_dir, "phaseG_tuned_final_refit"
        ),
    )

    print_test_metrics("PHASE G | TUNED FINAL REFIT ON 90% -> TEST ON 10%", tuned_test_metrics)

    metrics = {
        'threshold':   tuned_threshold,
        'cv_auroc':    tuned_summary['auroc_mean'],
        'cv_auprc':    tuned_summary['auprc_mean'],
        'cv_f1':       tuned_summary['f1_mean'],
        'cv_brier':    tuned_summary['brier_mean'],
        'test_auroc':  tuned_test_metrics['auroc'],
        'test_auprc':  tuned_test_metrics['auprc'],
        'test_f1':     tuned_test_metrics['f1'],
        'test_brier':  tuned_test_metrics['brier'],
        'sensitivity': tuned_test_metrics['sensitivity'],
        'specificity': tuned_test_metrics['specificity'],
        'accuracy':    tuned_test_metrics['accuracy'],
        'hidden_dim':  tuned_params['hidden_dim'],
        'dropout':     tuned_params['dropout'],
        'lr':          tuned_params['lr'],
        'weight_decay':tuned_params['weight_decay'],
        'batch_size':  tuned_params['batch_size'],
    }

    generate_all_outputs(
        y_true=test_true,
        y_prob=test_score,
        metrics=metrics,
        save_dir=window_output_dir,
        window_hours=args.window_hours,
    )

if __name__ == "__main__":
    main()
