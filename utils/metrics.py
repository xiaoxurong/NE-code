import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

# -------------------------
# Binary classification
# -------------------------

def binary_classification_metrics(
    y_true,
    y_prob,
    threshold=0.5
):
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        Ground-truth labels (0 or 1)
    y_prob : array-like, shape (N,)
        Predicted probabilities for the positive class
    threshold : float
        Threshold to convert probabilities to class labels

    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # probability-based metrics
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        # sensitivity / specificity (existing names kept for compatibility)
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        # per-class accuracy (explicit aliases — useful for focal loss tuning)
        # acc_class1 = how often NE cases are correctly identified  (= sensitivity)
        # acc_class0 = how often normal cases are correctly identified (= specificity)
        "acc_class1": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "acc_class0": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    })

    return metrics


# -------------------------
# Subject-level aggregation
# -------------------------

def aggregate_subject_predictions(subject_ids, y_prob, y_true):
    """
    Average window-level predicted probabilities per subject, then compute
    metrics at the subject level.

    Use this at test time when each subject contributes multiple windows.
    All windows from the same subject get the same label (inherited from
    the subject), so we average their predicted probabilities before
    computing AUROC / F1 / etc.

    Parameters
    ----------
    subject_ids : array-like, shape (N,)
        Subject identifier for each window (e.g. Study_ID strings).
    y_prob : array-like, shape (N,)
        Window-level predicted probabilities for the positive class.
    y_true : array-like, shape (N,)
        Window-level ground-truth labels (same for all windows of a subject).

    Returns
    -------
    metrics : dict   — subject-level metrics
    subj_prob : np.ndarray (n_subjects,)  — averaged probabilities
    subj_true : np.ndarray (n_subjects,)  — one label per subject
    """
    subject_ids = np.asarray(subject_ids)
    y_prob      = np.asarray(y_prob)
    y_true      = np.asarray(y_true)

    unique_subjects = np.unique(subject_ids)
    subj_prob = np.array([y_prob[subject_ids == s].mean() for s in unique_subjects])
    subj_true = np.array([y_true[subject_ids == s][0]    for s in unique_subjects])

    metrics = binary_classification_metrics(subj_true, subj_prob, threshold=0.5)
    return metrics, subj_prob, subj_true


# -------------------------
# Multi-class classification
# -------------------------

def multiclass_classification_metrics(
    y_true,
    y_prob,
    average="macro"
):
    """
    Compute metrics for multi-class classification.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        Ground-truth class labels
    y_prob : array-like, shape (N, C)
        Predicted class probabilities
    average : str
        Averaging strategy for precision/recall/F1

    Returns
    -------
    metrics : dict
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = y_prob.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics

def find_optimal_threshold(y_true, y_prob):
    """Find threshold maximising Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr + (1 - fpr) - 1      # sensitivity + specificity - 1
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])