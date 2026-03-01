"""
Shared evaluation utilities for the Uber ride cancellation prediction project.

Provides functions for model evaluation, threshold tuning, and diagnostic
plotting that are reused across multiple modeling notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_TARGETS = {
    "f2": 0.68,
    "recall": 0.70,
    "precision": 0.60,
}


def evaluate_model(y_true, y_pred, y_prob, dataset_name=""):
    """Compute and print key classification metrics against project targets.

    Returns a dict of metric name → value.
    """
    f2 = fbeta_score(y_true, y_pred, beta=2)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    def _status(value, target):
        return "✅" if value >= target else "❌"

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS - {dataset_name}")
    print(f"{'=' * 60}")
    print()
    print(f"{'Metric':<16}| {'Value':<8}| {'Target':<8}| Status")
    print("-" * 50)
    print(f"{'F2-Score':<16}| {f2:.4f}  | ≥ {METRIC_TARGETS['f2']:.2f}  | {_status(f2, METRIC_TARGETS['f2'])}")
    print(f"{'Recall':<16}| {recall:.4f}  | ≥ {METRIC_TARGETS['recall']:.2f}  | {_status(recall, METRIC_TARGETS['recall'])}")
    print(f"{'Precision':<16}| {precision:.4f}  | ≥ {METRIC_TARGETS['precision']:.2f}  | {_status(precision, METRIC_TARGETS['precision'])}")
    print(f"{'F1-Score':<16}| {f1:.4f}  |         |")
    print(f"{'PR-AUC':<16}| {pr_auc:.4f}  |         |")
    print(f"{'ROC-AUC':<16}| {roc_auc:.4f}  |         |")

    return {
        "f2": f2,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def find_optimal_threshold(y_true, y_prob, beta=2):
    """Grid-search over thresholds to maximise the F-beta score.

    Returns (best_threshold, best_f_beta, results_df).
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f_beta = 0.0

    results = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f_beta = fbeta_score(y_true, y_pred, beta=beta)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": thresh,
            "f_beta": f_beta,
            "precision": prec,
            "recall": rec,
        })

        if f_beta > best_f_beta:
            best_f_beta = f_beta
            best_threshold = thresh

    return best_threshold, best_f_beta, pd.DataFrame(results)


def plot_evaluation(y_true, y_pred, y_prob, title=""):
    """Plot confusion matrix, precision-recall curve, and ROC curve side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix - {title}")
    axes[0].set_xticklabels(["Completed", "Cancelled"])
    axes[0].set_yticklabels(["Completed", "Cancelled"])

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    axes[1].plot(rec_curve, prec_curve, "b-", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y=METRIC_TARGETS["precision"], color="r", linestyle="--", label=f"Precision target ({METRIC_TARGETS['precision']})")
    axes[1].axvline(x=METRIC_TARGETS["recall"], color="g", linestyle="--", label=f"Recall target ({METRIC_TARGETS['recall']})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="best")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    axes[2].plot(fpr, tpr, "b-", linewidth=2, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve")
    axes[2].legend(loc="best")

    plt.tight_layout()
    plt.show()
