from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_results(
    results: dict[str, Any],
    df_test: pd.DataFrame,
    plot_types: list[str],
    real_percentiles: bool,
) -> dict[str, Figure]:
    """Create Matplotlib figures based on model evaluation results.

    Args:
        results: Dictionary containing model results (metrics, predictions, etc.).
        df_test: Test dataset as a pandas DataFrame.
        plot_types: List of plot types to generate.
        real_percentiles: Whether thresholds represent cross-section-weighted percentiles.

    Returns:
        Dictionary mapping plot names to Matplotlib Figure objects.
    """
    plots: dict[str, Figure] = {}

    if "roc_curve" in plot_types:
        fpr, tpr, _ = roc_curve(df_test["target"], results["predictions"])
        plots["roc_curve"] = plot_roc_curve(fpr, tpr, results["metrics"]["roc_auc"])

    if "pr_curve" in plot_types:
        precision, recall, _ = precision_recall_curve(
            df_test["target"], results["predictions"]
        )
        plots["pr_curve"] = plot_precision_recall_curve(
            recall, precision, results["metrics"]["average_precision"]
        )

    if "weighted_pr_curve" in plot_types:
        plots["weighted_pr_curve"] = plot_weighted_precision_recall_curve(
            results["metrics"]["weighted_recall"],
            results["metrics"]["weighted_precision"],
            results["metrics"]["weighted_pr_thresholds"],
            results["metrics"]["weighted_pr_auc"],
            use_real_event_percentiles=real_percentiles,
        )

    if "feature_importances" in plot_types and "feature_importances" in results:
        feature_cols = [
            col for col in df_test.columns if col not in ("Truth", "target")
        ]
        importances = np.asarray(results["feature_importances"])
        if len(feature_cols) != len(importances):
            feature_cols = [f"feature_{i}" for i in range(len(importances))]
        plots["feature_importances"] = plot_feature_importances(
            feature_cols, importances
        )

    if "sig_bkg_metrics" in plot_types and "sig_bkg_metrics" in results:
        plots["sig_bkg_metrics"] = plot_signal_background_metrics(
            results["sig_bkg_metrics"], real_percentiles
        )

    if "scores_histogram" in plot_types:
        plots["scores_histogram"] = plot_scores_histogram(
            df_test["target"].to_numpy(), results["predictions"]
        )

    if "scores_weighted_histogram" in plot_types:
        plots["scores_weighted_histogram"] = plot_scores_weighted_histogram(
            df_test["target"].to_numpy(),
            results["predictions"],
            results["sample_weights"],
        )

    return plots


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=1.8, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.2, label="Random classifier")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(
    recall: np.ndarray, precision: np.ndarray, average_precision: float
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=1.8, label=f"PR curve (AP = {average_precision:.3f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_weighted_precision_recall_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    thresholds: np.ndarray,
    pr_auc: float,
    use_real_event_percentiles: bool,
) -> Figure:
    _ = thresholds
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=1.8, label=f"Weighted PR curve (AUC = {pr_auc:.3f})")
    ax.set_title("Weighted Precision-Recall Curve")
    if use_real_event_percentiles:
        ax.set_xlabel("Recall (real event percentiles)")
        ax.set_ylabel("Precision (real event percentiles)")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_feature_importances(
    feature_names: Sequence[str] | np.ndarray, importances: np.ndarray
) -> Figure:
    names_arr = np.asarray(feature_names)
    imp_arr = np.asarray(importances, dtype=float)
    sorted_idx = np.argsort(imp_arr)

    height = max(6.0, len(names_arr) * 0.22 + 1.5)
    fig, ax = plt.subplots(figsize=(9, height))
    y_pos = np.arange(len(sorted_idx))
    ax.barh(y_pos, imp_arr[sorted_idx], color="#1f77b4", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_arr[sorted_idx], fontsize=8)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_signal_background_metrics(
    metrics: dict[str, np.ndarray], use_real_event_percentiles: bool
) -> Figure:
    thresholds = np.asarray(metrics["thresholds"], dtype=float)
    sig_eff = np.asarray(metrics["signal_efficiency"], dtype=float)
    bkg_rej = np.asarray(metrics["background_rejection"], dtype=float)
    significance = np.asarray(metrics["significance"], dtype=float)
    s_over_b = np.asarray(metrics["s_over_b"], dtype=float)
    s_over_b_finite = np.where(np.isfinite(s_over_b), s_over_b, np.nan)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
    axes[0].plot(thresholds, sig_eff, marker="o", ms=3.5, label="Signal Efficiency")
    axes[0].plot(thresholds, bkg_rej, marker="s", ms=3.5, label="Background Rejection")
    axes[0].set_ylabel("Efficiency/Rejection")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        thresholds,
        significance,
        marker="o",
        ms=3.5,
        color="#2ca02c",
        label="Significance",
    )
    axes[1].set_ylabel("Significance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(
        thresholds, s_over_b_finite, marker="o", ms=3.5, color="#d62728", label="S/B"
    )
    axes[2].set_ylabel("S/B")
    axes[2].set_xlabel(
        "Percentile (real events)" if use_real_event_percentiles else "Threshold"
    )
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle("Signal-Background Metrics")
    fig.tight_layout()
    return fig


def plot_scores_histogram(y_true: np.ndarray, y_scores: np.ndarray) -> Figure:
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores, dtype=float)
    bins = np.linspace(0.0, 1.0, 51)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_val, label, color in (
        (0, "Background", "#1f77b4"),
        (1, "Signal", "#d62728"),
    ):
        mask = y_true_arr == cls_val
        if not np.any(mask):
            continue
        weights = np.full(int(mask.sum()), 1.0 / int(mask.sum()), dtype=float)
        ax.hist(
            y_scores_arr[mask],
            bins=bins,
            weights=weights,
            alpha=0.6,
            color=color,
            label=label,
        )

    ax.set_title("Classifier Scores")
    ax.set_xlabel("Score")
    ax.set_ylabel("Probability")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_scores_weighted_histogram(
    y_true: np.ndarray, y_scores: np.ndarray, weights: np.ndarray
) -> Figure:
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    bins = np.linspace(0.0, 1.0, 51)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_val, label, color in (
        (0, "Background", "#1f77b4"),
        (1, "Signal", "#d62728"),
    ):
        mask = y_true_arr == cls_val
        if not np.any(mask):
            continue
        ax.hist(
            y_scores_arr[mask],
            bins=bins,
            weights=weights_arr[mask],
            alpha=0.6,
            color=color,
            label=label,
        )

    ax.set_yscale("log")
    ax.set_title("Classifier Scores")
    ax.set_xlabel("Score")
    ax.set_ylabel("Cross Sections")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
