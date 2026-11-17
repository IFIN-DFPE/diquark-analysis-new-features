import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray,
    thresholds: np.ndarray,
    use_real_event_percentiles: bool,
) -> dict[str, float]:
    precision, recall, pr_thresholds = weighted_precision_recall_curve(
        y_true, y_pred, sample_weights, thresholds, use_real_event_percentiles
    )
    pr_auc = auc(recall, precision)

    return {
        "accuracy": accuracy_score(y_true, (y_pred > 0.5).astype(int)),
        "precision": precision_score(y_true, (y_pred > 0.5).astype(int)),
        "recall": recall_score(y_true, (y_pred > 0.5).astype(int)),
        "f1_score": f1_score(y_true, (y_pred > 0.5).astype(int)),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "average_precision": average_precision_score(y_true, y_pred),
        "weighted_pr_auc": pr_auc,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_pr_thresholds": pr_thresholds,
    }


def weighted_precision_recall_curve(
    y_true, y_pred, sample_weights, thresholds, use_real_event_percentiles
):
    if use_real_event_percentiles:
        sorted_indices = np.argsort(y_pred)
        cumulative_weights = np.cumsum(sample_weights[sorted_indices])
        total_weight = cumulative_weights[-1]
        percentile_thresholds = [
            np.searchsorted(cumulative_weights, threshold * total_weight)
            for threshold in thresholds
        ]
        thresholds = y_pred[sorted_indices[percentile_thresholds]]
    else:
        thresholds = np.quantile(y_pred, thresholds)

    thresholds = np.sort(np.unique(np.concatenate([thresholds, [0, 1]])))

    precision = []
    recall = []

    true_positives = np.sum(sample_weights[y_true == 1])

    for threshold in thresholds:
        y_pred_binary = y_pred >= threshold
        tp = np.sum(sample_weights[y_pred_binary & (y_true == 1)])
        fp = np.sum(sample_weights[y_pred_binary & (y_true == 0)])

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / true_positives if true_positives > 0 else 0

        precision.append(p)
        recall.append(r)

    # Ensure monotonicity
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # Remove duplicate points
    unique_recalls, unique_indices = np.unique(recall, return_index=True)
    unique_precisions = precision[unique_indices]

    # Add (1.0, 0.0) and (0.0, 1.0) points
    if unique_recalls[0] != 0:
        unique_recalls = np.r_[0, unique_recalls]
        unique_precisions = np.r_[1, unique_precisions]
    if unique_recalls[-1] != 1:
        unique_recalls = np.r_[unique_recalls, 1]
        unique_precisions = np.r_[unique_precisions, 0]

    return unique_precisions, unique_recalls, thresholds


# def calculate_signal_background_metrics(y_true, y_pred, thresholds, sample_weights, total_luminosity, cross_sections, use_real_event_percentiles):
#     signal = y_true == 1
#     background = y_true == 0

#     real_weights = sample_weights

#     if use_real_event_percentiles:
#         # Sort predictions and weights together
#         sorted_indices = np.argsort(y_pred)
#         sorted_pred = y_pred[sorted_indices]
#         sorted_weights = real_weights[sorted_indices]

#         # Calculate cumulative weights
#         cumulative_weights = np.cumsum(sorted_weights)
#         total_weight = cumulative_weights[-1]

#         # Calculate thresholds based on weighted percentiles
#         weighted_thresholds = []
#         for threshold in thresholds:
#             idx = np.searchsorted(cumulative_weights / total_weight, threshold)
#             weighted_thresholds.append(sorted_pred[idx])
#         thresholds = np.array(weighted_thresholds)
#     else:
#         # Use regular percentiles when the flag is False
#         thresholds = np.quantile(y_pred, thresholds)

#     signal_eff = []
#     bkg_rej = []
#     significance = []
#     s_over_b = []

#     for threshold in thresholds:
#         y_pred_binary = (y_pred > threshold).astype(int)

#         s = np.sum(y_pred_binary[signal] * real_weights[signal])
#         b = np.sum(y_pred_binary[background] * real_weights[background])

#         signal_eff.append(np.sum(y_pred_binary[signal] * real_weights[signal]) / np.sum(real_weights[signal]))
#         bkg_rej.append(1 - (np.sum(y_pred_binary[background] * real_weights[background]) / np.sum(real_weights[background])))
#         significance.append(s / np.sqrt(s + b) if (s + b) > 0 else 0)
#         s_over_b.append(s / b if b > 0 else np.inf)

#     return {
#         "signal_efficiency": np.array(signal_eff),
#         "background_rejection": np.array(bkg_rej),
#         "significance": np.array(significance),
#         "s_over_b": np.array(s_over_b),
#         "thresholds": thresholds
#     }

# def calculate_counts_for_score_cuts(y_true, y_pred, mnj, truth, cross_sections, total_luminosity, cuts, use_real_event_percentiles):
#     results = {}

#     if use_real_event_percentiles:
#         weights = np.array([cross_sections[t] for t in truth])
#         sorted_indices = np.argsort(y_pred)
#         sorted_pred = y_pred[sorted_indices]
#         sorted_weights = weights[sorted_indices]
#         cumulative_weights = np.cumsum(sorted_weights)
#         total_weight = cumulative_weights[-1]

#     for cut in cuts:
#         if use_real_event_percentiles:
#             idx = np.searchsorted(cumulative_weights / total_weight, cut)
#             threshold = sorted_pred[idx]
#         else:
#             threshold = np.quantile(y_pred, cut)

#         scores = {}
#         for key in np.unique(truth):
#             mask = (truth == key) & (y_pred > threshold)
#             scores[key] = mnj[mask]

#         counts = {
#             k: len(v) * total_luminosity * cross_sections[k] / np.sum(truth == k)
#             for k, v in scores.items()
#         }
#         results[cut] = counts

#     df_counts = pd.DataFrame(results)

#     # Ensure 'SIG:Suu' is the last row before summary rows
#     if 'SIG:Suu' in df_counts.index:
#         sig_row = df_counts.loc['SIG:Suu']
#         df_counts = pd.concat([df_counts.drop('SIG:Suu'), sig_row.to_frame().T])

#     bkg_counts = df_counts.loc[[idx for idx in df_counts.index if idx != 'SIG:Suu']].sum()
#     sig_counts = df_counts.loc['SIG:Suu'] if 'SIG:Suu' in df_counts.index else pd.Series(0, index=df_counts.columns)
#     s_over_b = sig_counts / bkg_counts

#     df_counts.loc['BKG:sum'] = bkg_counts
#     df_counts.loc['S/B'] = s_over_b

#     # Add row names as the first column
#     df_counts = df_counts.reset_index()
#     df_counts = df_counts.rename(columns={'index': 'Process'})

#     return df_counts


def calculate_signal_background_metrics(
    y_true,
    y_pred,
    thresholds,
    sample_weights,
    total_luminosity,
    cross_sections,
    use_real_event_percentiles,
    fixed_thresholds: dict[str, float] | None = None,
):
    signal = y_true == 1
    background = y_true == 0

    if fixed_thresholds is not None:
        thresholds = np.array([float(fixed_thresholds[str(t)]) for t in thresholds])
    elif use_real_event_percentiles:
        # Sort predictions and weights together
        sorted_indices = np.argsort(y_pred)
        sorted_pred = y_pred[sorted_indices]
        sorted_weights = sample_weights[sorted_indices]

        # Calculate cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]

        # Calculate thresholds based on weighted percentiles
        thresholds = np.array(
            [
                sorted_pred[np.searchsorted(cumulative_weights / total_weight, t)]
                for t in thresholds
            ]
        )
    else:
        # Use regular percentiles when the flag is False
        thresholds = np.quantile(y_pred, np.array(thresholds))

    signal_eff = []
    bkg_rej = []
    significance = []
    s_over_b = []

    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)

        s = np.sum(y_pred_binary[signal] * sample_weights[signal])
        b = np.sum(y_pred_binary[background] * sample_weights[background])

        signal_eff.append(
            np.sum(y_pred_binary[signal] * sample_weights[signal])
            / np.sum(sample_weights[signal])
        )
        bkg_rej.append(
            1
            - (
                np.sum(y_pred_binary[background] * sample_weights[background])
                / np.sum(sample_weights[background])
            )
        )
        significance.append(s / np.sqrt(s + b) if (s + b) > 0 else 0)
        s_over_b.append(s / b if b > 0 else np.inf)

    return {
        "signal_efficiency": np.array(signal_eff),
        "background_rejection": np.array(bkg_rej),
        "significance": np.array(significance),
        "s_over_b": np.array(s_over_b),
        "thresholds": thresholds,
    }


def calculate_counts_for_score_cuts(
    y_true,
    y_pred,
    mnj,
    truth,
    cross_sections,
    total_luminosity,
    cuts,
    use_real_event_percentiles,
    fixed_thresholds: dict[str, float] | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    results = {}
    uncertainties = {}
    raw_counts = {}

    truth_array = np.asarray(truth)
    keys = np.unique(truth_array)
    truth_masks = {key: truth_array == key for key in keys}
    truth_counts = {key: int(truth_masks[key].sum()) for key in keys}
    process_weights = {
        key: total_luminosity * cross_sections[key] / truth_counts[key] for key in keys
    }

    if fixed_thresholds is None and use_real_event_percentiles:
        weights = np.array([process_weights[t] for t in truth_array])
        sorted_indices = np.argsort(y_pred)
        sorted_pred = y_pred[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]

    for cut in cuts:
        if fixed_thresholds is not None and str(cut) in fixed_thresholds:
            threshold = float(fixed_thresholds[str(cut)])
        elif use_real_event_percentiles:
            threshold = sorted_pred[
                np.searchsorted(cumulative_weights / total_weight, cut)
            ]
        else:
            threshold = np.quantile(y_pred, cut)

        passed_mask = y_pred > threshold
        cut_raw_counts = {k: int((truth_masks[k] & passed_mask).sum()) for k in keys}
        counts = {k: cut_raw_counts[k] * process_weights[k] for k in keys}
        cut_uncertainties = {
            k: np.sqrt(cut_raw_counts[k] * (1.0 - cut_raw_counts[k] / truth_counts[k]))
            * process_weights[k]
            for k in keys
        }

        results[cut] = counts
        uncertainties[cut] = cut_uncertainties
        raw_counts[cut] = cut_raw_counts

    df_counts = pd.DataFrame(results)
    df_uncertainties = pd.DataFrame(uncertainties)
    df_raw_counts = pd.DataFrame(raw_counts)

    # Ensure 'SIG:Suu' is the last row before summary rows
    if "SIG:Suu" in df_counts.index:
        sig_row = df_counts.loc["SIG:Suu"]
        df_counts = pd.concat([df_counts.drop("SIG:Suu"), sig_row.to_frame().T])

        sig_unc_row = df_uncertainties.loc["SIG:Suu"]
        df_uncertainties = pd.concat(
            [df_uncertainties.drop("SIG:Suu"), sig_unc_row.to_frame().T]
        )

    bkg_indices = [idx for idx in df_counts.index if idx != "SIG:Suu"]
    bkg_counts = df_counts.loc[bkg_indices].sum()
    bkg_uncertainties = np.sqrt((df_uncertainties.loc[bkg_indices] ** 2).sum())

    sig_counts = (
        df_counts.loc["SIG:Suu"]
        if "SIG:Suu" in df_counts.index
        else pd.Series(0.0, index=df_counts.columns)
    )
    sig_uncertainties = (
        df_uncertainties.loc["SIG:Suu"]
        if "SIG:Suu" in df_uncertainties.index
        else pd.Series(0.0, index=df_uncertainties.columns)
    )

    s_over_b = sig_counts / bkg_counts
    s_over_b_uncertainties = pd.Series(
        np.where(
            bkg_counts > 0,
            np.sqrt(sig_uncertainties**2 + (s_over_b * bkg_uncertainties) ** 2)
            / bkg_counts,
            np.where(sig_counts > 0, np.inf, 0.0),
        ),
        index=df_counts.columns,
    )

    df_counts.loc["BKG:sum"] = bkg_counts
    df_counts.loc["S/B"] = s_over_b

    df_uncertainties.loc["BKG:sum"] = bkg_uncertainties
    df_uncertainties.loc["S/B"] = s_over_b_uncertainties

    # Add row names as the first column
    df_counts = df_counts.reset_index()
    df_counts = df_counts.rename(columns={"index": "Process"})

    df_uncertainties = df_uncertainties.reset_index()
    df_uncertainties = df_uncertainties.rename(columns={"index": "Process"})

    df_raw_counts = df_raw_counts.reset_index()
    df_raw_counts = df_raw_counts.rename(columns={"index": "Process"})

    return df_counts, df_raw_counts, df_uncertainties
