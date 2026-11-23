"Physics-weighted training weights and tail-sensitive evaluation metrics."

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

DEFAULT_SIGNAL_EFFICIENCIES: tuple[float, ...] = (
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
)

DEFAULT_PERCENTILE_CUTS: tuple[float, ...] = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.875,
    0.90,
    0.925,
    0.95,
)


def _is_signal_label(label: Any) -> bool:
    return "SIG" in str(label)


def compute_process_weights(
    truth: np.ndarray | Iterable[str],
    cross_sections: Mapping[str, float],
    total_luminosity: float = 3000.0,
) -> np.ndarray:
    """Compute physical event weights ``L * sigma_p / N_p`` for each process,
    where:
    - ``L`` is the total luminosity,
    - ``sigma_p`` is the cross-section for process ``p``,
    - ``N_p`` is the number of events for process ``p``
    ."""
    truth_arr = np.asarray(truth)
    if len(truth_arr) == 0:
        return np.empty(0, dtype=np.float64)

    keys, inverse, counts = np.unique(
        truth_arr, return_inverse=True, return_counts=True
    )
    per_key_weight = np.empty(len(keys), dtype=np.float64)
    lumi = float(total_luminosity)
    for idx, (key, count) in enumerate(zip(keys, counts)):
        key_str = str(key)
        if key_str not in cross_sections:
            raise KeyError(f"Missing cross-section for process '{key_str}'")
        per_key_weight[idx] = lumi * float(cross_sections[key_str]) / int(count)
    return per_key_weight[inverse]


def compute_training_weights(
    truth: np.ndarray | Iterable[str],
    cross_sections: Mapping[str, float],
    mode: str = "none",
    alpha: float = 1.0,
    balance_classes: bool = True,
    signal_weight_ratio: float = 1.0,
) -> np.ndarray | None:
    """Compute normalized per-event training weights.

    Modes:
    - ``"none"``: returns ``None``.
    - ``"class_balanced"``: uniform weights within signal and background, rescaled
      so signal-to-background weight ratio equals ``signal_weight_ratio`` when
      ``balance_classes=True`` and ``mean(weights) == 1.0``.
    - ``"physics"``: per-process weight proportional to ``(sigma_p / N_p) ** alpha``.
      ``alpha=1.0`` preserves physical cross-section proportions within each class;
      ``alpha=0.0`` matches ``"class_balanced"``. Normalized to ``mean(weights) == 1.0``.
    """
    if mode not in {"none", "class_balanced", "physics"}:
        raise ValueError(
            f"Unsupported training weight mode '{mode}'. "
            "Expected one of {'none', 'class_balanced', 'physics'}."
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    if float(signal_weight_ratio) <= 0.0:
        raise ValueError(f"signal_weight_ratio must be > 0, got {signal_weight_ratio}")

    if mode == "none":
        return None

    truth_arr = np.asarray(truth)
    n_events = len(truth_arr)
    if n_events == 0:
        return np.empty(0, dtype=np.float64)

    keys, inverse, counts = np.unique(
        truth_arr, return_inverse=True, return_counts=True
    )
    is_sig_key = np.array([_is_signal_label(k) for k in keys], dtype=bool)
    sig_mask = is_sig_key[inverse]
    bkg_mask = ~sig_mask

    alpha_f = float(alpha)
    if mode == "class_balanced" or alpha_f == 0.0:
        per_key_raw = np.ones(len(keys), dtype=np.float64)
    else:
        per_key_raw = np.empty(len(keys), dtype=np.float64)
        for idx, (key, count) in enumerate(zip(keys, counts)):
            key_str = str(key)
            if key_str not in cross_sections:
                raise KeyError(f"Missing cross-section for process '{key_str}'")
            xsec = float(cross_sections[key_str])
            if xsec < 0.0:
                raise ValueError(
                    f"Cross-section must be non-negative for '{key_str}', got {xsec}"
                )
            per_key_raw[idx] = np.power(xsec / int(count), alpha_f)

    raw_weights = per_key_raw[inverse]

    if balance_classes:
        if not np.any(sig_mask) or not np.any(bkg_mask):
            raise ValueError(
                "Cannot balance classes when either signal or background has zero events."
            )
        sig_sum = float(raw_weights[sig_mask].sum())
        bkg_sum = float(raw_weights[bkg_mask].sum())
        if sig_sum <= 0.0 or bkg_sum <= 0.0:
            raise ValueError("Signal and background raw weight sums must be positive.")

        ratio = float(signal_weight_ratio)
        target_sig = float(n_events) * ratio / (1.0 + ratio)
        target_bkg = float(n_events) / (1.0 + ratio)

        weights = np.empty(n_events, dtype=np.float64)
        weights[sig_mask] = raw_weights[sig_mask] * (target_sig / sig_sum)
        weights[bkg_mask] = raw_weights[bkg_mask] * (target_bkg / bkg_sum)
        return weights

    total_sum = float(raw_weights.sum())
    if total_sum <= 0.0:
        raise ValueError("Total raw weight sum must be positive.")
    return raw_weights * (float(n_events) / total_sum)


def summarize_weight_distribution(
    truth: np.ndarray | Iterable[str],
    weights: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Summarize weight shares and Kish effective sample sizes per process and class."""
    truth_arr = np.asarray(truth)
    n_events = len(truth_arr)
    if weights is None:
        w = np.ones(n_events, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != n_events:
            raise ValueError(
                f"Length mismatch between truth ({n_events}) and weights ({len(w)})"
            )

    if n_events == 0:
        return {}

    keys, inverse, _ = np.unique(truth_arr, return_inverse=True, return_counts=True)
    is_sig_key = np.array([_is_signal_label(k) for k in keys], dtype=bool)
    sig_mask = is_sig_key[inverse]
    bkg_mask = ~sig_mask

    total_w = float(w.sum())
    sig_w = float(w[sig_mask].sum())
    bkg_w = float(w[bkg_mask].sum())

    def _mask_stats(
        mask: np.ndarray, class_total: float | None = None
    ) -> dict[str, float]:
        count = int(mask.sum())
        wm = w[mask]
        wsum = float(wm.sum())
        w2sum = float(np.sum(wm * wm))
        n_eff = float((wsum * wsum) / w2sum) if w2sum > 0.0 else 0.0
        stats = {
            "count": count,
            "weight_sum": wsum,
            "weight_share": float(wsum / total_w) if total_w > 0.0 else 0.0,
            "mean_weight": float(wsum / count) if count > 0 else 0.0,
            "n_eff": n_eff,
        }
        if class_total is not None:
            stats["class_weight_share"] = (
                float(wsum / class_total) if class_total > 0.0 else 0.0
            )
        return stats

    summary: dict[str, dict[str, float]] = {}
    for idx, key in enumerate(keys):
        key_str = str(key)
        class_total = sig_w if is_sig_key[idx] else bkg_w
        summary[key_str] = _mask_stats(inverse == idx, class_total=class_total)

    summary["SIG"] = _mask_stats(sig_mask)
    summary["BKG"] = _mask_stats(bkg_mask)
    summary["ALL"] = _mask_stats(np.ones(n_events, dtype=bool))
    return summary


def asymptotic_z(s: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Asymptotic expected discovery significance sqrt(2*((S+B)*ln(1+S/B)-S))."""
    is_scalar = np.ndim(s) == 0 and np.ndim(b) == 0
    s_arr, b_arr = np.broadcast_arrays(
        np.atleast_1d(np.asarray(s, dtype=np.float64)),
        np.atleast_1d(np.asarray(b, dtype=np.float64)),
    )
    z = np.zeros_like(s_arr, dtype=np.float64)
    pos = s_arr > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        b_safe = np.where(b_arr > 0, b_arr, 1.0)
        f = (s_arr[pos] + b_safe[pos]) * np.log1p(s_arr[pos] / b_safe[pos]) - s_arr[pos]
        z[pos] = np.sqrt(2.0 * np.maximum(f, 0.0))
    z[pos & (b_arr <= 0)] = np.inf
    return float(z[0]) if is_scalar else z


def asymptotic_z_error(
    s: np.ndarray | float,
    b: np.ndarray | float,
    ds: np.ndarray | float,
    db: np.ndarray | float,
) -> np.ndarray | float:
    """Gaussian uncertainty propagation of (dS, dB) through ``asymptotic_z``."""
    is_scalar = (
        np.ndim(s) == 0 and np.ndim(b) == 0 and np.ndim(ds) == 0 and np.ndim(db) == 0
    )
    s_arr, b_arr, ds_arr, db_arr = np.broadcast_arrays(
        np.atleast_1d(np.asarray(s, dtype=np.float64)),
        np.atleast_1d(np.asarray(b, dtype=np.float64)),
        np.atleast_1d(np.asarray(ds, dtype=np.float64)),
        np.atleast_1d(np.asarray(db, dtype=np.float64)),
    )
    z = np.atleast_1d(np.asarray(asymptotic_z(s_arr, b_arr), dtype=np.float64))
    err = np.zeros_like(s_arr, dtype=np.float64)
    ok = (s_arr > 0) & (b_arr > 0) & np.isfinite(z) & (z > 0)
    x = np.divide(s_arr, b_arr, out=np.zeros_like(s_arr), where=b_arr > 0)
    df_ds = np.log1p(x)
    df_db = np.log1p(x) - x
    err[ok] = (
        np.sqrt((df_ds[ok] * ds_arr[ok]) ** 2 + (df_db[ok] * db_arr[ok]) ** 2) / z[ok]
    )
    return float(err[0]) if is_scalar else err


def weighted_partial_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray,
    max_fpr: float = 2e-3,
) -> float:
    """Normalized partial area under the weighted ROC curve for FPR in [0, max_fpr].

    Returns ``integral_0^{max_fpr} TPR(FPR) dFPR / max_fpr`` in ``[0, 1]``.
    """
    max_fpr_f = float(max_fpr)
    if not (0.0 < max_fpr_f <= 1.0):
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")

    fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=sample_weights)
    if len(fpr) == 0:
        return 0.0

    if fpr[0] > 0.0:
        fpr = np.r_[0.0, fpr]
        tpr = np.r_[0.0, tpr]

    mask = fpr <= max_fpr_f
    fpr_sub = fpr[mask]
    tpr_sub = tpr[mask]

    if len(fpr_sub) == 0:
        return 0.0

    if fpr_sub[-1] < max_fpr_f and fpr[-1] > fpr_sub[-1]:
        tpr_at_max = float(np.interp(max_fpr_f, fpr, tpr))
        fpr_sub = np.r_[fpr_sub, max_fpr_f]
        tpr_sub = np.r_[tpr_sub, tpr_at_max]

    area = float(np.trapezoid(tpr_sub, fpr_sub))
    return float(np.clip(area / max_fpr_f, 0.0, 1.0))


def _binary_logloss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> float:
    p_clip = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    bce = -(y_true * np.log(p_clip) + (1.0 - y_true) * np.log(1.0 - p_clip))
    if sample_weights is None:
        return float(np.mean(bce))
    return float(np.average(bce, weights=sample_weights))


def _compute_cut_row(
    passed_mask: np.ndarray,
    truth_masks: Mapping[str, np.ndarray],
    truth_counts: Mapping[str, int],
    process_weights: Mapping[str, float],
    sig_key: str,
    bkg_keys: list[str],
    total_sig_yield: float,
    total_bkg_yield: float,
) -> dict[str, Any]:
    raw_counts = {k: int((truth_masks[k] & passed_mask).sum()) for k in truth_masks}
    yields = {k: raw_counts[k] * process_weights[k] for k in truth_masks}
    uncs = {
        k: float(
            np.sqrt(raw_counts[k] * (1.0 - raw_counts[k] / truth_counts[k]))
            * process_weights[k]
        )
        for k in truth_masks
    }

    s = float(yields.get(sig_key, 0.0))
    ds = float(uncs.get(sig_key, 0.0))
    b = float(sum(yields[k] for k in bkg_keys))
    db = float(np.sqrt(sum(uncs[k] ** 2 for k in bkg_keys)))

    s_over_b = float(s / b) if b > 0.0 else (float("inf") if s > 0.0 else 0.0)
    z = float(asymptotic_z(s, b))
    dz = float(asymptotic_z_error(s, b, ds, db))

    sig_eff = float(s / total_sig_yield) if total_sig_yield > 0.0 else 0.0
    bkg_rej = float(1.0 - b / total_bkg_yield) if total_bkg_yield > 0.0 else 1.0

    raw_qcd = int(raw_counts.get("BKG:qcd", 0))
    raw_bkg_total = int(sum(raw_counts[k] for k in bkg_keys))

    return {
        "s": s,
        "ds": ds,
        "b": b,
        "db": db,
        "s_over_b": s_over_b,
        "z_asymp": z,
        "z_asymp_err": dz,
        "signal_efficiency": sig_eff,
        "bkg_rejection": bkg_rej,
        "raw_qcd": raw_qcd,
        "raw_bkg_total": raw_bkg_total,
        "per_process_yields": yields,
        "per_process_raw_counts": raw_counts,
    }


def evaluate_tail_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    truth: np.ndarray | Iterable[str],
    cross_sections: Mapping[str, float],
    total_luminosity: float = 3000.0,
    signal_efficiencies: Iterable[float] = DEFAULT_SIGNAL_EFFICIENCIES,
    max_fpr: float = 2e-3,
    percentile_cuts: Iterable[float] = DEFAULT_PERCENTILE_CUTS,
) -> dict[str, Any]:
    """Compute tail-sensitive physics and ML metrics on evaluation predictions."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    truth_arr = np.asarray(truth)

    if not (len(y_true_arr) == len(y_pred_arr) == len(truth_arr)):
        raise ValueError("y_true, y_pred, and truth must have identical lengths.")

    phys_weights = compute_process_weights(
        truth_arr, cross_sections, total_luminosity=total_luminosity
    )
    balanced_phys_weights = compute_training_weights(
        truth_arr,
        cross_sections,
        mode="physics",
        alpha=1.0,
        balance_classes=True,
    )

    roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    avg_precision = float(average_precision_score(y_true_arr, y_pred_arr))
    unweighted_ll = _binary_logloss(y_true_arr, y_pred_arr)
    weighted_ll = _binary_logloss(y_true_arr, y_pred_arr, balanced_phys_weights)
    partial_auc = weighted_partial_roc_auc(
        y_true_arr, y_pred_arr, phys_weights, max_fpr=max_fpr
    )

    keys, counts = np.unique(truth_arr, return_counts=True)
    key_strs = [str(k) for k in keys]
    truth_masks = {k_str: truth_arr == k for k, k_str in zip(keys, key_strs)}
    truth_counts = {k_str: int(c) for k_str, c in zip(key_strs, counts)}
    process_weights = {
        k_str: float(total_luminosity)
        * float(cross_sections[k_str])
        / truth_counts[k_str]
        for k_str in key_strs
    }

    sig_keys = [k for k in key_strs if _is_signal_label(k)]
    if len(sig_keys) != 1:
        raise ValueError(f"Expected exactly 1 signal key, found {sig_keys}")
    sig_key = sig_keys[0]
    bkg_keys = [k for k in key_strs if k != sig_key]

    total_sig_yield = float(truth_counts[sig_key] * process_weights[sig_key])
    total_bkg_yield = float(sum(truth_counts[k] * process_weights[k] for k in bkg_keys))
    max_bkg_process_weight = (
        max(process_weights[k] for k in bkg_keys) if bkg_keys else 0.0
    )

    sig_preds = y_pred_arr[y_true_arr == 1]

    eff_grid: list[dict[str, Any]] = []
    z_for_mean: list[float] = []
    has_zero_bkg_point = False

    for eps in signal_efficiencies:
        eps_f = float(eps)
        thr = float(np.quantile(sig_preds, 1.0 - eps_f))
        row = _compute_cut_row(
            y_pred_arr > thr,
            truth_masks,
            truth_counts,
            process_weights,
            sig_key,
            bkg_keys,
            total_sig_yield,
            total_bkg_yield,
        )
        row["target_signal_efficiency"] = eps_f
        row["threshold"] = thr
        eff_grid.append(row)

        if row["b"] <= 0.0:
            has_zero_bkg_point = True
            b_floor = max_bkg_process_weight if max_bkg_process_weight > 0.0 else 1e-6
            z_for_mean.append(float(asymptotic_z(row["s"], b_floor)))
        else:
            z_for_mean.append(row["z_asymp"])

    mean_z_eff_grid = float(np.mean(z_for_mean)) if z_for_mean else 0.0

    percentile_rows: list[dict[str, Any]] = []
    for cut in percentile_cuts:
        cut_f = float(cut)
        thr = float(np.quantile(y_pred_arr, cut_f))
        row = _compute_cut_row(
            y_pred_arr > thr,
            truth_masks,
            truth_counts,
            process_weights,
            sig_key,
            bkg_keys,
            total_sig_yield,
            total_bkg_yield,
        )
        row["percentile_cut"] = cut_f
        row["threshold"] = thr
        percentile_rows.append(row)

    return {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "unweighted_logloss": unweighted_ll,
        "weighted_logloss": weighted_ll,
        "partial_weighted_roc_auc": partial_auc,
        "max_fpr": float(max_fpr),
        "mean_z_eff_grid": mean_z_eff_grid,
        "has_zero_bkg_point": has_zero_bkg_point,
        "eff_grid": eff_grid,
        "percentile_cuts": percentile_rows,
    }


def evaluate_overfitting_diagnostics(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    truth_train: np.ndarray | Iterable[str],
    y_eval_true: np.ndarray,
    y_eval_pred: np.ndarray,
    truth_eval: np.ndarray | Iterable[str],
    cross_sections: Mapping[str, float],
    total_luminosity: float = 3000.0,
    max_fpr: float = 2e-3,
) -> dict[str, float]:
    """Compare train vs. evaluation discrimination and log-loss to flag overfitting."""
    y_tr = np.asarray(y_train_true, dtype=np.int64)
    p_tr = np.asarray(y_train_pred, dtype=np.float64)
    t_tr = np.asarray(truth_train)

    y_ev = np.asarray(y_eval_true, dtype=np.int64)
    p_ev = np.asarray(y_eval_pred, dtype=np.float64)
    t_ev = np.asarray(truth_eval)

    w_tr_phys = compute_process_weights(t_tr, cross_sections, total_luminosity)
    w_ev_phys = compute_process_weights(t_ev, cross_sections, total_luminosity)
    w_tr_bal = compute_training_weights(t_tr, cross_sections, mode="physics", alpha=1.0)
    w_ev_bal = compute_training_weights(t_ev, cross_sections, mode="physics", alpha=1.0)

    train_auc = float(roc_auc_score(y_tr, p_tr))
    eval_auc = float(roc_auc_score(y_ev, p_ev))
    train_ll = _binary_logloss(y_tr, p_tr)
    eval_ll = _binary_logloss(y_ev, p_ev)
    train_wll = _binary_logloss(y_tr, p_tr, w_tr_bal)
    eval_wll = _binary_logloss(y_ev, p_ev, w_ev_bal)
    train_pauc = weighted_partial_roc_auc(y_tr, p_tr, w_tr_phys, max_fpr=max_fpr)
    eval_pauc = weighted_partial_roc_auc(y_ev, p_ev, w_ev_phys, max_fpr=max_fpr)

    return {
        "train_roc_auc": train_auc,
        "eval_roc_auc": eval_auc,
        "roc_auc_gap": train_auc - eval_auc,
        "train_unweighted_logloss": train_ll,
        "eval_unweighted_logloss": eval_ll,
        "unweighted_logloss_gap": eval_ll - train_ll,
        "train_weighted_logloss": train_wll,
        "eval_weighted_logloss": eval_wll,
        "weighted_logloss_gap": eval_wll - train_wll,
        "train_partial_weighted_roc_auc": train_pauc,
        "eval_partial_weighted_roc_auc": eval_pauc,
        "partial_weighted_roc_auc_gap": train_pauc - eval_pauc,
    }
