"""Phase 1 benchmark and evaluation harness for MSuu hyperparameter tuning.

Loads cached (or freshly extracted) features, locks the outer holdout test split
matching the baseline run configuration, and evaluates XGBoost under three
validation protocols:
  - ``inner-cv``: Stratified K-fold CV strictly inside the 80% outer training set.
  - ``outer-eval``: Fit on the 80% outer training set (with internal val split) and
    evaluate on the locked 20% outer test set.
  - ``full-cv``: Full Stratified K-fold CV across 100% of events.

Usage::

    .venv/bin/python scripts/benchmark_hpo_baseline.py --mode outer-eval --n-jobs 32
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from sklearn.model_selection import StratifiedKFold

from diquark.data.feature_cache import load_or_extract_dataset
from diquark.data.preprocessor import Preprocessor
from diquark.evaluation.tuning_metrics import (
    compute_training_weights,
    evaluate_overfitting_diagnostics,
    evaluate_tail_metrics,
    summarize_weight_distribution,
)
from diquark.models.gradient_boosting import GradientBoostingModel

app = typer.Typer(
    help="Phase 1 benchmark harness for diquark XGBoost evaluation and HPO prep.",
    add_completion=False,
)


def _verify_reference_split(df_outer_test: pd.DataFrame, ref_dir: Path) -> dict[str, Any]:
    ref_path = ref_dir / "test_set.parquet"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference test_set.parquet not found at '{ref_path}'")

    ref_df = pd.read_parquet(ref_path)
    if len(df_outer_test) != len(ref_df):
        raise AssertionError(
            f"Outer test row count mismatch: got {len(df_outer_test)}, expected {len(ref_df)}"
        )
    if list(df_outer_test.columns) != list(ref_df.columns):
        raise AssertionError(
            f"Outer test column mismatch:\n  got: {list(df_outer_test.columns)}\n  ref: {list(ref_df.columns)}"
        )

    got_counts = {str(k): int(v) for k, v in df_outer_test["Truth"].value_counts().items()}
    ref_counts = {str(k): int(v) for k, v in ref_df["Truth"].value_counts().items()}
    if got_counts != ref_counts:
        raise AssertionError(
            f"Outer test Truth count mismatch:\n  got: {got_counts}\n  ref: {ref_counts}"
        )

    return {"verified": True, "n_rows": len(ref_df), "truth_counts": ref_counts}


def _fit_and_predict_split(
    df_fit_src: pd.DataFrame,
    df_eval_src: pd.DataFrame,
    df_val_src: pd.DataFrame | None,
    feature_cols: list[str],
    preprocessor: Preprocessor,
    cross_sections: dict[str, float],
    xgb_config: dict[str, Any],
    weight_mode: str,
    weight_alpha: float,
    balance_classes: bool,
    apply_weights_to_eval_set: bool,
    oversample: bool,
) -> dict[str, Any]:
    prepared = preprocessor.prepare_split_arrays(
        df_fit_src,
        df_eval_src,
        feature_cols,
        df_val=df_val_src,
        oversample_signal=oversample,
    )

    sample_weight = compute_training_weights(
        prepared["df_train"]["Truth"],
        cross_sections,
        mode=weight_mode,
        alpha=weight_alpha,
        balance_classes=balance_classes,
    )

    if df_val_src is not None:
        X_val, y_val = prepared["X_val"], prepared["y_val"]
        val_sample_weight = (
            compute_training_weights(
                prepared["df_val"]["Truth"],
                cross_sections,
                mode=weight_mode,
                alpha=weight_alpha,
                balance_classes=balance_classes,
            )
            if apply_weights_to_eval_set
            else None
        )
    else:
        X_val, y_val = prepared["X_test"], prepared["y_test"]
        val_sample_weight = (
            compute_training_weights(
                prepared["df_test"]["Truth"],
                cross_sections,
                mode=weight_mode,
                alpha=weight_alpha,
                balance_classes=balance_classes,
            )
            if (apply_weights_to_eval_set and weight_mode != "none")
            else None
        )

    model = GradientBoostingModel(xgb_config)
    model.build(prepared["X_train"].shape[1])

    t0 = time.perf_counter()
    model.train(
        prepared["X_train"],
        prepared["y_train"],
        X_val,
        y_val,
        sample_weight=sample_weight,
        val_sample_weight=val_sample_weight,
    )
    fit_time = time.perf_counter() - t0

    eval_pred = model.predict(prepared["X_test"])

    # Predict on un-duplicated fit events for clean train-vs-eval diagnostics
    if oversample:
        X_fit_unique = prepared["scaler"].transform(df_fit_src[feature_cols].to_numpy())
        y_fit_unique = df_fit_src["target"].to_numpy()
        truth_fit_unique = df_fit_src["Truth"].to_numpy()
    else:
        X_fit_unique = prepared["X_train"]
        y_fit_unique = prepared["y_train"]
        truth_fit_unique = prepared["df_train"]["Truth"].to_numpy()
    train_pred = model.predict(X_fit_unique)

    return {
        "eval_pred": eval_pred,
        "train_pred": train_pred,
        "y_train_unique": y_fit_unique,
        "truth_train_unique": truth_fit_unique,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
        "fit_time_s": fit_time,
        "weight_summary": summarize_weight_distribution(
            prepared["df_train"]["Truth"], sample_weight
        ),
    }


def _print_report(report: dict[str, Any]) -> None:
    metrics = report["tail_metrics"]
    ov = report["overfitting"]
    print("\n" + "=" * 80)
    print(
        f"Benchmark mode={report['mode']} | weight_mode={report['training']['weight_mode']} "
        f"(alpha={report['training']['weight_alpha']}) | oversample={report['training']['oversample']} | "
        f"device={report['xgb_config']['device']}"
    )
    print(
        f"Load/cache={report['timing']['data_load_s']:.2f}s (cache_hit={report['data']['cache_hit']}) | "
        f"Total fit={report['timing']['total_fit_s']:.2f}s"
    )
    if report["folds"]:
        iters = [f["best_iteration"] for f in report["folds"]]
        print(f"Fold best_iteration(s): {iters}")

    print("\n--- Summary Metrics ---")
    print(
        f"  mean_z_eff_grid          = {metrics['mean_z_eff_grid']:.4f} "
        f"(has_zero_bkg_point={metrics['has_zero_bkg_point']})"
    )
    print(
        f"  partial_weighted_roc_auc = {metrics['partial_weighted_roc_auc']:.6f} "
        f"(FPR <= {metrics['max_fpr']:.1e})"
    )
    print(f"  weighted_logloss         = {metrics['weighted_logloss']:.6f}")
    print(f"  unweighted_logloss       = {metrics['unweighted_logloss']:.6f}")
    print(f"  roc_auc                  = {metrics['roc_auc']:.6f}")
    print(f"  average_precision        = {metrics['average_precision']:.6f}")

    print("\n--- Overfitting Diagnostics (Train vs. Eval) ---")
    print(
        f"  ROC AUC:          train={ov['train_roc_auc']:.6f}, eval={ov['eval_roc_auc']:.6f}, "
        f"gap={ov['roc_auc_gap']:+.6f}"
    )
    print(
        f"  Partial wROC AUC: train={ov['train_partial_weighted_roc_auc']:.6f}, "
        f"eval={ov['eval_partial_weighted_roc_auc']:.6f}, "
        f"gap={ov['partial_weighted_roc_auc_gap']:+.6f}"
    )
    print(
        f"  Weighted logloss: train={ov['train_weighted_logloss']:.6f}, "
        f"eval={ov['eval_weighted_logloss']:.6f}, gap={ov['weighted_logloss_gap']:+.6f}"
    )
    print(
        f"  Unweighted LL:    train={ov['train_unweighted_logloss']:.6f}, "
        f"eval={ov['eval_unweighted_logloss']:.6f}, gap={ov['unweighted_logloss_gap']:+.6f}"
    )

    ws = report.get("weight_summary", {})
    if ws:
        print("\n--- Training Weight Distribution (first/outer split) ---")
        for key in sorted(ws.keys()):
            if key in {"ALL", "SIG", "BKG"}:
                continue
            s = ws[key]
            print(
                f"  {key:18s}: n={int(s['count']):8d}, share={s['weight_share']:.4f}, "
                f"class_share={s.get('class_weight_share', 0.0):.4f}, n_eff={s['n_eff']:.1f}"
            )
        for key in ("SIG", "BKG", "ALL"):
            if key in ws:
                s = ws[key]
                print(
                    f"  {key:18s}: n={int(s['count']):8d}, share={s['weight_share']:.4f}, "
                    f"n_eff={s['n_eff']:.1f}"
                )

    print("\n--- Fixed Signal-Efficiency Grid ---")
    print(
        f"  {'eps_target':>10s} {'eps_actual':>10s} {'S':>8s} {'B':>10s} "
        f"{'raw_QCD':>8s} {'S/B':>8s} {'Z_asymp':>14s} {'bkg_rej':>10s}"
    )
    for row in metrics["eff_grid"]:
        z_str = (
            f"{row['z_asymp']:6.3f}±{row['z_asymp_err']:5.3f}"
            if np.isfinite(row["z_asymp"])
            else "   inf±  nan"
        )
        sb_str = f"{row['s_over_b']:8.3f}" if np.isfinite(row["s_over_b"]) else "     inf"
        print(
            f"  {row['target_signal_efficiency']:10.2f} {row['signal_efficiency']:10.4f} "
            f"{row['s']:8.3f} {row['b']:10.3f} {row['raw_qcd']:8d} "
            f"{sb_str} {z_str:>14s} {row['bkg_rejection']:10.6f}"
        )

    print("\n--- Standard Percentile Cuts (Ref Comparison) ---")
    print(
        f"  {'cut':>7s} {'eps_S':>8s} {'S':>8s} {'B':>10s} "
        f"{'raw_QCD':>8s} {'S/B':>8s} {'Z_asymp':>14s}"
    )
    for row in metrics["percentile_cuts"]:
        z_str = (
            f"{row['z_asymp']:6.3f}±{row['z_asymp_err']:5.3f}"
            if np.isfinite(row["z_asymp"])
            else "   inf±  nan"
        )
        sb_str = f"{row['s_over_b']:8.3f}" if np.isfinite(row["s_over_b"]) else "     inf"
        print(
            f"  {row['percentile_cut']:7.3f} {row['signal_efficiency']:8.4f} "
            f"{row['s']:8.3f} {row['b']:10.3f} {row['raw_qcd']:8d} "
            f"{sb_str} {z_str:>14s}"
        )
    print("=" * 80)


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(help="Run config YAML file."),
    ] = Path("config/ChiChi_ManyJets/MSuu_8000/run_wb-ht_jjb-bbt.yaml"),
    cache_dir: Annotated[
        Path,
        typer.Option(help="Directory to cache extracted feature Parquet files."),
    ] = Path("cache/features"),
    mode: Annotated[
        str,
        typer.Option(help="Evaluation mode: 'inner-cv', 'outer-eval', or 'full-cv'."),
    ] = "inner-cv",
    n_folds: Annotated[
        int | None,
        typer.Option(help="Number of CV folds (default: 4 for inner-cv, 5 for full-cv)."),
    ] = None,
    val_size: Annotated[
        float,
        typer.Option(help="Fraction of training fold held out for early stopping."),
    ] = 0.1,
    cv_random_state: Annotated[
        int,
        typer.Option(help="Random seed for StratifiedKFold splitting."),
    ] = 42,
    weight_mode: Annotated[
        str,
        typer.Option(help="Training weight mode: 'none', 'class_balanced', or 'physics'."),
    ] = "none",
    weight_alpha: Annotated[
        float,
        typer.Option(help="Tempering exponent alpha in [0, 1] for physics weights."),
    ] = 1.0,
    balance_classes: Annotated[
        bool,
        typer.Option("--balance-classes/--no-balance-classes", help="Balance total S and B weight."),
    ] = True,
    apply_weights_to_eval_set: Annotated[
        bool,
        typer.Option(
            "--weight-eval-set/--no-weight-eval-set",
            help="Pass validation sample weights to XGBoost early-stopping eval_set.",
        ),
    ] = True,
    oversample: Annotated[
        bool | None,
        typer.Option(
            "--oversample/--no-oversample",
            help="Oversample signal rows in training split (default: True for weight-mode=none, else False).",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="XGBoost device ('cpu', 'cuda', 'cuda:0', or 'gpu')."),
    ] = "cpu",
    n_jobs: Annotated[
        int,
        typer.Option(help="Number of CPU threads for XGBoost."),
    ] = 32,
    n_estimators: Annotated[
        int,
        typer.Option(help="Maximum number of boosting rounds."),
    ] = 1024,
    learning_rate: Annotated[
        float,
        typer.Option(help="XGBoost learning rate (eta)."),
    ] = 0.1,
    max_depth: Annotated[
        int,
        typer.Option(help="Maximum tree depth."),
    ] = 8,
    min_child_weight: Annotated[
        float,
        typer.Option(help="Minimum sum of instance weight needed in a child."),
    ] = 1.0,
    subsample: Annotated[
        float,
        typer.Option(help="Row subsample ratio."),
    ] = 1.0,
    colsample_bytree: Annotated[
        float,
        typer.Option(help="Column subsample ratio per tree."),
    ] = 1.0,
    reg_lambda: Annotated[
        float,
        typer.Option(help="L2 regularization term on weights."),
    ] = 1.0,
    reg_alpha: Annotated[
        float,
        typer.Option(help="L1 regularization term on weights."),
    ] = 0.0,
    gamma: Annotated[
        float,
        typer.Option(help="Minimum loss reduction to make a split."),
    ] = 0.0,
    max_bin: Annotated[
        int,
        typer.Option(help="Histogram bin count."),
    ] = 256,
    early_stopping_rounds: Annotated[
        int,
        typer.Option(help="Early stopping patience rounds (<= 0 disables)."),
    ] = 40,
    eval_metric: Annotated[
        str,
        typer.Option(help="XGBoost eval_metric for early stopping."),
    ] = "logloss",
    scale_pos_weight: Annotated[
        float,
        typer.Option(help="XGBoost scale_pos_weight."),
    ] = 1.0,
    random_state: Annotated[
        int,
        typer.Option(help="XGBoost random_state."),
    ] = 17,
    max_fpr: Annotated[
        float,
        typer.Option(help="Max weighted FPR for partial_weighted_roc_auc."),
    ] = 2e-3,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--quiet", help="Whether XGBoost prints per-iteration logs."),
    ] = False,
    verify_reference_dir: Annotated[
        Path | None,
        typer.Option(help="Optional reference results dir to verify outer test split parity."),
    ] = None,
    output_json: Annotated[
        Path | None,
        typer.Option(help="Optional output path to save full JSON benchmark report."),
    ] = None,
) -> None:
    if mode not in {"inner-cv", "outer-eval", "full-cv"}:
        raise typer.BadParameter(
            f"Unsupported mode '{mode}'. Choose from 'inner-cv', 'outer-eval', 'full-cv'."
        )

    effective_oversample = (weight_mode == "none") if oversample is None else oversample
    effective_folds = (
        n_folds if n_folds is not None else (4 if mode == "inner-cv" else 5)
    )

    t_data_start = time.perf_counter()
    df, cross_sections, data_meta = load_or_extract_dataset(config, cache_dir=cache_dir)
    data_load_s = time.perf_counter() - t_data_start

    prep_cfg = dict(data_meta.get("preprocessing", {}))
    prep_cfg["val_size"] = val_size
    prep_cfg["oversample_signal"] = effective_oversample
    preprocessor = Preprocessor(prep_cfg)
    feature_cols = [c for c in df.columns if c not in ("Truth", "target")]

    df_outer_train, df_outer_test = preprocessor.split_train_test_df(df)
    ref_verification = (
        _verify_reference_split(df_outer_test, verify_reference_dir)
        if verify_reference_dir is not None
        else None
    )
    if ref_verification is not None:
        print(
            f"Verified exact outer test split parity with '{verify_reference_dir}' "
            f"({ref_verification['n_rows']} rows)."
        )

    xgb_config: dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "gamma": gamma,
        "max_bin": max_bin,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": device,
        "eval_metric": eval_metric,
        "early_stopping_rounds": early_stopping_rounds if early_stopping_rounds > 0 else None,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbose": verbose,
    }

    luminosity = float(data_meta.get("total_luminosity", 3000.0))
    fold_summaries: list[dict[str, Any]] = []
    weight_summary: dict[str, dict[str, float]] = {}

    if mode == "outer-eval":
        df_fit, df_val = preprocessor.split_train_val_df(
            df_outer_train, val_size=val_size, random_state=preprocessor.random_state
        )
        res = _fit_and_predict_split(
            df_fit,
            df_outer_test,
            df_val,
            feature_cols,
            preprocessor,
            cross_sections,
            xgb_config,
            weight_mode,
            weight_alpha,
            balance_classes,
            apply_weights_to_eval_set,
            effective_oversample,
        )
        weight_summary = res["weight_summary"]
        tail_metrics = evaluate_tail_metrics(
            df_outer_test["target"].to_numpy(),
            res["eval_pred"],
            df_outer_test["Truth"].to_numpy(),
            cross_sections,
            total_luminosity=luminosity,
            max_fpr=max_fpr,
        )
        overfitting = evaluate_overfitting_diagnostics(
            res["y_train_unique"],
            res["train_pred"],
            res["truth_train_unique"],
            df_outer_test["target"].to_numpy(),
            res["eval_pred"],
            df_outer_test["Truth"].to_numpy(),
            cross_sections,
            total_luminosity=luminosity,
            max_fpr=max_fpr,
        )
        fold_summaries.append(
            {
                "fold": 1,
                "best_iteration": res["best_iteration"],
                "best_score": res["best_score"],
                "fit_time_s": res["fit_time_s"],
                "roc_auc": tail_metrics["roc_auc"],
                "partial_weighted_roc_auc": tail_metrics["partial_weighted_roc_auc"],
                "weighted_logloss": tail_metrics["weighted_logloss"],
                "mean_z_eff_grid": tail_metrics["mean_z_eff_grid"],
            }
        )
        total_fit_s = float(res["fit_time_s"])
    else:
        cv_df = df_outer_train if mode == "inner-cv" else df
        skf = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=cv_random_state
        )
        oof_pred = np.empty(len(cv_df), dtype=np.float64)
        fold_overfits: list[dict[str, float]] = []
        total_fit_s = 0.0

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(cv_df[feature_cols], cv_df["Truth"]), 1
        ):
            df_fold_train = cv_df.iloc[train_idx]
            df_fold_test = cv_df.iloc[test_idx]
            df_fit, df_val = preprocessor.split_train_val_df(
                df_fold_train,
                val_size=val_size,
                random_state=preprocessor.random_state + fold_idx,
            )
            res = _fit_and_predict_split(
                df_fit,
                df_fold_test,
                df_val,
                feature_cols,
                preprocessor,
                cross_sections,
                xgb_config,
                weight_mode,
                weight_alpha,
                balance_classes,
                apply_weights_to_eval_set,
                effective_oversample,
            )
            if fold_idx == 1:
                weight_summary = res["weight_summary"]

            oof_pred[test_idx] = res["eval_pred"]
            total_fit_s += float(res["fit_time_s"])

            fold_tail = evaluate_tail_metrics(
                df_fold_test["target"].to_numpy(),
                res["eval_pred"],
                df_fold_test["Truth"].to_numpy(),
                cross_sections,
                total_luminosity=luminosity,
                max_fpr=max_fpr,
            )
            fold_ov = evaluate_overfitting_diagnostics(
                res["y_train_unique"],
                res["train_pred"],
                res["truth_train_unique"],
                df_fold_test["target"].to_numpy(),
                res["eval_pred"],
                df_fold_test["Truth"].to_numpy(),
                cross_sections,
                total_luminosity=luminosity,
                max_fpr=max_fpr,
            )
            fold_overfits.append(fold_ov)
            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "best_iteration": res["best_iteration"],
                    "best_score": res["best_score"],
                    "fit_time_s": res["fit_time_s"],
                    "roc_auc": fold_tail["roc_auc"],
                    "partial_weighted_roc_auc": fold_tail["partial_weighted_roc_auc"],
                    "weighted_logloss": fold_tail["weighted_logloss"],
                    "mean_z_eff_grid": fold_tail["mean_z_eff_grid"],
                }
            )

        tail_metrics = evaluate_tail_metrics(
            cv_df["target"].to_numpy(),
            oof_pred,
            cv_df["Truth"].to_numpy(),
            cross_sections,
            total_luminosity=luminosity,
            max_fpr=max_fpr,
        )
        overfitting = {
            k: float(np.mean([ov[k] for ov in fold_overfits]))
            for k in fold_overfits[0]
        }

    report: dict[str, Any] = {
        "config_path": str(config),
        "mode": mode,
        "n_folds": 1 if mode == "outer-eval" else effective_folds,
        "data": {
            "n_total": len(df),
            "n_outer_train": len(df_outer_train),
            "n_outer_test": len(df_outer_test),
            "n_features": len(feature_cols),
            "cache_hit": data_meta["cache_hit"],
            "cache_key": data_meta["cache_key"],
            "cache_path": data_meta["cache_path"],
            "reference_verification": ref_verification,
        },
        "training": {
            "weight_mode": weight_mode,
            "weight_alpha": weight_alpha,
            "balance_classes": balance_classes,
            "apply_weights_to_eval_set": apply_weights_to_eval_set,
            "oversample": effective_oversample,
            "val_size": val_size,
            "cv_random_state": cv_random_state,
        },
        "xgb_config": xgb_config,
        "timing": {
            "data_load_s": data_load_s,
            "total_fit_s": total_fit_s,
        },
        "weight_summary": weight_summary,
        "folds": fold_summaries,
        "tail_metrics": tail_metrics,
        "overfitting": overfitting,
    }

    _print_report(report)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Saved benchmark report to {output_json}")


if __name__ == "__main__":
    app()
