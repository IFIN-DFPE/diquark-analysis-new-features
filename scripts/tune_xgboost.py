"""Distributed Optuna hyperparameter sweep worker for diquark XGBoost tuning."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import optuna
import pandas as pd
import typer
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold

from diquark.data.feature_cache import load_or_extract_dataset, resolve_dataset_spec
from diquark.data.preprocessor import Preprocessor
from diquark.evaluation.tuning_metrics import (
    compute_training_weights,
    evaluate_overfitting_diagnostics,
    evaluate_tail_metrics,
)
from diquark.models.gradient_boosting import GradientBoostingModel

app = typer.Typer(
    help="Run Optuna hyperparameter tuning trials against a shared JournalStorage study.",
    add_completion=False,
)

OBJECTIVE_DIRECTIONS: dict[str, str] = {
    "mean_z_eff_grid": "maximize",
    "partial_weighted_roc_auc": "maximize",
    "weighted_logloss": "minimize",
}

ANCHOR_TRIALS: list[tuple[str, dict[str, Any]]] = [
    (
        "physics_a0.25_default",
        {
            "weight_alpha": 0.25,
            "grow_policy": "depthwise",
            "max_depth": 8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "colsample_bylevel": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 1e-4,
            "max_bin": 256,
        },
    ),
    (
        "physics_a1.00_reg",
        {
            "weight_alpha": 1.0,
            "grow_policy": "depthwise",
            "max_depth": 6,
            "min_child_weight": 20.0,
            "gamma": 0.0,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "colsample_bylevel": 1.0,
            "reg_lambda": 10.0,
            "reg_alpha": 1e-4,
            "max_bin": 256,
        },
    ),
    (
        "class_balanced_default",
        {
            "weight_alpha": 0.0,
            "grow_policy": "depthwise",
            "max_depth": 8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "colsample_bylevel": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 1e-4,
            "max_bin": 256,
        },
    ),
]


def build_journal_storage(journal_path: Path) -> JournalStorage:
    """Create NFS-safe Optuna JournalStorage backed by symlink locking."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    lock_obj = JournalFileSymlinkLock(str(journal_path))
    backend = JournalFileBackend(str(journal_path), lock_obj=lock_obj)
    return JournalStorage(backend)


def create_or_load_study(
    study_name: str,
    journal_path: Path,
    objective: str = "mean_z_eff_grid",
    sampler_seed: int = 42,
    prune: bool = True,
    enqueue_anchors: bool = True,
) -> optuna.Study:
    """Create or load the shared Optuna study and enqueue baseline anchors if empty."""
    if objective not in OBJECTIVE_DIRECTIONS:
        raise ValueError(
            f"Unsupported objective '{objective}'. Choose from {sorted(OBJECTIVE_DIRECTIONS)}."
        )

    storage = build_journal_storage(journal_path)
    sampler = TPESampler(seed=sampler_seed, multivariate=True, n_startup_trials=15)
    pruner: optuna.pruners.BasePruner = (
        MedianPruner(n_startup_trials=15, n_warmup_steps=2)
        if prune
        else NopPruner()
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=OBJECTIVE_DIRECTIONS[objective],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if enqueue_anchors and len(study.trials) == 0:
        for anchor_name, params in ANCHOR_TRIALS:
            study.enqueue_trial(
                params,
                user_attrs={"anchor_name": anchor_name},
                skip_if_exists=True,
            )

    return study


def sample_trial_params(
    trial: optuna.Trial,
    *,
    n_estimators: int = 3000,
    early_stopping_rounds: int = 40,
    device: str = "cpu",
    n_jobs: int = 64,
    random_state: int = 17,
) -> tuple[dict[str, Any], float]:
    """Sample XGBoost hyperparameters and physics-weighting exponent from Optuna trial."""
    weight_alpha = float(trial.suggest_float("weight_alpha", 0.0, 1.0))
    grow_policy = str(trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]))

    xgb_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping_rounds if early_stopping_rounds > 0 else None,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "n_jobs": n_jobs,
        "random_state": random_state,
        "verbose": False,
        "grow_policy": grow_policy,
    }
    if grow_policy == "depthwise":
        xgb_params["max_depth"] = int(trial.suggest_int("max_depth", 3, 9))
    else:
        xgb_params["max_depth"] = 0
        xgb_params["max_leaves"] = int(
            trial.suggest_int("max_leaves", 16, 256, log=True)
        )

    xgb_params["min_child_weight"] = float(
        trial.suggest_float("min_child_weight", 1.0, 200.0, log=True)
    )
    xgb_params["gamma"] = float(trial.suggest_float("gamma", 0.0, 5.0))
    xgb_params["learning_rate"] = float(
        trial.suggest_float("learning_rate", 0.02, 0.15, log=True)
    )
    xgb_params["subsample"] = float(trial.suggest_float("subsample", 0.6, 1.0))
    xgb_params["colsample_bytree"] = float(
        trial.suggest_float("colsample_bytree", 0.4, 1.0)
    )
    xgb_params["colsample_bylevel"] = float(
        trial.suggest_float("colsample_bylevel", 0.5, 1.0)
    )
    xgb_params["reg_lambda"] = float(
        trial.suggest_float("reg_lambda", 0.1, 100.0, log=True)
    )
    xgb_params["reg_alpha"] = float(
        trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True)
    )
    xgb_params["max_bin"] = int(trial.suggest_categorical("max_bin", [256, 512]))

    return xgb_params, weight_alpha


def _extract_objective_value(tail_metrics: dict[str, Any], objective: str) -> float:
    val = float(tail_metrics[objective])
    if not np.isfinite(val):
        raise ValueError(f"Objective '{objective}' produced non-finite value: {val}")
    return val


def _find_eff_row(tail_metrics: dict[str, Any], target_eps: float) -> dict[str, Any]:
    return min(
        tail_metrics["eff_grid"],
        key=lambda r: abs(float(r["target_signal_efficiency"]) - target_eps),
    )


def _find_pct_row(tail_metrics: dict[str, Any], target_cut: float) -> dict[str, Any]:
    return min(
        tail_metrics["percentile_cuts"],
        key=lambda r: abs(float(r["percentile_cut"]) - target_cut),
    )


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp_path, path)


def evaluate_cv_trial(
    trial: optuna.Trial,
    *,
    cv_df: pd.DataFrame,
    feature_cols: list[str],
    prep_cfg: dict[str, Any],
    cross_sections: dict[str, float],
    total_luminosity: float,
    xgb_params: dict[str, Any],
    weight_alpha: float,
    objective: str,
    n_folds: int,
    val_size: float,
    cv_random_state: int,
    max_fpr: float,
    study_dir: Path,
) -> float:
    """Run K-fold CV for one Optuna trial with cumulative OOF pruning and atomic reporting."""
    preprocessor = Preprocessor({**prep_cfg, "val_size": val_size, "oversample_signal": False})
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)

    oof_pred = np.empty(len(cv_df), dtype=np.float64)
    evaluated_indices: list[np.ndarray] = []
    fold_summaries: list[dict[str, Any]] = []
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
        prepared = preprocessor.prepare_split_arrays(
            df_fit,
            df_fold_test,
            feature_cols,
            df_val=df_val,
            oversample_signal=False,
        )

        sample_weight = compute_training_weights(
            prepared["df_train"]["Truth"],
            cross_sections,
            mode="physics",
            alpha=weight_alpha,
            balance_classes=True,
        )
        if df_val is not None:
            X_val, y_val = prepared["X_val"], prepared["y_val"]
            val_sample_weight = compute_training_weights(
                prepared["df_val"]["Truth"],
                cross_sections,
                mode="physics",
                alpha=weight_alpha,
                balance_classes=True,
            )
        else:
            X_val, y_val = prepared["X_test"], prepared["y_test"]
            val_sample_weight = compute_training_weights(
                prepared["df_test"]["Truth"],
                cross_sections,
                mode="physics",
                alpha=weight_alpha,
                balance_classes=True,
            )

        model = GradientBoostingModel(xgb_params)
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
        fit_s = time.perf_counter() - t0
        total_fit_s += fit_s

        eval_pred = model.predict(prepared["X_test"])
        train_pred = model.predict(prepared["X_train"])

        oof_pred[test_idx] = eval_pred
        evaluated_indices.append(test_idx)

        fold_tail = evaluate_tail_metrics(
            prepared["y_test"],
            eval_pred,
            prepared["df_test"]["Truth"].to_numpy(),
            cross_sections,
            total_luminosity=total_luminosity,
            max_fpr=max_fpr,
        )
        fold_ov = evaluate_overfitting_diagnostics(
            prepared["y_train"],
            train_pred,
            prepared["df_train"]["Truth"].to_numpy(),
            prepared["y_test"],
            eval_pred,
            prepared["df_test"]["Truth"].to_numpy(),
            cross_sections,
            total_luminosity=total_luminosity,
            max_fpr=max_fpr,
        )
        fold_overfits.append(fold_ov)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
                "fit_time_s": fit_s,
                "roc_auc": fold_tail["roc_auc"],
                "partial_weighted_roc_auc": fold_tail["partial_weighted_roc_auc"],
                "weighted_logloss": fold_tail["weighted_logloss"],
                "mean_z_eff_grid": fold_tail["mean_z_eff_grid"],
            }
        )

        if fold_idx < n_folds:
            cum_idx = np.concatenate(evaluated_indices)
            cum_df = cv_df.iloc[cum_idx]
            cum_tail = evaluate_tail_metrics(
                cum_df["target"].to_numpy(),
                oof_pred[cum_idx],
                cum_df["Truth"].to_numpy(),
                cross_sections,
                total_luminosity=total_luminosity,
                max_fpr=max_fpr,
            )
            running_obj = _extract_objective_value(cum_tail, objective)
            trial.report(running_obj, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    tail_metrics = evaluate_tail_metrics(
        cv_df["target"].to_numpy(),
        oof_pred,
        cv_df["Truth"].to_numpy(),
        cross_sections,
        total_luminosity=total_luminosity,
        max_fpr=max_fpr,
    )
    overfitting = {
        k: float(np.mean([ov[k] for ov in fold_overfits])) for k in fold_overfits[0]
    }
    obj_val = _extract_objective_value(tail_metrics, objective)
    trial.report(obj_val, step=n_folds)

    iters = [f["best_iteration"] for f in fold_summaries if f["best_iteration"] is not None]
    mean_best_iter = float(np.mean(iters)) if iters else float(xgb_params["n_estimators"])

    eff30 = _find_eff_row(tail_metrics, 0.30)
    eff40 = _find_eff_row(tail_metrics, 0.40)
    cut90 = _find_pct_row(tail_metrics, 0.90)
    cut925 = _find_pct_row(tail_metrics, 0.925)
    cut95 = _find_pct_row(tail_metrics, 0.95)

    trial.set_user_attr("mean_z_eff_grid", float(tail_metrics["mean_z_eff_grid"]))
    trial.set_user_attr(
        "partial_weighted_roc_auc", float(tail_metrics["partial_weighted_roc_auc"])
    )
    trial.set_user_attr("weighted_logloss", float(tail_metrics["weighted_logloss"]))
    trial.set_user_attr("roc_auc", float(tail_metrics["roc_auc"]))
    trial.set_user_attr("pauc_gap", float(overfitting["partial_weighted_roc_auc_gap"]))
    trial.set_user_attr("mean_best_iteration", mean_best_iter)
    trial.set_user_attr("total_fit_s", float(total_fit_s))
    trial.set_user_attr("eff30_b", float(eff30["b"]))
    trial.set_user_attr("eff30_z", float(eff30["z_asymp"]))
    trial.set_user_attr("eff40_b", float(eff40["b"]))
    trial.set_user_attr("eff40_z", float(eff40["z_asymp"]))
    trial.set_user_attr("cut90_b", float(cut90["b"]))
    trial.set_user_attr("cut90_z", float(cut90["z_asymp"]))
    trial.set_user_attr("cut925_b", float(cut925["b"]))
    trial.set_user_attr("cut925_z", float(cut925["z_asymp"]))
    trial.set_user_attr("cut95_b", float(cut95["b"]))
    trial.set_user_attr("cut95_z", float(cut95["z_asymp"]))

    trial_report = {
        "trial_number": trial.number,
        "objective": objective,
        "objective_value": obj_val,
        "weight_alpha": weight_alpha,
        "params": dict(trial.params),
        "xgb_config": xgb_params,
        "timing": {"total_fit_s": total_fit_s},
        "folds": fold_summaries,
        "tail_metrics": tail_metrics,
        "overfitting": overfitting,
    }
    _write_json_atomic(
        study_dir / "trials" / f"trial_{trial.number:04d}.json", trial_report
    )
    return obj_val


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(help="Run config YAML defining data, preprocessing, and luminosity."),
    ] = Path("config/ChiChi_ManyJets/MSuu_8000/run_wb-ht_jjb-bbt.yaml"),
    study_name: Annotated[
        str,
        typer.Option(help="Optuna study name."),
    ] = "msuu8000_wbht_xgb_hpo",
    study_dir: Annotated[
        Path,
        typer.Option(help="Directory for JournalStorage log, trial JSONs, and summaries."),
    ] = Path("results/ChiChi_ManyJets/MSuu_8000/experiments/hpo_wbht_study"),
    cache_dir: Annotated[
        Path,
        typer.Option(help="Feature extraction Parquet cache directory."),
    ] = Path("cache/features"),
    dataset_parquet: Annotated[
        Path | None,
        typer.Option(
            help="Optional pre-built feature Parquet override (skips ROOT/cache lookup)."
        ),
    ] = None,
    objective: Annotated[
        str,
        typer.Option(
            help="Optimization target: 'mean_z_eff_grid', 'partial_weighted_roc_auc', or 'weighted_logloss'."
        ),
    ] = "mean_z_eff_grid",
    mode: Annotated[
        str,
        typer.Option(help="CV mode: 'full-cv' or 'inner-cv'."),
    ] = "full-cv",
    n_folds: Annotated[
        int,
        typer.Option(help="Number of StratifiedKFold splits."),
    ] = 5,
    val_size: Annotated[
        float,
        typer.Option(help="Internal validation fraction held out from each train fold."),
    ] = 0.1,
    cv_random_state: Annotated[
        int,
        typer.Option(help="Random state for StratifiedKFold."),
    ] = 42,
    n_trials: Annotated[
        int,
        typer.Option(help="Maximum trials for this worker process."),
    ] = 15,
    max_total_trials: Annotated[
        int,
        typer.Option(help="Global cap on COMPLETE+PRUNED trials across all parallel workers."),
    ] = 100,
    n_estimators: Annotated[
        int,
        typer.Option(help="Maximum boosting rounds per fold (stopped early via val split)."),
    ] = 3000,
    early_stopping_rounds: Annotated[
        int,
        typer.Option(help="Early stopping patience rounds."),
    ] = 40,
    device: Annotated[
        str,
        typer.Option(help="XGBoost device ('cpu', 'cuda', 'cuda:0', or 'gpu')."),
    ] = "cpu",
    n_jobs: Annotated[
        int,
        typer.Option(help="CPU threads per worker."),
    ] = 64,
    random_state: Annotated[
        int,
        typer.Option(help="XGBoost random_state."),
    ] = 17,
    worker_seed: Annotated[
        int,
        typer.Option(help="Random seed for this worker's Optuna TPESampler."),
    ] = 1001,
    max_fpr: Annotated[
        float,
        typer.Option(help="Max FPR cutoff for partial_weighted_roc_auc."),
    ] = 2e-3,
    prune: Annotated[
        bool,
        typer.Option("--prune/--no-prune", help="Enable MedianPruner after startup trials."),
    ] = True,
) -> None:
    if mode not in {"full-cv", "inner-cv"}:
        raise typer.BadParameter(f"Unsupported mode '{mode}'. Choose 'full-cv' or 'inner-cv'.")
    if objective not in OBJECTIVE_DIRECTIONS:
        raise typer.BadParameter(
            f"Unsupported objective '{objective}'. Choose from {sorted(OBJECTIVE_DIRECTIONS)}."
        )

    if dataset_parquet is not None:
        spec = resolve_dataset_spec(config)
        df = pd.read_parquet(dataset_parquet)
        cross_sections = spec["cross_sections"]
        total_luminosity = float(spec["config"].get("data.total_luminosity", 3000.0))
        prep_cfg = dict(spec["config"].get("preprocessing", {}))
    else:
        df, cross_sections, data_meta = load_or_extract_dataset(config, cache_dir=cache_dir)
        total_luminosity = float(data_meta.get("total_luminosity", 3000.0))
        prep_cfg = dict(data_meta.get("preprocessing", {}))

    feature_cols = [c for c in df.columns if c not in ("Truth", "target")]
    if mode == "inner-cv":
        outer_prep = Preprocessor(prep_cfg)
        cv_df, _ = outer_prep.split_train_test_df(df)
    else:
        cv_df = df

    study_dir.mkdir(parents=True, exist_ok=True)
    journal_path = study_dir / "journal.log"
    study = create_or_load_study(
        study_name=study_name,
        journal_path=journal_path,
        objective=objective,
        sampler_seed=worker_seed,
        prune=prune,
        enqueue_anchors=True,
    )

    finished_states = (TrialState.COMPLETE, TrialState.PRUNED)
    existing_finished = sum(1 for t in study.trials if t.state in finished_states)
    if existing_finished >= max_total_trials:
        print(
            f"Study '{study_name}' already has {existing_finished} finished trials "
            f"(>= max_total_trials={max_total_trials}); exiting cleanly."
        )
        return

    def _objective_fn(trial: optuna.Trial) -> float:
        xgb_params, weight_alpha = sample_trial_params(
            trial,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            device=device,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        return evaluate_cv_trial(
            trial,
            cv_df=cv_df,
            feature_cols=feature_cols,
            prep_cfg=prep_cfg,
            cross_sections=cross_sections,
            total_luminosity=total_luminosity,
            xgb_params=xgb_params,
            weight_alpha=weight_alpha,
            objective=objective,
            n_folds=n_folds,
            val_size=val_size,
            cv_random_state=cv_random_state,
            max_fpr=max_fpr,
            study_dir=study_dir,
        )

    print(
        f"Starting Optuna worker (seed={worker_seed}, mode={mode}, n_folds={n_folds}, "
        f"objective={objective}, device={device}, n_jobs={n_jobs}) on study '{study_name}'..."
    )
    study.optimize(
        _objective_fn,
        n_trials=n_trials,
        callbacks=[MaxTrialsCallback(max_total_trials, states=finished_states)],
    )
    final_finished = sum(1 for t in study.trials if t.state in finished_states)
    print(
        f"Worker finished cleanly. Total finished trials in '{study_name}': {final_finished}"
    )


if __name__ == "__main__":
    app()
