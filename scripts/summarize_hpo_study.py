"""Summarize, compare, and plot results from a completed or running Optuna HPO study."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import typer
import yaml
from optuna.importance import get_param_importances
from optuna.trial import TrialState

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.tune_xgboost import OBJECTIVE_DIRECTIONS, build_journal_storage

app = typer.Typer(
    help="Summarize an Optuna XGBoost study, export best_params.yaml, and render summary PDF.",
    add_completion=False,
)


def _load_trial_reports(study_dir: Path) -> dict[int, dict[str, Any]]:
    reports: dict[int, dict[str, Any]] = {}
    trials_dir = study_dir / "trials"
    if not trials_dir.exists():
        return reports
    for p in sorted(trials_dir.glob("trial_*.json")):
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        reports[int(data["trial_number"])] = data
    return reports


def _compute_param_importances_safe(study: optuna.Study) -> dict[str, float]:
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(complete) < 4:
        return {}
    try:
        return dict(get_param_importances(study))
    except (ValueError, RuntimeError):
        common_params = sorted(
            set.intersection(*(set(t.params.keys()) for t in complete))
        )
        if not common_params:
            return {}
        try:
            return dict(get_param_importances(study, params=common_params))
        except (ValueError, RuntimeError):
            return {}


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


def _fmt_pm(val: float, err: float) -> str:
    return f"{val:.3f} ± {err:.3f}" if np.isfinite(val) else "inf"


def _build_best_yaml_dict(best_trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    p = best_trial.params
    gb: dict[str, Any] = {
        "n_estimators": 3000,
        "early_stopping_rounds": 40,
        "eval_metric": "logloss",
        "learning_rate": round(float(p["learning_rate"]), 6),
        "grow_policy": str(p["grow_policy"]),
        "min_child_weight": round(float(p["min_child_weight"]), 4),
        "gamma": round(float(p["gamma"]), 4),
        "subsample": round(float(p["subsample"]), 4),
        "colsample_bytree": round(float(p["colsample_bytree"]), 4),
        "colsample_bylevel": round(float(p["colsample_bylevel"]), 4),
        "reg_lambda": round(float(p["reg_lambda"]), 4),
        "reg_alpha": round(float(p["reg_alpha"]), 6),
        "max_bin": int(p["max_bin"]),
        "random_state": 17,
    }
    if p["grow_policy"] == "depthwise":
        gb["max_depth"] = int(p["max_depth"])
    else:
        gb["max_depth"] = 0
        gb["max_leaves"] = int(p["max_leaves"])

    return {
        "preprocessing": {"val_size": 0.1, "oversample_signal": False},
        "training_weights": {
            "mode": "physics",
            "alpha": round(float(p["weight_alpha"]), 4),
            "balance_classes": True,
            "apply_to_eval_set": True,
        },
        "models": {"gradient_boosting": gb},
    }


def _plot_summary(
    study: optuna.Study,
    complete_sorted: list[optuna.trial.FrozenTrial],
    reports: dict[int, dict[str, Any]],
    importances: dict[str, float],
    baseline_report: dict[str, Any] | None,
    objective: str,
    out_pdf: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)

    # Panel (a): Optimization history
    ax = axes[0, 0]
    by_num = sorted(
        [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None],
        key=lambda t: t.number,
    )
    if by_num:
        nums = np.array([t.number for t in by_num], dtype=int)
        vals = np.array([float(t.value) for t in by_num], dtype=float)
        if OBJECTIVE_DIRECTIONS[objective] == "maximize":
            running_best = np.maximum.accumulate(vals)
        else:
            running_best = np.minimum.accumulate(vals)
        ax.scatter(nums, vals, s=26, alpha=0.75, color="#1f77b4", label="Completed trials")
        ax.plot(nums, running_best, lw=2.0, color="#d62728", label="Running best")
    ax.set_title("(a) Optimization History")
    ax.set_xlabel("Trial number")
    ax.set_ylabel(objective)
    ax.legend(loc="best")

    # Panel (b): Hyperparameter importances
    ax = axes[0, 1]
    if importances:
        items = list(importances.items())[:10][::-1]
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        y = np.arange(len(names))
        ax.barh(y, vals, color="#2ca02c", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("Relative importance")
    else:
        ax.text(0.5, 0.5, "Need >= 4 completed trials\nfor importance analysis", ha="center", va="center")
    ax.set_title("(b) Hyperparameter Importances")

    # Panel (c): Objective vs weight_alpha colored by log10(min_child_weight)
    ax = axes[1, 0]
    if by_num:
        alphas = np.array([float(t.params["weight_alpha"]) for t in by_num], dtype=float)
        vals = np.array([float(t.value) for t in by_num], dtype=float)
        mcws = np.array([float(t.params["min_child_weight"]) for t in by_num], dtype=float)
        sc = ax.scatter(
            alphas,
            vals,
            c=np.log10(np.maximum(mcws, 1e-3)),
            cmap="viridis",
            s=36,
            alpha=0.88,
            edgecolors="k",
            linewidths=0.3,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\log_{10}(\mathrm{min\_child\_weight})$")
    ax.set_title(f"(c) {objective} vs. physics weight alpha")
    ax.set_xlabel(r"Physics weight exponent $\alpha$")
    ax.set_ylabel(objective)

    # Panel (d): Background yield vs signal efficiency (baseline vs top 3 trials)
    ax = axes[1, 1]
    if baseline_report is not None:
        rows = sorted(
            baseline_report["tail_metrics"]["eff_grid"],
            key=lambda r: r["target_signal_efficiency"],
        )
        eps = np.array([r["target_signal_efficiency"] for r in rows], dtype=float)
        b_vals = np.array([r["b"] for r in rows], dtype=float)
        db_vals = np.array([r["db"] for r in rows], dtype=float)
        ax.errorbar(
            eps,
            np.maximum(b_vals, 1e-4),
            yerr=np.minimum(db_vals, 0.9 * np.maximum(b_vals, 1e-4)),
            color="0.3",
            ls="--",
            marker="s",
            ms=4.5,
            lw=1.6,
            capsize=2.5,
            label="Baseline (none, oversample)",
        )

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for idx, t in enumerate(complete_sorted[:3]):
        rep = reports.get(t.number)
        if rep is None:
            continue
        rows = sorted(
            rep["tail_metrics"]["eff_grid"],
            key=lambda r: r["target_signal_efficiency"],
        )
        eps = np.array([r["target_signal_efficiency"] for r in rows], dtype=float)
        b_vals = np.array([r["b"] for r in rows], dtype=float)
        db_vals = np.array([r["db"] for r in rows], dtype=float)
        ax.errorbar(
            eps,
            np.maximum(b_vals, 1e-4),
            yerr=np.minimum(db_vals, 0.9 * np.maximum(b_vals, 1e-4)),
            color=colors[idx],
            marker="o",
            ms=4.5,
            lw=1.5,
            capsize=2.5,
            label=f"Trial #{t.number} ({objective}={t.value:.3f})",
        )

    ax.set_yscale("log")
    ax.set_title(r"(d) Background yield $B$ vs. $\varepsilon_S$")
    ax.set_xlabel(r"Target signal efficiency $\varepsilon_S$")
    ax.set_ylabel(r"Expected background yield $B$ [$3\ \mathrm{ab}^{-1}$]")
    ax.legend(loc="upper left", fontsize=7.8)

    fig.suptitle(f"Optuna HPO Study Summary: {study.study_name}")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    study_name: Annotated[
        str,
        typer.Option(help="Optuna study name."),
    ] = "msuu8000_wbht_xgb_hpo",
    study_dir: Annotated[
        Path,
        typer.Option(help="Study directory containing journal.log and trials/."),
    ] = Path("results/ChiChi_ManyJets/MSuu_8000/experiments/hpo_wbht_study"),
    baseline_json: Annotated[
        Path | None,
        typer.Option(help="Optional baseline JSON report for yield comparison."),
    ] = Path(
        "results/ChiChi_ManyJets/MSuu_8000/experiments/physics_weighting_study/baseline_oversample.json"
    ),
    objective: Annotated[
        str,
        typer.Option(help="Objective name used during the study."),
    ] = "mean_z_eff_grid",
    top_k: Annotated[
        int,
        typer.Option(help="Number of top completed trials to display."),
    ] = 10,
) -> None:
    if objective not in OBJECTIVE_DIRECTIONS:
        raise typer.BadParameter(
            f"Unsupported objective '{objective}'. Choose from {sorted(OBJECTIVE_DIRECTIONS)}."
        )

    journal_path = study_dir / "journal.log"
    if not journal_path.exists():
        raise FileNotFoundError(f"JournalStorage file not found at '{journal_path}'")

    storage = build_journal_storage(journal_path)
    study = optuna.load_study(study_name=study_name, storage=storage)
    reports = _load_trial_reports(study_dir)

    counts = Counter(t.state.name for t in study.trials)
    print("\n" + "=" * 100)
    print(f"Study '{study_name}' ({study_dir}) | states={dict(counts)}")

    complete = [
        t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        print("No completed trials yet.")
        return

    reverse = OBJECTIVE_DIRECTIONS[objective] == "maximize"
    complete_sorted = sorted(complete, key=lambda t: float(t.value or 0.0), reverse=reverse)
    best = complete_sorted[0]
    print(f"Best trial: #{best.number} | {objective}={best.value:.4f}")

    print(f"\n--- Top {min(top_k, len(complete_sorted))} Trials ---")
    print(
        f"{'#':>4} {'mean_Z':>7} {'pAUC':>8} {'wLL':>8} {'pGAP':>7} {'alpha':>6} "
        f"{'policy':>9} {'d/l':>5} {'lr':>6} {'mcw':>6} {'sub':>5} {'col':>5} {'lam':>7} {'gam':>5}"
    )
    for t in complete_sorted[:top_k]:
        p = t.params
        ua = t.user_attrs
        dl = str(p["max_depth"]) if p["grow_policy"] == "depthwise" else f"L{p['max_leaves']}"
        print(
            f"{t.number:4d} {ua.get('mean_z_eff_grid', float(t.value)):7.4f} "
            f"{ua.get('partial_weighted_roc_auc', float('nan')):8.5f} "
            f"{ua.get('weighted_logloss', float('nan')):8.5f} "
            f"{ua.get('pauc_gap', float('nan')):+7.4f} "
            f"{p['weight_alpha']:6.3f} {p['grow_policy']:>9} {dl:>5} "
            f"{p['learning_rate']:6.3f} {p['min_child_weight']:6.1f} "
            f"{p['subsample']:5.2f} {p['colsample_bytree']:5.2f} "
            f"{p['reg_lambda']:7.2f} {p['gamma']:5.2f}"
        )

    baseline_report: dict[str, Any] | None = None
    if baseline_json is not None and baseline_json.exists():
        with open(baseline_json, encoding="utf-8") as fh:
            baseline_report = json.load(fh)

    top_entries: list[tuple[str, dict[str, Any]]] = []
    if baseline_report is not None:
        top_entries.append(("Baseline (none, oversample)", baseline_report))
    for t in complete_sorted[:3]:
        if t.number in reports:
            top_entries.append((f"Trial #{t.number} (α={t.params['weight_alpha']:.2f})", reports[t.number]))

    if top_entries:
        for eps in (0.25, 0.30, 0.40, 0.50):
            print(f"\n--- Yield Comparison at eps_S = {eps:.2f} ---")
            print(f"{'Run':<30} {'S':>7} {'B ± dB':>16} {'raw_QCD':>8} {'S/B':>8} {'Z_asymp ± dZ':>16}")
            for label, rep in top_entries:
                row = _find_eff_row(rep["tail_metrics"], eps)
                sb = f"{row['s_over_b']:8.3f}" if np.isfinite(row["s_over_b"]) else "     inf"
                print(
                    f"{label:<30} {row['s']:7.3f} {_fmt_pm(row['b'], row['db']):>16} "
                    f"{row['raw_qcd']:8d} {sb} {_fmt_pm(row['z_asymp'], row['z_asymp_err']):>16}"
                )

        for cut in (0.90, 0.925, 0.95):
            print(f"\n--- Yield Comparison at Percentile Cut = {cut:.3f} ---")
            print(f"{'Run':<30} {'eps_S':>7} {'S':>7} {'B ± dB':>16} {'raw_QCD':>8} {'Z_asymp ± dZ':>16}")
            for label, rep in top_entries:
                row = _find_pct_row(rep["tail_metrics"], cut)
                print(
                    f"{label:<30} {row['signal_efficiency']:7.4f} {row['s']:7.3f} "
                    f"{_fmt_pm(row['b'], row['db']):>16} {row['raw_qcd']:8d} "
                    f"{_fmt_pm(row['z_asymp'], row['z_asymp_err']):>16}"
                )

    importances = _compute_param_importances_safe(study)
    if importances:
        print("\n--- Hyperparameter Importances ---")
        for k, v in importances.items():
            print(f"  {k:<22} {v:.4f}")

    best_yaml = _build_best_yaml_dict(best)
    best_yaml_path = study_dir / "best_params.yaml"
    with open(best_yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(best_yaml, fh, sort_keys=False)
    print(f"\nSaved best YAML snippet to {best_yaml_path}:")
    print(yaml.safe_dump(best_yaml, sort_keys=False))

    out_pdf = study_dir / "hpo_summary.pdf"
    _plot_summary(
        study, complete_sorted, reports, importances, baseline_report, objective, out_pdf
    )
    print(f"Saved summary plot to {out_pdf}")
    print("=" * 100)


if __name__ == "__main__":
    app()
