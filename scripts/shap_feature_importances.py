"""Compute and plot SHAP feature importances using ``shap.TreeExplainer``.

Loads a trained XGBoost model, its ``MinMaxScaler``/``StandardScaler``, and
``test_set.parquet`` from a run directory under ``results/``, evaluates exact
TreeSHAP values on a sample of test events, prints a ranked summary table, and
saves a two-panel figure (mean ``|SHAP|`` importance bar chart and SHAP
beeswarm summary plot).

Usage::

    uv run scripts/shap_feature_importances.py
    uv run scripts/shap_feature_importances.py --run-dir results/ChiChi_ManyJets/MSuu_8000/ht-ht/bbt-bbt
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer
import xgboost as xgb


def main(
    run_dir: Annotated[
        Path,
        typer.Option(help="Run directory containing gradient_boosting_model.json, scaler.pkl, and test_set.parquet."),
    ] = Path("results/ChiChi_ManyJets/MSuu_8000/wb-ht/jjb-bbt"),
    n_samples: Annotated[
        int,
        typer.Option(help="Number of test-set events to sample for SHAP evaluation (0 = use all)."),
    ] = 10000,
    top_k: Annotated[
        int,
        typer.Option(help="Number of top features to print and plot."),
    ] = 15,
    output: Annotated[
        Path,
        typer.Option(help="Output plot file path."),
    ] = Path("plots/shap_feature_importances.pdf"),
    seed: Annotated[
        int,
        typer.Option(help="Random seed for test-set sampling."),
    ] = 42,
) -> None:
    df = pd.read_parquet(run_dir / "test_set.parquet")
    if 0 < n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed)

    feature_cols = [c for c in df.columns if c not in ("Truth", "target")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(run_dir / "scaler.pkl", "rb") as fh:
            scaler = pickle.load(fh)

    X_raw = df[feature_cols].to_numpy()
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=feature_cols)

    model = xgb.XGBClassifier()
    model.load_model(str(run_dir / "gradient_boosting_model.json"))

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_scaled)
    # Attach unscaled feature values for interpretable beeswarm colorbars
    explanation = shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=X_raw,
        feature_names=feature_cols,
    )

    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    shap_share = mean_abs_shap / mean_abs_shap.sum()

    summary = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
            "shap_share_pct": 100.0 * shap_share,
        }
    )

    gain_path = run_dir / "gradient_boosting_feature_importances.json"
    if gain_path.exists():
        with open(gain_path, encoding="utf-8") as fh:
            gain_map = json.load(fh)
        summary["xgb_gain_pct"] = summary["feature"].map(lambda f: 100.0 * gain_map.get(f, 0.0))

    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"SHAP feature importances for {run_dir} (n={len(df)} test events):")
    print(summary.head(top_k).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Two-panel figure: (a) mean(|SHAP|) bar chart, (b) SHAP beeswarm plot
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9.5})
    fig, (ax_bar, ax_bee) = plt.subplots(
        1, 2, figsize=(14.5, max(5.2, 0.36 * top_k)), gridspec_kw={"width_ratios": [0.95, 1.15]}
    )

    top_df = summary.head(top_k).iloc[::-1]
    y_pos = np.arange(len(top_df))
    ax_bar.barh(y_pos, top_df["mean_abs_shap"], color="#1f77b4", alpha=0.88)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top_df["feature"], fontsize=8.5)
    ax_bar.set_xlabel("mean(|SHAP value|) [log-odds]")
    ax_bar.set_title("(a) Global SHAP importance")
    ax_bar.grid(axis="x", alpha=0.3, ls="--")
    ax_bar.set_axisbelow(True)
    x_max = top_df["mean_abs_shap"].max()
    ax_bar.set_xlim(0, x_max * 1.32)
    for yp, val, pct in zip(y_pos, top_df["mean_abs_shap"], top_df["shap_share_pct"]):
        ax_bar.text(val + x_max * 0.015, yp, f"{val:.3f} ({pct:.1f}%)", va="center", fontsize=7.8)

    top_idx = [feature_cols.index(f) for f in summary.head(top_k)["feature"]]
    plt.sca(ax_bee)
    shap.plots.beeswarm(explanation[:, top_idx], max_display=top_k, show=False, plot_size=None)
    ax_bee.set_title("(b) SHAP value distribution (color = feature value)")
    ax_bee.set_xlabel("SHAP value (impact on signal log-odds)")

    fig.suptitle(f"TreeExplainer SHAP Importances — {run_dir}", fontsize=11.5, y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"\nSaved {output}")
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
