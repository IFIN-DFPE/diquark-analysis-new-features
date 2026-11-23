"""Compare and visualize physics-weighting 5-fold CV study reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(
    help="Aggregate, compare, and plot physics-weighting benchmark JSON reports.",
    add_completion=False,
)

CANONICAL_ORDER: list[str] = [
    "baseline_oversample",
    "class_balanced",
    "physics_a0.25",
    "physics_a0.50",
    "physics_a0.75",
    "physics_a1.00",
    "physics_a1.00_reg",
    "baseline_reg",
]

VARIANT_LABELS: dict[str, str] = {
    "baseline_oversample": "Baseline (none, oversample)",
    "class_balanced": "Class-balanced (α=0.0)",
    "physics_a0.25": "Physics (α=0.25)",
    "physics_a0.50": "Physics (α=0.50)",
    "physics_a0.75": "Physics (α=0.75)",
    "physics_a1.00": "Physics (α=1.00)",
    "physics_a1.00_reg": "Physics (α=1.00, reg)",
    "baseline_reg": "Baseline (none, reg)",
}

SHORT_LABELS: dict[str, str] = {
    "baseline_oversample": "Base\nOS",
    "class_balanced": "CB\nα=0",
    "physics_a0.25": "Phys\n0.25",
    "physics_a0.50": "Phys\n0.50",
    "physics_a0.75": "Phys\n0.75",
    "physics_a1.00": "Phys\n1.00",
    "physics_a1.00_reg": "Phys\n1.00+R",
    "baseline_reg": "Base\nOS+R",
}

BKG_PROCESSES: tuple[str, ...] = (
    "BKG:qcd",
    "BKG:single_boson",
    "BKG:ttbar",
    "BKG:diboson",
)


def _load_reports(study_dir: Path) -> dict[str, dict[str, Any]]:
    json_files = {p.stem: p for p in study_dir.glob("*.json")}
    if not json_files:
        raise FileNotFoundError(f"No JSON benchmark reports found in '{study_dir}'")

    ordered_names = [name for name in CANONICAL_ORDER if name in json_files]
    ordered_names.extend(sorted(name for name in json_files if name not in CANONICAL_ORDER))

    reports: dict[str, dict[str, Any]] = {}
    for name in ordered_names:
        with open(json_files[name], encoding="utf-8") as fh:
            reports[name] = json.load(fh)
    return reports


def _find_eff_row(report: dict[str, Any], target_eps: float) -> dict[str, Any]:
    rows = report["tail_metrics"]["eff_grid"]
    return min(rows, key=lambda r: abs(float(r["target_signal_efficiency"]) - target_eps))


def _find_pct_row(report: dict[str, Any], target_cut: float) -> dict[str, Any]:
    rows = report["tail_metrics"]["percentile_cuts"]
    return min(rows, key=lambda r: abs(float(r["percentile_cut"]) - target_cut))


def _fmt_pm(val: float, err: float) -> str:
    if not np.isfinite(val):
        return "inf"
    return f"{val:.3f} ± {err:.3f}"


def _print_tables(reports: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY (5-Fold Pooled OOF)")
    print("=" * 100)
    print(
        f"{'Variant':<28} {'roc_auc':>9} {'pAUC_2e-3':>10} {'w_logloss':>10} "
        f"{'mean_Z':>8} {'pAUC_gap':>9} {'mean_iter':>10} {'fit_s':>8}"
    )
    for name, rep in reports.items():
        tm = rep["tail_metrics"]
        ov = rep["overfitting"]
        mean_iter = float(np.mean([f["best_iteration"] for f in rep["folds"]]))
        fit_s = float(rep["timing"]["total_fit_s"])
        label = VARIANT_LABELS.get(name, name)
        print(
            f"{label:<28} {tm['roc_auc']:9.6f} {tm['partial_weighted_roc_auc']:10.6f} "
            f"{tm['weighted_logloss']:10.6f} {tm['mean_z_eff_grid']:8.4f} "
            f"{ov['partial_weighted_roc_auc_gap']:+9.4f} {mean_iter:10.1f} {fit_s:8.1f}"
        )

    for eps in (0.25, 0.30, 0.40, 0.50, 0.60):
        sample_row = _find_eff_row(next(iter(reports.values())), eps)
        print(f"\n--- Fixed Signal Efficiency eps_S = {eps:.2f} (S ≈ {sample_row['s']:.3f}) ---")
        print(
            f"{'Variant':<28} {'S':>7} {'B ± dB':>16} {'raw_QCD':>8} "
            f"{'raw_BKG':>8} {'S/B':>8} {'Z_asymp ± dZ':>16}"
        )
        for name, rep in reports.items():
            row = _find_eff_row(rep, eps)
            label = VARIANT_LABELS.get(name, name)
            sb = f"{row['s_over_b']:8.3f}" if np.isfinite(row["s_over_b"]) else "     inf"
            print(
                f"{label:<28} {row['s']:7.3f} {_fmt_pm(row['b'], row['db']):>16} "
                f"{row['raw_qcd']:8d} {row['raw_bkg_total']:8d} {sb} "
                f"{_fmt_pm(row['z_asymp'], row['z_asymp_err']):>16}"
            )

    for cut in (0.85, 0.875, 0.90, 0.925, 0.95):
        print(f"\n--- Standard Percentile Cut = {cut:.3f} ---")
        print(
            f"{'Variant':<28} {'eps_S':>7} {'S':>7} {'B ± dB':>16} {'raw_QCD':>8} "
            f"{'S/B':>8} {'Z_asymp ± dZ':>16}"
        )
        for name, rep in reports.items():
            row = _find_pct_row(rep, cut)
            label = VARIANT_LABELS.get(name, name)
            sb = f"{row['s_over_b']:8.3f}" if np.isfinite(row["s_over_b"]) else "     inf"
            print(
                f"{label:<28} {row['signal_efficiency']:7.4f} {row['s']:7.3f} "
                f"{_fmt_pm(row['b'], row['db']):>16} {row['raw_qcd']:8d} {sb} "
                f"{_fmt_pm(row['z_asymp'], row['z_asymp_err']):>16}"
            )

    for title, getter in (
        ("Fixed eps_S = 0.30", lambda r: _find_eff_row(r, 0.30)),
        ("Fixed eps_S = 0.40", lambda r: _find_eff_row(r, 0.40)),
        ("Percentile Cut = 0.90", lambda r: _find_pct_row(r, 0.90)),
        ("Percentile Cut = 0.95", lambda r: _find_pct_row(r, 0.95)),
    ):
        print(f"\n--- Per-Process Background Breakdown ({title}) ---")
        print(
            f"{'Variant':<28} {'QCD (raw)':>15} {'single_boson':>16} "
            f"{'ttbar':>16} {'diboson':>16}"
        )
        for name, rep in reports.items():
            row = getter(rep)
            y = row["per_process_yields"]
            rc = row["per_process_raw_counts"]
            label = VARIANT_LABELS.get(name, name)
            cells = [
                f"{y[p]:.4f} ({rc[p]})" for p in BKG_PROCESSES
            ]
            print(f"{label:<28} {cells[0]:>15} {cells[1]:>16} {cells[2]:>16} {cells[3]:>16}")
    print("=" * 100)


def _plot_comparison(reports: dict[str, dict[str, Any]], out_pdf: Path) -> None:
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

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    names = list(reports.keys())

    # Panel (a) and (b): B vs eps_S and Z vs eps_S
    for idx, name in enumerate(names):
        rep = reports[name]
        rows = sorted(rep["tail_metrics"]["eff_grid"], key=lambda r: r["target_signal_efficiency"])
        eps = np.array([r["target_signal_efficiency"] for r in rows], dtype=float)
        b_vals = np.array([r["b"] for r in rows], dtype=float)
        db_vals = np.array([r["db"] for r in rows], dtype=float)
        z_vals = np.array([r["z_asymp"] for r in rows], dtype=float)
        dz_vals = np.array([r["z_asymp_err"] for r in rows], dtype=float)

        color = cmap(idx % 10)
        ls = "--" if name.endswith("_reg") else "-"
        marker = "s" if name.endswith("_reg") else "o"
        label = VARIANT_LABELS.get(name, name)

        axes[0].errorbar(
            eps,
            np.maximum(b_vals, 1e-4),
            yerr=np.minimum(db_vals, 0.9 * np.maximum(b_vals, 1e-4)),
            color=color,
            ls=ls,
            marker=marker,
            ms=4,
            lw=1.4,
            capsize=2.5,
            label=label,
        )
        axes[1].errorbar(
            eps,
            np.where(np.isfinite(z_vals), z_vals, np.nan),
            yerr=np.where(np.isfinite(dz_vals), dz_vals, 0.0),
            color=color,
            ls=ls,
            marker=marker,
            ms=4,
            lw=1.4,
            capsize=2.5,
            label=label,
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Target signal efficiency $\varepsilon_S$")
    axes[0].set_ylabel(r"Expected background yield $B$ [$3\ \mathrm{ab}^{-1}$]")
    axes[0].set_title(r"(a) Background yield vs. $\varepsilon_S$")
    axes[0].legend(loc="upper left", fontsize=7.2, framealpha=0.92)

    axes[1].set_xlabel(r"Target signal efficiency $\varepsilon_S$")
    axes[1].set_ylabel(r"Discovery significance $Z_{\mathrm{asymp}}$ [$\sigma$]")
    axes[1].set_title(r"(b) Discovery significance vs. $\varepsilon_S$")

    # Panel (c): Alpha/variant sweep at fixed eps_S = 0.30 and 0.40
    x = np.arange(len(names))
    ax_b = axes[2]
    ax_z = ax_b.twinx()
    ax_z.grid(False)

    for eps_target, color, marker, offset in (
        (0.30, "#1f77b4", "o", -0.12),
        (0.40, "#d62728", "^", +0.12),
    ):
        b_sweep = np.array([_find_eff_row(reports[n], eps_target)["b"] for n in names], dtype=float)
        db_sweep = np.array([_find_eff_row(reports[n], eps_target)["db"] for n in names], dtype=float)
        z_sweep = np.array([_find_eff_row(reports[n], eps_target)["z_asymp"] for n in names], dtype=float)
        dz_sweep = np.array([_find_eff_row(reports[n], eps_target)["z_asymp_err"] for n in names], dtype=float)

        ax_b.errorbar(
            x + offset,
            b_sweep,
            yerr=db_sweep,
            color=color,
            marker=marker,
            ms=5,
            lw=1.5,
            capsize=3,
            label=rf"$B$ ($\varepsilon_S={eps_target:.2f}$)",
        )
        ax_z.errorbar(
            x + offset,
            np.where(np.isfinite(z_sweep), z_sweep, np.nan),
            yerr=np.where(np.isfinite(dz_sweep), dz_sweep, 0.0),
            color=color,
            ls="--",
            marker=marker,
            mfc="white",
            ms=5,
            lw=1.2,
            capsize=2.5,
            label=rf"$Z$ ($\varepsilon_S={eps_target:.2f}$)",
        )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([SHORT_LABELS.get(n, n) for n in names], fontsize=7.5)
    ax_b.set_ylabel(r"Expected background yield $B$ [$3\ \mathrm{ab}^{-1}$]")
    ax_z.set_ylabel(r"$Z_{\mathrm{asymp}}$ [$\sigma$]")
    ax_b.set_title(r"(c) Sweep at fixed $\varepsilon_S = 0.30, 0.40$")

    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = ax_z.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.0, framealpha=0.92)

    fig.suptitle(r"Physics-Weighting Study ($M_{S_{uu}} = 8$ TeV, wb-ht / jjb-bbt, 5-fold pooled OOF)")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    study_dir: Annotated[
        Path,
        typer.Option(help="Directory containing benchmark JSON reports."),
    ] = Path("results/ChiChi_ManyJets/MSuu_8000/experiments/physics_weighting_study"),
    output_pdf: Annotated[
        Path | None,
        typer.Option(help="Output PDF figure path (defaults to <study_dir>/weighting_comparison.pdf)."),
    ] = None,
) -> None:
    reports = _load_reports(study_dir)
    _print_tables(reports)
    pdf_path = output_pdf if output_pdf is not None else (study_dir / "weighting_comparison.pdf")
    _plot_comparison(reports, pdf_path)
    print(f"\nSaved comparison figure to {pdf_path}")


if __name__ == "__main__":
    app()
