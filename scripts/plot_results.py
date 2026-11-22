"""Plot the machine-learning results of the ultraheavy diquark analysis.

Reads the per-run artifacts written by ``diquark/analysis.py`` (counts,
significance metrics, score thresholds, feature importances and the saved
test sets) under a results root directory, e.g. ``results/ChiChi_ManyJets``,
and renders a set of publication-style figures:

- ``discovery_reach.pdf``      discovery significance vs. diquark mass
- ``yields_vs_mass.pdf``       expected signal / background yields vs. mass
- ``significance_vs_cut.pdf``  significance, S/B and efficiency vs. score cut
- ``eff_vs_rej.pdf``           signal efficiency vs. background rejection
- ``score_distributions.pdf``  discriminator score distributions
- ``feature_importances.pdf``  top XGBoost feature importances
- ``systematics.pdf``          systematic variations at the anchor mass
- ``mass_distributions.pdf``   invariant mass of the event system (m_Nj)

All yields are pre-scaled to the total luminosity configured in the runs
(3000 fb^-1 = 3 ab^-1). The "fixed cut" working point refers to the
percentile of the discriminator score defined in the run configs (default
0.95); the corresponding score value is read from each run's
``*_score_thresholds.json``.

Usage::

    uv run scripts/plot_results.py --root results/ChiChi_ManyJets --out plots
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

LUMINOSITY_LABEL = "HL-LHC 3 ab$^{-1}$"
CHI_MASS_LABEL = r"$M_\chi = 2$ TeV"

# channel key -> nominal results sub-path, systematics sub-dir, display label, color
CHANNELS: dict[str, dict] = {
    "ht-ht": {
        "path": "ht-ht/bbt-bbt",
        "sys_dir": "ht-ht_bbt-bbt",
        "label": "ht-ht (bbt-bbt)",
        "color": "#1f77b4",
    },
    "wb-ht": {
        "path": "wb-ht/jjb-bbt",
        "sys_dir": "wb-ht_jjb-bbt",
        "label": "wb-ht (jjb-bbt)",
        "color": "#ff7f0e",
    },
    "wb-zt": {
        "path": "wb-zt",
        "sys_dir": "wb-zt",
        "label": "wb-zt",
        "color": "#2ca02c",
    },
    "zt-ht": {
        "path": "zt-ht_jjt-bbt",
        "sys_dir": "zt-ht_jjt-bbt",
        "label": "zt-ht (jjt-bbt)",
        "color": "#d62728",
    },
}

BKG_PROCESSES = ["BKG:qcd", "BKG:ttbar", "BKG:single_boson", "BKG:diboson"]
SIG_PROCESS = "SIG:Suu"
BKG_SUM = "BKG:sum"

PROCESS_LABELS = {
    "BKG:qcd": "QCD",
    "BKG:ttbar": r"$t\bar{t}$",
    "BKG:single_boson": "single boson",
    "BKG:diboson": "diboson",
    "SIG:Suu": "Suu signal",
    "BKG:sum": "total background",
}

SCORE_COLORS = {
    "SIG:Suu": "#d62728",
    "BKG:qcd": "#1f77b4",
    "BKG:ttbar": "#2ca02c",
    "BKG:single_boson": "#9467bd",
    "BKG:diboson": "#7f7f7f",
}

SYSTEMATIC_ORDER = [
    "JES_up",
    "JES_down",
    "JES_up_JER_down",
    "JES_down_JER_up",
    "JER_up",
    "JER_down",
    "both_up",
    "both_down",
    "cone_radius_1",
    "cone_radius_2",
    "pileup_0",
    "pileup_60",
    "pileup_100",
    "pileup_200",
]
#: single-parameter variations combined (max |delta|) into the systematic error bar
SYSTEMATIC_SINGLE = [
    "JES_up",
    "JES_down",
    "JER_up",
    "JER_down",
    "pileup_0",
    "pileup_200",
    "cone_radius_1",
    "cone_radius_2",
]
SYSTEMATIC_LABELS = {
    "JES_up": r"JES +",
    "JES_down": r"JES -",
    "JES_up_JER_down": r"JES+ / JER-",
    "JES_down_JER_up": r"JES- / JER+",
    "JER_up": r"JER +",
    "JER_down": r"JER -",
    "both_up": r"JES/JER +",
    "both_down": r"JES/JER -",
    "cone_radius_1": r"cone $R_1$",
    "cone_radius_2": r"cone $R_2$",
    "pileup_0": r"$\mu=0$",
    "pileup_60": r"$\mu=60$",
    "pileup_100": r"$\mu=100$",
    "pileup_200": r"$\mu=200$",
}

ALL_PLOTS = [
    "discovery_reach",
    "yields_vs_mass",
    "significance_vs_cut",
    "eff_vs_rej",
    "score_distributions",
    "feature_importances",
    "systematics",
    "mass_distributions",
]


@dataclass
class RunResult:
    """Loaded artifacts of one (mass, channel) analysis run."""

    mass: int
    channel: str
    directory: Path
    cuts: np.ndarray  # percentile cut values, ascending
    counts: pd.DataFrame  # process x cut, scaled to total luminosity
    counts_unc: pd.DataFrame | None  # same shape, statistical uncertainties
    sig_bkg: dict[str, np.ndarray]  # contents of *_sig_bkg_metrics.json
    score_thresholds: dict[str, float]  # cut percentile -> score value
    feature_importances: dict[str, float] | None
    predictions: np.ndarray | None
    test_set: pd.DataFrame | None  # "Truth" + "combined_invariant_mass"


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_counts(directory: Path) -> tuple[pd.DataFrame, np.ndarray]:
    counts = pd.read_csv(directory / "gradient_boosting_counts.csv", index_col=0)
    cuts = np.array(sorted(float(c) for c in counts.columns))
    counts = counts[[f"{c:g}" for c in cuts]]
    return counts, cuts


def _read_counts_uncertainty(directory: Path) -> pd.DataFrame | None:
    """Parse the 'mean +- std' strings of *_counts_summary.csv into the std part."""
    summary_path = directory / "gradient_boosting_counts_summary.csv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    if "Process" not in summary.columns:
        return None
    processes = summary.pop("Process")
    cols = list(summary.columns)
    cells = summary[cols].astype(str).to_numpy()
    stds = np.array(
        [
            float(c.split("\u00b1")[1]) if "\u00b1" in c else np.nan
            for c in cells.ravel()
        ]
    ).reshape(cells.shape)
    return pd.DataFrame(stds, index=processes.to_numpy(), columns=cols)


def load_run(directory: Path, mass: int, channel: str) -> RunResult | None:
    """Load the pooled (top-level) artifacts of one run; None if missing."""
    if not (directory / "gradient_boosting_counts.csv").exists():
        return None
    counts, cuts = _read_counts(directory)

    sig_bkg_raw = _load_json(directory / "gradient_boosting_sig_bkg_metrics.json")
    sig_bkg = {k: np.asarray(v, dtype=float) for k, v in sig_bkg_raw.items()}
    if len(cuts) != len(sig_bkg.get("thresholds", [])):
        raise ValueError(
            f"cut mismatch in {directory}: {len(cuts)} counts columns vs "
            f"{len(sig_bkg.get('thresholds', []))} metric thresholds"
        )

    thr_path = directory / "gradient_boosting_score_thresholds.json"
    score_thresholds = _load_json(thr_path) if thr_path.exists() else {}
    fi_path = directory / "gradient_boosting_feature_importances.json"
    feature_importances = _load_json(fi_path) if fi_path.exists() else None
    pred_path = directory / "gradient_boosting_predictions.npy"
    predictions = np.load(pred_path, allow_pickle=False) if pred_path.exists() else None
    par_path = directory / "test_set.parquet"
    test_set = (
        pd.read_parquet(par_path, columns=["Truth", "combined_invariant_mass"])
        if par_path.exists()
        else None
    )

    return RunResult(
        mass=mass,
        channel=channel,
        directory=directory,
        cuts=cuts,
        counts=counts,
        counts_unc=_read_counts_uncertainty(directory),
        sig_bkg=sig_bkg,
        score_thresholds=score_thresholds,
        feature_importances=feature_importances,
        predictions=predictions,
        test_set=test_set,
    )


def discover_masses(root: Path) -> list[int]:
    masses = [
        int(d.name.split("_", 1)[1])
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("MSuu_")
    ]
    return sorted(masses)


def nominal_dir(root: Path, mass: int, channel: str) -> Path:
    return root / f"MSuu_{mass}" / CHANNELS[channel]["path"]


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------


def asymptotic_z(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Asymptotic *expected* discovery significance for a known background.

    Z = sqrt(2 * ((S + B) * ln(1 + S/B) - S)),

    the profile-likelihood test statistic q0 = -2 ln(lambda(0)) evaluated at the
    expected event count n = S + B (Cowan et al. 2011, arXiv:1007.1727). The
    "asymptotic" label refers to the large-count (Wilks) limit of q0's
    distribution, under which Z is standard-normal under the background-only
    hypothesis. Reduces to S/sqrt(B) in the small-signal limit.
    """
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    z = np.zeros_like(s)
    pos = s > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        b_safe = np.where(b > 0, b, 1.0)
        f = (s[pos] + b_safe[pos]) * np.log1p(s[pos] / b_safe[pos]) - s[pos]
        z[pos] = np.sqrt(2.0 * np.maximum(f, 0.0))
    z[pos & (b <= 0)] = np.inf
    return z


def asymptotic_z_error(
    s: np.ndarray, b: np.ndarray, ds: np.ndarray, db: np.ndarray
) -> np.ndarray:
    """Gaussian propagation of the (stat.) uncertainties on S and B through ``asymptotic_z``.

    With f = (S+B) ln(1+S/B) - S and Z = sqrt(2f):
        df/dS = ln(1 + S/B)
        df/dB = ln(1 + S/B) - S/B
    """
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    ds = np.asarray(ds, dtype=float)
    db = np.asarray(db, dtype=float)
    z = asymptotic_z(s, b)
    err = np.zeros_like(s)
    ok = (s > 0) & (b > 0) & np.isfinite(z)
    x = s / b
    df_ds = np.log1p(x)  # d(S+B) ln(1+S/B) - S / dS
    df_db = np.log1p(x) - x  # d(S+B) ln(1+S/B) - S / dB
    err[ok] = np.sqrt((df_ds[ok] * ds[ok]) ** 2 + (df_db[ok] * db[ok]) ** 2) / z[ok]
    return err


@dataclass
class RunQuantities:
    """Per-cut physics quantities derived from one run's scaled counts."""

    cuts: np.ndarray
    s: np.ndarray
    b: np.ndarray
    ds: np.ndarray
    db: np.ndarray
    z_asymp: np.ndarray
    z_asymp_err: np.ndarray
    s_over_b: np.ndarray
    efficiency: np.ndarray
    bgk_pass_through: np.ndarray
    fixed_index: int


def run_quantities(run: RunResult, cut: float) -> RunQuantities:
    s = run.counts.loc[SIG_PROCESS].to_numpy(dtype=float)
    b = run.counts.loc[BKG_SUM].to_numpy(dtype=float)
    if run.counts_unc is not None:
        ds = run.counts_unc.loc[SIG_PROCESS].to_numpy(dtype=float)
        db = run.counts_unc.loc[BKG_SUM].to_numpy(dtype=float)
    else:
        ds = db = np.zeros_like(s)
    s_over_b = np.divide(s, b, out=np.full_like(s, np.inf), where=b > 0)
    return RunQuantities(
        cuts=run.cuts,
        s=s,
        b=b,
        ds=ds,
        db=db,
        z_asymp=asymptotic_z(s, b),
        z_asymp_err=asymptotic_z_error(s, b, ds, db),
        s_over_b=s_over_b,
        efficiency=run.sig_bkg["signal_efficiency"],
        bgk_pass_through=1.0 - run.sig_bkg["background_rejection"],
        fixed_index=int(np.argmin(np.abs(run.cuts - cut))),
    )


def fixed_score_threshold(run: RunResult, cut: float) -> float | None:
    """Score value corresponding to the fixed percentile cut of this run."""
    if not run.score_thresholds:
        return None
    keys = sorted(run.score_thresholds, key=float)
    idx = int(np.argmin(np.abs(np.array([float(k) for k in keys]) - cut)))
    return float(run.score_thresholds[keys[idx]])


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.35,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def annotate(ax: plt.Axes, text: str = "") -> None:
    full = f"{LUMINOSITY_LABEL}  \u2022  {CHI_MASS_LABEL}  {text}"
    ax.text(
        0.03,
        0.97,
        full,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )


def mass_colors(masses: list[int]) -> dict[int, tuple]:
    cmap = plt.get_cmap("coolwarm")
    n = len(masses)
    return {m: cmap(i / max(n - 1, 1)) for i, m in enumerate(masses)}


def pick_legend_masses(masses: list[int], n_max: int = 5) -> set[int]:
    if len(masses) <= n_max:
        return set(masses)
    idx = np.unique(np.linspace(0, len(masses) - 1, n_max).round().astype(int))
    return {masses[i] for i in idx}


def channel_grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    """One subplot per channel; a single channel gets a narrow single-panel figure."""
    if n == 1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
        return fig, np.array([ax])
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 4.5 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    if n < len(axes):
        for ax in axes[n:]:
            fig.delaxes(ax)
        axes = axes[:n]
    return fig, axes


# ---------------------------------------------------------------------------
# Mass-scan plots
# ---------------------------------------------------------------------------


def plot_discovery_reach(
    runs: dict[tuple[int, str], RunResult],
    sys_by_channel: dict[str, dict[str, RunResult]],
    masses: list[int],
    channels: list[str],
    cut: float,
    sys_mass: int,
) -> plt.Figure:
    """Discovery significance vs. M_Suu at a fixed working point, plus best-cut-per-mass."""
    fig, axes = channel_grid(len(channels))
    first = True
    for ax, channel in zip(axes, channels):
        meta = CHANNELS[channel]
        x, z_fix, z_err, z_sb, z_best = [], [], [], [], []
        for m in masses:
            run = runs.get((m, channel))
            if run is None:
                continue
            q = run_quantities(run, cut)
            i = q.fixed_index
            x.append(m / 1000.0)
            z_fix.append(q.z_asymp[i])
            z_err.append(q.z_asymp_err[i])
            z_sb.append(
                q.s[i] / np.sqrt(q.s[i] + q.b[i]) if q.s[i] + q.b[i] > 0 else 0.0
            )
            z_best.append(float(np.nanmax(q.z_asymp)))
        x = np.asarray(x)
        if len(x) < 2:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(meta["label"])
            continue
        ax.errorbar(
            x,
            z_fix,
            yerr=z_err,
            marker="o",
            ms=4.5,
            lw=1.4,
            color=meta["color"],
            label="Z (asymptotic), fixed cut",
            zorder=4,
        )
        ax.plot(
            x,
            z_sb,
            ls="--",
            lw=1.1,
            color=meta["color"],
            alpha=0.55,
            marker="s",
            ms=3.5,
            label=r"S/$\sqrt{S+B}$, fixed cut",
        )
        ax.plot(
            x,
            z_best,
            ls=":",
            lw=1.4,
            color=meta["color"],
            alpha=0.9,
            marker="^",
            ms=4,
            label="Z (asymptotic), best cut per mass",
        )
        # combined systematic error bar at the anchor mass
        sys_runs = sys_by_channel.get(channel, {})
        nom = runs.get((sys_mass, channel))
        if sys_runs and nom is not None:
            q_nom = run_quantities(nom, cut)
            z_nom = q_nom.z_asymp[q_nom.fixed_index]
            deltas = []
            for var in SYSTEMATIC_SINGLE:
                sys_run = sys_runs.get(var)
                if sys_run is None:
                    continue
                q_sys = run_quantities(sys_run, cut)
                deltas.append(abs(q_sys.z_asymp[q_sys.fixed_index] - z_nom))
            if deltas:
                ax.errorbar(
                    [sys_mass / 1000.0],
                    [z_nom],
                    yerr=[max(deltas)],
                    ls="none",
                    marker="none",
                    color="0.35",
                    lw=2.4,
                    capsize=5,
                    zorder=6,
                )
        ax.axhline(5, color="0.5", ls="--", lw=1.0)
        ax.axhline(3, color="0.5", ls=":", lw=1.0)
        ax.text(
            0.98,
            5,
            "5$\\sigma$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.5",
        )
        ax.set_xlim(x.min() - 0.15, x.max() + 0.15)
        ax.set_xlabel("M$_{Suu}$ [TeV]")
        ax.set_ylabel("Discovery significance [$\\sigma$]")
        ax.set_title(meta["label"])
        if first:
            annotate(ax)
            ax.legend(loc="upper right", framealpha=0.9)
            first = False
    fig.suptitle(
        f"Discovery reach vs. diquark mass (fixed {cut:.2f} percentile cut; "
        f"systematic error bar at {sys_mass // 1000} TeV)"
    )
    return fig


def plot_yields_vs_mass(
    runs: dict[tuple[int, str], RunResult],
    masses: list[int],
    channels: list[str],
    cut: float,
    show_backgrounds: bool = False,
    show_sb: bool = False,
) -> plt.Figure:
    """Expected signal and background yields vs. M_Suu at the fixed working point."""
    fig, axes = channel_grid(len(channels))
    for ax, channel in zip(axes, channels):
        meta = CHANNELS[channel]
        x, s_arr, ds_arr, b_arr, db_arr, sb_arr = [], [], [], [], [], []
        proc_arr = {p: [] for p in BKG_PROCESSES} if show_backgrounds else None
        for m in masses:
            run = runs.get((m, channel))
            if run is None:
                continue
            q = run_quantities(run, cut)
            i = q.fixed_index
            x.append(m / 1000.0)
            s_arr.append(q.s[i])
            ds_arr.append(q.ds[i])
            b_arr.append(q.b[i])
            db_arr.append(q.db[i])
            sb_arr.append(q.s_over_b[i])
            if show_backgrounds:
                col = f"{q.cuts[i]:g}"
                for p in BKG_PROCESSES:
                    proc_arr[p].append(
                        run.counts.loc[p, col] if p in run.counts.index else 0.0
                    )
        if not x:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(meta["label"])
            continue
        x = np.asarray(x)
        ax.set_yscale("log")
        ax.errorbar(
            x,
            s_arr,
            yerr=np.minimum(np.asarray(ds_arr), 0.95 * np.asarray(s_arr)),
            marker="o",
            ms=5,
            lw=1.4,
            color="#d62728",
            label=PROCESS_LABELS[SIG_PROCESS],
            zorder=5,
        )
        ax.errorbar(
            x,
            b_arr,
            yerr=np.minimum(np.asarray(db_arr), 0.95 * np.asarray(b_arr)),
            marker="s",
            ms=5,
            lw=1.4,
            color="0.35",
            label=PROCESS_LABELS[BKG_SUM],
            zorder=4,
        )
        if show_backgrounds:
            for p in BKG_PROCESSES:
                vals = np.where(np.asarray(proc_arr[p]) > 0, proc_arr[p], np.nan)
                ax.plot(
                    x,
                    vals,
                    lw=0.9,
                    alpha=0.55,
                    color=SCORE_COLORS[p],
                    label=PROCESS_LABELS[p],
                )
        ax2 = ax.twinx() if show_sb else None
        if ax2 is not None:
            ax2.set_yscale("log")
            ax2.plot(x, sb_arr, color="0.55", lw=1.1, label="S/B")
            ax2.set_ylabel("S/B", color="0.55", fontsize=8)
            ax2.tick_params(axis="y", colors="0.55", labelsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{m / 1000.0:g}" for m in masses if runs.get((m, channel)) is not None]
        )
        ax.set_xlabel("M$_{Suu}$ [TeV]")
        ax.set_ylabel("Expected yield [3 ab$^{-1}$]")
        ax.set_title(meta["label"])
        if channel == channels[0]:
            handles, labels = ax.get_legend_handles_labels()
            if ax2 is not None:
                handles += ax2.get_legend_handles_labels()[0]
                labels += ax2.get_legend_handles_labels()[1]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(labels), 4),
        fontsize=8,
        framealpha=0.9,
    )
    fig.suptitle(f"Expected yields at the {cut:.2f} percentile discriminator cut")
    return fig


def plot_significance_vs_cut(
    runs: dict[tuple[int, str], RunResult],
    masses: list[int],
    channels: list[str],
    cut: float,
) -> plt.Figure:
    """Significance, S/B and signal efficiency vs. percentile discriminator cut."""
    fig, axes = channel_grid(len(channels))
    colors = mass_colors(masses)
    legend_masses = pick_legend_masses(masses)
    for ax, channel in zip(axes, channels):
        meta = CHANNELS[channel]
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        for m in masses:
            run = runs.get((m, channel))
            if run is None:
                continue
            q = run_quantities(run, cut)
            c = colors[m]
            label = f"M$_{{Suu}}$ = {m} GeV" if m in legend_masses else None
            ax.errorbar(
                q.cuts,
                q.z_asymp,
                yerr=q.z_asymp_err,
                color=c,
                lw=1.2,
                marker=".",
                ms=3,
                elinewidth=0.8,
                capsize=2,
                label=label,
            )
            ax2.plot(
                q.cuts,
                np.where(q.s_over_b > 0, q.s_over_b, np.nan),
                ls="--",
                lw=0.9,
                color=c,
                alpha=0.7,
            )
            ax.plot(q.cuts, q.efficiency, ls=":", lw=0.9, color=c, alpha=0.9)
        ax.axvline(cut, color="0.5", ls="--", lw=0.8, alpha=0.8)
        ax.set_xlabel("Discriminator percentile cut")
        ax.set_ylabel("Z (asymptotic) [$\\sigma$]")
        ax2.set_ylabel("S/B", color="0.5", fontsize=8)
        ax2.tick_params(axis="y", colors="0.5", labelsize=8)
        ax.set_title(meta["label"])
        if channel == channels[0]:
            annotate(ax)
            ax.legend(
                loc="upper left",
                fontsize=7,
                title=r"solid: Z  •  dotted: $\varepsilon_S$  •  dashed (right): S/B",
            )
    fig.suptitle(f"Significance vs. discriminator cut (working point {cut:.2f} marked)")
    return fig


def plot_eff_vs_rej(
    runs: dict[tuple[int, str], RunResult],
    masses: list[int],
    channels: list[str],
    cut: float,
) -> plt.Figure:
    """Signal efficiency vs. background pass-through, one curve per mass."""
    fig, axes = channel_grid(len(channels))
    colors = mass_colors(masses)
    legend_masses = pick_legend_masses(masses)
    for ax, channel in zip(axes, channels):
        for m in masses:
            run = runs.get((m, channel))
            if run is None:
                continue
            q = run_quantities(run, cut)
            c = colors[m]
            label = f"M$_{{Suu}}$ = {m} GeV" if m in legend_masses else None
            ax.plot(
                q.bgk_pass_through,
                q.efficiency,
                marker=".",
                ms=3.5,
                lw=1.1,
                color=c,
                label=label,
            )
            ax.plot(
                [q.bgk_pass_through[q.fixed_index]],
                [q.efficiency[q.fixed_index]],
                marker="*",
                ms=11,
                color=c,
                mec="k",
                mew=0.5,
                ls="none",
            )
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1.0)
        ax.set_xlabel("Background pass-through $1-R_B$")
        ax.set_ylabel("Signal efficiency $\\varepsilon_S$")
        ax.set_title(CHANNELS[channel]["label"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(legend_masses), 5),
        fontsize=8,
        title="mass [GeV];  * = fixed working point",
    )
    fig.suptitle("Signal efficiency vs. background rejection")
    return fig


def plot_score_distributions(
    runs: dict[tuple[int, str], RunResult],
    channels: list[str],
    representative_mass: int,
    cut: float,
) -> plt.Figure:
    """Normalized discriminator score distributions, signal vs. backgrounds."""
    fig, axes = channel_grid(len(channels))
    for ax, channel in zip(axes, channels):
        meta = CHANNELS[channel]
        ax.set_title(f"{meta['label']} (M$_{{Suu}}$ = {representative_mass} GeV)")
        ax.set_xlabel("Discriminator score $D$")
        ax.set_ylabel("Normalised density")
        run = runs.get((representative_mass, channel))
        if (
            run is None
            or run.predictions is None
            or run.test_set is None
            or len(run.predictions) != len(run.test_set)
        ):
            ax.text(0.5, 0.5, "data unavailable", ha="center", va="center")
            continue
        scores = run.predictions
        truth = run.test_set["Truth"].to_numpy()
        for proc in [SIG_PROCESS, *BKG_PROCESSES]:
            mask = truth == proc
            if not mask.any():
                continue
            ax.hist(
                scores[mask],
                bins=np.linspace(0.0, 1.0, 51),
                density=True,
                color=SCORE_COLORS[proc],
                alpha=0.65,
                label=PROCESS_LABELS[proc],
            )
        ax.set_yscale("log")
        ax.set_ylim(1e-3, None)
        thr = fixed_score_threshold(run, cut)
        if thr is not None:
            ax.axvline(thr, color="0.2", ls="--", lw=1.2)
            ax.text(
                thr + 0.01,
                2e-3,
                f"fixed cut\n$D \\geq {thr:.4f}$",
                rotation=90,
                va="bottom",
                fontsize=8,
                color="0.2",
            )
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Discriminator score distributions (pooled out-of-fold / test scores)")
    return fig


def plot_feature_importances(
    runs: dict[tuple[int, str], RunResult],
    channels: list[str],
    representative_mass: int,
    top_n: int = 20,
) -> plt.Figure:
    """Top XGBoost feature importances per channel."""
    fig, axes = channel_grid(len(channels))
    for ax, channel in zip(axes, channels):
        meta = CHANNELS[channel]
        ax.set_title(f"{meta['label']} (M$_{{Suu}}$ = {representative_mass} GeV)")
        run = runs.get((representative_mass, channel))
        if run is None or not run.feature_importances:
            ax.text(0.5, 0.5, "data unavailable", ha="center", va="center")
            continue
        items = sorted(run.feature_importances.items(), key=lambda kv: -kv[1])[:top_n]
        names = [n for n, _ in items][::-1]
        values = [v for _, v in items][::-1]
        ax.barh(names, values, color=meta["color"], alpha=0.85)
        for i, v in enumerate(values):
            ax.text(v * 1.02, i, f"{v:.3f}", va="center", fontsize=6.5, color="0.3")
        ax.set_xlabel("XGBoost feature importance")
    fig.suptitle(f"Top {top_n} feature importances")
    return fig


def plot_systematics(
    sys_by_channel: dict[str, dict[str, RunResult]],
    runs: dict[tuple[int, str], RunResult],
    channels: list[str],
    cut: float,
    sys_mass: int,
) -> plt.Figure:
    """Asymptotic significance per systematic variation vs. nominal, at the anchor mass."""
    fig, axes = channel_grid(len(channels))
    for ax, channel in zip(axes, channels):
        meta = CHANNELS[channel]
        ax.set_title(f"{meta['label']} (M$_{{Suu}}$ = {sys_mass} GeV)")
        nom = runs.get((sys_mass, channel))
        sys_runs = sys_by_channel.get(channel, {})
        if nom is None or not sys_runs:
            ax.text(0.5, 0.5, "data unavailable", ha="center", va="center")
            continue
        q_nom = run_quantities(nom, cut)
        z_nom = q_nom.z_asymp[q_nom.fixed_index]
        names, values = [], []
        for var in SYSTEMATIC_ORDER:
            sys_run = sys_runs.get(var)
            if sys_run is None:
                continue
            q_sys = run_quantities(sys_run, cut)
            z_sys = q_sys.z_asymp[q_sys.fixed_index]
            if not np.isfinite(z_sys):
                continue
            names.append(SYSTEMATIC_LABELS[var])
            values.append(z_sys)
        if not values:
            ax.text(0.5, 0.5, "no variations", ha="center", va="center")
            continue
        ax.bar(names, values, color=meta["color"], alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, rotation=45)
        ax.axhline(z_nom, color="0.2", ls="--", lw=1.2, label="nominal")
        spread = max(abs(v - z_nom) for v in values) * 2.0
        ax.set_ylim(max(0.0, z_nom - spread), z_nom + spread)
        ax.set_ylabel("Z (asymptotic) [$\\sigma$]")
        ax.set_xlabel("Systematic variation")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Systematic variations of the discovery significance at the {cut:.2f} cut"
    )
    return fig


def plot_mass_distributions(
    runs: dict[tuple[int, str], RunResult],
    channels: list[str],
    representative_mass: int,
) -> plt.Figure:
    """Invariant mass of the event system (m_Nj), signal vs. backgrounds."""
    fig, axes = channel_grid(len(channels))
    bins = np.linspace(0.0, 1.5 * representative_mass, 150)
    for ax, channel in zip(axes, channels):
        meta = CHANNELS[channel]
        ax.set_title(f"{meta['label']} (M$_{{Suu}}$ = {representative_mass} GeV)")
        ax.set_xlabel("m$_{Nj}$ [GeV]")
        ax.set_ylabel("Normalised density")
        run = runs.get((representative_mass, channel))
        if run is None or run.test_set is None:
            ax.text(0.5, 0.5, "data unavailable", ha="center", va="center")
            continue
        masses = run.test_set["combined_invariant_mass"].to_numpy()
        truth = run.test_set["Truth"].to_numpy()
        for proc in [SIG_PROCESS, *BKG_PROCESSES]:
            mask = truth == proc
            if not mask.any():
                continue
            ax.hist(
                masses[mask],
                bins=bins,
                density=True,
                color=SCORE_COLORS[proc],
                alpha=0.65,
                label=PROCESS_LABELS[proc],
            )
        ax.set_yscale("log")
        ax.set_ylim(1e-6, None)
        ax.set_xlim(0.0, 1.5 * representative_mass)
        ax.axvline(representative_mass, color="0.4", ls=":", lw=1.2)
        ax.text(
            representative_mass + 0.01 * representative_mass,
            2e-6,
            f"M$_{{Suu}}$ = {representative_mass} GeV",
            rotation=90,
            va="bottom",
            fontsize=8,
            color="0.4",
        )
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Invariant mass of the reconstructed event system")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_summary_table(
    runs: dict[tuple[int, str], RunResult],
    masses: list[int],
    channels: list[str],
    cut: float,
) -> pd.DataFrame:
    rows = []
    for m in masses:
        for channel in channels:
            run = runs.get((m, channel))
            if run is None:
                continue
            q = run_quantities(run, cut)
            i = q.fixed_index
            rows.append(
                {
                    "mass_GeV": m,
                    "channel": channel,
                    "S": q.s[i],
                    "B": q.b[i],
                    "S/B": q.s_over_b[i],
                    "Z_asymp": q.z_asymp[i],
                    "Z_asymp_stat_err": q.z_asymp_err[i],
                    "Z_best_cut": float(np.nanmax(q.z_asymp)),
                }
            )
    return pd.DataFrame(rows)


def main(
    root: Annotated[Path, typer.Option(help="Results root directory.")] = Path(
        "results/ChiChi_ManyJets"
    ),
    out: Annotated[Path, typer.Option(help="Output directory for the figures.")] = Path(
        "plots"
    ),
    masses: Annotated[
        list[int] | None,
        typer.Option(help="Mass points (GeV) to include. Default: all found."),
    ] = None,
    channels: Annotated[
        list[str] | None, typer.Option(help="Channels to include. Default: all four.")
    ] = None,
    cut: Annotated[float, typer.Option(help="Fixed percentile working point.")] = 0.95,
    representative_mass: Annotated[
        int,
        typer.Option(
            help="Mass used for the per-event plots (scores, m_Nj, importances)."
        ),
    ] = 8000,
    systematic_mass: Annotated[
        int, typer.Option(help="Mass at which systematic variations are available.")
    ] = 8000,
    plots: Annotated[
        list[str] | None,
        typer.Option(help="Subset of plots to generate. Default: all."),
    ] = None,
    show_backgrounds: Annotated[
        bool,
        typer.Option(
            help="Also plot the individual background components in the "
            "yields-vs-mass figure."
        ),
    ] = False,
    show_sb: Annotated[
        bool,
        typer.Option(
            help="Plot the S/B ratio (twin axis) in the yields-vs-mass figure."
        ),
    ] = False,
) -> None:
    """Generate publication-style figures from the ML analysis results."""
    if not root.is_dir():
        typer.secho(
            f"error: results root '{root}' does not exist",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    all_masses = discover_masses(root)
    if not all_masses:
        typer.secho(
            f"error: no MSuu_* directories found under '{root}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    masses = [m for m in sorted(masses) if m in all_masses] if masses else all_masses
    channels = list(channels) if channels else list(CHANNELS)
    for channel in channels:
        if channel not in CHANNELS:
            typer.secho(
                f"error: unknown channel '{channel}'. Available: {', '.join(CHANNELS)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    wanted = list(plots) if plots else list(ALL_PLOTS)
    unknown = [p for p in wanted if p not in ALL_PLOTS]
    if unknown:
        typer.secho(
            f"error: unknown plot(s): {', '.join(unknown)}. "
            f"Available: {', '.join(ALL_PLOTS)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    setup_style()

    # Load nominal runs ------------------------------------------------------
    runs: dict[tuple[int, str], RunResult] = {}
    for m in masses:
        for channel in channels:
            run = load_run(nominal_dir(root, m, channel), m, channel)
            if run is None:
                typer.secho(
                    f"warning: no results for M={m} GeV, channel {channel} "
                    f"({nominal_dir(root, m, channel)})",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
            else:
                runs[(m, channel)] = run

    # Load systematic variations (anchor mass only) --------------------------
    sys_by_channel: dict[str, dict[str, RunResult]] = {}
    sys_root = root / f"MSuu_{systematic_mass}" / "systematics"
    if sys_root.is_dir():
        for channel in channels:
            sys_by_channel[channel] = {}
            for var_dir in sorted(sys_root.iterdir()):
                if not var_dir.is_dir():
                    continue
                run = load_run(
                    var_dir / CHANNELS[channel]["sys_dir"], systematic_mass, channel
                )
                if run is not None:
                    sys_by_channel[channel][var_dir.name] = run
    else:
        typer.secho(
            f"warning: no systematics under '{sys_root}'; "
            "systematic error bars / plots will be skipped.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    out.mkdir(parents=True, exist_ok=True)

    builders: dict[str, callable] = {
        "discovery_reach": lambda: plot_discovery_reach(
            runs, sys_by_channel, masses, channels, cut, systematic_mass
        ),
        "yields_vs_mass": lambda: plot_yields_vs_mass(
            runs, masses, channels, cut, show_backgrounds, show_sb
        ),
        "significance_vs_cut": lambda: plot_significance_vs_cut(
            runs, masses, channels, cut
        ),
        "eff_vs_rej": lambda: plot_eff_vs_rej(runs, masses, channels, cut),
        "score_distributions": lambda: plot_score_distributions(
            runs, channels, representative_mass, cut
        ),
        "feature_importances": lambda: plot_feature_importances(
            runs, channels, representative_mass
        ),
        "systematics": lambda: plot_systematics(
            sys_by_channel, runs, channels, cut, systematic_mass
        ),
        "mass_distributions": lambda: plot_mass_distributions(
            runs, channels, representative_mass
        ),
    }

    for name in wanted:
        fig = builders[name]()
        path = out / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
        plt.close(fig)

    table = build_summary_table(runs, masses, channels, cut)
    if not table.empty:
        print("\nSummary at the fixed working point (yields scaled to 3000 fb^-1):")
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
