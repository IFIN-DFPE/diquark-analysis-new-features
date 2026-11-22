"""Analyze why the 3-jet vector pT sum (`vector_sum_p_T_3j_mean`) dominates ML feature importance.

Reads the trained XGBoost models, feature importances, and saved test sets under
``results/ChiChi_ManyJets``, plus the raw Delphes ROOT files referenced by
``config/ChiChi_ManyJets``, and produces four explanatory figures in ``plots/``:

- ``3j_pt_overview_importance.pdf``: importance across (mass, channel) & Gain vs TreeSHAP vs AUC
- ``3j_pt_three_regimes.pdf``: distribution, jet multiplicity, and breakdown by N_jets
- ``3j_pt_vs_other_features.pdf``: ROC curves & scaling of 2j vs 3j vs pT_mean across N_jets
- ``3j_pt_physics_origin.pdf``: (E_tot, H_T) plane, raw jet |eta|, and mass scaling 7-9 TeV

Usage::

    uv run scripts/analyze_3jet_pt_sum.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import xgboost as xgb
import yaml
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score, roc_curve

ROOT_RESULTS = Path("results/ChiChi_ManyJets")
ROOT_CONFIG = Path("config/ChiChi_ManyJets")
OUT_DIR = Path("plots")

MASSES = [7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000]

CHANNELS = {
    "ht-ht": {
        "path": "ht-ht/bbt-bbt",
        "label": "ht-ht (bbt-bbt)",
        "sig_cfg": "signal_ht-ht_bbt-bbt.yaml",
    },
    "wb-ht": {
        "path": "wb-ht/jjb-bbt",
        "label": "wb-ht (jjb-bbt)",
        "sig_cfg": "signal_wb-ht_jjb-bbt.yaml",
    },
    "wb-zt": {"path": "wb-zt", "label": "wb-zt", "sig_cfg": "signal_wb-zt.yaml"},
    "zt-ht": {
        "path": "zt-ht_jjt-bbt",
        "label": "zt-ht (jjt-bbt)",
        "sig_cfg": "signal_zt-ht_jjt-bbt.yaml",
    },
}

PROCESS_ORDER = ["SIG:Suu", "BKG:qcd", "BKG:ttbar", "BKG:single_boson", "BKG:diboson"]
PROCESS_LABELS = {
    "SIG:Suu": r"$S_{uu}$ signal",
    "BKG:qcd": "QCD",
    "BKG:ttbar": r"$t\bar{t}$",
    "BKG:single_boson": "Single boson",
    "BKG:diboson": "Diboson",
}
PROCESS_COLORS = {
    "SIG:Suu": "#d62728",
    "BKG:qcd": "#1f77b4",
    "BKG:ttbar": "#2ca02c",
    "BKG:single_boson": "#9467bd",
    "BKG:diboson": "#7f7f7f",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fontsize": 8.5,
        }
    )


def save_figure(fig: plt.Figure, name: str, preview_dir: Path | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {pdf_path}")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        png_path = preview_dir / f"{name}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=160)
        print(f"Saved preview {png_path}")
    plt.close(fig)


def plot_overview_importance(
    df_anchor: pd.DataFrame,
    anchor_dir: Path,
    preview_dir: Path | None,
) -> None:
    """Plot (a) importance across (mass, channel) and (b) Gain vs TreeSHAP vs AUC."""
    # (a) Gather vector_sum_p_T_3j_mean importance across all masses and channels
    mat = np.zeros((len(CHANNELS), len(MASSES)))
    top_other_notes: list[tuple[int, int, str, float]] = []
    ch_keys = list(CHANNELS.keys())
    for i, ch in enumerate(ch_keys):
        for j, m in enumerate(MASSES):
            imp_path = (
                ROOT_RESULTS
                / f"MSuu_{m}"
                / CHANNELS[ch]["path"]
                / "gradient_boosting_feature_importances.json"
            )
            with open(imp_path, encoding="utf-8") as fh:
                imp = json.load(fh)
            val = imp.get("vector_sum_p_T_3j_mean", 0.0)
            mat[i, j] = val
            top_name, top_val = max(imp.items(), key=lambda x: x[1])
            if top_name != "vector_sum_p_T_3j_mean":
                top_other_notes.append((i, j, top_name, top_val))

    # (b) Compute Gain, TreeSHAP (25k sample), and single-feature ROC AUC on anchor
    with open(
        anchor_dir / "gradient_boosting_feature_importances.json", encoding="utf-8"
    ) as fh:
        gain_imp = json.load(fh)
    with open(anchor_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)

    feature_cols = [c for c in df_anchor.columns if c not in ["Truth", "target"]]
    model = xgb.XGBClassifier()
    model.load_model(str(anchor_dir / "gradient_boosting_model.json"))
    booster = model.get_booster()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(df_anchor), size=25000, replace=False)
    X_scaled = scaler.transform(df_anchor[feature_cols].values[idx])
    dm = xgb.DMatrix(X_scaled, feature_names=feature_cols)
    contribs = booster.predict(dm, pred_contribs=True)[:, :-1]
    mean_abs_shap = np.abs(contribs).mean(axis=0)
    shap_share = dict(zip(feature_cols, mean_abs_shap / mean_abs_shap.sum()))

    # Rank union of top Gain and top SHAP features
    top_features = list(
        dict.fromkeys(
            [k for k, _ in sorted(gain_imp.items(), key=lambda x: -x[1])[:8]]
            + [k for k, _ in sorted(shap_share.items(), key=lambda x: -x[1])[:8]]
        )
    )[:11]

    y = df_anchor["target"].values
    aucs = {}
    for c in top_features:
        a = roc_auc_score(y, df_anchor[c].values)
        aucs[c] = max(a, 1.0 - a)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    im = ax1.imshow(mat * 100, cmap="YlOrRd", aspect="auto", vmin=0, vmax=80)
    ax1.set_xticks(range(len(MASSES)))
    ax1.set_xticklabels([f"{m / 1000:.2f}".rstrip("0").rstrip(".") for m in MASSES])
    ax1.set_yticks(range(len(ch_keys)))
    ax1.set_yticklabels([CHANNELS[k]["label"] for k in ch_keys])
    ax1.set_xlabel(r"$M_{S_{uu}}$ [TeV]")
    ax1.set_title("(a) XGBoost Gain share of vector_sum_p_T_3j_mean (%)")
    for i in range(len(ch_keys)):
        for j in range(len(MASSES)):
            v = mat[i, j] * 100
            color = "white" if v > 48 else "black"
            txt = f"{v:.1f}%"
            for ni, nj, tname, tval in top_other_notes:
                if ni == i and nj == j:
                    txt += "*"
            ax1.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=color,
                fontsize=8.2,
                fontweight="bold" if "*" in txt else "normal",
            )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Gain importance [%]")

    # Place clean footnotes below panel (a) without colliding with ticks/xlabel
    note_parts = [
        f"* M={MASSES[nj]} ht-ht: top is {tname} ({tval * 100:.1f}%)"
        for _, nj, tname, tval in top_other_notes
    ]
    note_text = (
        ";  ".join(note_parts)
        + "\n"
        + r"Note: $M_{S_{uu}}=8$ TeV uses $p_T^{\min}=100$ GeV; all other masses use $p_T^{\min}=50$ GeV."
    )
    ax1.text(
        0.5,
        -0.22,
        note_text,
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
    )

    y_pos = np.arange(len(top_features))[::-1]
    gains = [gain_imp[c] * 100 for c in top_features]
    shaps = [shap_share[c] * 100 for c in top_features]
    h = 0.36
    ax2.barh(
        y_pos + h / 2,
        gains,
        height=h,
        color="#d62728",
        alpha=0.88,
        label="XGBoost Gain share [%]",
    )
    ax2.barh(
        y_pos - h / 2,
        shaps,
        height=h,
        color="#1f77b4",
        alpha=0.88,
        label="TreeSHAP share [%]",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{c} (AUC {aucs[c]:.3f})" for c in top_features], fontsize=8)
    ax2.set_xlabel("Relative importance [%]")
    ax2.set_xlim(0, 86)
    ax2.set_title(r"(b) $M_{S_{uu}}=8$ TeV, wb-ht: Gain vs. TreeSHAP vs. 1D AUC")
    ax2.legend(loc="lower right")
    for yp, g, s in zip(y_pos, gains, shaps):
        if g > 15:
            ax2.text(
                g + 1.0,
                yp + h / 2,
                f"{g:.1f}%",
                va="center",
                fontsize=7.5,
                color="#b2182b",
            )
            ax2.text(
                s + 1.0,
                yp - h / 2,
                f"{s:.1f}%",
                va="center",
                fontsize=7.5,
                color="#1f77b4",
            )

    fig.tight_layout()
    save_figure(fig, "3j_pt_overview_importance", preview_dir)


def plot_three_regimes(
    df_anchor: pd.DataFrame, root_split_gev: float, preview_dir: Path | None
) -> None:
    """Plot the three-regime anatomy of `vector_sum_p_T_3j_mean` at MSuu=8000, wb-ht."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # Panel (a): Full distribution with -1 bin clearly shown + root split line
    ax = axes[0]
    bins = np.concatenate(([-120, -0.5], np.linspace(0, 3200, 33)))
    for proc in PROCESS_ORDER:
        sub = df_anchor[df_anchor["Truth"] == proc]["vector_sum_p_T_3j_mean"].values
        # Map -1 sentinel to -60 so the bar is visible to the left of 0
        vals = np.where(sub < 0, -60.0, np.clip(sub, 0, 3199))
        counts, edges = np.histogram(vals, bins=bins)
        probs = counts / len(sub)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(
            edges[:-1],
            probs,
            where="post",
            color=PROCESS_COLORS[proc],
            lw=1.6,
            label=PROCESS_LABELS[proc],
        )

    ax.axvline(
        root_split_gev,
        color="black",
        ls="--",
        lw=1.3,
        label=f"Tree root split ({root_split_gev:.0f} GeV)",
    )
    ax.axvspan(-160, 0, color="#eeeeee", alpha=0.9, zorder=0)
    ax.text(
        -80,
        0.0025,
        r"$N_{\mathrm{jets}}<3$" "\n" r"($-1$)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#333333",
    )
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 0.95)
    ax.set_xlim(-160, 3200)
    ax.set_xlabel("vector_sum_p_T_3j_mean [GeV]")
    ax.set_ylabel("Fraction of events per bin")
    ax.set_title("(a) Full distribution & root split (8 TeV wb-ht)")
    ax.legend(loc="upper right", fontsize=7.5)

    # Panel (b): Jet multiplicity distribution
    ax = axes[1]
    nj_vals = np.arange(0, 9)
    width = 0.16
    offsets = np.linspace(-2 * width, 2 * width, len(PROCESS_ORDER))
    for off, proc in zip(offsets, PROCESS_ORDER):
        sub = df_anchor[df_anchor["Truth"] == proc]["jet_multiplicity"].values
        fracs = [np.mean(sub == k) for k in range(8)] + [np.mean(sub >= 8)]
        ax.bar(
            nj_vals + off,
            fracs,
            width=width,
            color=PROCESS_COLORS[proc],
            alpha=0.85,
            label=PROCESS_LABELS[proc],
        )
    ax.axvline(2.5, color="#555555", ls=":", lw=1.1)
    ax.axvline(3.5, color="#555555", ls=":", lw=1.1)
    ax.text(
        1.0,
        0.49,
        "Regime 1:\n<3 jets (-1)",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
    )
    ax.text(
        3.0,
        0.49,
        "Regime 2:\n=3 jets",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
    )
    ax.text(
        5.8,
        0.49,
        r"Regime 3: $\geq 4$ jets" "\n" "(94.2% of signal)",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#b2182b",
    )
    ax.set_xticks(nj_vals)
    ax.set_xticklabels([str(k) for k in range(8)] + ["8+"])
    ax.set_ylim(0, 0.54)
    ax.set_xlabel(
        r"Selected jet multiplicity $N_{\mathrm{jets}}$ ($p_T\geq 100$ GeV, $|\eta|\leq 2.5$)"
    )
    ax.set_ylabel("Fraction of events")
    ax.set_title(r"(b) Jet multiplicity by process")

    # Panel (c): Distribution conditioned on N_jets = 3, 4, >=5 for Signal vs QCD vs ttbar
    ax = axes[2]
    regimes = [
        ("= 3", df_anchor["jet_multiplicity"] == 3),
        ("= 4", df_anchor["jet_multiplicity"] == 4),
        (r"$\geq$ 5", df_anchor["jet_multiplicity"] >= 5),
    ]
    focus_procs = ["SIG:Suu", "BKG:qcd", "BKG:ttbar"]
    positions = []
    box_data = []
    box_colors = []
    tick_pos = []
    tick_labels = []
    base_x = 0
    for r_label, r_mask in regimes:
        sub_r = df_anchor[r_mask]
        group_xs = []
        for idx_p, proc in enumerate(focus_procs):
            vals = sub_r[sub_r["Truth"] == proc]["vector_sum_p_T_3j_mean"].values
            x_p = base_x + idx_p * 0.85
            positions.append(x_p)
            box_data.append(vals)
            box_colors.append(PROCESS_COLORS[proc])
            group_xs.append(x_p)
        tick_pos.append(np.mean(group_xs))
        tick_labels.append(f"$N_{{\\mathrm{{jets}}}}$ {r_label}")
        base_x += 3.2

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        whis=(10, 90),
        medianprops={"color": "black", "lw": 1.6},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.axhline(root_split_gev, color="black", ls="--", lw=1.2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("vector_sum_p_T_3j_mean [GeV]")
    ax.set_ylim(0, 2500)
    ax.set_title(r"(c) Conditional scale (10th–90th pct whiskers)")
    # Custom legend for panel c
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PROCESS_COLORS[p], alpha=0.72)
        for p in focus_procs
    ]
    ax.legend(
        handles, [PROCESS_LABELS[p] for p in focus_procs], loc="upper left", fontsize=8
    )

    fig.tight_layout()
    save_figure(fig, "3j_pt_three_regimes", preview_dir)


def plot_vs_other_features(df_anchor: pd.DataFrame, preview_dir: Path | None) -> None:
    """Compare ROC curves and behavior across N_jets vs 2j, p_T_mean, and m3j."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    y = df_anchor["target"].values
    compare_cols = [
        ("vector_sum_p_T_3j_mean", "#d62728", "-", 2.2),
        ("vector_sum_p_T_3j_max", "#ff7f0e", "--", 1.7),
        ("centrality", "#2ca02c", "-", 1.7),
        ("p_T_sum", "#9467bd", "-.", 1.7),
        ("vector_sum_p_T_2j_mean", "#1f77b4", "-", 1.7),
        ("jet_multiplicity", "#8c564b", ":", 1.8),
        ("p_T_mean", "#17becf", "--", 1.7),
        ("m3j_mean", "#7f7f7f", ":", 1.6),
    ]
    for col, color, ls, lw in compare_cols:
        scores = df_anchor[col].values
        auc = roc_auc_score(y, scores)
        if auc < 0.5:
            scores = -scores
            auc = 1.0 - auc
        fpr, tpr, _ = roc_curve(y, scores)
        # Subsample points for clean rendering
        step = max(1, len(fpr) // 500)
        ax1.plot(
            fpr[::step],
            tpr[::step],
            color=color,
            ls=ls,
            lw=lw,
            label=f"{col} (AUC = {auc:.3f})",
        )

    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("Background false positive rate (unweighted)")
    ax1.set_ylabel("Signal true positive rate")
    ax1.set_title(r"(a) Single-feature ROC curves ($M_{S_{uu}}=8$ TeV, wb-ht)")
    ax1.legend(loc="lower right", fontsize=7.8)

    # Panel (b): Median & IQR of vector_sum_p_T_3j_mean, vector_sum_p_T_2j_mean, p_T_mean vs N_jets
    nj_range = [2, 3, 4, 5, 6]
    features_b = [
        ("vector_sum_p_T_3j_mean", "3-jet vector $p_T$ sum (mean)", "-", "o"),
        ("vector_sum_p_T_2j_mean", "2-jet vector $p_T$ sum (mean)", "--", "s"),
        ("p_T_mean", "Mean single-jet $p_T$ (p_T_mean)", ":", "^"),
    ]
    for proc, color in [("SIG:Suu", "#d62728"), ("BKG:qcd", "#1f77b4")]:
        sub_p = df_anchor[df_anchor["Truth"] == proc]
        for feat, flabel, ls, marker in features_b:
            meds = []
            for n in nj_range:
                vals = sub_p[sub_p["jet_multiplicity"] == n][feat].values
                meds.append(np.median(vals) if len(vals) > 0 else np.nan)
            label = f"{PROCESS_LABELS[proc]}: {flabel}"
            ax2.plot(
                nj_range,
                meds,
                color=color,
                ls=ls,
                marker=marker,
                ms=5,
                lw=1.7,
                label=label,
            )

    ax2.annotate(
        r"At $N_{\mathrm{jets}}=3$, QCD 3-jet vector sum"
        "\n"
        r"collapses to 137 GeV ($\sum_{i=1}^3 \vec{p}_{T,i}\approx 0$),"
        "\n"
        r"while 2-jet & p_T_mean stay ~800 GeV!",
        xy=(3, 137),
        xytext=(3.3, 270),
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.1),
    )
    ax2.annotate(
        r"At $N_{\mathrm{jets}}=4$, momentum"
        "\n"
        r"conservation gives 3j mean $\approx$ p_T_mean",
        xy=(4, 1508),
        xytext=(4.15, 1920),
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
    )
    ax2.set_xticks(nj_range)
    ax2.set_xlabel(r"Selected jet multiplicity $N_{\mathrm{jets}}$")
    ax2.set_ylabel("Median value [GeV]")
    ax2.set_ylim(-100, 2650)
    ax2.set_title(r"(b) Why k=3 beats k=2 and p_T_mean across $N_{\mathrm{jets}}$")
    ax2.legend(loc="upper center", fontsize=7.0, ncol=2)

    fig.tight_layout()
    save_figure(fig, "3j_pt_vs_other_features", preview_dir)


def plot_physics_origin(df_anchor: pd.DataFrame, preview_dir: Path | None) -> None:
    """Plot the collider physics origin: (E_tot, H_T), raw jet |eta|, and mass scaling."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel (a): 2D contours / histograms of total_energy vs total_p_T (H_T)
    ax = axes[0]
    sig = df_anchor[df_anchor["Truth"] == "SIG:Suu"]
    qcd = df_anchor[df_anchor["Truth"] == "BKG:qcd"]

    xbins = np.linspace(3000, 10500, 45)
    ybins = np.linspace(500, 9000, 45)
    h_qcd, xe, ye = np.histogram2d(
        qcd["total_energy"], qcd["total_p_T"], bins=[xbins, ybins], density=True
    )
    h_sig, _, _ = np.histogram2d(
        sig["total_energy"], sig["total_p_T"], bins=[xbins, ybins], density=True
    )
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])

    # Filled contour for QCD, line contour for Signal
    q_levels = np.quantile(h_qcd[h_qcd > 0], [0.25, 0.5, 0.75, 0.9])
    s_levels = np.quantile(h_sig[h_sig > 0], [0.25, 0.5, 0.75, 0.9])
    ax.contourf(
        xc,
        yc,
        h_qcd.T,
        levels=np.append(q_levels, h_qcd.max() * 1.1),
        cmap="Blues",
        alpha=0.7,
    )
    ax.contour(xc, yc, h_qcd.T, levels=q_levels, colors="#1f77b4", linewidths=1.0)
    ax.contour(xc, yc, h_sig.T, levels=s_levels, colors="#d62728", linewidths=1.5)
    e_line = np.linspace(3000, 10500, 50)
    ax.plot(
        e_line,
        e_line,
        "k--",
        lw=0.9,
        alpha=0.6,
        label=r"$H_T = E_{\mathrm{tot}}$ (centrality = 1)",
    )
    ax.plot(
        e_line,
        0.33 * e_line,
        color="#1f77b4",
        ls=":",
        lw=1.0,
        label="centrality = 0.33 (QCD mean)",
    )
    ax.set_xlabel(r"Total jet energy $E_{\mathrm{tot}}$ [GeV]")
    ax.set_ylabel(r"Scalar $H_T = \sum p_T$ [GeV]")
    ax.set_title(r"(a) Same $\sqrt{\hat{s}}$ window, different centrality")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color="#d62728",
            lw=1.8,
            label=r"$S_{uu}$ signal (central $s$-channel)",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, color="#1f77b4", alpha=0.5, label=r"QCD (forward $t$-channel)"
        ),
        plt.Line2D([0], [0], color="k", ls="--", lw=0.9, label=r"centrality = 1.0"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5)

    # Panel (b): Raw leading-jet |eta| from Delphes ROOT files (before |eta|<=2.5 cut)
    ax = axes[1]
    with open(ROOT_CONFIG / "MSuu_8000" / "background.yaml", encoding="utf-8") as fh:
        bkg_cfg = yaml.safe_load(fh)["backgrounds"]
    with open(
        ROOT_CONFIG / "MSuu_8000" / "signal_wb-ht_jjb-bbt.yaml", encoding="utf-8"
    ) as fh:
        sig_cfg = yaml.safe_load(fh)["signal"]

    raw_sources = [("SIG:Suu", sig_cfg["file"])] + [
        (f"BKG:{k}", v["file"]) for k, v in bkg_cfg.items()
    ]
    eta_bins = np.linspace(0, 4.5, 37)
    for proc, path in raw_sources:
        with uproot.open(path) as f:
            arr = f["Delphes"].arrays(["Jet/Jet.Eta"], entry_stop=40000, library="ak")[
                "Jet/Jet.Eta"
            ]
        has1 = ak.num(arr) >= 1
        lead_eta = np.abs(ak.to_numpy(arr[has1, 0]))
        ax.hist(
            lead_eta,
            bins=eta_bins,
            density=True,
            histtype="step",
            lw=1.6,
            color=PROCESS_COLORS[proc],
            label=PROCESS_LABELS[proc],
        )

    ax.axvline(
        2.5, color="black", ls="--", lw=1.2, label=r"Pre-selection $|\eta| \leq 2.5$"
    )
    ax.set_xlabel(r"Raw leading jet $|\eta_1|$ (Delphes)")
    ax.set_ylabel("Probability density")
    ax.set_title(r"(b) Leading jet $|\eta_1|$ before cuts (8 TeV)")
    ax.legend(loc="upper right", fontsize=7.5)

    # Panel (c): Scaling across M_Suu = 7000..9000 GeV (wb-ht channel)
    ax = axes[2]
    sig_meds_4p = []
    qcd_meds_4p = []
    tt_meds_4p = []
    aucs_mass = []
    for m in MASSES:
        p = ROOT_RESULTS / f"MSuu_{m}" / "wb-ht/jjb-bbt/test_set.parquet"
        df_m = pd.read_parquet(
            p, columns=["Truth", "target", "jet_multiplicity", "vector_sum_p_T_3j_mean"]
        )
        aucs_mass.append(
            roc_auc_score(df_m["target"].values, df_m["vector_sum_p_T_3j_mean"].values)
        )
        sub4p = df_m[df_m["jet_multiplicity"] >= 4]
        sig_meds_4p.append(
            np.median(
                sub4p[sub4p["Truth"] == "SIG:Suu"]["vector_sum_p_T_3j_mean"].values
            )
        )
        qcd_meds_4p.append(
            np.median(
                sub4p[sub4p["Truth"] == "BKG:qcd"]["vector_sum_p_T_3j_mean"].values
            )
        )
        tt_meds_4p.append(
            np.median(
                sub4p[sub4p["Truth"] == "BKG:ttbar"]["vector_sum_p_T_3j_mean"].values
            )
        )

    x_tev = np.array(MASSES) / 1000.0
    ax.plot(
        x_tev,
        sig_meds_4p,
        "o-",
        color="#d62728",
        lw=1.8,
        label=r"$S_{uu}$ median ($N_{\mathrm{jets}}\geq 4$)",
    )
    ax.plot(
        x_tev,
        qcd_meds_4p,
        "s-",
        color="#1f77b4",
        lw=1.6,
        label=r"QCD median ($N_{\mathrm{jets}}\geq 4$)",
    )
    ax.plot(
        x_tev,
        tt_meds_4p,
        "^-",
        color="#2ca02c",
        lw=1.6,
        label=r"$t\bar{t}$ median ($N_{\mathrm{jets}}\geq 4$)",
    )
    ax.set_xlabel(r"$M_{S_{uu}}$ [TeV]")
    ax.set_ylabel(r"Median vector_sum_p_T_3j_mean ($N_{\mathrm{jets}}\geq 4$) [GeV]")
    ax.set_ylim(400, 2000)

    ax_r = ax.twinx()
    ax_r.plot(
        x_tev,
        aucs_mass,
        "D--",
        color="#444444",
        lw=1.3,
        ms=4,
        label="1D ROC AUC (all events)",
    )
    ax_r.set_ylabel("1D ROC AUC", color="#444444")
    ax_r.set_ylim(0.90, 0.98)
    ax_r.grid(False)

    ax.set_title("(c) Mass scaling across $M_{S_{uu}}=7\u20139$ TeV (wb-ht)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.3)

    fig.tight_layout()
    save_figure(fig, "3j_pt_physics_origin", preview_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory to also write PNG previews",
    )
    args = parser.parse_args()

    setup_style()
    anchor_dir = ROOT_RESULTS / "MSuu_8000" / "wb-ht/jjb-bbt"
    df_anchor = pd.read_parquet(anchor_dir / "test_set.parquet")
    preview_dir = args.preview_dir

    # Root split GeV from Tree 0
    with open(anchor_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    feature_cols = [c for c in df_anchor.columns if c not in ["Truth", "target"]]
    idx_3j = feature_cols.index("vector_sum_p_T_3j_mean")
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0, idx_3j] = 0.221650824
    root_split_gev = float(scaler.inverse_transform(dummy)[0, idx_3j])
    print(f"Anchor root split threshold: {root_split_gev:.2f} GeV")

    # plot_overview_importance(df_anchor, anchor_dir, preview_dir)
    plot_three_regimes(df_anchor, root_split_gev, preview_dir)
    # plot_vs_other_features(df_anchor, preview_dir)
    # plot_physics_origin(df_anchor, preview_dir)


if __name__ == "__main__":
    main()
