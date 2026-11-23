import os
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from diquark.evaluation.metrics import (
    calculate_metrics,
    calculate_signal_background_metrics,
)
from diquark.evaluation.visualizations import plot_results


class TestVisualizations(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        n_sig, n_bkg = 150, 250
        self.df_test = pd.DataFrame(
            {
                "f1": rng.normal(size=n_sig + n_bkg),
                "f2": rng.normal(size=n_sig + n_bkg),
                "f3": rng.normal(size=n_sig + n_bkg),
                "Truth": ["SIG:Suu"] * n_sig + ["BKG:qcd"] * n_bkg,
                "target": [1] * n_sig + [0] * n_bkg,
            }
        )
        preds = np.concatenate(
            [rng.beta(5, 2, size=n_sig), rng.beta(2, 5, size=n_bkg)]
        )
        weights = np.concatenate(
            [np.full(n_sig, 0.05), np.full(n_bkg, 2.0)]
        )
        thresholds = [0.5, 0.7, 0.9]
        cross_sections = {"SIG:Suu": 0.003, "BKG:qcd": 5.0}
        self.results = {
            "predictions": preds,
            "sample_weights": weights,
            "metrics": calculate_metrics(
                self.df_test["target"].to_numpy(),
                preds,
                weights,
                np.array(thresholds),
                use_real_event_percentiles=False,
            ),
            "sig_bkg_metrics": calculate_signal_background_metrics(
                self.df_test["target"].to_numpy(),
                preds,
                thresholds,
                weights,
                3000.0,
                cross_sections,
                use_real_event_percentiles=False,
            ),
            "feature_importances": np.array([0.6, 0.3, 0.1]),
        }

    def test_all_plot_types_render_headless_to_pdf(self) -> None:
        plot_types = [
            "roc_curve",
            "pr_curve",
            "weighted_pr_curve",
            "feature_importances",
            "sig_bkg_metrics",
            "scores_histogram",
            "scores_weighted_histogram",
        ]
        prev_display = os.environ.pop("DISPLAY", None)
        try:
            figs = plot_results(
                self.results, self.df_test, plot_types, real_percentiles=False
            )
            self.assertEqual(set(figs.keys()), set(plot_types))
            with tempfile.TemporaryDirectory() as tmp:
                for name, fig in figs.items():
                    self.assertIsInstance(fig, Figure)
                    out_pdf = Path(tmp) / f"{name}.pdf"
                    fig.savefig(out_pdf, bbox_inches="tight")
                    plt.close(fig)
                    self.assertTrue(out_pdf.exists() and out_pdf.stat().st_size > 0)
        finally:
            if prev_display is not None:
                os.environ["DISPLAY"] = prev_display


if __name__ == "__main__":
    unittest.main()
