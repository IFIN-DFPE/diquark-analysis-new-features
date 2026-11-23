import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from diquark.data.feature_cache import FeatureCache, compute_feature_cache_key
from diquark.data.preprocessor import Preprocessor
from diquark.evaluation.tuning_metrics import (
    asymptotic_z,
    asymptotic_z_error,
    compute_process_weights,
    compute_training_weights,
    evaluate_overfitting_diagnostics,
    evaluate_tail_metrics,
    summarize_weight_distribution,
    weighted_partial_roc_auc,
)
from diquark.models.gradient_boosting import GradientBoostingModel


class TestTrainingWeights(unittest.TestCase):
    def setUp(self) -> None:
        # Skewed MC counts vs cross-sections (mimics QCD vs diboson/ttbar)
        self.truth = np.array(
            ["SIG:Suu"] * 100
            + ["BKG:qcd"] * 100
            + ["BKG:ttbar"] * 200
            + ["BKG:diboson"] * 100
        )
        self.cross_sections = {
            "SIG:Suu": 0.003,
            "BKG:qcd": 5.0,
            "BKG:ttbar": 0.002,
            "BKG:diboson": 0.0001,
        }

    def test_none_mode_returns_none(self) -> None:
        self.assertIsNone(
            compute_training_weights(self.truth, self.cross_sections, mode="none")
        )

    def test_class_balanced_weights(self) -> None:
        w = compute_training_weights(
            self.truth, self.cross_sections, mode="class_balanced"
        )
        assert w is not None
        self.assertAlmostEqual(float(w.mean()), 1.0, places=10)
        sig_mask = self.truth == "SIG:Suu"
        self.assertAlmostEqual(float(w[sig_mask].sum()), float(w[~sig_mask].sum()), places=10)
        # Equal weight per MC event within background
        self.assertTrue(np.allclose(w[~sig_mask], w[~sig_mask][0]))

    def test_physics_weights_alpha_1_matches_cross_sections(self) -> None:
        w = compute_training_weights(
            self.truth, self.cross_sections, mode="physics", alpha=1.0, balance_classes=True
        )
        assert w is not None
        self.assertAlmostEqual(float(w.mean()), 1.0, places=10)
        summary = summarize_weight_distribution(self.truth, w)

        self.assertAlmostEqual(summary["SIG"]["weight_sum"], summary["BKG"]["weight_sum"], places=9)
        total_bkg_xsec = (
            self.cross_sections["BKG:qcd"]
            + self.cross_sections["BKG:ttbar"]
            + self.cross_sections["BKG:diboson"]
        )
        for proc in ("BKG:qcd", "BKG:ttbar", "BKG:diboson"):
            self.assertAlmostEqual(
                summary[proc]["class_weight_share"],
                self.cross_sections[proc] / total_bkg_xsec,
                places=12,
            )

    def test_physics_weights_alpha_interpolation_and_unbalanced(self) -> None:
        w_0 = compute_training_weights(
            self.truth, self.cross_sections, mode="physics", alpha=0.0
        )
        w_cb = compute_training_weights(
            self.truth, self.cross_sections, mode="class_balanced"
        )
        assert w_0 is not None and w_cb is not None
        self.assertTrue(np.allclose(w_0, w_cb))

        w_05 = compute_training_weights(
            self.truth, self.cross_sections, mode="physics", alpha=0.5
        )
        w_10 = compute_training_weights(
            self.truth, self.cross_sections, mode="physics", alpha=1.0
        )
        assert w_05 is not None and w_10 is not None
        s_0 = summarize_weight_distribution(self.truth, w_0)
        s_05 = summarize_weight_distribution(self.truth, w_05)
        s_10 = summarize_weight_distribution(self.truth, w_10)
        self.assertLess(
            s_0["BKG:qcd"]["class_weight_share"],
            s_05["BKG:qcd"]["class_weight_share"],
        )
        self.assertLess(
            s_05["BKG:qcd"]["class_weight_share"],
            s_10["BKG:qcd"]["class_weight_share"],
        )

        # balance_classes=False preserves global physical S/B ratio and mean(w)=1
        w_unbal = compute_training_weights(
            self.truth, self.cross_sections, mode="physics", alpha=1.0, balance_classes=False
        )
        assert w_unbal is not None
        self.assertAlmostEqual(float(w_unbal.mean()), 1.0, places=10)
        s_unbal = summarize_weight_distribution(self.truth, w_unbal)
        total_bkg_xsec = (
            self.cross_sections["BKG:qcd"]
            + self.cross_sections["BKG:ttbar"]
            + self.cross_sections["BKG:diboson"]
        )
        self.assertAlmostEqual(
            s_unbal["SIG"]["weight_sum"] / s_unbal["BKG"]["weight_sum"],
            self.cross_sections["SIG:Suu"] / total_bkg_xsec,
            places=12,
        )


class TestTailMetrics(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        n_sig, n_qcd, n_tt = 1000, 2000, 1000
        self.truth = np.array(
            ["SIG:Suu"] * n_sig + ["BKG:qcd"] * n_qcd + ["BKG:ttbar"] * n_tt
        )
        self.y_true = np.array([1] * n_sig + [0] * (n_qcd + n_tt))
        self.cross_sections = {"SIG:Suu": 0.003, "BKG:qcd": 5.0, "BKG:ttbar": 0.002}
        sig_scores = rng.beta(6, 1.5, size=n_sig)
        qcd_scores = rng.beta(1.5, 8, size=n_qcd)
        tt_scores = rng.beta(2, 6, size=n_tt)
        self.y_pred = np.concatenate([sig_scores, qcd_scores, tt_scores])

    def test_asymptotic_z_and_error(self) -> None:
        self.assertAlmostEqual(asymptotic_z(0.0, 10.0), 0.0)
        self.assertTrue(np.isinf(asymptotic_z(2.0, 0.0)))
        # Small signal limit: Z -> S / sqrt(B)
        s, b = 0.01, 1000.0
        self.assertAlmostEqual(asymptotic_z(s, b), s / np.sqrt(b), places=6)
        err = asymptotic_z_error(2.0, 1.0, 0.1, 0.2)
        self.assertTrue(np.isfinite(err) and err > 0.0)

    def test_partial_roc_auc_perfect_classifier(self) -> None:
        y = np.array([1, 1, 0, 0, 0, 0])
        p = np.array([0.95, 0.85, 0.4, 0.3, 0.2, 0.1])
        w = np.ones_like(y, dtype=float)
        self.assertAlmostEqual(weighted_partial_roc_auc(y, p, w, max_fpr=2e-3), 1.0)

    def test_eff_grid_calibration_invariance_and_zero_bkg_floor(self) -> None:
        res1 = evaluate_tail_metrics(
            self.y_true, self.y_pred, self.truth, self.cross_sections
        )
        # Monotone score re-calibration must not change eff_grid yields or partial ROC AUC
        res2 = evaluate_tail_metrics(
            self.y_true, self.y_pred**3, self.truth, self.cross_sections
        )
        self.assertAlmostEqual(
            res1["partial_weighted_roc_auc"],
            res2["partial_weighted_roc_auc"],
            places=10,
        )
        self.assertAlmostEqual(res1["mean_z_eff_grid"], res2["mean_z_eff_grid"], places=10)
        for r1, r2 in zip(res1["eff_grid"], res2["eff_grid"]):
            self.assertAlmostEqual(r1["s"], r2["s"], places=10)
            self.assertAlmostEqual(r1["b"], r2["b"], places=10)

        # Perfect separation triggers B=0 floor cleanly without inf/nan in mean_z_eff_grid
        perfect_pred = np.where(self.y_true == 1, 0.99, 0.01)
        res_perf = evaluate_tail_metrics(
            self.y_true, perfect_pred, self.truth, self.cross_sections
        )
        self.assertTrue(res_perf["has_zero_bkg_point"])
        self.assertTrue(np.isfinite(res_perf["mean_z_eff_grid"]))

    def test_overfitting_diagnostics_runs(self) -> None:
        w = compute_process_weights(self.truth, self.cross_sections)
        self.assertEqual(len(w), len(self.truth))
        ov = evaluate_overfitting_diagnostics(
            self.y_true,
            self.y_pred,
            self.truth,
            self.y_true,
            self.y_pred * 0.9,
            self.truth,
            self.cross_sections,
        )
        self.assertIn("weighted_logloss_gap", ov)
        self.assertIn("partial_weighted_roc_auc_gap", ov)


class TestGradientBoostingModel(unittest.TestCase):
    def test_defaults_and_custom_params(self) -> None:
        m_def = GradientBoostingModel({})
        m_def.build(10)
        params = m_def.model.get_params()
        self.assertEqual(params["max_depth"], 8)
        self.assertEqual(params["learning_rate"], 0.1)
        self.assertEqual(params["early_stopping_rounds"], 2)
        self.assertEqual(params["device"], "cpu")

        m_gpu = GradientBoostingModel(
            {
                "n_estimators": 20,
                "learning_rate": 0.05,
                "max_depth": 4,
                "max_leaves": 15,
                "grow_policy": "lossguide",
                "min_child_weight": 10.0,
                "gamma": 0.5,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "colsample_bylevel": 0.8,
                "reg_lambda": 5.0,
                "reg_alpha": 0.2,
                "max_bin": 512,
                "scale_pos_weight": 2.0,
                "device": "gpu",
                "eval_metric": "aucpr",
                "early_stopping_rounds": 15,
                "n_jobs": 2,
                "verbose": False,
            }
        )
        m_gpu.build(10)
        gp = m_gpu.model.get_params()
        self.assertEqual(gp["device"], "cuda")
        self.assertEqual(gp["max_leaves"], 15)
        self.assertEqual(gp["grow_policy"], "lossguide")
        self.assertEqual(gp["min_child_weight"], 10.0)
        self.assertEqual(gp["reg_lambda"], 5.0)
        self.assertEqual(gp["max_bin"], 512)
        self.assertEqual(gp["early_stopping_rounds"], 15)

        with self.assertRaises(ValueError):
            GradientBoostingModel({"nonexistent_param": 123})
        with self.assertRaises(ValueError):
            GradientBoostingModel({"device": "tpu"})

    def test_train_with_weights_early_stopping_and_roundtrip(self) -> None:
        rng = np.random.default_rng(11)
        X = rng.normal(size=(200, 6))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        w = rng.uniform(0.5, 2.0, size=200)

        m = GradientBoostingModel(
            {
                "n_estimators": 15,
                "max_depth": 3,
                "early_stopping_rounds": 5,
                "n_jobs": 2,
                "verbose": False,
            }
        )
        m.build(6)
        hist = m.train(
            X[:140], y[:140], X[140:], y[140:], sample_weight=w[:140], val_sample_weight=w[140:]
        )
        self.assertIn("validation_0", hist)
        self.assertIsNotNone(m.best_iteration)
        preds = m.predict(X[140:])

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.json")
            m.save(path)
            m2 = GradientBoostingModel({"n_jobs": 2})
            m2.load(path)
            self.assertTrue(np.allclose(preds, m2.predict(X[140:])))

        # Also test early_stopping disabled and X_val=None
        m_no_es = GradientBoostingModel(
            {"n_estimators": 5, "early_stopping_rounds": None, "n_jobs": 2, "verbose": False}
        )
        m_no_es.build(6)
        hist2 = m_no_es.train(X[:140], y[:140], None, None)
        self.assertEqual(hist2, {})


class TestPreprocessorAndCache(unittest.TestCase):
    def test_preprocessor_split_parity_and_no_val_oversample(self) -> None:
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "f1": rng.normal(size=200),
                "f2": rng.normal(size=200),
                "Truth": ["SIG:Suu"] * 40 + ["BKG:qcd"] * 100 + ["BKG:ttbar"] * 60,
            }
        )
        df["target"] = df["Truth"].apply(lambda s: 1 if "SIG" in s else 0)

        prep = Preprocessor({"scaler": "minmax", "test_size": 0.2, "random_state": 13, "oversample_signal": True})
        X_tr, X_te, _y_tr, _y_te, _df_tr, df_te = prep.split_data(df, ["f1", "f2"])

        prep2 = Preprocessor(
            {"scaler": "minmax", "test_size": 0.2, "val_size": 0.0, "random_state": 13, "oversample_signal": True}
        )
        df_tr2, df_te2 = prep2.split_train_test_df(df)
        out = prep2.prepare_split_arrays(df_tr2, df_te2, ["f1", "f2"])
        self.assertTrue(df_te.index.equals(df_te2.index))
        self.assertTrue(np.allclose(X_tr, out["X_train"]))
        self.assertTrue(np.allclose(X_te, out["X_test"]))

        # Now with val_size > 0: ensure df_val and df_test are NOT oversampled and remain disjoint
        df_fit, df_val = prep2.split_train_val_df(df_tr2, val_size=0.25, random_state=13)
        assert df_val is not None
        self.assertEqual(len(set(df_fit.index) & set(df_val.index)), 0)
        self.assertEqual(len(set(df_val.index) & set(df_te2.index)), 0)

        out_val = prep2.prepare_split_arrays(df_fit, df_te2, ["f1", "f2"], df_val=df_val)
        self.assertEqual(len(out_val["df_val"]), len(df_val))
        self.assertEqual(len(out_val["df_test"]), len(df_te2))
        self.assertGreater(len(out_val["df_train"]), len(df_fit))

    def test_feature_cache_hit_miss_and_key_invariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            f1 = tmp_path / "a.root"
            f2 = tmp_path / "b.root"
            f1.write_bytes(b"root1")
            f2.write_bytes(b"root2")

            path_dict = {"BKG:qcd": f1, "SIG:Suu": f2}
            fe_cfg = {"feature_set": "new", "n_jets": 32, "min_jet_pt": 100.0}
            key1 = compute_feature_cache_key(path_dict, 0, None, fe_cfg, ["f1", "f2"])
            key2 = compute_feature_cache_key(
                {"SIG:Suu": f2, "BKG:qcd": f1}, 0, None, fe_cfg, ["f1", "f2"]
            )
            self.assertEqual(key1, key2)

            key3 = compute_feature_cache_key(
                path_dict, 0, None, {**fe_cfg, "min_jet_pt": 50.0}, ["f1", "f2"]
            )
            self.assertNotEqual(key1, key3)

            cache = FeatureCache(tmp_path / "cache")
            self.assertIsNone(cache.load("test_run", key1))

            df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "Truth": ["SIG:Suu", "BKG:qcd"], "target": [1, 0]})
            cache.save(df, "test_run", key1, metadata={"n_rows": 2})
            loaded = cache.load("test_run", key1)
            assert loaded is not None
            pd.testing.assert_frame_equal(df, loaded)


if __name__ == "__main__":
    unittest.main()
