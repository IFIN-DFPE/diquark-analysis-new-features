import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from diquark.models.gradient_boosting import GradientBoostingModel
from scripts.launch_hpo_slurm import build_slurm_array_script
from scripts.tune_xgboost import create_or_load_study, sample_trial_params


class TestHpoSweep(unittest.TestCase):
    def test_search_space_builds_valid_xgb_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.log"
            study = create_or_load_study(
                "space_test", journal, enqueue_anchors=False, prune=False
            )
            # Test both depthwise and lossguide branches explicitly
            for idx, policy in enumerate(("depthwise", "lossguide", "depthwise")):
                trial = study.ask()
                if idx == 0:
                    study.enqueue_trial(
                        {
                            "weight_alpha": 0.3,
                            "grow_policy": policy,
                            "max_depth": 5,
                            "min_child_weight": 10.0,
                            "gamma": 0.5,
                            "learning_rate": 0.05,
                            "subsample": 0.8,
                            "colsample_bytree": 0.7,
                            "colsample_bylevel": 0.8,
                            "reg_lambda": 3.0,
                            "reg_alpha": 0.01,
                            "max_bin": 256,
                        }
                    )
                    trial = study.ask()
                xgb_params, alpha = sample_trial_params(
                    trial, n_estimators=20, early_stopping_rounds=5, n_jobs=2
                )
                self.assertTrue(0.0 <= alpha <= 1.0)
                model = GradientBoostingModel(xgb_params)
                model.build(10)
                study.tell(trial, float(idx + 1))

    def test_journal_storage_anchor_dedup_and_max_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = Path(tmp) / "journal.log"
            s1 = create_or_load_study("dedup_test", journal, enqueue_anchors=True)
            self.assertEqual(len(s1.trials), 3)

            # Reloading must not duplicate anchors
            s2 = create_or_load_study("dedup_test", journal, enqueue_anchors=True)
            self.assertEqual(len(s2.trials), 3)

            def _obj(trial: optuna.Trial) -> float:
                sample_trial_params(trial, n_estimators=10, n_jobs=1)
                return float(trial.number)

            s2.optimize(
                _obj,
                n_trials=10,
                callbacks=[
                    MaxTrialsCallback(4, states=(TrialState.COMPLETE, TrialState.PRUNED))
                ],
            )
            finished = [
                t
                for t in s2.trials
                if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
            ]
            self.assertEqual(len(finished), 4)

    def test_end_to_end_mini_sweep_and_summarize(self) -> None:
        rng = np.random.default_rng(123)
        n_sig, n_qcd, n_tt = 160, 200, 140
        n_total = n_sig + n_qcd + n_tt
        truth = ["SIG:Suu"] * n_sig + ["BKG:qcd"] * n_qcd + ["BKG:ttbar"] * n_tt
        target = [1] * n_sig + [0] * (n_qcd + n_tt)

        df = pd.DataFrame(
            {
                "f1": np.concatenate(
                    [
                        rng.normal(1.5, 1.0, n_sig),
                        rng.normal(-1.0, 1.0, n_qcd),
                        rng.normal(-0.5, 1.0, n_tt),
                    ]
                ),
                "f2": rng.normal(0.0, 1.0, n_total),
                "f3": rng.normal(0.0, 1.0, n_total),
                "combined_invariant_mass": rng.uniform(5000, 9000, n_total),
                "Truth": truth,
                "target": target,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dummy_qcd = tmp_path / "qcd.root"
            dummy_tt = tmp_path / "ttbar.root"
            dummy_sig = tmp_path / "sig.root"
            for f in (dummy_qcd, dummy_tt, dummy_sig):
                f.write_bytes(b"")

            bkg_yaml = tmp_path / "background.yaml"
            bkg_yaml.write_text(
                yaml.safe_dump(
                    {
                        "version": 1.0,
                        "backgrounds": {
                            "qcd": {"file": str(dummy_qcd), "cross_section": 5.283},
                            "ttbar": {"file": str(dummy_tt), "cross_section": 0.001581},
                        },
                    }
                )
            )
            sig_yaml = tmp_path / "signal.yaml"
            sig_yaml.write_text(
                yaml.safe_dump(
                    {
                        "version": 1.0,
                        "signal": {
                            "file": str(dummy_sig),
                            "cross_section": 0.002935,
                            "chi_mass": 2000,
                            "suu_mass": 8000,
                        },
                    }
                )
            )
            run_yaml = tmp_path / "run.yaml"
            run_yaml.write_text(
                yaml.safe_dump(
                    {
                        "version": 1.0,
                        "data": {
                            "total_luminosity": 3000,
                            "backgrounds_file": "background.yaml",
                            "signal_file": "signal.yaml",
                        },
                        "feature_extraction": {"n_jets": 32},
                        "preprocessing": {
                            "scaler": "minmax",
                            "test_size": 0.2,
                            "random_state": 13,
                            "oversample_signal": False,
                        },
                    }
                )
            )

            parquet_path = tmp_path / "features.parquet"
            df.to_parquet(parquet_path, index=False)
            study_dir = tmp_path / "study"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/tune_xgboost.py",
                    "--config",
                    str(run_yaml),
                    "--study-name",
                    "mini_test",
                    "--study-dir",
                    str(study_dir),
                    "--dataset-parquet",
                    str(parquet_path),
                    "--n-folds",
                    "2",
                    "--n-trials",
                    "3",
                    "--max-total-trials",
                    "3",
                    "--n-estimators",
                    "15",
                    "--early-stopping-rounds",
                    "5",
                    "--n-jobs",
                    "2",
                    "--no-prune",
                ],
                check=True,
            )

            trial_jsons = list((study_dir / "trials").glob("trial_*.json"))
            self.assertEqual(len(trial_jsons), 3)

            env = dict(os.environ)
            env.pop("DISPLAY", None)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/summarize_hpo_study.py",
                    "--study-name",
                    "mini_test",
                    "--study-dir",
                    str(study_dir),
                    "--baseline-json",
                    str(tmp_path / "nonexistent.json"),
                ],
                check=True,
                env=env,
            )
            self.assertTrue((study_dir / "best_params.yaml").exists())
            pdf = study_dir / "hpo_summary.pdf"
            self.assertTrue(pdf.exists() and pdf.stat().st_size > 0)

    def test_slurm_dry_run_renders_expected_directives(self) -> None:
        script = build_slurm_array_script(
            config=Path("config/ChiChi_ManyJets/MSuu_8000/run_wb-ht_jjb-bbt.yaml"),
            study_name="msuu8000_wbht_xgb_hpo",
            study_dir=Path("results/ChiChi_ManyJets/MSuu_8000/experiments/hpo_wbht_study"),
            cache_dir=Path("cache/features"),
            objective="mean_z_eff_grid",
            mode="full-cv",
            n_folds=5,
            val_size=0.1,
            cv_random_state=42,
            max_total_trials=100,
            trials_per_worker=15,
            n_workers=10,
            max_concurrent=10,
            n_estimators=3000,
            early_stopping_rounds=40,
            device="cpu",
            cpus_per_task=64,
            mem="64G",
            time_limit="04:00:00",
            partition="cpu-wide",
            account="acc-c-fiz-26-001",
            exclude="hpc-node02,hpc-db01",
            prune=True,
        )
        self.assertIn("#SBATCH --array=1-10%10", script)
        self.assertIn("#SBATCH --cpus-per-task=64", script)
        self.assertIn("#SBATCH --partition=cpu-wide", script)
        self.assertIn("#SBATCH --exclude=hpc-node02,hpc-db01", script)
        self.assertIn(".venv/bin/python\" -u scripts/tune_xgboost.py", script)
        self.assertNotIn("activate", script)


if __name__ == "__main__":
    unittest.main()
