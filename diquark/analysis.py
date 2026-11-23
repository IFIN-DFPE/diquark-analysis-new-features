import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.contrib.concurrent import thread_map

from diquark.config.config_manager import ConfigManager
from diquark.data.feature_cache import (
    FeatureCache,
    compute_feature_cache_key,
    resolve_dataset_spec,
)
from diquark.data.loader import DataLoader
from diquark.data.preprocessor import Preprocessor
from diquark.evaluation.metrics import (
    calculate_counts_for_score_cuts,
    calculate_metrics,
    calculate_signal_background_metrics,
)
from diquark.evaluation.tuning_metrics import compute_training_weights
from diquark.evaluation.visualizations import plot_results
from diquark.features import BaseFeatureExtractor
from diquark.models.gradient_boosting import GradientBoostingModel

# from diquark.models.neural_network import NeuralNetworkModel
from diquark.models.random_forest import RandomForestModel
from diquark.models.tabnet import TabNetModel
from diquark.utils.logger import setup_logger
from diquark.utils.results_manager import ResultsManager

_SUPPORTED_TRAINING_WEIGHT_KEYS = frozenset(
    {"mode", "alpha", "balance_classes", "signal_weight_ratio", "apply_to_eval_set"}
)


class Analysis:
    feature_extractor: BaseFeatureExtractor

    def __init__(self, config_path: str | Path):
        print("Reading config file...")
        print("Loading backgrounds config file...")
        print("Loading signal config file...")
        spec = resolve_dataset_spec(config_path)

        self.config: ConfigManager = spec["config"]
        self.logger = setup_logger(
            "experiment_logger",
            self.config.get("logging.file_path", "logs/experiment.log"),
            level=self.config.get("logging.level", "INFO"),
        )
        self.results_manager = ResultsManager(
            self.config.get("results.directory", "results")
        )

        self.cross_sections: dict[str, float] = spec["cross_sections"]
        self.path_dict: dict[str, Path] = spec["path_dict"]
        self.data_loader = DataLoader(
            self.path_dict,
            index_start=spec["index_start"],
            index_stop=spec["index_stop"],
        )

        self.use_cross_validation = self.config.get("cross_validation.enabled", False)
        self.n_folds = self.config.get("cross_validation.n_folds", 1)
        self.cv_random_state = int(self.config.get("cross_validation.random_state", 42))

        self.feature_extractor = spec["feature_extractor"]
        self.cache_dir: Path | None = spec["cache_dir"]
        self._cache_fe_cfg: dict[str, object] = spec["cache_fe_cfg"]
        self._cache_label: str = spec["label"]

        self.preprocessor = Preprocessor(self.config.get("preprocessing", {}))

        raw_tw_cfg = self.config.get("training_weights", {}) or {}
        if not isinstance(raw_tw_cfg, dict):
            raise TypeError("'training_weights' config section must be a mapping.")
        unknown_tw_keys = set(raw_tw_cfg.keys()) - _SUPPORTED_TRAINING_WEIGHT_KEYS
        if unknown_tw_keys:
            raise ValueError(
                f"Unknown training_weights config key(s): {sorted(unknown_tw_keys)}"
            )
        self.weight_mode = str(raw_tw_cfg.get("mode", "none"))
        self.weight_alpha = float(raw_tw_cfg.get("alpha", 1.0))
        self.weight_balance_classes = bool(raw_tw_cfg.get("balance_classes", True))
        self.weight_signal_ratio = float(raw_tw_cfg.get("signal_weight_ratio", 1.0))
        self.apply_weights_to_eval_set = bool(raw_tw_cfg.get("apply_to_eval_set", True))
        # Validate training_weights parameters early
        compute_training_weights(
            np.empty(0, dtype=str),
            self.cross_sections,
            mode=self.weight_mode,
            alpha=self.weight_alpha,
            balance_classes=self.weight_balance_classes,
            signal_weight_ratio=self.weight_signal_ratio,
        )

        pretrained_dir = (
            self.config.get("models.pretrained_results_dir", None)
            or self.config.get("evaluation.pretrained_results_dir", None)
            or self.config.get("results.pretrained_directory", None)
        )
        self.pretrained_results_dir: Path | None = (
            Path(pretrained_dir) if pretrained_dir else None
        )
        self.freeze_thresholds: bool = bool(
            self.config.get("evaluation.freeze_thresholds", False)
        )

        self.models = {
            # 'neural_network': NeuralNetworkModel(self.config.get('models.neural_network', {})) if self.config.get('models.neural_network', None) else None,
            "random_forest": RandomForestModel(
                self.config.get("models.random_forest", {})
            )
            if self.config.get("models.random_forest", None)
            else None,
            "gradient_boosting": GradientBoostingModel(
                self.config.get("models.gradient_boosting", {})
            )
            if self.config.get("models.gradient_boosting", None)
            else None,
            "tabnet": TabNetModel(self.config.get("models.tabnet", {}))
            if self.config.get("models.tabnet", None)
            else None,
        }

        if (
            self.weight_mode != "none"
            and self.weight_balance_classes
            and self.preprocessor.oversample_signal
        ):
            self.logger.warning(
                "training_weights.mode=%r with balance_classes=True is enabled alongside "
                "preprocessing.oversample_signal=True; consider setting oversample_signal: false "
                "to avoid redundant signal row duplication.",
                self.weight_mode,
            )

        gb_model = self.models.get("gradient_boosting")
        if (
            isinstance(gb_model, GradientBoostingModel)
            and self.weight_mode != "none"
            and self.weight_balance_classes
            and gb_model.scale_pos_weight != 1.0
        ):
            self.logger.warning(
                "Both training_weights (balance_classes=True) and gradient_boosting.scale_pos_weight=%s "
                "are set; this will double-scale the signal class.",
                gb_model.scale_pos_weight,
            )

    def _compute_training_weights(self, truth) -> np.ndarray | None:
        return compute_training_weights(
            truth,
            self.cross_sections,
            mode=self.weight_mode,
            alpha=self.weight_alpha,
            balance_classes=self.weight_balance_classes,
            signal_weight_ratio=self.weight_signal_ratio,
        )

    def get_feature_dataframe(self) -> pd.DataFrame:
        """Load cached feature DataFrame if available, else load ROOT data and extract."""
        cache: FeatureCache | None = None
        cache_key: str | None = None
        if self.cache_dir is not None:
            cache = FeatureCache(self.cache_dir)
            cache_key = compute_feature_cache_key(
                self.path_dict,
                self.data_loader.index_start,
                self.data_loader.index_stop,
                self._cache_fe_cfg,
                self.feature_extractor.feature_names,
            )
            df_cached = cache.load(self._cache_label, cache_key)
            if df_cached is not None:
                self.logger.info(
                    "Loaded extracted features from cache: %s",
                    cache.cache_path(self._cache_label, cache_key),
                )
                return df_cached

        data = self.load_data()
        features = self.extract_features(data)
        df = self.preprocessor.create_dataframe(features)

        if cache is not None and cache_key is not None:
            saved_path = cache.save(
                df,
                self._cache_label,
                cache_key,
                metadata={
                    "cache_key": cache_key,
                    "label": self._cache_label,
                    "n_rows": len(df),
                    "n_features": len(self.feature_extractor.feature_names),
                    "truth_counts": {
                        str(k): int(v) for k, v in df["Truth"].value_counts().items()
                    },
                    "feature_extraction": self._cache_fe_cfg,
                },
            )
            self.logger.info("Saved extracted features to cache: %s", saved_path)

        return df

    def run(self):
        self.logger.info("Starting analysis...")

        df = self.get_feature_dataframe()

        if self.use_cross_validation:
            self.run_cross_validation(df)
        else:
            self.run_single_fold(df)

        self.logger.info("Analysis completed.")

    def _to_dataframe(
        self, features_or_df: pd.DataFrame | dict[str, dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        if isinstance(features_or_df, pd.DataFrame):
            return features_or_df
        return self.preprocessor.create_dataframe(features_or_df)

    def run_single_fold(
        self, features: pd.DataFrame | dict[str, dict[str, np.ndarray]]
    ):
        df = self._to_dataframe(features)
        feature_cols = [col for col in df.columns if col not in ["Truth", "target"]]

        if self.pretrained_results_dir is not None:
            _, df_test = self.preprocessor.split_train_test_df(df)
            y_test = df_test["target"].to_numpy()
            scaler_dir = (
                self.pretrained_results_dir
                if (self.pretrained_results_dir / "scaler.pkl").exists()
                else self.pretrained_results_dir / "fold_1"
            )
            scaler = self.results_manager.load_pickle(
                "scaler.pkl", custom_dir=scaler_dir
            )
            X_test = scaler.transform(df_test[feature_cols].to_numpy())
            results = self.evaluate_pretrained_models(
                X_test, y_test, df_test, pretrained_dir=scaler_dir
            )
            self.visualize_results(results, df_test)
            self.save_results(results, df_test, save_models=False)
            return

        self.logger.info("Preprocessing data...")
        df_train_full, df_test = self.preprocessor.split_train_test_df(df)
        df_fit, df_val = self.preprocessor.split_train_val_df(df_train_full)
        prepared = self.preprocessor.prepare_split_arrays(
            df_fit, df_test, feature_cols, df_val=df_val
        )

        sample_weight = self._compute_training_weights(prepared["df_train"]["Truth"])
        if df_val is not None:
            X_val, y_val = prepared["X_val"], prepared["y_val"]
            val_sample_weight = (
                self._compute_training_weights(prepared["df_val"]["Truth"])
                if self.apply_weights_to_eval_set
                else None
            )
        else:
            X_val, y_val = prepared["X_test"], prepared["y_test"]
            val_sample_weight = (
                self._compute_training_weights(prepared["df_test"]["Truth"])
                if (self.apply_weights_to_eval_set and self.weight_mode != "none")
                else None
            )

        # Train and evaluate models
        results = self.train_and_evaluate_models(
            prepared["X_train"],
            prepared["X_test"],
            prepared["y_train"],
            prepared["y_test"],
            prepared["df_test"],
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
        )

        # Visualize results
        self.visualize_results(results, prepared["df_test"])

        # Save results
        self.save_results(results, prepared["df_test"], save_models=True)

    def run_cross_validation(
        self, features: pd.DataFrame | dict[str, dict[str, np.ndarray]]
    ):
        df = self._to_dataframe(features)
        feature_cols = [col for col in df.columns if col not in ["Truth", "target"]]
        X = df[feature_cols]

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.cv_random_state
        )

        all_fold_results = []
        oof_predictions = {
            model_name: np.empty(len(df), dtype=np.float64)
            for model_name, model in self.models.items()
            if model is not None
        }

        for fold, (train_index, test_index) in enumerate(skf.split(X, df["Truth"]), 1):
            self.logger.info(f"Processing fold {fold}")

            fold_dir = self.results_manager.create_subdir(f"fold_{fold}")

            df_train = df.iloc[train_index]
            df_test = df.iloc[test_index]

            if self.pretrained_results_dir is not None:
                pretrained_fold_dir = self.pretrained_results_dir / f"fold_{fold}"
                if not pretrained_fold_dir.exists():
                    raise FileNotFoundError(
                        f"Pretrained fold directory not found: '{pretrained_fold_dir}'. "
                        f"Please run the nominal configuration first to generate pretrained models."
                    )
                scaler = self.results_manager.load_pickle(
                    "scaler.pkl", custom_dir=pretrained_fold_dir
                )
                X_test_scaled = scaler.transform(df_test[feature_cols].to_numpy())
                y_test_arr = df_test["target"].to_numpy()
                results = self.evaluate_pretrained_models(
                    X_test_scaled,
                    y_test_arr,
                    df_test,
                    pretrained_dir=pretrained_fold_dir,
                )
                save_models = False
            else:
                df_fit, df_val = self.preprocessor.split_train_val_df(
                    df_train, random_state=self.preprocessor.random_state + fold
                )
                prepared = self.preprocessor.prepare_split_arrays(
                    df_fit, df_test, feature_cols, df_val=df_val
                )
                sample_weight = self._compute_training_weights(
                    prepared["df_train"]["Truth"]
                )
                if df_val is not None:
                    X_val, y_val = prepared["X_val"], prepared["y_val"]
                    val_sample_weight = (
                        self._compute_training_weights(prepared["df_val"]["Truth"])
                        if self.apply_weights_to_eval_set
                        else None
                    )
                else:
                    X_val, y_val = prepared["X_test"], prepared["y_test"]
                    val_sample_weight = (
                        self._compute_training_weights(prepared["df_test"]["Truth"])
                        if (
                            self.apply_weights_to_eval_set
                            and self.weight_mode != "none"
                        )
                        else None
                    )

                results = self.train_and_evaluate_models(
                    prepared["X_train"],
                    prepared["X_test"],
                    prepared["y_train"],
                    prepared["y_test"],
                    prepared["df_test"],
                    X_val=X_val,
                    y_val=y_val,
                    sample_weight=sample_weight,
                    val_sample_weight=val_sample_weight,
                )
                save_models = True

            for model_name, model_results in results.items():
                oof_predictions[model_name][test_index] = model_results["predictions"]

            self.visualize_results(results, df_test, fold_dir)
            self.save_results(results, df_test, fold_dir, save_models=save_models)

            all_fold_results.append(results)

        self.summarize_cross_validation_results(all_fold_results, df, oof_predictions)

    def load_data(self):
        self.logger.info("Loading data...")
        return self.data_loader.load_data()

    def extract_features(self, data):
        self.logger.info("Extracting features...")

        dataset_parallel_workers = self.config.get(
            "feature_extraction.dataset_parallel_workers",
            self.config.get("feature_extraction.max_workers", 4),
        )

        features = thread_map(
            self.feature_extractor.compute_all,
            data.values(),
            max_workers=dataset_parallel_workers,
            desc="Extracting features",
        )

        self.logger.info("Working with %d feature columns", len(features[0].keys()))

        assert len(features[0].keys()) == len(self.feature_extractor.feature_names), (
            f"Number of extracted features ({len(features[0].keys())}) doesn't match number of feature names defined on feature extractor object ({len(self.feature_extractor.feature_names)})"
        )

        return dict(zip(data.keys(), features))

    def preprocess_data(self, features):
        self.logger.info("Preprocessing data...")
        return self.preprocessor.prepare_data(features)

    def evaluate_predictions(
        self,
        y_true,
        y_pred,
        df_eval,
        fixed_thresholds: dict[str, float] | None = None,
    ):
        total_luminosity = self.config.get("data.total_luminosity", 3000)
        truth_series = df_eval["Truth"]
        truth_counts = truth_series.value_counts().to_dict()
        sample_weights = np.array(
            [
                total_luminosity * self.cross_sections[label] / truth_counts[label]
                for label in truth_series
            ],
            dtype=np.float64,
        )

        thresholds = self.config.get(
            "evaluation.thresholds",
            [0.2, 0.5, 0.8, 0.90, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99],
        )
        use_real_event_percentiles = self.config.get(
            "evaluation.use_real_event_percentiles", False
        )

        metrics = calculate_metrics(
            y_true, y_pred, sample_weights, thresholds, use_real_event_percentiles
        )
        sig_bkg_metrics = calculate_signal_background_metrics(
            y_true,
            y_pred,
            thresholds,
            sample_weights,
            total_luminosity,
            self.cross_sections,
            use_real_event_percentiles,
            fixed_thresholds=fixed_thresholds,
        )
        score_thresholds = {
            str(cut): float(thr)
            for cut, thr in zip(thresholds, sig_bkg_metrics["thresholds"])
        }

        cuts = thresholds
        df_counts, df_raw_counts, df_uncertainties = calculate_counts_for_score_cuts(
            y_true,
            y_pred,
            df_eval["combined_invariant_mass"],
            truth_series,
            self.cross_sections,
            total_luminosity,
            cuts,
            use_real_event_percentiles,
            fixed_thresholds=fixed_thresholds,
        )
        df_counts_summary = self.format_counts_summary(df_counts, df_uncertainties)

        return {
            "metrics": metrics,
            "sig_bkg_metrics": sig_bkg_metrics,
            "score_thresholds": score_thresholds,
            "predictions": y_pred,
            "sample_weights": sample_weights,
            "counts": df_counts,
            "raw_counts": df_raw_counts,
            "counts_uncertainties": df_uncertainties,
            "counts_summary": df_counts_summary,
        }

    def format_counts_summary(
        self, df_counts: pd.DataFrame, df_uncertainties: pd.DataFrame
    ) -> pd.DataFrame:
        numeric_cols = [col for col in df_counts.columns if col != "Process"]
        result_df = pd.DataFrame(
            index=df_counts.index, columns=[*numeric_cols, "Process"]
        )
        for idx in df_counts.index:
            for col in numeric_cols:
                mean_value = df_counts.loc[idx, col]
                std_value = df_uncertainties.loc[idx, col]
                result_df.loc[idx, col] = self.format_value(mean_value, std_value)
            result_df.loc[idx, "Process"] = df_counts.loc[idx, "Process"]
        return result_df

    def _load_fixed_thresholds(
        self, model_name: str, pretrained_dir: Path
    ) -> dict[str, float] | None:
        if not self.freeze_thresholds:
            return None
        thresholds_file = pretrained_dir / f"{model_name}_score_thresholds.json"
        if thresholds_file.exists():
            return self.results_manager.load_json(
                f"{model_name}_score_thresholds.json", custom_dir=pretrained_dir
            )
        return None

    def train_and_evaluate_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        df_test,
        X_val=None,
        y_val=None,
        sample_weight=None,
        val_sample_weight=None,
    ):
        eval_X = X_test if X_val is None else X_val
        eval_y = y_test if y_val is None else y_val
        results = {}
        for model_name, model in self.models.items():
            if model is None:
                continue
            self.logger.info(f"Training and evaluating {model_name}...")
            model.build(X_train.shape[1])
            model.train(
                X_train,
                y_train,
                eval_X,
                eval_y,
                sample_weight=sample_weight,
                val_sample_weight=val_sample_weight,
            )
            y_pred = model.predict(X_test)

            results[model_name] = self.evaluate_predictions(y_test, y_pred, df_test)
            if hasattr(model, "feature_importances"):
                results[model_name]["feature_importances"] = model.feature_importances()
        return results

    def evaluate_pretrained_models(self, X_test, y_test, df_test, pretrained_dir: Path):
        results = {}
        for model_name, model in self.models.items():
            if model is None:
                continue
            model_path = pretrained_dir / model.model_filename(model_name)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Pretrained model file not found: '{model_path}'. "
                    f"Please run the nominal configuration first to generate pretrained models."
                )
            self.logger.info(f"Loading pretrained {model_name} from {model_path}...")
            model.load(str(model_path))
            y_pred = model.predict(X_test)

            fixed_thresholds = self._load_fixed_thresholds(model_name, pretrained_dir)
            results[model_name] = self.evaluate_predictions(
                y_test, y_pred, df_test, fixed_thresholds=fixed_thresholds
            )
            if hasattr(model, "feature_importances"):
                results[model_name]["feature_importances"] = model.feature_importances()
        return results

    def visualize_results(self, results, df_test, custom_dir=None):
        plot_types = self.config.get("visualization.plots", [])
        if not plot_types:
            return
        self.logger.info("Visualizing results...")
        results_dir = Path(
            custom_dir if custom_dir else self.results_manager.results_dir
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        use_real_percentiles = bool(
            self.config.get("evaluation.use_real_event_percentiles", True)
        )
        for model_name, model_results in results.items():
            plots = plot_results(
                model_results,
                df_test,
                plot_types,
                use_real_percentiles,
            )
            for plot_name, fig in plots.items():
                fig.savefig(
                    results_dir / f"{model_name}_{plot_name}.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)

    def save_results(
        self,
        results,
        df_test,
        custom_dir=None,
        save_test_set: bool = True,
        save_models: bool = False,
    ):
        results_dir = Path(
            custom_dir if custom_dir else self.results_manager.results_dir
        )
        self.logger.info(f"Saving results to {results_dir}...")
        if save_models and self.config.get("results.save_models", True):
            self.results_manager.save_pickle(
                self.preprocessor.scaler, "scaler.pkl", custom_dir=results_dir
            )

        for model_name, model_results in results.items():
            if (
                save_models
                and self.config.get("results.save_models", True)
                and self.models.get(model_name) is not None
            ):
                model_obj = self.models[model_name]
                assert model_obj is not None
                model_obj.save(str(results_dir / model_obj.model_filename(model_name)))

            if "score_thresholds" in model_results:
                self.results_manager.save_json(
                    model_results["score_thresholds"],
                    f"{model_name}_score_thresholds.json",
                    custom_dir=results_dir,
                )

            self.results_manager.save_json(
                model_results["metrics"],
                f"{model_name}_metrics.json",
                custom_dir=results_dir,
            )
            self.results_manager.save_json(
                model_results["sig_bkg_metrics"],
                f"{model_name}_sig_bkg_metrics.json",
                custom_dir=results_dir,
            )
            self.results_manager.save_dataframe(
                model_results["counts"],
                f"{model_name}_counts.csv",
                custom_dir=results_dir,
            )
            self.results_manager.save_dataframe(
                model_results["raw_counts"],
                f"{model_name}_raw_counts.csv",
                custom_dir=results_dir,
            )
            if "counts_summary" in model_results:
                self.results_manager.save_dataframe(
                    model_results["counts_summary"],
                    f"{model_name}_counts_summary.csv",
                    custom_dir=results_dir,
                )

            if self.config.get("results.save_predictions", False):
                self.results_manager.save_numpy(
                    model_results["predictions"],
                    f"{model_name}_predictions.npy",
                    custom_dir=results_dir,
                )

            if (
                self.config.get("results.save_feature_importances", False)
                and "feature_importances" in model_results
            ):
                feature_importances = model_results["feature_importances"]
                if len(feature_importances) == len(
                    self.feature_extractor.feature_names
                ):
                    feature_importance_dict = dict(
                        zip(
                            self.feature_extractor.feature_names,
                            feature_importances.tolist(),
                        )
                    )
                else:
                    self.logger.warning(
                        f"Feature importances length ({len(feature_importances)}) does not match feature names length ({len(self.feature_extractor.feature_names)}). Saving without feature names."
                    )
                    feature_importance_dict = dict(
                        enumerate(feature_importances.tolist())
                    )
                    feature_importance_dict = {
                        f"feature_{i}": importance
                        for i, importance in feature_importance_dict.items()
                    }

                self.results_manager.save_json(
                    feature_importance_dict,
                    f"{model_name}_feature_importances.json",
                    custom_dir=results_dir,
                )

        if save_test_set:
            # Save test set
            self.results_manager.save_dataframe(
                df_test,
                "test_set.parquet",
                format="parquet",
                custom_dir=results_dir,
            )

    def summarize_cross_validation_results(
        self, all_fold_results, df: pd.DataFrame, oof_predictions: dict[str, np.ndarray]
    ):
        self.logger.info(
            "Evaluating pooled out-of-fold predictions across all folds..."
        )
        y_true = df["target"].to_numpy()
        pooled_results = {}

        for model_name in self.models:
            if self.models[model_name] is None:
                continue

            y_pred_oof = oof_predictions[model_name]
            fixed_thresholds = (
                self._load_fixed_thresholds(model_name, self.pretrained_results_dir)
                if self.pretrained_results_dir is not None
                else None
            )
            model_pooled_results = self.evaluate_predictions(
                y_true, y_pred_oof, df, fixed_thresholds=fixed_thresholds
            )

            fold_importances = [
                fold_result[model_name]["feature_importances"]
                for fold_result in all_fold_results
                if "feature_importances" in fold_result[model_name]
            ]
            if fold_importances:
                model_pooled_results["feature_importances"] = np.mean(
                    fold_importances, axis=0
                )

            pooled_results[model_name] = model_pooled_results

        self.visualize_results(pooled_results, df)
        self.save_results(pooled_results, df, save_test_set=False, save_models=False)

    def format_value(self, mean, std):
        if mean == 0 and std == 0:
            return "0.000e+00 ± 0.000e+00"

        if np.isinf(mean) or np.isinf(std):
            return f"{mean} ± {std}"

        if abs(mean) > 1e-10:
            mean_magnitude = int(np.floor(np.log10(abs(mean))))
        else:
            mean_magnitude = 0

        if abs(std) > 1e-10:
            std_magnitude = int(np.floor(np.log10(abs(std))))
        else:
            std_magnitude = mean_magnitude

        precision = max(0, mean_magnitude - std_magnitude + 3)

        mean_formatted = f"{mean:.{precision}e}"
        std_formatted = f"{std:.{precision}e}"

        return f"{mean_formatted} ± {std_formatted}"


def main() -> None:
    random.seed(17)
    np.random.seed(17)

    parser = argparse.ArgumentParser(
        description="Run diquark analysis with optional custom config file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="diquark/config/default_settings.yaml",
        help="Path to the configuration file (default: diquark/config/default_settings.yaml)",
    )

    args = parser.parse_args()

    analysis = Analysis(args.config)
    analysis.run()


if __name__ == "__main__":
    main()
