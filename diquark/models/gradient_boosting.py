from typing import Any

import numpy as np
import xgboost as xgb

from diquark.models.base import BaseModel

SUPPORTED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "max_leaves",
        "grow_policy",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "colsample_bynode",
        "reg_lambda",
        "reg_alpha",
        "max_delta_step",
        "max_bin",
        "scale_pos_weight",
        "tree_method",
        "device",
        "eval_metric",
        "early_stopping_rounds",
        "random_state",
        "n_jobs",
        "verbose",
    }
)


def _normalize_device(device: str | None) -> str:
    if device is None:
        return "cpu"
    normalized = str(device).strip().lower()
    if normalized == "gpu":
        return "cuda"
    if normalized in {"cpu", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        idx_str = normalized.split(":", 1)[1]
        if idx_str.isdigit():
            return normalized
    raise ValueError(
        f"Unsupported XGBoost device '{device}'. Expected 'cpu', 'cuda', 'cuda:<idx>', or 'gpu'."
    )


class GradientBoostingModel(BaseModel):
    """XGBoost binary classifier wrapper supporting HPO params, sample weights, and CPU/GPU."""

    model_extension: str = "json"

    def __init__(self, config: dict[str, Any]):
        super().__init__("GradientBoosting", config)
        unknown_keys = set(self.config.keys()) - SUPPORTED_CONFIG_KEYS
        if unknown_keys:
            raise ValueError(
                f"Unknown GradientBoosting config key(s): {sorted(unknown_keys)}"
            )

        self.n_estimators = int(self.config.get("n_estimators", 100))
        self.learning_rate = float(self.config.get("learning_rate", 0.1))
        self.max_depth = int(self.config.get("max_depth", 8))
        self.max_leaves = self.config.get("max_leaves", None)
        self.grow_policy = self.config.get("grow_policy", None)
        self.min_child_weight = float(self.config.get("min_child_weight", 1.0))
        self.gamma = float(self.config.get("gamma", 0.0))
        self.subsample = float(self.config.get("subsample", 1.0))
        self.colsample_bytree = float(self.config.get("colsample_bytree", 1.0))
        self.colsample_bylevel = self.config.get("colsample_bylevel", None)
        self.colsample_bynode = self.config.get("colsample_bynode", None)
        self.reg_lambda = float(self.config.get("reg_lambda", 1.0))
        self.reg_alpha = float(self.config.get("reg_alpha", 0.0))
        self.max_delta_step = float(self.config.get("max_delta_step", 0.0))
        self.max_bin = self.config.get("max_bin", None)
        self.scale_pos_weight = float(self.config.get("scale_pos_weight", 1.0))
        self.tree_method = str(self.config.get("tree_method", "hist"))
        self.device = _normalize_device(self.config.get("device", "cpu"))
        self.eval_metric = self.config.get("eval_metric", None)
        raw_es = self.config.get("early_stopping_rounds", 2)
        self.early_stopping_rounds: int | None = (
            int(raw_es) if raw_es is not None and int(raw_es) > 0 else None
        )
        self.random_state = int(self.config.get("random_state", 42))
        self.n_jobs = int(self.config.get("n_jobs", -1))
        self.verbose: bool | int = self.config.get("verbose", True)

    def build(self, input_shape: int) -> None:
        kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "max_delta_step": self.max_delta_step,
            "scale_pos_weight": self.scale_pos_weight,
            "tree_method": self.tree_method,
            "device": self.device,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        if self.max_leaves is not None:
            kwargs["max_leaves"] = int(self.max_leaves)
        if self.grow_policy is not None:
            kwargs["grow_policy"] = str(self.grow_policy)
        if self.colsample_bylevel is not None:
            kwargs["colsample_bylevel"] = float(self.colsample_bylevel)
        if self.colsample_bynode is not None:
            kwargs["colsample_bynode"] = float(self.colsample_bynode)
        if self.max_bin is not None:
            kwargs["max_bin"] = int(self.max_bin)
        if self.eval_metric is not None:
            kwargs["eval_metric"] = self.eval_metric
        if self.early_stopping_rounds is not None:
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBClassifier(**kwargs)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        sample_weight: np.ndarray | None = None,
        val_sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            self.build(X_train.shape[1])

        fit_kwargs: dict[str, Any] = {"verbose": self.verbose}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        has_eval_set = X_val is not None and y_val is not None
        if has_eval_set:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if val_sample_weight is not None:
                fit_kwargs["sample_weight_eval_set"] = [val_sample_weight]
        elif self.early_stopping_rounds is not None:
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self.model.evals_result() if has_eval_set else {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        self.model.save_model(path)

    def load(self, path: str) -> None:
        if self.model is None:
            self.build(0)
        assert self.model is not None, "Model must be built before loading."
        self.model.load_model(path)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    @property
    def best_iteration(self) -> int | None:
        if self.model is None:
            return None
        val = getattr(self.model, "best_iteration", None)
        return int(val) if val is not None else None

    @property
    def best_score(self) -> float | None:
        if self.model is None:
            return None
        val = getattr(self.model, "best_score", None)
        return float(val) if val is not None else None
