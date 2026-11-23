from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from diquark.models.base import BaseModel


class RandomForestModel(BaseModel):
    model_extension: str = "joblib"

    def __init__(self, config: dict[str, Any]):
        super().__init__("RandomForest", config)
        self.n_estimators = self.config.get("n_estimators", 100)
        self.max_depth = self.config.get("max_depth", None)
        self.min_samples_split = self.config.get("min_samples_split", 2)
        self.min_samples_leaf = self.config.get("min_samples_leaf", 1)
        self.random_state = self.config.get("random_state", 42)
        self.n_jobs = self.config.get("n_jobs", -1)

    def build(self, input_shape: int):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        sample_weight: np.ndarray | None = None,
        val_sample_weight: np.ndarray | None = None,
    ):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        train_score = self.model.score(X_train, y_train)
        val_score = (
            self.model.score(X_val, y_val)
            if X_val is not None and y_val is not None
            else None
        )
        return {"train_score": train_score, "val_score": val_score}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
