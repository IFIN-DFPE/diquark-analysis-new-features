from typing import Any

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score

from diquark.models.base import BaseModel


class TabNetModel(BaseModel):
    def __init__(self, config: dict[str, Any]):
        super().__init__("TabNet", config)
        self.random_state = self.config.get("random_state", 42)

    def build(self, input_shape: int):
        self.model = TabNetClassifier(n_d=16, n_a=16, n_steps=5, seed=self.random_state)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        train_score = accuracy_score(y_train, self.model.predict(X_train))
        val_score = accuracy_score(y_val, self.model.predict(X_val))
        return {"train_score": train_score, "val_score": val_score}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model.load_model(path)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
