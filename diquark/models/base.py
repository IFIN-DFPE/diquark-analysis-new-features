from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    model_extension: str = "pkl"

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.model: Any = None

    def model_filename(self, model_name: str) -> str:
        return f"{model_name}_model.{self.model_extension}"

    @abstractmethod
    def build(self, input_shape: int) -> None:
        "Build the model architecture."

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        sample_weight: np.ndarray | None = None,
        val_sample_weight: np.ndarray | None = None,
    ) -> Any:
        "Train the model."

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        "Make predictions using the trained model."

    @abstractmethod
    def save(self, path: str) -> None:
        "Save the trained model."

    @abstractmethod
    def load(self, path: str) -> None:
        "Load a trained model."
