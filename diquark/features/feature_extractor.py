from abc import ABC, abstractmethod

import awkward as ak
import numpy as np


class BaseFeatureExtractor(ABC):
    feature_names: list[str]

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or []

    @abstractmethod
    def compute_all(self, data: ak.Array) -> dict[str, np.ndarray]: ...
