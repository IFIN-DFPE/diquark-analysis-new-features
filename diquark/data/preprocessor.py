from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.scaler_type = config.get("scaler", "minmax")
        self.test_size = config.get("test_size", 0.2)
        self.random_state = config.get("random_state", 42)
        self.oversample_signal = config.get("oversample_signal", True)

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def create_dataframe(
        self, features: dict[str, dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """Creates a pandas DataFrame from the extracted features."""
        df_list = []
        for key, feature_dict in features.items():
            df = pd.DataFrame(feature_dict)
            df["Truth"] = key
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)
        df["target"] = df["Truth"].apply(lambda x: 1 if "SIG" in x else 0)
        return df

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scales the features using the specified scaler."""
        return self.scaler.fit_transform(X)

    def split_data(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame
    ]:
        """Splits the data into training and test sets."""
        X = df[feature_cols].values
        y = df["target"].values

        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X,
            y,
            df,
            test_size=self.test_size,
            stratify=df["Truth"],
            random_state=self.random_state,
        )

        X_train = cast(np.ndarray, X_train)
        X_test = cast(np.ndarray, X_test)
        y_train = cast(np.ndarray, y_train)
        y_test = cast(np.ndarray, y_test)
        df_train = cast(pd.DataFrame, df_train)
        df_test = cast(pd.DataFrame, df_test)

        if self.oversample_signal:
            df_train = self._oversample_signal(df_train)
            X_train = df_train[feature_cols].values
            y_train = df_train["target"].values
            y_train = cast(np.ndarray, y_train)

        X_train_scaled = self.scale_features(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = cast(np.ndarray, X_test_scaled)

        return X_train_scaled, X_test_scaled, y_train, y_test, df_train, df_test

    def _oversample_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oversamples the signal class to match the number of background instances."""
        df_sig = df[df["target"] == 1]
        df_bkg = df[df["target"] == 0]

        df_sig_oversampled = df_sig.sample(
            n=len(df_bkg), replace=True, random_state=self.random_state
        )
        df_oversampled = pd.concat([df_sig_oversampled, df_bkg])

        return df_oversampled.sample(frac=1, random_state=self.random_state)  # Shuffle

    def prepare_data(
        self, features: dict[str, dict[str, np.ndarray]]
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame
    ]:
        """Prepares the data for model training."""
        df = self.create_dataframe(features)
        feature_cols = [col for col in df.columns if col not in ["Truth", "target"]]
        return self.split_data(df, feature_cols)

    def prepare_fold_data(self, X_train, X_test, y_train, y_test, df_train, df_test):
        if self.oversample_signal:
            df_train = self._oversample_signal(df_train)
            X_train = df_train.drop(["target", "Truth"], axis=1).values
            y_train = df_train["target"].values
        else:
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)

        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        X_train_scaled = self.scale_features(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, df_train, df_test
