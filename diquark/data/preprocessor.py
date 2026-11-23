from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.scaler_type = config.get("scaler", "minmax")
        self.test_size = float(config.get("test_size", 0.2))
        self.val_size = float(config.get("val_size", 0.0))
        self.random_state = int(config.get("random_state", 42))
        self.oversample_signal = bool(config.get("oversample_signal", True))

        if not (0.0 <= self.val_size < 1.0):
            raise ValueError(f"val_size must be in [0.0, 1.0), got {self.val_size}")

        self.scaler = self._make_scaler()

    def _make_scaler(self) -> StandardScaler | MinMaxScaler:
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "minmax":
            return MinMaxScaler()
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

    def split_train_test_df(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform stratified train/test split on a feature DataFrame."""
        df_train, df_test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["Truth"],
            random_state=self.random_state,
        )
        return cast(pd.DataFrame, df_train), cast(pd.DataFrame, df_test)

    def split_train_val_df(
        self,
        df_train: pd.DataFrame,
        val_size: float | None = None,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Split a training DataFrame into stratified fit and validation subsets."""
        effective_val_size = self.val_size if val_size is None else float(val_size)
        if not (0.0 <= effective_val_size < 1.0):
            raise ValueError(
                f"val_size must be in [0.0, 1.0), got {effective_val_size}"
            )
        if effective_val_size <= 0.0:
            return df_train, None

        effective_seed = (
            self.random_state if random_state is None else int(random_state)
        )
        df_fit, df_val = train_test_split(
            df_train,
            test_size=effective_val_size,
            stratify=df_train["Truth"],
            random_state=effective_seed,
        )
        return cast(pd.DataFrame, df_fit), cast(pd.DataFrame, df_val)

    def prepare_split_arrays(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_cols: list[str],
        df_val: pd.DataFrame | None = None,
        oversample_signal: bool | None = None,
        fit_scaler: bool = True,
    ) -> dict[str, Any]:
        """Prepare scaled arrays for train, optional validation, and test splits.

        Signal oversampling and scaler fitting are restricted strictly to ``df_train``
        so neither ``df_val`` nor ``df_test`` is duplicated or leaked into scaling.
        """
        use_oversample = (
            self.oversample_signal
            if oversample_signal is None
            else bool(oversample_signal)
        )
        df_train_prep = self._oversample_signal(df_train) if use_oversample else df_train

        X_train = df_train_prep[feature_cols].to_numpy()
        y_train = df_train_prep["target"].to_numpy()
        X_test = df_test[feature_cols].to_numpy()
        y_test = df_test["target"].to_numpy()

        X_val = df_val[feature_cols].to_numpy() if df_val is not None else None
        y_val = df_val["target"].to_numpy() if df_val is not None else None

        if fit_scaler:
            scaler = self._make_scaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scaler = scaler
        else:
            scaler = self.scaler
            X_train_scaled = scaler.transform(X_train)

        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val) if X_val is not None else None

        return {
            "X_train": cast(np.ndarray, X_train_scaled),
            "y_train": cast(np.ndarray, y_train),
            "df_train": df_train_prep,
            "X_val": cast(np.ndarray, X_val_scaled) if X_val_scaled is not None else None,
            "y_val": cast(np.ndarray, y_val) if y_val is not None else None,
            "df_val": df_val,
            "X_test": cast(np.ndarray, X_test_scaled),
            "y_test": cast(np.ndarray, y_test),
            "df_test": df_test,
            "scaler": scaler,
        }

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
