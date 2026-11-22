from pathlib import Path

import awkward as ak
import pandas as pd

from ..analysis import Analysis


def load_raw_data(run_config_file_path: str | Path) -> dict[str, ak.Array]:
    """Load the default branches for the events from the run's ROOT files
    and returns them as a dictionary of Awkward `Array`s.
    """
    analysis = Analysis(run_config_file_path)
    analysis.logger.handlers.clear()

    return analysis.load_data()


def load_data_frame(
    run_config_file_path: str | Path, preprocess: bool = True
) -> pd.DataFrame:
    """Load events from the ROOT files specified in the configuration files
    associated with the given run config file, extract their features,
    preprocess them and return a single `DataFrame`.
    """

    analysis = Analysis(run_config_file_path)
    analysis.logger.handlers.clear()

    # Load data
    data = analysis.load_data()

    # Extract features
    features = analysis.extract_features(data)

    if preprocess:
        _, _, _, _, df_train, df_test = analysis.preprocess_data(features)
        return pd.concat([df_train, df_test], ignore_index=True)
    else:
        return analysis.preprocessor.create_dataframe(features)
