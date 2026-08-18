from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer


def extract_mean_and_std_from_string(value: str) -> tuple[float, float]:
    "Extract the mean and standard deviation from a string of the form 'mean ± std'."
    mean, std = value.split("±")
    return float(mean), float(std)


def main(
    csv_results_summary_file_path: Path,
    output_file_path: Annotated[
        Path, typer.Argument(help="Path where to save figure")
    ] = Path("yields.pdf"),
    process: Annotated[
        str | None,
        typer.Option(help="Description of the process we are plotting yields for"),
    ] = None,
) -> None:
    "Plot the evolution of the yields as a function of the discriminator threshold D."
    if not csv_results_summary_file_path.exists():
        raise FileNotFoundError(f"File {csv_results_summary_file_path} does not exist")

    results = pd.read_csv(csv_results_summary_file_path)

    signal_yields = results.iloc[-3]
    assert signal_yields["Process"] == "SIG:Suu"

    signal_yields_means, signal_yields_stds = zip(
        *(extract_mean_and_std_from_string(yld) for yld in signal_yields[:-2])
    )
    signal_yields_means = np.asarray(signal_yields_means)
    signal_yields_stds = np.asarray(signal_yields_stds)

    background_yields = results.iloc[-2]
    assert background_yields["Process"] == "BKG:sum"

    background_yields_means, background_yields_stds = zip(
        *(extract_mean_and_std_from_string(yld) for yld in background_yields[:-2])
    )
    background_yields_means = np.asarray(background_yields_means)
    background_yields_stds = np.asarray(background_yields_stds)

    signal_to_background_ratios = results.iloc[-1]
    assert signal_to_background_ratios["Process"] == "S/B"

    cuts = results.columns[:-2].astype(float)
    cuts = np.asarray(cuts)

    fig, ax = plt.subplots(figsize=(10, 6))

    fig.suptitle("Yields as a function of the discriminator threshold $D$")
    if process:
        ax.set_title(f"Process: {process}")

    start_index = 3

    ax.errorbar(
        cuts[start_index:],
        signal_yields_means[start_index:],
        yerr=signal_yields_stds[start_index:],
        fmt="o",
        label="Signal yield",
    )

    ax.errorbar(
        cuts[start_index:],
        background_yields_means[start_index:],
        yerr=background_yields_stds[start_index:],
        fmt="o",
        label="Background yield",
    )

    ax.axhline(y=1, color="gray", linestyle="--")

    ax.set_xlabel("Discriminator threshold $D$")
    ax.set_ylabel("Yield")

    ax.grid()
    ax.legend()

    fig.savefig(output_file_path)


if __name__ == "__main__":
    typer.run(main)
