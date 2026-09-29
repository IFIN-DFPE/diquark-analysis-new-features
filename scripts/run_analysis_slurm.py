#!/usr/bin/env python3
"""
Script to queue up and launch Slurm jobs for running the diquark ML analysis.

This script recursively finds YAML config files in a given directory (or subdirectories)
and launches each one as an independent Slurm job.

Usage:
    python run_analysis_slurm.py <config_directory> [--recursive] [--skip-existing] [--dry-run]

Examples:
    # Run all configs in current directory and subdirectories recursively
    python run_analysis_slurm.py config/ChiChi_ManyJets/systematics --recursive

    # Run only files in specified directory (not recursive)
    python run_analysis_slurm.py config/ChiChi_ManyJets/systematics/both_down

    # Skip jobs where output log already exists
    python run_analysis_slurm.py config/ChiChi_ManyJets/systematics --skip-existing

    # Preview what would be launched without actually submitting
    python run_analysis_slurm.py config/ChiChi_ManyJets/systematics --dry-run
"""

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from textwrap import dedent

import typer

app = typer.Typer(
    help="Queue up and launch Slurm jobs for diquark ML analysis",
    add_completion=False,
)


def _launch_slurm_job(config_file_path: Path, skip_existing: bool) -> None:
    """Launch a single Slurm job for the given config file."""
    config_file_path = config_file_path.resolve()
    config_name = config_file_path.stem  # filename without extension
    output_file_path = config_file_path.with_suffix(".log")

    if skip_existing and output_file_path.exists():
        print(
            f"Output log already exists for config {config_file_path}, skipping...",
            file=sys.stderr,
        )
        return

    print(f"Launching Slurm job for config: {config_file_path}")
    print(f"Redirecting output to: {output_file_path}")

    # Build the slurm script
    slurm_script = dedent(f"""
    #!/usr/bin/env bash

    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=120
    #SBATCH --time=01:00:00
    #SBATCH --mem=128G
    #SBATCH --job-name diquark_analysis_{config_name}
    #SBATCH --output={output_file_path}
    #SBATCH --error={output_file_path}.err

    # Activate virtual environment
    source /data/gmajeri/diquark-analysis/.venv/bin/activate

    set -ex

    # Run the analysis
    python diquark/analysis.py -c "{config_file_path}"
    """).strip()

    try:
        result = subprocess.run(
            "sbatch",
            input=slurm_script,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  ✓ Job submitted successfully: {result.stdout.strip()}")
    except subprocess.CalledProcessError as err:
        error_msg = err.stderr if err.stderr else err.stdout
        print(
            f"  ✗ Failed to submit Slurm job for config {config_file_path}: {error_msg}",
            file=sys.stderr,
        )
    except Exception as err:
        print(
            f"  ✗ An error occurred while launching Slurm job for config {config_file_path}: {err}",
            file=sys.stderr,
        )


def _launch_jobs(config_files: Iterable[Path], skip_existing: bool) -> None:
    """Launch all jobs through Slurm."""
    for config_file_path in config_files:
        _launch_slurm_job(config_file_path, skip_existing)


def _dry_run(config_files: Iterable[Path]) -> None:
    """Preview what would be launched without actually submitting."""
    print(f"{'=' * 80}")
    print("DRY RUN - Would launch the following Slurm jobs:")
    print(f"{'=' * 80}\n")

    for config_file_path in config_files:
        config_name = config_file_path.stem
        output_file_path = config_file_path.with_suffix(".log")

        print(f"Config: {config_file_path}")
        print(f'  Command: python diquark/analysis.py -c "{config_file_path}"')
        print(f"  Output: {output_file_path}")
        print("  Slurm resources:")
        print("    • CPUs: 120")
        print("    • Memory: 320GB")
        print("    • Time limit: 01:00:00")
        print(f"    • Job name: diquark_analysis_{config_name}")
        print()


def find_config_files(directory_path: Path, recursive: bool = False) -> list[Path]:
    """Find all YAML config files in the given directory."""
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")

    if not directory_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    yaml_extensions = (".yaml", ".yml")
    config_files = []

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for ext in yaml_extensions:
        files = list(directory_path.glob(f"{pattern}{ext}"))
        # Filter to only include run_* or signal_* or background.yaml files (not config dir definitions)
        config_files.extend(files)

    # Sort for consistent ordering
    config_files = sorted(config_files)

    return config_files


@app.command()
def main(
    directory_path: str = typer.Argument(
        ...,
        help="Directory containing YAML config files",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Search recursively in subdirectories",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        "-s",
        help="Skip jobs where output log already exists",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Preview what would be launched without submitting",
    ),
) -> None:
    """
    Queue up and launch Slurm jobs for diquark ML analysis.

    Args:
        directory_path: Directory containing YAML config files
        recursive: Search recursively in subdirectories
        skip_existing: Skip jobs where output log already exists
        dry_run: Preview what would be launched without submitting
    """
    directory = Path(directory_path).resolve()

    if not directory.exists():
        typer.echo(
            f"Error: Provided directory doesn't exist: {directory_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    if not directory.is_dir():
        typer.echo(
            f"Error: Provided path is not a directory: {directory_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    print(f"Searching for YAML config files in: {directory}")

    if recursive:
        print("  (recursive search enabled)")

    config_files = find_config_files(directory, recursive=recursive)

    if not config_files:
        typer.echo(
            f"No YAML config files found in {directory_path}",
            err=True,
        )
        raise typer.Exit(code=1)

    print(f"Found a total of {len(config_files)} config files to process\n")

    # Filter out background.yaml and signal_*.yaml (only run_*.yaml should be executed)
    run_configs = [f for f in config_files if f.stem.startswith("run_")]

    if not run_configs:
        typer.echo(
            f"No run_* config files found. Found {len(config_files)} YAML files but none match the pattern 'run_*'.",
            err=True,
        )
        typer.echo(
            "Tip: Create configs with names like 'run_ht-ht_bbt-bbt.yaml', 'run_wb-ht_jjb-bbt.yaml', etc.",
            err=True,
        )
        raise typer.Exit(code=1)

    print(f"Found {len(run_configs)} run_* config files to launch:")
    for cfg in run_configs:
        print(f"  • {cfg}")
    print()

    if dry_run:
        _dry_run(run_configs)
    else:
        print("Launching Slurm jobs...")
        print("-" * 80)
        _launch_jobs(run_configs, skip_existing)
        print("-" * 80)
        print(f"\n✓ Submitted {len(run_configs)} Slurm jobs successfully!")


if __name__ == "__main__":
    app()
