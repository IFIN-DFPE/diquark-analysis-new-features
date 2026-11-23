"""Launch parallel Optuna XGBoost tuning workers as a Slurm job array."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Annotated

import typer

from diquark.data.feature_cache import load_or_extract_dataset

app = typer.Typer(
    help="Generate and optionally submit a Slurm array job for Optuna XGBoost tuning.",
    add_completion=False,
)

REPO_ROOT = Path("/mnt/home/gabrielmajeri/Projects/Research/HEP/diquark-analysis")
VENV_PYTHON = REPO_ROOT / ".venv/bin/python"


def build_slurm_array_script(
    *,
    config: Path,
    study_name: str,
    study_dir: Path,
    cache_dir: Path,
    objective: str,
    mode: str,
    n_folds: int,
    val_size: float,
    cv_random_state: int,
    max_total_trials: int,
    trials_per_worker: int,
    n_workers: int,
    max_concurrent: int,
    n_estimators: int,
    early_stopping_rounds: int,
    device: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    partition: str,
    account: str,
    exclude: str,
    prune: bool,
) -> str:
    """Render the Slurm sbatch job-array script string."""
    log_dir = study_dir / "slurm_logs"
    prune_flag = "--prune" if prune else "--no-prune"
    exclude_line = f"#SBATCH --exclude={exclude}\n" if exclude.strip() else ""

    return dedent(f"""\
        #!/usr/bin/env bash
        #SBATCH --account={account}
        #SBATCH --partition={partition}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cpus_per_task}
        #SBATCH --mem={mem}
        #SBATCH --time={time_limit}
        #SBATCH --array=1-{n_workers}%{max_concurrent}
        #SBATCH --job-name={study_name}
        #SBATCH --output={log_dir}/worker_%A_%a.log
        #SBATCH --error={log_dir}/worker_%A_%a.err
        {exclude_line}
        set -euo pipefail
        cd "{REPO_ROOT}"

        WORKER_SEED=$((1000 + SLURM_ARRAY_TASK_ID))

        "{VENV_PYTHON}" -u scripts/tune_xgboost.py \\
          --config "{config}" \\
          --study-name "{study_name}" \\
          --study-dir "{study_dir}" \\
          --cache-dir "{cache_dir}" \\
          --objective "{objective}" \\
          --mode "{mode}" \\
          --n-folds {n_folds} \\
          --val-size {val_size} \\
          --cv-random-state {cv_random_state} \\
          --n-trials {trials_per_worker} \\
          --max-total-trials {max_total_trials} \\
          --n-estimators {n_estimators} \\
          --early-stopping-rounds {early_stopping_rounds} \\
          --device "{device}" \\
          --n-jobs {cpus_per_task} \\
          --worker-seed "${{WORKER_SEED}}" \\
          {prune_flag}
    """)


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(help="Run config YAML path."),
    ] = Path("config/ChiChi_ManyJets/MSuu_8000/run_wb-ht_jjb-bbt.yaml"),
    study_name: Annotated[
        str,
        typer.Option(help="Optuna study name."),
    ] = "msuu8000_wbht_xgb_hpo",
    study_dir: Annotated[
        Path,
        typer.Option(help="Output directory for journal, trial JSONs, and Slurm logs."),
    ] = Path("results/ChiChi_ManyJets/MSuu_8000/experiments/hpo_wbht_study"),
    cache_dir: Annotated[
        Path,
        typer.Option(help="Feature Parquet cache directory."),
    ] = Path("cache/features"),
    objective: Annotated[
        str,
        typer.Option(help="Optimization metric."),
    ] = "mean_z_eff_grid",
    mode: Annotated[
        str,
        typer.Option(help="CV mode: 'full-cv' or 'inner-cv'."),
    ] = "full-cv",
    n_folds: Annotated[
        int,
        typer.Option(help="Number of CV folds."),
    ] = 5,
    val_size: Annotated[
        float,
        typer.Option(help="Internal early-stopping validation fraction."),
    ] = 0.1,
    cv_random_state: Annotated[
        int,
        typer.Option(help="StratifiedKFold random seed."),
    ] = 42,
    max_total_trials: Annotated[
        int,
        typer.Option(help="Global cap on finished trials across all workers."),
    ] = 100,
    trials_per_worker: Annotated[
        int,
        typer.Option(help="Max trials per Slurm array worker."),
    ] = 15,
    n_workers: Annotated[
        int,
        typer.Option(help="Number of array tasks (parallel nodes)."),
    ] = 10,
    max_concurrent: Annotated[
        int,
        typer.Option(help="Max concurrent Slurm array tasks."),
    ] = 10,
    n_estimators: Annotated[
        int,
        typer.Option(help="Max XGBoost trees per fold."),
    ] = 3000,
    early_stopping_rounds: Annotated[
        int,
        typer.Option(help="Early stopping patience."),
    ] = 40,
    device: Annotated[
        str,
        typer.Option(help="XGBoost device ('cpu', 'cuda', etc.)."),
    ] = "cpu",
    cpus_per_task: Annotated[
        int,
        typer.Option(help="CPUs per Slurm array worker."),
    ] = 64,
    mem: Annotated[
        str,
        typer.Option(help="Memory limit per worker."),
    ] = "64G",
    time_limit: Annotated[
        str,
        typer.Option(help="Slurm wall-time limit per worker."),
    ] = "04:00:00",
    partition: Annotated[
        str,
        typer.Option(help="Slurm partition."),
    ] = "cpu-wide",
    account: Annotated[
        str,
        typer.Option(help="Slurm account."),
    ] = "acc-c-fiz-26-001",
    exclude: Annotated[
        str,
        typer.Option(help="Comma-separated Slurm nodes to exclude (empty string for none)."),
    ] = "hpc-node02,hpc-db01",
    prune: Annotated[
        bool,
        typer.Option("--prune/--no-prune", help="Enable MedianPruner."),
    ] = True,
    warm_cache: Annotated[
        bool,
        typer.Option(
            "--warm-cache/--no-warm-cache",
            help="Verify/build feature Parquet cache before launching workers.",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Print Slurm script without submitting."),
    ] = False,
) -> None:
    script = build_slurm_array_script(
        config=config,
        study_name=study_name,
        study_dir=study_dir,
        cache_dir=cache_dir,
        objective=objective,
        mode=mode,
        n_folds=n_folds,
        val_size=val_size,
        cv_random_state=cv_random_state,
        max_total_trials=max_total_trials,
        trials_per_worker=trials_per_worker,
        n_workers=n_workers,
        max_concurrent=max_concurrent,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        device=device,
        cpus_per_task=cpus_per_task,
        mem=mem,
        time_limit=time_limit,
        partition=partition,
        account=account,
        exclude=exclude,
        prune=prune,
    )

    if dry_run:
        print("=" * 80)
        print("DRY RUN — Generated Slurm Array Script (not submitted):")
        print("=" * 80)
        print(script)
        print("=" * 80)
        return

    if warm_cache:
        print(f"Checking/warming feature cache in '{cache_dir}' for '{config}'...")
        _, _, meta = load_or_extract_dataset(config, cache_dir=cache_dir)
        print(
            f"Feature cache ready: {meta['cache_path']} (cache_hit={meta['cache_hit']})"
        )

    (study_dir / "slurm_logs").mkdir(parents=True, exist_ok=True)
    (study_dir / "trials").mkdir(parents=True, exist_ok=True)
    script_path = study_dir / "hpo_array.sbatch"
    script_path.write_text(script, encoding="utf-8")

    result = subprocess.run(
        ["sbatch", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Saved Slurm script to {script_path}")
    print(f"Submitted: {result.stdout.strip()}")


if __name__ == "__main__":
    app()
