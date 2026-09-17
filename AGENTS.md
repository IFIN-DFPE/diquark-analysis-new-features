# Ultraheavy diquark analysis &mdash; Instructions for AI agents

## Summary

This repository contains a machine learning framework for doing signal vs. background classification in a discovery study for a hypothetical ultraheavy diquark. We provide a command-line interface (`diquark/analysis.py`) for running the code. Runs are configured using a YAML-based format (see the `config` subdirectories for examples).

## Development environment

We use [uv](https://docs.astral.sh/uv/) to manage our Python environment. Run all commands through `uv run` to avoid missing dependency errors.

## Configuration file format

To perform an analysis, three files are required:
- A background data configuration file, defining one or more source ROOT files and their cross-sections;
  see `config/ChiChi_ManyJets/MSuu_8000/background.yaml` for an example.
- A signal data configuration file, defining a signal process;
  see `config/ChiChi_ManyJets/MSuu_8000/chi-chi_wb-ht_jjb-bbt.yaml` for an example.
- A run config file, which ties the previous two together and defines the ML-specific hyperparameters;
  see `config/ChiChi_ManyJets/MSuu_8000/chi-chi_wb-ht_jjb-bbt.yaml` for an example.

## Workflow

We recommend running the analysis through a job runner, for example [Slurm](https://slurm.schedmd.com/overview.html).
See the [`run.sh`](run.sh) for an example job script.

The command which needs to be run is:

```shell
uv run diquark/analysis.py -c "config/.../path/to/config/file"
```

It can be launched through `sbatch` with CPU cores, memory and time limits as follows:

```shell
sbatch --ntasks=1 --cpus-per-task=64 --time=01:00:00 <<< "#!/bin/bash

source .venv/bin/activate

set -ex

python3 diquark/analysis.py -c "config/.../something.yaml"
"
```
