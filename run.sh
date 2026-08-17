#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120

source .venv/bin/activate

# Exit on error, print commands
set -ex

python3 diquark/analysis.py -c "diquark/config/New_Channels/wb-ht/jjb-bbt/MSuu_8000.yaml"
