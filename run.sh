#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256

source .venv/bin/activate

# Exit on error, print commands
set -ex

# python3 diquark/analysis.py -c "diquark/config/New_Channels/wb-zt/MSuu_8000.yaml"

# python3 diquark/analysis.py -c "diquark/config/MadGraph_Data/MSuu_8000.yaml"

python3 diquark/analysis.py -c "diquark/config/New_Channels/wb-ht/jjb-bbt/MSuu_8000.yaml"
