#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --cpus-per-task 120
#SBATCH --output slurm-%j.log

# Exit on error, print commands
set -ex

python3 diquark/analysis.py -c "diquark/config/ManyJets/MSuu_8000_wb-ht_jjb-bbt.yaml"
