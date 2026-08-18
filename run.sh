#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --output slurm-%j.log

source .venv/bin/activate

# Exit on error, print commands
set -ex

# python3 diquark/analysis.py -c "config/ChiChi_ManyJets/MSuu_8000/chi-chi_ht-ht_bbt-bbt.yaml"
# python3 diquark/analysis.py -c "config/ChiChi_ManyJets/MSuu_8000/chi-chi_wb-ht_jjb-bbt.yaml"
python3 diquark/analysis.py -c "config/ChiChi_ManyJets/MSuu_8000/chi-chi_wb-zt.yaml"
