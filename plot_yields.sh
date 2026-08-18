#!/usr/bin/env bash

set -e

uv run scripts/plot_yields.py results/ChiChi_ManyJets/MSuu_8000/ht-ht/bbt-bbt/gradient_boosting_counts_summary.csv yields_ht-ht_bbt-bbt.pdf --process 'Suu -> Chi Chi -> ht ht -> bbt bbt'

uv run scripts/plot_yields.py results/ChiChi_ManyJets/MSuu_8000/wb-ht/jjb-bbt/gradient_boosting_counts_summary.csv yields_wb-ht_jjb-bbt.pdf --process 'Suu -> Chi Chi -> W+b ht -> jjb bbt'

uv run scripts/plot_yields.py results/ChiChi_ManyJets/MSuu_8000/wb-zt/gradient_boosting_counts_summary.csv yields_wb-zt.pdf --process 'Suu -> Chi Chi -> W+b Zt'

