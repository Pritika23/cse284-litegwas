#!/usr/bin/env bash
set -euo pipefail

python -m litegwas.sim --outdir data/test --N 400 --M 5000 --K 5 --m_causal 25 --h2 0.3 --seed 42

python -m litegwas.run \
  --geno data/test/geno.npy \
  --pheno data/test/pheno.tsv \
  --covar data/test/covar.tsv \
  --snp data/test/snp.tsv \
  --out out/results.tsv \
  --plot_prefix out/test

python scripts/eval_recovery.py --results out/results.tsv --truth data/test/causal_truth.tsv --topk 50