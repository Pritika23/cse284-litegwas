# LiteGWAS – CSE284 Project

LiteGWAS is a lightweight Python implementation of covariate-adjusted linear regression for Genome-Wide Association Studies (GWAS). This project implements GWAS from scratch and applies it to both simulated data and real 1000 Genomes data.

---

# What Has Been Completed So Far

## 1. Implemented OLS-Based GWAS (from scratch)

For each SNP, we fit the model:

    y = intercept + PC1 + PC2 + PC3 + PC4 + PC5 + SNP + error

Features:
- Ordinary Least Squares (OLS)
- Covariate adjustment (top PCs)
- Minor Allele Frequency (MAF) filtering (MAF ≥ 0.01)
- Monomorphic SNP handling
- Singular matrix handling (pseudoinverse fallback)
- Outputs beta, standard error, t-statistic, p-value, degrees of freedom
- Manhattan and QQ plot generation

Core files:
- `litegwas/core.py` – regression logic
- `litegwas/run.py` – CLI entry point
- `litegwas/io.py` – input alignment
- `litegwas/plots.py` – visualization
- `litegwas/sim.py` – synthetic simulation

---

## 2. Built Synthetic Data Validation Pipeline

Created a full simulation workflow to verify correctness:

- Simulate genotype matrix
- Simulate phenotype using:
  - m causal SNPs
  - effect sizes ~ Normal(0, 0.1)
  - target heritability h²
- Run LiteGWAS
- Evaluate recovery of causal variants

Run synthetic demo:

    ./scripts/run_synth_demo.sh

Outputs:
- `out/results.tsv`
- Manhattan plot
- QQ plot
- Top-k overlap with true causal SNPs

This confirms that the regression implementation correctly recovers signal.

---

## 3. Applied LiteGWAS to Real 1000 Genomes Data

Dataset used:
- 1000 Genomes Phase 3
- Chromosome 22
- EUR ancestry subset
- SNP-only variants
- MAF ≥ 0.01
- 503 individuals
- 2,318 SNPs after filtering

Preprocessing performed with PLINK2:
- Subset to EUR samples
- Remove non-SNP variants
- Apply MAF filter
- Thin variants
- Assign stable variant IDs
- Compute top 5 principal components (PCs)
- Export dosage matrix

Converted PLINK2 outputs to LiteGWAS format using:

    scripts/from_plink2_to_litegwas.py

Generated:
- geno.npy
- covar.tsv
- snp.tsv
- pheno.tsv
- causal_truth.tsv

Ran LiteGWAS on real data:

    python -m litegwas.run \
      --geno <path>/geno.npy \
      --pheno <path>/pheno.tsv \
      --covar <path>/covar.tsv \
      --snp <path>/snp.tsv \
      --out out/real_results.tsv \
      --plot_prefix out/real

Results:
- No NaN p-values after filtering
- Stable regression
- Real Manhattan and QQ plots generated
- Strong signal detected (min p ≈ 3e-10)

---

## 4. GitHub Repository Setup

- SSH configured for GitHub
- Repository initialized and pushed
- `.gitignore` excludes:
  - venv/
  - realdata/
  - out/
- Code organized for reproducibility

---

# Current Status

Completed:
- Synthetic validation pipeline
- Real 1000G EUR GWAS run
- Stable covariate-adjusted regression
- Visualization
- GitHub setup

Next step:
- Benchmark LiteGWAS against PLINK2
- Perform comparative analysis
- Write final report

---

# Installation

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Dependencies:
- numpy
- pandas
- scipy
- matplotlib