# CSE284 Project

This project implements a lightweight Python implementation of Genome-Wide Association Studies (GWAS) from scratch and applies it to both simulated data and real 1000 Genomes data.

---

## 1. OLS-Based GWAS (from scratch)

For each SNP, we fit the model:

    y = intercept + PC1 + PC2 + PC3 + PC4 + PC5 + SNP + error

This system performs Ordinary Least Squares (OLS) regression with Covariate adjustment (using top PCs). We perform Minor Allele Frequency (MAF) filtering (MAF >= 0.01) and do monomorphic SNP handling. 

We also include a pseuroinverse fallback in our code incase we encounter a singular matrix, with the same genotype across all samples for a SNP. The algorithm outputs beta, standard error, t-statistic, p-value, degrees of freedom. And generates the Manhattan and QQ plots.

Important files:
- `litegwas/core.py` – regression logic
- `litegwas/run.py` – main run code
- `litegwas/io.py` – input alignment
- `litegwas/plots.py` – visualization

---

## 2. Synthetic Data Validation

We have also created an example simulation workflow to verify correctness, where we have simulated the genotype matrix and phenotypes and test if we can recover causal variants. 

Run synthetic demo:

    ./scripts/run_synth_demo.sh

The outputs are:
- `out/results.tsv`
- Manhattan plot
- QQ plot
- Top-k overlap with true causal SNPs

---

## 3. GWAS to Real 1000 Genomes Data

The dataset used is the 1000 Genomes Phase 3 dataset, where we focus on Chromosome 22, and use only the EUR ancestry subset. The initial preprocessing was performed with PLINK to get the EUR samples subset, remove non-SNP variants, apply MAF filter, thin variants, assign stable variant IDs and compute top 5 principal components (PCs). After filtering for MAF, we obtain 503 individuals and 2,318 SNPs.
<!-- - Chromosome 22
- EUR ancestry subset
- SNP-only variants
- MAF ≥ 0.01
- 503 individuals
- 2,318 SNPs after filtering -->

<!-- Preprocessing performed with PLINK2:
- Subset to EUR samples
- Remove non-SNP variants
- Apply MAF filter
- Thin variants
- Assign stable variant IDs
- Compute top 5 principal components (PCs)
- Export dosage matrix -->

We process the .raw file generated using PLINK to get inputs to run GWAS using:

    scripts/from_plink2_to_litegwas.py

This generates geno.py (genotype matrix), covar.tsv (covariates), snp.tsv (SNP metadata), pheno.tsv (phenotype information) and causal_truth.tsv (causal SNPs and effect sizes). It requires the .raw file and .pvar or .bim file (depending on use of specific versions of PLINK) as inputs.

To run on real data:

    python -m litegwas.run \
      --geno <path>/geno.npy \
      --pheno <path>/pheno.tsv \
      --covar <path>/covar.tsv \
      --snp <path>/snp.tsv \
      --out out/real_results.tsv \
      --plot_prefix out/real

In our results, we analyzed the Manhattan and QQ plots, and observed strong signals associated with a p-values close to 3e-10. 

---

## 4. Performed Case/control analysis with 1000 Genomes Data

We also used 1000 Genomes dataset for chromosome 15, focusing on the EUR ancestry subset. We applied MAF filtering of >= 0.01, and did LD-pruning. Here, we simulated binary phenotypes to perform case/control analysis.

We used Logistic regression and generated Manhattan and QQ plots.

To run:

    python -m litegwas.run \
      --geno <path>/geno.npy \
      --pheno <path>/pheno.tsv \
      --covar <path>/covar.tsv \
      --snp <path>/snp.tsv \
      --out out/real_results.tsv \
      --plot_prefix out/real \
      --type binary

Currently, this is not setup to use covariates, so the --covar option can be ignored.

# To do

We have completed most features of our algorithm, and have run it on our dataset, and generated results. Our next steps include:
- Benchmarking against PLINK and performing comparative analysis
- Simulating phenotypes using haptools
- Writing the final report

---

# Example notebooks

We have demonstrated some of our results on single SNPs and the case/control analysis in the notebooks/ folder.

# Installation

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Dependencies:
- numpy
- pandas
- scipy
- matplotlib
- statsmodels