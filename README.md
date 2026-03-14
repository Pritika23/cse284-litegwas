# CSE284 Project

This project implements a lightweight Python implementation of Genome-Wide Association Studies (GWAS) from scratch and applies it to both simulated data and real 1000 Genomes data.

---

## 1. OLS-Based GWAS

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

## Benchmarking LiteGWAS Against PLINK2

To validate our implementation, we compare LiteGWAS results with the standard GWAS implementation provided by **PLINK2**. Both tools run linear regression using the same genotype data, phenotype, and covariates.

---

## 1. Compute Principal Components (Covariates)

Population structure can confound GWAS results, so we compute the top 5 principal components using PLINK2:

```bash
plink2 \
  --pfile chr22_eur_20k_maf01_snps_ids \
  --pca 5 \
  --out chr22_eur_20k_maf01_snps_ids
```

This produces:

```
chr22_eur_20k_maf01_snps_ids.eigenvec
```

which contains the top principal components used as covariates.

---

## 2. Run GWAS with PLINK2

We run PLINK’s linear regression GWAS using the same phenotype and covariates used by LiteGWAS.

```bash
plink2 \
  --pfile chr22_eur_20k_maf01_snps_ids \
  --pheno litegwas_inputs/pheno.tsv \
  --pheno-name y \
  --covar litegwas_inputs/covar.tsv \
  --covar-name PC1,PC2,PC3,PC4,PC5 \
  --glm \
  --out plink_results
```

This generates the association results file:

```
plink_results.y.glm.linear
```

which contains effect sizes, standard errors, and p-values for each SNP.

Similarly, for logistic regression, we run,
```bash
plink \
  --bfile eur.qc.pruned \
  --pheno phenotypes_casecontrol.txt \
  --logistic \
  --out gwas_results \
  --allow-no-sex
```

This generates the association results file:

```
gwas_results.assoc.logistic
```

---

## 3. Run LiteGWAS

LiteGWAS can be run using the following command:

```bash
python -m litegwas.run \
  --geno litegwas_inputs/geno.npy \
  --pheno litegwas_inputs/pheno.tsv \
  --covar litegwas_inputs/covar.tsv \
  --snp litegwas_inputs/snp.tsv \
  --out out/real_results.tsv \
  --type type \
  --plot_prefix out/real
```

This produces:

```
out/real_results.tsv
out/real_manhattan.png
out/real_qq.png
```

---

## 4. Compare Results

We compare the results of LiteGWAS and PLINK using the provided comparison script:

```bash
python scripts/compare_to_plink2_alleleaware.py \
  --lite out/real_results.tsv \
  --plink plink_results.y.glm.linear \
  --topk 100
```
or 
```bash
python scripts/compare_to_plink2_alleleaware.py \
  --lite out/real_results.tsv \
  --plink gwas_results.assoc.logistic \
  --topk 100 \
  --type binary
```

The script reports:

- Number of shared variants between the two outputs
- Correlation between estimated SNP effect sizes
- Overlap among the top-ranked SNPs
- Plots showing the correlations between the effect sizes and the negative logarithm of the p values

## Runtime Benchmark: LiteGWAS vs PLINK2

To evaluate the runtime performance of LiteGWAS, we benchmark it against **PLINK2**, a widely used and highly optimized GWAS tool written in C/C++. Both tools were run on the same dataset with identical phenotype and covariates.

### Dataset

- **Individuals:** 503  
- **Variants (SNPs):** 2318  
- **Covariates:** 5 principal components (PC1–PC5)

---

## Running PLINK2

```bash
/usr/bin/time -v plink2 \
  --pfile realdata/chr22/chr22_eur_20k_maf01_snps_ids \
  --pheno litegwas_input/pheno.tsv \
  --pheno-name y \
  --covar litegwas_input/covar.tsv \
  --covar-name PC1,PC2,PC3,PC4,PC5 \
  --glm \
  --out plink_results
```

Output file:

```
plink_results.y.glm.linear
```

---

## Running LiteGWAS

```bash
/usr/bin/time -v python -m litegwas.run \
  --geno litegwas_input/geno.npy \
  --pheno litegwas_input/pheno.tsv \
  --covar litegwas_input/covar.tsv \
  --snp litegwas_input/snp.tsv \
  --out out/real_results.tsv \
  --plot_prefix out/real
```

Outputs:

```
out/real_results.tsv
out/real_manhattan.png
out/real_qq.png
```

---

## Runtime Results

| Tool | Runtime | Peak Memory |
|-----|------|------|
| **PLINK2** | 0.20 s | ~14 MB |
| **LiteGWAS** | 46.85 s | ~366 MB |


PLINK2 is significantly faster because it is implemented in **optimized C/C++ with multithreading and efficient genotype storage formats**.

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


