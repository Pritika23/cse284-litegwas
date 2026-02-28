import argparse
import os
import numpy as np
import pandas as pd


def simulate_genotypes(N, M, seed=0):
    rng = np.random.default_rng(seed)
    # allele frequencies away from extremes
    p = rng.uniform(0.05, 0.5, size=M)
    # dosage ~ Binomial(2, p) -> values 0/1/2
    G = rng.binomial(2, p, size=(N, M)).astype(np.float32)
    return G


def simulate_covariates(N, K, seed=0):
    rng = np.random.default_rng(seed + 1)
    # "PC-like" covariates (just Gaussian for MVP)
    C = rng.normal(size=(N, K)).astype(np.float32)
    return C


def simulate_phenotype(G, m_causal=25, h2=0.3, seed=0):
    rng = np.random.default_rng(seed + 2)
    N, M = G.shape

    causal_idx = rng.choice(M, size=m_causal, replace=False)
    effects = rng.normal(loc=0.0, scale=0.1, size=m_causal).astype(np.float32)

    g = G[:, causal_idx] @ effects  # genetic component

    vg = float(np.var(g, ddof=1))
    ve = vg * (1.0 - h2) / max(h2, 1e-6)  # noise variance to hit target h2
    y = g + rng.normal(loc=0.0, scale=np.sqrt(ve), size=N).astype(np.float32)

    return y.astype(np.float32), causal_idx, effects


def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic GWAS dataset (geno/pheno/covar/snp).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--m_causal", type=int, default=25)
    ap.add_argument("--h2", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    G = simulate_genotypes(args.N, args.M, seed=args.seed)
    C = simulate_covariates(args.N, args.K, seed=args.seed)
    y, causal_idx, effects = simulate_phenotype(
        G, m_causal=args.m_causal, h2=args.h2, seed=args.seed)

    iids = [f"sample_{i:04d}" for i in range(args.N)]
    snp_ids = [f"rs{i:07d}" for i in range(args.M)]

    # Save genotype matrix
    np.save(os.path.join(args.outdir, "geno.npy"), G)

    # Save phenotype
    pd.DataFrame({"IID": iids, "y": y}).to_csv(
        os.path.join(args.outdir, "pheno.tsv"), sep="\t", index=False
    )

    # Save covariates
    cov = {"IID": iids}
    for k in range(args.K):
        cov[f"PC{k+1}"] = C[:, k]
    pd.DataFrame(cov).to_csv(
        os.path.join(args.outdir, "covar.tsv"), sep="\t", index=False
    )

    # Save SNP metadata
    snp_df = pd.DataFrame({
        "snp_id": snp_ids,
        "chr": ["1"] * args.M,
        "pos": np.arange(1, args.M + 1),
        "a1": ["A"] * args.M,
        "a2": ["G"] * args.M,
    })
    snp_df.to_csv(os.path.join(args.outdir, "snp.tsv"), sep="\t", index=False)

    # Save ground truth (so you can verify top hits later)
    truth_df = pd.DataFrame({
        "causal_snp_id": [snp_ids[i] for i in causal_idx],
        "effect": effects
    })
    truth_df.to_csv(os.path.join(
        args.outdir, "causal_truth.tsv"), sep="\t", index=False)

    print(f"✅ Wrote synthetic dataset to: {args.outdir}")
    print("Files created: geno.npy, pheno.tsv, covar.tsv, snp.tsv, causal_truth.tsv")


if __name__ == "__main__":
    main()
