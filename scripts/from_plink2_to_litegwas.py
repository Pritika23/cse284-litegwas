import re
import argparse
import os
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

def load_raw_to_geno(raw_path: str):
    df = pd.read_csv(raw_path, sep=r"\s+", engine="python")
    # FID IID PAT MAT SEX PHENOTYPE then SNP columns
    id_cols = [c for c in ["FID", "IID"] if c in df.columns]
    if "IID" not in df.columns:
        raise ValueError("Expected IID column in .raw")

    meta = set(["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"])
    snp_cols = [c for c in df.columns if c not in meta]

    G = df[snp_cols].fillna(0).to_numpy(dtype=np.float32)

    iids = df["IID"].astype(str).tolist()
    return G, iids, snp_cols

# this returns covariate matrix for regression


def eigenvec_to_covar(eigenvec_path: str):
    ev = pd.read_csv(eigenvec_path, sep=r"\s+", engine="python")
    if "IID" not in ev.columns:
        # sometimes first column is '#IID'
        if "#IID" in ev.columns:
            ev = ev.rename(columns={"#IID": "IID"})
        else:
            # if it has FID IID
            cols = ev.columns.tolist()
            if len(cols) >= 2:
                ev = ev.rename(columns={cols[1]: "IID"})
            else:
                raise ValueError("Could not find IID in eigenvec.")
    # keep IID + PC columns
    pc_cols = [c for c in ev.columns if c.startswith("PC")]
    out = ev[["IID"] + pc_cols].copy()
    out["IID"] = out["IID"].astype(str)
    return out


def pvar_to_snp(pvar_path: str, raw_snp_cols):
    # to build snp.tsv aligned with geno.npy using variant order
    records = []
    with open(pvar_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            chr_, pos_, vid, ref, alt = parts[:5]

            if vid == ".":
                vid = f"{chr_}:{pos_}:{ref}:{alt}"

            records.append((vid, chr_, int(pos_), alt, ref))

    if len(records) < len(raw_snp_cols):
        raise ValueError(
            f".pvar has fewer variants ({len(records)}) than .raw columns ({len(raw_snp_cols)})"
        )

    # take only the number of SNPs present in geno matrix
    records = records[: len(raw_snp_cols)]

    snp_df = pd.DataFrame(
        records,
        columns=["snp_id", "chr", "pos", "a1", "a2"]
    )

    return snp_df

_suffix_re = re.compile(r"^(?P<id>.+?)_(?:[ACGT]+|<CN\d+>)$")

def normalize_raw_snp_cols(raw_snp_cols):
    """
    Convert PLINK .raw SNP column names like:
      rs123_A   -> rs123
      rs123_G   -> rs123
    """
    out = []
    for c in raw_snp_cols:
        m = _suffix_re.match(c)
        out.append(m.group("id") if m else c)
    return out

def bim_to_snp(bim_path: str, raw_snp_cols: list[str]) -> pd.DataFrame:
    bim = pd.read_csv(bim_path, sep=r"\s+", header=None,
                      names=["chr", "snp_id", "cm", "pos", "a1", "a2"])

    bim_idx = bim.set_index("snp_id")

    raw_base = normalize_raw_snp_cols(raw_snp_cols)

    try:
        snp = bim_idx.loc[raw_base].reset_index()
    except KeyError as e:
        missing = [c for c in raw_snp_cols if c not in bim_idx.index]
        raise KeyError(f"{len(missing)} SNP IDs in .raw not found in .bim. "
                       f"Example missing: {missing[:5]}") from e

    snp = snp[["snp_id", "chr", "pos", "a1", "a2"]]
    return snp

# simulate phenotype using the genotype matrix
def simulate_pheno_from_geno(G, iids, snp_ids, m_causal=25, h2=0.3, seed=42, type="quantitative"):
    rng = np.random.default_rng(seed)
    N, M = G.shape
    m = min(m_causal, M)
    causal_idx = rng.choice(M, size=m, replace=False)
    # we sample effects of the snps from this distribution
    effects = rng.normal(0.0, 0.1, size=m).astype(np.float32)
    if type=="quantitative":
        
        g = G[:, causal_idx] @ effects
        vg = float(np.var(g, ddof=1))
        if h2 is not None:
            # h2 term is heritability -> how much of the phenotype variation is explained by genetics and how much by noise. so this is to choose the epsilon value (noise) in the regression
            ve = vg * (1.0 - h2) / max(h2, 1e-6)
        else:
            ve = 1.0

        y = g + rng.normal(0.0, np.sqrt(ve), size=N).astype(np.float32)
        pheno = pd.DataFrame({"IID": list(map(str, iids)), "y": y})

        truth = pd.DataFrame({
            "causal_snp_index": causal_idx,
            "causal_snp_id": [snp_ids[i] for i in causal_idx],
            "effect": effects,
        })
    else:
        # polygenic score per individual
        PRS = (G[:, causal_idx] @ effects)

        prevalence = 0.3  # fraction of cases we want on average
        alpha = np.log(prevalence / (1 - prevalence))

        # compute probabilities and sample case/control labels
        p = expit(alpha + PRS)
        y = rng.binomial(1, p, size=N)
        pheno = pd.DataFrame({"IID": list(map(str, iids)), "y": y})

        truth = pd.DataFrame({
            "causal_snp_index": causal_idx,
            "causal_snp_id": [snp_ids[i] for i in causal_idx],
            "effect": effects,
        })

    return pheno, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="PLINK2 prefix, e.g. chr22_eur_20k")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for PyGWAS inputs")
    ap.add_argument("--m_causal", type=int, default=25)
    ap.add_argument("--h2", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--type", type=str, default="quantitative")
    args = ap.parse_args()

    raw_path = args.prefix + ".raw"
    eigenvec_path = args.prefix + ".eigenvec"
    pvar_path = args.prefix + ".pvar"
    bim_path = args.prefix + ".bim"
    type = args.type

    os.makedirs(args.outdir, exist_ok=True)

    # Load genotype matrix from .raw
    G, iids, raw_snp_cols = load_raw_to_geno(raw_path)
    print("Example .raw SNP columns:", raw_snp_cols[:5])
    np.save(os.path.join(args.outdir, "geno.npy"), G)

    # PCs not setup for case/control analysis yet
    if type == "quantitative":
        cov = eigenvec_to_covar(eigenvec_path)
        cov.to_csv(os.path.join(args.outdir, "covar.tsv"), sep="\t", index=False)

    if os.path.exists(pvar_path):
        snp = pvar_to_snp(pvar_path, raw_snp_cols)
        snp.to_csv(os.path.join(args.outdir, "snp.tsv"), sep="\t", index=False)
    elif os.path.exists(bim_path):
        snp = bim_to_snp(bim_path, raw_snp_cols)
        snp.to_csv(os.path.join(args.outdir, "snp.tsv"), sep="\t", index=False)

    snp_ids = snp["snp_id"].astype(str).tolist()

    pheno, truth = simulate_pheno_from_geno(
        G,
        iids,
        snp_ids,
        m_causal=args.m_causal,
        h2=args.h2,
        seed=args.seed,
        type=type
    )
    pheno.to_csv(os.path.join(args.outdir, "pheno.tsv"), sep="\t", index=False)
    truth.to_csv(os.path.join(args.outdir, "causal_truth.tsv"),
                 sep="\t", index=False)

    print(f"Wrote GWAS inputs to {args.outdir}")
    # print("Files: geno.npy, covar.tsv, snp.tsv, pheno.tsv, causal_truth.tsv")


if __name__ == "__main__":
    main()
