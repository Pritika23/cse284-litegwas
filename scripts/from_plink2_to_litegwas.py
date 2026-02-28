import argparse
import os
import numpy as np
import pandas as pd

def load_raw_to_geno(raw_path: str):
    df = pd.read_csv(raw_path, sep=r"\s+", engine="python")
    # PLINK .raw starts with: FID IID PAT MAT SEX PHENOTYPE then SNP columns
    # Some exports may omit PAT/MAT; handle generically:
    id_cols = [c for c in ["FID", "IID"] if c in df.columns]
    if "IID" not in df.columns:
        raise ValueError("Expected IID column in .raw")
    # SNP columns are everything except known metadata columns
    meta = set(["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"])
    snp_cols = [c for c in df.columns if c not in meta]

    G = df[snp_cols].to_numpy(dtype=np.float32)
    # replace missing values if any
    if np.isnan(G).any():
        col_means = np.nanmean(G, axis=0)
        inds = np.where(np.isnan(G))
        G[inds] = col_means[inds[1]]

    iids = df["IID"].astype(str).tolist()
    return G, iids, snp_cols

def eigenvec_to_covar(eigenvec_path: str):
    ev = pd.read_csv(eigenvec_path, sep=r"\s+", engine="python")
    # Usually columns: #IID PC1 PC2 ... or FID IID PC1...
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
    # Keep IID + PC columns
    pc_cols = [c for c in ev.columns if c.startswith("PC")]
    out = ev[["IID"] + pc_cols].copy()
    out["IID"] = out["IID"].astype(str)
    return out

def pvar_to_snp(pvar_path: str, raw_snp_cols):
    """
    Robust .pvar parser (no pandas):
    Reads first 5 fields: CHR POS ID REF ALT.
    Matches .raw SNP columns by stripping allele suffix after '_' (e.g. '...::_C' -> base id).
    """
    # Build set of base IDs from .raw columns
    raw_base = []
    for c in raw_snp_cols:
        raw_base.append(c.split("_")[0])  # strip allele suffix

    # Parse pvar into a dict: id -> (chr, pos, a1, a2)
    # a1 = ALT, a2 = REF
    meta = {}
    with open(pvar_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            if line.startswith("#"):
                # header line like: #CHROM POS ID REF ALT ...
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chr_, pos_, vid, ref, alt = parts[:5]
            # Use CHR:POS:REF:ALT if vid is '.' or missing
            if vid == "." or vid == "":
                vid = f"{chr_}:{pos_}:{ref}:{alt}"
            meta[vid] = (chr_, int(pos_), alt, ref)

    rows = []
    missing = 0
    for c, base in zip(raw_snp_cols, raw_base):
        if base in meta:
            chr_, pos_, a1, a2 = meta[base]
            rows.append((base, chr_, pos_, a1, a2))
        else:
            missing += 1
            rows.append((base, "NA", -1, "NA", "NA"))

    if missing > 0:
        print(f"Warning: {missing} / {len(raw_snp_cols)} SNPs not matched in pvar (placeholders kept).")

    import pandas as pd
    return pd.DataFrame(rows, columns=["snp_id", "chr", "pos", "a1", "a2"])

def simulate_pheno_from_geno(G, iids, m_causal=25, h2=0.3, seed=42):
    rng = np.random.default_rng(seed)
    N, M = G.shape
    causal_idx = rng.choice(M, size=min(m_causal, M), replace=False)
    effects = rng.normal(0.0, 0.1, size=causal_idx.size).astype(np.float32)
    g = G[:, causal_idx] @ effects
    vg = float(np.var(g, ddof=1))
    ve = vg * (1.0 - h2) / max(h2, 1e-6)
    y = g + rng.normal(0.0, np.sqrt(ve), size=N).astype(np.float32)
    pheno = pd.DataFrame({"IID": iids, "y": y})
    truth = pd.DataFrame({"causal_snp_index": causal_idx, "effect": effects})
    return pheno, truth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="PLINK2 prefix, e.g. chr22_eur_20k")
    ap.add_argument("--outdir", required=True, help="Output directory for LiteGWAS inputs")
    ap.add_argument("--m_causal", type=int, default=25)
    ap.add_argument("--h2", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_path = args.prefix + ".raw"
    eigenvec_path = args.prefix + ".eigenvec"
    pvar_path = args.prefix + ".pvar"

    os.makedirs(args.outdir, exist_ok=True)

    G, iids, raw_snp_cols = load_raw_to_geno(raw_path)
    print("Example .raw SNP columns:", raw_snp_cols[:5])
    np.save(os.path.join(args.outdir, "geno.npy"), G)

    cov = eigenvec_to_covar(eigenvec_path)
    cov.to_csv(os.path.join(args.outdir, "covar.tsv"), sep="\t", index=False)

    snp = pvar_to_snp(pvar_path, raw_snp_cols)
    snp.to_csv(os.path.join(args.outdir, "snp.tsv"), sep="\t", index=False)

    pheno, truth = simulate_pheno_from_geno(G, iids, m_causal=args.m_causal, h2=args.h2, seed=args.seed)
    pheno.to_csv(os.path.join(args.outdir, "pheno.tsv"), sep="\t", index=False)
    truth.to_csv(os.path.join(args.outdir, "causal_truth.tsv"), sep="\t", index=False)

    print(f"✅ Wrote LiteGWAS inputs to {args.outdir}")
    print("Files: geno.npy, covar.tsv, snp.tsv, pheno.tsv, causal_truth.tsv")

if __name__ == "__main__":
    main()