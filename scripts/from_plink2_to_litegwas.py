import re
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

# --- drop-in replacements / edits for from_plink_to_litegwas.py ---


def _base_variant_id(raw_col: str) -> str:
    """
    Turn a .raw SNP column name into a variant ID that should match the .pvar ID.

    Handles common cases:
      - "rs123_A" -> "rs123"
      - "chr:pos:ref:alt_A" -> "chr:pos:ref:alt"
      - but DOES NOT blindly split at the first "_" (since some IDs include underscores)

    Heuristic:
      - If the *last* underscore-separated token looks like an allele/dosage tag, drop it.
      - Otherwise, keep the full string.
    """
    parts = raw_col.split("_")
    if len(parts) <= 1:
        return raw_col

    last = parts[-1]
    # Common allele/dosage suffixes seen in exports
    if last in {"A", "C", "G", "T", "0", "1", "2"}:
        return "_".join(parts[:-1])

    # Sometimes suffix is longer but still allele-ish (e.g., "ALT", "REF", "HAP1")
    if re.fullmatch(r"[ACGT]{1,2}", last):
        return "_".join(parts[:-1])

    return raw_col


def pvar_to_snp(pvar_path: str, raw_snp_cols):
    """
    Build snp.tsv aligned with geno.npy using variant order.

    Since .raw SNP columns often look like '._ALT' when variant IDs are '.',
    we rely on the fact that PLINK writes variants in the SAME order in
    .raw and .pvar.
    """

    import pandas as pd

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


def simulate_pheno_from_geno(G, iids, snp_ids, m_causal=25, h2=0.3, seed=42):
    """
    Same simulation as before, but truth is recorded as SNP IDs (and indices),
    making evaluation robust even if ordering changes later.
    """
    rng = np.random.default_rng(seed)
    N, M = G.shape
    m = min(m_causal, M)
    causal_idx = rng.choice(M, size=m, replace=False)
    effects = rng.normal(0.0, 0.1, size=m).astype(np.float32)

    g = G[:, causal_idx] @ effects
    vg = float(np.var(g, ddof=1))
    ve = vg * (1.0 - h2) / max(h2, 1e-6)

    y = g + rng.normal(0.0, np.sqrt(ve), size=N).astype(np.float32)
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
                    help="Output directory for LiteGWAS inputs")
    ap.add_argument("--m_causal", type=int, default=25)
    ap.add_argument("--h2", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_path = args.prefix + ".raw"
    eigenvec_path = args.prefix + ".eigenvec"
    pvar_path = args.prefix + ".pvar"

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load genotype matrix from .raw
    G, iids, raw_snp_cols = load_raw_to_geno(raw_path)
    print("Example .raw SNP columns:", raw_snp_cols[:5])
    np.save(os.path.join(args.outdir, "geno.npy"), G)

    # 2) Load covariates (PCs) from .eigenvec
    cov = eigenvec_to_covar(eigenvec_path)
    cov.to_csv(os.path.join(args.outdir, "covar.tsv"), sep="\t", index=False)

    # 3) Build SNP metadata from .pvar in the same order as .raw SNP columns
    #    (critical when .raw columns look like '._G' etc and IDs are missing)
    snp = pvar_to_snp(pvar_path, raw_snp_cols)
    snp.to_csv(os.path.join(args.outdir, "snp.tsv"), sep="\t", index=False)

    # 4) Use the snp.tsv IDs (NOT raw column names) for simulation truth
    snp_ids = snp["snp_id"].astype(str).tolist()

    # 5) Simulate phenotype + causal truth (stores SNP IDs + indices)
    pheno, truth = simulate_pheno_from_geno(
        G,
        iids,
        snp_ids,
        m_causal=args.m_causal,
        h2=args.h2,
        seed=args.seed,
    )
    pheno.to_csv(os.path.join(args.outdir, "pheno.tsv"), sep="\t", index=False)
    truth.to_csv(os.path.join(args.outdir, "causal_truth.tsv"),
                 sep="\t", index=False)

    print(f"✅ Wrote LiteGWAS inputs to {args.outdir}")
    print("Files: geno.npy, covar.tsv, snp.tsv, pheno.tsv, causal_truth.tsv")


if __name__ == "__main__":
    main()
