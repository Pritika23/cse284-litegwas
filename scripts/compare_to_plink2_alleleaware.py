import argparse
import numpy as np
import pandas as pd

def _norm_chr(x):
    s = str(x)
    return s[3:] if s.lower().startswith("chr") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", required=True, help="LiteGWAS results TSV (results/litegwas.tsv)")
    ap.add_argument("--plink", required=True, help="PLINK2 .glm.linear file (results/plink2.y.glm.linear)")
    ap.add_argument("--topk", type=int, default=100, help="Top-k for overlap")
    args = ap.parse_args()

    lite = pd.read_csv(args.lite, sep="\t")
    plink = pd.read_csv(args.plink, sep=r"\s+", engine="python")

    # --- Lite requirements (from your litegwas output) ---
    for c in ["chr", "pos", "beta", "p", "a1", "a2"]:
        if c not in lite.columns:
            raise ValueError(f"LiteGWAS missing '{c}'. Columns={lite.columns.tolist()}")

    # --- PLINK requirements ---
    chrom_col = None
    for cand in ["#CHROM", "CHROM", "CHR"]:
        if cand in plink.columns:
            chrom_col = cand
            break
    if chrom_col is None or "POS" not in plink.columns:
        raise ValueError(f"PLINK missing CHROM/POS. Columns={plink.columns.tolist()}")

    for c in ["BETA", "P"]:
        if c not in plink.columns:
            raise ValueError(f"PLINK missing '{c}'. Columns={plink.columns.tolist()}")

    if "A1" not in plink.columns:
        raise ValueError(f"PLINK missing 'A1' (effect allele). Columns={plink.columns.tolist()}")

    # Keep additive test if multiple tests are present
    if "TEST" in plink.columns:
        plink = plink[plink["TEST"].astype(str).str.upper().isin(["ADD"])].copy()

    # Normalize + types
    lite2 = lite.copy()
    lite2["chr"] = lite2["chr"].map(_norm_chr)
    lite2["pos"] = pd.to_numeric(lite2["pos"], errors="coerce")
    lite2["a1"] = lite2["a1"].astype(str)
    lite2["a2"] = lite2["a2"].astype(str)

    plink2 = plink.copy()
    plink2["chr"] = plink2[chrom_col].map(_norm_chr)
    plink2["pos"] = pd.to_numeric(plink2["POS"], errors="coerce")
    plink2["A1"] = plink2["A1"].astype(str)

    # Merge on chr+pos (most robust given your ID weirdness)
    m = lite2.merge(
        plink2[["chr", "pos", "A1", "BETA", "P"]].rename(columns={"BETA": "BETA_P", "P": "P_P"}),
        on=["chr", "pos"],
        how="inner",
    ).replace([np.inf, -np.inf], np.nan)

    m = m.dropna(subset=["beta", "p", "BETA_P", "P_P", "A1", "a1", "a2"])
    if len(m) == 0:
        raise ValueError("Merge produced 0 rows.")

    # Flip LiteGWAS beta when PLINK's effect allele (A1) equals Lite's a2 (REF),
    # meaning the effect allele conventions are opposite.
    flip = (m["A1"] == m["a2"]) & (m["A1"] != m["a1"])
    m.loc[flip, "beta"] = -m.loc[flip, "beta"]

    # Metrics
    pearson_beta = m["beta"].corr(m["BETA_P"], method="pearson")

    m["lp_lite"] = -np.log10(m["p"].clip(lower=1e-300))
    m["lp_plink"] = -np.log10(m["P_P"].clip(lower=1e-300))
    spearman_lp = m["lp_lite"].corr(m["lp_plink"], method="spearman")

    def topk_idx(df, pcol, k):
        return set(df.nsmallest(k, pcol).index)

    k = args.topk
    jacc = len(topk_idx(m, "p", k) & topk_idx(m, "P_P", k)) / max(len(topk_idx(m, "p", k) | topk_idx(m, "P_P", k)), 1)

    print(f"Merged variants (chr+pos): {len(m)}")
    print(f"Flipped betas: {int(flip.sum())} / {len(m)}")
    print(f"Pearson corr(beta) after allele-align: {pearson_beta:.4f}")
    print(f"Spearman corr(-log10 p): {spearman_lp:.4f}")
    print(f"Top-{k} Jaccard overlap: {jacc:.4f}")

if __name__ == "__main__":
    main()
