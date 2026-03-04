import argparse
import numpy as np
import pandas as pd

def _norm_chr(x):
    s = str(x)
    return s[3:] if s.lower().startswith("chr") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", required=True)
    ap.add_argument("--plink", required=True)
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    lite = pd.read_csv(args.lite, sep="\t")
    plink = pd.read_csv(args.plink, sep=r"\s+", engine="python")

    # --- sanity: required cols ---
    for c in ["chr", "pos", "beta", "p"]:
        if c not in lite.columns:
            raise ValueError(f"LiteGWAS missing '{c}'. Columns={lite.columns.tolist()}")

    # PLINK header varies; these are common in --glm output
    # Try to find chromosome/position columns:
    chrom_col = None
    for cand in ["#CHROM", "CHROM", "CHR"]:
        if cand in plink.columns:
            chrom_col = cand
            break
    if chrom_col is None:
        raise ValueError(f"PLINK missing chromosome column (#CHROM/CHROM/CHR). Columns={plink.columns.tolist()}")

    if "POS" not in plink.columns:
        raise ValueError(f"PLINK missing POS column. Columns={plink.columns.tolist()}")

    # Effect/p-value columns
    if "BETA" not in plink.columns:
        raise ValueError(f"PLINK missing BETA column. Columns={plink.columns.tolist()}")
    if "P" not in plink.columns:
        raise ValueError(f"PLINK missing P column. Columns={plink.columns.tolist()}")

    # If PLINK has multiple tests per variant, keep additive
    if "TEST" in plink.columns:
        plink = plink[plink["TEST"].astype(str).str.upper().isin(["ADD"])].copy()

    # Normalize types
    lite2 = lite.copy()
    lite2["chr"] = lite2["chr"].map(_norm_chr)
    lite2["pos"] = pd.to_numeric(lite2["pos"], errors="coerce")

    plink2 = plink.copy()
    plink2["chr"] = plink2[chrom_col].map(_norm_chr)
    plink2["pos"] = pd.to_numeric(plink2["POS"], errors="coerce")

    # Merge on chr+pos
    m = lite2.merge(
        plink2[["chr", "pos", "BETA", "P"]].rename(columns={"BETA": "BETA_P", "P": "P_P"}),
        on=["chr", "pos"],
        how="inner",
    )

    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta", "p", "BETA_P", "P_P"])
    if len(m) == 0:
        raise ValueError("Merge produced 0 rows even on chr+pos. Check that PLINK and LiteGWAS used same variant set.")

    pearson_beta = m["beta"].corr(m["BETA_P"], method="pearson")
    m["lp_lite"] = -np.log10(m["p"].clip(lower=1e-300))
    m["lp_plink"] = -np.log10(m["P_P"].clip(lower=1e-300))
    spearman_lp = m["lp_lite"].corr(m["lp_plink"], method="spearman")

    def topk_mask(df, pcol, k):
        return df.nsmallest(k, pcol).index

    k = args.topk
    A = set(topk_mask(m, "p", k))
    B = set(topk_mask(m, "P_P", k))
    jacc = len(A & B) / max(len(A | B), 1)

    print(f"Merged variants (chr+pos): {len(m)}")
    print(f"Pearson corr(beta): {pearson_beta:.4f}")
    print(f"Spearman corr(-log10 p): {spearman_lp:.4f}")
    print(f"Top-{k} Jaccard overlap: {jacc:.4f}")

if __name__ == "__main__":
    main()
