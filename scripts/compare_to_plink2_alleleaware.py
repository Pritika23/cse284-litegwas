import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from litegwas.plots import log_odds_comp, logp_comp, beta_comp


def log_odds_comp(m, out_png: str, title: str = "Effect size"):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.log(m["OR_P"]), np.log(m["or"]), s=6, alpha=0.5)
    lims = [
        min(np.log(m["OR_P"]).min(), np.log(m["or"]).min()),
        max(np.log(m["OR_P"]).max(), np.log(m["or"]).max())
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("PLINK log-odds ratio")
    plt.ylabel("PyGWAS log-odds ratio")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def beta_comp(m, out_png: str, title: str = "Effect size"):
    plt.figure(figsize=(6, 6))
    plt.scatter(m["BETA_P"], m["beta"], s=6, alpha=0.5)
    lims = [
        min(m["BETA_P"].min(), m["beta"].min()),
        max(m["BETA_P"].max(), m["beta"].max())
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("PLINK beta")
    plt.ylabel("PyGWAS beta")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def logp_comp(m, out_png: str, title: str = "Association significance"):
    plt.figure(figsize=(6, 6))
    plt.scatter(m["lp_plink"], m["lp_lite"], s=6, alpha=0.5)
    lims = [
        min(m["lp_plink"].min(), m["lp_lite"].min()),
        max(m["lp_plink"].max(), m["lp_lite"].max())
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("PLINK -log10(p)")
    plt.ylabel("PythonGWAS -log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _norm_chr(x):
    s = str(x)
    return s[3:] if s.lower().startswith("chr") else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", required=True,
                    help="PyGWAS results TSV (results/litegwas.tsv)")
    ap.add_argument(
        "--plink",
        required=True,
        help="PLINK2 .glm.linear file (results/plink2.y.glm.linear) or .logistic file (notebooks/results/gwas_results.assoc.logistic)"
    )
    ap.add_argument("--type", default="quantitative",
                    help="Quantitative or binary (case/control)")
    ap.add_argument("--topk", type=int, default=100, help="Top-k for overlap")
    ap.add_argument("--plot_prefix", default=None,
                    help="If set, saves {prefix}_beta.png and {prefix}_logp.png")
    args = ap.parse_args()

    lite = pd.read_csv(args.lite, sep="\t")
    plink = pd.read_csv(args.plink, sep=r"\s+", engine="python")

    if args.type == "quantitative":
        for c in ["chr", "pos", "beta", "p", "a1", "a2"]:
            if c not in lite.columns:
                raise ValueError(
                    f"PyGWAS missing '{c}'. Columns={lite.columns.tolist()}")

        chrom_col = None
        for cand in ["#CHROM", "CHROM", "CHR"]:
            if cand in plink.columns:
                chrom_col = cand
                break
        if chrom_col is None or "POS" not in plink.columns:
            raise ValueError(
                f"PLINK missing CHROM/POS. Columns={plink.columns.tolist()}")

        for c in ["BETA", "P"]:
            if c not in plink.columns:
                raise ValueError(
                    f"PLINK missing '{c}'. Columns={plink.columns.tolist()}")

        if "A1" not in plink.columns:
            raise ValueError(
                f"PLINK missing 'A1' (effect allele). Columns={plink.columns.tolist()}")

        # Keep additive test if multiple tests are present
        if "TEST" in plink.columns:
            plink = plink[plink["TEST"].astype(
                str).str.upper().isin(["ADD"])].copy()

        # Normalize + types
        lite2 = lite.copy()
        lite2["chr"] = lite2["chr"].map(_norm_chr)
        lite2["pos"] = pd.to_numeric(lite2["pos"], errors="coerce")
        lite2["a1"] = lite2["a1"].astype(str).str.upper()
        lite2["a2"] = lite2["a2"].astype(str).str.upper()

        plink2 = plink.copy()
        plink2["chr"] = plink2[chrom_col].map(_norm_chr)
        plink2["pos"] = pd.to_numeric(plink2["POS"], errors="coerce")
        plink2["A1"] = plink2["A1"].astype(str).str.upper()

        # Merge on chr+pos
        m = lite2.merge(
            plink2[["chr", "pos", "A1", "BETA", "P"]].rename(
                columns={"BETA": "BETA_P", "P": "P_P"}),
            on=["chr", "pos"],
            how="inner",
        ).replace([np.inf, -np.inf], np.nan)

        m = m.dropna(subset=["beta", "p", "BETA_P", "P_P", "A1", "a1", "a2"])
        if len(m) == 0:
            raise ValueError("Merge produced 0 rows.")

        same_as_a1 = (m["A1"] == m["a1"]).sum()
        same_as_a2 = (m["A1"] == m["a2"]).sum()
        print(f"A1 matches Lite a1: {same_as_a1}")
        print(f"A1 matches Lite a2: {same_as_a2}")

        # Assume PyGWAS beta is oriented to a2 rather than a1.
        # If PLINK's A1 matches Lite's a1, signs are opposite and must be flipped.
        flip = (m["A1"] == m["a1"]) & (m["A1"] != m["a2"])
        m.loc[flip, "beta"] = -m.loc[flip, "beta"]

        # Metrics
        pearson_beta = m["beta"].corr(m["BETA_P"], method="pearson")

        m["lp_lite"] = -np.log10(m["p"].clip(lower=1e-300))
        m["lp_plink"] = -np.log10(m["P_P"].clip(lower=1e-300))
        spearman_lp = m["lp_lite"].corr(m["lp_plink"], method="spearman")

        def topk_idx(df, pcol, k):
            return set(df.nsmallest(k, pcol).index)

        k = args.topk
        jacc = len(topk_idx(m, "p", k) & topk_idx(m, "P_P", k)) / max(
            len(topk_idx(m, "p", k) | topk_idx(m, "P_P", k)), 1
        )

        print(f"Merged variants (chr+pos): {len(m)}")
        print(f"Flipped betas: {int(flip.sum())} / {len(m)}")
        print(f"Pearson corr(beta) after allele-align: {pearson_beta:.4f}")
        print(
            f"Pearson corr(beta) if all betas flipped: {(-m['beta']).corr(m['BETA_P'], method='pearson'):.4f}")
        print(f"Spearman corr(-log10 p): {spearman_lp:.4f}")
        print(f"Top-{k} Jaccard overlap: {jacc:.4f}")

        if args.plot_prefix is not None:
            beta_png = f"{args.plot_prefix}_beta.png"
            logp_png = f"{args.plot_prefix}_logp.png"
            beta_comp(m, beta_png)
            logp_comp(m, logp_png)

    else:
        for c in ["chr", "pos", "or", "p", "a1", "a2", "snp_id"]:
            if c not in lite.columns:
                raise ValueError(
                    f"PyGWAS missing '{c}'. Columns={lite.columns.tolist()}")

        chrom_col = None
        for cand in ["#CHROM", "CHROM", "CHR"]:
            if cand in plink.columns:
                chrom_col = cand
                break
        if chrom_col is None or "BP" not in plink.columns:
            raise ValueError(
                f"PLINK missing CHROM/POS. Columns={plink.columns.tolist()}")

        for c in ["OR", "P", "SNP"]:
            if c not in plink.columns:
                raise ValueError(
                    f"PLINK missing '{c}'. Columns={plink.columns.tolist()}")

        if "A1" not in plink.columns:
            raise ValueError(
                f"PLINK missing 'A1' (effect allele). Columns={plink.columns.tolist()}")

        # Keep additive test if multiple tests are present
        if "TEST" in plink.columns:
            plink = plink[plink["TEST"].astype(
                str).str.upper().isin(["ADD"])].copy()

        # Normalize + types
        lite2 = lite.copy()
        lite2["chr"] = lite2["chr"].map(_norm_chr)
        lite2["pos"] = pd.to_numeric(lite2["pos"], errors="coerce")
        lite2["a1"] = lite2["a1"].astype(str).str.upper()
        lite2["a2"] = lite2["a2"].astype(str).str.upper()
        lite2["snp_id"] = lite2["snp_id"].astype(str)

        plink2 = plink.copy()
        plink2["chr"] = plink2[chrom_col].map(_norm_chr)
        plink2["pos"] = pd.to_numeric(plink2["BP"], errors="coerce")
        plink2["A1"] = plink2["A1"].astype(str).str.upper()
        plink2["snp_id"] = plink2["SNP"].astype(str)

        # Merge on chr+pos+snp_id
        m = lite2.merge(
            plink2[["snp_id", "chr", "pos", "A1", "OR", "P"]].rename(
                columns={"OR": "OR_P", "P": "P_P"}),
            on=["chr", "pos", "snp_id"],
            how="inner",
        ).replace([np.inf, -np.inf], np.nan)

        m = m[(m["a1"].str.len() == 1) & (m["a2"].str.len() == 1)]
        m = m.dropna(subset=["or", "p", "OR_P", "P_P", "A1", "a1", "a2"])
        if len(m) == 0:
            raise ValueError("Merge produced 0 rows.")

        same_as_a1 = (m["A1"] == m["a1"]).sum()
        same_as_a2 = (m["A1"] == m["a2"]).sum()
        print(f"A1 matches Lite a1: {same_as_a1}")
        print(f"A1 matches Lite a2: {same_as_a2}")

        # Assume PyGWAS OR is oriented to a2 rather than a1.
        # If PLINK's A1 matches Lite's a1, directions are opposite.
        flip = (m["A1"] == m["a1"]) & (m["A1"] != m["a2"])
        m.loc[flip, "or"] = 1.0 / m.loc[flip, "or"]

        m["beta_lite"] = np.log(m["or"])
        m["beta_plink"] = np.log(m["OR_P"])
        pearson_beta = m["beta_lite"].corr(m["beta_plink"], method="pearson")

        m["lp_lite"] = -np.log10(m["p"].clip(lower=1e-300))
        m["lp_plink"] = -np.log10(m["P_P"].clip(lower=1e-300))
        spearman_lp = m["lp_lite"].corr(m["lp_plink"], method="spearman")

        def topk_idx(df, pcol, k):
            return set(df.nsmallest(k, pcol).index)

        k = args.topk
        jacc = len(topk_idx(m, "p", k) & topk_idx(m, "P_P", k)) / max(
            len(topk_idx(m, "p", k) | topk_idx(m, "P_P", k)), 1
        )

        print(f"Merged variants (chr+pos+snp_id): {len(m)}")
        print(f"Flipped odds ratios: {int(flip.sum())} / {len(m)}")
        print(f"Pearson corr(log-odds) after allele-align: {pearson_beta:.4f}")
        print(
            f"Pearson corr(log-odds) if all effects flipped: "
            f"{(-m['beta_lite']).corr(m['beta_plink'], method='pearson'):.4f}"
        )
        print(f"Spearman corr(-log10 p): {spearman_lp:.4f}")
        print(f"Top-{k} Jaccard overlap: {jacc:.4f}")

        if args.plot_prefix is not None:
            beta_png = f"{args.plot_prefix}_beta.png"
            logp_png = f"{args.plot_prefix}_logp.png"
            log_odds_comp(m, beta_png)
            logp_comp(m, logp_png)


if __name__ == "__main__":
    main()
