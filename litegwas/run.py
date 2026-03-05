import argparse
import os
import pandas as pd

from .core import gwas_ols, gwas_logistic
from .io import (
    load_geno_npy,
    load_pheno_tsv,
    load_covar_tsv,
    align_pheno_covar,
    load_snp_tsv,
)
from .plots import manhattan_plot, qq_plot


def main():
    ap = argparse.ArgumentParser(description="Python implementation of GWAS")
    ap.add_argument("--geno", required=True, help="Path to geno.npy (N,M)")
    ap.add_argument("--pheno", required=True, help="Path to pheno.tsv with columns IID, y")
    ap.add_argument("--covar", default=None, help="Optional covar.tsv with IID and covariate columns")
    ap.add_argument("--snp", required=True, help="Path to snp.tsv with SNP metadata (length M)")
    ap.add_argument("--out", required=True, help="Output results.tsv path")
    ap.add_argument("--type", default="quantitative", help="Quantitative or binary")
    ap.add_argument("--plot_prefix", default=None, help="If set, saves {prefix}_manhattan.png and {prefix}_qq.png")
    args = ap.parse_args()

    G = load_geno_npy(args.geno)
    pheno = load_pheno_tsv(args.pheno)
    covar = load_covar_tsv(args.covar)

    iids, y, C = align_pheno_covar(pheno, covar)

    # geno.npy rows are in the same order as pheno.tsv IIDs
    if len(iids) != G.shape[0]:
        raise ValueError(
            f"After aligning pheno/covar, N={len(iids)} but geno.npy has N={G.shape[0]}."
        )

    snp = load_snp_tsv(args.snp, M=G.shape[1])

    # Type = quantitative
    beta, se, tstat, pval, df = gwas_ols(G, y, C)

    out_df = snp.copy()
    out_df["beta"] = beta
    out_df["se"] = se
    out_df["t"] = tstat
    out_df["p"] = pval
    out_df["df"] = df

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"Results written to: {args.out}")
    if args.plot_prefix is not None:
        man_png = f"{args.plot_prefix}_manhattan.png"
        qq_png = f"{args.plot_prefix}_qq.png"
        manhattan_plot(out_df, man_png, title="GWAS Manhattan")
        qq_plot(out_df, qq_png, title="GWAS QQ")
        print(f"Saved plots: {man_png}, {qq_png}")

    # # Type = case/control
    # pval = gwas_logistic(G, y, C)
    # out_df = snp.copy()
    # out_df["p"] = pval
    # os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # out_df.to_csv(args.out, sep="\t", index=False)
    # print(f"Results written to: {args.out}")
    # if args.plot_prefix is not None:
    #     man_png = f"{args.plot_prefix}_manhattan.png"
    #     qq_png = f"{args.plot_prefix}_qq.png"
    #     manhattan_plot(out_df, man_png, title="GWAS Manhattan")
    #     qq_plot(out_df, qq_png, title="GWAS QQ")
    #     print(f"Saved plots: {man_png}, {qq_png}")


if __name__ == "__main__":
    main()