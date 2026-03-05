import numpy as np
import pandas as pd


def load_geno_npy(path: str) -> np.ndarray:
    G = np.load(path)
    if G.ndim != 2:
        raise ValueError("geno.npy must be a 2D array shaped (N, M)")
    return G


def load_pheno_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if not {"IID", "y"}.issubset(df.columns): # ensures iid and y are columns
        raise ValueError("pheno.tsv must contain columns: IID, y")
    return df[["IID", "y"]].copy()


def load_covar_tsv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path, sep="\t")
    if "IID" not in df.columns:
        raise ValueError("covar.tsv must contain column: IID")
    return df.copy()


def align_pheno_covar(pheno: pd.DataFrame, covar: pd.DataFrame | None):
    #this is to align individuals between phenotypes and genotypes
    if covar is None:
        iids = pheno["IID"].tolist()
        y = pheno["y"].to_numpy()
        return iids, y, None

    merged = pheno.merge(covar, on="IID", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping IIDs between phenotype and covariates.")

    y = merged["y"].to_numpy()
    cov_cols = [c for c in merged.columns if c not in ("IID", "y")]
    C = merged[cov_cols].to_numpy() if len(cov_cols) > 0 else None
    return merged["IID"].tolist(), y, C


def load_snp_tsv(path: str, M: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"snp_id", "chr", "pos", "a1", "a2"}
    if not required.issubset(df.columns):
        raise ValueError(f"snp.tsv must contain columns: {sorted(required)}")
    if len(df) != M:
        raise ValueError(f"snp.tsv has {len(df)} rows but genotype has M={M} SNPs")
    return df[["snp_id", "chr", "pos", "a1", "a2"]].copy()