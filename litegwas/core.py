import numpy as np
from scipy import stats
import statsmodels.api as sm

def gwas_ols(G: np.ndarray, y: np.ndarray, C: np.ndarray | None):
    if G.ndim != 2:
        raise ValueError("G must be 2D (N, M)")
    y = np.asarray(y, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    N, M = G.shape
    # allele frequency p = mean(G)/2 -> here G is the genotype value (0/1/2) 
    # and its expected value is 2p (after calculation)
    p = np.nanmean(G, axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p) # because the minor allele is the less common allele
    if y.shape[0] != N:
        raise ValueError(f"y length {y.shape[0]} != N {N}")

    # base design matrix X0 = [1, C] (intercept + covariates)
    if C is None:
        X0 = np.ones((N, 1), dtype=np.float64)
    else:
        C = np.asarray(C, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != N:
            raise ValueError("C must be (N, K)")
        X0 = np.concatenate([np.ones((N, 1), dtype=np.float64), C], axis=1)

    p0 = X0.shape[1]           # intercept + covariates
    p = p0 + 1                 # add SNP term
    df = N - p
    if df <= 0:
        raise ValueError(f"Not enough samples: N={N}, params={p} => df={df}")

    beta = np.empty(M, dtype=np.float64)
    se = np.empty(M, dtype=np.float64)
    tstat = np.empty(M, dtype=np.float64)
    pval = np.empty(M, dtype=np.float64)

    for j in range(M):
        gj = G[:, j].reshape(-1, 1)  # SNP j is taken here and linear regression is run on it
        # Skip very rare variants (MAF < 0.01) for numerical stability
        if maf[j] < 0.01 or not np.isfinite(maf[j]):
            beta[j] = np.nan
            se[j] = np.nan
            tstat[j] = np.nan
            pval[j] = np.nan
            continue
        X = np.concatenate([X0, gj], axis=1)  # (N, p) X = [X0, gj] full design matrix for this snp

        # ols solution
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # residuals + sigma^2
        resid = y - X @ b
        rss = float(resid.T @ resid)
        sigma2 = rss / df

        # standard error of SNP coefficient
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
        var_bj = sigma2 * XtX_inv[-1, -1]
        sej = np.sqrt(max(var_bj, 0.0))

        bj = float(b[-1])
        tj = bj / sej if sej > 0 else np.nan
        pj = 2.0 * stats.t.sf(np.abs(tj), df=df) if np.isfinite(tj) else np.nan

        beta[j] = bj
        se[j] = sej
        tstat[j] = tj
        pval[j] = pj

    return beta, se, tstat, pval, df

def gwas_logistic(G, y, C):
    p_values = []
    G_t = G.T

    for i in range(G_t.shape[0]):

        snp_genotypes = G_t[i, :].astype(float)

        # skip SNPs with no variation
        if np.std(snp_genotypes) == 0:
            p_values.append(1.0)
            continue

        X = sm.add_constant(snp_genotypes)

        try:
            result = sm.Logit(y, X).fit(disp=False)

            if result.mle_retvals["converged"]:
                p_values.append(result.pvalues[1])
            else:
                p_values.append(1.0)

        except:
            p_values.append(1.0)

    return p_values