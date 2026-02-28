import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def manhattan_plot(df: pd.DataFrame, out_png: str, title: str = "Manhattan Plot"):
    """
    Expects df columns: chr, pos, p
    """
    d = df.copy()
    d["chr"] = d["chr"].astype(str)
    d["pos"] = pd.to_numeric(d["pos"], errors="coerce")
    d["p"] = pd.to_numeric(d["p"], errors="coerce")
    d = d.dropna(subset=["pos", "p"])
    d = d.sort_values(["chr", "pos"])

    d["mlogp"] = -np.log10(np.clip(d["p"].values, 1e-300, 1.0))

    # Create a cumulative position for plotting across chromosomes
    chroms = d["chr"].unique().tolist()
    offset = 0
    ticks = []
    ticklabels = []
    cum_pos = np.zeros(len(d), dtype=np.int64)

    for c in chroms:
        mask = (d["chr"] == c).values
        positions = d.loc[mask, "pos"].values.astype(np.int64)
        cum_pos[mask] = positions + offset
        mid = (positions.min() + positions.max()) // 2 + offset
        ticks.append(mid)
        ticklabels.append(c)
        offset += positions.max()

    d["cum_pos"] = cum_pos

    plt.figure(figsize=(12, 4))
    plt.scatter(d["cum_pos"], d["mlogp"], s=6)
    plt.xticks(ticks, ticklabels)
    plt.xlabel("Chromosome")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def qq_plot(df: pd.DataFrame, out_png: str, title: str = "QQ Plot"):
    """
    Expects df column: p
    """
    p = pd.to_numeric(df["p"], errors="coerce").dropna().values
    p = np.clip(p, 1e-300, 1.0)
    p = np.sort(p)

    n = p.size
    exp = -np.log10(np.arange(1, n + 1) / (n + 1.0))
    obs = -np.log10(p)

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(exp, obs, s=8)
    maxv = max(exp.max(), obs.max())
    plt.plot([0, maxv], [0, maxv], linewidth=1)
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()