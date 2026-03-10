import argparse
import pandas as pd

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def compute_auprc(res, truth):

    causal = set(truth["causal_snp_id"])

    # label SNPs
    y_true = res["snp_id"].isin(causal).astype(int)

    # score = -log10(p)
    scores = -np.log10(res["p"].clip(lower=1e-300))

    precision, recall, _ = precision_recall_curve(y_true, scores)
    auprc = auc(recall, precision)

    print(f"AUPRC: {auprc:.4f}")

    return precision, recall

def causal_rank_distribution(res, truth):

    res = res.sort_values("p").reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)

    causal = set(truth["causal_snp_id"])

    causal_ranks = res[res["snp_id"].isin(causal)]["rank"]

    print("Causal SNP ranks summary:")
    print(causal_ranks.describe())

    return causal_ranks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="out/results.tsv")
    ap.add_argument("--truth", required=True,
                    help="data/test/causal_truth.tsv")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    res = pd.read_csv(args.results, sep="\t")
    # plink
    # res = pd.read_csv("plink_gwas.assoc.logistic", sep=r"\s+", engine="python")
    truth = pd.read_csv(args.truth, sep="\t")

    top = set(res.nsmallest(args.topk, "p")["snp_id"])
    causal = set(truth["causal_snp_id"])

    overlap = len(top & causal)
    print(f"Top-{args.topk} overlap with causal SNPs: {overlap}/{len(causal)}")
    if overlap > 0:
        print("Example overlaps:", list(top & causal)[:10])

    precision = overlap / args.topk
    recall = overlap / len(causal)

    print(f"Precision@{args.topk}: {precision:.3f}")
    print(f"Recall@{args.topk}: {recall:.3f}")

    precision, recall = compute_auprc(res, truth)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    ranks = causal_rank_distribution(res, truth)

    plt.hist(ranks, bins=20)
    plt.xlabel("Rank of causal SNP")
    plt.ylabel("Count")
    plt.title("Distribution of causal SNP ranks")
    plt.show()

if __name__ == "__main__":
    main()