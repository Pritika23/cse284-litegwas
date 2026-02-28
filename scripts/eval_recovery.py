import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="out/results.tsv")
    ap.add_argument("--truth", required=True,
                    help="data/test/causal_truth.tsv")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    res = pd.read_csv(args.results, sep="\t")
    truth = pd.read_csv(args.truth, sep="\t")

    top = set(res.nsmallest(args.topk, "p")["snp_id"])
    causal = set(truth["causal_snp_id"])

    overlap = len(top & causal)
    print(f"Top-{args.topk} overlap with causal SNPs: {overlap}/{len(causal)}")
    if overlap > 0:
        print("Example overlaps:", list(top & causal)[:10])


if __name__ == "__main__":
    main()
