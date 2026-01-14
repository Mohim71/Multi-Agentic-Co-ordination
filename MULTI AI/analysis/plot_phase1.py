import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import csv
from collections import defaultdict
import matplotlib.pyplot as plt

def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # cast numeric fields
            for k in ["N","regen","seed","U_total","collapsed","t_collapse"]:
                r[k] = int(r[k])
            for k in ["cooperation","gini","stability_var"]:
                r[k] = float(r[k])
            rows.append(r)
    return rows

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def plot_coop_by_N(rows):
    # average across regen+protocol+memory+seeds
    byN = defaultdict(list)
    for r in rows:
        byN[r["N"]].append(r["cooperation"])
    Ns = sorted(byN.keys())
    ys = [mean(byN[n]) for n in Ns]
    plt.figure()
    plt.plot(Ns, ys, marker="o")
    plt.xlabel("N agents")
    plt.ylabel("Mean Cooperation Index")
    plt.title("Cooperation vs Population Size (Phase 1)")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/plot_coop_vs_N.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_protocol_comparison(rows, metric="cooperation"):
    # average across N+regen+memory+seeds
    byP = defaultdict(list)
    for r in rows:
        byP[r["protocol"]].append(r[metric])
    prots = ["no_chat","roundtable","mediator"]
    ys = [mean(byP[p]) for p in prots]
    plt.figure()
    plt.bar(prots, ys)
    plt.ylabel(f"Mean {metric}")
    plt.title(f"{metric} by Protocol (Phase 1)")
    plt.savefig(f"results/plot_{metric}_by_protocol.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    csv_path = "results/phase1_metrics.csv"
    rows = read_rows(csv_path)
    plot_coop_by_N(rows)
    plot_protocol_comparison(rows, "cooperation")
    plot_protocol_comparison(rows, "gini")
    plot_protocol_comparison(rows, "stability_var")
    print("Plots saved to results/")
