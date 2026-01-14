import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import csv
from collections import defaultdict
import matplotlib.pyplot as plt

CSV_PATH = "results/phase2_metrics.csv"
OUT_DIR = "results"

def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ["N","regen","seed","U_total","collapsed","t_collapse"]:
                r[k] = int(r[k])
            for k in ["cooperation","gini","stability_var"]:
                r[k] = float(r[k])
            rows.append(r)
    return rows

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def plot_memory_effect(rows, metric="collapsed"):
    os.makedirs(OUT_DIR, exist_ok=True)
    prots = ["no_chat","roundtable","mediator"]
    mems = ["M0","M1","M2"]
    Ns = sorted({r["N"] for r in rows})

    # For each protocol, create a plot with lines for memory, x=N
    for p in prots:
        plt.figure()
        for mem in mems:
            ys = []
            for N in Ns:
                subset = [r for r in rows if r["protocol"]==p and r["memory"]==mem and r["N"]==N]
                if metric == "collapsed":
                    ys.append(mean([r["collapsed"] for r in subset]))
                else:
                    ys.append(mean([r[metric] for r in subset]))
            plt.plot(Ns, ys, marker="o", label=mem)

        plt.xlabel("N agents")
        plt.ylabel("Collapse rate" if metric=="collapsed" else f"Mean {metric}")
        plt.title(f"Phase-2 Stress Test: {metric} vs N (protocol={p}, regen=3)")
        plt.legend()
        out = os.path.join(OUT_DIR, f"phase2_{metric}_memory_effect_{p}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()

def main():
    rows = read_rows(CSV_PATH)
    plot_memory_effect(rows, "collapsed")
    plot_memory_effect(rows, "cooperation")
    plot_memory_effect(rows, "gini")
    plot_memory_effect(rows, "stability_var")
    print("Phase-2 plots saved to results/")

if __name__ == "__main__":
    main()
