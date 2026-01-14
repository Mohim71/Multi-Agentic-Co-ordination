import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import csv
from collections import defaultdict
import math
import matplotlib.pyplot as plt

CSV_PATH = "results/phase1_metrics.csv"
OUT_DIR = "results"

def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # ints
            for k in ["N", "regen", "seed", "U_total", "collapsed", "t_collapse"]:
                r[k] = int(r[k])
            # floats
            for k in ["cooperation", "gini", "stability_var"]:
                r[k] = float(r[k])
            rows.append(r)
    return rows

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def group(rows, keys):
    d = defaultdict(list)
    for r in rows:
        k = tuple(r[x] for x in keys)
        d[k].append(r)
    return d

def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)

def protocol_x_N_plot(rows):
    ensure_out()
    prots = ["no_chat", "roundtable", "mediator"]
    Ns = sorted({r["N"] for r in rows})

    plt.figure()
    for p in prots:
        ys = []
        for N in Ns:
            vals = [r["cooperation"] for r in rows if r["protocol"] == p and r["N"] == N]
            ys.append(mean(vals))
        plt.plot(Ns, ys, marker="o", label=p)

    plt.xlabel("N agents")
    plt.ylabel("Mean Cooperation Index")
    plt.title("Interaction: Protocol × N (Cooperation)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "plot_protocol_x_N_cooperation.png"), dpi=200, bbox_inches="tight")
    plt.close()

def protocol_x_scarcity_plot(rows):
    ensure_out()
    prots = ["no_chat", "roundtable", "mediator"]
    regens = sorted({r["regen"] for r in rows})  # 8,5,3

    plt.figure()
    for p in prots:
        ys = []
        for g in regens:
            vals = [r["cooperation"] for r in rows if r["protocol"] == p and r["regen"] == g]
            ys.append(mean(vals))
        plt.plot(regens, ys, marker="o", label=p)

    plt.xlabel("Regen (higher = less scarce)")
    plt.ylabel("Mean Cooperation Index")
    plt.title("Interaction: Protocol × Scarcity (Cooperation)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "plot_protocol_x_scarcity_cooperation.png"), dpi=200, bbox_inches="tight")
    plt.close()

def collapse_heatmaps(rows):
    """
    Create 3 heatmaps (one per protocol): x-axis regen, y-axis N, value = collapse rate.
    """
    ensure_out()
    prots = ["no_chat", "roundtable", "mediator"]
    Ns = sorted({r["N"] for r in rows})
    regens = sorted({r["regen"] for r in rows})

    for p in prots:
        # build matrix [len(Ns)][len(regens)]
        mat = []
        for N in Ns:
            row_vals = []
            for g in regens:
                subset = [r for r in rows if r["protocol"] == p and r["N"] == N and r["regen"] == g]
                if not subset:
                    row_vals.append(0.0)
                else:
                    row_vals.append(mean([r["collapsed"] for r in subset]))
            mat.append(row_vals)

        plt.figure()
        plt.imshow(mat, aspect="auto")  # default colormap
        plt.xticks(range(len(regens)), [str(g) for g in regens])
        plt.yticks(range(len(Ns)), [str(N) for N in Ns])
        plt.xlabel("Regen (scarcity)")
        plt.ylabel("N agents")
        plt.title(f"Collapse Rate Heatmap (protocol={p})")
        plt.colorbar(label="Collapse rate")
        plt.savefig(os.path.join(OUT_DIR, f"heatmap_collapse_{p}.png"), dpi=200, bbox_inches="tight")
        plt.close()

def write_summary_table(rows):
    """
    Write a CSV table with mean±std for key metrics grouped by (protocol, N, regen, memory).
    """
    ensure_out()
    keys = ["protocol", "N", "regen", "memory"]
    grouped = group(rows, keys)

    out_path = os.path.join(OUT_DIR, "phase1_summary_table.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = keys + [
            "cooperation_mean", "cooperation_std",
            "gini_mean", "gini_std",
            "stability_mean", "stability_std",
            "collapse_rate"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for k, items in sorted(grouped.items()):
            coop = [r["cooperation"] for r in items]
            gini = [r["gini"] for r in items]
            stab = [r["stability_var"] for r in items]
            coll = [r["collapsed"] for r in items]
            row = {
                "protocol": k[0],
                "N": k[1],
                "regen": k[2],
                "memory": k[3],
                "cooperation_mean": mean(coop),
                "cooperation_std": std(coop),
                "gini_mean": mean(gini),
                "gini_std": std(gini),
                "stability_mean": mean(stab),
                "stability_std": std(stab),
                "collapse_rate": mean(coll),
            }
            w.writerow(row)

    print("Saved summary table:", out_path)

def main():
    rows = read_rows(CSV_PATH)
    protocol_x_N_plot(rows)
    protocol_x_scarcity_plot(rows)
    collapse_heatmaps(rows)
    write_summary_table(rows)
    print("Step 7A outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
