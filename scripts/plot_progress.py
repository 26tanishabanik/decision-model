"""Training curves (mean dev accuracy and training loss) for one or more runs.

    python scripts/plot_progress.py --runs lora=path/to/log.jsonl frozen=path/to/log.jsonl --out results/progress.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=log.jsonl")
    ap.add_argument("--out", default="results/progress.png")
    a = ap.parse_args()
    runs = {k: [json.loads(l) for l in open(v) if l.strip()] for k, v in (r.split("=", 1) for r in a.runs)}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor(SURF)
    for (name, rows), c in zip(runs.items(), C):
        xs = [r["step"] for r in rows]
        ax[0].plot(xs, [r["dev"]["_macro_acc"] for r in rows], color=c, lw=2, marker="o", ms=5)
        ax[0].annotate(name, (xs[-1], rows[-1]["dev"]["_macro_acc"]), xytext=(6, 0), textcoords="offset points",
                       va="center", fontsize=9)
        lr = [r for r in rows if "train_loss" in r]
        ax[1].plot([r["step"] for r in lr], [r["train_loss"] for r in lr], color=c, lw=2, marker="o", ms=5)
        if lr:
            ax[1].annotate(name, (lr[-1]["step"], lr[-1]["train_loss"]), xytext=(6, 0), textcoords="offset points",
                           va="center", fontsize=9)
    for axx, title, yl in ((ax[0], "Mean dev accuracy", "dev accuracy (mean over sources)"),
                           (ax[1], "Training loss", "loss (smoothed CE + RPS)")):
        axx.set_facecolor(SURF); axx.grid(True, color=GRID, lw=0.8); axx.set_axisbelow(True)
        axx.set_title(title, loc="left", fontsize=11); axx.set_xlabel("optimizer step"); axx.set_ylabel(yl)
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)


if __name__ == "__main__":
    main()
