"""Frozen-test-set report for one or more trained models.

    python scripts/report.py --runs lora=cache/predictions/lora frozen=cache/predictions/frozen [--out results/report.json]
Each directory holds predictions.jsonl (+ meta.json) from scripts/predict.py. The first run is the reference for the paired
bootstrap (95% CI of the per-item accuracy difference; "win"/"loss" only when the interval excludes 0). Sets whose items all
have the same two answers also get AUROC, which measures the ranking independently of the decision threshold.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from decision_model.evaluate import load_preds, load_set, per_item, prob_vector
from decision_model.metrics import auroc

ROOT = Path(__file__).resolve().parents[1]


def boot(diff, n=2000):
    idx = np.random.default_rng(0).integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=prediction_dir")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    P = {k: load_preds(Path(v) / "predictions.jsonl") for k, v in runs.items()}
    meta = {k: json.loads((Path(v) / "meta.json").read_text()) for k, v in runs.items() if (Path(v) / "meta.json").exists()}
    ref = next(iter(P))
    man = json.loads((ROOT / "data/eval/manifest.json").read_text())["sets"]
    sets = {}
    for name in man:
        items = list(load_set(name))
        c = {k: np.array(per_item(items, p)["correct"], float) for k, p in P.items()}
        row = {k: round(float(v.mean()), 4) for k, v in c.items()}
        for k in P:
            if k != ref:
                d, lo, hi = boot(c[k] - c[ref])
                row[f"{k}_minus_{ref}"] = {"diff": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                                           "verdict": "win" if lo > 0 else ("loss" if hi < 0 else "tie")}
        keys = {tuple(it.criteria) for it in items}
        if len(keys) == 1 and len(next(iter(keys))) == 2:
            k0 = next(iter(keys))[0]
            y = np.array([it.gold == k0 for it in items], int)
            row["auroc"] = {k: round(auroc(y, np.array([prob_vector(it, p.get(it.id))[0] for it in items])), 4) for k, p in P.items()}
        sets[name] = row
    mean = {k: round(float(np.mean([sets[n][k] for n in man])), 4) for k in P}
    lat = {k: round(float(np.mean([t["ms_per_item"] for t in m["timing"].values()])), 1) for k, m in meta.items()}
    print(f"{'set':22s}" + "".join(f"{k:>16s}" for k in P))
    for n in man:
        print(f"{n:22s}" + "".join(f"{sets[n][k]:16.3f}" for k in P))
    print(f"{'MEAN':22s}" + "".join(f"{mean[k]:16.3f}" for k in P))
    print("ms per item:", lat)
    if a.out:
        Path(a.out).write_text(json.dumps({"sets": sets, "mean_accuracy": mean, "ms_per_item": lat}, indent=1))


if __name__ == "__main__":
    main()
