"""Predictions of a trained model on the frozen test sets, plus an optional jsonl of extra items (e.g. held-out
splits). Output rows: {"id", "probs"}.

    (inside Modal) python scripts/predict.py --final .../final.pt --out dir [--extra f.jsonl]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from decision_model import model as M
from decision_model.evaluate import load_set
from decision_model.schema import EVAL_DIR, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    cfg, bb, head = M.load_final(a.final)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.extra:
        items = list(read_jsonl(Path(a.extra)))
        P = M.predict(bb, head, items)
        with open(out / "extra.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps({"id": it.id, "probs": P[it.id]}) + "\n")
    man = json.loads((EVAL_DIR / "manifest.json").read_text())["sets"]
    timing = {}
    with open(out / "predictions.jsonl", "w") as f:
        for name in man:
            assert hashlib.sha256((EVAL_DIR / f"{name}.jsonl").read_bytes()).hexdigest() == man[name]["sha256"], name
            items = list(load_set(name))
            t0 = time.time()
            P = M.predict(bb, head, items)
            for it in items:
                f.write(json.dumps({"id": it.id, "probs": P[it.id]}) + "\n")
            timing[name] = {"n": len(items), "ms_per_item": round(1000 * (time.time() - t0) / len(items), 1)}
            print(name, timing[name], flush=True)
    (out / "meta.json").write_text(json.dumps({"final": a.final, "config": cfg.__dict__, "timing": timing,
                                               "gpu": torch.cuda.get_device_name(0)}, indent=1, default=str))


if __name__ == "__main__":
    main()
