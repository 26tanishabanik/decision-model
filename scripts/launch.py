"""Start a job on the deployed Modal app from this machine (it keeps running if this machine sleeps).

    python scripts/launch.py train --name lora [--set key=value ...]      # train, then predict on the frozen sets
    python scripts/launch.py queries --final /vol/runs/lora/final.pt --file data/queries/four_agents.json --out results/queries/four_agents/lora.json
Values for --set are parsed as JSON when possible (true/false/numbers), else kept as strings.
"""
import argparse
import json
from pathlib import Path

import modal


def parse_sets(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--set", nargs="*")
    q = sub.add_parser("queries")
    q.add_argument("--final", required=True)
    q.add_argument("--file", required=True)
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        c = modal.Function.from_name("decision-model", "train_then_predict").spawn(parse_sets(a.set), a.name)
        print("started", c.object_id, "-> runs/%s/chain.json on the volume" % a.name)
    else:
        r = modal.Function.from_name("decision-model", "answer_queries").remote(a.final, Path(a.file).read_text())
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(r)
        print("saved", a.out)


if __name__ == "__main__":
    main()
