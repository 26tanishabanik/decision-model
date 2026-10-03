"""Score answers to a query file: every <system>.json in a results directory ({query name: {option: p}}).
Reports scored accuracy overall, per type, per domain, bare vs described yes/no, and every scored miss.

    python scripts/queries_report.py data/queries/four_agents.json results/queries/four_agents
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np


def top(p):
    return max(p, key=p.get)


def main():
    qs = json.load(open(sys.argv[1]))
    d = Path(sys.argv[2])
    P = {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json")) if f.stem != "report"}
    agg = {s: collections.defaultdict(list) for s in P}
    for q in qs:
        if not q["scored"]:
            continue
        bare = q["type"] == "yes_no" and not any(v for v in q["options"].values())
        for s, R in P.items():
            ok = top(R[q["name"]]) == q["expect"]
            keys = ["ALL", f"type:{q['type']}", f"domain:{q['domain']}"]
            if q["type"] == "yes_no":
                keys += ["yes_no:bare" if bare else "yes_no:described", f"yes_no:gold_{q['expect']}"]
                agg[s]["yes_no:says_true"].append(top(R[q["name"]]) == "true")
            if q["type"] == "score":
                agg[s]["score:within_1"].append(abs(int(top(R[q["name"]])) - int(q["expect"])) <= 1)
            for k in keys:
                agg[s][k].append(ok)
    keys = sorted({k for s in agg for k in agg[s]}, key=lambda k: (k != "ALL", k.split(":")[0], k))
    table = {s: {k: (round(float(np.mean(v)), 3), len(v)) for k, v in agg[s].items()} for s in P}
    print(f"{'':28s}" + "".join(f"{s:>17s}" for s in P) + "      n")
    for k in keys:
        n = next(table[s][k][1] for s in P if k in table[s])
        print(f"{k:28s}" + "".join(f"{table[s].get(k, ('-', 0))[0]:>17}" for s in P) + f"{n:7d}")
    print("\nscored misses (expect -> wrong answers):")
    for q in qs:
        if q["scored"]:
            wrong = {s: top(R[q["name"]]) for s, R in P.items() if top(R[q["name"]]) != q["expect"]}
            if wrong:
                print(f"  {q['name']:40s} {q['expect']:18s} " + " ".join(f"{s}={x}" for s, x in wrong.items()))
    (d / "report.json").write_text(json.dumps({"table": table}, indent=1))


if __name__ == "__main__":
    main()
