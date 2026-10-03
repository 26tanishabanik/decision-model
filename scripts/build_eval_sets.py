"""Build the frozen test sets: data/eval/<name>.jsonl + manifest.json (revisions, counts, sha256, notes).

    python scripts/build_eval_sets.py [--sets a,b]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json

from decision_model import eval_sets as E
from decision_model.loaders import SCIFACT_TARBALL_SHA256, SCIFACT_URL
from decision_model.schema import EVAL_DIR, write_jsonl


def source_info(key):
    if key == "scifact":
        return {"dataset": f"SciFact release tarball ({SCIFACT_URL})",
                "revision": f"sha256:{SCIFACT_TARBALL_SHA256}", "config": None}
    repo, rev, cfg = E.REV[key]
    return {"dataset": repo, "revision": rev, "config": cfg}


def describe(items):
    n_opts = [len(it.criteria) for it in items]
    gold_share = collections.Counter(it.gold for it in items)
    roles = collections.Counter(r for it in items for r in it.roles.values())
    soft = sum(max(it.target.values()) < 1 for it in items)
    return {"count": len(items), "qtype": dict(collections.Counter(it.qtype for it in items)),
            "label_source": sorted({it.label_source for it in items}), "license": sorted({it.license for it in items}),
            "n_options": {"min": min(n_opts), "max": max(n_opts), "mean": round(sum(n_opts) / len(n_opts), 2)},
            "n_groups": len({it.group for it in items}), "n_soft_targets": soft,
            "gold_top5": gold_share.most_common(5), "roles": dict(roles)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(E.SETS))
    args = ap.parse_args()
    man_path = EVAL_DIR / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"sets": {}}
    all_ids = set()
    for name in args.sets.split(","):
        fn, src, notes = E.SETS[name]
        items = fn()
        ids = [it.id for it in items]
        assert len(set(ids)) == len(ids), f"duplicate ids in {name}"
        assert all(it.split == "eval" for it in items)
        dup = all_ids & set(ids)
        assert not dup, f"ids shared across sets: {list(dup)[:3]}"
        all_ids |= set(ids)
        w = write_jsonl(items, EVAL_DIR / f"{name}.jsonl")
        entry = {"file": w["path"], "sha256": w["sha256"], **source_info(src), **describe(items), "notes": notes}
        extra = {k: getattr(fn, k) for k in ("notes", "n_skip", "n_label_mismatch") if hasattr(fn, k)}
        if extra:
            entry["build_stats"] = extra
        if name == "bfcl_select":
            entry["skipped_categories"] = E.BFCL_SKIPPED
        manifest["sets"][name] = entry
        print(f"{name:22s} n={len(items):6d} qtype={entry['qtype']} opts={entry['n_options']} {w['sha256'][:12]}")
    manifest["built"] = datetime.date.today().isoformat()
    manifest["seed"] = E.SEED
    manifest["wordings"] = {k: getattr(E, k) for k in dir(E) if k.isupper() and k.endswith(("_Q", "_CRIT", "_TEXT", "_KEY"))}
    man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    print("wrote", man_path)


if __name__ == "__main__":
    main()
