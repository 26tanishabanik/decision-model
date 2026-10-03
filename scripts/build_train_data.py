"""Build every training source from its pinned upstream release.

    python scripts/build_train_data.py [source ...] [--dry-run]

Writes data/sources/<name>.jsonl (grouped train / dev / calib split), data/sources/<name>__native_test.jsonl (the
source's own test split, split = "eval", never used for training) and data/sources/manifest.json.
Next step: scripts/gate_sources.py.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import sys
import time
from collections import Counter, defaultdict

from decision_model.schema import DATA_DIR, ROOT, SOURCES_DIR, validate, write_jsonl
from decision_model.sources import DATASETS

SOURCES = list(DATASETS)
MANIFEST = SOURCES_DIR / "manifest.json"


def license_entry(dataset: str) -> dict:
    for e in json.loads((DATA_DIR / "licenses.json").read_text())["sources"]:
        if e["dataset"] == dataset:
            return e
    raise KeyError(dataset)


def _counts(items) -> dict:
    by = lambda f: dict(sorted(Counter(f(it) for it in items).items()))
    nopt = [len(it.criteria) for it in items]
    gold_pos = Counter()
    for it in items:
        if it.qtype == "choice" and len(it.criteria) <= 16 and \
                it.meta.get("format") not in ("claim_verification", "nli_pair"):
            gold_pos[list(it.criteria).index(it.gold)] += 1
    gsize = Counter(it.group for it in items)
    return {
        "n": len(items), "split": by(lambda it: it.split), "qtype": by(lambda it: it.qtype),
        "label_source": by(lambda it: it.label_source), "format": by(lambda it: it.meta.get("format", it.qtype)),
        "gold_top": Counter(it.gold for it in items).most_common(8),
        "n_options": {"min": min(nopt), "max": max(nopt), "mean": round(sum(nopt) / len(nopt), 2)} if nopt else {},
        "gold_position_choice_le16_options": dict(sorted(gold_pos.items())),
        "n_groups": len(gsize), "max_group_size": max(gsize.values()) if gsize else 0,
        "n_soft_targets": sum(len(it.target) > 1 for it in items),
    }


def _check(name, pool, test) -> dict:
    split_by_group = defaultdict(set)
    for it in pool:
        split_by_group[it.group].add(it.split)
    straddle = [g for g, s in split_by_group.items() if len(s) > 1]
    assert not straddle, f"{name}: {len(straddle)} groups straddle splits"
    assert all(it.split in ("train", "dev", "calib") for it in pool), name
    assert all(it.split == "eval" for it in test), name
    ids = Counter(it.id for it in pool + test)
    dup = [i for i, n in ids.items() if n > 1]
    assert not dup, f"{name}: duplicate ids {dup[:5]}"
    test_groups = {it.group for it in test}
    shared = {g for g in split_by_group if g in test_groups}
    return {"groups_straddling_pool_splits": 0, "duplicate_ids": 0,
            "pool_groups_also_in_native_test": len(shared),
            "pool_items_in_groups_shared_with_native_test": sum(it.group in shared for it in pool)}


def _write(items, path, dry: bool) -> dict:
    """write_jsonl, skipped when the file already holds identical content (or on --dry-run)."""
    text = "\n".join(it.to_json() for it in items) + "\n"
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = str(path.relative_to(ROOT))
    if dry:
        for it in items:
            validate(it)
        return {"path": rel, "n": len(items), "sha256": sha, "written": False}
    if path.exists() and path.stat().st_size == len(text.encode()) and \
            hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        for it in items:
            validate(it)
        return {"path": rel, "n": len(items), "sha256": sha, "written": "unchanged"}
    out = write_jsonl(items, path)
    assert out["sha256"] == sha
    return out


def build_one(name: str, dry: bool = False) -> dict:
    mod = importlib.import_module(f"decision_model.sources.{name}")
    t0 = time.time()
    pool, test, rep = mod.build()
    leak = _check(name, pool, test)
    test_groups = {it.group for it in test}
    for it in pool:
        if it.group in test_groups:
            it.meta["group_in_native_test"] = True
    pool.sort(key=lambda it: it.id)
    test.sort(key=lambda it: it.id)
    files = {"pool": _write(pool, SOURCES_DIR / f"{name}.jsonl", dry),
             "native_test": _write(test, SOURCES_DIR / f"{name}__native_test.jsonl", dry)}
    entry = {**rep, "item_licenses": sorted({it.license for it in pool + test}), "license": license_entry(DATASETS[name]),
             "files": files,
             "pool": _counts(pool), "native_test": _counts(test),
             "leak_checks": leak, "build_seconds": round(time.time() - t0, 1)}
    print(f"[{name}] pool={len(pool)} {entry['pool']['split']} native_test={len(test)} "
          f"qtype={entry['pool']['qtype']} drops={sum(rep['drops'].values())} ({entry['build_seconds']}s)",
          flush=True)
    return entry


def main(argv):
    dry = "--dry-run" in argv
    names = [a for a in argv if not a.startswith("--")] or SOURCES
    unknown = set(names) - set(SOURCES)
    assert not unknown, unknown
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"sources": {}}
    for n in names:
        entry = build_one(n, dry)
        if dry:
            print(json.dumps({k: entry[k] for k in ("pool", "native_test", "drops", "stats", "leak_checks")},
                             default=str)[:3000])
            continue
        manifest["sources"][n] = entry
        _save(manifest)


def _save(manifest):
    manifest.update({
        "built": datetime.date.today().isoformat(),
        "split_rule": "decision_model.schema.split_of(group, salt=<source name>), fractions train 0.90 / dev 0.05 / calib 0.05",
        "native_test_rule": "the source's native test (or validation when test is unlabelled), split='eval'; "
                            "never mixed into training",
        "sources": dict(sorted(manifest["sources"].items())),
    })
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
