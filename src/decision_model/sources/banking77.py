"""Banking77, the original PolyAI release (github PolyAI-LDN/task-specific-datasets, pinned commit): 77-way intent choice."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess

from decision_model.sources._common import (
    RAW_DIR,
    Report,
    assign_splits,
    checked,
    clean,
    drop_conflicting_duplicates,
    intent_items,
    readable,
    text_hash,
)

NAME = "banking77"
REPO = "PolyAI-LDN/task-specific-datasets (github), banking_data/"
REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
HF_REPO, HF_REVISION = "PolyAI/banking77", "90d4e2ee5521c04fc1488f065b8b083658768c57"   # loading script pointing here
LICENSE = "CC-BY-4.0"
RAW = RAW_DIR / "banking77"
FILES = {"train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
         "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
         "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63"}
GROUP_DEF = "normalized text hash (lowercase, punctuation stripped, whitespace collapsed)"


def _fetch(name: str):
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{REVISION}/banking_data/{name}"
        subprocess.run(["curl", "-sSfL", "-o", str(p), url], check=True)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == FILES[name], f"checksum mismatch: {p}"
    return p


def _csv(name: str) -> list[tuple[str, str]]:
    with open(_fetch(name), encoding="utf-8") as f:
        rd = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(rd)
        return [tuple(r) for r in rd]


def build():
    rep = Report(NAME, repo=REPO, revision=REVISION, hf_loader={"repo": HF_REPO, "revision": HF_REVISION},
                 file_sha256=FILES, license=LICENSE, label_source="human", group_definition=GROUP_DEF,
                 pool_native_splits=["train"], native_test_split="test")
    names = json.loads(_fetch("categories.json").read_text())
    assert len(names) == 77 and len(set(names)) == 77
    criteria = {n: readable(n) for n in names}
    out = {}
    for s in ("train", "test"):
        kept = []
        for i, (t, lab) in enumerate(_csv(f"{s}.csv")):
            t = clean(t)
            if not t:
                rep.drop(f"{s}:empty_text")
                continue
            if lab not in criteria:
                rep.drop(f"{s}:unknown_label")
                continue
            kept.append({"id": f"{s}-{i}", "text": t, "label": lab, "group": f"b77-g:{text_hash(t)}"})
        rep.stats[f"rows[{s}]"] = len(kept)
        out[s] = drop_conflicting_duplicates(kept, rep, s)
    pool = assign_splits(checked(intent_items(out["train"], NAME, LICENSE, "train", criteria), rep), NAME)
    test = checked(intent_items(out["test"], NAME, LICENSE, "eval", criteria), rep)
    rep.note("Rows come from the original PolyAI CSVs at a pinned GitHub commit (CC-BY-4.0), the files the "
             "PolyAI/banking77 HF loading script reads.")
    rep.note("Options in categories.json order; keys = native intent names; text = underscores->spaces, first "
             "letter capitalised.")
    return pool, test, rep.to_dict()
