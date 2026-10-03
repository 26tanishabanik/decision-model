"""Pair loaders for ANLI and SciFact.

Every example: uid, x (premise/evidence), y (hypothesis/claim), label, split, group.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANLI_REVISION = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_TARBALL = ROOT / "data" / "raw" / "scifact" / "data.tar.gz"
SCIFACT_DIR = SCIFACT_TARBALL.parent / "data"
SCIFACT_TARBALL_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"

ANLI_LABELS = ["entailment", "neutral", "contradiction"]
SCIFACT_LABELS = ["SUPPORT", "CONTRADICT"]


@dataclass
class Pair:
    uid: str
    x: str
    y: str
    label: int
    label_name: str
    split: str
    group: str
    meta: dict


def load_anli(splits=None) -> list[Pair]:
    from datasets import load_dataset
    ds = load_dataset("facebook/anli", revision=ANLI_REVISION)
    out = []
    for name in splits or list(ds.keys()):
        part, rnd = name.split("_")
        for r in ds[name]:
            out.append(Pair(r["uid"], r["premise"], r["hypothesis"], r["label"], ANLI_LABELS[r["label"]],
                            name, r["premise"], {"round": rnd, "part": part}))
    return out


def _read_jsonl(p: Path):
    with open(p) as f:
        return [json.loads(l) for l in f]


def fetch_scifact() -> None:
    """Download and unpack the SciFact release tarball once; its sha256 must match SCIFACT_TARBALL_SHA256."""
    if not SCIFACT_TARBALL.exists():
        SCIFACT_TARBALL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SCIFACT_URL, SCIFACT_TARBALL)
    digest = hashlib.sha256(SCIFACT_TARBALL.read_bytes()).hexdigest()
    assert digest == SCIFACT_TARBALL_SHA256, f"SciFact tarball changed upstream: {digest}"
    if not SCIFACT_DIR.exists():
        with tarfile.open(SCIFACT_TARBALL) as tar:
            tar.extractall(SCIFACT_TARBALL.parent, filter="data")


def load_scifact(premise: str = "rationale") -> list[Pair]:
    """One pair per (claim, evidence doc). premise='rationale': union of annotated rationale sentences in
    abstract order; 'abstract': title + full abstract. Label is per (claim, doc) in SciFact.
    Official dev is the final held-out split (test labels are not public)."""
    fetch_scifact()
    corpus = {d["doc_id"]: d for d in _read_jsonl(SCIFACT_DIR / "corpus.jsonl")}
    out = []
    for split in ("train", "dev"):
        for c in _read_jsonl(SCIFACT_DIR / f"claims_{split}.jsonl"):
            for doc_id, sets in c.get("evidence", {}).items():
                labels = {s["label"] for s in sets}
                assert len(labels) == 1, (c["id"], doc_id, labels)
                lab = labels.pop()
                doc = corpus[int(doc_id)]
                sents = sorted({i for s in sets for i in s["sentences"]})
                x = (" ".join(doc["abstract"][i] for i in sents) if premise == "rationale"
                     else doc["title"] + " " + " ".join(doc["abstract"]))
                out.append(Pair(f"sf-{c['id']}-{doc_id}", x, c["claim"], SCIFACT_LABELS.index(lab), lab,
                                split, str(doc_id),
                                {"claim_id": c["id"], "doc_id": int(doc_id), "rationale_idx": sents,
                                 "n_sets": len(sets)}))
    return out
