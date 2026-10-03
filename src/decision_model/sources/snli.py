"""SNLI, ORIGINAL 1.0 release (nlp.stanford.edu zip): captionID, pairID and per-annotator labels.

Soft targets (annotator distribution) wherever a pair has >= 2 annotator labels (all dev/test, ~39k train);
single-label train pairs are one-hot. Premises from the VisualGenome pilot collection (captionID 'vg_*') are dropped.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from collections import Counter

from decision_model.sources._common import (
    NLI_LABELS,
    RAW_DIR,
    Report,
    UnionFind,
    assign_splits,
    checked,
    clean,
    hf_parquet,
    nli_items,
    norm_text,
)

NAME = "snli"
ORIG_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
ORIG_ZIP = RAW_DIR / "snli" / "snli_1.0.zip"
ORIG_SHA256 = "afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e"
HF_REPO = "stanfordnlp/snli"
HF_REVISION = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"   # cross-check only (texts + gold match row by row)
LICENSE = "CC-BY-SA-4.0"
GROUP_DEF = ("connected component over {Flickr30k image id (captionID before '#'), normalized premise text}. "
             "Coarser than the caption: the 5 captions of one image describe the same scene, so they share a "
             "group; every hypothesis of a premise stays in its premise's group.")
SPLITS = {"train": "train", "dev": "validation", "test": "test"}   # original -> HF name


def _orig(split: str) -> list[dict]:
    if not ORIG_ZIP.exists():
        ORIG_ZIP.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSfL", "-o", str(ORIG_ZIP), ORIG_URL], check=True)
    h = hashlib.sha256()
    with open(ORIG_ZIP, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == ORIG_SHA256, "snli_1.0.zip checksum does not match the official file"
    with zipfile.ZipFile(ORIG_ZIP) as z, z.open(f"snli_1.0/snli_1.0_{split}.jsonl") as f:
        return [json.loads(l) for l in f]


def _load(rep: Report) -> dict:
    out = {}
    for split, hf_split in SPLITS.items():
        orig = _orig(split)
        hf = hf_parquet(HF_REPO, f"plain_text/{hf_split}-00000-of-00001.parquet", HF_REVISION)
        assert len(hf) == len(orig), (split, len(hf), len(orig))
        rows, n_vg = [], 0
        for o, hp, hh, hl in zip(orig, hf.premise, hf.hypothesis, hf.label):
            assert o["sentence1"] == hp and o["sentence2"] == hh, (split, o["pairID"])
            assert (o["gold_label"] == "-") == (hl == -1), (split, o["pairID"])
            if o["captionID"].startswith("vg_"):
                n_vg += 1
                rep.drop(f"{split}:visualgenome_premise")
                continue
            if o["gold_label"] not in NLI_LABELS:
                rep.drop(f"{split}:gold_label_-_no_majority")
                continue
            p, h = clean(o["sentence1"]), clean(o["sentence2"])
            if not p or not h:
                rep.drop(f"{split}:empty_text")
                continue
            anns = [a for a in o["annotator_labels"] if a in NLI_LABELS]
            rows.append({"id": o["pairID"], "premise": p, "hypothesis": h, "label": o["gold_label"],
                         "image": o["captionID"].split("#")[0], "caption_id": o["captionID"], "anns": anns,
                         "native_split": split})
        rep.stats[f"visualgenome_pairs_dropped[{split}]"] = n_vg
        out[split] = rows
    return out


def _group(splits: dict):
    uf = UnionFind()
    for rows in splits.values():
        for r in rows:
            uf.union(("img", r["image"]), ("prem", norm_text(r["premise"])))
    for rows in splits.values():
        for r in rows:
            r["group"] = uf.gid(("img", r["image"]), "snli-g")


def _with_targets(rows: list[dict], rep: Report) -> list[dict]:
    out = []
    for r in rows:
        meta = {"native_split": r["native_split"], "caption_id": r["caption_id"], "n_annotators": len(r["anns"])}
        if len(r["anns"]) >= 2:
            c = Counter(r["anns"])
            tgt = {k: v / len(r["anns"]) for k, v in c.items()}
            if tgt.get(r["label"], 0) != max(tgt.values()):
                rep.drop(f"{r['native_split']}:gold_not_annotator_argmax")
                continue
            out.append({**r, "label_source": "human_multi", "target": tgt, "meta": meta})
        else:
            out.append({**r, "label_source": "human", "target": None, "meta": meta})
    return out


def build():
    rep = Report(NAME, repo="nlp.stanford.edu SNLI 1.0 (original release)", revision=f"sha256:{ORIG_SHA256}",
                 url=ORIG_URL, hf_crosscheck={"repo": HF_REPO, "revision": HF_REVISION}, license=LICENSE,
                 label_source="human_multi (>=2 annotator labels, soft target) / human (single author label)",
                 group_definition=GROUP_DEF, pool_native_splits=["train", "dev"], native_test_split="test")
    splits = _load(rep)
    _group(splits)
    ids = Counter(r["id"] for rows in splits.values() for r in rows)
    assert max(ids.values()) == 1, "pairID not unique"
    pool_rows = _with_targets(splits["train"] + splits["dev"], rep)
    test_rows = _with_targets(splits["test"], rep)
    pool = assign_splits(checked(nli_items(pool_rows, NAME, LICENSE, "train", rep), rep), NAME)
    test = checked(nli_items(test_rows, NAME, LICENSE, "eval", rep), rep)
    rep.note("VisualGenome premises identified by captionID prefix 'vg_' (SNLI README: pilot collection from "
             "VisualGenome); all their pairs are dropped. Counts in stats.visualgenome_pairs_dropped.")
    rep.note("3-way pair items: target = empirical annotator distribution when >= 2 labels, else one-hot. "
             "Blank annotator labels are ignored. Per-premise choice items are one-hot.")
    rep.note("HF stanfordnlp/snli is used only as a row-by-row cross-check (texts and no-majority flags identical).")
    return pool, test, rep.to_dict()
