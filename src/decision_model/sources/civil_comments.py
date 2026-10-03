"""Civil Comments (Jigsaw Unintended Bias release, HF google/civil_comments @ pinned revision): yes/no toxicity.

`yes_no` items with SOFT targets from the rater fraction `toxicity` (share of crowd raters who marked the comment toxic
or very toxic): target = {"true": t, "false": 1 - t}, gold = "true" iff t >= 0.5. Natural class balance is kept.
Question wording is our own (3 paraphrases, chosen by a stable hash of the id) and deliberately differs from the
frozen toxic-chat eval wording.
"""
from __future__ import annotations

import hashlib

from decision_model.schema import Item
from decision_model.sources._common import (
    Report,
    assign_splits,
    checked,
    clean,
    drop_conflicting_duplicates,
    hf_parquet,
    norm_text,
    stable_rng,
    text_hash,
)

NAME = "civil_comments"
HF_REPO = "google/civil_comments"
HF_REVISION = "f2970eb3a55777454c94069077cc8d9b5866312d"
LICENSE = "CC0-1.0"
FILES = {"train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
         "test": ["data/test-00000-of-00001.parquet"]}
N_KEEP = {"train": 200_000, "test": 20_000}
MAX_WORDS = 3000
GROUP_DEF = "normalized text hash (lowercase, punctuation stripped, whitespace collapsed)"

TEMPLATES = (
    "Is this comment toxic (rude, disrespectful, or hateful)?",
    "Would a reasonable reader find this comment toxic, for example insulting, hateful, or abusive?",
    "Does this comment contain toxic language such as insults, threats, or hate?",
)
CRITERIA = {"false": "The comment is civil; it is not rude, insulting, hateful, or abusive.",
            "true": "The comment is toxic: rude, insulting, hateful, or abusive."}


def template_for(item_id: str) -> str:
    return TEMPLATES[stable_rng(f"{NAME}|template|{item_id}").randrange(len(TEMPLATES))]


def target_of(t: float) -> tuple[dict, str]:
    """Soft target from the rater fraction; zero-mass keys are dropped."""
    t = float(t)
    assert 0.0 <= t <= 1.0, t
    tgt = {k: v for k, v in (("true", t), ("false", 1.0 - t)) if v > 0}
    return tgt, ("true" if t >= 0.5 else "false")


def make_item(row: dict, split: str) -> Item:
    """row: id, text, toxicity, group, native_split."""
    tgt, gold = target_of(row["toxicity"])
    iid = f"{NAME}:{row['id']}"
    return Item(id=iid, source=NAME, split=split, label_source="human_multi", license=LICENSE, group=row["group"],
                state=row["text"], qtype="yes_no", instructions=template_for(iid), criteria=dict(CRITERIA),
                target=tgt, gold=gold, roles={},
                meta={"format": "toxicity_yes_no", "toxicity_fraction": round(float(row["toxicity"]), 6),
                      "native_split": row["native_split"]})


def _sort_key(text: str) -> str:
    return hashlib.sha256(f"{NAME}|keep|{norm_text(text)}".encode()).hexdigest()


def _rows(native: str, rep: Report) -> list[dict]:
    rows = []
    for fname in FILES[native]:
        df = hf_parquet(HF_REPO, fname, HF_REVISION)
        for i, (txt, tox) in enumerate(zip(df.text, df.toxicity)):
            t = clean(txt)
            if not t:
                rep.drop(f"{native}:empty_text")
                continue
            if len(t.split()) > MAX_WORDS:
                rep.drop(f"{native}:over_{MAX_WORDS}_words")
                continue
            rows.append({"id": f"{native}-{fname.split('/')[-1].split('.')[0]}-{i}", "text": t,
                         "toxicity": float(tox), "native_split": native})
    rep.stats[f"rows_native[{native}]"] = len(rows)
    # deterministic subsample: smallest stable hash of the normalized text (identical texts kept or dropped together)
    rows.sort(key=lambda r: (_sort_key(r["text"]), r["id"]))
    keep = rows[: N_KEEP[native]]
    last = _sort_key(keep[-1]["text"])
    keep += [r for r in rows[N_KEEP[native]:] if _sort_key(r["text"]) == last]   # never split a duplicate set
    for r in keep:
        r["group"] = f"cc-g:{text_hash(r['text'])}"
        r["label"] = "true" if r["toxicity"] >= 0.5 else "false"
    rep.stats[f"rows_kept[{native}]"] = len(keep)
    return drop_conflicting_duplicates(keep, rep, native)


def build():
    rep = Report(NAME, repo=HF_REPO, revision=HF_REVISION, files=FILES, license=LICENSE,
                 label_source="human_multi (toxicity = fraction of crowd raters marking toxic / very toxic)",
                 group_definition=GROUP_DEF, pool_native_splits=["train"], native_test_split="test",
                 subsample=dict(N_KEEP), templates=list(TEMPLATES), criteria=CRITERIA)
    out = {}
    for native in ("train", "test"):
        rows = _rows(native, rep)
        split = "train" if native == "train" else "eval"
        items = checked([make_item(r, split) for r in rows], rep)
        n = len(items)
        rep.stats[f"items[{native}]"] = n
        rep.stats[f"toxic_rate_gold_true[{native}]"] = round(sum(it.gold == "true" for it in items) / n, 4)
        rep.stats[f"exact_tie_t_0.5[{native}]"] = sum(it.meta["toxicity_fraction"] == 0.5 for it in items)
        rep.stats[f"soft_targets[{native}]"] = sum(len(it.target) > 1 for it in items)
        out[native] = items
    pool = assign_splits(out["train"], NAME)
    rep.note("Native validation split (97k) is unused. Pool = deterministic 200k subsample of native train "
             "(smallest sha256 of normalized text); native test -> 20k subsample the same way.")
    rep.note("Identical normalized texts whose gold (t >= 0.5) disagrees are dropped; agreeing duplicates share a group.")
    rep.note("t == 0.5 exactly -> gold 'true' with a 50/50 soft target (count in stats).")
    return pool, out["test"], rep.to_dict()
