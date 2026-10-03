"""AI2 ARC (allenai/ai2_arc), ARC-Easy + ARC-Challenge: grade-school science multiple choice."""
from __future__ import annotations

from decision_model.sources._common import (
    LETTERS,
    Report,
    assign_splits,
    checked,
    clean,
    dedup_options,
    hf_parquet,
    mc_item,
)

NAME = "ai2_arc"
REPO = "allenai/ai2_arc"
REVISION = "210d026faf9955653af8916fad021475a3f00453"
LICENSE = "CC-BY-SA-4.0"
GROUP_DEF = "question id (ARC has no finer shared-context unit; ids are unique across configs and splits)"
CONFIGS = ("ARC-Easy", "ARC-Challenge")


def _items(df, config, split, native, rep):
    out = []
    for r in df.itertuples(index=False):
        native_labels, texts = list(r.choices["label"]), [clean(t) for t in r.choices["text"]]
        if r.answerKey not in native_labels:
            rep.drop(f"{config}/{native}:no_answer_key")
            continue
        if not clean(r.question):
            rep.drop(f"{config}/{native}:empty_question")
            continue
        if not all(texts):
            rep.drop(f"{config}/{native}:empty_option")
            continue
        native_labels, texts, ok = dedup_options(native_labels, texts, r.answerKey)
        if not ok:
            rep.drop(f"{config}/{native}:gold_text_duplicated_by_distractor")
            continue
        if len(native_labels) < len(r.choices["label"]):
            rep.stats["duplicate_distractors_removed"] = rep.stats.get("duplicate_distractors_removed", 0) + 1
        # some items use 1..4 labels; key everything by letter
        labels = LETTERS[:len(native_labels)]
        ans = labels[native_labels.index(r.answerKey)]
        meta = {"config": config, "native_split": native}
        if native_labels != labels:
            meta["native_labels"] = native_labels
        out.append(mc_item(NAME, LICENSE, split, r.id, f"arc-g:{r.id}", r.question, labels, texts, ans, meta))
    return checked(out, rep)


def build():
    rep = Report(NAME, repo=REPO, revision=REVISION, license=LICENSE, label_source="human", group_definition=GROUP_DEF,
                 pool_native_splits=["train", "validation"], native_test_split="test", configs=list(CONFIGS))
    pool, test = [], []
    for c in CONFIGS:
        for s in ("train", "validation", "test"):
            df = hf_parquet(REPO, f"{c}/{s}-00000-of-00001.parquet", REVISION)
            its = _items(df, c, "eval" if s == "test" else "train", s, rep)
            (test if s == "test" else pool).extend(its)
    assign_splits(pool, NAME)
    rep.note("Numeric answer labels (1-4, ~4% of items) are re-keyed to letters A-D; originals in meta.native_labels.")
    return pool, test, rep.to_dict()
