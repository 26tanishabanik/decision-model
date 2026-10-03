"""CLINC150 (clinc/clinc_oos, config 'plus'): 150 intents + out-of-scope."""
from __future__ import annotations

from decision_model.sources._common import (
    Report,
    assign_splits,
    checked,
    clean,
    drop_conflicting_duplicates,
    hf_parquet,
    intent_items,
    parquet_label_names,
    readable,
    text_hash,
)

NAME = "clinc_oos"
REPO = "clinc/clinc_oos"
REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
CONFIG = "plus"
LICENSE = "CC-BY-3.0"
GROUP_DEF = "normalized text hash (lowercase, punctuation stripped, whitespace collapsed)"
OOS_KEY, OOS_TEXT = "out_of_scope", "none of the listed intents (out of scope)"


def build():
    rep = Report(NAME, repo=REPO, revision=REVISION, config=CONFIG, license=LICENSE, label_source="human",
                 group_definition=GROUP_DEF, pool_native_splits=["train", "validation"], native_test_split="test")
    names = parquet_label_names(REPO, f"{CONFIG}/train-00000-of-00001.parquet", REVISION, "intent")
    assert len(names) == 151 and "oos" in names
    key = {i: (OOS_KEY if n == "oos" else n) for i, n in enumerate(names)}
    criteria = {n: readable(n) for n in sorted(n for n in names if n != "oos")}
    criteria[OOS_KEY] = OOS_TEXT
    out = {}
    for s in ("train", "validation", "test"):
        df = hf_parquet(REPO, f"{CONFIG}/{s}-00000-of-00001.parquet", REVISION)
        kept = []
        for i, (t, lab) in enumerate(zip(df.text, df.intent)):
            t = clean(t)
            if not t:
                rep.drop(f"{s}:empty_text")
                continue
            k = key[int(lab)]
            kept.append({"id": f"{s}-{i}", "text": t, "label": k, "group": f"clinc-g:{text_hash(t)}",
                         "meta": {"oos": k == OOS_KEY, "native_split": s}})
        out[s] = kept
    pool_rows = drop_conflicting_duplicates(out["train"] + out["validation"], rep, "pool")
    test_rows = drop_conflicting_duplicates(out["test"], rep, "test")
    pool = assign_splits(checked(intent_items(pool_rows, NAME, LICENSE, "train", criteria), rep), NAME)
    test = checked(intent_items(test_rows, NAME, LICENSE, "eval", criteria), rep)
    rep.stats["oos_pool"] = sum(r["label"] == OOS_KEY for r in pool_rows)
    rep.stats["oos_test"] = sum(r["label"] == OOS_KEY for r in test_rows)
    rep.note("151 options: 150 intents alphabetical + out_of_scope last. OOS items kept with gold = out_of_scope "
             "(meta.oos).")
    return pool, test, rep.to_dict()
