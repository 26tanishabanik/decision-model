"""MASSIVE (AmazonScience/massive), en-US only: 60-way intent choice."""
from __future__ import annotations

from decision_model.sources._common import (
    Report,
    UnionFind,
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

NAME = "massive"
REPO = "AmazonScience/massive"
MAIN_REVISION = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
REVISION = "ed58ac423a2f4121720918bf5301577edce4ffd3"   # refs/convert/parquet (main is a loading script)
LICENSE = "CC-BY-4.0"
LOCALE = "en-US"
GROUP_DEF = ("connected component over {MASSIVE id (parallel across locales), normalized utterance text}; the text "
             "link merges the few en-US utterances that recur under different ids")


def build():
    rep = Report(NAME, repo=REPO, revision=REVISION, main_revision=MAIN_REVISION, locale=LOCALE, license=LICENSE,
                 label_source="human", group_definition=GROUP_DEF, pool_native_splits=["train", "validation"],
                 native_test_split="test")
    names = parquet_label_names(REPO, f"{LOCALE}/train/0000.parquet", REVISION, "intent")
    scen = parquet_label_names(REPO, f"{LOCALE}/train/0000.parquet", REVISION, "scenario")
    criteria = {n: readable(n) for n in names}
    data = {s: hf_parquet(REPO, f"{LOCALE}/{s}/0000.parquet", REVISION) for s in ("train", "validation", "test")}
    uf = UnionFind()
    for df in data.values():
        assert set(df.locale) == {LOCALE}
        for i, t in zip(df.id, df.utt):
            uf.union(("id", str(i)), ("txt", text_hash(t)))
    out = {}
    for s, df in data.items():
        kept = []
        for r in df.itertuples(index=False):
            t = clean(r.utt)
            if not t:
                rep.drop(f"{s}:empty_text")
                continue
            kept.append({"id": f"{LOCALE}:{r.id}", "text": t, "label": names[int(r.intent)],
                         "group": uf.gid(("id", str(r.id)), "massive-g"),
                         "meta": {"massive_id": str(r.id), "scenario": scen[int(r.scenario)], "native_split": s}})
        out[s] = kept
    pool_rows = drop_conflicting_duplicates(out["train"] + out["validation"], rep, "pool")
    test_rows = drop_conflicting_duplicates(out["test"], rep, "test")
    pool = assign_splits(checked(intent_items(pool_rows, NAME, LICENSE, "train", criteria), rep), NAME)
    test = checked(intent_items(test_rows, NAME, LICENSE, "eval", criteria), rep)
    rep.note(f"{len(names)} intents in label-id order; text = underscores->spaces (e.g. 'Iot hue lightchange').")
    return pool, test, rep.to_dict()
