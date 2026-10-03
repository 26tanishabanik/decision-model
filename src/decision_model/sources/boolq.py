"""BoolQ (Clark et al., NAACL 2019), HF google/boolq at a pinned revision: yes/no questions over Wikipedia paragraphs.

Yes/no items: state = passage, instructions = the question (capitalised, "?"), fixed yes/no criteria, one-hot target.
Native train -> pool (grouped train/dev/calib); native validation -> native test (split = "eval"); the native test split
has no public labels. The released files carry no page title, so groups are passage-level: identical passages and
passages with the same opening sentence share a group.
"""
from __future__ import annotations

from collections import Counter

from decision_model.schema import Item
from decision_model.sources._common import Report, UnionFind, assign_splits, checked, clean, hf_parquet, norm_text

NAME = "boolq"
HF_REPO = "google/boolq"
HF_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
LICENSE = "CC-BY-SA-3.0"
SPLITS = {"train": "train", "validation": "validation"}
INSTRUCTIONS_SUFFIX = "?"
CRITERIA = {"false": "No, the answer is no.", "true": "Yes, the answer is yes."}
GROUP_DEF = ("connected component over {normalized passage, normalized first sentence of the passage (first 120 chars)}. "
             "The release has no page titles, so items cannot be grouped by page.")


def question_text(q: str) -> str:
    q = clean(q).rstrip("?").strip()
    return (q[:1].upper() + q[1:] + INSTRUCTIONS_SUFFIX) if q else ""


def first_sentence(passage: str) -> str:
    return norm_text(passage.split(". ")[0])[:120]


def rows_from_frame(df, native_split: str, rep: Report) -> list[dict]:
    rows = []
    for i, (q, a, p) in enumerate(zip(df.question, df.answer, df.passage)):
        q, p = question_text(q), clean(p)
        if not q or not p:
            rep.drop(f"{native_split}:empty_text")
            continue
        rows.append({"id": f"{native_split}-{i}", "question": q, "answer": bool(a), "passage": p,
                     "native_split": native_split})
    # identical (question, passage) with different answers is a conflict; identical with the same answer is a duplicate
    by = {}
    for r in rows:
        by.setdefault((norm_text(r["question"]), norm_text(r["passage"])), []).append(r)
    out = []
    for group in by.values():
        if len({r["answer"] for r in group}) > 1:
            rep.drop(f"{native_split}:conflicting_duplicate", len(group))
            continue
        out.append(group[0])
        if len(group) > 1:
            rep.drop(f"{native_split}:exact_duplicate", len(group) - 1)
    return out


def assign_groups(rows: list[dict]) -> None:
    uf = UnionFind()
    for r in rows:
        uf.union(("p", norm_text(r["passage"])), ("s", first_sentence(r["passage"])))
    for r in rows:
        r["group"] = uf.gid(("p", norm_text(r["passage"])), "boolq-g")


def items(rows: list[dict], split: str) -> list[Item]:
    return [Item(id=f"{NAME}:{r['id']}", source=NAME, split=split, label_source="human", license=LICENSE, group=r["group"],
                 state=r["passage"], qtype="yes_no", instructions=r["question"], criteria=dict(CRITERIA),
                 target={("true" if r["answer"] else "false"): 1.0}, gold="true" if r["answer"] else "false",
                 meta={"format": "boolq_yes_no", "native_split": r["native_split"]})
            for r in rows]


def build():
    rep = Report(NAME, repo=HF_REPO, revision=HF_REVISION, license=LICENSE, label_source="human", group_definition=GROUP_DEF,
                 pool_native_splits=["train"], native_test_split="validation")
    splits = {}
    for s, hf in SPLITS.items():
        df = hf_parquet(HF_REPO, f"data/{hf}-00000-of-00001.parquet", HF_REVISION)
        rep.stats[f"rows_raw[{s}]"] = len(df)
        splits[s] = rows_from_frame(df, s, rep)
    assign_groups(splits["train"] + splits["validation"])      # one union-find: shared passages share a group id
    pool = assign_splits(checked(items(splits["train"], "train"), rep), NAME)
    test = checked(items(splits["validation"], "eval"), rep)
    pool_groups = {it.group for it in pool}
    shared = sum(it.group in pool_groups for it in test)
    rep.stats["native_test_items_sharing_a_group_with_pool"] = shared
    for name, its in (("pool", pool), ("native_test", test)):
        c = Counter(it.gold for it in its)
        rep.stats[f"yes_rate[{name}]"] = round(c["true"] / max(1, len(its)), 4)
    rep.stats["split_counts"] = dict(Counter(it.split for it in pool))
    rep.note(f"{shared} native-validation items share a passage group with the training pool.")
    rep.note("Yes-rate is kept as released (~62%); no rebalancing.")
    return pool, test, rep.to_dict()
