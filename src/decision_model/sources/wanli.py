"""WANLI (alisawuffles/WANLI): GPT-3-written NLI pairs, labelled (and optionally revised) by humans."""
from __future__ import annotations

from collections import defaultdict

from decision_model.sources._common import (
    NLI_LABELS,
    Report,
    assign_splits,
    checked,
    clean,
    hf_jsonl,
    nli_items,
    text_hash,
)

NAME = "wanli"
REPO = "alisawuffles/WANLI"
REVISION = "61c95318fd71c55b6ba355d76253254615f387ec"
LICENSE = "CC-BY-4.0"
NOTE = "AI-generated text, human labels"
GROUP_DEF = ("normalized premise text (all hypotheses of a premise stay together). pairID is NOT a leak unit: "
             "it is the MNLI seed example id used for generation and is shared by unrelated generations.")


def build():
    rep = Report(NAME, repo=REPO, revision=REVISION, license=LICENSE, label_source="human", group_definition=GROUP_DEF,
                 pool_native_splits=["train"], native_test_split="test", flag=NOTE)
    ann = defaultdict(list)
    for a in hf_jsonl(REPO, "anonymized_annotations.jsonl", REVISION):
        ann[a["id"]].append(a["gold"])
    data = {s: hf_jsonl(REPO, f"{s}.jsonl", REVISION) for s in ("train", "test")}
    out = {}
    for s, rows in data.items():
        kept = []
        for r in rows:
            if r["gold"] not in NLI_LABELS:
                rep.drop(f"{s}:label_not_nli")
                continue
            p, h = clean(r["premise"]), clean(r["hypothesis"])
            if not p or not h:
                rep.drop(f"{s}:empty_text")
                continue
            # Soft target from votes: the final gold plus each worker's label (disagreement is kept, not discarded).
            votes = [r["gold"]] + [g for g in ann.get(r["id"], []) if g in NLI_LABELS]
            tgt = {k: votes.count(k) / len(votes) for k in NLI_LABELS if k in votes}
            kept.append({"id": r["id"], "premise": p, "hypothesis": h, "label": r["gold"], "target": tgt,
                         "label_source": "human_multi" if len(votes) >= 3 else "human",
                         "group": f"wanli-g:{text_hash(p)}",
                         "meta": {"note": NOTE, "genre": r["genre"], "seed_mnli_pair_id": r["pairID"],
                                  "annotator_golds": ann.get(r["id"], []), "native_split": s}})
        out[s] = kept
    pool = assign_splits(checked(nli_items(out["train"], NAME, LICENSE, "train", rep), rep), NAME)
    test = checked(nli_items(out["test"], NAME, LICENSE, "eval", rep), rep)
    for it in pool + test:
        it.meta.setdefault("note", NOTE)
    rep.note("Every item carries meta.note = 'AI-generated text, human labels'. meta.annotator_golds holds the two "
             "per-worker gold labels from anonymized_annotations.jsonl; the target is the vote distribution over the final "
             "gold and both workers' labels.")
    return pool, test, rep.to_dict()

