"""VitaminC (tals/vitaminc): contrastive claim verification over Wikipedia revisions."""
from __future__ import annotations

from collections import defaultdict

from decision_model.schema import Item
from decision_model.sources._common import (
    CV_CRITERIA,
    CV_KEYS,
    Report,
    UnionFind,
    assign_splits,
    checked,
    cv_instructions,
    hf_jsonl,
    norm_text,
    one_hot,
    shuffled_options,
    unptb,
)

NAME = "vitaminc"
REPO = "tals/vitaminc"
REVISION = "be6febb761b0b2807687e61e0b5282e459df2fa0"
LICENSE = "CC-BY-SA-3.0"
GROUP_DEF = ("connected component over {wiki page, normalized claim text}, computed over all native splits. "
             "case_id is nested in page (verified: 0 cases span >1 page), so every contrastive sibling set stays in "
             "one group; the claim link merges pages that share an identical claim. Evidence text is NOT linked: "
             "generic sentences ('The film received mixed reviews from critics', on 31 pages) would chain "
             "unrelated pages into one 17k-item component.")
CONTRAST_Q = "Which claim does the evidence support?"


def _rows(rep: Report):
    out = {}
    for split in ("train", "dev", "test"):
        rows = hf_jsonl(REPO, f"{split}.jsonl", REVISION)
        kept = []
        for r in rows:
            r["claim"], r["evidence"] = unptb(r["claim"]), unptb(r["evidence"])
            if r["label"] not in CV_KEYS:
                rep.drop(f"{split}:unknown_label")
                continue
            if not r["claim"] or not r["evidence"]:
                rep.drop(f"{split}:empty_claim_or_evidence")
                continue
            kept.append(r)
        out[split] = kept
    return out


def _groups(splits: dict) -> dict:
    uf = UnionFind()
    for rows in splits.values():
        for r in rows:
            p = ("page", r["page"])
            uf.union(p, ("claim", norm_text(r["claim"])))
    return {r["unique_id"]: uf.gid(('page', r['page']), 'vitaminc-g') for rows in splits.values() for r in rows}


def _items(rows: list[dict], gid: dict, split: str, rep: Report) -> list[Item]:
    items = []
    for r in rows:
        key = CV_KEYS[r["label"]]
        items.append(Item(
            id=f"{NAME}:{r['unique_id']}", source=NAME, split=split, label_source="human", license=LICENSE,
            group=gid[r["unique_id"]], state=r["evidence"], qtype="choice", instructions=cv_instructions(r["claim"]),
            criteria=dict(CV_CRITERIA), target=one_hot(key), gold=key, roles={},
            meta={"format": "claim_verification", "case_id": r["case_id"], "page": r["page"],
                  "revision_type": r["revision_type"], "fever_id": r["FEVER_id"] or None,
                  "native_split": r["_split"]}))

    # (b) contrastive: same case, same evidence, >=1 SUPPORTS and >=1 REFUTES claim
    by_ce = defaultdict(list)
    for r in rows:
        by_ce[(r["case_id"], r["evidence"])].append(r)
    for (case_id, evidence), rs in by_ce.items():
        sup = [r for r in rs if r["label"] == "SUPPORTS"]
        ref = [r for r in rs if r["label"] == "REFUTES"]
        nei = [r for r in rs if r["label"] == "NOT ENOUGH INFO"]
        if not sup or not ref:
            continue
        if len(sup) > 1:
            rep.stats[f"contrast_sets_with_multiple_supported[{split}]"] = \
                rep.stats.get(f"contrast_sets_with_multiple_supported[{split}]", 0) + 1
        for s in sup:
            cands = [s] + ref + nei
            texts = [c["claim"] for c in cands]
            if len({norm_text(t) for t in texts}) < len(texts):
                rep.drop("contrast:duplicate_claim_text")
                continue
            perm = shuffled_options(f"{NAME}:contrast:{s['unique_id']}", texts)
            crit, roles, gold = {}, {}, None
            for k, orig in perm:
                c = cands[orig]
                crit[k] = c["claim"]
                roles[k] = {"SUPPORTS": "correct", "REFUTES": "hard", "NOT ENOUGH INFO": "in_question"}[c["label"]]
                if orig == 0:
                    gold = k
            items.append(Item(
                id=f"{NAME}:contrast:{s['unique_id']}", source=NAME, split=split, label_source="human", license=LICENSE,
                group=gid[s["unique_id"]], state=evidence, qtype="choice", instructions=CONTRAST_Q, criteria=crit,
                target=one_hot(gold), gold=gold, roles=roles,
                meta={"format": "contrastive_claims", "case_id": case_id, "page": s["page"],
                      "n_options": len(crit), "option_ids": {k: cands[o]["unique_id"] for k, o in perm},
                      "native_split": s["_split"]}))
    return checked(items, rep)


def build():
    rep = Report(NAME, repo=REPO, revision=REVISION, license=LICENSE, label_source="human", group_definition=GROUP_DEF,
                 pool_native_splits=["train", "dev"], native_test_split="test",
                 contrastive_instructions=CONTRAST_Q)
    splits = _rows(rep)
    for s, rows in splits.items():
        for r in rows:
            r["_split"] = s
    gid = _groups(splits)
    pool = assign_splits(_items(splits["train"] + splits["dev"], gid, "train", rep), NAME)
    test = _items(splits["test"], gid, "eval", rep)
    rep.note("PTB bracket tokens (-LRB- etc.) in claims/evidence are unescaped to plain brackets.")
    rep.note("revision_type='synthetic' rows are human-written edits of FEVER claims/evidence (not LLM text); "
             "meta.fever_id links them to FEVER.")
    rep.note("Contrastive items: per (case_id, evidence) holding >=1 SUPPORTS and >=1 REFUTES claim; one item per "
             "supported claim; options = that claim + all refuted (hard) + NEI (in_question) claims of that "
             "evidence; option order deterministically shuffled.")
    return pool, test, rep.to_dict()
