"""LocalLLaMA/typed-decisions: synthetic multi-question decision states (four workflows) with LLM-teacher soft targets.

Yes/no questions use the dataset's own answer descriptions; the few that ship none get YES_NO_FALLBACK (the frozen
test set uses the same rule).
"""
from __future__ import annotations

import json

from decision_model.schema import Item
from decision_model.sources._common import Report, assign_splits, checked, hf_parquet

NAME = "typed_decisions"
REPO = "LocalLLaMA/typed-decisions"
REVISION = "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"
LICENSE = "Apache-2.0"
WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")
GROUP_DEF = "row id (one state; all of its questions stay together)"
YES_NO_FALLBACK = {"false": "The statement is false.", "true": "The statement is true."}


def question_type(q: dict, gold: dict) -> str | None:
    """choice / score as published; any other question whose answers are only true / false is a yes/no question."""
    if q["type"] in ("choice", "score"):
        return q["type"]
    keys = {str(k) for k in gold["probabilities"]} | {str(gold["label"])}
    return "yes_no" if keys <= {"true", "false"} else None


def _text(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _items(df, split: str, rep: Report) -> list[Item]:
    out = []
    for r in df.itertuples(index=False):
        state = r.state if isinstance(r.state, str) else _text(r.state)
        if not state.strip():
            rep.drop("empty_state")
            continue
        qs, gold = json.loads(r.questions), json.loads(r.gold)
        agree = json.loads(r.label_agreement) if isinstance(r.label_agreement, str) else {}
        for qid, q in qs.items():
            if qid not in gold:
                rep.drop("question_without_gold")
                continue
            g = gold[qid]
            qt = question_type(q, g)
            meta = {"workflow": r.workflow, "question": qid, "row_id": r.id, "teacher_label": g["label"],
                    "native_split": r.split, "label_agreement": agree.get(qid)}
            if qt == "choice":
                crit = {str(k): _text(v) for k, v in q["criteria"].items()}
            elif qt == "yes_no":
                crit = {str(k): _text(v) for k, v in q["criteria"].items()} if q.get("criteria") else dict(YES_NO_FALLBACK)
            elif qt == "score":
                c = q["criteria"]
                crit = {str(i): _text(v) for i, v in enumerate(c)} if isinstance(c, list) else \
                    {str(k): _text(v) for k, v in sorted(c.items(), key=lambda kv: int(kv[0]))}
            else:
                rep.drop(f"unknown_question_type:{q['type']}")
                continue
            p = {str(k): float(v) for k, v in g["probabilities"].items() if float(v) > 0}
            if not p or set(p) - set(crit):
                rep.drop(f"{qt}:probabilities_outside_criteria")
                continue
            z = sum(p.values())
            tgt = {k: v / z for k, v in p.items()}
            m = max(tgt.values())
            lab = str(g["label"])
            gk = lab if tgt.get(lab, 0) == m else max(tgt, key=tgt.get)
            if gk != lab:
                rep.stats["label_not_argmax"] = rep.stats.get("label_not_argmax", 0) + 1
            if abs(z - 1) > 1e-6:
                rep.stats["renormalised_targets"] = rep.stats.get("renormalised_targets", 0) + 1
            meta["n_options"] = len(crit)
            out.append(Item(
                id=f"{NAME}:{r.id}:{qid}", source=NAME, split=split, label_source="llm", license=LICENSE,
                group=f"td-g:{r.id}", state=state, qtype=qt, instructions=q["instructions"].strip(),
                criteria=crit, target=tgt, gold=gk, roles={}, meta=meta))
    return checked(out, rep)


def build():
    rep = Report(NAME, repo=REPO, revision=REVISION, license=LICENSE, label_source="llm", group_definition=GROUP_DEF,
                 pool_native_splits=["train"], native_test_split="test", configs=list(WORKFLOWS))
    pool, test = [], []
    for w in WORKFLOWS:
        pool += _items(hf_parquet(REPO, f"{w}/train-00000-of-00001.parquet", REVISION), "train", rep)
        test += _items(hf_parquet(REPO, f"{w}/test-00000-of-00001.parquet", REVISION), "eval", rep)
    assign_splits(pool, NAME)
    rep.note("The 'all' config is the exact union of the 4 workflow configs (same 1600 ids) and is not read "
             "separately.")
    rep.note("Targets = teacher gold probabilities renormalised to sum 1 (released values are rounded to 6 d.p.); "
             "gold = released label, one of the highest-probability answers (it breaks ties).")
    rep.note("Yes/no questions keep the dataset's answer descriptions; questions that ship none use YES_NO_FALLBACK.")
    return pool, test, rep.to_dict()
