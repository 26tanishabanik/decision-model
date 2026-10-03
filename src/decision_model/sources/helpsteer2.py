"""HelpSteer2 (nvidia/HelpSteer2, pinned revision): 5-level ordinal `score` questions on an assistant response.

One HF row = (prompt, response, five 0-4 attribute ratings). One `score` item per (row, attribute).
Labels are human (Scale AI annotators, >= 3 per sample); the RESPONSES being judged are LLM-generated text
(NVIDIA in-house models) and most prompts come from ShareGPT. The response is the state under judgement, never a label.

Targets: the released label is the rounded mean of the three most-agreeing annotators. The per-annotator ratings
(disagreements/disagreements.jsonl.gz) give an empirical distribution; it is the SOFT target whenever the released
label is one of its argmax levels, else the item keeps a one-hot target on the released label (counted).
"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter

from decision_model.schema import Item
from decision_model.sources._common import Report, assign_splits, checked, hf_file, norm_text, text_hash

NAME = "helpsteer2"
HF_REPO = "nvidia/HelpSteer2"
HF_REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"
LICENSE = "CC-BY-4.0"
MAX_STATE_WORDS = 3000
GROUP_DEF = ("normalized PROMPT text hash: both responses to a prompt and all five attribute items of each response "
             "share a group")
ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")

# Our own question wording, one per attribute.
QUESTIONS = {
    "helpfulness": "How helpful is the assistant's response to the user's request?",
    "correctness": "How correct and complete is the assistant's response?",
    "coherence": "How clear and self-consistent is the assistant's response?",
    "complexity": "How sophisticated is the language of the assistant's response?",
    "verbosity": "How long and detailed is the assistant's response, relative to what the user asked for?",
}
# Short level descriptions condensed from the HelpSteer2 annotation guidelines
# (Wang et al. 2024, arXiv:2406.08673v1, Appendix G.3.1 "Detailed Rating Breakdown"). Lowest level first.
LEVELS = {
    "helpfulness": [
        "Not helpful at all; it completely misses what the user wanted.",
        "Borderline unhelpful; it mostly misses what the user wanted but is useful in a small way.",
        "Partially helpful; it misses the user's overall goal in some way.",
        "Mostly helpful and aligned with what the user wanted, with some room for improvement.",
        "Extremely helpful and completely aligned with what the user asked for.",
    ],
    "correctness": [
        "Completely incorrect or irrelevant; the requested task is not attempted.",
        "Mostly wrong or incomplete, with several false or misleading statements.",
        "A mix of correct and incorrect information, or clear gaps in the task.",
        "Mostly correct, with a small amount of missing information and no false statements.",
        "Completely correct and complete, with nothing false or missing.",
    ],
    "coherence": [
        "Completely incoherent; no clear meaning can be found.",
        "Mostly hard to follow, with contradictions or confusing logic throughout.",
        "A little unclear, with some inconsistencies or hard-to-follow parts.",
        "Mostly clear and coherent, with one or two confusing places.",
        "Perfectly clear and self-consistent throughout.",
    ],
    "complexity": [
        "Basic language that anyone, including children, can understand.",
        "Simple language that needs some elementary or middle school education.",
        "Intermediate language that someone with a high school education understands.",
        "Advanced vocabulary, as a university student of the subject would write.",
        "Expert language, as a specialist in the field would write.",
    ],
    "verbosity": [
        "Succinct: as short and to the point as possible.",
        "Pretty short, though a little could still be cut.",
        "Average length for what was asked.",
        "Moderately long, on the longer side.",
        "Verbose: particularly lengthy or wordy for what was asked.",
    ],
}
SPLITS = {"train": "train.jsonl.gz", "validation": "validation.jsonl.gz"}
DISAGREEMENTS = "disagreements/disagreements.jsonl.gz"


def keep_lines(s: str) -> str:
    """Whitespace clean-up that keeps line structure (clean() would flatten code and lists)."""
    s = str(s or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for ln in s.split("\n"):
        body = ln.lstrip(" \t")
        indent = ln[: len(ln) - len(body)].replace("\t", "    ")          # keep code indentation
        lines.append((indent + re.sub(r"[ \t\f\v]+", " ", body)).rstrip() if body.strip() else "")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def render_prompt(prompt: str) -> str:
    """Multi-turn prompts use NeMo turn markers; render them as plain role-labelled turns."""
    if "<extra_id_1>" not in prompt:
        return keep_lines(prompt)
    p = re.sub(r"\n?<extra_id_1>Assistant\n?", "\n\nAssistant: ", prompt)
    p = re.sub(r"\n?<extra_id_1>User\n?", "\n\nUser: ", p)
    return keep_lines("User: " + p)


def state_text(prompt: str, response: str) -> str:
    return f"User request:\n{render_prompt(prompt)}\n\nAssistant response:\n{keep_lines(response)}"


def target_for(ratings, released: int):
    """-> (target dict, kind). kind: 'soft' | 'onehot_fallback' | 'onehot_no_ratings'."""
    gold = str(released)
    if not ratings:
        return {gold: 1.0}, "onehot_no_ratings"
    c = Counter(int(v) for v in ratings)
    if c.get(released, 0) != max(c.values()):
        return {gold: 1.0}, "onehot_fallback"
    n = sum(c.values())
    return {str(k): v / n for k, v in sorted(c.items())}, "soft"


def make_items(rows: list[dict], ann: dict, split: str, rep: Report) -> list[Item]:
    out = []
    for r in rows:
        state = state_text(r["prompt"], r["response"])
        if len(state.split()) > MAX_STATE_WORDS:
            rep.drop(f"{r['native_split']}:state_over_{MAX_STATE_WORDS}_words", len(ATTRIBUTES))
            continue
        a = ann.get((r["prompt"], r["response"]))
        for attr in ATTRIBUTES:
            released = int(r[attr])
            if not 0 <= released <= 4:
                rep.drop(f"{r['native_split']}:label_out_of_range")
                continue
            tgt, kind = target_for(a[attr] if a is not None else None, released)
            rep.stats.setdefault("target_kind", Counter())[kind] += 1
            out.append(Item(
                id=f"{NAME}:{r['native_split']}-{r['row']}:{attr}", source=NAME, split=split,
                label_source="human_multi" if a is not None else "human", license=LICENSE, group=r["group"], state=state, qtype="score",
                instructions=QUESTIONS[attr], criteria={str(i): t for i, t in enumerate(LEVELS[attr])},
                target=tgt, gold=str(released),
                meta={"format": "helpsteer2_score", "attribute": attr, "native_split": r["native_split"],
                      "target_kind": kind, "n_annotators": len(a[attr]) if a is not None else 1}))
    return out


def annotations(dis_rows: list[dict], rep: Report) -> dict:
    """(prompt, response) -> per-annotator ratings; keys with conflicting duplicate entries are dropped (-> one-hot)."""
    by = {}
    bad = set()
    for d in dis_rows:
        k = (d["prompt"], d["response"])
        v = {a: [int(x) for x in d[a]] for a in ATTRIBUTES}
        if k in by and by[k] != v:
            bad.add(k)
        by[k] = v
    for k in bad:
        del by[k]
    rep.stats["disagreement_keys_conflicting_duplicates"] = len(bad)
    return by


def load_rows(rep: Report) -> dict:
    out = {}
    for split, fname in SPLITS.items():
        with gzip.open(hf_file(HF_REPO, fname, HF_REVISION), "rt", encoding="utf-8") as f:
            raw = [json.loads(l) for l in f if l.strip()]
        rows = []
        for i, r in enumerate(raw):
            if not str(r["prompt"]).strip() or not str(r["response"]).strip():
                rep.drop(f"{split}:empty_text")
                continue
            rows.append({**r, "row": i, "native_split": split, "group": f"hs2-g:{text_hash(r['prompt'])}"})
        rep.stats[f"rows[{split}]"] = len(raw)
        out[split] = rows
    return out


def dedup_rows(rows: list[dict], rep: Report, tag: str) -> list[dict]:
    """Exact (prompt, response) duplicates: keep one if labels agree, drop all if they conflict."""
    by = {}
    for r in rows:
        by.setdefault((norm_text(r["prompt"]), norm_text(r["response"])), []).append(r)
    out = []
    for grp in by.values():
        labels = {tuple(int(g[a]) for a in ATTRIBUTES) for g in grp}
        if len(labels) > 1:
            rep.drop(f"{tag}:conflicting_duplicate_rows", len(grp))
            continue
        if len(grp) > 1:
            rep.drop(f"{tag}:duplicate_row", len(grp) - 1)
        out.append(grp[0])
    return out


def build():
    rep = Report(NAME, repo=HF_REPO, revision=HF_REVISION, license=LICENSE,
                 label_source="human_multi (>= 2 human annotators per sample; soft target = empirical rating distribution)",
                 group_definition=GROUP_DEF, pool_native_splits=["train"], native_test_split="validation",
                 level_descriptions_source="arXiv:2406.08673v1 Appendix G.3.1 (Detailed Rating Breakdown)")
    with gzip.open(hf_file(HF_REPO, DISAGREEMENTS, HF_REVISION), "rt", encoding="utf-8") as f:
        ann = annotations([json.loads(l) for l in f if l.strip()], rep)
    rows = load_rows(rep)
    train = dedup_rows(rows["train"], rep, "train")
    val = dedup_rows(rows["validation"], rep, "validation")
    train_groups = {r["group"] for r in train}
    leak = [r for r in val if r["group"] in train_groups]
    rep.drop("validation:prompt_group_also_in_train", len(leak) * len(ATTRIBUTES))
    val = [r for r in val if r["group"] not in train_groups]
    for tag, rs in (("train", train), ("validation", val)):
        rep.stats[f"rows_matched_to_individual_ratings[{tag}]"] = sum((r["prompt"], r["response"]) in ann for r in rs)
        rep.stats[f"multi_turn_rows[{tag}]"] = sum("<extra_id_1>" in r["prompt"] for r in rs)
    pool = assign_splits(checked(make_items(train, ann, "train", rep), rep), NAME)
    test = checked(make_items(val, ann, "eval", rep), rep)
    rep.stats["target_kind"] = dict(rep.stats.get("target_kind", {}))
    rep.stats["gold_distribution"] = {
        a: dict(sorted(Counter(it.gold for it in pool + test if it.meta["attribute"] == a).items())) for a in ATTRIBUTES}
    rep.note("LABELS are human (Scale AI, ~1000 US annotators, >= 3 per sample); RESPONSES are LLM-generated (10 NVIDIA "
             "in-house models) and ~95% of prompts come from user-contributed ShareGPT conversations. Multi-turn prompts "
             "contain earlier assistant turns of unstated origin. The response is the state being judged, not a label.")
    rep.note("Soft target = empirical distribution of the individual annotations (disagreements file); if the released "
             "label (rounded mean of the 3 most-agreeing annotators) is not an argmax level of it, one-hot on the "
             "released label (meta.target_kind = onehot_fallback). Label distributions are not rebalanced.")
    rep.note("Multi-turn prompts: NeMo '<extra_id_1>User/Assistant' markers rendered as 'User:' / 'Assistant:' turns.")
    rep.note("Native validation rows whose prompt group also occurs in train are dropped from the native test.")
    return pool, test, rep.to_dict()
