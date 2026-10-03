"""Shared helpers for the training-source builders."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from decision_model.schema import ROOT, Item, split_of, validate

RAW_DIR = ROOT / "data" / "raw"
LETTERS = [chr(ord("A") + i) for i in range(26)]


def hf_file(repo: str, filename: str, revision: str) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def hf_parquet(repo: str, filename: str, revision: str) -> pd.DataFrame:
    return pd.read_parquet(hf_file(repo, filename, revision))


def hf_jsonl(repo: str, filename: str, revision: str) -> list[dict]:
    with open(hf_file(repo, filename, revision)) as f:
        return [json.loads(l) for l in f if l.strip()]


def parquet_label_names(repo: str, filename: str, revision: str, column: str) -> list[str]:
    import pyarrow.parquet as pq
    md = pq.read_schema(hf_file(repo, filename, revision)).metadata
    return json.loads(md[b"huggingface"])["info"]["features"][column]["names"]


def clean(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


_PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def unptb(s: str) -> str:
    """Unescape PTB bracket tokens used in FEVER/VitaminC Wikipedia text."""
    for k, v in _PTB.items():
        s = s.replace(k, v)
    return clean(s)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_hash(s: str, n: int = 16) -> str:
    return hashlib.sha1(norm_text(s).encode()).hexdigest()[:n]


def stable_rng(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


class UnionFind:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def gid(self, x, prefix: str) -> str:
        return f"{prefix}:{hashlib.sha1(repr(self.find(x)).encode()).hexdigest()[:16]}"

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


class Report:
    """Collects drops, notes and stats for one source build."""

    def __init__(self, source: str, **info):
        self.source = source
        self.info = dict(info)
        self.drops: Counter = Counter()
        self.notes: list[str] = []
        self.stats: dict = {}

    def drop(self, reason: str, n: int = 1):
        self.drops[reason] += n

    def note(self, s: str):
        self.notes.append(s)

    def to_dict(self) -> dict:
        return {**self.info, "drops": dict(self.drops), "notes": self.notes, "stats": self.stats}


def assign_splits(items: list[Item], salt: str) -> list[Item]:
    for it in items:
        it.split = split_of(it.group, salt=salt)
    return items


def one_hot(key: str) -> dict:
    return {key: 1.0}


def checked(items: list[Item], rep: Report) -> list[Item]:
    """Validate; drop (and record) anything invalid rather than crash mid-build."""
    out = []
    for it in items:
        try:
            validate(it)
        except AssertionError:
            rep.drop(f"{it.qtype}:failed_validate")
            continue
        out.append(it)
    return out


def shuffled_options(key: str, texts: list[str]) -> list[tuple[str, int]]:
    """Deterministic shuffle of constructed option lists -> [(letter, original_index)]."""
    order = list(range(len(texts)))
    stable_rng(key).shuffle(order)
    return [(LETTERS[pos], orig) for pos, orig in enumerate(order)]


# ---------------- NLI (SNLI / WANLI) ----------------

NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_CRITERIA = {
    "entailment": "The hypothesis is definitely true given the text.",
    "neutral": "The hypothesis might be true or false; the text does not say.",
    "contradiction": "The hypothesis is definitely false given the text.",
}
NLI_PREMISE_Q = "Which statement is definitely true given the text?"
MIN_AGREEMENT = 0.8


def nli_pair_instructions(h: str) -> str:
    return f'Hypothesis: "{h}"\nIs the hypothesis true, undetermined, or false given the text?'


def nli_items(rows: list[dict], source: str, license: str, split: str, rep: Report) -> list[Item]:
    """rows: dicts with id, premise, hypothesis, label, group, label_source, [target], [meta].

    Emits (a) one 3-way choice per pair and (b) one per-premise choice per entailment hypothesis whose premise
    also has >= 1 contradiction (options: that entailment + all contradictions + all neutrals of the premise).
    """
    items: list[Item] = []
    by_premise: dict[str, list[dict]] = {}
    for r in rows:
        tgt = r.get("target") or one_hot(r["label"])
        items.append(Item(
            id=f"{source}:{r['id']}", source=source, split=split, label_source=r["label_source"], license=license,
            group=r["group"], state=r["premise"], qtype="choice",
            instructions=nli_pair_instructions(r["hypothesis"]), criteria=dict(NLI_CRITERIA),
            target=tgt, gold=r["label"], roles={}, meta={"format": "nli_pair", **r.get("meta", {})}))
        by_premise.setdefault(r["premise"], []).append(r)

    n_multi_ent = 0
    for premise, rs in by_premise.items():
        # "Definitely true" items need confident labels: with annotator distributions, keep a hypothesis only if
        # >= MIN_AGREEMENT of the mass is on its label (contested pairs stay only in the 3-way soft-target format).
        confident = [r for r in rs if (r.get("target") or one_hot(r["label"])).get(r["label"], 0) >= MIN_AGREEMENT]
        if len(confident) < len(rs):
            rep.drop("premise_choice:contested_hypothesis_excluded", len(rs) - len(confident))
        rs = confident
        ents = [r for r in rs if r["label"] == "entailment"]
        cons = [r for r in rs if r["label"] == "contradiction"]
        neus = [r for r in rs if r["label"] == "neutral"]
        if not ents or not cons:
            continue
        if len(ents) > 1:
            n_multi_ent += 1
        groups = {r["group"] for r in rs}
        assert len(groups) == 1, (source, premise, groups)
        for e in ents:
            cands = [e] + cons + neus
            texts = [c["hypothesis"] for c in cands]
            if len({norm_text(t) for t in texts}) < len(texts):
                rep.drop("premise_choice:duplicate_hypothesis_text")
                continue
            perm = shuffled_options(f"{source}:premise:{e['id']}", texts)
            crit, roles, gold = {}, {}, None
            for k, orig in perm:
                c = cands[orig]
                crit[k] = c["hypothesis"]
                roles[k] = {"entailment": "correct", "contradiction": "hard", "neutral": "in_question"}[c["label"]]
                if orig == 0:
                    gold = k
            items.append(Item(
                id=f"{source}:premise:{e['id']}", source=source, split=split, label_source="human", license=license,
                group=e["group"], state=premise, qtype="choice", instructions=NLI_PREMISE_Q, criteria=crit,
                target=one_hot(gold), gold=gold, roles=roles,
                meta={"format": "nli_premise_choice", "n_options": len(crit),
                      "option_ids": {k: cands[o]["id"] for k, o in perm}}))
    rep.stats.setdefault(f"premises_with_multiple_entailments[{split}]", n_multi_ent)
    return items


# ---------------- intent classification (banking77 / CLINC / MASSIVE) ----------------

def readable(name: str) -> str:
    s = name.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def drop_conflicting_duplicates(rows: list[dict], rep: Report, tag: str) -> list[dict]:
    """Drop every row whose normalized text also occurs with a different label (contradictory supervision)."""
    labels: dict[str, set] = {}
    for r in rows:
        labels.setdefault(norm_text(r["text"]), set()).add(r["label"])
    out = []
    n_dup = Counter(norm_text(r["text"]) for r in rows)
    rep.stats[f"duplicate_text_rows[{tag}]"] = sum(v for v in n_dup.values() if v > 1)
    for r in rows:
        if len(labels[norm_text(r["text"])]) > 1:
            rep.drop(f"{tag}:duplicate_text_conflicting_labels")
            continue
        out.append(r)
    return out


def intent_items(rows: list[dict], source: str, license: str, split: str, criteria: dict,
                 instructions: str = "Which intent does this message express?") -> list[Item]:
    """rows: dicts with id, text, label (a criteria key), group, [meta]."""
    return [Item(
        id=f"{source}:{r['id']}", source=source, split=split, label_source="human", license=license, group=r["group"],
        state=r["text"], qtype="choice", instructions=instructions, criteria=criteria,
        target=one_hot(r["label"]), gold=r["label"], roles={r["label"]: "correct"},
        meta={"format": "intent", "n_options": len(criteria), **r.get("meta", {})}) for r in rows]


# ---------------- multiple choice (ARC) ----------------

def dedup_options(labels: list[str], texts: list[str], answer: str):
    """Remove later duplicate distractors (case-insensitive). ok=False when the gold text is itself duplicated."""
    gold_t = texts[labels.index(answer)].lower()
    if sum(t.lower() == gold_t for t in texts) > 1:
        return labels, texts, False
    seen, keep = set(), []
    for i, t in enumerate(texts):
        if t.lower() not in seen:
            seen.add(t.lower())
            keep.append(i)
    return [labels[i] for i in keep], [texts[i] for i in keep], True


def mc_item(source: str, license: str, split: str, nid: str, group: str, question: str,
            labels: list[str], texts: list[str], answer: str, meta: dict) -> Item:
    crit = {lab: clean(t) for lab, t in zip(labels, texts)}
    roles = {lab: ("correct" if lab == answer else "hard") for lab in labels}
    return Item(id=f"{source}:{nid}", source=source, split=split, label_source="human", license=license, group=group,
                state=clean(question), qtype="choice", instructions="Which answer is correct?", criteria=crit,
                target=one_hot(answer), gold=answer, roles=roles,
                meta={"format": "multiple_choice", "n_options": len(crit), **meta})


# ---------------- claim verification (VitaminC) ----------------

CV_KEYS = {"SUPPORTS": "supports", "REFUTES": "refutes", "NOT ENOUGH INFO": "not_enough_info"}
CV_CRITERIA = {
    "supports": "The evidence supports the claim.",
    "refutes": "The evidence refutes the claim.",
    "not_enough_info": "The evidence does not give enough information to decide.",
}


def cv_instructions(claim: str) -> str:
    return f'Claim: "{claim}"\nDoes the evidence support the claim, refute it, or not give enough information?'
