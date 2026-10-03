"""Item schema shared by every training and evaluation file (one JSON object per line).

Every item is a typed question over a state:
  qtype == "choice": criteria = {key: option_text}   (ordered; keys are what target/gold refer to)
  qtype == "yes_no": criteria = {"false": text, "true": text}
  qtype == "score":  criteria = {"0": level_text, "1": ..., ...}   (ordered levels)
Targets are distributions over criteria keys (soft when multiple annotators, one-hot otherwise).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"      # one file per source after building (all splits)
TRAIN_DIR = DATA_DIR / "train"          # the same sources after the quality gates: what training reads
EVAL_DIR = DATA_DIR / "eval"            # frozen test sets + manifest
# who produced the target: several human annotators, one human label, humans labelling LLM-written text, or an LLM
LABEL_SOURCES = ("human_multi", "human", "human_on_llm_text", "llm")
QTYPES = ("choice", "yes_no", "score")
ROLES = ("correct", "hard", "in_question", "unrelated")   # hard = a human-written wrong answer close to the correct one
SPLITS = ("train", "dev", "calib", "eval")


@dataclass
class Item:
    id: str                    # globally unique: "<source>:<native id>[:<question id>]"
    source: str                # registry key, e.g. "vitaminc"
    split: str                 # train | dev | calib (training sources) ; eval (held-out eval sets)
    label_source: str          # one of LABEL_SOURCES
    license: str               # SPDX id of the source dataset
    group: str                 # leak unit (premise / doc / claim family / template / task); splits never cross groups
    state: str
    qtype: str
    instructions: str
    criteria: dict             # see module docstring
    target: dict               # key -> probability, sums to 1
    gold: str                  # argmax key of target
    roles: dict = field(default_factory=dict)   # key -> one of ROLES (optional but required for relational sources)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def validate(it: Item) -> None:
    assert it.split in SPLITS, it.split
    assert it.label_source in LABEL_SOURCES, it.label_source
    assert it.qtype in QTYPES, it.qtype
    assert it.state.strip() and it.instructions.strip(), it.id
    keys = list(it.criteria)
    assert len(keys) >= 2 and len(set(keys)) == len(keys), it.id
    if it.qtype == "yes_no":
        assert set(keys) == {"false", "true"}, it.id
    if it.qtype == "score":
        assert keys == [str(i) for i in range(len(keys))], it.id
    assert all(isinstance(v, str) and v.strip() for v in it.criteria.values()), it.id
    assert set(it.target) <= set(keys) and abs(sum(it.target.values()) - 1) < 1e-6, it.id
    assert it.gold in keys and it.target.get(it.gold, 0) == max(it.target.values()), it.id
    assert set(it.roles) <= set(keys) and set(it.roles.values()) <= set(ROLES), it.id


def split_of(group: str, salt: str, fractions=(("train", 0.90), ("dev", 0.05), ("calib", 0.05))) -> str:
    """Deterministic grouped split: every item of a group lands in the same split."""
    h = int(hashlib.sha256(f"{salt}|{group}".encode()).hexdigest()[:12], 16) / 16 ** 12
    acc = 0.0
    for name, frac in fractions:
        acc += frac
        if h < acc:
            return name
    return fractions[-1][0]


def write_jsonl(items, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for it in items:
        validate(it)
        lines.append(it.to_json())
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return {"path": str(path.relative_to(ROOT)), "n": len(lines), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield Item(**json.loads(line))
