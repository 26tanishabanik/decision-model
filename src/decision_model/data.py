"""Training data: source mixture, splits, option formatting and token layout.

Rules enforced here (tests in tests/test_data.py):
  * WANLI enters only as its unanimous subset (wanli_unanimous); the full build is never read.
  * Mixture shares are fixed in advance; no source repeats more than MAX_EPOCHS times. If a cap binds, the realised mix
    is reported, not silently rebalanced.
  * Training uses split == "train" only; dev / calib are separate, and no id or leak group crosses splits (asserted).
  * Every yes/no item can also appear as a bare question with no answer descriptions (the "bare_yes_no" augmentation).
  * Options are never cut for content; a long input keeps the question whole and trims the input's beginning.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from decision_model.schema import TRAIN_DIR, Item, read_jsonl

BASE_MIX = {"vitaminc": 0.22, "snli": 0.15, "wanli_unanimous": 0.08, "intents": 0.12, "ai2_arc": 0.03,
            "civil_comments": 0.07, "boolq": 0.06, "nli_yes_no": 0.06, "helpsteer2": 0.11, "typed_decisions": 0.10}
# base shares scaled to 92%, plus 4% each of prompt-injection and agent-run safety data
DEFAULT_MIX = {**{k: round(v * 0.92, 4) for k, v in BASE_MIX.items()}, "injection": 0.04, "assebench": 0.04}
MIXES = {"default": DEFAULT_MIX, "base": BASE_MIX}
INTENT_SOURCES = ("clinc_oos", "massive", "banking77")
EXCLUDED = ("wanli",)       # the full WANLI build; training reads wanli_unanimous
MAX_EPOCHS = 3
MAX_PREFIX_TOKENS = 2048    # input + question budget; a longer input keeps its tail
MAX_OPTION_TOKENS = 128     # cap per option; truncations are counted
QTYPE_ID = {"choice": 0, "score": 1, "yes_no": 2}


def source_files(name: str) -> list[str]:
    return list(INTENT_SOURCES) if name == "intents" else [name]


def bare_yes_no_copy(it: Item) -> Item:
    """The same yes/no item with no answer descriptions, i.e. what a bare question ("Is this urgent?") looks like at
    read time (option_text fills a default). Same group, so splits never separate the pair."""
    return Item(**{**it.__dict__, "id": f"{it.id}:bare", "criteria": {"false": "", "true": ""},
                   "meta": {**it.meta, "bare": True}})


AUGMENTATIONS = {"bare_yes_no": lambda it: [bare_yes_no_copy(it)] if it.qtype == "yes_no" else []}


def load_all_splits(splits=("train", "dev", "calib"), sources=None, augment=()) -> dict[str, dict[str, list[Item]]]:
    """One pass over each allowed file -> {split: {source_file: [items]}}. `augment`: names in AUGMENTATIONS, applied
    to every split alike."""
    out = {sp: {} for sp in splits}
    for s in sources or DEFAULT_MIX:
        for f in source_files(s):
            assert f not in EXCLUDED, f
            for sp in splits:
                out[sp][f] = []
            for it in read_jsonl(TRAIN_DIR / f"{f}.jsonl"):
                if it.split in out:
                    out[it.split][f].append(it)
                    for name in augment:
                        out[it.split][f].extend(AUGMENTATIONS[name](it))
    return out


@dataclass
class Plan:
    items: list[Item]                      # training order
    report: dict = field(default_factory=dict)

    def digest(self) -> str:
        h = hashlib.sha256()
        for it in self.items:
            h.update(f"{it.id}\n".encode())
        return h.hexdigest()


def sample_plan(pools: dict[str, list[Item]], n_total: int, seed: int, mix: dict | None = None) -> Plan:
    """Mixture sampling with per-source epoch caps. Deterministic in (pools, n_total, seed, mix)."""
    rng = random.Random(seed)
    order, report = [], {"requested_total": n_total, "sources": {}}
    for name, share in (mix or DEFAULT_MIX).items():
        files = [f for f in source_files(name) if pools.get(f)]
        if not files:
            report["sources"][name] = {"available": 0, "sampled": 0}
            continue
        sizes = {f: len(pools[f]) for f in files}           # intents: share split across files by size
        tot = sum(sizes.values())
        for f in files:
            want = int(round(n_total * share * sizes[f] / tot))
            cap = MAX_EPOCHS * sizes[f]
            n = min(want, cap)
            picked = []
            while len(picked) < n:                          # whole shuffled passes, then a partial one
                pass_ = pools[f][:]
                rng.shuffle(pass_)
                picked += pass_[: n - len(picked)]
            order += picked
            report["sources"][f] = {"available": sizes[f], "requested": want, "sampled": n,
                                    "epochs": round(n / sizes[f], 3), "cap_binding": want > cap}
    rng.shuffle(order)
    got = len(order)
    for f, r in report["sources"].items():
        if r.get("sampled"):
            r["realised_share"] = round(r["sampled"] / got, 4)
    report["sampled_total"] = got
    return Plan(order, report)


def assert_disjoint(train_items, *other_splits):
    """No id and no leak group may appear in both training and dev/calib/eval."""
    tr_ids = {it.id for it in train_items}
    tr_groups = {(it.source, it.group) for it in train_items}
    for other in other_splits:
        ids = {it.id for it in other}
        assert not (tr_ids & ids), f"{len(tr_ids & ids)} ids shared between train and a held-out split"
        groups = {(it.source, it.group) for it in other}
        assert not (tr_groups & groups), f"{len(tr_groups & groups)} leak groups shared with a held-out split"


def option_keys(it: Item) -> list[str]:
    """Option keys: criteria order for choice, level order for score, (false, true) for yes/no."""
    if it.qtype == "score":
        return sorted(it.criteria, key=int)
    if it.qtype == "yes_no":
        return ["false", "true"]
    return list(it.criteria)


def option_text(it: Item, key: str) -> str:
    """The text read for one option: choice -> its description (or the key if empty); score -> the level description;
    yes/no -> "<key>: <description>", with a default description built from the question when none is given."""
    d = it.criteria.get(key)
    if it.qtype == "yes_no":
        if d in (None, ""):
            ins = it.instructions.strip()
            d = (f"Yes. This is true: {ins}" if key == "true" else f"No. This is false: {ins}") if ins else key
        return f"{key}: {d}"
    return d if d not in (None, "") else key


def state_text(it: Item) -> str:
    """Input first, question last."""
    s, q = it.state.strip(), it.instructions.strip()
    return f"{s}\n\n{q}" if s and q else (s or q)


def prefix_ids(tok, it: Item) -> tuple[list[int], bool]:
    """Token ids of input + question. If too long, keep the question whole and trim the input's beginning (keep its
    tail). -> (ids, truncated)."""
    ids = tok(state_text(it), add_special_tokens=False).input_ids
    if len(ids) <= MAX_PREFIX_TOKENS:
        return ids, False
    q = tok("\n\n" + it.instructions.strip(), add_special_tokens=False).input_ids
    s = tok(it.state.strip(), add_special_tokens=False).input_ids
    keep = MAX_PREFIX_TOKENS - len(q)
    assert keep > 0, f"question alone exceeds {MAX_PREFIX_TOKENS} tokens: {it.id}"
    return s[-keep:] + q, True


def option_suffix_ids(tok, texts: list[str]) -> tuple[list[int], list[tuple[int, int]], int]:
    """Options as one suffix, each wrapped in identical delimiters (" <option> text </option>") so an option's tokens
    never depend on its position. -> (ids, [(start, end) per option, relative to the suffix], n_truncated)."""
    ids, spans, n_trunc = [], [], 0
    for t in texts:
        o = tok(f" <option> {t} </option>", add_special_tokens=False).input_ids
        if len(o) > MAX_OPTION_TOKENS:
            o, n_trunc = o[:MAX_OPTION_TOKENS], n_trunc + 1
        spans.append((len(ids), len(ids) + len(o)))
        ids += o
    return ids, spans, n_trunc


def restricted_target(it: Item, keys: list[str]) -> list[float]:
    """Target distribution over the given keys, renormalised (zero mass stays zero)."""
    t = [it.target.get(k, 0.0) for k in keys]
    z = sum(t)
    return [v / z for v in t] if z > 0 else t


def plan_manifest(plan: Plan, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"digest": plan.digest(), **plan.report}, indent=1))
