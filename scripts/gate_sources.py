"""Quality gates: turn built sources (data/sources/<name>.jsonl) into training files (data/train/<name>.jsonl).

For every built source listed for training in data/licenses.json:
  * at most GROUP_CAP[name] items per leak group (Civil Comments repeats short texts such as "Spot on!" many times);
  * near-duplicates of any test item are removed (5-gram Jaccard >= 0.8 on state + question, or an exact normalised
    match of a state with >= 8 words);
  * a state-blind audit (a classifier that sees only question + options) flags sources answerable without the state;
  * unrelated-option negatives are added (add_unrelated_negatives);
  * a 100-item review sample is written per source;
  * WANLI is written as its unanimous subset (wanli_unanimous.jsonl): pair items whose two annotators both chose the
    final label, plus the premise-choice items.
Reports: data/gate_reports/gates.json, data/gate_reports/removed_<name>.jsonl, data/gate_reports/review/<name>.md.

    python scripts/gate_sources.py [name ...]
"""
import hashlib
import json
import sys
from collections import defaultdict

from decision_model.quality import (
    add_unrelated_negatives,
    budget_stats,
    dedup_against_eval,
    gold_position_stats,
    review_pack,
    state_blind_audit,
)
from decision_model.schema import DATA_DIR, EVAL_DIR, SOURCES_DIR, TRAIN_DIR, read_jsonl, write_jsonl
from decision_model.sources import DATASETS

GROUP_CAP = {"civil_comments": 3}
UNANIMOUS_POLICY = "generated text, unanimous human label (both annotators = final gold)"
REPORTS = DATA_DIR / "gate_reports"


def unanimous_subset(items):
    """WANLI items whose two annotators both chose the final label, plus the premise-choice items."""
    out = []
    for it in items:
        votes = it.meta.get("annotator_golds") or []
        if it.meta.get("format") == "nli_premise_choice" or (len(votes) == 2 and all(v == it.gold for v in votes)):
            out.append(type(it)(**{**it.__dict__, "label_source": "human_on_llm_text",
                                   "meta": {**it.meta, "policy": UNANIMOUS_POLICY}}))
    return out


def training_sources() -> set:
    reg = json.loads((DATA_DIR / "licenses.json").read_text())["sources"]
    return {e["dataset"] for e in reg if "train" in e["use"]}


def cap_groups(items, cap):
    by = defaultdict(list)
    for it in items:
        by[it.group].append(it)
    out = []
    for g in by.values():
        out += sorted(g, key=lambda it: hashlib.sha256(it.id.encode()).hexdigest())[:cap]
    return out


def main(names):
    allowed = training_sources()
    eval_items = [it for p in sorted(EVAL_DIR.glob("*.jsonl")) for it in read_jsonl(p)]
    (REPORTS / "review").mkdir(parents=True, exist_ok=True)
    report = {"n_eval_items": len(eval_items), "sources": {}}
    for src in names or sorted(DATASETS):
        if DATASETS[src] not in allowed:
            print(src, "skipped: not listed for training in data/licenses.json", flush=True)
            continue
        items = list(read_jsonl(SOURCES_DIR / f"{src}.jsonl"))
        n0 = len(items)
        if src in GROUP_CAP:
            items = cap_groups(items, GROUP_CAP[src])
        kept, removed = dedup_against_eval(items, eval_items)
        split = {s: [it for it in kept if it.split == s] for s in ("train", "dev", "calib")}
        entry = {"n_in": n0, "n_after_group_cap": len(items), "n_removed_near_dup_eval": len(removed),
                 "removed_by_eval_set": {}, "split_counts": {s: len(v) for s, v in split.items()},
                 "gold_position": gold_position_stats(kept), "budget": budget_stats(kept),
                 "state_blind_audit": state_blind_audit(split["train"], split["dev"])}
        for _, eid, _ in removed:
            s = eid.split(":")[0]
            entry["removed_by_eval_set"][s] = entry["removed_by_eval_set"].get(s, 0) + 1
        (REPORTS / f"removed_{src}.jsonl").write_text(
            "\n".join(json.dumps({"train_id": a, "eval_id": b, "jaccard": j}) for a, b, j in removed))
        (REPORTS / "review" / f"{src}.md").write_text(review_pack(split["train"] or kept))
        aug, entry["augmentation"] = add_unrelated_negatives(kept)
        if src == "wanli":
            aug = unanimous_subset(aug)
        out_name = "wanli_unanimous" if src == "wanli" else src
        entry["written"] = write_jsonl(aug, TRAIN_DIR / f"{out_name}.jsonl")
        report["sources"][src] = entry
        print(src, "| in", n0, "| kept", len(kept), "| removed near-dup of eval", len(removed),
              "| audit margin", entry["state_blind_audit"].get("margin_over_baseline"), flush=True)
    (REPORTS / "gates.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
