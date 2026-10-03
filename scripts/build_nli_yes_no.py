"""Re-ask NLI and claim-verification items as yes/no questions -> data/train/nli_yes_no.jsonl.

Why: every question type must come from >= 2 sources, and the SAME content must appear under two types, so the type
embedding learns the question format, not a topic.

Rules:
  * Derived items inherit the source item's split, group, label source and licence (asserted), so no content crosses splits.
  * p(true) = the source target's mass on entailment / supports; p(false) = the rest. Exact 0.5 ties are dropped
    (no well-defined gold). gold = argmax.
  * Deterministic subsample by stable hash of the source id: 20,000 train + 1,000 dev + 1,000 calib per source.
  * Question / option wording is ours (fixed per format), not copied from any eval set.

    python scripts/build_nli_yes_no.py   (after scripts/gate_sources.py)
"""
import hashlib
import json
from collections import Counter

from decision_model.schema import DATA_DIR, TRAIN_DIR, Item, read_jsonl, validate, write_jsonl

SOURCES = {"snli": "nli_pair", "wanli_unanimous": "nli_pair", "vitaminc": "claim_verification"}
PER_SPLIT = {"train": 20_000, "dev": 1_000, "calib": 1_000}
TEXT = {
    "nli_pair": ("Is the hypothesis definitely true given the text?", "entailment",
                 {"false": "No, the text does not show that the hypothesis is true.",
                  "true": "Yes, the text shows that the hypothesis is true."}),
    "claim_verification": ("Does the evidence support the claim?", "supports",
                           {"false": "No, the evidence does not support the claim.",
                            "true": "Yes, the evidence supports the claim."}),
}


def rank(item_id: str) -> int:
    return int(hashlib.sha256(f"nli_yes_no|{item_id}".encode()).hexdigest()[:12], 16)


def derive(it: Item) -> Item | None:
    q, pos, crit = TEXT[it.meta["format"]]
    first = it.instructions.split("\n")[0].strip()
    assert first.startswith(("Hypothesis:", "Claim:")), (it.id, first)
    p = float(it.target.get(pos, 0.0))
    if abs(p - 0.5) < 1e-9:
        return None
    target = {k: v for k, v in (("true", p), ("false", 1.0 - p)) if v > 0}
    new = Item(id=f"nli_yes_no:{it.id}", source="nli_yes_no", split=it.split, label_source=it.label_source, license=it.license,
               group=it.group, state=it.state, qtype="yes_no", instructions=f"{first}\n{q}", criteria=dict(crit),
               target=target, gold="true" if p > 0.5 else "false", roles={},
               meta={"format": "nli_yes_no", "derived_from": it.id, "derived_source": it.source,
                     "native_split": it.meta.get("native_split")})
    validate(new)
    assert (new.split, new.group) == (it.split, it.group)
    return new


def main():
    out, stats = [], {}
    for src, fmt in SOURCES.items():
        by = {s: [] for s in PER_SPLIT}
        for it in read_jsonl(TRAIN_DIR / f"{src}.jsonl"):
            if it.meta.get("format") == fmt and it.split in by:
                by[it.split].append(it)
        for split, n in PER_SPLIT.items():
            picked, ties = [], 0
            for it in sorted(by[split], key=lambda x: rank(x.id)):
                if len(picked) >= n:
                    break
                d = derive(it)
                if d is None:
                    ties += 1
                    continue
                picked.append(d)
            out += picked
            stats[f"{src}:{split}"] = {"available": len(by[split]), "kept": len(picked), "dropped_0.5_ties": ties,
                                       "true_rate": round(sum(d.gold == "true" for d in picked) / max(1, len(picked)), 3)}
    ids = Counter(it.id for it in out)
    assert max(ids.values()) == 1
    info = write_jsonl(out, TRAIN_DIR / "nli_yes_no.jsonl")
    (DATA_DIR / "gate_reports" / "nli_yes_no_build.json").write_text(json.dumps({"stats": stats, "file": info}, indent=1,
                                                                        default=str))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
