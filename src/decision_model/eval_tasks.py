"""Task definitions behind five frozen test sets: prompt injection (deepset), toxicity (toxic-chat), SciFact claim
verification, SciFact claim selection and ANLI. Every item: {"id", "state", "questions": {"q": {...}}, "gold", "meta"}."""
from __future__ import annotations

import collections
import random

from decision_model.loaders import load_anli, load_scifact

DEEPSET = ("deepset/prompt-injections", "4f61ecb038e9c3fb77e21034b22511b523772cdd")
TOXIC = ("lmsys/toxic-chat", "29df8e4dba60e1f4af4b4075c0705c5b313548a8", "toxicchat0124")
SEED = 0

INJECTION_Q = "Is this text a prompt injection or jailbreak attempt that tries to override or ignore an AI assistant's instructions?"
TOXICITY_Q = "Is this user message toxic, meaning harmful, offensive, sexual, or otherwise inappropriate?"
VERIFY_CRIT = {"support": "The evidence supports the claim.", "contradict": "The evidence contradicts the claim."}
SELECT_Q = "Which claim does this abstract support?"
NLI_CRIT = {"entailment": "The hypothesis is definitely true given the text.",
           "neutral": "The hypothesis might be true or false; the text does not say.",
           "contradiction": "The hypothesis is definitely false given the text."}


def prompt_injection_items():
    from datasets import load_dataset
    ds = load_dataset(DEEPSET[0], revision=DEEPSET[1], split="test")
    return [{"id": str(i), "state": r["text"],
             "questions": {"q": {"type": "yes_no", "instructions": INJECTION_Q}}, "gold": int(r["label"]), "meta": {}}
            for i, r in enumerate(ds)]


def toxicity_items():
    from datasets import load_dataset
    ds = load_dataset(TOXIC[0], TOXIC[2], revision=TOXIC[1], split="test")
    return [{"id": r["conv_id"], "state": r["user_input"],
             "questions": {"q": {"type": "yes_no", "instructions": TOXICITY_Q}}, "gold": int(r["toxicity"]), "meta": {}}
            for r in ds]


def _scifact_all():
    return [p for p in load_scifact(premise="rationale")]  # train + official dev


def scifact_verify_items():
    return [{"id": p.uid, "state": p.x,
             "questions": {"q": {"type": "choice", "instructions": f'Claim: "{p.y}"\nDoes the evidence support or contradict this claim?',
                                 "criteria": dict(VERIFY_CRIT)}},
             "gold": "support" if p.label == 0 else "contradict", "meta": {"doc_id": p.meta["doc_id"]}}
            for p in _scifact_all()]


def scifact_select_items():
    pairs = load_scifact(premise="abstract")
    by_doc = collections.OrderedDict()
    for p in pairs:
        by_doc.setdefault(p.meta["doc_id"], []).append(p)
    all_claims = sorted({p.y for p in pairs})
    rng = random.Random(SEED)
    items = []
    for doc, ps in by_doc.items():
        sup = [p.y for p in ps if p.label == 0]
        con = [p.y for p in ps if p.label == 1 and p.y not in sup]
        if not sup:
            continue
        own = set(p.y for p in ps)
        opts = [("support", sup[0])] + ([("contradict", con[0])] if con else [])
        while len(opts) < 4:
            c = all_claims[rng.randrange(len(all_claims))]
            if c not in own and c not in [o for _, o in opts]:
                opts.append(("unrelated", c))
        rng.shuffle(opts)
        keys = "ABCD"
        items.append({"id": str(doc), "state": ps[0].x,
                      "questions": {"q": {"type": "choice", "instructions": SELECT_Q,
                                          "criteria": {keys[i]: t for i, (_, t) in enumerate(opts)}}},
                      "gold": keys[[r for r, _ in opts].index("support")],
                      "meta": {"roles": {keys[i]: r for i, (r, _) in enumerate(opts)}, "doc_id": doc}})
    return items


def anli_items():
    names = ["entailment", "neutral", "contradiction"]
    return [{"id": p.uid, "state": p.x,
             "questions": {"q": {"type": "choice", "instructions": f'Hypothesis: "{p.y}"\nIs the hypothesis true, undetermined, or false given the text?',
                                 "criteria": dict(NLI_CRIT)}},
             "gold": names[p.label], "meta": {"round": p.meta["round"]}}
            for p in load_anli() if p.split.startswith("test_")]
