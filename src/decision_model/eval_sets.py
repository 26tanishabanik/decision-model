"""Frozen evaluation sets. Every item: split="eval", unified Item schema.

The SciFact / ANLI / toxic-chat / prompt-injection sets use the task definitions in eval_tasks.py. All wordings below
are frozen: changing any of them changes the eval, so bump the manifest and re-baseline.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from decision_model import eval_tasks as T
from decision_model.schema import Item
from decision_model.sources import typed_decisions as TD

SEED = 0

# ---- frozen wordings -------------------------------------------------------------------------------------------------
INJECTION_Q = T.INJECTION_Q
INJECTION_CRIT = {"false": "The text is not a prompt injection or jailbreak attempt.",
           "true": "The text is a prompt injection or jailbreak attempt."}
VERIFY_CRIT = T.VERIFY_CRIT
SELECT_Q = T.SELECT_Q
NLI_CRIT = T.NLI_CRIT
TOXICITY_Q = T.TOXICITY_Q
TOXICITY_CRIT = {"false": "The user message is not toxic.",
           "true": "The user message is toxic."}
BANKING_Q = "Which intent does this online-banking customer message express?"
CLINC_Q = ("Which intent does this request to a virtual assistant express? "
           "Choose out of scope if none of the listed intents applies.")
CLINC_OOS_KEY = "out_of_scope"
CLINC_OOS_TEXT = "Out of scope: the request matches none of the listed intents."
BFCL_Q = "Which function should be called to fulfil the user's request?"
HELLASWAG_Q = "Which ending most plausibly continues the text?"
OBQA_Q = "Which option correctly answers the question?"
SCIQ_Q = "Which option correctly answers the question?"
MC_KEYS = "ABCD"

# ---- pinned sources --------------------------------------------------------------------------------------------------
REV = {
    "anli": ("facebook/anli", "8e4813d81f46d313dac7892e1c28076917cfcdf9", None),
    "toxic_chat": (T.TOXIC[0], T.TOXIC[1], T.TOXIC[2]),
    "prompt_injection": (T.DEEPSET[0], T.DEEPSET[1], None),
    "banking77": ("mteb/banking77", "18072d2685ea682290f7b8924d94c62acc19c0b2", None),
    "clinc": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", "plus"),
    "typed_decisions": ("LocalLLaMA/typed-decisions", "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8", "all"),
    "bfcl": ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda", None),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", None),
    "openbookqa": ("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", "main"),
    "sciq": ("allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815", None),
}
BFCL_USED = ("multiple", "live_multiple", "exec_multiple")
BFCL_SKIPPED = {
    "simple, parallel, live_simple, live_parallel, java, javascript, sql, exec_simple, exec_parallel, rest":
        "exactly one candidate function (live_parallel: 1-2), so there is no choice to make",
    "parallel_multiple, live_parallel_multiple, exec_parallel_multiple":
        "gold is a set of 2-5 calls, usually to different functions: no single correct option",
    "irrelevance, live_irrelevance":
        "gold is 'call nothing', so there is no correct function to choose",
    "live_relevance": "no ground truth released",
    "multi_turn_*": "multi-step: gold is a sequence of calls per turn, state changes between turns",
    "chatable": "no functions",
}


def _h(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def _rng(key: str) -> random.Random:
    return random.Random(f"{SEED}|{key}")


def _item(**kw) -> Item:
    kw.setdefault("split", "eval")
    kw.setdefault("roles", {})
    kw.setdefault("meta", {})
    if "target" not in kw:
        kw["target"] = {kw["gold"]: 1.0}
    return Item(**kw)


def _hf(name, split):
    from datasets import load_dataset
    repo, rev, cfg = REV[name]
    return load_dataset(repo, cfg, revision=rev, split=split) if cfg else load_dataset(repo, revision=rev, split=split)


# ---- relational: SciFact, ANLI ---------------------------------------------------------------------------------------
def scifact_verify():
    out = []
    for r in T.scifact_verify_items():
        q = r["questions"]["q"]
        out.append(_item(id=f"scifact:{r['id']}:verify", source="scifact", label_source="human", license="CC-BY-NC-2.0",
                         group=f"doc:{r['meta']['doc_id']}", state=r["state"], qtype="choice",
                         instructions=q["instructions"], criteria=q["criteria"], gold=r["gold"],
                         meta={"doc_id": r["meta"]["doc_id"], "f1_space": ""}))
    return out


ROLE_MAP = {"support": "correct", "contradict": "hard", "unrelated": "unrelated"}


def scifact_select():
    out = []
    for r in T.scifact_select_items():
        q = r["questions"]["q"]
        out.append(_item(id=f"scifact:doc{r['meta']['doc_id']}:select", source="scifact", label_source="human",
                         license="CC-BY-NC-2.0", group=f"doc:{r['meta']['doc_id']}", state=r["state"], qtype="choice",
                         instructions=q["instructions"], criteria=q["criteria"], gold=r["gold"],
                         roles={k: ROLE_MAP[v] for k, v in r["meta"]["roles"].items()},
                         meta={"doc_id": r["meta"]["doc_id"]}))
    return out


def anli_test():
    return [_item(id=f"anli:{r['id']}", source="anli", label_source="human", license="CC-BY-NC-4.0",
                  group=f"premise:{_h(r['state'])}", state=r["state"], qtype="choice",
                  instructions=r["questions"]["q"]["instructions"], criteria=r["questions"]["q"]["criteria"],
                  gold=r["gold"], meta={"round": r["meta"]["round"], "f1_space": ""})
            for r in T.anli_items()]


# ---- safety yes_no ---------------------------------------------------------------------------------------------------
def toxic_chat():
    return [_item(id=f"toxic_chat:{r['id']}", source="toxic_chat", label_source="human", license="CC-BY-NC-4.0",
                  group=f"conv:{r['id']}", state=r["state"], qtype="yes_no", instructions=TOXICITY_Q,
                  criteria=dict(TOXICITY_CRIT), gold="true" if r["gold"] else "false",
                  meta={"f1_space": ""})
            for r in T.toxicity_items()]


def prompt_injection():
    return [_item(id=f"prompt_injection:test-{r['id']}", source="deepset_prompt_injections", label_source="human",
                  license="Apache-2.0", group=f"text:{_h(r['state'])}", state=r["state"], qtype="yes_no",
                  instructions=INJECTION_Q, criteria=dict(INJECTION_CRIT), gold="true" if r["gold"] else "false",
                  meta={"f1_space": ""})
            for r in T.prompt_injection_items()]


# ---- many-option intents ---------------------------------------------------------------------------------------------
def readable_intent(name: str) -> str:
    s = name.replace("_", " ").strip().lower()
    return s[:1].upper() + s[1:]


def banking77_test():
    ds = _hf("banking77", "test")
    names = sorted({(r["label"], r["label_text"]) for r in ds})
    assert [i for i, _ in names] == list(range(77)), "banking77 label ids not 0..76"
    crit = {n: readable_intent(n) for _, n in names}
    assert len(set(crit.values())) == 77
    return [_item(id=f"banking77:test-{i}", source="banking77", label_source="human", license="CC-BY-4.0",
                  group=f"text:{_h(r['text'])}", state=r["text"], qtype="choice", instructions=BANKING_Q,
                  criteria=dict(crit), gold=r["label_text"], meta={"f1_space": ""})
            for i, r in enumerate(ds)]


def clinc_test():
    ds = _hf("clinc", "test")
    names = ds.features["intent"].names
    assert "oos" in names and len(names) == 151
    crit = {n: readable_intent(n) for n in sorted(n for n in names if n != "oos")}
    crit[CLINC_OOS_KEY] = CLINC_OOS_TEXT
    assert len(set(crit.values())) == 151
    out = []
    for i, r in enumerate(ds):
        lab = names[r["intent"]]
        key = CLINC_OOS_KEY if lab == "oos" else lab
        out.append(_item(id=f"clinc150:test-{i}", source="clinc150", label_source="human", license="CC-BY-3.0",
                         group=f"text:{_h(r['text'])}", state=r["text"], qtype="choice", instructions=CLINC_Q,
                         criteria=dict(crit), gold=key, meta={"oos": lab == "oos", "f1_space": ""}))
    return out


# ---- typed decisions (LLM-labelled) ----------------------------------------------------------------------------------
def typed_decisions_test():
    ds = _hf("typed_decisions", "test")
    out, n_label_mismatch = [], 0
    for r in ds:
        state = json.loads(r["state"])
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        for qn, q in qs.items():
            qtype = TD.question_type(q, gold[qn])
            filled = "criteria" not in q
            crit = dict(TD.YES_NO_FALLBACK) if filled and qtype == "yes_no" else q["criteria"]
            if isinstance(crit, list):
                crit = {str(i): t for i, t in enumerate(crit)}
            probs = {str(k): float(v) for k, v in gold[qn]["probabilities"].items()}
            z = sum(probs.values())
            target = {k: v / z for k, v in probs.items()}
            top = max(target.values())
            tied = [k for k in crit if target.get(k, 0) == top]
            label = str(gold[qn]["label"])
            g = label if label in tied else tied[0]
            n_label_mismatch += label not in tied
            out.append(_item(id=f"typed_decisions:{r['id']}:{qn}", source="typed_decisions", label_source="llm",
                             license="Apache-2.0", group=f"case:{r['id']}", state=state, qtype=qtype,
                             instructions=q["instructions"], criteria=crit, target=target, gold=g,
                             meta={"workflow": r["workflow"], "question": qn, "teacher_label": label,
                                   "f1_space": f"{r['workflow']}/{qn}|", "criteria_filled": filled,
                                   "label_agreement": json.loads(r["label_agreement"]).get(qn)}))
    typed_decisions_test.n_label_mismatch = n_label_mismatch
    return out


# ---- BFCL tool selection ---------------------------------------------------------------------------------------------
def _bfcl_load(fname):
    from huggingface_hub import hf_hub_download
    repo, rev, _ = REV["bfcl"]
    p = hf_hub_download(repo, fname, repo_type="dataset", revision=rev)
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def _bfcl_idx(i: str) -> str:
    return re.match(r"^[a-z_]+?_(\d+)", i).group(1)


def bfcl_select():
    out, notes = [], {}
    for cat in BFCL_USED:
        qs = _bfcl_load(f"BFCL_v3_{cat}.json")
        if cat.startswith("exec"):
            golds = {x["id"]: [re.match(r"\s*([\w.]+)\(", g).group(1) for g in x["ground_truth"]] for x in qs}
        else:
            ans = _bfcl_load(f"possible_answer/BFCL_v3_{cat}.json")
            assert len(ans) == len(qs)
            by_idx = {_bfcl_idx(a["id"]): a for a in ans}
            assert len(by_idx) == len(ans)
            golds, n_id_fix = {}, 0
            for x in qs:           # match on the leading index: one live_multiple answer id has a typo in its group
                a = by_idx[_bfcl_idx(x["id"])]
                n_id_fix += a["id"] != x["id"]
                golds[x["id"]] = [next(iter(c)) for c in a["ground_truth"]]
            notes[cat] = {"answer_id_mismatches_matched_by_index": n_id_fix}
        n_skip = 0
        for x in qs:
            g = golds[x["id"]]
            fns = x["function"]
            names = [f["name"] for f in fns]
            if len(g) != 1 or len(fns) < 2 or len(set(names)) != len(names) or g[0] not in names \
                    or len(x["question"]) != 1:
                n_skip += 1
                continue
            msgs = x["question"][0]
            state = msgs[0]["content"] if len(msgs) == 1 and msgs[0]["role"] == "user" else \
                "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs)
            order = list(range(len(fns)))
            _rng(x["id"]).shuffle(order)
            crit = {fns[j]["name"]: f"{fns[j]['name']}: {' '.join(fns[j].get('description', '').split())}".rstrip(": ")
                    for j in order}
            m = re.match(r"^live_multiple_\d+-(\d+)-", x["id"])
            out.append(_item(id=f"bfcl:{x['id']}", source="bfcl", label_source="human", license="Apache-2.0",
                             group=f"{cat}:{m.group(1)}" if m else f"{cat}:{x['id']}", state=state, qtype="choice",
                             instructions=BFCL_Q, criteria=crit, gold=g[0],
                             meta={"category": cat, "native_gold_pos": names.index(g[0]), "n_functions": len(fns)}))
        notes.setdefault(cat, {})["skipped_items"] = n_skip
    bfcl_select.notes = notes
    return out


# ---- native multiple choice (eval-only) ------------------------------------------------------------------------------
def hellaswag_val():
    ds = _hf("hellaswag", "validation")
    out = []
    for i, r in enumerate(ds):   # native "ind" is not unique within validation
        crit = {MC_KEYS[j]: e.strip() for j, e in enumerate(r["endings"])}
        out.append(_item(id=f"hellaswag:val-{i}", source="hellaswag", label_source="human", license="MIT",
                         group=f"src:{r['source_id']}", state=f"{r['activity_label']}: {r['ctx']}".strip(),
                         qtype="choice", instructions=HELLASWAG_Q, criteria=crit, gold=MC_KEYS[int(r["label"])],
                         meta={"split_type": r["split_type"], "ind": r["ind"]}))
    return out


def openbookqa_test():
    ds = _hf("openbookqa", "test")
    out = []
    for r in ds:
        labs, texts = r["choices"]["label"], r["choices"]["text"]
        assert labs == list(MC_KEYS)
        crit = dict(zip(labs, (t.strip() for t in texts)))
        out.append(_item(id=f"openbookqa:{r['id']}", source="openbookqa", label_source="human", license="Apache-2.0",
                         group=f"q:{r['id']}", state=r["question_stem"].strip(), qtype="choice", instructions=OBQA_Q,
                         criteria=crit, gold=r["answerKey"],
                         roles={k: "correct" if k == r["answerKey"] else "hard" for k in labs}))
    return out


def sciq_test():
    ds = _hf("sciq", "test")
    out, n_skip = [], 0
    for i, r in enumerate(ds):
        opts = [("correct", r["correct_answer"])] + [("hard", r[f"distractor{j}"]) for j in (1, 2, 3)]
        opts = [(role, t.strip()) for role, t in opts]
        if len({t.lower() for _, t in opts}) < 4 or not all(t for _, t in opts):
            n_skip += 1
            continue
        _rng(f"sciq:{i}").shuffle(opts)
        out.append(_item(id=f"sciq:test-{i}", source="sciq", label_source="human", license="CC-BY-NC-3.0",
                         group=f"q:{_h(r['question'])}", state=r["question"].strip(), qtype="choice",
                         instructions=SCIQ_Q, criteria={MC_KEYS[j]: t for j, (_, t) in enumerate(opts)},
                         gold=MC_KEYS[[role for role, _ in opts].index("correct")],
                         roles={MC_KEYS[j]: role for j, (role, _) in enumerate(opts)}))
    sciq_test.n_skip = n_skip
    return out


# name -> (builder, pinned source, construction notes)
SETS = {
    "scifact_verify": (scifact_verify, "scifact",
                       "All labelled SciFact claim-doc pairs from official train + dev; "
                       "state = annotated rationale sentences; claim in instructions; criteria support/contradict."),
    "scifact_select": (scifact_select, "scifact",
                       "Seed 0. State = title + abstract; options = first SUPPORT claim "
                       "(correct), first CONTRADICT claim if any (hard), unrelated claims from other docs up to 4 "
                       "options (unrelated). Docs without a SUPPORT claim are dropped."),
    "anli_test": (anli_test, "anli", "Rounds 1-3 test sets; the hypothesis is in the instructions."),
    "toxic_chat": (toxic_chat, "toxic_chat", "toxicchat0124 test split; true = toxic."),
    "prompt_injection": (prompt_injection, "prompt_injection", "Test split; true = prompt injection."),
    "banking77_test": (banking77_test, "banking77",
                       "Choice over all 77 intents in label-id order, keys = native intent names, text = readable "
                       "name. Rows from the mteb/banking77 parquet copy: 3,076 test rows (the original release has "
                       "3,080)."),
    "clinc_test": (clinc_test, "clinc",
                   "'plus' config test: 150 intents x 30 + 1000 out-of-scope. Choice over 150 intents (alphabetical) "
                   "+ out_of_scope last; meta.oos marks OOS items."),
    "typed_decisions_test": (typed_decisions_test, "typed_decisions",
                             "One item per (case, question): 400 cases x 5. Target = teacher gold probabilities "
                             "(renormalised); gold = argmax, ties broken towards the released label. The two yes_no "
                             "questions that ship no criteria (invoice duplicate, security credential_compromise) get "
                             "typed_decisions.YES_NO_FALLBACK (meta.criteria_filled). Synthetic states, LLM-teacher "
                             "labels (mean of 3 samples)."),
    "bfcl_select": (bfcl_select, "bfcl",
                    "Tool selection. Categories multiple, live_multiple, exec_multiple: one ground-truth call, >= 2 "
                    "candidate functions with unique names. State = user query (live: 'Role: content' lines when a "
                    "system message is present); options = 'name: description', keyed by function name, order "
                    "shuffled per item (seed 0). Only the function NAME is scored, not arguments."),
    "hellaswag_val": (hellaswag_val, "hellaswag",
                      "Validation split (test labels are hidden). State = 'activity_label: ctx'; 4 endings A-D in "
                      "native order. Endings are machine-generated adversarial distractors, so no roles."),
    "openbookqa_test": (openbookqa_test, "openbookqa",
                        "'main' config test, closed book (the fact is not given). Native A-D order; distractors = hard."),
    "sciq_test": (sciq_test, "sciq",
                  "Test, closed book (support paragraph not given). Correct + 3 human distractors shuffled per item "
                  "(seed 0); distractors = hard."),
}
