"""Metrics on the frozen test sets.

Predictions: JSONL, one object per item: {"id": ..., "probs": {criteria_key: p}}. Missing keys get p = 0; probs are
renormalised.

Conventions (fixed; tests pin them):
- Argmax ties are scored by expected correctness under a uniform random tie-break: an item whose gold shares the top
  probability with m other options scores 1/(m+1). The same rule gives relevance / relational / full-decision.
  Hard predictions (for macro-F1, McNemar, what-is-on-top) break ties by criteria order.
- log-loss = cross-entropy against the (possibly soft) target, p clipped at 1e-15. KL = log-loss - H(target).
- Brier = sum over options of (p_k - t_k)^2 (multi-class form; for yes_no it is 2x the binary Brier).
- ECE: 15 equal-width bins on max p, bin (lo, hi], weighted |mean correctness - mean confidence|.
- Macro-F1 only where keys are class labels (item.meta["f1_space"] present; label = f1_space + key).
- Abstention: items ranked by max p; tied confidences are averaged as a block, so the curve does not depend on input
  order. AURC = mean selective risk over coverages k/N, k = 1..N.
- Standardized margins: log p(correct) - max log p(negatives), divided by the SD of all option log-probs in the set.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from decision_model.metrics import boot_ci, macro_f1, mcnemar, paired_boot_diff
from decision_model.schema import EVAL_DIR, Item, read_jsonl, validate

EPS = 1e-15
TIE_TOL = 1e-12
N_ECE_BINS = 15
COVERAGES = (0.5, 0.8, 1.0)
HARD_ROLES = ("hard", "in_question")


# ---- io --------------------------------------------------------------------------------------------------------------
def load_set(name: str, eval_dir: Path = EVAL_DIR) -> list[Item]:
    items = list(read_jsonl(Path(eval_dir) / f"{name}.jsonl"))
    for it in items:
        validate(it)
    return items


def load_preds(path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["id"] in out:
                    raise ValueError(f"duplicate prediction id {r['id']}")
                out[r["id"]] = r
    return out


def prob_vector(it: Item, pred: dict | None) -> np.ndarray:
    keys = list(it.criteria)
    if pred is None:
        return np.full(len(keys), 1 / len(keys))
    probs = pred["probs"]
    bad = set(probs) - set(keys)
    if bad:
        raise ValueError(f"{it.id}: prediction keys not in criteria: {sorted(bad)[:5]}")
    p = np.array([float(probs.get(k, 0.0)) for k in keys])
    if not np.all(np.isfinite(p)) or (p < 0).any() or p.sum() <= 0:
        raise ValueError(f"{it.id}: invalid probabilities")
    return p / p.sum()


# ---- per-item primitives ---------------------------------------------------------------------------------------------
def top_share(p_pos: float, p_neg) -> float:
    """Expected probability that the positive ranks above every negative under uniform random tie-breaking."""
    p_neg = np.asarray(p_neg, dtype=float)
    if p_neg.size == 0:
        return 1.0
    m = p_neg.max()
    if p_pos > m + TIE_TOL:
        return 1.0
    if p_pos < m - TIE_TOL:
        return 0.0
    return 1.0 / (1 + int((np.abs(p_neg - p_pos) <= TIE_TOL).sum()))


def ece(conf, correct, n_bins: int = N_ECE_BINS) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(tot / len(conf))


def coverage_curve(conf, correct):
    """-> accuracy at each coverage k/N (k = 1..N), ranking by conf desc with tied blocks averaged."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    c, y = conf[order], correct[order].copy()
    i = 0
    while i < len(c):
        j = i
        while j + 1 < len(c) and c[j + 1] == c[i]:
            j += 1
        y[i:j + 1] = y[i:j + 1].mean()
        i = j + 1
    return np.cumsum(y) / np.arange(1, len(y) + 1)


def aurc(conf, correct) -> float:
    return float(np.mean(1 - coverage_curve(conf, correct)))


def acc_at_coverage(conf, correct, cov: float) -> float:
    curve = coverage_curve(conf, correct)
    k = max(1, math.ceil(cov * len(curve)))
    return float(curve[k - 1])


def per_item(items: list[Item], preds: dict, allow_missing: bool = False) -> dict:
    missing = [it.id for it in items if it.id not in preds]
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} items have no prediction, e.g. {missing[:3]}")
    rows = {k: [] for k in ("correct", "pred", "gold", "ce", "kl", "brier", "conf", "f1_gold", "f1_pred",
                            "relevance", "relational", "full", "err_on_top", "m_relevance", "m_relational", "m_full")}
    logps = []
    for it in items:
        pr = preds.get(it.id)
        p = prob_vector(it, pr)
        keys = list(it.criteria)
        t = np.array([it.target.get(k, 0.0) for k in keys])
        g = keys.index(it.gold)
        lp = np.log(np.clip(p, EPS, None))
        others = np.delete(p, g)
        rows["correct"].append(top_share(p[g], others))
        hard = keys[int(np.argmax(p))]
        rows["pred"].append(hard)
        rows["gold"].append(it.gold)
        ce = float(-(t * lp).sum())
        rows["ce"].append(ce)
        rows["kl"].append(ce + float((t[t > 0] * np.log(t[t > 0])).sum()))
        rows["brier"].append(float(((p - t) ** 2).sum()))
        rows["conf"].append(float(p.max()))
        sp = it.meta.get("f1_space")
        rows["f1_gold"].append(None if sp is None else sp + it.gold)
        rows["f1_pred"].append(None if sp is None else sp + hard)
        if it.roles:
            pos = [k for k, r in it.roles.items() if r == "correct"]
            assert pos == [it.gold], f"{it.id}: roles 'correct' must be exactly the gold"
            unrel = [keys.index(k) for k, r in it.roles.items() if r == "unrelated"]
            hardn = [keys.index(k) for k, r in it.roles.items() if r in HARD_ROLES]
            rows["relevance"].append(top_share(p[g], p[unrel]) if unrel else np.nan)
            rows["relational"].append(top_share(p[g], p[hardn]) if hardn else np.nan)
            rows["full"].append(rows["correct"][-1])
            rows["err_on_top"].append(it.roles.get(hard, "unknown") if hard != it.gold else None)
            rows["m_relevance"].append(lp[g] - lp[unrel].max() if unrel else np.nan)
            rows["m_relational"].append(lp[g] - lp[hardn].max() if hardn else np.nan)
            rows["m_full"].append(lp[g] - np.delete(lp, g).max())
            logps.append(lp)
        else:
            for k in ("relevance", "relational", "full", "m_relevance", "m_relational", "m_full"):
                rows[k].append(np.nan)
            rows["err_on_top"].append(None)
    out = {k: np.array(v, dtype=object if k in ("pred", "gold", "f1_gold", "f1_pred", "err_on_top") else float)
           for k, v in rows.items()}
    sd = float(np.std(np.concatenate(logps))) + 1e-12 if logps else 1.0
    for k in ("m_relevance", "m_relational", "m_full"):
        out[k] = out[k] / sd
    out["n_missing"] = len(missing)
    return out


# ---- set summary -----------------------------------------------------------------------------------------------------
def _ci(fn, *arrays, n_boot):
    return boot_ci(fn, *arrays, n=n_boot) if len(arrays[0]) else None


def _mean(a):
    return float(np.mean(a))


def summarize(items: list[Item], preds: dict, n_boot: int = 2000, allow_missing: bool = False) -> dict:
    r = per_item(items, preds, allow_missing)
    n = len(items)
    res = {"n": n, "n_missing": r["n_missing"], "n_options": {"min": min(len(i.criteria) for i in items),
                                                             "max": max(len(i.criteria) for i in items)},
           "accuracy": _ci(_mean, r["correct"], n_boot=n_boot),
           "log_loss": _ci(_mean, r["ce"], n_boot=n_boot),
           "brier": _ci(_mean, r["brier"], n_boot=n_boot),
           "ece": _ci(ece, r["conf"], r["correct"], n_boot=n_boot),
           "majority_acc": float(max(np.unique(r["gold"], return_counts=True)[1]) / n)}
    if any(max(i.target.values()) < 1 for i in items):
        res["kl"] = _ci(_mean, r["kl"], n_boot=n_boot)
    if all(g is not None for g in r["f1_gold"]):
        res["macro_f1"] = _ci(lambda y, p: macro_f1(list(y), list(p)), r["f1_gold"], r["f1_pred"], n_boot=n_boot)
    else:
        res["macro_f1"] = None   # keys are option positions / per-item names, not class labels
    qtypes = sorted({i.qtype for i in items})
    if len(qtypes) > 1:
        res["accuracy_by_qtype"] = {q: _mean(r["correct"][np.array([i.qtype == q for i in items])]) for q in qtypes}

    has_roles = ~np.isnan(r["full"])
    if has_roles.any():
        rr = {"n_items": int(has_roles.sum())}
        for name, key in (("relevance", "relevance"), ("relational", "relational"), ("full_decision", "full"),
                          ("margin_relevance", "m_relevance"), ("margin_relational", "m_relational"),
                          ("margin_full", "m_full")):
            v = r[key][~np.isnan(r[key])]
            rr[name] = _ci(_mean, v, n_boot=n_boot) if len(v) else None
            if name in ("relevance", "relational"):
                rr[f"n_{name}"] = int(len(v))
        errs = [e for e in r["err_on_top"][has_roles] if e is not None]
        rr["n_errors"] = len(errs)
        rr["hard_on_top_rate"] = (sum(e in HARD_ROLES for e in errs) / len(errs)) if errs else None
        rr["unrelated_on_top_rate"] = (sum(e == "unrelated" for e in errs) / len(errs)) if errs else None
        res["roles"] = rr

    conf = r["conf"]
    curve = coverage_curve(conf, r["correct"])
    res["abstention"] = {"confidence": "max_p",
                         **{f"acc@{int(c * 100)}": acc_at_coverage(conf, r["correct"], c) for c in COVERAGES},
                         "aurc": _ci(aurc, conf, r["correct"], n_boot=n_boot),
                         "curve": [[round((k + 1) / n, 4), round(float(curve[k]), 6)]
                                   for k in sorted({max(0, math.ceil(f * n) - 1) for f in np.linspace(0.05, 1, 20)})]}
    return res


# ---- paired comparison -----------------------------------------------------------------------------------------------
def compare(items: list[Item], preds_a: dict, preds_b: dict, n_boot: int = 2000, allow_missing: bool = False) -> dict:
    """System A vs B on the same items: paired bootstrap CI of A - B (accuracy, log-loss, Brier, full decision)
    and exact McNemar on hard-prediction correctness."""
    a, b = per_item(items, preds_a, allow_missing), per_item(items, preds_b, allow_missing)
    dummy = np.zeros(len(items))
    mean = lambda _y, s: float(np.mean(s))
    out = {"n": len(items)}
    for name, key in (("accuracy", "correct"), ("log_loss", "ce"), ("brier", "brier")):
        out[name] = paired_boot_diff(mean, dummy, a[key], b[key], n=n_boot)
    m = ~np.isnan(a["full"])
    if m.any():
        out["full_decision"] = paired_boot_diff(mean, dummy[m], a["full"][m], b["full"][m], n=n_boot)
    out["mcnemar"] = mcnemar(a["gold"], a["pred"], b["pred"])
    return out


def evaluate(set_names, preds: dict, n_boot: int = 2000, allow_missing: bool = False, eval_dir: Path = EVAL_DIR):
    """-> {set: summary} for every set with at least one prediction; sets with none are listed under _skipped."""
    out, skipped = {}, []
    for name in set_names:
        items = load_set(name, eval_dir)
        if not any(it.id in preds for it in items):
            skipped.append(name)
            continue
        out[name] = summarize(items, preds, n_boot, allow_missing)
    if skipped:
        out["_skipped"] = skipped
    return out
