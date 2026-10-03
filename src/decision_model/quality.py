"""Quality gates: near-duplicate removal against the test sets, state-blind
artifact audit, gold-position balance, budget checks, unrelated-negative augmentation, human review packs."""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict

import numpy as np

from decision_model.schema import Item

_WORD = re.compile(r"[a-z0-9]+")
N_PERM = 64
BANDS, ROWS = 16, 4          # 16 bands x 4 rows: candidate pairs at Jaccard ~0.5+, verified exactly afterwards
_P = np.uint64((1 << 31) - 1)          # 31-bit prime: a*x + b < 2^63, so uint64 never overflows
_rng = np.random.default_rng(12345)
_A = _rng.integers(1, int(_P), N_PERM, dtype=np.uint64)
_B = _rng.integers(0, int(_P), N_PERM, dtype=np.uint64)


def norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def shingles(text: str, n: int = 5) -> set[str]:
    w = norm(text).split()
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


SHORT_OPTION_WORDS = 4


def stem_words(text: str) -> set[str]:
    """Crude plural stripping so 'cell' and 'Cells' match; used only to exclude near-synonymous negatives."""
    out = set()
    for w in norm(text).split():
        for suf in ("ies", "es", "s"):
            if len(w) > 3 and w.endswith(suf):
                w = w[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.add(w)
    return out


def same_answer(a: str, b: str, threshold: float = 0.5) -> bool:
    """Word-level overlap test for short option texts, where 5-gram shingles are meaningless."""
    wa, wb = stem_words(a), stem_words(b)
    if not wa or not wb or min(len(wa), len(wb)) > SHORT_OPTION_WORDS:
        return False  # longer options are covered by 5-gram shingle Jaccard
    if wa <= wb or wb <= wa:
        return True
    return len(wa & wb) / len(wa | wb) >= threshold


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def minhash(sh: set[str]) -> np.ndarray:
    if not sh:
        return np.full(N_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    x = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little") % int(_P)
                  for s in sh], dtype=np.uint64)
    return ((np.outer(_A, x) + _B[:, None]) % _P).min(axis=1)


class NearDupIndex:
    """LSH over reference texts; query returns reference ids with exact 5-gram Jaccard >= threshold."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.buckets = defaultdict(list)
        self.sh = {}

    def add(self, key: str, text: str):
        sh = shingles(text)
        self.sh[key] = sh
        sig = minhash(sh)
        for b in range(BANDS):
            self.buckets[(b, sig[b * ROWS:(b + 1) * ROWS].tobytes())].append(key)

    def query(self, text: str) -> list[tuple[str, float]]:
        sh = shingles(text)
        sig = minhash(sh)
        cands = {k for b in range(BANDS) for k in self.buckets.get((b, sig[b * ROWS:(b + 1) * ROWS].tobytes()), [])}
        hits = [(k, jaccard(sh, self.sh[k])) for k in cands]
        return [(k, j) for k, j in hits if j >= self.threshold]




def dedup_against_eval(train_items: list[Item], eval_items: list[Item], threshold: float = 0.8):
    """-> (kept, removed[(train_id, eval_id, jaccard)]). Exact normalized-text matches are caught too (Jaccard 1)."""
    idx = NearDupIndex(threshold)
    exact = {}
    for ev in eval_items:
        key = ev.id
        text = f"{ev.state} {ev.instructions}"
        idx.add(key, text)
        exact.setdefault(norm(text), key)
        exact.setdefault(norm(ev.state), key) if len(norm(ev.state).split()) >= 8 else None
    kept, removed = [], []
    for it in train_items:
        hit = None
        full = f"{it.state} {it.instructions}"
        for t in (full, it.state):
            n = norm(t)
            if n in exact and len(n.split()) >= 8:
                hit = (exact[n], 1.0)
                break
        if hit is None:
            q = idx.query(full)
            if q:
                hit = max(q, key=lambda z: z[1])
        if hit:
            removed.append((it.id, hit[0], round(hit[1], 3)))
        else:
            kept.append(it)
    return kept, removed


def gold_position_stats(items: list[Item]) -> dict:
    """Distribution of the gold key's index in criteria order, per option count."""
    by_k = defaultdict(Counter)
    for it in items:
        keys = list(it.criteria)
        by_k[len(keys)][keys.index(it.gold)] += 1
    out = {}
    for k, c in sorted(by_k.items()):
        n = sum(c.values())
        top = max(c.values()) / n
        out[str(k)] = {"n": n, "max_position_share": round(top, 4), "uniform": round(1 / k, 4),
                       "imbalanced": top > 1 / k + 0.10}
    return out


def state_blind_audit(train: list[Item], dev: list[Item], seed: int = 0) -> dict:
    """Artifact audit: a model that never sees the state. Per option, TF-IDF of (instructions + option text) ->
    P(correct); pick argmax per item. Compare dev accuracy to the chance / majority baseline. A large margin means
    the answer is predictable without reading the state (hypothesis-only style artifact)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def rows(items):
        X, y, owner = [], [], []
        for i, it in enumerate(items):
            for k, v in it.criteria.items():
                X.append(f"{it.instructions} || {v}")
                y.append(int(k == it.gold))
                owner.append((i, k))
        return X, y, owner

    rng = random.Random(seed)
    train = rng.sample(train, min(len(train), 60000))
    dev = rng.sample(dev, min(len(dev), 10000))
    if len(train) < 50 or len(dev) < 20:
        return {"skipped": "too few items"}
    Xt, yt, _ = rows(train)
    Xd, yd, od = rows(dev)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(vec.fit_transform(Xt), yt)
    p = clf.predict_proba(vec.transform(Xd))[:, 1]
    best = {}
    for (i, k), s in zip(od, p):
        if i not in best or s > best[i][1]:
            best[i] = (k, s)
    acc = float(np.mean([best[i][0] == it.gold for i, it in enumerate(dev)]))
    chance = float(np.mean([1 / len(it.criteria) for it in dev]))
    maj_key = Counter(it.gold for it in train).most_common(1)[0][0]
    majority = float(np.mean([it.gold == maj_key for it in dev]))
    base = max(chance, majority)
    return {"state_blind_acc": round(acc, 4), "chance": round(chance, 4), "majority": round(majority, 4),
            "margin_over_baseline": round(acc - base, 4), "flag": acc - base > 0.15, "n_train": len(train),
            "n_dev": len(dev)}


def budget_stats(items: list[Item], state_words=3000, option_words=48, max_k=16) -> dict:
    sw = np.array([len(it.state.split()) for it in items])
    ow = np.array([max(len(v.split()) for v in it.criteria.values()) for it in items])
    k = np.array([len(it.criteria) for it in items])
    q = lambda a: {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": int(a.max())}
    return {"state_words": q(sw), "max_option_words": q(ow), "n_options": q(k),
            "frac_state_over_budget": round(float((sw > state_words).mean()), 4),
            "frac_option_over_budget": round(float((ow > option_words).mean()), 4),
            "frac_over_max_options": round(float((k > max_k).mean()), 4)}


def add_unrelated_negatives(items: list[Item], n_neg: int = 2, seed: int = 4321,
                            exclude_threshold: float = 0.5) -> tuple[list[Item], dict]:
    """For content-candidate `choice` items (those with roles), append n_neg options taken from the
    correct options of OTHER groups in the same source, excluding near-duplicates of the item's own options.
    Items without roles (fixed label sets: NLI labels, intents, yes_no/score) are returned unchanged."""
    pool = [(it.group, it.criteria[it.gold]) for it in items if it.qtype == "choice" and it.roles]
    pool_sh = [shingles(t) for _, t in pool]
    out, n_aug, n_skip = [], 0, 0
    for idx, it in enumerate(items):
        if not (it.qtype == "choice" and it.roles) or not pool:
            out.append(it)
            continue
        rng = random.Random(seed + idx)
        own = [shingles(v) for v in it.criteria.values()]
        new, tries = [], 0
        while len(new) < n_neg and tries < 50:
            tries += 1
            j = rng.randrange(len(pool))
            g, t = pool[j]
            if g == it.group or t in it.criteria.values() or t in new:
                continue
            if any(jaccard(pool_sh[j], o) >= exclude_threshold for o in own):
                continue
            if any(same_answer(t, v) for v in it.criteria.values()):
                continue
            new.append(t)
        if len(new) < n_neg:
            n_skip += 1
            out.append(it)
            continue
        crit, roles = dict(it.criteria), dict(it.roles)
        for t in new:
            key = f"u{len(crit)}"
            crit[key] = t
            roles[key] = "unrelated"
        out.append(Item(**{**it.__dict__, "criteria": crit, "roles": roles,
                           "meta": {**it.meta, "unrelated_added": len(new)}}))
        n_aug += 1
    return out, {"augmented": n_aug, "skipped_no_clean_negative": n_skip}


def review_pack(items: list[Item], n: int = 100, seed: int = 0) -> str:
    """Markdown pack of n random items for human keep/fix/drop review."""
    rng = random.Random(seed)
    pick = rng.sample(items, min(n, len(items)))
    lines = ["| # | id | decision (keep/fix/drop) | note |", "|---|---|---|---|"]
    body = []
    for i, it in enumerate(pick, 1):
        lines.append(f"| {i} | `{it.id}` |  |  |")
        opts = "\n".join(f"  - `{k}`{' ← gold' if k == it.gold else ''}{f' [{it.roles[k]}]' if k in it.roles else ''}: "
                         f"{v[:200]}" for k, v in it.criteria.items())
        body.append(f"### {i}. `{it.id}` ({it.qtype}, labels: {it.label_source})\n**State:** {it.state[:800]}\n\n"
                    f"**Question:** {it.instructions[:400]}\n\n**Options:**\n{opts}\n\n**Target:** {it.target}\n")
    return "\n".join(lines) + "\n\n---\n\n" + "\n".join(body)
