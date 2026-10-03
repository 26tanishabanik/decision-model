"""Metrics with bootstrap CIs. All bootstraps resample examples (paired across methods when comparing)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

N_BOOT = 2000


def auroc(y_bin, score):
    """Mann-Whitney AUROC with average ranks for ties (same result as sklearn roc_auc_score)."""
    from scipy.stats import rankdata
    y_bin = np.asarray(y_bin)
    n1 = int(y_bin.sum())
    n0 = len(y_bin) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y_bin == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(fn, *arrays, n=N_BOOT, seed=0, alpha=0.05):
    """Percentile bootstrap CI of fn(*arrays) resampling rows jointly."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n_ex = len(arrays[0])
    point = fn(*arrays)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, n_ex, n_ex)
        v = fn(*[a[idx] for a in arrays])
        if np.isfinite(v):
            stats.append(v)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def paired_boot_diff(fn, y, s_a, s_b, n=N_BOOT, seed=0):
    """CI and two-sided bootstrap p for fn(y, s_a) - fn(y, s_b)."""
    rng = np.random.default_rng(seed)
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    d0 = fn(y, s_a) - fn(y, s_b)
    ds = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        d = fn(y[idx], s_a[idx]) - fn(y[idx], s_b[idx])
        if np.isfinite(d):
            ds.append(d)
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"diff": float(d0), "lo": float(lo), "hi": float(hi), "p_boot": float(min(1.0, p))}


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro"))


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test on discordant correctness."""
    from scipy.stats import binomtest
    a, b = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p": float(p)}
