"""Losses, and the fixed dev / calibration samples used during training."""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from decision_model.reader import NEG


def smooth_one_hot(tgt, opt_mask, eps):
    """(1 - eps) * one-hot + eps / k over the k options read, for rows whose target is one-hot; other rows unchanged."""
    if not eps:
        return tgt
    one_hot = (tgt.max(-1).values > 1 - 1e-6) & (tgt.sum(-1) > 0)
    k = opt_mask.sum(-1, keepdim=True).clamp(min=1).float()
    sm = (1 - eps) * tgt + eps * opt_mask.float() / k
    return torch.where(one_hot[:, None], sm, tgt)


def rps_sum(logits, tgt, opt_mask, is_score):
    """Ranked probability score summed over learnable `score` rows: sum_j (CDF_p - CDF_t)^2 / (k - 1), in level order.
    Zero for other question types."""
    ok = is_score & (tgt.sum(-1) > 0)
    if not ok.any():
        return logits.new_zeros(())
    p = torch.softmax(logits.masked_fill(~opt_mask, NEG), -1)[ok]
    t, m = tgt[ok], opt_mask[ok].float()
    k = m.sum(-1).clamp(min=2)
    return ((((torch.cumsum(p, -1) - torch.cumsum(t, -1)) ** 2) * m).sum(-1) / (k - 1)).sum()


def soft_ce_sum(logits, tgt, opt_mask):
    """-> (summed loss over learnable rows, number of learnable rows); rows with zero target mass are excluded."""
    logp = F.log_softmax(logits.masked_fill(~opt_mask, NEG), -1)
    ok = tgt.sum(-1) > 0
    return -(tgt[ok] * logp[ok]).sum(), int(ok.sum())


def fixed_sample(pools, per_source, seed):
    """At most `per_source` items per source file, the same for every run with this seed."""
    out = []
    for f, items in sorted(pools.items()):
        out += random.Random(f"{seed}-{f}").sample(items, min(per_source, len(items)))
    return out
