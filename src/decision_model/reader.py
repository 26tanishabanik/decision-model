"""Joint reader: every option of a question reads every other option before it is scored.

Input: backbone states of the option tokens [B, To, H]. Linear(H -> HEAD_D) + question-type embedding, then 2
bidirectional pre-norm Transformer layers over all option tokens of the question (padding masked).
"""
from __future__ import annotations

import torch.nn as nn

HEAD_D = 1024
N_TYPES = 3                       # choice / score / yes_no
BUCKETS = (2, 5, 10, 16, 1024)    # option-count buckets for temperatures: 2 | 3-5 | 6-10 | 11-16 | 17+
NEG = -1e4


def bucket(k: int) -> int:
    for i, b in enumerate(BUCKETS):
        if k <= b:
            return i
    return len(BUCKETS) - 1


class JointReader(nn.Module):
    def __init__(self, hidden: int, d: int = HEAD_D, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden, d)
        self.type_emb = nn.Embedding(N_TYPES, d)
        layer = nn.TransformerEncoderLayer(d, d // 64, 4 * d, dropout, batch_first=True, norm_first=True)
        self.extra = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, opt_h, opt_pad, qtype):
        """opt_h [B,To,H] option-token states; opt_pad [B,To] True = padding."""
        W = self.down(opt_h) + self.type_emb(qtype)[:, None, :]
        return self.extra(W, src_key_padding_mask=opt_pad)
