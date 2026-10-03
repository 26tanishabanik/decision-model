import numpy as np
from sklearn.metrics import roc_auc_score

from decision_model.metrics import auroc


def test_fast_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 200)
        s = np.round(rng.normal(size=200) + 0.3 * y, 1)  # rounding forces ties
        assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-12
