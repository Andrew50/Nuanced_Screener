from __future__ import annotations

import numpy as np
import pandas as pd

from screener_loader.weak_supervision.label_model import fit_independent_label_model


def test_independent_label_model_recovers_reasonable_params() -> None:
    rng = np.random.default_rng(1337)
    n = 500
    pi_true = 0.3
    y = (rng.random(n) < pi_true).astype(int)

    def gen_votes(coverage: float, acc: float) -> np.ndarray:
        emit = rng.random(n) < coverage
        v = np.zeros(n, dtype=int)
        # When emitted, vote matches y with prob acc.
        correct = rng.random(n) < acc
        v_emit = np.where(y == 1, 1, -1)
        v_wrong = np.where(y == 1, -1, 1)
        v[emit] = np.where(correct[emit], v_emit[emit], v_wrong[emit])
        return v

    lf1 = gen_votes(coverage=0.8, acc=0.8)
    lf2 = gen_votes(coverage=0.6, acc=0.7)

    df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "lf1": lf1, "lf2": lf2})
    post, params = fit_independent_label_model(df, max_iter=50)

    assert not post.empty
    assert 0.05 <= params.class_prior <= 0.95
    # Should be directionally correct.
    assert abs(params.class_prior - pi_true) < 0.15
    assert params.accuracy["lf1"] >= 0.6
    assert params.accuracy["lf2"] >= 0.55

