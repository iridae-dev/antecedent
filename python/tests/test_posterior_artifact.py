"""posterior artifact encode/decode round-trip from Python."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("causal")
import causal


def _confounded_scm(n: int = 400, seed: int = 11):
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return ["t", "y", "z"], [t, y, z], [("z", "t"), ("z", "y"), ("t", "y")]


def test_posterior_artifact_round_trip():
    names, cols, edges = _confounded_scm()
    result = causal.analyze_ate(
        names,
        cols,
        edges,
        treatment="t",
        outcome="y",
        inference="bayesian",
        n_draws=128,
        seed=3,
        refute=False,
    )
    assert result.posterior_artifact is not None
    assert result.posterior_n_draws == 128
    assert result.posterior_effect_mean is not None

    art = causal.decode_posterior_artifact(result.posterior_artifact)
    assert art.n_draws == 128
    effect_idx = art.quantity_names.index("ate") if "ate" in art.quantity_names else -1
    assert abs(art.mean[effect_idx] - result.posterior_effect_mean) < 1e-12
    assert art.backend_id
    assert len(art.draws) == art.n_draws * len(art.quantity_names)

    again = causal.encode_posterior_artifact(art)
    art2 = causal.decode_posterior_artifact(again)
    assert art2.n_draws == art.n_draws
    assert art2.draws == art.draws
    assert art2.mean == art.mean
    assert abs(art2.mean[effect_idx] - 2.0) < 0.5
