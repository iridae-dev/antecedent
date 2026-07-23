"""posterior artifact encode/decode + no-default-artifact (BACKLOG E)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


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
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_bayesian_default_omits_posterior_artifact():
    data, edges = _confounded_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=128),
        seed=3,
        refute=False,
    )
    assert result.posterior is not None
    assert result.posterior.artifact is None
    assert result.posterior.n_draws == 128
    assert result.posterior.effect_mean is not None
    assert math.isfinite(result.posterior.effect_mean)


def test_posterior_artifact_round_trip_opt_in():
    data, edges = _confounded_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=128),
        seed=3,
        refute=False,
        return_posterior_artifact=True,
    )
    assert result.posterior is not None
    assert result.posterior.artifact is not None
    assert result.posterior.n_draws == 128
    assert result.posterior.effect_mean is not None

    art = antecedent.decode_posterior_artifact(result.posterior.artifact)
    assert art.n_draws == 128
    effect_idx = art.quantity_names.index("ate") if "ate" in art.quantity_names else -1
    assert abs(art.mean[effect_idx] - result.posterior.effect_mean) < 1e-12
    assert art.backend_id
    assert len(art.draws) == art.n_draws * len(art.quantity_names)

    again = antecedent.encode_posterior_artifact(art)
    art2 = antecedent.decode_posterior_artifact(again)
    assert art2.n_draws == art.n_draws
    assert art2.draws == art.draws
    assert art2.mean == art.mean
    assert abs(art2.mean[effect_idx] - 2.0) < 0.5


def test_posterior_artifact_payload_size_vs_summaries():
    data, edges = _confounded_scm(n=200, seed=5)
    summary = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=2000),
        seed=5,
        refute=False,
    )
    full = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=2000),
        seed=5,
        refute=False,
        return_posterior_artifact=True,
    )
    assert summary.posterior.artifact is None
    assert full.posterior.artifact is not None
    art = antecedent.decode_posterior_artifact(full.posterior.artifact)
    assert len(art.draws) == art.n_draws * len(art.quantity_names)
    assert len(art.draws) > 0
    assert len(full.posterior.artifact) > len(art.mean) * 8
