"""Sequential Bayes: posterior artifact → next prior (P1-C)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded(n: int = 160, seed: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_sequential_prior_from_artifact():
    data_a, edges = _confounded(seed=1)
    data_b, _ = _confounded(seed=2)
    query = causal.AverageEffect(treatment="t", outcome="y")

    a = causal.analyze(
        data_a,
        graph=edges,
        query=query,
        inference=causal.Bayesian(n_draws=64),
        refute=False,
        seed=1,
        return_posterior_artifact=True,
    )
    assert a.posterior is not None
    artifact = bytes(a.posterior.artifact)

    b = causal.analyze(
        data_b,
        graph=edges,
        query=query,
        inference=causal.Bayesian(n_draws=64, prior_from=artifact),
        refute=False,
        seed=2,
    )
    assert b.posterior is not None
    assert np.isfinite(b.posterior.effect_mean)
    assert b.identification.assumption_count >= 1


def test_sequential_prior_named_subset_when_design_shrinks():
    """Named coefficient priors apply to the overlapping subspace when Z is dropped."""
    data, edges = _confounded()
    a = causal.analyze(
        data,
        graph=edges,
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=48),
        refute=False,
        seed=3,
        return_posterior_artifact=True,
    )
    assert a.posterior is not None
    artifact = bytes(a.posterior.artifact)
    names = list(causal.decode_posterior_artifact(artifact).quantity_names)
    assert "coef_z" in names

    data2 = {"t": data["t"], "y": data["y"]}
    b = causal.analyze(
        data2,
        graph=[("t", "y")],
        query=causal.AverageEffect(treatment="t", outcome="y"),
        inference=causal.Bayesian(n_draws=48, prior_from=artifact),
        refute=False,
        seed=4,
    )
    assert b.posterior is not None
    assert np.isfinite(b.posterior.effect_mean)


def test_sequential_prior_rejects_corrupt_artifact():
    data, edges = _confounded()
    with pytest.raises(Exception, match="(?i)artifact|posterior|cbor|format|magic"):
        causal.analyze(
            data,
            graph=edges,
            query=causal.AverageEffect(treatment="t", outcome="y"),
            inference=causal.Bayesian(n_draws=48, prior_from=b"not-a-posterior"),
            refute=False,
            seed=4,
        )
