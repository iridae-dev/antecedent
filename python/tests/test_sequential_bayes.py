"""Sequential Bayes: posterior artifact → next prior (P1-C)."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest


def _confounded(n: int = 160, seed: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def test_sequential_prior_from_artifact():
    data_a, edges = _confounded(seed=1)
    data_b, _ = _confounded(seed=2)
    query = antecedent.AverageEffect(treatment="t", outcome="y")

    a = antecedent.analyze(
        data_a,
        graph=edges,
        query=query,
        inference=antecedent.Bayesian(n_draws=128, backend="conjugate"),
        refute=False,
        seed=1,
        return_posterior_artifact=True,
    )
    assert a.posterior is not None
    artifact = bytes(a.posterior.artifact)

    flat_b = antecedent.analyze(
        data_b,
        graph=edges,
        query=query,
        inference=antecedent.Bayesian(n_draws=128, backend="conjugate", prior_scale=1e4),
        refute=False,
        seed=2,
    )
    b = antecedent.analyze(
        data_b,
        graph=edges,
        query=query,
        inference=antecedent.Bayesian(n_draws=128, backend="conjugate", prior_from=artifact),
        refute=False,
        seed=2,
    )
    assert b.posterior is not None
    assert np.isfinite(b.posterior.effect_mean)
    assert b.identification.assumption_count >= 1
    # Diagonal / sequential disclosure must be visible (estimate-bayes-transport-response-2).
    assumption_text = " ".join(str(a) for a in (b.assumptions or [])).lower()
    assert "diagonal" in assumption_text or "sequential" in assumption_text

    # Sequential must move relative to a flat-prior fit on the same batch B
    # (prior ignored ⇒ means equal).
    assert flat_b.posterior is not None
    assert (
        abs(b.posterior.effect_mean - flat_b.posterior.effect_mean) > 1e-3
        or abs((b.posterior.effect_sd or 0.0) - (flat_b.posterior.effect_sd or 0.0)) > 1e-3
    )


def test_prior_and_data_both_influence_posterior():
    """Fail if the prior is ignored or the data is ignored (tests-quality-7)."""
    data, edges = _confounded(n=200, seed=11)
    query = antecedent.AverageEffect(treatment="t", outcome="y")

    flat = antecedent.analyze(
        data,
        graph=edges,
        query=query,
        inference=antecedent.Bayesian(n_draws=200, backend="conjugate", prior_scale=1e4),
        refute=False,
        seed=7,
    )
    strong = antecedent.analyze(
        data,
        graph=edges,
        query=query,
        inference=antecedent.Bayesian(n_draws=200, backend="conjugate", prior_scale=0.05),
        refute=False,
        seed=7,
    )
    assert flat.posterior is not None and strong.posterior is not None
    m_flat = flat.posterior.effect_mean
    m_strong = strong.posterior.effect_mean
    # Strong isotropic prior at 0 must pull the ATE toward 0 vs flat.
    assert abs(m_flat - m_strong) > 0.1, f"prior ignored? flat={m_flat} strong={m_strong}"
    assert abs(m_strong) < abs(m_flat), f"strong should shrink: flat={m_flat} strong={m_strong}"
    # Informative data must move the strong prior off its mean (0).
    assert abs(m_strong) > 0.15, f"data ignored? strong posterior {m_strong} ≈ 0"


def test_sequential_prior_named_subset_when_design_shrinks():
    """Named coefficient priors apply to the overlapping subspace when Z is dropped."""
    data, edges = _confounded()
    a = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=48, backend="conjugate"),
        refute=False,
        seed=3,
        return_posterior_artifact=True,
    )
    assert a.posterior is not None
    artifact = bytes(a.posterior.artifact)
    names = list(antecedent.inference.decode_posterior_artifact(artifact).quantity_names)
    assert "coef_z" in names

    data2 = {"t": data["t"], "y": data["y"]}
    b = antecedent.analyze(
        data2,
        graph=[("t", "y")],
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Bayesian(n_draws=48, backend="conjugate", prior_from=artifact),
        refute=False,
        seed=4,
    )
    assert b.posterior is not None
    assert np.isfinite(b.posterior.effect_mean)


def test_sequential_prior_rejects_corrupt_artifact():
    data, edges = _confounded()
    with pytest.raises(Exception, match="(?i)artifact|posterior|cbor|format|magic"):
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=48, prior_from=b"not-a-posterior"),
            refute=False,
            seed=4,
        )
