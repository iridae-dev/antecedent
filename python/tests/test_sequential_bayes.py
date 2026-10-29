"""Sequential Bayes: posterior artifact → next prior (P1-C)."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalSerializationError
from antecedent.outcomes import require_success


def _confounded(n: int = 160, seed: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def _ols_effect(data):
    """Least squares coefficient of t in y ~ 1 + t + z and its standard error."""
    design = np.column_stack([np.ones(len(data["y"])), data["t"], data["z"]])
    beta = np.linalg.lstsq(design, data["y"], rcond=None)[0]
    residual = data["y"] - design @ beta
    sigma2 = residual @ residual / (len(residual) - 3)
    return beta[1], float(np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1]))


def test_sequential_prior_from_artifact():
    data_a, edges = _confounded(seed=1)
    data_b, _ = _confounded(seed=2)
    query = antecedent.AverageEffect(treatment="t", outcome="y")

    a = require_success(
        antecedent.analyze(
            data_a,
            graph=edges,
            query=query,
            inference=antecedent.Bayesian(n_draws=4000, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert a.posterior is not None
    artifact = bytes(a.export(quantity="posterior"))

    flat_b = require_success(
        antecedent.analyze(
            data_b,
            graph=edges,
            query=query,
            inference=antecedent.Bayesian(n_draws=4000, backend="conjugate", prior_scale=1e4),
            refute=False,
            execution=antecedent.Execution(seed=2),
        )
    )
    b = require_success(
        antecedent.analyze(
            data_b,
            graph=edges,
            query=query,
            inference=antecedent.Bayesian(n_draws=4000, backend="conjugate", prior_from=artifact),
            refute=False,
            execution=antecedent.Execution(seed=2),
        )
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
    # Independent references. A flat prior returns the least-squares fit of batch B (observed
    # 2.0332 vs 2.0328, sd 2 percent off). Batch A's posterior is a (diagonal) normal prior, so
    # the sequential fit lands on the pooled A+B least squares (observed 1.9865 vs 1.9869, sd
    # 4 percent off) and is more precise than B alone.
    beta_b, se_b = _ols_effect(data_b)
    pooled = {key: np.concatenate([data_a[key], data_b[key]]) for key in data_a}
    beta_pooled, se_pooled = _ols_effect(pooled)
    assert flat_b.posterior.effect_mean == pytest.approx(beta_b, abs=0.01)
    assert flat_b.posterior.effect_sd == pytest.approx(se_b, rel=0.08)
    assert b.posterior.effect_mean == pytest.approx(beta_pooled, abs=0.01)
    assert b.posterior.effect_sd == pytest.approx(se_pooled, rel=0.08)
    assert b.posterior.effect_sd < flat_b.posterior.effect_sd


def test_prior_and_data_both_influence_posterior():
    """Fail if the prior is ignored or the data is ignored (tests-quality-7)."""
    data, edges = _confounded(n=200, seed=11)
    query = antecedent.AverageEffect(treatment="t", outcome="y")

    flat = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            inference=antecedent.Bayesian(n_draws=200, backend="conjugate", prior_scale=1e4),
            refute=False,
            execution=antecedent.Execution(seed=7),
        )
    )
    strong = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=query,
            inference=antecedent.Bayesian(n_draws=200, backend="conjugate", prior_scale=0.05),
            refute=False,
            execution=antecedent.Execution(seed=7),
        )
    )
    assert flat.posterior is not None and strong.posterior is not None
    m_flat = flat.posterior.effect_mean
    m_strong = strong.posterior.effect_mean
    # Independent references: the flat fit is least squares; the N(0, 0.05^2) prior on every
    # coefficient with unit noise precision is a ridge fit with penalty 1 / 0.05^2 (observed
    # 0.4235 vs 0.4225).
    design = np.column_stack([np.ones(200), data["t"], data["z"]])
    ols = np.linalg.lstsq(design, data["y"], rcond=None)[0][1]
    ridge = np.linalg.solve(
        design.T @ design + np.eye(3) / 0.05**2, design.T @ data["y"]
    )[1]
    assert m_flat == pytest.approx(ols, abs=0.01)
    assert m_strong == pytest.approx(ridge, abs=0.01)
    # Strong isotropic prior at 0 must pull the ATE toward 0 vs flat.
    assert abs(m_flat - m_strong) > 0.1, f"prior ignored? flat={m_flat} strong={m_strong}"
    assert abs(m_strong) < abs(m_flat), f"strong should shrink: flat={m_flat} strong={m_strong}"
    # Informative data must move the strong prior off its mean (0).
    assert abs(m_strong) > 0.15, f"data ignored? strong posterior {m_strong} ≈ 0"


def test_sequential_prior_named_subset_when_design_shrinks():
    """Named coefficient priors apply to the overlapping subspace when Z is dropped."""
    data, edges = _confounded()
    a = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=48, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=3),
        )
    )
    assert a.posterior is not None
    artifact = bytes(a.export(quantity="posterior"))
    names = list(require_success(antecedent.load(artifact)).quantity_names)
    assert "coef_z" in names

    data2 = {"t": data["t"], "y": data["y"]}
    b = require_success(
        antecedent.analyze(
            data2,
            graph=[("t", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=48, backend="conjugate", prior_from=artifact),
            refute=False,
            execution=antecedent.Execution(seed=4),
        )
    )
    assert b.posterior is not None
    assert np.isfinite(b.posterior.effect_mean)
    # The overlapping coefficients (intercept, t) carry A's posterior as a prior. Without z the
    # flat fit is confounded (the naive slope, 3.23); A's t coefficient is 1.90 +- 0.06, so the
    # sequential posterior sits strictly between the two and is more precise than the flat one.
    flat = require_success(
        antecedent.analyze(
            data2,
            graph=[("t", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=48, backend="conjugate"),
            refute=False,
            execution=antecedent.Execution(seed=4),
        )
    )
    naive = np.polyfit(data["t"], data["y"], 1)[0]
    assert flat.posterior.effect_mean == pytest.approx(naive, abs=0.3)
    assert a.posterior.effect_mean + 0.2 < b.posterior.effect_mean < flat.posterior.effect_mean - 0.2
    assert b.posterior.effect_sd < flat.posterior.effect_sd


def test_sequential_prior_rejects_corrupt_artifact():
    data, edges = _confounded()
    with pytest.raises(CausalSerializationError, match="bad artifact magic"):
        require_success(
            antecedent.analyze(
                data,
                graph=edges,
                query=antecedent.AverageEffect(treatment="t", outcome="y"),
                inference=antecedent.Bayesian(n_draws=48, prior_from=b"not-a-posterior"),
                refute=False,
                execution=antecedent.Execution(seed=4),
            )
        )
