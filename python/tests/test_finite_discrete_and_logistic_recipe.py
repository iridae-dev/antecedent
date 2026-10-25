"""The finite-discrete adjustment contrast and the logistic g-computation recipe.

Oracles are worked out independently of the library: the adjustment contrast of an eight-cell
table is ``sum_z P(z) (P(y=1 | t=1, z) - P(y=1 | t=0, z))`` computed with numpy, and the logistic
fit of a saturated two-by-two trial has the closed form log odds and Wald variance of its cells.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import finite_discrete as fd
from antecedent import recipe_glm as rg
from antecedent.errors import CausalError, CausalTypeError, CausalValueError

# Cell 4t + 2z + y.
COUNTS = [10, 10, 5, 15, 4, 16, 2, 18]
OTHER = [10, 10, 5, 15, 8, 12, 2, 18]


def contrast(counts: list[int]) -> float:
    cells = np.asarray(counts, dtype=float).reshape(2, 2, 2)  # [t][z][y]
    p_z = cells.sum(axis=(0, 2)) / cells.sum()
    p_y1 = cells[:, :, 1] / cells.sum(axis=2)
    return float(((p_y1[1] - p_y1[0]) * p_z).sum())


def test_the_plug_in_contrast_is_the_adjustment_formula_over_the_table():
    assert contrast(COUNTS) == pytest.approx(0.225, abs=1e-12)
    result = fd.finite_discrete_contrast(COUNTS, [(COUNTS, 1.0)])
    assert result.plug_in_effect == pytest.approx(contrast(COUNTS), abs=1e-9)
    assert result.sample_size == sum(COUNTS)
    assert result.plan_compilations >= 1


def test_the_posterior_cells_are_the_weighted_contrast_of_the_draws():
    draws = [(COUNTS, 0.25), (OTHER, 0.75)]
    result = fd.finite_discrete_contrast(COUNTS, draws)
    per_draw = [contrast(COUNTS), contrast(OTHER)]
    assert per_draw[0] != pytest.approx(per_draw[1], abs=1e-3)
    assert result.posterior_mean == pytest.approx(0.25 * per_draw[0] + 0.75 * per_draw[1], abs=1e-9)
    assert min(per_draw) - 1e-9 <= result.posterior_lower <= result.posterior_mean
    assert result.posterior_mean <= result.posterior_upper <= max(per_draw) + 1e-9
    assert result.plug_in_effect == pytest.approx(contrast(COUNTS), abs=1e-9)


def test_tables_and_weights_that_cannot_be_planned_are_refused():
    with pytest.raises(CausalError):
        fd.finite_discrete_contrast([0] * 8, [(COUNTS, 1.0)])
    with pytest.raises(CausalError):
        fd.finite_discrete_contrast(COUNTS, [(COUNTS, 0.5)])  # weights do not sum to one
    with pytest.raises(CausalError):
        fd.finite_discrete_contrast(COUNTS, [])
    with pytest.raises(CausalValueError):
        fd.finite_discrete_contrast(COUNTS[:7], [(COUNTS, 1.0)])
    with pytest.raises(CausalValueError):
        fd.finite_discrete_contrast(COUNTS, [(COUNTS, float("nan"))])


def trial() -> dict[str, np.ndarray]:
    """100 control rows with 30 responders and 100 treated rows with 60."""
    treatment = np.repeat([0.0, 1.0], 100)
    within = np.tile(np.arange(100), 2)
    responders = np.where(treatment == 0.0, 30, 60)
    return {"t": treatment, "y": (within < responders).astype(float)}


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def test_the_provider_declares_what_the_recipe_requires():
    provider = rg.logistic_provider(7, "logistic")
    assert provider.id == 7
    assert provider.name == "logistic"
    assert {"differentiable_log_density", "estimating_equations"} <= set(provider.capabilities)
    with pytest.raises(CausalError):
        rg.logistic_provider(1, "   ")
    with pytest.raises(CausalTypeError):
        rg.logistic_provider(-1, "logistic")


def test_the_recipe_fit_recovers_the_two_by_two_log_odds_and_risk_difference():
    fit = rg.logistic_standardized_effect(
        trial(), treatment="t", outcome="y", provider=rg.logistic_provider(1, "logistic")
    )
    assert fit.coefficients[0] == pytest.approx(logit(0.3), abs=1e-8)
    assert fit.coefficients[1] == pytest.approx(logit(0.6) - logit(0.3), abs=1e-8)
    assert fit.treated_mean == pytest.approx(0.6, abs=1e-9)
    assert fit.control_mean == pytest.approx(0.3, abs=1e-9)
    assert fit.effect == pytest.approx(0.3, abs=1e-9)
    assert fit.rows == 200
    assert fit.covariance.shape == (2, 2)
    # Var(beta_1) = 1 / (n1 p1 (1 - p1)) + 1 / (n0 p0 (1 - p0)) for the saturated model.
    variance = 1.0 / (100 * 0.6 * 0.4) + 1.0 / (100 * 0.3 * 0.7)
    assert fit.covariance[1, 1] == pytest.approx(variance, rel=1e-6)
    assert np.allclose(fit.covariance, fit.covariance.T)


def test_an_adjuster_widens_the_design_and_keeps_the_standardized_means_between_zero_and_one():
    rng = np.random.default_rng(5)
    z = rng.integers(0, 2, size=400).astype(float)
    t = rng.integers(0, 2, size=400).astype(float)
    y = (rng.random(400) < 0.2 + 0.3 * t + 0.1 * z).astype(float)
    fit = rg.logistic_standardized_effect(
        {"t": t, "y": y, "z": z},
        treatment="t",
        outcome="y",
        adjustment=["z"],
        provider=rg.logistic_provider(1, "logistic"),
    )
    assert len(fit.coefficients) == 3
    assert fit.covariance.shape == (3, 3)
    assert 0.0 < fit.control_mean < fit.treated_mean < 1.0
    assert fit.effect == pytest.approx(fit.treated_mean - fit.control_mean, abs=1e-12)


def test_unusable_data_and_providers_are_refused_before_a_fit():
    provider = rg.logistic_provider(1, "logistic")
    data = trial()
    with pytest.raises(CausalError):
        rg.logistic_standardized_effect(
            {**data, "y": data["y"] * 2.0}, treatment="t", outcome="y", provider=provider
        )
    missing = data["t"].copy()
    missing[3] = float("nan")
    with pytest.raises(CausalError):
        rg.logistic_standardized_effect(
            {**data, "t": missing}, treatment="t", outcome="y", provider=provider
        )
    with pytest.raises(CausalValueError):
        rg.logistic_standardized_effect(data, treatment="t", outcome="absent", provider=provider)
    with pytest.raises(CausalTypeError):
        rg.logistic_standardized_effect(
            data,
            treatment="t",
            outcome="y",
            provider="logistic",  # type: ignore[arg-type]
        )
