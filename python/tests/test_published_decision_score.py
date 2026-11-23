"""Scoring a decision demand from what a published answer carries."""

from __future__ import annotations

import pytest
from antecedent import decision
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success


def test_an_affine_posterior_demand_reads_the_posterior_mean_not_the_point_answer():
    score = require_success(
        decision.score_published_answer(
            "affine_posterior", mean=0.1, posterior_mean=0.4, slope=2.0, intercept=-0.5
        )
    )
    assert score.value == pytest.approx(2.0 * 0.4 - 0.5, abs=1e-15)
    assert score.carrier == "parameter_posterior"


def test_an_affine_mean_demand_reads_the_published_point_answer():
    score = require_success(
        decision.score_published_answer("affine_mean", mean=0.1, slope=2.0, intercept=-0.5)
    )
    assert score.value == pytest.approx(2.0 * 0.1 - 0.5, abs=1e-15)
    assert score.carrier == "mean"
    # A posterior mean that is also published is not read by a mean demand.
    again = require_success(
        decision.score_published_answer(
            "affine_mean", mean=0.1, posterior_mean=9.0, slope=2.0, intercept=-0.5
        )
    )
    assert again.value == score.value


def test_a_posterior_demand_on_an_answer_without_a_posterior_names_the_missing_carrier():
    with pytest.raises(decision.DecisionRefusal) as info:
        require_success(
            decision.score_published_answer("affine_posterior", mean=0.1, slope=2.0, intercept=-0.5)
        )
    assert info.value.offending == "parameter_posterior"
    assert info.value.reason_code == "distribution_meaning_mismatch"


@pytest.mark.parametrize("demand", ["nonlinear", "risk"])
def test_a_mean_cannot_carry_a_nonlinear_utility_or_an_outcome_risk(demand):
    with pytest.raises(decision.DecisionRefusal) as info:
        require_success(decision.score_published_answer(demand, mean=0.1, posterior_mean=0.4))
    assert info.value.offending == "outcome_law"


def test_malformed_demands_are_refused():
    with pytest.raises(CausalValueError, match='unknown decision demand "quadratic"'):
        require_success(decision.score_published_answer("quadratic", mean=0.1))  # type: ignore[arg-type]
    with pytest.raises(CausalValueError, match="needs a finite slope and a finite intercept"):
        require_success(decision.score_published_answer("affine_mean", mean=0.1, slope=1.0))
    with pytest.raises(CausalValueError, match="mean must be a finite number"):
        require_success(
            decision.score_published_answer(
                "affine_mean", mean=float("nan"), slope=1.0, intercept=0.0
            )
        )
