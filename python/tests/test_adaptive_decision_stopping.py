"""A Thompson-sampling trial's declared posterior expected-loss rule, read at an interim look.

Reference and tolerance. With a Beta(1, 1) prior and Bernoulli outcomes the arm posteriors are
``Beta(1 + successes, 1 + failures)``. The probability that the second arm is best and the
expected loss of choosing an arm, ``E[(theta_other - theta_chosen)^+]``, are computed here in
pure Python (the Evan Miller finite sum and Simpson integration of ``F_chosen (1 - F_other)`` in
``test_adaptive_inference``) and are independent of the library. The rule stops exactly when the
best arm's expected loss is below the declared threshold, so a threshold at twice the reference
loss stops and one at half of it does not. The reference loss carries Simpson error well below
``1e-7`` (2000 panels of a smooth polynomial), which is the tolerance. The rule is a decision
procedure, not an error-rate guarantee: the result states the unsupported claim and offers no
interval.
"""

from __future__ import annotations

import pytest
from antecedent import _reason_details as reason_details
from antecedent import adaptive_inference as ai
from antecedent.outcomes import require_success

from test_adaptive_inference import (
    counts_of,
    expected_loss,
    p_second_exceeds_first,
    posterior_shapes,
    prepare,
    replace,
    thompson_design,
)


def design_with(rule: ai.StoppingRule) -> ai.AdaptiveDesign:
    return replace(thompson_design(), stopping="decision_theoretic", stopping_rule=rule)


def reference(look_units: int) -> tuple[int, float, float]:
    """Best arm, its probability of being best and its expected loss, from the raw counts."""
    design = thompson_design()
    counts = counts_of(
        list(design.assigned_arm), [list(o) for o in design.outcomes], list(range(look_units))
    )
    shapes = posterior_shapes(counts)
    p_second = p_second_exceeds_first(shapes[0], shapes[1])
    best = 1 if p_second > 0.5 else 0
    other = 1 - best
    return best, max(p_second, 1.0 - p_second), expected_loss(shapes[best], shapes[other])


def test_the_expected_loss_rule_stops_exactly_when_the_best_arms_loss_is_below_the_threshold():
    best, probability_best, loss = reference(60)
    assert 0.0 < loss < 0.5

    generous = require_success(
        prepare(design_with(ai.StoppingRule.posterior_expected_loss(2.0 * loss))).analyse(
            ai.PosteriorStopping(look=1)
        )
    )
    assert generous.paradigm == "decision_theoretic_stopping"
    assert generous.claim == "point_only"
    summary = generous.payload
    assert summary.best_arm == best
    assert summary.probability_best == pytest.approx(probability_best, abs=1e-7)
    assert summary.expected_loss == pytest.approx(loss, abs=1e-7)
    assert summary.stop is True
    # A decision rule is not an error-rate guarantee, and the result says so.
    assert generous.unsupported_claim is not None
    assert generous.stopping_rule is not None

    strict = require_success(
        prepare(design_with(ai.StoppingRule.posterior_expected_loss(0.5 * loss))).analyse(
            ai.PosteriorStopping(look=1)
        )
    )
    assert strict.payload.best_arm == best
    assert strict.payload.expected_loss == pytest.approx(loss, abs=1e-7)
    assert strict.payload.stop is False


def test_the_first_look_reads_only_the_units_available_by_then():
    best, probability_best, loss = reference(30)
    early = require_success(
        prepare(design_with(ai.StoppingRule.posterior_expected_loss(2.0 * loss))).analyse(
            ai.PosteriorStopping(look=0)
        )
    )
    assert early.look == 0
    assert early.payload.best_arm == best
    assert early.payload.probability_best == pytest.approx(probability_best, abs=1e-7)
    assert early.payload.expected_loss == pytest.approx(loss, abs=1e-7)
    assert early.payload.stop is True
    # The same history at its second look has strictly more evidence, so the loss moves.
    later = reference(60)[2]
    assert later != pytest.approx(loss, abs=1e-6)


def test_a_cell_without_the_decision_theoretic_paradigm_refuses_the_request():
    # The efficacy cell of the same Thompson design supports the posterior and the anytime-valid
    # sequence, not the decision-theoretic reading; a guarantee never transfers across paradigms.
    session = prepare(thompson_design())
    with pytest.raises(ai.AdaptiveInferenceRefusal) as caught:
        require_success(session.analyse(ai.PosteriorStopping(look=1)))
    assert caught.value.reason_code == "route_not_supported"
    assert caught.value.detail == reason_details.adaptive_inference.PARADIGM_NOT_SUPPORTED_FOR_CELL
