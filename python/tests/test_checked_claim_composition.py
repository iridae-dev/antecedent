"""Checked synthesis of completed average-effect studies into one conditional point claim.

Hand example. A dose ``t`` observed at 0, 1 and 2 with ``y = 3 t`` plus a zero-mean
perturbation: the dose-1 and dose-2 average effects are about 3 and 6. The contrast
``1 * ate(1) - 1 * ate(2)`` is about -3 and the aggregate ``0.25 * ate(1) + 0.75 * ate(2)``
about 5.25; each is exactly the declared linear combination of the parents' points.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent import program_claims
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalTypeError,
    CausalValueError,
)
from antecedent.outcomes import require_success
from antecedent.program_claims import CheckedClaimLimits, ClaimParent, compose_study_claims

N = 300


def _data():
    t = np.tile([0.0, 1.0, 2.0], N // 3)
    y = 3.0 * t + 0.1 * ((np.arange(N) % 5) - 2.0)
    return {"t": t, "y": y}


def _prepared(level):
    return ant.prepare(
        _data(),
        graph=[("t", "y")],
        query=ant.AverageEffect("t", "y", active_level=level),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=17, threads=1),
    )


def _completed(level):
    outcome = ant.estimate(_prepared(level))
    return outcome, require_success(outcome).ate


def _pair():
    (first, a), (second, b) = _completed(1.0), _completed(2.0)
    return first, second, a, b


@pytest.mark.parametrize(
    ("operation", "weights"), [("contrast", (1.0, -1.0)), ("aggregate", (0.25, 0.75))]
)
def test_a_composition_is_exactly_the_declared_linear_combination_of_its_parents(
    operation, weights
):
    first, second, a, b = _pair()
    parents = [ClaimParent(first, weights[0]), ClaimParent(second, weights[1])]
    derived = compose_study_claims(parents, operation=operation)
    assert derived.point == pytest.approx(weights[0] * a + weights[1] * b, abs=1e-12)
    assert len(set(derived.parents)) == 2 and derived.claim_id not in derived.parents
    # A derived claim is conditional and in-process: it stamps no identification of its own.
    assert derived.identification == "Unknown"
    assert "portable_export" in derived.unavailable_operations
    assert derived.provenance_valid
    assert all(len(h) == 64 for h in (derived.claim_id, derived.proof_identity))


def test_the_derived_claim_does_not_depend_on_the_order_of_the_parents():
    first, second, _, _ = _pair()
    parents = [ClaimParent(first, 0.25), ClaimParent(second, 0.75)]
    forward = compose_study_claims(parents, operation="aggregate")
    reverse = compose_study_claims(parents[::-1], operation="aggregate")
    assert (forward.proof_identity, forward.claim_id, forward.parents) == (
        reverse.proof_identity,
        reverse.claim_id,
        reverse.parents,
    )
    assert forward.point == reverse.point
    # The studies themselves are untouched by being composed.
    assert math.isfinite(require_success(first).ate)


def test_a_prepared_study_that_was_estimated_is_a_parent_as_its_outcome_is():
    first, second, _, _ = _pair()
    by_outcome = compose_study_claims(
        [ClaimParent(first, 1.0), ClaimParent(second, -1.0)], operation="contrast"
    )
    by_study = compose_study_claims(
        [ClaimParent(first.study, 1.0), ClaimParent(second.study, -1.0)], operation="contrast"
    )
    assert by_outcome.point == by_study.point


def test_coefficients_or_output_weights_the_parents_do_not_realize_are_refused():
    first, second, _, _ = _pair()
    with pytest.raises(CausalValueError):
        # The arithmetic would compute 1*a + 1*b, but the declared output is a - b.
        compose_study_claims(
            [
                ClaimParent(first, 1.0, output_weight=1.0),
                ClaimParent(second, 1.0, output_weight=-1.0),
            ],
            operation="contrast",
        )
    with pytest.raises(CausalValueError):
        # The arithmetic is a - b but the declared output is the 0.25/0.75 aggregate.
        compose_study_claims(
            [
                ClaimParent(first, 1.0, output_weight=0.25),
                ClaimParent(second, -1.0, output_weight=0.75),
            ],
            operation="contrast",
        )


def test_a_parent_without_a_completed_estimate_or_a_duplicate_parent_is_refused():
    first, second, _, _ = _pair()
    with pytest.raises(CausalValueError, match="completed estimate"):
        compose_study_claims(
            [ClaimParent(_prepared(1.0), 1.0), ClaimParent(second, -1.0)], operation="contrast"
        )
    with pytest.raises(CausalValueError):
        compose_study_claims(
            [ClaimParent(first, 0.5), ClaimParent(first, 0.5)], operation="aggregate"
        )


def test_a_parent_that_is_not_an_average_effect_contrast_is_refused():
    rng = np.random.default_rng(61)
    z = rng.normal(size=500)
    x = 0.5 * z + rng.normal(size=500)
    y = 1 + 2 * x + z + rng.normal(scale=0.3, size=500)
    derivative = ant.estimate(
        ant.prepare(
            {"x": x, "z": z, "y": y},
            graph=[("z", "x"), ("z", "y"), ("x", "y")],
            query=ant.PointDerivative("x", "y", at=0),
            estimator_config={"bandwidth": 0.6},
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
            execution=ant.Execution(seed=61, threads=1),
        )
    )
    first, _ = _completed(1.0)
    with pytest.raises(CausalValueError, match="contrast"):
        compose_study_claims(
            [ClaimParent(first, 1.0), ClaimParent(derivative, -1.0)], operation="contrast"
        )


def test_declarations_are_typed_before_any_native_work():
    first, second, _, _ = _pair()
    pair = [ClaimParent(first, 1.0), ClaimParent(second, -1.0)]
    with pytest.raises(CausalValueError, match="operation"):
        compose_study_claims(pair, operation="compare")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        compose_study_claims([], operation="contrast")
    with pytest.raises(CausalTypeError):
        compose_study_claims([first, second], operation="contrast")  # type: ignore[list-item]
    with pytest.raises(CausalTypeError):
        compose_study_claims(pair, operation="contrast", limits=object())  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        compose_study_claims(pair, operation="contrast", execution=object())  # type: ignore
    with pytest.raises(CausalTypeError):
        ClaimParent(first, True)
    with pytest.raises(CausalValueError):
        ClaimParent(first, math.nan)
    with pytest.raises(CausalValueError):
        ClaimParent(first, 1.0, output_weight=math.inf)
    with pytest.raises(CausalValueError):
        CheckedClaimLimits(max_terms=-1)
    with pytest.raises(CausalTypeError, match="completed prepared study"):
        compose_study_claims(
            [ClaimParent(object(), 1.0), ClaimParent(second, -1.0)], operation="contrast"
        )


def test_the_composition_runs_under_the_callers_budget_and_cancellation():
    first, second, _, _ = _pair()
    pair = [ClaimParent(first, 1.0), ClaimParent(second, -1.0)]
    with pytest.raises(CausalResourceError):
        compose_study_claims(
            pair, operation="contrast", limits=CheckedClaimLimits(max_memory_bytes=1)
        )
    tight = ant.Execution(limits=ant.ResourceLimits(memory_bytes=1))
    with pytest.raises(CausalResourceError):
        compose_study_claims(pair, operation="contrast", execution=tight)
    token = ant.state.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        compose_study_claims(pair, operation="contrast", execution=ant.Execution(cancel=token))


def test_the_surface_is_exported():
    for name in ("ClaimParent", "CheckedStudyClaim", "CheckedClaimLimits", "compose_study_claims"):
        assert name in program_claims.__all__
