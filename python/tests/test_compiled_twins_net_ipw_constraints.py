"""Policy capacity, budget and availability on the compiled net-value route.

``evaluate_policy`` checks a ``BinaryPolicy``'s ``capacity``, ``max_treatment_rate`` and
``budget`` (and the reference's) and the ``available`` rows before it evaluates the policy.
``analyze`` with a ``PolicyValue`` carrying the same policies and ``available`` rows and
``estimator="randomized.ipw_policy_net"`` accepts the same constraints: a policy that keeps them
has the same values to the bit, a policy that breaks one is refused with the same error, and the
compiled route itself receives and checks the limits.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import _prepare_route, policy
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.errors import CausalError
from antecedent.outcomes import require_success

N = 40
AVAILABLE = [i % 7 != 6 for i in range(N)]
ASSIGNMENT = [bool((i * 7) % 5 < 2) for i in range(N)]
OUTCOME = np.array([1.0 + 0.37 * (i % 11) + 0.13 * float(ASSIGNMENT[i]) for i in range(N)])
PROPENSITY = [0.3 + 0.01 * (i % 9) for i in range(N)]
DATA = {"y": OUTCOME}
SUBJECTS = [f"eval-{i}" for i in range(N)]
ACTIONS = [i % 3 != 0 and AVAILABLE[i] for i in range(N)]
REFERENCE_ACTIONS = [i % 4 == 0 and AVAILABLE[i] for i in range(N)]
TREATED = sum(ACTIONS)
REFERENCE_TREATED = sum(REFERENCE_ACTIONS)
POINTS = (
    "policy_value",
    "reference_value",
    "incremental_value",
    "relative_value_gap",
    "treatment_rate",
    "total_treatment_cost",
)


def _policies(**overrides):
    """Policy and reference that meet every limit with equality, with ``overrides`` applied."""
    limits = {
        "capacity": TREATED,
        "budget": 0.1 * TREATED + 1e-9,
        "reference_capacity": REFERENCE_TREATED,
        "reference_budget": 0.07 * REFERENCE_TREATED + 1e-9,
        "max_treatment_rate": None,
    }
    limits.update(overrides)
    chosen = policy.BinaryPolicy(
        ACTIONS,
        capacity=limits["capacity"],
        budget=limits["budget"],
        max_treatment_rate=limits["max_treatment_rate"],
        costs=0.1,
    )
    reference = policy.BinaryPolicy(
        REFERENCE_ACTIONS,
        capacity=limits["reference_capacity"],
        budget=limits["reference_budget"],
        costs=0.07,
    )
    return chosen, reference


def _direct(chosen, reference, available):
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_policy"):
        return policy.evaluate_policy(
            DATA,
            outcome="y",
            assignment=ASSIGNMENT,
            propensity=PROPENSITY,
            policy=chosen,
            reference=reference,
            available=available,
        )


def _compiled(chosen, reference, available):
    query = policy.PolicyValue(
        "y",
        ASSIGNMENT,
        PROPENSITY,
        chosen,
        SUBJECTS,
        reference=reference,
        available=available,
    )
    return require_success(
        ant.analyze(DATA, query=query, refute="none", estimator="randomized.ipw_policy_net")
    ).policy_value


def test_limits_that_hold_give_the_direct_values_bit_for_bit():
    chosen, reference = _policies()
    direct = _direct(chosen, reference, AVAILABLE)
    compiled = _compiled(chosen, reference, AVAILABLE)
    for name in POINTS:
        assert getattr(compiled, name) == getattr(direct, name), name


BROKEN = {
    "capacity": ({"capacity": TREATED - 1}, AVAILABLE),
    "budget": ({"budget": 0.1 * TREATED - 0.5}, AVAILABLE),
    "reference capacity": ({"reference_capacity": REFERENCE_TREATED - 1}, AVAILABLE),
    "reference budget": ({"reference_budget": 0.07 * REFERENCE_TREATED - 0.5}, AVAILABLE),
    "treatment rate": ({"max_treatment_rate": (TREATED - 1) / N}, AVAILABLE),
    "policy availability": ({}, [i != ACTIONS.index(True) for i in range(N)]),
    "reference availability": (
        {},
        [
            i != next(j for j in range(N) if REFERENCE_ACTIONS[j] and not ACTIONS[j])
            for i in range(N)
        ],
    ),
}


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_a_broken_limit_is_refused_by_both_paths_with_the_same_error(case):
    overrides, available = BROKEN[case]
    chosen, reference = _policies(**overrides)
    with pytest.raises((CausalError, ValueError)) as direct:
        _direct(chosen, reference, available)
    with pytest.raises((CausalError, ValueError)) as compiled:
        _compiled(chosen, reference, available)
    assert type(compiled.value) is type(direct.value)
    assert str(compiled.value) == str(direct.value)


@pytest.mark.parametrize("case", ["capacity", "budget", "treatment rate", "policy availability"])
def test_the_compiled_route_itself_checks_the_limits(case, monkeypatch):
    # Without the evaluator's own pre-check in analyze, the limits still reach the compiled
    # route, whose point-only kernel refuses the same policies.
    monkeypatch.setattr(_prepare_route, "_evaluate_policy_direct", lambda *_, **__: None)
    overrides, available = BROKEN[case]
    chosen, reference = _policies(**overrides)
    with pytest.raises(CausalError, match="exceeds the declared|unavailable"):
        _compiled(chosen, reference, available)
    held_policy, held_reference = _policies()
    compiled = _compiled(held_policy, held_reference, AVAILABLE)
    with pytest.warns(AntecedentDeprecationWarning):
        direct = policy.evaluate_policy(
            DATA,
            outcome="y",
            assignment=ASSIGNMENT,
            propensity=PROPENSITY,
            policy=held_policy,
            reference=held_reference,
            available=AVAILABLE,
        )
    for name in POINTS:
        assert getattr(compiled, name) == getattr(direct, name), name


def test_the_unconstrained_net_value_query_passes_no_limits():
    chosen = policy.BinaryPolicy(ACTIONS, costs=0.1)
    query = policy.PolicyValue("y", ASSIGNMENT, PROPENSITY, chosen, SUBJECTS)
    assert policy._net_policy_constraint_options(query, "randomized.ipw_policy_net") == {}
    constrained = policy.PolicyValue(
        "y", ASSIGNMENT, PROPENSITY, chosen, SUBJECTS, available=AVAILABLE
    )
    assert policy._net_policy_constraint_options(constrained, None) == {}
    assert policy._net_policy_constraint_options(constrained, "randomized.ipw_policy_net") == {
        "capacity": None,
        "budget": None,
        "reference_capacity": None,
        "reference_budget": None,
        "available": AVAILABLE,
    }


LABELS = ("control", "A", "B")
MULTI_ASSIGNMENT = [LABELS[(0, 1, 2, 0, 2)[i % 5]] for i in range(N)]
MULTI_RECOMMENDATIONS = [LABELS[(0, 1, 1, 2)[i % 4]] for i in range(N)]
MULTI_PROPENSITIES = [[0.4, 0.35, 0.25] if i % 2 == 0 else [0.5, 0.3, 0.2] for i in range(N)]
# Action B is available exactly where the policy recommends it.
MULTI_AVAILABLE = [[True, True, action == "B"] for action in MULTI_RECOMMENDATIONS]
MULTI_COSTS = [0.0, 0.15, 0.4]


def _multi_policy(capacities=None, budget=None):
    counts = [MULTI_RECOMMENDATIONS.count(label) for label in LABELS]
    total = sum(MULTI_COSTS[LABELS.index(action)] for action in MULTI_RECOMMENDATIONS)
    return policy.MultiActionPolicy(
        LABELS,
        MULTI_RECOMMENDATIONS,
        costs=MULTI_COSTS,
        capacities=counts if capacities is None else capacities,
        budget=total + 1e-9 if budget is None else budget,
    )


def _multi_direct(chosen, available):
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_multi_action_policy"):
        return policy.evaluate_multi_action_policy(
            DATA,
            outcome="y",
            assignment=MULTI_ASSIGNMENT,
            propensities=MULTI_PROPENSITIES,
            policy=chosen,
            available=available,
        )


def _multi_compiled(chosen, available):
    query = policy.MultiActionPolicyValue(
        "y", MULTI_ASSIGNMENT, MULTI_PROPENSITIES, chosen, SUBJECTS, available=available
    )
    return require_success(
        ant.analyze(
            DATA, query=query, refute="none", estimator="randomized.multi_action_ipw_policy_net"
        )
    ).policy_value


def test_multi_action_limits_that_bind_give_the_direct_values_bit_for_bit():
    chosen = _multi_policy()
    direct = _multi_direct(chosen, MULTI_AVAILABLE)
    compiled = _multi_compiled(chosen, MULTI_AVAILABLE)
    for name in POINTS:
        assert getattr(compiled, name) == getattr(direct, name), name


@pytest.mark.parametrize("case", ["capacity", "budget", "availability"])
def test_a_broken_multi_action_limit_is_refused_by_both_paths_with_the_same_error(case):
    counts = [MULTI_RECOMMENDATIONS.count(label) for label in LABELS]
    available = MULTI_AVAILABLE
    if case == "capacity":
        chosen = _multi_policy(capacities=[counts[0], counts[1] - 1, counts[2]])
    elif case == "budget":
        chosen = _multi_policy(budget=0.5)
    else:
        chosen = _multi_policy()
        available = [[True, row != 1, row % 4 == 3] for row in range(N)]
    with pytest.raises((CausalError, ValueError)) as direct:
        _multi_direct(chosen, available)
    with pytest.raises((CausalError, ValueError)) as compiled:
        _multi_compiled(chosen, available)
    assert type(compiled.value) is type(direct.value)
    assert str(compiled.value) == str(direct.value)
