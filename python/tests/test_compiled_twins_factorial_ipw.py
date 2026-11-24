"""The point-only policy evaluators warn, and the declared net-value estimators reproduce them.

``evaluate_policy`` and ``evaluate_multi_action_policy`` run the point-only inverse-probability
kernels. ``analyze`` with ``estimator="randomized.ipw_policy_net"`` (binary) or
``estimator="randomized.multi_action_ipw_policy_net"`` (multi-action) runs the same kernels
inside the compiled ``PolicyValue`` route, so every point value is equal to the bit.
"""

from __future__ import annotations

import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent import policy
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success

N = 40
ASSIGNMENT = [bool((i * 7) % 5 < 2) for i in range(N)]
OUTCOME = np.array(
    [1.0 + 0.37 * (i % 11) + 0.13 * float(ASSIGNMENT[i]) + 0.01 * (i % 7) for i in range(N)]
)
PROPENSITY = [0.3 + 0.01 * (i % 9) for i in range(N)]
DATA = {"y": OUTCOME}
SUBJECTS = [f"eval-{i}" for i in range(N)]
POLICY = policy.BinaryPolicy([i % 3 != 0 for i in range(N)], costs=0.1)
REFERENCE = policy.BinaryPolicy([i % 4 == 0 for i in range(N)], costs=0.07)
POINTS = (
    "policy_value",
    "reference_value",
    "incremental_value",
    "relative_value_gap",
    "treatment_rate",
    "total_treatment_cost",
)


def test_binary_direct_path_warns_and_the_net_value_estimator_matches_bit_for_bit():
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_policy"):
        direct = policy.evaluate_policy(
            DATA,
            outcome="y",
            assignment=ASSIGNMENT,
            propensity=PROPENSITY,
            policy=POLICY,
            reference=REFERENCE,
        )
    query = policy.PolicyValue("y", ASSIGNMENT, PROPENSITY, POLICY, SUBJECTS, reference=REFERENCE)
    compiled = require_success(
        ant.analyze(DATA, query=query, refute="none", estimator="randomized.ipw_policy_net")
    ).policy_value
    for name in POINTS:
        assert getattr(compiled, name) == getattr(direct, name), name


def test_multi_action_direct_path_warns_and_the_net_value_estimator_matches_bit_for_bit():
    labels = ("control", "A", "B")
    assignment = [labels[(0, 1, 2, 0, 2)[i % 5]] for i in range(N)]
    recommendations = [labels[(0, 1, 1, 2)[i % 4]] for i in range(N)]
    propensities = [[0.4, 0.35, 0.25] if i % 2 == 0 else [0.5, 0.3, 0.2] for i in range(N)]
    outcome = np.array([1.0 + 0.31 * (i % 7) + 0.11 * (i % 3) for i in range(N)])
    chosen = policy.MultiActionPolicy(labels, recommendations, costs=[0.0, 0.15, 0.4])
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_multi_action_policy"):
        direct = policy.evaluate_multi_action_policy(
            {"y": outcome},
            outcome="y",
            assignment=assignment,
            propensities=propensities,
            policy=chosen,
        )
    query = policy.MultiActionPolicyValue("y", assignment, propensities, chosen, SUBJECTS)
    compiled = require_success(
        ant.analyze(
            {"y": outcome},
            query=query,
            refute="none",
            estimator="randomized.multi_action_ipw_policy_net",
        )
    ).policy_value
    for name in POINTS:
        assert getattr(compiled, name) == getattr(direct, name), name


def test_the_net_value_estimator_on_a_doubly_robust_query_is_refused():
    query = policy.PolicyValue(
        "y",
        ASSIGNMENT,
        0.5,
        POLICY,
        SUBJECTS,
        mu0=[1.0] * N,
        mu1=[2.0] * N,
        training_subject_ids=["train-0"],
    )
    # The estimator answers a policy without outcome predictions: the study is refused, as a
    # raised refusal or a refused outcome, and never returns a value.
    try:
        result = ant.analyze(
            DATA, query=query, refute="none", estimator="randomized.ipw_policy_net"
        )
    except CausalUnsupportedError:
        return
    assert result.refusal is not None and result.value is None


def test_the_compiled_net_value_query_does_not_warn():
    query = policy.PolicyValue("y", ASSIGNMENT, PROPENSITY, POLICY, SUBJECTS, reference=REFERENCE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        require_success(
            ant.analyze(DATA, query=query, refute="none", estimator="randomized.ipw_policy_net")
        )
