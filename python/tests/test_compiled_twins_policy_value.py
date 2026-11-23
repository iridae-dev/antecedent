"""The direct policy-value utilities warn, and the compiled ``PolicyValue`` query reproduces them."""

from __future__ import annotations

import warnings

import antecedent as ant
import numpy as np
import pytest
from antecedent import policy
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.outcomes import require_success

N = 40
ASSIGNMENT = [bool(i % 2) for i in range(N)]
OUTCOME = np.array(
    [1.0 + 2.0 * float(ASSIGNMENT[i]) + 0.1 * (i % 5) + 0.01 * (i % 7) for i in range(N)]
)
DATA = {"y": OUTCOME}
SUBJECTS = [f"eval-{i}" for i in range(N)]
TRAINING = [f"train-{i}" for i in range(6)]
POLICY = policy.BinaryPolicy([i % 3 == 0 for i in range(N)], costs=0.25)
REFERENCE = policy.BinaryPolicy([i % 4 == 0 for i in range(N)], costs=0.25)
MU0 = [1.0 + 0.01 * (i % 7) for i in range(N)]
MU1 = [3.0 + 0.02 * (i % 5) for i in range(N)]


def _compiled(**fields):
    query = policy.PolicyValue(
        "y", ASSIGNMENT, 0.5, POLICY, SUBJECTS, reference=REFERENCE, **fields
    )
    return require_success(ant.analyze(DATA, query=query, refute="none")).policy_value


def test_doubly_robust_direct_path_warns_and_the_compiled_query_matches_bit_for_bit():
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_policy_doubly_robust"):
        direct = policy.evaluate_policy_doubly_robust(
            DATA,
            outcome="y",
            assignment=ASSIGNMENT,
            propensity=0.5,
            mu0=MU0,
            mu1=MU1,
            policy=POLICY,
            reference=REFERENCE,
            evaluation_subject_ids=SUBJECTS,
            training_subject_ids=TRAINING,
        )
    compiled = _compiled(mu0=MU0, mu1=MU1, training_subject_ids=TRAINING)
    for name in (
        "policy_value",
        "reference_value",
        "incremental_value",
        "treatment_rate",
        "policy_value_standard_error",
        "reference_value_standard_error",
        "incremental_value_standard_error",
        "propensity_min",
        "propensity_max",
    ):
        assert getattr(compiled, name) == getattr(direct, name), name
    # The direct path reports the gap as the negated paired increment; the compiled path
    # as reference minus policy value. They are equal in exact arithmetic and round apart.
    assert compiled.relative_value_gap == pytest.approx(direct.relative_value_gap, rel=1e-12)
    assert compiled.relative_value_gap == compiled.reference_value - compiled.policy_value
    assert direct.relative_value_gap == -direct.incremental_value
    assert compiled.total_treatment_cost == direct.total_treatment_cost


def test_ipw_direct_path_warns_and_the_compiled_query_agrees_to_rounding():
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_policy"):
        direct = policy.evaluate_policy(
            DATA,
            outcome="y",
            assignment=ASSIGNMENT,
            propensity=0.5,
            policy=POLICY,
            reference=REFERENCE,
        )
    compiled = _compiled()
    assert compiled.policy_value == pytest.approx(direct.policy_value, rel=1e-12)
    assert compiled.reference_value == pytest.approx(direct.reference_value, rel=1e-12)
    assert compiled.incremental_value == pytest.approx(direct.incremental_value, rel=1e-12)
    assert compiled.treatment_rate == direct.treatment_rate


def test_uplift_direct_path_warns_and_the_compiled_bins_match_bit_for_bit():
    scores = [float((i * 7) % 11) + 0.01 * i for i in range(N)]
    with pytest.warns(AntecedentDeprecationWarning, match="uplift_by_score"):
        direct = policy.uplift_by_score(
            DATA,
            outcome="y",
            assignment=ASSIGNMENT,
            propensity=0.5,
            scores=scores,
            bins=4,
        )
    compiled = _compiled(
        uplift_scores=scores, uplift_bin_count=4, uplift_training_subject_ids=TRAINING
    ).uplift_bins
    assert len(compiled) == len(direct) == 4
    for got, want in zip(compiled, direct, strict=True):
        assert got.rank == want.rank
        assert got.evaluation_rows == want.evaluation_rows
        assert got.effect == want.effect
        assert got.standard_error == want.standard_error


def test_multi_action_direct_path_warns_and_the_compiled_query_agrees_to_rounding():
    labels = ("control", "A", "B")
    assignment = [labels[(0, 0, 1, 2)[i % 4]] for i in range(N)]
    recommendations = [labels[(0, 1, 1, 2)[i % 4]] for i in range(N)]
    propensities = [[0.5, 0.25, 0.25]] * N
    outcome = np.array([1.0 + 0.5 * (i % 4) for i in range(N)])
    chosen = policy.MultiActionPolicy(labels, recommendations, costs=[0.0, 0.25, 0.5])
    with pytest.warns(AntecedentDeprecationWarning, match="evaluate_multi_action_policy"):
        direct = policy.evaluate_multi_action_policy(
            {"y": outcome},
            outcome="y",
            assignment=assignment,
            propensities=propensities,
            policy=chosen,
        )
    query = policy.MultiActionPolicyValue("y", assignment, propensities, chosen, SUBJECTS)
    compiled = require_success(ant.analyze({"y": outcome}, query=query, refute="none")).policy_value
    assert compiled.policy_value == pytest.approx(direct.policy_value, rel=1e-12)
    assert compiled.reference_value == pytest.approx(direct.reference_value, rel=1e-12)
    assert compiled.treatment_rate == pytest.approx(direct.treatment_rate, rel=1e-12)


def test_the_compiled_policy_query_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        _compiled(mu0=MU0, mu1=MU1, training_subject_ids=TRAINING)
        _compiled()
