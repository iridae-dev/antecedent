from __future__ import annotations

import numpy as np
import pytest
from antecedent import policy
from antecedent.errors import CausalValueError
from antecedent.results._execution import answer_from_artifact


def _call(**overrides):
    n = 8
    args = {
        "evaluation_data": {"y": 5.0 + 2.0 * np.array([True, False] * 4, dtype=float)},
        "outcome": "y",
        "assignment": [True, False] * 4,
        "propensity": 0.5,
        "mu0": [5.0] * n,
        "mu1": [7.0] * n,
        "policy": policy.BinaryPolicy([True] * n, costs=0.25),
        "reference": policy.BinaryPolicy([False] * n),
        "evaluation_subject_ids": [f"eval-{i}" for i in range(n)],
        "training_subject_ids": [f"train-{i}" for i in range(5)],
    }
    args.update(overrides)
    return policy.evaluate_policy_doubly_robust(**args)


def test_doubly_robust_policy_value_recovers_known_truth_and_score_se():
    result = _call()
    assert result.policy_value == pytest.approx(6.75)
    assert result.reference_value == pytest.approx(5.0)
    assert result.incremental_value == pytest.approx(1.75)
    assert result.relative_value_gap == pytest.approx(-1.75)
    assert result.treatment_rate == pytest.approx(1.0)
    assert result.total_treatment_cost == pytest.approx(2.0)
    assert result.policy_value_standard_error == pytest.approx(0.0)
    assert result.reference_value_standard_error == pytest.approx(0.0)
    assert result.incremental_value_standard_error == pytest.approx(0.0)
    assert result.prediction_ownership == "held_out_disjoint_subject_ids"
    assert result.propensity_min == result.propensity_max == pytest.approx(0.5)
    assert result.uncertainty == "row_score_standard_error_independent_subjects"
    assert result.support_status == "unlicensed_point_utility"


def test_doubly_robust_policy_supports_caller_declared_cross_fitting():
    result = _call(
        training_subject_ids=None,
        fold_ids=[i % 2 for i in range(8)],
        prediction_excluded_fold_ids=[i % 2 for i in range(8)],
    )
    assert result.prediction_ownership == "caller_declared_cross_fitted_excluded_fold_ids"


def test_doubly_robust_policy_refuses_missing_or_leaking_ownership_metadata():
    with pytest.raises(CausalValueError, match="provide disjoint training_subject_ids"):
        _call(training_subject_ids=None)
    with pytest.raises(CausalValueError, match="overlap"):
        _call(training_subject_ids=["eval-0"])
    with pytest.raises(CausalValueError, match="exclude its evaluation row's fold"):
        _call(
            training_subject_ids=None,
            fold_ids=[i % 2 for i in range(8)],
            prediction_excluded_fold_ids=[0] * 8,
        )


def test_doubly_robust_policy_refuses_invalid_propensity_and_duplicate_subjects():
    with pytest.raises((CausalValueError, ValueError), match="propensities"):
        _call(propensity=1.0)
    with pytest.raises(CausalValueError, match="unique"):
        _call(evaluation_subject_ids=["same"] * 8)


def test_policy_artifact_answer_preserves_typed_values_and_uncertainty():
    payload = {
        "policy_value": {
            "policy_value": 6.75,
            "reference_value": 5.0,
            "incremental_value": 1.75,
            "treatment_rate": 1.0,
            "incremental_standard_error": 0.0,
            "uncertainty": "row_score_standard_error_independent_subjects",
        },
        "estimate": None,
    }
    answer = answer_from_artifact({"claim": {"kind": "policy_value"}}, payload)
    assert answer.kind == "policy_value"
    assert answer.structured == payload["policy_value"]
