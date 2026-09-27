from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import policy
from antecedent.errors import CausalUnsupportedError, CausalValueError
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


def test_retained_policy_value_honors_capacity_budget_and_availability():
    assignment = [True, False] * 4
    outcome = 5.0 + 2.0 * np.asarray(assignment, dtype=float)
    common = dict(
        outcome="y",
        assignment=assignment,
        propensity=0.5,
        mu0=[5.0] * 8,
        mu1=[7.0] * 8,
        evaluation_subject_ids=[f"eval-{i}" for i in range(8)],
        training_subject_ids=[f"train-{i}" for i in range(8)],
    )

    valid = ant.policy.PolicyValue(
        **common,
        policy=policy.BinaryPolicy([True] * 4 + [False] * 4, capacity=4, costs=0.25, budget=1.0),
        available=[True] * 4 + [False] * 4,
    )
    result = ant.analyze({"y": outcome}, query=valid, refute="none")
    assert result.study is not None
    assert result.policy_value.policy_value == pytest.approx(5.875)
    assert result.answer.kind == "policy_value"
    loaded = ant.load(result.export(artifact_id="retained-policy"))
    assert loaded.answer.kind == "policy_value"
    assert loaded.answer.structured["policy_value"] == pytest.approx(5.875)
    with pytest.raises(CausalUnsupportedError, match="bound to the prepared evaluation rows"):
        result.refresh({"y": outcome})

    for constrained in (
        policy.BinaryPolicy([True] * 4 + [False] * 4, capacity=3),
        policy.BinaryPolicy([True] * 4 + [False] * 4, costs=0.25, budget=0.5),
    ):
        with pytest.raises(ValueError, match="capacity|budget"):
            ant.analyze(
                {"y": outcome},
                query=ant.policy.PolicyValue(**common, policy=constrained),
                refute="none",
            )

    with pytest.raises(ValueError, match="unavailable"):
        ant.analyze(
            {"y": outcome},
            query=ant.policy.PolicyValue(
                **common,
                policy=policy.BinaryPolicy([True] * 4 + [False] * 4),
                available=[False] + [True] * 7,
            ),
            refute="none",
        )


def test_retained_randomized_ipw_policy_needs_no_nuisance_predictions():
    query = policy.PolicyValue(
        outcome="y",
        assignment=[False, True, False, True],
        propensity=0.5,
        policy=policy.BinaryPolicy([False, True, False, True]),
        evaluation_subject_ids=["a", "b", "c", "d"],
    )
    result = ant.analyze({"y": [1.0, 3.0, 1.0, 3.0]}, query=query, refute="none")
    assert result.policy_value.policy_value == pytest.approx(4.0)
    assert result.policy_value.reference_value == pytest.approx(1.0)
    assert result.policy_value.incremental_value == pytest.approx(3.0)
    assert result.policy_value.prediction_ownership == "no_outcome_nuisance_predictions"
    assert result.policy_value.uncertainty == "ipw_row_score_standard_error_independent_subjects"
    assert result.policy_value.policy_value_standard_error >= 0
    direct = policy.evaluate_policy(
        {"y": [1.0, 3.0, 1.0, 3.0]}, outcome="y",
        assignment=query.assignment, propensity=query.propensity, policy=query.policy,
    )
    assert result.policy_value.policy_value == pytest.approx(direct.policy_value)
    assert result.policy_value.incremental_value == pytest.approx(direct.incremental_value)
    artifact = ant.load(result.export(artifact_id="policy-ipw"))
    assert artifact.answer.structured["uncertainty"] == result.policy_value.uncertainty

    with pytest.raises(CausalValueError, match="supplied together"):
        policy.PolicyValue(
            outcome="y", assignment=[False, True], propensity=0.5,
            policy=policy.BinaryPolicy([False, True]),
            evaluation_subject_ids=["a", "b"], mu0=[1.0, 1.0],
        )
    with pytest.raises(CausalValueError, match="requires mu0 and mu1"):
        policy.PolicyValue(
            outcome="y", assignment=[False, True], propensity=0.5,
            policy=policy.BinaryPolicy([False, True]),
            evaluation_subject_ids=["a", "b"], training_subject_ids=["train"],
        )


def test_retained_policy_reports_held_out_ranked_uplift_bins():
    assignment = [True, False] * 4
    query = policy.PolicyValue(
        outcome="y", assignment=assignment, propensity=0.5,
        policy=policy.BinaryPolicy([True] * 4 + [False] * 4),
        evaluation_subject_ids=[f"eval-{i}" for i in range(8)],
        uplift_scores=[8, 7, 6, 5, 4, 3, 2, 1],
        uplift_bin_count=2,
        uplift_training_subject_ids=["rank-train"],
    )
    result = ant.analyze({"y": [9, 5, 9, 5, 5, 5, 5, 5]}, query=query, refute="none")
    bins = result.policy_value.uplift_bins
    assert len(bins) == 2
    assert [bin.effect for bin in bins] == pytest.approx([4.0, 0.0])
    assert [bin.evaluation_rows for bin in bins] == [4, 4]
    assert all(bin.standard_error >= 0 for bin in bins)
    assert result.policy_value.support_status == "unlicensed_point_utility"
    assert ant.load(result.export(artifact_id="ranked-policy")).answer.structured["uplift_bins"][0]["effect"] == pytest.approx(4.0)

    with pytest.raises(CausalValueError, match="disjoint"):
        policy.PolicyValue(
            outcome="y", assignment=assignment, propensity=0.5,
            policy=policy.BinaryPolicy([False] * 8),
            evaluation_subject_ids=[f"eval-{i}" for i in range(8)],
            uplift_scores=[8, 7, 6, 5, 4, 3, 2, 1], uplift_bin_count=2,
            uplift_training_subject_ids=["eval-0"],
        )


def test_retained_multi_action_policy_matches_direct_native_value_and_artifact():
    labels = ("control", "A", "B")
    assigned = ["control", "A", "B"] * 3
    outcomes = [1.0, 2.0, 4.0] * 3
    recommendations = ["control"] * 3 + ["A"] * 3 + ["B"] * 3
    fixed = policy.MultiActionPolicy(labels, recommendations, capacities=[9, 3, 3])
    query = policy.MultiActionPolicyValue(
        outcome="y", assignment=assigned,
        propensities=[[1.0 / 3.0] * 3 for _ in assigned],
        policy=fixed, evaluation_subject_ids=[f"s{i}" for i in range(9)],
    )
    result = ant.analyze({"y": outcomes}, query=query, refute="none")
    direct = policy.evaluate_multi_action_policy(
        {"y": outcomes}, outcome="y", assignment=assigned,
        propensities=query.propensities, policy=fixed,
    )
    assert result.policy_value.policy_value == pytest.approx(7.0 / 3.0)
    assert result.policy_value.policy_value == pytest.approx(direct.policy_value)
    assert result.policy_value.incremental_value == pytest.approx(4.0 / 3.0)
    assert result.policy_value.treatment_rate == pytest.approx(2.0 / 3.0)
    assert result.policy_value.uncertainty == "multi_action_ipw_row_score_standard_error_independent_subjects"
    assert result.policy_value.support_status == "unlicensed_point_utility"
    assert any("action probabilities" in assumption for assumption in result.policy_value.assumptions)
    loaded = ant.load(result.export(artifact_id="retained-multi-policy"))
    assert loaded.answer.structured["policy_value"] == pytest.approx(7.0 / 3.0)
    assert loaded.answer.structured["uncertainty"] == result.policy_value.uncertainty
    with pytest.raises(CausalUnsupportedError, match="bound to the prepared evaluation rows"):
        result.refresh({"y": outcomes})

    with pytest.raises(CausalValueError, match="probability"):
        policy.MultiActionPolicyValue(
            outcome="y", assignment=assigned,
            propensities=[[0.0, 0.5, 0.5]] * 9, policy=fixed,
            evaluation_subject_ids=[f"s{i}" for i in range(9)],
        )
    over_capacity = policy.MultiActionPolicy(labels, recommendations, capacities=[9, 2, 3])
    with pytest.raises(ValueError, match="capacity"):
        ant.analyze(
            {"y": outcomes},
            query=policy.MultiActionPolicyValue(
                outcome="y", assignment=assigned, propensities=query.propensities,
                policy=over_capacity, evaluation_subject_ids=[f"s{i}" for i in range(9)],
            ),
            refute="none",
        )
