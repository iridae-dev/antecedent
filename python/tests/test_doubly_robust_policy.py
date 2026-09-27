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


def test_retained_finite_class_regret_has_simultaneous_bound_and_artifact():
    n = 400
    assignment = [i % 4 < 2 for i in range(n)]
    outcome = [1.0 + ((2.0 if i % 2 == 0 else -1.0) if assigned else 0.0)
               + 0.01 * (i % 7)
               for i, assigned in enumerate(assignment)]
    selected = policy.BinaryPolicy([True] * n, costs=0.2)
    candidates = (
        policy.BinaryPolicy([False] * n, costs=0.2), selected,
        policy.BinaryPolicy([i % 2 == 0 for i in range(n)], costs=0.2),
        policy.BinaryPolicy([i % 2 == 1 for i in range(n)], costs=0.2),
    )
    query = policy.PolicyValue(
        "y", assignment, 0.5, selected, [f"eval-{i}" for i in range(n)],
        regret_candidates=candidates,
        regret_training_subject_ids=["selection-training"],
    )
    result = ant.analyze({"y": outcome}, query=query, refute="none")
    regret = result.policy_value.finite_class_regret
    assert regret is not None
    assert regret.target == "best_in_prespecified_candidate_class_minus_selected"
    assert regret.regret > 0.0
    assert regret.interval_95[0] <= regret.regret <= regret.interval_95[1]
    assert result.policy_value.support_status == "off_axis_simultaneous_95"
    loaded = ant.load(result.export(artifact_id="finite-class-regret"))
    assert loaded.answer.structured["regret"]["selected_index"] == 1
    with pytest.raises(CausalValueError, match="regret training IDs"):
        policy.PolicyValue(
            "y", assignment, 0.5, selected, [f"eval-{i}" for i in range(n)],
            regret_candidates=candidates, regret_training_subject_ids=["eval-0"],
        )


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


def test_top_k_policy_uses_retained_native_value_and_frozen_ranking():
    query = policy.PolicyValue.top_k(
        [8, 7, 6, 5, 4, 3, 2, 1], 4,
        outcome="y", assignment=[True, False] * 4, propensity=0.5,
        evaluation_subject_ids=[f"eval-{i}" for i in range(8)],
        ranking_training_subject_ids=["rank-train"], uplift_bins=2,
    )
    assert query.policy.actions == (True, True, True, True, False, False, False, False)
    result = ant.analyze({"y": [9, 5, 9, 5, 5, 5, 5, 5]}, query=query, refute="none")
    assert result.policy_value.policy_value == pytest.approx(7.0)
    assert result.policy_value.incremental_value == pytest.approx(2.0)
    assert result.policy_value.treatment_rate == pytest.approx(0.5)
    assert [bin.effect for bin in result.policy_value.uplift_bins] == pytest.approx([4.0, 0.0])
    loaded = ant.load(result.export(artifact_id="top-k-policy"))
    assert loaded.answer.structured["policy_value"] == pytest.approx(7.0)

    with pytest.raises(CausalValueError, match="disjoint"):
        policy.PolicyValue.top_k(
            [8, 7, 6, 5, 4, 3, 2, 1], 4,
            outcome="y", assignment=[True, False] * 4, propensity=0.5,
            evaluation_subject_ids=[f"eval-{i}" for i in range(8)],
            ranking_training_subject_ids=["eval-0"], uplift_bins=2,
        )


def test_retained_top_k_uplift_intervals_require_held_out_bin_support():
    rows = 600
    assignment = [i % 2 == 0 for i in range(rows)]
    effects = [2.0] * 300 + [0.5] * 300
    outcome = [1.0 + effects[i] * assignment[i] for i in range(rows)]
    query = policy.PolicyValue.top_k(
        list(range(rows, 0, -1)), 300,
        outcome="y", assignment=assignment, propensity=0.5,
        evaluation_subject_ids=[f"eval-{i}" for i in range(rows)],
        ranking_training_subject_ids=["rank-train"], uplift_bins=2,
    )
    result = ant.analyze({"y": outcome}, query=query, refute="none")
    assert result.policy_value.support_status == "off_axis_pointwise_95"
    assert result.policy_value.policy_value_interval_95 is None
    assert result.policy_value.incremental_value_interval_95 is None
    bins = result.policy_value.uplift_bins
    assert [bin.evaluation_rows for bin in bins] == [300, 300]
    for point, truth in zip(bins, (2.0, 0.5), strict=True):
        assert point.interval_95 is not None
        assert point.interval_95[0] < truth < point.interval_95[1]
    fixed_query = policy.PolicyValue(
        outcome="y", assignment=assignment, propensity=0.5,
        policy=policy.BinaryPolicy([i < 300 for i in range(rows)]),
        evaluation_subject_ids=[f"fixed-eval-{i}" for i in range(rows)],
        uplift_scores=list(range(rows, 0, -1)), uplift_bin_count=2,
        uplift_training_subject_ids=["fixed-rank-train"],
    )
    fixed_result = ant.analyze({"y": outcome}, query=fixed_query, refute="none")
    assert fixed_result.policy_value.support_status == "licensed"
    assert fixed_result.policy_value.policy_value_interval_95 is not None
    assert all(point.interval_95 is not None for point in fixed_result.policy_value.uplift_bins)
    loaded = ant.load(result.export(artifact_id="top-k-uplift-interval"))
    assert loaded.answer.structured["uplift_bins"][0]["interval_95"] == pytest.approx(bins[0].interval_95)

    thin_rows = 598
    thin_query = policy.PolicyValue.top_k(
        list(range(thin_rows, 0, -1)), 299,
        outcome="y", assignment=assignment[:thin_rows], propensity=0.5,
        evaluation_subject_ids=[f"eval-{i}" for i in range(thin_rows)],
        ranking_training_subject_ids=["rank-train"], uplift_bins=2,
    )
    thin = ant.analyze({"y": outcome[:thin_rows]}, query=thin_query, refute="none")
    assert all(point.interval_95 is None for point in thin.policy_value.uplift_bins)
    direct = policy.uplift_by_score(
        {"y": outcome}, outcome="y", assignment=assignment, propensity=0.5,
        scores=list(range(rows, 0, -1)), bins=2,
    )
    assert all(point.interval_95 is None for point in direct)


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
    grouped = policy.MultiActionPolicyValue(
        outcome="y", assignment=assigned, propensities=query.propensities,
        policy=fixed, evaluation_subject_ids=query.evaluation_subject_ids,
        baseline_groups=["x"] * 3 + ["y"] * 3 + ["z"] * 3,
    )
    grouped_result = ant.analyze({"y": outcomes}, query=grouped, refute="none")
    points = grouped_result.policy_value.multi_action_cate
    assert [(point.group, point.action, point.effect) for point in points] == [
        (group, action, effect)
        for group in ("x", "y", "z")
        for action, effect in (("A", 1.0), ("B", 3.0))
    ]
    assert all(point.uncertainty == "point_only" for point in points)
    grouped_artifact = ant.load(grouped_result.export(artifact_id="grouped-multi-policy"))
    assert grouped_artifact.answer.structured["multi_action_cate"][0]["effect"] == pytest.approx(1.0)
    with pytest.raises(CausalValueError, match="baseline_groups"):
        policy.MultiActionPolicyValue(
            outcome="y", assignment=assigned, propensities=query.propensities,
            policy=fixed, evaluation_subject_ids=query.evaluation_subject_ids,
            baseline_groups=["x"],
        )
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


def test_retained_multi_action_cate_interval_and_sparse_refusal():
    n = 300
    labels = ("control", "A", "B")
    assigned = [labels[i % 3] for i in range(n)]
    outcomes = [1.0 + (0.0, 1.0, 3.0)[i % 3] + (i % 7) / 10.0 for i in range(n)]
    fixed = policy.MultiActionPolicy(labels, [labels[(i // 3) % 3] for i in range(n)])
    query = policy.MultiActionPolicyValue(
        outcome="y", assignment=assigned,
        propensities=[[1.0 / 3.0] * 3 for _ in range(n)],
        policy=fixed, evaluation_subject_ids=[f"cate-{i}" for i in range(n)],
        baseline_groups=["g0"] * n,
    )
    result = ant.analyze({"y": outcomes}, query=query, refute="none")
    assert result.policy_value.support_status == "licensed"
    points = result.policy_value.multi_action_cate
    assert len(points) == 2
    for point, truth in zip(points, (1.0, 3.0), strict=True):
        assert point.uncertainty == "pointwise_95"
        assert point.standard_error > 0.0
        assert point.interval_95[0] < truth < point.interval_95[1]
    loaded = ant.load(result.export(artifact_id="multi-cate-interval"))
    assert loaded.answer.structured["multi_action_cate"][0]["interval_95"] == pytest.approx(points[0].interval_95)
    assert loaded.answer.structured["graphless_support_status"] == "licensed"

    thin = policy.MultiActionPolicyValue(
        outcome="y", assignment=assigned[:299],
        propensities=[[1.0 / 3.0] * 3 for _ in range(299)],
        policy=policy.MultiActionPolicy(labels, ["control"] * 299),
        evaluation_subject_ids=[f"thin-cate-{i}" for i in range(299)],
        baseline_groups=["g0"] * 299,
    )
    thin_result = ant.analyze({"y": outcomes[:299]}, query=thin, refute="none")
    assert all(point.interval_95 is None and point.uncertainty == "point_only"
               for point in thin_result.policy_value.multi_action_cate)


def test_retained_held_out_policy_intervals_and_crossfit_refusal():
    n = 300
    assignment = [i % 2 == 1 for i in range(n)]
    actions = [i % 3 == 0 for i in range(n)]
    y = [1.0 + 2.0 * int(assignment[i]) + (i % 5) / 10 for i in range(n)]
    common = dict(
        outcome="y", assignment=assignment, propensity=0.5,
        policy=policy.BinaryPolicy(actions, costs=0.1),
        evaluation_subject_ids=[f"eval-{i}" for i in range(n)],
        mu0=[1.0] * n, mu1=[3.0] * n,
    )
    held_out = policy.PolicyValue(**common, training_subject_ids=["train-1", "train-2"])
    result = ant.analyze({"y": y}, query=held_out, refute="none")
    answer = result.policy_value
    assert answer.policy_value_interval_95 is not None
    assert answer.incremental_value_interval_95 is not None
    assert answer.policy_value_interval_95[0] < answer.policy_value < answer.policy_value_interval_95[1]
    assert answer.incremental_value_interval_95[0] < answer.incremental_value < answer.incremental_value_interval_95[1]
    assert answer.support_status == "licensed"
    artifact = ant.load(result.export(artifact_id="policy-interval"))
    assert artifact.answer.structured["policy_interval_95"] == pytest.approx(answer.policy_value_interval_95)
    assert artifact.answer.structured["incremental_interval_95"] == pytest.approx(answer.incremental_value_interval_95)

    crossfit = policy.PolicyValue(
        **common,
        fold_ids=[i % 2 for i in range(n)],
        prediction_excluded_fold_ids=[i % 2 for i in range(n)],
    )
    point_only = ant.analyze({"y": y}, query=crossfit, refute="none").policy_value
    assert point_only.policy_value_interval_95 is None
    assert point_only.incremental_value_interval_95 is None
    assert point_only.support_status == "unlicensed_point_utility"

    constrained = policy.PolicyValue(
        **dict(common, policy=policy.BinaryPolicy(actions, costs=0.1, capacity=sum(actions))),
        training_subject_ids=["train-1", "train-2"],
    )
    constrained_result = ant.analyze({"y": y}, query=constrained, refute="none").policy_value
    assert constrained_result.policy_value_interval_95 is None
    assert constrained_result.incremental_value_interval_95 is None


def test_retained_multi_action_policy_interval_and_global_constraint_refusal():
    n = 300
    labels = ("control", "A", "B")
    assigned = [labels[i % 3] for i in range(n)]
    y = [[1.0, 2.0, 4.0][i % 3] + (i % 7) / 10 for i in range(n)]
    common = dict(
        outcome="y", assignment=assigned,
        propensities=[[1 / 3] * 3 for _ in range(n)],
        evaluation_subject_ids=[f"multi-{i}" for i in range(n)],
    )
    independent = policy.MultiActionPolicyValue(
        **common,
        policy=policy.MultiActionPolicy(labels, assigned, costs=[0, 0.1, 0.2],
                                        capacities=[n] * 3, budget=n * 0.2),
    )
    result = ant.analyze({"y": y}, query=independent, refute="none")
    assert result.policy_value.policy_value_interval_95 is not None
    assert result.policy_value.incremental_value_interval_95 is not None
    artifact = ant.load(result.export(artifact_id="multi-policy-interval"))
    assert artifact.answer.structured["policy_interval_95"] == pytest.approx(result.policy_value.policy_value_interval_95)

    constrained = policy.MultiActionPolicyValue(
        **common,
        policy=policy.MultiActionPolicy(labels, assigned, costs=[0, 0.1, 0.2],
                                        capacities=[n] * 3, budget=40.0),
    )
    point_only = ant.analyze({"y": y}, query=constrained, refute="none").policy_value
    assert point_only.policy_value_interval_95 is None
