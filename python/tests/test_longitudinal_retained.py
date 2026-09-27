"""Retained longitudinal regime value through the ordinary analysis lifecycle."""

import antecedent
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.regimes import LongitudinalRegimeQuery


def fixture():
    data = {"y": [4.0, 0.0, 0.0, 0.0], "id": ["a", "b", "c", "d"]}
    query = LongitudinalRegimeQuery(
        outcome="y",
        treatment_history=[[True, True], [True, False], [False, True], [False, False]],
        actions=[True, True],
        treatment_probabilities=[[0.5, 0.5]] * 4,
        subject_ids=["a", "b", "c", "d"],
    )
    return data, query


def test_known_truth_regime_value_prepare_analyze_and_artifact():
    data, query = fixture()
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.longitudinal_regime is not None
    assert result.longitudinal_regime.value == pytest.approx(4.0)
    assert result.longitudinal_regime.effective_sample_size == pytest.approx(1.0)
    assert result.longitudinal_regime.matched_observed_fraction == pytest.approx(0.25)
    assert result.longitudinal_regime.maximum_weight == pytest.approx(4.0)
    assert result.longitudinal_regime.uncertainty == "point_only_no_interval"
    assert result.longitudinal_regime.value_interval_95 is None
    assert result.longitudinal_regime.interval_reason == "insufficient_independent_subject_support_or_degenerate_score"
    assert result.longitudinal_regime.probability_ownership == "known_sequential_randomization"
    assert result.longitudinal_regime.support_status == "unlicensed_point_utility"
    assert "known_sequential_randomization" in " ".join(result.assumptions or [])
    assert result.answer.kind == "structured"
    assert antecedent.analyze(data, query=query).longitudinal_regime == result.longitudinal_regime

    loaded = antecedent.load(prepared.export())
    assert loaded.artifact.payload_kind == "analysis_result"
    section = loaded.artifact.payload["longitudinal_regime"]
    assert section["value"] == pytest.approx(4.0)
    assert section["uncertainty"] == "point_only_no_interval"
    query_artifact = antecedent.artifacts.loads(prepared.export_artifact(payload="query"))
    assert query_artifact.payload_kind == "query"
    query_wire = query_artifact.payload["longitudinal_regime"]
    assert query_wire["subject_ids"] == ["a", "b", "c", "d"]
    assert query_wire["fold_ids"] == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="align with subject histories"):
        prepared.refresh({"y": [1.0, 2.0]})


def test_retained_ipw_interval_on_independent_subject_histories_round_trips():
    n = 500
    history = [
        [(i % 4 in (0, 1)), (i % 4 in (0, 2))]
        for i in range(n)
    ]
    data = {"y": [4.0 if i % 4 == 0 else 0.0 for i in range(n)]}
    query = LongitudinalRegimeQuery(
        outcome="y", treatment_history=history, actions=[True, True],
        treatment_probabilities=[[0.5, 0.5]] * n,
        subject_ids=[f"subject-{i}" for i in range(n)],
        fold_ids=[i % 5 for i in range(n)],
    )
    result = antecedent.analyze(data, query=query)
    value = result.longitudinal_regime
    assert value.value == pytest.approx(4.0)
    assert value.value_standard_error > 0.0
    assert value.value_interval_95[0] < 4.0 < value.value_interval_95[1]
    assert value.uncertainty == "pointwise_subject_score_95"
    assert value.interval_reason is None
    assert value.support_status == "off_axis_pointwise_95"
    artifact = antecedent.load(result.export(artifact_id="regime-ipw-interval"))
    assert artifact.artifact.payload["longitudinal_regime"]["value_interval_95"] == pytest.approx(value.value_interval_95)


def test_retained_msm_intercept_and_period_intervals_round_trip():
    n = 300
    treatment = [[bool((i % 4) // 2), bool((i % 4) % 2)] for i in range(n)]
    y = [1.0 + 2.0 * int(a0) + 3.0 * int(a1) + (i % 7) / 10
         for i, (a0, a1) in enumerate(treatment)]
    query = LongitudinalRegimeQuery.marginal_structural_model(
        outcome="y", treatment_history=treatment,
        treatment_probabilities=[[0.5, 0.5]] * n,
        stabilizing_numerator_probabilities=[0.5, 0.5],
        subject_ids=[f"msm-{i}" for i in range(n)],
        fold_ids=[i % 5 for i in range(n)],
    )
    result = antecedent.analyze({"y": y}, query=query)
    value = result.longitudinal_regime
    assert value.value_standard_error > 0.0
    assert value.value_interval_95[0] < value.value < value.value_interval_95[1]
    assert len(value.period_intervals_95) == 2
    assert result.answer.detail == "longitudinal_msm_pointwise_cr1_interval"
    assert "Pointwise 95% independent-subject CR1 intervals" in result.claim()
    for interval, effect in zip(value.period_intervals_95, value.period_effects, strict=True):
        assert interval[0] < effect < interval[1]
    artifact = antecedent.load(result.export(artifact_id="msm-interval"))
    for actual, expected in zip(artifact.artifact.payload["longitudinal_regime"]["period_intervals_95"],
                                value.period_intervals_95, strict=True):
        assert actual == pytest.approx(expected)


def test_dynamic_rule_freezes_only_available_history_with_identity_and_point_value():
    data, static = fixture()
    seen = []

    def rule(period, past_actions, covariates_through_period):
        seen.append((period, past_actions, covariates_through_period))
        return bool(covariates_through_period[0][0] > 0) if period == 0 else past_actions[0]

    query = LongitudinalRegimeQuery.from_dynamic_rule(
        outcome="y", treatment_history=static.treatment_history,
        predecision_covariates=[[[1.0], [99.0]], [[1.0], [99.0]],
                                [[-1.0], [99.0]], [[-1.0], [99.0]]],
        rule=rule, rule_id="adaptive-threshold", rule_version="v1",
        rule_provenance="study-protocol-7",
        treatment_probabilities=static.treatment_probabilities,
        subject_ids=static.subject_ids, fold_ids=[0, 0, 1, 1],
    )
    assert query.actions == ((True, True), (True, True), (False, False), (False, False))
    assert seen[0] == (0, (), ((1.0,),))
    assert seen[1] == (1, (True,), ((1.0,), (99.0,)))
    prepared = antecedent.prepare(data, query=query)
    fit = prepared.estimate().longitudinal_regime
    assert fit.value == pytest.approx(4.0)
    assert fit.rule_id == "adaptive-threshold"
    assert fit.rule_version == "v1"
    assert fit.rule_provenance == "study-protocol-7"
    assert fit.uncertainty == "point_only_no_interval"
    assert fit.support_status == "unlicensed_point_utility"
    assert antecedent.analyze(data, query=query).longitudinal_regime == fit
    result_wire = antecedent.load(prepared.export()).artifact.payload["longitudinal_regime"]
    assert result_wire["rule_provenance"] == "study-protocol-7"
    query_wire = antecedent.artifacts.loads(prepared.export_artifact(payload="query")).payload["longitudinal_regime"]
    assert query_wire["rule_id"] == "adaptive-threshold"
    assert query_wire["regime_actions"] == [True, True, True, True, False, False, False, False]


def test_dynamic_rule_refuses_invalid_identity_future_covariate_shape_and_nonbinary_return():
    _, static = fixture()
    args = dict(outcome="y", treatment_history=static.treatment_history,
                predecision_covariates=[[[1.0], [2.0]]] * 4,
                treatment_probabilities=static.treatment_probabilities,
                subject_ids=static.subject_ids, rule_id="rule", rule_version="v1",
                rule_provenance="protocol")
    with pytest.raises(ValueError, match="rule_provenance"):
        LongitudinalRegimeQuery.from_dynamic_rule(**{**args, "rule_provenance": ""}, rule=lambda *_: True)
    with pytest.raises(ValueError, match="subject-by-period-by-feature"):
        LongitudinalRegimeQuery.from_dynamic_rule(**{**args, "predecision_covariates": [[1.0, 2.0]] * 4}, rule=lambda *_: True)
    with pytest.raises(ValueError, match="binary bool"):
        LongitudinalRegimeQuery.from_dynamic_rule(**args, rule=lambda *_: 1)


def test_dynamic_rule_skips_decisions_and_covariates_after_dropout():
    _, static = fixture()
    calls = []

    def rule(period, past, covariates):
        calls.append((period, past, covariates))
        return True

    query = LongitudinalRegimeQuery.from_dynamic_rule(
        outcome="y", treatment_history=static.treatment_history,
        predecision_covariates=[[[1.0], [2.0]], [[1.0], [float("nan")]],
                                [[1.0], [2.0]], [[1.0], [2.0]]],
        rule=rule, rule_id="dropout-aware", rule_version="v1", rule_provenance="protocol",
        treatment_probabilities=static.treatment_probabilities,
        subject_ids=static.subject_ids, method="sequential_dr",
        q_predictions=[[1.0, 1.0]] * 4,
        observation_history=[[True, True], [True, False], [True, True], [True, True]],
        outcome_observed=[True, False, True, True],
        fold_ids=[0, 0, 1, 1], prediction_fold_ids=[0, 0, 1, 1],
        excluded_fold_predictions=True,
    )
    assert len(calls) == 7
    assert query.actions[1] == (True, False)
    with pytest.raises(ValueError, match="monotone after dropout"):
        LongitudinalRegimeQuery.from_dynamic_rule(
            outcome="y", treatment_history=static.treatment_history,
            predecision_covariates=[[[1.0], [2.0]]] * 4,
            rule=rule, rule_id="bad-dropout", rule_version="v1", rule_provenance="protocol",
            treatment_probabilities=static.treatment_probabilities,
            subject_ids=static.subject_ids, method="sequential_dr",
            q_predictions=[[1.0, 1.0]] * 4,
            observation_history=[[True, True], [False, True], [True, True], [True, True]],
            outcome_observed=[True] * 4, fold_ids=[0, 0, 1, 1],
            prediction_fold_ids=[0, 0, 1, 1], excluded_fold_predictions=True,
        )


def test_longitudinal_refuses_unsupported_nuisance_ownership_and_bad_subjects():
    data, query = fixture()
    with pytest.raises(ValueError, match="distinct non-empty"):
        LongitudinalRegimeQuery(
            outcome="y", treatment_history=query.treatment_history,
            actions=query.actions, treatment_probabilities=query.treatment_probabilities,
            subject_ids=["a", "a", "c", "d"],
        )
    unsupported = LongitudinalRegimeQuery(
        outcome="y", treatment_history=query.treatment_history,
        actions=query.actions, treatment_probabilities=query.treatment_probabilities,
        subject_ids=query.subject_ids, fold_ids=[0, 1, 0, 1],
        probabilities_known_by_design=False, excluded_fold_predictions=True,
    )
    with pytest.raises(CausalUnsupportedError, match="known sequential randomization"):
        antecedent.analyze(data, query=unsupported)


def test_g_formula_regime_uses_main_analysis_and_round_trips_method():
    data, base = fixture()
    query = LongitudinalRegimeQuery(
        outcome="y",
        treatment_history=base.treatment_history,
        actions=base.actions,
        treatment_probabilities=base.treatment_probabilities,
        subject_ids=base.subject_ids,
        method="g_formula",
        period_outcome_predictions=[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
        fold_ids=[0, 1, 0, 1],
        excluded_fold_predictions=True,
    )
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.longitudinal_regime.value == pytest.approx(3.5)
    assert result.longitudinal_regime.method == "g_formula"
    assert result.longitudinal_regime.uncertainty == "point_only_no_interval"
    assert result.longitudinal_regime.support_status == "unlicensed_point_utility"
    assert "conditional_period_reward_validity" in " ".join(result.assumptions or [])
    assert antecedent.analyze(data, query=query).longitudinal_regime == result.longitudinal_regime
    artifact = antecedent.load(prepared.export()).artifact
    assert artifact.payload["longitudinal_regime"]["method"] == "g_formula"
    query_artifact = antecedent.artifacts.loads(prepared.export_artifact(payload="query"))
    assert query_artifact.payload["longitudinal_regime"]["method"] == "g_formula"


def test_g_formula_query_refuses_missing_or_nonfinite_predictions():
    _, base = fixture()
    args = dict(
        outcome="y", treatment_history=base.treatment_history, actions=base.actions,
        treatment_probabilities=base.treatment_probabilities, subject_ids=base.subject_ids,
        method="g_formula",
    )
    with pytest.raises(ValueError, match="requires finite"):
        LongitudinalRegimeQuery(**args)
    with pytest.raises(ValueError, match="requires finite"):
        LongitudinalRegimeQuery(**args, period_outcome_predictions=[[float("nan"), 1.0]] * 4)


def test_sequential_dr_retained_matches_direct_kernel_and_artifact():
    from antecedent.regimes import evaluate_sequential_doubly_robust

    data = {"y": [7.0, 99.0, float("nan"), 8.0]}
    query = LongitudinalRegimeQuery(
        outcome="y", method="sequential_dr",
        treatment_history=[[False, False], [False, True], [False, False], [True, True]],
        actions=[False, False],
        treatment_probabilities=[[0.5, 0.5]] * 4,
        censoring_probabilities=[[0.8, 0.8]] * 4,
        outcome_observed=[True, True, False, True],
        observation_history=[[True, True], [True, True], [True, False], [True, True]],
        q_predictions=[[1.0, 3.0], [5.0, 4.0], [2.0, 6.0], [1.0, 1.0]],
        subject_ids=["s1", "s2", "s3", "s4"], fold_ids=[0, 1, 0, 1],
        prediction_fold_ids=[0, 1, 0, 1], excluded_fold_predictions=True,
    )
    direct = evaluate_sequential_doubly_robust(
        outcomes=data["y"], outcome_observed=query.outcome_observed,
        observation_history=query.observation_history,
        treatment_history=query.treatment_history, regime=[False, False],
        q_predictions=query.q_predictions, treatment_probabilities=query.treatment_probabilities,
        censoring_probabilities=query.censoring_probabilities,
        subject_ids=query.subject_ids, fold_ids=query.fold_ids,
        prediction_fold_ids=query.prediction_fold_ids,
    )
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.longitudinal_regime.method == "sequential_dr"
    assert result.longitudinal_regime.value == pytest.approx(direct.value)
    assert result.longitudinal_regime.uncertainty == "point_only_no_interval"
    assert result.longitudinal_regime.support_status == "unlicensed_point_utility"
    assert result.longitudinal_regime.interval_reason == "insufficient_calibrated_horizon_or_trajectory_support_for_sequential_dr_interval"
    assert "subject_excluded_fold_predictions" in " ".join(result.assumptions or [])
    assert antecedent.analyze(data, query=query).longitudinal_regime == result.longitudinal_regime
    artifact = antecedent.load(prepared.export()).artifact
    assert artifact.payload["longitudinal_regime"]["method"] == "sequential_dr"
    query_artifact = antecedent.artifacts.loads(prepared.export_artifact(payload="query"))
    assert query_artifact.payload["longitudinal_regime"]["q_predictions"] == [1.0, 3.0, 5.0, 4.0, 2.0, 6.0, 1.0, 1.0]


def test_sequential_dr_supported_interval_keeps_q_fold_ownership_in_artifact():
    n = 300
    treatment = [[bool((i % 4) // 2), bool((i % 4) % 2)] for i in range(n)]
    outcomes = [2.0 + int(a0) + int(a1) + ((i % 7) - 3) / 10
                for i, (a0, a1) in enumerate(treatment)]
    folds = [i % 5 for i in range(n)]
    query = LongitudinalRegimeQuery(
        outcome="y", method="sequential_dr", treatment_history=treatment,
        actions=[True, True], treatment_probabilities=[[0.5, 0.5]] * n,
        q_predictions=[[4.0, 4.0]] * n, observation_history=[[True, True]] * n,
        subject_ids=[f"dr-{i}" for i in range(n)], fold_ids=folds,
        prediction_fold_ids=folds, excluded_fold_predictions=True,
    )
    result = antecedent.analyze({"y": outcomes}, query=query)
    value = result.longitudinal_regime
    assert value.value_standard_error is None
    assert value.value_interval_95 is None
    assert value.uncertainty == "point_only_no_interval"
    assert value.support_status == "unlicensed_point_utility"
    assert result.answer.detail == "longitudinal_regime_point_only"
    artifact = antecedent.load(result.export(artifact_id="sequential-dr-interval"))
    assert artifact.artifact.payload["longitudinal_regime"]["value_interval_95"] is None
    query_wire = artifact.artifact.payload["query"]["longitudinal_regime"]
    assert query_wire["prediction_fold_ids"] == query_wire["fold_ids"] == folds
    assert query_wire["excluded_fold_predictions"] is True


def test_three_period_sequential_dr_interval_and_four_period_refusal():
    n = 800
    treatment = [[bool(i & 1), bool(i & 2), bool(i & 4)] for i in range(n)]
    outcomes = [2.0 + sum(actions) + ((i % 11) - 5) / 20
                for i, actions in enumerate(treatment)]
    folds = [i % 5 for i in range(n)]
    args = dict(
        outcome="y", method="sequential_dr", treatment_history=treatment,
        actions=[True] * 3, treatment_probabilities=[[0.5] * 3] * n,
        censoring_probabilities=[[0.95] * 3] * n,
        q_predictions=[[4.4, 4.7, 5.1]] * n,
        observation_history=[[True] * 3] * n,
        subject_ids=[f"three-{i}" for i in range(n)], fold_ids=folds,
        prediction_fold_ids=folds, excluded_fold_predictions=True,
    )
    result = antecedent.analyze({"y": outcomes}, query=LongitudinalRegimeQuery(**args))
    value = result.longitudinal_regime
    assert value.value_interval_95[0] < 5.0 < value.value_interval_95[1]
    assert value.uncertainty == "pointwise_subject_score_conditional_excluded_fold_q_95"
    assert value.support_status == "licensed"
    assert result.support_status == "licensed"
    assert result.reasoning.support.payload["matrix_coordinate"].startswith("graphless:longitudinal_regime/")
    assert "sequential_q_validity" in " ".join(result.assumptions or [])
    artifact = antecedent.load(result.export(artifact_id="three-period-dr"))
    assert artifact.artifact.payload["longitudinal_regime"]["value_interval_95"] == pytest.approx(value.value_interval_95)

    four = dict(args)
    four.update(
        treatment_history=[actions + [bool(i & 8)] for i, actions in enumerate(treatment)],
        actions=[True] * 4,
        treatment_probabilities=[[0.5] * 4] * n,
        censoring_probabilities=[[0.95] * 4] * n,
        q_predictions=[[6.0] * 4] * n,
        observation_history=[[True] * 4] * n,
    )
    unsupported = antecedent.analyze({"y": outcomes}, query=LongitudinalRegimeQuery(**four)).longitudinal_regime
    assert unsupported.value_interval_95 is None
    assert unsupported.uncertainty == "point_only_no_interval"
    assert unsupported.support_status == "unlicensed_point_utility"


def test_two_period_sequential_dr_graphless_license_requires_calibrated_subject_support():
    n = 500
    treatment = [[i % 5 < 2, (i // 5) % 5 < 2] for i in range(n)]
    outcomes = [2.0 + sum(actions) + ((i % 7) - 3) / 10
                for i, actions in enumerate(treatment)]
    folds = [i % 5 for i in range(n)]
    args = dict(
        outcome="y", method="sequential_dr", treatment_history=treatment,
        actions=[True, True], treatment_probabilities=[[0.4, 0.4]] * n,
        censoring_probabilities=[[0.85, 0.85]] * n,
        q_predictions=[[4.0, 4.0]] * n,
        observation_history=[[True, True]] * n,
        subject_ids=[f"two-{i}" for i in range(n)], fold_ids=folds,
        prediction_fold_ids=folds, excluded_fold_predictions=True,
    )
    result = antecedent.analyze({"y": outcomes}, query=LongitudinalRegimeQuery(**args))
    value = result.longitudinal_regime
    assert value.value_interval_95 is not None
    assert value.support_status == "licensed"
    assert result.support_status == "licensed"
    artifact = antecedent.load(result.export(artifact_id="two-period-dr-license"))
    assert artifact.artifact.payload["longitudinal_regime"]["graphless_support_status"] == "licensed"
    args["subject_ids"] = args["subject_ids"][:499]
    args["treatment_history"] = args["treatment_history"][:499]
    args["treatment_probabilities"] = args["treatment_probabilities"][:499]
    args["censoring_probabilities"] = args["censoring_probabilities"][:499]
    args["q_predictions"] = args["q_predictions"][:499]
    args["observation_history"] = args["observation_history"][:499]
    args["fold_ids"] = args["fold_ids"][:499]
    args["prediction_fold_ids"] = args["prediction_fold_ids"][:499]
    small = antecedent.analyze({"y": outcomes[:499]}, query=LongitudinalRegimeQuery(**args))
    assert small.longitudinal_regime.value_interval_95 is None
    assert small.longitudinal_regime.support_status == "unlicensed_point_utility"


def test_sequential_dr_retained_refuses_split_fold_and_reappearing_observation():
    _, base = fixture()
    args = dict(outcome="y", method="sequential_dr", treatment_history=base.treatment_history,
                actions=base.actions, treatment_probabilities=base.treatment_probabilities,
                subject_ids=base.subject_ids, fold_ids=[0, 1, 0, 1],
                excluded_fold_predictions=True, q_predictions=[[1.0, 1.0]] * 4,
                observation_history=[[True, True]] * 4)
    with pytest.raises(ValueError, match="fold ownership"):
        LongitudinalRegimeQuery(**args, prediction_fold_ids=[1, 1, 0, 1])
    with pytest.raises(ValueError, match="monotone"):
        LongitudinalRegimeQuery(**{**args, "observation_history": [[False, True]] + [[True, True]] * 3},
                                prediction_fold_ids=[0, 1, 0, 1])


def test_msm_retained_matches_direct_and_round_trips_pointwise_uncertainty():
    import numpy as np
    from antecedent.regimes import fit_marginal_structural_model

    histories = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 4, axis=0)
    residuals = np.tile([-1.0, 0.0, 0.0, 1.0], 4)
    y = 10.0 + 2.0 * histories[:, 0] + 3.0 * histories[:, 1] + residuals
    ids = [f"s{i}" for i in range(16)]
    direct = fit_marginal_structural_model(
        y, histories, np.full((16, 2), 0.5),
        stabilizing_numerator_probabilities=[0.5, 0.5], subject_ids=ids,
        fold_ids=[i % 4 for i in range(16)],
    )
    query = LongitudinalRegimeQuery.marginal_structural_model(
        outcome="y", treatment_history=histories,
        treatment_probabilities=np.full((16, 2), 0.5),
        stabilizing_numerator_probabilities=[0.5, 0.5],
        subject_ids=ids, fold_ids=[i % 4 for i in range(16)],
    )
    prepared = antecedent.prepare({"y": y}, query=query)
    result = prepared.estimate()
    fit = result.longitudinal_regime
    assert fit.method == "marginal_structural_model"
    assert fit.value == pytest.approx(direct.intercept)
    assert fit.period_effects == pytest.approx(direct.period_effects)
    assert fit.standard_errors == pytest.approx(direct.standard_errors)
    assert fit.observed_subjects == direct.observed_subjects
    assert fit.uncertainty == "pointwise_subject_clustered_cr1_no_interval"
    assert fit.support_status == "unlicensed_point_utility"
    assert "additive_marginal_structural_mean" in " ".join(result.assumptions or [])
    assert antecedent.analyze({"y": y}, query=query).longitudinal_regime == fit
    artifact = antecedent.load(prepared.export()).artifact
    assert artifact.payload["longitudinal_regime"]["period_effects"] == pytest.approx([2.0, 3.0])
    query_artifact = antecedent.artifacts.loads(prepared.export_artifact(payload="query"))
    assert query_artifact.payload["longitudinal_regime"]["stabilizing_numerator_probabilities"] == [0.5, 0.5]


def test_msm_retained_refuses_invalid_numerators_and_unowned_probabilities():
    histories = [[False, False], [False, True], [True, False], [True, True]] * 2
    args = dict(outcome="y", method="marginal_structural_model", treatment_history=histories,
                actions=[False, False], treatment_probabilities=[[0.5, 0.5]] * 8,
                subject_ids=[f"s{i}" for i in range(8)])
    with pytest.raises(ValueError, match="requires one finite"):
        LongitudinalRegimeQuery(**args)
    with pytest.raises(ValueError, match="positivity floor"):
        LongitudinalRegimeQuery(**args, stabilizing_numerator_probabilities=[0.0, 0.5])
    query = LongitudinalRegimeQuery(**args, stabilizing_numerator_probabilities=[0.5, 0.5],
                                    probabilities_known_by_design=False,
                                    excluded_fold_predictions=True, fold_ids=[0, 1] * 4)
    with pytest.raises(CausalUnsupportedError, match="known sequential randomization"):
        antecedent.analyze({"y": list(range(8))}, query=query)
