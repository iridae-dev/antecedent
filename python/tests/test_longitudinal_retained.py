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
    assert "subject_excluded_fold_predictions" in " ".join(result.assumptions or [])
    assert antecedent.analyze(data, query=query).longitudinal_regime == result.longitudinal_regime
    artifact = antecedent.load(prepared.export()).artifact
    assert artifact.payload["longitudinal_regime"]["method"] == "sequential_dr"
    query_artifact = antecedent.artifacts.loads(prepared.export_artifact(payload="query"))
    assert query_artifact.payload["longitudinal_regime"]["q_predictions"] == [1.0, 3.0, 5.0, 4.0, 2.0, 6.0, 1.0, 1.0]


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
