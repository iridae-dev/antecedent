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
