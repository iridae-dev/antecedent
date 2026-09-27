"""Retained survival studies use analyze and the native result contract."""

from __future__ import annotations

import antecedent
import pytest
from antecedent.errors import CausalUnsupportedError
from antecedent.observation import IndependentGiven
from antecedent.survival import CompetingRisksOutcome, KnownCensoringSurvival, SurvivalOutcome


def test_survival_analyze_retains_rmst_curve_and_artifact() -> None:
    data = {
        "duration": [1.0, 2.0, 2.0, 2.0],
        "event": [1.0, 0.0, 0.0, 0.0],
        "treatment": [0.0, 0.0, 1.0, 1.0],
    }
    query = SurvivalOutcome(
        "duration", "event", "treatment", 2.0,
        randomized=True, observation_assumption=IndependentGiven(()),
    )
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.survival is not None
    assert result.survival.times == (0.0, 1.0, 2.0)
    assert result.survival.control_survival == (1.0, 0.5, 0.5)
    assert result.survival.treated_survival == (1.0, 1.0, 1.0)
    assert result.survival.rmst_control == pytest.approx(1.5)
    assert result.survival.rmst_treated == pytest.approx(2.0)
    assert result.survival.uncertainty == "point_only_no_interval"
    assert result.estimate.ate is None
    assert prepared.refresh(data).survival == result.survival
    assert antecedent.analyze(data, query=query).survival == result.survival
    loaded = antecedent.load(prepared.export(artifact_id="survival-study"))
    assert loaded.answer.kind == "structured"
    assert loaded.answer.structured["rmst_treated"] == pytest.approx(2.0)


def test_competing_risks_analyze_and_missing_observation_claim_refusal() -> None:
    data = {
        "duration": [1.0, 2.0, 1.0, 2.0],
        "cause": [1.0, 2.0, 1.0, 1.0],
        "treatment": [0.0, 0.0, 1.0, 1.0],
    }
    query = CompetingRisksOutcome(
        "duration", "cause", "treatment", 1, 2.0,
        randomized=True, observation_assumption=IndependentGiven(()),
    )
    result = antecedent.analyze(data, query=query)
    assert result.survival is not None
    assert result.survival.control_incidence == (0.0, 0.5, 0.5)
    assert result.survival.treated_incidence == (0.0, 0.5, 1.0)
    assert result.survival.target_cause == 1
    assert result.survival.uncertainty == "point_only_no_interval"

    no_claim = SurvivalOutcome("duration", "cause", "treatment", 2.0, randomized=True)
    with pytest.raises(CausalUnsupportedError, match="IndependentGiven"):
        antecedent.analyze(data, query=no_claim)


def test_known_censoring_survival_analyze_matches_native_ipcw_and_artifact() -> None:
    data = {
        "duration": [1.0, 3.0, 1.0, 3.0],
        "event": [1.0, 0.0, 1.0, 0.0],
        "treatment": [0.0, 0.0, 1.0, 1.0],
        "g0": [1.0] * 4,
        "g1": [1.0, 0.5, 0.5, 0.5],
        "g3": [1.0, 0.5, 0.5, 0.5],
        "baseline": [0.0, 1.0, 0.0, 1.0],
    }
    query = SurvivalOutcome(
        "duration", "event", "treatment", 3.0, randomized=True,
        observation_assumption=IndependentGiven(("baseline",)),
        known_censoring=KnownCensoringSurvival((0.0, 1.0, 3.0), ("g0", "g1", "g3")),
    )
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.survival.control_survival == pytest.approx((1.0, 2 / 3, 2 / 3))
    assert result.survival.treated_survival == pytest.approx((1.0, 0.5, 0.5))
    assert result.survival.rmst_difference == pytest.approx(-1 / 3)
    assert result.survival.uncertainty == "point_only_no_interval"
    assert result.estimate.ate is None
    assert prepared.refresh(data).survival == result.survival
    loaded = antecedent.load(prepared.export(artifact_id="ipcw-survival-study"))
    assert loaded.answer.structured["rmst_treated"] == pytest.approx(2.0)


def test_known_censoring_survival_refuses_bad_grid_and_probability() -> None:
    with pytest.raises(Exception, match="time grid must end at tau"):
        SurvivalOutcome(
            "duration", "event", "treatment", 3.0, randomized=True,
            observation_assumption=IndependentGiven(()),
            known_censoring=KnownCensoringSurvival((0.0, 1.0, 2.0), ("g0", "g1", "g2")),
        )
    data = {
        "duration": [1.0, 3.0, 1.0, 3.0], "event": [1.0, 0.0, 1.0, 0.0],
        "treatment": [0.0, 0.0, 1.0, 1.0],
        "g0": [1.0] * 4, "g1": [1.0, 0.001, 0.5, 0.5],
        "g3": [1.0, 0.001, 0.5, 0.5],
    }
    query = SurvivalOutcome(
        "duration", "event", "treatment", 3.0, randomized=True,
        observation_assumption=IndependentGiven(()),
        known_censoring=KnownCensoringSurvival((0.0, 1.0, 3.0), ("g0", "g1", "g3")),
    )
    with pytest.raises(Exception, match="positivity"):
        antecedent.analyze(data, query=query)


def test_known_censoring_competing_risk_retained_incidence() -> None:
    data = {
        "duration": [1.0, 1.0, 1.0, 2.0] * 2,
        "cause": [1.0, 2.0, 0.0, 0.0] * 2,
        "treatment": [0.0] * 4 + [1.0] * 4,
        "g0": [1.0] * 8,
        "g1": [1.0, 0.5, 0.5, 0.5] + [1.0] * 4,
        "g2": [1.0, 0.5, 0.5, 0.5] + [1.0] * 4,
    }
    query = CompetingRisksOutcome(
        "duration", "cause", "treatment", target_cause=1, tau=2.0,
        randomized=True, observation_assumption=IndependentGiven(()),
        known_censoring=KnownCensoringSurvival((0.0, 1.0, 2.0), ("g0", "g1", "g2")),
    )
    result = antecedent.analyze(data, query=query)
    assert result.survival.control_incidence[-1] == pytest.approx(1 / 7)
    assert result.survival.treated_incidence[-1] == pytest.approx(1 / 4)
    assert result.survival.uncertainty == "point_only_no_interval"
