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


def test_retained_survival_pointwise_intervals_round_trip_and_refuse_thin_draws() -> None:
    data = {
        "duration": [1.0 if i % 5 == 0 else 3.0 for i in range(40)]
        + [1.0 if i % 10 == 0 else 3.0 for i in range(40)],
        "event": [float(i % 5 == 0) for i in range(40)]
        + [float(i % 10 == 0) for i in range(40)],
        "treatment": [0.0] * 40 + [1.0] * 40,
    }
    query = SurvivalOutcome(
        "duration", "event", "treatment", 3.0,
        randomized=True, observation_assumption=IndependentGiven(()),
    )
    prepared = antecedent.prepare(data, query=query, bootstrap=399, seed=17)
    result = prepared.estimate()
    section = result.survival
    assert section is not None
    assert section.uncertainty == "subject_stratified_percentile_bootstrap_pointwise_95"
    assert section.bootstrap_replicates_requested == 399
    assert section.bootstrap_replicates_ok == 399
    assert section.rmst_difference_interval[0] <= section.rmst_difference <= section.rmst_difference_interval[1]
    assert section.survival_at_tau_difference_interval[0] <= 0.1 <= section.survival_at_tau_difference_interval[1]
    assert result.answer.detail == "randomized_survival_pointwise_bootstrap"
    assert "Pointwise 95% subject-bootstrap interval" in result.claim()
    assert prepared.refresh(data).survival == section
    loaded = antecedent.load(prepared.export(artifact_id="survival-pointwise-interval"))
    assert loaded.answer.structured["rmst_difference_interval"] == list(section.rmst_difference_interval)
    assert loaded.answer.structured["bootstrap_replicates_ok"] == 399
    assert loaded.answer.detail == "randomized_survival_pointwise_bootstrap"
    with pytest.raises(Exception, match="199"):
        antecedent.analyze(data, query=query, bootstrap=198)


def test_retained_competing_risk_known_g_pointwise_interval() -> None:
    n = 40
    duration = [1.0 if i % 5 == 0 or i % 7 == 0 else 3.0 for i in range(n)] * 2
    causes = [1.0 if i % 5 == 0 else 2.0 if i % 7 == 0 else 0.0 for i in range(n)] * 2
    data = {
        "duration": duration,
        "cause": causes,
        "treatment": [0.0] * n + [1.0] * n,
        "g0": [1.0] * (2 * n),
        "g1": [0.8] * (2 * n),
        "g3": [0.7] * (2 * n),
    }
    query = CompetingRisksOutcome(
        "duration", "cause", "treatment", 1, 3.0,
        randomized=True, observation_assumption=IndependentGiven(()),
        known_censoring=KnownCensoringSurvival((0.0, 1.0, 3.0), ("g0", "g1", "g3")),
    )
    section = antecedent.analyze(data, query=query, bootstrap=399, seed=19).survival
    assert section is not None
    assert section.incidence_difference_interval[0] <= 0.0 <= section.incidence_difference_interval[1]
    assert section.bootstrap_replicates_ok == 399
