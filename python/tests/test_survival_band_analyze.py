"""Public retained simultaneous survival and competing-risk curve bands."""

from __future__ import annotations

import antecedent as ant
import pytest
from antecedent.observation import IndependentGiven
from antecedent.survival import CompetingRisksOutcome, KnownCensoringSurvival, SurvivalOutcome


def _study_data(*, competing: bool = False) -> dict[str, list[float]]:
    duration = []
    event = []
    treatment = []
    for arm in (0.0, 1.0):
        for i in range(100):
            duration.append(1.0 if i % (4 if arm else 5) == 0 else 2.0 if i % 7 == 0 else 3.0)
            event.append(
                (1.0 if i % 2 == 0 else 2.0)
                if competing and duration[-1] < 3.0
                else 1.0
                if duration[-1] < 3.0
                else 0.0
            )
            treatment.append(arm)
    return {"duration": duration, "event": event, "treatment": treatment}


@pytest.mark.parametrize("competing", [False, True])
def test_simultaneous_difference_band_round_trip_and_determinism(competing: bool) -> None:
    data = _study_data(competing=competing)
    query = (
        CompetingRisksOutcome(
            "duration",
            "event",
            "treatment",
            target_cause=1,
            tau=3.0,
            randomized=True,
            observation_assumption=IndependentGiven(()),
        )
        if competing
        else SurvivalOutcome(
            "duration",
            "event",
            "treatment",
            3.0,
            randomized=True,
            observation_assumption=IndependentGiven(()),
        )
    )
    first = ant.analyze(data, query=query, bootstrap=399, seed=923)
    second = ant.analyze(data, query=query, bootstrap=399, seed=923)
    band = first.survival.difference_band
    assert band is not None
    assert band == second.survival.difference_band
    assert band.level == 0.95
    assert band.replicates_ok >= 399
    assert band.times == first.survival.times
    assert len(band.lower) == len(band.upper) == len(band.difference)
    assert all(
        lower <= point <= upper
        for lower, point, upper in zip(band.lower, band.difference, band.upper, strict=True)
    )
    assert first.survival.band_unavailable_reason is None
    body = ant.load(first.export()).artifact.payload
    assert body["survival"]["difference_band"]["lower"] == pytest.approx(band.lower)
    assert body["survival"]["difference_band"]["upper"] == pytest.approx(band.upper)


def test_simultaneous_band_refuses_fixed_censoring_weights_and_delayed_entry() -> None:
    data = _study_data()
    data.update(
        {
            "g0": [1.0] * 200,
            "g1": [1.0] * 200,
            "g2": [1.0] * 200,
            "g3": [1.0] * 200,
            "entry": [0.0] * 200,
        }
    )
    weighted = SurvivalOutcome(
        "duration",
        "event",
        "treatment",
        3.0,
        randomized=True,
        observation_assumption=IndependentGiven(()),
        known_censoring=KnownCensoringSurvival((0.0, 1.0, 2.0, 3.0), ("g0", "g1", "g2", "g3")),
    )
    weighted_result = ant.analyze(data, query=weighted, bootstrap=399, seed=923).survival
    assert weighted_result.difference_band is None
    assert "censoring" in weighted_result.band_unavailable_reason
    assert weighted_result.rmst_difference_interval is not None

    delayed = SurvivalOutcome(
        "duration",
        "event",
        "treatment",
        3.0,
        randomized=True,
        delayed_entry="entry",
        observation_assumption=IndependentGiven(()),
    )
    delayed_result = ant.analyze(data, query=delayed, bootstrap=399, seed=923).survival
    assert delayed_result.difference_band is None
    assert "delayed entry" in delayed_result.band_unavailable_reason
    assert delayed_result.rmst_difference_interval is not None


def test_simultaneous_band_withholds_thin_arms_but_keeps_scalar_interval() -> None:
    data = _study_data()
    for column in data:
        data[column] = data[column][:60] + data[column][100:160]
    query = SurvivalOutcome(
        "duration",
        "event",
        "treatment",
        3.0,
        randomized=True,
        observation_assumption=IndependentGiven(()),
    )
    result = ant.analyze(data, query=query, bootstrap=399, seed=923).survival
    assert result.difference_band is None
    assert "80 subjects" in result.band_unavailable_reason
    assert result.rmst_difference_interval is not None
