"""Retained delayed entry with fixed known censoring survival."""

from __future__ import annotations

import json
import subprocess
import sys

import antecedent as ant
import pandas as pd
import pytest
from antecedent import _reason_details as reason_details
from antecedent.errors import CausalValueError, StructuredRefusal
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success
from antecedent.survival import (
    CompetingRisksOutcome,
    KnownCensoringSurvival,
    StratifiedCensoringSurvival,
    SurvivalOutcome,
)


def _data(arm_size: int = 80) -> pd.DataFrame:
    rows = []
    for arm in (0, 1):
        for i in range(arm_size):
            entry = 1.0 if i % 3 == 0 else 0.0
            if entry == 0.0 and i % 7 == 0:
                duration, cause = 1.0, 1
            elif i % 11 == 0:
                duration, cause = 1.5, 0
            elif i % 5 == 0:
                duration, cause = 2.0, 2
            elif i % (4 if arm else 3) == 0:
                duration, cause = 2.0, 1
            else:
                duration, cause = 3.0, 0
            rows.append((duration, cause, int(cause > 0), arm, entry, 1.0, 1.0, 1.0, 0.8, 0.8))
    return pd.DataFrame(
        rows,
        columns=(
            "duration",
            "cause",
            "event",
            "treated",
            "entry",
            "g0",
            "g1",
            "g15",
            "g2",
            "g3",
        ),
    )


def _g() -> KnownCensoringSurvival:
    return KnownCensoringSurvival((0.0, 1.0, 1.5, 2.0, 3.0), ("g0", "g1", "g15", "g2", "g3"))


@pytest.mark.parametrize("competing", [False, True])
def test_combined_entry_fixed_g_pointwise_interval_round_trip(competing: bool) -> None:
    data = _data()
    if competing:
        query = CompetingRisksOutcome(
            "duration",
            "cause",
            "treated",
            1,
            3.0,
            randomized=True,
            delayed_entry="entry",
            known_censoring=_g(),
            observation_assumption=IndependentGiven(()),
        )
    else:
        data.loc[data["cause"] == 2, "event"] = 1
        query = SurvivalOutcome(
            "duration",
            "event",
            "treated",
            3.0,
            randomized=True,
            delayed_entry="entry",
            known_censoring=_g(),
            observation_assumption=IndependentGiven(()),
        )
    result = require_success(
        ant.analyze(
            data,
            query=query,
            inference=ant.Frequentist(bootstrap=299),
            execution=ant.Execution(seed=297),
        )
    )
    section = result.survival
    assert section is not None
    assert section.uncertainty == "subject_stratified_percentile_bootstrap_pointwise_95"
    assert section.censoring_survival_provenance == "caller_supplied_fixed_not_fitted_or_verified"
    assert section.difference_band is None
    assert "delayed entry" in section.band_unavailable_reason
    assert section.bootstrap_replicates_ok == 299
    if competing:
        assert section.incidence_difference_interval is not None
    else:
        assert section.rmst_difference_interval is not None
        assert section.survival_at_tau_difference_interval is not None
    body = require_success(ant.load(bytes(result.export()))).decoded_artifact.payload
    assert body["query"]["survival"]["delayed_entry"] is not None
    assert len(body["query"]["survival"]["censoring_columns"]) == 5
    assert body["survival"]["difference_at_tau_interval"] is not None


@pytest.mark.parametrize("fixed_known", [False, True])
def test_delayed_entry_incidence_support_round_trips(fixed_known: bool) -> None:
    # Mirrors the Rust retained route fixture, including the seed and subject-bootstrap size.
    data = (
        _data(200 if fixed_known else 160)
        .drop(columns="event")
        .rename(columns={"cause": "event", "treated": "treatment"})
        .astype(float)
    )
    query = CompetingRisksOutcome(
        "duration",
        "event",
        "treatment",
        target_cause=1,
        tau=3.0,
        randomized=True,
        delayed_entry="entry",
        known_censoring=_g() if fixed_known else None,
        observation_assumption=IndependentGiven(()),
    )
    result = require_success(
        ant.analyze(
            data,
            query=query,
            inference=ant.Frequentist(bootstrap=299),
            execution=ant.Execution(seed=297),
        )
    )
    section = result.survival
    assert section.target_cause == query.target_cause
    assert section.tau == query.tau
    assert section.support_status == "supported"
    assert section.difference_band is None
    assert section.incidence_difference_interval is not None
    compact = require_success(ant.load(bytes(result.export()))).replay()
    assert "dependencies.checked_survival_incidence_operation" in compact.acceptance.not_replayable
    reader = """
import json
import sys
import antecedent as ant
from antecedent.composition_bundle import Bundle, consume_bundle
from antecedent.outcomes import require_success
raw = sys.stdin.buffer.read()
loaded = require_success(ant.load(raw)).replay()
assert loaded.acceptance.recognized
assert loaded.acceptance.replayed, loaded.acceptance
bundle = Bundle.builder().add_artifact(raw, node_id="incidence").build()
consumed = require_success(consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity))
consumed.require_verified()
assert consumed.value("incidence", "horizon") == 3.0
print(json.dumps(dict(loaded.decoded_artifact.payload["survival"])))
"""
    retained = subprocess.run(
        [sys.executable, "-c", reader],
        input=bytes(result.export(source_data=data)),
        capture_output=True,
        check=True,
        timeout=30,
    )
    curve = json.loads(retained.stdout)
    assert curve["target_cause"] == query.target_cause
    assert curve["tau"] == query.tau
    assert curve["graphless_support_status"] == "supported"
    assert curve["difference_at_tau_interval"] == pytest.approx(section.incidence_difference_interval)


@pytest.mark.parametrize("fixed_known", [False, True])
@pytest.mark.parametrize("incidence", [False, True])
def test_delayed_entry_band_replays_in_a_bundle(fixed_known: bool, incidence: bool) -> None:
    rows = []
    for arm in (0, 1):
        first = 40 if arm else 60
        for index in range(200):
            if index < first:
                exit_time, event, entry, g = 1.0, 1.0, 0.0, 0.5
            elif index < first + 40:
                exit_time, event, entry, g = 2.0, (2.0 if incidence else 1.0), 1.0, 0.5
            else:
                exit_time, event, entry, g = 3.0, 0.0, 0.0, 0.8
            rows.append((exit_time, event, float(arm), entry, 1.0, g, g, g))
    data = pd.DataFrame(
        rows, columns=("duration", "event", "treatment", "entry", "g0", "g1", "g2", "g3")
    )
    known = (
        KnownCensoringSurvival((0.0, 1.0, 2.0, 3.0), ("g0", "g1", "g2", "g3"), 0.05)
        if fixed_known else None
    )
    observation = {
        "randomized": True,
        "delayed_entry": "entry",
        "known_censoring": known,
        "observation_assumption": IndependentGiven(()),
    }
    query = (
        CompetingRisksOutcome("duration", "event", "treatment", 1, 3.0, **observation)
        if incidence else SurvivalOutcome("duration", "event", "treatment", 3.0, **observation)
    )
    result = require_success(
        ant.analyze(data, query=query, inference=ant.Frequentist(bootstrap=399),
                    execution=ant.Execution(seed=5_167))
    )
    section = result.survival
    band = section.difference_band
    assert band.minimum_horizon_risk_set == 100
    assert tuple(band.times) == (0.0, 1.0, 2.0, 3.0)
    assert band.replicates_ok == 399
    compact = require_success(ant.load(bytes(result.export()))).replay()
    assert "dependencies.checked_survival_band_operation" in compact.acceptance.not_replayable
    reader = """
import json
import sys
import antecedent as ant
from antecedent.composition_bundle import Bundle, consume_bundle
from antecedent.outcomes import require_success
source = sys.stdin.buffer.read()
assert require_success(ant.load(source)).replay().acceptance.replayed
bundle = Bundle.builder().add_artifact(source, node_id="band").build()
consumed = require_success(consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity))
consumed.require_verified()
print(json.dumps([consumed.value("band", key) for key in (
    "horizon", "horizon_lower_95", "whole_grid_upper_95"
)]))
"""
    printed = subprocess.run(
        [sys.executable, "-c", reader], input=bytes(result.export(source_data=data)),
        capture_output=True, check=True, timeout=30,
    )
    assert json.loads(printed.stdout) == pytest.approx([3.0, band.lower[-1], max(band.upper)])


def test_fitted_censoring_arrays_cannot_enter_delayed_entry_incidence_support() -> None:
    with pytest.raises(CausalValueError, match="known_censoring must be KnownCensoringSurvival"):
        CompetingRisksOutcome(
            "duration",
            "cause",
            "treated",
            target_cause=1,
            tau=3.0,
            randomized=True,
            delayed_entry="entry",
            observation_assumption=IndependentGiven(()),
            known_censoring=_data()[["g0", "g1", "g15", "g2", "g3"]].to_numpy(),
        )


def test_conditional_censoring_refits_the_pre_entry_population_in_a_bundle() -> None:
    # Exact finite product law shared with the Rust fixture.
    events = ((((2, 1), (3, 2)), ((1, 1), (2, 1))),
              (((1, 1), (2, 1)), ((2, 2), (4, 2))))
    rows = []
    for arm in range(2):
        for left in range(2):
            for censor in range(2):
                early, later = events[arm][left][censor]
                for event_draw in range(8):
                    event_time = 1.0 if event_draw < early else 2.0 if event_draw < early + later else 3.0
                    for entry_draw in range(8):
                        entry_time = float(entry_draw < (1, 6)[left])
                        for censor_draw in range(8):
                            censor_time = 1.5 if censor_draw < (1, 3)[censor] else 3.0
                            exit_time = min(event_time, censor_time)
                            if entry_time < exit_time:
                                rows.append((exit_time, float(event_time < 3.0 and event_time <= censor_time),
                                             float(arm), entry_time, float(left), float(censor)))
    data = pd.DataFrame(rows, columns=("duration", "event", "treatment", "entry", "entry_history", "censor_history"))
    learner = StratifiedCensoringSurvival("entry_history", "censor_history", (0.5, 0.5))
    assumption = IndependentGiven(("entry_history", "censor_history"))
    query = SurvivalOutcome("duration", "event", "treatment", 3.0, randomized=True,
                            delayed_entry="entry", observation_assumption=assumption,
                            stratified_censoring=learner)
    inference = ant.Frequentist(bootstrap=399)
    context = ant.Execution(seed=9_732)
    result = require_success(ant.analyze(data, query=query, inference=inference, execution=context))
    curve = result.survival
    band = curve.difference_band
    assert curve.censoring_survival_provenance == "same_subject_stratified_reverse_product_limit_refit"
    assert curve.support_status == "supported_simultaneous_band"
    assert "fixed_pre_entry_source_population" in curve.assumptions
    assert band.times == (0.0, 1.0, 2.0, 3.0)
    assert band.difference == pytest.approx((0.0, -0.03125, -0.0625, -0.0625))
    compact = require_success(ant.load(bytes(result.export()))).replay()
    assert "dependencies.checked_survival_conditional_training" in compact.acceptance.not_replayable
    replay = """
import json
import sys
from antecedent.composition_bundle import Bundle, consume_bundle
from antecedent.outcomes import require_success
source = sys.stdin.buffer.read()
bundle = Bundle.builder().add_artifact(source, node_id="fit").build()
consumed = require_success(consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity))
consumed.require_verified()
print(json.dumps(consumed.value("fit", "horizon_lower_95")))
"""
    printed = subprocess.run(
        [sys.executable, "-c", replay], input=bytes(result.export(source_data=data)),
        capture_output=True, check=True, timeout=30,
    )
    assert json.loads(printed.stdout) == pytest.approx(band.lower[-1])

    with pytest.raises(CausalValueError, match="conditional delayed entry"):
        SurvivalOutcome("duration", "event", "treatment", 3.0, randomized=True,
                        delayed_entry="entry", observation_assumption=assumption)
    with pytest.raises(StructuredRefusal) as missing:
        require_success(ant.analyze(data.drop(columns="censor_history"), query=query, inference=inference))
    assert missing.value.detail == reason_details.survival.CONDITIONAL_OBSERVATION_IDENTIFICATION_INPUTS
    revised = data.copy()
    revised.loc[(revised["treatment"] == 1.0) & (revised["entry_history"] == 1.0), "censor_history"] = 0.0
    with pytest.raises(Exception, match="inadequate entry or follow-up overlap"):
        require_success(ant.analyze(revised, query=query, inference=inference, execution=context))


def test_combined_entry_fixed_g_refuses_conditional_entry_and_bad_g() -> None:
    with pytest.raises(CausalValueError, match="conditional delayed entry"):
        SurvivalOutcome(
            "duration",
            "event",
            "treated",
            3.0,
            randomized=True,
            delayed_entry="entry",
            known_censoring=_g(),
            observation_assumption=IndependentGiven(("baseline",)),
        )
    data = _data()
    data.loc[data["cause"] == 2, "event"] = 1
    data.loc[0, "g2"] = 0.0
    query = SurvivalOutcome(
        "duration",
        "event",
        "treated",
        3.0,
        randomized=True,
        delayed_entry="entry",
        known_censoring=_g(),
        observation_assumption=IndependentGiven(()),
    )
    with pytest.raises(Exception, match="positivity|censoring survival"):
        require_success(
            ant.analyze(
                data,
                query=query,
                inference=ant.Frequentist(bootstrap=299),
                execution=ant.Execution(seed=297),
            )
        )
