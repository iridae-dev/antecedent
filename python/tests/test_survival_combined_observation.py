"""Retained delayed entry with fixed known censoring survival."""

from __future__ import annotations

import json
import subprocess
import sys

import antecedent as ant
import pandas as pd
import pytest
from antecedent.errors import CausalValueError
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success
from antecedent.survival import CompetingRisksOutcome, KnownCensoringSurvival, SurvivalOutcome


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
