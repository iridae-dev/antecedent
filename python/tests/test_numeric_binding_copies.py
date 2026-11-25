"""Contiguous numeric columns snapshot once at the five native boundaries."""

import dataclasses
import json

import antecedent as ant
import numpy as np
from antecedent import _native, regimes, survival
from antecedent._temporal_design import temporal_design_payload
from antecedent.temporal_designs import TwoArmExcursion

from test_canonical_trial_transport import fixture as trial_fixture
from test_canonical_trial_transport import prepare as prepare_trial
from test_joint_panel import declarations as panel_fixture
from test_joint_panel import request as prepare_panel
from test_posterior_boundaries import _one_period
from test_temporal_designs import _two_arm


def _copies(call, columns, *, adapted=()):
    value, error, raw = _native.observe_native_attempts(call)
    if error is not None:
        raise error
    report = json.loads(raw)
    assert report["input_copies"] == len(columns)
    assert report["input_copied_bytes"] == sum(columns)
    assert report["copies"] == len(columns) + len(adapted)
    assert report["copied_bytes"] == sum(columns) + sum(adapted)
    return value


def test_regime_takes_float64_indicators_without_a_python_conversion():
    data, arm, _ = _one_period(subjects=40)
    data["treatment"] = arm.astype(np.float64).reshape(-1, 1)
    trajectory = regimes._trajectories(
        data["treatment"],
        data["event"],
        data["at_risk"],
        data["treatment_probability"],
        data["censoring_weight"],
        data["minimum_probability"],
    )
    assert trajectory.treatment.dtype == np.float64
    assert np.shares_memory(trajectory.treatment, data["treatment"])
    _copies(lambda: regimes.trajectory_survival_contrast(**data), [40] * 3 + [40 * 8] * 2)


def test_survival_converts_float64_events_directly_into_native_codes():
    data = {
        "duration": np.array([1.0, 3.0, 1.0, 3.0]),
        "event": np.array([1.0, 0.0, 1.0, 0.0]),
        "treated": np.array([0.0, 0.0, 1.0, 1.0]),
    }
    query = survival.SurvivalOutcome("duration", "event", "treated", 3, randomized=True)
    times = np.array([0.0, 1.0, 3.0])
    g = np.ones((4, 3))
    _copies(
        lambda: survival.estimate_survival_ipcw(data, query, times=times, censoring_survival=g),
        [32, 24, 4, 32, 96],
    )


def test_survival_converts_float64_cause_codes_directly():
    data = {
        "duration": np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        "cause": np.array([1.0, 2.0, 0.0, 1.0, 2.0, 0.0]),
        "treated": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    }
    assert np.shares_memory(survival._cause_array(data["cause"]), data["cause"])
    query = survival.CompetingRisksOutcome(
        "duration", "cause", "treated", target_cause=1, tau=3, randomized=True
    )
    times = np.array([0.0, 1.0, 2.0, 3.0])
    g = np.ones((6, 4))
    _copies(
        lambda: survival.estimate_cumulative_incidence_ipcw(
            data, query, times=times, censoring_survival=g
        ),
        [48, 32, 6, 48, 192],
    )


def test_joint_panel_preparation_snapshots_numeric_source_columns():
    groups, worlds = panel_fixture()
    arrays = {
        group: {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
        for group, columns in groups.items()
    }
    source_columns = [values for columns in arrays.values() for values in columns.values()]
    # The fit adapter reads the owned native snapshot. It separates the two history rows
    # from follow-up and converts season levels to integer codes on that shorter grid.
    # Those writes remain visible in the aggregate, alongside one transfer per input.
    _copies(
        lambda: prepare_panel(arrays, worlds),
        [values.nbytes for values in source_columns],
        adapted=[(values.size - 2) * values.itemsize for values in source_columns],
    )


def test_temporal_design_snapshots_assignment_and_probability_columns():
    units, periods, assignment, probability, outcome = _two_arm(units=25, periods=5)
    data = {
        "unit": units,
        "period": np.asarray(periods, dtype=np.float64),
        "a": np.asarray(assignment, dtype=np.float64),
        "p": np.asarray(probability, dtype=np.float64),
        "y": outcome,
    }
    query = TwoArmExcursion("y", "unit", "period", assignment="a", probability="p", lag=1)
    _, _, bound = temporal_design_payload(data, query)
    assert np.shares_memory(bound["buffers"]["assignment"], data["a"])
    assert np.shares_memory(bound["periods"], data["period"])
    recoded = {**data, "period": data["period"].astype(np.int64), "a": data["a"].astype(bool)}
    _, _, rebound = temporal_design_payload(recoded, query)
    assert rebound["period_identity"] == bound["period_identity"]
    assert rebound["buffer_identity"] == bound["buffer_identity"]

    _copies(
        lambda: ant.prepare(data, query=query),
        [data["period"].nbytes, data["a"].size, data["p"].nbytes],
    )


def test_transport_snapshots_float64_input_columns():
    query, data = trial_fixture()
    data = dataclasses.replace(
        data,
        outcome=np.asarray(data.outcome, dtype=np.float64),
        treatment=np.asarray(data.treatment, dtype=bool),
        source=np.asarray(data.source, dtype=bool),
        randomization=np.asarray(data.randomization, dtype=np.float64),
    )
    _copies(
        lambda: prepare_trial(query, data),
        [data.outcome.nbytes, data.treatment.nbytes, data.source.nbytes, data.randomization.nbytes],
    )
