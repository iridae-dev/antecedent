"""Released original-source producers and independent temporal interval consumers."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalCancelledError, CausalError, CausalTypeError
from antecedent.inference import MeasuredInference
from antecedent.recalc_temporal import (
    CheckedTemporalIntervalConfig,
    TemporalEffect,
    TemporalResponse,
    TemporalSession,
)
from antecedent.transport.advanced import InitialStateLaw, temporal_dependent_interval

from test_checked_temporal_interval import GAMMA, draw, fold, panel_digest, request, scores
from test_temporal_interval_activation import independent_replicates, panel


def replay(result, tmp_path):
    path = tmp_path / "measured-temporal.cbor"
    path.write_bytes(result.export())
    loaded = MeasuredInference.load(path.read_bytes(), expected=result.expected_identity)
    assert loaded.inspect() == result.inspect()
    assert loaded.source_report() == result.source_report()
    script = """import json,pathlib,sys
from antecedent.inference import MeasuredInference as M,MeasuredInferenceIdentity as I
r=M.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(r.inspect(),sort_keys=True))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path), json.dumps(result.expected_identity._wire())],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(child.stdout) == result.inspect()
    with pytest.raises(CausalError):
        MeasuredInference.load(
            result.export(), expected=replace(result.expected_identity, data_digest="f" * 64)
        )


def expected_interval(values, selected, method):
    points = selected.mean(axis=1)
    if method == "studentized":
        se = values.std(ddof=1) / len(values) ** 0.5
        per_draw = selected.std(axis=1, ddof=1) / len(values) ** 0.5
        pivots = (points - values.mean()) / per_draw
        tails = np.quantile(pivots, [0.025, 0.975])
        return (values.mean() - tails[1] * se, values.mean() - tails[0] * se), per_draw, pivots
    low, high = np.quantile(points, [0.025, 0.975])
    return (
        (
            (low, high)
            if method == "percentile"
            else (2 * values.mean() - high, 2 * values.mean() - low)
        ),
        None,
        None,
    )


@pytest.mark.parametrize(
    "functional,method",
    [
        (TemporalResponse((0, 0)), "studentized"),
        (TemporalEffect((1, 1), (0, 0)), "studentized"),
        (TemporalEffect((1, 1), (0, 0)), "percentile"),
        (TemporalEffect((1, 1), (0, 0)), "basic"),
    ],
)
def test_normal_checked_source_measured_scalar_independent_math_and_fresh_consumer(
    functional, method, tmp_path
):
    req = request(functional)
    session = TemporalSession()
    session.execute(req, seed=41)
    result = session.dependent_interval(
        config=CheckedTemporalIntervalConfig(method=method, bootstrap_seed=9171)
    )
    name = "effect" if isinstance(functional, TemporalEffect) else "response"
    scalar = result.scalar(name)
    source = result.source_report()
    values = scores(req)
    assert isinstance(result, MeasuredInference)
    assert scalar.calibration == "calibrated" and scalar.record_id.startswith(
        "cov.temporal_transport."
    )
    assert len(scalar.calibration_sha) == 40
    assert source["calibration"] == "unmeasured"
    assert source["source"]["request"]["functional"] == json.loads(
        json.dumps(req._wire()["functional"])
    )
    assert source["source"]["seed"] == 41 and source["config"]["seed"] == 9171
    assert (
        result.inspect()["validated_scope"]["functional"]
        == source["source"]["request"]["functional"]
    )
    base = fold(fold(GAMMA, 9171), panel_digest(req))
    selected = np.array([values[draw(len(values), fold(base, i))] for i in range(500)])
    expected, per_draw, pivots = expected_interval(values, selected, method)
    assert scalar.point == pytest.approx(values.mean(), abs=1e-12)
    assert scalar.interval == pytest.approx(expected, abs=1e-12)
    assert [r["point"] for r in source["result"]["replicates"]] == pytest.approx(
        selected.mean(axis=1), abs=1e-12
    )
    if method == "studentized":
        assert source["result"]["studentization"]["replicate_standard_errors"] == pytest.approx(
            per_draw, abs=1e-12
        )
        assert source["result"]["studentization"]["pivots"] == pytest.approx(pivots, abs=1e-10)
    assert not hasattr(session._handle, "interval_candidate")
    replay(result, tmp_path)


def test_normal_direct_whole_unit_measured_response_independent_math_and_fresh_consumer(tmp_path):
    result = temporal_dependent_interval(
        panel(),
        sequence=(0, 0),
        target_law=InitialStateLaw("fixed_target", {0: 0.3, 1: 0.7}),
        seed=901,
    )
    scalar = result.scalar("response")
    source = result.source_report()
    assert source["result"]["calibration"] == "unmeasured"
    assert scalar.record_id.endswith("temporal_two_step_units_studentized_l95")
    points = independent_replicates(source)
    values = np.array([0.49 + (0.2 if i % 2 else -0.2) for i in range(100)])
    selected = np.array(
        [values[draw(100, r["replicate_id"])] for r in source["result"]["replicates"]]
    )
    assert selected.mean(axis=1) == pytest.approx(points, abs=1e-12)
    expected, per_draw, pivots = expected_interval(values, selected, "studentized")
    assert scalar.point == pytest.approx(0.49, abs=1e-12)
    assert scalar.interval == pytest.approx(expected, abs=1e-12)
    assert source["result"]["studentization"]["replicate_standard_errors"] == pytest.approx(
        per_draw, abs=1e-12
    )
    assert source["result"]["studentization"]["pivots"] == pytest.approx(pivots, abs=1e-10)
    assert not hasattr(_native, "temporal_dependent_interval_candidate")
    replay(result, tmp_path)


def test_normal_checked_and_direct_neighboring_protocols_cannot_borrow_evidence():
    session = TemporalSession()
    session.execute(request(TemporalResponse((0, 0))), seed=41)
    for config in [
        CheckedTemporalIntervalConfig(method="percentile"),
        CheckedTemporalIntervalConfig(method="basic"),
        CheckedTemporalIntervalConfig(method="studentized", level=0.9),
        CheckedTemporalIntervalConfig(method="studentized", replicates=400),
    ]:
        with pytest.raises(CausalError):
            session.dependent_interval(config=config)
    for args in [
        {"method": "percentile"},
        {"sequence": (1, 1)},
        {"target_law": InitialStateLaw("different_target", {0: 0.4, 1: 0.6})},
    ]:
        options = {
            "sequence": (0, 0),
            "target_law": InitialStateLaw("fixed_target", {0: 0.3, 1: 0.7}),
            **args,
        }
        with pytest.raises(CausalError):
            temporal_dependent_interval(panel(), **options)
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        session.dependent_interval(cancel=token)
    with pytest.raises(CausalError):
        session.dependent_interval(memory_limit_bytes=0)
    calls = []
    session._handle = SimpleNamespace(measured_interval=lambda *a, **k: calls.append(1))
    with pytest.raises(CausalTypeError):
        session.dependent_interval()
    assert calls == []


@pytest.mark.parametrize("checked", [False, True])
def test_normal_temporal_measured_artifact_requires_original_source_expectations(checked, tmp_path):
    if checked:
        original = request(TemporalEffect((1, 1), (0, 0)))
        req = replace(
            original,
            units=tuple(
                replace(unit, unit_id=1009 + 17 * index)
                for index, unit in reversed(tuple(enumerate(original.units)))
            ),
        )
        session = TemporalSession()
        session.execute(req, seed=41)
        measured = session.dependent_interval(config=CheckedTemporalIntervalConfig(method="basic"))
        assert measured.source_report()["source"]["request"]["units"] == json.loads(
            json.dumps(req._wire()["units"])
        )
    else:
        measured = temporal_dependent_interval(
            panel(),
            sequence=(0, 0),
            target_law=InitialStateLaw("fixed_target", {0: 0.3, 1: 0.7}),
            seed=901,
        )
    replay(measured, tmp_path)
    loaded = MeasuredInference.load(measured.export(), expected=measured.expected_identity)
    assert loaded.source_artifact() == measured.source_artifact()
    assert loaded.source_report() == measured.source_report()
    for artifact, identity in (
        (measured.export(), replace(measured.expected_identity, candidate_digest="0" * 64)),
        (measured.export()[:-1], measured.expected_identity),
        (measured.source_artifact(), measured.expected_identity),
    ):
        with pytest.raises(CausalError):
            MeasuredInference.load(artifact, expected=identity)


@pytest.mark.parametrize("method", ["percentile", "basic"])
def test_checked_duplicate_history_cells_cannot_borrow_measured_protocol(method):
    req = request(TemporalEffect((1, 1), (0, 0)))
    unit = req.units[0]
    histories = list(unit.histories)
    histories[1] = replace(histories[0], time_id=histories[1].time_id)
    changed = replace(req, units=(replace(unit, histories=tuple(histories)), *req.units[1:]))
    session = TemporalSession()
    session.execute(changed, seed=41)
    with pytest.raises(CausalError):
        session.dependent_interval(config=CheckedTemporalIntervalConfig(method=method))
