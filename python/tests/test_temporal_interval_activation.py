"""Public activation preparation, not coverage or released-positive evidence.

Run on an installed calibration-internal wheel. Default wheels retain frozen
refusals; positive cases must be rerun on the normal wheel after actual activation.
The finite SCM uses the exact frozen target law and shared-unit-shock design.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.transport._temporal_extensions import (
    _temporal_dependent_interval_candidate as temporal_dependent_interval,
)
from antecedent.transport.advanced import (
    InitialStateLaw,
    TemporalIntervalCandidate,
    TemporalUnitPanel,
)

_INTERNAL = hasattr(_native, "temporal_dependent_interval_candidate")
_CANDIDATE = pytest.mark.skipif(
    not _INTERNAL, reason="requires calibration-internal acceptance wheel"
)
_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def panel(*, snapshot="activation-unit-panel", unit_offset=0, time_offset=0, lift=0.0):
    rows = []
    for unit in range(100):
        shock = 0.2 if unit % 2 else -0.2
        time = time_offset
        for s0 in range(2):
            for a1 in range(2):
                for l2 in range(2):
                    for a2 in range(2):
                        mean = 0.3 + 0.1 * l2 + 0.1 * s0 + 0.2 * s0 * l2 + 0.05 * (a1 + a2)
                        rows.append((unit_offset + unit, time, s0, a1, l2, a2, mean + shock + lift))
                        time += 1
    return TemporalUnitPanel.from_rows(snapshot, rows)


def produce(method="studentized", **panel_args):
    return temporal_dependent_interval(
        panel(**panel_args),
        sequence=(0, 0),
        estimand="marginalized_initial_state",
        target_law=InitialStateLaw("fixed_target", {0: 0.3, 1: 0.7}),
        method=method,
        replicates=500,
        min_units=20,
        seed=901,
        level=0.95,
    )


def mix(word):
    word = ((word ^ (word >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & _MASK
    return word ^ (word >> 31)


def fold(state, word):
    return (mix(state ^ word) + _GAMMA) & _MASK


def independent_replicates(payload):
    """Independent unit sampling plus SCM algebra, never the native estimator.

    Matching RNG is replay bookkeeping. Scientific reference uses E[Y|s0=0]=.35,
    E[Y|s0=1]=.55, hence .3*.35+.7*.55=.49 plus the selected unit shock mean.
    """
    base = fold(fold(_GAMMA, 901), int(payload["result"]["panel_digest"], 16))
    points = []
    for index, record in enumerate(payload["result"]["replicates"]):
        identity = fold(base, index)
        assert record["replicate_id"] == identity
        state = mix((identity + _GAMMA) & _MASK)
        selected = []
        digest = fold(_GAMMA, identity)
        for _ in range(100):
            state = (state + _GAMMA) & _MASK
            unit = (mix(state) * 100) >> 64
            selected.append(unit)
            digest = fold(digest, unit)
        assert record["selection_digest"] == digest
        truth = 0.49 + sum(0.2 if unit % 2 else -0.2 for unit in selected) / 100
        assert record["point"] == pytest.approx(truth, abs=1e-12)
        points.append(truth)
    return points


def test_temporal_candidate_requires_original_factory():
    with pytest.raises(CausalTypeError, match="temporal_dependent_interval"):
        TemporalIntervalCandidate()
    with pytest.raises(CausalTypeError, match="original native"):
        TemporalIntervalCandidate._from_native(SimpleNamespace(payload=lambda: "{}"))


@pytest.mark.skipif(_INTERNAL, reason="normal released wheel refusal boundary")
@pytest.mark.parametrize("method", ["studentized"])
def test_temporal_default_route_remains_frozen(method):
    with pytest.raises(CausalUnsupportedError) as caught:
        produce(method)
    assert caught.value.reason_code == "cell_not_licensed"
    assert caught.value.detail == "temporal_interval.route_frozen"


@_CANDIDATE
@pytest.mark.parametrize("method", ["studentized"])
def test_temporal_actual_public_candidate_matches_independent_unit_scm_and_fresh_consumer(
    method, tmp_path
):
    result = produce(method)
    assert isinstance(result, TemporalIntervalCandidate)
    assert result.point == pytest.approx(0.49, abs=1e-12)
    assert result.calibration == "unmeasured"
    assert result.claim == "dependence_preserving_calibration_unmeasured"
    with pytest.raises(FrozenInstanceError):
        result.calibration = "calibrated"
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        replace(result, calibration="calibrated")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        replace(result, point=9.0)
    payload = result.to_dict()
    assert payload["estimator"]["sequence"] == [0, 0]
    assert payload["config"]["seed"] == 901
    assert payload["panel"]["snapshot_id"] == "activation-unit-panel"
    assert payload["result"]["failed"] == 0
    draws = independent_replicates(payload)
    original_se = 0.2 / np.sqrt(99)
    resample_se = np.sqrt((0.04 - (np.array(draws) - 0.49) ** 2) / 99)
    pivots = (np.array(draws) - 0.49) / resample_se
    qlow, qhigh = np.quantile(pivots, [0.025, 0.975], method="linear")
    bounds = (0.49 - qhigh * original_se, 0.49 - qlow * original_se)
    receipt = payload["result"]["studentization"]
    assert receipt["standard_error"] == pytest.approx(original_se, abs=1e-12)
    np.testing.assert_allclose(
        receipt["replicate_standard_errors"], resample_se, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(receipt["pivots"], pivots, atol=1e-11, rtol=0)
    assert (result.lower, result.upper) == pytest.approx(bounds, abs=1e-12)
    artifact = tmp_path / "whole-unit.cbor"
    artifact.write_bytes(result.export())
    code = """
import json,sys
from pathlib import Path
from antecedent.transport.advanced import TemporalIntervalCandidate
r=TemporalIntervalCandidate.load(Path(sys.argv[1]).read_bytes(),expected_identity=sys.argv[2])
print(json.dumps(r.to_dict()))
"""
    fresh = subprocess.run(
        [sys.executable, "-c", code, str(artifact), result.identity],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(fresh.stdout) == payload


@_CANDIDATE
@pytest.mark.parametrize(
    "change",
    [
        {"snapshot": "another-snapshot"},
        {"unit_offset": 10_000},
        {"time_offset": 100},
        {"lift": 0.01},
    ],
)
def test_temporal_changed_panel_refuses_retained_identity(change):
    original = produce()
    changed = produce(**change)
    assert changed.identity != original.identity
    with pytest.raises(CausalUnsupportedError) as caught:
        TemporalIntervalCandidate.load(changed.export(), expected_identity=original.identity)
    assert caught.value.reason_code == "invalid_argument"
    assert caught.value.detail == "temporal_interval_artifact.expected_identity_mismatch"


@_CANDIDATE
def test_temporal_candidate_keeps_unknown_units_and_incompatible_history_refusals():
    with pytest.raises(CausalUnsupportedError) as caught:
        temporal_dependent_interval(TemporalUnitPanel("no-map", None), sequence=(0, 0))
    assert caught.value.detail == "temporal_interval.unknown_units"
    with pytest.raises(CausalUnsupportedError) as caught:
        temporal_dependent_interval(panel(), sequence=(7, 7))
    assert caught.value.detail == "temporal_interval.unsupported_history"


@pytest.mark.parametrize("method", ["percentile", "basic"])
def test_direct_temporal_retired_methods_are_not_public_options(method):
    from antecedent.errors import CausalValueError

    with pytest.raises(CausalValueError, match="require method='studentized'"):
        produce(method)
