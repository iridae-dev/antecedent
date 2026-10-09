"""Intended 90%/95% public lifecycle on internal builds, never coverage measurement.

Normal installed wheels retain frozen refusal. Internal successes must be rerun
on the released wheel after actual calibration and explicit activation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.graph import Admg
from antecedent.transport.advanced import (
    NestedFisherCandidate,
    binary_nested_markov_fisher_interval,
)

_INTERNAL = hasattr(_native, "nested_markov_fisher_candidate")
_CANDIDATE = pytest.mark.skipif(not _INTERNAL, reason="requires calibration-internal acceptance wheel")


def graph():
    return Admg.from_edges(
        ["X1", "X2", "X3", "X4"],
        [("X1", "X2"), ("X2", "X3"), ("X3", "X4")],
        [("X2", "X4")],
    )


def cells(n=16000):
    # Independently sum the structural factors: fair X1 and X2, mediating X3,
    # and Y depending on X3. No native probability/fitted receipt supplies counts.
    return [
        round(n * 0.25 * (c if m == 0 else 1-c) * (q if y == 0 else 1-q))
        for _x1 in range(2)
        for c in (0.75, 0.25)
        for m, q in enumerate((0.8, 0.4))
        for y in range(2)
    ]


def produce(level=0.95, n=16000):
    return binary_nested_markov_fisher_interval(
        graph=graph(), regimes=[{"counts": cells(n)}], nominal_level=level,
    )


def original_law(p):
    a, c0, c1, q20, q21, q40, q41, g00, g01, g10, g11 = p
    margins2 = (q20, q21)
    margins4 = (q40, q41)
    associations = ((g00, g01), (g10, g11))
    out = []
    for x1 in range(2):
        for x2 in range(2):
            for x3 in range(2):
                g = associations[x1][x3]
                table = (g, margins2[x1]-g, margins4[x3]-g, 1-margins2[x1]-margins4[x3]+g)
                c = (c0, c1)[x2]
                for x4 in range(2):
                    out.append((a if x1 == 0 else 1-a) * (c if x3 == 0 else 1-c) * table[x2*2+x4])
    return np.array(out)


@pytest.mark.skipif(_INTERNAL, reason="normal released wheel refusal boundary")
@pytest.mark.parametrize("level", [0.90, 0.95])
def test_nested_fisher_normal_route_remains_frozen(level):
    with pytest.raises(CausalUnsupportedError) as caught:
        produce(level)
    assert caught.value.reason_code == "cell_not_licensed"
    assert "nested_markov.route_frozen" in str(caught.value)


@_CANDIDATE
@pytest.mark.parametrize("level", [0.90, 0.95])
def test_nested_fisher_actual_public_candidate_has_independent_full_covariance_and_fresh_consumer(level, tmp_path):
    result = produce(level)
    assert isinstance(result, NestedFisherCandidate)
    assert result.calibration == "unmeasured"
    assert result.inference == "interval_withheld_calibration_unmeasured"
    assert result.identification == "nonparametrically_identified"
    assert result.values == pytest.approx((0.3, 0.5, 0.2), abs=1e-12)
    oracle = json.loads((Path(__file__).resolve().parents[2] / "conformance/transport/nested_fisher/expected.json").read_text())
    covariance = np.array(oracle["covariance_times_n"]) / oracle["sample_size"]
    np.testing.assert_allclose(result.covariance, covariance, atol=1e-12, rtol=0)
    z = NormalDist().inv_cdf(0.5+level/2)
    radius = z*np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(result.intervals, np.column_stack((np.array(result.values)-radius, np.array(result.values)+radius)), atol=1e-10, rtol=0)
    payload = result.to_dict()
    p = payload["point"]["receipt"]["parameters"]
    coordinates = np.array([p["a"], *p["c"], *p["q2"], *p["q4"], *p["g"][0], *p["g"][1]])
    # Independent numerical differentiation of the ORIGINAL model, followed by
    # NumPy inversion: validates all 121 covariance entries, not inverse residuals.
    jacobian = np.empty((16, 11))
    step = 1e-6
    for k in range(11):
        delta = np.zeros(11)
        delta[k] = step
        jacobian[:, k] = (original_law(coordinates+delta)-original_law(coordinates-delta))/(2*step)
    fisher = oracle["sample_size"]*(jacobian.T/original_law(coordinates)) @ jacobian
    np.testing.assert_allclose(np.array(payload["parameter_covariance"]).reshape(11, 11), np.linalg.inv(fisher), atol=2e-12, rtol=1e-8)
    artifact = tmp_path / "nested-fisher.cbor"
    artifact.write_bytes(result.export())
    code = """
import json,sys
from pathlib import Path
from antecedent.transport.advanced import NestedFisherCandidate
r=NestedFisherCandidate.load(Path(sys.argv[1]).read_bytes(),expected_identity=sys.argv[2])
print(json.dumps(r.to_dict()))
"""
    fresh = subprocess.run([sys.executable, "-c", code, str(artifact), result.identity], capture_output=True, text=True, check=True)
    assert json.loads(fresh.stdout) == payload


@_CANDIDATE
def test_nested_fisher_candidate_retains_original_identity_and_scope_refusals():
    original = produce()
    changed = produce(n=32000)
    assert original.identity != changed.identity
    with pytest.raises(CausalValueError) as caught:
        NestedFisherCandidate.load(changed.export(), expected_identity=original.identity)
    assert caught.value.reason_code == "invalid_argument"
    assert "identity_mismatch" in str(caught.value)
    fractional = cells()
    fractional[0] += 0.5
    with pytest.raises(CausalValueError) as caught:
        binary_nested_markov_fisher_interval(graph=graph(), regimes=[{"counts": fractional}])
    assert caught.value.reason_code == "invalid_argument"
    assert "fisher_sampling_design" in str(caught.value)
    with pytest.raises(CausalValueError) as caught:
        produce(0.80)
    assert caught.value.reason_code == "invalid_argument"
    assert "fisher_invalid_level" in str(caught.value)


def test_nested_fisher_python_validation_preserves_domain_errors_for_overflow_and_malformed_tables():
    from antecedent.errors import CausalTypeError

    for arguments in ({"nominal_level": 10**400}, {"max_iterations": 10**400}, {"tolerance": 10**400}):
        with pytest.raises(CausalValueError) as caught:
            binary_nested_markov_fisher_interval(graph=graph(), regimes=[{"counts": cells()}], **arguments)
        assert caught.value.reason_code == "invalid_argument"
    with pytest.raises(CausalValueError) as caught:
        binary_nested_markov_fisher_interval(graph=graph(), regimes=[{"counts": [10**400]*16}])
    assert caught.value.reason_code == "invalid_argument"
    with pytest.raises(CausalTypeError) as caught:
        binary_nested_markov_fisher_interval(graph=graph(), regimes=[{"counts": cells(), "levels": 2}])
    assert caught.value.reason_code == "invalid_argument"
