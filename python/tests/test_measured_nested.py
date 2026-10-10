"""Normal nested scalar licensing retains original unmeasured model artifacts."""

import json
import subprocess
import sys
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalCancelledError, CausalError
from antecedent.inference import MeasuredInference
from antecedent.transport import advanced as tr

from test_nested_bayesian_activation import graph


def produce(bayesian, shape=1, **changes):
    kwargs = {"graph": graph(), "regimes": [{"counts": [250] * 16}]} | changes
    if bayesian:
        prior = tr.NestedMarkovPrior(
            **{name: (shape, shape) for name in tr.NestedMarkovPrior().coordinates}
        )
        return tr.binary_nested_markov(**({"prior": prior, "seed": 817} | kwargs))
    return tr.binary_nested_markov_fisher_interval(**kwargs)


@pytest.mark.parametrize("bayesian,shape", [(False, 1), (True, 1), (True, 2)])
def test_normal_nested_all_three_measured_scalars_replay_original_source(bayesian, shape, tmp_path):
    assert hasattr(
        _native, "nested_markov_posterior_measured" if bayesian else "nested_markov_fisher_measured"
    )
    measured = produce(bayesian, shape)
    assert isinstance(measured, MeasuredInference)
    assert [scalar.name for scalar in measured.scalars] == ["mean0", "mean1", "contrast"]
    source = measured.source_report()
    assert source["calibration"] == "unmeasured"
    if bayesian:
        posterior = source["posterior"]
        samples = np.asarray(posterior["samples"]).reshape(4 * 4096, 14)
        assert samples[:, 13] == pytest.approx(samples[:, 12] - samples[:, 11], abs=1e-14)
        points = np.asarray(posterior["mean"])[11:14]
        intervals = np.quantile(samples[:, 11:14], [0.025, 0.975], axis=0).T
        assert points == pytest.approx(samples[:, 11:14].mean(axis=0), abs=2e-12)
        assert source["inference"] == "posterior_candidate_withheld_calibration_unmeasured"
    else:
        contrast = source["point"]["receipt"]["contrast"]
        points = np.array([*contrast["model_means"], contrast["model_contrast"]])
        covariance = np.asarray(source["effect_covariance"]).reshape(3, 3)
        radius = NormalDist().inv_cdf(0.975) * np.sqrt(np.diag(covariance))
        intervals = np.column_stack((points - radius, points + radius))
    for index, scalar in enumerate(measured.scalars):
        assert scalar.point == pytest.approx(points[index], abs=2e-12)
        assert scalar.interval == pytest.approx(intervals[index], abs=2e-12)
        assert scalar.calibration == "calibrated" and scalar.level == 0.95
        assert scalar.record_id and len(scalar.calibration_sha) == 40
        assert scalar.basis["scope"]["row_count"] == 4000
    path = tmp_path / "measured-nested.cbor"
    path.write_bytes(measured.export())
    loaded = MeasuredInference.load(path.read_bytes(), expected=measured.expected_identity)
    assert loaded.inspect() == measured.inspect()
    assert loaded.source_artifact() == measured.source_artifact()
    code = """
import json,pathlib,sys
from antecedent.inference import MeasuredInference as M,MeasuredInferenceIdentity as I
r=M.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(r.inspect(),sort_keys=True))
"""
    observed = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(path), json.dumps(measured.expected_identity._wire())],
            text=True,
        )
    )
    assert observed == measured.inspect()
    with pytest.raises(CausalError):
        MeasuredInference.load(
            measured.export(), expected=replace(measured.expected_identity, scalars=("contrast",))
        )
    with pytest.raises(CausalError):
        MeasuredInference.load(measured.source_artifact(), expected=measured.expected_identity)
    cancelled = _native.CancellationToken()
    cancelled.cancel()
    with pytest.raises(CausalCancelledError):
        MeasuredInference.load(
            measured.export(), expected=measured.expected_identity, cancel=cancelled
        )


@pytest.mark.parametrize("bayesian", [False, True])
def test_normal_nested_unmeasured_protocol_refusals_are_not_candidate_successes(bayesian):
    for kwargs in ({"max_iterations": 49999}, {"tolerance": 2e-11}):
        with pytest.raises(CausalError):
            produce(bayesian, **kwargs)
    if bayesian:
        with pytest.raises(CausalError):
            produce(True, shape=3)
    else:
        with pytest.raises(CausalError):
            produce(False, nominal_level=0.90)
    cancelled = _native.CancellationToken()
    cancelled.cancel()
    with pytest.raises(CausalCancelledError):
        produce(bayesian, cancel=cancelled)
    with pytest.raises(CausalError):
        produce(bayesian, memory_limit_bytes=0)


@pytest.mark.parametrize("bayesian", [False, True])
def test_normal_nested_measured_artifact_preserves_retained_source_expectations(bayesian, tmp_path):
    measured = produce(bayesian)
    artifact = measured.export()
    loaded = MeasuredInference.load(artifact, expected=measured.expected_identity)
    assert loaded.source_artifact() == measured.source_artifact()
    assert loaded.source_report() == measured.source_report()
    assert loaded.inspect() == measured.inspect()
    path = tmp_path / "nested-original.cbor"
    path.write_bytes(artifact)
    script = """import json,pathlib,sys
from antecedent.inference import MeasuredInference as M,MeasuredInferenceIdentity as I
r=M.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(r.source_report(),sort_keys=True))
"""
    receipt = subprocess.check_output(
        [sys.executable, "-c", script, str(path), json.dumps(measured.expected_identity._wire())],
        text=True,
    )
    assert json.loads(receipt) == measured.source_report()
    for changed, expected in (
        (artifact, replace(measured.expected_identity, candidate_digest="0" * 64)),
        (artifact[:-1], measured.expected_identity),
        (measured.source_artifact(), measured.expected_identity),
    ):
        with pytest.raises(CausalError):
            MeasuredInference.load(changed, expected=expected)
