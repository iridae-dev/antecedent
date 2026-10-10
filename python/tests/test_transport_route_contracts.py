"""Each measured factory replays after every producer input and handle is discarded."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest
from antecedent.errors import CausalError
from antecedent.inference import MeasuredInference
from antecedent.transport import advanced as tr

from test_measured_joint_transport import options as joint_options
from test_measured_recovery import options as recovery_options
from test_nested_bayesian_activation import graph
from test_temporal_interval_activation import panel


def independent_replay(payload, identity, original, tmp_path):
    loaded = MeasuredInference.load(payload, expected=identity)
    assert loaded.inspect() == original
    path = tmp_path / "detached-measured.cbor"
    path.write_bytes(payload)
    script = """
import json,pathlib,sys
from antecedent.inference import MeasuredInference as M,MeasuredInferenceIdentity as I
result=M.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(result.inspect(),sort_keys=True))
"""
    observed = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", script, str(path), json.dumps(identity._wire())], text=True
        )
    )
    assert observed == original
    assert all(scalar.calibration == "calibrated" for scalar in loaded.scalars)
    source = loaded.source_report()
    source = source["result"] if loaded.route == "temporal_interval" else source
    assert source["calibration"] == "unmeasured"
    with pytest.raises(CausalError):
        MeasuredInference.load(payload, expected=replace(identity, data_digest="0" * 64))
    with pytest.raises(CausalError):
        MeasuredInference.load(loaded.source_artifact(), expected=identity)
    return loaded


def test_joint_bayesian_factory_replays_without_producer_inputs(tmp_path):
    source_inputs = joint_options(False, "one")
    measured = tr.joint_bayesian_transport(**source_inputs)
    payload, identity, original = measured.export(), measured.expected_identity, measured.inspect()
    del source_inputs
    del measured
    loaded = independent_replay(payload, identity, original, tmp_path)
    assert loaded.route == "joint_bayesian"


def test_learned_joint_factory_replays_without_producer_inputs(tmp_path):
    source_inputs = joint_options(True, "one")
    measured = tr.learned_joint_transport(**source_inputs)
    payload, identity, original = measured.export(), measured.expected_identity, measured.inspect()
    del source_inputs
    del measured
    loaded = independent_replay(payload, identity, original, tmp_path)
    assert loaded.route == "learned_joint"


def test_nested_bayesian_factory_replays_without_producer_inputs(tmp_path):
    source_inputs = {"graph": graph(), "regimes": [{"counts": [250] * 16}], "seed": 817}
    measured = tr.binary_nested_markov(**source_inputs)
    payload, identity, original = measured.export(), measured.expected_identity, measured.inspect()
    del source_inputs
    del measured
    loaded = independent_replay(payload, identity, original, tmp_path)
    assert loaded.route == "nested_bayesian"


def test_nested_fisher_factory_replays_without_producer_inputs(tmp_path):
    source_inputs = {"graph": graph(), "regimes": [{"counts": [250] * 16}]}
    measured = tr.binary_nested_markov_fisher_interval(**source_inputs)
    payload, identity, original = measured.export(), measured.expected_identity, measured.inspect()
    del source_inputs
    del measured
    loaded = independent_replay(payload, identity, original, tmp_path)
    assert loaded.route == "nested_fisher"


def test_direct_temporal_factory_replays_without_producer_inputs(tmp_path):
    source_inputs = {
        "panel": panel(),
        "sequence": (0, 0),
        "target_law": tr.InitialStateLaw("fixed_target", {0: 0.3, 1: 0.7}),
        "seed": 901,
    }
    measured = tr.temporal_dependent_interval(**source_inputs)
    payload, identity, original = measured.export(), measured.expected_identity, measured.inspect()
    del source_inputs
    del measured
    loaded = independent_replay(payload, identity, original, tmp_path)
    assert loaded.route == "temporal_interval"


def test_sampled_recovery_factory_replays_without_producer_inputs(tmp_path):
    source_inputs = recovery_options()
    measured = tr.sampled_observation_recovery(**source_inputs)
    payload, identity, original = measured.export(), measured.expected_identity, measured.inspect()
    del source_inputs
    del measured
    loaded = independent_replay(payload, identity, original, tmp_path)
    assert loaded.route == "sampled_recovery"
