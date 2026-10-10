"""Normal BCa scalar authorization with independent whole-row recovery arithmetic."""

import json
import subprocess
import sys
from dataclasses import replace
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalCancelledError, CausalError, CausalTypeError
from antecedent.inference import MeasuredInference
from antecedent.transport import advanced as tr

from test_observation_recovery import catalog, graph, query
from test_sampled_recovery_activation import oracle, resample, rows


def stage():
    return tr.identify_observation_recovery(
        graph=graph(),
        query=query(),
        catalog=catalog(),
        effect_outcomes=["y"],
        effect_treatments=["t"],
    )


def options():
    return {
        "stage": stage(),
        "query": query(),
        "rows": rows(),
        "snapshot": "snap-1",
        "seed": 7,
    }


def test_normal_recovered_effect_bca_original_source_and_fresh_measured_replay(tmp_path):
    opts = options()
    measured = tr.sampled_observation_recovery(**opts)
    assert isinstance(measured, MeasuredInference)
    assert measured.route == "sampled_recovery" and len(measured.scalars) == 1
    scalar = measured.scalar("recovered_effect")
    assert scalar.calibration == "calibrated" and scalar.level == 0.95
    assert scalar.record_id.endswith("binary_missingness_whole_row_recovery_bca_l95")
    assert scalar.basis["key"]["interval_method"] == "bootstrap_bca"
    assert scalar.basis["scope"]["row_count"] == 2000
    assert len(scalar.calibration_sha) == 40
    source = measured.source_report()
    assert source["calibration"] == "unmeasured"
    receipt = source["receipt"]
    assert receipt["config"]["replicates"] == 2000
    assert receipt["config"]["max_failed_fraction"] == 0
    assert len(receipt["replicates"]) == 2000
    assert all(
        row["failure"] is None and row["effect"] is not None for row in receipt["replicates"]
    )
    sample = opts["rows"]
    _, point = oracle(sample)
    assert scalar.point == pytest.approx(point, abs=1e-12)
    jack = np.array([oracle(sample[:i] + sample[i + 1 :])[1] for i in range(len(sample))])
    differences = jack.mean() - jack
    acceleration = np.sum(differences**3) / (6 * np.sum(differences**2) ** 1.5)
    effects = np.array([oracle(resample(sample, index, 7))[1] for index in range(2000)])
    normal = NormalDist()
    z0 = normal.inv_cdf(
        (np.count_nonzero(effects < point) + 0.5 * np.count_nonzero(effects == point)) / 2000
    )
    adjusted = []
    for tail in (0.025, 0.975):
        z = z0 + normal.inv_cdf(tail)
        adjusted.append(normal.cdf(z0 + z / (1 - acceleration * z)))
    bca = receipt["bca"]
    assert bca["acceleration"] == pytest.approx(acceleration, abs=1e-10)
    assert bca["bias_correction"] == pytest.approx(z0, abs=1e-7)
    assert scalar.interval == pytest.approx(np.quantile(effects, adjusted), abs=1e-7)
    path = tmp_path / "measured-recovery.cbor"
    path.write_bytes(measured.export())
    loaded = MeasuredInference.load(path.read_bytes(), expected=measured.expected_identity)
    assert loaded.inspect() == measured.inspect()
    assert loaded.source_artifact() == measured.source_artifact()
    script = """
import json,pathlib,sys
from antecedent.inference import MeasuredInference as M,MeasuredInferenceIdentity as I
r=M.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(r.inspect(),sort_keys=True))
"""
    actual = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                script,
                str(path),
                json.dumps(measured.expected_identity._wire()),
            ],
            text=True,
        )
    )
    assert actual == measured.inspect()
    for artifact, identity in [
        (measured.source_artifact(), measured.expected_identity),
        (measured.export(), replace(measured.expected_identity, data_digest="0" * 64)),
    ]:
        with pytest.raises(CausalError):
            MeasuredInference.load(artifact, expected=identity)


def test_normal_recovery_legacy_scope_fake_authority_and_resource_refusals():
    opts = options()
    for change in (
        {"replicates": 500},
        {"interval_method": "bootstrap_percentile"},
        {"memory_limit_bytes": 0},
        {"rows": opts["rows"][:500]},
    ):
        with pytest.raises(CausalError):
            tr.sampled_observation_recovery(**(opts | change))
    provider = catalog()
    weighted = replace(
        provider,
        bindings=tuple(
            replace(binding, weights_snapshot=binding.snapshot_identity)
            for binding in provider.bindings
        ),
    )
    weighted_stage = tr.identify_observation_recovery(
        graph=graph(),
        query=query(),
        catalog=weighted,
        effect_outcomes=["y"],
        effect_treatments=["t"],
    )
    with pytest.raises(CausalError, match="weighted"):
        tr.sampled_observation_recovery(**(opts | {"stage": weighted_stage}))
    for design in ("unknown", "clustered"):
        provider = catalog()
        changed = replace(
            provider,
            bindings=tuple(replace(binding, sampling=design) for binding in provider.bindings),
        )
        declared_stage = tr.identify_observation_recovery(
            graph=graph(),
            query=query(),
            catalog=changed,
            effect_outcomes=["y"],
            effect_treatments=["t"],
        )
        with pytest.raises(CausalError):
            tr.sampled_observation_recovery(**(opts | {"stage": declared_stage}))
    calls = []
    fake = SimpleNamespace(outcome="recovered", sampled_measured=lambda *a, **k: calls.append(1))
    with pytest.raises(CausalTypeError):
        tr.sampled_observation_recovery(**(opts | {"stage": fake}))
    assert calls == []
    cancelled = _native.CancellationToken()
    cancelled.cancel()
    with pytest.raises(CausalCancelledError):
        tr.sampled_observation_recovery(**(opts | {"cancel": cancelled}))


def test_normal_sampled_recovery_measured_artifact_requires_original_expected_identity(tmp_path):
    measured = tr.sampled_observation_recovery(**options())
    loaded = MeasuredInference.load(measured.export(), expected=measured.expected_identity)
    assert loaded.source_report() == measured.source_report()
    assert loaded.source_artifact() == measured.source_artifact()
    path = tmp_path / "recovery-original.cbor"
    path.write_bytes(measured.export())
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
    for artifact, identity in (
        (measured.export(), replace(measured.expected_identity, premises_digest="0" * 64)),
        (measured.export()[:-1], measured.expected_identity),
        (measured.source_artifact(), measured.expected_identity),
    ):
        with pytest.raises(CausalError):
            MeasuredInference.load(artifact, expected=identity)
