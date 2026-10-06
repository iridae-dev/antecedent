"""Typed Python consumption of the bounded 2.3 joint artifact."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from antecedent.artifacts import DistributionIdentity, JointDistributionArtifact, ScientificQuantity


def identity() -> DistributionIdentity:
    quantities = tuple(
        ScientificQuantity(
            variable_id=f"schema:{name}",
            variable_name=name,
            role="outcome",
            units="dimensionless",
            population_id="target",
            regime_id="do(a=1)",
            horizon=0,
            functional_id="outcome",
        )
        for name in ("x", "y")
    )
    return DistributionIdentity(
        semantic="interventional_predictive",
        quantities=quantities,
        alignment="joint",
        source_id="study",
        provider_id="provider",
        rng_id="deterministic_exact",
        snapshot_id="snapshot",
        causal_contract_id="checked-contract",
    )


def test_typed_numpy_copy_and_round_trip() -> None:
    expected = identity()
    artifact = JointDistributionArtifact(expected, np.array([[0.0, 0.0], [1.0, 2.0]]))
    assert artifact.semantic == "interventional_predictive"
    assert artifact.axes == ("draw", "quantity")
    assert artifact.shape == (2, 2)
    assert artifact.n_draws == 2
    assert artifact.quantities == expected.quantities
    assert artifact.covariance(0, 1) == 0.5
    assert artifact.joint_product_expectation(0, 1) == 1.0
    values = np.asarray(artifact)
    assert values.shape == (2, 2)
    values[0, 0] = 99.0
    assert np.asarray(artifact)[0, 0] == 0.0
    with pytest.raises(ValueError, match="bounded copy"):
        np.asarray(artifact, copy=False)
    reloaded = JointDistributionArtifact.load(
        artifact.export("joint-python"), expected_identity=expected
    )
    assert reloaded.identity == expected
    np.testing.assert_array_equal(np.asarray(reloaded), [[0.0, 0.0], [1.0, 2.0]])
    assert np.cov(np.asarray(reloaded).T, bias=True)[0, 1] == 0.5


def test_load_refuses_changed_identity_and_bad_shape() -> None:
    expected = identity()
    artifact = JointDistributionArtifact(expected, np.array([[0.0, 0.0], [1.0, 2.0]]))
    changed = DistributionIdentity(
        semantic=expected.semantic,
        quantities=expected.quantities,
        alignment=expected.alignment,
        source_id=expected.source_id,
        provider_id=expected.provider_id,
        rng_id=expected.rng_id,
        snapshot_id="changed",
        causal_contract_id=expected.causal_contract_id,
    )
    with pytest.raises(Exception, match="identity"):
        JointDistributionArtifact.load(artifact.export("joint-python"), expected_identity=changed)
    with pytest.raises(ValueError, match="two-dimensional"):
        JointDistributionArtifact(expected, np.array([0.0, 1.0]))
    with pytest.raises(Exception, match="shape"):
        JointDistributionArtifact(expected, np.array([[0.0], [1.0]]))
    oversized_view = np.broadcast_to(np.array([1.0]), (100_001, 2))
    with pytest.raises(ValueError, match="bounds"):
        JointDistributionArtifact(expected, oversized_view)


def test_weights_and_coordinate_mask_survive_python_export() -> None:
    expected = identity()
    artifact = JointDistributionArtifact(
        expected,
        np.array([[0.0, 0.0], [1.0, 2.0]]),
        weights=(1.0, 3.0),
        supported=(True, False),
    )
    loaded = JointDistributionArtifact.load(
        artifact.export("weighted-python"), expected_identity=expected
    )
    assert loaded.weights == (1.0, 3.0)
    assert loaded.supported == (True, False)
    assert loaded.trust == "unverified"
    assert loaded.calibration == "unmeasured"


def test_independent_marginals_refuse_python_joint_operations() -> None:
    expected = identity()
    marginal = DistributionIdentity(
        semantic=expected.semantic,
        quantities=expected.quantities,
        alignment="independent_marginals",
        source_id=expected.source_id,
        provider_id=expected.provider_id,
        rng_id=expected.rng_id,
        snapshot_id=expected.snapshot_id,
        causal_contract_id=expected.causal_contract_id,
    )
    artifact = JointDistributionArtifact(marginal, np.array([[0.0, 0.0], [1.0, 2.0]]))
    with pytest.raises(Exception, match="joint_law_required"):
        artifact.covariance(0, 1)
    with pytest.raises(Exception, match="joint_law_required"):
        artifact.joint_product_expectation(0, 1)


def test_fresh_python_process_consumes_with_independent_identity(tmp_path) -> None:
    data = JointDistributionArtifact(
        identity(), np.array([[0.0, 0.0], [1.0, 2.0]])
    ).export("joint-python")
    path = tmp_path / "joint.bin"
    path.write_bytes(data)
    script = """
import numpy as np
from antecedent.artifacts import DistributionIdentity, JointDistributionArtifact, ScientificQuantity
from pathlib import Path
quantities = tuple(ScientificQuantity(variable_id=f'schema:{name}', variable_name=name,
    role='outcome', units='dimensionless', population_id='target', regime_id='do(a=1)',
    horizon=0, functional_id='outcome') for name in ('x', 'y'))
expected = DistributionIdentity(semantic='interventional_predictive', quantities=quantities,
    alignment='joint', source_id='study', provider_id='provider',
    rng_id='deterministic_exact', snapshot_id='snapshot', causal_contract_id='checked-contract')
artifact = JointDistributionArtifact.load(Path(__import__('sys').argv[1]).read_bytes(), expected_identity=expected)
values = np.asarray(artifact)
assert artifact.shape == (2, 2)
assert np.mean(values[:, 0] * values[:, 1]) == 1.0
assert np.cov(values.T, bias=True)[0, 1] == 0.5
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
