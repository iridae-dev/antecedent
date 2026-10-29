"""Bayesian randomized-dose grid posterior through the Python facade.

The oracle is the conjugate ridge posterior of the rescaled polynomial design, computed here with
numpy from the normal equations (not by the library). The row is unlicensed and its calibration is
unmeasured; the joint covariance across grid points is retained and exported.
"""

from __future__ import annotations

import subprocess
import sys

import antecedent
import numpy as np
import pytest
from antecedent.dose_grid import (
    DoseGridRefusal,
    consume_dose_grid_posterior_artifact,
    dose_grid_posterior,
)
from antecedent.errors import CausalSerializationError, CausalUnsupportedError

DOSE = np.array([0.1, 0.5, 0.9, 1.4, 1.8, 2.2, 2.7, 3.1, 3.5, 3.9])
OUTCOME = np.array([1.2, 1.9, 2.1, 3.0, 3.2, 3.1, 3.0, 2.6, 2.2, 1.5])
SCALES = (3.0, 2.0, 1.0)
LOW, HIGH = 0.0, 4.0
SIGMA2 = 0.25


def _rescale(d):
    return (2.0 * np.asarray(d, dtype=float) - LOW - HIGH) / (HIGH - LOW)


def _posterior():
    design = np.vander(_rescale(DOSE), 3, increasing=True)
    precision = design.T @ design + np.diag(1.0 / np.asarray(SCALES) ** 2)
    mean = np.linalg.solve(precision, design.T @ OUTCOME)
    return mean, SIGMA2 * np.linalg.inv(precision)


def _run(**kwargs):
    arguments = {
        "support": (LOW, HIGH),
        "coefficient_scales": SCALES,
        "residual_variance": SIGMA2,
        "functional": "level",
        "grid": (0.5, 2.0, 3.5),
        "draws": 40_000,
        "seed": 21,
    }
    arguments.update(kwargs)
    controls = {name: arguments.pop(name) for name in ("threads", "cancel") if name in arguments}
    if "max_bytes" in arguments:
        controls["limits"] = antecedent.ResourceLimits(memory_bytes=arguments.pop("max_bytes"))
    if controls:
        arguments["execution"] = antecedent.Execution(**controls)
    return dose_grid_posterior(DOSE, OUTCOME, **arguments)


def test_level_posterior_matches_the_conjugate_reference_and_keeps_the_covariance():
    mean, covariance = _posterior()
    basis = np.vander(_rescale([0.5, 2.0, 3.5]), 3, increasing=True)
    result = _run()
    row = result.posterior
    assert row.estimator == "bayesian.dose_grid"
    assert (row.uncertainty, row.calibration, row.licensed) == ("posterior", "unmeasured", False)
    assert row.exact_mean == pytest.approx(basis @ mean, abs=1e-9)
    exact = row.exact_covariance_matrix()
    assert exact == pytest.approx(basis @ covariance @ basis.T, abs=1e-9)
    assert row.mean == pytest.approx(row.exact_mean, abs=0.01)
    joint = row.covariance_matrix()
    scale = np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
    assert np.all(np.abs(joint - exact) < 0.05 * scale)
    assert joint[0, 1] > 0.01 and joint[1, 2] > 0.01  # independent draws would leave ~0
    assert row.vectors["points"] == (0.5, 2.0, 3.5)
    assert bytes(result.export())


def test_derivative_and_contrast_use_the_same_joint_draws():
    mean, covariance = _posterior()
    slope = np.array(
        [[0.0, 2.0 / (HIGH - LOW), 2.0 * _rescale(d) * 2.0 / (HIGH - LOW)] for d in (0.5, 2.0, 3.5)]
    )
    derivative = _run(functional="derivative").posterior
    assert derivative.exact_mean == pytest.approx(slope @ mean, abs=1e-9)
    contrast = _run(functional="contrast", grid=None, contrast=(1.0, 3.0)).posterior
    row = np.vander(_rescale([3.0, 1.0]), 3, increasing=True)
    direction = row[0] - row[1]
    assert contrast.exact_mean == pytest.approx((direction @ mean,), abs=1e-9)
    assert contrast.exact_covariance == pytest.approx(
        (direction @ covariance @ direction,), abs=1e-9
    )
    # The paired variance, not the larger one independent draws at the two doses would give.
    assert contrast.covariance[0] == pytest.approx(contrast.exact_covariance[0], rel=0.1)


def test_artifact_replays_in_a_fresh_process_and_refuses_corruption(tmp_path):
    result = _run(draws=2_000)
    artifact = bytes(result.export())
    consumed = consume_dose_grid_posterior_artifact(artifact)
    assert consumed.posterior == result.posterior
    assert consumed.premises_digest == result.premises_digest
    path = tmp_path / "posterior.antecedent"
    path.write_bytes(artifact)
    script = (
        "import sys\n"
        "from antecedent.dose_grid import consume_dose_grid_posterior_artifact as c\n"
        "print(repr(c(open(sys.argv[1], 'rb').read()).posterior.exact_mean))\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path)], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == repr(result.posterior.exact_mean)
    with pytest.raises(CausalSerializationError):
        consume_dose_grid_posterior_artifact(artifact[: len(artifact) // 2])
    with pytest.raises(DoseGridRefusal) as caught:
        consume_dose_grid_posterior_artifact(artifact, max_rows=5)
    assert caught.value.detail == "bayesian_row.consumer_limit_exceeded"


def test_refusals_are_typed():
    with pytest.raises(DoseGridRefusal) as caught:
        _run(design="observational")
    assert isinstance(caught.value, CausalUnsupportedError)
    assert caught.value.detail == "dose_grid.graph_not_certified"
    with pytest.raises(DoseGridRefusal) as caught:
        _run(grid=(0.01,))
    assert caught.value.detail == "dose_grid.unsupported_dose"
    with pytest.raises(Exception, match="residual"):
        _run(residual_variance=None)


def test_caller_policy_bounds_evaluation_and_independent_consumption():
    from antecedent import _native
    from antecedent.errors import CausalCancelledError, CausalResourceError

    serial = _run(draws=200, threads=1)
    assert _run(draws=200, threads=2).posterior == serial.posterior
    artifact = bytes(serial.export())
    assert (
        consume_dose_grid_posterior_artifact(
            artifact, execution=antecedent.Execution(threads=2)
        ).posterior
        == serial.posterior
    )

    with pytest.raises(CausalResourceError):
        _run(draws=200, max_bytes=1)
    with pytest.raises(CausalResourceError):
        consume_dose_grid_posterior_artifact(
            artifact,
            execution=antecedent.Execution(limits=antecedent.ResourceLimits(memory_bytes=1)),
        )

    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        _run(draws=200, cancel=token)
    with pytest.raises(CausalCancelledError):
        consume_dose_grid_posterior_artifact(artifact, execution=antecedent.Execution(cancel=token))


def test_native_execution_facts_survive_success_refusal_cancellation_and_replay_failure():
    from antecedent import _native
    from antecedent.errors import CausalCancelledError, CausalResourceError

    result = _run(draws=200)
    facts = result.execution_facts
    assert facts.is_native
    assert facts.status() == "completed"
    assert facts.peak_admitted_bytes > 0
    assert facts.observed_model_fits == 1
    replay = consume_dose_grid_posterior_artifact(result.artifact)
    assert replay.execution_facts.is_native
    assert replay.execution_facts.status() == "completed"

    with pytest.raises(CausalResourceError) as refused:
        _run(draws=200, max_bytes=1)
    assert refused.value.execution_facts.is_native
    assert refused.value.execution_facts.status() == "failed"

    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError) as cancelled:
        _run(draws=200, cancel=token)
    assert cancelled.value.execution_facts.is_native
    assert cancelled.value.execution_facts.status() == "cancelled"

    with pytest.raises(CausalSerializationError) as corrupt:
        consume_dose_grid_posterior_artifact(b"corrupt")
    assert corrupt.value.execution_facts.is_native
    assert corrupt.value.execution_facts.status() == "failed"
