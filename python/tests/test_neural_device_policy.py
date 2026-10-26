"""Public cpu/gpu/auto policy of the neural learner: validation, wire form, provenance, refusal.

``ANTECEDENT_EXPECT_ACCELERATOR=1`` marks a run against the accelerator distribution on a
host with a usable adapter; the default run is the driver-free CPU wheel.
"""

import os

import antecedent as ac
import numpy as np
import pytest
from antecedent.estimators import DML
from antecedent.learners import Logistic, NeuralNet
from antecedent.outcomes import require_success

ACCELERATOR = os.environ.get("ANTECEDENT_EXPECT_ACCELERATOR") == "1"


def data_fixture():
    rng = np.random.default_rng(19)
    z = rng.normal(size=160)
    t = rng.binomial(1, 0.5, size=160).astype(float)
    y = t * (2 + z) + z + rng.normal(size=160) * 0.1
    return {"z": z, "t": t, "y": y}


def analyze(device):
    return require_success(
        ac.analyze(
            data_fixture(),
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ac.AverageEffect(treatment="t", outcome="y"),
            estimator=DML(
                outcome=NeuralNet(hidden=4, epochs=3, learning_rate=0.01, device=device),
                treatment=Logistic(),
                folds=2,
            ),
            bootstrap=0,
            refute=False,
            seed=3,
        )
    )


def neural_implementations(result):
    return {impl for spec, impl, _ in result.estimate.learner_provenance if spec == "neural_net"}


@pytest.mark.parametrize("device", ["cuda", "", "GPU", None, 1])
def test_unknown_device_is_refused_at_construction(device):
    with pytest.raises(ValueError, match="device must be one of"):
        NeuralNet(device=device)


def test_cpu_is_the_default_and_keeps_the_wire_form():
    assert NeuralNet().device == "cpu"
    assert "device" not in NeuralNet()._wire()
    assert NeuralNet(device="auto")._wire()["device"] == "auto"
    assert NeuralNet(device="gpu")._wire()["device"] == "gpu"


def test_default_fit_keeps_the_plain_provenance_label():
    assert neural_implementations(analyze("cpu")) == {"burn"}


def test_auto_runs_a_small_fit_on_the_cpu_and_records_the_reason():
    result = analyze("auto")
    (label,) = neural_implementations(result)
    assert "device=host-cpu" in label and "requested=auto" in label
    assert "fallback=" in label
    assert "wgpu" not in label
    assert result.estimate.ate == analyze("cpu").estimate.ate


@pytest.mark.skipif(ACCELERATOR, reason="this build links the device runtime")
def test_gpu_is_refused_without_the_device_runtime():
    with pytest.raises(Exception, match="runtime_not_compiled"):
        analyze("gpu")


@pytest.mark.skipif(not ACCELERATOR, reason="needs the accelerator distribution and an adapter")
def test_gpu_fit_names_the_device_and_agrees_with_the_cpu_estimate():
    gpu = analyze("gpu")
    cpu = analyze("cpu")
    (label,) = neural_implementations(gpu)
    assert "burn-wgpu" in label and "requested=gpu" in label and "fallback" not in label
    assert abs(gpu.estimate.ate - cpu.estimate.ate) < 0.5 * cpu.estimate.se_analytic
