"""Normal public measured scalar lifecycle; original full posterior remains unmeasured."""

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
from antecedent.transport import advanced as tr

from test_joint_transport_candidate_lifecycle import fixture, oracle


def options(learned, variant):
    varying = "intercept_and_covariates" if variant == "varying" else "intercept"
    sharing = "shared_varying_block" if variant == "shared" else "independent_varying_blocks"
    result = fixture(learned, varying, sharing)
    rng = np.random.default_rng(221971)
    sources = []
    for j in range(1 if variant == "one" else 2):
        old = result["sources"][j]
        x = rng.uniform(-1, 1, 150)
        a = (rng.random(150) < (0.5 if j == 0 else 0.35)).astype(float)
        block = 0 if variant == "shared" else j
        slope = 0.5 + (0.3 * block if varying != "intercept" else 0)
        variance = 1.0 if j == 0 else 2.25
        y = (
            0.4
            + 0.7 * block
            + slope * x
            + a * (2 + 0.4 * x + (0.2 * x * x if learned else 0))
            + np.sqrt(variance) * rng.normal(size=150)
        )
        sources.append(
            replace(
                old,
                data={"a": a, "x": x, "y": y},
                noise_variance=variance,
                identity=tr.JointTransportIdentity(
                    f"measured-source-{j}", tuple(f"source-{j}:{i}" for i in range(150))
                ),
            )
        )

    def prior(width):
        return tr.GaussianTransportPrior(
            tuple([0.0] * width), tuple(map(tuple, np.eye(width) * 1000.0))
        )

    priors = result["priors"]
    result.update(
        sources=sources,
        draws=4096,
        priors=tr.JointTransportPriors(
            prior(len(priors.invariant.mean)), prior(len(priors.varying.mean))
        ),
        target=replace(
            result["target"],
            data={"x": np.array([-0.25, 0.25, 0.25, 0.55])},
            identity=tr.JointTransportIdentity(
                "fixed_target_x", tuple(f"target-{i}" for i in range(4))
            ),
        ),
    )
    return result


def produce(learned, kwargs):
    return (tr.learned_joint_transport if learned else tr.joint_bayesian_transport)(**kwargs)


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("variant", ["one", "independent", "shared", "varying"])
def test_normal_public_measured_joint_scalar_and_original_source_fresh_replay(
    learned, variant, tmp_path
):
    assert hasattr(_native, "joint_transport_measured"), (
        "normal wheel must expose measured producer"
    )
    opts = options(learned, variant)
    measured = produce(learned, opts)
    assert isinstance(measured, MeasuredInference)
    assert measured.calibration == "calibrated"
    assert len(measured.scalars) == 1
    scalar = measured.scalar("target_effect")
    mean, covariance, contrast = oracle(opts, learned)
    assert scalar.point == pytest.approx(contrast @ mean, abs=2e-10)
    assert scalar.level == 0.95
    assert scalar.record_id.endswith(
        ("learned_degree2_gaussian_" if learned else "joint_gaussian_")
        + {
            "one": "one_source_intercept_l95",
            "independent": "two_independent_intercepts_l95",
            "shared": "two_shared_intercepts_l95",
            "varying": "two_varying_covariates_l95",
        }[variant]
    )
    assert len(scalar.calibration_sha) == 40
    assert scalar.basis["scope"]["row_count"] == 150 * len(opts["sources"])
    report = measured.source_report()
    assert report["calibration"] == "unmeasured"
    result = report["result"]
    assert result["posterior_mean"] == pytest.approx(mean, abs=2e-10)
    assert np.asarray(result["posterior_covariance"]).reshape(
        len(mean), len(mean)
    ) == pytest.approx(covariance, abs=2e-10)
    draw_report = result["draws"]
    rows = np.asarray(draw_report["values"]).reshape(
        draw_report["n_draws"], len(draw_report["names"])
    )
    effects = rows[:, draw_report["names"].index("effect.target")]
    assert effects == pytest.approx(rows[:, : len(mean)] @ contrast, abs=2e-10)
    assert scalar.interval == pytest.approx(np.quantile(effects, [0.025, 0.975]), abs=2e-10)
    scope = measured.validated_scope
    assert scope["sampling_assumptions_authenticated_from_rows"] is False
    assert scope["reported_scalar"] == "target_effect"
    artifact = tmp_path / "measured-joint.cbor"
    artifact.write_bytes(measured.export())
    loaded = MeasuredInference.load(artifact.read_bytes(), expected=measured.expected_identity)
    assert loaded.inspect() == measured.inspect()
    assert loaded.source_artifact() == measured.source_artifact()
    script = """
import json,pathlib,sys
from antecedent.inference import MeasuredInference as M,MeasuredInferenceIdentity as I
r=M.load(pathlib.Path(sys.argv[1]).read_bytes(),expected=I._from_wire(json.loads(sys.argv[2])))
print(json.dumps(r.inspect(),sort_keys=True))
"""
    observed = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                script,
                str(artifact),
                json.dumps(measured.expected_identity._wire()),
            ],
            text=True,
        )
    )
    assert observed == measured.inspect()
    detached = measured.inspect()
    detached["scalars"][0]["point"] = 999
    assert scalar.point == pytest.approx(contrast @ mean, abs=2e-10)
    with pytest.raises(CausalTypeError):
        replace(measured, _native=object())
    with pytest.raises(CausalTypeError):
        replace(scalar, _body={"calibration": "calibrated"})
    for expected in [
        replace(measured.expected_identity, data_digest="0" * 64),
        replace(measured.expected_identity, route="unknown"),
        replace(measured.expected_identity, level=0.9),
        replace(measured.expected_identity, scalars=("unmeasured_coefficient",)),
    ]:
        with pytest.raises(CausalError):
            MeasuredInference.load(measured.export(), expected=expected)
    with pytest.raises(CausalError):
        MeasuredInference.load(measured.source_artifact(), expected=measured.expected_identity)
    with pytest.raises(CausalError):
        MeasuredInference.load(
            measured.export(), expected=measured.expected_identity, memory_limit_bytes=0
        )
    with pytest.raises(CausalError):
        MeasuredInference.load(
            measured.export(),
            expected=measured.expected_identity,
            max_bytes=len(measured.export()) - 1,
        )
    cancelled = _native.CancellationToken()
    cancelled.cancel()
    with pytest.raises(CausalCancelledError):
        MeasuredInference.load(
            measured.export(), expected=measured.expected_identity, cancel=cancelled
        )


@pytest.mark.parametrize("learned", [False, True])
def test_nearby_unmeasured_joint_protocols_refuse_without_candidate_fallback(learned):
    opts = options(learned, "independent")
    first = opts["sources"][0]
    priors = opts["priors"]
    cases = [
        {"draws": 4095},
        {"level": 0.90},
        {"target": replace(opts["target"], data={"x": [-0.2, 0.25, 0.25, 0.55]})},
        {"sources": [replace(first, noise_variance=1.1), opts["sources"][1]]},
        {
            "priors": replace(
                priors,
                invariant=replace(
                    priors.invariant, mean=tuple([0.01] * len(priors.invariant.mean))
                ),
            )
        },
        {"priors": replace(priors, invariant=replace(priors.invariant, bank_id="unmeasured-bank"))},
    ]
    if learned:
        cases.append({"basis_degree": 3})
    for changes in cases:
        with pytest.raises(CausalError):
            produce(learned, opts | changes)


@pytest.mark.parametrize("learned", [False, True])
def test_normal_joint_original_proof_bounds_and_cancel_are_required(learned):
    opts = options(learned, "independent")
    calls = []
    fake = SimpleNamespace(payload=lambda: calls.append("payload"))
    for native in (None, fake):
        with pytest.raises(CausalTypeError):
            produce(
                learned, opts | {"identification": replace(opts["identification"], _native=native)}
            )
    assert calls == []
    with pytest.raises(CausalError):
        produce(learned, opts | {"memory_limit_bytes": 0})
    cancelled = _native.CancellationToken()
    cancelled.cancel()
    with pytest.raises(CausalCancelledError):
        produce(learned, opts | {"cancel": cancelled})
    with pytest.raises(CausalTypeError):
        produce(learned, opts | {"level": True})
    with pytest.raises(CausalError):
        produce(learned, opts | {"level": 10**1000})


def test_normal_measured_hooks_do_not_publish_internal_candidate_factories():
    for name in (
        "joint_transport_candidate",
        "consume_joint_transport_candidate",
        "nested_markov_fisher_candidate",
        "nested_markov_posterior_candidate",
    ):
        assert not hasattr(_native, name), name
