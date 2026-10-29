"""Registered external cyclic mechanisms: fit, portable export, independent replay, refusals.

The reference mechanism is ``gain * tanh(x1)``. With the second variable hard-clamped in every
row, the maximum-likelihood gain of the first equation is the least-squares slope of the first
variable on ``tanh(x1)``, which the test computes directly; the equilibrium of the fitted system
is checked against the closed form of the two-equation fixed point (solved by bisection here, not
by the library). The artifact is replayed in a fresh Python process with a fresh registry.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap

import pytest
from antecedent import external_mechanisms as em
from antecedent.cyclic import Equation
from antecedent.errors import CausalValueError

from _refusal import assert_registered_refusal
from _repo_text import REPO_ROOT, load_json

CONTENT = "gain * tanh(x1) :: v1 :: reference mechanism"
CLAMPED = (-2.0, -1.0, 0.5, 1.0, 2.0)


def make_registry(content: str = CONTENT, version: str = "1.0") -> em.MechanismRegistry:
    registry = em.MechanismRegistry()
    em.register_mechanism(
        "lab/saturating_gain",
        version,
        lambda x, p: p[0] * math.tanh(x[0]),
        reads=[1],
        parameters=["gain"],
        content=content,
        input_gradient=lambda x, p: [p[0] * (1.0 - math.tanh(x[0]) ** 2)],
        parameter_gradient=lambda x, p: [math.tanh(x[0])],
        input_lipschitz=[0.9],
        registry=registry,
    )
    return registry


def model(registry: em.MechanismRegistry) -> em.ExternalFeedbackModel:
    implementation = registry.resolve(
        {"implementation_id": "lab/saturating_gain", "version": "1.0"}
    )
    return em.ExternalFeedbackModel(
        equations=(
            em.ExternalEquation("response", implementation, parameters=(0.0,)),
            Equation("feedback", 0.0, (0.2, 0.0), "tanh"),
        ),
        noise_covariance=((1.0, 0.0), (0.0, 1.0)),
    )


def regime_rows() -> list[dict]:
    return [
        {
            "values": [0.7 * math.tanh(x) + noise, x],
            "clamps": [{"variable": 1, "value": x}],
        }
        for x in CLAMPED
        for noise in (-0.1, 0.1)
    ]


def fit_request() -> dict:
    return em.ExternalFeedbackModel.fit_spec(
        [{"kind": "parameter", "equation": 0, "parameter": 0}], regime_rows()
    )


WORLDS = [[], [{"variable": 1, "value": 1.0}]]
NOISE = [[0.1, 0.2], [-0.1, -0.2]]


def export(registry: em.MechanismRegistry) -> em.ExternalArtifact:
    return model(registry).export(
        "external-fit", WORLDS, NOISE, resolver=registry, fit=fit_request(), factual=[0.3, 0.7]
    )


def reference_gain() -> float:
    rows = regime_rows()
    t = [math.tanh(r["values"][1]) for r in rows]
    y = [r["values"][0] for r in rows]
    return sum(a * b for a, b in zip(t, y, strict=True)) / sum(a * a for a in t)


def reference_fixed_point(gain: float, noise: list[float]) -> list[float]:
    """Solve x0 = gain*tanh(x1)+u0, x1 = tanh(0.2*x0)+u1 by bisection on x1."""

    def residual(x1: float) -> float:
        x0 = gain * math.tanh(x1) + noise[0]
        return x1 - (math.tanh(0.2 * x0) + noise[1])

    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    x1 = 0.5 * (lo + hi)
    return [gain * math.tanh(x1) + noise[0], x1]


def test_fit_with_a_registered_mechanism_recovers_the_closed_form_and_solves_the_fixed_point():
    registry = make_registry()
    artifact = export(registry)
    fitted = artifact.report["system"]["equations"][0]["parameters"][0]
    assert fitted == pytest.approx(reference_gain(), abs=1e-6)
    assert artifact.report["fit"]["converged"]
    expected = reference_fixed_point(fitted, NOISE[0])
    assert artifact.report["worlds"][0][0]["values"] == pytest.approx(expected, abs=1e-8)
    assert artifact.report["worlds"][0][0]["globally_unique"] is True
    assert artifact.trust == "verified_external"
    assert artifact.report["claim"] == "verified_external_model_conditional"
    [mechanism] = artifact.report["mechanisms"]
    assert mechanism["trust"] == "verified_external"
    assert mechanism["derivative_basis"] == "analytic"


def test_finite_difference_declaration_reproduces_the_analytic_fit():
    analytic = export(make_registry()).report["system"]["equations"][0]["parameters"][0]
    registry = em.MechanismRegistry()
    em.register_mechanism(
        "lab/saturating_gain",
        "1.0",
        lambda x, p: p[0] * math.tanh(x[0]),
        reads=[1],
        parameters=["gain"],
        content=CONTENT,
        derivative=em.FiniteDifference(1e-6, "central"),
        input_lipschitz=[0.9],
        registry=registry,
    )
    difference = model(registry).execute(WORLDS, NOISE, resolver=registry, fit=fit_request())[
        "system"
    ]["equations"][0]["parameters"][0]
    assert difference == pytest.approx(analytic, abs=1e-6)


def test_independent_reload_in_a_fresh_process_with_a_fresh_registry_reproduces_the_results():
    artifact = export(make_registry())
    script = textwrap.dedent(
        """
        import json, math, sys
        from antecedent import external_mechanisms as em

        registry = em.MechanismRegistry()
        em.register_mechanism(
            "lab/saturating_gain", "1.0",
            lambda x, p: p[0] * math.tanh(x[0]),
            reads=[1], parameters=["gain"], content=sys.argv[3],
            input_gradient=lambda x, p: [p[0] * (1.0 - math.tanh(x[0]) ** 2)],
            parameter_gradient=lambda x, p: [math.tanh(x[0])],
            input_lipschitz=[0.9], registry=registry,
        )
        report = em.consume_artifact(bytes.fromhex(sys.argv[1]), sys.argv[2], resolver=registry)
        print(json.dumps(report))
        """
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, artifact.data.hex(), artifact.identity, CONTENT],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(out.stdout) == artifact.report
    replay = em.consume_artifact(artifact.data, artifact.identity, resolver=make_registry())
    assert replay == artifact.report
    assert replay["trust"] == "verified_external"


def refusal(artifact: em.ExternalArtifact, resolver) -> CausalValueError:
    with pytest.raises(CausalValueError) as caught:
        em.consume_artifact(artifact.data, artifact.identity, resolver=resolver)
    assert_registered_refusal(caught.value)
    return caught.value


def test_replay_without_a_resolver_an_implementation_or_with_a_changed_one_is_refused():
    artifact = export(make_registry())
    assert refusal(artifact, None).reason_code == "mechanism_resolver_missing"
    assert (
        refusal(artifact, em.MechanismRegistry()).reason_code
        == "mechanism_implementation_unavailable"
    )
    assert (
        refusal(artifact, make_registry(content=CONTENT + " edited")).reason_code
        == "mechanism_implementation_changed"
    )
    # A nearby version is a different mechanism, not the recorded one.
    assert (
        refusal(artifact, make_registry(version="1.1")).reason_code
        == "mechanism_implementation_unavailable"
    )
    # A resolver callable that returns an implementation of another content is refused too.
    stale = make_registry(content="other content")
    assert (
        refusal(artifact, lambda spec: stale.resolve(spec)).reason_code
        == "mechanism_implementation_changed"
    )


def test_changed_declarations_are_refused_even_when_the_content_is_identical():
    artifact = export(make_registry())
    registry = em.MechanismRegistry()
    em.register_mechanism(
        "lab/saturating_gain",
        "1.0",
        lambda x, p: p[0] * math.tanh(x[0]),
        reads=[1],
        parameters=["gain"],
        content=CONTENT,
        input_gradient=lambda x, p: [p[0] * (1.0 - math.tanh(x[0]) ** 2)],
        input_lipschitz=[0.95],  # a different Lipschitz declaration
        registry=registry,
    )
    assert refusal(artifact, registry).reason_code == "mechanism_implementation_changed"


def test_artifact_bytes_contain_no_executable_code_and_inspection_needs_no_resolver():
    marker = "UNIQUE_IMPLEMENTATION_BODY_MARKER"
    registry = make_registry(content=f"{CONTENT} {marker}")
    artifact = model(registry).export(
        "external-fit", WORLDS, NOISE, resolver=registry, fit=fit_request()
    )
    assert marker.encode() not in artifact.data
    inspected = em.inspect_artifact(artifact.data)
    assert inspected["trust"] == "verified_external"
    [recorded] = inspected["mechanisms"]
    spec = recorded["spec"]
    assert spec["content_digest"] == registry.resolve(spec).spec["content_digest"]
    assert set(spec) <= {
        "implementation_id",
        "version",
        "content_digest",
        "dependencies",
        "reads",
        "parameter_names",
        "derivative",
        "input_lipschitz",
    }
    text = json.dumps(inspected)
    for forbidden in ("lambda", "def ", "bytecode", "pickle", marker):
        assert forbidden not in text


def test_a_resealed_request_is_refused_against_the_retained_identity():
    artifact = export(make_registry())
    with pytest.raises(CausalValueError, match="identity/replay mismatch"):
        em.consume_artifact(artifact.data, "0" * 64, resolver=make_registry())


def test_an_analytic_declaration_requires_a_gradient_and_malformed_specs_are_refused():
    with pytest.raises(CausalValueError, match="analytic derivative declaration needs input_gradient"):
        em.register_mechanism(
            "lab/x",
            "1",
            lambda x, p: 0.0,
            reads=[0],
            content="x",
            derivative="analytic",
            registry=em.MechanismRegistry(),
        )
    with pytest.raises(CausalValueError) as duplicate_read:
        em.register_mechanism(
            "lab/x",
            "1",
            lambda x, p: 0.0,
            reads=[0, 0],
            content="x",
            registry=em.MechanismRegistry(),
        )
    assert duplicate_read.value.reason_code == "mechanism_declaration_invalid"


def test_registering_the_same_id_and_version_twice_is_refused():
    registry = make_registry()
    with pytest.raises(CausalValueError, match="version '1.0' is already registered"):
        em.register_mechanism(
            "lab/saturating_gain",
            "1.0",
            lambda x, p: 0.0,
            reads=[1],
            content="again",
            registry=registry,
        )


def test_external_ode_matches_the_closed_form_and_stays_verified_external():
    registry = em.MechanismRegistry()
    implementation = em.register_mechanism(
        "lab/linear_drive",
        "1",
        lambda x, p: p[0] * x[0],
        reads=[1],
        parameters=["rate"],
        content="rate * x1",
        input_gradient=lambda x, p: [p[0]],
        input_lipschitz=[0.5],
        registry=registry,
    )
    ode = em.ExternalOdeModel(
        equations=(
            em.ExternalEquation("a", implementation, parameters=(0.5,)),
            Equation("b", 0.0, (0.0, 0.0), "identity"),
        ),
        decay=(1.0, 0.0),
    )
    out = ode.trajectory([3.0, 2.0], [0.0, 1.0, 2.0], resolver=registry)
    for state, t in zip(out["trajectory"]["states"], [0.0, 1.0, 2.0], strict=True):
        assert state[0] == pytest.approx(1.0 + 2.0 * math.exp(-t), abs=1e-6)
    assert out["trust"] == "verified_external"
    assert out["drift_lipschitz"] == pytest.approx(1.5)
    with pytest.raises(CausalValueError) as caught:
        ode.trajectory([3.0, 2.0], [0.0, 1.0], resolver=None)
    assert caught.value.reason_code == "mechanism_resolver_missing"


def test_a_process_local_callback_model_still_refuses_export():
    from antecedent.cyclic import CyclicRefusal, LocalCyclicModel, LocalMechanism
    from antecedent.outcomes import require_success

    local = LocalCyclicModel((LocalMechanism("a", (0,), lambda x: (0.5 * x[0] + 1.0, [0.5])),))
    answer = require_success(local.execute([0.0], []))
    assert answer["result"]["values"][0] == pytest.approx(2.0)
    assert not answer["inspection"]["portable"]
    with pytest.raises(CyclicRefusal) as refused:
        bytes(local.export())
    assert refused.value.reason_code == "policy_callback_not_portable"


def test_fit_and_fixed_points_match_the_committed_independent_reference():
    expected = load_json(REPO_ROOT / "conformance/gcm/external_mechanism_feedback/expected.json")
    report = export(make_registry()).report
    assert report["system"]["equations"][0]["parameters"][0] == pytest.approx(
        expected["fitted_gain"], abs=1e-6
    )
    for row, truth in zip(fixed_point_rows(report), expected["fixed_points"], strict=True):
        # The fixed point moves with the fitted gain, which is compared at 1e-6 above.
        assert row == pytest.approx(truth, abs=1e-6)


def fixed_point_rows(report):  # noqa: N802
    return [result["values"] for result in report["worlds"][0]]


def test_a_resolver_that_raises_is_a_resolver_failure_not_an_absent_implementation():
    artifact = export(make_registry())

    def broken(spec):
        raise RuntimeError("registry backend down")

    error = refusal(artifact, broken)
    assert error.reason_code == "mechanism_resolver_failed"
    assert "registry backend down" in str(error)
    assert refusal(artifact, lambda spec: object()).reason_code == "mechanism_resolver_failed"
    assert refusal(artifact, lambda spec: None).reason_code == (
        "mechanism_implementation_unavailable"
    )


def test_the_generic_artifact_loader_refuses_without_a_resolver():
    from antecedent import load

    artifact = export(make_registry())
    with pytest.raises(CausalValueError) as caught:
        load(artifact.data, expected=artifact.identity)
    assert caught.value.reason_code == "mechanism_resolver_missing"
