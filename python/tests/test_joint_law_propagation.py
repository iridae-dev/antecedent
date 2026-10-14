"""Composition-aware uncertainty through the Python surface.

Truth is closed form or enumerated and computed in this file in pure Python / numpy: the product
moment of a correlated Gaussian pair (``rho s1 s2 + mu1 mu2``) inside a band predeclared from the
closed-form ``Var(XY)``, ``Var(X - Y) = sx^2 + sy^2 - 2 cov``, a known-weight mixture, and the
Makarov and comonotone/countermonotone endpoints of Bernoulli and uniform marginals. Every
simulation runs at a fixed seed.
"""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from antecedent import joint_law_propagation as jl
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

R = jl.JointLawPropagationRefusal


def _artifact(
    snapshot: str, x: np.ndarray, y: np.ndarray, alignment: str = "joint"
) -> JointDistributionArtifact:
    quantities = tuple(
        ScientificQuantity.outcome(name, units="units", population="target", regime="do(a=1)")
        for name in ("x", "y")
    )
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=quantities,
        alignment=alignment,  # type: ignore[arg-type]
        source_id=f"study-{snapshot}",
        provider_id="engine",
        rng_id=f"seed-{snapshot}",
        snapshot_id=snapshot,
        causal_contract_id="checked-contract",
    )
    draws = np.column_stack([x, y]).astype(np.float64)
    return JointDistributionArtifact(identity, draws)


def _gaussian_pair(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    mu_x, mu_y, s_x, s_y, rho = 1.0, 2.0, 1.5, 0.5, 0.6
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    x = mu_x + s_x * z1
    y = mu_y + s_y * (rho * z1 + math.sqrt(1.0 - rho * rho) * z2)
    truth = rho * s_x * s_y + mu_x * mu_y
    var = (
        mu_x**2 * s_y**2
        + mu_y**2 * s_x**2
        + 2 * mu_x * mu_y * rho * s_x * s_y
        + s_x**2 * s_y**2 * (1 + rho**2)
    )
    return x, y, truth, 5.0 * math.sqrt(var / n)


def _draws(art: JointDistributionArtifact) -> tuple[jl.PropagationDraws, jl.PropagationDraws]:
    return (
        jl.PropagationDraws.from_artifact_column("x", art, 0),
        jl.PropagationDraws.from_artifact_column("y", art, 1),
    )


# ------------------------------------------------------------ aligned joint law


def test_aligned_product_matches_the_joint_moment_within_the_predeclared_band() -> None:
    x, y, truth, band = _gaussian_pair(100_000, 20_240_917)
    art = _artifact("snap-joint", x, y)
    dx, dy = _draws(art)
    claim = jl.aligned_functional("product", dx, dy, "product")
    assert abs(claim.value("mean") - truth) < band
    assert claim.route == "aligned_joint_law"
    assert claim.label == "point_only"
    assert not claim.is_envelope
    assert claim.n_rows == 100_000
    assert claim.alignment_id == dx.alignment_id == dy.alignment_id
    # The same columns paired against an unrelated shift miss the joint truth.
    shifted = np.roll(y, 7_919)
    miss = float(np.mean(x * shifted))
    assert abs(miss - truth) > 5.0 * band


def test_enumerated_joint_law_nonlinear_functionals() -> None:
    rows = [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)]
    x = np.array([a for a, _, c in rows for _ in range(c)])
    y = np.array([b for _, b, c in rows for _ in range(c)])
    dx, dy = _draws(_artifact("snap-enum", x, y))
    assert jl.aligned_functional("product", dx, dy, "product").value("mean") == pytest.approx(0.4)
    assert jl.aligned_functional("nonlinear_functional", dx, dy, "minimum").value(
        "mean"
    ) == pytest.approx(0.2)
    # Covariance-aware contrast from the aligned rows equals the closed form.
    var_x, var_y = float(np.var(x)), float(np.var(y))
    cov = float(np.mean(x * y) - np.mean(x) * np.mean(y))
    claim = jl.aligned_linear("contrast", (1.0, -1.0), dx, dy)
    assert claim.value("variance") == pytest.approx(var_x + var_y - 2.0 * cov, abs=1e-12)
    assert claim.value("mean") == pytest.approx(float(np.mean(x) - np.mean(y)), abs=1e-12)


def test_index_pairing_of_unrelated_joint_artifacts_is_refused() -> None:
    x, y, _, _ = _gaussian_pair(2_000, 5)
    left = _artifact("snap-left", x, y)
    right = _artifact("snap-right", y, x)
    lx = jl.PropagationDraws.from_artifact_column("left_x", left, 0)
    ry = jl.PropagationDraws.from_artifact_column("right_y", right, 1)
    with pytest.raises(R) as error:
        jl.aligned_functional("product", lx, ry, "product")
    assert error.value.reason_code == "joint_law_required"
    assert error.value.detail == "joint_law_propagation.alignment_mismatch"
    assert error.value.remedy


def test_independent_marginal_artifacts_are_never_paired_by_index() -> None:
    x, y, _, _ = _gaussian_pair(2_000, 6)
    marginals = _artifact("snap-marginals", x, y, alignment="independent_marginals")
    mx, my = _draws(marginals)
    assert mx.alignment_id is None
    with pytest.raises(R) as error:
        jl.aligned_functional("nonlinear_functional", mx, my, "minimum")
    assert error.value.reason_code == "joint_law_required"
    assert error.value.detail == "joint_law_propagation.index_pairing_refused"
    with pytest.raises(CausalValueError):
        jl.aligned_functional(
            "product",
            jl.PropagationDraws.independent_marginal("a", x, evidence_id="s"),
            my,
            "product",
        )


# ---------------------------------------------------------------- linear moments


def test_covariance_aware_contrast_matches_the_closed_form() -> None:
    inputs = [
        jl.MomentInput("effect_a", "snapshot_1", 5.0),
        jl.MomentInput("effect_b", "snapshot_1", 2.0),
    ]
    cov = [[4.0, 3.0], [3.0, 9.0]]
    claim = jl.linear_moments(
        "contrast", inputs, [1.0, -1.0], cov, dependence="covariance", covariance_id="cov_ab"
    )
    assert claim.value("mean") == pytest.approx(3.0)
    assert claim.value("variance") == pytest.approx(4.0 + 9.0 - 2.0 * 3.0)
    assert claim.route == "linear_moments"
    weighted = jl.linear_moments(
        "weighted_sum", inputs, [0.25, 0.75], cov, dependence="covariance", covariance_id="c"
    )
    assert weighted.value("variance") == pytest.approx(
        0.0625 * 4.0 + 0.5625 * 9.0 + 2 * 0.25 * 0.75 * 3.0
    )


def test_undeclared_or_shared_evidence_dependence_is_refused() -> None:
    shared = [jl.MomentInput("a", "snap", 1.0), jl.MomentInput("b", "snap", 2.0)]
    diag = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(R) as error:
        jl.linear_moments("sum", shared, [1.0, 1.0], diag, dependence="declared_independent")
    assert error.value.reason_code == "evidence_double_counted"
    with pytest.raises(R) as error:
        jl.linear_moments("sum", shared, [1.0, 1.0], diag, dependence="unknown")
    assert error.value.reason_code == "sampling_dependence_unknown"
    distinct = [jl.MomentInput("a", "s1", 1.0), jl.MomentInput("b", "s2", 2.0)]
    ok = jl.linear_moments("sum", distinct, [1.0, 1.0], diag, dependence="declared_independent")
    assert ok.value("variance") == pytest.approx(2.0)


# --------------------------------------------------------------------- mixture


def test_known_weight_mixture_matches_the_closed_form() -> None:
    parts = [
        jl.MixtureComponent("a", "s1", 0.25, 0.0, 1.0),
        jl.MixtureComponent("b", "s2", 0.75, 4.0, 4.0),
    ]
    claim = jl.mixture(parts)
    assert claim.value("mean") == pytest.approx(3.0)
    # E[X^2] = 0.25 * 1 + 0.75 * (4 + 16) = 15.25
    assert claim.value("variance") == pytest.approx(15.25 - 9.0)
    with pytest.raises(R) as error:
        jl.mixture(parts, weights_declared=False)
    assert error.value.reason_code == "invalid_argument"
    assert error.value.detail == "joint_law_propagation.mixture_weights_unknown"


# ------------------------------------------------------------------- envelopes


def test_bernoulli_envelope_endpoints_are_the_closed_form() -> None:
    # Bernoulli(0.3), Bernoulli(0.6) as 10-draw marginals (exact equal-mass atoms).
    x = jl.PropagationDraws.independent_marginal("x", [1.0] * 3 + [0.0] * 7, evidence_id="s1")
    y = jl.PropagationDraws.independent_marginal("y", [1.0] * 6 + [0.0] * 4, evidence_id="s2")
    prod = jl.dependence_envelope("product", jl.EnvelopeQuery.product_moment(), x, y)
    assert prod.label == "structural_envelope" and prod.is_envelope
    assert prod.value("product_moment_lower") == pytest.approx(max(0.0, 0.3 + 0.6 - 1.0))
    assert prod.value("product_moment_upper") == pytest.approx(min(0.3, 0.6))
    cdf = jl.dependence_envelope("sum", jl.EnvelopeQuery.sum_cdf(0.0), x, y)
    # P(X+Y<=0) = 1 - p - q + P(1,1), P(1,1) in [0, 0.3].
    assert cdf.value("sum_cdf_lower") == pytest.approx(0.1)
    assert cdf.value("sum_cdf_upper") == pytest.approx(0.4)
    var = jl.dependence_envelope("contrast", jl.EnvelopeQuery.linear_variance(1.0, -1.0), x, y)
    assert var.value("linear_variance_lower") == pytest.approx(0.21 + 0.24 - 2 * 0.12)
    assert var.value("linear_variance_upper") == pytest.approx(0.21 + 0.24 + 2 * 0.18)


def test_equal_mass_envelope_equals_enumeration_over_permutations() -> None:
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [0.0, 0.125, 0.5, 0.625, 1.5]
    x = jl.PropagationDraws.independent_marginal("x", xs, evidence_id="s1")
    y = jl.PropagationDraws.independent_marginal("y", ys, evidence_id="s2")
    perms = list(itertools.permutations(range(len(xs))))
    for t in (0.5, 1.0, 1.125, 1.5, 2.25):
        hits = [sum(1 for i, j in enumerate(p) if xs[i] + ys[j] <= t) / len(xs) for p in perms]
        claim = jl.dependence_envelope("sum", jl.EnvelopeQuery.sum_cdf(t), x, y)
        assert claim.value("sum_cdf_lower") == pytest.approx(min(hits))
        assert claim.value("sum_cdf_upper") == pytest.approx(max(hits))
    prods = [sum(xs[i] * ys[j] for i, j in enumerate(p)) / len(xs) for p in perms]
    claim = jl.dependence_envelope("product", jl.EnvelopeQuery.product_moment(), x, y)
    assert claim.value("product_moment_lower") == pytest.approx(min(prods))
    assert claim.value("product_moment_upper") == pytest.approx(max(prods))


def test_uniform_marginals_makarov_and_monotone_couplings() -> None:
    m = 1_000
    atoms = (np.arange(m) + 0.5) / m
    u = jl.PropagationDraws.independent_marginal("x", atoms, evidence_id="s1")
    v = jl.PropagationDraws.independent_marginal("y", atoms, evidence_id="s2")
    cdf = jl.dependence_envelope("sum", jl.EnvelopeQuery.sum_cdf(1.0), u, v)
    assert abs(cdf.value("sum_cdf_lower") - 0.0) < 2.0 / m
    assert abs(cdf.value("sum_cdf_upper") - 1.0) < 2.0 / m
    prod = jl.dependence_envelope("product", jl.EnvelopeQuery.product_moment(), u, v)
    assert abs(prod.value("product_moment_upper") - 1.0 / 3.0) < 1e-3
    assert abs(prod.value("product_moment_lower") - 1.0 / 6.0) < 1e-3


def test_envelope_is_never_a_point_or_an_interval_and_ranges_are_labelled() -> None:
    x = jl.PropagationDraws.independent_marginal("x", [0.0, 1.0], evidence_id="s1")
    y = jl.PropagationDraws.independent_marginal("y", [0.0, 1.0], evidence_id="s2")
    claim = jl.dependence_envelope("sum", jl.EnvelopeQuery.sum_cdf(1.0), x, y)
    for closed in (claim.interval, claim.posterior):
        with pytest.raises(R) as error:
            closed()
        assert error.value.reason_code == "cell_not_licensed"
    with pytest.raises(R) as error:
        claim.interval()
    assert error.value.detail == "joint_law_propagation.interval_route_closed"
    with pytest.raises(R) as error:
        claim.posterior()
    assert error.value.detail == "joint_law_propagation.posterior_route_closed"
    # A value function without a verified sign structure is refused.
    bad = jl.PropagationDraws.independent_marginal("x", [-1.0, 1.0, 2.0], evidence_id="s1")
    other = jl.PropagationDraws.independent_marginal("y", [-1.0, 1.0, 2.0], evidence_id="s2")
    with pytest.raises(R) as error:
        jl.dependence_envelope(
            "nonlinear_functional", jl.EnvelopeQuery.of_functional("ratio"), bad, other
        )
    assert error.value.detail == "joint_law_propagation.structure_not_verified"
    ranged = jl.correlation_range(
        "sum",
        ("x", "s1"),
        ("y", "s2"),
        x_mean=1.0,
        x_sd=2.0,
        y_mean=3.0,
        y_sd=0.5,
        rho_lo=-0.5,
        rho_hi=0.5,
    )
    assert ranged.label == "assumption_range"
    assert ranged.value("linear_variance_lower") == pytest.approx(4.0 + 0.25 - 1.0)
    assert ranged.value("linear_variance_upper") == pytest.approx(4.0 + 0.25 + 1.0)


# ---------------------------------------------------------------- transported


def test_transported_composition_needs_an_aligned_law_or_a_covariance() -> None:
    cov = [[1.0, 0.25], [0.25, 1.0]]
    transported = jl.MomentInput("transported", "source-study", 1.0, role="transported_interval")
    other = jl.MomentInput("other_claim", "source-study", 2.0)
    diag = [[1.0, 0.0], [0.0, 1.0]]
    for dependence in ("unknown", "declared_independent"):
        with pytest.raises(R) as error:
            jl.linear_moments("sum", [transported, other], [1.0, 1.0], diag, dependence=dependence)
        assert error.value.reason_code == "joint_law_required"
        assert error.value.detail == "joint_law_propagation.transported_dependence_required"
    claim = jl.linear_moments(
        "sum",
        [transported, other],
        [1.0, 1.0],
        cov,
        dependence="covariance",
        covariance_id="shared-source-evidence",
    )
    assert claim.value("variance") == pytest.approx(2.5)


# ------------------------------------------------------------ artifact surface


def test_artifact_round_trip_mutation_and_fresh_process(tmp_path) -> None:  # type: ignore[no-untyped-def]
    x, y, _, _ = _gaussian_pair(1_000, 99)
    dx, dy = _draws(_artifact("snap-replay", x, y))
    claim = jl.aligned_functional("product", dx, dy, "product")
    data = claim.export()
    consumed = jl.consume_joint_law_propagation_artifact(data, expected_identity=claim.identity)
    assert consumed.identity == claim.identity
    assert consumed.values["mean"] == claim.value("mean")
    assert consumed.label == "point_only"
    with pytest.raises(R) as error:
        jl.consume_joint_law_propagation_artifact(data, expected_identity="not-the-identity")
    assert error.value.detail == "joint_law_propagation.artifact_refused"
    # Sampled single-byte edits never replay to a different accepted claim.
    for position in range(0, len(data), max(1, len(data) // 64)):
        mutated = bytearray(data)
        mutated[position] ^= 0x55
        try:
            other = jl.consume_joint_law_propagation_artifact(bytes(mutated))
        except (R, CausalValueError, CausalTypeError):
            continue
        assert other.identity == claim.identity
        assert other.values == consumed.values
    # An envelope artifact keeps its label through replay.
    env = jl.dependence_envelope(
        "sum",
        jl.EnvelopeQuery.sum_cdf(1.0),
        jl.PropagationDraws.independent_marginal("x", [0.0, 1.0], evidence_id="s1"),
        jl.PropagationDraws.independent_marginal("y", [0.0, 1.0], evidence_id="s2"),
    )
    replay = jl.consume_joint_law_propagation_artifact(env.export())
    assert replay.is_envelope and replay.label == "structural_envelope"

    path = tmp_path / "claim.bin"
    path.write_bytes(data)
    script = textwrap.dedent(
        """
        import json, sys
        from antecedent import joint_law_propagation as jl
        data = open(sys.argv[1], "rb").read()
        ok = jl.consume_joint_law_propagation_artifact(data, expected_identity=sys.argv[2])
        try:
            jl.consume_joint_law_propagation_artifact(data, expected_identity="other")
            refused = False
        except jl.JointLawPropagationRefusal:
            refused = True
        print(json.dumps({"mean": ok.values["mean"], "label": ok.label, "refused": refused}))
        """
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", script, str(path), claim.identity],
        capture_output=True,
        text=True,
        check=True,
    )
    seen = json.loads(completed.stdout)
    assert seen == {"mean": claim.value("mean"), "label": "point_only", "refused": True}


def test_inputs_are_typed() -> None:
    with pytest.raises(CausalTypeError):
        jl.PropagationDraws.from_artifact_column("x", "not an artifact", 0)  # type: ignore[arg-type]
    with pytest.raises(CausalValueError):
        jl.EnvelopeQuery.of_functional("bogus")
    with pytest.raises(CausalValueError):
        jl.linear_moments(
            "sum",
            [jl.MomentInput("a", "s", 1.0)],
            [1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            dependence="covariance",
            covariance_id="c",
        )
