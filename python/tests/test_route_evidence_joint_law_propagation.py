"""Joint-law propagation: one test per Python entry and per native entry.

Every value is hand-derived. Enumerated joint law over ten aligned rows, counts
``(0,0) x1, (0,2) x4, (1,0) x3, (1,2) x2``: ``E[X] = 0.5``, ``E[Y] = 1.2``,
``Var X = 0.25``, ``Var Y = 0.96``, ``E[XY] = 0.4``, ``Cov = -0.2``, and
``E[max(X, Y)] = 0.4 * 2 + 0.3 * 1 + 0.2 * 2 = 1.5``. For binary marginals with
``p = 0.4`` and ``q = 0.7`` every coupling is fixed by ``c = P(1, 1)`` in
``[0.1, 0.4]``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from antecedent import _native
from antecedent import joint_law_propagation as jl
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

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
    return JointDistributionArtifact(identity, np.column_stack([x, y]).astype(np.float64))


def _rows() -> tuple[np.ndarray, np.ndarray]:
    cells = [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)]
    x = np.array([a for a, _, c in cells for _ in range(c)])
    y = np.array([b for _, b, c in cells for _ in range(c)])
    return x, y


def _columns(snapshot: str):  # type: ignore[no-untyped-def]
    x, y = _rows()
    art = _artifact(snapshot, x, y)
    return (
        art,
        jl.PropagationDraws.from_artifact_column("x", art, 0),
        jl.PropagationDraws.from_artifact_column("y", art, 1),
    )


def _bernoulli(p: float, n: int = 10) -> list[float]:
    ones = round(p * n)
    return [1.0] * ones + [0.0] * (n - ones)


def _marginals() -> tuple[jl.PropagationDraws, jl.PropagationDraws]:
    return (
        jl.PropagationDraws.independent_marginal("x", _bernoulli(0.4), evidence_id="s1"),
        jl.PropagationDraws.independent_marginal("y", _bernoulli(0.7), evidence_id="s2"),
    )


# --------------------------------------------------------------- aligned joint law


def test_aligned_functional_is_the_row_mean_of_a_maximum_under_the_joint_law() -> None:
    _, dx, dy = _columns("snap-max")
    claim = require_success(jl.aligned_functional("nonlinear_functional", dx, dy, "maximum"))
    assert claim.value("mean") == pytest.approx(1.5, abs=1e-12)
    assert claim.route == "aligned_joint_law"
    assert claim.n_rows == 10
    # Rows of different joint laws are never paired.
    _, _, other_y = _columns("snap-max-other")
    with pytest.raises(R) as error:
        require_success(jl.aligned_functional("nonlinear_functional", dx, other_y, "maximum"))
    assert error.value.detail == "joint_law_propagation.alignment_mismatch"


def test_the_native_aligned_functional_entry_returns_a_sealed_claim_of_the_row_mean() -> None:
    _, dx, dy = _columns("snap-native-functional")
    xa, xc = dx._require_artifact()
    ya, yc = dy._require_artifact()
    assert xa is not None and ya is not None
    claim = jl._claim(
        _native.joint_law_aligned_functional("product", "x", xa, xc, "y", ya, yc, "product", None)
    )
    assert claim.value("mean") == pytest.approx(0.4, abs=1e-12)
    assert claim.n_rows == 10
    assert claim.alignment_id == dx.alignment_id == dy.alignment_id


def test_aligned_linear_takes_the_covariance_from_the_rows() -> None:
    _, dx, dy = _columns("snap-linear")
    claim = require_success(jl.aligned_linear("weighted_sum", (2.0, 3.0), dx, dy))
    # mean 2 * 0.5 + 3 * 1.2 = 4.6; variance 4 * 0.25 + 9 * 0.96 + 12 * (-0.2) = 7.24.
    assert claim.value("mean") == pytest.approx(4.6, abs=1e-12)
    assert claim.value("variance") == pytest.approx(7.24, abs=1e-12)
    assert claim.value("covariance") == pytest.approx(-0.2, abs=1e-12)


def test_the_native_aligned_linear_entry_returns_the_sealed_sum_moments() -> None:
    _, dx, dy = _columns("snap-native-linear")
    xa, xc = dx._require_artifact()
    ya, yc = dy._require_artifact()
    claim = jl._claim(
        _native.joint_law_aligned_linear("sum", 1.0, 1.0, "x", xa, xc, "y", ya, yc, None)
    )
    # mean 0.5 + 1.2; Var(X + Y) = 0.25 + 0.96 + 2 * (-0.2) = 0.81.
    assert claim.value("mean") == pytest.approx(1.7, abs=1e-12)
    assert claim.value("variance") == pytest.approx(0.81, abs=1e-12)
    assert claim.label == "point_only"


def test_the_native_column_entry_reports_the_alignment_of_a_joint_column() -> None:
    x, y = _rows()
    joint = _artifact("snap-column", x, y)
    parsed, refusal = _native.joint_law_column("y", joint._native, 1)
    assert refusal is None and parsed is not None
    info = json.loads(parsed)
    assert info["evidence_id"] == "snap-column"
    # The entry reports the draw alignment kind ("joint"), plus the alignment identity the
    # propagation rule derives from it; the wrapper carries both through unchanged.
    assert info["alignment"] == "joint"
    assert info["alignment_id"]
    wrapped = jl.PropagationDraws.from_artifact_column("y", joint, 1)
    assert wrapped.alignment == "joint"
    assert info["alignment_id"] == wrapped.alignment_id
    marginals = _artifact("snap-column-marginals", x, y, alignment="independent_marginals")
    parsed, refusal = _native.joint_law_column("y", marginals._native, 1)
    assert refusal is None and parsed is not None
    info = json.loads(parsed)
    assert info["alignment"] == "independent_marginals" and info["alignment_id"] is None


# ---------------------------------------------------------------- linear moments


def test_linear_moments_of_three_independent_inputs_add_scaled_variances() -> None:
    inputs = [
        jl.MomentInput("a", "s1", 1.0),
        jl.MomentInput("b", "s2", 0.0),
        jl.MomentInput("c", "s3", 2.0),
    ]
    covariance = [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]
    claim = require_success(
        jl.linear_moments(
            "weighted_sum", inputs, [1.0, 2.0, -1.0], covariance, dependence="declared_independent"
        )
    )
    # mean 1 + 0 - 2 = -1; variance 1 + 4 * 4 + 9 = 26.
    assert claim.value("mean") == pytest.approx(-1.0, abs=1e-12)
    assert claim.value("variance") == pytest.approx(26.0, abs=1e-12)
    shared = [jl.MomentInput("a", "s1", 1.0), jl.MomentInput("b", "s1", 0.0), inputs[2]]
    with pytest.raises(R) as error:
        require_success(
            jl.linear_moments(
                "weighted_sum",
                shared,
                [1.0, 2.0, -1.0],
                covariance,
                dependence="declared_independent",
            )
        )
    assert error.value.reason_code == "evidence_double_counted"


def test_the_native_linear_moments_entry_uses_the_declared_covariance() -> None:
    inputs = [jl.MomentInput("a", "s1", 1.0), jl.MomentInput("b", "s2", 2.0)]
    claim = jl._claim(
        _native.joint_law_linear_moments(
            "sum",
            json.dumps([item._wire() for item in inputs]),
            [1.0, 1.0],
            [1.0, 0.25, 0.25, 1.0],
            "covariance",
            "pair_cov",
            None,
        )
    )
    # mean 1 + 2 = 3; variance 1 + 1 + 2 * 0.25 = 2.5.
    assert claim.value("mean") == pytest.approx(3.0, abs=1e-12)
    assert claim.value("variance") == pytest.approx(2.5, abs=1e-12)
    assert claim.route == "linear_moments"


# --------------------------------------------------------------------- mixture


def test_a_known_weight_mixture_splits_variance_into_within_and_between() -> None:
    parts = [
        jl.MixtureComponent("a", "s1", 0.5, 1.0, 2.0),
        jl.MixtureComponent("b", "s2", 0.5, 3.0, 2.0),
    ]
    claim = require_success(jl.mixture(parts))
    # mean 2, within 2, between 1, variance 3.
    assert claim.value("mean") == pytest.approx(2.0, abs=1e-12)
    assert claim.value("within") == pytest.approx(2.0, abs=1e-12)
    assert claim.value("variance") == pytest.approx(3.0, abs=1e-12)


def test_the_native_mixture_entry_applies_the_law_of_total_variance() -> None:
    parts = [
        jl.MixtureComponent("a", "s1", 0.2, 0.0, 1.0),
        jl.MixtureComponent("b", "s2", 0.8, 5.0, 1.0),
    ]
    claim = jl._claim(_native.joint_law_mixture(json.dumps([p._wire() for p in parts]), None))
    # mean 4; within 1; between 0.2 * 16 + 0.8 * 1 = 4; variance 5.
    assert claim.value("mean") == pytest.approx(4.0, abs=1e-12)
    assert claim.value("within") == pytest.approx(1.0, abs=1e-12)
    assert claim.value("variance") == pytest.approx(5.0, abs=1e-12)
    assert claim.route == "known_weight_mixture"


# ------------------------------------------------------------------- envelopes


def test_the_dependence_envelope_is_the_makarov_range_of_the_deciding_cell() -> None:
    x, y = _marginals()
    at0 = require_success(jl.dependence_envelope("sum", jl.EnvelopeQuery.sum_cdf(0.0), x, y))
    # P(X + Y <= 0) = P(0, 0) = c - 0.1 in [0, 0.3].
    assert at0.value("sum_cdf_lower") == pytest.approx(0.0, abs=1e-12)
    assert at0.value("sum_cdf_upper") == pytest.approx(0.3, abs=1e-12)
    at1 = require_success(jl.dependence_envelope("sum", jl.EnvelopeQuery.sum_cdf(1.0), x, y))
    # P(X + Y <= 1) = 1 - c in [0.6, 0.9].
    assert at1.value("sum_cdf_lower") == pytest.approx(0.6, abs=1e-12)
    assert at1.value("sum_cdf_upper") == pytest.approx(0.9, abs=1e-12)
    assert at1.label == "structural_envelope" and at1.is_envelope


def test_the_native_envelope_entry_returns_the_product_moment_range() -> None:
    claim = jl._claim(
        _native.joint_law_envelope(
            "product",
            jl.EnvelopeQuery.product_moment()._json(),
            "x",
            "s1",
            _bernoulli(0.4),
            "y",
            "s2",
            _bernoulli(0.7),
            None,
        )
    )
    # E[XY] = c in [0.1, 0.4].
    assert claim.value("product_moment_lower") == pytest.approx(0.1, abs=1e-12)
    assert claim.value("product_moment_upper") == pytest.approx(0.4, abs=1e-12)
    assert claim.route == "frechet_envelope"


def test_a_declared_correlation_range_is_linear_in_the_correlation() -> None:
    claim = require_success(
        jl.correlation_range(
            "weighted_sum",
            ("x", "s1"),
            ("y", "s2"),
            x_mean=0.0,
            x_sd=1.0,
            y_mean=2.0,
            y_sd=3.0,
            a=1.0,
            b=2.0,
            rho_lo=-0.2,
            rho_hi=0.6,
        )
    )
    # s_x s_y = 3: E[XY] in [-0.6, 1.8]; Var = 1 + 36 + 12 rho in [34.6, 44.2].
    assert claim.label == "assumption_range"
    assert claim.value("product_moment_lower") == pytest.approx(-0.6, abs=1e-12)
    assert claim.value("product_moment_upper") == pytest.approx(1.8, abs=1e-12)
    assert claim.value("linear_variance_lower") == pytest.approx(34.6, abs=1e-12)
    assert claim.value("linear_variance_upper") == pytest.approx(44.2, abs=1e-12)


def test_the_native_correlation_range_entry_returns_the_assumption_range() -> None:
    claim = jl._claim(
        _native.joint_law_correlation_range(
            "weighted_sum", "x", "s1", "y", "s2", 2.0, 3.0, -1.0, 2.0, 2.0, 1.0, 0.0, 0.5, None
        )
    )
    # centre -2, scale 6: E[XY] in [-2, 1]; Var(2X + Y) = 40 + 24 rho in [40, 52].
    assert claim.value("product_moment_lower") == pytest.approx(-2.0, abs=1e-12)
    assert claim.value("product_moment_upper") == pytest.approx(1.0, abs=1e-12)
    assert claim.value("linear_variance_lower") == pytest.approx(40.0, abs=1e-12)
    assert claim.value("linear_variance_upper") == pytest.approx(52.0, abs=1e-12)
    assert claim.label == "assumption_range"


# ------------------------------------------------------------ artifact surface


def test_consume_replays_a_sealed_claim_against_the_retained_identity() -> None:
    parts = [
        jl.MixtureComponent("a", "s1", 0.5, 1.0, 2.0),
        jl.MixtureComponent("b", "s2", 0.5, 3.0, 2.0),
    ]
    claim = require_success(jl.mixture(parts))
    consumed = require_success(
        jl.consume_joint_law_propagation_artifact(
            bytes(claim.export()), expected_identity=claim.identity
        )
    )
    assert consumed.identity == claim.identity
    assert consumed.values["variance"] == pytest.approx(3.0, abs=1e-12)
    assert consumed.route == "known_weight_mixture"
    with pytest.raises(R) as error:
        require_success(
            jl.consume_joint_law_propagation_artifact(
                bytes(claim.export()), expected_identity="0" * 64
            )
        )
    assert error.value.detail == "joint_law_propagation.artifact_refused"


def test_the_native_consume_entry_returns_a_summary_or_a_refusal_payload() -> None:
    x, y = _marginals()
    claim = require_success(
        jl.dependence_envelope("product", jl.EnvelopeQuery.product_moment(), x, y)
    )
    data = bytes(claim.export())
    summary, refusal = _native.joint_law_consume(data, claim.identity, None)
    assert refusal is None and summary is not None
    info = json.loads(summary)
    assert info["identity"] == claim.identity
    assert info["is_envelope"] is True
    assert info["label"] == "structural_envelope"
    values = {name: value for name, value in info["values"]}
    assert values["product_moment_upper"] == pytest.approx(0.4, abs=1e-12)
    # Truncated bytes are a refusal payload, not a summary.
    summary, refusal = _native.joint_law_consume(data[: len(data) // 2], None, None)
    assert summary is None and refusal is not None
