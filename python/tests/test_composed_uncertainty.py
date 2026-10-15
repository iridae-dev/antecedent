"""Composed-uncertainty readings through the Python surface.

Truth is enumerated and computed here in plain Python / numpy:

* ten rows ``(x, y)``: ``(0,0)`` x1, ``(0,2)`` x4, ``(1,0)`` x3, ``(1,2)`` x2, so
  ``Var(X - Y) = 1.61`` (population), ``g = x y`` has two 2s and eight 0s (mean ``0.4``, sample
  variance ``6.4 / 9``) and ``P(g > 1) = 0.2``;
* a mixture of ``[0, 1, 2, 3]`` (weight 0.4) and ``[0, 0, 0, 2]`` (weight 0.6) has
  ``P(X > 1.5) = 0.35``;
* transported pieces ``s = [1, 2, 3, 4]``, ``t = [2, 1, 4, 3]``, ``f = [0, 0, 1, 2]`` sum to
  ``[3, 3, 8, 9]`` (population variance ``7.6875``).
"""

from __future__ import annotations

import math

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
Z_95 = 1.959963984540054


def _artifact(
    snapshot: str, columns: list[np.ndarray], alignment: str = "joint"
) -> JointDistributionArtifact:
    names = ("x", "y", "z")[: len(columns)]
    quantities = tuple(
        ScientificQuantity.outcome(name, units="units", population="target", regime="do(a=1)")
        for name in names
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
    return JointDistributionArtifact(identity, np.column_stack(columns).astype(np.float64))


def _enumerated(scale: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rows = [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)]
    x = np.array([a for a, _, c in rows for _ in range(c * scale)])
    y = np.array([b for _, b, c in rows for _ in range(c * scale)])
    return x, y


def _pair(snapshot: str, scale: int = 1) -> tuple[jl.PropagationDraws, jl.PropagationDraws]:
    x, y = _enumerated(scale)
    art = _artifact(snapshot, [x, y])
    return (
        jl.PropagationDraws.from_artifact_column("x", art, 0),
        jl.PropagationDraws.from_artifact_column("y", art, 1),
    )


_PIECES = (
    np.array([1.0, 2.0, 3.0, 4.0]),
    np.array([2.0, 1.0, 4.0, 3.0]),
    np.array([0.0, 0.0, 1.0, 2.0]),
)


def _piece_draws(snapshot: str, alignment: str = "joint") -> list[jl.PropagationDraws]:
    art = _artifact(snapshot, list(_PIECES), alignment=alignment)
    return [
        jl.PropagationDraws.from_artifact_column(name, art, column)
        for column, name in enumerate(("source", "target", "functional"))
    ]


# ------------------------------------------------------------------- spread


def test_aligned_functional_spread_matches_the_enumerated_rows() -> None:
    x, y = _pair("snap-enum")
    claim = jl.spread("product", x, y, "product")
    assert claim.label == "uncalibrated_spread"
    assert claim.route == "aligned_joint_law"
    assert not claim.is_envelope
    assert claim.n_rows == 10
    assert claim.value("mean") == pytest.approx(0.4, abs=1e-12)
    assert claim.value("variance") == pytest.approx(6.4 / 9.0, abs=1e-12)
    assert claim.value("mean_standard_error") == pytest.approx(4.0 / 15.0, abs=1e-12)
    assert claim.verdict is None
    # The point reading of the same columns keeps its own label and identity.
    point = jl.aligned_functional("product", x, y, "product")
    assert point.label == "point_only"
    assert point.identity != claim.identity


def test_aligned_linear_spread_is_the_population_variance_of_the_contrast() -> None:
    x, y = _pair("snap-enum")
    xs, ys = _enumerated()
    claim = jl.aligned_linear_spread("contrast", (1.0, -1.0), x, y)
    assert claim.label == "uncalibrated_spread"
    assert claim.value("variance") == pytest.approx(1.61, abs=1e-12)
    assert claim.value("variance") == pytest.approx(float(np.var(xs - ys)), abs=1e-12)
    assert claim.value("mean") == pytest.approx(-0.7, abs=1e-12)


def test_linear_moments_spread_agrees_with_the_aligned_rows() -> None:
    inputs = [jl.MomentInput("x", "snap-enum", 0.5), jl.MomentInput("y", "snap-enum", 1.2)]
    claim = jl.linear_moments_spread(
        "contrast",
        inputs,
        [1.0, -1.0],
        [[0.25, -0.2], [-0.2, 0.96]],
        dependence="covariance",
        covariance_id="enumeration",
    )
    assert claim.label == "uncalibrated_spread"
    assert claim.route == "linear_moments"
    assert claim.value("variance") == pytest.approx(1.61, abs=1e-12)
    with pytest.raises(R) as error:
        jl.linear_moments_spread(
            "contrast",
            inputs,
            [1.0, -1.0],
            [[0.25, 0.0], [0.0, 0.96]],
            dependence="declared_independent",
        )
    assert error.value.reason_code == "evidence_double_counted"
    assert error.value.detail == "joint_law_propagation.shared_evidence_declared_independent"


def test_spread_of_unpaired_draws_is_refused() -> None:
    x, y = _enumerated()
    marginals = _artifact("snap-marginals", [x, y], alignment="independent_marginals")
    mx = jl.PropagationDraws.from_artifact_column("mx", marginals, 0)
    my = jl.PropagationDraws.from_artifact_column("my", marginals, 1)
    with pytest.raises(R) as error:
        jl.spread("nonlinear_functional", mx, my, "minimum")
    assert error.value.reason_code == "joint_law_required"
    assert error.value.detail == "joint_law_propagation.index_pairing_refused"
    assert error.value.remedy
    px, _ = _pair("snap-enum")
    _, other_y = _pair("snap-other")
    with pytest.raises(R) as error:
        jl.aligned_linear_spread("sum", (1.0, 1.0), px, other_y)
    assert error.value.detail == "joint_law_propagation.alignment_mismatch"


# ------------------------------------------------------------------ interval


def test_composed_interval_opens_only_with_a_measured_record_at_its_coordinate() -> None:
    x, y = _pair("snap-enum")
    claim = jl.aligned_linear_spread("contrast", (1.0, -1.0), x, y)
    coordinate = claim.coordinate_id()
    assert coordinate == "contrast|aligned_joint_law|estimate,estimate"

    # The caller supplies no record: the registry decides. A claim whose composed key has no
    # currently attesting record is closed, whatever construction is named.
    assert not hasattr(jl, "CoverageRecord")
    # Inside the measured scope the interval is bound to its attesting record.
    interval = claim.check_composed_interval("normal_approximation", evidence_rows=200)
    assert interval.label == "measured_interval"
    assert interval.record_id.startswith("cov.composed_quantity.")
    assert interval.nominal_level == 0.95
    half = 1.959963984540054 * math.sqrt(1.61)
    assert interval.lower == pytest.approx(-0.7 - half, abs=1e-6)
    assert interval.upper == pytest.approx(-0.7 + half, abs=1e-6)
    # Another construction or a smaller evidence size has no attesting record.
    with pytest.raises(R) as error:
        claim.check_composed_interval("aligned_draw_equal_tail_exchangeable_rank", evidence_rows=200)
    assert error.value.reason_code == "cell_not_licensed"
    assert error.value.detail == "joint_law_propagation.interval_route_closed"
    with pytest.raises(R) as error:
        claim.check_composed_interval("normal_approximation", evidence_rows=100)
    assert error.value.detail == "joint_law_propagation.coverage_record_unmeasured"
    assert error.value.remedy
    with pytest.raises(CausalValueError):
        claim.check_composed_interval("normal_approximation", evidence_rows=0)

    # Component-interval and posterior readings stay closed.
    with pytest.raises(R) as error:
        claim.interval()
    assert error.value.detail == "joint_law_propagation.interval_route_closed"
    with pytest.raises(TypeError):
        claim.check_composed_interval("normal_approximation")  # type: ignore[call-arg]


def test_an_envelope_has_no_composed_interval() -> None:
    env = jl.dependence_envelope(
        "sum",
        jl.EnvelopeQuery.sum_cdf(1.0),
        jl.PropagationDraws.independent_marginal("x", [0.0, 1.0], evidence_id="s1"),
        jl.PropagationDraws.independent_marginal("y", [0.0, 1.0], evidence_id="s2"),
    )
    with pytest.raises(R) as error:
        env.check_composed_interval("normal_approximation", evidence_rows=200)
    assert error.value.reason_code == "cell_not_licensed"
    assert error.value.detail == "joint_law_propagation.spread_needs_licensed_dependence"
    with pytest.raises(R):
        env.coordinate_id()


# --------------------------------------------------------------- exceedance


def test_composed_exceedance_reports_probability_band_and_verdict() -> None:
    x, y = _pair("snap-enum-large", scale=100)

    def decide(cutoff: float) -> jl.PropagationClaim:
        return jl.composed_exceedance(
            "product", x, y, "product", threshold=1.0, cutoff=cutoff, band_level=0.95
        )

    claim = decide(0.15)
    assert claim.label == "uncalibrated_spread"
    assert claim.value("probability") == pytest.approx(0.2, rel=1e-9, abs=1e-12)
    assert claim.n_rows == 1000
    adjusted = (200.0 + Z_95**2 / 2.0) / (1000.0 + Z_95**2)
    standard_error = math.sqrt(adjusted * (1.0 - adjusted) / (1000.0 + Z_95**2))
    assert claim.value("standard_error") == pytest.approx(standard_error, abs=1e-6)
    assert claim.value("band") == pytest.approx(Z_95 * standard_error, abs=1e-6)
    assert "verdict_code" not in claim.values
    assert claim.verdict == "act"
    assert decide(0.2).verdict == "unresolved"
    assert decide(0.25).verdict == "withhold"

    # Ten rows: the band is wide and the same cutoff is unresolved.
    sx, sy = _pair("snap-enum")
    small = jl.composed_exceedance("product", sx, sy, "product", threshold=1.0, cutoff=0.15)
    assert small.verdict == "unresolved"

    # The sealed artifact replays the decision, including its verdict.
    consumed = jl.consume_joint_law_propagation_artifact(
        claim.export(), expected_identity=claim.identity
    )
    assert consumed.label == "uncalibrated_spread"
    assert consumed.verdict == "act"
    assert consumed.values["probability"] == pytest.approx(0.2, rel=1e-9, abs=1e-12)
    assert "verdict_code" not in consumed.values


def test_exceedance_cutoff_outside_the_unit_interval_is_refused() -> None:
    x, y = _pair("snap-enum")
    with pytest.raises(R) as error:
        jl.composed_exceedance("product", x, y, "product", threshold=1.0, cutoff=1.5)
    assert error.value.reason_code == "invalid_argument"
    assert error.value.detail == "joint_law_propagation.invalid_request"
    with pytest.raises(CausalTypeError):
        jl.composed_exceedance("product", x, y, "product", threshold="1", cutoff=0.5)  # type: ignore[arg-type]


def test_known_weight_mixture_exceedance_matches_the_weighted_fractions() -> None:
    components = [
        jl.MixtureDraws("a", "s1", 0.4, [0.0, 1.0, 2.0, 3.0]),
        jl.MixtureDraws("b", "s2", 0.6, [0.0, 0.0, 0.0, 2.0]),
    ]
    claim = jl.mixture_exceedance(components, threshold=1.5)
    assert claim.label == "uncalibrated_spread"
    assert claim.route == "known_weight_mixture"
    assert claim.value("probability") == pytest.approx(0.35, abs=1e-15)
    expected = math.sqrt(0.16 * 0.25 / 4.0 + 0.36 * 0.1875 / 4.0)
    assert claim.value("standard_error") == pytest.approx(expected, abs=1e-15)
    changed = [components[0], jl.MixtureDraws("b", "s2", 0.6, [0.0, 0.0, 0.0, 1.0])]
    other = jl.mixture_exceedance(changed, threshold=1.5)
    assert other.identity != claim.identity
    assert other.value("probability") == pytest.approx(0.2, abs=1e-15)
    skewed = [jl.MixtureDraws("a", "s1", 0.9, [0.0, 1.0]), components[1]]
    with pytest.raises(R) as error:
        jl.mixture_exceedance(skewed, threshold=1.5)
    assert error.value.reason_code == "invalid_argument"


# --------------------------------------------------------------- transported


def test_transported_spread_needs_all_three_pieces_and_their_dependence() -> None:
    covariance = [[1.0, 0.5, 0.0], [0.5, 4.0, 0.0], [0.0, 0.0, 9.0]]
    pieces = [
        jl.MomentInput("source", "snap-source", 0.0, role="source_sampling"),
        jl.MomentInput("target", "snap-target", 0.0, role="target_sampling"),
        jl.MomentInput("functional", "snap-source", 0.0, role="transport_functional"),
    ]
    claim = jl.transported_spread(
        pieces, [1.0, 1.0, 1.0], covariance, covariance_id="piece-covariance"
    )
    assert claim.label == "uncalibrated_spread"
    assert claim.operation == "transported_composition"
    assert claim.value("variance") == pytest.approx(15.0, abs=1e-12)  # 1 + 4 + 9 + 2 * 0.5

    with pytest.raises(R) as error:
        jl.transported_spread(
            pieces[:2], [1.0, 1.0], [[1.0, 0.5], [0.5, 4.0]], covariance_id="piece-covariance"
        )
    assert error.value.reason_code == "joint_law_required"
    assert error.value.detail == "joint_law_propagation.transport_pieces_incomplete"
    assert error.value.offending == "transport_functional"

    for dependence in ("declared_independent", "unknown"):
        with pytest.raises(R) as error:
            jl.transported_spread(
                pieces,
                [1.0, 1.0, 1.0],
                [[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]],
                dependence=dependence,
            )
        assert error.value.reason_code == "joint_law_required"
        assert error.value.detail == "joint_law_propagation.transported_dependence_required"


def test_transported_spread_from_aligned_pieces_matches_the_row_sums() -> None:
    source, target, functional = _piece_draws("snap-pieces")
    rows = _PIECES[0] + _PIECES[1] + _PIECES[2]
    claim = jl.transported_spread_aligned(source, target, functional)
    assert claim.label == "uncalibrated_spread"
    assert claim.value("mean") == pytest.approx(float(np.mean(rows)), abs=1e-12)
    assert claim.value("variance") == pytest.approx(float(np.var(rows)), abs=1e-12)
    assert claim.value("variance") == pytest.approx(7.6875, abs=1e-12)

    _, other_target, _ = _piece_draws("snap-other-pieces")
    with pytest.raises(R) as error:
        jl.transported_spread_aligned(source, other_target, functional)
    assert error.value.reason_code == "joint_law_required"
    assert error.value.detail == "joint_law_propagation.alignment_mismatch"

    ms, mt, mf = _piece_draws("snap-marginals", alignment="independent_marginals")
    with pytest.raises(R) as error:
        jl.transported_spread_aligned(ms, mt, mf)
    assert error.value.detail == "joint_law_propagation.index_pairing_refused"

    with pytest.raises(CausalValueError):
        jl.transported_spread_aligned(
            jl.PropagationDraws.independent_marginal("s", [1.0, 2.0], evidence_id="e"),
            target,
            functional,
        )
