"""Bridges that carry distributions, through the Python facade.

Every asserted number has an oracle computed here with numpy, never by native code:

* an affine map with an uncertain slope and intercept over the draws of a posterior has the
  first-order delta-method mean and variance ``a + b mu + c`` and
  ``Va + mu^2 Vb + b^2 V + 2 mu b c + 2 mu d + Vb V + c^2``, evaluated on the empirical moments of
  the supplied draws;
* an exact map is bit for bit the map of each draw, and the bridged table keeps the source's
  shape, its alignment and every other column;
* a downstream chain of transport and exponential utility is the numpy expectation over the same
  bridged rows, not the utility of the mean.

An interval of a bridged distribution is published only through a measured coverage record, and
these tests assert that it refuses outside one.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import propagation_chain as pc
from antecedent import scientific_bridge as sb
from antecedent.errors import CausalTypeError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.scientific_bridge import ScientificBridgeRefusal

from _refusal import assert_registered_refusal

ROWS = 20_000
_NORMAL = NormalDist()


def outcome(variable, *, units):
    return ScientificQuantity.outcome(
        variable, units=units, population="adults", regime="do(t=1)", functional="average_effect"
    )


SOURCE = outcome("trial:biomarker", units="mmol_per_l")
COST = outcome("trial:cost", units="usd")
TARGET = outcome("trial:survival", units="log_hazard")


def grid(mean, sd, rows=ROWS):
    return np.array([mean + sd * _NORMAL.inv_cdf((i + 0.5) / rows) for i in range(rows)])


def table(rows=ROWS, *, semantic="causal_functional_posterior", alignment="joint"):
    source = grid(1.0, 0.3, rows)
    partner = grid(0.2, 0.3, rows)[(np.arange(rows) * 12_487) % rows]
    identity = DistributionIdentity(
        semantic=semantic,
        quantities=(SOURCE, COST),
        alignment=alignment,
        source_id="study-1",
        provider_id="posterior.provider",
        rng_id="seed:11",
        snapshot_id="snapshot-1",
        causal_contract_id="contract-1",
    )
    return JointDistributionArtifact(identity, np.column_stack([source, partner]))


def assumption():
    return sb.Assumption(
        "validation",
        "surrogate_validation",
        "the surrogate is validated at the trial level",
        evidence="meta-analysis:ckd-2024",
    )


def contract(*, uncertain=True, covariance=None, value_kind="posterior", **overrides):
    fields = dict(
        kind="surrogate_terminal_outcome",
        source=SOURCE,
        source_class="surrogate_endpoint",
        target=TARGET,
        target_class="terminal_outcome",
        evidence_scope=sb.Scope("adults", "do(t=1)"),
        mapping=sb.AffineMapping(intercept=0.5, slope=2.0),
        assumptions=(assumption(),),
        source_value=value_kind,
        target_value=value_kind,
    )
    if uncertain:
        fields["mapping_uncertainty"] = sb.MappingUncertainty(
            "calibration-fit",
            intercept_variance=0.01,
            slope_variance=0.04,
            intercept_slope_covariance=-0.005,
        )
        fields["source_dependence"] = (
            sb.SourceDependence.independent("disjoint-samples")
            if covariance is None
            else sb.SourceDependence.slope_covariance(covariance, "shared-sample")
        )
    fields.update(overrides)
    return sb.BridgeContract(**fields)


def refused(call, code, detail):
    with pytest.raises(ScientificBridgeRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == code, (caught.value.reason_code, str(caught.value))
    assert caught.value.detail == detail, (caught.value.detail, str(caught.value))
    return caught.value


def delta(source, covariance=0.0):
    mu, v = source.mean(), source.var()
    b, va, vb, d = 2.0, 0.01, 0.04, -0.005
    mean = 0.5 + b * mu + covariance
    variance = (
        va + mu**2 * vb + b**2 * v + 2 * mu * b * covariance + 2 * mu * d + vb * v + covariance**2
    )
    return mean, variance


def test_a_bridge_of_draws_keeps_the_alignment_and_the_declared_uncertainty():
    draws = table()
    source = np.asarray(draws)[:, 0]
    bridge = sb.prepare_distribution_bridge(contract(uncertain=False))
    exact = bridge.derive(draws, seed=3)
    out = np.asarray(exact.distribution())
    # Draws in, draws out: the same rows, the sibling column untouched, an exact map exact.
    assert out.shape == (ROWS, 2)
    np.testing.assert_array_equal(out[:, 1], np.asarray(draws)[:, 1])
    np.testing.assert_array_equal(out[:, 0], 0.5 + 2.0 * source)
    identity = exact.distribution().identity
    assert identity.alignment == "joint"
    assert identity.source_id == draws.identity.source_id
    assert identity.snapshot_id == draws.identity.snapshot_id
    assert identity.quantities[0] == TARGET
    assert identity.quantities[1] == COST
    assert exact.reading == "uncalibrated_spread"
    assert exact.calibration == "unmeasured"
    assert exact.value_kind == "posterior"
    np.testing.assert_array_equal(np.asarray(exact.source_distribution()), np.asarray(draws))

    # An uncertain slope and intercept: the delta method and its dependence.
    independent = sb.prepare_distribution_bridge(contract()).derive(draws, seed=3)
    mapped = np.asarray(independent.distribution())[:, 0]
    mean, variance = delta(source)
    assert mapped.mean() == pytest.approx(mean, abs=0.02)
    assert mapped.var() == pytest.approx(variance, rel=0.06)
    assert independent.variance == pytest.approx(mapped.var(), rel=1e-9)

    covariance = 0.02
    dependent = sb.prepare_distribution_bridge(contract(covariance=covariance)).derive(
        draws, seed=3
    )
    shifted = np.asarray(dependent.distribution())[:, 0]
    mean_c, variance_c = delta(source, covariance)
    assert shifted.mean() == pytest.approx(mean_c, abs=0.02)
    assert shifted.var() == pytest.approx(variance_c, rel=0.06)
    # The declared dependence is not the same as independence.
    assert shifted.mean() - mapped.mean() == pytest.approx(covariance, abs=0.012)
    assert shifted.var() - mapped.var() > 0.04
    assert dependent.source_dependence == sb.SourceDependence.slope_covariance(
        covariance, "shared-sample"
    )
    assert dependent.mapping_uncertainty == contract().mapping_uncertainty
    # Another seed is another realization of the mapping's coefficients; the same seed replays.
    again = sb.prepare_distribution_bridge(contract()).derive(draws, seed=3)
    other = sb.prepare_distribution_bridge(contract()).derive(draws, seed=4)
    assert again.draws_digest == independent.draws_digest
    assert other.draws_digest != independent.draws_digest
    assert other.claim_id != independent.claim_id


def test_the_refusals_of_a_bridge_of_draws_name_what_would_license_them():
    draws = table(2_000)
    bridge = sb.prepare_distribution_bridge(contract())
    # An associational claim is never a bridge source.
    refused(
        lambda: bridge.derive(draws, seed=1, source_kind="associational"),
        "cell_not_licensed",
        "scientific_bridge.source_not_draws",
    )
    # A bootstrap is neither a posterior nor a predictive law.
    refused(
        lambda: bridge.derive(table(2_000, semantic="bootstrap"), seed=1),
        "distribution_meaning_mismatch",
        "scientific_bridge.draws_meaning_mismatch",
    )
    # Independent marginals have no row pairing to keep.
    refused(
        lambda: bridge.derive(table(2_000, alignment="independent_marginals"), seed=1),
        "joint_law_required",
        "scientific_bridge.draws_not_aligned",
    )
    # An uncertain map declares how it depends on the source.
    refused(
        lambda: sb.prepare_distribution_bridge(replace(contract(), source_dependence=None)),
        "sampling_dependence_unknown",
        "scientific_bridge.source_dependence_missing",
    )
    # A bridged mean is the point route's; a distribution is a posterior, never a mean.
    result = bridge.derive(draws, seed=1)
    refused(
        lambda: result.require_value_kind("mean"),
        "bridge_substitution_refused",
        "scientific_bridge.value_kind_substitution",
    )
    result.require_value_kind("posterior")
    refused(
        result.interval,
        "cell_not_licensed",
        "scientific_bridge.interval_not_licensed",
    )
    # An interval needs a measured record; an uncertain map has none.
    refused(
        lambda: result.interval(0.95),
        "cell_not_licensed",
        "scientific_bridge.interval_scope_outside_measured",
    )
    with pytest.raises(CausalTypeError):
        bridge.derive(np.zeros((3, 2)), seed=1)


def test_an_exact_map_publishes_its_interval_as_an_exact_transform():
    draws = table(4_000)
    exact = sb.prepare_distribution_bridge(contract(uncertain=False)).derive(draws, seed=2)
    interval = exact.interval(0.95)
    sd = np.asarray(draws)[:, 0].std()
    assert interval.licensed_by == "exact_transform_of_declared_source_interval"
    assert interval.source_standard_error == pytest.approx(sd, rel=1e-9)
    z = _NORMAL.inv_cdf(0.975)
    assert interval.upper - interval.lower == pytest.approx(2 * z * 2.0 * sd, rel=1e-9)


def test_a_bridge_is_a_stage_of_a_chain_that_propagates_the_mapping_and_the_source_jointly():
    draws = table(4_001)
    bridged = sb.prepare_distribution_bridge(contract(covariance=0.02)).derive(draws, seed=9)
    parts = bridged.chain_stage("terminal")
    assert [source.id for source in parts.sources] == [
        "terminal.source",
        "terminal.slope",
        "terminal.intercept",
    ]
    assert parts.stage.kind == "measurement_mapping"
    assert parts.stage.op["scale"] == {"kind": "input", "position": 1}
    assert parts.stage.op["shift"] == {"kind": "input", "position": 2}
    chain = pc.propagate(
        list(parts.sources),
        [
            parts.stage,
            pc.affine("transported", "transport", ["terminal"], scale=0.9, shift=0.0),
            pc.unary(
                "utility",
                "decision_utility",
                ["transported"],
                "exponential_utility",
                risk_aversion=0.5,
            ),
        ],
        output="utility",
    )
    mapped = np.asarray(bridged.distribution())[:, 0]
    assert chain.stage("terminal").value("mean") == pytest.approx(mapped.mean(), abs=1e-9)
    expectation = (1.0 - np.exp(-0.5 * (0.9 * mapped))).mean()
    assert chain.stage("utility").value("mean") == pytest.approx(expectation, abs=1e-9)
    assert abs(expectation - (1.0 - np.exp(-0.5 * 0.9 * mapped.mean()))) > 1e-4


def test_the_artifact_carries_the_draws_and_a_fresh_consumer_replays_them_bit_for_bit():
    draws = table(3_000)
    bridged = sb.prepare_distribution_bridge(contract(covariance=0.02)).derive(draws, seed=5)
    exported = bridged.artifact("draws-bridge")
    consumed = sb.consume_scientific_bridge_artifact(
        exported.data, exported.expectation.identity
    )
    assert isinstance(consumed, sb.ConsumedBridgeDistribution)
    assert consumed.claim.draws_digest == bridged.draws_digest
    assert consumed.claim.claim_id == bridged.claim_id
    np.testing.assert_array_equal(
        np.asarray(consumed.claim.distribution()), np.asarray(bridged.distribution())
    )
    # Another identity retained by the consumer, or a flipped byte, is refused.
    with pytest.raises(Exception):  # noqa: B017 - a typed refusal or a serialization error
        sb.consume_scientific_bridge_artifact(exported.data, "0" * 64)
    flipped = bytearray(exported.data)
    flipped[len(flipped) // 2] ^= 0xFF
    with pytest.raises(Exception):  # noqa: B017
        sb.consume_scientific_bridge_artifact(bytes(flipped))


def test_the_extended_declarations_ride_the_point_artifact_and_a_plain_one_is_unchanged():
    estimand = sb.EstimandBridge(
        "geo-to-panel",
        "geo-2024",
        sb.ExposureSetting("adults", "2024-01", "2024-04", 10.0, "usd", 12.0, "week", "do(t=1)"),
        sb.ExposureSetting("adults", "2024-06", "2024-09", 25.0, "usd", 8.0, "week", "do(t=1)"),
        recency_delta=8.0,
        recency_units="week",
        spend_delta=15.0,
        spend_units="usd",
    )
    declared = contract(
        value_kind="mean",
        covariance=0.001,
        estimand_bridge=estimand,
        mapping=sb.AffineMapping(intercept=0.06, slope=0.92),
    )
    bridge = sb.prepare_scientific_bridge(declared)
    claim = bridge.derive(sb.SourceClaim.declared("biomarker-effect", 0.8, SOURCE))
    assert claim.value == pytest.approx(0.06 + 0.92 * 0.8, abs=1e-12)
    assert claim.mapping_uncertainty == declared.mapping_uncertainty
    assert claim.source_dependence == declared.source_dependence
    assert claim.estimand_bridge == estimand
    exported = claim.artifact("uncertain-bridge")
    again = sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
    assert isinstance(again, sb.ConsumedBridge)
    assert again.claim.contract_identity == claim.contract_identity
    assert again.claim.mapping_uncertainty == declared.mapping_uncertainty
    assert again.claim.source_dependence == declared.source_dependence
    assert again.claim.estimand_bridge == estimand

    plain = sb.prepare_scientific_bridge(
        contract(uncertain=False, value_kind="mean", mapping=sb.AffineMapping(0.06, 0.92))
    ).derive(sb.SourceClaim.declared("biomarker-effect", 0.8, SOURCE))
    assert plain.mapping_uncertainty is None
    assert plain.source_dependence is None
    assert plain.estimand_bridge is None
    again = sb.consume_scientific_bridge_artifact(plain.export())
    assert isinstance(again, sb.ConsumedBridge)
