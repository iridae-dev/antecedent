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

import subprocess
import sys
from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import measurement_models as mm
from antecedent import propagation_chain as pc
from antecedent import scientific_bridge as sb
from antecedent.errors import CausalTypeError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success
from antecedent.scientific_bridge import ScientificBridgeRefusal

from _refusal import assert_registered_refusal

R = pc.PropagationChainRefusal
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


def table(rows=ROWS, *, semantic="causal_functional_posterior", alignment="joint", source_mean=1.0):
    source = grid(source_mean, 0.3, rows)
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
    exact = require_success(bridge.derive(draws, seed=3))
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
    assert exact.calibration == "not_measured"
    assert exact.value_kind == "posterior"
    np.testing.assert_array_equal(np.asarray(exact.source_distribution()), np.asarray(draws))

    # An uncertain slope and intercept: the delta method and its dependence.
    independent = require_success(sb.prepare_distribution_bridge(contract()).derive(draws, seed=3))
    mapped = np.asarray(independent.distribution())[:, 0]
    mean, variance = delta(source)
    assert mapped.mean() == pytest.approx(mean, abs=0.02)
    assert mapped.var() == pytest.approx(variance, rel=0.06)
    assert independent.variance == pytest.approx(mapped.var(), rel=1e-9)

    covariance = 0.02
    dependent = require_success(
        sb.prepare_distribution_bridge(contract(covariance=covariance)).derive(draws, seed=3)
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
    again = require_success(sb.prepare_distribution_bridge(contract()).derive(draws, seed=3))
    other = require_success(sb.prepare_distribution_bridge(contract()).derive(draws, seed=4))
    assert again.draws_digest == independent.draws_digest
    assert other.draws_digest != independent.draws_digest
    assert other.claim_id != independent.claim_id


def test_the_refusals_of_a_bridge_of_draws_name_what_would_support_them():
    draws = table(2_000)
    bridge = sb.prepare_distribution_bridge(contract())
    # An associational claim is never a bridge source.
    refused(
        lambda: require_success(bridge.derive(draws, seed=1, source_kind="associational")),
        "bridge_mapping_not_supported",
        reason_details.scientific_bridge.SOURCE_NOT_DRAWS,
    )
    # A bootstrap is neither a posterior nor a predictive law.
    refused(
        lambda: require_success(bridge.derive(table(2_000, semantic="bootstrap"), seed=1)),
        "distribution_meaning_mismatch",
        reason_details.scientific_bridge.DRAWS_MEANING_MISMATCH,
    )
    # Independent marginals have no row pairing to keep.
    refused(
        lambda: require_success(
            bridge.derive(table(2_000, alignment="independent_marginals"), seed=1)
        ),
        "joint_law_required",
        reason_details.scientific_bridge.DRAWS_NOT_ALIGNED,
    )
    # An uncertain map declares how it depends on the source.
    refused(
        lambda: sb.prepare_distribution_bridge(replace(contract(), source_dependence=None)),
        "sampling_dependence_unknown",
        reason_details.scientific_bridge.SOURCE_DEPENDENCE_MISSING,
    )
    # A bridged mean is the point route's; a distribution is a posterior, never a mean.
    result = require_success(bridge.derive(draws, seed=1))
    refused(
        lambda: result.require_value_kind("mean"),
        "bridge_substitution_refused",
        reason_details.scientific_bridge.VALUE_KIND_SUBSTITUTION,
    )
    result.require_value_kind("posterior")
    refused(
        lambda: require_success(result.interval()),
        "route_not_supported",
        reason_details.scientific_bridge.INTERVAL_NOT_SUPPORTED,
    )
    # An interval needs a measured record; an uncertain map has none.
    refused(
        lambda: require_success(result.interval(0.95)),
        "route_not_supported",
        reason_details.scientific_bridge.INTERVAL_SCOPE_OUTSIDE_MEASURED,
    )
    with pytest.raises(CausalTypeError):
        require_success(bridge.derive(np.zeros((3, 2)), seed=1))


def test_an_exact_map_publishes_its_interval_as_an_exact_transform():
    draws = table(4_000)
    exact = require_success(
        sb.prepare_distribution_bridge(contract(uncertain=False)).derive(draws, seed=2)
    )
    interval = require_success(exact.interval(0.95))
    sd = np.asarray(draws)[:, 0].std()
    assert interval.supported_by == "exact_transform_of_declared_source_interval"
    assert interval.source_standard_error == pytest.approx(sd, rel=1e-9)
    z = _NORMAL.inv_cdf(0.975)
    assert interval.upper - interval.lower == pytest.approx(2 * z * 2.0 * sd, rel=1e-9)


def test_a_bridge_is_a_stage_of_a_chain_that_propagates_the_mapping_and_the_source_jointly():
    draws = table(4_001)
    bridged = require_success(
        sb.prepare_distribution_bridge(contract(covariance=0.02)).derive(draws, seed=9)
    )
    parts = bridged.chain_stage("terminal")
    assert [source.id for source in parts.sources] == [
        "terminal.source",
        "terminal.slope",
        "terminal.intercept",
    ]
    assert parts.stage.kind == "measurement_mapping"
    assert parts.stage.op["scale"] == {"kind": "input", "position": 1}
    assert parts.stage.op["shift"] == {"kind": "input", "position": 2}
    chain = require_success(
        pc.propagate(
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
    )
    mapped = np.asarray(bridged.distribution())[:, 0]
    assert chain.stage("terminal").value("mean") == pytest.approx(mapped.mean(), abs=1e-9)
    expectation = (1.0 - np.exp(-0.5 * (0.9 * mapped))).mean()
    assert chain.stage("utility").value("mean") == pytest.approx(expectation, abs=1e-9)
    assert abs(expectation - (1.0 - np.exp(-0.5 * 0.9 * mapped.mean()))) > 1e-4


def test_the_artifact_carries_the_draws_and_a_fresh_consumer_replays_them_bit_for_bit():
    draws = table(3_000)
    bridged = require_success(
        sb.prepare_distribution_bridge(contract(covariance=0.02)).derive(draws, seed=5)
    )
    exported = bridged.export("draws-bridge")
    consumed = require_success(
        sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
    )
    assert isinstance(consumed, sb.ConsumedBridgeDistribution)
    assert consumed.claim.draws_digest == bridged.draws_digest
    assert consumed.claim.claim_id == bridged.claim_id
    np.testing.assert_array_equal(
        np.asarray(consumed.claim.distribution()), np.asarray(bridged.distribution())
    )
    # Another identity retained by the consumer, or a flipped byte, is refused.
    with pytest.raises(ScientificBridgeRefusal):
        require_success(sb.consume_scientific_bridge_artifact(exported.data, "0" * 64))
    flipped = bytearray(exported.data)
    flipped[len(flipped) // 2] ^= 0xFF
    with pytest.raises(ScientificBridgeRefusal):
        require_success(sb.consume_scientific_bridge_artifact(bytes(flipped)))


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
    claim = require_success(bridge.derive(sb.SourceClaim.declared("biomarker-effect", 0.8, SOURCE)))
    assert claim.value == pytest.approx(0.06 + 0.92 * 0.8, abs=1e-12)
    assert claim.mapping_uncertainty == declared.mapping_uncertainty
    assert claim.source_dependence == declared.source_dependence
    assert claim.estimand_bridge == estimand
    exported = claim.export("uncertain-bridge")
    again = require_success(
        sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
    )
    assert isinstance(again, sb.ConsumedBridge)
    assert again.claim.contract_identity == claim.contract_identity
    assert again.claim.mapping_uncertainty == declared.mapping_uncertainty
    assert again.claim.source_dependence == declared.source_dependence
    assert again.claim.estimand_bridge == estimand

    plain = require_success(
        sb.prepare_scientific_bridge(
            contract(uncertain=False, value_kind="mean", mapping=sb.AffineMapping(0.06, 0.92))
        ).derive(sb.SourceClaim.declared("biomarker-effect", 0.8, SOURCE))
    )
    assert plain.mapping_uncertainty is None
    assert plain.source_dependence is None
    assert plain.estimand_bridge is None
    again = require_success(sb.consume_scientific_bridge_artifact(bytes(plain.export())))
    assert isinstance(again, sb.ConsumedBridge)


TRIALS = [(0.0, 0.1), (1.0, 1.1), (2.0, 1.9), (3.0, 3.2), (4.0, 3.9)]


def test_a_predictive_line_is_a_stage_whose_residual_variance_is_added_once():
    bridge = contract(
        uncertain=False,
        value_kind="predictive_law",
        mapping=sb.TrialLevelMapping(TRIALS, min_trials=5, min_r_squared=0.9),
        source_dependence=sb.SourceDependence.independent("disjoint-samples"),
    )
    # Every source draw must lie inside the validated surrogate range of the trial-level line
    # (0 to 4): the draws are centered so their tails stay inside it.
    draws = table(4_001, semantic="posterior_predictive", source_mean=1.5)
    bridged = require_success(sb.prepare_distribution_bridge(bridge).derive(draws, seed=5))
    parts = bridged.chain_stage("line")
    assert [source.id for source in parts.sources] == [
        "line.source",
        "line.slope",
        "line.intercept",
        "line.residual",
    ]
    assert parts.stage.op["op"] == "expression"
    # The residual declares a disturbance of its own and no other input does.
    assert any(kind == "disturbance" for kind, _ in parts.sources[3].factors)
    assert all(kind != "disturbance" for source in parts.sources[:3] for kind, _ in source.factors)
    # Predictive source draws and an innovation residual are published only as a predictive law
    # of the bridge's target coordinate.
    assert parts.reading == "predictive"
    assert parts.output_quantity == bridged.quantity
    chain = require_success(
        pc.propagate(
            list(parts.sources),
            [parts.stage],
            output="line",
            reading=parts.reading,
            output_quantity=parts.output_quantity,
        )
    )
    mapped = np.asarray(bridged.distribution())[:, 0]
    assert chain.stage("line").value("mean") == pytest.approx(mapped.mean(), abs=1e-12)
    assert chain.stage("line").value("variance") == pytest.approx(mapped.var(ddof=1), rel=1e-12)
    # The mean line of the same coefficients, without the residual column.
    mean_line = require_success(
        pc.propagate(
            list(parts.sources[:3]),
            [
                pc.affine(
                    "mean_line",
                    "measurement_mapping",
                    ["line.source", "line.slope", "line.intercept"],
                    scale=pc.input_coefficient(1),
                    shift=pc.input_coefficient(2),
                )
            ],
            output="mean_line",
            reading=parts.reading,
            output_quantity=parts.output_quantity,
        )
    )
    residual = np.asarray(parts.sources[3].values)
    line = np.asarray(parts.sources[0].values) * np.asarray(parts.sources[1].values) + np.asarray(
        parts.sources[2].values
    )
    covariance = np.cov(line, residual, ddof=1)[0, 1]
    expected = (
        mean_line.stage("mean_line").value("variance") + residual.var(ddof=1) + 2 * covariance
    )
    assert chain.stage("line").value("variance") == pytest.approx(expected, rel=1e-9)
    assert residual.var(ddof=1) > 0.0
    # The consumer replays the stage to the stored bits.
    consumed = require_success(
        pc.consume_propagation_chain(bytes(chain.export()), expected_identity=chain.identity)
    )
    assert dict(consumed.stage("line").values) == dict(chain.stage("line").values)


def _clinic_correction():
    model = mm.MeasurementModel(
        indicator=mm.MeasurementVariable("schema:flag", "noisy_misclassified", "binary"),
        construct=mm.MeasurementVariable("construct:disease", "latent_scientific", "binary"),
        parameters=mm.BinaryParameters((mm.BinaryRates(0.9, 0.8, 0.0004, 0.0009),), min_youden=0.1),
        population_id="clinic-a",
        regime_id="observational",
        evidence=mm.ParameterEvidence.declared("lab"),
    )

    def prevalence(variable):
        return ScientificQuantity.outcome(
            variable,
            units="probability",
            population="clinic-a",
            regime="observational",
            functional="prevalence",
        )

    flag = prevalence("schema:flag")
    bridge = sb.BridgeContract(
        kind="latent_measured_outcome",
        source=flag,
        source_class="measured_outcome",
        target=prevalence("construct:disease"),
        target_class="latent_construct",
        evidence_scope=sb.Scope("clinic-a", "observational"),
        mapping=sb.MeasurementCorrection(model),
        assumptions=(
            sb.Assumption(
                "flag-model",
                "measurement_model",
                "the flag relates to the disease by the declared misclassification model",
                evidence=mm.model_identities(model).model,
            ),
        ),
        source_value="posterior",
        target_value="posterior",
        source_dependence=sb.SourceDependence.independent("validation-sample-disjoint"),
    )
    rows = 4_000
    partner = grid(0.2, 0.3, rows)[(np.arange(rows) * 1_247) % rows]
    identity = DistributionIdentity(
        semantic="causal_functional_posterior",
        quantities=(flag, COST),
        alignment="joint",
        source_id="study-1",
        provider_id="posterior.provider",
        rng_id="seed:11",
        snapshot_id="snapshot-1",
        causal_contract_id="contract-1",
    )
    draws = JointDistributionArtifact(identity, np.column_stack([grid(0.31, 0.01, rows), partner]))
    return bridge, draws


def test_a_measurement_correction_is_a_ratio_stage_and_a_zero_denominator_is_refused():
    bridge, draws = _clinic_correction()
    bridged = require_success(sb.prepare_distribution_bridge(bridge).derive(draws, seed=13))
    parts = bridged.chain_stage("correction")
    assert [source.id for source in parts.sources] == [
        "correction.source",
        "correction.sensitivity",
        "correction.specificity",
    ]
    assert parts.stage.op["op"] == "expression"
    chain = require_success(pc.propagate(list(parts.sources), [parts.stage], output="correction"))
    corrected = np.asarray(bridged.distribution())[:, 0]
    # The same inputs give the same rows: the stage is the bridged draws, to the last bit.
    assert chain.stage("correction").value("mean") == pytest.approx(corrected.mean(), rel=1e-12)
    assert chain.stage("correction").value("variance") == pytest.approx(
        corrected.var(ddof=1), rel=1e-12
    )
    observed, sensitivity, specificity = (np.asarray(source.values) for source in parts.sources)
    np.testing.assert_array_equal(
        corrected, (observed + specificity - 1.0) / (sensitivity + specificity - 1.0)
    )
    # 0.31 recorded at sensitivity 0.9 and specificity 0.8 is 11/70 latent.
    assert chain.stage("correction").value("mean") == pytest.approx(11 / 70, abs=4e-3)

    # A sensitivity that leaves the Youden index at or below zero on some rows is not a ratio.
    crossing = np.where(np.arange(len(sensitivity)) % 7 == 0, 0.1, sensitivity)
    sources = [
        parts.sources[0],
        pc.draws(
            "correction.sensitivity",
            crossing,
            meaning="posterior",
            alignment=parts.sources[1].alignment,
        ),
        parts.sources[2],
    ]
    with pytest.raises(R) as caught:
        require_success(pc.propagate(sources, [parts.stage], output="correction"))
    assert caught.value.detail == reason_details.propagation_chain.DENOMINATOR_NOT_BOUNDED
    assert caught.value.reason_code == "invalid_argument"


def test_a_guarded_division_refuses_a_denominator_that_reaches_zero_and_replays():
    alignment = "ratio-table"
    numerator = pc.draws("a", grid(2.0, 0.1), meaning="posterior", alignment=alignment)
    positive = pc.draws("b", grid(4.0, 0.1), meaning="posterior", alignment=alignment)
    chain = require_success(
        pc.propagate(
            [numerator, positive],
            [pc.divide("ratio", "derived_quantity", ["a", "b"])],
            output="ratio",
        )
    )
    expected = grid(2.0, 0.1) / grid(4.0, 0.1)
    assert chain.stage("ratio").value("mean") == pytest.approx(expected.mean(), rel=1e-12)
    replayed = require_success(
        pc.consume_propagation_chain(bytes(chain.export()), expected_identity=chain.identity)
    )
    assert dict(replayed.answer.values) == dict(chain.answer.values)
    # A denominator whose draws cross zero is refused, never a silent ratio.
    crossing = pc.draws("b", grid(0.0, 1.0), meaning="posterior", alignment=alignment)
    with pytest.raises(R) as caught:
        require_success(
            pc.propagate(
                [numerator, crossing],
                [pc.divide("ratio", "derived_quantity", ["a", "b"])],
                output="ratio",
            )
        )
    assert caught.value.detail == reason_details.propagation_chain.DENOMINATOR_NOT_BOUNDED
    # A declared bound refuses a denominator that comes closer to zero than the model supports.
    near_zero = pc.draws("b", grid(0.5, 0.05), meaning="posterior", alignment=alignment)
    with pytest.raises(R):
        require_success(
            pc.propagate(
                [numerator, near_zero],
                [pc.divide("ratio", "derived_quantity", ["a", "b"], min_magnitude=0.6)],
                output="ratio",
            )
        )


@pytest.mark.parametrize("meaning", ["estimator_sampling", "bootstrap"])
def test_fixed_sampling_bridge_replays_mapped_replicates_and_preserves_law(meaning):
    source = table(128, semantic=meaning)
    prepared = sb.prepare_distribution_bridge(contract(uncertain=False, value_kind=meaning))
    result = require_success(prepared.derive(source, seed=7))
    exported = result.export("fixed-sampling-bridge")
    from antecedent import load

    replay = require_success(load(exported.data, expected=exported.expectation)).claim
    direct = require_success(
        sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
    ).claim
    assert replay.claim_id == direct.claim_id
    assert replay.value_kind == meaning
    mapped = np.asarray(replay.distribution())
    np.testing.assert_array_equal(mapped[:, 0], 0.5 + 2.0 * np.asarray(source)[:, 0])
    np.testing.assert_array_equal(mapped[:, 1], np.asarray(source)[:, 1])
    assert replay.chain_stage("terminal").reading == "spread"
    refused(
        lambda: require_success(replay.interval(0.95)),
        "route_not_supported",
        reason_details.scientific_bridge.INTERVAL_SCOPE_OUTSIDE_MEASURED,
    )


def test_trial_bootstrap_replays_source_and_map_on_the_same_trials(tmp_path):
    from antecedent import bridge_transport_decision as btd
    from antecedent import composition_bundle as cb
    from antecedent.execution import Execution

    trials = [(float(i), i + i**2 / 48.0) for i in range(24)]
    declaration = contract(
        uncertain=False,
        value_kind="bootstrap",
        mapping=sb.TrialLevelMapping(trials, min_trials=5, min_r_squared=0.0),
        source_dependence=sb.SourceDependence.shared_trial_mean("meta-analysis:ckd-2024"),
    )
    prepared = sb.prepare_distribution_bridge(declaration)
    source = require_success(prepared.shared_trial_bootstrap_source(199, seed=515))
    compiled = prepared.compile_sampling(source, seed=515)
    assert compiled.plan_view()["route"] == "SamplingDistributionBridge"
    assert compiled.unit_support()["unit"] == "scientific_bridge.distribution_bridges"
    bridged = require_success(compiled.execute())
    assert require_success(prepared.derive(source, seed=515)).claim_id == bridged.claim_id
    parts = bridged.chain_stage("fit")
    squared = require_success(
        pc.propagate(
            list(parts.sources),
            [parts.stage, pc.unary("squared_effect", "derived_quantity", ["fit"], "square")],
            output="squared_effect",
        )
    )
    reference = np.square(np.asarray(bridged.distribution())[:, 0])
    assert squared.stage("squared_effect").value("variance") == pytest.approx(reference.var(ddof=1))
    terminal = bridged.export("refitted-trial-terminal")
    bundle = (
        cb.Bundle.builder()
        .add_artifact("distribution", bytes(source.export("trial-source")), node_id="source")
        .add_artifact("scientific_bridge", bytes(terminal), node_id="terminal")
        .connect("source", "terminal")
        .build()
    )
    plan = btd.compile_bridge_bundle_recomputation(
        bundle, "terminal", declaration, source, stream=515
    )
    execution = Execution(seed=515)
    receipt = plan.execute(execution=execution)
    assert receipt.terminal_identity == terminal.expectation.identity
    replay = btd.consume_bridge_bundle_recomputation(
        receipt.export(),
        expected_bundle=bundle.identity,
        expected_identity=plan.program_identity,
        execution=execution,
    )
    assert replay == receipt
    path = tmp_path / "trial-bootstrap-bundle"
    path.write_bytes(bytes(receipt.export()))
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from antecedent.bridge_transport_decision import consume_bridge_bundle_recomputation; "
            "from antecedent.execution import Execution; "
            "r = consume_bridge_bundle_recomputation(Path(sys.argv[1]).read_bytes(), "
            "expected_bundle=sys.argv[2], expected_identity=sys.argv[3], execution=Execution(seed=515)); "
            "print(r.terminal_identity)",
            str(path),
            bundle.identity,
            plan.program_identity,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert loaded.stdout.strip() == terminal.expectation.identity
    with pytest.raises(ScientificBridgeRefusal):
        require_success(prepared.derive(source, seed=514))


def test_identified_image_bounds_flow_into_the_decision_and_replay(tmp_path):
    from antecedent import decision, decision_robust

    source = sb.IdentifiedSetSource(SOURCE, -2.0, 3.0, "ab" * 32)
    declaration = contract(
        uncertain=False, value_kind="identified_set", mapping=sb.AffineMapping(1.0, -2.0)
    )
    plan = sb.compile_identified_set_image(declaration, source, domain=(-2.0, 4.0))
    result = require_success(plan.execute())
    assert (result.quantity, result.lower, result.upper) == (TARGET, -5.0, 5.0)
    contract_for_decision = decision_robust.admissible_contract(
        decision.Contract(
            actions=(
                decision.Action("act", inputs=(TARGET,), utility=decision.x(0) + 10.0),
                decision.Action("wait", inputs=(TARGET,), utility=decision.x(0) * 0.0),
            ),
            utility_units="utility",
            criterion=decision.Criterion.expected_utility(),
            target_population="adults",
            structural_policy="require_invariant_best_action",
        )
    )
    utilities = [
        decision_robust.IdentifiedUtility.from_inputs(contract_for_decision, name, [result])
        for name in ("act", "wait")
    ]
    assert (utilities[0].lower, utilities[0].upper) == (5.0, 15.0)
    assert (
        require_success(decision_robust.identified_sets(contract_for_decision, utilities)).selected
        == "act"
    )
    artifact = tmp_path / "sharp-set.bin"
    from antecedent import load

    exported = plan.export()
    assert exported.expectation is not None
    restored = require_success(load(exported, expected=exported.expectation))
    assert (restored.identity, restored.lower, restored.upper) == (result.identity, -5.0, 5.0)

    from antecedent import CancellationToken, Execution, ResourceLimits
    from antecedent.errors import CausalCancelledError, CausalResourceError

    with pytest.raises(CausalResourceError):
        require_success(
            load(
                exported,
                expected=exported.expectation,
                execution=Execution(limits=ResourceLimits(memory_bytes=1)),
            )
        )
    cancellation = CancellationToken()
    cancellation.cancel()
    with pytest.raises(CausalCancelledError):
        require_success(
            load(exported, expected=exported.expectation, execution=Execution(cancel=cancellation))
        )
    artifact.write_bytes(bytes(exported))
    script = """
import pathlib, sys
from antecedent import scientific_bridge as sb
from antecedent.outcomes import require_success
result = require_success(sb.consume_bridge_image(pathlib.Path(sys.argv[1]).read_bytes(),
                         expected_program=sys.argv[2], expected_result=sys.argv[3]))
print(result.lower, result.upper)
"""
    done = subprocess.run(
        [sys.executable, "-c", script, str(artifact), plan.program_identity, result.identity],
        check=True,
        text=True,
        capture_output=True,
    )
    assert done.stdout.strip() == "-5.0 5.0"
    with pytest.raises(ScientificBridgeRefusal):
        require_success(
            sb.compile_identified_set_image(declaration, source, domain=(-1.0, 4.0)).execute()
        )


def test_joint_outcome_image_consumes_paired_weighted_rows_and_feeds_affine_utility():
    from antecedent import decision

    source = JointDistributionArtifact(
        replace(table(3).identity, semantic="empirical_outcome"),
        np.array([[1.0, 3.0], [2.0, 2.0], [4.0, 1.0]]),
        weights=(2.0, 1.0, 3.0),
    )
    second = outcome("trial:difference", units="log_hazard")

    def term(q, target, a, b):
        return sb.OutcomeMapTerm(
            contract(
                uncertain=False,
                value_kind="joint_outcome_law",
                source=q,
                target=target,
                mapping=sb.AffineMapping(a, b),
            ),
            (0.0, 5.0),
        )

    outputs = [
        [term(SOURCE, TARGET, 1.0, 1.0), term(COST, TARGET, 0.0, 2.0)],
        [term(SOURCE, second, 0.0, 1.0), term(COST, second, 0.0, -1.0)],
    ]
    plan = sb.compile_joint_outcome_image(source, outputs)
    observed = require_success(plan.execute())
    np.testing.assert_array_equal(
        np.asarray(observed.distribution), [[8.0, -2.0], [7.0, 0.0], [7.0, 3.0]]
    )
    contract_for_decision = decision.Contract(
        actions=(
            decision.Action("act", inputs=(second,), utility=decision.x(0)),
            decision.Action("wait", inputs=(second,), utility=decision.x(0) * 0.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="adults",
    )
    result = require_success(contract_for_decision.evaluate(observed))
    assert result.selected == ("act",)
    assert result.actions[0].expected_utility == pytest.approx(5 / 6)
    raw = bytes(plan.export())
    replayed = require_success(
        sb.consume_bridge_image(
            raw, expected_program=plan.program_identity, expected_result=observed.identity
        )
    )
    np.testing.assert_array_equal(
        np.asarray(replayed.distribution), np.asarray(observed.distribution)
    )
    from antecedent import composition_bundle as cb

    bundle = (
        cb.Bundle.builder()
        .add_artifact(source, node_id="source")
        .add_artifact(raw, node_id="image")
        .connect("source", "image")
        .build()
    )
    report = require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    )
    assert report.all_replayed
    assert report.value("image", "n_draws") == 3
    changed = JointDistributionArtifact(source.identity, np.asarray(source) + 0.125)
    changed_bundle = (
        cb.Bundle.builder()
        .add_artifact(changed, node_id="source")
        .add_artifact(raw, node_id="image")
        .connect("source", "image")
        .build()
    )
    with pytest.raises(cb.CompositionBundleRefusal):
        require_success(
            cb.consume_bundle(
                bytes(changed_bundle.export()), expected_identity=changed_bundle.identity
            )
        )
    unpaired = JointDistributionArtifact(
        replace(source.identity, alignment="independent_marginals"), np.asarray(source)
    )
    with pytest.raises(ScientificBridgeRefusal):
        require_success(sb.compile_joint_outcome_image(unpaired, outputs).execute())


def test_joint_posterior_bridge_reports_attained_mass_and_replays_the_threshold(tmp_path):
    from antecedent import composition_bundle as cb

    offset = outcome("map:intercept_shift", units=TARGET.units)
    slope = outcome("map:slope_shift", units=f"{TARGET.units}/{SOURCE.units}")
    joint = JointDistributionArtifact(
        replace(table(4).identity, quantities=(SOURCE, offset, slope)),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 0.0, -1.0], [3.0, 0.0, 3.0]]),
        weights=(1.0, 2.0, 3.0, 2.0),
    )
    declaration = contract(
        uncertain=False, value_kind="posterior", mapping=sb.AffineMapping(0.0, 1.0)
    )
    plan = sb.compile_posterior_bridge_interval(
        declaration, joint, intercept_shift=offset, slope_shift=slope, domain=(0.0, 3.0), level=0.5
    )
    interval = require_success(plan.execute())
    assert isinstance(interval, sb.PosteriorBridgeInterval)
    assert (interval.lower, interval.upper, interval.level, interval.attained_mass) == (
        0.0,
        4.0,
        0.5,
        0.75,
    )
    assert interval.quantity == TARGET
    # Equality at either endpoint leaves the direction unresolved. The posterior atoms assign
    # weight to both 0 and 4, so treating an endpoint as a strict preference would change it.
    for cutoff, verdict in [
        (-1.0, "above"),
        (0.0, "unresolved"),
        (4.0, "unresolved"),
        (4.5, "below"),
    ]:
        decision = require_success(plan.compile_threshold(TARGET, cutoff).execute())
        assert (decision.verdict, decision.reading, decision.attained_mass) == (
            verdict,
            "posterior_credible",
            0.75,
        )
        assert decision.interval_identity == interval.identity
        assert decision.decision_calibration == "not_measured"
        assert decision.interval_coverage_record is None
    decision_plan = plan.compile_threshold(TARGET, 0.0)
    result = require_success(decision_plan.execute())
    raw = bytes(decision_plan.export())
    file = tmp_path / "interval-decision.bin"
    file.write_bytes(raw)
    script = """
import pathlib, sys
from antecedent import scientific_bridge as sb
from antecedent.outcomes import require_success
result=require_success(sb.consume_bridge_image(pathlib.Path(sys.argv[1]).read_bytes(),expected_program=sys.argv[2],expected_result=sys.argv[3]))
print(result.reading,result.verdict,result.decision_calibration)
"""
    done = subprocess.run(
        [sys.executable, "-c", script, str(file), decision_plan.program_identity, result.identity],
        check=True,
        text=True,
        capture_output=True,
    )
    assert done.stdout.strip() == "posterior_credible unresolved not_measured"
    bundle = (
        cb.Bundle.builder()
        .add_artifact(plan, node_id="interval")
        .add_artifact(raw, node_id="decision")
        .connect("interval", "decision")
        .build()
    )
    report = require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    )
    assert report.all_replayed
    with pytest.raises(ScientificBridgeRefusal):
        require_success(plan.compile_threshold(COST, 0.0).execute())


@pytest.mark.parametrize("shared", [True, False], ids=["shared-trials", "independent-studies"])
def test_measured_trial_bridge_keeps_interval_and_decision_calibration_separate(shared):
    from antecedent import composition_bundle as cb

    # Balanced trial study on the interior of the validation range, matching the calibrated
    # refit construction. The external source and shared validation means are different.
    rng = np.random.default_rng(276)
    x = rng.uniform(-1, 1, size=80)
    y = 0.1 + 0.9 * x + 0.35 * rng.normal(size=80)
    trials = tuple(zip(x.tolist(), y.tolist(), strict=True))
    if shared:
        dependence = sb.SourceDependence.shared_trial_mean("meta-analysis:ckd-2024")
        study = sb.SharedValidationTrials()
    else:
        dependence = sb.SourceDependence.independent("independent-validation-and-source")
        study = sb.IndependentTrialStudy(
            "external-source", tuple(np.random.default_rng(277).uniform(-0.25, 0.75, size=80))
        )
    declaration = contract(
        uncertain=False,
        value_kind="bootstrap",
        mapping=sb.TrialLevelMapping(trials, min_trials=3, min_r_squared=0.0),
        source_dependence=dependence,
    )
    plan = sb.compile_trial_bridge_confidence_interval(
        declaration,
        study,
        replicates=399,
        level=0.95,
        seed=167,
        release_design="shared_uniform_gaussian_v1" if shared else "independent_shifted_uniform_v1",
    )
    interval = require_success(plan.execute())
    assert isinstance(interval, sb.TrialBridgeConfidenceInterval)
    expected = np.quantile(np.asarray(interval.bootstrap)[:, 0], [0.025, 0.975], method="linear")
    np.testing.assert_allclose([interval.lower, interval.upper], expected, rtol=1e-12)
    assert interval.quantity == TARGET and interval.coverage_record and interval.calibration_sha
    terminal = plan.compile_threshold(TARGET, interval.upper)
    decision = require_success(terminal.execute())
    assert (decision.verdict, decision.reading, decision.interval_identity) == (
        "unresolved",
        "sampling_confidence",
        interval.identity,
    )
    assert decision.interval_coverage_record == interval.coverage_record
    assert decision.interval_calibration_sha == interval.calibration_sha
    assert decision.decision_calibration == "not_measured"
    bundle = (
        cb.Bundle.builder()
        .add_artifact(plan, node_id="interval")
        .add_artifact(terminal, node_id="decision")
        .connect("interval", "decision")
        .build()
    )
    assert require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    ).all_replayed
    off_design = sb.compile_trial_bridge_confidence_interval(
        declaration, study, replicates=399, level=0.8, seed=167
    )
    with pytest.raises(ScientificBridgeRefusal):
        require_success(off_design.execute())
