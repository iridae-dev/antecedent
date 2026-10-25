"""A bridge of a single number carries the propagated variance of an uncertain mapping.

Every asserted number has an oracle computed here in plain Python, never by native code:

* an affine map with a declared slope and intercept uncertainty has the first-order
  delta-method variance
  ``Va + x^2 Vb + b^2 V + 2 x b c + 2 x d + Vb V + c^2`` at the source value ``x`` and the squared
  standard error ``V`` of the source claim, and the mean ``a + b x + c``;
* the sign of the declared slope-source covariance ``c`` moves the variance and the mean;
* the expectation of a square of the propagated law is ``mean^2 + variance``;
* a declared uncertainty with no declared dependence is refused, never assumed independent;
* the artifact replays the propagation in a fresh consumption.

The propagated variance is an uncalibrated spread: an interval of the bridged value stays gated
by the measured-coverage rule and refuses here.
"""

from __future__ import annotations

import math

import pytest
from antecedent import scientific_bridge as sb
from antecedent.joint_distribution import ScientificQuantity
from antecedent.scientific_bridge import ScientificBridgeRefusal

from _refusal import assert_registered_refusal

INTERCEPT, SLOPE = 0.1, 0.6
VA, VB, D = 0.0004, 0.01, -0.0015


def outcome(variable):
    return ScientificQuantity.outcome(
        variable,
        units="probability",
        population="panel",
        regime="observational",
        functional="uptake",
    )


STATED = outcome("survey:intent")
REALIZED = outcome("behavior:uptake")


def contract(dependence):
    return sb.BridgeContract(
        kind="preference_realized_behavior",
        source=STATED,
        source_class="stated_preference",
        target=REALIZED,
        target_class="realized_behavior",
        evidence_scope=sb.Scope("panel", "observational"),
        mapping=sb.AffineMapping(intercept=INTERCEPT, slope=SLOPE),
        assumptions=(
            sb.Assumption(
                "linkage",
                "behavioral_linkage",
                "stated intent maps to uptake by the declared line",
                evidence="linkage-study:2023",
            ),
        ),
        mapping_uncertainty=sb.MappingUncertainty(
            "linkage-fit",
            intercept_variance=VA,
            slope_variance=VB,
            intercept_slope_covariance=D,
        ),
        source_dependence=dependence,
    )


def independent():
    return sb.SourceDependence.independent("disjoint-samples")


def derive(dependence, *, standard_error=0.05, nonlinear=None, value=0.5):
    bridge = sb.prepare_scientific_bridge(contract(dependence))
    source = sb.SourceClaim.declared(
        "intent", value, STATED, standard_error=standard_error, nonlinear=nonlinear
    )
    return bridge.derive(source)


def delta_variance(x, se, covariance=0.0):
    v = se * se
    return (
        VA
        + x * x * VB
        + SLOPE * SLOPE * v
        + 2.0 * x * SLOPE * covariance
        + 2.0 * x * D
        + VB * v
        + covariance * covariance
    )


def refused(call, code, detail):
    with pytest.raises(ScientificBridgeRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == code, (caught.value.reason_code, str(caught.value))
    assert caught.value.detail == detail, (caught.value.detail, str(caught.value))
    return caught.value


def test_an_affine_map_carries_the_hand_delta_variance_beside_the_plug_in_value():
    claim = derive(independent())
    propagation = claim.propagated_uncertainty()
    assert propagation.variance == pytest.approx(delta_variance(0.5, 0.05), rel=1e-12)
    assert claim.propagated_variance == propagation.variance
    assert claim.value == pytest.approx(INTERCEPT + SLOPE * 0.5)
    assert propagation.plug_in_value == claim.value
    assert propagation.mean == pytest.approx(claim.value)
    assert claim.propagated_mean == pytest.approx(claim.value)
    assert propagation.reading == "uncalibrated_spread"
    assert propagation.calibration == "unmeasured"
    assert propagation.standard_deviation == pytest.approx(math.sqrt(propagation.variance))
    assert propagation.dependence["kind"] == "independent"
    components = {c["target"]: c["omitted"] for c in claim.components}
    assert components["propagated_variance:uncalibrated_spread:calibration_unmeasured"] is False
    assert components["source_standard_error"] is False


def test_the_interval_stays_gated_by_the_measured_coverage_rule():
    claim = derive(independent())
    refused(
        claim.interval,
        "cell_not_licensed",
        "scientific_bridge.interval_not_licensed",
    )
    refused(
        lambda: claim.interval(0.95),
        "cell_not_licensed",
        "scientific_bridge.interval_scope_outside_measured",
    )


def test_the_sign_of_the_declared_covariance_moves_the_variance_and_the_mean():
    base = derive(independent())
    positive = derive(sb.SourceDependence.slope_covariance(0.0004, "shared-rows"))
    negative = derive(sb.SourceDependence.slope_covariance(-0.0004, "shared-rows"))
    assert positive.propagated_variance == pytest.approx(delta_variance(0.5, 0.05, 0.0004))
    assert negative.propagated_variance == pytest.approx(delta_variance(0.5, 0.05, -0.0004))
    assert positive.propagated_variance > base.propagated_variance > negative.propagated_variance
    assert positive.propagated_mean - base.propagated_mean == pytest.approx(0.0004)
    assert negative.propagated_mean - base.propagated_mean == pytest.approx(-0.0004)
    assert positive.value == base.value == negative.value


def test_a_square_reports_the_second_moment_expectation():
    claim = derive(independent(), nonlinear="square")
    propagation = claim.propagated_uncertainty()
    assert propagation.nonlinear["map"] == "square"
    assert propagation.nonlinear["map_of_mean"] == pytest.approx(propagation.mean**2)
    assert propagation.nonlinear["expectation"] == pytest.approx(
        propagation.mean**2 + propagation.variance
    )
    assert claim.propagated_mean == pytest.approx(propagation.nonlinear["expectation"])
    assert claim.value == pytest.approx(INTERCEPT + SLOPE * 0.5)


def test_an_undeclared_dependence_is_refused_not_assumed_independent():
    error = refused(
        lambda: derive(None),
        "sampling_dependence_unknown",
        "scientific_bridge.source_dependence_missing",
    )
    assert error.remedy
    # Without a standard error the plug-in derivation stands and the propagation is refused.
    claim = derive(None, standard_error=None)
    assert claim.propagated_variance is None
    refused(
        claim.propagated_uncertainty,
        "scientific_bridge_invalid",
        "scientific_bridge.source_variance_invalid",
    )


def test_the_artifact_replays_the_propagation_in_a_fresh_consumption():
    claim = derive(sb.SourceDependence.slope_covariance(0.0004, "shared-rows"), nonlinear="square")
    exported = claim.artifact("propagated-bridge")
    consumed = sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
    replayed = consumed.claim.propagated_uncertainty()
    held = claim.propagated_uncertainty()
    assert replayed.variance == held.variance
    assert replayed.mean == held.mean
    assert replayed.nonlinear == held.nonlinear
    assert consumed.claim.claim_id == claim.claim_id
    other = derive(
        sb.SourceDependence.slope_covariance(0.0004, "shared-rows"),
        standard_error=0.06,
        nonlinear="square",
    )
    assert other.claim_id != claim.claim_id
    assert (
        other.artifact("propagated-bridge").expectation.identity != exported.expectation.identity
    )
