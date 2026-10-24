//! A bridge of a single number whose mapping declares its uncertainty publishes the
//! propagated variance beside the plug-in value: hand delta formulas for an affine map, a
//! trial-level line and a correction, the sign of the declared covariance, the second-moment
//! expectation of a nonlinear map, the refusal of an undeclared dependence and the bit-for-bit
//! replay of the artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, MappingUncertainty, NonlinearMap, QuantityClass,
    SourceMappingDependence, TrialEffectPair, TrialLevelValidation, ValueKind,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{ContractIdentities, SemanticDigest};

use super::{
    PreparedScientificBridge, ScientificBridgeRefusal, ScientificBridgeResult,
    consume_scientific_bridge_artifact,
};

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * (1.0 + left.abs().max(right.abs()))
}

fn quantity(variable: &str, population: &str, regime: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "uptake".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn claim(value: f64) -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(1),
        ContractIdentities::new(
            digest(10),
            digest(11),
            Some(digest(12)),
            Some(digest(13)),
            digest(14),
            digest(15),
            digest(16),
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from("probability")),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        None,
        [Arc::<str>::from("prior:p0")],
    )
}

fn independent() -> SourceMappingDependence {
    SourceMappingDependence::Independent { evidence: "disjoint-samples".into() }
}

fn affine_contract(dependence: Option<SourceMappingDependence>) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::PreferenceRealizedBehavior,
        source: BridgeEndpoint {
            quantity: quantity("survey:intent", "panel", "observational", "probability"),
            class: QuantityClass::StatedPreference,
        },
        target: BridgeEndpoint {
            quantity: quantity("behavior:uptake", "panel", "observational", "probability"),
            class: QuantityClass::RealizedBehavior,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: "panel".into(),
            regime_id: "observational".into(),
        },
        mapping: BridgeMapping::Affine { intercept: 0.1, slope: 0.6 },
        assumptions: vec![BridgeAssumption {
            id: "linkage".into(),
            kind: AssumptionKind::BehavioralLinkage,
            statement: "stated intent maps to uptake by the declared line".into(),
            evidence: Some("linkage-study:2023".into()),
        }],
        transport: None,
        mapping_uncertainty: Some(MappingUncertainty {
            id: "linkage-fit".into(),
            intercept_variance: 0.0004,
            slope_variance: 0.01,
            intercept_slope_covariance: -0.0015,
        }),
        source_dependence: dependence,
        estimand_bridge: None,
    }
}

fn surrogate_contract(dependence: Option<SourceMappingDependence>) -> BridgeContract {
    let trials = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5), (0.8, 0.7)]
        .into_iter()
        .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
            surrogate_effect,
            terminal_effect,
        })
        .collect();
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity("trial:biomarker", "adults", "do(t=1)", "probability"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity("trial:survival", "adults", "do(t=1)", "log_hazard"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials,
            min_trials: 4,
            min_r_squared: 0.8,
        }),
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: dependence,
        estimand_bridge: None,
    }
}

fn correction_contract(dependence: Option<SourceMappingDependence>) -> BridgeContract {
    let model = MeasurementModel {
        model_id: "screening-flag".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "schema:flag".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "binary".into(),
        },
        construct: MeasurementVariable {
            id: "construct:disease".into(),
            kind: MeasurementKind::LatentScientific,
            units: "binary".into(),
        },
        binding: MeasurementBinding {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.9,
                specificity: 0.8,
                sensitivity_variance: 0.0004,
                specificity_variance: 0.0004,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::ValidationSample {
            sample_id: "validation-sample".into(),
            units: 400,
        },
    };
    BridgeContract {
        kind: BridgeKind::LatentMeasuredOutcome,
        source: BridgeEndpoint {
            quantity: quantity("schema:flag", "clinic-a", "observational", "probability"),
            class: QuantityClass::MeasuredOutcome,
        },
        target: BridgeEndpoint {
            quantity: quantity("construct:disease", "clinic-a", "observational", "probability"),
            class: QuantityClass::LatentConstruct,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
        },
        assumptions: vec![BridgeAssumption::for_measurement_model("flag-model", &model)],
        mapping: BridgeMapping::MeasurementCorrection(Box::new(model)),
        transport: None,
        mapping_uncertainty: None,
        source_dependence: dependence,
        estimand_bridge: None,
    }
}

fn derive(
    contract: &BridgeContract,
    value: f64,
    standard_error: Option<f64>,
    nonlinear: Option<NonlinearMap>,
) -> Result<ScientificBridgeResult, ScientificBridgeRefusal> {
    let envelope = claim(value);
    let source = BridgeSource { claim: &envelope, quantity: &contract.source.quantity };
    PreparedScientificBridge::prepare(contract.clone()).expect("licensed").derive_propagating(
        &source,
        standard_error,
        nonlinear,
    )
}

#[test]
fn an_affine_map_publishes_the_hand_delta_variance_beside_the_plug_in_value() {
    // a = 0.1, b = 0.6, x = 0.5, se = 0.05; Va = 0.0004, Vb = 0.01, Cov(a, b) = -0.0015.
    let result = derive(&affine_contract(Some(independent())), 0.5, Some(0.05), None).unwrap();
    let (x, b, v) = (0.5_f64, 0.6_f64, 0.05_f64 * 0.05);
    let expected = 0.0004 + x * x * 0.01 + b * b * v + 2.0 * x * -0.0015 + 0.01 * v;
    let propagation = result.propagated_uncertainty().unwrap();
    assert!(close(propagation.variance, expected), "{} vs {expected}", propagation.variance);
    assert_eq!(result.value().to_bits(), (0.1 + 0.6 * 0.5_f64).to_bits(), "the plug-in value");
    assert_eq!(result.propagated_mean().unwrap().to_bits(), result.value().to_bits());
    assert_eq!(result.propagated_variance(), Some(propagation.variance));
    assert_eq!(propagation.reading(), "uncalibrated_spread");
    assert_eq!(propagation.calibration(), "unmeasured");

    // The variance is a recorded uncertainty component, not an interval.
    let components = result.components();
    assert!(components.iter().any(|(_, target, omitted)| {
        !omitted && target == "propagated_variance:uncalibrated_spread:calibration_unmeasured"
    }));
    assert!(
        components.iter().any(|(_, target, omitted)| !omitted && target == "source_standard_error")
    );
    let unscoped = result.interval().unwrap_err();
    assert_eq!(unscoped.detail, "scientific_bridge.interval_not_licensed");
    // An affine map with declared uncertainty is no exact transform of the source interval.
    let scoped = result.bridged_interval(0.95).unwrap_err();
    assert_eq!(scoped.code, "cell_not_licensed");
    assert_eq!(scoped.detail, "scientific_bridge.interval_scope_outside_measured");
}

#[test]
fn the_declared_covariance_sign_moves_the_variance_and_the_mean() {
    let with = |covariance: f64| {
        let dependence =
            SourceMappingDependence::SlopeCovariance { covariance, evidence: "shared-rows".into() };
        let result = derive(&affine_contract(Some(dependence)), 0.5, Some(0.05), None).unwrap();
        (result.propagated_variance().unwrap(), result.propagated_mean().unwrap())
    };
    let (zero, mean_zero) = {
        let result = derive(&affine_contract(Some(independent())), 0.5, Some(0.05), None).unwrap();
        (result.propagated_variance().unwrap(), result.propagated_mean().unwrap())
    };
    let (positive, mean_positive) = with(0.0004);
    let (negative, mean_negative) = with(-0.0004);
    // 2 x b c + c^2 with x = 0.5, b = 0.6, c = +-0.0004.
    assert!(close(positive - zero, 2.0 * 0.5 * 0.6 * 0.0004 + 0.0004 * 0.0004));
    assert!(close(negative - zero, -2.0 * 0.5 * 0.6 * 0.0004 + 0.0004 * 0.0004));
    assert!(positive > zero && zero > negative);
    assert!(close(mean_positive - mean_zero, 0.0004) && close(mean_negative - mean_zero, -0.0004));
    // A covariance no joint law has is refused with its own detail.
    let impossible = SourceMappingDependence::SlopeCovariance {
        covariance: 0.01,
        evidence: "shared-rows".into(),
    };
    let error = derive(&affine_contract(Some(impossible)), 0.5, Some(0.05), None).unwrap_err();
    assert_eq!(error.detail, "scientific_bridge.dependence_not_a_covariance");
}

#[test]
fn a_nonlinear_square_reports_the_second_moment_expectation_not_the_square_of_the_mean() {
    let result =
        derive(&affine_contract(Some(independent())), 0.5, Some(0.05), Some(NonlinearMap::Square))
            .unwrap();
    let propagation = result.propagated_uncertainty().unwrap();
    let nonlinear = propagation.nonlinear.expect("the square was asked for");
    let (mean, variance) = (propagation.mean, propagation.variance);
    // E[X^2] = (E X)^2 + Var X for any law with those moments.
    assert_eq!(nonlinear.map_of_mean.to_bits(), (mean * mean).to_bits());
    assert_eq!(nonlinear.expectation.to_bits(), (mean * mean + variance).to_bits());
    assert!(nonlinear.expectation > nonlinear.map_of_mean);
    assert_eq!(result.propagated_mean().unwrap().to_bits(), nonlinear.expectation.to_bits());
    // The claim value stays the plug-in value of the linear map.
    assert_eq!(result.value().to_bits(), (0.1 + 0.6 * 0.5_f64).to_bits());

    // A nonlinear map of a mean alone is refused, and so is an exponential of an unspecified law.
    let error =
        derive(&affine_contract(Some(independent())), 0.5, None, Some(NonlinearMap::Square))
            .unwrap_err();
    assert_eq!(error.detail, "scientific_bridge.nonlinear_mean_not_expectation");
    let error = derive(
        &affine_contract(Some(independent())),
        0.5,
        Some(0.05),
        Some(NonlinearMap::Exponential),
    )
    .unwrap_err();
    assert_eq!(error.detail, "scientific_bridge.nonlinear_law_insufficient");
}

#[test]
fn an_undeclared_dependence_is_refused_with_a_typed_next_attempt_never_assumed_independent() {
    // A declared affine uncertainty with no declared dependence refuses at the derivation.
    let error = derive(&affine_contract(None), 0.5, Some(0.05), None).unwrap_err();
    assert_eq!(error.code, "sampling_dependence_unknown");
    assert_eq!(error.detail, "scientific_bridge.source_dependence_missing");
    assert!(error.remedy.is_some());
    // The same bridge without a standard error is the plug-in derivation, and the propagated
    // uncertainty it was asked for refuses by naming the missing standard error.
    let plain = derive(&affine_contract(None), 0.5, None, None).unwrap();
    assert!(plain.propagation().is_none() && plain.propagated_variance().is_none());
    assert_eq!(
        plain.propagated_uncertainty().unwrap_err().detail,
        "scientific_bridge.source_variance_invalid"
    );
    // A trial-level line with a standard error and no declared dependence keeps its measured
    // interval route and carries no propagated variance; asking for it refuses.
    let line = derive(&surrogate_contract(None), 0.8, Some(0.05), None).unwrap();
    assert!(line.propagation().is_none());
    assert_eq!(
        line.propagated_uncertainty().unwrap_err().detail,
        "scientific_bridge.source_dependence_missing"
    );
    // A correction has no slope estimator: only independence can be declared.
    let covariance = SourceMappingDependence::SlopeCovariance {
        covariance: 0.0,
        evidence: "validation-sample".into(),
    };
    let error = derive(&correction_contract(Some(covariance)), 0.31, Some(0.02), None).unwrap_err();
    assert_eq!(error.detail, "scientific_bridge.source_dependence_invalid");
}

#[test]
fn a_trial_level_line_carries_the_variance_of_its_fitted_mean_line() {
    let contract = surrogate_contract(Some(SourceMappingDependence::Independent {
        evidence: "disjoint-trials".into(),
    }));
    let result = derive(&contract, 0.8, Some(0.05), None).unwrap();
    let trials = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5), (0.8, 0.7)];
    let k = trials.len() as f64;
    let mean_x = trials.iter().map(|t| t.0).sum::<f64>() / k;
    let mean_y = trials.iter().map(|t| t.1).sum::<f64>() / k;
    let sxx = trials.iter().map(|t| (t.0 - mean_x).powi(2)).sum::<f64>();
    let syy = trials.iter().map(|t| (t.1 - mean_y).powi(2)).sum::<f64>();
    let sxy = trials.iter().map(|t| (t.0 - mean_x) * (t.1 - mean_y)).sum::<f64>();
    let slope = sxy / sxx;
    let s2 = (syy - sxy * sxy / sxx) / (k - 2.0);
    let v = 0.05_f64 * 0.05;
    let line = s2 * (1.0 / k + (0.8 - mean_x).powi(2) / sxx);
    let expected = line + slope * slope * v + (s2 / sxx) * v;
    let propagation = result.propagated_uncertainty().unwrap();
    assert!(close(propagation.variance, expected), "{} vs {expected}", propagation.variance);
    assert_eq!(propagation.construction, "trial_level_mean_line_delta_method");
}

#[test]
fn a_correction_carries_the_delta_variance_of_the_source_and_both_rates() {
    let contract = correction_contract(Some(SourceMappingDependence::Independent {
        evidence: "validation-sample".into(),
    }));
    let result = derive(&contract, 0.31, Some(0.02), None).unwrap();
    let p = 11.0_f64 / 70.0;
    let expected = (0.02_f64 * 0.02 + p * p * 0.0004 + (1.0 - p) * (1.0 - p) * 0.0004) / 0.49;
    let propagation = result.propagated_uncertainty().unwrap();
    assert!(close(propagation.variance, expected), "{} vs {expected}", propagation.variance);
    assert!(close(result.value(), p));
    assert_eq!(propagation.mean.to_bits(), result.value().to_bits());
}

#[test]
fn the_exported_artifact_replays_the_propagation_bit_for_bit() {
    let contract = affine_contract(Some(SourceMappingDependence::SlopeCovariance {
        covariance: 0.0004,
        evidence: "shared-rows".into(),
    }));
    let result = derive(&contract, 0.5, Some(0.05), Some(NonlinearMap::Square)).unwrap();
    let exported = result.export("propagated-bridge").unwrap();
    let consumed =
        consume_scientific_bridge_artifact(&exported.bytes, Some(&exported.identity)).unwrap();
    let held = result.propagation().unwrap();
    let replayed = consumed.bridged.propagation.as_ref().expect("the propagation replays");
    assert_eq!(replayed.variance.to_bits(), held.variance.to_bits());
    assert_eq!(replayed.mean.to_bits(), held.mean.to_bits());
    assert_eq!(replayed.nonlinear, held.nonlinear);
    assert_eq!(consumed.bridged.claim.claim_id, result.claim().claim_id);
    assert_eq!(consumed.bridged.value.to_bits(), result.value().to_bits());
    // Another standard error is another claim and another artifact.
    let other = derive(&contract, 0.5, Some(0.06), Some(NonlinearMap::Square)).unwrap();
    assert_ne!(other.claim().claim_id, result.claim().claim_id);
    assert_ne!(other.export("propagated-bridge").unwrap().identity, exported.identity);
}
