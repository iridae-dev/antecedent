//! The interval of a bridged value: hand-computed oracles, the refusals, and the
//! absence of a licence where no coverage record measured the construction.
//!
//! Oracles that are not the code under test:
//!
//! * four validation trials `(0, 0.1), (0.5, 0.5), (1, 0.9), (1.5, 1.5)`: slope `0.92`,
//!   intercept `0.06`, residuals `0.04, -0.02, -0.08, 0.06`, residual variance
//!   `0.012 / 2 = 0.006`, `xbar = 0.75`, `Sxx = 1.25`. A source value `0.8` with standard
//!   error `0.1` has leverage `1/4 + 0.05^2 / 1.25 = 0.252`, variance
//!   `0.006 * 1.252 + 0.92^2 * 0.01 = 0.0159752`, so a half width of `t_{2, 0.975} sd`
//!   with `t = 4.302653`;
//! * a declared affine map `0.1 + 0.6 x` of a source value `0.5` with standard error
//!   `0.05` is `0.4 +- 1.959964 * 0.6 * 0.05`;
//! * the Rogan-Gladen correction of a recorded `0.31` under sensitivity 0.9 and
//!   specificity 0.8 is `11/70`; with source standard error `0.02` and rate variances
//!   `0.0004` the delta-method variance is
//!   `(0.0004 + (11/70)^2 0.0004 + (59/70)^2 0.0004) / 0.7^2`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{PreparedScientificBridge, ScientificBridgeRefusal};
use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, QuantityClass, TrialEffectPair, TrialLevelValidation, ValueKind,
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
use antecedent_io::bridge_interval::construct_bridged_claim_interval;

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

fn quantity(
    variable: &str,
    units: &str,
    functional: &str,
    population: &str,
    regime: &str,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn claim(value: f64, units: &str) -> ClaimEnvelope {
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
        Some(Arc::from(units)),
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

fn surrogate_contract() -> BridgeContract {
    let trials = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
        .into_iter()
        .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
            surrogate_effect,
            terminal_effect,
        })
        .collect();
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity(
                "trial:biomarker",
                "mmol_per_l",
                "average_effect",
                "adults",
                "do(t=1)",
            ),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "trial:survival",
                "log_hazard",
                "average_effect",
                "adults",
                "do(t=1)",
            ),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials,
            min_trials: 4,
            min_r_squared: 0.9,
        }),
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,

        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn affine_contract() -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::PreferenceRealizedBehavior,
        source: BridgeEndpoint {
            quantity: quantity("survey:intent", "probability", "uptake", "panel", "observational"),
            class: QuantityClass::StatedPreference,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "behavior:uptake",
                "probability",
                "uptake",
                "panel",
                "observational",
            ),
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

        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn correction_contract() -> BridgeContract {
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
            quantity: quantity(
                "schema:flag",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
            class: QuantityClass::MeasuredOutcome,
        },
        target: BridgeEndpoint {
            quantity: quantity(
                "construct:disease",
                "probability",
                "prevalence",
                "clinic-a",
                "observational",
            ),
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
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn derive(
    contract: &BridgeContract,
    value: f64,
    units: &str,
    standard_error: Option<f64>,
) -> antecedent::analysis::scientific_bridge::ScientificBridgeResult {
    let envelope = claim(value, units);
    let source = BridgeSource { claim: &envelope, quantity: &contract.source.quantity };
    PreparedScientificBridge::prepare(contract.clone())
        .expect("licensed")
        .derive_with_standard_error(&source, standard_error)
        .expect("derived")
}

fn refusal(
    result: Result<impl std::fmt::Debug, ScientificBridgeRefusal>,
) -> ScientificBridgeRefusal {
    result.expect_err("the request must be refused")
}

#[test]
fn a_trial_level_prediction_interval_matches_the_hand_computed_variance_and_t_quantile() {
    let contract = surrogate_contract();
    let result = derive(&contract, 0.8, "mmol_per_l", Some(0.1));
    let built = construct_bridged_claim_interval(result.bridged(), Some(0.1), 0.95).expect("built");
    let expected_sd = (0.006_f64 * 1.252 + 0.92_f64 * 0.92 * 0.01).sqrt();
    let half = 4.302_653 * expected_sd;
    assert!(close(built.interval.value, 0.796, 1e-12));
    assert!(
        close(built.interval.standard_error, expected_sd, 1e-9),
        "{}",
        built.interval.standard_error
    );
    assert!(close(built.interval.lower, 0.796 - half, 1e-4), "{}", built.interval.lower);
    assert!(close(built.interval.upper, 0.796 + half, 1e-4), "{}", built.interval.upper);
    assert_eq!(built.interval.degrees_of_freedom, Some(2.0));
    assert!(built.interval.coverage_record.is_empty(), "construction is not a licence");
    // The record's scope is the number of validation trials.
    assert_eq!(built.basis.scope.row_count, 4);
    assert_eq!(built.basis.key.estimator, "scientific_bridge_trial_level_linear");
}

#[test]
fn an_affine_map_transforms_the_declared_source_interval_exactly() {
    let contract = affine_contract();
    let result = derive(&contract, 0.5, "probability", Some(0.05));
    let interval = result.bridged_interval(0.95).expect("an exact transform needs no record");
    let half = 1.959_964 * 0.6 * 0.05;
    assert!(close(interval.value, 0.4, 1e-12));
    assert!(close(interval.lower, 0.4 - half, 1e-5));
    assert!(close(interval.upper, 0.4 + half, 1e-5));
    assert_eq!(interval.licensed_by, "exact_transform_of_declared_source_interval");
    assert!(interval.coverage_record.is_empty());
}

#[test]
fn a_correction_propagates_the_source_and_both_rate_variances_by_the_delta_method() {
    let contract = correction_contract();
    let result = derive(&contract, 0.31, "probability", Some(0.02));
    let built =
        construct_bridged_claim_interval(result.bridged(), Some(0.02), 0.95).expect("built");
    let p = 11.0_f64 / 70.0;
    let variance = (0.02_f64 * 0.02 + p * p * 0.0004 + (1.0 - p) * (1.0 - p) * 0.0004) / 0.49;
    assert!(close(built.interval.value, p, 1e-12));
    assert!(close(built.interval.standard_error, variance.sqrt(), 1e-9));
    assert!(close(built.interval.lower, p - 1.959_964 * variance.sqrt(), 1e-5));
    assert!(close(built.interval.upper, p + 1.959_964 * variance.sqrt(), 1e-5));
    assert_eq!(built.basis.scope.row_count, 400, "the stated validation units");
}

#[test]
fn a_correction_inside_a_measured_record_is_published_bound_to_it() {
    // Rates estimated from a stated validation sample of 400 units (measured: 100 to 400),
    // Youden 0.7, rate standard errors 0.02, level 0.95.
    let contract = correction_contract();
    let result = derive(&contract, 0.31, "probability", Some(0.02));
    let interval = result.bridged_interval(0.95).expect("measured");
    let p = 11.0_f64 / 70.0;
    assert!(close(interval.value, p, 1e-12));
    assert!(interval.lower < p && p < interval.upper);
    assert_eq!(interval.licensed_by, "coverage_record");
    assert!(
        interval.coverage_record.starts_with("cov.bridged_claim_interval."),
        "{}",
        interval.coverage_record
    );
    assert!(!interval.calibration_sha.is_empty());
    assert!(interval.assumptions.contains("unresolved obligations"));
}

#[test]
fn a_trial_level_prediction_interval_inside_a_measured_record_is_published_bound_to_it() {
    // Twenty validation trials (measured: 15 to 40) on a noisy line.
    let trials: Vec<TrialEffectPair> = (0..20)
        .map(|i| {
            let x = 0.5 + 1.5 * f64::from(i) / 19.0;
            TrialEffectPair {
                surrogate_effect: x,
                terminal_effect: 0.1 + 0.9 * x + 0.1 * f64::from(i).sin(),
            }
        })
        .collect();
    let mut contract = surrogate_contract();
    if let BridgeMapping::TrialLevelLinear(validation) = &mut contract.mapping {
        validation.trials = trials;
        validation.min_trials = 4;
        validation.min_r_squared = 0.5;
    }
    let result = derive(&contract, 1.2, "mmol_per_l", Some(0.1));
    let interval = result.bridged_interval(0.95).expect("measured");
    assert!(interval.lower < interval.value && interval.value < interval.upper);
    assert_eq!(interval.mapping.as_str(), "trial_level_linear");
    assert_eq!(interval.licensed_by, "coverage_record");
    assert!(interval.degrees_of_freedom == Some(18.0));
    assert!(interval.coverage_record.contains("student_t_prediction_interval"));
}

#[test]
fn an_interval_without_the_source_standard_error_is_refused_with_its_remedy() {
    let contract = affine_contract();
    let result = derive(&contract, 0.5, "probability", None);
    let refused = refusal(result.bridged_interval(0.95));
    assert_eq!(refused.code, antecedent_core::reason_code!("cell_not_licensed"));
    assert_eq!(refused.detail, "scientific_bridge.interval_source_uncertainty_required");
    assert!(refused.remedy.as_deref().is_some_and(|remedy| remedy.contains("standard error")));
}

#[test]
fn an_unscoped_request_a_posterior_and_an_unmeasured_level_stay_refused() {
    let contract = surrogate_contract();
    let result = derive(&contract, 0.8, "mmol_per_l", Some(0.1));
    assert_eq!(
        refusal(result.interval()).detail,
        "scientific_bridge.interval_not_licensed",
        "an unscoped request"
    );
    // No record measures the trial-level construction at level 0.8, so the scoped request
    // is refused whatever else has been measured.
    let refused = refusal(result.bridged_interval(0.8));
    assert_eq!(refused.code, antecedent_core::reason_code!("cell_not_licensed"));
    assert_eq!(refused.detail, "scientific_bridge.interval_scope_outside_measured");
    // A level outside the admitted range is outside the measured scope.
    assert_eq!(
        refusal(result.bridged_interval(1.5)).detail,
        "scientific_bridge.interval_scope_outside_measured"
    );
}

#[test]
fn rates_declared_without_a_validation_sample_are_outside_the_measured_scope() {
    let mut contract = correction_contract();
    if let BridgeMapping::MeasurementCorrection(model) = &mut contract.mapping {
        model.evidence = ParameterEvidence::Declared { declarer: "lab".into() };
        contract.assumptions = vec![BridgeAssumption::for_measurement_model("flag-model", model)];
    }
    let result = derive(&contract, 0.31, "probability", Some(0.02));
    let refused = refusal(result.bridged_interval(0.95));
    assert_eq!(refused.detail, "scientific_bridge.interval_scope_outside_measured");
}
