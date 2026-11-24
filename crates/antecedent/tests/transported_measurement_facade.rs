//! A measurement model carried to another population under a declared invariance premise:
//! corrected prevalence against the inverted misclassification algebra, the typed receipt and
//! identity, the existing artifact of the re-bound model, fresh re-derivation, and the named
//! refusal of every unmet premise.
//!
//! Oracle. For a binary indicator with sensitivity `Se`, specificity `Sp` and observed positive
//! fraction `p*`, `pi = (p* + Sp - 1) / (Se + Sp - 1)` and its delta-method variance is
//! `(p*(1 - p*)/n + (1 - pi)^2 Var(Sp) + pi^2 Var(Se)) / (Se + Sp - 1)^2`. The carried
//! parameters are the source population's, so the transported value is that algebra at the
//! source's rates and the target's observed positives: `(0.52 + 0.8 - 1) / 0.7 = 0.32 / 0.7`
//! for `Se = 0.9`, `Sp = 0.8`, 520 of 1000; a model established for the target is used as is.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use antecedent::analysis::transported_measurement::{
    DETAIL_TRANSPORT_RECEIPT_MISMATCH, ErrorParameter, InvarianceDeclaration,
    PopulationErrorStructure, TransportBasis, transport_measurement,
    transport_measurement_estimate, verify_measurement_transport,
};
use antecedent::{
    MeasurementAnalysisBinding, MeasurementInput, consume_measurement_model_artifact,
};
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementInference,
    MeasurementKind, MeasurementModel, MeasurementParameters, MeasurementRoute,
    MeasurementVariable, ParameterEvidence, RepresentationVersion,
};

const BOTH: [ErrorParameter; 2] = [ErrorParameter::Sensitivity, ErrorParameter::Specificity];
const EVENTS: u64 = 520;
const UNITS: u64 = 1000;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn scalar() -> RepresentationVersion {
    RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 }
}

fn model(population: &str, se: f64, sp: f64) -> MeasurementModel {
    MeasurementModel {
        model_id: format!("screen-{population}"),
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
            population_id: population.into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: se,
                specificity: sp,
                sensitivity_variance: 1e-4,
                specificity_variance: 4e-4,
            }],
            min_youden: 0.1,
        },
        representation: scalar(),
        evidence: ParameterEvidence::ValidationSample {
            sample_id: format!("val-{population}"),
            units: 500,
        },
    }
}

fn binding(population: &str, regime: &str) -> MeasurementAnalysisBinding {
    MeasurementAnalysisBinding {
        construct_id: "construct:disease".into(),
        population_id: population.into(),
        regime_id: regime.into(),
        arm_id: None,
        representation: scalar(),
        representation_mapping: None,
        assumed_structure: ErrorStructure::NonDifferential,
    }
}

fn premise(parameters: &[ErrorParameter]) -> PopulationErrorStructure {
    PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
        "clinic_a", "clinic_b", "analyst", parameters,
    ))
}

fn data() -> MeasurementInput {
    MeasurementInput::BinaryPrevalence { events: EVENTS, total: UNITS }
}

fn algebra(se: f64, sp: f64) -> (f64, f64) {
    let j = se + sp - 1.0;
    let p_star = EVENTS as f64 / UNITS as f64;
    let pi = (p_star + sp - 1.0) / j;
    let variance =
        (p_star * (1.0 - p_star) / UNITS as f64 + (1.0 - pi).powi(2) * 4e-4 + pi.powi(2) * 1e-4)
            / j.powi(2);
    (pi, variance)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() < tolerance, "actual={actual}, expected={expected}");
}

#[test]
fn carried_rates_correct_the_target_prevalence_with_a_typed_receipt() {
    let result = transport_measurement_estimate(
        &[model("clinic_a", 0.9, 0.8)],
        Some(&premise(&BOTH)),
        binding("clinic_b", "observational"),
        data(),
        None,
        &ctx(),
    )
    .unwrap();
    let (pi, variance) = algebra(0.9, 0.8);
    close(pi, 0.32 / 0.7, 1e-15);
    close(result.point(), pi, 1e-12);
    close(result.standard_error().unwrap().powi(2), variance, 1e-12);
    close(result.estimate().naive_point().unwrap(), 0.52, 1e-15);
    assert_eq!(result.inference(), MeasurementInference::UncalibratedStandardError);

    let receipt = result.receipt();
    assert!(result.carried());
    assert_eq!(receipt.source_population, "clinic_a");
    assert_eq!(receipt.target_population, "clinic_b");
    assert!(matches!(receipt.basis, TransportBasis::DeclaredInvariance(_)));
    assert_eq!(result.identity_hex(), receipt.identity.to_hex());
    assert_eq!(result.identity_hex().len(), 64);
    assert!(result.assumptions().iter().any(|line| line.contains("not propagated")));
    // The corrected value is bound to the re-bound model, not to the source model.
    assert_ne!(receipt.source_model, result.estimate().model_identity());
    // The interval is the estimate's: the binary misclassification variance has an executed
    // independent closed-form check, so the Wald interval is open.
    let (lo, hi) = result.interval(0.95).unwrap();
    let half = 1.959_963_984_540_054 * variance.sqrt();
    close(lo, pi - half, 1e-9);
    close(hi, pi + half, 1e-9);
}

#[test]
fn the_existing_artifact_replays_and_the_receipt_is_rederived() {
    let models = [model("clinic_a", 0.9, 0.8)];
    let structure = premise(&BOTH);
    let target = binding("clinic_b", "observational");
    let result = transport_measurement_estimate(
        &models,
        Some(&structure),
        target.clone(),
        data(),
        None,
        &ctx(),
    )
    .unwrap();
    let bytes = result.export("transported").unwrap();
    let consumed =
        consume_measurement_model_artifact(&bytes, &result.estimate().model_identity().to_hex())
            .unwrap();
    close(consumed.replayed().unwrap().point, result.point(), 1e-12);
    assert_eq!(consumed.model().binding.population_id, "clinic_b");

    let receipt =
        verify_measurement_transport(&models, Some(&structure), &target, &result.identity_hex())
            .unwrap();
    assert_eq!(&receipt, result.receipt());
    let error = verify_measurement_transport(&models, Some(&structure), &target, &"0".repeat(64))
        .unwrap_err();
    assert_eq!(error.code, "measurement_binding_mismatch");
    assert_eq!(error.detail, DETAIL_TRANSPORT_RECEIPT_MISMATCH);
}

#[test]
fn changed_numbers_and_declarations_move_the_identity() {
    let run = |se: f64, parameters: &[ErrorParameter]| {
        transport_measurement_estimate(
            &[model("clinic_a", se, 0.8)],
            Some(&premise(parameters)),
            binding("clinic_b", "observational"),
            data(),
            None,
            &ctx(),
        )
        .unwrap()
    };
    let base = run(0.9, &BOTH);
    let shifted = run(0.85, &BOTH);
    close(shifted.point(), algebra(0.85, 0.8).0, 1e-12);
    assert_ne!(shifted.identity_hex(), base.identity_hex());
    let wider = run(
        0.9,
        &[ErrorParameter::Sensitivity, ErrorParameter::Specificity, ErrorParameter::ErrorVariance],
    );
    close(wider.point(), base.point(), 1e-15);
    assert_ne!(wider.identity_hex(), base.identity_hex());
    assert_eq!(wider.receipt().source_model, base.receipt().source_model);
}

#[test]
fn a_model_established_for_the_target_borrows_nothing() {
    let models = [model("clinic_a", 0.9, 0.8), model("clinic_b", 0.7, 0.95)];
    let differential = PopulationErrorStructure::Differential { declarer: "analyst".into() };
    let result = transport_measurement_estimate(
        &models,
        Some(&differential),
        binding("clinic_b", "observational"),
        data(),
        None,
        &ctx(),
    )
    .unwrap();
    close(result.point(), 0.47 / 0.65, 1e-12);
    assert!(!result.carried());
    assert!(matches!(result.receipt().basis, TransportBasis::DifferentialEstablished { .. }));
    assert!(result.assumptions().is_empty());
}

#[test]
fn every_unmet_premise_is_named() {
    let source = [model("clinic_a", 0.9, 0.8)];
    let target = binding("clinic_b", "observational");
    let run = |structure: Option<&PopulationErrorStructure>,
               binding: MeasurementAnalysisBinding| {
        transport_measurement_estimate(&source, structure, binding, data(), None, &ctx())
            .unwrap_err()
    };

    let missing = run(None, target.clone());
    assert_eq!(missing.code, "transport_missing_evidence");
    assert_eq!(
        missing.detail,
        antecedent_core::reason_details::measurement_model::INVARIANCE_PREMISE_REQUIRED
    );
    assert!(missing.remedy.is_some());

    let partial = run(Some(&premise(&[ErrorParameter::Specificity])), target.clone());
    assert_eq!(partial.code, "transport_not_certified");
    assert_eq!(
        partial.detail,
        antecedent_core::reason_details::measurement_model::INVARIANCE_NOT_DECLARED
    );
    assert!(partial.message.contains("sensitivity"), "{}", partial.message);

    let differential = PopulationErrorStructure::Differential { declarer: "analyst".into() };
    let unpooled = run(Some(&differential), target.clone());
    assert_eq!(unpooled.code, "differential_error_not_pooled");
    assert_eq!(
        unpooled.detail,
        antecedent_core::reason_details::measurement_model::DIFFERENTIAL_POPULATION_UNESTABLISHED
    );

    let elsewhere = PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
        "clinic_a", "clinic_c", "analyst", &BOTH,
    ));
    let wrong_target = run(Some(&elsewhere), target.clone());
    assert_eq!(wrong_target.code, "transport_not_certified");
    assert_eq!(
        wrong_target.detail,
        antecedent_core::reason_details::measurement_model::INVARIANCE_TARGET_MISMATCH
    );

    let unknown = PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
        "clinic_z", "clinic_b", "analyst", &BOTH,
    ));
    let source_unknown = run(Some(&unknown), target.clone());
    assert_eq!(
        source_unknown.detail,
        antecedent_core::reason_details::measurement_model::INVARIANCE_SOURCE_UNKNOWN
    );

    let regime = run(Some(&premise(&BOTH)), binding("clinic_b", "do(a=1)"));
    assert_eq!(regime.code, "measurement_binding_mismatch");
    assert_eq!(regime.detail, antecedent_core::reason_details::measurement_model::REGIME_MISMATCH);
}

#[test]
fn a_route_with_nothing_to_carry_is_refused_by_the_transport_decision() {
    let mut replicated = model("clinic_a", 0.9, 0.8);
    replicated.route = MeasurementRoute::RepeatedMeasurements;
    replicated.parameters =
        MeasurementParameters::Repeated { min_replicated_units: 20, min_reliability: 0.1 };
    let error = transport_measurement(
        &[replicated],
        Some(&premise(&[ErrorParameter::ErrorVariance])),
        &binding("clinic_b", "observational"),
    )
    .unwrap_err();
    assert_eq!(error.code, "route_not_supported");
    assert_eq!(
        error.detail,
        antecedent_core::reason_details::measurement_model::NOTHING_TO_TRANSPORT
    );
    assert!(error.remedy.is_some());
}

/// Logit of `successes / (successes + failures)` with the Haldane correction, and its variance.
fn haldane_logit(successes: f64, failures: f64) -> (f64, f64) {
    (((successes + 0.5) / (failures + 0.5)).ln(), 1.0 / (successes + 0.5) + 1.0 / (failures + 0.5))
}

/// Paule-Mandel pooling of two populations in closed form. With weights `1 / (v_k + t)` the
/// generalized statistic of two estimates is `(y1 - y2)^2 / (v1 + v2 + 2 t)`; it equals its one
/// degree of freedom at `t = ((y1 - y2)^2 - v1 - v2) / 2`, or the variance is zero when that is
/// negative. Returns the pooled mean, its variance and `t`.
fn two_population_pool(first: (f64, f64), second: (f64, f64)) -> (f64, f64, f64) {
    let between = (((first.0 - second.0).powi(2) - first.1 - second.1) / 2.0).max(0.0);
    let (w1, w2) = (1.0 / (first.1 + between), 1.0 / (second.1 + between));
    ((w1 * first.0 + w2 * second.0) / (w1 + w2), 1.0 / (w1 + w2), between)
}

/// The pooled differential route against the two-population closed form. Sensitivities 180/200
/// and 164/200 differ by more than their sampling error, so the between-population variance is
/// positive; specificities 170/200 and 172/200 do not, so it is zero and the pooled mean is the
/// inverse-variance mean. The target is validated, which keeps its own rates identified. The
/// corrected prevalence draws are one seeded simulation, so their interval is checked for what
/// the construction fixes: it lies in `[0, 1]`, contains the mean of the draws and leaves 2.5%
/// of the retained draws on each side.
#[test]
fn pooled_differential_rates_equal_the_two_population_paule_mandel_closed_form() {
    use antecedent::analysis::transported_measurement::{
        PooledDifferentialRequest, PopulationValidation, transport_pooled_differential_prevalence,
    };
    let validation = |population: &str, counts: [u64; 4]| PopulationValidation {
        population: population.into(),
        true_positive: counts[0],
        false_negative: counts[1],
        true_negative: counts[2],
        false_positive: counts[3],
    };
    let sources =
        [validation("north", [180, 20, 170, 30]), validation("south", [164, 36, 172, 28])];
    let target = validation("east", [172, 28, 171, 29]);
    let out = transport_pooled_differential_prevalence(
        PooledDifferentialRequest {
            sources: &sources,
            target_validation: Some(&target),
            target_population: "east",
            observed_positive: 400,
            observed_total: 1000,
            draws: 4000,
        },
        &ctx(),
    )
    .unwrap();

    let sensitivity = two_population_pool(haldane_logit(180.0, 20.0), haldane_logit(164.0, 36.0));
    assert!(sensitivity.2 > 0.1, "the sensitivities are heterogeneous: {}", sensitivity.2);
    close(out.sensitivity.mean_logit, sensitivity.0, 1e-9);
    close(out.sensitivity.mean_variance, sensitivity.1, 1e-9);
    close(out.sensitivity.between_population_variance, sensitivity.2, 1e-9);
    let specificity = two_population_pool(haldane_logit(170.0, 30.0), haldane_logit(172.0, 28.0));
    assert!(specificity.2 <= 0.0, "the specificities are homogeneous");
    close(out.specificity.mean_logit, specificity.0, 1e-12);
    close(out.specificity.mean_variance, specificity.1, 1e-12);
    close(out.specificity.between_population_variance, 0.0, 1e-12);

    assert!(out.interval.0 >= 0.0 && out.interval.1 <= 1.0);
    assert!(out.interval.0 <= out.prevalence && out.prevalence <= out.interval.1);
    let n = out.draws.len() as f64;
    let below = out.draws.iter().filter(|d| **d < out.interval.0).count() as f64;
    let above = out.draws.iter().filter(|d| **d > out.interval.1).count() as f64;
    assert!((below / n - 0.025).abs() < 0.002 && (above / n - 0.025).abs() < 0.002);
}
