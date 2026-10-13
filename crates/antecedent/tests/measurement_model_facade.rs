//! Facade integration of corrected functionals under a declared measurement
//! model: checked preparation, estimation, refresh, export and independent
//! consumption, against the package oracles.
//!
//! Every value asserted has an oracle that is not the code under test: the
//! closed-form attenuation factor of a simulated DGP with a known coefficient,
//! the algebra of misclassification (`p* = Se pi + (1 - Sp)(1 - pi)`), the
//! exact class summaries of a confusion model, and the conjugate normal
//! posterior of a latent regressor. Simulations use fixed seeds and tolerances
//! written in reported standard errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    CategoricalTarget, MeasurementAnalysisBinding, MeasurementDetail, MeasurementInput,
    StudyBuilder, consume_measurement_model_artifact,
};
use antecedent_contract::measurement::{LatentPosteriorTrust, MeasurementOperation};
use antecedent_contract::{
    ExternalCapability, ExternalPosteriorKind, ExternalScientificObject, ExternalTrustState,
    PosteriorProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, DifferentialAxis, ErrorStructure, MappingSemantics,
    MeasurementBinding, MeasurementInference, MeasurementKind, MeasurementModel,
    MeasurementParameters, MeasurementRoute, MeasurementVariable, ParameterEvidence,
    ParameterUncertainty, PosteriorReference, RepresentationMapping, RepresentationVersion,
};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_estimate::measurement::{
    DataBinding, LinearMeasuredData, correct_classical_linear_effect,
};
use antecedent_io::measurement_model_wire::MeasurementModelArtifact;
use antecedent_stats::measurement_error::{ObservedClassSummary, TwoByTwoCounts};

fn normal(rng: &mut CausalRng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn scalar() -> RepresentationVersion {
    RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 }
}

fn binding() -> MeasurementAnalysisBinding {
    MeasurementAnalysisBinding {
        construct_id: "construct:latent".into(),
        population_id: "cohort".into(),
        regime_id: "observational".into(),
        arm_id: None,
        representation: scalar(),
        representation_mapping: None,
        assumed_structure: ErrorStructure::NonDifferential,
    }
}

fn variable(id: &str, kind: MeasurementKind) -> MeasurementVariable {
    MeasurementVariable { id: id.into(), kind, units: "unit".into() }
}

fn model(
    route: MeasurementRoute,
    indicator_kind: MeasurementKind,
    structure: ErrorStructure,
    parameters: MeasurementParameters,
) -> MeasurementModel {
    MeasurementModel {
        model_id: "facade-model".into(),
        route,
        indicator: variable("schema:indicator", indicator_kind),
        construct: variable("construct:latent", MeasurementKind::LatentScientific),
        binding: MeasurementBinding {
            population_id: "cohort".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure,
        parameters,
        representation: scalar(),
        evidence: ParameterEvidence::Declared { declarer: "test".into() },
    }
}

fn classical_model(error_variance: f64, standard_error: f64) -> MeasurementModel {
    model(
        MeasurementRoute::ClassicalContinuousError,
        MeasurementKind::NoisyMisclassified,
        ErrorStructure::NonDifferential,
        MeasurementParameters::ClassicalError {
            error_variance,
            error_variance_standard_error: standard_error,
            min_reliability: 0.1,
        },
    )
}

fn rates(arm: Option<&str>, se: f64, sp: f64, vse: f64, vsp: f64) -> BinaryArmRates {
    BinaryArmRates {
        arm_id: arm.map(str::to_owned),
        sensitivity: se,
        specificity: sp,
        sensitivity_variance: vse,
        specificity_variance: vsp,
    }
}

fn binary_model(rates: Vec<BinaryArmRates>, structure: ErrorStructure) -> MeasurementModel {
    model(
        MeasurementRoute::BinaryMisclassification,
        MeasurementKind::NoisyMisclassified,
        structure,
        MeasurementParameters::Binary { rates, min_youden: 0.1 },
    )
}

/// `X ~ N(0,1)`, `Z = 0.5 X + N(0,1)`, `W = X + N(0, 0.6)`,
/// `Y = 1 + 2 X + 0.5 Z + N(0,1)`.
struct Simulated {
    y: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
}

fn simulate(n: usize, seed: u64) -> Simulated {
    let mut rng = CausalRng::from_seed(seed);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let x = normal(&mut rng);
        let zi = 0.5 * x + normal(&mut rng);
        w.push(x + 0.6_f64.sqrt() * normal(&mut rng));
        y.push(1.0 + 2.0 * x + 0.5 * zi + normal(&mut rng));
        z.push(zi);
    }
    Simulated { y, w, z }
}

fn classical_input(sim: &Simulated) -> MeasurementInput {
    MeasurementInput::ClassicalLinear {
        outcome: sim.y.clone(),
        regressor: sim.w.clone(),
        covariates: sim.z.clone(),
        covariate_count: 1,
        replicate_counts: None,
    }
}

fn prepare_classical(
    m: &MeasurementModel,
    sim: &Simulated,
) -> antecedent::PreparedMeasurementModel {
    StudyBuilder::measurement_model(m.clone(), binding(), classical_input(sim), None, &ctx())
        .unwrap()
}

// ---------------------------------------------------------------------------
// Classical continuous error
// ---------------------------------------------------------------------------

#[test]
fn classical_error_recovers_the_coefficient_and_carries_its_labels() {
    let sim = simulate(20_000, 424_242);
    let m = classical_model(0.6, 0.0);
    let prepared = prepare_classical(&m, &sim);
    assert_eq!(prepared.operation(), MeasurementOperation::LinearAdjustedEffect);
    let estimate = prepared.estimate(&ctx()).unwrap();

    // The facade is the estimator: the same record, bit for bit.
    let direct = correct_classical_linear_effect(
        &m,
        &DataBinding {
            population_id: "cohort".into(),
            regime_id: "observational".into(),
            arm_id: None,
            representation: scalar(),
        },
        &LinearMeasuredData {
            outcome: &sim.y,
            regressor: &sim.w,
            covariates: &sim.z,
            covariate_count: 1,
            replicate_counts: None,
        },
        &ctx(),
    )
    .unwrap();
    assert_eq!(estimate.record(), &direct.record);

    // Truth: within 4 reported standard errors of 2; the naive fit is the
    // closed-form attenuation lambda = 0.8 / 1.4 of it.
    let se = estimate.standard_error().unwrap();
    assert!((estimate.point() - 2.0).abs() < 4.0 * se, "{} se {se}", estimate.point());
    assert!(se < 0.08);
    let lambda = 0.8 / 1.4;
    let MeasurementDetail::Linear(effect) = estimate.detail() else {
        panic!("a classical route reports its linear effect");
    };
    let naive_se = effect.fit.naive_covariance[effect.fit.dim + 1].sqrt();
    assert!((estimate.naive_point().unwrap() - 2.0 * lambda).abs() < 4.0 * naive_se);
    assert!((2.0 - estimate.naive_point().unwrap()) > 10.0 * naive_se);

    // Identity, label and the closed interval.
    assert_eq!(estimate.model_identity(), m.identities().model);
    assert_eq!(prepared.model_identity_hex(), m.identities().model.to_hex());
    assert_eq!(estimate.route(), MeasurementRoute::ClassicalContinuousError);
    assert_eq!(estimate.inference(), MeasurementInference::UncalibratedStandardError);
    assert_eq!(estimate.inference().as_str(), "uncalibrated_standard_error");
    assert_eq!(estimate.parameter_uncertainty(), ParameterUncertainty::DeclaredExact);
    let refusal = estimate.interval().unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "measurement_model.interval_closed");
    let refusal = prepared.interval().unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
}

#[test]
fn the_measurement_parameter_uncertainty_is_in_the_variance_and_the_functional() {
    let sim = simulate(5_000, 7);
    let exact = prepare_classical(&classical_model(0.6, 0.0), &sim).estimate(&ctx()).unwrap();
    let uncertain = prepare_classical(&classical_model(0.6, 0.06), &sim).estimate(&ctx()).unwrap();
    assert_eq!(exact.point().to_bits(), uncertain.point().to_bits());
    assert!(uncertain.standard_error().unwrap() > exact.standard_error().unwrap());
    assert_eq!(uncertain.parameter_uncertainty(), ParameterUncertainty::Propagated);
    // The added variance is (d beta / d sigma^2)^2 Var(sigma^2), with
    // d beta / d sigma^2 = beta / Var(X | Z) = beta / 0.8.
    let added = uncertain.standard_error().unwrap().powi(2)
        - exact.standard_error().unwrap().powi(2);
    let slope = uncertain.point() / 0.8;
    assert!((added - slope * slope * 0.06 * 0.06).abs() / added < 0.1, "added {added}");
    // A downstream linear functional uses the joint covariance.
    let (value, variance) = uncertain.downstream_linear_functional(&[0.0, 1.5, 0.0]).unwrap();
    assert!((value - 1.5 * uncertain.point()).abs() < 1e-12);
    assert!((variance - 2.25 * uncertain.standard_error().unwrap().powi(2)).abs() < 1e-12);
    // A wrong-width weight vector refuses.
    assert!(uncertain.downstream_linear_functional(&[1.0]).is_err());
}

#[test]
fn row_order_does_not_change_the_correction_and_the_same_inputs_give_the_same_bits() {
    let sim = simulate(2_000, 99);
    let m = classical_model(0.6, 0.03);
    let forward = prepare_classical(&m, &sim).estimate(&ctx()).unwrap();
    let rev = Simulated {
        y: sim.y.iter().rev().copied().collect(),
        w: sim.w.iter().rev().copied().collect(),
        z: sim.z.iter().rev().copied().collect(),
    };
    let backward = prepare_classical(&m, &rev).estimate(&ctx()).unwrap();
    assert!((forward.point() - backward.point()).abs() < 1e-9);
    assert!(
        (forward.standard_error().unwrap() - backward.standard_error().unwrap()).abs() < 1e-9
    );
    let again = prepare_classical(&m, &sim).estimate(&ctx()).unwrap();
    assert_eq!(forward.record(), again.record());
}

#[test]
fn estimator_refusals_keep_their_registered_codes() {
    let sim = simulate(500, 5);
    // An error variance that leaves no latent variance.
    let prepared = prepare_classical(&classical_model(5.0, 0.0), &sim);
    let refusal = prepared.estimate(&ctx()).unwrap_err();
    assert_eq!(refusal.code, "measurement_not_identified");
    assert_eq!(refusal.detail, "measurement_model.reliability_too_small");
}

// ---------------------------------------------------------------------------
// Prepare: every check runs before any number
// ---------------------------------------------------------------------------

#[test]
fn population_regime_arm_and_representation_are_checked_before_estimation() {
    let sim = simulate(60, 3);
    let m = classical_model(0.6, 0.0);
    let refuse = |model: &MeasurementModel, b: MeasurementAnalysisBinding| {
        StudyBuilder::measurement_model(model.clone(), b, classical_input(&sim), None, &ctx())
            .unwrap_err()
    };
    let mut other = binding();
    other.population_id = "elsewhere".into();
    let refusal = refuse(&m, other);
    assert_eq!(refusal.code, "measurement_binding_mismatch");
    assert_eq!(refusal.detail, "measurement_model.population_mismatch");

    let mut other = binding();
    other.regime_id = "do(a=1)".into();
    assert_eq!(refuse(&m, other).detail, "measurement_model.regime_mismatch");

    let mut arm_model = classical_model(0.6, 0.0);
    arm_model.binding.arm = ArmBinding::Arm("treated".into());
    let mut other = binding();
    other.arm_id = Some("control".into());
    assert_eq!(refuse(&arm_model, other).detail, "measurement_model.arm_mismatch");

    let mut other = binding();
    other.representation.version = 2;
    let refusal = refuse(&m, other);
    assert_eq!(refusal.code, "representation_mapping_required");
    assert_eq!(refusal.detail, "measurement_model.representation_unmapped");

    // A proxy is never the construct: the measured column named as the
    // construct, and a proxy kind declared as the construct.
    let mut other = binding();
    other.construct_id = "schema:indicator".into();
    let refusal = refuse(&m, other);
    assert_eq!(refusal.code, "proxy_offered_as_construct");
    assert_eq!(refusal.detail, "measurement_model.indicator_as_construct");
    let mut proxy_construct = classical_model(0.6, 0.0);
    proxy_construct.construct.kind = MeasurementKind::LearnedVectorRepresentation;
    let refusal = refuse(&proxy_construct, binding());
    assert_eq!(refusal.code, "proxy_offered_as_construct");
    assert_eq!(refusal.detail, "measurement_model.proxy_as_construct");

    // The refusals are reason-coded for the facade error as well.
    let error: antecedent::CausalError = refuse(&m, {
        let mut o = binding();
        o.population_id = "elsewhere".into();
        o
    })
    .into();
    assert_eq!(error.reason_code(), Some("measurement_binding_mismatch"));
}

#[test]
fn a_declared_structure_is_never_pooled_with_the_other() {
    let table = TwoByTwoCounts {
        exposed_events: 480,
        exposed_total: 1000,
        unexposed_events: 180,
        unexposed_total: 1000,
    };
    let input = MeasurementInput::BinaryRiskDifference {
        exposed_arm_id: "treated".into(),
        unexposed_arm_id: "control".into(),
        table,
    };
    let differential = binary_model(
        vec![
            rates(Some("treated"), 0.9, 0.8, 0.0, 0.0),
            rates(Some("control"), 0.7, 0.95, 0.0, 0.0),
        ],
        ErrorStructure::Differential(DifferentialAxis::Arm),
    );
    // Analysed as non-differential: refused.
    let refusal =
        StudyBuilder::measurement_model(differential.clone(), binding(), input.clone(), None, &ctx())
            .unwrap_err();
    assert_eq!(refusal.code, "differential_error_not_pooled");
    assert_eq!(refusal.detail, "measurement_model.differential_declared");
    // Analysed under its own declaration: the per-arm rates apply.
    let right = StudyBuilder::measurement_model(
        differential.clone(),
        binding().assuming(ErrorStructure::Differential(DifferentialAxis::Arm)),
        input.clone(),
        None,
        &ctx(),
    )
    .unwrap()
    .estimate(&ctx())
    .unwrap();
    // Observed 0.9*0.4 + 0.2*0.6 = 0.48 and 0.7*0.2 + 0.05*0.8 = 0.18.
    assert!((right.point() - 0.2).abs() < 1e-12);
    assert!((right.naive_point().unwrap() - 0.3).abs() < 1e-12);
    // The shared-rate model is a different identity and a different answer.
    let pooled = binary_model(vec![rates(None, 0.9, 0.8, 0.0, 0.0)], ErrorStructure::NonDifferential);
    let wrong = StudyBuilder::measurement_model(pooled.clone(), binding(), input.clone(), None, &ctx())
        .unwrap()
        .estimate(&ctx())
        .unwrap();
    assert!((wrong.point() - 0.3 / 0.7).abs() < 1e-12);
    assert_ne!(right.model_identity(), wrong.model_identity());

    // An arm the differential model has no rates for refuses; no fallback.
    let unknown = MeasurementInput::BinaryRiskDifference {
        exposed_arm_id: "treated".into(),
        unexposed_arm_id: "placebo".into(),
        table,
    };
    let refusal = StudyBuilder::measurement_model(
        differential,
        binding().assuming(ErrorStructure::Differential(DifferentialAxis::Arm)),
        unknown,
        None,
        &ctx(),
    )
    .unwrap()
    .estimate(&ctx())
    .unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.arm_mismatch");

    // Differential by time has no bounded route: refused, not pooled.
    let mut by_time = classical_model(0.6, 0.0);
    by_time.structure = ErrorStructure::Differential(DifferentialAxis::Time);
    let sim = simulate(60, 3);
    let refusal = StudyBuilder::measurement_model(
        by_time,
        binding().assuming(ErrorStructure::Differential(DifferentialAxis::Time)),
        classical_input(&sim),
        None,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "differential_error_not_pooled");
}

#[test]
fn the_route_licenses_only_its_own_operations() {
    let sim = simulate(60, 3);
    let prepared = prepare_classical(&classical_model(0.6, 0.0), &sim);
    assert!(prepared.check_operation(MeasurementOperation::LinearFunctionalOfCoefficients).is_ok());
    for refused in [
        MeasurementOperation::NonlinearOutcomeModel,
        MeasurementOperation::ConditionOnConstruct,
        MeasurementOperation::TransportToOtherPopulation,
        MeasurementOperation::Interval,
    ] {
        let refusal = prepared.check_operation(refused).unwrap_err();
        assert_eq!(refusal.code, "cell_not_licensed", "{refused:?}");
        assert_eq!(refusal.detail, "measurement_model.operation_not_licensed");
        assert!(refusal.remedy.is_some());
    }
    // A binary input sent at a classical model is not an operation of its route.
    let refusal = StudyBuilder::measurement_model(
        classical_model(0.6, 0.0),
        binding(),
        MeasurementInput::BinaryPrevalence { events: 10, total: 100 },
        None,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(prepared.route_card().route, MeasurementRoute::ClassicalContinuousError);
}

#[test]
fn a_mapping_carries_values_and_intervention_meaning_only_when_declared() {
    let sim = simulate(2_000, 11);
    let m = classical_model(0.6, 0.0);
    let relabel = RepresentationMapping {
        from: scalar(),
        to: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 2 },
        mapping_id: "map-relabel".into(),
        semantics: MappingSemantics::CoordinateRelabeling,
    };
    let mut mapped = binding().with_mapping(relabel.clone());
    mapped.representation = relabel.to.clone();
    let prepared = StudyBuilder::measurement_model(
        m.clone(),
        mapped,
        classical_input(&sim),
        None,
        &ctx(),
    )
    .unwrap();
    let estimate = prepared.estimate(&ctx()).unwrap();
    assert_eq!(estimate.representation_mapping_id(), Some("map-relabel"));
    // Values crossed unchanged: the same correction as the unmapped one.
    let plain = prepare_classical(&m, &sim).estimate(&ctx()).unwrap();
    assert_eq!(estimate.record(), plain.record());
    // A relabeling carries values only.
    let refusal =
        prepared.check_operation(MeasurementOperation::InterveneOnRepresentation).unwrap_err();
    assert_eq!(refusal.code, "representation_mapping_required");
    assert_eq!(refusal.detail, "measurement_model.intervention_meaning_not_carried");
    // A mapping between other versions is refused outright.
    let mut wrong = relabel.clone();
    wrong.from.version = 7;
    let mut b = binding().with_mapping(wrong);
    b.representation = relabel.to.clone();
    let refusal =
        StudyBuilder::measurement_model(m.clone(), b, classical_input(&sim), None, &ctx())
            .unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.mapping_endpoints");
    // Declared evidence of intervention preservation licenses the meaning.
    let preserving = RepresentationMapping {
        semantics: MappingSemantics::InterventionPreserving { evidence_id: "ev:trial-9".into() },
        ..relabel.clone()
    };
    let mut b = binding().with_mapping(preserving.clone());
    b.representation = preserving.to.clone();
    let ok = StudyBuilder::measurement_model(m, b, classical_input(&sim), None, &ctx()).unwrap();
    // The mapping now carries the meaning; the route itself still refuses an
    // intervention on the recorded indicator (it is declared on the construct).
    let refusal =
        ok.check_operation(MeasurementOperation::InterveneOnRepresentation).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "measurement_model.operation_not_licensed");
}

// ---------------------------------------------------------------------------
// Repeated measurements
// ---------------------------------------------------------------------------

fn repeated_world(units: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    // X ~ N(0,1), replicates W_ij = X_i + N(0, 0.5): ICC = 1 / 1.5.
    let mut rng = CausalRng::from_seed(seed);
    let mut y = Vec::new();
    let mut counts = Vec::new();
    let mut values = Vec::new();
    for i in 0..units {
        let x = normal(&mut rng);
        y.push(1.0 + 1.5 * x + normal(&mut rng));
        let k = 2 + (i % 2) as u32; // 2 or 3 replicates
        counts.push(k);
        for _ in 0..k {
            values.push(x + 0.5_f64.sqrt() * normal(&mut rng));
        }
    }
    (y, values, counts)
}

fn repeated_model(min_replicated_units: u32) -> MeasurementModel {
    model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementKind::NoisyMisclassified,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Repeated { min_replicated_units, min_reliability: 0.1 },
    )
}

#[test]
fn repeated_measurements_estimate_the_icc_and_recover_the_coefficient() {
    let (y, values, counts) = repeated_world(6_000, 2024);
    let input = |y: Vec<f64>, values: Vec<f64>, counts: Vec<u32>| MeasurementInput::RepeatedLinear {
        outcome: y,
        replicate_values: values,
        replicate_counts: counts,
        covariates: vec![],
        covariate_count: 0,
    };
    let prepared = StudyBuilder::measurement_model(
        repeated_model(50),
        binding(),
        input(y, values, counts),
        None,
        &ctx(),
    )
    .unwrap();
    let estimate = prepared.estimate(&ctx()).unwrap();
    let MeasurementDetail::Linear(effect) = estimate.detail() else {
        panic!("the repeated route reports its linear effect");
    };
    let components = effect.components.as_ref().unwrap();
    assert!((components.intraclass_correlation - 1.0 / 1.5).abs() < 0.03);
    assert!((components.within_variance - 0.5).abs() < 0.03);
    let se = estimate.standard_error().unwrap();
    assert!((estimate.point() - 1.5).abs() < 4.0 * se, "{} se {se}", estimate.point());
    // The mean of 2.5 replicates still attenuates: the naive slope is below truth.
    assert!(estimate.naive_point().unwrap() < 1.5 - 5.0 * se);
    assert_eq!(estimate.parameter_uncertainty(), ParameterUncertainty::Propagated);
    assert_eq!(estimate.route(), MeasurementRoute::RepeatedMeasurements);

    // Too few replicated units for the model's declared minimum.
    let (y, values, counts) = repeated_world(200, 5);
    let refusal = StudyBuilder::measurement_model(
        repeated_model(10_000),
        binding(),
        input(y, values, counts),
        None,
        &ctx(),
    )
    .unwrap()
    .estimate(&ctx())
    .unwrap_err();
    assert_eq!(refusal.code, "measurement_not_identified");
    assert_eq!(refusal.detail, "measurement_model.insufficient_replicates");
}

// ---------------------------------------------------------------------------
// Binary and categorical routes: exact algebra
// ---------------------------------------------------------------------------

#[test]
fn binary_prevalence_inverts_the_misclassification_algebra() {
    // pi = 0.3, Se = 0.9, Sp = 0.8: p* = 0.27 + 0.14 = 0.41.
    let m = binary_model(
        vec![rates(None, 0.9, 0.8, 0.0001, 0.0004)],
        ErrorStructure::NonDifferential,
    );
    let estimate = StudyBuilder::measurement_model(
        m,
        binding(),
        MeasurementInput::BinaryPrevalence { events: 410, total: 1000 },
        None,
        &ctx(),
    )
    .unwrap()
    .estimate(&ctx())
    .unwrap();
    assert!((estimate.point() - 0.3).abs() < 1e-12);
    assert!((estimate.naive_point().unwrap() - 0.41).abs() < 1e-15);
    let expected = (0.41 * 0.59 / 1000.0 + 0.49 * 0.0004 + 0.09 * 0.0001) / 0.49;
    let se = estimate.standard_error().unwrap();
    assert!((se * se - expected).abs() < 1e-12);
    assert_eq!(estimate.parameter_uncertainty(), ParameterUncertainty::Propagated);
    assert_eq!(estimate.operation(), MeasurementOperation::LatentPrevalence);
    // No coefficient vector to take a functional of.
    assert!(estimate.downstream_linear_functional(&[1.0]).is_err());
}

#[test]
fn shared_rates_keep_the_arm_covariance_that_split_rates_miss() {
    let table = TwoByTwoCounts {
        exposed_events: 480,
        exposed_total: 1000,
        unexposed_events: 340,
        unexposed_total: 1000,
    };
    let m = binary_model(
        vec![rates(None, 0.9, 0.8, 0.0004, 0.0009)],
        ErrorStructure::NonDifferential,
    );
    let estimate = StudyBuilder::measurement_model(
        m,
        binding(),
        MeasurementInput::BinaryRiskDifference {
            exposed_arm_id: "treated".into(),
            unexposed_arm_id: "control".into(),
            table,
        },
        None,
        &ctx(),
    )
    .unwrap()
    .estimate(&ctx())
    .unwrap();
    assert!((estimate.point() - 0.2).abs() < 1e-12);
    // Hand calculation: sampling [0.48*0.52 + 0.34*0.66]/1000 / 0.49 and
    // rates 0.2^2 (0.0004 + 0.0009) / 0.49.
    let sampling = (0.48 * 0.52 / 1000.0 + 0.34 * 0.66 / 1000.0) / 0.49;
    let rate_part = 0.04 * 0.0013 / 0.49;
    let se = estimate.standard_error().unwrap();
    assert!((se * se - (sampling + rate_part)).abs() < 1e-12);
    assert_eq!(estimate.operation(), MeasurementOperation::RiskDifference);
}

#[test]
fn youden_index_at_or_below_the_minimum_refuses() {
    for (se, sp) in [(0.5, 0.5), (0.3, 0.6), (0.55, 0.5)] {
        let m = binary_model(vec![rates(None, se, sp, 0.0, 0.0)], ErrorStructure::NonDifferential);
        let refusal = StudyBuilder::measurement_model(
            m,
            binding(),
            MeasurementInput::BinaryPrevalence { events: 40, total: 100 },
            None,
            &ctx(),
        )
        .unwrap()
        .estimate(&ctx())
        .unwrap_err();
        assert_eq!(refusal.code, "measurement_not_identified");
        assert_eq!(refusal.detail, "measurement_model.youden_too_small");
    }
}

fn categorical_model(column_counts: Option<Vec<u64>>) -> MeasurementModel {
    model(
        MeasurementRoute::ProbabilisticCategorical,
        MeasurementKind::ProbabilisticFeatureAssignment,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Categorical {
            levels: vec!["low".into(), "mid".into(), "high".into()],
            confusion: vec![0.8, 0.1, 0.0, 0.1, 0.8, 0.2, 0.1, 0.1, 0.8],
            column_counts,
            max_condition: 50.0,
        },
    )
}

fn exact_classes() -> Vec<ObservedClassSummary> {
    // Rows are the observed label, columns the true category.
    let c = [0.8, 0.1, 0.0, 0.1, 0.8, 0.2, 0.1, 0.1, 0.8];
    let pi = [0.5, 0.3, 0.2];
    let mu = [1.0, 2.0, 4.0];
    let n = 10_000.0;
    (0..3)
        .map(|j| {
            let p: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t]).sum();
            let ybar: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t] * mu[t]).sum::<f64>() / p;
            ObservedClassSummary { count: (p * n).round() as u64, mean: ybar, variance: 1.0 }
        })
        .collect()
}

#[test]
fn categorical_inversion_recovers_proportions_and_outcome_means() {
    let m = categorical_model(Some(vec![400, 400, 400]));
    let run = |target: CategoricalTarget| {
        StudyBuilder::measurement_model(
            m.clone(),
            binding(),
            MeasurementInput::Categorical { classes: exact_classes(), target },
            None,
            &ctx(),
        )
        .unwrap()
        .estimate(&ctx())
        .unwrap()
    };
    for (level, truth) in [("low", 0.5), ("mid", 0.3), ("high", 0.2)] {
        let e = run(CategoricalTarget::Proportion(level.into()));
        assert!((e.point() - truth).abs() < 1e-9, "{level}: {}", e.point());
        assert_eq!(e.operation(), MeasurementOperation::LatentPrevalence);
        assert!(e.standard_error().unwrap() > 0.0);
    }
    let contrast = run(CategoricalTarget::OutcomeContrast {
        first: "high".into(),
        second: "low".into(),
    });
    assert!((contrast.point() - 3.0).abs() < 1e-6, "{}", contrast.point());
    assert!(contrast.naive_point().unwrap() < 2.5);
    assert_eq!(contrast.operation(), MeasurementOperation::CategoryOutcomeMeans);
    assert_eq!(contrast.parameter_uncertainty(), ParameterUncertainty::Propagated);
    // An undeclared category refuses.
    let refusal = StudyBuilder::measurement_model(
        m,
        binding(),
        MeasurementInput::Categorical {
            classes: exact_classes(),
            target: CategoricalTarget::Proportion("absent".into()),
        },
        None,
        &ctx(),
    )
    .unwrap()
    .estimate(&ctx())
    .unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.unknown_level");
}

// ---------------------------------------------------------------------------
// External latent posterior
// ---------------------------------------------------------------------------

fn posterior_model(conditioned: bool) -> MeasurementModel {
    model(
        MeasurementRoute::ExternalLatentPosterior,
        MeasurementKind::LearnedVectorRepresentation,
        ErrorStructure::NonDifferential,
        MeasurementParameters::ExternalPosterior {
            reference: PosteriorReference {
                provider_id: "vendor".into(),
                object_id: "latent-1".into(),
                version_id: "v3".into(),
                snapshot_id: "snap".into(),
                request_id: "req-1".into(),
            },
            conditioned_on_outcome: conditioned,
            min_draws: 20,
        },
    )
}

fn provider_object(request_id: &str) -> ExternalScientificObject {
    ExternalScientificObject::Posterior(PosteriorProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "vendor".into(),
            object_id: "latent-1".into(),
            version_id: "v3".into(),
            snapshot_id: "snap".into(),
            request_id: request_id.into(),
        },
        kind: ExternalPosteriorKind::Parameter,
        quantities: vec![ScientificQuantity {
            variable_id: "construct:latent".into(),
            variable_name: "latent".into(),
            role: QuantityRole::Covariate,
            units: "unit".into(),
            population_id: "cohort".into(),
            regime_id: "observational".into(),
            horizon: 0,
            functional_id: "value".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }],
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
    })
}

/// `X ~ N(0,1)`, `W = X + N(0,1)`, `Y = 1 + 2 X + N(0,1)`; the posterior of
/// `X` given `W` alone has mean `W/2` and variance 1/2, given `(W, Y)`
/// precision 6.
fn posterior_world(n: usize, draws: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for _ in 0..n {
        let x = normal(&mut rng);
        w.push(x + normal(&mut rng));
        y.push(1.0 + 2.0 * x + normal(&mut rng));
    }
    let mut marginal = vec![0.0; n * draws];
    let mut conditional = vec![0.0; n * draws];
    for s in 0..draws {
        for i in 0..n {
            marginal[s * n + i] = w[i] / 2.0 + 0.5_f64.sqrt() * normal(&mut rng);
            let mean = (w[i] + 2.0 * (y[i] - 1.0)) / 6.0;
            conditional[s * n + i] = mean + (1.0_f64 / 6.0).sqrt() * normal(&mut rng);
        }
    }
    (y, marginal, conditional)
}

#[test]
fn a_posterior_route_needs_the_exact_attested_provider_object() {
    let (y, marginal, _) = posterior_world(500, 30, 31);
    let input = || MeasurementInput::LatentPosterior {
        outcome: y.clone(),
        draws: marginal.clone(),
        draw_count: 30,
        covariates: vec![],
        covariate_count: 0,
    };
    let m = posterior_model(false);
    let good = provider_object("req-1");
    let attested = ExternalTrustState::attest(&good, "vendor").unwrap();

    // No provider object at all.
    let refusal =
        StudyBuilder::measurement_model(m.clone(), binding(), input(), None, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "measurement_posterior_not_trusted");
    // Metadata cannot confer native trust.
    let refusal = StudyBuilder::measurement_model(
        m.clone(),
        binding(),
        input(),
        Some((&good, &ExternalTrustState::NativeLicensed)),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.posterior_native_trust");
    // Another request fingerprint is another object.
    let other = provider_object("req-2");
    let attested_other = ExternalTrustState::attest(&other, "vendor").unwrap();
    let refusal = StudyBuilder::measurement_model(
        m.clone(),
        binding(),
        input(),
        Some((&other, &attested_other)),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.posterior_identity");
    // The exact object, attested, is consumed and labelled as such.
    let prepared = StudyBuilder::measurement_model(
        m,
        binding(),
        input(),
        Some((&good, &attested)),
        &ctx(),
    )
    .unwrap();
    assert_eq!(prepared.provider_trust(), Some(LatentPosteriorTrust::ExternallyAttested));
    // A learned representation's coordinates are not intervenable.
    let refusal =
        prepared.check_operation(MeasurementOperation::InterveneOnRepresentation).unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.learned_coordinates_not_intervenable");
}

#[test]
fn posterior_mean_substitution_and_rubins_rules_replay_independently() {
    let n = 4_000;
    let draws = 100;
    let (y, marginal, conditional) = posterior_world(n, draws, 31);
    let good = provider_object("req-1");
    let attested = ExternalTrustState::attest(&good, "vendor").unwrap();

    // Substitution: the posterior does not condition on the outcome.
    let m = posterior_model(false);
    let prepared = StudyBuilder::measurement_model(
        m.clone(),
        binding(),
        MeasurementInput::LatentPosterior {
            outcome: y.clone(),
            draws: marginal,
            draw_count: draws,
            covariates: vec![],
            covariate_count: 0,
        },
        Some((&good, &attested)),
        &ctx(),
    )
    .unwrap();
    let estimate = prepared.estimate(&ctx()).unwrap();
    let se = estimate.standard_error().unwrap();
    assert!((estimate.point() - 2.0).abs() < 4.0 * se + 0.03, "{}", estimate.point());
    assert_eq!(estimate.parameter_uncertainty(), ParameterUncertainty::NotPropagated);
    assert_eq!(estimate.provider_trust(), Some(LatentPosteriorTrust::ExternallyAttested));
    let bytes = estimate.export("posterior-substitution").unwrap();
    let consumed =
        consume_measurement_model_artifact(&bytes, &prepared.model_identity_hex()).unwrap();
    assert_eq!(consumed.provider_trust(), Some("externally_attested"));
    let replayed = consumed.replayed().unwrap();
    assert!((replayed.point - estimate.point()).abs() < 1e-9);
    assert!((replayed.variance.sqrt() - se).abs() < 1e-9);

    // Outcome-conditioned draws combine by Rubin's rules.
    let mc = posterior_model(true);
    let prepared = StudyBuilder::measurement_model(
        mc.clone(),
        binding(),
        MeasurementInput::LatentPosterior {
            outcome: y,
            draws: conditional,
            draw_count: draws,
            covariates: vec![],
            covariate_count: 0,
        },
        Some((&good, &attested)),
        &ctx(),
    )
    .unwrap();
    let estimate = prepared.estimate(&ctx()).unwrap();
    let se = estimate.standard_error().unwrap();
    assert!((estimate.point() - 2.0).abs() < 4.0 * se + 0.02, "{}", estimate.point());
    assert_eq!(estimate.parameter_uncertainty(), ParameterUncertainty::Propagated);
    let names: Vec<&str> =
        estimate.record().variance_components.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["within_imputation", "between_imputation"]);
    let bytes = estimate.export("posterior-imputation").unwrap();
    let consumed =
        consume_measurement_model_artifact(&bytes, &prepared.model_identity_hex()).unwrap();
    let replayed = consumed.replayed().unwrap();
    assert!((replayed.point - estimate.point()).abs() < 1e-9);
    // Fewer draws than the model declares refuses before a claim.
    let (y2, marginal2, _) = posterior_world(200, 10, 4);
    let refusal = StudyBuilder::measurement_model(
        m,
        binding(),
        MeasurementInput::LatentPosterior {
            outcome: y2,
            draws: marginal2,
            draw_count: 10,
            covariates: vec![],
            covariate_count: 0,
        },
        Some((&good, &attested)),
        &ctx(),
    )
    .unwrap()
    .estimate(&ctx())
    .unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.insufficient_draws");
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[test]
fn refresh_keeps_the_model_recomputes_the_numbers_and_refuses_other_questions() {
    let first = simulate(3_000, 21);
    let second = simulate(3_000, 22);
    let m = classical_model(0.6, 0.05);
    let prepared = prepare_classical(&m, &first);
    let before = prepared.estimate(&ctx()).unwrap();

    let refreshed = prepared.refresh(classical_input(&second), &ctx()).unwrap();
    // Kept: the model and its identities.
    assert_eq!(refreshed.prepared.model(), prepared.model());
    assert_eq!(refreshed.prepared.model_identity_hex(), prepared.model_identity_hex());
    assert_eq!(refreshed.estimate.model_identity(), before.model_identity());
    // Recomputed: the point, the standard error and the data identity.
    assert_ne!(refreshed.estimate.point().to_bits(), before.point().to_bits());
    assert_ne!(refreshed.estimate.record().data_identity, before.record().data_identity);
    let direct = prepare_classical(&m, &second).estimate(&ctx()).unwrap();
    assert_eq!(refreshed.estimate.record(), direct.record());

    // Refused: another question. And the state it was called on is untouched.
    let refusal = prepared
        .refresh(MeasurementInput::BinaryPrevalence { events: 1, total: 10 }, &ctx())
        .unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(prepared.estimate(&ctx()).unwrap().record(), before.record());
    // A refresh that fails inside the estimator leaves the prior state alone.
    let tiny = Simulated { y: vec![0.0; 3], w: vec![0.0; 3], z: vec![0.0; 3] };
    assert!(prepared.refresh(classical_input(&tiny), &ctx()).is_err());
    assert_eq!(prepared.estimate(&ctx()).unwrap().record(), before.record());
}

#[test]
fn changed_validation_parameters_rebuild_the_identities() {
    let sim = simulate(3_000, 21);
    let prepared = prepare_classical(&classical_model(0.6, 0.05), &sim);
    let before = prepared.estimate(&ctx()).unwrap();
    let rebuilt = prepared
        .with_validation_parameters(
            MeasurementParameters::ClassicalError {
                error_variance: 0.5,
                error_variance_standard_error: 0.05,
                min_reliability: 0.1,
            },
            ParameterEvidence::ValidationSample { sample_id: "val-2".into(), units: 400 },
            &ctx(),
        )
        .unwrap();
    assert_ne!(rebuilt.prepared.model_identity_hex(), prepared.model_identity_hex());
    assert_ne!(rebuilt.prepared.identities().measurement, prepared.identities().measurement);
    assert_ne!(rebuilt.prepared.identities().evidence, prepared.identities().evidence);
    // The binding and representation identities are untouched.
    assert_eq!(rebuilt.prepared.identities().binding, prepared.identities().binding);
    assert_eq!(rebuilt.prepared.identities().representation, prepared.identities().representation);
    assert_eq!(rebuilt.estimate.model_identity(), rebuilt.prepared.model().identities().model);
    assert_ne!(rebuilt.estimate.point().to_bits(), before.point().to_bits());
    // Parameters of another route are not a validation update.
    let refusal = prepared
        .with_validation_parameters(
            MeasurementParameters::Binary {
                rates: vec![rates(None, 0.9, 0.8, 0.0, 0.0)],
                min_youden: 0.1,
            },
            ParameterEvidence::Declared { declarer: "test".into() },
            &ctx(),
        )
        .unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    // The original is untouched.
    assert_eq!(prepared.estimate(&ctx()).unwrap().record(), before.record());
}

#[test]
fn cancellation_stops_before_a_claim() {
    let sim = simulate(200, 5);
    let prepared = prepare_classical(&classical_model(0.6, 0.0), &sim);
    let context = ctx();
    context.cancellation.cancel();
    let refusal = prepared.estimate(&context).unwrap_err();
    assert_eq!(refusal.code, "cancelled_no_claim");
    let refusal = prepared.refresh(classical_input(&sim), &context).unwrap_err();
    assert_eq!(refusal.code, "cancelled_no_claim");
    let refusal = StudyBuilder::measurement_model(
        classical_model(0.6, 0.0),
        binding(),
        classical_input(&sim),
        None,
        &context,
    )
    .unwrap_err();
    assert_eq!(refusal.code, "cancelled_no_claim");
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

#[test]
fn artifact_replays_independently_and_refuses_semantic_mutation() {
    let sim = simulate(3_000, 8);
    let m = classical_model(0.6, 0.05);
    let prepared = prepare_classical(&m, &sim);
    let estimate = prepared.estimate(&ctx()).unwrap();
    let bytes = estimate.export("replay-1").unwrap();
    let identity = prepared.model_identity_hex();

    let consumed = consume_measurement_model_artifact(&bytes, &identity).unwrap();
    assert_eq!(consumed.record(), Some(estimate.record()));
    let replayed = consumed.replayed().unwrap();
    assert!((replayed.point - estimate.point()).abs() < 1e-10);
    assert!((replayed.variance.sqrt() - estimate.standard_error().unwrap()).abs() < 1e-10);
    assert!(consumed.covers_data_identity(&estimate.record().data_identity));
    assert_eq!(consumed.model(), prepared.model());
    // The consumer applies its own binding to the stored result.
    assert!(consumed
        .check_use(&binding(), MeasurementOperation::LinearAdjustedEffect)
        .is_ok());
    let mut elsewhere = binding();
    elsewhere.population_id = "elsewhere".into();
    let refusal = consumed
        .check_use(&elsewhere, MeasurementOperation::LinearAdjustedEffect)
        .unwrap_err();
    assert_eq!(refusal.detail, "measurement_model.population_mismatch");

    // Semantic mutations are different identities and are refused; the
    // original result cannot be attached to a mutated model either.
    let mut moved = m.clone();
    moved.binding.population_id = "elsewhere".into();
    let mut changed = m.clone();
    changed.parameters = MeasurementParameters::ClassicalError {
        error_variance: 0.7,
        error_variance_standard_error: 0.05,
        min_reliability: 0.1,
    };
    let mut versioned = m.clone();
    versioned.representation.version = 2;
    let result_of_original = consumed.record().unwrap().clone();
    for mutated in [moved, changed, versioned] {
        let forged =
            MeasurementModelArtifact::new(&mutated, None).unwrap().to_bytes("forged").unwrap();
        let refusal = consume_measurement_model_artifact(&forged, &identity).unwrap_err();
        assert_eq!(refusal.code, "measurement_binding_mismatch");
        assert_ne!(mutated.identities().model, result_of_original.model_identity);
    }
    // A flipped byte is refused by the container.
    let mut damaged = bytes.clone();
    let at = damaged.len() / 2;
    damaged[at] ^= 0x55;
    assert!(consume_measurement_model_artifact(&damaged, &identity).is_err());
    // A malformed retained identity is refused, not ignored.
    assert!(consume_measurement_model_artifact(&bytes, "not-an-identity").is_err());
}

#[test]
fn every_route_exports_a_result_a_consumer_recomputes() {
    // Binary prevalence.
    let m = binary_model(
        vec![rates(None, 0.9, 0.8, 0.0001, 0.0004)],
        ErrorStructure::NonDifferential,
    );
    let prepared = StudyBuilder::measurement_model(
        m,
        binding(),
        MeasurementInput::BinaryPrevalence { events: 410, total: 1000 },
        None,
        &ctx(),
    )
    .unwrap();
    let estimate = prepared.estimate(&ctx()).unwrap();
    let bytes = estimate.export("binary").unwrap();
    let consumed =
        consume_measurement_model_artifact(&bytes, &prepared.model_identity_hex()).unwrap();
    assert!((consumed.replayed().unwrap().point - 0.3).abs() < 1e-12);

    // Categorical contrast.
    let m = categorical_model(None);
    let prepared = StudyBuilder::measurement_model(
        m,
        binding(),
        MeasurementInput::Categorical {
            classes: exact_classes(),
            target: CategoricalTarget::OutcomeContrast {
                first: "high".into(),
                second: "low".into(),
            },
        },
        None,
        &ctx(),
    )
    .unwrap();
    let estimate = prepared.estimate(&ctx()).unwrap();
    let bytes = estimate.export("categorical").unwrap();
    let consumed =
        consume_measurement_model_artifact(&bytes, &prepared.model_identity_hex()).unwrap();
    assert!((consumed.replayed().unwrap().point - 3.0).abs() < 1e-6);
    assert_eq!(consumed.record().unwrap().route, MeasurementRoute::ProbabilisticCategorical);
}
