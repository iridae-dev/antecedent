//! Continuous classical measurement error: regression calibration and SIMEX
//! for linear and logistic outcome models, against simulated data with a
//! known coefficient.
//!
//! Oracles are the data-generating coefficients and the closed-form
//! attenuation factor `Var(X | Z) / (Var(X | Z) + sigma_u^2)`; tolerances are
//! written in reported standard errors or as absolute bands stated per test.
//! Simulations use fixed seeds.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroU32;

use antecedent_core::execution::AdmissionPool;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, ParameterUncertainty, RepresentationVersion,
};
use antecedent_core::{CausalRng, ExecutionContext, Parallelism};
use antecedent_estimate::measurement::continuous::{
    MIN_BOOTSTRAP_REPLICATES, SUPPLY_REPLICATES_OR_VALIDATION, USE_BINARY_MISCLASSIFICATION,
};
use antecedent_estimate::measurement::{
    ContinuousCorrectionSpec, ContinuousErrorData, ContinuousMethodKind, CorrectedContinuousEffect,
    CorrectionMethod, DataBinding, ErrorLocation, MeasurementError, OutcomeLink, SimexExtrapolant,
    correct_continuous_error,
};

fn normal(rng: &mut CausalRng) -> f64 {
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn sigmoid(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn scalar() -> RepresentationVersion {
    RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 }
}

fn data_binding() -> DataBinding {
    DataBinding {
        population_id: "cohort".into(),
        regime_id: "observational".into(),
        arm_id: None,
        representation: scalar(),
    }
}

fn model(route: MeasurementRoute, parameters: MeasurementParameters) -> MeasurementModel {
    MeasurementModel {
        model_id: "continuous-model".into(),
        route,
        indicator: MeasurementVariable {
            id: "schema:indicator".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "unit".into(),
        },
        construct: MeasurementVariable {
            id: "construct:latent".into(),
            kind: MeasurementKind::LatentScientific,
            units: "unit".into(),
        },
        binding: MeasurementBinding {
            population_id: "cohort".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters,
        representation: scalar(),
        evidence: ParameterEvidence::Declared { declarer: "test".into() },
    }
}

fn classical(error_variance: f64, standard_error: f64) -> MeasurementModel {
    model(
        MeasurementRoute::ClassicalContinuousError,
        MeasurementParameters::ClassicalError {
            error_variance,
            error_variance_standard_error: standard_error,
            min_reliability: 0.1,
        },
    )
}

fn spec(
    location: ErrorLocation,
    link: OutcomeLink,
    method: CorrectionMethod,
) -> ContinuousCorrectionSpec {
    ContinuousCorrectionSpec { location, link, method, bootstrap_replicates: 50, seed: 2024 }
}

fn simex(replicates: u32) -> CorrectionMethod {
    CorrectionMethod::Simex {
        lambdas: vec![0.5, 1.0, 1.5, 2.0],
        replicates,
        extrapolant: SimexExtrapolant::Quadratic,
    }
}

/// `X ~ N(0, 1)`, `W = X + N(0, 0.25)` (reliability 0.8), `Z ~ N(0, 1)`.
/// Linear: `Y = 1 + 2 X + 0.5 Z + N(0, 0.25)`. Logistic: `P(Y = 1) =
/// sigmoid(-0.5 + X + 0.3 Z)`.
struct Sim {
    y: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
}

fn simulate(n: usize, seed: u64, logistic: bool) -> Sim {
    let mut rng = CausalRng::from_seed(seed);
    let mut sim = Sim { y: Vec::with_capacity(n), w: Vec::with_capacity(n), z: Vec::new() };
    for _ in 0..n {
        let x = normal(&mut rng);
        let z = normal(&mut rng);
        sim.w.push(x + 0.5 * normal(&mut rng));
        sim.z.push(z);
        if logistic {
            let p = sigmoid(-0.5 + x + 0.3 * z);
            sim.y.push(if rng.next_f64() < p { 1.0 } else { 0.0 });
        } else {
            sim.y.push(1.0 + 2.0 * x + 0.5 * z + 0.5 * normal(&mut rng));
        }
    }
    sim
}

fn data(sim: &Sim) -> ContinuousErrorData<'_> {
    ContinuousErrorData {
        outcome: &sim.y,
        regressor: &sim.w,
        covariates: &sim.z,
        covariate_count: 1,
        replicate_values: None,
        replicate_counts: None,
    }
}

fn run(
    m: &MeasurementModel,
    data: &ContinuousErrorData<'_>,
    spec: &ContinuousCorrectionSpec,
    ctx: &ExecutionContext,
) -> Result<CorrectedContinuousEffect, MeasurementError> {
    correct_continuous_error(m, &data_binding(), data, spec, ctx)
}

fn assert_recovers(effect: &CorrectedContinuousEffect, truth: f64, band: f64) {
    let point = effect.record.point;
    let naive = effect.record.naive_point.unwrap();
    let se = effect.record.standard_error.unwrap();
    assert!(se > 0.0 && se.is_finite(), "se {se}");
    assert!((point - truth).abs() < band, "point {point}, truth {truth}");
    assert!((point - truth).abs() < 4.0 * se + 0.1 * truth.abs(), "point {point} se {se}");
    assert!((point - truth).abs() < (naive - truth).abs(), "point {point} naive {naive}");
    assert!(!effect.record.has_interval());
}

#[test]
fn linear_calibration_recovers_the_attenuated_slope() {
    let sim = simulate(2000, 11, false);
    let effect = run(
        &classical(0.25, 0.0),
        &data(&sim),
        &spec(
            ErrorLocation::Treatment,
            OutcomeLink::Identity,
            CorrectionMethod::RegressionCalibration,
        ),
        &ctx(),
    )
    .unwrap();
    // Observed 2.006 against the true 2 (the naive slope is 1.6): a band of 5 observed errors.
    assert_recovers(&effect, 2.0, 0.03);
    let naive = effect.record.naive_point.unwrap();
    assert!((naive - 1.6).abs() < 0.1, "naive {naive}");
    assert_eq!(effect.record.functional_id, "linear_coefficient.regression_calibration");
    assert_eq!(effect.summary.method, ContinuousMethodKind::RegressionCalibration);
    assert!((effect.summary.reliability.unwrap() - 0.8).abs() < 0.05);
    assert_eq!(effect.summary.bootstrap_replicates, 0);
    assert_eq!(effect.record.parameter_uncertainty, ParameterUncertainty::DeclaredExact);
}

#[test]
fn linear_simex_recovers_the_attenuated_slope() {
    let sim = simulate(2000, 12, false);
    let effect = run(
        &classical(0.25, 0.0),
        &data(&sim),
        &spec(ErrorLocation::Treatment, OutcomeLink::Identity, simex(20)),
        &ctx(),
    )
    .unwrap();
    // The quadratic extrapolant under-corrects a rational path: at
    // reliability 0.8 its large-sample limit is 1.94, not 2 (observed 1.960).
    assert_recovers(&effect, 2.0, 0.08);
    assert_eq!(effect.record.functional_id, "linear_coefficient.simex");
    let path = effect.simex.as_ref().unwrap();
    assert_eq!(path.lambdas, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    assert!(path.lambda_means.windows(2).all(|w| w[1][1] < w[0][1]));
    assert_eq!(
        effect.summary.bootstrap_replicates + effect.summary.failed_bootstrap_replicates,
        50
    );
    assert_eq!(
        u32::try_from(effect.bootstrap_estimates.len()).unwrap(),
        effect.summary.bootstrap_replicates
    );
}

#[test]
fn logistic_calibration_and_simex_recover_the_slope() {
    let sim = simulate(4000, 13, true);
    let m = classical(0.25, 0.0);
    let rc = run(
        &m,
        &data(&sim),
        &spec(
            ErrorLocation::Treatment,
            OutcomeLink::Logit,
            CorrectionMethod::RegressionCalibration,
        ),
        &ctx(),
    )
    .unwrap();
    assert_recovers(&rc, 1.0, 0.15);
    assert_eq!(rc.record.functional_id, "logit_coefficient.regression_calibration");
    let sx = run(
        &m,
        &data(&sim),
        &spec(ErrorLocation::Treatment, OutcomeLink::Logit, simex(10)),
        &ctx(),
    )
    .unwrap();
    assert_recovers(&sx, 1.0, 0.15);
    assert_eq!(sx.record.functional_id, "logit_coefficient.simex");
}

#[test]
fn declared_error_variance_uncertainty_is_split_out_of_the_bootstrap() {
    let sim = simulate(1500, 14, true);
    let effect = run(
        &classical(0.25, 0.05),
        &data(&sim),
        &spec(
            ErrorLocation::Treatment,
            OutcomeLink::Logit,
            CorrectionMethod::RegressionCalibration,
        ),
        &ctx(),
    )
    .unwrap();
    assert_eq!(effect.record.parameter_uncertainty, ParameterUncertainty::Propagated);
    let parts = &effect.record.variance_components;
    assert_eq!(parts.len(), 2);
    let sum: f64 = parts.iter().map(|c| c.variance).sum();
    let se = effect.record.standard_error.unwrap();
    assert!((sum - se * se).abs() < 1e-12 * (1.0 + se * se));
    assert!(parts[1].variance > 0.0, "{parts:?}");
    assert!((effect.summary.error_variance_standard_error.unwrap() - 0.05).abs() < 1e-12);
}

#[test]
fn error_in_a_covariate_corrects_the_treatment_coefficient() {
    // X ~ N(0,1), T = 0.6 X + N(0,1), W = X + N(0, 0.5), Y = 1 + T + 2 X + N(0,1).
    // Adjusting for W instead of X leaves confounding the treatment absorbs.
    let mut rng = CausalRng::from_seed(15);
    let n = 3000;
    let (mut y, mut w, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let x = normal(&mut rng);
        let ti = 0.6 * x + normal(&mut rng);
        w.push(x + 0.5_f64.sqrt() * normal(&mut rng));
        t.push(ti);
        y.push(1.0 + ti + 2.0 * x + normal(&mut rng));
    }
    let data = ContinuousErrorData {
        outcome: &y,
        regressor: &w,
        covariates: &t,
        covariate_count: 1,
        replicate_values: None,
        replicate_counts: None,
    };
    let effect = run(
        &classical(0.5, 0.0),
        &data,
        &spec(
            ErrorLocation::Covariate { treatment_column: 0 },
            OutcomeLink::Identity,
            CorrectionMethod::RegressionCalibration,
        ),
        &ctx(),
    )
    .unwrap();
    assert_eq!(effect.coefficient_index, 2);
    assert_recovers(&effect, 1.0, 0.05);
    assert!(effect.record.naive_point.unwrap() > 1.2, "naive should absorb confounding");
    let out_of_range = run(
        &classical(0.5, 0.0),
        &data,
        &spec(
            ErrorLocation::Covariate { treatment_column: 3 },
            OutcomeLink::Identity,
            CorrectionMethod::RegressionCalibration,
        ),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(out_of_range.code, "invalid_argument");
}

#[test]
fn repeated_measurements_identify_the_error_variance() {
    let mut rng = CausalRng::from_seed(16);
    let n = 1500;
    let (mut y, mut values, mut counts) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let x = normal(&mut rng);
        y.push(1.0 + 2.0 * x + normal(&mut rng));
        let k = 2 + u32::from(i % 2 == 1);
        counts.push(k);
        for _ in 0..k {
            values.push(x + 0.5_f64.sqrt() * normal(&mut rng));
        }
    }
    let m = model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementParameters::Repeated { min_replicated_units: 50, min_reliability: 0.1 },
    );
    let data = ContinuousErrorData {
        outcome: &y,
        regressor: &[],
        covariates: &[],
        covariate_count: 0,
        replicate_values: Some(&values),
        replicate_counts: Some(&counts),
    };
    let rc = run(
        &m,
        &data,
        &spec(
            ErrorLocation::Treatment,
            OutcomeLink::Identity,
            CorrectionMethod::RegressionCalibration,
        ),
        &ctx(),
    )
    .unwrap();
    assert_recovers(&rc, 2.0, 0.03);
    assert!((rc.summary.error_variance.unwrap() - 0.5).abs() < 0.05);
    assert!(rc.replicate_components.is_some());
    assert_eq!(rc.record.parameter_uncertainty, ParameterUncertainty::Propagated);
    let sx =
        run(&m, &data, &spec(ErrorLocation::Treatment, OutcomeLink::Identity, simex(10)), &ctx())
            .unwrap();
    assert_recovers(&sx, 2.0, 0.06);
}

#[test]
fn outcome_error_under_an_identity_link_passes_through() {
    // Y* = Y + N(0, 1): the slope on the error-free treatment is unbiased.
    let mut rng = CausalRng::from_seed(17);
    let n = 2000;
    let (mut y, mut t) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let ti = normal(&mut rng);
        t.push(ti);
        y.push(1.0 + 1.5 * ti + 0.5 * normal(&mut rng) + normal(&mut rng));
    }
    let data = ContinuousErrorData {
        outcome: &y,
        regressor: &t,
        covariates: &[],
        covariate_count: 0,
        replicate_values: None,
        replicate_counts: None,
    };
    let effect = run(
        &classical(1.0, 0.0),
        &data,
        &spec(
            ErrorLocation::Outcome,
            OutcomeLink::Identity,
            CorrectionMethod::RegressionCalibration,
        ),
        &ctx(),
    )
    .unwrap();
    // Oracle: the simple least-squares slope.
    let nf = f64::from(n);
    let (tb, yb) = (t.iter().sum::<f64>() / nf, y.iter().sum::<f64>() / nf);
    let sxy: f64 = t.iter().zip(&y).map(|(a, b)| (a - tb) * (b - yb)).sum();
    let sxx: f64 = t.iter().map(|a| (a - tb) * (a - tb)).sum();
    assert!((effect.record.point - sxy / sxx).abs() < 1e-10);
    assert!((effect.record.point - effect.record.naive_point.unwrap()).abs() < 1e-12);
    assert!((effect.record.point - 1.5).abs() < 4.0 * effect.record.standard_error.unwrap());
    assert_eq!(effect.summary.method, ContinuousMethodKind::NoCorrectionNeeded);
    assert_eq!(effect.record.functional_id, "linear_coefficient.outcome_error_no_correction");
    assert_eq!(effect.summary.error_variance, None);
}

#[test]
fn a_logit_outcome_with_continuous_error_is_refused() {
    let sim = simulate(500, 18, true);
    let err = run(
        &classical(0.25, 0.0),
        &data(&sim),
        &spec(ErrorLocation::Outcome, OutcomeLink::Logit, CorrectionMethod::RegressionCalibration),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, "route_not_supported");
    assert_eq!(err.detail, "measurement_model.binary_outcome_continuous_error");
    assert!(err.message.ends_with(USE_BINARY_MISCLASSIFICATION.sentence()), "{}", err.message);
}

#[test]
fn an_unidentified_error_variance_is_refused_with_the_next_attempt() {
    let sim = simulate(500, 19, false);
    let rc = spec(
        ErrorLocation::Treatment,
        OutcomeLink::Identity,
        CorrectionMethod::RegressionCalibration,
    );
    let suffix = format!("next attempt: {}", SUPPLY_REPLICATES_OR_VALIDATION.sentence());

    // A route that carries no error variance for a continuous column.
    let binary = model(
        MeasurementRoute::BinaryMisclassification,
        MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.9,
                specificity: 0.9,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
    );
    let err = run(&binary, &data(&sim), &rc, &ctx()).unwrap_err();
    assert_eq!(err.code, "measurement_not_identified");
    assert_eq!(err.detail, "measurement_model.error_variance_not_identified");
    assert!(err.message.ends_with(&suffix), "{}", err.message);

    // The repeated route with no replicates, and with no replicated unit.
    let repeated = model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementParameters::Repeated { min_replicated_units: 2, min_reliability: 0.1 },
    );
    let err = run(&repeated, &data(&sim), &rc, &ctx()).unwrap_err();
    assert_eq!(err.code, "measurement_not_identified");
    assert!(err.message.ends_with(&suffix), "{}", err.message);
    let ones = vec![1_u32; sim.w.len()];
    let singles = ContinuousErrorData {
        replicate_values: Some(&sim.w),
        replicate_counts: Some(&ones),
        ..data(&sim)
    };
    let err = run(&repeated, &singles, &rc, &ctx()).unwrap_err();
    assert_eq!(err.code, "measurement_not_identified");
    assert_eq!(err.detail, "measurement_model.insufficient_replicates");
    assert!(err.message.ends_with(&suffix), "{}", err.message);
}

#[test]
fn the_bootstrap_is_bitwise_reproducible_across_thread_counts() {
    let sim = simulate(800, 20, true);
    let m = classical(0.25, 0.02);
    let request = spec(ErrorLocation::Treatment, OutcomeLink::Logit, simex(5));
    let serial = run(&m, &data(&sim), &request, &ctx()).unwrap();
    let again = run(&m, &data(&sim), &request, &ctx()).unwrap();
    assert_eq!(serial, again);
    let mut wide = ctx();
    wide.parallelism = Parallelism::bounded(NonZeroU32::new(4).unwrap());
    let parallel = run(&m, &data(&sim), &request, &wide).unwrap();
    assert_eq!(serial, parallel);
    let reseeded = ContinuousCorrectionSpec { seed: 7, ..request };
    let other = run(&m, &data(&sim), &reseeded, &ctx()).unwrap();
    assert_ne!(serial.bootstrap_estimates, other.bootstrap_estimates);
}

#[test]
fn an_unfunded_bootstrap_is_refused_before_allocation() {
    let sim = simulate(500, 21, true);
    let tiny = ExecutionContext::for_tests(1).with_admission(AdmissionPool::fixed(1, 64));
    let err = run(
        &classical(0.25, 0.0),
        &data(&sim),
        &spec(
            ErrorLocation::Treatment,
            OutcomeLink::Logit,
            CorrectionMethod::RegressionCalibration,
        ),
        &tiny,
    )
    .unwrap_err();
    assert_eq!(err.code, "invalid_argument");
    assert_eq!(err.detail, "measurement_model.admission_refused");
}

#[test]
fn too_few_bootstrap_replicates_are_refused() {
    let sim = simulate(500, 22, true);
    let request = ContinuousCorrectionSpec {
        bootstrap_replicates: MIN_BOOTSTRAP_REPLICATES - 1,
        ..spec(
            ErrorLocation::Treatment,
            OutcomeLink::Logit,
            CorrectionMethod::RegressionCalibration,
        )
    };
    let err = run(&classical(0.25, 0.0), &data(&sim), &request, &ctx()).unwrap_err();
    assert_eq!(err.code, "invalid_argument");
}
