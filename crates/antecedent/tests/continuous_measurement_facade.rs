//! Facade integration of continuous classical measurement error: a prepared
//! measurement model corrects a linear or logistic treatment coefficient by
//! regression calibration or SIMEX, against simulated data with a known
//! coefficient; the nonlinear outcome operation stays refused by
//! `check_operation` and names the continuous entry point; interval and
//! export refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::continuous_measurement::{
    ContinuousCorrectionSpec, CorrectionMethod, ErrorLocation, OutcomeLink, SimexExtrapolant,
};
use antecedent::{
    ContinuousErrorInput, MeasurementAnalysisBinding, MeasurementInput, PreparedMeasurementModel,
    StudyBuilder,
};
use antecedent_contract::measurement::MeasurementOperation;
use antecedent_core::measurement_model::{
    ArmBinding, ErrorStructure, MeasurementBinding, MeasurementInference, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{CausalRng, ExecutionContext};

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

fn model(route: MeasurementRoute, parameters: MeasurementParameters) -> MeasurementModel {
    MeasurementModel {
        model_id: "facade-continuous".into(),
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

fn classical(error_variance: f64) -> MeasurementModel {
    model(
        MeasurementRoute::ClassicalContinuousError,
        MeasurementParameters::ClassicalError {
            error_variance,
            error_variance_standard_error: 0.0,
            min_reliability: 0.1,
        },
    )
}

/// `X ~ N(0, 1)`, `W = X + N(0, 0.25)`; linear `Y = 1 + 2 X + N(0, 0.25)` or
/// logistic `P(Y = 1) = sigmoid(-0.5 + X)`.
fn simulate(n: usize, seed: u64, logistic: bool) -> (Vec<f64>, Vec<f64>) {
    let mut rng = CausalRng::from_seed(seed);
    let (mut y, mut w) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let x = normal(&mut rng);
        w.push(x + 0.5 * normal(&mut rng));
        if logistic {
            let p = 1.0 / (1.0 + (0.5 - x).exp());
            y.push(if rng.next_f64() < p { 1.0 } else { 0.0 });
        } else {
            y.push(1.0 + 2.0 * x + 0.5 * normal(&mut rng));
        }
    }
    (y, w)
}

fn prepare(m: &MeasurementModel, y: &[f64], w: &[f64]) -> PreparedMeasurementModel {
    let input = MeasurementInput::ClassicalLinear {
        outcome: y.to_vec(),
        regressor: w.to_vec(),
        covariates: Vec::new(),
        covariate_count: 0,
        replicate_counts: None,
    };
    StudyBuilder::measurement_model(m.clone(), binding(), input, None, &ctx()).unwrap()
}

fn input(
    y: &[f64],
    w: &[f64],
    link: OutcomeLink,
    method: CorrectionMethod,
) -> ContinuousErrorInput {
    ContinuousErrorInput {
        outcome: y.to_vec(),
        regressor: w.to_vec(),
        covariates: Vec::new(),
        covariate_count: 0,
        replicate_values: None,
        replicate_counts: None,
        spec: ContinuousCorrectionSpec {
            location: ErrorLocation::Treatment,
            link,
            method,
            bootstrap_replicates: 50,
            seed: 31,
        },
    }
}

fn simex() -> CorrectionMethod {
    CorrectionMethod::Simex {
        lambdas: vec![0.5, 1.0, 1.5, 2.0],
        replicates: 10,
        extrapolant: SimexExtrapolant::Quadratic,
    }
}

#[test]
fn logistic_calibration_and_linear_simex_recover_the_slope() {
    let (y, w) = simulate(4000, 41, true);
    let prepared = prepare(&classical(0.25), &y, &w);
    let logistic = prepared
        .estimate_continuous(
            input(&y, &w, OutcomeLink::Logit, CorrectionMethod::RegressionCalibration),
            &ctx(),
        )
        .unwrap();
    let naive = logistic.naive_point().unwrap();
    assert!((logistic.point() - 1.0).abs() < 0.15, "point {}", logistic.point());
    assert!((logistic.point() - 1.0).abs() < (naive - 1.0).abs());
    assert_eq!(logistic.record().functional_id, "logit_coefficient.regression_calibration");
    assert_eq!(logistic.inference(), MeasurementInference::UncalibratedStandardError);
    assert_eq!(
        logistic.summary().bootstrap_replicates + logistic.summary().failed_bootstrap_replicates,
        50
    );
    assert_eq!(logistic.model().identities().model, prepared.identities().model);

    let (y, w) = simulate(2000, 42, false);
    let prepared = prepare(&classical(0.25), &y, &w);
    let linear = prepared
        .estimate_continuous(input(&y, &w, OutcomeLink::Identity, simex()), &ctx())
        .unwrap();
    assert!((linear.point() - 2.0).abs() < 0.15, "point {}", linear.point());
    assert!((linear.naive_point().unwrap() - 1.6).abs() < 0.1);
    assert_eq!(linear.record().functional_id, "linear_coefficient.simex");
    assert!(linear.effect().simex.is_some());
}

#[test]
fn the_nonlinear_operation_stays_refused_and_names_the_continuous_entry_point() {
    let (y, w) = simulate(100, 43, false);
    let prepared = prepare(&classical(0.25), &y, &w);
    let refusal =
        prepared.check_operation(MeasurementOperation::NonlinearOutcomeModel).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert!(refusal.remedy.as_deref().is_some_and(|r| r.contains("estimate_continuous")));
}

#[test]
fn interval_and_export_refuse() {
    let (y, w) = simulate(500, 44, false);
    let prepared = prepare(&classical(0.25), &y, &w);
    let estimate = prepared
        .estimate_continuous(
            input(&y, &w, OutcomeLink::Identity, CorrectionMethod::RegressionCalibration),
            &ctx(),
        )
        .unwrap();
    assert_eq!(estimate.interval().unwrap_err().code, "cell_not_licensed");
    let export = estimate.export("continuous").unwrap_err();
    assert_eq!(export.code, "route_not_supported");
    assert_eq!(export.detail, "measurement_model.continuous_export");
}

#[test]
fn unidentified_error_variance_and_binary_outcome_error_carry_their_remedies() {
    let (y, w) = simulate(300, 45, true);
    let repeated = model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementParameters::Repeated { min_replicated_units: 2, min_reliability: 0.1 },
    );
    let ones = vec![1_u32; w.len()];
    let prepared = StudyBuilder::measurement_model(
        repeated,
        binding(),
        MeasurementInput::RepeatedLinear {
            outcome: y.clone(),
            replicate_values: w.clone(),
            replicate_counts: ones,
            covariates: Vec::new(),
            covariate_count: 0,
        },
        None,
        &ctx(),
    )
    .unwrap();
    let refusal = prepared
        .estimate_continuous(
            input(&y, &w, OutcomeLink::Identity, CorrectionMethod::RegressionCalibration),
            &ctx(),
        )
        .unwrap_err();
    assert_eq!(refusal.code, "measurement_not_identified");
    assert_eq!(refusal.remedy.as_deref(), Some("supply replicates or a validation sample"));
    assert!(refusal.message.ends_with("next attempt: supply replicates or a validation sample"));

    let classical_prepared = prepare(&classical(0.25), &y, &w);
    let mut outcome_error =
        input(&y, &w, OutcomeLink::Logit, CorrectionMethod::RegressionCalibration);
    outcome_error.spec.location = ErrorLocation::Outcome;
    let refusal = classical_prepared.estimate_continuous(outcome_error, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert!(refusal.remedy.as_deref().is_some_and(|r| r.contains("binary misclassification")));
}
