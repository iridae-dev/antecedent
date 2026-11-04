//! Corrected functionals under declared measurement models, against
//! independent truth.
//!
//! Every value asserted here has an oracle that is not the code under test:
//! the closed-form attenuation factor `lambda = Var(X | Z) / (Var(X | Z) +
//! Var(U))` of a simulated DGP with a known coefficient, the exact algebra of
//! misclassification (`p* = Se pi + (1 - Sp)(1 - pi)`), and the conjugate
//! normal posterior of a latent regressor. Simulations use fixed seeds and
//! predeclared tolerances written in terms of the reported standard error.

#![allow(
    clippy::needless_range_loop,
    reason = "simulation fixtures index parallel columns by row and draw"
)]

use antecedent_contract::measurement::{
    LatentPosteriorTrust, MeasurementOperation, MeasurementRequest, check_latent_posterior,
    check_measurement_request,
};
use antecedent_contract::{
    ExternalCapability, ExternalPosteriorKind, ExternalScientificObject, ExternalTrustState,
    PosteriorProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, DifferentialAxis, ErrorStructure, MeasurementBinding,
    MeasurementInference, MeasurementKind, MeasurementModel, MeasurementParameters,
    MeasurementRoute, MeasurementVariable, ParameterEvidence, ParameterUncertainty,
    PosteriorReference, RepresentationVersion,
};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_estimate::measurement::{
    CategoricalFunctional, DataBinding, LinearMeasuredData, PosteriorCombination,
    PosteriorLinearData, RiskDifferenceInput, correct_binary_prevalence,
    correct_binary_risk_difference, correct_categorical, correct_classical_linear_effect,
    correct_latent_posterior_linear_effect, correct_repeated_linear_effect,
};
use antecedent_io_core::measurement_model_wire::{
    MeasurementModelArtifact, MeasurementReplay, MeasurementResult, replay_values,
};
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

fn data_binding() -> DataBinding {
    DataBinding {
        population_id: "cohort".into(),
        regime_id: "observational".into(),
        arm_id: None,
        representation: scalar(),
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
        model_id: "test-model".into(),
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

fn binary_rates(arm: Option<&str>, se: f64, sp: f64, vse: f64, vsp: f64) -> BinaryArmRates {
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

fn linear_data(sim: &Simulated) -> LinearMeasuredData<'_> {
    LinearMeasuredData {
        outcome: &sim.y,
        regressor: &sim.w,
        covariates: &sim.z,
        covariate_count: 1,
        replicate_counts: None,
    }
}

// ---------------------------------------------------------------------------
// Classical continuous error
// ---------------------------------------------------------------------------

#[test]
fn classical_error_recovers_the_coefficient_and_the_naive_fit_is_attenuated() {
    let sim = simulate(20_000, 424_242);
    let m = classical_model(0.6, 0.0);
    let result =
        correct_classical_linear_effect(&m, &data_binding(), &linear_data(&sim), &ctx()).unwrap();
    let se = result.record.standard_error.unwrap();
    // Predeclared: the corrected coefficient is within 4 reported standard errors of 2.
    assert!((result.record.point - 2.0).abs() < 4.0 * se, "{} se {se}", result.record.point);
    assert!(se < 0.08, "se {se}");
    // Closed-form attenuation: Var(X | Z) = 1 - 0.25 / 1.25 = 0.8, lambda = 0.8 / 1.4.
    let lambda = 0.8 / 1.4;
    let naive = result.record.naive_point.unwrap();
    let naive_se = result.fit.naive_covariance[result.fit.dim + 1].sqrt();
    assert!((naive - 2.0 * lambda).abs() < 4.0 * naive_se, "naive {naive}");
    assert!((result.fit.reliability - lambda).abs() < 0.02, "{}", result.fit.reliability);
    // The naive estimate is demonstrably biased: far from truth in its own SEs.
    assert!((2.0 - naive) > 10.0 * naive_se);
    assert_eq!(result.record.model_identity, m.identities().model);
    assert_eq!(result.record.route, MeasurementRoute::ClassicalContinuousError);
    assert_eq!(result.record.inference, MeasurementInference::UncalibratedStandardError);
    assert_eq!(result.record.parameter_uncertainty, ParameterUncertainty::DeclaredExact);
    assert!(!result.record.has_interval());
}

#[test]
fn variance_includes_the_uncertainty_of_the_error_variance() {
    let sim = simulate(5_000, 7);
    let exact = correct_classical_linear_effect(
        &classical_model(0.6, 0.0),
        &data_binding(),
        &linear_data(&sim),
        &ctx(),
    )
    .unwrap();
    let uncertain = correct_classical_linear_effect(
        &classical_model(0.6, 0.06),
        &data_binding(),
        &linear_data(&sim),
        &ctx(),
    )
    .unwrap();
    assert_eq!(exact.record.point.to_bits(), uncertain.record.point.to_bits());
    let se_exact = exact.record.standard_error.unwrap();
    let se_uncertain = uncertain.record.standard_error.unwrap();
    assert!(se_uncertain > se_exact, "{se_uncertain} vs {se_exact}");
    assert_eq!(uncertain.record.parameter_uncertainty, ParameterUncertainty::Propagated);
    // The added variance is (d beta / d sigma^2)^2 Var(sigma^2) with
    // d beta / d sigma^2 = beta / (Var(W | Z) - sigma^2) = beta / Var(X | Z).
    let d = uncertain.fit.dim;
    let added =
        uncertain.fit.covariance[d + 1] - uncertain.fit.covariance_given_error_variance[d + 1];
    let slope = uncertain.record.point / 0.8;
    assert!((added - slope * slope * 0.06 * 0.06).abs() / added < 0.1, "added {added}");
    // A downstream linear functional uses the joint covariance, not the
    // sampling variance alone: shift of 1.5 units of the latent regressor.
    let (value, variance) = uncertain.downstream_linear_functional(&[0.0, 1.5, 0.0]).unwrap();
    assert!((value - 1.5 * uncertain.record.point).abs() < 1e-12);
    assert!((variance - 2.25 * uncertain.fit.covariance[d + 1]).abs() < 1e-12);
    assert!(variance > 2.25 * uncertain.fit.covariance_given_error_variance[d + 1]);
}

#[test]
fn row_order_does_not_change_the_correction() {
    let sim = simulate(2_000, 99);
    let m = classical_model(0.6, 0.03);
    let forward =
        correct_classical_linear_effect(&m, &data_binding(), &linear_data(&sim), &ctx()).unwrap();
    let rev = Simulated {
        y: sim.y.iter().rev().copied().collect(),
        w: sim.w.iter().rev().copied().collect(),
        z: sim.z.iter().rev().copied().collect(),
    };
    let backward =
        correct_classical_linear_effect(&m, &data_binding(), &linear_data(&rev), &ctx()).unwrap();
    assert!((forward.record.point - backward.record.point).abs() < 1e-9);
    let (a, b) = (forward.record.standard_error.unwrap(), backward.record.standard_error.unwrap());
    assert!((a - b).abs() < 1e-9);
    // Same inputs, same bits.
    let again =
        correct_classical_linear_effect(&m, &data_binding(), &linear_data(&sim), &ctx()).unwrap();
    assert_eq!(forward.record, again.record);
}

#[test]
fn reliability_at_or_below_the_floor_refuses_with_a_registered_code() {
    let sim = simulate(500, 5);
    // Var(W | Z) is about 1.6; an error variance of 5 leaves no latent variance.
    let m = classical_model(5.0, 0.0);
    let err = correct_classical_linear_effect(&m, &data_binding(), &linear_data(&sim), &ctx())
        .unwrap_err();
    assert_eq!(err.code, "measurement_not_identified");
    assert_eq!(
        err.detail,
        antecedent_core::reason_details::measurement_model::RELIABILITY_TOO_SMALL
    );
}

// ---------------------------------------------------------------------------
// Repeated measurements
// ---------------------------------------------------------------------------

#[test]
fn repeated_measurements_estimate_the_icc_and_recover_the_coefficient() {
    // X ~ N(0,1), replicates W_ij = X_i + N(0, 0.5): ICC = 1 / 1.5.
    let n = 6_000;
    let mut rng = CausalRng::from_seed(2024);
    let mut y = Vec::new();
    let mut counts = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        let x = normal(&mut rng);
        y.push(1.0 + 1.5 * x + normal(&mut rng));
        let k = 2 + u32::from(i % 2 == 1); // 2 or 3 replicates
        counts.push(k);
        for _ in 0..k {
            values.push(x + 0.5_f64.sqrt() * normal(&mut rng));
        }
    }
    let m = model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementKind::NoisyMisclassified,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Repeated { min_replicated_units: 50, min_reliability: 0.1 },
    );
    let result =
        correct_repeated_linear_effect(&m, &data_binding(), &y, &values, &counts, &[], 0, &ctx())
            .unwrap();
    let components = result.components.as_ref().unwrap();
    assert!((components.intraclass_correlation - 1.0 / 1.5).abs() < 0.03);
    assert!((components.within_variance - 0.5).abs() < 0.03);
    let se = result.record.standard_error.unwrap();
    assert!((result.record.point - 1.5).abs() < 4.0 * se, "{} se {se}", result.record.point);
    // Mean of 2.5 replicates still attenuates: the naive slope is below truth.
    assert!(result.record.naive_point.unwrap() < 1.5 - 5.0 * se);
    assert_eq!(result.record.parameter_uncertainty, ParameterUncertainty::Propagated);
}

#[test]
fn too_few_replicated_units_refuse_with_measurement_not_identified() {
    let n = 200;
    let mut rng = CausalRng::from_seed(5);
    let mut y = Vec::new();
    let mut counts = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        let x = normal(&mut rng);
        y.push(1.0 + 1.5 * x + normal(&mut rng));
        let k = 2 + u32::from(i % 2 == 1);
        counts.push(k);
        for _ in 0..k {
            values.push(x + 0.5_f64.sqrt() * normal(&mut rng));
        }
    }
    let few = model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementKind::NoisyMisclassified,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Repeated { min_replicated_units: 10_000, min_reliability: 0.1 },
    );
    let err =
        correct_repeated_linear_effect(&few, &data_binding(), &y, &values, &counts, &[], 0, &ctx())
            .unwrap_err();
    assert_eq!(err.code, "measurement_not_identified");
    assert_eq!(
        err.detail,
        antecedent_core::reason_details::measurement_model::INSUFFICIENT_REPLICATES
    );
}

// ---------------------------------------------------------------------------
// Binary misclassification
// ---------------------------------------------------------------------------

#[test]
fn binary_prevalence_inverts_the_misclassification_algebra() {
    // pi = 0.3, Se = 0.9, Sp = 0.8: p* = 0.27 + 0.14 = 0.41.
    let m = binary_model(
        vec![binary_rates(None, 0.9, 0.8, 0.0001, 0.0004)],
        ErrorStructure::NonDifferential,
    );
    let result = correct_binary_prevalence(&m, &data_binding(), 410, 1000, &ctx()).unwrap();
    assert!((result.record.point - 0.3).abs() < 1e-12);
    assert!((result.record.naive_point.unwrap() - 0.41).abs() < 1e-15);
    let expect = (0.41 * 0.59 / 1000.0 + 0.49 * 0.0004 + 0.09 * 0.0001) / 0.49;
    let se = result.record.standard_error.unwrap();
    assert!((se * se - expect).abs() < 1e-12);
    // The variance carries the sensitivity/specificity uncertainty.
    let rate_part = result
        .record
        .variance_components
        .iter()
        .find(|c| c.name == "sensitivity_specificity_uncertainty")
        .unwrap();
    assert!((rate_part.variance - (0.49 * 0.0004 + 0.09 * 0.0001) / 0.49).abs() < 1e-12);
    assert_eq!(result.record.parameter_uncertainty, ParameterUncertainty::Propagated);
}

#[test]
fn binary_risk_difference_recovers_truth_and_pooling_differential_error_fails() {
    // Exposed arm: Se 0.9 / Sp 0.8; unexposed: Se 0.7 / Sp 0.95; truth 0.4 vs 0.2.
    // Observed 0.9*0.4 + 0.2*0.6 = 0.48 and 0.7*0.2 + 0.05*0.8 = 0.18.
    let table = TwoByTwoCounts {
        exposed_events: 480,
        exposed_total: 1000,
        unexposed_events: 180,
        unexposed_total: 1000,
    };
    let input =
        RiskDifferenceInput { exposed_arm_id: "treated", unexposed_arm_id: "control", table };
    let differential = binary_model(
        vec![
            binary_rates(Some("treated"), 0.9, 0.8, 0.0, 0.0),
            binary_rates(Some("control"), 0.7, 0.95, 0.0, 0.0),
        ],
        ErrorStructure::Differential(DifferentialAxis::Arm),
    );
    let right =
        correct_binary_risk_difference(&differential, &data_binding(), &input, &ctx()).unwrap();
    assert!(right.differential);
    assert!((right.record.point - 0.2).abs() < 1e-12);
    assert!((right.record.naive_point.unwrap() - 0.3).abs() < 1e-12);

    // The non-differential route with the treated arm's rates applied to both
    // arms is a different, wrong answer: it is never reached from the
    // differential model, and run explicitly it misses the truth.
    let pooled =
        binary_model(vec![binary_rates(None, 0.9, 0.8, 0.0, 0.0)], ErrorStructure::NonDifferential);
    let wrong = correct_binary_risk_difference(&pooled, &data_binding(), &input, &ctx()).unwrap();
    assert!(!wrong.differential);
    assert!((wrong.record.point - 0.3 / 0.7).abs() < 1e-12);
    assert!((wrong.record.point - 0.2).abs() > 0.2);
    assert_ne!(right.record.model_identity, wrong.record.model_identity);

    // An arm the differential model has no rates for refuses; it never falls
    // back to another arm's rates.
    let unknown =
        RiskDifferenceInput { exposed_arm_id: "treated", unexposed_arm_id: "placebo", table };
    let err = correct_binary_risk_difference(&differential, &data_binding(), &unknown, &ctx())
        .unwrap_err();
    assert_eq!(err.detail, antecedent_core::reason_details::measurement_model::ARM_MISMATCH);
}

#[test]
fn shared_rates_keep_the_arm_covariance_that_split_rates_miss() {
    let table = TwoByTwoCounts {
        exposed_events: 480,
        exposed_total: 1000,
        unexposed_events: 340,
        unexposed_total: 1000,
    };
    let input =
        RiskDifferenceInput { exposed_arm_id: "treated", unexposed_arm_id: "control", table };
    let shared = binary_model(
        vec![binary_rates(None, 0.9, 0.8, 0.0004, 0.0009)],
        ErrorStructure::NonDifferential,
    );
    let r = correct_binary_risk_difference(&shared, &data_binding(), &input, &ctx()).unwrap();
    assert!((r.record.point - 0.2).abs() < 1e-12);
    // Hand calculation: sampling [0.48*0.52 + 0.34*0.66]/1000 / 0.49,
    // rates 0.2^2 (0.0004 + 0.0009) / 0.49.
    let sampling = (0.48 * 0.52 / 1000.0 + 0.34 * 0.66 / 1000.0) / 0.49;
    let rates = 0.04 * 0.0013 / 0.49;
    let se = r.record.standard_error.unwrap();
    assert!((se * se - (sampling + rates)).abs() < 1e-12);
}

#[test]
fn youden_index_at_or_below_the_minimum_refuses() {
    for (se, sp) in [(0.5, 0.5), (0.3, 0.6), (0.55, 0.5)] {
        let m = binary_model(
            vec![binary_rates(None, se, sp, 0.0, 0.0)],
            ErrorStructure::NonDifferential,
        );
        let err = correct_binary_prevalence(&m, &data_binding(), 40, 100, &ctx()).unwrap_err();
        assert_eq!(err.code, "measurement_not_identified");
        assert_eq!(
            err.detail,
            antecedent_core::reason_details::measurement_model::YOUDEN_TOO_SMALL
        );
    }
}

// ---------------------------------------------------------------------------
// Probabilistic categorical
// ---------------------------------------------------------------------------

#[test]
fn categorical_inversion_recovers_proportions_and_outcome_means() {
    // Rows are the observed label, columns the true category.
    let c = vec![
        0.8, 0.1, 0.0, //
        0.1, 0.8, 0.2, //
        0.1, 0.1, 0.8,
    ];
    let pi = [0.5, 0.3, 0.2];
    let mu = [1.0, 2.0, 4.0];
    let m = model(
        MeasurementRoute::ProbabilisticCategorical,
        MeasurementKind::ProbabilisticFeatureAssignment,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Categorical {
            levels: vec!["low".into(), "mid".into(), "high".into()],
            confusion: c.clone(),
            column_counts: Some(vec![400, 400, 400]),
            max_condition: 50.0,
        },
    );
    // Exact class summaries from the algebra: p*_j = sum_t C[j][t] pi_t and
    // ybar_j = sum_t C[j][t] pi_t mu_t / p*_j; n = 10 000.
    let n = 10_000.0;
    let classes: Vec<ObservedClassSummary> = (0..3)
        .map(|j| {
            let p: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t]).sum();
            let ybar: f64 = (0..3).map(|t| c[j * 3 + t] * pi[t] * mu[t]).sum::<f64>() / p;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the rounded expected count is a small non-negative integer by construction"
            )]
            let count = (p * n).round() as u64;
            ObservedClassSummary { count, mean: ybar, variance: 1.0 }
        })
        .collect();
    let out = correct_categorical(&m, &data_binding(), &classes, &ctx()).unwrap();
    for (level, truth) in [("low", 0.5), ("mid", 0.3), ("high", 0.2)] {
        let (value, _, variance) =
            out.functional(&CategoricalFunctional::Proportion(level)).unwrap();
        assert!((value - truth).abs() < 1e-9, "{level}: {value}");
        assert!(variance > 0.0);
    }
    let (contrast, naive, var) = out
        .functional(&CategoricalFunctional::OutcomeContrast { first: "high", second: "low" })
        .unwrap();
    assert!((contrast - 3.0).abs() < 1e-6, "{contrast}");
    // The observed-label contrast is attenuated: 3.125 - 1.1 ...
    assert!(naive < 2.5, "{naive}");
    assert!(var > 0.0);
    let record = out
        .record(&CategoricalFunctional::OutcomeContrast { first: "high", second: "low" })
        .unwrap();
    assert_eq!(record.parameter_uncertainty, ParameterUncertainty::Propagated);
    assert_eq!(record.model_identity, m.identities().model);
    let unknown = out.functional(&CategoricalFunctional::Proportion("absent")).unwrap_err();
    assert_eq!(unknown.code, "invalid_argument");
    assert_eq!(unknown.detail, antecedent_core::reason_details::measurement_model::UNKNOWN_LEVEL);

    // A confusion matrix worse conditioned than declared refuses.
    let singular = model(
        MeasurementRoute::ProbabilisticCategorical,
        MeasurementKind::ProbabilisticFeatureAssignment,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Categorical {
            levels: vec!["a".into(), "b".into()],
            confusion: vec![0.5, 0.5, 0.5, 0.5],
            column_counts: None,
            max_condition: 100.0,
        },
    );
    let two = [
        ObservedClassSummary { count: 50, mean: 0.0, variance: 1.0 },
        ObservedClassSummary { count: 50, mean: 0.0, variance: 1.0 },
    ];
    let err = correct_categorical(&singular, &data_binding(), &two, &ctx()).unwrap_err();
    assert_eq!(
        err.detail,
        antecedent_core::reason_details::measurement_model::CONFUSION_ILL_CONDITIONED
    );
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

/// `X ~ N(0,1)`, `W = X + N(0,1)`, `Y = 1 + 2 X + N(0,1)`; returns outcome and
/// the posterior of `X` given `W` alone (mean `W / 2`, variance 1/2) and given
/// `(W, Y)` (precision 1 + 1 + 4 = 6).
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
fn posterior_mean_substitution_recovers_the_slope_that_a_single_draw_attenuates() {
    let n = 4_000;
    let draws = 100;
    let (y, marginal, _) = posterior_world(n, draws, 31);
    let m = posterior_model(false);
    let data = PosteriorLinearData {
        outcome: &y,
        draws: &marginal,
        draw_count: draws,
        covariates: &[],
        covariate_count: 0,
    };
    let result =
        correct_latent_posterior_linear_effect(&m, &data_binding(), &data, &ctx()).unwrap();
    assert_eq!(result.combination, PosteriorCombination::PosteriorMeanSubstitution);
    let se = result.record.standard_error.unwrap();
    // Predeclared: 4 SE plus 0.03 for the finite-draw (v / S) attenuation.
    assert!((result.record.point - 2.0).abs() < 4.0 * se + 0.03, "{}", result.record.point);
    // The provider's own parameter uncertainty is not representable here.
    assert_eq!(result.record.parameter_uncertainty, ParameterUncertainty::NotPropagated);
    // One draw regressed on is Berkson-noised: slope about 2 * 0.5 / (0.5 + 0.5).
    let single = PosteriorLinearData {
        outcome: &y,
        draws: &marginal[..n],
        draw_count: 1,
        covariates: &[],
        covariate_count: 0,
    };
    let one = antecedent_stats::measurement_error::LinearMoments::accumulate(
        &antecedent_stats::measurement_error::LinearErrorData {
            y: single.outcome,
            w: single.draws,
            covariates: &[],
            covariate_count: 0,
            replicate_counts: None,
        },
        &antecedent_stats::measurement_error::ErrorVariance::known(0.0, 0.0).unwrap(),
        0.0,
    )
    .unwrap()
    .solve()
    .unwrap();
    assert!(one.naive_coefficients[1] < 1.2, "{}", one.naive_coefficients[1]);

    // Fewer draws than declared refuses.
    let few = PosteriorLinearData { draw_count: 10, draws: &marginal[..10 * n], ..data };
    let err =
        correct_latent_posterior_linear_effect(&m, &data_binding(), &few, &ctx()).unwrap_err();
    assert_eq!(err.detail, antecedent_core::reason_details::measurement_model::INSUFFICIENT_DRAWS);
}

#[test]
fn outcome_conditioned_draws_combine_by_rubins_rules() {
    let n = 3_000;
    let draws = 30;
    let (y, _, conditional) = posterior_world(n, draws, 77);
    let m = posterior_model(true);
    let data = PosteriorLinearData {
        outcome: &y,
        draws: &conditional,
        draw_count: draws,
        covariates: &[],
        covariate_count: 0,
    };
    let result =
        correct_latent_posterior_linear_effect(&m, &data_binding(), &data, &ctx()).unwrap();
    assert_eq!(result.combination, PosteriorCombination::MultipleImputation);
    let se = result.record.standard_error.unwrap();
    assert!((result.record.point - 2.0).abs() < 4.0 * se + 0.02, "{} se {se}", result.record.point);
    assert_eq!(result.record.parameter_uncertainty, ParameterUncertainty::Propagated);
    let names: Vec<&str> =
        result.record.variance_components.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["within_imputation", "between_imputation"]);
    assert!(result.imputations.as_ref().unwrap().between[3] > 0.0);
}

#[test]
fn a_latent_posterior_needs_an_exactly_matching_provider_object() {
    let m = posterior_model(false);
    let identity = ProviderObjectIdentity {
        provider_id: "vendor".into(),
        object_id: "latent-1".into(),
        version_id: "v3".into(),
        snapshot_id: "snap".into(),
        request_id: "req-1".into(),
    };
    let quantity = |population: &str| ScientificQuantity {
        variable_id: "construct:latent".into(),
        variable_name: "latent".into(),
        role: QuantityRole::Covariate,
        units: "unit".into(),
        population_id: population.into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "value".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    };
    let object = |identity: ProviderObjectIdentity, population: &str| {
        ExternalScientificObject::Posterior(PosteriorProviderContract {
            identity,
            kind: ExternalPosteriorKind::Parameter,
            quantities: vec![quantity(population)],
            capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
        })
    };
    let good = object(identity.clone(), "cohort");
    let attested = ExternalTrustState::attest(&good, "vendor").unwrap();
    assert_eq!(
        check_latent_posterior(&m, &good, &attested).unwrap(),
        LatentPosteriorTrust::ExternallyAttested
    );
    // Metadata cannot confer native trust.
    let refusal =
        check_latent_posterior(&m, &good, &ExternalTrustState::NativeSupported).unwrap_err();
    assert_eq!(
        refusal.detail,
        antecedent_core::reason_details::measurement_model::POSTERIOR_NATIVE_TRUST
    );
    // Another request fingerprint is another object.
    let mut other_identity = identity.clone();
    other_identity.request_id = "req-2".into();
    let other = object(other_identity, "cohort");
    let refusal = check_latent_posterior(&m, &other, &attested).unwrap_err();
    assert_eq!(
        refusal.detail,
        antecedent_core::reason_details::measurement_model::POSTERIOR_IDENTITY
    );
    // The posterior must declare the construct in the model's population.
    let wrong_population = object(identity, "other-cohort");
    let refusal = check_latent_posterior(&m, &wrong_population, &attested).unwrap_err();
    assert_eq!(
        refusal.detail,
        antecedent_core::reason_details::measurement_model::POSTERIOR_QUANTITY
    );
    assert_eq!(refusal.validate(), Ok(()));
}

// ---------------------------------------------------------------------------
// Negatives: kind, binding, representation, route, structure
// ---------------------------------------------------------------------------

fn small() -> Simulated {
    simulate(60, 3)
}

#[test]
fn a_proxy_offered_as_the_construct_is_refused_before_any_number() {
    let sim = small();
    for offered in [
        MeasurementKind::NoisyMisclassified,
        MeasurementKind::ProbabilisticFeatureAssignment,
        MeasurementKind::LearnedVectorRepresentation,
        MeasurementKind::Derived,
    ] {
        let mut m = classical_model(0.6, 0.0);
        m.construct.kind = offered;
        let err = correct_classical_linear_effect(&m, &data_binding(), &linear_data(&sim), &ctx())
            .unwrap_err();
        assert_eq!(err.code, "proxy_offered_as_construct");
        assert_eq!(
            err.detail,
            antecedent_core::reason_details::measurement_model::PROXY_AS_CONSTRUCT
        );
    }
    let mut observed = classical_model(0.6, 0.0);
    observed.indicator.kind = MeasurementKind::Observed;
    let err =
        correct_classical_linear_effect(&observed, &data_binding(), &linear_data(&sim), &ctx())
            .unwrap_err();
    assert_eq!(err.detail, antecedent_core::reason_details::measurement_model::INDICATOR_OBSERVED);
}

#[test]
fn population_regime_arm_and_representation_must_match_the_data() {
    let sim = small();
    let m = classical_model(0.6, 0.0);
    let run = |model: &MeasurementModel, binding: &DataBinding| {
        correct_classical_linear_effect(model, binding, &linear_data(&sim), &ctx()).unwrap_err()
    };
    let mut other = data_binding();
    other.population_id = "elsewhere".into();
    assert_eq!(
        run(&m, &other).detail,
        antecedent_core::reason_details::measurement_model::POPULATION_MISMATCH
    );
    let mut other = data_binding();
    other.regime_id = "do(a=1)".into();
    assert_eq!(
        run(&m, &other).detail,
        antecedent_core::reason_details::measurement_model::REGIME_MISMATCH
    );
    let mut arm_model = classical_model(0.6, 0.0);
    arm_model.binding.arm = ArmBinding::Arm("treated".into());
    let mut other = data_binding();
    other.arm_id = Some("control".into());
    assert_eq!(
        run(&arm_model, &other).detail,
        antecedent_core::reason_details::measurement_model::ARM_MISMATCH
    );
    let mut other = data_binding();
    other.representation.version = 2;
    let err = run(&m, &other);
    assert_eq!(err.code, "representation_mapping_required");
    assert_eq!(
        err.detail,
        antecedent_core::reason_details::measurement_model::REPRESENTATION_UNMAPPED
    );
}

#[test]
fn route_and_structure_mismatches_are_never_pooled() {
    let sim = small();
    // A binary model sent down the classical route.
    let binary =
        binary_model(vec![binary_rates(None, 0.9, 0.8, 0.0, 0.0)], ErrorStructure::NonDifferential);
    let err = correct_classical_linear_effect(&binary, &data_binding(), &linear_data(&sim), &ctx())
        .unwrap_err();
    assert_eq!(err.code, "route_not_supported");
    // A differential-by-time declaration has no bounded route and is not pooled.
    let mut by_time = classical_model(0.6, 0.0);
    by_time.structure = ErrorStructure::Differential(DifferentialAxis::Time);
    let err =
        correct_classical_linear_effect(&by_time, &data_binding(), &linear_data(&sim), &ctx())
            .unwrap_err();
    assert_eq!(err.code, "differential_error_not_pooled");
    // Differential by population on the binary route is refused, not pooled.
    let mut binary_by_population = binary.clone();
    binary_by_population.structure = ErrorStructure::Differential(DifferentialAxis::Population);
    let err = correct_binary_prevalence(&binary_by_population, &data_binding(), 10, 100, &ctx())
        .unwrap_err();
    assert_eq!(err.code, "differential_error_not_pooled");
}

#[test]
fn cancellation_stops_before_a_claim() {
    let sim = small();
    let context = ctx();
    context.cancellation.cancel();
    let err = correct_classical_linear_effect(
        &classical_model(0.6, 0.0),
        &data_binding(),
        &linear_data(&sim),
        &context,
    )
    .unwrap_err();
    assert_eq!(err.code, "cancelled_no_claim");
}

#[test]
fn estimator_and_contract_refuse_the_same_requests_with_the_same_details() {
    let m = classical_model(0.6, 0.0);
    let rep = scalar();
    let request = |population: &'static str| MeasurementRequest {
        construct_id: "construct:latent",
        population_id: population,
        regime_id: "observational",
        arm_id: None,
        representation: &rep,
        assumed_structure: ErrorStructure::NonDifferential,
        operation: MeasurementOperation::LinearAdjustedEffect,
    };
    assert!(check_measurement_request(&m, &request("cohort"), None).is_ok());
    let contract = check_measurement_request(&m, &request("elsewhere"), None).unwrap_err();
    let sim = small();
    let mut binding = data_binding();
    binding.population_id = "elsewhere".into();
    let estimate =
        correct_classical_linear_effect(&m, &binding, &linear_data(&sim), &ctx()).unwrap_err();
    assert_eq!(contract.code, estimate.code);
    assert_eq!(contract.detail, estimate.detail);
    // An interval is never supported for a measurement-corrected functional.
    let mut interval = request("cohort");
    interval.operation = MeasurementOperation::Interval;
    let refusal = check_measurement_request(&m, &interval, None).unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(
        antecedent_estimate::measurement::interval_not_supported().code,
        "route_not_supported"
    );
}

// ---------------------------------------------------------------------------
// Artifact: independent replay and semantic mutation
// ---------------------------------------------------------------------------

#[test]
fn artifact_replays_independently_and_refuses_semantic_mutation() {
    let sim = simulate(3_000, 8);
    let m = classical_model(0.6, 0.05);
    let effect =
        correct_classical_linear_effect(&m, &data_binding(), &linear_data(&sim), &ctx()).unwrap();
    let result = MeasurementResult {
        record: effect.record.clone(),
        replay: MeasurementReplay::Linear { moments: effect.moments.clone(), coefficient_index: 1 },
        provider_trust: None,
    };
    let artifact = MeasurementModelArtifact::new(&m, Some(&result)).unwrap();
    let bytes = artifact.to_bytes("replay-1").unwrap();
    let identity = m.identities().model.to_hex();
    let loaded = MeasurementModelArtifact::from_bytes(&bytes, &identity).unwrap();
    let back = loaded.result().unwrap().unwrap();
    assert_eq!(back.record, effect.record);

    // The consumer recomputes from the moments alone, with no estimator.
    let replayed = replay_values(&loaded.model().unwrap(), &back.replay).unwrap();
    assert!((replayed.point - effect.record.point).abs() < 1e-10);
    let se = effect.record.standard_error.unwrap();
    assert!((replayed.variance.sqrt() - se).abs() < 1e-10);
    // The consumer who holds the data can also check the data identity.
    assert!(loaded.covers_data_identity(&effect.record.data_identity));

    // Semantic mutations: another population, another error variance, another
    // representation version are different identities and are refused.
    let mut moved = m.clone();
    moved.binding.population_id = "elsewhere".into();
    let mut variance = m.clone();
    variance.parameters = MeasurementParameters::ClassicalError {
        error_variance: 0.7,
        error_variance_standard_error: 0.05,
        min_reliability: 0.1,
    };
    let mut versioned = m.clone();
    versioned.representation.version = 2;
    for mutated in [moved, variance, versioned] {
        let forged = MeasurementModelArtifact::new(&mutated, None).unwrap().to_bytes("x").unwrap();
        let err = MeasurementModelArtifact::from_bytes(&forged, &identity).unwrap_err();
        assert_eq!(err.reason_code(), Some("measurement_binding_mismatch"));
        // And the original result cannot be attached to the mutated model.
        let attached = MeasurementModelArtifact::new(&mutated, Some(&result)).unwrap_err();
        assert_eq!(attached.reason_code(), Some("measurement_binding_mismatch"));
    }
    // Bytes damaged in transit are refused by the container checksums.
    let mut damaged = bytes.clone();
    let at = damaged.len() / 2;
    damaged[at] ^= 0x55;
    let damaged_error = MeasurementModelArtifact::from_bytes(&damaged, &identity).unwrap_err();
    assert!(damaged_error.to_string().contains("checksum mismatch"), "{damaged_error}");
}

#[test]
fn corrected_prevalence_interval_is_the_closed_form_wald_interval() {
    use antecedent_estimate::measurement::corrected_functional_interval;
    // As the algebra test above: pi = 0.3, closed-form variance of the corrected prevalence.
    let m = binary_model(
        vec![binary_rates(None, 0.9, 0.8, 0.0001, 0.0004)],
        ErrorStructure::NonDifferential,
    );
    let result = correct_binary_prevalence(&m, &data_binding(), 410, 1000, &ctx()).unwrap();
    let variance: f64 = (0.41 * 0.59 / 1000.0 + 0.49 * 0.0004 + 0.09 * 0.0001) / 0.49;
    let half = 1.959_963_984_540_054 * variance.sqrt();
    let (lo, hi) = corrected_functional_interval(&result.record, 0.95).unwrap();
    assert!((lo - (0.3 - half)).abs() < 1e-9 && (hi - (0.3 + half)).abs() < 1e-9);
    assert_eq!(
        corrected_functional_interval(&result.record, 1.0).unwrap_err().code,
        "invalid_argument"
    );
}

/// Classical-error corrected slope on eight hand-written rows (no covariates, error variance
/// 0.3, standard error 0.05 of that estimate), against the moment-correction sandwich written
/// out with explicit 2 x 2 arithmetic:
/// `A = M - diag(0, s2)`, `beta = A^-1 r`, `psi_i = v_i e_i + (0, s2 beta_1)`,
/// `Var = A^-1 mean(psi psi') A^-T / n + c c' v_est` with `c = A^-1[:, 1] beta_1`.
#[test]
fn classical_slope_interval_equals_the_hand_sandwich() {
    use antecedent_estimate::measurement::corrected_functional_interval;
    let w = [0.1, 0.9, 1.7, 2.4, 3.1, 3.8, 4.6, 5.2];
    let y = [1.0, 2.1, 2.8, 4.3, 4.9, 6.2, 6.8, 8.1];
    let data = LinearMeasuredData {
        outcome: &y,
        regressor: &w,
        covariates: &[],
        covariate_count: 0,
        replicate_counts: None,
    };
    let (s2, se_s2) = (0.3, 0.05);
    let result = correct_classical_linear_effect(
        &classical_model(s2, se_s2),
        &data_binding(),
        &data,
        &ctx(),
    )
    .unwrap();
    let n = 8.0;
    let mw = w.iter().sum::<f64>() / n;
    let mww = w.iter().map(|v| v * v).sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mwy = w.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / n;
    // A = [[1, mw], [mw, mww - s2]]; inverse by the 2 x 2 formula.
    let (a00, a01, a11) = (1.0, mw, mww - s2);
    let det = a00 * a11 - a01 * a01;
    let inv = [[a11 / det, -a01 / det], [-a01 / det, a00 / det]];
    let beta = [inv[0][0] * my + inv[0][1] * mwy, inv[1][0] * my + inv[1][1] * mwy];
    let mut meat = [[0.0; 2]; 2];
    for (wi, yi) in w.iter().zip(&y) {
        let e = yi - beta[0] - beta[1] * wi;
        let psi = [e, wi * e + s2 * beta[1]];
        for a in 0..2 {
            for b in 0..2 {
                meat[a][b] += psi[a] * psi[b] / n;
            }
        }
    }
    // given = inv * meat * inv' / n, entry (1, 1).
    let row = [
        inv[1][0] * meat[0][0] + inv[1][1] * meat[1][0],
        inv[1][0] * meat[0][1] + inv[1][1] * meat[1][1],
    ];
    let given = (row[0] * inv[1][0] + row[1] * inv[1][1]) / n;
    let c1 = inv[1][1] * beta[1];
    let total = given + c1 * c1 * se_s2 * se_s2;
    let record = &result.record;
    assert!((record.point - beta[1]).abs() < 1e-9, "{} vs {}", record.point, beta[1]);
    assert!((record.standard_error.unwrap() - total.sqrt()).abs() < 1e-9);
    let half = 1.959_963_984_540_054 * total.sqrt();
    let (lo, hi) = corrected_functional_interval(record, 0.95).unwrap();
    assert!((lo - (beta[1] - half)).abs() < 1e-8 && (hi - (beta[1] + half)).abs() < 1e-8);
}

/// Two-level categorical proportion: with `C = [[0.9, 0.2], [0.1, 0.8]]`, validation column
/// counts (500, 400) and 410 of 1000 units observed as `a`, the corrected proportion is the
/// Rogan-Gladen `pi = (p - c12) / (c11 - c12) = 0.3`, whose delta variance is
/// `[p(1-p)/n + pi^2 c11 (1-c11)/n1 + (1-pi)^2 c12 (1-c12)/n2] / (c11 - c12)^2`.
#[test]
fn categorical_proportion_interval_equals_the_two_level_closed_form() {
    use antecedent_estimate::measurement::corrected_functional_interval;
    let m = model(
        MeasurementRoute::ProbabilisticCategorical,
        MeasurementKind::ProbabilisticFeatureAssignment,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Categorical {
            levels: vec!["a".into(), "b".into()],
            confusion: vec![0.9, 0.2, 0.1, 0.8],
            column_counts: Some(vec![500, 400]),
            max_condition: 50.0,
        },
    );
    let classes = [
        ObservedClassSummary { count: 410, mean: 1.0, variance: 1.0 },
        ObservedClassSummary { count: 590, mean: 2.0, variance: 1.0 },
    ];
    let out = correct_categorical(&m, &data_binding(), &classes, &ctx()).unwrap();
    let record = out.record(&CategoricalFunctional::Proportion("a")).unwrap();
    let (p, n, c11, c12) = (0.41, 1000.0, 0.9, 0.2);
    let den = c11 - c12;
    let pi = (p - c12) / den;
    let variance = (p * (1.0 - p) / n
        + pi * pi * c11 * (1.0 - c11) / 500.0
        + (1.0 - pi) * (1.0 - pi) * c12 * (1.0 - c12) / 400.0)
        / (den * den);
    assert!((record.point - pi).abs() < 1e-12);
    assert!((record.standard_error.unwrap().powi(2) - variance).abs() < 1e-12);
    let half = 1.959_963_984_540_054 * variance.sqrt();
    let (lo, hi) = corrected_functional_interval(&record, 0.95).unwrap();
    assert!((lo - (pi - half)).abs() < 1e-9 && (hi - (pi + half)).abs() < 1e-9);
}

/// Repeated-measurement corrected slope on six units of two replicates each, against the
/// moment-correction sandwich with the retained influence of the within-variance estimate
/// written out in explicit arithmetic.
#[test]
fn repeated_slope_interval_equals_the_hand_sandwich_with_retained_influence() {
    use antecedent_estimate::measurement::corrected_functional_interval;
    let pairs = [(0.2, 0.6), (1.1, 0.9), (1.8, 2.3), (2.9, 2.6), (3.4, 3.9), (4.6, 4.3)];
    let y = [1.2, 2.0, 3.1, 4.2, 5.3, 6.1];
    let values: Vec<f64> = pairs.iter().flat_map(|p| [p.0, p.1]).collect();
    let counts = [2_u32; 6];
    let m = model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementKind::NoisyMisclassified,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Repeated { min_replicated_units: 2, min_reliability: 0.1 },
    );
    let result =
        correct_repeated_linear_effect(&m, &data_binding(), &y, &values, &counts, &[], 0, &ctx())
            .unwrap();
    let n = 6.0;
    let w: Vec<f64> = pairs.iter().map(|p| 0.5 * (p.0 + p.1)).collect();
    // One-way within sum of squares per unit, (a - b)^2 / 2; df_within = 6, so msw = mean ssw.
    let ssw: Vec<f64> = pairs.iter().map(|p| (p.0 - p.1) * (p.0 - p.1) / 2.0).collect();
    let s2 = ssw.iter().sum::<f64>() / 6.0;
    // nu_bar = 1, so the influence of unit i is ssw_i - (k - 1) msw.
    let xi: Vec<f64> = ssw.iter().map(|s| s - s2).collect();
    let shift = s2 * 0.5; // mean(1 / k) = 1/2
    let mw = w.iter().sum::<f64>() / n;
    let mww = w.iter().map(|v| v * v).sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mwy = w.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>() / n;
    let (a00, a01, a11) = (1.0, mw, mww - shift);
    let det = a00 * a11 - a01 * a01;
    let inv = [[a11 / det, -a01 / det], [-a01 / det, a00 / det]];
    let beta = [inv[0][0] * my + inv[0][1] * mwy, inv[1][0] * my + inv[1][1] * mwy];
    let mut meat = [[0.0; 2]; 2];
    let mut cross = [0.0; 2];
    for ((wi, yi), xii) in w.iter().zip(&y).zip(&xi) {
        let e = yi - beta[0] - beta[1] * wi;
        let psi = [e, wi * e + s2 * 0.5 * beta[1]];
        for a in 0..2 {
            cross[a] += psi[a] * xii / n;
            for b in 0..2 {
                meat[a][b] += psi[a] * psi[b] / n;
            }
        }
    }
    let row = [
        inv[1][0] * meat[0][0] + inv[1][1] * meat[1][0],
        inv[1][0] * meat[0][1] + inv[1][1] * meat[1][1],
    ];
    let given = (row[0] * inv[1][0] + row[1] * inv[1][1]) / n;
    let c1 = inv[1][1] * beta[1] * 0.5;
    let u1 = inv[1][0] * cross[0] + inv[1][1] * cross[1];
    let v_xi = xi.iter().map(|v| v * v).sum::<f64>() / n;
    let total = given + (2.0 * c1 * u1 + c1 * c1 * v_xi) / n;
    let record = &result.record;
    assert!((record.point - beta[1]).abs() < 1e-9, "{} vs {}", record.point, beta[1]);
    assert!((record.standard_error.unwrap().powi(2) - total).abs() < 1e-9);
    let half = 1.959_963_984_540_054 * total.sqrt();
    let (lo, hi) = corrected_functional_interval(record, 0.95).unwrap();
    assert!((lo - (beta[1] - half)).abs() < 1e-8 && (hi - (beta[1] + half)).abs() < 1e-8);
}

/// Simple-regression slope and its HC0 variance `sum (x - mean)^2 e^2 / Sxx^2`, from the
/// definition.
fn ols_slope_hc0(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let meat: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| {
            let e = b - intercept - slope * a;
            (a - mx) * (a - mx) * e * e
        })
        .sum();
    (slope, meat / (sxx * sxx))
}

fn small_posterior_model(conditioned: bool) -> MeasurementModel {
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
            min_draws: 2,
        },
    )
}

#[test]
fn posterior_slope_intervals_equal_the_hand_rubin_and_hc0_constructions() {
    use antecedent_estimate::measurement::corrected_functional_interval;
    let y = [1.2, 2.0, 3.1, 4.2, 5.3, 6.1];
    let draws = [
        [0.1, 0.8, 1.6, 2.2, 3.1, 3.9],
        [0.3, 1.1, 1.4, 2.6, 2.8, 4.2],
        [0.2, 0.9, 1.9, 2.4, 3.3, 4.0],
    ];
    let flat: Vec<f64> = draws.iter().flatten().copied().collect();
    let data = PosteriorLinearData {
        outcome: &y,
        draws: &flat,
        draw_count: 3,
        covariates: &[],
        covariate_count: 0,
    };
    let z = 1.959_963_984_540_054;

    // Outcome-conditioned draws: Rubin's rules, W + (1 + 1/S) B on the slope.
    let fits: Vec<(f64, f64)> = draws.iter().map(|d| ols_slope_hc0(d, &y)).collect();
    let mean = fits.iter().map(|f| f.0).sum::<f64>() / 3.0;
    let within = fits.iter().map(|f| f.1).sum::<f64>() / 3.0;
    let between = fits.iter().map(|f| (f.0 - mean) * (f.0 - mean)).sum::<f64>() / 2.0;
    let total = within + (1.0 + 1.0 / 3.0) * between;
    let result = correct_latent_posterior_linear_effect(
        &small_posterior_model(true),
        &data_binding(),
        &data,
        &ctx(),
    )
    .unwrap();
    assert!((result.record.point - mean).abs() < 1e-9);
    assert!((result.record.standard_error.unwrap().powi(2) - total).abs() < 1e-9);
    let (lo, hi) = corrected_functional_interval(&result.record, 0.95).unwrap();
    assert!((lo - (mean - z * total.sqrt())).abs() < 1e-8);
    assert!((hi - (mean + z * total.sqrt())).abs() < 1e-8);

    // Outcome-free posterior: regression on the posterior means, HC0 variance.
    let means: Vec<f64> = (0..6).map(|i| draws.iter().map(|d| d[i]).sum::<f64>() / 3.0).collect();
    let (slope, variance) = ols_slope_hc0(&means, &y);
    let result = correct_latent_posterior_linear_effect(
        &small_posterior_model(false),
        &data_binding(),
        &data,
        &ctx(),
    )
    .unwrap();
    assert!((result.record.point - slope).abs() < 1e-9);
    assert!((result.record.standard_error.unwrap().powi(2) - variance).abs() < 1e-9);
    let (lo, hi) = corrected_functional_interval(&result.record, 0.95).unwrap();
    assert!((lo - (slope - z * variance.sqrt())).abs() < 1e-8);
    assert!((hi - (slope + z * variance.sqrt())).abs() < 1e-8);
}

/// Outcome means by true category for two levels, from `theta = (ybar_0, ybar_1, p_0, p_1,
/// C_00, C_01, C_10, C_11)`: `pi = C^-1 p`, `Q[j][t] = C[j][t] pi_t / p_j`, `mu = Q^-1 ybar`,
/// written with explicit 2 x 2 inverses.
fn two_level_outcome_means(t: &[f64; 8]) -> [f64; 2] {
    let inv = |a: f64, b: f64, c: f64, d: f64| {
        let det = a * d - b * c;
        [[d / det, -b / det], [-c / det, a / det]]
    };
    let ci = inv(t[4], t[5], t[6], t[7]);
    let pi = [ci[0][0] * t[2] + ci[0][1] * t[3], ci[1][0] * t[2] + ci[1][1] * t[3]];
    let q =
        [[t[4] * pi[0] / t[2], t[5] * pi[1] / t[2]], [t[6] * pi[0] / t[3], t[7] * pi[1] / t[3]]];
    let qi = inv(q[0][0], q[0][1], q[1][0], q[1][1]);
    [qi[0][0] * t[0] + qi[0][1] * t[1], qi[1][0] * t[0] + qi[1][1] * t[1]]
}

/// The corrected outcome means and contrast against an independently differentiated map: the
/// Jacobian of `two_level_outcome_means` is taken by Richardson-extrapolated central
/// differences (steps `h` and `h / 2`, error `O(h^4)`), the covariance of `theta` is written
/// from the multinomial and binomial variances, and the variances are `J Sigma J'`.
#[test]
fn categorical_outcome_mean_and_contrast_variances_equal_an_independent_jacobian() {
    use antecedent_estimate::measurement::corrected_functional_interval;
    let m = model(
        MeasurementRoute::ProbabilisticCategorical,
        MeasurementKind::ProbabilisticFeatureAssignment,
        ErrorStructure::NonDifferential,
        MeasurementParameters::Categorical {
            levels: vec!["a".into(), "b".into()],
            confusion: vec![0.9, 0.2, 0.1, 0.8],
            column_counts: Some(vec![500, 400]),
            max_condition: 50.0,
        },
    );
    let classes = [
        ObservedClassSummary { count: 410, mean: 1.0, variance: 0.8 },
        ObservedClassSummary { count: 590, mean: 2.0, variance: 1.3 },
    ];
    let out = correct_categorical(&m, &data_binding(), &classes, &ctx()).unwrap();
    let theta = [1.0, 2.0, 0.41, 0.59, 0.9, 0.2, 0.1, 0.8];
    let n = 1000.0;
    let mut sigma = [[0.0_f64; 8]; 8];
    sigma[0][0] = 0.8 / 410.0;
    sigma[1][1] = 1.3 / 590.0;
    sigma[2][2] = 0.41 * 0.59 / n;
    sigma[3][3] = 0.41 * 0.59 / n;
    sigma[2][3] = -0.41 * 0.59 / n;
    sigma[3][2] = sigma[2][3];
    // Column t = 0 holds C_00 and C_10 (500 validation units); column 1 holds C_01 and C_11 (400).
    for (idx, count) in [([4, 6], 500.0), ([5, 7], 400.0)] {
        let p = theta[idx[0]];
        sigma[idx[0]][idx[0]] = p * (1.0 - p) / count;
        sigma[idx[1]][idx[1]] = p * (1.0 - p) / count;
        sigma[idx[0]][idx[1]] = -p * (1.0 - p) / count;
        sigma[idx[1]][idx[0]] = -p * (1.0 - p) / count;
    }
    let derivative = |i: usize, level: usize| {
        let central = |h: f64| {
            let (mut up, mut down) = (theta, theta);
            up[i] += h;
            down[i] -= h;
            (two_level_outcome_means(&up)[level] - two_level_outcome_means(&down)[level])
                / (2.0 * h)
        };
        let h = 1e-3;
        (4.0 * central(h / 2.0) - central(h)) / 3.0
    };
    let jac: Vec<Vec<f64>> =
        (0..2).map(|level| (0..8).map(|i| derivative(i, level)).collect()).collect();
    let cov = |a: &[f64], b: &[f64]| {
        (0..8).map(|i| (0..8).map(|j| a[i] * sigma[i][j] * b[j]).sum::<f64>()).sum::<f64>()
    };
    let mu = two_level_outcome_means(&theta);
    let (v00, v11, v01) = (cov(&jac[0], &jac[0]), cov(&jac[1], &jac[1]), cov(&jac[0], &jac[1]));
    let z = 1.959_963_984_540_054;
    let cases = [
        (CategoricalFunctional::OutcomeMean("a"), mu[0], v00),
        (CategoricalFunctional::OutcomeMean("b"), mu[1], v11),
        (
            CategoricalFunctional::OutcomeContrast { first: "a", second: "b" },
            mu[0] - mu[1],
            v00 + v11 - 2.0 * v01,
        ),
    ];
    for (functional, point, variance) in cases {
        let record = out.record(&functional).unwrap();
        assert!((record.point - point).abs() < 1e-9, "point");
        let got = record.standard_error.unwrap().powi(2);
        assert!((got - variance).abs() <= 1e-6 * variance, "variance {got} vs {variance}");
        let (lo, hi) = corrected_functional_interval(&record, 0.95).unwrap();
        let half = z * got.sqrt();
        assert!((lo - (point - half)).abs() < 1e-8 && (hi - (point + half)).abs() < 1e-8);
    }
}
