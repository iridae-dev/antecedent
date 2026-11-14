//! Repeated-sampling coverage of the intervals published by the measurement-correction routes
//! and by the posterior composition route, each scored against an analytically known truth.
//!
//! Every test scores the interval the route reports at 0.95 and binds the construction it
//! reports. A replicate whose correction refuses, or whose route reports no interval, is a skip
//! and counts as a miss.
//!
//! Static measurement correction (`PreparedMeasurementModel::estimate` then
//! `MeasurementEstimate::interval`, the equal-tailed Wald interval `point -/+ z * se` of the
//! delta-method standard error that includes the measurement-parameter uncertainty):
//!
//! * binary prevalence (Rogan-Gladen): sensitivity 0.9 and specificity 0.8 estimated from
//!   validation samples, a recorded prevalence from a separate sample, truth the latent
//!   prevalence 0.3;
//! * binary risk difference: the same shared validation rates applied to two arms with latent
//!   risks 0.4 and 0.2, truth 0.2;
//! * classical error, linear outcome: `X ~ N(0,1)`, `Z = 0.5 X + N(0,1)`,
//!   `W = X + N(0, 0.6)`, `Y = 1 + 2 X + 0.5 Z + N(0,1)`, truth the coefficient 2, with the
//!   error variance declared exactly and with the error variance estimated from replicate pairs;
//! * repeated measurements: two or three replicates per unit of `X + N(0, 0.5)`,
//!   `Y = 1 + 1.5 X + N(0,1)`, truth 1.5;
//! * categorical confusion matrix: three true categories with proportions 0.5, 0.3, 0.2 and
//!   outcome means 1, 2, 4, labels assigned through a confusion matrix estimated from a
//!   validation sample per true category; truths the proportion 0.5, the outcome mean 4 and the
//!   outcome contrast 3.
//!
//! Continuous classical error (`PreparedMeasurementModel::estimate_continuous` then
//! `ContinuousMeasurementEstimate::interval`): `X ~ N(0,1)`, `W = X + N(0, 0.25)` with the error
//! variance declared exactly, corrected by regression calibration and by SIMEX on the default
//! grid. Linear outcome `Y = 1 + 2 X + N(0, 0.25)` (truth 2), the error in a confounder of an
//! error-free treatment `T = 0.6 X + N(0,1)`, `Y = 1 + T + 1.5 X + N(0,1)` (truth 1), and the
//! logistic outcome `P(Y = 1) = expit(-0.5 + X)` (truth 1). Every interval is scored against the
//! generating coefficient; logistic calibration and SIMEX are approximate corrections.
//!
//! Latent posterior route (`correct_latent_posterior_linear_effect`): `X ~ N(0,1)`,
//! `W = X + N(0,1)`, `Y = 1 + 2 X + N(0,1)`. The provider's posterior is the conditional law of
//! `X` given `W` alone (posterior-mean substitution), or joint draws of `X` given `(W, Y)` that
//! first draw the outcome model's parameters from their posterior (proper imputations, combined
//! under Rubin's rules). Truth is the coefficient 2.
//!
//! Pooled differential transport (`transport_pooled_differential_prevalence`): sensitivity and
//! specificity of every population, the target included, are drawn on the logit scale around
//! 0.9 and 0.85; six validated sources; the scored interval is the equal-tailed simulation
//! interval of the target prevalence 0.3, without and with a validation sample in the target and
//! with homogeneous error. With homogeneous error the between-population variance is at its
//! boundary zero, where every variance the route draws is at least the truth, so that record is
//! a one-sided (at least nominal) target.
//!
//! Temporal misclassification: a sequentially randomized unit-period panel with a binary
//! outcome `P(Y = 1) = 0.3 + 0.25 A` recorded with sensitivity 0.9 and specificity 0.85 and an
//! independent validation sample, through `TemporalDesignQuery::outcome_misclassification`.
//! The frequentist cluster-robust interval and the Bayesian-bootstrap credible interval are
//! scored on the excursion contrast (truth 0.25) and on the control arm mean (truth 0.3). The
//! frequentist route appends the replicate's estimated Youden index to its functional label;
//! the key drops that value. The frequentist interval is also scored on the three-arm,
//! ordered-level, factorial and interference designs with the same recording error.
//!
//! Posterior composition: aligned Bayesian-bootstrap posterior draws of two sample means
//! composed into their contrast through `propagate_aligned_linear_spread`, with the normal
//! interval built on the composed spread, keyed as the claim keys itself. The sum, a weighted
//! sum, the product and the ratio of the two means are scored the same way, the linear ones
//! also on the linear-moments route with the covariance of the means estimated from the rows.
//!
//! The measurement and temporal routes publish no runtime calibration key for the interval, so
//! their constructions are declared here in the form the other facade-level harnesses without a
//! runtime key use; the estimator and functional of the temporal routes are read from the result.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded replicate indices in a calibration fixture"
)]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]
#![allow(
    clippy::needless_range_loop,
    reason = "simulation fixtures index parallel columns by row and draw"
)]

mod common;

use antecedent::analysis::continuous_measurement::{
    ContinuousCorrectionSpec, CorrectionMethod, DEFAULT_BOOTSTRAP_REPLICATES, ErrorLocation,
    OutcomeLink,
};
use antecedent::analysis::joint_law_propagation::{
    LinearDependence, MomentInput, PropagationClaim, PropagationDraws, ValueFunctional,
    propagate_aligned_linear_spread, propagate_linear_moments_spread, propagate_spread,
};
use antecedent::{
    BayesianConfig, CategoricalTarget, ContinuousErrorInput, InferenceMode,
    MeasurementAnalysisBinding, MeasurementInput, Study, StudyBuilder,
};
use antecedent_contract::{InputRole, PropagationOperation};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, PosteriorReference, RepresentationVersion,
};
use antecedent_core::{
    CalibrationBasis, CausalQuery, ExecutionContext, OutcomeMisclassification, TemporalAssignment,
    TemporalDesignFunctional, TemporalDesignQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::measurement::{
    DataBinding, PosteriorLinearData, correct_latent_posterior_linear_effect,
    corrected_functional_interval,
};
use antecedent_estimate::measurement_transport_pooled::{
    PooledDifferentialRequest, PopulationValidation, transport_pooled_differential_prevalence,
};
use antecedent_stats::measurement_error::{ObservedClassSummary, TwoByTwoCounts};
use antecedent_stats::{NORMAL_APPROXIMATION_CONSTRUCTION, normal_interval_candidate};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian,
    grid_n, grid_seed, map_replicates, n_sim, uniform,
};

// ---------------------------------------------------------------------------
// Shared scoring
// ---------------------------------------------------------------------------

/// One replicate's reported interval, the construction it was reported under and its row count.
struct Reading {
    interval: (f64, f64),
    construction: Construction,
    rows: u64,
}

/// Score `replicate` over `n_sim()` datasets against `truth` and assert nominal coverage;
/// `None` is a skip.
fn measure(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    truth: f64,
    seed_base: u64,
    replicate: impl Fn(u64) -> Option<Reading> + Sync,
) {
    tally_of(test, dgp, interval, truth, seed_base, replicate).assert();
}

/// The tally of `replicate` over `n_sim()` datasets scored against `truth`; `None` is a skip.
fn tally_of(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    truth: f64,
    seed_base: u64,
    replicate: impl Fn(u64) -> Option<Reading> + Sync,
) -> CoverageTally {
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    let readings = map_replicates(n_sim(), |rep| replicate(grid_seed(seed_base + rep)));
    for reading in readings {
        let Some(reading) = reading else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &reading.construction,
            ScopeFacts {
                row_count: reading.rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(reading.interval), truth);
    }
    tally
}

/// The declared construction of a measurement-route interval.
#[allow(
    clippy::too_many_arguments,
    reason = "one value per axis of the calibration key, all of them required"
)]
fn construction(
    query: &str,
    graph_class: &str,
    estimator: &str,
    interval_method: &str,
    se_kind: &str,
    dependence: &str,
    inference: &str,
    functional: &str,
) -> Construction {
    Construction {
        query: query.into(),
        graph_class: graph_class.into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: inference.into(),
        estimator: estimator.into(),
        interval_method: interval_method.into(),
        se_kind: se_kind.into(),
        dependence: dependence.into(),
        posterior: String::new(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn binomial(next: &mut impl FnMut() -> f64, n: u64, p: f64) -> u64 {
    (0..n).filter(|_| next() < p).count() as u64
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

// ---------------------------------------------------------------------------
// Static measurement correction
// ---------------------------------------------------------------------------

const WALD: &str = "delta_method_wald";

fn scalar() -> RepresentationVersion {
    RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 }
}

fn analysis_binding() -> MeasurementAnalysisBinding {
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
    parameters: MeasurementParameters,
    evidence: ParameterEvidence,
) -> MeasurementModel {
    MeasurementModel {
        model_id: "calibration-model".into(),
        route,
        indicator: variable("schema:indicator", indicator_kind),
        construct: variable("construct:latent", MeasurementKind::LatentScientific),
        binding: MeasurementBinding {
            population_id: "cohort".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters,
        representation: scalar(),
        evidence,
    }
}

fn validation_evidence(units: u64) -> ParameterEvidence {
    ParameterEvidence::ValidationSample { sample_id: "validation".into(), units }
}

/// A binary model whose shared rates were estimated from `m` true positives and `m` true
/// negatives, with the binomial variances of the estimates.
fn estimated_binary_model(
    next: &mut impl FnMut() -> f64,
    m: u64,
    sensitivity: f64,
    specificity: f64,
) -> MeasurementModel {
    let se_hat = binomial(next, m, sensitivity) as f64 / m as f64;
    let sp_hat = binomial(next, m, specificity) as f64 / m as f64;
    let rates = BinaryArmRates {
        arm_id: None,
        sensitivity: se_hat,
        specificity: sp_hat,
        sensitivity_variance: se_hat * (1.0 - se_hat) / m as f64,
        specificity_variance: sp_hat * (1.0 - sp_hat) / m as f64,
    };
    model(
        MeasurementRoute::BinaryMisclassification,
        MeasurementKind::NoisyMisclassified,
        MeasurementParameters::Binary { rates: vec![rates], min_youden: 0.1 },
        validation_evidence(2 * m),
    )
}

fn binary_prevalence_replicate(seed: u64) -> Option<Reading> {
    let (sensitivity, specificity, prevalence) = (0.9, 0.8, 0.3);
    let m = grid_n(200) as u64;
    let total = grid_n(500) as u64;
    let mut next = uniform(seed, 1);
    let model = estimated_binary_model(&mut next, m, sensitivity, specificity);
    let recorded = sensitivity * prevalence + (1.0 - specificity) * (1.0 - prevalence);
    let events = binomial(&mut next, total, recorded);
    let estimate = StudyBuilder::measurement_model(
        model,
        analysis_binding(),
        MeasurementInput::BinaryPrevalence { events, total },
        None,
        &ctx(seed),
    )
    .ok()?
    .estimate(&ctx(seed))
    .ok()?;
    Some(Reading {
        interval: estimate.interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            "measurement.binary_misclassification.rogan_gladen",
            WALD,
            "delta_method_with_validation_variance",
            "declared_non_differential",
            "Frequentist",
            "latent_prevalence",
        ),
        rows: total,
    })
}

fn binary_risk_difference_replicate(seed: u64) -> Option<Reading> {
    let (sensitivity, specificity) = (0.9, 0.8);
    let (exposed_risk, unexposed_risk) = (0.4, 0.2);
    let m = grid_n(200) as u64;
    let per_arm = grid_n(500) as u64;
    let mut next = uniform(seed, 1);
    let model = estimated_binary_model(&mut next, m, sensitivity, specificity);
    let recorded = |risk: f64| sensitivity * risk + (1.0 - specificity) * (1.0 - risk);
    let table = TwoByTwoCounts {
        exposed_events: binomial(&mut next, per_arm, recorded(exposed_risk)),
        exposed_total: per_arm,
        unexposed_events: binomial(&mut next, per_arm, recorded(unexposed_risk)),
        unexposed_total: per_arm,
    };
    let estimate = StudyBuilder::measurement_model(
        model,
        analysis_binding(),
        MeasurementInput::BinaryRiskDifference {
            exposed_arm_id: "treated".into(),
            unexposed_arm_id: "control".into(),
            table,
        },
        None,
        &ctx(seed),
    )
    .ok()?
    .estimate(&ctx(seed))
    .ok()?;
    Some(Reading {
        interval: estimate.interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            "measurement.binary_misclassification.risk_difference",
            WALD,
            "delta_method_with_validation_variance",
            "declared_non_differential",
            "Frequentist",
            "latent_risk_difference",
        ),
        rows: 2 * per_arm,
    })
}

/// `X ~ N(0,1)`, `Z = 0.5 X + N(0,1)`, `W = X + N(0, 0.6)`, `Y = 1 + 2 X + 0.5 Z + N(0,1)`.
fn classical_world(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut normal = gaussian(seed);
    let (mut y, mut w, mut z) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::new());
    for _ in 0..n {
        let x = normal();
        let zi = 0.5 * x + normal();
        w.push(x + 0.6_f64.sqrt() * normal());
        y.push(1.0 + 2.0 * x + 0.5 * zi + normal());
        z.push(zi);
    }
    (y, w, z)
}

fn classical_model(
    error_variance: f64,
    standard_error: f64,
    evidence: ParameterEvidence,
) -> MeasurementModel {
    model(
        MeasurementRoute::ClassicalContinuousError,
        MeasurementKind::NoisyMisclassified,
        MeasurementParameters::ClassicalError {
            error_variance,
            error_variance_standard_error: standard_error,
            min_reliability: 0.1,
        },
        evidence,
    )
}

fn classical_replicate(seed: u64, estimate_error_variance: bool) -> Option<Reading> {
    let n = grid_n(500);
    let (y, w, z) = classical_world(n, seed);
    let model = if estimate_error_variance {
        // The error variance from `m` replicate pairs: `d = e1 - e2 ~ N(0, 2 sigma^2)`.
        let m = grid_n(200);
        let mut normal = gaussian(seed ^ 0x5eed_0001);
        let mean_square =
            (0..m).map(|_| (0.6_f64.sqrt() * (normal() - normal())).powi(2)).sum::<f64>()
                / m as f64;
        let variance = 0.5 * mean_square;
        classical_model(variance, variance * (2.0 / m as f64).sqrt(), validation_evidence(m as u64))
    } else {
        classical_model(0.6, 0.0, ParameterEvidence::Declared { declarer: "calibration".into() })
    };
    let estimate = StudyBuilder::measurement_model(
        model,
        analysis_binding(),
        MeasurementInput::ClassicalLinear {
            outcome: y,
            regressor: w,
            covariates: z,
            covariate_count: 1,
            replicate_counts: None,
        },
        None,
        &ctx(seed),
    )
    .ok()?
    .estimate(&ctx(seed))
    .ok()?;
    Some(Reading {
        interval: estimate.interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            "measurement.classical_continuous_error.linear",
            WALD,
            if estimate_error_variance {
                "delta_method_with_estimated_error_variance"
            } else {
                "delta_method_with_declared_error_variance"
            },
            "declared_non_differential",
            "Frequentist",
            "linear_coefficient",
        ),
        rows: n as u64,
    })
}

fn repeated_replicate(seed: u64) -> Option<Reading> {
    let units = grid_n(500);
    let mut normal = gaussian(seed);
    let (mut y, mut counts, mut values) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        let x = normal();
        y.push(1.0 + 1.5 * x + normal());
        let k = 2 + u32::from(i % 2 == 1);
        counts.push(k);
        for _ in 0..k {
            values.push(x + 0.5_f64.sqrt() * normal());
        }
    }
    let model = model(
        MeasurementRoute::RepeatedMeasurements,
        MeasurementKind::NoisyMisclassified,
        MeasurementParameters::Repeated { min_replicated_units: 50, min_reliability: 0.1 },
        ParameterEvidence::ReplicateSample { sample_id: "analysis-replicates".into() },
    );
    let estimate = StudyBuilder::measurement_model(
        model,
        analysis_binding(),
        MeasurementInput::RepeatedLinear {
            outcome: y,
            replicate_values: values,
            replicate_counts: counts,
            covariates: vec![],
            covariate_count: 0,
        },
        None,
        &ctx(seed),
    )
    .ok()?
    .estimate(&ctx(seed))
    .ok()?;
    Some(Reading {
        interval: estimate.interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            "measurement.repeated_measurements.linear",
            WALD,
            "delta_method_with_replicate_error_variance",
            "declared_non_differential",
            "Frequentist",
            "linear_coefficient",
        ),
        rows: units as u64,
    })
}

/// True-category names, in the order of the confusion axes.
const CATEGORY_LEVELS: [&str; 3] = ["low", "mid", "high"];
/// `P(observed | true)`, row-major `[observed][true]`.
const CONFUSION: [f64; 9] = [0.8, 0.1, 0.0, 0.1, 0.8, 0.2, 0.1, 0.1, 0.8];
const CATEGORY_PROPORTIONS: [f64; 3] = [0.5, 0.3, 0.2];
/// Mean outcome of each true category; the outcome adds unit normal noise.
const CATEGORY_MEANS: [f64; 3] = [1.0, 2.0, 4.0];

fn category_of(u: f64, probabilities: [f64; 3]) -> usize {
    if u < probabilities[0] {
        0
    } else if u < probabilities[0] + probabilities[1] {
        1
    } else {
        2
    }
}

fn observed_label(u: f64, truth: usize) -> usize {
    category_of(u, [CONFUSION[truth], CONFUSION[3 + truth], CONFUSION[6 + truth]])
}

/// Labels assigned through [`CONFUSION`], whose columns are estimated from `m` validation units
/// per true category; the observed-label summaries are of a separate sample.
fn categorical_replicate(
    seed: u64,
    target: CategoricalTarget,
    functional: &'static str,
) -> Option<Reading> {
    let n = grid_n(1000);
    let m = grid_n(300) as u64;
    let mut next = uniform(seed, 1);
    let mut normal = gaussian(seed);
    let mut validation = [0_u64; 9];
    for truth in 0..3 {
        for _ in 0..m {
            validation[observed_label(next(), truth) * 3 + truth] += 1;
        }
    }
    let confusion: Vec<f64> = validation.iter().map(|&count| count as f64 / m as f64).collect();
    let (mut count, mut sum, mut squares) = ([0_u64; 3], [0.0_f64; 3], [0.0_f64; 3]);
    for _ in 0..n {
        let truth = category_of(next(), CATEGORY_PROPORTIONS);
        let label = observed_label(next(), truth);
        let y = CATEGORY_MEANS[truth] + normal();
        count[label] += 1;
        sum[label] += y;
        squares[label] += y * y;
    }
    let classes: Vec<ObservedClassSummary> = (0..3)
        .map(|j| {
            let units = count[j] as f64;
            let mean = sum[j] / units;
            ObservedClassSummary {
                count: count[j],
                mean,
                variance: (squares[j] - units * mean * mean) / (units - 1.0),
            }
        })
        .collect();
    let model = model(
        MeasurementRoute::ProbabilisticCategorical,
        MeasurementKind::ProbabilisticFeatureAssignment,
        MeasurementParameters::Categorical {
            levels: CATEGORY_LEVELS.map(String::from).to_vec(),
            confusion,
            column_counts: Some(vec![m; 3]),
            max_condition: 50.0,
        },
        validation_evidence(3 * m),
    );
    let estimate = StudyBuilder::measurement_model(
        model,
        analysis_binding(),
        MeasurementInput::Categorical { classes, target },
        None,
        &ctx(seed),
    )
    .ok()?
    .estimate(&ctx(seed))
    .ok()?;
    Some(Reading {
        interval: estimate.interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            "measurement.probabilistic_categorical.confusion_inversion",
            WALD,
            "delta_method_with_validation_variance",
            "declared_non_differential",
            "Frequentist",
            functional,
        ),
        rows: n as u64,
    })
}

/// The data law of a continuous correction; `W = X + N(0, 0.25)` in each.
#[derive(Clone, Copy)]
enum ContinuousWorld {
    /// `Y = 1 + 2 X + N(0, 0.25)`: the error is in the treatment, coefficient 2.
    LinearTreatment,
    /// `T = 0.6 X + N(0,1)`, `Y = 1 + T + 1.5 X + N(0,1)`: the error is in the confounder `X` of
    /// the error-free treatment `T`, coefficient 1.
    LinearCovariate,
    /// `P(Y = 1) = expit(-0.5 + X)`: the error is in the treatment, logit coefficient 1.
    Logistic,
}

impl ContinuousWorld {
    /// The generating coefficient of the treatment.
    fn truth(self) -> f64 {
        match self {
            Self::LinearTreatment => 2.0,
            Self::LinearCovariate | Self::Logistic => 1.0,
        }
    }
}

/// Regression calibration or default-grid SIMEX of one world, with the error variance 0.25
/// declared exactly and the route's default bootstrap replicate count.
fn continuous_replicate(seed: u64, world: ContinuousWorld, simex: bool) -> Option<Reading> {
    let n = grid_n(500);
    let mut normal = gaussian(seed);
    let mut next = uniform(seed, 5);
    let (mut y, mut w, mut covariates) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::new());
    for _ in 0..n {
        let x = normal();
        w.push(x + 0.5 * normal());
        match world {
            ContinuousWorld::LinearTreatment => y.push(1.0 + 2.0 * x + 0.5 * normal()),
            ContinuousWorld::LinearCovariate => {
                let treatment = 0.6 * x + normal();
                covariates.push(treatment);
                y.push(1.0 + treatment + 1.5 * x + normal());
            }
            ContinuousWorld::Logistic => y.push(f64::from(u8::from(next() < expit(-0.5 + x)))),
        }
    }
    let covariate_count = usize::from(matches!(world, ContinuousWorld::LinearCovariate));
    let prepared = StudyBuilder::measurement_model(
        classical_model(0.25, 0.0, ParameterEvidence::Declared { declarer: "calibration".into() }),
        analysis_binding(),
        MeasurementInput::ClassicalLinear {
            outcome: y.clone(),
            regressor: w.clone(),
            covariates: covariates.clone(),
            covariate_count,
            replicate_counts: None,
        },
        None,
        &ctx(seed),
    )
    .ok()?;
    let spec = ContinuousCorrectionSpec {
        location: match world {
            ContinuousWorld::LinearCovariate => ErrorLocation::Covariate { treatment_column: 0 },
            _ => ErrorLocation::Treatment,
        },
        link: match world {
            ContinuousWorld::Logistic => OutcomeLink::Logit,
            _ => OutcomeLink::Identity,
        },
        method: if simex {
            CorrectionMethod::default_simex()
        } else {
            CorrectionMethod::RegressionCalibration
        },
        bootstrap_replicates: DEFAULT_BOOTSTRAP_REPLICATES,
        seed,
    };
    let estimate = prepared
        .estimate_continuous(
            ContinuousErrorInput {
                outcome: y,
                regressor: w,
                covariates,
                covariate_count,
                replicate_values: None,
                replicate_counts: None,
                spec,
            },
            &ctx(seed),
        )
        .ok()?;
    let summary = estimate.summary();
    Some(Reading {
        interval: estimate.interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            &format!("measurement.classical_continuous_error.{}", summary.method.as_str()),
            WALD,
            if summary.bootstrap_replicates > 0 {
                "whole_procedure_bootstrap"
            } else {
                "delta_method_with_declared_error_variance"
            },
            "declared_non_differential",
            "Frequentist",
            &format!("{}|error_in_{}", estimate.record().functional_id, summary.location.as_str()),
        ),
        rows: n as u64,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_binary_prevalence_wald_l95() {
    measure(
        "measurement_binary_prevalence_wald_l95",
        "latent_prevalence_0_3_sensitivity_0_9_specificity_0_8_estimated_validation_rates",
        WALD,
        0.3,
        0x4D01_0000,
        binary_prevalence_replicate,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_binary_risk_difference_wald_l95() {
    measure(
        "measurement_binary_risk_difference_wald_l95",
        "latent_risks_0_4_and_0_2_shared_estimated_validation_rates",
        WALD,
        0.2,
        0x4D02_0000,
        binary_risk_difference_replicate,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_classical_error_declared_variance_wald_l95() {
    measure(
        "measurement_classical_error_declared_variance_wald_l95",
        "linear_outcome_classical_error_variance_0_6_declared_exactly_coefficient_2",
        WALD,
        2.0,
        0x4D03_0000,
        |seed| classical_replicate(seed, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_classical_error_estimated_variance_wald_l95() {
    measure(
        "measurement_classical_error_estimated_variance_wald_l95",
        "linear_outcome_classical_error_variance_estimated_from_replicate_pairs_coefficient_2",
        WALD,
        2.0,
        0x4D04_0000,
        |seed| classical_replicate(seed, true),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_repeated_measurements_wald_l95() {
    measure(
        "measurement_repeated_measurements_wald_l95",
        "linear_outcome_two_or_three_replicates_per_unit_coefficient_1_5",
        WALD,
        1.5,
        0x4D05_0000,
        repeated_replicate,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_categorical_proportion_wald_l95() {
    // Truth: the declared proportion of the true category `low`.
    measure(
        "measurement_categorical_proportion_wald_l95",
        "three_categories_proportions_0_5_0_3_0_2_confusion_estimated_from_validation_proportion_low",
        WALD,
        CATEGORY_PROPORTIONS[0],
        0x4D20_0000,
        |seed| {
            categorical_replicate(
                seed,
                CategoricalTarget::Proportion("low".into()),
                "category_proportion",
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_categorical_outcome_mean_wald_l95() {
    // Truth: the mean outcome among units whose true category is `high`.
    measure(
        "measurement_categorical_outcome_mean_wald_l95",
        "three_categories_outcome_means_1_2_4_confusion_estimated_from_validation_mean_high",
        WALD,
        CATEGORY_MEANS[2],
        0x4D21_0000,
        |seed| {
            categorical_replicate(
                seed,
                CategoricalTarget::OutcomeMean("high".into()),
                "category_outcome_mean",
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_categorical_outcome_contrast_wald_l95() {
    // Truth: the difference of the true-category outcome means, high minus low.
    measure(
        "measurement_categorical_outcome_contrast_wald_l95",
        "three_categories_outcome_means_1_2_4_confusion_estimated_from_validation_high_minus_low",
        WALD,
        CATEGORY_MEANS[2] - CATEGORY_MEANS[0],
        0x4D22_0000,
        |seed| {
            categorical_replicate(
                seed,
                CategoricalTarget::OutcomeContrast { first: "high".into(), second: "low".into() },
                "category_outcome_contrast",
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_regression_calibration_linear_wald_l95() {
    measure(
        "measurement_regression_calibration_linear_wald_l95",
        "linear_outcome_classical_error_variance_0_25_declared_exactly_coefficient_2",
        WALD,
        ContinuousWorld::LinearTreatment.truth(),
        0x4D23_0000,
        |seed| continuous_replicate(seed, ContinuousWorld::LinearTreatment, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_regression_calibration_covariate_error_wald_l95() {
    measure(
        "measurement_regression_calibration_covariate_error_wald_l95",
        "linear_outcome_error_prone_confounder_variance_0_25_declared_exactly_treatment_coefficient_1",
        WALD,
        ContinuousWorld::LinearCovariate.truth(),
        0x4D24_0000,
        |seed| continuous_replicate(seed, ContinuousWorld::LinearCovariate, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_regression_calibration_logistic_wald_l95() {
    // Scored against the generating logit coefficient; calibration is first-order for a logit.
    measure(
        "measurement_regression_calibration_logistic_wald_l95",
        "logistic_outcome_classical_error_variance_0_25_declared_exactly_logit_coefficient_1",
        WALD,
        ContinuousWorld::Logistic.truth(),
        0x4D25_0000,
        |seed| continuous_replicate(seed, ContinuousWorld::Logistic, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_simex_linear_wald_l95() {
    // Scored against the generating coefficient; the quadratic extrapolant keeps its bias.
    measure(
        "measurement_simex_linear_wald_l95",
        "linear_outcome_classical_error_variance_0_25_declared_exactly_default_simex_coefficient_2",
        WALD,
        ContinuousWorld::LinearTreatment.truth(),
        0x4D26_0000,
        |seed| continuous_replicate(seed, ContinuousWorld::LinearTreatment, true),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_simex_logistic_wald_l95() {
    // Scored against the generating logit coefficient; the quadratic extrapolant keeps its bias.
    measure(
        "measurement_simex_logistic_wald_l95",
        "logistic_outcome_classical_error_variance_0_25_declared_exactly_default_simex_logit_coefficient_1",
        WALD,
        ContinuousWorld::Logistic.truth(),
        0x4D27_0000,
        |seed| continuous_replicate(seed, ContinuousWorld::Logistic, true),
    );
}

// ---------------------------------------------------------------------------
// Latent posterior route
// ---------------------------------------------------------------------------

fn posterior_model(conditioned: bool) -> MeasurementModel {
    model(
        MeasurementRoute::ExternalLatentPosterior,
        MeasurementKind::LearnedVectorRepresentation,
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
        ParameterEvidence::ExternalProvider { provider_id: "vendor".into() },
    )
}

/// `X ~ N(0,1)`, `W = X + N(0,1)`, `Y = 1 + 2 X + N(0,1)`.
///
/// Outcome-free posterior: the provider knows the measurement law, so `X` given `W` has mean
/// `W / 2` and variance 1/2.
///
/// Outcome-conditioned posterior: the provider knows the measurement law but not the outcome
/// model `Y = alpha + beta X + N(0, sigma^2)`, so a joint draw is a proper imputation: first
/// `(alpha, beta, sigma^2)` from their posterior given `(W, Y)`, then `X` given `(W, Y)` and that
/// draw. Marginally `Y | W ~ N(alpha + gamma W, tau^2)` with `gamma = beta / 2` and
/// `tau^2 = sigma^2 + beta^2 / 2`; under the prior `1 / tau^2` on `(alpha, gamma, tau^2)`
/// restricted to `sigma^2 > 0` the posterior is `tau^2 = RSS / chi^2(n - 2)`,
/// `gamma ~ N(slope, tau^2 / Sww)`, `alpha ~ N(ybar - gamma wbar, tau^2 / n)`, draws with
/// `sigma^2 <= 0` rejected. Given the parameters `X` has precision `2 + beta^2 / sigma^2` and mean
/// `(W + beta (Y - alpha) / sigma^2) / precision`. Imputing at the true coefficient instead
/// would leave out the between-imputation variance Rubin's rules count.
fn posterior_replicate(seed: u64, conditioned: bool) -> Option<Reading> {
    const DRAWS: usize = 40;
    let n = grid_n(300);
    let mut normal = gaussian(seed);
    let (mut y, mut w) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let x = normal();
        w.push(x + normal());
        y.push(1.0 + 2.0 * x + normal());
    }
    let mut draws = vec![0.0; n * DRAWS];
    if conditioned {
        let rows = n as f64;
        let w_mean = w.iter().sum::<f64>() / rows;
        let y_mean = y.iter().sum::<f64>() / rows;
        let sww: f64 = w.iter().map(|v| (v - w_mean).powi(2)).sum();
        let slope = w.iter().zip(&y).map(|(wi, yi)| (wi - w_mean) * yi).sum::<f64>() / sww;
        let rss: f64 =
            w.iter().zip(&y).map(|(wi, yi)| (yi - y_mean - slope * (wi - w_mean)).powi(2)).sum();
        let (mut s, mut attempts) = (0, 0);
        while s < DRAWS {
            attempts += 1;
            if attempts > 100 * DRAWS {
                return None;
            }
            let chi_square: f64 = (0..n - 2).map(|_| normal().powi(2)).sum();
            let tau2 = rss / chi_square;
            let gamma = slope + (tau2 / sww).sqrt() * normal();
            let alpha = y_mean - gamma * w_mean + (tau2 / rows).sqrt() * normal();
            let beta = 2.0 * gamma;
            let sigma2 = tau2 - 2.0 * gamma * gamma;
            if sigma2 <= 0.0 {
                continue;
            }
            let precision = 2.0 + beta * beta / sigma2;
            for i in 0..n {
                draws[s * n + i] = (w[i] + beta * (y[i] - alpha) / sigma2) / precision
                    + normal() / precision.sqrt();
            }
            s += 1;
        }
    } else {
        for s in 0..DRAWS {
            for i in 0..n {
                draws[s * n + i] = w[i] / 2.0 + 0.5_f64.sqrt() * normal();
            }
        }
    }
    let binding = DataBinding {
        population_id: "cohort".into(),
        regime_id: "observational".into(),
        arm_id: None,
        representation: scalar(),
    };
    let data = PosteriorLinearData {
        outcome: &y,
        draws: &draws,
        draw_count: DRAWS,
        covariates: &[],
        covariate_count: 0,
    };
    let effect = correct_latent_posterior_linear_effect(
        &posterior_model(conditioned),
        &binding,
        &data,
        &ctx(seed),
    )
    .ok()?;
    Some(Reading {
        interval: corrected_functional_interval(&effect.record, REPORTED_LEVEL).ok()?,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            if conditioned {
                "measurement.external_latent_posterior.multiple_imputation"
            } else {
                "measurement.external_latent_posterior.posterior_mean_substitution"
            },
            WALD,
            if conditioned {
                "rubin_total_variance"
            } else {
                "ordinary_least_squares_given_posterior_means"
            },
            "declared_non_differential",
            "Bayesian",
            "linear_coefficient",
        ),
        rows: n as u64,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_latent_posterior_mean_substitution_wald_l95() {
    measure(
        "measurement_latent_posterior_mean_substitution_wald_l95",
        "gaussian_latent_exact_posterior_given_indicator_coefficient_2",
        WALD,
        2.0,
        0x4D06_0000,
        |seed| posterior_replicate(seed, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_latent_posterior_multiple_imputation_wald_l95() {
    measure(
        "measurement_latent_posterior_multiple_imputation_wald_l95",
        "gaussian_latent_proper_posterior_given_indicator_and_outcome_coefficient_2",
        WALD,
        2.0,
        0x4D07_0000,
        |seed| posterior_replicate(seed, true),
    );
}

// ---------------------------------------------------------------------------
// Pooled differential transport
// ---------------------------------------------------------------------------

const SIMULATION: &str = "equal_tailed_simulation_interval";

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// A validation sample of `m` true positives and `m` true negatives of a population whose
/// error rates are `sensitivity` and `specificity`.
fn validation_of(
    next: &mut impl FnMut() -> f64,
    population: &str,
    m: u64,
    sensitivity: f64,
    specificity: f64,
) -> PopulationValidation {
    let true_positive = binomial(next, m, sensitivity);
    let true_negative = binomial(next, m, specificity);
    PopulationValidation {
        population: population.to_owned(),
        true_positive,
        false_negative: m - true_positive,
        true_negative,
        false_positive: m - true_negative,
    }
}

fn pooled_replicate(seed: u64, tau: f64, target_validated: bool) -> Option<Reading> {
    const SOURCES: usize = 6;
    const DRAWS: usize = 2000;
    let prevalence = 0.3;
    let (mu_sensitivity, mu_specificity) = (logit(0.9), logit(0.85));
    let m = grid_n(200) as u64;
    let total = grid_n(1000) as u64;
    let mut normal = gaussian(seed);
    let mut next = uniform(seed, 7);
    let mut sources = Vec::with_capacity(SOURCES);
    for k in 0..SOURCES {
        let sensitivity = expit(mu_sensitivity + tau * normal());
        let specificity = expit(mu_specificity + tau * normal());
        sources.push(validation_of(&mut next, &format!("source-{k}"), m, sensitivity, specificity));
    }
    let target_sensitivity = expit(mu_sensitivity + tau * normal());
    let target_specificity = expit(mu_specificity + tau * normal());
    let recorded =
        target_sensitivity * prevalence + (1.0 - target_specificity) * (1.0 - prevalence);
    let observed_positive = binomial(&mut next, total, recorded);
    let own = target_validated
        .then(|| validation_of(&mut next, "target", m, target_sensitivity, target_specificity));
    let correction = transport_pooled_differential_prevalence(
        PooledDifferentialRequest {
            sources: &sources,
            target_validation: own.as_ref(),
            target_population: "target",
            observed_positive,
            observed_total: total,
            draws: DRAWS,
        },
        &ctx(seed),
    )
    .ok()?;
    Some(Reading {
        interval: correction.interval,
        construction: construction(
            "MeasurementCorrectedFunctional",
            "Dag",
            "measurement_transport.pooled_differential",
            SIMULATION,
            "",
            if target_validated {
                "random_effects_logit_pool_with_target_validation"
            } else {
                "random_effects_logit_pool"
            },
            "Frequentist",
            "latent_prevalence_target",
        ),
        rows: total,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_pooled_differential_heterogeneous_l95() {
    measure(
        "measurement_pooled_differential_heterogeneous_l95",
        "six_sources_logit_error_rates_between_population_sd_0_3_target_unvalidated",
        SIMULATION,
        0.3,
        0x4D08_0000,
        |seed| pooled_replicate(seed, 0.3, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_pooled_differential_target_validated_l95() {
    measure(
        "measurement_pooled_differential_target_validated_l95",
        "six_sources_logit_error_rates_between_population_sd_0_3_target_validated",
        SIMULATION,
        0.3,
        0x4D09_0000,
        |seed| pooled_replicate(seed, 0.3, true),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_pooled_differential_homogeneous_l95() {
    // Conservative by construction: the between-population variance is at its boundary zero and
    // every drawn variance is at least zero, so no draw's predictive law is narrower than the
    // true one.
    tally_of(
        "measurement_pooled_differential_homogeneous_l95",
        "six_sources_identical_error_rates_target_unvalidated",
        SIMULATION,
        0.3,
        0x4D0A_0000,
        |seed| pooled_replicate(seed, 0.0, false),
    )
    .assert_one_sided(
        CoverageSide::Upper,
        "the target prevalence covered at least nominally when the error rates do not differ by population",
    );
}

// ---------------------------------------------------------------------------
// Temporal misclassification
// ---------------------------------------------------------------------------

const PERIODS: usize = 6;

/// Which published interval of the temporal design a replicate scores.
#[derive(Clone, Copy)]
struct Coordinate {
    /// Index of the coordinate in the published section.
    index: usize,
    /// The coordinate's value under the data law.
    truth: f64,
}

/// The excursion contrast `0.25`.
const CONTRAST: Coordinate = Coordinate { index: 0, truth: 0.25 };
/// The control arm mean `0.3`.
const CONTROL_MEAN: Coordinate = Coordinate { index: 2, truth: 0.3 };

/// The functional label without the estimated Youden index the frequentist route appends to it:
/// the index is a value of the replicate, not part of the construction.
fn functional_label(functional: &str) -> String {
    const MARK: &str = " (Youden index ";
    let Some(start) = functional.find(MARK) else {
        return functional.to_owned();
    };
    let end = functional[start..].find(')').map_or(functional.len(), |close| start + close + 1);
    format!("{}{}", &functional[..start], &functional[end..])
}

fn temporal_replicate(
    seed: u64,
    inference: InferenceMode,
    coordinate: Coordinate,
    method: &'static str,
) -> Option<Reading> {
    let (sensitivity, specificity) = (0.9, 0.85);
    let units = grid_n(200);
    let m = grid_n(200) as u64;
    let mut next = uniform(seed, 3);
    let (mut labels, mut periods, mut assignment, mut recorded) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        let label: std::sync::Arc<str> = format!("u{i}").into();
        for t in 0..PERIODS {
            let treated = next() < 0.5;
            let truth = next() < 0.3 + if treated { 0.25 } else { 0.0 };
            let observed = if truth { next() < sensitivity } else { next() >= specificity };
            labels.push(label.clone());
            periods.push(t as i64);
            assignment.push(treated);
            recorded.push(f64::from(u8::from(observed)));
        }
    }
    let rows = labels.len();
    let true_positive = binomial(&mut next, m, sensitivity);
    let true_negative = binomial(&mut next, m, specificity);
    let counts = OutcomeMisclassification {
        true_positive,
        false_negative: m - true_positive,
        true_negative,
        false_positive: m - true_negative,
    };
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels.into(),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.as_slice().into(),
            probability: vec![0.5; rows].into(),
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.05,
        outcome_misclassification: Some(counts),
    };
    temporal_reading(&recorded, query, inference, coordinate, method, seed)
}

/// Run one temporal design on the recorded outcome and read the scored coordinate's interval.
fn temporal_reading(
    recorded: &[f64],
    query: TemporalDesignQuery,
    inference: InferenceMode,
    coordinate: Coordinate,
    method: &'static str,
    seed: u64,
) -> Option<Reading> {
    let data = TabularData::from_f64_columns([("y", recorded)]).ok()?;
    let result = Study::tabular(data)
        .query(CausalQuery::TemporalDesign(query))
        .inference(inference)
        .build()
        .ok()?
        .run(&ctx(seed))
        .ok()?;
    let fit = result.temporal_design()?;
    let (lower, upper) = (*fit.lower.get(coordinate.index)?, *fit.upper.get(coordinate.index)?);
    let name = fit.coordinates.get(coordinate.index)?;
    let bayesian = method == TEMPORAL_CREDIBLE;
    Some(Reading {
        interval: (lower, upper),
        construction: Construction {
            query: "TemporalDesign".into(),
            graph_class: "RandomizedTrial".into(),
            structure: "design".into(),
            modality: "tabular".into(),
            inference: if bayesian { "Bayesian" } else { "Frequentist" }.into(),
            estimator: result.logical_plan.estimator.as_deref().unwrap_or("").to_owned(),
            interval_method: method.into(),
            se_kind: fit.uncertainty.to_string(),
            dependence: "cluster".into(),
            posterior: String::new(),
            functional: format!("{}|{name}", functional_label(&fit.functional.to_string())),
            identification: "point".into(),
            reported_level: REPORTED_LEVEL,
        },
        rows: fit.pairs,
    })
}

const TEMPORAL_NORMAL: &str = "temporal_design_cluster_robust_normal";
const TEMPORAL_CREDIBLE: &str = "temporal_design_posterior_quantile";

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1000))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_frequentist_contrast_l95() {
    measure(
        "temporal_misclassification_frequentist_contrast_l95",
        "randomized_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_contrast_0_25",
        TEMPORAL_NORMAL,
        CONTRAST.truth,
        0x4D0B_0000,
        |seed| temporal_replicate(seed, InferenceMode::Frequentist, CONTRAST, TEMPORAL_NORMAL),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_frequentist_control_mean_l95() {
    measure(
        "temporal_misclassification_frequentist_control_mean_l95",
        "randomized_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_control_mean_0_3",
        TEMPORAL_NORMAL,
        CONTROL_MEAN.truth,
        0x4D0C_0000,
        |seed| temporal_replicate(seed, InferenceMode::Frequentist, CONTROL_MEAN, TEMPORAL_NORMAL),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_bayesian_contrast_l95() {
    measure(
        "temporal_misclassification_bayesian_contrast_l95",
        "randomized_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_contrast_0_25_bayesian_bootstrap",
        TEMPORAL_CREDIBLE,
        CONTRAST.truth,
        0x4D0D_0000,
        |seed| temporal_replicate(seed, bayesian(), CONTRAST, TEMPORAL_CREDIBLE),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_bayesian_control_mean_l95() {
    measure(
        "temporal_misclassification_bayesian_control_mean_l95",
        "randomized_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_control_mean_0_3_bayesian_bootstrap",
        TEMPORAL_CREDIBLE,
        CONTROL_MEAN.truth,
        0x4D0E_0000,
        |seed| temporal_replicate(seed, bayesian(), CONTROL_MEAN, TEMPORAL_CREDIBLE),
    );
}

/// A sequentially randomized design other than the two-arm one.
#[derive(Clone, Copy)]
enum Panel {
    /// Three arms; `ordered` declares them as ordered categorical levels.
    Arms {
        ordered: bool,
    },
    Factorial,
    Interference,
}

/// `P(Y = 1)` in the arms `low, mid, high`, assigned with probabilities 0.4, 0.3, 0.3.
const ARM_RISKS: [f64; 3] = [0.2, 0.35, 0.5];
const ARM_PROBABILITIES: [f64; 3] = [0.4, 0.3, 0.3];

/// `P(Y = 1)` in the factorial cell `(first, second)`.
fn factorial_risk(first: bool, second: bool) -> f64 {
    let (a, b) = (f64::from(u8::from(first)), f64::from(u8::from(second)));
    0.2 + 0.2 * a + 0.1 * b + 0.1 * a * b
}

/// `P(Y = 1)` under a unit's own assignment and its cluster peer's.
fn exposure_risk(own: bool, peer: bool) -> f64 {
    0.2 + 0.25 * f64::from(u8::from(own)) + 0.15 * f64::from(u8::from(peer))
}

/// Contrast of the middle arm against the reference: `0.35 - 0.2`.
const MID_CONTRAST: Coordinate = Coordinate { index: 0, truth: 0.15 };
/// Reference arm mean of the three-arm design (after its two contrasts).
const REFERENCE_MEAN: Coordinate = Coordinate { index: 2, truth: 0.2 };
/// Ordered step from the middle to the top level (after the two contrasts): `0.5 - 0.35`.
const TOP_STEP: Coordinate = Coordinate { index: 2, truth: 0.15 };
/// Main effect of the first factor averaged over the second: `(0.2 + 0.3) / 2`.
const FIRST_MAIN_EFFECT: Coordinate = Coordinate { index: 0, truth: 0.25 };
/// Interaction `p11 - p10 - p01 + p00 = 0.6 - 0.4 - 0.3 + 0.2`.
const INTERACTION: Coordinate = Coordinate { index: 2, truth: 0.1 };
/// Direct effect with no treated peer: `0.45 - 0.2`.
const DIRECT_EFFECT: Coordinate = Coordinate { index: 0, truth: 0.25 };
/// Spillover effect on an untreated unit: `0.35 - 0.2`.
const SPILLOVER_EFFECT: Coordinate = Coordinate { index: 1, truth: 0.15 };

/// A binary outcome with the given risk, recorded with sensitivity 0.9 and specificity 0.85.
fn recorded_outcome(next: &mut impl FnMut() -> f64, risk: f64) -> f64 {
    let truth = next() < risk;
    let observed = if truth { next() < 0.9 } else { next() >= 0.85 };
    f64::from(u8::from(observed))
}

/// The frequentist interval of one coordinate of `panel` on a misclassified binary outcome
/// with an independent validation sample.
fn temporal_panel_replicate(seed: u64, panel: Panel, coordinate: Coordinate) -> Option<Reading> {
    let units = grid_n(200);
    let m = grid_n(200) as u64;
    let mut next = uniform(seed, 3);
    let (mut labels, mut periods, mut recorded): (Vec<std::sync::Arc<str>>, Vec<i64>, Vec<f64>) =
        (Vec::new(), Vec::new(), Vec::new());
    let (assignment, min_probability) = match panel {
        Panel::Arms { ordered } => {
            let (mut arm_of, mut probabilities) = (Vec::new(), Vec::new());
            for i in 0..units {
                let label: std::sync::Arc<str> = format!("u{i}").into();
                for t in 0..PERIODS {
                    let arm = category_of(next(), ARM_PROBABILITIES);
                    labels.push(label.clone());
                    periods.push(t as i64);
                    arm_of.push(arm);
                    probabilities.push(ARM_PROBABILITIES.to_vec());
                    recorded.push(recorded_outcome(&mut next, ARM_RISKS[arm]));
                }
            }
            let arms: Vec<std::sync::Arc<str>> =
                ["low", "mid", "high"].map(std::sync::Arc::<str>::from).to_vec();
            (
                TemporalAssignment::MultiArm {
                    assignment: arm_of.as_slice().into(),
                    probabilities: probabilities.into(),
                    arms: arms.into(),
                    categorical: ordered,
                    ordered,
                },
                0.1,
            )
        }
        Panel::Factorial => {
            let (mut first, mut second) = (Vec::new(), Vec::new());
            for i in 0..units {
                let label: std::sync::Arc<str> = format!("u{i}").into();
                for t in 0..PERIODS {
                    let (a, b) = (next() < 0.5, next() < 0.4);
                    labels.push(label.clone());
                    periods.push(t as i64);
                    first.push(a);
                    second.push(b);
                    recorded.push(recorded_outcome(&mut next, factorial_risk(a, b)));
                }
            }
            let rows = labels.len();
            (
                TemporalAssignment::Factorial {
                    first: first.as_slice().into(),
                    first_probability: vec![0.5; rows].into(),
                    second: second.as_slice().into(),
                    second_probability: vec![0.4; rows].into(),
                },
                0.05,
            )
        }
        Panel::Interference => {
            // `units` clusters of two units each.
            let (mut treated, mut clusters): (Vec<bool>, Vec<std::sync::Arc<str>>) =
                (Vec::new(), Vec::new());
            for c in 0..units {
                let cluster: std::sync::Arc<str> = format!("c{c}").into();
                for t in 0..PERIODS {
                    let pair = [next() < 0.5, next() < 0.5];
                    for member in 0..2 {
                        labels.push(format!("u{c}_{member}").into());
                        periods.push(t as i64);
                        clusters.push(cluster.clone());
                        treated.push(pair[member]);
                        recorded.push(recorded_outcome(
                            &mut next,
                            exposure_risk(pair[member], pair[1 - member]),
                        ));
                    }
                }
            }
            let rows = labels.len();
            (
                TemporalAssignment::Interference {
                    assignment: treated.as_slice().into(),
                    probability: vec![0.5; rows].into(),
                    clusters: clusters.into(),
                },
                0.05,
            )
        }
    };
    let true_positive = binomial(&mut next, m, 0.9);
    let true_negative = binomial(&mut next, m, 0.85);
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels.into(),
        periods: periods.as_slice().into(),
        lag: 0,
        assignment,
        functional: TemporalDesignFunctional::Mean,
        min_probability,
        outcome_misclassification: Some(OutcomeMisclassification {
            true_positive,
            false_negative: m - true_positive,
            true_negative,
            false_positive: m - true_negative,
        }),
    };
    temporal_reading(
        &recorded,
        query,
        InferenceMode::Frequentist,
        coordinate,
        TEMPORAL_NORMAL,
        seed,
    )
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_three_arm_contrast_l95() {
    measure(
        "temporal_misclassification_three_arm_contrast_l95",
        "randomized_three_arm_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_mid_minus_low_0_15",
        TEMPORAL_NORMAL,
        MID_CONTRAST.truth,
        0x4D30_0000,
        |seed| temporal_panel_replicate(seed, Panel::Arms { ordered: false }, MID_CONTRAST),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_three_arm_reference_mean_l95() {
    measure(
        "temporal_misclassification_three_arm_reference_mean_l95",
        "randomized_three_arm_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_reference_mean_0_2",
        TEMPORAL_NORMAL,
        REFERENCE_MEAN.truth,
        0x4D31_0000,
        |seed| temporal_panel_replicate(seed, Panel::Arms { ordered: false }, REFERENCE_MEAN),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_ordered_levels_step_l95() {
    measure(
        "temporal_misclassification_ordered_levels_step_l95",
        "randomized_ordered_levels_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_high_minus_mid_0_15",
        TEMPORAL_NORMAL,
        TOP_STEP.truth,
        0x4D32_0000,
        |seed| temporal_panel_replicate(seed, Panel::Arms { ordered: true }, TOP_STEP),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_factorial_main_effect_l95() {
    measure(
        "temporal_misclassification_factorial_main_effect_l95",
        "randomized_factorial_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_first_main_effect_0_25",
        TEMPORAL_NORMAL,
        FIRST_MAIN_EFFECT.truth,
        0x4D33_0000,
        |seed| temporal_panel_replicate(seed, Panel::Factorial, FIRST_MAIN_EFFECT),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_factorial_interaction_l95() {
    measure(
        "temporal_misclassification_factorial_interaction_l95",
        "randomized_factorial_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_interaction_0_1",
        TEMPORAL_NORMAL,
        INTERACTION.truth,
        0x4D34_0000,
        |seed| temporal_panel_replicate(seed, Panel::Factorial, INTERACTION),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_interference_direct_effect_l95() {
    measure(
        "temporal_misclassification_interference_direct_effect_l95",
        "randomized_paired_interference_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_direct_effect_0_25",
        TEMPORAL_NORMAL,
        DIRECT_EFFECT.truth,
        0x4D35_0000,
        |seed| temporal_panel_replicate(seed, Panel::Interference, DIRECT_EFFECT),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_misclassification_interference_spillover_effect_l95() {
    measure(
        "temporal_misclassification_interference_spillover_effect_l95",
        "randomized_paired_interference_panel_binary_outcome_recorded_with_sensitivity_0_9_specificity_0_85_spillover_effect_0_15",
        TEMPORAL_NORMAL,
        SPILLOVER_EFFECT.truth,
        0x4D36_0000,
        |seed| temporal_panel_replicate(seed, Panel::Interference, SPILLOVER_EFFECT),
    );
}

// ---------------------------------------------------------------------------
// Posterior composition
// ---------------------------------------------------------------------------

const COMPOSED_LEVEL: f64 = 0.95;
const POSTERIOR_DRAWS: usize = 200;

#[derive(Clone, Copy)]
enum Law {
    /// Bivariate normal, means 1.0 and 0.4, unit variances, correlation 0.6; contrast truth 0.6.
    Gaussian,
    /// Shared-evidence skewed pair `x = E1`, `y = 0.5 E1 + 0.5 E2`, exponential `E`; truth 0.
    Skewed,
}

impl Law {
    fn truth(self) -> f64 {
        match self {
            Self::Gaussian => 0.6,
            Self::Skewed => 0.0,
        }
    }
}

fn rows_of(law: Law, n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    match law {
        Law::Gaussian => {
            let mut normal = gaussian(seed);
            let rho = 0.6_f64;
            for _ in 0..n {
                let (z1, z2) = (normal(), normal());
                x.push(1.0 + z1);
                y.push(0.4 + rho * z1 + (1.0 - rho * rho).sqrt() * z2);
            }
        }
        Law::Skewed => {
            let mut next = uniform(seed, 1);
            for _ in 0..n {
                let (e1, e2) = (-next().ln(), -next().ln());
                x.push(e1);
                y.push(0.5 * e1 + 0.5 * e2);
            }
        }
    }
    (x, y)
}

fn bind_basis(tally: &mut CoverageTally, basis: &CalibrationBasis) {
    tally.bind_validation(
        &Construction {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            identification: basis.identification.to_string(),
            reported_level: basis.level,
        },
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
        &basis.validation,
    );
}

/// The normal interval a claim reports and the key it is measured under.
fn reading_of(claim: &PropagationClaim, rows: usize) -> (Option<(f64, f64)>, CalibrationBasis) {
    let basis = claim
        .composed_interval_basis(NORMAL_APPROXIMATION_CONSTRUCTION, rows as u64)
        .expect("composed key");
    let interval = match (claim.value("mean"), claim.value("variance")) {
        (Some(m), Some(v)) => {
            normal_interval_candidate(m, v, COMPOSED_LEVEL).ok().map(|c| (c.lower, c.upper))
        }
        _ => None,
    };
    (interval, basis)
}

/// Aligned Bayesian-bootstrap posterior draws of two sample means: one set of Exp(1) row weights
/// per draw, shared by both means.
fn posterior_mean_draws(x: &[f64], y: &[f64], seed: u64) -> (PropagationDraws, PropagationDraws) {
    let mut next = uniform(seed, 3);
    let (mut px, mut py) =
        (Vec::with_capacity(POSTERIOR_DRAWS), Vec::with_capacity(POSTERIOR_DRAWS));
    for _ in 0..POSTERIOR_DRAWS {
        let (mut total, mut sx, mut sy) = (0.0, 0.0, 0.0);
        for i in 0..x.len() {
            let weight = -next().ln();
            total += weight;
            sx += weight * x[i];
            sy += weight * y[i];
        }
        px.push(sx / total);
        py.push(sy / total);
    }
    let alignment = format!("posterior-{seed}");
    (
        PropagationDraws::aligned("x", "shared_rows", alignment.clone(), px),
        PropagationDraws::aligned("y", "shared_rows", alignment, py),
    )
}

/// The contrast of the two posterior means.
fn posterior_composition(test: &'static str, dgp: &'static str, law: Law) {
    let n = grid_n(400);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: NORMAL_APPROXIMATION_CONSTRUCTION },
        COMPOSED_LEVEL,
    );
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x4D10_0000 + replicate);
        let (x, y) = rows_of(law, n, seed);
        let (dx, dy) = posterior_mean_draws(&x, &y, seed);
        let claim = propagate_aligned_linear_spread(
            PropagationOperation::Contrast,
            (1.0, -1.0),
            &dx,
            &dy,
            &ctx(seed),
        )
        .expect("claim");
        reading_of(&claim, n)
    });
    for (interval, basis) in results {
        bind_basis(&mut tally, &basis);
        tally.record(interval, law.truth());
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_contrast_aligned_posterior_gaussian_l95() {
    posterior_composition(
        "composed_contrast_aligned_posterior_gaussian_l95",
        "bivariate_normal_rho_0.6_aligned_bayesian_bootstrap_posterior_of_the_sample_means",
        Law::Gaussian,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_contrast_aligned_posterior_skewed_pair_l95() {
    posterior_composition(
        "composed_contrast_aligned_posterior_skewed_pair_l95",
        "exponential_pair_shared_evidence_aligned_bayesian_bootstrap_posterior_of_the_sample_means",
        Law::Skewed,
    );
}

/// A composition of the two means other than their contrast.
#[derive(Clone, Copy)]
enum Composition {
    /// `x + y`.
    Sum,
    /// `0.7 x + 0.3 y`.
    WeightedSum,
    /// `x y`.
    Product,
    /// `y / x`.
    Ratio,
}

impl Composition {
    /// The composition of the population means `(mx, my)`: (1.0, 0.4) for the Gaussian law and
    /// (1.0, 1.0) for the skewed pair.
    fn truth(self, law: Law) -> f64 {
        let (mx, my) = match law {
            Law::Gaussian => (1.0, 0.4),
            Law::Skewed => (1.0, 1.0),
        };
        match self {
            Self::Sum => mx + my,
            Self::WeightedSum => 0.7 * mx + 0.3 * my,
            Self::Product => mx * my,
            Self::Ratio => my / mx,
        }
    }

    /// The operation and coefficients of a linear composition.
    fn linear(self) -> Option<(PropagationOperation, (f64, f64))> {
        match self {
            Self::Sum => Some((PropagationOperation::Sum, (1.0, 1.0))),
            Self::WeightedSum => Some((PropagationOperation::WeightedSum, (0.7, 0.3))),
            Self::Product | Self::Ratio => None,
        }
    }
}

fn score_composition(
    test: &'static str,
    dgp: &'static str,
    truth: f64,
    results: Vec<(Option<(f64, f64)>, CalibrationBasis)>,
) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: NORMAL_APPROXIMATION_CONSTRUCTION },
        COMPOSED_LEVEL,
    );
    for (interval, basis) in results {
        bind_basis(&mut tally, &basis);
        tally.record(interval, truth);
    }
    tally.assert();
}

/// `composition` of the aligned posterior draws of the two means.
fn aligned_composition(
    test: &'static str,
    dgp: &'static str,
    law: Law,
    composition: Composition,
    seed_base: u64,
) {
    let n = grid_n(400);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(seed_base + replicate);
        let (x, y) = rows_of(law, n, seed);
        let (dx, dy) = posterior_mean_draws(&x, &y, seed);
        let claim = match (composition.linear(), composition) {
            (Some((operation, coefficients)), _) => {
                propagate_aligned_linear_spread(operation, coefficients, &dx, &dy, &ctx(seed))
            }
            (None, Composition::Ratio) => propagate_spread(
                PropagationOperation::NonlinearFunctional,
                &dy,
                &dx,
                ValueFunctional::Ratio,
                &ctx(seed),
            ),
            (None, _) => propagate_spread(
                PropagationOperation::Product,
                &dx,
                &dy,
                ValueFunctional::Product,
                &ctx(seed),
            ),
        }
        .expect("claim");
        reading_of(&claim, n)
    });
    score_composition(test, dgp, composition.truth(law), results);
}

/// A linear `composition` of the two sample means on the linear-moments route, the covariance
/// of the means estimated from the shared rows.
fn shared_rows_composition(
    test: &'static str,
    dgp: &'static str,
    law: Law,
    composition: Composition,
    seed_base: u64,
) {
    let n = grid_n(400);
    let (operation, (a, b)) = composition.linear().expect("a linear composition");
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(seed_base + replicate);
        let (x, y) = rows_of(law, n, seed);
        let rows = n as f64;
        let (mx, my) = (x.iter().sum::<f64>() / rows, y.iter().sum::<f64>() / rows);
        let (mut vx, mut vy, mut cross) = (0.0, 0.0, 0.0);
        for i in 0..n {
            vx += (x[i] - mx) * (x[i] - mx);
            vy += (y[i] - my) * (y[i] - my);
            cross += (x[i] - mx) * (y[i] - my);
        }
        // Covariance of the two means: the sample covariance of the rows over `n`.
        let scale = 1.0 / ((rows - 1.0) * rows);
        let input = |id: &str, mean: f64| MomentInput {
            id: id.into(),
            evidence_id: "shared_rows".into(),
            role: InputRole::Estimate,
            mean,
        };
        let claim = propagate_linear_moments_spread(
            operation,
            &[input("x", mx), input("y", my)],
            &[a, b],
            &[vx * scale, cross * scale, cross * scale, vy * scale],
            &LinearDependence::Covariance { id: "estimated_from_the_rows".into() },
            &ctx(seed),
        )
        .expect("claim");
        reading_of(&claim, n)
    });
    score_composition(test, dgp, composition.truth(law), results);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_sum_shared_rows_gaussian_l95() {
    shared_rows_composition(
        "composed_sum_shared_rows_gaussian_l95",
        "bivariate_normal_rho_0.6_sum_of_the_sample_means_covariance_estimated_from_the_rows",
        Law::Gaussian,
        Composition::Sum,
        0x4D40_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_weighted_sum_shared_rows_gaussian_l95() {
    shared_rows_composition(
        "composed_weighted_sum_shared_rows_gaussian_l95",
        "bivariate_normal_rho_0.6_weights_0.7_0.3_of_the_sample_means_covariance_estimated_from_the_rows",
        Law::Gaussian,
        Composition::WeightedSum,
        0x4D41_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_sum_aligned_posterior_gaussian_l95() {
    aligned_composition(
        "composed_sum_aligned_posterior_gaussian_l95",
        "bivariate_normal_rho_0.6_sum_of_the_aligned_bayesian_bootstrap_posterior_means",
        Law::Gaussian,
        Composition::Sum,
        0x4D42_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_sum_aligned_posterior_skewed_pair_l95() {
    aligned_composition(
        "composed_sum_aligned_posterior_skewed_pair_l95",
        "exponential_pair_shared_evidence_sum_of_the_aligned_bayesian_bootstrap_posterior_means",
        Law::Skewed,
        Composition::Sum,
        0x4D43_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_weighted_sum_aligned_posterior_gaussian_l95() {
    aligned_composition(
        "composed_weighted_sum_aligned_posterior_gaussian_l95",
        "bivariate_normal_rho_0.6_weights_0.7_0.3_of_the_aligned_bayesian_bootstrap_posterior_means",
        Law::Gaussian,
        Composition::WeightedSum,
        0x4D44_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_product_aligned_posterior_gaussian_l95() {
    aligned_composition(
        "composed_product_aligned_posterior_gaussian_l95",
        "bivariate_normal_rho_0.6_product_of_the_aligned_bayesian_bootstrap_posterior_means",
        Law::Gaussian,
        Composition::Product,
        0x4D45_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_product_aligned_posterior_skewed_pair_l95() {
    aligned_composition(
        "composed_product_aligned_posterior_skewed_pair_l95",
        "exponential_pair_shared_evidence_product_of_the_aligned_bayesian_bootstrap_posterior_means",
        Law::Skewed,
        Composition::Product,
        0x4D46_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn composed_ratio_aligned_posterior_gaussian_l95() {
    aligned_composition(
        "composed_ratio_aligned_posterior_gaussian_l95",
        "bivariate_normal_rho_0.6_ratio_y_over_x_of_the_aligned_bayesian_bootstrap_posterior_means",
        Law::Gaussian,
        Composition::Ratio,
        0x4D47_0000,
    );
}
