//! Coverage designs for the 95% credible intervals of the Bayesian randomized-design routes,
//! scored through the public study facade against known truths.
//!
//! Every design runs one route on simulated data and scores the equal-tailed 95% posterior
//! interval the route publishes against an analytically known target. The record's construction
//! is read from the runtime's own calibration key (`common::calibration_bind::constructions`).
//! A graphless design route is keyed on the design axes (its query, structure `design`,
//! validation `not_applicable`), as the randomized design routes are; nothing else is declared. A replicate the route refuses or publishes no posterior for is a skip,
//! which counts as a miss.
//!
//! * `bayesian.randomized_finite_population`: the fixed units of the randomized-trial design
//!   (effect `1.4 + 0.2 * x1`), completely randomized; the truth is the finite-population
//!   average effect of those units and only the assignment is random. The route publishes the
//!   contrast under additive unit effects (Neyman's variance bound); the unit effects here vary
//!   by `0.2 * x1` only, under half a percent of the bound, so the bound is attained and the
//!   interval is held to the nominal band on both sides.
//! * `bayesian.randomized_superpopulation`: independent units with Gaussian covariates, so each
//!   arm's outcome is exactly Gaussian, and independent Bernoulli arms; the truths are the
//!   population contrasts (1.4 and 0.2).
//! * `bayesian.switchback_effect`: sequences with compound-symmetric errors at the declared
//!   correlation and a constant direct effect of 0.7.
//! * `bayesian.complier_effect`: Bernoulli encouragement with one-sided noncompliance and a
//!   constant effect of 2 on compliers.
//! * `bayesian.piecewise_survival`: exponential hazards 0.4 (control) and 0.25 (treated) with
//!   random and administrative censoring; the truths are the survival difference and the
//!   restricted mean survival time difference at the horizon.
//! * `bayesian.continuous_dose_policy`: two baseline groups with exactly quadratic dose-response
//!   curves; the truth is the Epanechnikov window average of each curve at the policy and
//!   reference doses, weighted by the observed group shares.
//! * `interference.cluster_bayesian_bootstrap`: two-unit clusters, complete cluster
//!   randomization, a constant total effect of 3, from nine clusters per arm upward; below
//!   fifty clusters per arm the route publishes its Student-t reference law.
//! * `bayesian.gcomp.model_based_arm_effect`: a probit-confounded binary treatment whose effect
//!   `1 + 2 z` grows in the confounder; the truth is the mean of that effect over the treated
//!   (or untreated) rows of the replicate, the model's own estimand.
//!
//! The designs are `#[ignore]`d measurement designs; the fixed-seed checks of the same routes are
//! `bayesian_randomized_routes.rs`, `bayesian_design_posterior_routes.rs`,
//! `bayesian_arm_effect_routes.rs` and `interference_estimators_known_truth.rs`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "simulation dimensions are small counts"
)]

mod common;

use std::sync::Arc;

use antecedent::{
    BayesianConfig, DesignPosteriorModel, EstimatorId, InferenceMode, InterferenceSpec,
    RandomizedEffectQuery, RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    AssignmentDesign, AverageEffectQuery, CausalQuery, ContinuousDoseResponseQuery,
    ExecutionContext, ExposureLevel, ExposureMapping, FixedGroupDosePolicy, InterferenceFunctional,
    InterferenceQuery, ObservationAssumption, RandomizationDesign, SurvivalFunctional,
    SurvivalQuery, TargetPopulation, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::dose_basis_regression::ResidualPrior;
use antecedent_estimate::randomized_posterior::{ArmBetaPrior, ArmNormalPrior, ArmOutcomeModel};
use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel, SwitchbackPrior,
};
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, stream_seed, unit_uniform,
};
use common::calibration_bind::{bind_reported, constructions};

const INTERVAL: &str = "posterior_quantile";
const DRAWS: usize = 2_000;

type Reported = Vec<(Construction, ScopeFacts, &'static str)>;

/// One scored replicate: the published interval, the known truth and the runtime's construction.
struct Replicate {
    interval: (f64, f64),
    truth: f64,
    reported: Reported,
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn labels(n: usize, prefix: &str) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

/// Prepare and estimate one study, returning its result and the construction the runtime reports.
/// A refusal is `None`.
fn execute(
    builder: StudyBuilder,
    data: &TabularData,
    seed: u64,
) -> Option<(StudyResult, Reported)> {
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = builder.build().ok()?.prepare(&ctx).ok()?;
    let result = prepared.estimate(data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("the execution contract");
    let reported = constructions(&contract, &result);
    Some((result, reported))
}

/// The posterior's published 95% interval of one column, `None` when it is absent or not ordered.
fn posterior_interval(result: &StudyResult, column: usize) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let (lower, upper) = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some((lower, upper))
}

/// Score every replicate against its truth under one record.
///
/// A graphless design route has no graph and runs no validation suite, and the runtime keys
/// it with no query. Its record is keyed on the design axes instead: `design_query` names the
/// query, the structure is `design` and no validation mode applies. Every other field of the
/// construction is the runtime's. A route on a graph (`None`) binds the runtime's key as is.
fn measure(
    test: &'static str,
    dgp: &'static str,
    design_query: Option<&'static str>,
    runs: &[Option<Replicate>],
) {
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL);
    for run in runs {
        let Some(replicate) = run else {
            tally.skip();
            continue;
        };
        match design_query {
            Some(query) => {
                let (construction, scope, _) = replicate
                    .reported
                    .iter()
                    .find(|(construction, _, _)| construction.interval_method == INTERVAL)
                    .expect("the execution reports its posterior interval");
                let mut construction = construction.clone();
                construction.query = query.into();
                construction.structure = "design".into();
                tally.bind_validation(&construction, *scope, "not_applicable");
            }
            None => bind_reported(&mut tally, &replicate.reported),
        }
        tally.record(Some(replicate.interval), replicate.truth);
    }
    tally.assert();
}

/// `treated` units chosen uniformly without replacement.
fn complete_assignment(n: usize, treated: usize, seed: u64) -> Vec<bool> {
    let keys: Vec<f64> = (0..n).map(|i| unit_uniform(stream_seed(seed, i as u64))).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
    let mut assigned = vec![false; n];
    for &unit in &order[..treated] {
        assigned[unit] = true;
    }
    assigned
}

// ------------------------------------------------------------------ randomized arm contrasts

fn complete_query(assigned: Vec<bool>) -> RandomizedEffectQuery {
    let n = assigned.len();
    let treated = assigned.iter().filter(|a| **a).count();
    RandomizedEffectQuery::with_design(
        RandomizationDesign::Complete { treated_units: treated },
        v(0),
        assigned,
        vec![treated as f64 / n as f64; n],
        labels(n, "unit"),
        labels(n, "row"),
        ("control", "treated"),
    )
}

fn arm_inference(model: ArmOutcomeModel) -> InferenceMode {
    InferenceMode::Bayesian(
        BayesianConfig::conjugate()
            .n_draws(DRAWS)
            .design_model(DesignPosteriorModel::ArmOutcome { model }),
    )
}

fn weak_gaussian_arms() -> ArmOutcomeModel {
    ArmOutcomeModel::Gaussian(ArmNormalPrior {
        mean: 0.0,
        pseudo_units: 0.1,
        variance_shape: 1.0,
        variance_scale: 1.0,
    })
}

/// The fixed units of the design: baseline covariates are deterministic sequences and the unit
/// effect is `1.4 + 0.2 * x1`. The randomization variance of the arm-mean contrast is
/// `S1^2 / n1 + S0^2 / n0 - S_tau^2 / n` with `S_tau^2 = 0.04 Var(x1)`, about 0.02 against arm
/// variances above one, so Neyman's bound exceeds it by under half a percent.
fn fixed_unit_outcomes(n: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let x1: Vec<f64> = (0..n).map(|i| (0.17 * i as f64).sin()).collect();
    let x2: Vec<f64> = (0..n).map(|i| (0.11 * i as f64).cos()).collect();
    let control: Vec<f64> =
        (0..n).map(|i| 2.0 + 1.2 * x1[i] - 0.7 * x2[i] + 0.5 * (0.29 * i as f64).sin()).collect();
    let treated: Vec<f64> = (0..n).map(|i| control[i] + 1.4 + 0.2 * x1[i]).collect();
    let truth = x1.iter().map(|x| 1.4 + 0.2 * x).sum::<f64>() / n as f64;
    (control, treated, truth)
}

fn arm_replicate(
    outcome: &[f64],
    assigned: Vec<bool>,
    model: ArmOutcomeModel,
    estimator: Option<EstimatorId>,
    truth: f64,
    seed: u64,
) -> Option<Replicate> {
    let data = TabularData::from_f64_columns([("outcome", outcome)]).ok()?;
    let mut builder = Study::tabular(data.clone())
        .query(complete_query(assigned))
        .inference(arm_inference(model));
    if let Some(estimator) = estimator {
        builder = builder.estimator(estimator);
    }
    let (result, reported) = execute(builder, &data, seed)?;
    Some(Replicate { interval: posterior_interval(&result, 0)?, truth, reported })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn finite_population_gaussian_arms_credible_interval() {
    let n = grid_n(300);
    let (control, treated, truth) = fixed_unit_outcomes(n);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0xB1A0_0000 + rep);
        let assigned = complete_assignment(n, n / 2, seed);
        let outcome: Vec<f64> =
            (0..n).map(|i| if assigned[i] { treated[i] } else { control[i] }).collect();
        arm_replicate(&outcome, assigned, weak_gaussian_arms(), None, truth, seed)
    });
    measure(
        "finite_population_gaussian_arms_credible_interval",
        "complete_randomization_fixed_units_known_finite_population_effect",
        Some("RandomizedEffect"),
        &runs,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn superpopulation_gaussian_arms_credible_interval() {
    let n = grid_n(300);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0xB1A1_0000 + rep);
        let assigned = complete_assignment(n, n / 2, seed);
        let mut noise = gaussian(stream_seed(seed, 0xA11));
        let outcome: Vec<f64> = (0..n)
            .map(|i| {
                let (x1, x2) = (noise(), noise());
                let baseline = 2.0 + 1.2 * x1 - 0.7 * x2 + 0.5 * noise();
                baseline + if assigned[i] { 1.4 + 0.2 * x1 } else { 0.0 }
            })
            .collect();
        arm_replicate(
            &outcome,
            assigned,
            weak_gaussian_arms(),
            Some(EstimatorId::BayesianRandomizedSuperpopulation),
            1.4,
            seed,
        )
    });
    measure(
        "superpopulation_gaussian_arms_credible_interval",
        "complete_randomization_gaussian_population_known_mean_effect",
        Some("RandomizedEffect"),
        &runs,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn superpopulation_bernoulli_arms_credible_interval() {
    let n = grid_n(300);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0xB1A2_0000 + rep);
        let assigned = complete_assignment(n, n / 2, seed);
        let outcome: Vec<f64> = (0..n)
            .map(|i| {
                let success = if assigned[i] { 0.55 } else { 0.35 };
                f64::from(u8::from(unit_uniform(stream_seed(seed, (n + i) as u64)) < success))
            })
            .collect();
        arm_replicate(
            &outcome,
            assigned,
            ArmOutcomeModel::Bernoulli(ArmBetaPrior { alpha: 1.0, beta: 1.0 }),
            Some(EstimatorId::BayesianRandomizedSuperpopulation),
            0.2,
            seed,
        )
    });
    measure(
        "superpopulation_bernoulli_arms_credible_interval",
        "complete_randomization_bernoulli_population_known_risk_difference",
        Some("RandomizedEffect"),
        &runs,
    );
}

// ------------------------------------------------------------------------------- switchback

const SWITCHBACK_PERIODS: usize = 12;
const SWITCHBACK_RHO: f64 = 0.4;
const SWITCHBACK_EFFECT: f64 = 0.7;

fn switchback_replicate(sequences: usize, seed: u64) -> Option<Replicate> {
    let rows = sequences * SWITCHBACK_PERIODS;
    let mut noise = gaussian(stream_seed(seed, 0x5B));
    let (mut outcome, mut assigned, mut ids, mut periods) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for s in 0..sequences {
        let shared = noise();
        for t in 0..SWITCHBACK_PERIODS {
            let treated =
                unit_uniform(stream_seed(seed, (s * SWITCHBACK_PERIODS + t) as u64)) < 0.5;
            let error = SWITCHBACK_RHO.sqrt() * shared + (1.0 - SWITCHBACK_RHO).sqrt() * noise();
            outcome.push(0.5 + SWITCHBACK_EFFECT * f64::from(u8::from(treated)) + error);
            assigned.push(treated);
            ids.push(Arc::<str>::from(format!("s{s}")));
            periods.push(Arc::<str>::from(format!("p{t}")));
        }
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).ok()?;
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Switchback { periods: periods.into() },
        v(0),
        assigned,
        vec![0.5; rows],
        ids,
        labels(rows, "row"),
        ("off", "on"),
    );
    let model = DesignPosteriorModel::Switchback {
        model: SwitchbackModel {
            carryover: CarryoverDeclaration::None,
            dependence: SequenceDependence::CompoundSymmetric { rho: SWITCHBACK_RHO },
            periods: PeriodAdjustment::None,
            prior: SwitchbackPrior {
                effect_precision: 0.5,
                nuisance_precision: 0.01,
                variance_shape: 2.0,
                variance_scale: 1.0,
            },
        },
    };
    let builder = Study::tabular(data.clone()).query(query).inference(InferenceMode::Bayesian(
        BayesianConfig::conjugate().n_draws(DRAWS).design_model(model),
    ));
    let (result, reported) = execute(builder, &data, seed)?;
    Some(Replicate {
        interval: posterior_interval(&result, 0)?,
        truth: SWITCHBACK_EFFECT,
        reported,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn switchback_direct_effect_credible_interval() {
    let sequences = grid_n(30);
    let runs = map_replicates(n_sim(), |rep| {
        switchback_replicate(sequences, grid_seed(0xB1A3_0000 + rep))
    });
    measure(
        "switchback_direct_effect_credible_interval",
        "independent_sequences_compound_symmetric_errors_constant_direct_effect",
        Some("RandomizedEffect"),
        &runs,
    );
}

// ---------------------------------------------------------------------------- complier effect

const COMPLIER_EFFECT: f64 = 2.0;

fn complier_replicate(n: usize, seed: u64) -> Option<Replicate> {
    let mut noise = gaussian(stream_seed(seed, 0xC0));
    let assigned: Vec<bool> =
        (0..n).map(|i| unit_uniform(stream_seed(seed, i as u64)) < 0.5).collect();
    let complier: Vec<bool> =
        (0..n).map(|i| unit_uniform(stream_seed(seed, (n + i) as u64)) < 0.7).collect();
    let received: Vec<bool> = (0..n).map(|i| assigned[i] && complier[i]).collect();
    // Compliers sit at a different baseline than never-takers, so a naive treated-versus-control
    // contrast of the received arms is biased and only the ratio of the two score means is not.
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 0.8 * f64::from(u8::from(complier[i]))
                + COMPLIER_EFFECT * f64::from(u8::from(received[i]))
                + noise()
        })
        .collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).ok()?;
    let query = RandomizedEffectQuery::bernoulli_itt(
        v(0),
        assigned,
        vec![0.5; n],
        labels(n, "account"),
        labels(n, "row"),
        ("control", "encouraged"),
    )
    .with_received_treatment(received);
    let builder = Study::tabular(data.clone())
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)));
    let (result, reported) = execute(builder, &data, seed)?;
    let column = result.posterior.as_ref()?.effect_column()?;
    Some(Replicate {
        interval: posterior_interval(&result, column)?,
        truth: COMPLIER_EFFECT,
        reported,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn complier_effect_credible_interval() {
    let n = grid_n(600);
    let runs = map_replicates(n_sim(), |rep| complier_replicate(n, grid_seed(0xB1A4_0000 + rep)));
    measure(
        "complier_effect_credible_interval",
        "bernoulli_encouragement_one_sided_noncompliance_constant_complier_effect",
        Some("RandomizedEffect"),
        &runs,
    );
}

// ----------------------------------------------------------------------- piecewise survival

const CONTROL_HAZARD: f64 = 0.4;
const TREATED_HAZARD: f64 = 0.25;
const CENSORING_HAZARD: f64 = 0.15;
const HORIZON: f64 = 3.0;

fn survival_truths() -> [f64; 2] {
    let survival = |hazard: f64| (-hazard * HORIZON).exp();
    let restricted_mean = |hazard: f64| (1.0 - (-hazard * HORIZON).exp()) / hazard;
    [
        survival(TREATED_HAZARD) - survival(CONTROL_HAZARD),
        restricted_mean(TREATED_HAZARD) - restricted_mean(CONTROL_HAZARD),
    ]
}

fn survival_replicate(n: usize, coordinate: usize, seed: u64) -> Option<Replicate> {
    let exponential =
        |stream: u64, hazard: f64| -(1.0 - unit_uniform(stream_seed(seed, stream))).ln() / hazard;
    let (mut duration, mut event, mut treatment) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let base = 3 * i as u64;
        let treated = unit_uniform(stream_seed(seed, base)) < 0.5;
        let hazard = if treated { TREATED_HAZARD } else { CONTROL_HAZARD };
        let failure = exponential(base + 1, hazard);
        let censor = exponential(base + 2, CENSORING_HAZARD).min(HORIZON);
        duration.push(failure.min(censor).max(1e-9));
        event.push(f64::from(u8::from(failure <= censor)));
        treatment.push(f64::from(u8::from(treated)));
    }
    let data = TabularData::from_f64_columns([
        ("duration", duration.as_slice()),
        ("event", event.as_slice()),
        ("treatment", treatment.as_slice()),
    ])
    .ok()?;
    let query = CausalQuery::Survival(SurvivalQuery {
        duration: v(0),
        event: v(1),
        treatment: v(2),
        tau: HORIZON,
        delayed_entry: None,
        stratified_censoring: None,
        known_censoring: None,
        observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
        functional: SurvivalFunctional::SurvivalAndRmst,
    });
    let model = DesignPosteriorModel::PiecewiseHazard {
        cuts: Arc::from([1.0, 2.0, 3.0].as_slice()),
        prior_shape: 1.0,
        prior_rate: 1.0,
    };
    let builder = Study::tabular(data.clone()).query(query).inference(InferenceMode::Bayesian(
        BayesianConfig::conjugate().n_draws(DRAWS).design_model(model),
    ));
    let (result, reported) = execute(builder, &data, seed)?;
    let row = result.posterior_row()?;
    let (lower, upper) = (row.lower[coordinate], row.upper[coordinate]);
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Replicate {
        interval: (lower, upper),
        truth: survival_truths()[coordinate],
        reported,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn piecewise_survival_difference_credible_interval() {
    let n = grid_n(400);
    let runs =
        map_replicates(n_sim(), |rep| survival_replicate(n, 0, grid_seed(0xB1A5_0000 + rep)));
    measure(
        "piecewise_survival_difference_credible_interval",
        "constant_hazards_random_and_administrative_censoring_known_survival_difference",
        Some("Survival"),
        &runs,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn piecewise_restricted_mean_difference_credible_interval() {
    let n = grid_n(400);
    let runs =
        map_replicates(n_sim(), |rep| survival_replicate(n, 1, grid_seed(0xB1A6_0000 + rep)));
    measure(
        "piecewise_restricted_mean_difference_credible_interval",
        "constant_hazards_random_and_administrative_censoring_known_restricted_mean_difference",
        Some("Survival"),
        &runs,
    );
}

// ---------------------------------------------------------------------- continuous dose policy

const WINDOW: f64 = 1.0;
/// Intercept, slope and curvature of each baseline group's exactly quadratic dose response.
const GROUP_CURVES: [[f64; 3]; 2] = [[1.0, 0.5, 0.25], [2.0, -0.3, 0.4]];
const POLICY_DOSES: [f64; 2] = [1.5, 2.5];
const REFERENCE_DOSES: [f64; 2] = [2.5, 1.5];

fn curve(group: usize, dose: f64) -> f64 {
    let [a, b, c] = GROUP_CURVES[group];
    a + b * dose + c * dose * dose
}

/// Average of the group's curve over the Epanechnikov window of half-width [`WINDOW`] centred at
/// `target`: the second moment of that kernel is one fifth of the squared half-width.
fn window_average(group: usize, target: f64) -> f64 {
    let [a, b, c] = GROUP_CURVES[group];
    a + b * target + c * (target * target + WINDOW * WINDOW / 5.0)
}

/// The policy value, the reference value and their difference, with equal group shares.
fn dose_truths() -> [f64; 3] {
    let value =
        |doses: [f64; 2]| 0.5 * window_average(0, doses[0]) + 0.5 * window_average(1, doses[1]);
    let (policy, reference) = (value(POLICY_DOSES), value(REFERENCE_DOSES));
    [policy, reference, policy - reference]
}

fn dose_replicate(n: usize, coordinate: usize, seed: u64) -> Option<Replicate> {
    let mut noise = gaussian(stream_seed(seed, 0xD0));
    let (mut dose, mut outcome, mut groups) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let group = i % 2;
        let d = 4.0 * unit_uniform(stream_seed(seed, i as u64));
        dose.push(d);
        outcome.push(curve(group, d) + noise());
        groups.push(Arc::<str>::from(if group == 0 { "A" } else { "B" }));
    }
    let density = vec![1.0; n];
    let data = TabularData::from_f64_columns([
        ("outcome", outcome.as_slice()),
        ("dose", dose.as_slice()),
        ("density", density.as_slice()),
    ])
    .ok()?;
    let rule = |doses: [f64; 2]| -> Arc<[(Arc<str>, f64)]> {
        [(Arc::<str>::from("A"), doses[0]), (Arc::<str>::from("B"), doses[1])].into_iter().collect()
    };
    let query = CausalQuery::ContinuousDoseResponse(ContinuousDoseResponseQuery {
        outcome: v(0),
        dose: v(1),
        dose_density: v(2),
        baseline_groups: groups.into(),
        target_doses: Arc::from([]),
        bandwidth: WINDOW,
        min_local_support: 20,
        density_provenance: Arc::from("known"),
        fixed_policy: Some(FixedGroupDosePolicy {
            policy_doses: rule(POLICY_DOSES),
            reference_doses: rule(REFERENCE_DOSES),
        }),
    });
    let model = DesignPosteriorModel::DosePolynomial {
        support: (0.0, 4.0),
        coefficient_scales: Arc::from([10.0, 10.0, 10.0].as_slice()),
        residual: ResidualPrior::InverseGamma { shape: 1.0, scale: 1.0 },
    };
    let builder = Study::tabular(data.clone()).query(query).inference(InferenceMode::Bayesian(
        BayesianConfig::conjugate().n_draws(DRAWS).design_model(model),
    ));
    let (result, reported) = execute(builder, &data, seed)?;
    let row = result.posterior_row()?;
    let (lower, upper) = (row.lower[coordinate], row.upper[coordinate]);
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Replicate {
        interval: (lower, upper),
        truth: dose_truths()[coordinate],
        reported,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dose_policy_value_credible_interval() {
    let n = grid_n(400);
    let runs = map_replicates(n_sim(), |rep| dose_replicate(n, 0, grid_seed(0xB1A7_0000 + rep)));
    measure(
        "dose_policy_value_credible_interval",
        "randomized_dose_two_groups_exact_quadratic_curves_known_window_value",
        Some("ContinuousDoseResponse"),
        &runs,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn dose_policy_incremental_value_credible_interval() {
    let n = grid_n(400);
    let runs = map_replicates(n_sim(), |rep| dose_replicate(n, 2, grid_seed(0xB1A8_0000 + rep)));
    measure(
        "dose_policy_incremental_value_credible_interval",
        "randomized_dose_two_groups_exact_quadratic_curves_known_window_increment",
        Some("ContinuousDoseResponse"),
        &runs,
    );
}

// ------------------------------------------------------------------ cluster Bayesian bootstrap

const CLUSTER_TOTAL_EFFECT: f64 = 3.0;

fn cluster_replicate(clusters: usize, seed: u64) -> Option<Replicate> {
    let treated = complete_assignment(clusters, clusters / 2, stream_seed(seed, 0xC1));
    let mut noise = gaussian(seed);
    let (mut outcome, mut assignment) = (Vec::new(), Vec::new());
    for &arm in &treated {
        let shared = noise();
        for _ in 0..2 {
            outcome.push(shared + 0.5 * noise() + CLUSTER_TOTAL_EFFECT * f64::from(u8::from(arm)));
            assignment.push(arm);
        }
    }
    let units = 2 * clusters;
    let edges: Vec<NetworkEdge> = (0..clusters as u64)
        .flat_map(|c| {
            [NetworkEdge::new(2 * c, 2 * c + 1, 1.0), NetworkEdge::new(2 * c + 1, 2 * c, 1.0)]
        })
        .collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).ok()?;
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from((0..units).map(|u| (u / 2) as u32).collect::<Vec<_>>()),
            treated_clusters: clusters / 2,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let network = NetworkData::try_new(data.clone(), edges).ok()?;
    let builder = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .refute(RefuteSuite::None);
    let (result, reported) = execute(builder, &data, seed)?;
    let column = result.posterior.as_ref()?.effect_column()?;
    Some(Replicate {
        interval: posterior_interval(&result, column)?,
        truth: CLUSTER_TOTAL_EFFECT,
        reported,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn cluster_bayesian_bootstrap_small_cluster_credible_interval() {
    // Twenty-four clusters at the base point, so the grid runs from nine clusters per arm upward.
    let clusters = grid_n(24);
    let runs =
        map_replicates(n_sim(), |rep| cluster_replicate(clusters, grid_seed(0xB1A9_0000 + rep)));
    measure(
        "cluster_bayesian_bootstrap_small_cluster_credible_interval",
        "paired_cluster_randomization_constant_total_effect_few_clusters",
        None,
        &runs,
    );
}

// --------------------------------------------------------------------- model-based arm effect

/// Columns `t, y, z`: a probit-confounded binary treatment and a linear outcome model with a
/// treatment-by-confounder interaction, `y = z + t (1 + 2 z) + 0.5 e`. The known targets are the
/// mean of `1 + 2 z` over the treated and over the untreated rows.
fn gcomp_data(n: usize, seed: u64) -> (TabularData, f64, f64) {
    let mut noise = gaussian(seed);
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let confounder = noise();
        let treated = f64::from(u8::from(confounder + noise() > 0.0));
        y.push(confounder + treated * (1.0 + 2.0 * confounder) + 0.5 * noise());
        t.push(treated);
        z.push(confounder);
    }
    let arm_effect = |arm: f64| {
        let rows: Vec<usize> = (0..n).filter(|&i| (t[i] - arm).abs() < 0.5).collect();
        rows.iter().map(|&i| 1.0 + 2.0 * z[i]).sum::<f64>() / rows.len() as f64
    };
    let (treated, untreated) = (arm_effect(1.0), arm_effect(0.0));
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned");
    (data, treated, untreated)
}

fn gcomp_graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph
            .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .expect("an acyclic confounding graph");
    }
    graph
}

fn gcomp_replicate(n: usize, population: TargetPopulation, seed: u64) -> Option<Replicate> {
    let (data, treated, untreated) = gcomp_data(n, seed);
    let truth = if matches!(population, TargetPopulation::Treated) { treated } else { untreated };
    let query = AverageEffectQuery::binary_ate(v(0), v(1)).with_target_population(population);
    let builder = Study::tabular(data.clone())
        .graph(gcomp_graph())
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(100.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let (result, reported) = execute(builder, &data, seed)?;
    let column = result.posterior.as_ref()?.effect_column()?;
    Some(Replicate { interval: posterior_interval(&result, column)?, truth, reported })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn gcomp_effect_on_the_treated_credible_interval() {
    let n = grid_n(400);
    let runs = map_replicates(n_sim(), |rep| {
        gcomp_replicate(n, TargetPopulation::Treated, grid_seed(0xB1AA_0000 + rep))
    });
    measure(
        "gcomp_effect_on_the_treated_credible_interval",
        "probit_confounded_linear_outcome_known_effect_on_treated_rows",
        None,
        &runs,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn gcomp_effect_on_the_untreated_credible_interval() {
    let n = grid_n(400);
    let runs = map_replicates(n_sim(), |rep| {
        gcomp_replicate(n, TargetPopulation::Untreated, grid_seed(0xB1AB_0000 + rep))
    });
    measure(
        "gcomp_effect_on_the_untreated_credible_interval",
        "probit_confounded_linear_outcome_known_effect_on_untreated_rows",
        None,
        &runs,
    );
}

// ---------------------------------------------------------------------------- quick checks

/// The known truths are the ones the designs state.
#[test]
fn the_stated_truths_are_the_ones_the_generators_imply() {
    let [survival, restricted] = survival_truths();
    assert!((survival - (-0.75_f64).exp() + (-1.2_f64).exp()).abs() < 1e-12);
    assert!(
        (restricted - (4.0 * (1.0 - (-0.75_f64).exp()) - 2.5 * (1.0 - (-1.2_f64).exp()))).abs()
            < 1e-12
    );
    let [policy, reference, increment] = dose_truths();
    assert!((policy - reference - increment).abs() < 1e-12);
    // Group A at 1.5: 1 + 0.75 + 0.25 * (2.25 + 0.2); group B at 2.5: 2 - 0.75 + 0.4 * (6.25 + 0.2).
    let expected = 0.5 * (1.0 + 0.75 + 0.25 * 2.45) + 0.5 * (2.0 - 0.75 + 0.4 * 6.45);
    assert!((policy - expected).abs() < 1e-12, "{policy} vs {expected}");
    let (control, treated, truth) = fixed_unit_outcomes(300);
    let mean_difference = treated.iter().zip(&control).map(|(t, c)| t - c).sum::<f64>() / 300.0;
    assert!((mean_difference - truth).abs() < 1e-12);
}
