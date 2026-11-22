//! Repeated-sampling coverage of the intervals published by the retained sequential-contrast,
//! sequential-regime, policy-value and within-period history routes.
//!
//! Every design draws a fresh sample of a known data generating process in each replicate, runs the
//! public study flow, and scores the interval the route publishes at 0.95 against the analytic
//! target of that process:
//!
//! * the cross-fit Wald interval and the modular posterior of the sequential contrast of the two
//!   static regimes (treat at both decisions against treat at neither), on a panel whose
//!   propensities are logistic in bounded histories and whose outcome is linear in the measured
//!   history, so the contrast is `1.32 + 1.5`;
//! * the Bayesian-bootstrap posterior of a prespecified sequential regime value under known
//!   sequential randomization, with frozen Q predictions that are functions of the subject's own
//!   history only;
//! * the Bayesian-bootstrap posterior of a fixed binary policy's incremental value under known
//!   randomized assignment (independent score draws, so the target is the population value);
//! * the Newey-West fixed-b period interval of the within-period history match, scored against
//!   the population mean effect on the treated (the interval's sampling units are the periods, so
//!   its target is the expected matched contrast, not the effect on the units one sample
//!   happened to match), and the posterior of the within-period history outcome model, scored
//!   against the model's own effect on the treated units.
//!
//! These are design routes: they declare no graph and run no validation suite, and the runtime
//! keys them with no query. Each record is therefore keyed on the design axes: the query is
//! named here, the structure is `design` and no validation mode applies. The routes publish no
//! runtime interval binding for the Frequentist estimates (their primary estimate is not an
//! effect), so a Frequentist record also states the interval method, the standard-error kind
//! and the dependence of the interval the route's own fit publishes. Every other field of a
//! construction is the runtime's.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, HistoryMatchQuery, LongitudinalRegimeMethod,
    LongitudinalRegimeQuery, PolicyValueQuery, SequentialContrastQuery, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, stream_seed, uniform,
};
use common::calibration_bind::constructions;

/// Interval method of the interval a route forms from its own standard error.
const NORMAL_INTERVAL: &str = "analytic_se";

/// Interval method of the equal-tailed posterior interval.
const POSTERIOR_INTERVAL: &str = "posterior_quantile";

/// Posterior draws of every Bayesian design.
const DRAWS: usize = 2_000;

/// Standard normal density at zero, `E[x 1{x > 0}]` for a standard normal `x`.
const NORMAL_DENSITY_AT_ZERO: f64 = 0.398_942_280_401_432_7;

type Reported = Vec<(Construction, ScopeFacts, &'static str)>;

/// One replicate that published an interval: the interval, the runtime's constructions of the
/// execution and the target the interval is scored against.
struct Scored {
    interval: (f64, f64),
    reported: Reported,
    truth: f64,
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS))
}

/// The equal-tailed posterior interval of the effect column, as the posterior summarizes it.
fn posterior_interval(result: &StudyResult) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let column = posterior.effect_column()?;
    Some((posterior.summaries.q025[column], posterior.summaries.q975[column]))
}

/// The runtime's construction of a Frequentist execution with the interval method, the
/// standard-error kind and the dependence of the interval the route's fit publishes.
fn normal_interval_construction(
    reported: Reported,
    se_kind: &'static str,
    dependence: &'static str,
) -> Reported {
    reported
        .into_iter()
        .take(1)
        .map(|(mut construction, scope, validation)| {
            construction.interval_method = NORMAL_INTERVAL.into();
            construction.se_kind = se_kind.into();
            construction.dependence = dependence.into();
            construction.reported_level = REPORTED_LEVEL;
            (construction, scope, validation)
        })
        .collect()
}

/// What one record is: the test, its data generating process, the query the design route
/// answers, and the interval method scored.
#[derive(Clone, Copy)]
struct Record {
    test: &'static str,
    dgp: &'static str,
    query: &'static str,
    interval: &'static str,
}

/// Score `run` over the replicates against the record's interval method; a replicate that
/// publishes no interval is a skip, never a free pass.
fn measure(record: Record, seed: u64, run: impl Fn(u64) -> Option<Scored> + Sync) {
    let Record { test, dgp, query, interval } = record;
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    // Each grid point draws its own replicate data.
    let runs = map_replicates(n_sim(), |rep| run(grid_seed(stream_seed(seed, rep))));
    for scored in runs {
        let Some(scored) = scored else {
            tally.skip();
            continue;
        };
        let (construction, scope, _) = scored
            .reported
            .iter()
            .find(|(construction, _, _)| construction.interval_method == interval)
            .unwrap_or_else(|| panic!("the execution reported no {interval} interval"));
        // A design route has no graph and runs no validation suite.
        let mut construction = construction.clone();
        construction.query = query.into();
        construction.structure = "design".into();
        tally.bind_validation(&construction, *scope, "not_applicable");
        tally.record(Some(scored.interval), scored.truth);
    }
    tally.assert();
}

// ------------------------------------------------------------ sequential contrast

const SEQUENTIAL_BASE_N: usize = 400;

/// Contrast of the regime that treats at both decisions against the regime that treats at neither:
/// `E[Y(1, 1)] - E[Y(0, 0)] = (1 + 1.32 + 1.5) - 1`.
const SEQUENTIAL_CONTRAST_TRUTH: f64 = 1.32 + 1.5;

/// Evaluation folds of the cross-fit: each nuisance model is fitted on four fifths of the rows,
/// so the product of the propensity and conditional-mean errors the route's
/// `nuisance_product_rate` premise needs small stays small at the smallest sample measured.
const SEQUENTIAL_FOLDS: usize = 5;

/// A draw on `[-sqrt 3, sqrt 3]` (mean zero, unit variance) from a uniform draw on `[0, 1)`.
fn bounded(unit: f64) -> f64 {
    3.0_f64.sqrt() * (2.0 * unit - 1.0)
}

/// Columns `l0, a0, l1, a1, y`: `l0` uniform with unit variance,
/// `a0 ~ Bernoulli(expit(.5 l0))`, `l1 = .5 l0 + .8 a0 + .5 u` with `u` uniform with unit
/// variance, `a1 ~ Bernoulli(expit(.5 (l1 - .4)))` and `y = 1 + .5 l0 + a0 + 1.5 a1 + .4 l1 + e`.
/// Under sequential exchangeability given the measured history,
/// `E[Y(a0, a1)] = 1 + .5 E[l0] + a0 + 1.5 a1 + .4 (.5 E[l0] + .8 a0) = 1 + 1.32 a0 + 1.5 a1`.
///
/// The histories are bounded so that every action probability lies in `[.25, .75]`: the route
/// refuses a sample in which a fitted action probability leaves its declared floor, and an
/// unbounded history puts an extreme row's fitted probability outside `[.05, .95]` in a few
/// samples in a hundred.
fn sequential_panel(n: usize, seed: u64) -> TabularData {
    let mut normal = gaussian(stream_seed(seed, 0x5E01));
    let mut unit = uniform(seed, 0x5E02);
    let mut history = uniform(seed, 0x5E03);
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let l0 = bounded(history());
        let a0 = f64::from(unit() < expit(0.5 * l0));
        let l1 = 0.5 * l0 + 0.8 * a0 + 0.5 * bounded(history());
        let a1 = f64::from(unit() < expit(0.5 * (l1 - 0.4)));
        let y = 1.0 + 0.5 * l0 + a0 + 1.5 * a1 + 0.4 * l1 + normal();
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
        ys.push(y);
    }
    TabularData::from_f64_columns([
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned")
}

fn sequential_query(n: usize) -> SequentialContrastQuery {
    SequentialContrastQuery {
        outcome: VariableId::from_raw(4),
        first_treatment: VariableId::from_raw(1),
        second_treatment: VariableId::from_raw(3),
        baseline: Arc::from([VariableId::from_raw(0)]),
        intermediate: Arc::from([VariableId::from_raw(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: (0..n)
            .map(|row| u32::try_from(row % SEQUENTIAL_FOLDS).expect("a fold index fits"))
            .collect(),
        fold_count: u32::try_from(SEQUENTIAL_FOLDS).expect("the fold count fits"),
        min_probability: 0.05,
    }
}

/// One sequential-contrast replicate; the Frequentist interval is the published Wald interval.
fn run_sequential_contrast(seed: u64, bayesian_inference: bool) -> Option<Scored> {
    let n = grid_n(SEQUENTIAL_BASE_N);
    let data = sequential_panel(n, seed);
    let builder =
        Study::tabular(data.clone()).query(CausalQuery::SequentialContrast(sequential_query(n)));
    let builder = if bayesian_inference { builder.inference(bayesian()) } else { builder };
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = builder.build().ok()?.prepare(&ctx).ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    let reported = constructions(&contract, &result);
    let (interval, reported) = if bayesian_inference {
        (posterior_interval(&result)?, reported)
    } else {
        let [lower, upper] = result.sequential_contrast()?.interval?;
        ((lower, upper), normal_interval_construction(reported, "crossfit_score", "iid"))
    };
    Some(Scored { interval, reported, truth: SEQUENTIAL_CONTRAST_TRUTH })
}

/// The cross-fit Wald interval of the sequential contrast.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn sequential_crossfit_contrast_wald_interval_coverage() {
    measure(
        Record {
            test: "sequential_crossfit_contrast_wald_interval_coverage",
            dgp: "sequential_panel",
            query: "SequentialContrast",
            interval: NORMAL_INTERVAL,
        },
        0x5E0_0001,
        |seed| run_sequential_contrast(seed, false),
    );
}

/// The modular Bayesian-bootstrap posterior of the sequential contrast.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_sequential_crossfit_contrast_posterior_coverage() {
    measure(
        Record {
            test: "bayesian_sequential_crossfit_contrast_posterior_coverage",
            dgp: "sequential_panel",
            query: "SequentialContrast",
            interval: POSTERIOR_INTERVAL,
        },
        0x5E0_0002,
        |seed| run_sequential_contrast(seed, true),
    );
}

// ---------------------------------------------------------------- sequential regime

const REGIME_BASE_N: usize = 400;

/// `E[Y(1, 1)]` of the regime that treats at both decisions: `1 + 1 + 1.5 + .4 (.8)`.
const REGIME_TRUTH: f64 = 3.5 + 0.32;

/// Fixed shrinkage of the true Q functions. The frozen predictions are wrong by a constant factor
/// and depend only on the subject's own history, never on any fitted model, so they are
/// subject-excluded and the augmented score stays unbiased under the known randomization.
const Q_SHRINK: f64 = 0.75;

/// A two-decision trial with known sequential randomization (each action is a fair coin): the
/// outcome column and the retained regime query. `l1 = .5 l0 + .8 a0 + .5 e` and
/// `y = 1 + .5 l0 + a0 + 1.5 a1 + .4 l1 + e`.
fn regime_trial(n: usize, seed: u64) -> (TabularData, LongitudinalRegimeQuery) {
    let mut normal = gaussian(stream_seed(seed, 0x4E01));
    let mut unit = uniform(seed, 0x4E02);
    let mut outcome = Vec::with_capacity(n);
    let mut treatment = Vec::with_capacity(2 * n);
    let mut q = Vec::with_capacity(2 * n);
    for _ in 0..n {
        let l0 = normal();
        let a0 = unit() < 0.5;
        let l1 = 0.5 * l0 + 0.8 * f64::from(a0) + 0.5 * normal();
        let a1 = unit() < 0.5;
        outcome.push(1.0 + 0.5 * l0 + f64::from(a0) + 1.5 * f64::from(a1) + 0.4 * l1 + normal());
        treatment.extend([a0, a1]);
        // Q at each decision under the regime's action: E[Y | history, a1 = 1] and
        // E[Q1 | l0, a0 = 1], shrunk.
        q.push(Q_SHRINK * (3.82 + 0.7 * l0));
        q.push(Q_SHRINK * (2.5 + 0.5 * l0 + f64::from(a0) + 0.4 * l1));
    }
    let folds: Vec<u32> = (0..n).map(|i| u32::from(i % 2 == 1)).collect();
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())])
        .expect("the simulated column is finite");
    let query = LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(0),
        method: LongitudinalRegimeMethod::SequentialDoublyRobust,
        period_outcome_predictions: None,
        known_fixed_outcome_predictions: false,
        stabilizing_numerator_probabilities: None,
        q_predictions: Some(Arc::from(q)),
        observation_history: Some(Arc::from(vec![true; 2 * n])),
        prediction_fold_ids: Some(Arc::from(folds.clone())),
        periods: 2,
        treatment_history: Arc::from(treatment),
        regime_actions: Arc::from(vec![true; 2 * n]),
        treatment_probabilities: Arc::from(vec![0.5; 2 * n]),
        censoring_probabilities: Arc::from(vec![1.0; 2 * n]),
        outcome_observed: Arc::from(vec![true; n]),
        subject_ids: (0..n).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect(),
        fold_ids: Arc::from(folds),
        excluded_fold_predictions: true,
        probabilities_known_by_design: true,
        minimum_probability: 0.1,
        rule_id: None,
        rule_version: None,
        rule_provenance: None,
    };
    (data, query)
}

fn run_regime(seed: u64) -> Option<Scored> {
    let (data, query) = regime_trial(grid_n(REGIME_BASE_N), seed);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::LongitudinalRegime(query))
        .inference(bayesian())
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    Some(Scored {
        interval: posterior_interval(&result)?,
        reported: constructions(&contract, &result),
        truth: REGIME_TRUTH,
    })
}

/// The Bayesian-bootstrap posterior of the sequential doubly robust regime value.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_sequential_regime_value_posterior_coverage() {
    measure(
        Record {
            test: "bayesian_sequential_regime_value_posterior_coverage",
            dgp: "regime_trial",
            query: "LongitudinalRegime",
            interval: POSTERIOR_INTERVAL,
        },
        0x4E0_0001,
        run_regime,
    );
}

// ---------------------------------------------------------------------- policy value

const POLICY_BASE_N: usize = 400;

/// Incremental value of treating when `x > 0` against treating no one:
/// `E[1{x > 0} (1 + .5 x)] = .5 + .5 E[x 1{x > 0}]`.
const POLICY_INCREMENTAL_TRUTH: f64 = 0.5 + 0.5 * NORMAL_DENSITY_AT_ZERO;

/// A randomized trial with a fair-coin assignment: `x ~ N(0, 1)` and
/// `y = 1 + x + a (1 + .5 x) + e`. The fixed policy treats when `x > 0`; the reference treats no
/// one. The units are drawn independently, so the scores are independent and the population value
/// is the target.
fn policy_trial(n: usize, seed: u64) -> (TabularData, PolicyValueQuery) {
    let mut normal = gaussian(stream_seed(seed, 0x7A01));
    let mut unit = uniform(seed, 0x7A02);
    let (mut outcome, mut assignment, mut actions) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let x = normal();
        let assigned = unit() < 0.5;
        outcome.push(1.0 + x + f64::from(assigned) * (1.0 + 0.5 * x) + normal());
        assignment.push(assigned);
        actions.push(x > 0.0);
    }
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())])
        .expect("the simulated column is finite");
    let query = PolicyValueQuery {
        outcome: VariableId::from_raw(0),
        assignment: Arc::from(assignment),
        propensity: Arc::from([0.5]),
        actions: Arc::from(actions),
        reference: Arc::from(vec![false; n]),
        mu0: Arc::from([]),
        mu1: Arc::from([]),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: (0..n).map(|i| Arc::<str>::from(format!("subject-{i}"))).collect(),
        disjoint_training_subjects: false,
        crossfit_fold_ownership_valid: false,
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
        regret: None,
    };
    (data, query)
}

fn run_policy(seed: u64) -> Option<Scored> {
    let (data, query) = policy_trial(grid_n(POLICY_BASE_N), seed);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(query))
        .inference(bayesian())
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    Some(Scored {
        interval: posterior_interval(&result)?,
        reported: constructions(&contract, &result),
        truth: POLICY_INCREMENTAL_TRUTH,
    })
}

/// The Bayesian-bootstrap posterior of the incremental value of a fixed binary policy.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_binary_policy_value_posterior_coverage() {
    measure(
        Record {
            test: "bayesian_binary_policy_value_posterior_coverage",
            dgp: "policy_trial",
            query: "PolicyValue",
            interval: POSTERIOR_INTERVAL,
        },
        0x7A0_0001,
        run_policy,
    );
}

// ------------------------------------------------------------------ history routes

/// `E[s | treated]` for a score `s ~ N(0, 1)` treated with probability `expit(.8 s)`:
/// `2 E[s expit(.8 s)]` (the treated share is one half by symmetry), by the trapezoid rule on
/// `[-10, 10]`.
fn treated_mean_score() -> f64 {
    const STEPS: u32 = 200_000;
    let width = 20.0 / f64::from(STEPS);
    let integrand = |s: f64| s * expit(0.8 * s) * (-0.5 * s * s).exp();
    let inner: f64 = (1..STEPS).map(|k| integrand(-10.0 + width * f64::from(k))).sum();
    let ends = 0.5 * (integrand(-10.0) + integrand(10.0));
    2.0 * width * (inner + ends) / (2.0 * std::f64::consts::PI).sqrt()
}

#[test]
fn the_treated_mean_score_is_the_tilted_normal_mean() {
    // An independent quadrature of the same integral gives 0.350452056355.
    assert!((treated_mean_score() - 0.350_452_056_355).abs() < 1e-9);
}

/// A panel of calendar periods of independent units.
struct HistoryPanel {
    data: TabularData,
    score: Vec<f64>,
    treated: Vec<bool>,
    period: Vec<usize>,
}

/// Columns `h0, h1, y, treated, period`: a unit's score `s ~ N(0, 1)` is the mean of its two
/// history values `s +- .3 e`, treatment is `Bernoulli(expit(.8 s))`, and
/// `y = .5 period + .5 s + treated (1 + 2 s) + e`. The effect of a treated unit is `1 + 2 s`.
fn history_panel(periods: usize, per_period: usize, seed: u64) -> HistoryPanel {
    let mut normal = gaussian(stream_seed(seed, 0x4801));
    let mut unit = uniform(seed, 0x4802);
    let (mut h0, mut h1, mut y, mut t, mut p) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut score, mut treated, mut period) = (vec![], vec![], vec![]);
    for index in 0..periods {
        for _ in 0..per_period {
            let s = normal();
            let spread = 0.3 * normal();
            let arm = unit() < expit(0.8 * s);
            let outcome =
                0.5 * index as f64 + 0.5 * s + if arm { 1.0 + 2.0 * s } else { 0.0 } + normal();
            h0.push(s + spread);
            h1.push(s - spread);
            y.push(outcome);
            t.push(f64::from(arm));
            p.push(index as f64);
            score.push(f64::midpoint(s + spread, s - spread));
            treated.push(arm);
            period.push(index);
        }
    }
    let data = TabularData::from_f64_columns([
        ("h0", h0.as_slice()),
        ("h1", h1.as_slice()),
        ("y", y.as_slice()),
        ("treated", t.as_slice()),
        ("period", p.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned");
    HistoryPanel { data, score, treated, period }
}

fn history_query() -> HistoryMatchQuery {
    HistoryMatchQuery {
        outcome: VariableId::from_raw(2),
        treated: VariableId::from_raw(3),
        history: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        period: Some(VariableId::from_raw(4)),
        weight: None,
        caliper: None,
    }
}

const MATCH_PERIODS: usize = 8;
const MATCH_PERIOD_BASE_UNITS: usize = 100;

/// One matched-contrast replicate: the period interval against `truth`, the population mean
/// effect `1 + 2 E[s | treated]` on the treated. Every period holds controls, so every treated
/// unit is matched and the matched contrast estimates that mean; the interval treats the periods
/// as the sampling units, so the variation of the treated units' own effects between periods is
/// part of what it covers.
fn run_history_match(seed: u64, truth: f64) -> Option<Scored> {
    let panel = history_panel(MATCH_PERIODS, grid_n(MATCH_PERIOD_BASE_UNITS), seed);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(panel.data.clone())
        .query(CausalQuery::HistoryMatch(history_query()))
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&panel.data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    let fit = result.history_match()?;
    let interval = (fit.interval_lower?, fit.interval_upper?);
    Some(Scored {
        interval,
        reported: normal_interval_construction(
            constructions(&contract, &result),
            "newey_west_fixed_b",
            "period",
        ),
        truth,
    })
}

/// The Newey-West fixed-b period interval of the within-period history match.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn period_history_contrast_period_interval_coverage() {
    let truth = 1.0 + 2.0 * treated_mean_score();
    measure(
        Record {
            test: "period_history_contrast_period_interval_coverage",
            dgp: "history_panel",
            query: "HistoryMatch",
            interval: NORMAL_INTERVAL,
        },
        0x480_0001,
        |seed| run_history_match(seed, truth),
    );
}

const MODEL_PERIODS: usize = 3;
const MODEL_PERIOD_BASE_UNITS: usize = 60;

/// The model's effect on the treated units of the periods that hold both arms: the unit effect
/// `1 + 2 s` averaged over those treated units, which is the treatment coefficient of the model
/// whose interaction is centred at their mean score.
fn effect_on_treated(panel: &HistoryPanel) -> f64 {
    let holds_both = |period: usize| {
        let arms = panel.period.iter().zip(&panel.treated).filter(|(p, _)| **p == period);
        let (mut treated, mut control) = (false, false);
        for (_, &arm) in arms {
            treated |= arm;
            control |= !arm;
        }
        treated && control
    };
    let used: Vec<f64> = (0..panel.score.len())
        .filter(|&row| panel.treated[row] && holds_both(panel.period[row]))
        .map(|row| 1.0 + 2.0 * panel.score[row])
        .collect();
    used.iter().sum::<f64>() / used.len() as f64
}

fn run_history_outcome_model(seed: u64) -> Option<Scored> {
    let panel = history_panel(MODEL_PERIODS, grid_n(MODEL_PERIOD_BASE_UNITS), seed);
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = Study::tabular(panel.data.clone())
        .query(CausalQuery::HistoryMatch(history_query()))
        .inference(bayesian())
        .estimator(EstimatorId::BayesianPeriodHistoryOutcomeModel)
        .build()
        .ok()?
        .prepare(&ctx)
        .ok()?;
    let result = prepared.estimate(&panel.data, &ctx).ok()?;
    let contract = prepared.contract_for_result(&result).expect("execution contract");
    Some(Scored {
        interval: posterior_interval(&result)?,
        reported: constructions(&contract, &result),
        truth: effect_on_treated(&panel),
    })
}

/// The posterior of the within-period history outcome model's effect on the treated.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_period_history_outcome_model_posterior_coverage() {
    measure(
        Record {
            test: "bayesian_period_history_outcome_model_posterior_coverage",
            dgp: "history_panel",
            query: "HistoryMatch",
            interval: POSTERIOR_INTERVAL,
        },
        0x480_0002,
        run_history_outcome_model,
    );
}
