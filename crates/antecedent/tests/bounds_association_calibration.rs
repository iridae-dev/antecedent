//! Coverage designs for the intervals of the binary-instrument bounds route and of the
//! descriptive association routes, each scored against a known truth.
//!
//! Bounds. The population is a mixture of always-takers, never-takers and compliers with fixed
//! potential-outcome probabilities, so the observed law of the instrument, treatment and outcome
//! is known exactly, and so are the sharp (Balke-Pearl) endpoints of the identified set of
//! average effects, written here in closed form and independent of the route's solver. The
//! interval a replicate publishes (the Imbens-Manski row-bootstrap interval, or the endpoint
//! posterior quantiles of the Bayesian bootstrap) is scored against each true extremal bound in
//! turn: an interval built to cover every value of the set must cover the values at its edges,
//! and the edge is where the width correction matters. Over-coverage is the construction's
//! claim, so each record is a one-sided target (`assert_one_sided`).
//!
//! Association. The target is the population slope of a linear regression of the outcome on one
//! predictor, scored against the interval each route publishes for the slope term:
//!
//! * a table of independent rows with heteroskedastic errors: the leverage-adjusted sandwich
//!   standard error (`sandwich_leverage_student`: the influence root sum of squares times the
//!   Student-t over normal quantile ratio at the Satterthwaite-effective row count) as
//!   `estimate +- z * se`, and the studentized spread of row-bootstrap replicates;
//! * one regular series with autoregressive predictor and errors: the Newey-West standard error
//!   as `estimate +- z * se` (the estimator scales a series error by the ratio of its own
//!   reference quantile to the normal one, so the normal interval of the published error is the
//!   interval the route states), and the fixed-b spread of circular-block bootstrap replicates;
//! * the posterior interval of the conjugate Normal-Inverse-Gamma marginal (homoskedastic
//!   Gaussian rows) and of the block Bayesian bootstrap (the series).
//!
//! The runtime reports no interval method for either query (the result is no effect estimate),
//! so the fields of a record's construction that the runtime does report (query, graph class,
//! modality, inference, estimator, posterior, identification) are read from its own
//! calibration key; the interval method, standard-error kind, dependence and functional are set
//! here from what the result states about the quantity scored. Both routes are graphless, so a
//! record is keyed on the design axes (structure `design`, validation `not_applicable`), as the
//! randomized designs are, and its scope carries the replicates or draws the result records.
//! A replicate counts only when its result states the graphless support row (families
//! `iv_bounds` and `association`), so a record measures the fits the row admits and no others.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "row indices and counts convert exactly at these sizes"
)]

mod common;

use antecedent::analysis::StudyBuilder;
use antecedent::{
    AssociationFunctional, AssociationQuery, BayesianConfig, CausalQuery, InferenceMode,
    IvBoundsQuery, PreparedStudy, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{DeclaredSamplingDesign, ExecutionContext, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, Z95,
    ar1_noise, gaussian, grid_n, grid_seed, map_replicates, n_sim, normal_interval, stream_seed,
    uniform,
};
use common::calibration_bind::{bind_reported, constructions};

/// One replicate's construction, scope facts and validation mode as the bind helpers take them.
type Reported = (Construction, ScopeFacts, &'static str);

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// What a design states about the interval it scores.
struct Scored<'a> {
    interval_method: &'a str,
    level: f64,
    se_kind: &'a str,
    dependence: &'a str,
    functional: &'a str,
    /// Bootstrap replicates that fitted, when the interval is a resampling one.
    replicates: Option<u32>,
    /// Posterior draws behind the interval, when it is drawn.
    draws: Option<u32>,
    /// The posterior a drawn interval comes from, when the runtime key names another.
    posterior: Option<&'a str>,
}

/// The construction the runtime keys the result under, on the design axes, with the interval
/// fields set to the interval scored.
fn reported_as(prepared: &PreparedStudy, result: &StudyResult, scored: &Scored<'_>) -> Reported {
    let contract = prepared.contract_for_result(result).expect("execution contract");
    let (base, scope, _) = constructions(&contract, result)
        .into_iter()
        .next()
        .expect("the runtime reports a calibration basis");
    let posterior = scored.posterior.map_or(base.posterior.clone(), Into::into);
    let construction = Construction {
        structure: "design".into(),
        posterior,
        interval_method: scored.interval_method.into(),
        se_kind: scored.se_kind.into(),
        dependence: scored.dependence.into(),
        functional: scored.functional.into(),
        reported_level: scored.level,
        unit: None,
        ..base
    };
    let scope =
        ScopeFacts { replicates_ok: scored.replicates, posterior_draws: scored.draws, ..scope };
    (construction, scope, "not_applicable")
}

// ------------------------------------------------------------------ bounds

const IV_ROWS: usize = 1000;
const IV_REPLICATES: u32 = 399;
const IV_DRAWS: usize = 400;
const IV_FUNCTIONAL: &str = "ate_identified_set";
const IV_FREQUENTIST_INTERVAL: &str = "imbens_manski_row_bootstrap";
const IV_BAYESIAN_INTERVAL: &str = "posterior_endpoint_quantiles";

/// Population shares of the always-takers, never-takers and compliers.
const TYPE_SHARE: [f64; 3] = [0.2, 0.3, 0.5];
/// `P(Y(1) = 1)` within each type.
const OUTCOME_UNDER_TREATMENT: [f64; 3] = [0.7, 0.4, 0.8];
/// `P(Y(0) = 1)` within each type.
const OUTCOME_UNDER_CONTROL: [f64; 3] = [0.5, 0.3, 0.4];

/// Treatment taken by a type (0 always, 1 never, 2 complier) under instrument value `z`.
fn treatment_of(kind: usize, z: usize) -> usize {
    match kind {
        0 => 1,
        1 => 0,
        _ => z,
    }
}

/// The exact observed law `P(Y=y, D=d | Z=z)` of the mixture; cells within an arm are ordered
/// `(0,0), (1,0), (0,1), (1,1)` as `(y, d)`.
fn response_type_mixture_population_law() -> [[f64; 4]; 2] {
    let mut cells = [[0.0; 4]; 2];
    for z in 0..2 {
        for kind in 0..3 {
            let d = treatment_of(kind, z);
            let p1 =
                if d == 1 { OUTCOME_UNDER_TREATMENT[kind] } else { OUTCOME_UNDER_CONTROL[kind] };
            cells[z][1 + 2 * d] += TYPE_SHARE[kind] * p1;
            cells[z][2 * d] += TYPE_SHARE[kind] * (1.0 - p1);
        }
    }
    cells
}

/// Lower Balke-Pearl bound of `E[Y(1) - Y(0)]` from the observed law, in closed form.
fn balke_pearl_lower(cells: &[[f64; 4]; 2]) -> f64 {
    let p = |y: usize, d: usize, z: usize| cells[z][y + 2 * d];
    [
        p(1, 1, 1) + p(0, 0, 0) - 1.0,
        p(1, 1, 0) + p(0, 0, 1) - 1.0,
        p(1, 1, 0) - p(1, 1, 1) - p(1, 0, 1) - p(0, 1, 0) - p(1, 0, 0),
        p(1, 1, 1) - p(1, 1, 0) - p(1, 0, 0) - p(0, 1, 1) - p(1, 0, 1),
        -p(0, 1, 1) - p(1, 0, 1),
        -p(0, 1, 0) - p(1, 0, 0),
        p(0, 0, 1) - p(0, 1, 1) - p(1, 0, 1) - p(0, 1, 0) - p(0, 0, 0),
        p(0, 0, 0) - p(0, 1, 0) - p(1, 0, 0) - p(0, 1, 1) - p(0, 0, 1),
    ]
    .into_iter()
    .fold(f64::NEG_INFINITY, f64::max)
}

/// The true sharp lower and upper bounds of the average effect on the population law. The
/// upper bound is the negated lower bound of the law with the outcome relabelled.
fn true_effect_bounds() -> (f64, f64) {
    let law = response_type_mixture_population_law();
    let mut flipped = [[0.0; 4]; 2];
    for z in 0..2 {
        for d in 0..2 {
            for y in 0..2 {
                flipped[z][(1 - y) + 2 * d] = law[z][y + 2 * d];
            }
        }
    }
    (balke_pearl_lower(&law), -balke_pearl_lower(&flipped))
}

/// Instrument, treatment and outcome columns of `n` units drawn from the mixture.
fn response_type_mixture_sample(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut next = uniform(seed, 0x1B01);
    let (mut z, mut d, mut y) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let instrument = usize::from(next() >= 0.5);
        let draw = next();
        let kind = if draw < TYPE_SHARE[0] {
            0
        } else if draw < TYPE_SHARE[0] + TYPE_SHARE[1] {
            1
        } else {
            2
        };
        let treated = treatment_of(kind, instrument);
        let p1 =
            if treated == 1 { OUTCOME_UNDER_TREATMENT[kind] } else { OUTCOME_UNDER_CONTROL[kind] };
        z.push(instrument as f64);
        d.push(treated as f64);
        y.push(f64::from(next() < p1));
    }
    (z, d, y)
}

/// The interval one simulated study publishes for the identified set, with its construction;
/// `None` when the route refuses the sample's law (it violates the instrumental inequality),
/// publishes no interval, or does not state its support row: a fit outside the row is not
/// evidence for it.
fn iv_interval(n: usize, seed: u64, bayesian: bool) -> Option<([f64; 2], Reported)> {
    let (z, d, y) = response_type_mixture_sample(n, seed);
    let data = TabularData::from_f64_columns([
        ("y", y.as_slice()),
        ("d", d.as_slice()),
        ("z", z.as_slice()),
    ])
    .expect("the simulated columns are finite and aligned");
    let query = CausalQuery::IvBounds(IvBoundsQuery::new(v(0), v(1), v(2)));
    let builder = Study::tabular(data.clone()).query(query).refute(RefuteSuite::None);
    let builder = if bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(IV_DRAWS)))
    } else {
        builder.bootstrap_replicates(IV_REPLICATES)
    };
    let study = builder.build().expect("the bounds study is valid");
    let ctx = ExecutionContext::for_tests(seed);
    let prepared = study.prepare(&ctx).expect("the bounds study prepares");
    let result = prepared.estimate(&data, &ctx).ok()?;
    let interval = result.iv_bounds().filter(|_| result.states_section_row())?.interval?;
    let wanted = if bayesian { IV_BAYESIAN_INTERVAL } else { IV_FREQUENTIST_INTERVAL };
    assert_eq!(interval.method.as_str(), wanted, "the route formed another interval");
    assert!((interval.level - REPORTED_LEVEL).abs() < 1e-12, "the route reports another level");
    let resamples = u32::try_from(interval.replicates).ok();
    let reported = reported_as(
        &prepared,
        &result,
        &Scored {
            interval_method: wanted,
            level: interval.level,
            se_kind: "",
            dependence: "iid",
            functional: IV_FUNCTIONAL,
            replicates: if bayesian { None } else { resamples },
            draws: if bayesian { resamples } else { None },
            posterior: bayesian.then_some("bayesian_bootstrap.row_weights"),
        },
    );
    Some(([interval.lower, interval.upper], reported))
}

#[derive(Clone, Copy)]
enum Edge {
    Lower,
    Upper,
}

fn measure_iv(test: &'static str, bayesian: bool, edge: Edge, seed_base: u64) {
    let n = grid_n(IV_ROWS);
    let (lower, upper) = true_effect_bounds();
    let (truth, side, target) = match edge {
        Edge::Lower => (lower, CoverageSide::Lower, "true sharp lower bound of the average effect"),
        Edge::Upper => (upper, CoverageSide::Upper, "true sharp upper bound of the average effect"),
    };
    let interval = if bayesian { IV_BAYESIAN_INTERVAL } else { IV_FREQUENTIST_INTERVAL };
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "response_type_mixture_population_law", interval },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| iv_interval(n, grid_seed(seed_base + rep), bayesian));
    for run in runs {
        // A sample whose law the route refuses, or that publishes no interval or states no
        // support, is a miss.
        let Some((band, reported)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &[reported]);
        tally.record(Some((band[0], band[1])), truth);
    }
    tally.assert_one_sided(side, target);
}

#[test]
fn the_population_bounds_are_the_sharp_set_of_the_mixture() {
    // The sharp endpoints of this law (the optima of the linear program over the 16 response
    // types, which the closed form reproduces) and the average effect of the mixture inside them.
    let (lower, upper) = true_effect_bounds();
    assert!((lower - 0.05).abs() < 1e-12 && (upper - 0.55).abs() < 1e-12, "{lower}, {upper}");
    let truth: f64 = (0..3)
        .map(|k| TYPE_SHARE[k] * (OUTCOME_UNDER_TREATMENT[k] - OUTCOME_UNDER_CONTROL[k]))
        .sum();
    assert!(lower < truth && truth < upper);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn iv_bounds_imbens_manski_interval_covers_the_lower_bound() {
    measure_iv(
        "iv_bounds_imbens_manski_interval_covers_the_lower_bound",
        false,
        Edge::Lower,
        301_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn iv_bounds_imbens_manski_interval_covers_the_upper_bound() {
    measure_iv(
        "iv_bounds_imbens_manski_interval_covers_the_upper_bound",
        false,
        Edge::Upper,
        302_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_iv_bounds_endpoint_interval_covers_the_lower_bound() {
    measure_iv(
        "bayesian_iv_bounds_endpoint_interval_covers_the_lower_bound",
        true,
        Edge::Lower,
        303_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_iv_bounds_endpoint_interval_covers_the_upper_bound() {
    measure_iv(
        "bayesian_iv_bounds_endpoint_interval_covers_the_upper_bound",
        true,
        Edge::Upper,
        304_000,
    );
}

// ------------------------------------------------------------- association

const TABLE_ROWS: usize = 200;
const SERIES_ROWS: usize = 200;
const REPLICATES: u32 = 399;
const DRAWS: usize = 400;
const SLOPE: f64 = 2.0;
const SERIES_SLOPE: f64 = 0.5;
const SERIES_PERSISTENCE: f64 = 0.5;
const ASSOCIATION_FUNCTIONAL: &str = "linear_regression_slope";

/// Rows `y = 1 + 2 x + e`, `x ~ N(0, 1)`, with error standard deviation `0.5 + 0.5 |x|`.
fn heteroskedastic_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 3));
    let mut draw_e = gaussian(stream_seed(seed, 4));
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x.iter().map(|x| 1.0 + SLOPE * x + (0.5 + 0.5 * x.abs()) * draw_e()).collect();
    (x, y)
}

/// Rows `y = 1 + 2 x + e`, `x ~ N(0, 1)`, `e ~ N(0, 1)`.
fn homoskedastic_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 5));
    let mut draw_e = gaussian(stream_seed(seed, 6));
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x.iter().map(|x| 1.0 + SLOPE * x + draw_e()).collect();
    (x, y)
}

/// A series `y_t = 1 + 0.5 x_t + u_t` with the predictor `x` and the error `u` independent
/// stationary AR(1) processes of persistence 0.5.
fn autoregressive_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let x = ar1_noise(n, SERIES_PERSISTENCE, 1.0, stream_seed(seed, 1));
    let u = ar1_noise(n, SERIES_PERSISTENCE, 1.0, stream_seed(seed, 2));
    let y = x.iter().zip(&u).map(|(x, u)| 1.0 + SERIES_SLOPE * x + u).collect();
    (x, y)
}

/// Which interval of the slope term a design scores.
#[derive(Clone, Copy)]
enum Source {
    /// `estimate +- z * standard_error`, at the level the facade reports.
    StandardError,
    /// The resampling spread the estimator publishes.
    Spread,
    /// The equal-tailed posterior interval.
    Posterior,
}

#[derive(Clone, Copy)]
struct Design {
    temporal: bool,
    bayesian: bool,
    homoskedastic: bool,
    replicates: u32,
    source: Source,
    interval: &'static str,
    dependence: &'static str,
}

fn association_study(builder: StudyBuilder, design: Design) -> Study {
    let query =
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Coefficients);
    let builder = builder.query(CausalQuery::Association(query)).refute(RefuteSuite::None);
    let builder = if design.bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
    } else if design.replicates > 0 {
        builder.bootstrap_replicates(design.replicates)
    } else {
        builder
    };
    let builder = if design.temporal && !design.bayesian && design.replicates > 0 {
        builder.sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
    } else {
        builder
    };
    builder.build().expect("the association study is valid")
}

/// The interval one simulated study publishes for the slope, with its construction; `None` when
/// the result states none or does not state its support row.
fn association_interval(n: usize, seed: u64, design: Design) -> Option<([f64; 2], Reported)> {
    let ctx = ExecutionContext::for_tests(seed);
    let (prepared, result) = if design.temporal {
        let (x, y) = autoregressive_line_known_slope(n, seed);
        let data = TimeSeriesData::from_f64_columns(
            [("x", x.as_slice()), ("y", y.as_slice())],
            1_000_000_000,
        )
        .expect("the simulated series is finite and aligned");
        let prepared = association_study(Study::series(data.clone()), design)
            .prepare(&ctx)
            .expect("the association study prepares");
        let result = prepared.estimate_series(&data, &ctx).expect("the association runs");
        (prepared, result)
    } else {
        let (x, y) = if design.homoskedastic {
            homoskedastic_line_known_slope(n, seed)
        } else {
            heteroskedastic_line_known_slope(n, seed)
        };
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("the simulated rows are finite and aligned");
        let prepared = association_study(Study::tabular(data.clone()), design)
            .prepare(&ctx)
            .expect("the association study prepares");
        let result = prepared.estimate(&data, &ctx).expect("the association runs");
        (prepared, result)
    };
    let payload = result.association().filter(|_| result.states_section_row())?;
    assert_eq!(payload.estimate.terms.len(), 2, "an intercept and one slope");
    let term = &payload.estimate.terms[1];
    // The facade reports 0.95 for a standard-error interval and the estimator forms its spread
    // and credible interval at the same level.
    let band = match design.source {
        Source::StandardError => normal_interval(term.estimate, term.standard_error, Z95),
        Source::Spread => term.spread,
        Source::Posterior => term.posterior.map(|summary| (summary.lower, summary.upper)),
    }?;
    let reported = reported_as(
        &prepared,
        &result,
        &Scored {
            interval_method: design.interval,
            level: REPORTED_LEVEL,
            // A posterior interval has no standard-error construction of its own.
            se_kind: match design.source {
                Source::Posterior => "",
                _ => term.standard_error_method.unwrap_or(""),
            },
            dependence: design.dependence,
            functional: ASSOCIATION_FUNCTIONAL,
            replicates: payload.estimate.resampling.map(|record| record.replicates_ok),
            draws: payload
                .estimate
                .posterior
                .filter(|record| record.draws > 0)
                .and_then(|record| u32::try_from(record.draws).ok()),
            // The series posterior is the block Bayesian bootstrap, not the conjugate model.
            posterior: (design.bayesian && design.temporal)
                .then_some("bayesian_bootstrap.contiguous_block_weights"),
        },
    );
    Some(([band.0, band.1], reported))
}

fn measure_association(
    test: &'static str,
    dgp: &'static str,
    design: Design,
    base_rows: usize,
    truth: f64,
    seed_base: u64,
) {
    let n = grid_n(base_rows);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: design.interval },
        REPORTED_LEVEL,
    );
    let runs =
        map_replicates(n_sim(), |rep| association_interval(n, grid_seed(seed_base + rep), design));
    for run in runs {
        let Some((band, reported)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &[reported]);
        tally.record(Some((band[0], band[1])), truth);
    }
    tally.assert();
}

const STATIC: Design = Design {
    temporal: false,
    bayesian: false,
    homoskedastic: false,
    replicates: 0,
    source: Source::StandardError,
    interval: "analytic_se",
    dependence: "iid",
};

const TEMPORAL: Design = Design { temporal: true, dependence: "serial_hac", ..STATIC };

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_sandwich_slope_interval() {
    measure_association(
        "association_static_sandwich_slope_interval",
        "heteroskedastic_line_known_slope",
        STATIC,
        TABLE_ROWS,
        SLOPE,
        311_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_bootstrap_spread_slope_interval() {
    measure_association(
        "association_static_bootstrap_spread_slope_interval",
        "heteroskedastic_line_known_slope",
        Design {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "studentized_spread",
            ..STATIC
        },
        TABLE_ROWS,
        SLOPE,
        312_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_temporal_newey_west_slope_interval() {
    measure_association(
        "association_temporal_newey_west_slope_interval",
        "autoregressive_line_known_slope",
        TEMPORAL,
        SERIES_ROWS,
        SERIES_SLOPE,
        313_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_temporal_block_bootstrap_spread_slope_interval() {
    measure_association(
        "association_temporal_block_bootstrap_spread_slope_interval",
        "autoregressive_line_known_slope",
        Design {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "fixed_b_percentile_spread",
            dependence: "circular_block:row_block",
            ..TEMPORAL
        },
        SERIES_ROWS,
        SERIES_SLOPE,
        314_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_static_slope_credible_interval() {
    measure_association(
        "bayesian_association_static_slope_credible_interval",
        "homoskedastic_line_known_slope",
        Design {
            bayesian: true,
            homoskedastic: true,
            source: Source::Posterior,
            interval: "posterior_equal_tailed",
            ..STATIC
        },
        TABLE_ROWS,
        SLOPE,
        315_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_temporal_slope_credible_interval() {
    measure_association(
        "bayesian_association_temporal_slope_credible_interval",
        "autoregressive_line_known_slope",
        Design {
            bayesian: true,
            source: Source::Posterior,
            interval: "posterior_equal_tailed",
            dependence: "contiguous_block:row_block",
            ..TEMPORAL
        },
        SERIES_ROWS,
        SERIES_SLOPE,
        316_000,
    );
}
