//! Coverage of the further analyses of the descriptive association routes: the mean or
//! probability of a stratum, a logistic or Poisson coefficient, a Pearson or distance
//! correlation, the row Bayesian bootstrap, and a stratum mean of one series. The support
//! row covers only the linear slope, so a replicate counts whenever the term publishes.
//!
//! Builders mirror the family's first harness file, parametrized where a statement needs it.
//! Records are keyed on the design axes (structure `design`, validation `not_applicable`). A
//! replicate that publishes nothing to score is a skip (a miss).
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "row indices and counts convert exactly at these sizes"
)]

mod common;

use antecedent::analysis::StudyBuilder;
use antecedent::{
    AssociationFunctional, AssociationMeasure, AssociationOutcome, AssociationQuery,
    BayesianConfig, CausalQuery, InferenceMode, PreparedStudy, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{DeclaredSamplingDesign, ExecutionContext, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, Z95, ar1_noise, gaussian,
    grid_n, grid_seed, map_replicates, n_sim, normal_interval, stream_seed, uniform,
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
        ..base
    };
    let scope =
        ScopeFacts { replicates_ok: scored.replicates, posterior_draws: scored.draws, ..scope };
    (construction, scope, "not_applicable")
}
// ------------------------------------------------------------------ bounds

// ------------------------------------------------------------- association

const TABLE_ROWS: usize = 200;

const SERIES_ROWS: usize = 200;

const REPLICATES: u32 = 399;

const DRAWS: usize = 400;

const SERIES_PERSISTENCE: f64 = 0.5;

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
// ---------------------------------------------- association: other analyses

const STRATUM_SHARE: f64 = 0.4;

const STRATUM_MEAN: f64 = 3.0;

const STRATUM_PROBABILITY: f64 = 0.6;

const LOGISTIC_SLOPE: f64 = 1.0;

const POISSON_SLOPE: f64 = 0.4;

const CORRELATION: f64 = 0.5;

const REGIME_MEAN: f64 = 2.0;

const REGIME_STAY: f64 = 0.8;

/// Rows with a 0/1 group `g` (share 0.4 in group 1) and `y = 1 + 2 g + (1 + g) e`,
/// `e ~ N(0, 1)`: the mean of `y` in group 1 is 3.
fn two_group_known_mean(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut next = uniform(seed, 0x1B11);
    let mut draw_e = gaussian(stream_seed(seed, 7));
    let g: Vec<f64> = (0..n).map(|_| f64::from(next() < STRATUM_SHARE)).collect();
    let y = g.iter().map(|g| 1.0 + 2.0 * g + (1.0 + g) * draw_e()).collect();
    (g, y)
}

/// Rows with a 0/1 group `g` (share 0.4 in group 1) and a 0/1 outcome of probability 0.3 in
/// group 0 and 0.6 in group 1.
fn two_group_known_probability(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut next = uniform(seed, 0x1B12);
    let g: Vec<f64> = (0..n).map(|_| f64::from(next() < STRATUM_SHARE)).collect();
    let y = g
        .iter()
        .map(|g| f64::from(next() < if *g > 0.5 { STRATUM_PROBABILITY } else { 0.3 }))
        .collect();
    (g, y)
}

/// Rows `x ~ N(0, 1)` and a 0/1 outcome with log odds `-0.5 + x`: the model is the law, so
/// the population slope is 1.
fn logistic_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 8));
    let mut next = uniform(seed, 0x1B13);
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x
        .iter()
        .map(|x| f64::from(next() < 1.0 / (1.0 + (0.5 - LOGISTIC_SLOPE * x).exp())))
        .collect();
    (x, y)
}

/// Rows `x ~ N(0, 1)` and a Poisson count of log rate `0.5 + 0.4 x`: the population slope is
/// 0.4. The count is drawn by multiplying uniforms down to `exp(-rate)`.
fn poisson_line_known_slope(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 9));
    let mut next = uniform(seed, 0x1B14);
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x
        .iter()
        .map(|x| {
            let floor = (-(0.5 + POISSON_SLOPE * x).exp()).exp();
            let (mut count, mut product) = (0.0, next());
            while product > floor {
                count += 1.0;
                product *= next();
            }
            count
        })
        .collect();
    (x, y)
}

/// Bivariate normal rows of unit variances and correlation 0.5.
fn bivariate_normal_known_correlation(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut draw_x = gaussian(stream_seed(seed, 10));
    let mut draw_e = gaussian(stream_seed(seed, 11));
    let x: Vec<f64> = (0..n).map(|_| draw_x()).collect();
    let y = x
        .iter()
        .map(|x| CORRELATION * x + (1.0 - CORRELATION * CORRELATION).sqrt() * draw_e())
        .collect();
    (x, y)
}

/// Distance correlation of a bivariate normal law of correlation `rho` (Szekely, Rizzo and
/// Bakirov 2007, theorem 7): the root of
/// `(rho asin rho + sqrt(1 - rho^2) - rho asin(rho / 2) - sqrt(4 - rho^2) + 1) / (1 + pi / 3 - sqrt 3)`.
fn bivariate_normal_distance_correlation(rho: f64) -> f64 {
    let numerator = rho * rho.asin() + (1.0 - rho * rho).sqrt()
        - rho * (rho / 2.0).asin()
        - (4.0 - rho * rho).sqrt()
        + 1.0;
    (numerator / (1.0 + std::f64::consts::PI / 3.0 - 3.0_f64.sqrt())).sqrt()
}

/// A series with a 0/1 regime `g` (a symmetric two-state chain that stays with probability
/// 0.8) and `y_t = 1 + g_t + u_t`, `u` a stationary AR(1) of persistence 0.5 independent of the
/// regime: the mean of `y` in regime 1 is 2.
fn regime_series_known_mean(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut next = uniform(seed, 0x1B15);
    let u = ar1_noise(n, SERIES_PERSISTENCE, 1.0, stream_seed(seed, 12));
    let mut state = next() < 0.5;
    let g: Vec<f64> = (0..n)
        .map(|_| {
            if next() >= REGIME_STAY {
                state = !state;
            }
            f64::from(state)
        })
        .collect();
    let y = g.iter().zip(&u).map(|(g, u)| 1.0 + g + u).collect();
    (g, y)
}

/// One analysis of the selection rule and the term of it a design scores.
#[derive(Clone, Copy)]
struct Analysis {
    temporal: bool,
    bayesian: bool,
    replicates: u32,
    /// Predictor and outcome columns of `n` rows.
    sample: fn(usize, u64) -> (Vec<f64>, Vec<f64>),
    requested: AssociationFunctional,
    scale: AssociationOutcome,
    /// The analysis the selection rule must pick.
    analysis: &'static str,
    /// Index of the scored term among `terms` published ones.
    term: usize,
    terms: usize,
    source: Source,
    interval: &'static str,
    dependence: &'static str,
    functional: &'static str,
}

/// The interval one simulated study publishes for the scored term, with its construction;
/// `None` when the route refuses the sample or the term states no such interval.
fn analysis_interval(n: usize, seed: u64, design: Analysis) -> Option<([f64; 2], Reported)> {
    let ctx = ExecutionContext::for_tests(seed);
    let (x, y) = (design.sample)(n, seed);
    let columns = [("x", x.as_slice()), ("y", y.as_slice())];
    let query = AssociationQuery::new(v(1), [v(0)])
        .with_functional(design.requested)
        .with_outcome_scale(design.scale);
    let configure = |builder: StudyBuilder| {
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
    };
    let (prepared, result) = if design.temporal {
        let data = TimeSeriesData::from_f64_columns(columns, 1_000_000_000)
            .expect("the simulated series is finite and aligned");
        let prepared = configure(Study::series(data.clone()))
            .prepare(&ctx)
            .expect("the association study prepares");
        let result = prepared.estimate_series(&data, &ctx).ok()?;
        (prepared, result)
    } else {
        let data = TabularData::from_f64_columns(columns)
            .expect("the simulated rows are finite and aligned");
        let prepared = configure(Study::tabular(data.clone()))
            .prepare(&ctx)
            .expect("the association study prepares");
        let result = prepared.estimate(&data, &ctx).ok()?;
        (prepared, result)
    };
    let payload = result.association()?;
    assert_eq!(payload.estimate.selection.analysis.as_str(), design.analysis);
    assert_eq!(payload.estimate.terms.len(), design.terms, "the published terms");
    let term = &payload.estimate.terms[design.term];
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
            se_kind: match design.source {
                Source::Posterior => "",
                _ => term.standard_error_method.unwrap_or(""),
            },
            dependence: design.dependence,
            functional: design.functional,
            replicates: payload.estimate.resampling.map(|record| record.replicates_ok),
            draws: payload
                .estimate
                .posterior
                .filter(|record| record.draws > 0)
                .and_then(|record| u32::try_from(record.draws).ok()),
            // No conjugate form: Dirichlet weights on rows, or on blocks of a series.
            posterior: design.bayesian.then_some(if design.temporal {
                "bayesian_bootstrap.contiguous_block_weights"
            } else {
                "bayesian_bootstrap.row_weights"
            }),
        },
    );
    Some(([band.0, band.1], reported))
}

fn measure_analysis(
    test: &'static str,
    dgp: &'static str,
    design: Analysis,
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
        map_replicates(n_sim(), |rep| analysis_interval(n, grid_seed(seed_base + rep), design));
    for run in runs {
        // A refused sample or a term with no interval is a miss.
        let Some((band, reported)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &[reported]);
        tally.record(Some((band[0], band[1])), truth);
    }
    tally.assert();
}

/// The mean of the second of two strata on a table, by its analytic error.
const STRATUM: Analysis = Analysis {
    temporal: false,
    bayesian: false,
    replicates: 0,
    sample: two_group_known_mean,
    requested: AssociationFunctional::Conditional,
    scale: AssociationOutcome::Continuous,
    analysis: "stratified_mean",
    term: 1,
    terms: 2,
    source: Source::StandardError,
    interval: "analytic_se",
    dependence: "iid",
    functional: "stratum_mean",
};

/// The slope of a logistic regression on a table, by its analytic error.
const LOGISTIC: Analysis = Analysis {
    sample: logistic_line_known_slope,
    requested: AssociationFunctional::Coefficients,
    scale: AssociationOutcome::Binary,
    analysis: "logistic_regression",
    functional: "logistic_regression_slope",
    ..STRATUM
};

/// The slope of a Poisson regression on a table, by its analytic error.
const POISSON: Analysis = Analysis {
    sample: poisson_line_known_slope,
    scale: AssociationOutcome::Count,
    analysis: "poisson_regression",
    functional: "poisson_regression_slope",
    ..LOGISTIC
};

/// The Pearson correlation on a table, by its analytic error.
const PEARSON: Analysis = Analysis {
    sample: bivariate_normal_known_correlation,
    requested: AssociationFunctional::Measure(AssociationMeasure::Pearson),
    analysis: "pearson_correlation",
    term: 0,
    terms: 1,
    functional: "pearson_correlation",
    ..STRATUM
};

/// The mean of the second of two regimes of a series, by its analytic error.
const SERIES_STRATUM: Analysis = Analysis {
    temporal: true,
    sample: regime_series_known_mean,
    dependence: "serial_hac",
    functional: "stratum_mean",
    ..STRATUM
};

#[test]
fn the_distance_correlation_of_the_normal_law_matches_its_closed_form() {
    // Independence gives 0 and a perfect line 1; at correlation 0.5 the value is 0.4541...
    assert!(bivariate_normal_distance_correlation(0.0).abs() < 1e-12);
    assert!((bivariate_normal_distance_correlation(1.0) - 1.0).abs() < 1e-12);
    let value = bivariate_normal_distance_correlation(CORRELATION);
    assert!((value - 0.454_126_504_440_247_7).abs() < 1e-12, "{value}");
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_stratum_mean_interval() {
    measure_analysis(
        "association_static_stratum_mean_interval",
        "two_group_known_mean",
        STRATUM,
        TABLE_ROWS,
        STRATUM_MEAN,
        321_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_stratum_probability_interval() {
    measure_analysis(
        "association_static_stratum_probability_interval",
        "two_group_known_probability",
        Analysis {
            sample: two_group_known_probability,
            scale: AssociationOutcome::Binary,
            analysis: "stratified_probability",
            functional: "stratum_probability",
            ..STRATUM
        },
        TABLE_ROWS,
        STRATUM_PROBABILITY,
        322_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_stratum_mean_bootstrap_spread_interval() {
    measure_analysis(
        "association_static_stratum_mean_bootstrap_spread_interval",
        "two_group_known_mean",
        Analysis {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "percentile_spread",
            ..STRATUM
        },
        TABLE_ROWS,
        STRATUM_MEAN,
        323_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_logistic_slope_interval() {
    measure_analysis(
        "association_static_logistic_slope_interval",
        "logistic_line_known_slope",
        LOGISTIC,
        TABLE_ROWS,
        LOGISTIC_SLOPE,
        324_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_logistic_bootstrap_spread_slope_interval() {
    measure_analysis(
        "association_static_logistic_bootstrap_spread_slope_interval",
        "logistic_line_known_slope",
        Analysis {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "studentized_spread",
            ..LOGISTIC
        },
        TABLE_ROWS,
        LOGISTIC_SLOPE,
        325_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_poisson_slope_interval() {
    measure_analysis(
        "association_static_poisson_slope_interval",
        "poisson_line_known_slope",
        POISSON,
        TABLE_ROWS,
        POISSON_SLOPE,
        326_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_poisson_bootstrap_spread_slope_interval() {
    measure_analysis(
        "association_static_poisson_bootstrap_spread_slope_interval",
        "poisson_line_known_slope",
        Analysis {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "studentized_spread",
            ..POISSON
        },
        TABLE_ROWS,
        POISSON_SLOPE,
        327_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_pearson_correlation_interval() {
    measure_analysis(
        "association_static_pearson_correlation_interval",
        "bivariate_normal_known_correlation",
        PEARSON,
        TABLE_ROWS,
        CORRELATION,
        328_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_pearson_correlation_bootstrap_spread_interval() {
    measure_analysis(
        "association_static_pearson_correlation_bootstrap_spread_interval",
        "bivariate_normal_known_correlation",
        Analysis {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "percentile_spread",
            ..PEARSON
        },
        TABLE_ROWS,
        CORRELATION,
        329_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_static_distance_correlation_bootstrap_spread_interval() {
    // The distance correlation has no analytic error: its only interval is the spread, the
    // normal interval of the jackknife error around the unbiased point.
    measure_analysis(
        "association_static_distance_correlation_bootstrap_spread_interval",
        "bivariate_normal_known_correlation",
        Analysis {
            requested: AssociationFunctional::Measure(AssociationMeasure::Distance),
            analysis: "distance_correlation",
            functional: "distance_correlation",
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "jackknife_normal_spread",
            ..PEARSON
        },
        TABLE_ROWS,
        bivariate_normal_distance_correlation(CORRELATION),
        330_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_static_bootstrap_logistic_slope_credible_interval() {
    measure_analysis(
        "bayesian_association_static_bootstrap_logistic_slope_credible_interval",
        "logistic_line_known_slope",
        Analysis {
            bayesian: true,
            source: Source::Posterior,
            interval: "posterior_equal_tailed",
            ..LOGISTIC
        },
        TABLE_ROWS,
        LOGISTIC_SLOPE,
        331_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_static_bootstrap_stratum_probability_credible_interval() {
    measure_analysis(
        "bayesian_association_static_bootstrap_stratum_probability_credible_interval",
        "two_group_known_probability",
        Analysis {
            bayesian: true,
            sample: two_group_known_probability,
            scale: AssociationOutcome::Binary,
            analysis: "stratified_probability",
            functional: "stratum_probability",
            source: Source::Posterior,
            interval: "posterior_equal_tailed",
            ..STRATUM
        },
        TABLE_ROWS,
        STRATUM_PROBABILITY,
        332_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_association_static_bootstrap_pearson_correlation_credible_interval() {
    measure_analysis(
        "bayesian_association_static_bootstrap_pearson_correlation_credible_interval",
        "bivariate_normal_known_correlation",
        Analysis {
            bayesian: true,
            source: Source::Posterior,
            interval: "posterior_equal_tailed",
            ..PEARSON
        },
        TABLE_ROWS,
        CORRELATION,
        333_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_temporal_stratum_mean_newey_west_interval() {
    measure_analysis(
        "association_temporal_stratum_mean_newey_west_interval",
        "regime_series_known_mean",
        SERIES_STRATUM,
        SERIES_ROWS,
        REGIME_MEAN,
        334_000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_temporal_stratum_mean_block_bootstrap_spread_interval() {
    measure_analysis(
        "association_temporal_stratum_mean_block_bootstrap_spread_interval",
        "regime_series_known_mean",
        Analysis {
            replicates: REPLICATES,
            source: Source::Spread,
            interval: "fixed_b_percentile_spread",
            dependence: "circular_block:row_block",
            ..SERIES_STRATUM
        },
        SERIES_ROWS,
        REGIME_MEAN,
        335_000,
    );
}
