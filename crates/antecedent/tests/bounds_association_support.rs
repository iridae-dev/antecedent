//! Support rows of the binary-instrument bounds and of the descriptive association.
//!
//! Each route states its support row only inside the measured range of that graphless row: the
//! interval for the endpoints of the identified set (500 unweighted rows, 399 row-bootstrap
//! replicates or 400 posterior draws, none left out), and the uncertainty of the slope of a
//! linear regression of a real outcome on one numeric predictor (100 rows or steps, 399
//! replicates of a resampling spread, 400 draws of a block posterior). A fit inside the range
//! states the row in a coded diagnostic of the result and of the exported artifact (the
//! section's own `supported` field stays false); an artifact that states the row below a
//! threshold or under another method is refused; a fit outside the range runs and states no
//! support.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "fixture indices and counts convert exactly at these sizes"
)]

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

use antecedent::{
    AssociationQuery, BayesianConfig, CausalQuery, InferenceMode, IvBoundsQuery, PreparedStudy,
    RefuteSuite, Study, StudyResult,
};
use antecedent_core::{DeclaredSamplingDesign, ExecutionContext, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_io::analysis_result_artifact::AnalysisResultWire;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// The exported artifact of a result: its variable names and its body.
fn exported(
    label: &str,
    prepared: &PreparedStudy,
    result: &StudyResult,
    ctx: &ExecutionContext,
) -> (Vec<String>, AnalysisResultWire) {
    let bytes = prepared.encode_contracted_result(result, label, ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    (header.variable_names, body)
}

/// Whether the artifact check refuses `forged` for a support row its section does not meet.
fn refused(names: &[String], forged: &AnalysisResultWire) -> bool {
    antecedent_io::encode_analysis_result_artifact(forged, names.to_vec(), "forged-support")
        .is_err_and(|error| error.to_string().contains("states a graphless support row"))
}

/// Whether the artifact check refuses `forged` for the section's own `supported` flag, which
/// no producer sets.
fn refused_flag(names: &[String], forged: &AnalysisResultWire) -> bool {
    antecedent_io::encode_analysis_result_artifact(forged, names.to_vec(), "forged-flag")
        .is_err_and(|error| error.to_string().contains("claims no support"))
}

/// Whether the exported body states the support row `code` names.
fn stated(body: &AnalysisResultWire, code: &str) -> bool {
    body.diagnostics.iter().any(|diagnostic| diagnostic.code == code)
}

/// `body` without the statement of the row `code` names.
fn without_statement(body: &AnalysisResultWire, code: &str) -> AnalysisResultWire {
    let mut unclaimed = body.clone();
    unclaimed.diagnostics.retain(|diagnostic| diagnostic.code != code);
    unclaimed
}

const BOUNDS_ROW: &str = antecedent_io::graphless_support::IV_BOUNDS_ROW_CODE;
const SLOPE_ROW: &str = antecedent_io::graphless_support::ASSOCIATION_SLOPE_ROW_CODE;

// ------------------------------------------------------------------ bounds

/// Rows per instrument arm of each `(y, d)` cell, order `(0,0), (1,0), (0,1), (1,1)`, for 100
/// rows per arm: the exact law of 20% always-takers, 30% never-takers and 50% compliers with
/// outcome probabilities 0.7, 0.4, 0.8 under treatment and 0.5, 0.3, 0.4 under control. Its
/// sharp bounds on the average effect are 0.05 and 0.55.
const MIXTURE_CELLS: [[usize; 4]; 2] = [[51, 29, 6, 14], [21, 9, 16, 54]];
const MIXTURE_LOWER: f64 = 0.05;
const MIXTURE_UPPER: f64 = 0.55;

/// The mixture's law as `copies * 200` rows with columns y, d, z and a unit weight.
fn mixture_table(copies: usize) -> TabularData {
    let (mut y, mut d, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for (arm, cells) in MIXTURE_CELLS.iter().enumerate() {
        for (cell, &count) in cells.iter().enumerate() {
            for _ in 0..count * copies {
                y.push((cell % 2) as f64);
                d.push((cell / 2) as f64);
                z.push(arm as f64);
            }
        }
    }
    let w = vec![1.0; y.len()];
    TabularData::from_f64_columns([("y", &y[..]), ("d", &d[..]), ("z", &z[..]), ("w", &w[..])])
        .unwrap()
}

/// How one bounds study is run.
#[derive(Clone, Copy)]
struct BoundsRun {
    copies: usize,
    /// Row-bootstrap replicates (frequentist) or posterior draws (Bayesian).
    resamples: u32,
    bayesian: bool,
    weighted: bool,
}

impl BoundsRun {
    /// Prepare the study and estimate it from its retained handle.
    fn run(self, ctx: &ExecutionContext) -> (PreparedStudy, StudyResult) {
        let data = mixture_table(self.copies);
        let query = IvBoundsQuery::new(v(0), v(1), v(2));
        let query = if self.weighted { query.with_weight(v(3)) } else { query };
        let builder = Study::tabular(data.clone())
            .query(CausalQuery::IvBounds(query))
            .refute(RefuteSuite::None);
        let builder = if self.bayesian {
            builder.inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(self.resamples as usize),
            ))
        } else {
            builder.bootstrap_replicates(self.resamples)
        };
        let prepared = builder.build().unwrap().prepare(ctx).unwrap();
        let result = prepared.estimate(&data, ctx).unwrap();
        (prepared, result)
    }
}

/// A bounds fit inside its row: supported, with the sharp bounds of the law and an interval
/// about them; the artifact keeps the row and refuses each forged condition; fits below the
/// rows or the resamples, and a weighted fit, run unsupported.
fn bounds_row_round_trips(label: &str, bayesian: bool, resamples: u32, method: &str) {
    let ctx = ExecutionContext::for_tests(23);
    let inside = BoundsRun { copies: 3, resamples, bayesian, weighted: false };
    let (prepared, result) = inside.run(&ctx);
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let fit = result.iv_bounds().expect("a bounds payload").clone();
    assert!(result.states_section_row(), "{label}");
    assert_eq!((fit.rows, fit.rows_dropped), (600, 0), "{label}");
    assert!((fit.lower - MIXTURE_LOWER).abs() < 1e-9, "{label}: {}", fit.lower);
    assert!((fit.upper - MIXTURE_UPPER).abs() < 1e-9, "{label}: {}", fit.upper);
    let interval = fit.interval.expect("an interval about the set");
    assert_eq!(interval.method.as_str(), method, "{label}");
    assert_eq!((interval.replicates, interval.infeasible), (resamples as usize, 0), "{label}");
    assert!((interval.level - 0.95).abs() < 1e-12, "{label}");
    assert!(interval.width_retained, "{label}: the set is wider than its sampling noise");
    assert!(interval.lower < fit.lower && fit.upper < interval.upper, "{label}: {interval:?}");

    let (names, body) = exported(label, &prepared, &result, &ctx);
    let wire = body.iv_bounds.as_ref().expect("the section is exported");
    assert!(stated(&body, BOUNDS_ROW), "{label}");
    assert!(!wire.supported && wire.calibration == "not_measured", "{label}");
    assert!((wire.interval.as_ref().unwrap().lower - interval.lower).abs() < 1e-12, "{label}");
    let forge = |change: &dyn Fn(&mut antecedent_io::analysis_result_artifact::IvBoundsWire)| {
        let mut forged = body.clone();
        change(forged.iv_bounds.as_mut().unwrap());
        forged
    };
    assert!(refused(&names, &forge(&|section| section.rows = 499)), "{label}: rows");
    assert!(
        refused(&names, &forge(&|section| section.interval.as_mut().unwrap().replicates -= 1)),
        "{label}: resamples"
    );
    assert!(
        refused(&names, &forge(&|section| section.interval.as_mut().unwrap().infeasible = 1)),
        "{label}: a resample left out"
    );
    assert!(
        refused(&names, &forge(&|section| section.interval.as_mut().unwrap().level = 0.9)),
        "{label}: level"
    );
    assert!(
        refused(&names, &forge(&|section| "another_method".clone_into(&mut section.uncertainty))),
        "{label}: method"
    );
    assert!(refused(&names, &forge(&|section| section.rows_dropped = 1)), "{label}: dropped");
    assert!(
        refused(
            &names,
            &forge(&|section| section.interval.as_mut().unwrap().width_retained = false)
        ),
        "{label}: a set within its sampling noise"
    );
    // The section's own flag is never a statement: a body that sets it is refused.
    assert!(refused_flag(&names, &forge(&|section| section.supported = true)), "{label}: flag");
    // A result may state less than its row grants; it then loads as an unsupported result.
    let unclaimed = without_statement(&body, BOUNDS_ROW);
    antecedent_io::encode_analysis_result_artifact(&unclaimed, names, "unclaimed-support").unwrap();

    // Outside the row the same route runs and states no support.
    for outside in [
        BoundsRun { copies: 2, ..inside },
        BoundsRun { resamples: resamples - 1, ..inside },
        BoundsRun { weighted: true, ..inside },
    ] {
        let (outside_prepared, result) = outside.run(&ctx);
        let fit = result.iv_bounds().unwrap();
        assert!(
            !result.states_section_row(),
            "{label}: {} rows, weighted {}",
            fit.rows,
            outside.weighted
        );
        assert!((fit.lower - MIXTURE_LOWER).abs() < 1e-9, "{label}");
        // The statement of the supported fit, carried onto a fit outside the row, is refused.
        let (outside_names, mut claimed) = exported(label, &outside_prepared, &result, &ctx);
        assert!(!stated(&claimed, BOUNDS_ROW), "{label}");
        claimed.diagnostics.extend(
            body.diagnostics.iter().filter(|diagnostic| diagnostic.code == BOUNDS_ROW).cloned(),
        );
        assert!(refused(&outside_names, &claimed), "{label}: a statement outside the row");
    }
}

#[test]
fn supported_iv_bounds_endpoint_interval_round_trips_and_refuses_forgery() {
    bounds_row_round_trips(
        "supported_iv_bounds_endpoint_interval",
        false,
        399,
        "imbens_manski_row_bootstrap",
    );
    // The identified set alone, with no sampling interval, has no row.
    let ctx = ExecutionContext::for_tests(23);
    let point = BoundsRun { copies: 3, resamples: 0, bayesian: false, weighted: false };
    let (_, result) = point.run(&ctx);
    let fit = result.iv_bounds().unwrap();
    assert!(!result.states_section_row() && fit.interval.is_none());
    assert!((fit.upper - fit.lower - 0.5).abs() < 1e-9, "{} {}", fit.lower, fit.upper);
}

#[test]
fn supported_bayesian_iv_bounds_endpoint_posterior_round_trips_and_refuses_forgery() {
    bounds_row_round_trips(
        "supported_bayesian_iv_bounds_endpoint_posterior",
        true,
        400,
        "posterior_endpoint_quantiles",
    );
    let ctx = ExecutionContext::for_tests(23);
    let inside = BoundsRun { copies: 3, resamples: 400, bayesian: true, weighted: false };
    let (_, result) = inside.run(&ctx);
    let posterior = result.iv_bounds().unwrap().posterior.expect("a posterior summary");
    assert_eq!((posterior.n_draws, posterior.infeasible_draws), (400, 0));
    assert!((posterior.lower_mean - MIXTURE_LOWER).abs() < 0.05, "{posterior:?}");
    assert!((posterior.upper_mean - MIXTURE_UPPER).abs() < 0.05, "{posterior:?}");
}

// ------------------------------------------------------------- association

/// Deterministic pseudo-noise in `[-0.5, 0.5)`.
fn noise(i: usize) -> f64 {
    ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
}

/// Columns x, y with `y = 1 + 2 x + spread * noise`, a 0/1 column b and a unit weight w.
fn line_columns(rows: usize, spread: f64) -> [Vec<f64>; 4] {
    let x: Vec<f64> = (0..rows).map(|i| i as f64 / 12.0).collect();
    let y = x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + spread * noise(i)).collect();
    // A 0/1 column whose share of ones rises with the row index without separating on it.
    let b = (0..rows).map(|i| f64::from(noise(i + 500) + 0.004 * (i as f64) > 0.2)).collect();
    [x, y, b, vec![1.0; rows]]
}

/// An autoregressive predictor and an outcome that follows it.
fn series_columns(steps: usize) -> [Vec<f64>; 2] {
    let (mut x, mut y) = (vec![0.0; steps], vec![0.0; steps]);
    for t in 1..steps {
        x[t] = 0.5 * x[t - 1] + noise(t);
        y[t] = 0.5 * x[t] + noise(t + 1000);
    }
    [x, y]
}

/// Least-squares slope of `y` on `x`.
fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / sxx
}

/// How one association study is run.
#[derive(Clone, Copy)]
struct SlopeRun {
    temporal: bool,
    rows: usize,
    replicates: u32,
    /// Posterior draws; zero is frequentist inference.
    draws: usize,
    /// The outcome column: 1 is the real outcome, 2 the 0/1 column.
    outcome: u32,
    /// Width of the table outcome's noise about the line.
    spread: f64,
    /// Scale of the conjugate prior; 10 is the built-in scale.
    prior_scale: f64,
    weighted: bool,
    conditioned: bool,
}

const SLOPE: SlopeRun = SlopeRun {
    temporal: false,
    rows: 120,
    replicates: 0,
    draws: 0,
    outcome: 1,
    spread: 8.0,
    prior_scale: 10.0,
    weighted: false,
    conditioned: false,
};

impl SlopeRun {
    /// Prepare the study and estimate it from its retained handle; returns the handle, the
    /// result and the least-squares slope of the columns.
    fn run(self, ctx: &ExecutionContext) -> (PreparedStudy, StudyResult, f64) {
        let settle = |builder: antecedent::analysis::StudyBuilder, query: AssociationQuery| {
            let builder = builder.query(CausalQuery::Association(query)).refute(RefuteSuite::None);
            let builder = if self.draws > 0 {
                builder.inference(InferenceMode::Bayesian(
                    BayesianConfig::conjugate().n_draws(self.draws).prior_scale(self.prior_scale),
                ))
            } else if self.replicates > 0 {
                builder.bootstrap_replicates(self.replicates)
            } else {
                builder
            };
            let builder = if self.temporal && self.replicates > 0 {
                builder.sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
            } else {
                builder
            };
            builder.build().unwrap()
        };
        if self.temporal {
            let [x, y] = series_columns(self.rows);
            let data =
                TimeSeriesData::from_f64_columns([("x", &x[..]), ("y", &y[..])], 1_000_000_000)
                    .unwrap();
            let query = AssociationQuery::new(v(1), [v(0)]);
            let prepared = settle(Study::series(data.clone()), query).prepare(ctx).unwrap();
            let result = prepared.estimate_series(&data, ctx).unwrap();
            return (prepared, result, ols_slope(&x, &y));
        }
        let [x, y, b, w] = line_columns(self.rows, self.spread);
        let data = TabularData::from_f64_columns([
            ("x", &x[..]),
            ("y", &y[..]),
            ("b", &b[..]),
            ("w", &w[..]),
        ])
        .unwrap();
        let query = AssociationQuery::new(v(self.outcome), [v(0)]);
        let query = if self.weighted { query.with_weight(v(3)) } else { query };
        let query = if self.conditioned { query.with_conditioning([v(2)]) } else { query };
        let prepared = settle(Study::tabular(data.clone()), query).prepare(ctx).unwrap();
        let result = prepared.estimate(&data, ctx).unwrap();
        (prepared, result, ols_slope(&x, &y))
    }
}

/// The interval the slope term publishes under `run`: posterior, spread, else the standard
/// error about the estimate.
fn slope_band(result: &StudyResult) -> (f64, f64) {
    let term = &result.association().expect("an association payload").estimate.terms[1];
    if let Some(posterior) = term.posterior {
        (posterior.lower, posterior.upper)
    } else if let Some(spread) = term.spread {
        spread
    } else {
        let error = term.standard_error.expect("a standard error");
        (term.estimate - error, term.estimate + error)
    }
}

/// A slope fit inside its row: supported, with the least-squares slope inside its published
/// interval; the artifact keeps the row and refuses forged rows; every run of `outside` states
/// no support.
fn slope_row_round_trips(
    label: &str,
    inside: SlopeRun,
    method: &str,
    outside: &[SlopeRun],
) -> (Vec<String>, AnalysisResultWire) {
    let ctx = ExecutionContext::for_tests(29);
    let (prepared, result, slope) = inside.run(&ctx);
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let payload = result.association().expect("an association payload");
    assert!(result.states_section_row(), "{label}");
    assert_eq!(payload.estimate.selection.analysis.as_str(), "linear_regression", "{label}");
    assert_eq!((payload.estimate.rows, payload.estimate.rows_dropped), (inside.rows, 0));
    let term = &payload.estimate.terms[1];
    let named = payload.estimate.posterior.map_or(term.standard_error_method, |r| Some(r.method));
    assert_eq!(named, Some(method), "{label}");
    let tolerance = if inside.draws > 0 { 0.05 } else { 1e-8 };
    assert!((term.estimate - slope).abs() < tolerance, "{label}: {}", term.estimate);
    let band = slope_band(&result);
    assert!(band.0 < slope && slope < band.1, "{label}: {band:?} about {slope}");

    let (names, body) = exported(label, &prepared, &result, &ctx);
    let wire = body.association.as_ref().expect("the section is exported");
    assert!(stated(&body, SLOPE_ROW), "{label}");
    assert!(!wire.supported && wire.calibration == "not_measured", "{label}");
    assert!(!wire.ranks_actions && !wire.is_intervention_effect, "{label}");
    assert!((wire.terms[1].estimate - term.estimate).abs() < 1e-12, "{label}");
    let mut thin = body.clone();
    thin.association.as_mut().unwrap().rows = 99;
    assert!(refused(&names, &thin), "{label}: rows");
    let mut dropped = body.clone();
    dropped.association.as_mut().unwrap().rows_dropped = 1;
    assert!(refused(&names, &dropped), "{label}: dropped");
    // The section's own flag is never a statement: a body that sets it is refused.
    let mut flagged = body.clone();
    flagged.association.as_mut().unwrap().supported = true;
    assert!(refused_flag(&names, &flagged), "{label}: flag");
    // A result may state less than its row grants; it then loads as an unsupported result.
    let unclaimed = without_statement(&body, SLOPE_ROW);
    antecedent_io::encode_analysis_result_artifact(&unclaimed, names.clone(), "unclaimed-support")
        .unwrap();

    for run in outside {
        let (_, result, _) = run.run(&ctx);
        let payload = result.association().unwrap();
        assert!(!result.states_section_row(), "{label}: {} rows", payload.estimate.rows);
        assert!(payload.estimate.terms.iter().all(|term| term.estimate.is_finite()), "{label}");
    }
    (names, body)
}

#[test]
fn supported_static_association_slope_round_trips_and_refuses_forgery() {
    // The standard error: 80 rows, a 0/1 outcome (logistic), a weight column and a
    // conditioning column are outside the row.
    let outside = [
        SlopeRun { rows: 80, ..SLOPE },
        SlopeRun { outcome: 2, ..SLOPE },
        SlopeRun { weighted: true, ..SLOPE },
        SlopeRun { conditioned: true, ..SLOPE },
    ];
    let (names, body) = slope_row_round_trips(
        "supported_static_association_standard_error",
        SLOPE,
        "sandwich_leverage_student",
        &outside,
    );
    let mut other = body;
    other.association.as_mut().unwrap().terms[1].standard_error_method =
        Some("sandwich_hc1".to_owned());
    assert!(refused(&names, &other), "another standard-error construction has no row");

    // The studentized spread of 399 row-bootstrap replicates; 398 are below the row.
    let spread = SlopeRun { replicates: 399, ..SLOPE };
    let (names, body) = slope_row_round_trips(
        "supported_static_association_studentized_spread",
        spread,
        "bootstrap_row",
        &[SlopeRun { replicates: 398, ..SLOPE }, SlopeRun { rows: 80, ..spread }],
    );
    let record = body.association.as_ref().unwrap().resampling.as_ref().unwrap();
    assert_eq!((record.replicates_requested, record.replicates_ok), (399, 399));
    let mut failed = body;
    failed.association.as_mut().unwrap().resampling.as_mut().unwrap().replicates_ok = 398;
    assert!(refused(&names, &failed), "a replicate that did not fit leaves the row");
}

#[test]
fn supported_temporal_association_slope_round_trips_and_refuses_forgery() {
    let series = SlopeRun { temporal: true, ..SLOPE };
    let (names, body) = slope_row_round_trips(
        "supported_temporal_association_standard_error",
        series,
        "sandwich_newey_west",
        &[SlopeRun { rows: 90, ..series }],
    );
    // The series row is not the table row: the same section read as a cross-section has none.
    let mut table = body;
    table.association.as_mut().unwrap().modality = "cross_section".to_owned();
    assert!(refused(&names, &table), "a series error is not a row of independent units");

    // The fixed-b spread of 399 circular-block replicates under the declared design.
    let spread = SlopeRun { replicates: 399, ..series };
    let (names, body) = slope_row_round_trips(
        "supported_temporal_association_fixed_b_spread",
        spread,
        "bootstrap_circular_block",
        &[SlopeRun { replicates: 398, ..series }, SlopeRun { rows: 90, ..spread }],
    );
    let record = body.association.as_ref().unwrap().resampling.as_ref().unwrap();
    assert!(record.block_length.is_some_and(|length| length >= 1), "{record:?}");
    let mut short = body;
    let forged = short.association.as_mut().unwrap().resampling.as_mut().unwrap();
    (forged.replicates_requested, forged.replicates_ok) = (398, 398);
    assert!(refused(&names, &short), "398 replicates are below the row");
}

#[test]
fn supported_bayesian_static_association_slope_round_trips_and_refuses_forgery() {
    // The conjugate marginal of a real outcome; a 0/1 outcome or a weight column draws the
    // Bayesian bootstrap, which has no row. A prior of scale 1 moves this slope by about 0.19
    // posterior standard deviations (the built-in scale 10 by about 0.002), far past the
    // hundredth the row allows. A tighter line alone does not leave the row: the prior's
    // variance scale floors the posterior spread, so the move stays near 0.005.
    let conjugate = SlopeRun { draws: 400, ..SLOPE };
    let outside = [
        SlopeRun { rows: 80, ..conjugate },
        SlopeRun { outcome: 2, ..conjugate },
        SlopeRun { weighted: true, ..conjugate },
        SlopeRun { prior_scale: 1.0, ..conjugate },
    ];
    let (names, body) = slope_row_round_trips(
        "supported_bayesian_static_association_posterior",
        conjugate,
        "conjugate_normal_inverse_gamma",
        &outside,
    );
    let record = body.association.as_ref().unwrap().posterior.as_ref().unwrap();
    assert_eq!(record.draws, 0, "the marginal is closed form");
    assert!((record.level - 0.95).abs() < 1e-12);
    let slope = &body.association.as_ref().unwrap().terms[1];
    let shift = (slope.posterior_mean.unwrap() - slope.estimate).abs();
    assert!(shift < 0.01 * slope.posterior_sd.unwrap(), "the prior leaves the slope: {shift}");
    let mut moved = body.clone();
    let forged = &mut moved.association.as_mut().unwrap().terms[1];
    forged.posterior_mean = Some(forged.estimate + 0.02 * forged.posterior_sd.unwrap());
    assert!(refused(&names, &moved), "a slope the prior moved has no row");
    let mut drawn = body;
    drawn.association.as_mut().unwrap().posterior.as_mut().unwrap().method =
        "bayesian_bootstrap".to_owned();
    assert!(refused(&names, &drawn), "the row bootstrap posterior of a table has no row");
}

#[test]
fn supported_bayesian_temporal_association_slope_round_trips_and_refuses_forgery() {
    let blocks = SlopeRun { temporal: true, draws: 400, ..SLOPE };
    let (names, body) = slope_row_round_trips(
        "supported_bayesian_temporal_association_posterior",
        blocks,
        "bayesian_bootstrap",
        &[SlopeRun { draws: 399, ..blocks }, SlopeRun { rows: 90, ..blocks }],
    );
    let record = body.association.as_ref().unwrap().posterior.as_ref().unwrap();
    assert_eq!(record.draws, 400);
    assert!((record.level - 0.95).abs() < 1e-12);
    let mut few = body;
    few.association.as_mut().unwrap().posterior.as_mut().unwrap().draws = 399;
    assert!(refused(&names, &few), "399 draws are below the row");
}
