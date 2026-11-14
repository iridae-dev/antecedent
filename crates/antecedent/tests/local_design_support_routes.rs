//! Support rows of the local-ratio design routes through the public study facade: the fuzzy
//! discontinuity and regression kink at a selected or diagnosed bandwidth, the same designs in
//! calendar time, and their Bayesian counterparts.
//!
//! Each route's test runs a fit that meets its row and checks that the result states support,
//! that the statement survives the contracted artifact, that a forged statement (too few
//! window rows, a weaker first stage, too few replicates or draws, another interval
//! procedure) is refused by the loader, and that the same route below its row runs and states
//! no support. The kink in calendar time is also checked against the analytic standard
//! deviation of its estimator under the generated error law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "the artifact must carry every published number exactly; a route's supported fit, its forgeries and its unsupported neighbours read as one sequence"
)]

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, InferenceMode, PreparedStudy, Study, StudyResult};
use antecedent_core::{
    LocalBandwidthRule, LocalPolynomialRatioQuery, LocalRatioOptions, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::local_polynomial_ratio::fit_local_polynomial_ratio;
use antecedent_io::analysis_result_artifact::{AnalysisResultWire, QuasiRolloutWire};

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

/// Normal 97.5% quantile.
const Z: f64 = 1.959_963_984_540_054;
/// Code of the diagnostic by which a design posterior states its support row.
const POSTERIOR_ROW: &str = antecedent_io::graphless_support::DESIGN_POSTERIOR_ROW_CODE;
/// What the loader says of a rollout section whose support claim no row admits.
const NO_ROW: &str = "claims no support";

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        (antecedent_core::splitmix64(&mut self.0) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Row-aligned columns of one fixture.
struct Sample {
    running: Vec<f64>,
    treatment: Vec<f64>,
    outcome: Vec<f64>,
}

impl Sample {
    fn data(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("x", self.running.as_slice()),
            ("t", self.treatment.as_slice()),
            ("y", self.outcome.as_slice()),
        ])
        .unwrap()
    }
}

fn query(
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
    options: Option<LocalRatioOptions>,
) -> LocalPolynomialRatioQuery {
    LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(1),
        running: VariableId::from_raw(0),
        cutoff,
        bandwidth,
        kink,
        options,
    }
}

fn selected() -> Option<LocalRatioOptions> {
    Some(LocalRatioOptions {
        bandwidth_rule: LocalBandwidthRule::MseOptimal,
        ..LocalRatioOptions::default()
    })
}

fn in_time(replicates: u32) -> Option<LocalRatioOptions> {
    Some(LocalRatioOptions {
        temporal_replicates: Some(replicates),
        ..LocalRatioOptions::default()
    })
}

fn bayes(prior_scale: f64, draws: usize) -> Option<InferenceMode> {
    Some(InferenceMode::Bayesian(
        BayesianConfig::conjugate().prior_scale(prior_scale).n_draws(draws),
    ))
}

fn run(
    sample: &Sample,
    query: LocalPolynomialRatioQuery,
    inference: Option<InferenceMode>,
    seed: u64,
) -> (PreparedStudy, StudyResult, ExecutionContext, TabularData) {
    let data = sample.data();
    let ctx = ExecutionContext::for_tests(seed);
    let study = match inference {
        Some(mode) => Study::tabular(data.clone()).query(query).inference(mode).build(),
        None => Study::tabular(data.clone()).query(query).build(),
    }
    .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    (prepared, result, ctx, data)
}

/// The decoded contracted artifact of a result: its variable names and body.
fn artifact(
    prepared: &PreparedStudy,
    result: &StudyResult,
    ctx: &ExecutionContext,
    label: &str,
) -> (Vec<String>, AnalysisResultWire) {
    let bytes = prepared.encode_contracted_result(result, label, ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    (header.variable_names, body)
}

/// The loader's refusal of a forged body.
fn refusal(names: &[String], forged: &AnalysisResultWire) -> String {
    antecedent_io::encode_analysis_result_artifact(forged, names.to_vec(), "forged-support")
        .unwrap_err()
        .to_string()
}

fn section(body: &mut AnalysisResultWire) -> &mut QuasiRolloutWire {
    body.quasi_rollout.as_mut().expect("the rollout section is exported")
}

/// Overwrite one named diagnostic of a rollout section.
fn set_diagnostic(body: &mut AnalysisResultWire, name: &str, value: f64) {
    let entry = section(body)
        .diagnostics
        .iter_mut()
        .find(|(key, _)| key == name)
        .unwrap_or_else(|| panic!("the section reports {name}"));
    entry.1 = value;
}

fn supported(result: &StudyResult) -> bool {
    result.quasi_rollout().expect("a rollout payload").supported
}

// ------------------------------------------------------------------ fixtures

/// Independent rows on an even grid over `[-1, 1]`: a binary treatment received with
/// probability `0.1 + 0.02 x + 0.8 [x > 0]` and the outcome
/// `0.7 + 0.5 x + 0.3 x^2 + 0.1 x^3 + 2 t + noise`; the jump ratio is 2.
fn fuzzy_rows(seed: u64, n: usize) -> Sample {
    let mut rng = Rng(seed);
    let half = n as f64 / 2.0 + 0.5;
    let (mut running, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for i in 1..=n {
        let x = -1.0 + i as f64 / half;
        let probability = 0.1 + 0.02 * x + if x > 0.0 { 0.8 } else { 0.0 };
        let t = f64::from(rng.uniform() < probability);
        running.push(x);
        treatment.push(t);
        outcome.push(0.7 + 0.5 * x + 0.3 * x * x + 0.1 * x.powi(3) + 2.0 * t + rng.normal());
    }
    Sample { running, treatment, outcome }
}

/// Independent rows on the same grid with the kinked dose
/// `1 + 0.2 x + 3 max(x, 0) + 0.02 noise` and the outcome
/// `0.7 + 0.5 x + 0.3 x^2 + 0.1 x^3 + 2 dose + 0.5 noise`; the slope-change ratio is 2.
fn kink_rows(seed: u64, n: usize) -> Sample {
    let mut rng = Rng(seed);
    let half = n as f64 / 2.0 + 0.5;
    let (mut running, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for i in 1..=n {
        let x = -1.0 + i as f64 / half;
        let dose = 1.0 + 0.2 * x + 3.0 * x.max(0.0) + 0.02 * rng.normal();
        running.push(x);
        treatment.push(dose);
        outcome
            .push(0.7 + 0.5 * x + 0.3 * x * x + 0.1 * x.powi(3) + 2.0 * dose + 0.5 * rng.normal());
    }
    Sample { running, treatment, outcome }
}

/// Periods `1..=n` with a treatment that steps at the middle period (plus `dose_noise` times
/// a standard normal) and AR(1) errors (coefficient 0.7, innovation sd 0.5) about a linear
/// trend; the jump is 1. Returns the sample, the cutoff and the bandwidth `0.15 n`.
fn time_jump(seed: u64, n: usize, dose_noise: f64) -> (Sample, f64, f64) {
    let mut rng = Rng(seed);
    let cutoff = n as f64 / 2.0;
    let (mut running, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    let mut error = 0.0;
    for period in 1..=n {
        error = 0.7 * error + 0.5 * rng.normal();
        let t = period as f64;
        let mut d = f64::from(t >= cutoff);
        if dose_noise > 0.0 {
            d += dose_noise * rng.normal();
        }
        running.push(t);
        treatment.push(d);
        outcome.push(2.0 * t / n as f64 + d + error);
    }
    (Sample { running, treatment, outcome }, cutoff, 0.15 * n as f64)
}

/// Marginal standard deviation of the kink-in-time error.
const KINK_ERROR_SD: f64 = 0.5;

/// Periods `1..=n` with a dose that kinks at the middle period
/// (`1 + 0.5 s + 3 max(s, 0) + 0.02 noise`, `s` the time from the cutoff in bandwidths) and
/// the outcome `0.7 + 0.5 s + 0.3 s^2 + 0.1 s^3 + 2 dose` plus AR(1) errors (coefficient 0.7,
/// marginal sd 0.5); the slope-change ratio is 2.
fn time_kink(seed: u64, n: usize) -> (Sample, f64, f64) {
    let mut rng = Rng(seed);
    let cutoff = n as f64 / 2.0;
    let bandwidth = 0.15 * n as f64;
    let innovation = KINK_ERROR_SD * (1.0_f64 - 0.49).sqrt();
    let (mut running, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    let mut error = 0.0;
    for period in 1..=n {
        error = 0.7 * error + innovation * rng.normal();
        let t = period as f64;
        let s = (t - cutoff) / bandwidth;
        let dose = 1.0 + 0.5 * s + 3.0 * s.max(0.0) + 0.02 * rng.normal();
        running.push(t);
        treatment.push(dose);
        outcome.push(0.7 + 0.5 * s + 0.3 * s * s + 0.1 * s.powi(3) + 2.0 * dose + error);
    }
    (Sample { running, treatment, outcome }, cutoff, bandwidth)
}

/// Rows on an even grid over `[-1, 1]` drawn from the posterior's own Gaussian local-linear
/// model: a jump (`0.2 + 0.1 x + 0.6 [x >= 0] + 0.3 noise`, outcome `1 + 0.3 x + 2 dose +
/// 0.2 noise`) or a kink (`0.2 + 0.5 x + 0.6 max(x, 0) + 0.1 noise`, outcome `1 + 2 dose +
/// 0.2 noise`); the ratio is 2.
fn model_rows(seed: u64, n: usize, kink: bool) -> Sample {
    let mut rng = Rng(seed);
    let (mut running, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
        let (dose, trend) = if kink {
            (0.2 + 0.5 * x + 0.6 * x.max(0.0) + 0.1 * rng.normal(), 0.0)
        } else {
            (0.2 + 0.1 * x + if x >= 0.0 { 0.6 } else { 0.0 } + 0.3 * rng.normal(), 0.3 * x)
        };
        running.push(x);
        treatment.push(dose);
        outcome.push(1.0 + trend + 2.0 * dose + 0.2 * rng.normal());
    }
    Sample { running, treatment, outcome }
}

// ------------------------------------------------------- independent rows

/// A supported independent-row design: the section states support, its interval is the
/// normal interval of the robust bias-corrected fit at the reported bandwidth, and the
/// statement survives the artifact.
fn assert_supported_design(
    label: &str,
    sample: &Sample,
    query: LocalPolynomialRatioQuery,
) -> (Vec<String>, AnalysisResultWire) {
    let (prepared, result, ctx, _) = run(sample, query.clone(), None, 31);
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let rollout = result.quasi_rollout().unwrap();
    assert!(rollout.supported, "{label}");
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_ref() == "estimate.quasi.rollout.unsupported"),
        "{label}: a supported result carries no unsupported note"
    );
    let overall = rollout.overall().unwrap();
    let bandwidth = rollout.diagnostic("bandwidth").unwrap();
    assert!(bandwidth <= query.bandwidth, "{label}");
    let direct = fit_local_polynomial_ratio(
        &sample.running,
        &sample.outcome,
        &sample.treatment,
        query.cutoff,
        bandwidth,
        query.kink,
    )
    .unwrap();
    assert!((overall.estimate - direct.estimate).abs() < 1e-12, "{label}");
    let se = overall.standard_error.unwrap();
    assert!((se - direct.standard_error).abs() < 1e-12, "{label}");
    let [lower, upper] = overall.interval.unwrap();
    assert!((lower - (overall.estimate - Z * se)).abs() < 1e-9, "{label}");
    assert!((upper - (overall.estimate + Z * se)).abs() < 1e-9, "{label}");
    assert!((overall.estimate - 2.0).abs() < 4.0 * se, "{label}: {}", overall.estimate);
    let (names, body) = artifact(&prepared, &result, &ctx, label);
    let wire = body.quasi_rollout.as_ref().unwrap();
    assert!(wire.supported && wire.calibration == "not_measured", "{label}");
    assert_eq!(wire.effects[0].interval, overall.interval, "{label}");
    (names, body)
}

#[test]
fn supported_design_layer_local_ratio_round_trips_and_refuses_forgery() {
    // The fuzzy jump at the MSE-optimal bandwidth.
    let fuzzy = fuzzy_rows(41, 4_800);
    let (names, body) = assert_supported_design(
        "design-layer-fuzzy-mse",
        &fuzzy,
        query(0.0, 0.5, false, selected()),
    );
    let mut thin = body.clone();
    set_diagnostic(&mut thin, "rows_left", 100.0);
    assert!(refusal(&names, &thin).contains(NO_ROW));
    let mut few = body.clone();
    set_diagnostic(&mut few, "rows", 500.0);
    assert!(refusal(&names, &few).contains(NO_ROW));
    // A first stage four standard errors from zero is below the row's strength.
    let mut weak = body.clone();
    let first_stage = section(&mut weak).diagnostic("first_stage").unwrap();
    set_diagnostic(&mut weak, "first_stage_standard_error", first_stage.abs() / 4.0);
    assert!(refusal(&names, &weak).contains(NO_ROW));
    let mut other = body.clone();
    section(&mut other).uncertainty = "rbc_hc0_delta_normal_fixed_bandwidth".to_owned();
    assert!(refusal(&names, &other).contains(NO_ROW));
    // A section may state less than its row grants; it then loads as an unsupported result.
    let mut unclaimed = body;
    section(&mut unclaimed).supported = false;
    antecedent_io::encode_analysis_result_artifact(&unclaimed, names, "unclaimed-support").unwrap();

    // The kink at the MSE-optimal bandwidth and the fuzzy jump at a declared bandwidth with
    // the density test each have their own row.
    let kink = kink_rows(42, 4_800);
    assert_supported_design("design-layer-kink-mse", &kink, query(0.0, 1.0, true, selected()));
    let declared = Some(LocalRatioOptions { density_test: true, ..LocalRatioOptions::default() });
    let (_, body) = assert_supported_design(
        "design-layer-fuzzy-declared",
        &fuzzy_rows(43, 2_400),
        query(0.0, 0.5, false, declared.clone()),
    );
    assert_eq!(
        body.quasi_rollout.as_ref().unwrap().uncertainty,
        "rbc_hc0_delta_normal_fixed_bandwidth"
    );

    // The same routes outside their rows run and state no support: 560 rows are fewer than
    // the selected-bandwidth row was measured at, and the declared-bandwidth kink has no row.
    let (_, small, _, _) = run(&fuzzy_rows(44, 560), query(0.0, 0.5, false, selected()), None, 31);
    assert!(!supported(&small));
    assert!(small.quasi_rollout().unwrap().overall().unwrap().interval.is_some());
    let (_, unmeasured, _, _) = run(&kink, query(0.0, 1.0, true, declared), None, 31);
    assert!(!supported(&unmeasured));
    assert!(
        unmeasured
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_ref() == "estimate.quasi.rollout.unsupported")
    );
}

// ------------------------------------------------------------ calendar time

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let k = b.len();
    for col in 0..k {
        let pivot = (col..k).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..k {
            let f = a[row][col] / a[col][col];
            for c in col..k {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let tail: f64 = (row + 1..k).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// Signed row weights of the reference slope change on the window rows, left side then right
/// side in running order: the slope change of a column per unit of the running variable is
/// its weighted sum. A side's weight on a row is the slope loading `w (G^-1 x)_1 / h` of the
/// triangular-kernel quartic fit in the scaled distance, negative left of the cutoff.
fn reference_slope_weights(running: &[f64], cutoff: f64, h: f64) -> Vec<f64> {
    let basis = |d: f64| [1.0, d, d * d, d.powi(3), d.powi(4)];
    let mut weights = Vec::new();
    for right in [false, true] {
        let distances: Vec<f64> = running
            .iter()
            .map(|r| (r - cutoff) / h)
            .filter(|d| d.abs() < 1.0 && if right { *d >= 0.0 } else { *d < 0.0 })
            .collect();
        let mut gram = vec![vec![0.0; 5]; 5];
        for &d in &distances {
            let (x, w) = (basis(d), 1.0 - d.abs());
            for a in 0..5 {
                for b in 0..5 {
                    gram[a][b] += w * x[a] * x[b];
                }
            }
        }
        // The Gram matrix is symmetric, so this is the slope row of its inverse.
        let bread1 = solve(gram, vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        let sign = if right { 1.0 } else { -1.0 };
        weights.extend(distances.iter().map(|&d| {
            let x = basis(d);
            sign * (1.0 - d.abs()) * (0..5).map(|a| bread1[a] * x[a]).sum::<f64>() / h
        }));
    }
    weights
}

/// The frequentist section of a calendar-time fit: its standard error and interval.
fn time_interval(result: &StudyResult) -> (f64, f64, [f64; 2]) {
    let overall = result.quasi_rollout().unwrap().overall().unwrap();
    (overall.estimate, overall.standard_error.unwrap(), overall.interval.unwrap())
}

/// The interval of a kink in calendar time against the known error law, for the bootstrap
/// and for the block posterior.
///
/// Time is a fixed design, so the estimate is the reference slope weights applied to the
/// outcome over the same weights applied to the dose. The outcome is twice the dose plus a
/// cubic the quartic fit removes plus the error, so the estimate is 2 plus the weighted
/// error over the weighted dose, and the weighted dose is the slope change `3 / h` up to a
/// dose noise thirty times smaller. Under the generated AR(1) errors (coefficient 0.7,
/// marginal sd 0.5, one recursion through the cutoff) the standard deviation of the estimate
/// is therefore `sqrt(a' G a) / (3 / h)` with `G[i][j] = 0.25 * 0.7^|i - j|`.
#[test]
fn a_kink_in_time_interval_matches_the_analytic_serial_standard_error() {
    let (sample, cutoff, bandwidth) = time_kink(25, 4_000);
    let weights = reference_slope_weights(&sample.running, cutoff, bandwidth);
    let window: Vec<usize> = (0..sample.running.len())
        .filter(|&row| (sample.running[row] - cutoff).abs() < bandwidth)
        .collect();
    assert_eq!(weights.len(), window.len());
    assert_eq!(window.len(), 1_199);
    let weighted = |column: &[f64]| -> f64 {
        window.iter().zip(&weights).map(|(&row, weight)| weight * column[row]).sum()
    };
    let reference = weighted(&sample.outcome) / weighted(&sample.treatment);
    let mut serial_variance = 0.0;
    for (i, a) in weights.iter().enumerate() {
        for (j, b) in weights.iter().enumerate() {
            serial_variance +=
                a * b * KINK_ERROR_SD * KINK_ERROR_SD * 0.7_f64.powf(i.abs_diff(j) as f64);
        }
    }
    let serial_sd = serial_variance.sqrt() / (3.0 / bandwidth);

    let (_, result, _, _) = run(&sample, query(cutoff, bandwidth, true, in_time(199)), None, 5);
    let (estimate, se, [lower, upper]) = time_interval(&result);
    assert!(
        (estimate - reference).abs() < 1e-7 * (1.0 + reference.abs()),
        "{estimate} {reference}"
    );
    // The route estimates the standard deviation from an AR(1) model of the window residuals
    // and the interquartile range of 199 replicates. The coefficient's sampling sd (about
    // 0.03 on 1,199 rows) moves the long-run sd by 1 / (1 - 0.7) per unit (10%), the
    // innovation variance adds 2% and the interquartile range of 199 replicates 8%: 13%
    // together, so a factor of 1.6 either way is more than three standard deviations.
    assert!(
        (se / serial_sd).ln().abs() < 1.6_f64.ln(),
        "block bootstrap SE {se} vs the analytic AR(1) standard deviation {serial_sd}"
    );
    // The interval is the estimate plus or minus a Student-t quantile times that standard
    // error: symmetric, and wider than the normal interval by a reference with at least ten
    // degrees of freedom (quantile 2.23).
    assert!(((upper - estimate) - (estimate - lower)).abs() < 1e-9 * (1.0 + se));
    let critical = (upper - estimate) / se;
    assert!(critical > Z && critical < 2.23, "{critical}");
    assert!((estimate - 2.0).abs() < 4.0 * serial_sd, "{estimate}");

    let (_, posterior, _, _) =
        run(&sample, query(cutoff, bandwidth, true, in_time(199)), bayes(100.0, 500), 5);
    let rollout = posterior.quasi_rollout().unwrap();
    let posterior_sd = rollout.diagnostic("posterior_sd").unwrap();
    // The posterior redraws the same fitted error model with the coefficient's posterior
    // spread, so against the analytic sd it carries the same estimation error.
    assert!(
        (posterior_sd / serial_sd).ln().abs() < 1.6_f64.ln(),
        "posterior sd {posterior_sd} vs the analytic AR(1) standard deviation {serial_sd}"
    );
    let [low, high] = rollout.overall().unwrap().interval.unwrap();
    assert!(low < reference && reference < high);
    assert!(high - low > 2.0 * Z * serial_sd / 1.6 && high - low < 2.0 * Z * serial_sd * 1.6);
}

#[test]
fn supported_temporal_local_ratio_round_trips_and_refuses_forgery() {
    // A sharp jump in time at the measured replicate count.
    let (jump, cutoff, bandwidth) = time_jump(25, 2_000, 0.0);
    let (prepared, result, ctx, _) =
        run(&jump, query(cutoff, bandwidth, false, in_time(199)), None, 7);
    retained_graphless::assert_retained_graphless_program(
        "temporal-jump-supported",
        &prepared,
        &result,
        &ctx,
    );
    assert!(supported(&result));
    let (estimate, se, [lower, upper]) = time_interval(&result);
    assert!((estimate - 1.0).abs() < 4.0 * se, "{estimate} {se}");
    assert!(lower < estimate && estimate < upper);
    let rollout = result.quasi_rollout().unwrap();
    assert_eq!(
        (rollout.diagnostic("rows_left"), rollout.diagnostic("rows_right")),
        (Some(299.0), Some(300.0))
    );
    assert_eq!(rollout.replicates_ok, 199);
    let (names, body) = artifact(&prepared, &result, &ctx, "temporal-jump-supported");
    assert!(body.quasi_rollout.as_ref().unwrap().supported);
    assert_eq!(body.quasi_rollout.as_ref().unwrap().effects[0].interval, Some([lower, upper]));
    let mut thin = body.clone();
    set_diagnostic(&mut thin, "rows_left", 120.0);
    assert!(refusal(&names, &thin).contains(NO_ROW));
    let mut few = body.clone();
    section(&mut few).replicates_requested = 150;
    section(&mut few).replicates_ok = 150;
    assert!(refusal(&names, &few).contains(NO_ROW));
    // The jump's row is a sharp first stage: one with a standard error is outside it.
    let mut fuzzy = body.clone();
    set_diagnostic(&mut fuzzy, "first_stage_standard_error", 0.01);
    assert!(refusal(&names, &fuzzy).contains(NO_ROW));
    let mut unclaimed = body;
    section(&mut unclaimed).supported = false;
    antecedent_io::encode_analysis_result_artifact(&unclaimed, names, "unclaimed-support").unwrap();

    // A kink in time with a first stage far from zero.
    let (kink, cutoff, bandwidth) = time_kink(25, 4_000);
    let (prepared, result, ctx, _) =
        run(&kink, query(cutoff, bandwidth, true, in_time(199)), None, 7);
    retained_graphless::assert_retained_graphless_program(
        "temporal-kink-supported",
        &prepared,
        &result,
        &ctx,
    );
    assert!(supported(&result));
    let (names, body) = artifact(&prepared, &result, &ctx, "temporal-kink-supported");
    let mut weak = body;
    let first_stage = section(&mut weak).diagnostic("first_stage").unwrap();
    set_diagnostic(&mut weak, "first_stage_standard_error", first_stage.abs() / 10.0);
    assert!(refusal(&names, &weak).contains(NO_ROW));

    // Outside the rows the route runs and states no support: fewer replicates, a jump whose
    // first stage has error, and a window below the measured rows.
    let (jump, cutoff, bandwidth) = time_jump(25, 2_000, 0.0);
    let (_, fewer, _, _) = run(&jump, query(cutoff, bandwidth, false, in_time(99)), None, 7);
    assert!(!supported(&fewer));
    let (noisy, cutoff, bandwidth) = time_jump(25, 2_000, 0.05);
    let (_, fuzzy, _, _) = run(&noisy, query(cutoff, bandwidth, false, in_time(199)), None, 7);
    assert!(!supported(&fuzzy));
    assert!(fuzzy.quasi_rollout().unwrap().overall().unwrap().interval.is_some());
    let (short, cutoff, bandwidth) = time_jump(25, 600, 0.0);
    let (_, thin, _, _) = run(&short, query(cutoff, bandwidth, false, in_time(199)), None, 7);
    assert!(!supported(&thin));
}

#[test]
fn supported_bayesian_temporal_local_ratio_round_trips_and_refuses_forgery() {
    for kink in [false, true] {
        let label = if kink { "bayesian-temporal-kink" } else { "bayesian-temporal-jump" };
        let (sample, cutoff, bandwidth) =
            if kink { time_kink(25, 4_000) } else { time_jump(25, 2_000, 0.0) };
        let truth = if kink { 2.0 } else { 1.0 };
        let (prepared, result, ctx, _) =
            run(&sample, query(cutoff, bandwidth, kink, in_time(199)), bayes(100.0, 500), 9);
        retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
        assert!(supported(&result), "{label}");
        assert!(
            !result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code.as_ref() == "estimate.quasi.bayesian.unsupported"),
            "{label}"
        );
        let rollout = result.quasi_rollout().unwrap();
        assert_eq!(rollout.replicates_ok, 500, "{label}");
        let overall = rollout.overall().unwrap();
        let [lower, upper] = overall.interval.unwrap();
        let posterior_sd = rollout.diagnostic("posterior_sd").unwrap();
        assert!((overall.estimate - truth).abs() < 4.0 * posterior_sd, "{label}");
        assert!(lower < overall.estimate && overall.estimate < upper, "{label}");
        let (names, body) = artifact(&prepared, &result, &ctx, label);
        assert!(body.posterior_artifact.is_some(), "{label}");
        assert!(body.quasi_rollout.as_ref().unwrap().supported, "{label}");
        let mut few = body.clone();
        section(&mut few).replicates_requested = 400;
        section(&mut few).replicates_ok = 400;
        assert!(refusal(&names, &few).contains(NO_ROW), "{label}");
        let mut thin = body.clone();
        set_diagnostic(&mut thin, "rows_right", 100.0);
        assert!(refusal(&names, &thin).contains(NO_ROW), "{label}");
        let mut unclaimed = body;
        section(&mut unclaimed).supported = false;
        antecedent_io::encode_analysis_result_artifact(&unclaimed, names, "unclaimed-support")
            .unwrap();

        // 400 draws are fewer than the row was measured at.
        let (_, fewer, _, _) =
            run(&sample, query(cutoff, bandwidth, kink, in_time(199)), bayes(100.0, 400), 9);
        assert!(!supported(&fewer), "{label}");
    }
}

// ------------------------------------------------- fixed-bandwidth posterior

/// The fields of the diagnostic by which a design posterior states its support row.
fn stated_row(result: &StudyResult) -> Option<Vec<(String, String)>> {
    result.diagnostics.iter().find(|diagnostic| diagnostic.code.as_ref() == POSTERIOR_ROW).map(
        |diagnostic| {
            diagnostic
                .fields
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect()
        },
    )
}

fn field<'a>(fields: &'a [(String, String)], name: &str) -> &'a str {
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map_or_else(|| panic!("the row statement carries {name}"), |(_, value)| value.as_str())
}

/// A fixed-bandwidth posterior on rows drawn from its own model: it states its row, the
/// statement survives the artifact, a forged statement is refused, and a posterior outside
/// the row (another prior scale, fewer draws, a selected bandwidth, a thinner window) states
/// none.
fn assert_supported_posterior(label: &str, kink: bool, design: &str) {
    let sample = model_rows(if kink { 52 } else { 51 }, 800, kink);
    let (prepared, result, ctx, _) =
        run(&sample, query(0.0, 0.8, kink, None), bayes(100.0, 2_000), 13);
    retained_graphless::assert_retained_graphless_program(label, &prepared, &result, &ctx);
    let fields = stated_row(&result).unwrap_or_else(|| panic!("{label}: no row stated"));
    assert_eq!(field(&fields, "family"), "local_polynomial_ratio", "{label}");
    assert_eq!(field(&fields, "design"), design, "{label}");
    assert_eq!(
        field(&fields, "method"),
        "joint_local_linear_gaussian_posterior_prior_scale_100",
        "{label}"
    );
    assert_eq!((field(&fields, "control"), field(&fields, "treated")), ("320", "320"), "{label}");
    assert_eq!(field(&fields, "interval_resamples"), "2000", "{label}");
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_ref() == "estimate.quasi.bayesian.unsupported"),
        "{label}"
    );
    let posterior = result.posterior.as_ref().unwrap();
    let column = posterior.effect_column().unwrap();
    let (lower, upper) = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
    assert!(lower < upper && upper - lower < 1.5, "{label}: {lower} {upper}");
    assert!((result.effect() - 2.0).abs() < 2.0 * (upper - lower), "{label}: {}", result.effect());

    let (names, body) = artifact(&prepared, &result, &ctx, label);
    assert!(body.posterior_artifact.is_some(), "{label}");
    let stated = |body: &AnalysisResultWire| {
        body.diagnostics.iter().position(|diagnostic| diagnostic.code == POSTERIOR_ROW)
    };
    let index = stated(&body).unwrap_or_else(|| panic!("{label}: the artifact states no row"));
    let forge = |name: &str, value: &str| {
        let mut forged = body.clone();
        let entry = forged.diagnostics[index]
            .fields
            .iter_mut()
            .find(|(key, _)| key == name)
            .unwrap_or_else(|| panic!("{label}: no {name}"));
        entry.1 = value.to_owned();
        refusal(&names, &forged)
    };
    for (name, value) in [
        ("control", "40"),
        ("interval_resamples", "1000"),
        ("first_stage_t", "3.0"),
        ("method", "another_method"),
    ] {
        assert!(
            forge(name, value).contains("does not admit"),
            "{label}: a forged {name} must be refused"
        );
    }

    // Outside the row the route runs, publishes its interval and states nothing.
    for (what, sample, query, inference) in [
        ("another prior scale", &sample, query(0.0, 0.8, kink, None), bayes(10.0, 2_000)),
        ("fewer draws", &sample, query(0.0, 0.8, kink, None), bayes(100.0, 1_000)),
        ("a selected bandwidth", &sample, query(0.0, 0.8, kink, selected()), bayes(100.0, 2_000)),
        (
            "a thinner window",
            &model_rows(53, 160, kink),
            query(0.0, 0.8, kink, None),
            bayes(100.0, 2_000),
        ),
    ] {
        // A refusal (a selected window too thin for the posterior) states nothing either.
        let ctx = ExecutionContext::for_tests(13);
        let Ok(outside) = Study::tabular(sample.data())
            .query(query)
            .inference(inference.expect("a Bayesian mode"))
            .build()
            .and_then(|study| study.run(&ctx))
        else {
            assert_eq!(what, "a selected bandwidth", "{label}: only the selection may refuse");
            continue;
        };
        assert!(stated_row(&outside).is_none(), "{label}: {what} states no row");
        assert!(
            outside
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code.as_ref() == "estimate.quasi.bayesian.unsupported"),
            "{label}: {what}"
        );
    }
}

#[test]
fn supported_bayesian_fuzzy_rd_posterior_round_trips_and_refuses_forgery() {
    assert_supported_posterior("bayesian-fuzzy-rd-supported", false, "fuzzy_jump");
}

#[test]
fn supported_bayesian_regression_kink_posterior_round_trips_and_refuses_forgery() {
    assert_supported_posterior("bayesian-regression-kink-supported", true, "regression_kink");
}
