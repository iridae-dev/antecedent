//! The sampling intervals of the binary-instrument bounds against their definitions.
//!
//! The estimator's resamples are inputs here: the test regenerates the row-bootstrap indexes
//! and the exponential row weights from the streams the estimator draws (pinned below by stream
//! index, so a changed stream is a visible failure). Everything after the draw is rebuilt
//! independently: each resample's bounds come from the closed-form Balke-Pearl expressions (not
//! the estimator's vertex search), and the interval from its definition: the replicate standard
//! deviations of the two endpoints and of the width, the `sqrt(ln n)` retention rule, the root
//! of `Phi(c + r) - Phi(-c) = level` by bisection on a series evaluation of the normal
//! distribution function, the endpoints `bound -+ c * sd` of the frequentist interval, and the
//! exchangeable-rank tail quantiles of the endpoint draws at probability `1 - Phi(c)` of the
//! posterior interval.
//!
//! Two laws: a partial-compliance mixture whose set is many standard deviations wide (the
//! one-sided limit of the critical value) and a law with 10% always-takers and 10% never-takers
//! whose set is a few standard deviations wide (a critical value above that limit). Both are
//! far enough inside the instrumental inequality that no resample is left out.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "row and replicate counts convert exactly at these sizes"
)]

use antecedent_core::{ExecutionContext, IvBoundsQuery, StreamDomain, VariableId, derive_stream};
use antecedent_data::{ResamplingPlan, TabularData, fill_resample_index_replicate};
use antecedent_estimate::{IvBoundsEstimator, IvBoundsIntervalMethod};
use antecedent_kernels::bayesian_bootstrap_weights;

/// The stream indices the estimator draws from.
const BOOTSTRAP_STREAM: u64 = 0x1B0D_5001;
const POSTERIOR_STREAM: u64 = 0x1B0D_5002;
const SEED: u64 = 31;
const LEVEL: f64 = 0.95;
const TOLERANCE: f64 = 1e-9;

/// Rows per `(z, y + 2 d)` cell of the two laws.
const MIXTURE: [[usize; 4]; 2] = [[153, 87, 18, 42], [63, 27, 48, 162]];
const NEAR_COMPLIANCE: [[usize; 4]; 2] = [[78, 57, 6, 9], [12, 3, 39, 96]];

/// One row per unit: the instrument arm and the cell `y + 2 d` of each row, in table order.
fn rows_of(counts: &[[usize; 4]; 2]) -> Vec<(usize, usize)> {
    let mut rows = Vec::new();
    for (arm, cells) in counts.iter().enumerate() {
        for (cell, &count) in cells.iter().enumerate() {
            rows.extend(std::iter::repeat_n((arm, cell), count));
        }
    }
    rows
}

fn table(rows: &[(usize, usize)]) -> TabularData {
    let column = |value: &dyn Fn(&(usize, usize)) -> usize| -> Vec<f64> {
        rows.iter().map(|row| value(row) as f64).collect()
    };
    let (z, d, y) = (column(&|row| row.0), column(&|row| row.1 / 2), column(&|row| row.1 % 2));
    TabularData::from_f64_columns([("z", &z[..]), ("d", &d[..]), ("y", &y[..])]).unwrap()
}

fn query() -> IvBoundsQuery {
    IvBoundsQuery::new(VariableId::from_raw(2), VariableId::from_raw(1), VariableId::from_raw(0))
}

/// Conditional law `P(y, d | z)` of the rows under the row masses `mass`.
fn law(rows: &[(usize, usize)], mass: &[f64]) -> [[f64; 4]; 2] {
    let mut cells = [[0.0; 4]; 2];
    for (&(arm, cell), weight) in rows.iter().zip(mass) {
        cells[arm][cell] += weight;
    }
    for arm in &mut cells {
        let total: f64 = arm.iter().sum();
        for cell in arm.iter_mut() {
            *cell /= total;
        }
    }
    cells
}

/// Lower Balke-Pearl bound of `E[Y(1) - Y(0)]`, in closed form.
fn closed_form_lower(cells: &[[f64; 4]; 2]) -> f64 {
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

/// Both bounds; the upper one is the negated lower bound of the law with the outcome
/// relabelled.
fn closed_form_bounds(cells: &[[f64; 4]; 2]) -> (f64, f64) {
    let mut flipped = [[0.0; 4]; 2];
    for z in 0..2 {
        for d in 0..2 {
            for y in 0..2 {
                flipped[z][(1 - y) + 2 * d] = cells[z][y + 2 * d];
            }
        }
    }
    (closed_form_lower(cells), -closed_form_lower(&flipped))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation with the `n - 1` divisor.
fn sd(values: &[f64]) -> f64 {
    let centre = mean(values);
    let squares: f64 = values.iter().map(|value| (value - centre).powi(2)).sum();
    (squares / (values.len() as f64 - 1.0)).sqrt()
}

/// Standard normal distribution function, from the all-positive series
/// `erf(x) = 2 / sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (2n+1)!!`.
fn normal_cdf(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    if x.abs() > 6.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let magnitude = x.abs();
    let (mut term, mut sum, mut n) = (magnitude, magnitude, 0.0_f64);
    while term > 1e-18 * sum {
        n += 1.0;
        term *= 2.0 * magnitude * magnitude / (2.0 * n + 1.0);
        sum += term;
    }
    let erf = 2.0 / std::f64::consts::PI.sqrt() * (-magnitude * magnitude).exp() * sum;
    0.5 * (1.0 + erf.copysign(x))
}

/// The root `c >= 0` of `Phi(c + r) - Phi(-c) = level`.
fn critical_value(r: f64) -> f64 {
    if normal_cdf(r) - 0.5 >= LEVEL {
        return 0.0;
    }
    let (mut low, mut high) = (0.0_f64, 10.0_f64);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if normal_cdf(middle + r) - normal_cdf(-middle) < LEVEL {
            low = middle;
        } else {
            high = middle;
        }
    }
    0.5 * (low + high)
}

/// Critical value of the points `[lower, upper]` with endpoint deviations `sds` and width
/// deviation `width_sd`, and whether the width was retained.
fn selection(points: (f64, f64), sds: (f64, f64), width_sd: f64, rows: usize) -> (f64, bool) {
    let width = points.1 - points.0;
    let threshold = (rows as f64).ln().sqrt().max(1.0);
    let retained = width > threshold * width_sd && width > 0.0;
    let scale = sds.0.max(sds.1);
    let standardized = if retained && scale > 0.0 { width / scale } else { 0.0 };
    (critical_value(standardized), retained)
}

/// Exchangeable-rank quantile: zero-based rank `clamp(p (D + 1), 1, D) - 1`, interpolated.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "a non-negative rank below the draw count; an exact zero fraction"
)]
fn rank_quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len() as f64;
    let rank = (p * (count + 1.0)).clamp(1.0, count) - 1.0;
    let below = rank.floor();
    let fraction = rank - below;
    let index = below as usize;
    if fraction == 0.0 {
        sorted[index]
    } else {
        sorted[index] + fraction * (sorted[index + 1] - sorted[index])
    }
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= TOLERANCE, "{what}: {got} vs {want}");
}

#[test]
fn the_row_bootstrap_interval_is_the_imbens_manski_construction_of_its_replicates() {
    for (name, counts) in [("mixture", MIXTURE), ("near compliance", NEAR_COMPLIANCE)] {
        let rows = rows_of(&counts);
        let n = rows.len();
        let replicates = 399_u32;
        let ctx = ExecutionContext::for_tests(SEED);
        let estimator = IvBoundsEstimator::new().with_bootstrap_replicates(replicates);
        let prepared = estimator.prepare(&table(&rows), &query()).unwrap();
        let estimate = estimator.estimate(&prepared, &ctx).unwrap();
        let interval = estimate.interval.expect("an interval was requested");
        assert_eq!(interval.method, IvBoundsIntervalMethod::ImbensManskiRowBootstrap, "{name}");
        assert_eq!((interval.replicates, interval.infeasible), (399, 0), "{name}");

        let points = closed_form_bounds(&law(&rows, &vec![1.0; n]));
        close(estimate.lower, points.0, &format!("{name} lower bound"));
        close(estimate.upper, points.1, &format!("{name} upper bound"));
        let (mut lows, mut highs) = (Vec::new(), Vec::new());
        let mut drawn = Vec::new();
        let streams = ExecutionContext::for_tests(SEED);
        for replicate in 0..u64::from(replicates) {
            fill_resample_index_replicate(
                ResamplingPlan::IidBootstrap,
                n,
                None,
                &streams,
                BOOTSTRAP_STREAM,
                replicate,
                &mut drawn,
            )
            .unwrap();
            let mut multiplicity = vec![0.0; n];
            for id in &drawn {
                multiplicity[usize::try_from(id.get()).unwrap()] += 1.0;
            }
            let (low, high) = closed_form_bounds(&law(&rows, &multiplicity));
            assert!(low <= high + 1e-12, "{name}: replicate {replicate} has no feasible law");
            lows.push(low);
            highs.push(high);
        }
        let widths: Vec<f64> = lows.iter().zip(&highs).map(|(low, high)| high - low).collect();
        let sds = (sd(&lows), sd(&highs));
        let (c, retained) = selection(points, sds, sd(&widths), n);
        let lower = (points.0 - c * sds.0).min(points.1 - c * sds.1);
        let upper = (points.0 + c * sds.0).max(points.1 + c * sds.1);
        assert_eq!(interval.width_retained, retained, "{name}");
        assert!((interval.level - LEVEL).abs() < 1e-15, "{name}");
        close(interval.critical_value, c, &format!("{name} critical value"));
        close(interval.lower, lower, &format!("{name} lower endpoint"));
        close(interval.upper, upper, &format!("{name} upper endpoint"));
        close(interval.lower_se, (points.0 - lower) / c, &format!("{name} lower error"));
        close(interval.upper_se, (upper - points.1) / c, &format!("{name} upper error"));
        assert!(retained, "{name}: both sets are wider than their sampling noise");
        // The wide set sits at the one-sided limit; the narrower one stays above it.
        if name == "mixture" {
            close(c, 1.644_853_626_951_472, "one-sided critical value");
        } else {
            assert!(c > 1.644_853_626_951_472 && c < 1.959_963_984_540_054, "{name}: {c}");
        }
    }
}

#[test]
fn the_endpoint_posterior_interval_is_the_tail_quantile_of_its_draws() {
    for (name, counts) in [("mixture", MIXTURE), ("near compliance", NEAR_COMPLIANCE)] {
        let rows = rows_of(&counts);
        let n = rows.len();
        let draws = 400_usize;
        let ctx = ExecutionContext::for_tests(SEED);
        let estimator = IvBoundsEstimator::new();
        let prepared = estimator.prepare(&table(&rows), &query()).unwrap();
        let estimate = estimator.estimate_bayesian(&prepared, draws, &ctx).unwrap();
        let interval = estimate.interval.expect("a posterior interval");
        let posterior = estimate.posterior.expect("a posterior summary");
        assert_eq!(interval.method, IvBoundsIntervalMethod::PosteriorEndpointQuantiles, "{name}");
        assert_eq!((interval.replicates, interval.infeasible), (draws, 0), "{name}");
        assert_eq!((posterior.n_draws, posterior.infeasible_draws), (draws, 0), "{name}");

        let (mut lows, mut highs) = (Vec::new(), Vec::new());
        let streams = ExecutionContext::for_tests(SEED);
        for draw in 0..draws {
            let mut rng = streams
                .rng
                .stream_for(StreamDomain::Estimate, derive_stream(POSTERIOR_STREAM, draw as u64));
            let weights = bayesian_bootstrap_weights(&mut rng, n, false);
            let (low, high) = closed_form_bounds(&law(&rows, &weights));
            assert!(low <= high + 1e-12, "{name}: draw {draw} has no feasible law");
            lows.push(low);
            highs.push(high);
        }
        let points = (mean(&lows), mean(&highs));
        let sds = (sd(&lows), sd(&highs));
        close(posterior.lower_mean, points.0, &format!("{name} posterior lower mean"));
        close(posterior.upper_mean, points.1, &format!("{name} posterior upper mean"));
        close(posterior.lower_sd, sds.0, &format!("{name} posterior lower deviation"));
        close(posterior.upper_sd, sds.1, &format!("{name} posterior upper deviation"));
        let widths: Vec<f64> = lows.iter().zip(&highs).map(|(low, high)| high - low).collect();
        let (c, retained) = selection(points, sds, sd(&widths), n);
        let tail = 1.0 - normal_cdf(c);
        let lower = rank_quantile(&lows, tail);
        let upper = rank_quantile(&highs, 1.0 - tail);
        assert_eq!(interval.width_retained, retained, "{name}");
        close(interval.critical_value, c, &format!("{name} critical value"));
        close(interval.lower, lower, &format!("{name} lower endpoint"));
        close(interval.upper, upper, &format!("{name} upper endpoint"));
        close(interval.lower_se, (points.0 - lower) / c, &format!("{name} lower error"));
        close(interval.upper_se, (upper - points.1) / c, &format!("{name} upper error"));
    }
}
