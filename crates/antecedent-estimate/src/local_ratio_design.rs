//! Design layer of the local-polynomial fuzzy discontinuity and regression kink: a
//! data-driven bandwidth, manipulation and covariate-balance diagnostics, and a
//! dependence-preserving interval when the running variable is calendar time.
//!
//! **Bandwidth.** The MSE-optimal plug-in bandwidth of the local quadratic jump (or slope
//! change) of the linearized outcome `Y − τ̃·T`, with `τ̃` the ratio at the declared pilot
//! bandwidth, in the style of Imbens–Kalyanaraman and Calonico–Cattaneo–Titiunik: the
//! triangular-kernel boundary constants are computed exactly, the residual variance on each
//! side comes from a local linear pilot fit, the density at the cutoff from a pilot window,
//! and the third derivative on each side from a global quartic. The selected bandwidth is
//! capped at the declared bandwidth, which acts as the largest admissible window. Because the
//! reported interval is robust bias-corrected, it stays valid at the MSE-optimal bandwidth,
//! which is why the selection is paired with it. The plug-in is not the full rdrobust
//! selector (no iterated pilot or regularization term).
//!
//! **Diagnostics.** A `McCrary` binned-density test of a discontinuity in the running
//! variable's density at the cutoff (a manipulation check), and the robust bias-corrected
//! jump of each declared baseline covariate at the cutoff (a balance check). Neither is part
//! of identification: a clean density or balance check does not establish continuity of the
//! potential outcomes.
//!
//! **Calendar-time running variable.** A regression discontinuity in time has serially
//! dependent rows, so the independent-row HC0 variance does not apply. Time is not sampled,
//! so the design is held fixed and the errors are redrawn: the reported ratio is linear in the
//! outcome and the treatment at a fixed design, and the errors of the structural series
//! `Y − τ̂·T` and of the treatment about a cubic trend on each side are prewhitened by an
//! AR(1) fit, as in a prewhitened long-run variance (Andrews and Monahan). The trend absorbs
//! the low-frequency part of a persistent error, so the AR(1) coefficient and the error
//! variance are the ones whose expected trend residuals match the observed residuals. The
//! interval is a moving-block bootstrap of the innovations within each side of the cutoff
//! (blocks of consecutive rows in time order, length the rounded cube root of the side's
//! window rows), recolored through the cutoff, refitting the whole robust bias-corrected
//! local ratio per replicate; the interval is the estimate plus or minus a Student-t quantile
//! times the interquartile bootstrap standard error, with Satterthwaite degrees of freedom
//! for the estimated error model and the finite number of replicates.
//!
//! Resampling whole rows in blocks is not used: it moves the time points with the outcomes,
//! so a replicate can lose the rows nearest the cutoff, and its spread then reflects the
//! redrawn design rather than the dependence of the errors.
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
#[path = "local_ratio_design/tests.rs"]
mod tests;

use antecedent_core::{CausalRng, ExecutionContext};

use crate::EstimationError;
use crate::local_polynomial_ratio::{LocalPolynomialRatioFit, fit_local_polynomial_ratio};
use crate::rollout_resampling::{
    ROLLOUT_LEVEL, ReplicateQuantity, UnitWeighting, cube_root_block_length, resample_draws,
    summarize_replicates, uniform_index, unit_weights,
};

/// Semantic operation tag of the temporal block bootstrap.
const BLOCK_OPERATION: u64 = 0x4c52_5442_0001;

/// Order of the local polynomial whose MSE the bandwidth minimizes.
const ORDER: usize = 2;

/// Fewest rows on each side the selected window must keep.
const MIN_SIDE_ROWS: usize = 8;

/// The row-aligned columns of a local ratio.
#[derive(Clone, Copy, Debug)]
pub struct LocalRatioColumns<'a> {
    /// Running variable.
    pub running: &'a [f64],
    /// Outcome.
    pub outcome: &'a [f64],
    /// Treatment receipt or dose.
    pub treatment: &'a [f64],
}

/// What the design layer runs around the local ratio.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalRatioDesign {
    /// Select the MSE-optimal plug-in bandwidth, capped at the declared one.
    pub select_bandwidth: bool,
    /// The running variable is calendar time: replicates of the within-side block bootstrap.
    pub temporal_replicates: Option<u32>,
    /// Run the `McCrary` density test at the cutoff.
    pub density_test: bool,
}

/// The bandwidth the fit used and how it was chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandwidthChoice {
    /// Bandwidth of the reported fit.
    pub bandwidth: f64,
    /// The plug-in MSE-optimal bandwidth before the cap, when selection ran.
    pub mse_optimal: Option<f64>,
    /// Whether the declared bandwidth capped the selection.
    pub capped: bool,
}

/// `McCrary` density discontinuity test at the cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityTest {
    /// `ln f̂₊ − ln f̂₋`.
    pub log_difference: f64,
    /// Its standard error.
    pub standard_error: f64,
    /// Two-sided normal p-value; a diagnostic, not support.
    pub p_value: f64,
    /// Bin width.
    pub bin_width: f64,
}

/// Robust bias-corrected jump of one baseline covariate at the cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CovariateBalance {
    /// Position of the covariate in the declared list.
    pub index: usize,
    /// Bias-corrected jump (or slope change).
    pub jump: f64,
    /// Its HC0 standard error.
    pub standard_error: f64,
    /// `jump / standard_error`.
    pub z: f64,
}

/// Block bootstrap record of a calendar-time running variable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalBlockRecord {
    /// Rows per block.
    pub block_length: usize,
    /// Requested replicates.
    pub requested: u32,
    /// Replicates whose refit produced a finite ratio.
    pub replicates_ok: u32,
    /// Replicates without one.
    pub replicates_failed: u32,
    /// The bootstrap summary of the ratio.
    pub ratio: ReplicateQuantity,
}

/// The local ratio with its design layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalRatioDesignFit {
    /// The robust bias-corrected local ratio at the chosen bandwidth.
    pub fit: LocalPolynomialRatioFit,
    /// Bandwidth choice.
    pub bandwidth: BandwidthChoice,
    /// Density test, when requested and estimable.
    pub density: Option<DensityTest>,
    /// Covariate balance, one per estimable declared covariate.
    pub balance: Vec<CovariateBalance>,
    /// Block bootstrap of a calendar-time running variable, when requested.
    pub temporal: Option<TemporalBlockRecord>,
}

fn refused(message: impl Into<String>) -> EstimationError {
    EstimationError::refused("option_not_applicable", message)
}

/// Fit the local ratio with the declared design layer.
///
/// # Errors
///
/// The local ratio fails at the pilot or selected bandwidth (support, weak first stage), the
/// selection leaves fewer than eight rows on a side, the block bootstrap is refused
/// (cancellation, admission), or the bootstrap yields no standard error.
pub fn fit_local_ratio_design(
    columns: &LocalRatioColumns<'_>,
    covariates: &[&[f64]],
    cutoff: f64,
    declared_bandwidth: f64,
    kink: bool,
    design: LocalRatioDesign,
    ctx: &ExecutionContext,
) -> Result<LocalRatioDesignFit, EstimationError> {
    let fit_at = |bandwidth: f64| {
        fit_local_polynomial_ratio(
            columns.running,
            columns.outcome,
            columns.treatment,
            cutoff,
            bandwidth,
            kink,
        )
        .map_err(|message| {
            refused(format!(
                "{message}; next attempt: a declared fixed bandwidth with more local support"
            ))
        })
    };
    let bandwidth = if design.select_bandwidth {
        let pilot = fit_at(declared_bandwidth)?;
        let linearized: Vec<f64> = columns
            .outcome
            .iter()
            .zip(columns.treatment)
            .map(|(y, t)| y - pilot.estimate * t)
            .collect();
        let optimal = mse_optimal_bandwidth(
            columns.running,
            &linearized,
            cutoff,
            usize::from(kink),
        )
        .ok_or_else(|| {
            refused(
                "the plug-in bandwidth is not estimable (too few rows near the cutoff or no \
                     curvature on either side); next attempt: a declared fixed bandwidth",
            )
        })?;
        let chosen = optimal.min(declared_bandwidth);
        let (left, right) = side_counts(columns.running, cutoff, chosen);
        if left < MIN_SIDE_ROWS || right < MIN_SIDE_ROWS {
            return Err(refused(format!(
                "the selected bandwidth {chosen} keeps {left} rows left and {right} right of the \
                 cutoff, fewer than {MIN_SIDE_ROWS}; next attempt: a declared fixed bandwidth"
            )));
        }
        BandwidthChoice {
            bandwidth: chosen,
            mse_optimal: optimal.is_finite().then_some(optimal),
            capped: optimal > declared_bandwidth,
        }
    } else {
        BandwidthChoice { bandwidth: declared_bandwidth, mse_optimal: None, capped: false }
    };
    let fit = fit_at(bandwidth.bandwidth)?;
    let density = design
        .density_test
        .then(|| density_test(columns.running, cutoff, bandwidth.bandwidth))
        .flatten();
    let balance = covariates
        .iter()
        .enumerate()
        .filter_map(|(index, covariate)| {
            let balance = fit_local_polynomial_ratio(
                columns.running,
                covariate,
                columns.treatment,
                cutoff,
                bandwidth.bandwidth,
                kink,
            )
            .ok()?;
            let se = balance.reduced_form_standard_error;
            (se > 0.0 && se.is_finite()).then(|| CovariateBalance {
                index,
                jump: balance.reduced_form,
                standard_error: se,
                z: balance.reduced_form / se,
            })
        })
        .collect();
    let temporal = design
        .temporal_replicates
        .map(|replicates| {
            temporal_block_bootstrap(
                columns,
                cutoff,
                bandwidth.bandwidth,
                kink,
                &fit,
                replicates,
                ctx,
            )
        })
        .transpose()?;
    Ok(LocalRatioDesignFit { fit, bandwidth, density, balance, temporal })
}

fn side_counts(running: &[f64], cutoff: f64, bandwidth: f64) -> (usize, usize) {
    let left = running.iter().filter(|r| **r < cutoff && cutoff - **r < bandwidth).count();
    let right = running.iter().filter(|r| **r >= cutoff && **r - cutoff < bandwidth).count();
    (left, right)
}

/// `∫₀¹ (1 − u) uᵐ du`.
fn kernel_moment(m: usize) -> f64 {
    1.0 / ((m + 1) as f64 * (m + 2) as f64)
}

/// `∫₀¹ (1 − u)² uᵐ du`.
fn squared_kernel_moment(m: usize) -> f64 {
    2.0 / ((m + 1) as f64 * (m + 2) as f64 * (m + 3) as f64)
}

/// The triangular-kernel boundary bias and variance constants `(B_ν, V_ν)` of a local
/// polynomial of order [`ORDER`].
fn boundary_constants(nu: usize) -> Option<(f64, f64)> {
    let size = ORDER + 1;
    let gamma: Vec<Vec<f64>> =
        (0..size).map(|j| (0..size).map(|k| kernel_moment(j + k)).collect()).collect();
    let psi: Vec<Vec<f64>> =
        (0..size).map(|j| (0..size).map(|k| squared_kernel_moment(j + k)).collect()).collect();
    let lambda: Vec<f64> = (0..size).map(|j| kernel_moment(ORDER + 1 + j)).collect();
    let inverse = crate::util::invert_rows(&gamma)?;
    let bias: f64 = (0..size).map(|k| inverse[nu][k] * lambda[k]).sum();
    let mut variance = 0.0;
    for (j, psi_row) in psi.iter().enumerate() {
        for (k, psi_jk) in psi_row.iter().enumerate() {
            variance += inverse[nu][j] * psi_jk * inverse[k][nu];
        }
    }
    Some((bias, variance))
}

/// The plug-in MSE-optimal bandwidth of the jump (`nu = 0`) or slope change (`nu = 1`) of
/// `values` at `cutoff`; `None` when a pilot quantity is not estimable, `+∞` when neither side
/// shows measurable curvature.
#[must_use]
pub fn mse_optimal_bandwidth(
    running: &[f64],
    values: &[f64],
    cutoff: f64,
    nu: usize,
) -> Option<f64> {
    let n = running.len();
    if n < 4 * MIN_SIDE_ROWS || values.len() != n {
        return None;
    }
    let mean = running.iter().sum::<f64>() / n as f64;
    let sd = (running.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let pilot = 1.84 * sd * (n as f64).powf(-0.2);
    if !(pilot > 0.0 && pilot.is_finite()) {
        return None;
    }
    let (left, right) = side_counts(running, cutoff, pilot);
    if left < MIN_SIDE_ROWS || right < MIN_SIDE_ROWS {
        return None;
    }
    let density = (left + right) as f64 / (2.0 * n as f64 * pilot);
    let side_variance = |right_side: bool| -> Option<f64> {
        let rows: Vec<(Vec<f64>, f64, f64)> = running
            .iter()
            .zip(values)
            .filter(|(r, _)| (**r >= cutoff) == right_side && (**r - cutoff).abs() < pilot)
            .map(|(r, v)| (vec![1.0, r - cutoff], *v, 1.0))
            .collect();
        let beta = crate::heterogeneity_robust_did::weighted_least_squares(&rows)?;
        let residual: f64 =
            rows.iter().map(|(x, v, _)| (v - beta[0] - beta[1] * x[1]).powi(2)).sum();
        Some(residual / (rows.len() - 2) as f64)
    };
    let third_derivative = |right_side: bool| -> Option<f64> {
        let rows: Vec<(Vec<f64>, f64, f64)> = running
            .iter()
            .zip(values)
            .filter(|(r, _)| (**r >= cutoff) == right_side)
            .map(|(r, v)| {
                let x = r - cutoff;
                (vec![1.0, x, x * x, x.powi(3), x.powi(4)], *v, 1.0)
            })
            .collect();
        (rows.len() > 5)
            .then(|| crate::heterogeneity_robust_did::weighted_least_squares(&rows))
            .flatten()
            .map(|beta| 6.0 * beta[3])
    };
    let variance = side_variance(false)? + side_variance(true)?;
    let (b_nu, v_nu) = boundary_constants(nu)?;
    let sign = if (ORDER + 1 + nu).is_multiple_of(2) { 1.0 } else { -1.0 };
    let curvature = (third_derivative(true)? - sign * third_derivative(false)?) / 6.0;
    let bias = b_nu * curvature;
    let variance_constant = v_nu * variance / density;
    if !(variance_constant > 0.0 && variance_constant.is_finite() && bias.is_finite()) {
        return None;
    }
    if bias.abs() <= 1e-12 {
        // No measurable curvature: the MSE-optimal window is unbounded and the cap binds.
        return Some(f64::INFINITY);
    }
    let order = (2 * ORDER + 3) as f64;
    let ratio = (1.0 + 2.0 * nu as f64) * variance_constant
        / (2.0 * (ORDER + 1 - nu) as f64 * bias * bias * n as f64);
    let bandwidth = ratio.powf(1.0 / order);
    (bandwidth.is_finite() && bandwidth > 0.0).then_some(bandwidth)
}

/// `McCrary`'s binned-density test of a jump in the running variable's density at `cutoff`,
/// with local linear triangular smoothing of the bin heights within `bandwidth`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a bin index is the floor of a bounded finite ratio"
)]
#[must_use]
pub fn density_test(running: &[f64], cutoff: f64, bandwidth: f64) -> Option<DensityTest> {
    let n = running.len();
    if n < 4 * MIN_SIDE_ROWS || bandwidth.is_nan() || bandwidth <= 0.0 {
        return None;
    }
    let mean = running.iter().sum::<f64>() / n as f64;
    let sd = (running.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    let width = 2.0 * sd / (n as f64).sqrt();
    if !(width > 0.0 && width.is_finite()) {
        return None;
    }
    let mut bins: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    for r in running {
        let index = ((r - cutoff) / width).floor();
        if index.is_finite() && index.abs() < 1e12 {
            *bins.entry(index as i64).or_default() += 1;
        }
    }
    let side = |right_side: bool| -> Option<f64> {
        let rows: Vec<(Vec<f64>, f64, f64)> = bins
            .iter()
            .filter_map(|(index, count)| {
                let midpoint = (*index as f64 + 0.5) * width;
                let inside = (*index >= 0) == right_side && midpoint.abs() < bandwidth;
                inside.then(|| {
                    let height = *count as f64 / (n as f64 * width);
                    (vec![1.0, midpoint], height, 1.0 - midpoint.abs() / bandwidth)
                })
            })
            .collect();
        if rows.len() < 3 {
            return None;
        }
        let beta = crate::heterogeneity_robust_did::weighted_least_squares(&rows)?;
        (beta[0] > 0.0).then_some(beta[0])
    };
    let (left, right) = (side(false)?, side(true)?);
    let log_difference = right.ln() - left.ln();
    let standard_error =
        ((24.0 / 5.0) / (n as f64 * bandwidth) * (1.0 / right + 1.0 / left)).sqrt();
    let z = log_difference / standard_error;
    Some(DensityTest {
        log_difference,
        standard_error,
        p_value: crate::util::normal_two_sided_p(z),
        bin_width: width,
    })
}

// Retain time order within each side after filtering to the bandwidth. A block
// is consecutive in this retained side sequence, not necessarily in all source rows.
fn temporal_side_rows(running: &[f64], cutoff: f64, bandwidth: f64) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (i, &r) in running.iter().enumerate() {
        if (r - cutoff).abs() < bandwidth {
            if r < cutoff {
                left.push(i);
            } else {
                right.push(i);
            }
        }
    }
    (left, right)
}

/// Order of the per-side trend whose residuals are the error series.
const TREND_ORDER: usize = 3;

/// Largest prewhitening coefficient in absolute value.
const MAX_PREWHITENING: f64 = 0.97;

/// Step of the numerical derivatives in the prewhitening coefficient.
const COEFFICIENT_STEP: f64 = 0.01;

/// Relative variance of the interquartile range of one Gaussian replicate:
/// `0.25 / (φ(z₀.₇₅)² · (2 z₀.₇₅)²)`.
const IQR_RELATIVE_VARIANCE: f64 = 1.360_5;

/// A residual sum of squares below this fraction of the series' own sum of squares is an
/// exact trend fit: the series carries no error.
const EXACT_FIT_FRACTION: f64 = 1e-18;

/// Monomials `1, d, …, d^order`.
fn monomials(distance: f64, order: usize) -> Vec<f64> {
    let mut basis = Vec::with_capacity(order + 1);
    let mut power = 1.0;
    for _ in 0..=order {
        basis.push(power);
        power *= distance;
    }
    basis
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Inverse Gram matrix of the order-`order` monomials of `distances` under `weight`.
fn inverse_gram(
    distances: &[f64],
    order: usize,
    weight: impl Fn(f64) -> f64,
) -> Option<Vec<Vec<f64>>> {
    let size = order + 1;
    let mut gram = vec![vec![0.0; size]; size];
    for &distance in distances {
        let basis = monomials(distance, order);
        let row_weight = weight(distance);
        for (gram_row, left) in gram.iter_mut().zip(&basis) {
            for (cell, right) in gram_row.iter_mut().zip(&basis) {
                *cell += row_weight * left * right;
            }
        }
    }
    crate::util::invert_rows(&gram)
}

/// The weight each row of one side carries in the reported bias-corrected jump (or slope
/// change) of a column: the estimator is linear in the column at a fixed design. `distances`
/// are the side's rows in bandwidths from the cutoff.
fn estimator_weights(distances: &[f64], bandwidth: f64, kink: bool) -> Option<Vec<f64>> {
    let triangular = |distance: f64| 1.0 - distance.abs();
    if kink {
        let quartic = inverse_gram(distances, 4, triangular)?;
        return Some(
            distances
                .iter()
                .map(|&d| triangular(d) * dot(&quartic[1], &monomials(d, 4)) / bandwidth)
                .collect(),
        );
    }
    let quadratic = inverse_gram(distances, 2, triangular)?;
    let cubic = inverse_gram(distances, 3, triangular)?;
    let mut cross = [0.0; 3];
    for &d in distances {
        for (value, basis) in cross.iter_mut().zip(monomials(d, 2)) {
            *value += triangular(d) * basis * d.powi(3);
        }
    }
    let projection = dot(&quadratic[0], &cross);
    Some(
        distances
            .iter()
            .map(|&d| {
                triangular(d)
                    * (dot(&quadratic[0], &monomials(d, 2))
                        - projection * dot(&cubic[3], &monomials(d, 3)))
            })
            .collect(),
    )
}

/// The unweighted polynomial trend of one side: its design rows and inverse Gram matrix.
struct SideTrend {
    basis: Vec<Vec<f64>>,
    inverse: Vec<Vec<f64>>,
}

impl SideTrend {
    /// Least-squares residuals of `values` about the trend.
    fn residuals(&self, values: &[f64]) -> Vec<f64> {
        let mut moment = vec![0.0; self.inverse.len()];
        for (row, value) in self.basis.iter().zip(values) {
            for (cell, basis) in moment.iter_mut().zip(row) {
                *cell += basis * value;
            }
        }
        let beta: Vec<f64> = self.inverse.iter().map(|row| dot(row, &moment)).collect();
        self.basis.iter().zip(values).map(|(row, value)| value - dot(row, &beta)).collect()
    }

    /// Expected lag-one cross product and sum of squares of the trend residuals of a
    /// unit-variance AR(1) series with coefficient `rho`: the sums of `(MΓM)[i][i+1]` and of
    /// `(MΓM)[i][i]`, with `M` the trend's residual maker and `Γ[i][j] = rho^|i−j|`.
    #[allow(
        clippy::needless_range_loop,
        reason = "the index pairs each row with its successor across aligned row sets"
    )]
    fn residual_moments(&self, rho: f64) -> (f64, f64) {
        let n = self.basis.len();
        let size = self.inverse.len();
        // Γ·X row by row, from a forward and a backward geometric sum.
        let mut smoothed = vec![vec![0.0; size]; n];
        let mut carry = vec![0.0; size];
        for (row, target) in self.basis.iter().zip(smoothed.iter_mut()) {
            for k in 0..size {
                carry[k] = row[k] + rho * carry[k];
                target[k] = carry[k];
            }
        }
        carry.fill(0.0);
        for (row, target) in self.basis.iter().zip(smoothed.iter_mut()).rev() {
            for k in 0..size {
                carry[k] = row[k] + rho * carry[k];
                target[k] += carry[k] - row[k];
            }
        }
        // X·(XᵀX)⁻¹ and XᵀΓX.
        let projected: Vec<Vec<f64>> = self
            .basis
            .iter()
            .map(|row| self.inverse.iter().map(|inverse_row| dot(inverse_row, row)).collect())
            .collect();
        let mut quadratic = vec![vec![0.0; size]; size];
        for (row, smooth) in self.basis.iter().zip(&smoothed) {
            for (quadratic_row, basis) in quadratic.iter_mut().zip(row) {
                for (cell, value) in quadratic_row.iter_mut().zip(smooth) {
                    *cell += basis * value;
                }
            }
        }
        let through: Vec<Vec<f64>> = projected
            .iter()
            .map(|row| {
                (0..size)
                    .map(|q| (0..size).map(|p| row[p] * quadratic[p][q]).sum::<f64>())
                    .collect()
            })
            .collect();
        let mut cross = 0.0;
        let mut square = 0.0;
        for i in 0..n {
            square +=
                1.0 - 2.0 * dot(&projected[i], &smoothed[i]) + dot(&through[i], &projected[i]);
            if i + 1 < n {
                cross += rho
                    - dot(&projected[i], &smoothed[i + 1])
                    - dot(&smoothed[i], &projected[i + 1])
                    + dot(&through[i], &projected[i + 1]);
            }
        }
        (cross, square)
    }
}

/// The AR(1) model of one error series over the window.
struct ErrorSeries {
    /// Centered innovations in time order, left side then right side, on the scale of the
    /// fitted model.
    innovations: Vec<f64>,
    /// AR(1) coefficient, corrected for the trend fit.
    coefficient: f64,
    /// Standard deviation of the coefficient.
    coefficient_sd: f64,
}

/// AR(1) model of the trend residuals of one series, `values` holding each side's rows.
///
/// The residuals of a fitted trend understate both the variance and the autocorrelation of a
/// persistent error, because the trend absorbs its low-frequency part. The coefficient is
/// therefore the one whose expected residual lag-one ratio equals the observed ratio, and the
/// marginal variance the residual sum of squares over its expectation at that coefficient
/// (both expectations exact for the trend design). The innovations are whitened with the
/// uncorrected ratio, which leaves them empirically white, and rescaled to the corrected
/// model. `None` when the trend reproduces the series exactly.
fn error_series(trends: &[SideTrend; 2], values: [&[f64]; 2]) -> Option<ErrorSeries> {
    let residuals = [trends[0].residuals(values[0]), trends[1].residuals(values[1])];
    let energy: f64 = values.iter().flat_map(|side| side.iter()).map(|v| v * v).sum();
    let square: f64 = residuals.iter().flatten().map(|r| r * r).sum();
    if square <= EXACT_FIT_FRACTION * energy {
        return None;
    }
    let cross: f64 = residuals
        .iter()
        .map(|side| side.windows(2).map(|pair| pair[0] * pair[1]).sum::<f64>())
        .sum();
    let observed = cross / square;
    let expected = |rho: f64| {
        let (left, right) = (trends[0].residual_moments(rho), trends[1].residual_moments(rho));
        ((left.0 + right.0) / (left.1 + right.1), left.1 + right.1)
    };
    let coefficient = if expected(MAX_PREWHITENING).0 <= observed {
        MAX_PREWHITENING
    } else if expected(-MAX_PREWHITENING).0 >= observed {
        -MAX_PREWHITENING
    } else {
        let (mut lower, mut upper) = (-MAX_PREWHITENING, MAX_PREWHITENING);
        for _ in 0..48 {
            let middle = 0.5 * (lower + upper);
            if expected(middle).0 < observed {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        0.5 * (lower + upper)
    };
    let marginal_variance = square / expected(coefficient).1;
    // Delta method through the correction: the observed ratio's large-sample variance over
    // the squared slope of its expectation in the coefficient.
    let below = (coefficient - COEFFICIENT_STEP).max(-MAX_PREWHITENING);
    let above = (coefficient + COEFFICIENT_STEP).min(MAX_PREWHITENING);
    let slope = (expected(above).0 - expected(below).0) / (above - below);
    let pairs = residuals.iter().map(|side| side.len().saturating_sub(1)).sum::<usize>();
    let ratio_sd = ((1.0 - observed * observed).max(0.0) / pairs.max(1) as f64).sqrt();
    let coefficient_sd = if slope.is_finite() && slope > 0.0 { ratio_sd / slope } else { ratio_sd };
    let whitening = observed.clamp(-MAX_PREWHITENING, MAX_PREWHITENING);
    let mut innovations = Vec::with_capacity(residuals[0].len() + residuals[1].len());
    for side in &residuals {
        let start = innovations.len();
        innovations.push((1.0 - whitening * whitening).sqrt() * side[0]);
        innovations.extend(side.windows(2).map(|pair| pair[1] - whitening * pair[0]));
        let mean = innovations[start..].iter().sum::<f64>() / side.len() as f64;
        for value in &mut innovations[start..] {
            *value -= mean;
        }
    }
    let variance = innovations.iter().map(|e| e * e).sum::<f64>() / innovations.len() as f64;
    let scale = if variance > 0.0 {
        (marginal_variance * (1.0 - coefficient * coefficient) / variance).sqrt()
    } else {
        0.0
    };
    for value in &mut innovations {
        *value *= scale;
    }
    Some(ErrorSeries { innovations, coefficient, coefficient_sd })
}

/// The estimator's weight on each innovation of an AR(1) error with coefficient `rho`
/// started from its stationary law: `Σ_{i ≥ j} weights[i] · rho^{i−j}`.
fn recolored_loadings(weights: &[f64], rho: f64) -> Vec<f64> {
    let mut loadings = vec![0.0; weights.len()];
    let mut carry = 0.0;
    for (loading, weight) in loadings.iter_mut().zip(weights).rev() {
        carry = weight + rho * carry;
        *loading = carry;
    }
    if let Some(first) = loadings.first_mut() {
        *first /= (1.0 - rho * rho).sqrt();
    }
    loadings
}

/// The fixed-design error model of a local ratio in calendar time.
///
/// The reported ratio is `(a·Y)/(a·T)` for fixed row weights `a`, so a redrawn error series
/// refits it in closed form. The errors are those of the structural series `Y − τ̂·T` and of
/// the treatment about a cubic trend on each side, each prewhitened by its own AR(1) fit;
/// one recursion runs through the cutoff, so dependence across it is kept.
struct TemporalErrors {
    /// Signed estimator weights of the window rows in time order, left side then right side.
    weights: Vec<f64>,
    /// Rows left of the cutoff.
    split: usize,
    /// Structural and treatment error; `None` for a series without error.
    series: [Option<ErrorSeries>; 2],
    /// Rows per block of innovations.
    block_length: usize,
    /// Residual degrees of freedom of the two trends.
    residual_df: f64,
}

impl TemporalErrors {
    fn new(
        columns: &LocalRatioColumns<'_>,
        cutoff: f64,
        bandwidth: f64,
        kink: bool,
        estimate: f64,
    ) -> Result<Self, EstimationError> {
        let unsupported = || {
            refused(
                "the window has too few rows on a side for the serial-dependence model of the \
                 errors; next attempt: a wider bandwidth",
            )
        };
        let (left, right) = temporal_side_rows(columns.running, cutoff, bandwidth);
        let mut weights = Vec::with_capacity(left.len() + right.len());
        let mut trends = Vec::with_capacity(2);
        let mut structural = Vec::with_capacity(2);
        let mut treatment = Vec::with_capacity(2);
        for (rows, sign) in [(&left, -1.0), (&right, 1.0)] {
            if rows.len() < TREND_ORDER + 3 {
                return Err(unsupported());
            }
            let distances: Vec<f64> =
                rows.iter().map(|&row| (columns.running[row] - cutoff) / bandwidth).collect();
            let side_weights =
                estimator_weights(&distances, bandwidth, kink).ok_or_else(unsupported)?;
            weights.extend(side_weights.iter().map(|weight| sign * weight));
            trends.push(SideTrend {
                inverse: inverse_gram(&distances, TREND_ORDER, |_| 1.0).ok_or_else(unsupported)?,
                basis: distances.iter().map(|&d| monomials(d, TREND_ORDER)).collect(),
            });
            structural.push(
                rows.iter()
                    .map(|&row| columns.outcome[row] - estimate * columns.treatment[row])
                    .collect::<Vec<f64>>(),
            );
            treatment.push(rows.iter().map(|&row| columns.treatment[row]).collect::<Vec<f64>>());
        }
        let Ok(trends) = <[SideTrend; 2]>::try_from(trends) else {
            return Err(unsupported());
        };
        let series = [
            error_series(&trends, [structural[0].as_slice(), structural[1].as_slice()]),
            error_series(&trends, [treatment[0].as_slice(), treatment[1].as_slice()]),
        ];
        Ok(Self {
            residual_df: (weights.len() - 2 * (TREND_ORDER + 1)) as f64,
            block_length: cube_root_block_length(left.len().min(right.len())),
            split: left.len(),
            weights,
            series,
        })
    }

    /// First row and length of each side in the window sequence.
    fn sides(&self) -> [(usize, usize); 2] {
        [(0, self.split), (self.split, self.weights.len() - self.split)]
    }

    /// The loadings of each series at its fitted coefficient.
    fn loadings(&self) -> [Option<Vec<f64>>; 2] {
        self.series.each_ref().map(|series| {
            series.as_ref().map(|series| recolored_loadings(&self.weights, series.coefficient))
        })
    }

    /// The ratio refit on the errors redrawn from window rows `sources`.
    fn redrawn_ratio(
        &self,
        fit: &LocalPolynomialRatioFit,
        sources: &[usize],
        loadings: &[Option<Vec<f64>>; 2],
    ) -> f64 {
        let shift = |index: usize| -> f64 {
            match (&self.series[index], &loadings[index]) {
                (Some(series), Some(loadings)) => loadings
                    .iter()
                    .zip(sources)
                    .map(|(loading, source)| loading * series.innovations[*source])
                    .sum(),
                _ => 0.0,
            }
        };
        let (structural, treatment) = (shift(0), shift(1));
        (fit.reduced_form + structural + fit.estimate * treatment) / (fit.first_stage + treatment)
    }

    /// Satterthwaite degrees of freedom of the bootstrap variance: its relative variance
    /// from the innovation variance, the prewhitening coefficient and the finite number of
    /// replicates behind the interquartile range. `None` without a structural error.
    fn reference_degrees_of_freedom(&self, replicates_ok: u32) -> Option<f64> {
        let series = self.series[0].as_ref()?;
        let log_variance = |rho: f64| {
            recolored_loadings(&self.weights, rho).iter().map(|c| c * c).sum::<f64>().ln()
        };
        let below = (series.coefficient - COEFFICIENT_STEP).max(-MAX_PREWHITENING);
        let above = (series.coefficient + COEFFICIENT_STEP).min(MAX_PREWHITENING);
        let sensitivity = (log_variance(above) - log_variance(below)) / (above - below);
        let relative_variance = 2.0 / self.residual_df
            + (sensitivity * series.coefficient_sd).powi(2)
            + 4.0 * IQR_RELATIVE_VARIANCE / f64::from(replicates_ok.max(1));
        (relative_variance.is_finite() && relative_variance > 0.0).then(|| 2.0 / relative_variance)
    }
}

/// Appends the `len` redrawn source rows of the side that starts at window row `offset`:
/// whole blocks of `block` consecutive rows from `next_start` until the side is filled.
fn fill_side(
    sources: &mut Vec<usize>,
    offset: usize,
    len: usize,
    block: usize,
    mut next_start: impl FnMut() -> usize,
) {
    let end = sources.len() + len;
    while sources.len() < end {
        let start = next_start().min(len - 1);
        let take = block.min(len - start).min(end - sources.len());
        sources.extend(offset + start..offset + start + take);
    }
}

/// Moving-block bootstrap of the prewhitened errors within each side of the cutoff.
#[allow(
    clippy::too_many_arguments,
    reason = "the fit's columns, design and resample size are distinct inputs"
)]
fn temporal_block_bootstrap(
    columns: &LocalRatioColumns<'_>,
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
    fit: &LocalPolynomialRatioFit,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<TemporalBlockRecord, EstimationError> {
    let errors = TemporalErrors::new(columns, cutoff, bandwidth, kink, fit.estimate)?;
    let loadings = errors.loadings();
    let draws = resample_draws(ctx, BLOCK_OPERATION, replicates, 1, |rng| {
        let mut sources = Vec::with_capacity(errors.weights.len());
        for (offset, len) in errors.sides() {
            let block = errors.block_length.min(len);
            fill_side(&mut sources, offset, len, block, || uniform_index(rng, len + 1 - block));
        }
        Some(vec![errors.redrawn_ratio(fit, &sources, &loadings)])
    })?;
    let summary = summarize_replicates(&[fit.estimate], &draws, replicates);
    let mut ratio = summary.quantities[0];
    let Some(standard_error) = ratio.standard_error else {
        return Err(refused(
            "the temporal block bootstrap produced too few finite replicates for a standard \
             error; next attempt: a wider bandwidth or more replicates",
        ));
    };
    if let Some(degrees) = errors.reference_degrees_of_freedom(summary.replicates_ok) {
        let critical = antecedent_stats::special::student_t_ppf(0.5 + ROLLOUT_LEVEL / 2.0, degrees);
        let interval =
            [fit.estimate - critical * standard_error, fit.estimate + critical * standard_error];
        ratio.interval = Some(interval);
        ratio.band = ratio.band.map(|band| [band[0].min(interval[0]), band[1].max(interval[1])]);
    }
    Ok(TemporalBlockRecord {
        block_length: errors.block_length,
        requested: replicates,
        replicates_ok: summary.replicates_ok,
        replicates_failed: summary.replicates_failed,
        ratio,
    })
}

/// Semantic operation tag of the within-side block Bayesian bootstrap.
const BLOCK_POSTERIOR_OPERATION: u64 = 0x4c52_5442_0002;

/// Posterior of a local ratio whose running variable is calendar time.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalRatioPosterior {
    /// Posterior draws of the ratio.
    pub draws: Vec<f64>,
    /// Posterior summary.
    pub summary: crate::rollout_resampling::PosteriorQuantity,
    /// Rows per block.
    pub block_length: usize,
    /// Requested draws.
    pub requested: u32,
    /// Draws whose weighted refit produced a finite ratio.
    pub draws_ok: u32,
}

/// A standard normal draw.
fn standard_normal(rng: &mut CausalRng) -> f64 {
    // 1 − U lies in (0, 1], so the logarithm is finite.
    let radius = (-2.0 * (1.0 - rng.next_f64()).ln()).sqrt();
    radius * (std::f64::consts::TAU * rng.next_f64()).cos()
}

/// Within-side block Bayesian bootstrap of the local ratio in calendar time.
///
/// The errors are those of the fixed-design model of the frequentist interval: the structural
/// series and the treatment about a cubic trend on each side, each prewhitened by an AR(1)
/// fit corrected for the trend. The innovations on each side are cut, in time order, into
/// consecutive non-overlapping blocks of the rounded cube root of the side's rows. Each draw
/// puts flat-Dirichlet weights on a side's blocks, redraws the side's innovations in whole
/// blocks from that weighted block distribution, draws each prewhitening coefficient from its
/// large-sample posterior, recolors the innovations through the cutoff and refits the robust
/// bias-corrected ratio on the redrawn series. Dependence the AR(1) fit leaves within a block
/// is kept; across block boundaries only the AR(1) part is. The design's continuity
/// assumptions stay declared.
///
/// # Errors
///
/// The unweighted fit fails, a side has too few rows for the error model, the resample is
/// refused (cancellation, admission), or fewer than 20 draws produce a finite ratio.
pub fn posterior_temporal_local_ratio(
    columns: &LocalRatioColumns<'_>,
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
    draws: u32,
    ctx: &ExecutionContext,
) -> Result<TemporalRatioPosterior, EstimationError> {
    let fit = fit_local_polynomial_ratio(
        columns.running,
        columns.outcome,
        columns.treatment,
        cutoff,
        bandwidth,
        kink,
    )
    .map_err(|message| refused(format!("{message}; next attempt: a wider bandwidth")))?;
    let errors = TemporalErrors::new(columns, cutoff, bandwidth, kink, fit.estimate)?;
    let raw = resample_draws(ctx, BLOCK_POSTERIOR_OPERATION, draws, 1, |rng| {
        let loadings = errors.series.each_ref().map(|series| {
            series.as_ref().map(|series| {
                let coefficient = (series.coefficient
                    + series.coefficient_sd * standard_normal(rng))
                .clamp(-MAX_PREWHITENING, MAX_PREWHITENING);
                recolored_loadings(&errors.weights, coefficient)
            })
        });
        let mut sources = Vec::with_capacity(errors.weights.len());
        for (offset, len) in errors.sides() {
            let block = errors.block_length.min(len);
            // Normalized exponential weights are flat-Dirichlet block probabilities.
            let weights = unit_weights(UnitWeighting::Multiplier, len.div_ceil(block), rng);
            let total: f64 = weights.iter().sum();
            fill_side(&mut sources, offset, len, block, || {
                let mut target = rng.next_f64() * total;
                let mut chosen = weights.len() - 1;
                for (index, weight) in weights.iter().enumerate() {
                    if target < *weight {
                        chosen = index;
                        break;
                    }
                    target -= weight;
                }
                chosen * block
            });
        }
        Some(vec![errors.redrawn_ratio(&fit, &sources, &loadings)])
    })?;
    let summary = crate::rollout_resampling::summarize_posterior(&raw, 1, draws);
    let Some(quantity) = summary.quantities[0] else {
        return Err(refused(
            "fewer than 20 block Bayesian-bootstrap draws produced a finite ratio; next \
             attempt: more draws or a wider bandwidth",
        ));
    };
    Ok(TemporalRatioPosterior {
        draws: raw
            .iter()
            .flatten()
            .filter(|draw| draw[0].is_finite())
            .map(|draw| draw[0])
            .collect(),
        summary: crate::rollout_resampling::PosteriorQuantity { band: None, ..quantity },
        block_length: errors.block_length,
        requested: draws,
        draws_ok: summary.draws_ok,
    })
}
