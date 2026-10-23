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
//! **Diagnostics.** A McCrary binned-density test of a discontinuity in the running
//! variable's density at the cutoff (a manipulation check), and the robust bias-corrected
//! jump of each declared baseline covariate at the cutoff (a balance check). Neither is part
//! of identification: a clean density or balance check does not establish continuity of the
//! potential outcomes.
//!
//! **Calendar-time running variable.** A regression discontinuity in time has serially
//! dependent rows, so the independent-row HC0 variance does not apply. The interval is instead
//! a moving-block bootstrap within each side of the cutoff (blocks of consecutive rows in time
//! order, length the rounded cube root of the side's window rows), refitting the whole robust
//! bias-corrected local ratio per replicate; the interval is the estimate plus or minus the
//! normal quantile times the interquartile bootstrap standard error.
//!
//! No coverage of these intervals has been measured.
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
#[path = "local_ratio_design/tests.rs"]
mod tests;

use antecedent_core::ExecutionContext;

use crate::EstimationError;
use crate::local_polynomial_ratio::{LocalPolynomialRatioFit, fit_local_polynomial_ratio};
use crate::rollout_resampling::{
    ReplicateQuantity, cube_root_block_length, moving_block_starts, resample_draws,
    summarize_replicates,
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
    /// Run the McCrary density test at the cutoff.
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

/// McCrary density discontinuity test at the cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityTest {
    /// `ln f̂₊ − ln f̂₋`.
    pub log_difference: f64,
    /// Its standard error.
    pub standard_error: f64,
    /// Two-sided normal p-value; a diagnostic, not a licence.
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
                fit.estimate,
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
    for j in 0..size {
        for k in 0..size {
            variance += inverse[nu][j] * psi[j][k] * inverse[k][nu];
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
    let sign = if (ORDER + 1 + nu) % 2 == 0 { 1.0 } else { -1.0 };
    let curvature = (third_derivative(true)? - sign * third_derivative(false)?) / 6.0;
    let bias = b_nu * curvature;
    let variance_constant = v_nu * variance / density;
    if !(variance_constant > 0.0 && variance_constant.is_finite()) || !bias.is_finite() {
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

/// McCrary's binned-density test of a jump in the running variable's density at `cutoff`,
/// with local linear triangular smoothing of the bin heights within `bandwidth`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a bin index is the floor of a bounded finite ratio"
)]
#[must_use]
pub fn density_test(running: &[f64], cutoff: f64, bandwidth: f64) -> Option<DensityTest> {
    let n = running.len();
    if n < 4 * MIN_SIDE_ROWS || !(bandwidth > 0.0) {
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

/// Moving-block bootstrap of the local ratio within each side of the cutoff, rows in time
/// order.
#[allow(
    clippy::too_many_arguments,
    reason = "the fit's columns, design and resample size are distinct inputs"
)]
fn temporal_block_bootstrap(
    columns: &LocalRatioColumns<'_>,
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
    estimate: f64,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<TemporalBlockRecord, EstimationError> {
    let window: Vec<usize> = {
        let mut rows: Vec<usize> = (0..columns.running.len())
            .filter(|i| (columns.running[*i] - cutoff).abs() < bandwidth)
            .collect();
        rows.sort_by(|a, b| columns.running[*a].total_cmp(&columns.running[*b]).then(a.cmp(b)));
        rows
    };
    let left: Vec<usize> =
        window.iter().copied().filter(|i| columns.running[*i] < cutoff).collect();
    let right: Vec<usize> =
        window.iter().copied().filter(|i| columns.running[*i] >= cutoff).collect();
    let block_length = cube_root_block_length(left.len().min(right.len()));
    let draws = resample_draws(ctx, BLOCK_OPERATION, replicates, 1, |rng| {
        let mut running = Vec::with_capacity(window.len());
        let mut outcome = Vec::with_capacity(window.len());
        let mut treatment = Vec::with_capacity(window.len());
        for side in [&left, &right] {
            let mut taken = 0usize;
            for start in moving_block_starts(side.len(), block_length, rng) {
                for row in &side[start..(start + block_length).min(side.len())] {
                    if taken == side.len() {
                        break;
                    }
                    running.push(columns.running[*row]);
                    outcome.push(columns.outcome[*row]);
                    treatment.push(columns.treatment[*row]);
                    taken += 1;
                }
            }
        }
        fit_local_polynomial_ratio(&running, &outcome, &treatment, cutoff, bandwidth, kink)
            .ok()
            .map(|fit| vec![fit.estimate])
    })?;
    let summary = summarize_replicates(&[estimate], &draws, replicates);
    let ratio = summary.quantities[0];
    if ratio.standard_error.is_none() {
        return Err(refused(
            "the temporal block bootstrap produced too few finite replicates for a standard \
             error; next attempt: a wider bandwidth or more replicates",
        ));
    }
    Ok(TemporalBlockRecord {
        block_length,
        requested: replicates,
        replicates_ok: summary.replicates_ok,
        replicates_failed: summary.replicates_failed,
        ratio,
    })
}
