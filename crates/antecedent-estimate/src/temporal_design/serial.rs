//! Serial-dependence-aware inference for the arm designs: the ratio estimate's covariance when
//! the unit-period pairs are one dependent series rather than independent units.
//!
//! Declaring a dependent time series says no unit (or cluster) is an independent draw: shocks
//! shared by the units at a period make their pair scores dependent across units and across
//! periods. The linearized residual of the ratio estimate is aggregated over every pair that
//! decides in the same period, and the long-run variance of that period series is the
//! Newey-West (Bartlett kernel) estimator: `Gamma_0 + sum_{l=1}^{L} w_l (Gamma_l + Gamma_l')`
//! with `w_l = 1 - l / (L + 1)` and `Gamma_l = sum_t r_t r_{t-l}'`, where periods enter by their
//! labels (a missing period invents no adjacency). The covariance is
//! `T / (T - 1) * meat / (sum of weights)^2` for the `T` observed periods.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_core::{ExecutionContext, StreamDomain};

use super::{ClusterScores, NOMINAL_LEVEL, SamplingSummary, attach_band, invalid};
use crate::error::EstimationError;
use crate::util::refuse;

/// Stream key of the serial-band draws.
const SERIAL_BAND_STREAM: u64 = 0x7E5D_5E71_0000_0000;

/// Method label of a serial-dependence-aware summary.
pub const SERIAL_METHOD: &str = "newey_west_ratio_normal_over_periods";

/// The Newey-West truncation lag `floor(4 (T / 100)^(2/9))`, at least 1 and below `T`, for `T`
/// observed periods.
#[must_use]
pub fn newey_west_bandwidth(periods: usize) -> usize {
    if periods < 2 {
        return 0;
    }
    let rule = (4.0 * (periods as f64 / 100.0).powf(2.0 / 9.0)).floor().max(1.0) as usize;
    rule.min(periods - 1)
}

/// Newey-West covariance (row-major `k x k`) of the ratio estimate of period-aggregated
/// scores. `scores` clusters are the periods in ascending order and `periods` their labels.
///
/// # Errors
/// Period labels that do not match the clusters, or fewer than two periods.
pub fn serial_covariance(
    scores: &ClusterScores,
    estimate: &[f64],
    periods: &[i64],
    bandwidth: usize,
) -> Result<Vec<f64>, EstimationError> {
    let k = scores.coordinates;
    let t_count = scores.clusters();
    if periods.len() != t_count || t_count < 2 || estimate.len() != k {
        return Err(invalid("serial covariance needs one period label per aggregated cluster"));
    }
    if periods.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid("period labels must be strictly increasing"));
    }
    let residuals = scores.residuals(estimate);
    let index: HashMap<i64, usize> = periods.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let total: f64 = scores.counts.iter().sum();
    let mut meat = vec![0.0; k * k];
    for t in 0..t_count {
        for a in 0..k {
            for b in 0..k {
                meat[a * k + b] += residuals[t * k + a] * residuals[t * k + b];
            }
        }
    }
    for lag in 1..=bandwidth {
        let weight = 1.0 - lag as f64 / (bandwidth as f64 + 1.0);
        let Ok(step) = i64::try_from(lag) else { continue };
        for t in 0..t_count {
            let Some(&s) = periods[t].checked_sub(step).and_then(|p| index.get(&p)) else {
                continue;
            };
            for a in 0..k {
                for b in 0..k {
                    let cross = residuals[t * k + a] * residuals[s * k + b]
                        + residuals[s * k + a] * residuals[t * k + b];
                    meat[a * k + b] += weight * cross;
                }
            }
        }
    }
    let scale = t_count as f64 / (t_count as f64 - 1.0) / (total * total);
    Ok(meat.into_iter().map(|value| value * scale).collect())
}

/// The pointwise normal summary of a ratio of period-aggregated scores under serial dependence.
///
/// # Errors
/// As [`serial_covariance`], and a non-finite estimate or covariance.
pub fn serial_sampling_summary(
    scores: &ClusterScores,
    periods: &[i64],
) -> Result<SamplingSummary, EstimationError> {
    let estimate = scores.estimate();
    let covariance =
        serial_covariance(scores, &estimate, periods, newey_west_bandwidth(periods.len()))?;
    let k = scores.coordinates;
    let standard_error: Vec<f64> = (0..k).map(|j| covariance[j * k + j].max(0.0).sqrt()).collect();
    if estimate.iter().chain(&standard_error).any(|value| !value.is_finite()) {
        return Err(EstimationError::stats_msg("the design estimate is not finite"));
    }
    let z = antecedent_stats::NORMAL_Q975;
    Ok(SamplingSummary {
        lower: estimate.iter().zip(&standard_error).map(|(e, s)| e - z * s).collect(),
        upper: estimate.iter().zip(&standard_error).map(|(e, s)| e + z * s).collect(),
        estimate,
        standard_error,
        covariance,
        band: None,
        holm_adjusted_p: None,
        smoothing_bias_bound: None,
        method: SERIAL_METHOD,
    })
}

/// Lower-triangular factor of a positive semi-definite matrix; a pivot that is not positive
/// zeroes its column (a rank-deficient covariance has no variation in that direction).
fn psd_cholesky(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                l[i * n + i] = (matrix[i * n + i] - dot).max(0.0).sqrt();
            } else if l[j * n + j] > 1e-300 {
                l[i * n + j] = (matrix[i * n + j] - dot) / l[j * n + j];
            }
        }
    }
    l
}

/// Attach a simultaneous sup-t band whose critical value is the rank of the largest standardized
/// coordinate among Gaussian draws with the summary's own covariance.
///
/// Under independent clusters the band multiplies cluster residuals; those multipliers would
/// discard the cross-period dependence the covariance carries, so a serial summary draws from
/// its Newey-West covariance instead.
///
/// # Errors
/// Fewer than two replicates, a coordinate out of range, a zero standard error, or cancellation.
pub fn serial_sup_t_band(
    summary: &mut SamplingSummary,
    band_coordinates: &[usize],
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<(), EstimationError> {
    let k = summary.standard_error.len();
    if replicates < 2 || band_coordinates.is_empty() || band_coordinates.iter().any(|j| *j >= k) {
        return Err(invalid("a band needs at least two replicates and coordinates in range"));
    }
    if band_coordinates.iter().any(|j| {
        let se = summary.standard_error[*j];
        !se.is_finite() || se <= 0.0
    }) {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "temporal_design.degenerate_band",
            "a band coordinate has zero standard error",
        ));
    }
    let m = band_coordinates.len();
    let sub: Vec<f64> = band_coordinates
        .iter()
        .flat_map(|&a| band_coordinates.iter().map(move |&b| (a, b)))
        .map(|(a, b)| summary.covariance[a * k + b])
        .collect();
    let factor = psd_cholesky(&sub, m);
    let mut maxima = Vec::with_capacity(replicates);
    let mut normals = vec![0.0; m];
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, SERIAL_BAND_STREAM | replicate as u64);
        antecedent_kernels::fill_standard_normal(&mut rng, &mut normals);
        let mut largest = 0.0_f64;
        for (row, &j) in band_coordinates.iter().enumerate() {
            let draw: f64 = (0..=row).map(|c| factor[row * m + c] * normals[c]).sum();
            largest = largest.max((draw / summary.standard_error[j]).abs());
        }
        maxima.push(largest);
    }
    maxima.sort_by(f64::total_cmp);
    let critical = crate::util::monte_carlo_critical(&maxima, NOMINAL_LEVEL);
    attach_band(summary, band_coordinates, critical, replicates);
    Ok(())
}
