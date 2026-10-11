//! Multiplier simultaneous band for a G×n influence matrix.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, IntervalInterpretation, ResponseUncertainty};
use antecedent_stats::normal_ppf;

use crate::EstimationError;
use crate::util::monte_carlo_critical;

pub(super) fn simultaneous_multiplier_band(
    mean: &[f64],
    influences: &[f64],
    sample_size: usize,
    standard_errors: &[f64],
    level: f64,
    replicates: u32,
    seed: u64,
) -> Result<ResponseUncertainty, EstimationError> {
    let grid_len = mean.len();
    if sample_size == 0
        || grid_len == 0
        || influences.len() != grid_len * sample_size
        || standard_errors.len() != grid_len
    {
        return Err(EstimationError::unsupported(
            "simultaneous bands require a non-empty response grid",
        ));
    }
    if standard_errors.iter().any(|se| !se.is_finite() || *se <= f64::EPSILON) {
        return Err(EstimationError::unsupported(
            "simultaneous bands require finite non-degenerate influence standard errors",
        ));
    }
    let mut rng = CausalRng::from_seed(seed);
    let mut maxima = Vec::with_capacity(replicates as usize);
    let mut multipliers = vec![0.0; sample_size];
    for _ in 0..replicates {
        for multiplier in &mut multipliers {
            *multiplier = if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
        }
        let maximum = (0..grid_len)
            .zip(standard_errors)
            .map(|(g_idx, se)| {
                let row = &influences[g_idx * sample_size..(g_idx + 1) * sample_size];
                row.iter()
                    .zip(&multipliers)
                    .map(|(influence, multiplier)| influence * multiplier)
                    .sum::<f64>()
                    .abs()
                    / se
            })
            .fold(0.0_f64, f64::max);
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    // The population max-|t| quantile is never below the marginal one, but the
    // finite multiplier quantile can dip under it when grid columns are nearly
    // collinear (the shared covariate-marginalization term makes them so); a
    // simultaneous band narrower than the pointwise band would be incoherent.
    let critical = monte_carlo_critical(&maxima, level).max(normal_ppf(0.5 + level / 2.0));
    let lower = mean
        .iter()
        .zip(standard_errors)
        .map(|(estimate, se)| estimate - critical * se)
        .collect::<Vec<_>>();
    let upper = mean
        .iter()
        .zip(standard_errors)
        .map(|(estimate, se)| estimate + critical * se)
        .collect::<Vec<_>>();
    Ok(ResponseUncertainty::SimultaneousBand {
        level,
        lower: Arc::from(lower),
        upper: Arc::from(upper),
        replicates,
        interpretation: antecedent_core::IntervalInterpretation::Confidence,
    })
}

/// A fixed-grid credible band from coherent joint posterior response draws.
///
/// `columns[g][draw]` must refer to the same posterior draw at every grid point.
/// The maximum standardized deviation is calculated once per draw, retaining
/// the posterior dependence between the coordinates of the response curve.
pub(super) fn simultaneous_posterior_band(
    mean: &[f64],
    columns: &[Vec<f64>],
    pointwise_lower: &[f64],
    pointwise_upper: &[f64],
    standard_deviations: &[f64],
    level: f64,
) -> Result<ResponseUncertainty, EstimationError> {
    let grid_len = mean.len();
    let n_draws = columns.first().map_or(0, Vec::len);
    if grid_len == 0
        || columns.len() != grid_len
        || pointwise_lower.len() != grid_len
        || pointwise_upper.len() != grid_len
        || standard_deviations.len() != grid_len
        || n_draws < 100
        || !level.is_finite()
        || !(0.0..1.0).contains(&level)
        || columns.iter().any(|column| column.len() != n_draws)
        || mean.iter().any(|x| !x.is_finite())
        || standard_deviations.iter().any(|sd| !sd.is_finite() || *sd <= f64::EPSILON)
    {
        return Err(EstimationError::unsupported(
            "simultaneous Bayesian bands require at least 100 coherent finite posterior draws and non-degenerate grid uncertainty",
        ));
    }
    let mut maxima = Vec::with_capacity(n_draws);
    for draw in 0..n_draws {
        let mut maximum = 0.0_f64;
        for grid in 0..grid_len {
            let value = columns[grid][draw];
            if !value.is_finite() {
                return Err(EstimationError::unsupported(
                    "simultaneous Bayesian band contains a non-finite posterior draw",
                ));
            }
            maximum = maximum.max((value - mean[grid]).abs() / standard_deviations[grid]);
        }
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    // Finite-draw marginal quantiles can extend beyond the empirical maximum's
    // requested quantile. Containing them keeps the joint statement coherent
    // with the pointwise intervals published for the same draws.
    let mut critical = monte_carlo_critical(&maxima, level);
    for grid in 0..grid_len {
        critical = critical.max((mean[grid] - pointwise_lower[grid]) / standard_deviations[grid]);
        critical = critical.max((pointwise_upper[grid] - mean[grid]) / standard_deviations[grid]);
    }
    let lower: Vec<f64> =
        mean.iter().zip(standard_deviations).map(|(m, sd)| m - critical * sd).collect();
    let upper: Vec<f64> =
        mean.iter().zip(standard_deviations).map(|(m, sd)| m + critical * sd).collect();
    Ok(ResponseUncertainty::SimultaneousBand {
        level,
        lower: Arc::from(lower),
        upper: Arc::from(upper),
        replicates: u32::try_from(n_draws).map_err(|_| {
            EstimationError::unsupported("posterior draw count exceeds simultaneous band capacity")
        })?,
        interpretation: IntervalInterpretation::Credible,
    })
}
