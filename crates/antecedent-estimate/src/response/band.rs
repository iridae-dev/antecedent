//! Multiplier simultaneous band for a G×n influence matrix.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ResponseUncertainty;
use antecedent_stats::normal_ppf;

use crate::EstimationError;
use crate::util::monte_carlo_critical;

use super::splitmix64;

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
    let mut state = seed;
    let mut maxima = Vec::with_capacity(replicates as usize);
    let mut multipliers = vec![0.0; sample_size];
    for _ in 0..replicates {
        for multiplier in &mut multipliers {
            state = splitmix64(state);
            *multiplier = if state & 1 == 0 { -1.0 } else { 1.0 };
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
    })
}
