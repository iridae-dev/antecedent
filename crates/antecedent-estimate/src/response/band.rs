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
    if standard_errors.iter().any(|se| !se.is_finite() || *se <= 0.0) {
        return Err(EstimationError::unsupported(
            "simultaneous bands require finite non-degenerate influence standard errors",
        ));
    }
    let mut rng = CausalRng::from_seed(seed);
    let _admitted = crate::util::admit_bound_array::<f64>(replicates as usize, 1)?;
    let mut maxima = Vec::with_capacity(replicates as usize);
    let mut multipliers = vec![0.0; sample_size];
    for _ in 0..replicates {
        for multiplier in &mut multipliers {
            *multiplier = if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
        }
        let mut maximum = 0.0_f64;
        for (g_idx, se) in (0..grid_len).zip(standard_errors) {
            let row = &influences[g_idx * sample_size..(g_idx + 1) * sample_size];
            let standardized = row
                .iter()
                .zip(&multipliers)
                .map(|(influence, multiplier)| influence * multiplier)
                .sum::<f64>()
                .abs()
                / se;
            if !standardized.is_finite() {
                return Err(EstimationError::unsupported(
                    "simultaneous band contains a non-finite standardized influence",
                ));
            }
            maximum = maximum.max(standardized);
        }
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
    if !critical.is_finite() || lower.iter().chain(&upper).any(|v| !v.is_finite()) {
        return Err(EstimationError::unsupported("simultaneous band has non-finite endpoints"));
    }
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
        || standard_deviations.iter().any(|sd| !sd.is_finite() || *sd <= 0.0)
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
            let standardized = (value - mean[grid]).abs() / standard_deviations[grid];
            if !standardized.is_finite() {
                return Err(EstimationError::unsupported(
                    "simultaneous Bayesian band has non-finite standardized draws",
                ));
            }
            maximum = maximum.max(standardized);
        }
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    // Finite-draw marginal quantiles can extend beyond the empirical maximum's
    // requested quantile. Containing them keeps the joint statement coherent
    // with the pointwise intervals published for the same draws.
    let mut critical = monte_carlo_critical(&maxima, level);
    for grid in 0..grid_len {
        let low = (mean[grid] - pointwise_lower[grid]) / standard_deviations[grid];
        let high = (pointwise_upper[grid] - mean[grid]) / standard_deviations[grid];
        if !low.is_finite() || !high.is_finite() {
            return Err(EstimationError::unsupported(
                "simultaneous Bayesian band has non-finite pointwise bounds",
            ));
        }
        critical = critical.max(low).max(high);
    }
    let lower: Vec<f64> =
        mean.iter().zip(standard_deviations).map(|(m, sd)| m - critical * sd).collect();
    let upper: Vec<f64> =
        mean.iter().zip(standard_deviations).map(|(m, sd)| m + critical * sd).collect();
    if !critical.is_finite() || lower.iter().chain(&upper).any(|v| !v.is_finite()) {
        return Err(EstimationError::unsupported("simultaneous band has non-finite endpoints"));
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    fn endpoints(band: ResponseUncertainty) -> (Vec<f64>, Vec<f64>) {
        match band {
            ResponseUncertainty::SimultaneousBand { lower, upper, .. } => {
                (lower.to_vec(), upper.to_vec())
            }
            _ => panic!("expected band"),
        }
    }
    #[test]
    fn simultaneous_bands_respect_small_units_and_reject_nonfinite_inputs() {
        let scale = 1e-18;
        let influences = [0.1, -0.2, 0.3, -0.1];
        let scaled: Vec<_> = influences.iter().map(|v| v * scale).collect();
        let a = endpoints(
            simultaneous_multiplier_band(&[1.0], &influences, 4, &[0.2], 0.95, 200, 4).unwrap(),
        );
        let b = endpoints(
            simultaneous_multiplier_band(&[scale], &scaled, 4, &[0.2 * scale], 0.95, 200, 4)
                .unwrap(),
        );
        assert!((a.0[0] - b.0[0] / scale).abs() < 1e-12);
        assert!((a.1[0] - b.1[0] / scale).abs() < 1e-12);
        assert!(
            simultaneous_multiplier_band(&[1.0], &influences, 4, &[0.0], 0.95, 200, 4).is_err()
        );
        assert!(
            simultaneous_multiplier_band(
                &[1.0],
                &[f64::NAN, 0.0, 0.0, 0.0],
                4,
                &[0.2],
                0.95,
                200,
                4
            )
            .is_err()
        );
        let draws: Vec<f64> = (0..120).map(|i| 1.0 + (f64::from(i) - 60.0) / 100.0).collect();
        let scaled_draws: Vec<f64> = draws.iter().map(|v| v * scale).collect();
        let a = endpoints(
            simultaneous_posterior_band(&[1.0], &[draws], &[0.5], &[1.5], &[0.35], 0.95).unwrap(),
        );
        let b = endpoints(
            simultaneous_posterior_band(
                &[scale],
                &[scaled_draws],
                &[0.5 * scale],
                &[1.5 * scale],
                &[0.35 * scale],
                0.95,
            )
            .unwrap(),
        );
        assert!((a.0[0] - b.0[0] / scale).abs() < 1e-12);
        assert!((a.1[0] - b.1[0] / scale).abs() < 1e-12);
        assert!(
            simultaneous_posterior_band(&[1.0], &[vec![1.0; 120]], &[0.5], &[1.5], &[0.0], 0.95)
                .is_err()
        );
        assert!(
            simultaneous_posterior_band(
                &[1.0],
                &[vec![f64::NAN; 120]],
                &[0.5],
                &[1.5],
                &[1.0],
                0.95
            )
            .is_err()
        );
    }
}
