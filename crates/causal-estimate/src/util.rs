//! Shared estimation helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use causal_core::CausalRng;
use causal_stats::{StatsError, form_xtx, invert_square};

use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;

/// Map a stats-layer error into [`EstimationError::Stats`].
#[allow(clippy::needless_pass_by_value)] // StatsError is small / owned at call sites
pub(crate) fn stats_err(e: StatsError) -> EstimationError {
    EstimationError::from(e)
}

/// Require [`OverlapPolicy::ExplicitOverride`] (linear / IV / RD / front-door / GLM paths).
pub(crate) fn require_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if overlap != OverlapPolicy::ExplicitOverride {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// Refuse [`OverlapPolicy::ExplicitOverride`] (propensity / AIPW paths — positivity mandatory).
pub(crate) fn refuse_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if matches!(overlap, OverlapPolicy::ExplicitOverride) {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// Unbiased sample standard deviation; `NaN` if fewer than 2 observations.
pub(crate) fn sample_std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    var.sqrt()
}

/// IID bootstrap standard error: draw `replicates` estimates via `estimate`, then [`sample_std`].
pub(crate) fn bootstrap_se(
    replicates: u32,
    rng: &mut CausalRng,
    n: usize,
    mut estimate: impl FnMut(&[usize]) -> Result<Option<f64>, EstimationError>,
) -> Result<Option<f64>, EstimationError> {
    if replicates == 0 || n == 0 {
        return Ok(None);
    }
    let mut ates = Vec::with_capacity(replicates as usize);
    let mut idx = vec![0usize; n];
    for _ in 0..replicates {
        for slot in &mut idx {
            *slot = (rng.next_u64() as usize) % n;
        }
        if let Some(ate) = estimate(&idx)? {
            ates.push(ate);
        }
    }
    if ates.len() < 2 {
        return Ok(Some(f64::NAN));
    }
    Ok(Some(sample_std(&ates)))
}

/// OLS via normal equations on a column-major design matrix.
pub(crate) fn ols_colmajor(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    if nrows == 0 || ncols == 0 || y.len() != nrows || x_colmajor.len() != nrows * ncols {
        return Err(EstimationError::stats_msg("ols_colmajor: shape mismatch"));
    }
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    let mut xty = vec![0.0; ncols];
    for c in 0..ncols {
        let mut s = 0.0;
        let col = &x_colmajor[c * nrows..(c + 1) * nrows];
        for r in 0..nrows {
            s += col[r] * y[r];
        }
        xty[c] = s;
    }
    let inv = invert_square(&xtx, ncols)
        .ok_or_else(|| EstimationError::stats_msg("singular design in OLS"))?;
    let mut beta = vec![0.0; ncols];
    for i in 0..ncols {
        let mut s = 0.0;
        for j in 0..ncols {
            s += inv[i * ncols + j] * xty[j];
        }
        beta[i] = s;
    }
    Ok(beta)
}
