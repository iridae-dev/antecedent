// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kernel residualization and covariate scaling for sensitivity analysis.

use super::{
    ExecutionContext, MAX_NEIGHBORHOOD_WORK, MAX_NONPARAMETRIC_ROWS, RefutationProblem,
    ValidationError, neighborhood_loo_predict_pair, sample_sd,
};

/// Nadaraya–Watson leave-one-out prediction of two targets sharing one
/// covariate matrix (`n × dim`, row-major).
///
/// The kernel weight depends only on the covariates, so predicting `y1` and
/// `y2` in one pass halves the O(n²·dim) distance/`exp` work versus two calls;
/// per-target accumulation order is unchanged, so each output matches the
/// single-target form bit for bit.
///
/// Rows are independent, so the outer loop runs on `ctx`'s pool (results come back in row order,
/// each row summed in `j` order, so the output does not depend on the thread count). Distances are
/// squared Euclidean, which is exact for covariates already on a common scale (the caller
/// standardizes). Each row stores those squared distances once and reuses the same `f64` values
/// for the minimum and the weights; the weight shift `exp(−½ (d² − min d²) / h²)` keeps the
/// nearest observations at weight 1 even for a tiny bandwidth.
pub(super) fn nw_loo_predict_pair(
    y1: &[f64],
    y2: &[f64],
    cov_rowmajor: &[f64],
    dim: usize,
    bandwidth: f64,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<f64>), ValidationError> {
    let n = y1.len();
    let h2 = bandwidth * bandwidth;
    let rows = ctx.map_indexed(n, |i, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let xi = &cov_rowmajor[i * dim..(i + 1) * dim];
        // Slot `i` is unused; every other slot is the squared distance to that row.
        let mut dist2 = vec![0.0; n];
        let mut nearest = f64::INFINITY;
        for (j, slot) in dist2.iter_mut().enumerate() {
            if i == j {
                continue;
            }
            let xj = &cov_rowmajor[j * dim..(j + 1) * dim];
            let mut acc = 0.0;
            for d in 0..dim {
                let diff = xi[d] - xj[d];
                acc += diff * diff;
            }
            *slot = acc;
            nearest = nearest.min(acc);
        }
        let mut num1 = 0.0;
        let mut num2 = 0.0;
        let mut den = 0.0;
        for (j, &d2) in dist2.iter().enumerate() {
            if i == j {
                continue;
            }
            // The nearest observations always have weight 1, even for a tiny bandwidth.
            #[allow(
                clippy::float_cmp,
                reason = "nearest is the minimum of these very squared distances, so the nearest observations compare exactly equal"
            )]
            let w = if d2 == nearest { 1.0 } else { (-0.5 * (d2 - nearest) / h2).exp() };
            num1 += w * y1[j];
            num2 += w * y2[j];
            den += w;
        }
        Ok((num1 / den, num2 / den))
    })?;
    Ok(rows.into_iter().unzip())
}

/// Row-major covariate matrix over the complete-case rows of `mask` (adjustment ∪ {T, Y};
/// the caller computes that mask once and shares it with the residualization step), each
/// column divided by its sample SD (a constant column is left as is) so one isotropic
/// bandwidth means the same number of standard deviations in every direction.
pub(super) fn covariate_matrix(
    problem: &RefutationProblem<'_>,
    mask: &[bool],
) -> Result<(Vec<f64>, usize, usize), ValidationError> {
    let ids = problem.estimand.adjustment_set.to_vec();
    let n = mask.iter().filter(|&&k| k).count();
    if ids.is_empty() {
        return Ok((vec![1.0; n], n, 1));
    }
    let dim = ids.len();
    let mut cov = vec![0.0; n * dim];
    for (c, &z) in ids.iter().enumerate() {
        let col = problem.data.float64_masked(z, mask).map_err(ValidationError::from)?;
        for (r, &v) in col.iter().enumerate() {
            cov[r * dim + c] = v;
        }
    }
    standardize_columns(&mut cov, n, dim);
    Ok((cov, n, dim))
}

/// Divide each column of a row-major `n × dim` matrix by its sample SD (a constant or
/// non-finite-SD column is left unscaled).
pub(super) fn standardize_columns(cov: &mut [f64], n: usize, dim: usize) {
    for d in 0..dim {
        let column: Vec<f64> = (0..n).map(|r| cov[r * dim + d]).collect();
        let sd = sample_sd(&column);
        if sd.is_finite() && sd > 0.0 {
            for r in 0..n {
                cov[r * dim + d] /= sd;
            }
        }
    }
}

/// Which leave-one-out smoother residualizes treatment and outcome.
#[derive(Clone, Copy)]
pub(super) enum ResidualSmoother {
    /// Every other row enters the kernel sums (quadratic in the sample size) up to
    /// [`MAX_NONPARAMETRIC_ROWS`] complete cases; a larger sample runs the neighbourhood
    /// smoother over `neighbors_beyond` neighbours instead.
    ExactUpTo { neighbors_beyond: usize },
    /// Only the `neighbors` nearest other covariate rows enter them.
    Neighborhood { neighbors: usize },
}

/// Leave-one-out Nadaraya–Watson residuals of treatment and outcome on the adjustment
/// covariates, over the complete cases, with the complete-case row count and whether the
/// neighbourhood smoother ran in place of the exact one.
#[allow(
    clippy::type_complexity,
    reason = "residuals, row count and the route flag are returned together"
)]
pub(super) fn kernel_residuals(
    problem: &RefutationProblem<'_>,
    bandwidth: Option<f64>,
    ctx: &ExecutionContext,
    smoother: ResidualSmoother,
) -> Result<((Vec<f64>, Vec<f64>, usize), bool), ValidationError> {
    // Estimation complete cases (adjustment ∪ {T, Y} ∪ modifiers), shared by the
    // covariate matrix and the residualization pulls.
    let ids = problem.estimation_ids(true);
    let mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
    let (cov, n, dim) = covariate_matrix(problem, &mask)?;
    let t =
        problem.data.float64_masked(problem.treatment(), &mask).map_err(ValidationError::from)?;
    let y = problem.data.float64_masked(problem.outcome(), &mask).map_err(ValidationError::from)?;
    if t.len() != n || y.len() != n {
        return Err(ValidationError::data_msg("nonparametric sensitivity row mismatch"));
    }
    if n < 2 {
        return Err(ValidationError::data_msg(
            "nonparametric sensitivity requires at least 2 complete cases",
        ));
    }
    let routed =
        matches!(smoother, ResidualSmoother::ExactUpTo { .. }) && n > MAX_NONPARAMETRIC_ROWS;
    let smoother = match smoother {
        ResidualSmoother::ExactUpTo { neighbors_beyond } if routed => {
            ResidualSmoother::Neighborhood { neighbors: neighbors_beyond }
        }
        other => other,
    };
    // Covariates are standardized to unit SD, so Silverman's (1986) multivariate
    // normal-reference factor is the bandwidth. With no covariates every kernel distance
    // is zero and the width is immaterial.
    let h = bandwidth.unwrap_or_else(|| {
        antecedent_stats::silverman_normal_reference_factor(n, dim).unwrap_or(1.0)
    });
    if !h.is_finite() || h <= 0.0 {
        return Err(ValidationError::data_msg(
            "nonparametric sensitivity bandwidth must be finite and positive",
        ));
    }
    let (t_hat, y_hat) = match smoother {
        ResidualSmoother::ExactUpTo { .. } => nw_loo_predict_pair(&t, &y, &cov, dim, h, ctx)?,
        ResidualSmoother::Neighborhood { neighbors } => {
            // Neighbours beyond the other rows add nothing, so the work is bounded by them.
            if (n as u64).saturating_mul(neighbors.min(n) as u64) > MAX_NEIGHBORHOOD_WORK {
                return Err(ValidationError::NotApplicable {
                    message: "neighborhood sensitivity work (rows x neighbors) exceeds its \
                              budget; use fewer neighbors",
                });
            }
            neighborhood_loo_predict_pair(&t, &y, &cov, dim, h, neighbors, ctx)?
        }
    };
    let t_res: Vec<f64> = t.iter().zip(&t_hat).map(|(&a, &b)| a - b).collect();
    let y_res: Vec<f64> = y.iter().zip(&y_hat).map(|(&a, &b)| a - b).collect();
    Ok(((t_res, y_res, n), routed))
}
