//! Multinomial logit fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::binomial::fit_logistic;
use super::{GlmDesignRef, GlmOptions, SATURATION_MARGIN, design_column};
use crate::error::{GlmRefusalKind, StatsError};
use crate::gram::invert_square;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresWorkspace};

/// Multinomial logit design: column-major `X` and integer category codes.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialDesignRef<'a> {
    /// Column-major design matrix (typically intercept + covariates).
    pub x_colmajor: &'a [f64],
    /// Rows.
    pub nrows: usize,
    /// Columns.
    pub ncols: usize,
    /// Category index in `0..n_categories` per row.
    pub y_category: &'a [u32],
    /// Number of categories `K ≥ 1`.
    pub n_categories: usize,
}

/// Multinomial logit fit (softmax / baseline-category logits).
///
/// Coefficients are stored row-major as `[K * ncols]` with **reference category 0**
/// pinned to zero (identifying constraint). Categories `1..K-1` hold free MLE logits.
#[derive(Clone, Debug)]
pub struct MultinomialFit {
    /// Row-major `[n_categories * ncols]`; category 0 is all zeros.
    pub coefficients: Vec<f64>,
    /// Fisher-scoring iterations used.
    pub iterations: u32,
    /// Whether the score updates converged.
    pub converged: bool,
    /// Whether fitted probabilities hit the soft clamp band (separation signal), or the
    /// Fisher information became singular and had to be ridge-regularized to continue.
    pub separated: bool,
    /// Final deviance (`−2` log-likelihood).
    pub deviance: f64,
    /// `K`.
    pub n_categories: usize,
    /// Design width.
    pub ncols: usize,
}

impl MultinomialFit {
    /// Error if scoring failed to converge or separation was detected.
    ///
    /// # Errors
    ///
    /// Non-converged or separated fits.
    pub fn require_ok(&self) -> Result<(), StatsError> {
        let refuse = |kind, message: &'static str| StatsError::GlmRefused {
            kind,
            message,
            iterations: self.iterations,
            boundary_margin_bits: None,
            boundary_count: None,
        };
        if !self.converged {
            return Err(refuse(
                GlmRefusalKind::NonConverged,
                "multinomial logit IRLS did not converge",
            ));
        }
        if self.separated {
            return Err(refuse(
                GlmRefusalKind::Separated,
                "multinomial logit indicates (quasi-)complete separation",
            ));
        }
        Ok(())
    }
}

/// Fit a baseline-category multinomial logit via Fisher scoring.
///
/// For `K = 2` this delegates to binomial logit IRLS (same MLE). For `K > 2` it
/// runs Fisher scoring on the `(K−1)·ncols` free parameters with expected Hessian
/// `Xᵀ W X` from the multinomial covariance.
///
/// # Errors
///
/// Shape mismatch, invalid category codes, a rank-deficient design
/// ([`StatsError::RankDeficient`]), or linear-algebra failure (binary path).
pub fn fit_multinomial_logit(
    design: MultinomialDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<MultinomialFit, StatsError> {
    fit_multinomial_logit_weighted(design, None, backend, workspace, options)
}

/// Multinomial logit with nonnegative observation weights.
///
/// # Errors
/// Invalid weights or any unweighted multinomial fit error.
pub fn fit_multinomial_logit_weighted(
    design: MultinomialDesignRef<'_>,
    weights: Option<&[f64]>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<MultinomialFit, StatsError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::GlmFit,
        || {
            if let Some(w) = weights {
                let sum: f64 = w.iter().sum();
                if w.len() != design.nrows
                    || w.iter().any(|v| !v.is_finite() || *v < 0.0)
                    || !sum.is_finite()
                    || sum <= 0.0
                {
                    return Err(StatsError::Shape {
                        message: "invalid multinomial observation weights",
                    });
                }
            }
            let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, n_categories: k } =
                design;
            if k == 0 {
                return Err(StatsError::Shape { message: "multinomial requires K ≥ 1" });
            }
            if y_category.len() != nrows {
                return Err(StatsError::Shape { message: "y_category length != nrows" });
            }
            if x_colmajor.len() < nrows.saturating_mul(ncols) {
                return Err(StatsError::Shape { message: "X buffer too short" });
            }
            for &yi in y_category {
                if (yi as usize) >= k {
                    return Err(StatsError::Shape { message: "y_category out of range" });
                }
            }

            if k == 1 {
                return Ok(MultinomialFit {
                    coefficients: vec![0.0; ncols],
                    iterations: 0,
                    converged: true,
                    separated: false,
                    deviance: 0.0,
                    n_categories: 1,
                    ncols,
                });
            }

            if k == 2 && weights.is_none() {
                return fit_multinomial_binary(design, backend, workspace, options);
            }

            fit_multinomial_fisher(design, options, weights)
        },
    )
}

fn fit_multinomial_binary(
    design: MultinomialDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<MultinomialFit, StatsError> {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, .. } = design;
    let y: Vec<f64> = y_category.iter().map(|&c| f64::from(c)).collect();
    let fit = fit_logistic(
        GlmDesignRef { x_colmajor, nrows, ncols, y: &y },
        backend,
        workspace,
        options,
    )?;
    let mut coefficients = vec![0.0; 2 * ncols];
    coefficients[ncols..].copy_from_slice(&fit.coefficients[..ncols]);
    Ok(MultinomialFit {
        coefficients,
        iterations: fit.iterations,
        converged: fit.converged,
        separated: fit.separated,
        deviance: fit.deviance,
        n_categories: 2,
        ncols,
    })
}

/// Whether `XᵀWX` (`W = diag(weights)`, identity when `None`) is numerically nonsingular.
///
/// Each Gram entry is summed down a pair of contiguous columns. The product
/// `(w_r x_{r c1}) x_{r c2}` matches the old per-row update.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn weighted_gram_is_full_rank(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    weights: Option<&[f64]>,
) -> bool {
    let mut gram = vec![0.0; ncols * ncols];
    if nrows > 0 {
        for c1 in 0..ncols {
            let col1 = design_column(x_colmajor, nrows, c1);
            for c2 in c1..ncols {
                let col2 = design_column(x_colmajor, nrows, c2);
                let mut acc = 0.0;
                for r in 0..nrows {
                    let w = weights.map_or(1.0, |w| w[r]);
                    acc += (w * col1[r]) * col2[r];
                }
                gram[c1 * ncols + c2] = acc;
            }
        }
    }
    for c1 in 0..ncols {
        for c2 in 0..c1 {
            gram[c1 * ncols + c2] = gram[c2 * ncols + c1];
        }
    }
    invert_square(&gram, ncols).is_some()
}

#[allow(
    clippy::too_many_lines,
    reason = "Fisher scoring loop over classes; the iteration state is read by every step"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn fit_multinomial_fisher(
    design: MultinomialDesignRef<'_>,
    options: &GlmOptions,
    weights: Option<&[f64]>,
) -> Result<MultinomialFit, StatsError> {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, n_categories: k } = design;
    let n_free = k - 1;
    let m = n_free * ncols;
    let mut beta_free = vec![0.0; m];
    let mut h = vec![0.0; m * m];
    let mut score = vec![0.0; m];
    let mut pi = vec![0.0; k];
    let mut eta = vec![0.0; k];
    let mut eta_free = vec![0.0; nrows * n_free];
    let mut pi_rows = vec![0.0; nrows * k];
    let mut delta = vec![0.0; m];
    let mut converged = false;
    let mut iterations = 0u32;
    let mut prev_deviance = f64::INFINITY;
    let mut regularized = false;

    // Identifiability is a property of the design, not of the sample size: a rank-deficient
    // (weighted) design is refused outright instead of being made invertible by a ridge.
    if ncols > 0 && !weighted_gram_is_full_rank(x_colmajor, nrows, ncols, weights) {
        return Err(StatsError::RankDeficient { rank: 0, ncols });
    }

    for iter in 1..=options.max_iter {
        iterations = iter;
        fill_multinomial_probabilities(
            design,
            &beta_free,
            &mut eta_free,
            &mut pi_rows,
            &mut eta,
            &mut pi,
        );
        let mut loop_deviance = 0.0;
        for r in 0..nrows {
            let yi = y_category[r] as usize;
            // In-loop deviance is for convergence only; final diagnostics recomputed below.
            loop_deviance +=
                weights.map_or(1.0, |w| w[r]) * -2.0 * pi_rows[r * k + yi].max(1e-300).ln();
        }
        accumulate_multinomial_score_and_information(design, &pi_rows, weights, &mut score, &mut h);

        // At interior probabilities the information is nonsingular exactly when the design
        // has full column rank (checked before the loop), so a singular `h` here means the
        // fitted probabilities have saturated (quasi-separation). Only then is a small ridge
        // applied, and the fit is flagged so `require_ok` refuses it.
        let h_inv = if let Some(inv) = invert_square(&h, m) {
            inv
        } else {
            let scale = (0..m).fold(0.0_f64, |acc, i| acc.max(h[i * m + i]));
            if !(scale.is_finite() && scale > 0.0) {
                return Err(StatsError::RankDeficient { rank: 0, ncols: m });
            }
            for i in 0..m {
                h[i * m + i] += 1e-8 * scale;
            }
            regularized = true;
            invert_square(&h, m).ok_or(StatsError::RankDeficient { rank: 0, ncols: m })?
        };
        let mut max_delta = 0.0_f64;
        let mut score_norm = 0.0_f64;
        for i in 0..m {
            score_norm = score_norm.max(score[i].abs());
            let mut d = 0.0;
            for j in 0..m {
                d += h_inv[i * m + j] * score[j];
            }
            delta[i] = d;
            max_delta = max_delta.max(d.abs());
        }
        // Damped Newton step when the update is large (common under quasi-separation).
        let scale = if max_delta > 5.0 { 5.0 / max_delta } else { 1.0 };
        for i in 0..m {
            beta_free[i] += scale * delta[i];
        }
        let dev_delta = (prev_deviance - loop_deviance).abs();
        prev_deviance = loop_deviance;
        if max_delta * scale < options.tol
            || score_norm < options.tol
            || (iter > 2 && dev_delta < options.tol * (1.0 + loop_deviance.abs()))
        {
            converged = true;
            break;
        }
    }

    // Diagnostics at the returned coefficients (not the pre-update scoring iterate).
    let (deviance, saturated) = multinomial_diagnostics_at(design, &beta_free, weights);
    let separated = saturated || regularized;

    let mut coefficients = vec![0.0; k * ncols];
    for j in 1..k {
        let src = (j - 1) * ncols;
        let dst = j * ncols;
        coefficients[dst..dst + ncols].copy_from_slice(&beta_free[src..src + ncols]);
    }

    Ok(MultinomialFit {
        coefficients,
        iterations,
        converged,
        separated,
        deviance,
        n_categories: k,
        ncols,
    })
}

/// Baseline-category probabilities for every row.
///
/// Free linear predictors are accumulated one contiguous column at a time, in
/// column order, so row `r` of class `j` matches `Σ_c x_{rc} β_{jc}`. Class 0
/// stays at `η = 0`. `η₀ = 0` is always a candidate for the maximum, so the
/// softmax normaliser is at least 1.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn fill_multinomial_probabilities(
    design: MultinomialDesignRef<'_>,
    beta_free: &[f64],
    eta_free: &mut [f64],
    pi_rows: &mut [f64],
    row_eta: &mut [f64],
    row_pi: &mut [f64],
) {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, n_categories: k, .. } = design;
    let n_free = k - 1;
    eta_free[..nrows * n_free].fill(0.0);
    if nrows > 0 && n_free > 0 && ncols > 0 {
        let covered = nrows * ncols;
        for (c, column) in x_colmajor[..covered].chunks_exact(nrows).take(ncols).enumerate() {
            for (r, &x) in column.iter().enumerate() {
                for j in 0..n_free {
                    eta_free[j * nrows + r] += x * beta_free[j * ncols + c];
                }
            }
        }
    }
    for r in 0..nrows {
        row_eta[0] = 0.0;
        for j in 0..n_free {
            row_eta[j + 1] = eta_free[j * nrows + r];
        }
        row_pi[..k].copy_from_slice(&row_eta[..k]);
        antecedent_kernels::softmax_in_place(&mut row_pi[..k]);
        pi_rows[r * k..(r + 1) * k].copy_from_slice(&row_pi[..k]);
    }
}

/// Score and Fisher information for free categories `1..K-1`.
///
/// One pass over each contiguous column pair updates every class block. The
/// score for column `c1` is added on the first partner column, in row order:
/// `(m_r (y_{rj} - π_{rj})) x_{r c1}`. The information entry is
/// `((m_r w_{jj'}) x_{r c1}) x_{r c2}`. A non-finite probability is left in
/// the matrix so the factorisation still refuses it.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
pub(super) fn accumulate_multinomial_score_and_information(
    design: MultinomialDesignRef<'_>,
    pi_rows: &[f64],
    weights: Option<&[f64]>,
    score: &mut [f64],
    h: &mut [f64],
) {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, n_categories: k } = design;
    let n_free = k - 1;
    let m = n_free * ncols;
    score[..m].fill(0.0);
    h[..m * m].fill(0.0);
    if nrows == 0 || ncols == 0 || n_free == 0 {
        return;
    }
    for c1 in 0..ncols {
        let col1 = design_column(x_colmajor, nrows, c1);
        for c2 in 0..ncols {
            let col2 = design_column(x_colmajor, nrows, c2);
            for r in 0..nrows {
                let obs = weights.map_or(1.0, |w| w[r]);
                let xc1 = col1[r];
                let xc2 = col2[r];
                let yi = y_category[r] as usize;
                if c2 == 0 {
                    for j in 0..n_free {
                        let pj = pi_rows[r * k + j + 1];
                        let resid = if yi == j + 1 { 1.0 } else { 0.0 } - pj;
                        score[j * ncols + c1] += (obs * resid) * xc1;
                    }
                }
                for j in 0..n_free {
                    let pj = pi_rows[r * k + j + 1];
                    for jp in 0..n_free {
                        let pjp = pi_rows[r * k + jp + 1];
                        let class_w = if j == jp { pj * (1.0 - pj) } else { -pj * pjp };
                        h[(j * ncols + c1) * m + (jp * ncols + c2)] +=
                            ((obs * class_w) * xc1) * xc2;
                    }
                }
            }
        }
    }
}

/// Multinomial deviance and separation at free coefficients (reference category 0 pinned).
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn multinomial_diagnostics_at(
    design: MultinomialDesignRef<'_>,
    beta_free: &[f64],
    weights: Option<&[f64]>,
) -> (f64, bool) {
    let MultinomialDesignRef { nrows, y_category, n_categories: k, .. } = design;
    let n_free = k - 1;
    let mut eta_free = vec![0.0; nrows * n_free];
    let mut pi_rows = vec![0.0; nrows * k];
    let mut row_eta = vec![0.0; k];
    let mut row_pi = vec![0.0; k];
    fill_multinomial_probabilities(
        design,
        beta_free,
        &mut eta_free,
        &mut pi_rows,
        &mut row_eta,
        &mut row_pi,
    );
    let mut deviance = 0.0;
    let mut separated = false;
    for r in 0..nrows {
        for p in &pi_rows[r * k..(r + 1) * k] {
            if !(SATURATION_MARGIN..=1.0 - SATURATION_MARGIN).contains(p) {
                separated = true;
            }
        }
        let yi = y_category[r] as usize;
        deviance += weights.map_or(1.0, |w| w[r]) * -2.0 * pi_rows[r * k + yi].max(1e-300).ln();
    }
    (deviance, separated)
}
