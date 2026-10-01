//! Analytic conjugate Gaussian linear regression.
//!
//! Normal–Inv-Gamma (or known-σ² Normal) posterior with diagonal Gaussian
//! coefficient prior. Draws are columnar; no object-per-draw storage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_kernels::standard_normal;
pub use antecedent_kernels::{sample_gamma, sample_inv_gamma};

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::InferenceDiagnostics;
use crate::error::ProbError;
use crate::likelihood_terms::validate_design;
use crate::linalg::{cholesky_spd, condition_from_chol, invert_spd_from_chol, solve_spd_into};
use crate::posterior::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use crate::prior::{CoefficientPrecision, GaussianCoefficientPrior, InvGammaPrior, PriorSet};

/// Analytic conjugate Gaussian linear backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConjugateGaussianBackend;

impl InferenceBackend for ConjugateGaussianBackend {
    fn fit(
        &self,
        likelihood: BayesLikelihood,
        design: BayesDesignRef<'_>,
        prior: &PriorSet,
        options: &BayesFitOptions,
        workspace: &mut LaplaceWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<BayesFitResult, ProbError> {
        if likelihood != BayesLikelihood::GaussianIdentity {
            return Err(ProbError::Inference {
                message: "conjugate backend supports GaussianIdentity only",
            });
        }
        prior.validate()?;
        fit_conjugate_gaussian(design, prior, options, workspace)
    }
}

/// Fit conjugate Gaussian linear regression and draw from the posterior.
///
/// # Errors
///
/// Shape, prior, or singular posterior precision.
pub fn fit_conjugate_gaussian(
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    options: &BayesFitOptions,
    workspace: &mut LaplaceWorkspace,
) -> Result<BayesFitResult, ProbError> {
    fit_with_absolute_scale_resolved(design, prior, |prior| {
        fit_conjugate_gaussian_resolved(design, prior, options, workspace)
    })
}

/// Run a Gaussian-identity fit on `prior` with any absolute-coefficient-scale
/// marker ([`PriorSet::mark_absolute_coefficient_scale`]) resolved first.
///
/// The target's residual variance converts the marked absolute variances to the
/// conjugate `V0` (`β | σ² ~ N(m, σ² V0)`): the known residual variance when the
/// prior fixes one, else the target's OLS residual variance
/// `Σ wᵢ rᵢ² / (n − p)` over rows with `wᵢ > 0`. The fit's diagnostics record the
/// plug-in and its `σ̂²`. A prior without the marker is passed through untouched.
///
/// # Errors
///
/// A marked prior on a design whose OLS residual variance is not estimable
/// (`n ≤ p`, singular `X'WX`, or zero residuals), or the fit's own errors.
pub(crate) fn fit_with_absolute_scale_resolved(
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    fit: impl FnOnce(&PriorSet) -> Result<BayesFitResult, ProbError>,
) -> Result<BayesFitResult, ProbError> {
    let Some(indices) = prior.absolute_coefficient_scale() else {
        return fit(prior);
    };
    let (sigma2, source) = match prior.known_residual_variance() {
        Some(v) => (v, "known residual variance"),
        None => (ols_residual_variance(design)?, "target OLS residual variance, df n-p"),
    };
    let resolved = prior
        .resolve_absolute_coefficient_scale(sigma2)?
        .ok_or(ProbError::InvalidPrior { message: "absolute coefficient scale marker vanished" })?;
    let mut result = fit(&resolved)?;
    let list = indices.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
    result.diagnostics.notes.push(Arc::from(format!(
        "absolute_coefficient_scale_plug_in: source posterior recorded no residual variance; \
         absolute prior variances of coefficients [{list}] converted to V0 with the target's \
         sigma2_hat={sigma2} ({source})"
    )));
    Ok(result)
}

/// Target residual-variance estimate `Σ wᵢ (yᵢ − offsetᵢ − xᵢ'β̂)² / (n − p)`.
///
/// `β̂` is the weighted least-squares solution and `n` counts rows with a positive
/// weight (the WLS residual quadratic form has expectation `σ² (n − p)`).
fn ols_residual_variance(design: BayesDesignRef<'_>) -> Result<f64, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    let n_pos = (0..nrows).filter(|&r| design.weights.map_or(1.0, |w| w[r]) > 0.0).count();
    if n_pos <= ncols {
        return Err(ProbError::Inference {
            message: "absolute-scale prior plug-in needs the target residual variance, which \
                      requires more observations than coefficients (n > p)",
        });
    }
    let mut xtx = vec![0.0; ncols * ncols];
    let mut xty = vec![0.0; ncols];
    for r in 0..nrows {
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        if w == 0.0 {
            continue;
        }
        let yr = design.y[r] - design.offsets.map_or(0.0, |oo| oo[r]);
        for c1 in 0..ncols {
            let x1 = design.x_colmajor[c1 * nrows + r];
            xty[c1] += w * x1 * yr;
            for c2 in c1..ncols {
                xtx[c1 * ncols + c2] += w * x1 * design.x_colmajor[c2 * nrows + r];
            }
        }
    }
    for c1 in 0..ncols {
        for c2 in 0..c1 {
            xtx[c1 * ncols + c2] = xtx[c2 * ncols + c1];
        }
    }
    let beta = solve_posterior_mean(&xtx, ncols, &xty)?.mean;
    #[allow(clippy::cast_precision_loss, reason = "row counts are far below 2^52")]
    let df = (n_pos - ncols) as f64;
    let sigma2 = residual_ss_from_design(design, &beta) / df;
    if !(sigma2 > 0.0) || !sigma2.is_finite() {
        return Err(ProbError::Inference {
            message: "absolute-scale prior plug-in: the target OLS residual variance is zero \
                      or non-finite",
        });
    }
    Ok(sigma2)
}

fn fit_conjugate_gaussian_resolved(
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    options: &BayesFitOptions,
    workspace: &mut LaplaceWorkspace,
) -> Result<BayesFitResult, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    validate_design(BayesLikelihood::GaussianIdentity, design)?;

    workspace.prepare(nrows, ncols, options.n_draws);

    let (yty, n_eff) = ensure_conjugate_gram(design, workspace);

    let coef_prior = match prior.gaussian_coefficients() {
        Some(p) => p.clone(),
        None => GaussianCoefficientPrior::isotropic(ncols, 10.0),
    };
    if coef_prior.len() != ncols {
        return Err(ProbError::InvalidPrior { message: "coefficient prior length != ncols" });
    }
    coef_prior.validate()?;

    let xtx = &workspace.neg_hessian[..ncols * ncols];
    let xty = &workspace.grad[..ncols];

    let known_sigma2 = prior.known_residual_variance();
    let ig = prior.residual_inv_gamma().unwrap_or_else(InvGammaPrior::weakly_informative);

    // Conjugate-scale precision V0⁻¹: dense when the prior set carries a
    // coefficient correlation (a hydrated full-covariance prior).
    let prec = prior.coefficient_precision(&coef_prior)?;

    let (map, draws, include_sigma2, condition, exact_cov) = if let Some(sigma2) = known_sigma2 {
        let (mean, cov, condition) =
            posterior_known_sigma2(ncols, &coef_prior, &prec, xtx, xty, sigma2)?;
        let draws =
            draw_mvn_known_sigma(&mean, &cov, sigma2, options.n_draws, options.seed, workspace)?;
        (mean, draws, false, condition, Some(cov))
    } else {
        let post =
            posterior_nig(ncols, &coef_prior, &prec, xtx, xty, ig, yty, n_eff, Some(design))?;
        let draws = draw_nig(
            &post.mean,
            &post.scale_chol,
            post.alpha_n,
            post.beta_n,
            options.n_draws,
            options.seed,
            workspace,
        )?;
        (post.mean, draws, true, post.condition, None)
    };

    let schema = if include_sigma2 {
        let mut q: Vec<_> = (0..ncols)
            .map(|i| PosteriorQuantityKind::Coefficient { index: i, name: None })
            .collect();
        q.push(PosteriorQuantityKind::ResidualVariance);
        PosteriorSchema { quantities: Arc::from(q) }
    } else {
        PosteriorSchema::coefficients(ncols)
    };

    let posterior = PosteriorDraws::from_column_major(schema, options.n_draws, draws)?;
    // Condition lower bound of the equilibrated posterior precision V_n^{-1} (σ²
    // scaling does not change it), so the MAX_HESSIAN_CONDITION publication gate
    // refuses a numerically singular conjugate solve but not one whose columns are
    // merely on different scales.
    let mut diagnostics = InferenceDiagnostics::analytic("conjugate_gaussian");
    diagnostics.hessian_condition = condition;
    // The known-σ² posterior is exactly Gaussian, so its covariance is published
    // (the NIG coefficient marginal is Student-t and has none).
    Ok(BayesFitResult { draws: posterior, map, diagnostics, cov: exact_cov })
}

fn ensure_conjugate_gram(
    design: BayesDesignRef<'_>,
    workspace: &mut LaplaceWorkspace,
) -> (f64, f64) {
    let key = crate::backend::ConjugateGramKey::from_design(&design);
    let n2 = design.ncols.saturating_mul(design.ncols);
    let nrows = design.nrows;
    let ncols = design.ncols;

    // Cache only XᵀX (+ n_eff from weights). Always recompute Xᵀy / yᵀy: SBC
    // reallocates equal-length outcome buffers that allocators often recycle to
    // the same address, so a y-pointer key would restore a stale likelihood.
    let xtx_hit = workspace.conjugate_key == Some(key) && workspace.conjugate_xtx.len() == n2;
    let n_eff = if xtx_hit {
        workspace.neg_hessian[..n2].copy_from_slice(&workspace.conjugate_xtx);
        workspace.conjugate_n_eff
    } else {
        let xtx = &mut workspace.neg_hessian[..n2];
        xtx.fill(0.0);
        let mut n_eff = 0.0;
        for r in 0..nrows {
            let w = design.weights.map_or(1.0, |ww| ww[r]);
            n_eff += w;
        }
        for c1 in 0..ncols {
            for c2 in c1..ncols {
                let mut acc = 0.0;
                for r in 0..nrows {
                    let w = design.weights.map_or(1.0, |ww| ww[r]);
                    let x1 = design.x_colmajor[c1 * nrows + r];
                    let x2 = design.x_colmajor[c2 * nrows + r];
                    acc += w * x1 * x2;
                }
                xtx[c1 * ncols + c2] = acc;
                xtx[c2 * ncols + c1] = acc;
            }
        }
        workspace.conjugate_xtx.clear();
        workspace.conjugate_xtx.extend_from_slice(xtx);
        workspace.conjugate_n_eff = n_eff;
        workspace.conjugate_key = Some(key);
        n_eff
    };

    let xty = &mut workspace.grad[..ncols];
    xty.fill(0.0);
    let mut yty = 0.0;
    for r in 0..nrows {
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        let offset = design.offsets.map_or(0.0, |oo| oo[r]);
        let yr = design.y[r] - offset;
        yty += w * yr * yr;
    }
    for c1 in 0..ncols {
        let mut acc = 0.0;
        for r in 0..nrows {
            let w = design.weights.map_or(1.0, |ww| ww[r]);
            let offset = design.offsets.map_or(0.0, |oo| oo[r]);
            let x = design.x_colmajor[c1 * nrows + r];
            acc += w * x * (design.y[r] - offset);
        }
        xty[c1] = acc;
    }
    (yty, n_eff)
}

fn posterior_known_sigma2(
    ncols: usize,
    prior: &GaussianCoefficientPrior,
    prec: &CoefficientPrecision,
    xtx: &[f64],
    xty: &[f64],
    sigma2: f64,
) -> Result<(Vec<f64>, Vec<f64>, f64), ProbError> {
    // Conjugate known-σ²: Cov(β|σ²) = σ² V0 (diagonal, or dense under a
    // coefficient correlation), so `prec` = V0^{-1}. Matches [`GaussianCoefficientPrior`] docs.
    // Λn = (V0^{-1} + X'X) / σ² ; mn = Λn^{-1} (V0^{-1} μ0 + X'y) / σ²
    let mut lam = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            lam[i * ncols + j] = xtx[i * ncols + j] / sigma2;
        }
    }
    prec.add_divided_to(&mut lam, ncols, sigma2);
    let mut prec_mean = vec![0.0; ncols];
    prec.mul_into(&prior.mean, &mut prec_mean);
    let mut rhs = vec![0.0; ncols];
    for i in 0..ncols {
        rhs[i] = (prec_mean[i] + xty[i]) / sigma2;
    }
    let solved = solve_posterior_mean(&lam, ncols, &rhs)?;
    let cov = solved.inverse();
    Ok((solved.mean, cov, solved.condition))
}

/// Posterior mean, precision factor and condition from [`solve_posterior_mean`].
struct ScaledSolve {
    /// Posterior mean `m` in the original coordinates.
    mean: Vec<f64>,
    /// Cholesky factor of the equilibrated precision `D⁻¹ A D⁻¹`.
    chol: Vec<f64>,
    /// Equilibration scales `D` (powers of two).
    scales: Vec<f64>,
    /// Condition lower bound of the equilibrated precision.
    condition: f64,
}

impl ScaledSolve {
    /// `A⁻¹ = D⁻¹ (D⁻¹ A D⁻¹)⁻¹ D⁻¹` in the original coordinates.
    fn inverse(&self) -> Vec<f64> {
        let n = self.scales.len();
        let mut inv = invert_spd_from_chol(&self.chol, n);
        for i in 0..n {
            for j in 0..n {
                inv[i * n + j] /= self.scales[i] * self.scales[j];
            }
        }
        inv
    }
}

/// Solve `A m = rhs` for the posterior mean through the Cholesky of the SPD posterior
/// precision `A` (no explicit inverse), after symmetric diagonal equilibration.
///
/// With `D = diag(d_j)`, `d_j` the power of two nearest `sqrt(A_jj)`, the solve runs on
/// `Ã = D⁻¹ A D⁻¹` (unit-order diagonal) in the coordinates `β̃ = D β`, and the mean is
/// mapped back as `m = D⁻¹ m̃`. This is the design-column rescaling that puts every
/// coefficient on a common scale, with the prior precision carried along exactly. Power-of-
/// two scales make the transform exact in floating point, so a design whose columns are
/// merely in different units (say `1e8` and `1e-3`) is judged by the condition of the
/// equilibrated system, not by its units.
fn solve_posterior_mean(a: &[f64], ncols: usize, rhs: &[f64]) -> Result<ScaledSolve, ProbError> {
    let scales: Vec<f64> = (0..ncols)
        .map(|j| {
            let diag = a[j * ncols + j];
            if diag.is_finite() && diag > 0.0 {
                // `2^round(log2 sqrt(diag))`, within the normal exponent range.
                #[allow(clippy::cast_possible_truncation, reason = "exponent of a finite f64")]
                let e = (0.5 * diag.log2()).round().clamp(-1000.0, 1000.0) as i32;
                2.0_f64.powi(e)
            } else {
                1.0
            }
        })
        .collect();
    let mut scaled = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            scaled[i * ncols + j] = a[i * ncols + j] / (scales[i] * scales[j]);
        }
    }
    let scaled_rhs: Vec<f64> = rhs.iter().zip(&scales).map(|(r, d)| r / d).collect();
    let mut mean = vec![0.0; ncols];
    let mut chol = vec![0.0; ncols * ncols];
    let mut scratch = vec![0.0; ncols];
    solve_spd_into(&scaled, ncols, &scaled_rhs, &mut mean, &mut chol, &mut scratch)?;
    for (m, d) in mean.iter_mut().zip(&scales) {
        *m /= d;
    }
    let condition = condition_from_chol(&chol, ncols);
    Ok(ScaledSolve { mean, chol, scales, condition })
}

/// Normal–Inv-Gamma posterior pieces from [`posterior_nig`].
struct NigPosterior {
    /// Posterior coefficient mean `m_n`.
    mean: Vec<f64>,
    /// Cholesky of the scale matrix `V_n` (`cov(β|σ²) = σ² V_n`).
    scale_chol: Vec<f64>,
    /// Inv-Gamma shape `α_n`.
    alpha_n: f64,
    /// Inv-Gamma scale `β_n`.
    beta_n: f64,
    /// Condition lower bound of the equilibrated posterior precision `V_n^{-1}`.
    condition: f64,
}

fn posterior_nig(
    ncols: usize,
    prior: &GaussianCoefficientPrior,
    prec: &CoefficientPrecision,
    xtx: &[f64],
    xty: &[f64],
    ig: InvGammaPrior,
    yty: f64,
    n_eff: f64,
    design: Option<BayesDesignRef<'_>>,
) -> Result<NigPosterior, ProbError> {
    // Vn^{-1} = V0^{-1} + X'X ; mn = Vn (V0^{-1} m0 + X'y)
    // βn = β0 + ½ [ ‖y − X mn‖² + (mn − m0)' Λ0 (mn − m0) ]
    // Prefer residual RSS from the design (stable on uncentred y); fall back to
    // the three-term Gram form when only moments are available.
    let mut vn_inv = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            vn_inv[i * ncols + j] = xtx[i * ncols + j];
        }
    }
    prec.add_divided_to(&mut vn_inv, ncols, 1.0);
    let mut prec_mean = vec![0.0; ncols];
    prec.mul_into(&prior.mean, &mut prec_mean);
    let mut rhs = vec![0.0; ncols];
    for i in 0..ncols {
        rhs[i] = prec_mean[i] + xty[i];
    }
    let solved = solve_posterior_mean(&vn_inv, ncols, &rhs)?;
    let vn = solved.inverse();
    let (mean, condition) = (solved.mean, solved.condition);

    let rss = match design {
        Some(d) => residual_ss_from_design(d, &mean),
        None => residual_ss_from_moments(ncols, &mean, xtx, xty, yty),
    };
    let prior_quad = prec.quadratic(&mean, &prior.mean);
    let alpha_n = ig.shape + 0.5 * n_eff;
    let beta_n = ig.scale + 0.5 * (rss + prior_quad);
    if !(beta_n > 0.0) || !(alpha_n > 0.0) || !beta_n.is_finite() || !alpha_n.is_finite() {
        return Err(ProbError::Numerical {
            message: format!("invalid NIG posterior: alpha={alpha_n} beta={beta_n}"),
        });
    }

    // Cholesky of Vn (scale matrix for β | σ²): cov(β|σ²) = σ² Vn
    let chol = cholesky_spd(&vn, ncols)?;
    Ok(NigPosterior { mean, scale_chol: chol, alpha_n, beta_n, condition })
}

/// Weighted residual sum of squares `Σ w_i (y_i − offset_i − x_i' m)²`.
fn residual_ss_from_design(design: BayesDesignRef<'_>, mean: &[f64]) -> f64 {
    let nrows = design.nrows;
    let ncols = mean.len();
    let mut rss = 0.0;
    for r in 0..nrows {
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        if w == 0.0 {
            continue;
        }
        let offset = design.offsets.map_or(0.0, |oo| oo[r]);
        let mut pred = offset;
        for c in 0..ncols {
            pred += design.x_colmajor[c * nrows + r] * mean[c];
        }
        let resid = design.y[r] - pred;
        rss += w * resid * resid;
    }
    rss.max(0.0)
}

/// RSS = ‖y − X m‖² from Gram moments: `y'y − 2 m'X'y + m'(X'X)m`.
///
/// Subtracts the cross term in two passes. Prefer [`residual_ss_from_design`]
/// when `y` and `X` are available — this form still loses digits for huge `|y|`.
fn residual_ss_from_moments(ncols: usize, mean: &[f64], xtx: &[f64], xty: &[f64], yty: f64) -> f64 {
    let mut mn_xty = 0.0;
    for i in 0..ncols {
        mn_xty += mean[i] * xty[i];
    }
    let mut mn_xtx_mn = 0.0;
    for i in 0..ncols {
        let mut row = 0.0;
        for j in 0..ncols {
            row += xtx[i * ncols + j] * mean[j];
        }
        mn_xtx_mn += mean[i] * row;
    }
    let mut rss = yty;
    rss -= mn_xty;
    rss -= mn_xty;
    rss += mn_xtx_mn;
    rss.max(0.0)
}

fn draw_mvn_known_sigma(
    mean: &[f64],
    cov: &[f64],
    sigma2: f64,
    n_draws: usize,
    seed: u64,
    workspace: &mut LaplaceWorkspace,
) -> Result<Arc<[f64]>, ProbError> {
    let ncols = mean.len();
    let chol = cholesky_spd(cov, ncols)?;
    let mut rng = crate::streams::direct_draw_rng(seed);
    let mut values = vec![0.0; n_draws * ncols];
    let z = &mut workspace.draw_scratch[..ncols];
    for d in 0..n_draws {
        for j in 0..ncols {
            z[j] = standard_normal(&mut rng);
        }
        // β = mean + chol * z (chol lower)
        for i in 0..ncols {
            let mut acc = mean[i];
            for j in 0..=i {
                acc += chol[i * ncols + j] * z[j];
            }
            values[i * n_draws + d] = acc;
        }
    }
    let _ = sigma2; // cov already includes σ²
    Ok(Arc::from(values))
}

fn draw_nig(
    mean: &[f64],
    scale_chol: &[f64],
    alpha_n: f64,
    beta_n: f64,
    n_draws: usize,
    seed: u64,
    workspace: &mut LaplaceWorkspace,
) -> Result<Arc<[f64]>, ProbError> {
    let ncols = mean.len();
    let mut rng = crate::streams::direct_draw_rng(seed);
    let mut values = vec![0.0; n_draws * (ncols + 1)];
    let z = &mut workspace.draw_scratch[..ncols];
    for d in 0..n_draws {
        let sigma2 = sample_inv_gamma(alpha_n, beta_n, &mut rng);
        let sigma = sigma2.sqrt();
        for j in 0..ncols {
            z[j] = standard_normal(&mut rng);
        }
        for i in 0..ncols {
            let mut acc = mean[i];
            for j in 0..=i {
                acc += sigma * scale_chol[i * ncols + j] * z[j];
            }
            values[i * n_draws + d] = acc;
        }
        values[ncols * n_draws + d] = sigma2;
    }
    Ok(Arc::from(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::PriorSpec;
    use antecedent_core::{CausalRng, ExecutionContext};

    fn simple_design() -> (Vec<f64>, Vec<f64>) {
        // y = 1 + 2x + noise; x = 0..9
        let n = 20;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = r as f64;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 1.0 + 2.0 * xi;
        }
        (x, y)
    }

    #[test]
    fn conjugate_recovers_ols_mean() {
        let (x, y) = simple_design();
        let n = y.len();
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 100.0)),
                PriorSpec::KnownResidualVariance(1e-6),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let mut ws = LaplaceWorkspace::default();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let opts = BayesFitOptions { n_draws: 500, seed: 42, ..BayesFitOptions::default() };
        let fit = ConjugateGaussianBackend
            .fit(
                BayesLikelihood::GaussianIdentity,
                design,
                &prior,
                &opts,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert!(fit.diagnostics.allows_posterior());
        assert!((fit.map[0] - 1.0).abs() < 1e-3);
        assert!((fit.map[1] - 2.0).abs() < 1e-3);
        let s = fit.draws.summarize();
        assert!((s.mean[0] - 1.0).abs() < 0.05);
        assert!((s.mean[1] - 2.0).abs() < 0.05);
    }

    fn fit_with_prior(x: &[f64], y: &[f64], ncols: usize, prior: &PriorSet) -> BayesFitResult {
        let mut ws = LaplaceWorkspace::default();
        let design = BayesDesignRef {
            x_colmajor: x,
            nrows: y.len(),
            ncols,
            y,
            weights: None,
            offsets: None,
        };
        let opts = BayesFitOptions { n_draws: 50, seed: 3, ..BayesFitOptions::default() };
        ConjugateGaussianBackend
            .fit(
                BayesLikelihood::GaussianIdentity,
                design,
                prior,
                &opts,
                &mut ws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap()
    }

    #[test]
    fn conjugate_reports_posterior_precision_condition() {
        // Well-conditioned design: a finite condition estimate that publishes.
        let (x, y) = simple_design();
        for prior in [
            PriorSet::weakly_informative(2),
            PriorSet {
                specs: vec![
                    PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 10.0)),
                    PriorSpec::KnownResidualVariance(1.0),
                ],
                contrast: None,
                categorical: Vec::new(),
                restrictions: Vec::new(),
            },
        ] {
            let fit = fit_with_prior(&x, &y, 2, &prior);
            let k = fit.diagnostics.hessian_condition;
            assert!(k.is_finite() && k > 1.0, "condition {k}");
            assert!(fit.diagnostics.allows_posterior());
        }

        // Columns on scales 1e8 and 1e-3 are well posed: the equilibrated
        // precision is well conditioned and the posterior publishes, with a mean
        // matching the same fit on the rescaled (unit-scale) design.
        let n = 20;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let t = r as f64 + 1.0;
            x[r] = 1e8 * t;
            x[n + r] = 1e-3 * (t * 0.7).sin();
            y[r] = 1e-8 * x[r] + x[n + r];
        }
        for prior in [
            PriorSet::weakly_informative(2),
            PriorSet {
                specs: vec![
                    PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 10.0)),
                    PriorSpec::KnownResidualVariance(1.0),
                ],
                contrast: None,
                categorical: Vec::new(),
                restrictions: Vec::new(),
            },
        ] {
            let fit = fit_with_prior(&x, &y, 2, &prior);
            let k = fit.diagnostics.hessian_condition;
            assert!(k.is_finite() && k < 10.0, "condition {k}");
            assert!(fit.diagnostics.allows_posterior());
            // Same model on unit-scale columns: β_unit = D β with the prior
            // variance rescaled by D², so the posterior means correspond exactly.
            let d = [1e8, 1e-3];
            let x_unit: Vec<f64> = x.iter().enumerate().map(|(i, v)| v / d[i / n]).collect();
            let unit_prior = PriorSet {
                specs: prior
                    .specs
                    .iter()
                    .map(|spec| match spec {
                        PriorSpec::GaussianCoefficients(c) => {
                            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
                                mean: c.mean.iter().zip(d).map(|(m, s)| m * s).collect(),
                                variance: c
                                    .variance
                                    .iter()
                                    .zip(d)
                                    .map(|(v, s)| v * s * s)
                                    .collect(),
                            })
                        }
                        other => other.clone(),
                    })
                    .collect(),
                ..prior.clone()
            };
            let unit = fit_with_prior(&x_unit, &y, 2, &unit_prior);
            for j in 0..2 {
                let rescaled = fit.map[j] * d[j];
                let rel = (rescaled - unit.map[j]).abs() / unit.map[j].abs().max(1e-300);
                assert!(rel < 1e-9, "coef {j}: {rescaled} vs {}", unit.map[j]);
            }
        }
    }

    /// Equilibration is exact (power-of-two scales), so a well-scaled design gets
    /// the same posterior as the unequilibrated solve to rounding.
    #[test]
    fn equilibrated_solve_matches_direct_solve_on_well_scaled_precision() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let rhs = [1.0, 2.0, -0.5];
        let solved = solve_posterior_mean(&a, 3, &rhs).unwrap();
        let mut direct = vec![0.0; 3];
        let mut chol = vec![0.0; 9];
        let mut scratch = vec![0.0; 3];
        solve_spd_into(&a, 3, &rhs, &mut direct, &mut chol, &mut scratch).unwrap();
        let direct_inv = invert_spd_from_chol(&chol, 3);
        for (m, d) in solved.mean.iter().zip(&direct) {
            assert!((m - d).abs() <= 1e-12 * d.abs(), "{m} vs {d}");
        }
        for (m, d) in solved.inverse().iter().zip(&direct_inv) {
            assert!((m - d).abs() <= 1e-12 * d.abs(), "{m} vs {d}");
        }
    }

    #[test]
    fn nig_draws_include_sigma2() {
        let (x, y) = simple_design();
        let n = y.len();
        let prior = PriorSet::weakly_informative(2);
        let mut ws = LaplaceWorkspace::default();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let opts = BayesFitOptions { n_draws: 200, seed: 7, ..BayesFitOptions::default() };
        let fit = fit_conjugate_gaussian(design, &prior, &opts, &mut ws).unwrap();
        assert_eq!(fit.draws.n_quantities(), 3);
        let sig = fit.draws.column(2).unwrap();
        assert!(sig.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn conjugate_gram_cache_is_bit_identical_on_refit() {
        let (x, y) = simple_design();
        let n = y.len();
        let prior = PriorSet::weakly_informative(2);
        let mut ws = LaplaceWorkspace::default();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let opts = BayesFitOptions { n_draws: 64, seed: 3, ..BayesFitOptions::default() };
        let first = fit_conjugate_gaussian(design, &prior, &opts, &mut ws).unwrap();
        assert!(ws.conjugate_key.is_some());
        let second = fit_conjugate_gaussian(design, &prior, &opts, &mut ws).unwrap();
        assert_eq!(first.map, second.map);
        assert_eq!(first.draws.values, second.draws.values);
    }

    #[test]
    fn conjugate_gram_cache_respects_different_outcomes_on_shared_workspace() {
        // SBC reallocates equal-length y buffers; allocators often recycle the
        // address. The XᵀX cache must not restore a stale Xᵀy.
        let (x, y1) = simple_design();
        let n = y1.len();
        let y2: Vec<f64> = y1.iter().map(|&v| v + 3.0).collect();
        let prior = PriorSet::weakly_informative(2);
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 64, seed: 3, ..BayesFitOptions::default() };
        let first = fit_conjugate_gaussian(
            BayesDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y: &y1,
                weights: None,
                offsets: None,
            },
            &prior,
            &opts,
            &mut ws,
        )
        .unwrap();
        let second = fit_conjugate_gaussian(
            BayesDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y: &y2,
                weights: None,
                offsets: None,
            },
            &prior,
            &opts,
            &mut ws,
        )
        .unwrap();
        assert!(
            (first.map[0] - second.map[0]).abs() > 0.5
                || (first.map[1] - second.map[1]).abs() > 0.5,
            "MAP must move when y shifts by a constant under an intercept design; \
             got {:?} vs {:?}",
            first.map,
            second.map
        );
    }

    #[test]
    fn conjugate_gram_cache_does_not_reuse_stale_xtx_after_realloc() {
        let n = 32usize;
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(1, 1e6)),
                PriorSpec::KnownResidualVariance(1.0),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let y = vec![2.0; n];
        let opts = BayesFitOptions { n_draws: 8, seed: 1, ..BayesFitOptions::default() };
        let mut reused = LaplaceWorkspace::default();
        let first_ptr = {
            let x = vec![1.0; n];
            let ptr = x.as_ptr() as usize;
            let _ = fit_conjugate_gaussian(
                BayesDesignRef {
                    x_colmajor: &x,
                    nrows: n,
                    ncols: 1,
                    y: &y,
                    weights: None,
                    offsets: None,
                },
                &prior,
                &opts,
                &mut reused,
            )
            .unwrap();
            ptr
        };
        let x2 = vec![2.0; n];
        let recycled = x2.as_ptr() as usize == first_ptr;
        let reused_fit = fit_conjugate_gaussian(
            BayesDesignRef {
                x_colmajor: &x2,
                nrows: n,
                ncols: 1,
                y: &y,
                weights: None,
                offsets: None,
            },
            &prior,
            &opts,
            &mut reused,
        )
        .unwrap();
        let fresh = fit_conjugate_gaussian(
            BayesDesignRef {
                x_colmajor: &x2,
                nrows: n,
                ncols: 1,
                y: &y,
                weights: None,
                offsets: None,
            },
            &prior,
            &opts,
            &mut LaplaceWorkspace::default(),
        )
        .unwrap();
        assert_eq!(
            reused_fit.map, fresh.map,
            "fresh/reused MAP must match after a new X allocation (allocator_recycled={recycled})"
        );
    }

    #[test]
    fn inv_gamma_moment_matches_mean() {
        // InvGamma(α=5, β=4) has mean β/(α−1) = 1.0
        let mut rng = CausalRng::from_seed(42);
        let n = 20_000usize;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += sample_inv_gamma(5.0, 4.0, &mut rng);
        }
        let mean = sum / n as f64;
        assert!((mean - 1.0).abs() < 0.05, "empirical mean {mean} far from 1.0");
    }

    /// Closed-form NIG on an uncentred intercept-only design with non-zero prior mean.
    ///
    /// Earns `prob-1`: the old `m0'Λ0 m0 + y'y − mn'Vn^{-1} mn` scale cancelled on
    /// large-mean outcomes; the residual-plus-prior-quadratic form must match the
    /// analytic `(αn, βn)` and the Student-t marginal half-width.
    #[test]
    fn nig_scale_matches_analytic_on_uncentred_design() {
        for &sigma in &[0.1_f64, 10.0] {
            let n = 64usize;
            let true_mu = 1.0e6;
            let mut y = vec![0.0; n];
            // Deterministic residuals so the Gram moments are exact.
            for r in 0..n {
                let e = ((r as f64 + 0.5) / n as f64 - 0.5) * sigma * 12.0_f64.sqrt();
                y[r] = true_mu + e;
            }
            let x = vec![1.0; n];
            let m0 = 1.0e6 + 3.0;
            let v0 = 4.0;
            let alpha0 = 3.0;
            let beta0 = 2.0 * sigma * sigma; // prior scale near the true σ²
            let prior = PriorSet {
                specs: vec![
                    PriorSpec::GaussianCoefficients(
                        GaussianCoefficientPrior::shared(1, m0, v0).unwrap(),
                    ),
                    PriorSpec::ResidualInvGamma(InvGammaPrior { shape: alpha0, scale: beta0 }),
                ],
                contrast: None,
                categorical: Vec::new(),
                restrictions: Vec::new(),
            };

            let yty: f64 = y.iter().map(|&yi| yi * yi).sum();
            let xty = y.iter().sum::<f64>();
            let xtx = n as f64;
            let lam0 = 1.0 / v0;
            let vn_inv = lam0 + xtx;
            let vn = 1.0 / vn_inv;
            let mn = vn * (lam0 * m0 + xty);
            let rss: f64 = y
                .iter()
                .map(|&yi| {
                    let d = yi - mn;
                    d * d
                })
                .sum();
            let prior_quad = (mn - m0) * (mn - m0) * lam0;
            let alpha_n = alpha0 + 0.5 * n as f64;
            let beta_n = beta0 + 0.5 * (rss + prior_quad);
            // Marginal Var(β) under NIG: (βn / (αn − 1)) · Vn
            let marg_var = (beta_n / (alpha_n - 1.0)) * vn;
            let half_width = 1.96 * marg_var.sqrt();

            let super::NigPosterior {
                mean: post_mean, alpha_n: post_alpha, beta_n: post_beta, ..
            } = super::posterior_nig(
                1,
                prior.gaussian_coefficients().unwrap(),
                &CoefficientPrecision::Diagonal(prior.gaussian_coefficients().unwrap().precision()),
                &[xtx],
                &[xty],
                InvGammaPrior { shape: alpha0, scale: beta0 },
                yty,
                n as f64,
                Some(BayesDesignRef {
                    x_colmajor: &x,
                    nrows: n,
                    ncols: 1,
                    y: &y,
                    weights: None,
                    offsets: None,
                }),
            )
            .unwrap();
            assert!(
                (post_mean[0] - mn).abs() < 1e-9,
                "sigma={sigma}: mean {} vs analytic {mn}",
                post_mean[0]
            );
            assert!(
                (post_alpha - alpha_n).abs() < 1e-12,
                "sigma={sigma}: alpha {post_alpha} vs {alpha_n}"
            );
            assert!(
                (post_beta - beta_n).abs() / beta_n < 1e-10,
                "sigma={sigma}: beta {post_beta} vs analytic {beta_n}"
            );

            let mut ws = LaplaceWorkspace::default();
            let fit = fit_conjugate_gaussian(
                BayesDesignRef {
                    x_colmajor: &x,
                    nrows: n,
                    ncols: 1,
                    y: &y,
                    weights: None,
                    offsets: None,
                },
                &prior,
                &BayesFitOptions { n_draws: 8_000, seed: 11, ..BayesFitOptions::default() },
                &mut ws,
            )
            .unwrap();
            let s = fit.draws.summarize();
            // coefficient column 0; residual variance column 1
            assert!(
                (s.mean[0] - mn).abs() < 0.05 * marg_var.sqrt().max(1e-6),
                "sigma={sigma}: draw mean {} vs {mn}",
                s.mean[0]
            );
            let emp_half = 1.96 * s.sd[0];
            assert!(
                (emp_half - half_width).abs() / half_width < 0.15,
                "sigma={sigma}: credible half-width {emp_half} vs analytic {half_width}"
            );
            let sig_mean = beta_n / (alpha_n - 1.0);
            assert!(
                (s.mean[1] - sig_mean).abs() / sig_mean < 0.1,
                "sigma={sigma}: sigma2 mean {} vs analytic {sig_mean}",
                s.mean[1]
            );
        }
    }

    #[test]
    fn nig_scale_stable_when_outcome_mean_is_huge() {
        // Would cancel under m0'Λ0 m0 + y'y − mn'Vn^{-1} mn when |y| ~ 1e8.
        let n = 32usize;
        let mu = 1.0e8;
        let y: Vec<f64> = (0..n).map(|r| mu + (r as f64 - 15.5) * 0.01).collect();
        let x = vec![1.0; n];
        let yty: f64 = y.iter().map(|&yi| yi * yi).sum();
        let xty = y.iter().sum::<f64>();
        let xtx = n as f64;
        let prior = GaussianCoefficientPrior::shared(1, mu, 1.0).unwrap();
        let ig = InvGammaPrior { shape: 2.0, scale: 1.0 };
        let super::NigPosterior { alpha_n, beta_n, .. } = super::posterior_nig(
            1,
            &prior,
            &CoefficientPrecision::Diagonal(prior.precision()),
            &[xtx],
            &[xty],
            ig,
            yty,
            n as f64,
            Some(BayesDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 1,
                y: &y,
                weights: None,
                offsets: None,
            }),
        )
        .unwrap();
        assert!(alpha_n.is_finite() && alpha_n > 0.0);
        assert!(beta_n.is_finite() && beta_n > 0.0 && beta_n < 1e6, "beta_n={beta_n}");
    }

    /// Intercept + N(0,1) covariate, `y = 1 + 0.5 x + σ ε`.
    fn noisy_design(n: usize, sigma: f64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = CausalRng::from_seed(17);
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = standard_normal(&mut rng);
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 1.0 + 0.5 * xi + sigma * standard_normal(&mut rng);
        }
        (x, y)
    }

    /// Two-coefficient prior with variances `var`, optionally marked absolute.
    fn coef_prior(var: [f64; 2], residual: PriorSpec, absolute: bool) -> PriorSet {
        let mut prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
                    mean: Arc::from(vec![0.0, 0.0]),
                    variance: Arc::from(var.to_vec()),
                }),
                residual,
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        if absolute {
            prior.mark_absolute_coefficient_scale(&[0, 1]);
        }
        prior
    }

    fn plug_in_sigma2(fit: &BayesFitResult) -> Option<f64> {
        let note = fit
            .diagnostics
            .notes
            .iter()
            .find(|n| n.starts_with("absolute_coefficient_scale_plug_in"))?;
        let tail = note.split("sigma2_hat=").nth(1)?;
        tail.split_whitespace().next()?.parse().ok()
    }

    #[test]
    fn absolute_scale_plug_in_yields_the_stated_absolute_prior_width() {
        // Source absolute sd s on both coefficients; target noise σ = 2 (σ² = 4).
        let (n, sigma, s) = (4000_usize, 2.0_f64, 0.3_f64);
        let (x, y) = noisy_design(n, sigma);
        let ig = PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative());
        let marked = coef_prior([s * s; 2], ig.clone(), true);
        let fit = fit_with_prior(&x, &y, 2, &marked);
        let sigma2_hat = plug_in_sigma2(&fit).expect("plug-in note with sigma2_hat");
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        assert_eq!(sigma2_hat.to_bits(), ols_residual_variance(design).unwrap().to_bits());
        // The fit consumed V0 = s² / σ̂²: identical to an unmarked prior written in V0.
        let v0 = s * s / sigma2_hat;
        let reference = fit_with_prior(&x, &y, 2, &coef_prior([v0; 2], ig, false));
        assert_eq!(fit.map, reference.map);
        assert!(plug_in_sigma2(&reference).is_none());
        // Effective prior variance on β under the true noise: σ² V0 = s² σ²/σ̂². With
        // σ̂²/σ² ~ χ²_{n−p}/(n−p) (sd √(2/(n−p))), a 5-sd band on σ̂²/σ² bounds the
        // ratio to s² within e/(1−e), e = 5√(2/(n−p)).
        let effective = sigma * sigma * v0;
        let e = 5.0 * (2.0 / (n - 2) as f64).sqrt();
        let tol = e / (1.0 - e);
        let ratio = effective / (s * s);
        assert!((ratio - 1.0).abs() < tol, "effective {effective} vs s² {} (tol {tol})", s * s);
    }

    #[test]
    fn absolute_scale_plug_in_uses_a_known_residual_variance_exactly() {
        let (x, y) = noisy_design(200, 2.0);
        let s2 = 0.09;
        let marked = coef_prior([s2; 2], PriorSpec::KnownResidualVariance(4.0), true);
        let fit = fit_with_prior(&x, &y, 2, &marked);
        assert_eq!(plug_in_sigma2(&fit), Some(4.0));
        let reference = fit_with_prior(
            &x,
            &y,
            2,
            &coef_prior([s2 / 4.0; 2], PriorSpec::KnownResidualVariance(4.0), false),
        );
        assert_eq!(fit.map, reference.map);
        assert_eq!(fit.draws.values, reference.draws.values);
    }

    #[test]
    fn absolute_scale_marker_is_already_in_glm_units() {
        // A GLM reads V0 at σ² ≡ 1, so an absolute-scale prior needs no conversion:
        // the marked fit is the unmarked fit, with no plug-in.
        let n = 300;
        let mut rng = CausalRng::from_seed(5);
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let z = standard_normal(&mut rng);
            x[r] = 1.0;
            x[n + r] = z;
            let p = 1.0 / (1.0 + (-(0.3 + 0.8 * z)).exp());
            y[r] = f64::from(rng.next_f64() < p);
        }
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let opts = BayesFitOptions { n_draws: 50, seed: 3, ..BayesFitOptions::default() };
        let fit = |prior: &PriorSet| {
            crate::laplace::fit_laplace_glm(
                BayesLikelihood::BernoulliLogit,
                design,
                prior,
                &opts,
                &mut LaplaceWorkspace::default(),
            )
            .unwrap()
        };
        let ig = PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative());
        let marked = fit(&coef_prior([0.25; 2], ig.clone(), true));
        let plain = fit(&coef_prior([0.25; 2], ig, false));
        assert_eq!(marked.map, plain.map);
        assert!(plug_in_sigma2(&marked).is_none());
    }

    #[test]
    fn absolute_scale_plug_in_refuses_an_unestimable_target_variance() {
        let (x, y) = noisy_design(2, 1.0);
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 2,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let marked = coef_prior(
            [1.0; 2],
            PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative()),
            true,
        );
        let opts = BayesFitOptions { n_draws: 10, seed: 1, ..BayesFitOptions::default() };
        let err = fit_conjugate_gaussian(design, &marked, &opts, &mut LaplaceWorkspace::default())
            .unwrap_err();
        assert!(matches!(err, ProbError::Inference { .. }), "{err:?}");
    }
}
