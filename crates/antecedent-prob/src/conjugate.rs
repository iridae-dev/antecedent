//! Analytic conjugate Gaussian linear regression.
//!
//! Normal–Inv-Gamma (or known-σ² Normal) posterior with diagonal Gaussian
//! coefficient prior. Draws are columnar; no object-per-draw storage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::needless_range_loop)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_kernels::standard_normal;
pub use antecedent_kernels::{sample_gamma, sample_inv_gamma};

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::InferenceDiagnostics;
use crate::error::ProbError;
use crate::likelihood_terms::validate_design;
use crate::linalg::{cholesky_spd, invert_spd};
use crate::posterior::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use crate::prior::{GaussianCoefficientPrior, InvGammaPrior, PriorSet};

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

    let (map, draws, include_sigma2) = if let Some(sigma2) = known_sigma2 {
        let (mean, cov) = posterior_known_sigma2(ncols, &coef_prior, xtx, xty, sigma2)?;
        let draws =
            draw_mvn_known_sigma(&mean, &cov, sigma2, options.n_draws, options.seed, workspace)?;
        (mean, draws, false)
    } else {
        let (mean, scale_chol, alpha_n, beta_n) =
            posterior_nig(ncols, &coef_prior, xtx, xty, ig, yty, n_eff, Some(design))?;
        let draws = draw_nig(
            &mean,
            &scale_chol,
            alpha_n,
            beta_n,
            options.n_draws,
            options.seed,
            workspace,
        )?;
        (mean, draws, true)
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
    Ok(BayesFitResult {
        draws: posterior,
        map,
        diagnostics: InferenceDiagnostics::analytic("conjugate_gaussian"),
        cov: None,
    })
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
    xtx: &[f64],
    xty: &[f64],
    sigma2: f64,
) -> Result<(Vec<f64>, Vec<f64>), ProbError> {
    // Conjugate known-σ²: Cov(β|σ²) = σ² V0 with V0 = diag(prior.variance),
    // so prior.precision() = V0^{-1}. Matches [`GaussianCoefficientPrior`] docs.
    // Λn = (V0^{-1} + X'X) / σ² ; mn = Λn^{-1} (V0^{-1} μ0 + X'y) / σ²
    let mut lam = vec![0.0; ncols * ncols];
    let prec = prior.precision();
    for i in 0..ncols {
        for j in 0..ncols {
            lam[i * ncols + j] = xtx[i * ncols + j] / sigma2;
        }
        lam[i * ncols + i] += prec[i] / sigma2;
    }
    let mut rhs = vec![0.0; ncols];
    for i in 0..ncols {
        rhs[i] = (prec[i] * prior.mean[i] + xty[i]) / sigma2;
    }
    let cov = invert_spd(&lam, ncols)?;
    let mut mean = vec![0.0; ncols];
    for i in 0..ncols {
        let mut acc = 0.0;
        for j in 0..ncols {
            acc += cov[i * ncols + j] * rhs[j];
        }
        mean[i] = acc;
    }
    Ok((mean, cov))
}

fn posterior_nig(
    ncols: usize,
    prior: &GaussianCoefficientPrior,
    xtx: &[f64],
    xty: &[f64],
    ig: InvGammaPrior,
    yty: f64,
    n_eff: f64,
    design: Option<BayesDesignRef<'_>>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), ProbError> {
    // Vn^{-1} = V0^{-1} + X'X ; mn = Vn (V0^{-1} m0 + X'y)
    // βn = β0 + ½ [ ‖y − X mn‖² + (mn − m0)' Λ0 (mn − m0) ]
    // Prefer residual RSS from the design (stable on uncentred y); fall back to
    // the three-term Gram form when only moments are available.
    let mut vn_inv = vec![0.0; ncols * ncols];
    let prec = prior.precision();
    for i in 0..ncols {
        for j in 0..ncols {
            vn_inv[i * ncols + j] = xtx[i * ncols + j];
        }
        vn_inv[i * ncols + i] += prec[i];
    }
    let mut rhs = vec![0.0; ncols];
    for i in 0..ncols {
        rhs[i] = prec[i] * prior.mean[i] + xty[i];
    }
    let vn = invert_spd(&vn_inv, ncols)?;
    let mut mean = vec![0.0; ncols];
    for i in 0..ncols {
        let mut acc = 0.0;
        for j in 0..ncols {
            acc += vn[i * ncols + j] * rhs[j];
        }
        mean[i] = acc;
    }

    let rss = match design {
        Some(d) => residual_ss_from_design(d, &mean),
        None => residual_ss_from_moments(ncols, &mean, xtx, xty, yty),
    };
    let mut prior_quad = 0.0;
    for i in 0..ncols {
        let d = mean[i] - prior.mean[i];
        prior_quad += prec[i] * d * d;
    }
    let alpha_n = ig.shape + 0.5 * n_eff;
    let beta_n = ig.scale + 0.5 * (rss + prior_quad);
    if !(beta_n > 0.0) || !(alpha_n > 0.0) || !beta_n.is_finite() || !alpha_n.is_finite() {
        return Err(ProbError::Numerical {
            message: format!("invalid NIG posterior: alpha={alpha_n} beta={beta_n}"),
        });
    }

    // Cholesky of Vn (scale matrix for β | σ²): cov(β|σ²) = σ² Vn
    let chol = cholesky_spd(&vn, ncols)?;
    Ok((mean, chol, alpha_n, beta_n))
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
    let mut rng = CausalRng::from_seed(seed);
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
    let mut rng = CausalRng::from_seed(seed);
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
    use antecedent_core::ExecutionContext;

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

            let (post_mean, _chol, post_alpha, post_beta) = super::posterior_nig(
                1,
                prior.gaussian_coefficients().unwrap(),
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
        let (_m, _c, alpha_n, beta_n) = super::posterior_nig(
            1,
            &prior,
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
}
