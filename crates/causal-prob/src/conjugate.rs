//! Analytic conjugate Gaussian linear regression (DESIGN.md §14.4).
//!
//! Normal–Inv-Gamma (or known-σ² Normal) posterior with diagonal Gaussian
//! coefficient prior. Draws are columnar; no object-per-draw storage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use std::sync::Arc;

use causal_core::{CausalRng, ExecutionContext};

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::InferenceDiagnostics;
use crate::error::ProbError;
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
    if design.y.len() != nrows {
        return Err(ProbError::Shape { message: "y length != nrows" });
    }
    if design.x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(ProbError::Shape { message: "X buffer too short" });
    }
    if nrows == 0 || ncols == 0 {
        return Err(ProbError::Shape { message: "empty design" });
    }
    if let Some(w) = design.weights {
        if w.len() != nrows {
            return Err(ProbError::Shape { message: "weights length != nrows" });
        }
    }
    if let Some(o) = design.offsets {
        if o.len() != nrows {
            return Err(ProbError::Shape { message: "offsets length != nrows" });
        }
    }

    workspace.prepare(nrows, ncols, options.n_draws);

    let coef_prior = match prior.gaussian_coefficients() {
        Some(p) => p.clone(),
        None => GaussianCoefficientPrior::isotropic(ncols, 10.0),
    };
    if coef_prior.len() != ncols {
        return Err(ProbError::InvalidPrior { message: "coefficient prior length != ncols" });
    }
    coef_prior.validate()?;

    // Form XtWX and XtWy (weights default to 1; offsets subtracted from y).
    let xtx = &mut workspace.neg_hessian[..ncols * ncols];
    xtx.fill(0.0);
    let xty = &mut workspace.grad[..ncols];
    xty.fill(0.0);

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
        let mut acc = 0.0;
        for r in 0..nrows {
            let w = design.weights.map_or(1.0, |ww| ww[r]);
            let offset = design.offsets.map_or(0.0, |oo| oo[r]);
            let x = design.x_colmajor[c1 * nrows + r];
            acc += w * x * (design.y[r] - offset);
        }
        xty[c1] = acc;
    }

    let known_sigma2 = prior.known_residual_variance();
    let ig = prior.residual_inv_gamma().unwrap_or_else(InvGammaPrior::weakly_informative);

    let (map, draws, include_sigma2) = if let Some(sigma2) = known_sigma2 {
        let (mean, cov) = posterior_known_sigma2(ncols, &coef_prior, xtx, xty, sigma2)?;
        let draws =
            draw_mvn_known_sigma(&mean, &cov, sigma2, options.n_draws, options.seed, workspace)?;
        (mean, draws, false)
    } else {
        let (mean, scale_chol, alpha_n, beta_n) =
            posterior_nig(ncols, nrows, &coef_prior, xtx, xty, ig, design)?;
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
    })
}

fn posterior_known_sigma2(
    ncols: usize,
    prior: &GaussianCoefficientPrior,
    xtx: &[f64],
    xty: &[f64],
    sigma2: f64,
) -> Result<(Vec<f64>, Vec<f64>), ProbError> {
    // Λn = Λ0 + XtX/σ² ; mn = Λn^{-1}(Λ0 μ0 + XtY/σ²)
    let mut lam = vec![0.0; ncols * ncols];
    let prec = prior.precision();
    for i in 0..ncols {
        for j in 0..ncols {
            lam[i * ncols + j] = xtx[i * ncols + j] / sigma2;
        }
        lam[i * ncols + i] += prec[i];
    }
    let mut rhs = vec![0.0; ncols];
    for i in 0..ncols {
        rhs[i] = prec[i] * prior.mean[i] + xty[i] / sigma2;
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
    nrows: usize,
    prior: &GaussianCoefficientPrior,
    xtx: &[f64],
    xty: &[f64],
    ig: InvGammaPrior,
    design: BayesDesignRef<'_>,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), ProbError> {
    // Use prior precision on coefficients as if σ²=1 scaling in the NIG location update:
    // Vn^{-1} = V0^{-1} + X'X ; mn = Vn (V0^{-1} m0 + X'y)
    // Then αn = α0 + n/2, βn = β0 + 0.5 (m0' V0^{-1} m0 + y'y - mn' Vn^{-1} mn)
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

    let mut yty = 0.0;
    for r in 0..nrows {
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        let offset = design.offsets.map_or(0.0, |oo| oo[r]);
        let yr = design.y[r] - offset;
        yty += w * yr * yr;
    }
    let mut m0_term = 0.0;
    for i in 0..ncols {
        m0_term += prec[i] * prior.mean[i] * prior.mean[i];
    }
    let mut mn_term = 0.0;
    for i in 0..ncols {
        for j in 0..ncols {
            mn_term += mean[i] * vn_inv[i * ncols + j] * mean[j];
        }
    }
    let alpha_n = ig.shape + 0.5 * nrows as f64;
    let beta_n = ig.scale + 0.5 * (m0_term + yty - mn_term);
    if !(beta_n > 0.0) || !(alpha_n > 0.0) {
        return Err(ProbError::Numerical {
            message: format!("invalid NIG posterior: alpha={alpha_n} beta={beta_n}"),
        });
    }

    // Cholesky of Vn (scale matrix for β | σ²): cov(β|σ²) = σ² Vn
    let chol = cholesky_spd(&vn, ncols)?;
    Ok((mean, chol, alpha_n, beta_n))
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
        // β = mean + chol * z  (chol lower)
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

fn standard_normal(rng: &mut CausalRng) -> f64 {
    // Box–Muller
    let u1 = rng.next_f64().max(f64::EPSILON);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn sample_inv_gamma(shape: f64, scale: f64, rng: &mut CausalRng) -> f64 {
    // InvGamma(α, β) = 1 / Gamma(α, rate=1/β) with mean β/(α-1)
    let g = sample_gamma(shape, 1.0 / scale, rng);
    1.0 / g.max(f64::MIN_POSITIVE)
}

fn sample_gamma(shape: f64, rate: f64, rng: &mut CausalRng) -> f64 {
    // Marsaglia–Tsang for shape >= 1; boost for shape < 1
    if shape < 1.0 {
        let u = rng.next_f64().max(f64::EPSILON);
        return sample_gamma(shape + 1.0, rate, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = standard_normal(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v / rate;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v / rate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::PriorSpec;
    use causal_core::ExecutionContext;

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
}
