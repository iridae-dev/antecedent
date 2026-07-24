//! Fixed Gaussian posterior targets for HMC and Laplace.
//!
//! Unlike the historical plug-in residual variance, these targets keep one
//! potential for the entire run: known σ², or joint `(β, λ = log σ²)` under an
//! inverse-gamma residual prior. Coefficient priors follow
//! `β | σ² ~ N(m₀, σ² V₀)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use crate::backend::BayesDesignRef;
use crate::error::ProbError;
use crate::prior::{GaussianCoefficientPrior, GaussianVarianceModel};

/// Unconstrained posterior density and gradient used by HMC leapfrog steps.
pub trait PosteriorTarget {
    /// Dimension of the unconstrained state `q`.
    fn dim(&self) -> usize;

    /// Evaluate `log π(q)` and write `∇_q log π` into `grad`.
    ///
    /// # Errors
    ///
    /// Shape mismatches or non-finite intermediate values that make the density undefined.
    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError>;
}

/// Shared weighted residual / prior quadratic pieces for Gaussian linear models.
#[derive(Clone, Debug)]
pub struct GaussianSufficientStats {
    /// Effective sample size `Σ w_i`.
    pub n_eff: f64,
    /// Number of coefficients `p`.
    pub p: usize,
}

impl GaussianSufficientStats {
    /// Compute `n_eff` and validate design / coefficient lengths.
    ///
    /// # Errors
    ///
    /// Empty design or coefficient length mismatch.
    pub fn from_design(design: BayesDesignRef<'_>, ncols: usize) -> Result<Self, ProbError> {
        if design.ncols != ncols {
            return Err(ProbError::Shape {
                message: "Gaussian target ncols does not match design",
            });
        }
        if ncols == 0 || design.nrows == 0 {
            return Err(ProbError::Shape { message: "empty design" });
        }
        let mut n_eff = 0.0;
        for r in 0..design.nrows {
            n_eff += design.weights.map_or(1.0, |w| w[r]);
        }
        if !(n_eff > 0.0) || !n_eff.is_finite() {
            return Err(ProbError::Numerical {
                message: "non-positive or non-finite effective sample size".into(),
            });
        }
        Ok(Self { n_eff, p: ncols })
    }
}

/// Weighted residual sum of squares and score `X' W r` at `beta`.
///
/// `r_i = y_i - offset_i - x_i' β`, `RSS_w = Σ w_i r_i²`.
pub(crate) fn rss_and_xtwr(
    design: BayesDesignRef<'_>,
    beta: &[f64],
    xtwr: &mut [f64],
) -> Result<f64, ProbError> {
    let nrows = design.nrows;
    let ncols = beta.len();
    if xtwr.len() != ncols {
        return Err(ProbError::Shape { message: "xtwr length != ncols" });
    }
    xtwr.fill(0.0);
    let mut rss = 0.0;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut pred = offset;
        for c in 0..ncols {
            pred += design.x_colmajor[c * nrows + r] * beta[c];
        }
        let resid = design.y[r] - pred;
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        rss += w * resid * resid;
        for c in 0..ncols {
            xtwr[c] += w * design.x_colmajor[c * nrows + r] * resid;
        }
    }
    if !rss.is_finite() {
        return Err(ProbError::Numerical { message: "non-finite RSS".into() });
    }
    Ok(rss)
}

/// Prior quadratic `Q(β) = (β − m₀)' P (β − m₀)` and `P(β − m₀)` into `p_diff`.
pub(crate) fn prior_quadratic(
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    beta: &[f64],
    p_diff: &mut [f64],
) -> Result<f64, ProbError> {
    let p = beta.len();
    if p_diff.len() != p || prec.len() != p || coef_prior.len() != p {
        return Err(ProbError::Shape { message: "prior quadratic length mismatch" });
    }
    let mut q = 0.0;
    for i in 0..p {
        let diff = beta[i] - coef_prior.mean[i];
        p_diff[i] = prec[i] * diff;
        q += diff * p_diff[i];
    }
    if !q.is_finite() {
        return Err(ProbError::Numerical { message: "non-finite prior quadratic".into() });
    }
    Ok(q)
}

/// Known-σ² Gaussian target: state `q = β`.
#[derive(Clone, Debug)]
pub struct GaussianKnownTarget<'a> {
    design: BayesDesignRef<'a>,
    coef_prior: GaussianCoefficientPrior,
    prec: Vec<f64>,
    sigma2: f64,
    stats: GaussianSufficientStats,
    xtwr: Vec<f64>,
    p_diff: Vec<f64>,
}

impl<'a> GaussianKnownTarget<'a> {
    /// Build a known-variance target.
    ///
    /// # Errors
    ///
    /// Invalid σ² or design / prior shape errors.
    pub fn new(
        design: BayesDesignRef<'a>,
        coef_prior: GaussianCoefficientPrior,
        sigma2: f64,
    ) -> Result<Self, ProbError> {
        if !(sigma2 > 0.0) || !sigma2.is_finite() {
            return Err(ProbError::InvalidPrior {
                message: "known residual variance must be finite and > 0",
            });
        }
        coef_prior.validate()?;
        let ncols = design.ncols;
        if coef_prior.len() != ncols {
            return Err(ProbError::InvalidPrior { message: "coefficient prior length != ncols" });
        }
        let stats = GaussianSufficientStats::from_design(design, ncols)?;
        let prec = coef_prior.precision();
        Ok(Self {
            design,
            coef_prior,
            prec,
            sigma2,
            stats,
            xtwr: vec![0.0; ncols],
            p_diff: vec![0.0; ncols],
        })
    }

    /// Fixed residual variance.
    #[must_use]
    pub const fn sigma2(&self) -> f64 {
        self.sigma2
    }

    /// Effective sample size.
    #[must_use]
    pub const fn n_eff(&self) -> f64 {
        self.stats.n_eff
    }
}

impl PosteriorTarget for GaussianKnownTarget<'_> {
    fn dim(&self) -> usize {
        self.stats.p
    }

    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        let p = self.dim();
        if q.len() != p || grad.len() != p {
            return Err(ProbError::Shape { message: "known target state/grad length != p" });
        }
        let rss = rss_and_xtwr(self.design, q, &mut self.xtwr)?;
        let quad = prior_quadratic(&self.coef_prior, &self.prec, q, &mut self.p_diff)?;
        let inv_s2 = 1.0 / self.sigma2;
        let logp = -0.5 * inv_s2 * (rss + quad);
        for i in 0..p {
            grad[i] = inv_s2 * (self.xtwr[i] - self.p_diff[i]);
        }
        if !logp.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return Err(ProbError::Numerical {
                message: "non-finite known-Gaussian logp or gradient".into(),
            });
        }
        Ok(logp)
    }
}

/// Inverse-gamma residual target: state `q = [β, λ]` with `λ = log(σ²)`.
#[derive(Clone, Debug)]
pub struct GaussianInvGammaTarget<'a> {
    design: BayesDesignRef<'a>,
    coef_prior: GaussianCoefficientPrior,
    prec: Vec<f64>,
    shape: f64,
    scale: f64,
    /// `A = a₀ + (n_eff + p) / 2`.
    a_const: f64,
    stats: GaussianSufficientStats,
    xtwr: Vec<f64>,
    p_diff: Vec<f64>,
}

impl<'a> GaussianInvGammaTarget<'a> {
    /// Build a joint `(β, λ)` target.
    ///
    /// # Errors
    ///
    /// Invalid InvGamma params or design / prior shape errors.
    pub fn new(
        design: BayesDesignRef<'a>,
        coef_prior: GaussianCoefficientPrior,
        shape: f64,
        scale: f64,
    ) -> Result<Self, ProbError> {
        if !(shape > 0.0) || !(scale > 0.0) || !shape.is_finite() || !scale.is_finite() {
            return Err(ProbError::InvalidPrior {
                message: "InvGamma shape and scale must be finite and > 0",
            });
        }
        coef_prior.validate()?;
        let ncols = design.ncols;
        if coef_prior.len() != ncols {
            return Err(ProbError::InvalidPrior { message: "coefficient prior length != ncols" });
        }
        let stats = GaussianSufficientStats::from_design(design, ncols)?;
        let a_const = shape + 0.5 * (stats.n_eff + stats.p as f64);
        let prec = coef_prior.precision();
        Ok(Self {
            design,
            coef_prior,
            prec,
            shape,
            scale,
            a_const,
            stats,
            xtwr: vec![0.0; ncols],
            p_diff: vec![0.0; ncols],
        })
    }

    /// Constant `A` in the joint log-density.
    #[must_use]
    pub const fn a_const(&self) -> f64 {
        self.a_const
    }

    /// Prior shape α₀.
    #[must_use]
    pub const fn shape(&self) -> f64 {
        self.shape
    }

    /// Prior scale β₀.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
}

impl PosteriorTarget for GaussianInvGammaTarget<'_> {
    fn dim(&self) -> usize {
        self.stats.p + 1
    }

    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        let p = self.stats.p;
        if q.len() != p + 1 || grad.len() != p + 1 {
            return Err(ProbError::Shape {
                message: "InvGamma target state/grad length != p + 1",
            });
        }
        let beta = &q[..p];
        let lambda = q[p];
        if !lambda.is_finite() {
            return Err(ProbError::Numerical { message: "non-finite log_sigma2".into() });
        }
        let rss = rss_and_xtwr(self.design, beta, &mut self.xtwr)?;
        let quad = prior_quadratic(&self.coef_prior, &self.prec, beta, &mut self.p_diff)?;
        let b_beta = self.scale + 0.5 * (rss + quad);
        let exp_neg_l = (-lambda).exp();
        if !b_beta.is_finite() || !exp_neg_l.is_finite() {
            return Err(ProbError::Numerical {
                message: "non-finite InvGamma B(β) or exp(-λ)".into(),
            });
        }
        // log π = -A λ - B(β) e^{-λ}  (+ const omitted)
        let logp = -self.a_const * lambda - b_beta * exp_neg_l;
        for i in 0..p {
            grad[i] = exp_neg_l * (self.xtwr[i] - self.p_diff[i]);
        }
        grad[p] = -self.a_const + b_beta * exp_neg_l;
        if !logp.is_finite() || grad.iter().any(|g| !g.is_finite()) {
            return Err(ProbError::Numerical {
                message: "non-finite InvGamma-Gaussian logp or gradient".into(),
            });
        }
        Ok(logp)
    }
}

/// Build the Gaussian HMC / Laplace target from a resolved variance model.
///
/// # Errors
///
/// Propagates design / prior construction errors.
pub fn gaussian_target_from_model(
    design: BayesDesignRef<'_>,
    coef_prior: GaussianCoefficientPrior,
    model: GaussianVarianceModel,
) -> Result<GaussianTarget<'_>, ProbError> {
    match model {
        GaussianVarianceModel::Known { sigma2 } => {
            Ok(GaussianTarget::Known(GaussianKnownTarget::new(design, coef_prior, sigma2)?))
        }
        GaussianVarianceModel::InvGamma { shape, scale } => Ok(GaussianTarget::InvGamma(
            GaussianInvGammaTarget::new(design, coef_prior, shape, scale)?,
        )),
    }
}

/// Owned Gaussian target enum (avoids `dyn` across the HMC loop).
#[derive(Clone, Debug)]
pub enum GaussianTarget<'a> {
    /// Known residual variance.
    Known(GaussianKnownTarget<'a>),
    /// Joint inverse-gamma residual prior.
    InvGamma(GaussianInvGammaTarget<'a>),
}

impl PosteriorTarget for GaussianTarget<'_> {
    fn dim(&self) -> usize {
        match self {
            Self::Known(t) => t.dim(),
            Self::InvGamma(t) => t.dim(),
        }
    }

    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        match self {
            Self::Known(t) => t.logp_and_grad(q, grad),
            Self::InvGamma(t) => t.logp_and_grad(q, grad),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::GaussianCoefficientPrior;

    fn tiny_design() -> (Vec<f64>, Vec<f64>) {
        // n=4, p=2: intercept + x
        let x = vec![
            1.0, 1.0, 1.0, 1.0, // col 0
            0.0, 1.0, 2.0, 3.0, // col 1
        ];
        let y = vec![1.0, 2.5, 3.8, 5.1];
        (x, y)
    }

    #[test]
    fn known_gradient_matches_central_differences() {
        let (x, y) = tiny_design();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = GaussianCoefficientPrior::shared(2, 0.0, 4.0).unwrap();
        let mut target = GaussianKnownTarget::new(design, prior, 2.5).unwrap();
        let q = [0.2, 1.4];
        let mut grad = [0.0; 2];
        let lp = target.logp_and_grad(&q, &mut grad).unwrap();
        assert!(lp.is_finite());
        let eps = 1e-6;
        for i in 0..2 {
            let mut qp = q;
            let mut qm = q;
            qp[i] += eps;
            qm[i] -= eps;
            let mut g = [0.0; 2];
            let lp_p = target.logp_and_grad(&qp, &mut g).unwrap();
            let lp_m = target.logp_and_grad(&qm, &mut g).unwrap();
            let fd = (lp_p - lp_m) / (2.0 * eps);
            assert!((fd - grad[i]).abs() < 1e-5, "coef {i}: fd={fd} analytic={}", grad[i]);
        }
    }

    #[test]
    fn invgamma_gradient_matches_central_differences() {
        let (x, y) = tiny_design();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = GaussianCoefficientPrior::shared(2, 0.1, 2.0).unwrap();
        let mut target = GaussianInvGammaTarget::new(design, prior, 2.0, 1.5).unwrap();
        let q = [0.3, 1.2, 0.4]; // λ = log σ²
        let mut grad = [0.0; 3];
        let lp = target.logp_and_grad(&q, &mut grad).unwrap();
        assert!(lp.is_finite());
        let eps = 1e-6;
        for i in 0..3 {
            let mut qp = q;
            let mut qm = q;
            qp[i] += eps;
            qm[i] -= eps;
            let mut g = [0.0; 3];
            let lp_p = target.logp_and_grad(&qp, &mut g).unwrap();
            let lp_m = target.logp_and_grad(&qm, &mut g).unwrap();
            let fd = (lp_p - lp_m) / (2.0 * eps);
            assert!((fd - grad[i]).abs() < 1e-5, "coord {i}: fd={fd} analytic={}", grad[i]);
        }
    }

    #[test]
    fn known_sigma2_scales_likelihood_and_prior_curvature() {
        let (x, y) = tiny_design();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = GaussianCoefficientPrior::shared(2, 0.0, 1.0).unwrap();
        let q = [0.5, 1.0];
        let mut g1 = [0.0; 2];
        let mut g2 = [0.0; 2];
        let mut t1 = GaussianKnownTarget::new(design, prior.clone(), 1.0).unwrap();
        let mut t2 = GaussianKnownTarget::new(design, prior, 4.0).unwrap();
        let lp1 = t1.logp_and_grad(&q, &mut g1).unwrap();
        let lp2 = t2.logp_and_grad(&q, &mut g2).unwrap();
        // Both likelihood and prior quadratic scale by 1/σ², so logp and grad scale by 1/4.
        assert!((lp2 - lp1 / 4.0).abs() < 1e-12);
        for i in 0..2 {
            assert!((g2[i] - g1[i] / 4.0).abs() < 1e-12);
        }
    }
}
