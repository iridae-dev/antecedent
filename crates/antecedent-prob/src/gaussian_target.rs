//! Fixed Gaussian posterior targets for HMC and Laplace.
//!
//! Unlike the historical plug-in residual variance, these targets keep one
//! potential for the entire run: known σ², or joint `(β, λ = log σ²)` under an
//! inverse-gamma residual prior. Coefficient priors follow
//! `β | σ² ~ N(m₀, σ² V₀)`. Leapfrog reads a weighted, offset-aware cross-product
//! summary, so it does not rescan the design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::backend::BayesDesignRef;
use crate::error::ProbError;
use crate::prior::{CoefficientPrecision, GaussianCoefficientPrior, GaussianVarianceModel};

/// Unconstrained posterior density and gradient used by HMC leapfrog steps.
pub(crate) trait PosteriorTarget {
    /// Dimension of the unconstrained state `q`.
    fn dim(&self) -> usize;

    /// Evaluate `log π(q)` and write `∇_q log π` into `grad`.
    ///
    /// # Errors
    ///
    /// Shape mismatches or non-finite intermediate values that make the density undefined.
    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError>;
}

/// Weighted, offset-aware cross-products for a Gaussian linear model.
///
/// The summary stores `n_eff`, `X' W X`, `X' W ỹ`, and `ỹ' W ỹ`, where
/// `ỹ = y − offset` and a missing offset is zero. Known-σ² and joint
/// `(β, log σ²)` targets both hold this summary, so a later HMC step evaluates
/// the residual quadratic and `X' W r` without rescanning rows.
#[derive(Clone, Debug)]
pub(crate) struct GaussianSufficientStats {
    /// Effective sample size `Σ w_i`.
    pub n_eff: f64,
    /// Number of coefficients `p`.
    pub p: usize,
    /// `X' W X`, entry `(c1, c2)` at `c1 * p + c2` (`p²`).
    pub xtwx: Vec<f64>,
    /// `X' W (y − offset)` (`p`).
    pub xtwy: Vec<f64>,
    /// `(y − offset)' W (y − offset)`.
    pub ytwy: f64,
}

impl GaussianSufficientStats {
    /// Cross-products of a weighted, offset-adjusted design.
    ///
    /// A non-finite weight, offset, or design row is refused. The next attempt
    /// is to drop that row from the summary and rebuild; the weight is not skipped.
    ///
    /// # Errors
    ///
    /// Empty design, a length mismatch, a non-finite weight, offset, or row, a
    /// negative weight, a non-finite cross-product, or a non-positive effective
    /// sample size.
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
    )]
    #[must_use = "a non-finite weight, offset, or row is refused"]
    pub fn from_design(design: BayesDesignRef<'_>, ncols: usize) -> Result<Self, ProbError> {
        if design.ncols != ncols {
            return Err(ProbError::Shape {
                message: "Gaussian target ncols does not match design",
            });
        }
        if ncols == 0 || design.nrows == 0 {
            return Err(ProbError::Shape { message: "empty design" });
        }
        screened_design_lengths(design)?;
        let Some(gram) = ncols.checked_mul(ncols) else {
            return Err(ProbError::Shape { message: "coefficient dimension overflow" });
        };
        let mut stats =
            Self { n_eff: 0.0, p: ncols, xtwx: vec![0.0; gram], xtwy: vec![0.0; ncols], ytwy: 0.0 };
        for row in 0..design.nrows {
            accumulate_summary_row(design, row, &mut stats)?;
        }
        if !(stats.n_eff > 0.0) || !stats.n_eff.is_finite() {
            return Err(ProbError::Numerical {
                message: "non-positive or non-finite effective sample size".into(),
            });
        }
        Ok(stats)
    }

    /// Residual sum of squares and score `X' W r` from the stored cross-products.
    ///
    /// `ỹ = y − offset`, `RSS = ỹ' W ỹ − 2 β' X' W ỹ + β' X' W X β`, and
    /// `X' W r = X' W ỹ − X' W X β`. No row is revisited.
    ///
    /// # Errors
    ///
    /// `beta` / `xtwr` length mismatch, or a non-finite residual sum of squares.
    #[must_use = "a non-finite residual sum of squares is refused"]
    pub(crate) fn rss_and_score(&self, beta: &[f64], xtwr: &mut [f64]) -> Result<f64, ProbError> {
        let p = self.p;
        if beta.len() != p || xtwr.len() != p {
            return Err(ProbError::Shape { message: "sufficient-stat score length != p" });
        }
        for (i, score) in xtwr.iter_mut().enumerate() {
            let mut acc = self.xtwy[i];
            for (&gram, &coef) in self.xtwx[i * p..(i + 1) * p].iter().zip(beta) {
                acc -= gram * coef;
            }
            *score = acc;
        }
        let mut rss = self.ytwy;
        for ((&coef, &cross), &score) in beta.iter().zip(&self.xtwy).zip(xtwr.iter()) {
            rss -= coef * (cross + score);
        }
        if !rss.is_finite() || xtwr.iter().any(|score| !score.is_finite()) {
            return Err(ProbError::Numerical {
                message: "non-finite residual sum of squares from cross-products".into(),
            });
        }
        Ok(rss)
    }
}

/// Lengths required before a row can be accumulated. A long `X` buffer is kept;
/// only the `nrows * ncols` prefix is read.
#[must_use = "a short design buffer is refused"]
fn screened_design_lengths(design: BayesDesignRef<'_>) -> Result<(), ProbError> {
    if design.y.len() != design.nrows {
        return Err(ProbError::Shape { message: "y length != nrows" });
    }
    if design.x_colmajor.len() < design.nrows.saturating_mul(design.ncols) {
        return Err(ProbError::Shape { message: "X buffer too short" });
    }
    if let Some(weights) = design.weights {
        if weights.len() != design.nrows {
            return Err(ProbError::Shape { message: "weights length != nrows" });
        }
    }
    if let Some(offsets) = design.offsets {
        if offsets.len() != design.nrows {
            return Err(ProbError::Shape { message: "offsets length != nrows" });
        }
    }
    Ok(())
}

fn non_finite_summary_input(what: &str, row: usize) -> ProbError {
    ProbError::Numerical {
        message: format!(
            "non-finite {what} at row {row}; next attempt: drop that row from the summary and \
             rebuild"
        ),
    }
}

fn negative_weight(row: usize) -> ProbError {
    ProbError::Numerical {
        message: format!(
            "negative weight at row {row}; next attempt: drop that row from the summary and \
             rebuild"
        ),
    }
}

#[must_use = "a non-finite cross-product is refused"]
fn add_finite(acc: &mut f64, term: f64, row: usize) -> Result<(), ProbError> {
    let sum = *acc + term;
    if !sum.is_finite() {
        return Err(non_finite_summary_input("weighted cross-product", row));
    }
    *acc = sum;
    Ok(())
}

/// One design row. A zero weight still screens the row; it does not drop a
/// non-finite entry, and it contributes nothing to the products.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
#[must_use = "a non-finite weight, offset, or row is refused"]
fn accumulate_summary_row(
    design: BayesDesignRef<'_>,
    row: usize,
    stats: &mut GaussianSufficientStats,
) -> Result<(), ProbError> {
    let weight = design.weights.map_or(1.0, |weights| weights[row]);
    if !weight.is_finite() {
        return Err(non_finite_summary_input("weight", row));
    }
    if weight < 0.0 {
        return Err(negative_weight(row));
    }
    let offset = design.offsets.map_or(0.0, |offsets| offsets[row]);
    if !offset.is_finite() {
        return Err(non_finite_summary_input("offset", row));
    }
    let y = design.y[row];
    if !y.is_finite() {
        return Err(non_finite_summary_input("design row", row));
    }
    let adjusted = y - offset;
    if !adjusted.is_finite() {
        return Err(non_finite_summary_input("design row", row));
    }
    let nrows = design.nrows;
    let p = stats.p;
    for c1 in 0..p {
        let x1 = design.x_colmajor[c1 * nrows + row];
        if !x1.is_finite() {
            return Err(non_finite_summary_input("design row", row));
        }
        add_finite(&mut stats.xtwy[c1], weight * x1 * adjusted, row)?;
        for c2 in c1..p {
            let x2 = design.x_colmajor[c2 * nrows + row];
            if c1 != c2 && !x2.is_finite() {
                return Err(non_finite_summary_input("design row", row));
            }
            let prod = weight * x1 * x2;
            add_finite(&mut stats.xtwx[c1 * p + c2], prod, row)?;
            if c1 != c2 {
                add_finite(&mut stats.xtwx[c2 * p + c1], prod, row)?;
            }
        }
    }
    add_finite(&mut stats.ytwy, weight * adjusted * adjusted, row)?;
    stats.n_eff += weight;
    Ok(())
}

/// Weighted residual sum of squares and score `X' W r` at `beta`.
///
/// `r_i = y_i - offset_i - x_i' β`, `RSS_w = Σ w_i r_i²`.
/// The library targets call `GaussianSufficientStats::rss_and_score`. The Laplace
/// tests still compare this row scan with that summary.
#[cfg_attr(not(test), allow(dead_code))]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
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
    prec: &CoefficientPrecision,
    beta: &[f64],
    p_diff: &mut [f64],
) -> Result<f64, ProbError> {
    let p = beta.len();
    if p_diff.len() != p || prec.len() != p || coef_prior.len() != p {
        return Err(ProbError::Shape { message: "prior quadratic length mismatch" });
    }
    let mut q = 0.0;
    match prec {
        CoefficientPrecision::Diagonal(d) => {
            for i in 0..p {
                let diff = beta[i] - coef_prior.mean[i];
                p_diff[i] = d[i] * diff;
                q += diff * p_diff[i];
            }
        }
        CoefficientPrecision::Dense { .. } => {
            let diff: Vec<f64> = (0..p).map(|i| beta[i] - coef_prior.mean[i]).collect();
            prec.mul_into(&diff, p_diff);
            for i in 0..p {
                q += diff[i] * p_diff[i];
            }
        }
    }
    if !q.is_finite() {
        return Err(ProbError::Numerical { message: "non-finite prior quadratic".into() });
    }
    Ok(q)
}

/// Known-σ² Gaussian target: state `q = β`.
#[derive(Clone, Debug)]
pub(crate) struct GaussianKnownTarget<'a> {
    design: BayesDesignRef<'a>,
    coef_prior: GaussianCoefficientPrior,
    prec: CoefficientPrecision,
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
    #[cfg(test)]
    pub fn new(
        design: BayesDesignRef<'a>,
        coef_prior: GaussianCoefficientPrior,
        sigma2: f64,
    ) -> Result<Self, ProbError> {
        let prec = CoefficientPrecision::Diagonal(coef_prior.precision());
        Self::with_precision(design, coef_prior, prec, sigma2)
    }

    /// Build a known-variance target with an explicit (possibly dense) `V0⁻¹`.
    ///
    /// # Errors
    ///
    /// Invalid σ², design / prior shape errors, or a precision of the wrong size.
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
    )]
    pub fn with_precision(
        design: BayesDesignRef<'a>,
        coef_prior: GaussianCoefficientPrior,
        prec: CoefficientPrecision,
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
        if prec.len() != ncols {
            return Err(ProbError::InvalidPrior { message: "coefficient precision size != ncols" });
        }
        let stats = GaussianSufficientStats::from_design(design, ncols)?;
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
}

impl PosteriorTarget for GaussianKnownTarget<'_> {
    fn dim(&self) -> usize {
        self.design.ncols
    }

    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        let p = self.dim();
        if q.len() != p || grad.len() != p {
            return Err(ProbError::Shape { message: "known target state/grad length != p" });
        }
        let rss = self.stats.rss_and_score(q, &mut self.xtwr)?;
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
pub(crate) struct GaussianInvGammaTarget<'a> {
    design: BayesDesignRef<'a>,
    coef_prior: GaussianCoefficientPrior,
    prec: CoefficientPrecision,
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
    /// Invalid `InvGamma` params or design / prior shape errors.
    #[cfg(test)]
    pub fn new(
        design: BayesDesignRef<'a>,
        coef_prior: GaussianCoefficientPrior,
        shape: f64,
        scale: f64,
    ) -> Result<Self, ProbError> {
        let prec = CoefficientPrecision::Diagonal(coef_prior.precision());
        Self::with_precision(design, coef_prior, prec, shape, scale)
    }

    /// Build a joint `(β, λ)` target with an explicit (possibly dense) `V0⁻¹`.
    ///
    /// # Errors
    ///
    /// Invalid `InvGamma` params, design / prior shape errors, or a precision of
    /// the wrong size.
    #[allow(
        clippy::neg_cmp_op_on_partial_ord,
        reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
    )]
    pub fn with_precision(
        design: BayesDesignRef<'a>,
        coef_prior: GaussianCoefficientPrior,
        prec: CoefficientPrecision,
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
        if prec.len() != ncols {
            return Err(ProbError::InvalidPrior { message: "coefficient precision size != ncols" });
        }
        let a_const = shape + 0.5 * (stats.n_eff + stats.p as f64);
        Ok(Self {
            design,
            coef_prior,
            prec,
            scale,
            a_const,
            stats,
            xtwr: vec![0.0; ncols],
            p_diff: vec![0.0; ncols],
        })
    }
}

impl PosteriorTarget for GaussianInvGammaTarget<'_> {
    fn dim(&self) -> usize {
        self.design.ncols + 1
    }

    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        let p = self.stats.p;
        if q.len() != p + 1 || grad.len() != p + 1 {
            return Err(ProbError::Shape { message: "InvGamma target state/grad length != p + 1" });
        }
        let beta = &q[..p];
        let lambda = q[p];
        if !lambda.is_finite() {
            return Err(ProbError::Numerical { message: "non-finite log_sigma2".into() });
        }
        let rss = self.stats.rss_and_score(beta, &mut self.xtwr)?;
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
#[cfg(test)]
pub(crate) fn gaussian_target_from_model(
    design: BayesDesignRef<'_>,
    coef_prior: GaussianCoefficientPrior,
    model: GaussianVarianceModel,
) -> Result<GaussianTarget<'_>, ProbError> {
    let prec = CoefficientPrecision::Diagonal(coef_prior.precision());
    gaussian_target_from_model_with_precision(design, coef_prior, prec, model)
}

/// [`gaussian_target_from_model`] with an explicit (possibly dense) `V0⁻¹`,
/// e.g. [`crate::PriorSet::coefficient_precision`].
///
/// # Errors
///
/// Propagates design / prior construction errors.
pub(crate) fn gaussian_target_from_model_with_precision(
    design: BayesDesignRef<'_>,
    coef_prior: GaussianCoefficientPrior,
    prec: CoefficientPrecision,
    model: GaussianVarianceModel,
) -> Result<GaussianTarget<'_>, ProbError> {
    match model {
        GaussianVarianceModel::Known { sigma2 } => Ok(GaussianTarget::Known(
            GaussianKnownTarget::with_precision(design, coef_prior, prec, sigma2)?,
        )),
        GaussianVarianceModel::InvGamma { shape, scale } => Ok(GaussianTarget::InvGamma(
            GaussianInvGammaTarget::with_precision(design, coef_prior, prec, shape, scale)?,
        )),
    }
}

/// Owned Gaussian target enum (avoids `dyn` across the HMC loop).
#[derive(Clone, Debug)]
pub(crate) enum GaussianTarget<'a> {
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

    /// Both Gaussian targets differentiate the dense prior kernel correctly.
    #[test]
    fn dense_prior_gradient_matches_finite_differences() {
        let (x, y) = tiny_design();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let coef = GaussianCoefficientPrior {
            mean: std::sync::Arc::from(vec![0.3, 0.8]),
            variance: std::sync::Arc::from(vec![0.5, 0.2]),
        };
        let prec = CoefficientPrecision::Dense {
            dim: 2,
            matrix: crate::linalg::invert_spd(&[0.5, -0.25, -0.25, 0.2], 2).unwrap(),
        };
        for model in [
            GaussianVarianceModel::Known { sigma2: 0.7 },
            GaussianVarianceModel::InvGamma { shape: 2.0, scale: 1.0 },
        ] {
            let mut t = gaussian_target_from_model_with_precision(
                design,
                coef.clone(),
                prec.clone(),
                model,
            )
            .unwrap();
            let q: Vec<f64> = (0..t.dim()).map(|i| 0.2 + 0.3 * i as f64).collect();
            let mut grad = vec![0.0; q.len()];
            t.logp_and_grad(&q, &mut grad).unwrap();
            for i in 0..q.len() {
                let h = 1e-6;
                let (mut up, mut dn) = (q.clone(), q.clone());
                up[i] += h;
                dn[i] -= h;
                let mut g = vec![0.0; q.len()];
                let fd = (t.logp_and_grad(&up, &mut g).unwrap()
                    - t.logp_and_grad(&dn, &mut g).unwrap())
                    / (2.0 * h);
                assert!(
                    (fd - grad[i]).abs() < 1e-5 * (1.0 + fd.abs()),
                    "{model:?} {i}: {fd} vs {}",
                    grad[i]
                );
            }
        }
    }

    #[test]
    fn weighted_summary_matches_explicit_cross_product() {
        let (x, y) = tiny_design();
        let weights = [0.5, 1.5, 0.0, 2.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: None,
        };
        let stats = GaussianSufficientStats::from_design(design, 2).unwrap();
        let mut xtwx = [0.0; 4];
        let mut xtwy = [0.0; 2];
        let mut ytwy = 0.0;
        let mut n_eff = 0.0;
        for r in 0..4 {
            let w = weights[r];
            n_eff += w;
            ytwy += w * y[r] * y[r];
            for c1 in 0..2 {
                let x1 = x[c1 * 4 + r];
                xtwy[c1] += w * x1 * y[r];
                for c2 in 0..2 {
                    xtwx[c1 * 2 + c2] += w * x1 * x[c2 * 4 + r];
                }
            }
        }
        assert!((stats.n_eff - n_eff).abs() < 1e-12);
        assert!((stats.ytwy - ytwy).abs() < 1e-12);
        for (got, expected) in stats.xtwx.iter().zip(&xtwx) {
            assert!((got - expected).abs() < 1e-12);
        }
        for (got, expected) in stats.xtwy.iter().zip(&xtwy) {
            assert!((got - expected).abs() < 1e-12);
        }

        let beta = [0.25, -0.5];
        let mut score = [0.0; 2];
        let rss = stats.rss_and_score(&beta, &mut score).unwrap();
        let mut explicit_rss = 0.0;
        let mut explicit_score = [0.0; 2];
        for r in 0..4 {
            let pred = x[r] * beta[0] + x[4 + r] * beta[1];
            let resid = y[r] - pred;
            explicit_rss += weights[r] * resid * resid;
            explicit_score[0] += weights[r] * x[r] * resid;
            explicit_score[1] += weights[r] * x[4 + r] * resid;
        }
        assert!((rss - explicit_rss).abs() < 1e-12);
        assert!((score[0] - explicit_score[0]).abs() < 1e-12);
        assert!((score[1] - explicit_score[1]).abs() < 1e-12);
    }

    #[test]
    fn offset_shifts_cross_product_like_centering() {
        let (x, y) = tiny_design();
        let weights = [1.0, 2.0, 1.0, 3.0];
        let mass: f64 = weights.iter().sum();
        let mean = y.iter().zip(&weights).map(|(yi, wi)| wi * yi).sum::<f64>() / mass;
        let offsets = [mean, mean, mean, mean];
        let raw = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: None,
        };
        let centered = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: Some(&offsets),
        };
        let raw_stats = GaussianSufficientStats::from_design(raw, 2).unwrap();
        let centered_stats = GaussianSufficientStats::from_design(centered, 2).unwrap();
        for (got, expected) in centered_stats.xtwx.iter().zip(&raw_stats.xtwx) {
            assert!((got - expected).abs() < 1e-12);
        }
        for c in 0..2 {
            let column_mass: f64 = (0..4).map(|r| weights[r] * x[c * 4 + r]).sum();
            let shifted = raw_stats.xtwy[c] - mean * column_mass;
            assert!((centered_stats.xtwy[c] - shifted).abs() < 1e-12);
            let mut explicit = 0.0;
            for r in 0..4 {
                explicit += weights[r] * x[c * 4 + r] * (y[r] - mean);
            }
            assert!((centered_stats.xtwy[c] - explicit).abs() < 1e-12);
        }
        // Column 0 is an intercept, so centering removes its cross-product.
        assert!(centered_stats.xtwy[0].abs() < 1e-12);
    }

    #[test]
    fn non_finite_weight_is_refused() {
        let (x, y) = tiny_design();
        let mut weights = [1.0, 1.0, 1.0, 1.0];
        weights[1] = f64::NAN;
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: None,
        };
        let err = GaussianSufficientStats::from_design(design, 2).unwrap_err();
        let ProbError::Numerical { message } = err else {
            panic!("expected a numerical refusal");
        };
        assert!(message.contains("non-finite weight at row 1"), "{message}");
        assert!(
            message.contains("next attempt: drop that row from the summary and rebuild"),
            "{message}"
        );

        weights[1] = f64::INFINITY;
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: None,
        };
        let err = GaussianSufficientStats::from_design(design, 2).unwrap_err();
        let ProbError::Numerical { message } = err else {
            panic!("expected a numerical refusal");
        };
        assert!(message.contains("non-finite weight at row 1"), "{message}");

        let weights = [1.0, 0.0, 1.0, 1.0];
        let mut offsets = [0.0, 0.0, 0.0, 0.0];
        offsets[2] = f64::NAN;
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: Some(&offsets),
        };
        let err = GaussianSufficientStats::from_design(design, 2).unwrap_err();
        let ProbError::Numerical { message } = err else {
            panic!("expected a numerical refusal");
        };
        assert!(message.contains("non-finite offset at row 2"), "{message}");
        assert!(message.contains("drop that row from the summary and rebuild"), "{message}");

        let mut x_bad = x.clone();
        x_bad[4 + 1] = f64::NAN;
        let design = BayesDesignRef {
            x_colmajor: &x_bad,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: None,
        };
        let err = GaussianSufficientStats::from_design(design, 2).unwrap_err();
        let ProbError::Numerical { message } = err else {
            panic!("expected a numerical refusal");
        };
        assert!(message.contains("non-finite design row at row 1"), "{message}");
        assert!(message.contains("drop that row from the summary and rebuild"), "{message}");
    }
}
