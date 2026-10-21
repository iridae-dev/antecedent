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

    /// Evaluate the density-difference kernel and write `∇_q log π` into `grad`.
    ///
    /// The value omits additive terms that do not depend on `q`. Differences of
    /// it match the normalized log joint, which is what leapfrog and Laplace
    /// compare. It is not model evidence: the normalized log joint, including
    /// those terms, is [`GaussianTarget::log_evidence`].
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
///
/// The raw products cancel catastrophically when `y` sits far from its spread
/// (`ỹ' W ỹ` and `β' X' W ỹ` agree to every digit the residual would need), so
/// the summary also stores the residual products at an anchor `β₀`, the
/// weighted least-squares solution when the Gram matrix is positive definite
/// and zero otherwise. Evaluation expands about that anchor instead.
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
    /// Anchor `β₀` the residual products below are taken at (`p`).
    anchor: Vec<f64>,
    /// `X' W r₀` with `r₀ = ỹ − X β₀` (`p`).
    anchor_score: Vec<f64>,
    /// `r₀' W r₀`.
    anchor_rss: f64,
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
        let mut stats = Self {
            n_eff: 0.0,
            p: ncols,
            xtwx: vec![0.0; gram],
            xtwy: vec![0.0; ncols],
            ytwy: 0.0,
            anchor: vec![0.0; ncols],
            anchor_score: vec![0.0; ncols],
            anchor_rss: 0.0,
        };
        let blocked_gram = design.weights.is_none()
            && design.nrows >= 256
            && ncols >= 16
            && design.nrows.saturating_mul(ncols).saturating_mul(ncols) >= 524_288
            && design.x_colmajor[..design.nrows * ncols]
                .iter()
                .all(|v| v.is_finite() && v.abs() < (f64::MAX / design.nrows as f64).sqrt());
        for row in 0..design.nrows {
            accumulate_summary_row(design, row, &mut stats, !blocked_gram)?;
        }
        if blocked_gram {
            antecedent_stats::gram::form_xtx(
                design.x_colmajor,
                design.nrows,
                ncols,
                &mut stats.xtwx,
            );
        }
        if !(stats.n_eff > 0.0) || !stats.n_eff.is_finite() {
            return Err(ProbError::Numerical {
                message: "non-positive or non-finite effective sample size".into(),
            });
        }
        stats.anchor_at_least_squares(design)?;
        Ok(stats)
    }

    /// Set the anchor to the weighted least-squares solution (one refinement
    /// step) and store the residual products there. A Gram matrix that is not
    /// positive definite keeps the zero anchor.
    fn anchor_at_least_squares(&mut self, design: BayesDesignRef<'_>) -> Result<(), ProbError> {
        let p = self.p;
        let mut factor = vec![0.0; p * p];
        if crate::linalg::cholesky_spd_into(&self.xtwx, p, &mut factor).is_ok() {
            let mut scratch = vec![0.0; p];
            let mut solved = vec![0.0; p];
            crate::linalg::solve_chol_into(&factor, p, &self.xtwy, &mut scratch, &mut solved);
            if solved.iter().all(|b| b.is_finite()) {
                self.anchor.copy_from_slice(&solved);
            }
            self.anchor_rss = residual_products(design, &self.anchor, &mut self.anchor_score)?;
            crate::linalg::solve_chol_into(
                &factor,
                p,
                &self.anchor_score,
                &mut scratch,
                &mut solved,
            );
            if solved.iter().all(|b| b.is_finite()) {
                for (b, step) in self.anchor.iter_mut().zip(&solved) {
                    *b += step;
                }
            }
        }
        self.anchor_rss = residual_products(design, &self.anchor, &mut self.anchor_score)?;
        Ok(())
    }

    /// Residual sum of squares and score `X' W r` from the stored cross-products.
    ///
    /// With `δ = β − β₀` and the stored anchor products,
    /// `RSS = r₀' W r₀ − 2 δ' X' W r₀ + δ' X' W X δ` and
    /// `X' W r = X' W r₀ − X' W X δ`. No row is revisited. A negative result
    /// within rounding of zero is clamped to zero; a larger one is refused.
    ///
    /// # Errors
    ///
    /// `beta` / `xtwr` length mismatch, a non-finite residual sum of squares, or
    /// a negative one beyond rounding.
    #[must_use = "a non-finite residual sum of squares is refused"]
    pub(crate) fn rss_and_score(&self, beta: &[f64], xtwr: &mut [f64]) -> Result<f64, ProbError> {
        let p = self.p;
        if beta.len() != p || xtwr.len() != p {
            return Err(ProbError::Shape { message: "sufficient-stat score length != p" });
        }
        let delta: Vec<f64> = beta.iter().zip(&self.anchor).map(|(b, b0)| b - b0).collect();
        for (i, score) in xtwr.iter_mut().enumerate() {
            let mut acc = self.anchor_score[i];
            for (&gram, &d) in self.xtwx[i * p..(i + 1) * p].iter().zip(&delta) {
                acc -= gram * d;
            }
            *score = acc;
        }
        let mut rss = self.anchor_rss;
        let mut scale = self.anchor_rss;
        for ((&d, &cross), &score) in delta.iter().zip(&self.anchor_score).zip(xtwr.iter()) {
            rss -= d * (cross + score);
            scale += (d * (cross + score)).abs();
        }
        if !rss.is_finite() || xtwr.iter().any(|score| !score.is_finite()) {
            return Err(ProbError::Numerical {
                message: "non-finite residual sum of squares from cross-products".into(),
            });
        }
        if rss < 0.0 {
            if rss < -1e-9 * scale {
                return Err(ProbError::Numerical {
                    message: "negative residual sum of squares from cross-products".into(),
                });
            }
            return Ok(0.0);
        }
        Ok(rss)
    }
}

/// `r' W r` and `X' W r` at `beta`, where `r = y − offset − X beta`, scanned row
/// by row so no raw cross-product has to cancel.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn residual_products(
    design: BayesDesignRef<'_>,
    beta: &[f64],
    xtwr: &mut [f64],
) -> Result<f64, ProbError> {
    let nrows = design.nrows;
    xtwr.fill(0.0);
    let mut rss = 0.0;
    for row in 0..nrows {
        let weight = design.weights.map_or(1.0, |weights| weights[row]);
        let offset = design.offsets.map_or(0.0, |offsets| offsets[row]);
        let mut resid = design.y[row] - offset;
        for (c, &coef) in beta.iter().enumerate() {
            resid -= design.x_colmajor[c * nrows + row] * coef;
        }
        add_finite(&mut rss, weight * resid * resid, row)?;
        for c in 0..beta.len() {
            xtwr[c] += weight * design.x_colmajor[c * nrows + row] * resid;
        }
    }
    if xtwr.iter().any(|score| !score.is_finite()) {
        return Err(non_finite_summary_input("weighted cross-product", nrows));
    }
    Ok(rss)
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
    accumulate_gram: bool,
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
        for c2 in c1..if accumulate_gram { p } else { c1 } {
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
/// The library targets call `GaussianSufficientStats::rss_and_score`. The tests
/// compare this row scan with that summary.
#[cfg(test)]
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
        let offset = design.offsets.map_or(0.0, |offsets| offsets[r]);
        if !offset.is_finite() {
            return Err(non_finite_summary_input("offset", r));
        }
        let y = design.y[r];
        if !y.is_finite() {
            return Err(non_finite_summary_input("design row", r));
        }
        let adjusted = y - offset;
        if !adjusted.is_finite() {
            return Err(non_finite_summary_input("design row", r));
        }
        let mut pred = 0.0;
        for c in 0..ncols {
            let x = design.x_colmajor[c * nrows + r];
            if !x.is_finite() {
                return Err(non_finite_summary_input("design row", r));
            }
            pred += x * beta[c];
        }
        let resid = adjusted - pred;
        let weight = design.weights.map_or(1.0, |weights| weights[r]);
        if !weight.is_finite() {
            return Err(non_finite_summary_input("weight", r));
        }
        if weight < 0.0 {
            return Err(negative_weight(r));
        }
        rss += weight * resid * resid;
        for c in 0..ncols {
            xtwr[c] += weight * design.x_colmajor[c * nrows + r] * resid;
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

impl GaussianKnownTarget<'_> {
    /// Normalized log joint of `(y, β)` at known `σ²`.
    ///
    /// This is [`PosteriorTarget::logp_and_grad`] plus the Gaussian normalizing
    /// constants in `β` and in the weighted observations. The gradient matches
    /// the density-difference kernel. Weights are frequency weights: `n` in the
    /// constant is `n_eff = Σ w_i`.
    ///
    /// # Errors
    ///
    /// The same failures as the density-difference kernel, or a prior precision
    /// whose log determinant is not finite.
    #[allow(
        dead_code,
        reason = "the HMC kernel stays the density difference; tests call the normalized joint"
    )]
    pub(crate) fn log_evidence(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        let kernel = self.logp_and_grad(q, grad)?;
        let offset = known_evidence_offset(
            self.stats.n_eff,
            self.stats.p,
            self.sigma2,
            log_det_v0(&self.prec)?,
        )?;
        finish_evidence(kernel, offset)
    }
}

/// Inverse-gamma residual target: state `q = [β, λ]` with `λ = log(σ²)`.
#[derive(Clone, Debug)]
pub(crate) struct GaussianInvGammaTarget<'a> {
    design: BayesDesignRef<'a>,
    coef_prior: GaussianCoefficientPrior,
    prec: CoefficientPrecision,
    scale: f64,
    /// `A = a₀ + (n_eff + p) / 2`. `n_eff = Σ w_i` treats the weights as
    /// frequency weights: a weight of `k` counts as `k` observations in the
    /// residual-variance posterior.
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
        // Density-difference kernel: -A λ - B(β) e^{-λ}. The normalizing
        // constants that do not depend on (β, λ) are added by `log_evidence`.
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

impl GaussianInvGammaTarget<'_> {
    /// Normalized log joint of `(y, β, λ)` with `λ = log σ²`.
    ///
    /// This is [`PosteriorTarget::logp_and_grad`] plus the constants of the
    /// Gaussian likelihood, the conditional coefficient prior, and the
    /// inverse-gamma prior on `σ²` after the change of variables to `λ`. The
    /// gradient matches the density-difference kernel. `n` in the Gaussian
    /// constant is the frequency-weighted `n_eff`.
    ///
    /// # Errors
    ///
    /// The same failures as the density-difference kernel, or a prior precision
    /// or inverse-gamma term whose constant is not finite.
    #[allow(
        dead_code,
        reason = "the HMC kernel stays the density difference; tests call the normalized joint"
    )]
    pub(crate) fn log_evidence(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        let kernel = self.logp_and_grad(q, grad)?;
        let shape = self.a_const - 0.5 * (self.stats.n_eff + self.stats.p as f64);
        let offset = invgamma_evidence_offset(
            self.stats.n_eff,
            self.stats.p,
            shape,
            self.scale,
            log_det_v0(&self.prec)?,
        )?;
        finish_evidence(kernel, offset)
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

impl GaussianTarget<'_> {
    /// Normalized log joint used as model evidence.
    ///
    /// [`PosteriorTarget::logp_and_grad`] stays the density-difference kernel.
    /// This adds the constants that kernel omits. The gradient is the same.
    ///
    /// # Errors
    ///
    /// Propagates the kernel and the normalizing-constant failures.
    #[allow(
        dead_code,
        reason = "the HMC kernel stays the density difference; tests call the normalized joint"
    )]
    pub(crate) fn log_evidence(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        match self {
            Self::Known(t) => t.log_evidence(q, grad),
            Self::InvGamma(t) => t.log_evidence(q, grad),
        }
    }
}

/// `log det V0` from a precision `V0⁻¹`.
#[allow(
    dead_code,
    reason = "the HMC kernel stays the density difference; tests call the normalized joint"
)]
fn log_det_v0(prec: &CoefficientPrecision) -> Result<f64, ProbError> {
    let det = match prec {
        CoefficientPrecision::Diagonal(diag) => {
            let mut sum = 0.0;
            for &precision in diag {
                if precision <= 0.0 || !precision.is_finite() {
                    return Err(ProbError::Numerical {
                        message: "coefficient precision is not a finite positive scale".into(),
                    });
                }
                sum -= precision.ln();
            }
            sum
        }
        CoefficientPrecision::Dense { dim, matrix } => {
            let chol = antecedent_stats::cholesky_spd(matrix, *dim).ok_or_else(|| {
                ProbError::Numerical {
                    message: "coefficient precision is not positive definite".into(),
                }
            })?;
            -antecedent_stats::chol_log_det(&chol, *dim)
        }
    };
    if !det.is_finite() {
        return Err(ProbError::Numerical { message: "log determinant of V0 is not finite".into() });
    }
    Ok(det)
}

#[allow(
    dead_code,
    reason = "the HMC kernel stays the density difference; tests call the normalized joint"
)]
fn known_evidence_offset(
    n_eff: f64,
    p: usize,
    sigma2: f64,
    log_det_v0: f64,
) -> Result<f64, ProbError> {
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let offset = -0.5 * (n_eff + p as f64) * (ln_2pi + sigma2.ln()) - 0.5 * log_det_v0;
    finite_offset(offset)
}

#[allow(
    dead_code,
    reason = "the HMC kernel stays the density difference; tests call the normalized joint"
)]
fn invgamma_evidence_offset(
    n_eff: f64,
    p: usize,
    shape: f64,
    scale: f64,
    log_det_v0: f64,
) -> Result<f64, ProbError> {
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let offset = -0.5 * (n_eff + p as f64) * ln_2pi - 0.5 * log_det_v0 + shape * scale.ln()
        - antecedent_stats::ln_gamma(shape);
    finite_offset(offset)
}

#[allow(
    dead_code,
    reason = "the HMC kernel stays the density difference; tests call the normalized joint"
)]
fn finite_offset(offset: f64) -> Result<f64, ProbError> {
    if !offset.is_finite() {
        return Err(ProbError::Numerical {
            message: "Gaussian evidence constant is not finite".into(),
        });
    }
    Ok(offset)
}

fn finish_evidence(kernel: f64, offset: f64) -> Result<f64, ProbError> {
    let logp = kernel + offset;
    if !logp.is_finite() {
        return Err(ProbError::Numerical { message: "non-finite Gaussian evidence".into() });
    }
    Ok(logp)
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

    /// Row-wise `RSS` and `X' W r`, with the offset removed before `x'β`.
    ///
    /// Putting the offset into the predictor first cancels a shared `1e12` level.
    /// [`rss_and_xtwr`] keeps that order for the Laplace checks; this reference is
    /// the centered scan the summary is compared with.
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    fn row_wise_rss_and_score(design: BayesDesignRef<'_>, beta: &[f64]) -> (f64, Vec<f64>) {
        let nrows = design.nrows;
        let p = beta.len();
        let mut xtwr = vec![0.0; p];
        let mut rss = 0.0;
        for row in 0..nrows {
            let weight = design.weights.map_or(1.0, |weights| weights[row]);
            let offset = design.offsets.map_or(0.0, |offsets| offsets[row]);
            let mut pred = 0.0;
            for col in 0..p {
                pred += design.x_colmajor[col * nrows + row] * beta[col];
            }
            let resid = (design.y[row] - offset) - pred;
            rss += weight * resid * resid;
            for col in 0..p {
                xtwr[col] += weight * design.x_colmajor[col * nrows + row] * resid;
            }
        }
        (rss, xtwr)
    }

    fn row_wise_prior(mean: &[f64], prec: &CoefficientPrecision, beta: &[f64]) -> (f64, Vec<f64>) {
        let p = beta.len();
        let diff: Vec<f64> = beta.iter().zip(mean).map(|(coef, m)| coef - m).collect();
        let mut p_diff = vec![0.0; p];
        match prec {
            CoefficientPrecision::Diagonal(diag) => {
                for (slot, (precision, gap)) in p_diff.iter_mut().zip(diag.iter().zip(&diff)) {
                    *slot = precision * gap;
                }
            }
            CoefficientPrecision::Dense { matrix, .. } => {
                for (i, slot) in p_diff.iter_mut().enumerate() {
                    let row = &matrix[i * p..(i + 1) * p];
                    *slot = row.iter().zip(&diff).map(|(entry, gap)| entry * gap).sum();
                }
            }
        }
        let quad = diff.iter().zip(&p_diff).map(|(gap, precision_gap)| gap * precision_gap).sum();
        (quad, p_diff)
    }

    fn assert_close(got: f64, expected: f64, tol: f64, what: &str) {
        let gap = (got - expected).abs();
        assert!(gap <= tol, "{what}: summary {got} vs row-wise {expected} (gap {gap}, tol {tol})");
    }

    struct TargetCase<'a> {
        design: BayesDesignRef<'a>,
        prior: GaussianCoefficientPrior,
        prec: CoefficientPrecision,
        beta: &'a [f64],
        sigma2: f64,
        shape: f64,
        scale: f64,
        lambda: f64,
        tol: f64,
    }

    fn assert_summary_matches_rows(case: &TargetCase<'_>) {
        let (rss, xtwr) = row_wise_rss_and_score(case.design, case.beta);
        let (quad, p_diff) = row_wise_prior(&case.prior.mean, &case.prec, case.beta);
        let inv_s2 = 1.0 / case.sigma2;
        let mut known = gaussian_target_from_model_with_precision(
            case.design,
            case.prior.clone(),
            case.prec.clone(),
            GaussianVarianceModel::Known { sigma2: case.sigma2 },
        )
        .unwrap();
        let mut grad = vec![0.0; known.dim()];
        let logp = known.logp_and_grad(case.beta, &mut grad).unwrap();
        assert_close(logp, -0.5 * inv_s2 * (rss + quad), case.tol, "known log density");
        for i in 0..case.beta.len() {
            assert_close(grad[i], inv_s2 * (xtwr[i] - p_diff[i]), case.tol, "known gradient");
        }

        let n_eff = match case.design.weights {
            Some(weights) => weights.iter().sum(),
            None => case.design.nrows as f64,
        };
        let a_const = case.shape + 0.5 * (n_eff + case.beta.len() as f64);
        let b_beta = case.scale + 0.5 * (rss + quad);
        let exp_neg = (-case.lambda).exp();
        let mut state = case.beta.to_vec();
        state.push(case.lambda);
        let mut joint = gaussian_target_from_model_with_precision(
            case.design,
            case.prior.clone(),
            case.prec.clone(),
            GaussianVarianceModel::InvGamma { shape: case.shape, scale: case.scale },
        )
        .unwrap();
        let mut jgrad = vec![0.0; joint.dim()];
        let jlogp = joint.logp_and_grad(&state, &mut jgrad).unwrap();
        assert_close(
            jlogp,
            -a_const * case.lambda - b_beta * exp_neg,
            case.tol,
            "joint log density",
        );
        for i in 0..case.beta.len() {
            assert_close(
                jgrad[i],
                exp_neg * (xtwr[i] - p_diff[i]),
                case.tol,
                "joint coefficient gradient",
            );
        }
        assert_close(
            jgrad[case.beta.len()],
            -a_const + b_beta * exp_neg,
            case.tol,
            "joint log-variance gradient",
        );
    }

    #[test]
    fn summary_log_density_matches_row_wise_reference() {
        let (x, y) = tiny_design();
        let weights = [0.5, 1.5, 0.0, 2.0];
        let offsets = [0.25, -0.5, 1.0, 0.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: Some(&offsets),
        };
        let prior = GaussianCoefficientPrior::shared(2, 0.3, 2.5).unwrap();
        let prec = CoefficientPrecision::Diagonal(prior.precision());
        let beta = [0.2, 1.4];
        assert_summary_matches_rows(&TargetCase {
            design,
            prior,
            prec,
            beta: &beta,
            sigma2: 2.5,
            shape: 2.0,
            scale: 1.5,
            lambda: 0.35,
            tol: 1e-9,
        });
    }

    #[test]
    fn dense_prior_summary_matches_row_wise_reference() {
        let (x, y) = tiny_design();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = GaussianCoefficientPrior {
            mean: std::sync::Arc::from(vec![0.3, 0.8]),
            variance: std::sync::Arc::from(vec![0.5, 0.2]),
        };
        let prec = CoefficientPrecision::Dense {
            dim: 2,
            matrix: crate::linalg::invert_spd(&[0.5, -0.25, -0.25, 0.2], 2).unwrap(),
        };
        let beta = [0.4, -0.2];
        assert_summary_matches_rows(&TargetCase {
            design,
            prior,
            prec,
            beta: &beta,
            sigma2: 0.7,
            shape: 2.0,
            scale: 1.0,
            lambda: -0.2,
            tol: 1e-9,
        });
    }

    #[test]
    fn rank_deficient_summary_matches_row_wise_reference() {
        // Second column is twice the first, so `X' W X` is singular. A proper
        // coefficient prior still defines the log density and gradient.
        let x = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let y = vec![1.0, 2.0, 0.5, 1.5];
        let weights = [1.0, 0.25, 2.0, 1.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 4,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: None,
        };
        let stats = GaussianSufficientStats::from_design(design, 2).unwrap();
        let det = stats.xtwx[0] * stats.xtwx[3] - stats.xtwx[1] * stats.xtwx[2];
        assert!(det.abs() <= 1e-12, "Gram det {det}");
        let prior = GaussianCoefficientPrior::shared(2, 0.0, 1.0).unwrap();
        let prec = CoefficientPrecision::Diagonal(prior.precision());
        let beta = [0.4, -0.2];
        assert_summary_matches_rows(&TargetCase {
            design,
            prior,
            prec,
            beta: &beta,
            sigma2: 1.2,
            shape: 3.0,
            scale: 0.5,
            lambda: 0.1,
            tol: 1e-9,
        });
    }

    #[test]
    fn shared_offset_matches_row_reference_at_rounding_scale() {
        let offset = 1e12;
        let n = 6;
        let mut x = vec![1.0; n];
        x.extend([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let weights = [1.0, 0.5, 1.5, 0.0, 2.0, 1.0];
        let offsets = vec![offset; n];
        let prior = GaussianCoefficientPrior::shared(2, -0.1, 1.5).unwrap();
        let prec = CoefficientPrecision::Diagonal(prior.precision());

        // Deltas sit at `ulp(1e12)`, so `y − offset` is the rounding of that level.
        let deltas = [1.0e-4, -2.0e-4, 0.5e-4, 1.5e-4, -0.5e-4, 0.0];
        let y: Vec<f64> = deltas.iter().map(|delta| offset + delta).collect();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: Some(&offsets),
        };
        let stats = GaussianSufficientStats::from_design(design, 2).unwrap();
        let rounding = offset * f64::EPSILON;
        let max_cross = stats.xtwy.iter().fold(0.0_f64, |max, value| max.max(value.abs()));
        assert!(max_cross > 0.0 && max_cross <= rounding * 8.0, "centered moment {max_cross}");
        assert!(stats.ytwy.abs() <= rounding, "centered square {}", stats.ytwy);
        let beta = [0.25, -0.1];
        assert_summary_matches_rows(&TargetCase {
            design,
            prior: prior.clone(),
            prec: prec.clone(),
            beta: &beta,
            sigma2: 1.0,
            shape: 2.0,
            scale: 1.0,
            lambda: 0.0,
            tol: 1e-9,
        });

        // An O(1) signal on the same offset cancels in an uncentered residual.
        // The summary log density still matches the centered row scan.
        let eps = [-1.5, 0.5, -0.25, 1.25, -0.75, 0.75];
        let y: Vec<f64> = eps
            .iter()
            .enumerate()
            .map(|(i, shock)| offset + 1.0 + 2.0 * i as f64 + shock)
            .collect();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: Some(&offsets),
        };
        let beta = [0.4, 1.7];
        assert_summary_matches_rows(&TargetCase {
            design,
            prior,
            prec,
            beta: &beta,
            sigma2: 1.7,
            shape: 2.5,
            scale: 0.8,
            lambda: -0.3,
            tol: 1e-8,
        });
    }

    #[test]
    fn row_scanner_centers_before_the_predictor_and_refuses_a_negative_weight() {
        let offset = 1.0e12;
        let x = vec![1.0, 1.0, 0.0, 1.0];
        let y = vec![offset + 0.25, offset - 0.5];
        let offsets = vec![offset, offset];
        let weights = [1.0, 2.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 2,
            ncols: 2,
            y: &y,
            weights: Some(&weights),
            offsets: Some(&offsets),
        };
        let mut score = [0.0, 0.0];
        let rss = rss_and_xtwr(design, &[0.0, 0.0], &mut score).unwrap();
        let centered: f64 = 0.25 * 0.25 + 2.0 * 0.5 * 0.5;
        assert_eq!(rss.to_bits(), centered.to_bits());
        let negative = [-1.0, 1.0];
        let refused = BayesDesignRef { weights: Some(&negative), ..design };
        let error = rss_and_xtwr(refused, &[0.0, 0.0], &mut score).unwrap_err();
        assert!(error.to_string().contains("negative weight"));
    }

    /// Far-from-zero outcome with a deterministic spread of about one. Raw
    /// cross-products lose every digit of the residual here.
    fn far_from_zero_design() -> (Vec<f64>, Vec<f64>) {
        let n = 1000;
        let mut x = vec![1.0; n];
        let mut y = Vec::with_capacity(n);
        let mut state = 12345_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
        };
        for i in 0..n {
            let xi = (i as f64 / n as f64) * 2.0 - 1.0;
            x.push(xi);
            y.push(1.0e8 + 0.5 * xi + 3.0 * next());
        }
        (x, y)
    }

    #[test]
    fn summary_residual_matches_the_row_scan_far_from_zero() {
        let (x, y) = far_from_zero_design();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 1000,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let stats = GaussianSufficientStats::from_design(design, 2).unwrap();
        for beta in [[1.0e8, 0.0], [1.0e8 + 0.1, 0.5], [1.0e8 - 2.0, -1.0]] {
            let mut got_score = [0.0; 2];
            let mut want_score = [0.0; 2];
            let got = stats.rss_and_score(&beta, &mut got_score).unwrap();
            let want = rss_and_xtwr(design, &beta, &mut want_score).unwrap();
            assert!(want > 100.0, "the reference residual must be non-trivial: {want}");
            assert!((got - want).abs() < 1e-9 * want, "rss {got} vs row scan {want}");
            for c in 0..2 {
                assert!(
                    (got_score[c] - want_score[c]).abs() < 1e-9 * (1.0 + want_score[c].abs()),
                    "score {c}: {} vs {}",
                    got_score[c],
                    want_score[c]
                );
            }
        }
    }

    #[test]
    fn targets_match_the_row_scan_far_from_zero() {
        let (x, y) = far_from_zero_design();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 1000,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = GaussianCoefficientPrior {
            mean: std::sync::Arc::from(vec![1.0e8, 0.0]),
            variance: std::sync::Arc::from(vec![4.0, 4.0]),
        };
        let beta = [1.0e8 + 0.1, 0.5];
        let prec = CoefficientPrecision::Diagonal(prior.precision());
        let mut score = [0.0; 2];
        let rss = rss_and_xtwr(design, &beta, &mut score).unwrap();
        let mut p_diff = [0.0; 2];
        let quad = prior_quadratic(&prior, &prec, &beta, &mut p_diff).unwrap();

        let sigma2 = 2.0;
        let mut known = GaussianKnownTarget::new(design, prior.clone(), sigma2).unwrap();
        let mut grad = [0.0; 2];
        let logp = known.logp_and_grad(&beta, &mut grad).unwrap();
        let want_logp = -0.5 * (rss + quad) / sigma2;
        assert!((logp - want_logp).abs() < 1e-9 * want_logp.abs(), "{logp} vs {want_logp}");
        for c in 0..2 {
            let want = (score[c] - p_diff[c]) / sigma2;
            assert!(
                (grad[c] - want).abs() < 1e-9 * (1.0 + want.abs()),
                "{c}: {} vs {want}",
                grad[c]
            );
        }

        let (shape, scale, lambda) = (2.0, 1.5, 0.4);
        let mut joint = GaussianInvGammaTarget::new(design, prior, shape, scale).unwrap();
        let q = [beta[0], beta[1], lambda];
        let mut grad = [0.0; 3];
        let logp = joint.logp_and_grad(&q, &mut grad).unwrap();
        let a = shape + 0.5 * (1000.0 + 2.0);
        let b = scale + 0.5 * (rss + quad);
        let want_logp = -a * lambda - b * (-lambda).exp();
        assert!((logp - want_logp).abs() < 1e-9 * want_logp.abs(), "{logp} vs {want_logp}");
        for c in 0..2 {
            let want = (-lambda).exp() * (score[c] - p_diff[c]);
            assert!(
                (grad[c] - want).abs() < 1e-9 * (1.0 + want.abs()),
                "{c}: {} vs {want}",
                grad[c]
            );
        }
        let want_lambda = -a + b * (-lambda).exp();
        assert!((grad[2] - want_lambda).abs() < 1e-9 * (1.0 + want_lambda.abs()));
    }

    /// The density-difference kernel stays the pinned quadratic. Evidence adds
    /// the normalizing constants, so it is not that kernel.
    #[test]
    fn evidence_adds_normalizing_constants_without_moving_the_kernel() {
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
        let q = [0.2, 1.4];
        let mut known = GaussianKnownTarget::new(design, prior.clone(), 2.5).unwrap();
        let mut kernel_grad = [0.0; 2];
        let kernel = known.logp_and_grad(&q, &mut kernel_grad).unwrap();
        let mut evidence_grad = [0.0; 2];
        let known_evidence = known.log_evidence(&q, &mut evidence_grad).unwrap();
        assert_eq!(kernel_grad, evidence_grad);
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_det_v0 = 2.0 * 4.0_f64.ln();
        let offset = -0.5 * (4.0 + 2.0) * (ln_2pi + 2.5_f64.ln()) - 0.5 * log_det_v0;
        assert!(
            (known_evidence - (kernel + offset)).abs() < 1e-9,
            "{known_evidence} vs {}",
            kernel + offset
        );
        assert!(
            (known_evidence - kernel).abs() > 1.0,
            "evidence collapsed to the density difference"
        );

        let mut joint = GaussianInvGammaTarget::new(design, prior, 2.0, 1.5).unwrap();
        let state = [0.2, 1.4, 0.35];
        let mut kernel_grad = [0.0; 3];
        let kernel = joint.logp_and_grad(&state, &mut kernel_grad).unwrap();
        let mut evidence_grad = [0.0; 3];
        let evidence = joint.log_evidence(&state, &mut evidence_grad).unwrap();
        assert_eq!(kernel_grad, evidence_grad);
        let offset = -0.5 * (4.0 + 2.0) * ln_2pi - 0.5 * log_det_v0 + 2.0 * 1.5_f64.ln()
            - antecedent_stats::ln_gamma(2.0);
        assert!((evidence - (kernel + offset)).abs() < 1e-9, "{evidence} vs {}", kernel + offset);
        assert!((evidence - kernel).abs() > 1.0, "evidence collapsed to the density difference");
        let mut wrapped = GaussianTarget::Known(known);
        let mut wrapped_grad = [0.0; 2];
        let wrapped_evidence = wrapped.log_evidence(&q, &mut wrapped_grad).unwrap();
        assert!((wrapped_evidence - known_evidence).abs() < 1e-9);
    }
}
