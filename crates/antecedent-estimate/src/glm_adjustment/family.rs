//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Floor on a probit probability and on a log-mean inside Fisher weights and the NB moment.
pub(super) const FISHER_MEAN_FLOOR: f64 = 1e-12;

/// Bernoulli mean clamp for the sandwich score. Wider than [`FISHER_MEAN_FLOOR`]: the score
/// divides by `V(μ)`, which blows up at the boundary, while the information weight does not.
pub(super) const SCORE_PROBABILITY_CLAMP: f64 = 1e-9;

/// Floor on a variance or a probit density inside the sandwich score.
pub(super) const SCORE_VARIANCE_FLOOR: f64 = 1e-12;

/// Mean derivative, Fisher weight, and sandwich score for one GLM family.
///
/// The response mean is [`GlmFamily::mean_from_eta`]. The two clamps above stay on
/// different paths: collapsing them changes probit information weights and Bernoulli scores.
#[derive(Clone, Copy)]
pub(crate) struct FamilyKernel {
    family: GlmFamily,
}

impl FamilyKernel {
    pub(crate) const fn new(family: GlmFamily) -> Self {
        Self { family }
    }

    /// Response-scale derivative `dμ/dη` at `eta`.
    pub(crate) fn mean_derivative(self, eta: f64) -> f64 {
        match self.family {
            GlmFamily::BinomialLogit => {
                let mu = antecedent_kernels::sigmoid(eta);
                mu * (1.0 - mu)
            }
            GlmFamily::BinomialProbit => {
                // φ(η) = dμ/dη for probit.
                antecedent_kernels::norm_pdf(eta)
            }
            GlmFamily::GaussianIdentity => 1.0,
            GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => eta.exp(),
        }
    }

    /// IRLS / Fisher information weight `W_ii` at `eta`.
    ///
    /// For canonical Bernoulli/logit and Poisson this equals `dμ/dη`. For probit the
    /// Bernoulli Fisher weight is `φ(η)² / (μ(1−μ))`, not `φ(η)`. For NB2 with log
    /// link, `V = μ + α μ²` so `w = (μ')² / V = μ / (1 + α μ)`.
    pub(crate) fn fisher_weight(self, eta: f64, nb_alpha: f64) -> f64 {
        match self.family {
            GlmFamily::BinomialProbit => {
                let phi = self.mean_derivative(eta);
                // Φ(η), clamped away from {0, 1} for numerical stability.
                let mu = antecedent_kernels::norm_cdf(eta)
                    .clamp(FISHER_MEAN_FLOOR, 1.0 - FISHER_MEAN_FLOOR);
                (phi * phi) / (mu * (1.0 - mu))
            }
            GlmFamily::NegativeBinomial => {
                let mu = eta.exp().max(FISHER_MEAN_FLOOR);
                let alpha = nb_alpha.max(0.0);
                mu / (1.0 + alpha * mu)
            }
            _ => self.mean_derivative(eta),
        }
    }

    /// Sandwich multiplier `u` and Fisher weight `w` at one row.
    ///
    /// `u = (y−μ) V(μ)⁻¹ μ'(η)`. Bernoulli `μ` uses [`SCORE_PROBABILITY_CLAMP`]; the
    /// information weight of [`Self::fisher_weight`] uses [`FISHER_MEAN_FLOOR`] instead.
    pub(super) fn score_at(self, eta: f64, y: f64, nb_alpha: f64) -> (f64, f64) {
        use antecedent_kernels::norm_pdf;
        let mu = self.family.mean_from_eta(eta);
        let alpha = nb_alpha.max(0.0);
        match self.family {
            GlmFamily::GaussianIdentity => {
                // V=1, μ'=1 → u = y−μ, w = 1
                (y - mu, 1.0)
            }
            GlmFamily::BinomialLogit => {
                let mu = mu.clamp(SCORE_PROBABILITY_CLAMP, 1.0 - SCORE_PROBABILITY_CLAMP);
                let var = (mu * (1.0 - mu)).max(SCORE_VARIANCE_FLOOR);
                // μ' = var, u = (y−μ)/var * var = y−μ; w = var
                (y - mu, var)
            }
            GlmFamily::BinomialProbit => {
                let mu = mu.clamp(SCORE_PROBABILITY_CLAMP, 1.0 - SCORE_PROBABILITY_CLAMP);
                let phi = norm_pdf(eta).max(SCORE_VARIANCE_FLOOR);
                let var = (mu * (1.0 - mu)).max(SCORE_VARIANCE_FLOOR);
                // u = (y−μ) φ / V; w = φ² / V
                ((y - mu) * phi / var, (phi * phi) / var)
            }
            GlmFamily::PoissonLog => {
                let mu = mu.max(FISHER_MEAN_FLOOR);
                // V=μ, μ'=μ → u = y−μ; w = μ
                (y - mu, mu)
            }
            GlmFamily::NegativeBinomial => {
                let mu = mu.max(FISHER_MEAN_FLOOR);
                let var = (mu + alpha * mu * mu).max(SCORE_VARIANCE_FLOOR);
                // μ'=μ, u = (y−μ) μ / V; w = μ² / V
                ((y - mu) * mu / var, (mu * mu) / var)
            }
        }
    }
}

/// Response-scale derivative `dμ/dη` at `eta`.
pub(crate) fn mean_derivative(family: GlmFamily, eta: f64) -> f64 {
    FamilyKernel::new(family).mean_derivative(eta)
}

/// IRLS / Fisher information weight `W_ii` at `eta`.
///
/// For canonical Bernoulli/logit and Poisson this equals `dμ/dη`. For probit the
/// Bernoulli Fisher weight is `φ(η)² / (μ(1−μ))`, not `φ(η)`. For NB2 with log
/// link, `V = μ + α μ²` so `w = (μ')² / V = μ / (1 + α μ)`.
pub(crate) fn fisher_weight(family: GlmFamily, eta: f64, nb_alpha: f64) -> f64 {
    FamilyKernel::new(family).fisher_weight(eta, nb_alpha)
}

/// Fitted NB2 `α`, or a moment estimate from working residuals when the fit did
/// not report one. Non-NB families return `0.0` (ignored by callers).
pub(crate) fn resolve_nb_alpha(
    family: GlmFamily,
    fitted: Option<f64>,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
    y: &[f64],
) -> f64 {
    if !matches!(family, GlmFamily::NegativeBinomial) {
        return 0.0;
    }
    if let Some(a) = fitted {
        if a.is_finite() && a > 0.0 {
            return a;
        }
    }
    moment_nb_alpha(x_colmajor, nrows, ncols, coefficients, y)
}

/// Method-of-moments NB2 `α` from Pearson residuals of a log-mean fit.
pub(super) fn moment_nb_alpha(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
    y: &[f64],
) -> f64 {
    let mut pearson_ss = 0.0;
    let mut sum_mu = 0.0;
    for r in 0..nrows {
        let mut eta = 0.0;
        for c in 0..ncols {
            eta += x_colmajor[c * nrows + r] * coefficients[c];
        }
        let mu = eta.exp().max(FISHER_MEAN_FLOOR);
        let e = (y[r] - mu) / mu.sqrt();
        pearson_ss += e * e;
        sum_mu += mu;
    }
    let df = (nrows as f64 - ncols as f64).max(1.0);
    let excess = (pearson_ss - df).max(0.0);
    (excess / sum_mu.max(FISHER_MEAN_FLOOR)).max(1e-8)
}
