//! GLM fitting: logistic, probit, multinomial logit, Gaussian, Poisson, and NB2 IRLS.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use crate::error::{GlmRefusalKind, StatsError};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresWorkspace};
use antecedent_kernels::{norm_cdf, sigmoid};

mod binomial;
mod count;
mod gaussian;
mod multinomial;

pub use binomial::{
    GlmLaplaceFit, fit_glm_ridge, fit_glm_ridge_weighted, fit_logistic_weighted, fit_logit_laplace,
    logit_working_row,
};
pub use multinomial::{
    MultinomialDesignRef, MultinomialFit, fit_multinomial_logit, fit_multinomial_logit_weighted,
};

#[cfg(test)]
use self::{binomial::*, multinomial::*};

use binomial::{fit_logistic, fit_probit, fit_probit_weighted};
use count::{fit_negbin, fit_negbin_prior, fit_poisson, fit_poisson_weighted};
use gaussian::{fit_gaussian, fit_gaussian_weighted};

/// Family for the GLM path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GlmFamily {
    /// Binomial / Bernoulli with logit link.
    BinomialLogit,
    /// Binomial / Bernoulli with probit link.
    BinomialProbit,
    /// Gaussian with identity link (OLS).
    GaussianIdentity,
    /// Poisson with log link.
    PoissonLog,
    /// Negative binomial (NB2) with log link; dispersion via [`GlmOptions::nb_alpha`].
    NegativeBinomial,
}

/// Policy for NB2 dispersion `α` (`Var = μ + α μ²`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NbAlphaPolicy {
    /// Fixed dispersion.
    Fixed(f64),
    /// Pearson method-of-moments after a Poisson warm-start (default).
    MethodOfMoments,
    /// Alternating β-IRLS and 1-D MLE for `α`.
    NestedMle {
        /// Maximum outer (α, β) cycles.
        max_outer: u32,
        /// Absolute tolerance on `|Δα|`.
        tol_alpha: f64,
    },
}

impl Default for NbAlphaPolicy {
    fn default() -> Self {
        Self::MethodOfMoments
    }
}

impl GlmFamily {
    /// Mean on the response scale given linear predictor `eta`.
    #[must_use]
    pub fn mean_from_eta(self, eta: f64) -> f64 {
        match self {
            Self::BinomialLogit => sigmoid(eta),
            Self::BinomialProbit => norm_cdf(eta),
            Self::GaussianIdentity => eta,
            Self::PoissonLog | Self::NegativeBinomial => eta.exp(),
        }
    }
}

/// Borrowed column-major design + outcome used by [`fit_glm`].
#[derive(Clone, Copy, Debug)]
pub struct GlmDesignRef<'a> {
    /// Column-major design matrix.
    pub x_colmajor: &'a [f64],
    /// Rows.
    pub nrows: usize,
    /// Columns.
    pub ncols: usize,
    /// Outcome aligned with rows.
    pub y: &'a [f64],
}

/// Default ridge λ applied on binomial separation fallback ([`GlmOptions::ridge_on_separation`]).
pub const DEFAULT_RIDGE_ON_SEPARATION: f64 = 1e-4;

/// Fitting options for [`fit_glm`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlmOptions {
    /// Maximum IRLS iterations.
    pub max_iter: u32,
    /// Convergence tolerance on the change of the linear predictor, relative to its own
    /// magnitude: `max |Δη| ≤ tol · (1 + max |η|)`.
    pub tol: f64,
    /// NB2 dispersion policy ([`NbAlphaPolicy`]); ignored for other families.
    pub nb_alpha: NbAlphaPolicy,
    /// When set, logistic/probit that hit separation are refit with this ridge λ on `β`.
    ///
    /// Default is [`DEFAULT_RIDGE_ON_SEPARATION`].
    pub ridge_on_separation: Option<f64>,
    /// Propensity fits only: accept a converged, unseparated fit whose probabilities reach
    /// within `1e-8` of 0 or 1. A caller sets this when its declared overlap policy (a clip or
    /// a trim) governs those rows; the fit still records `boundary_saturated`, the margin and
    /// the count. Every other fit ignores it. Default `false`.
    pub admit_boundary_saturation: bool,
}

impl Default for GlmOptions {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-8,
            nb_alpha: NbAlphaPolicy::MethodOfMoments,
            ridge_on_separation: Some(DEFAULT_RIDGE_ON_SEPARATION),
            admit_boundary_saturation: false,
        }
    }
}

impl GlmOptions {
    /// Construct options (`MoM` NB α, default ridge-on-separation).
    #[must_use]
    pub const fn new(max_iter: u32, tol: f64) -> Self {
        Self {
            max_iter,
            tol,
            nb_alpha: NbAlphaPolicy::MethodOfMoments,
            ridge_on_separation: Some(DEFAULT_RIDGE_ON_SEPARATION),
            admit_boundary_saturation: false,
        }
    }

    /// These options admitting a boundary-saturated propensity fit when `admit` holds (see
    /// [`Self::admit_boundary_saturation`]).
    #[must_use]
    pub const fn admitting_boundary_saturation(self, admit: bool) -> Self {
        Self { admit_boundary_saturation: admit, ..self }
    }

    /// These options with the separation-ridge refit turned off.
    ///
    /// Estimation paths refuse every separated fit ([`GlmFit::require_ok`]), so the ridge
    /// refit [`Self::ridge_on_separation`] triggers would be computed and then discarded.
    /// Only diagnostic paths, which keep the ridge fit's scores, use the caller's option.
    #[must_use]
    pub const fn without_separation_ridge(self) -> Self {
        Self { ridge_on_separation: None, ..self }
    }
}

/// Convergence / iteration diagnostics from a GLM fit.
#[derive(Clone, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent diagnostic flags, each a distinct verdict"
)]
pub struct GlmFit {
    /// Coefficient vector.
    pub coefficients: Vec<f64>,
    /// Iterations used.
    pub iterations: u32,
    /// Whether the IRLS loop converged.
    pub converged: bool,
    /// (Quasi-)complete separation: fitted probabilities saturate *and* either the iteration
    /// did not converge or the fitted linear predictor classifies every row perfectly, so no
    /// finite maximum-likelihood estimate exists. A ridge refit keeps the unpenalized verdict.
    pub separated: bool,
    /// Some fitted probability lies within `1e-8` of 0 or 1. Weaker than `separated`: a
    /// converged fit with a strong predictor can saturate while its MLE exists, but such
    /// scores are still unusable as propensities, so [`GlmFit::require_ok`] refuses them.
    pub boundary_saturated: bool,
    /// Smallest distance of a fitted probability to 0 or 1, `min(mu, 1 - mu)` over the rows,
    /// at the returned coefficients (a binomial fit only; `None` for every other family,
    /// where it is not measured). For a separation fallback these are the penalized fit's
    /// scores, not the unpenalized fit's.
    pub boundary_margin: Option<f64>,
    /// Rows whose fitted probability lies within `1e-8` of 0 or 1 (a binomial fit only).
    pub boundary_count: Option<u64>,
    /// The coefficients minimise a ridge-penalized objective (the deliberate
    /// [`fit_glm_ridge`] or the separation fallback), not the unpenalized likelihood.
    pub penalized: bool,
    /// Final deviance.
    pub deviance: f64,
    /// Estimated or fixed NB2 `α` when family is [`GlmFamily::NegativeBinomial`].
    pub nb_alpha: Option<f64>,
    /// Rank / condition / backend / allocation diagnostics.
    pub diagnostics: FitDiagnostics,
}

impl GlmFit {
    /// Error if IRLS failed to converge or logistic separation was detected.
    ///
    /// # Errors
    ///
    /// Non-converged or separated fits.
    pub fn require_ok(&self) -> Result<(), StatsError> {
        let refuse = |kind, message: &'static str| StatsError::GlmRefused {
            kind,
            message,
            iterations: self.iterations,
            boundary_margin_bits: self.boundary_margin.map(f64::to_bits),
            boundary_count: self.boundary_count,
        };
        if !self.converged {
            return Err(refuse(
                GlmRefusalKind::NonConverged,
                "GLM IRLS did not converge; refuse propensity/outcome scores",
            ));
        }
        if self.separated {
            return Err(refuse(
                GlmRefusalKind::Separated,
                "GLM indicates (quasi-)complete separation; refuse propensity/outcome scores",
            ));
        }
        if self.boundary_saturated {
            return Err(refuse(
                GlmRefusalKind::BoundarySaturated,
                "GLM fitted probabilities lie within 1e-8 of 0 or 1 (extreme scores, not \
                 necessarily separation); refuse propensity/outcome scores",
            ));
        }
        Ok(())
    }
}

/// Fit a GLM on a compiled column-major design via IRLS + least squares.
///
/// # Errors
///
/// Shape mismatch, invalid outcomes for the family, or linear-algebra failure.
pub fn fit_glm(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::GlmFit,
        || match family {
            GlmFamily::BinomialLogit => fit_logistic(design, backend, workspace, options),
            GlmFamily::BinomialProbit => fit_probit(design, backend, workspace, options),
            GlmFamily::GaussianIdentity => fit_gaussian(design, backend, workspace),
            GlmFamily::PoissonLog => fit_poisson(design, backend, workspace, options),
            GlmFamily::NegativeBinomial => fit_negbin(design, backend, workspace, options),
        },
    )
}

/// Column `col` of a column-major design, a contiguous `nrows` block.
///
/// An empty design (`nrows == 0`) has no block. `chunks_exact(0)` would panic,
/// so that case is an empty slice rather than a zero-stride chunk.
#[inline]
fn design_column(x_colmajor: &[f64], nrows: usize, col: usize) -> &[f64] {
    if nrows == 0 {
        return &[];
    }
    let start = col * nrows;
    &x_colmajor[start..start + nrows]
}

/// `η = X β`, one contiguous pass per column.
///
/// Column `c` is added to every row before column `c + 1`, which is the same
/// sum each row of `Σ_c x_{rc} β_c` produced when the product was formed per row.
fn linear_predictor_into(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    beta: &[f64],
    eta: &mut [f64],
) {
    let eta = &mut eta[..nrows];
    eta.fill(0.0);
    if nrows == 0 {
        return;
    }
    let cols = ncols.min(beta.len());
    let covered = nrows * cols;
    for (column, &coef) in x_colmajor[..covered].chunks_exact(nrows).zip(beta) {
        for (slot, &x) in eta.iter_mut().zip(column) {
            *slot += x * coef;
        }
    }
}

/// Validate that `y` matches `nrows` and `x_colmajor` holds at least `nrows * ncols` entries.
///
/// Shared shape guard for the `fit_*` entry points; family-specific outcome-domain
/// checks (0/1, non-negativity, category range) are validated separately by each caller.
fn validate_glm_shape(design: GlmDesignRef<'_>) -> Result<(), StatsError> {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    Ok(())
}

/// Floor on a fitted mean, variance or density so a working weight stays representable.
const MEAN_FLOOR: f64 = 1e-12;

/// Clip applied to a fitted probability before it enters a working weight or deviance
/// (`μ ∈ [PROB_CLIP, 1 − PROB_CLIP]`).
const PROB_CLIP: f64 = 1e-9;

/// A fitted probability within this distance of 0 or 1 counts as saturated.
const SATURATION_MARGIN: f64 = 1e-8;

/// Step halvings tried before an IRLS step that fails to decrease the deviance is abandoned.
const MAX_STEP_HALVINGS: u32 = 30;

/// A deviance may rise by this relative amount between accepted iterates (rounding noise).
const DEVIANCE_NOISE: f64 = 1e-12;

/// Iteratively reweighted least squares shared by every non-Gaussian family.
///
/// `working(eta, y)` returns `(√w, √w · z)`: the square-root Fisher weight and the weighted
/// working response for one row. `data_start` initialises `η₀ = ln(y + 0.5)` (count
/// families, which diverge from `β = 0` at ordinary magnitudes) instead of `η₀ = 0`.
/// Returns `(β, iterations, converged)`.
fn irls_fit(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    data_start: bool,
    working: impl Fn(f64, f64) -> (f64, f64),
) -> Result<(Vec<f64>, u32, bool), StatsError> {
    irls_fit_scaled(design, backend, workspace, options, data_start, None, None, working)
}

/// [`irls_fit`] with optional square-root prior weights: row `r` of the working design and
/// response is multiplied by `row_scale[r]`, which solves the prior-weighted IRLS step.
///
/// With an `objective` (the prior-weighted deviance as a function of the linear predictor)
/// every step is checked: a step whose deviance is not finite, or rises, is halved toward
/// the previous iterate until it decreases, so a large predictor cannot overflow the mean
/// or send the iteration uphill. Convergence is judged on the change of the linear predictor
/// relative to its own magnitude (`max |Δη| ≤ tol · (1 + max |η|)`), which does not depend
/// on the units of the columns or on directions the data cannot identify.
#[allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "index-based loop over parallel arrays; one routine owns the damped update"
)]
fn irls_fit_scaled(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    data_start: bool,
    row_scale: Option<&[f64]>,
    objective: Option<&dyn Fn(&[f64]) -> f64>,
    working: impl Fn(f64, f64) -> (f64, f64),
) -> Result<(Vec<f64>, u32, bool), StatsError> {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut beta = vec![0.0; ncols];
    let mut z = vec![0.0; nrows];
    let mut eta = vec![0.0; nrows];
    let mut eta_next = vec![0.0; nrows];
    let mut eta_try = vec![0.0; nrows];
    let mut beta_try = vec![0.0; ncols];
    let mut weight = vec![0.0; nrows];
    let mut converged = false;
    let mut iterations = 0u32;
    // `eta` always holds `X β` for the accepted `β`, except before the first iteration of a
    // data-started fit, whose linearization point is the data-derived predictor instead.
    let mut current_deviance = objective.map(|f| f(&eta));

    for iter in 1..=options.max_iter {
        iterations = iter;
        // Predictor, then one scale of each column. `w` depends on the whole row,
        // so the two passes cannot collapse further; both walk contiguous columns.
        let linearization: &[f64] = if data_start && iter == 1 {
            for (slot, &yi) in eta_try.iter_mut().zip(y) {
                *slot = (yi + 0.5).ln();
            }
            &eta_try
        } else {
            &eta
        };
        for r in 0..nrows {
            let (mut w, mut zw) = working(linearization[r], y[r]);
            if let Some(scale) = row_scale {
                w *= scale[r];
                zw *= scale[r];
            }
            if !(w.is_finite() && zw.is_finite()) {
                return Err(StatsError::Backend(
                    "GLM IRLS: non-finite working weight or response (the linear predictor left \
                     the representable range of the mean)"
                        .into(),
                ));
            }
            weight[r] = w;
            z[r] = zw;
        }
        let fit =
            backend.row_scaled_least_squares(x_colmajor, nrows, ncols, &weight, &z, workspace)?;
        linear_predictor_into(x_colmajor, nrows, ncols, &fit.coefficients, &mut eta_next);
        let eta_scale = 1.0 + eta_next.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let full_step_change =
            eta_next.iter().zip(&eta).fold(0.0_f64, |m, (new, old)| m.max((new - old).abs()));
        if full_step_change <= options.tol * eta_scale {
            // The full step is already below tolerance: accept it, nothing to damp.
            beta.copy_from_slice(&fit.coefficients[..ncols]);
            converged = true;
            break;
        }
        let Some(objective) = objective else {
            beta.copy_from_slice(&fit.coefficients[..ncols]);
            std::mem::swap(&mut eta, &mut eta_next);
            continue;
        };
        let reference = current_deviance.unwrap_or(f64::INFINITY);
        let ceiling = reference + DEVIANCE_NOISE * reference.abs().max(1.0);
        let mut step = 1.0_f64;
        let mut accepted = None;
        for _ in 0..=MAX_STEP_HALVINGS {
            for c in 0..ncols {
                beta_try[c] = beta[c] + step * (fit.coefficients[c] - beta[c]);
            }
            for r in 0..nrows {
                eta_try[r] = eta[r] + step * (eta_next[r] - eta[r]);
            }
            let candidate = objective(&eta_try);
            if candidate.is_finite() && candidate <= ceiling {
                accepted = Some(candidate);
                break;
            }
            step *= 0.5;
        }
        let Some(candidate) = accepted else {
            // No damped step decreases the deviance: the iterate is as good as this
            // linearization can make it, but the fit did not reach the tolerance.
            break;
        };
        beta.copy_from_slice(&beta_try);
        eta.copy_from_slice(&eta_try);
        current_deviance = Some(candidate);
    }
    Ok((beta, iterations, converged))
}

/// Prior-weighted GLM fit: the maximizer of `Σ w_i ℓ_i(β)` for nonnegative case weights `w`,
/// whose estimating equation `Σ w_i x_i s_i(β) = 0` is the weighted-population score when the
/// weights are sampling weights.
///
/// Logit, probit, Poisson, negative-binomial and Gaussian families are supported. The
/// negative-binomial dispersion is updated from the weighted data (weighted Pearson moment or
/// weighted profile likelihood, per [`GlmOptions::nb_alpha`]). Separation diagnostics are those
/// of the unweighted rows and the separation ridge refit is not offered. The Gaussian
/// `deviance` is the weighted residual sum of squares `Σ w_i r_i²`.
///
/// # Errors
///
/// Shape mismatch, a weight that is negative or not finite, outcomes invalid for the family,
/// or a linear-algebra failure.
pub fn fit_glm_weighted(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    match family {
        GlmFamily::BinomialLogit => {
            fit_logistic_weighted(design, weights, backend, workspace, options)
        }
        GlmFamily::BinomialProbit => {
            fit_probit_weighted(design, weights, backend, workspace, options)
        }
        GlmFamily::PoissonLog => fit_poisson_weighted(design, weights, backend, workspace, options),
        GlmFamily::GaussianIdentity => fit_gaussian_weighted(design, weights, backend, workspace),
        GlmFamily::NegativeBinomial => {
            fit_negbin_prior(design, Some(weights), backend, workspace, options)
        }
    }
}

fn check_prior_weights(design: GlmDesignRef<'_>, weights: &[f64]) -> Result<(), StatsError> {
    validate_glm_shape(design)?;
    if weights.len() != design.nrows {
        return Err(StatsError::Shape { message: "weights length != nrows" });
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(StatsError::Shape { message: "prior weights must be finite and non-negative" });
    }
    Ok(())
}

#[cfg(test)]
mod binomial_tests;
#[cfg(test)]
mod count_tests;
#[cfg(test)]
mod fused_tests;
#[cfg(test)]
mod gaussian_tests;
#[cfg(test)]
mod multinomial_tests;
