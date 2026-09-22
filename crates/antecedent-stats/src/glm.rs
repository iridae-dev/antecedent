//! GLM fitting: logistic, probit, multinomial logit, Gaussian, Poisson, and NB2 IRLS.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::too_many_lines)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_kernels::{norm_cdf, norm_pdf};

use crate::error::StatsError;
use crate::gram::{column_is_constant, invert_square};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresWorkspace};

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
            Self::BinomialLogit => 1.0 / (1.0 + (-eta).exp()),
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
    /// Coefficient change tolerance for convergence.
    pub tol: f64,
    /// NB2 dispersion policy ([`NbAlphaPolicy`]); ignored for other families.
    pub nb_alpha: NbAlphaPolicy,
    /// When set, logistic/probit that hit separation are refit with this ridge λ on `β`.
    ///
    /// Default is [`DEFAULT_RIDGE_ON_SEPARATION`].
    pub ridge_on_separation: Option<f64>,
}

impl Default for GlmOptions {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-8,
            nb_alpha: NbAlphaPolicy::MethodOfMoments,
            ridge_on_separation: Some(DEFAULT_RIDGE_ON_SEPARATION),
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
        }
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
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent, separately documented fit verdict, part of the public result shape"
)]
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)] // independent diagnostic flags, each a distinct verdict
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
        if !self.converged {
            return Err(StatsError::Backend(
                "GLM IRLS did not converge; refuse propensity/outcome scores".into(),
            ));
        }
        if self.separated {
            return Err(StatsError::Backend(
                "GLM indicates (quasi-)complete separation; refuse propensity/outcome scores"
                    .into(),
            ));
        }
        if self.boundary_saturated {
            return Err(StatsError::Backend(
                "GLM fitted probabilities lie within 1e-8 of 0 or 1 (extreme scores, not \
                 necessarily separation); refuse propensity/outcome scores"
                    .into(),
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
    match family {
        GlmFamily::BinomialLogit => fit_logistic(design, backend, workspace, options),
        GlmFamily::BinomialProbit => fit_probit(design, backend, workspace, options),
        GlmFamily::GaussianIdentity => fit_gaussian(design, backend, workspace),
        GlmFamily::PoissonLog => fit_poisson(design, backend, workspace, options),
        GlmFamily::NegativeBinomial => fit_negbin(design, backend, workspace, options),
    }
}

/// Linear predictor `X β` for one row of a column-major design.
#[inline]
fn eta_at(x_colmajor: &[f64], nrows: usize, ncols: usize, beta: &[f64], row: usize) -> f64 {
    let mut eta = 0.0;
    for c in 0..ncols {
        eta += x_colmajor[c * nrows + row] * beta[c];
    }
    eta
}

/// Poisson deviance at `β` (log link).
fn poisson_deviance_at(design: GlmDesignRef<'_>, beta: &[f64]) -> f64 {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut deviance = 0.0;
    for r in 0..nrows {
        let mu = eta_at(x_colmajor, nrows, ncols, beta, r).exp().max(1e-12);
        let yi = y[r];
        if yi > 0.0 {
            deviance += 2.0 * (yi * (yi / mu).ln() - (yi - mu));
        } else {
            deviance += 2.0 * mu;
        }
    }
    deviance
}

/// Binomial deviance and separation evidence at `β`.
struct BinomialDiagnostics {
    deviance: f64,
    /// Some fitted probability is outside `[1e-8, 1 − 1e-8]`.
    saturated: bool,
    /// `η > 0` for every `y = 1` row and `η < 0` for every `y = 0` row: `β` itself is a
    /// separating direction, which proves complete separation.
    perfectly_classified: bool,
}

impl BinomialDiagnostics {
    /// Separation verdict: saturated probabilities that either failed to converge or sit on
    /// a perfectly separating hyperplane. Saturation alone (a strong predictor) is not
    /// separation.
    fn separated(&self, converged: bool) -> bool {
        self.saturated && (!converged || self.perfectly_classified)
    }
}

fn binomial_diagnostics_at(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    beta: &[f64],
) -> BinomialDiagnostics {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut deviance = 0.0;
    let mut saturated = false;
    let mut perfectly_classified = true;
    for r in 0..nrows {
        let eta = eta_at(x_colmajor, nrows, ncols, beta, r);
        let mu = match family {
            GlmFamily::BinomialLogit => 1.0 / (1.0 + (-eta).exp()),
            GlmFamily::BinomialProbit => norm_cdf(eta),
            _ => unreachable!("binomial diagnostics require a binomial family"),
        };
        if !(1e-8..=1.0 - 1e-8).contains(&mu) {
            saturated = true;
        }
        if (y[r] > 0.0 && eta <= 0.0) || (y[r] <= 0.0 && eta >= 0.0) {
            perfectly_classified = false;
        }
        let mu_clamped = mu.clamp(1e-9, 1.0 - 1e-9);
        if y[r] > 0.0 {
            deviance += -2.0 * mu_clamped.ln();
        } else {
            deviance += -2.0 * (1.0 - mu_clamped).ln();
        }
    }
    BinomialDiagnostics { deviance, saturated, perfectly_classified }
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

/// NB2 deviance at `β` with fixed dispersion `α` (log link).
fn negbin_deviance_at(design: GlmDesignRef<'_>, beta: &[f64], alpha: f64) -> f64 {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let theta = 1.0 / alpha;
    let mut deviance = 0.0;
    for r in 0..nrows {
        let mu = eta_at(x_colmajor, nrows, ncols, beta, r).exp().max(1e-12);
        let yi = y[r];
        if yi > 0.0 {
            deviance +=
                2.0 * (yi * (yi / mu).ln() - (yi + theta) * ((yi + theta) / (mu + theta)).ln());
        } else {
            deviance += 2.0 * theta * ((theta + mu) / theta).ln();
        }
    }
    deviance
}

fn fit_gaussian(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let fit = backend.least_squares(x_colmajor, nrows, ncols, y, workspace)?;
    Ok(GlmFit {
        coefficients: fit.coefficients,
        iterations: 1,
        converged: true,
        separated: false,
        boundary_saturated: false,
        penalized: false,
        deviance: fit.rss,
        nb_alpha: None,
        diagnostics: fit.diagnostics,
    })
}

fn fit_poisson(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { ncols, y, .. } = design;
    for &yi in y {
        if !(yi.is_finite() && yi >= 0.0) {
            return Err(StatsError::Shape {
                message: "Poisson GLM requires non-negative outcomes",
            });
        }
    }

    let (beta, iterations, converged) =
        irls_fit(design, backend, workspace, options, true, |eta, yi| {
            let mu = eta.exp().max(1e-12);
            let w = mu.sqrt();
            (w, (eta + (yi - mu) / mu) * w)
        })?;

    // Diagnostics at the returned coefficients (not the pre-update IRLS iterate).
    let deviance = poisson_deviance_at(design, &beta);

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated: false,
        boundary_saturated: false,
        penalized: false,
        deviance,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(ncols, None, "glm-irls", workspace.grow_count),
    })
}

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
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut beta = vec![0.0; ncols];
    let mut x_w = vec![0.0; nrows * ncols];
    let mut z = vec![0.0; nrows];
    let mut converged = false;
    let mut iterations = 0u32;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        for r in 0..nrows {
            let eta = if data_start && iter == 1 {
                (y[r] + 0.5).ln()
            } else {
                eta_at(x_colmajor, nrows, ncols, &beta, r)
            };
            let (w, zw) = working(eta, y[r]);
            z[r] = zw;
            for c in 0..ncols {
                x_w[c * nrows + r] = x_colmajor[c * nrows + r] * w;
            }
        }
        let fit = backend.least_squares(&x_w, nrows, ncols, &z, workspace)?;
        for c in 0..ncols {
            max_delta = max_delta.max((fit.coefficients[c] - beta[c]).abs());
            beta[c] = fit.coefficients[c];
        }
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }
    Ok((beta, iterations, converged))
}

#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
fn fit_logistic(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { y, .. } = design;
    for &yi in y {
        if !(yi == 0.0 || yi == 1.0) {
            return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
        }
    }

    let (beta, iterations, converged) =
        irls_fit(design, backend, workspace, options, false, |eta, yi| {
            let mu = 1.0 / (1.0 + (-eta).exp());
            let mu_clamped = mu.clamp(1e-9, 1.0 - 1e-9);
            let w = (mu_clamped * (1.0 - mu_clamped)).sqrt();
            let z = eta + (yi - mu_clamped) / (mu_clamped * (1.0 - mu_clamped));
            (w, z * w)
        })?;

    finish_binomial(
        GlmFamily::BinomialLogit,
        design,
        workspace,
        options,
        beta,
        iterations,
        converged,
    )
}

/// Deviance and separation diagnostics for an unpenalized binomial IRLS result, with the
/// optional ridge refit when the fitted probabilities saturate.
#[allow(clippy::too_many_arguments)]
fn finish_binomial(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    beta: Vec<f64>,
    iterations: u32,
    converged: bool,
) -> Result<GlmFit, StatsError> {
    // Diagnostics at the returned coefficients (not the pre-update IRLS iterate).
    let diag = binomial_diagnostics_at(family, design, &beta);
    let separated = diag.separated(converged);

    if diag.saturated {
        if let Some(lambda) = options.ridge_on_separation {
            if lambda > 0.0 {
                return fit_binomial_ridge(
                    family,
                    design,
                    workspace,
                    options,
                    lambda,
                    Some(separated),
                );
            }
        }
    }

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated,
        boundary_saturated: diag.saturated,
        penalized: false,
        deviance: diag.deviance,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(design.ncols, None, "glm-irls", workspace.grow_count),
    })
}

#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
fn fit_probit(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { y, .. } = design;
    for &yi in y {
        if !(yi == 0.0 || yi == 1.0) {
            return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
        }
    }

    let (beta, iterations, converged) =
        irls_fit(design, backend, workspace, options, false, |eta, yi| {
            let mu = norm_cdf(eta);
            let mu_clamped = mu.clamp(1e-9, 1.0 - 1e-9);
            let phi = norm_pdf(eta).max(1e-12);
            let denom = (mu_clamped * (1.0 - mu_clamped)).max(1e-12);
            let w = ((phi * phi) / denom).sqrt();
            (w, (eta + (yi - mu_clamped) / phi) * w)
        })?;

    finish_binomial(
        GlmFamily::BinomialProbit,
        design,
        workspace,
        options,
        beta,
        iterations,
        converged,
    )
}

/// NB2 IRLS with fixed, `MoM`, or nested-MLE dispersion α (`Var = μ + α μ²`).
fn fit_negbin(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    for &yi in y {
        if !(yi.is_finite() && yi >= 0.0) {
            return Err(StatsError::Shape {
                message: "negative-binomial GLM requires non-negative outcomes",
            });
        }
    }

    let mut mom_alpha = || -> Result<f64, StatsError> {
        let poisson = fit_poisson(design, backend, workspace, options)?;
        let mut pearson_ss = 0.0;
        let mut sum_mu = 0.0;
        for r in 0..nrows {
            let mut eta = 0.0;
            for c in 0..ncols {
                eta += x_colmajor[c * nrows + r] * poisson.coefficients[c];
            }
            let mu = eta.exp().max(1e-12);
            let e = (y[r] - mu) / mu.sqrt();
            pearson_ss += e * e;
            sum_mu += mu;
        }
        let df = (nrows as f64 - ncols as f64).max(1.0);
        let excess = (pearson_ss - df).max(0.0);
        Ok((excess / sum_mu.max(1e-12)).max(1e-8))
    };

    match options.nb_alpha {
        NbAlphaPolicy::Fixed(a) => {
            if !(a.is_finite() && a > 0.0) {
                return Err(StatsError::Shape { message: "nb_alpha Fixed must be finite and > 0" });
            }
            fit_negbin_fixed_alpha(design, backend, workspace, options, a)
        }
        NbAlphaPolicy::MethodOfMoments => {
            let alpha = mom_alpha()?;
            fit_negbin_fixed_alpha(design, backend, workspace, options, alpha)
        }
        NbAlphaPolicy::NestedMle { max_outer, tol_alpha } => {
            let mut alpha = mom_alpha()?;
            let mut fit = fit_negbin_fixed_alpha(design, backend, workspace, options, alpha)?;
            let mut outer_ok = false;
            for _ in 0..max_outer.max(1) {
                let mut mu = vec![0.0; nrows];
                for r in 0..nrows {
                    let mut eta = 0.0;
                    for c in 0..ncols {
                        eta += x_colmajor[c * nrows + r] * fit.coefficients[c];
                    }
                    mu[r] = eta.exp().max(1e-12);
                }
                let alpha_new = mle_nb_alpha(y, &mu, alpha);
                let beta_prev = fit.coefficients.clone();
                fit = fit_negbin_fixed_alpha(design, backend, workspace, options, alpha_new)?;
                let mut max_db = 0.0_f64;
                for c in 0..ncols {
                    max_db = max_db.max((fit.coefficients[c] - beta_prev[c]).abs());
                }
                let da = (alpha_new - alpha).abs();
                alpha = alpha_new;
                if da < tol_alpha && (fit.converged || max_db < options.tol) {
                    outer_ok = true;
                    break;
                }
            }
            fit.converged = fit.converged && outer_ok;
            fit.nb_alpha = Some(alpha);
            Ok(fit)
        }
    }
}

/// 1-D MLE for NB2 α given fitted means (digamma/trigamma Newton on θ=1/α, safeguarded).
fn mle_nb_alpha(y: &[f64], mu: &[f64], alpha0: f64) -> f64 {
    use crate::special::{digamma, trigamma};

    // Work in θ = 1/α (more stable); clamp to a wide but finite range.
    let mut theta = (1.0 / alpha0.max(1e-8)).clamp(1e-4, 1e6);
    for _ in 0..80 {
        let mut score = 0.0;
        let mut hess = 0.0;
        for (&yi, &mui) in y.iter().zip(mu.iter()) {
            let d_theta = digamma(yi + theta) - digamma(theta)
                + (theta / (theta + mui)).ln()
                + (mui - yi) / (theta + mui);
            let d2_theta = trigamma(yi + theta) - trigamma(theta) + 1.0 / theta
                - 1.0 / (theta + mui)
                - (mui - yi) / ((theta + mui) * (theta + mui));
            score += d_theta;
            hess += d2_theta;
        }
        if !score.is_finite() {
            break;
        }
        // Maximize ⇒ Newton uses −score/hess when hess < 0; otherwise gradient ascent.
        let step = if hess < -1e-12 {
            -score / hess
        } else if score.abs() > 1e-12 {
            0.1 * score.signum() * theta
        } else {
            0.0
        };
        let mut theta_new = (theta + step).clamp(1e-4, 1e6);
        // Damp large relative steps.
        if (theta_new - theta).abs() > 0.5 * theta {
            theta_new = theta + 0.25 * (theta_new - theta);
            theta_new = theta_new.clamp(1e-4, 1e6);
        }
        if (theta_new - theta).abs() < 1e-10 * (1.0 + theta) || score.abs() < 1e-10 {
            theta = theta_new;
            break;
        }
        theta = theta_new;
    }
    (1.0 / theta).clamp(1e-8, 1e4)
}

fn fit_negbin_fixed_alpha(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    alpha: f64,
) -> Result<GlmFit, StatsError> {
    let ncols = design.ncols;
    let (beta, iterations, converged) =
        irls_fit(design, backend, workspace, options, true, |eta, yi| {
            let mu = eta.exp().max(1e-12);
            let var = mu * (1.0 + alpha * mu);
            let w = (mu * mu / var).sqrt();
            (w, (eta + (yi - mu) / mu) * w)
        })?;

    // Diagnostics at the returned coefficients (not the pre-update IRLS iterate).
    let deviance = negbin_deviance_at(design, &beta, alpha);

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated: false,
        boundary_saturated: false,
        penalized: false,
        deviance,
        nb_alpha: Some(alpha),
        diagnostics: FitDiagnostics::new(ncols, None, "glm-irls", workspace.grow_count),
    })
}

/// Ridge-penalized binomial IRLS: the separation fallback (`fallback_separated` is the
/// unpenalized fit's verdict, which the penalized fit must not clear) or the deliberate
/// always-on ridge of [`fit_glm_ridge`] (`None`: the verdict is taken at the ridge solution).
fn fit_binomial_ridge(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    lambda: f64,
    fallback_separated: Option<bool>,
) -> Result<GlmFit, StatsError> {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut beta = vec![0.0; ncols];
    let mut xtwx = vec![0.0; ncols * ncols];
    let mut xtwz = vec![0.0; ncols];
    let mut converged = false;
    let mut iterations = 0u32;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        xtwx.fill(0.0);
        xtwz.fill(0.0);
        for r in 0..nrows {
            let mut eta = 0.0;
            for c in 0..ncols {
                eta += x_colmajor[c * nrows + r] * beta[c];
            }
            let (_mu, w_fisher, working) = match family {
                GlmFamily::BinomialLogit => {
                    let mu = (1.0 / (1.0 + (-eta).exp())).clamp(1e-9, 1.0 - 1e-9);
                    let w = mu * (1.0 - mu);
                    let z = eta + (y[r] - mu) / w.max(1e-12);
                    (mu, w, z)
                }
                GlmFamily::BinomialProbit => {
                    let mu = norm_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
                    let phi = norm_pdf(eta).max(1e-12);
                    let w = (phi * phi) / (mu * (1.0 - mu)).max(1e-12);
                    let z = eta + (y[r] - mu) / phi;
                    (mu, w, z)
                }
                _ => unreachable!("ridge fallback is binomial-only"),
            };
            for i in 0..ncols {
                let xi = x_colmajor[i * nrows + r];
                xtwz[i] += xi * w_fisher * working;
                for j in 0..ncols {
                    xtwx[i * ncols + j] += xi * w_fisher * x_colmajor[j * nrows + r];
                }
            }
        }
        let unpenalize0 = ncols > 0 && column_is_constant(x_colmajor, nrows, 0);
        for c in 0..ncols {
            if c == 0 && unpenalize0 {
                continue;
            }
            xtwx[c * ncols + c] += lambda;
        }
        let Some(inv) = invert_square(&xtwx, ncols) else {
            return Err(StatsError::Backend("ridge binomial: singular X'WX+λI".into()));
        };
        let mut beta_new = vec![0.0; ncols];
        for i in 0..ncols {
            let mut s = 0.0;
            for j in 0..ncols {
                s += inv[i * ncols + j] * xtwz[j];
            }
            beta_new[i] = s;
            max_delta = max_delta.max((s - beta[i]).abs());
        }
        beta = beta_new;
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }

    // A separation fallback keeps the unpenalized fit's verdict: penalized scores are not a
    // substitute for an unpenalized MLE, so estimation paths (`require_ok`) still refuse
    // them. A deliberate ridge fit is judged at its own solution and is `penalized`.
    let diag = binomial_diagnostics_at(family, design, &beta);
    let separated = fallback_separated.unwrap_or_else(|| diag.separated(converged));

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated,
        boundary_saturated: diag.saturated || fallback_separated.is_some(),
        penalized: true,
        deviance: diag.deviance,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(ncols, None, "glm-irls", workspace.grow_count),
    })
}

/// Fit a binomial GLM with an always-active ridge penalty.
///
/// A constant first design column is treated as an intercept and is not
/// penalized. Unlike [`GlmOptions::ridge_on_separation`], this entry point
/// applies `lambda` whether or not the unpenalized fit is separated.
///
/// # Errors
///
/// Invalid shapes, outcomes, family, penalty, or a numerical solver failure.
#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
pub fn fit_glm_ridge(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    _backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    lambda: f64,
) -> Result<GlmFit, StatsError> {
    if !matches!(family, GlmFamily::BinomialLogit | GlmFamily::BinomialProbit) {
        return Err(StatsError::Shape {
            message: "ridge GLM currently requires a binomial family",
        });
    }
    validate_glm_shape(design)?;
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(StatsError::Shape { message: "ridge lambda must be finite and positive" });
    }
    if design.y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
        return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
    }
    fit_binomial_ridge(family, design, workspace, options, lambda, None)
}

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
        if !self.converged {
            return Err(StatsError::Backend("multinomial logit IRLS did not converge".into()));
        }
        if self.separated {
            return Err(StatsError::Backend(
                "multinomial logit indicates (quasi-)complete separation".into(),
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
    if let Some(w) = weights {
        let sum: f64 = w.iter().sum();
        if w.len() != design.nrows
            || w.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !sum.is_finite()
            || sum <= 0.0
        {
            return Err(StatsError::Shape { message: "invalid multinomial observation weights" });
        }
    }
    let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, n_categories: k } = design;
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
fn weighted_gram_is_full_rank(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    weights: Option<&[f64]>,
) -> bool {
    let mut gram = vec![0.0; ncols * ncols];
    for r in 0..nrows {
        let w = weights.map_or(1.0, |w| w[r]);
        for c1 in 0..ncols {
            let a = w * x_colmajor[c1 * nrows + r];
            for c2 in c1..ncols {
                gram[c1 * ncols + c2] += a * x_colmajor[c2 * nrows + r];
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
        h.fill(0.0);
        score.fill(0.0);
        let mut loop_deviance = 0.0;

        for r in 0..nrows {
            softmax_row(design, &beta_free, r, &mut eta, &mut pi);
            let yi = y_category[r] as usize;
            // In-loop deviance is for convergence only; final diagnostics recomputed below.
            loop_deviance += weights.map_or(1.0, |w| w[r]) * -2.0 * pi[yi].max(1e-300).ln();

            // Score and Fisher information over free categories 1..K-1.
            for j in 1..k {
                let pj = pi[j];
                let yj = if yi == j { 1.0 } else { 0.0 };
                let resid = yj - pj;
                let jb = (j - 1) * ncols;
                for c in 0..ncols {
                    let xc = x_colmajor[c * nrows + r];
                    score[jb + c] += weights.map_or(1.0, |w| w[r]) * resid * xc;
                }
                for jp in 1..k {
                    let pjp = pi[jp];
                    let w = if j == jp { pj * (1.0 - pj) } else { -pj * pjp };
                    let jpb = (jp - 1) * ncols;
                    for c1 in 0..ncols {
                        let xc1 = x_colmajor[c1 * nrows + r];
                        for c2 in 0..ncols {
                            let xc2 = x_colmajor[c2 * nrows + r];
                            h[(jb + c1) * m + (jpb + c2)] +=
                                weights.map_or(1.0, |w| w[r]) * w * xc1 * xc2;
                        }
                    }
                }
            }
        }

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
        let mut delta = vec![0.0; m];
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

/// Baseline-category softmax probabilities for row `r`: `η₀ = 0`, `ηⱼ = x·βⱼ` for
/// `j = 1..K-1`, computed against the maximum `η` so the exponentials cannot overflow.
/// `η₀ = 0` is always a candidate for the maximum, so the normaliser is at least 1.
fn softmax_row(
    design: MultinomialDesignRef<'_>,
    beta_free: &[f64],
    r: usize,
    eta: &mut [f64],
    pi: &mut [f64],
) {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, n_categories: k, .. } = design;
    eta[0] = 0.0;
    let mut max_eta = 0.0_f64;
    for j in 1..k {
        let mut acc = 0.0;
        let base = (j - 1) * ncols;
        for c in 0..ncols {
            acc += x_colmajor[c * nrows + r] * beta_free[base + c];
        }
        eta[j] = acc;
        if acc > max_eta {
            max_eta = acc;
        }
    }
    let mut zsum = 0.0;
    for j in 0..k {
        let e = (eta[j] - max_eta).exp();
        pi[j] = e;
        zsum += e;
    }
    let inv_z = 1.0 / zsum;
    for p in pi.iter_mut().take(k) {
        *p *= inv_z;
    }
}

/// Multinomial deviance and separation at free coefficients (reference category 0 pinned).
fn multinomial_diagnostics_at(
    design: MultinomialDesignRef<'_>,
    beta_free: &[f64],
    weights: Option<&[f64]>,
) -> (f64, bool) {
    let MultinomialDesignRef { nrows, y_category, n_categories: k, .. } = design;
    let mut deviance = 0.0;
    let mut separated = false;
    let mut pi = vec![0.0; k];
    let mut eta = vec![0.0; k];
    for r in 0..nrows {
        softmax_row(design, beta_free, r, &mut eta, &mut pi);
        for &p in &pi {
            if !(1e-8..=1.0 - 1e-8).contains(&p) {
                separated = true;
            }
        }
        let yi = y_category[r] as usize;
        deviance += weights.map_or(1.0, |w| w[r]) * -2.0 * pi[yi].max(1e-300).ln();
    }
    (deviance, separated)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert exactly representable values and copied inputs, not measurements"
)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;
    use antecedent_kernels::norm_cdf;

    #[test]
    fn logistic_separates_simple_signal() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = if i % 10 == 0 { 1.0 - t } else { t };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-6),
        )
        .unwrap();
        assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
        assert!(fit.coefficients[1] > 0.5);
        assert!(!fit.separated);
    }

    #[test]
    fn logistic_flags_complete_separation() {
        let n = 60usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = t;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let opts = GlmOptions { ridge_on_separation: None, ..GlmOptions::new(100, 1e-6) };
        let fit = fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &opts,
        )
        .unwrap();
        assert!(fit.separated);
        assert!(fit.require_ok().is_err());
    }

    #[test]
    fn gaussian_recovers_linear_slope() {
        let n = 100usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i % 2) as f64;
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = 1.0 + 2.0 * t;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::GaussianIdentity,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
        )
        .unwrap();
        assert!((fit.coefficients[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn poisson_recovers_positive_association() {
        let n = 120usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = if t < 0.5 { 2.0 } else { 4.0 };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.coefficients[1] > 0.3);
    }

    #[test]
    fn poisson_intercept_only_converges_to_log_mean() {
        let n = 40usize;
        let x = vec![1.0; n];
        let y = vec![100.0; n];
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
        )
        .unwrap();
        assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
        assert!((fit.coefficients[0] - 100.0_f64.ln()).abs() < 1e-6, "b0={}", fit.coefficients[0]);
    }

    #[test]
    fn probit_recovers_positive_slope() {
        let n = 200usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            // Soft signal: mostly follows t with a few flips.
            y[i] = if i % 8 == 0 { 1.0 - t } else { t };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::BinomialProbit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-6),
        )
        .unwrap();
        assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
        assert!(fit.coefficients[1] > 0.3, "slope={}", fit.coefficients[1]);
        assert!(!fit.separated);
    }

    #[test]
    fn negbin_fixed_alpha_recovers_positive_association() {
        let n = 120usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            // Overdispersed counts: base 2 vs 5 with occasional spikes.
            y[i] = if t < 0.5 {
                if i % 11 == 0 { 8.0 } else { 2.0 }
            } else if i % 11 == 0 {
                15.0
            } else {
                5.0
            };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let opts = GlmOptions { nb_alpha: NbAlphaPolicy::Fixed(0.5), ..GlmOptions::new(100, 1e-8) };
        let fit = fit_glm(
            GlmFamily::NegativeBinomial,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &opts,
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.coefficients[1] > 0.3, "slope={}", fit.coefficients[1]);
    }

    #[test]
    fn negbin_mom_alpha_matches_pearson_formula() {
        // Intercept-only Poisson μ̂ = ȳ; Pearson MoM α̂ = (Σ(y−μ)²/μ − (n−1)) / (nμ).
        let y = [1.0_f64, 2.0, 0.0, 4.0, 3.0, 1.0, 5.0, 2.0];
        let n = y.len();
        let mu = y.iter().sum::<f64>() / n as f64;
        let pearson: f64 = y.iter().map(|&yi| (yi - mu).powi(2) / mu).sum();
        let expected = ((pearson - (n as f64 - 1.0)).max(0.0) / (n as f64 * mu)).max(1e-8);

        let x = vec![1.0; n];
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::NegativeBinomial,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-10),
        )
        .unwrap();
        assert!(fit.converged);
        // Recompute MoM from the same Poisson μ path used internally.
        let poisson = fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-10),
        )
        .unwrap();
        let mu_hat = poisson.coefficients[0].exp();
        assert!((mu_hat - mu).abs() < 1e-8);
        assert!(
            (expected - ((pearson - (n as f64 - 1.0)).max(0.0) / (n as f64 * mu_hat)).max(1e-8))
                .abs()
                < 1e-12
        );
        // Fixed-α fit at that MoM should match default MethodOfMoments path coefficients.
        let opts =
            GlmOptions { nb_alpha: NbAlphaPolicy::Fixed(expected), ..GlmOptions::new(100, 1e-10) };
        let fit_fixed = fit_glm(
            GlmFamily::NegativeBinomial,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &opts,
        )
        .unwrap();
        assert!((fit.coefficients[0] - fit_fixed.coefficients[0]).abs() < 1e-6);
    }

    #[test]
    fn negbin_nested_mle_recovers_overdispersion() {
        // Intercept-only NB2 with known α≈0.5 via gamma-poisson mixture (deterministic spikes).
        let n = 400usize;
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            // Mean ~3 with heavy tails: mostly 2, occasional 10.
            y[i] = if i % 8 == 0 { 10.0 } else { 2.0 };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let opts = GlmOptions {
            nb_alpha: NbAlphaPolicy::NestedMle { max_outer: 20, tol_alpha: 1e-6 },
            ..GlmOptions::new(100, 1e-8)
        };
        let fit = fit_glm(
            GlmFamily::NegativeBinomial,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &opts,
        )
        .unwrap();
        assert!(fit.converged);
        let alpha = fit.nb_alpha.expect("nb alpha");
        assert!(alpha > 0.05, "alpha={alpha}");
        assert!(alpha < 5.0, "alpha={alpha}");
        // Nested MLE α should be in the ballpark of MoM.
        let mom = fit_glm(
            GlmFamily::NegativeBinomial,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        let mom_a = mom.nb_alpha.unwrap();
        assert!((alpha - mom_a).abs() / mom_a.max(1e-6) < 2.0, "nested={alpha} mom={mom_a}");
    }

    #[test]
    fn negbin_nested_mle_recovers_alpha_above_two() {
        // Heavy overdispersion so NestedMle Newton enters θ = 1/α < 0.5 (trigamma reflection).
        let n = 500usize;
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            // Mean ~4 with extreme spikes ⇒ α ≫ 2.
            y[i] = if i % 5 == 0 { 30.0 } else { 1.0 };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let opts = GlmOptions {
            nb_alpha: NbAlphaPolicy::NestedMle { max_outer: 30, tol_alpha: 1e-6 },
            ..GlmOptions::new(100, 1e-8)
        };
        let fit = fit_glm(
            GlmFamily::NegativeBinomial,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &opts,
        )
        .unwrap();
        assert!(fit.converged);
        let alpha = fit.nb_alpha.expect("nb alpha");
        assert!(alpha > 2.0, "expected alpha > 2 for trigamma reflection path, got {alpha}");
        // MoM should also land in the same high-α regime.
        let mom = fit_glm(
            GlmFamily::NegativeBinomial,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        let mom_a = mom.nb_alpha.unwrap();
        assert!(mom_a > 2.0, "mom alpha={mom_a}");
        assert!((alpha - mom_a).abs() / mom_a.max(1e-6) < 2.0, "nested={alpha} mom={mom_a}");
    }

    #[test]
    fn ridge_on_separation_keeps_separated_flag() {
        let n = 60usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = t;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let opts = GlmOptions { ridge_on_separation: Some(1.0), ..GlmOptions::new(100, 1e-6) };
        let fit = fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &opts,
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.separated, "ridge fallback must not clear separation");
        assert!(fit.require_ok().is_err());
        assert!(fit.coefficients[1] > 0.0);
    }

    #[test]
    fn estimation_options_skip_the_discarded_separation_ridge_refit() {
        // Perfectly separated data: with the caller's ridge option the returned fit is the
        // ridge refit (`penalized`); without it the unpenalized IRLS iterate comes back.
        // Both are flagged separated, so `require_ok` refuses either, and the second spares
        // the refit whose result an estimation path would throw away.
        let n = 40usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = t;
        }
        let with_ridge = GlmOptions::new(100, 1e-6);
        let without = with_ridge.without_separation_ridge();
        assert_eq!(without.ridge_on_separation, None);
        assert_eq!((without.max_iter, without.tol), (with_ridge.max_iter, with_ridge.tol));
        let mut ws = LeastSquaresWorkspace::default();
        let mut fit = |opts: &GlmOptions| {
            fit_glm(
                GlmFamily::BinomialLogit,
                GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
                &FaerBackend,
                &mut ws,
                opts,
            )
            .unwrap()
        };
        let ridge_fit = fit(&with_ridge);
        let plain_fit = fit(&without);
        assert!(ridge_fit.penalized && !plain_fit.penalized);
        assert!(ridge_fit.separated && plain_fit.separated);
        assert!(ridge_fit.require_ok().is_err() && plain_fit.require_ok().is_err());
    }

    #[test]
    fn multinomial_binary_matches_logistic_slope_sign() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0u32; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = if i % 10 == 0 { 1 - (t as u32) } else { t as u32 };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_multinomial_logit(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y_category: &y,
                n_categories: 2,
            },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-6),
        )
        .unwrap();
        assert!(fit.converged);
        assert!(!fit.separated);
        assert!(fit.coefficients[..2].iter().all(|&c| c == 0.0));
        assert!(fit.coefficients[3] > 0.5, "slope={}", fit.coefficients[3]);
    }

    #[test]
    fn multinomial_intercept_only_three_class() {
        let n = 90usize;
        let x = vec![1.0; n];
        let mut y = vec![0u32; n];
        for i in 0..n {
            y[i] = (i % 3) as u32;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_multinomial_logit(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 1,
                y_category: &y,
                n_categories: 3,
            },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        assert!(fit.converged);
        // Equal class sizes → free intercepts ≈ 0.
        assert!(fit.coefficients[1].abs() < 0.2, "{:?}", fit.coefficients);
        assert!(fit.coefficients[2].abs() < 0.2, "{:?}", fit.coefficients);
    }

    #[test]
    fn multinomial_three_class_recovers_parent_signal() {
        // Y ∈ {0,1,2}; when X=0 prefer class 0, when X=1 prefer class 2.
        // Enough class mixing to avoid complete separation.
        let n = 180usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0u32; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = match (t < 0.5, i % 5) {
                (true, 0) | (false, 1) => 1,
                (true, 1) => 2,
                (true, _) | (false, 0) => 0,
                (false, _) => 2,
            };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_multinomial_logit(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y_category: &y,
                n_categories: 3,
            },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
        assert!(fit.coefficients[..2].iter().all(|&c| c == 0.0));
        // Class 2 vs 0: positive slope on X.
        let slope_class2 = fit.coefficients[2 * 2 + 1];
        assert!(slope_class2 > 0.5, "class2 slope={slope_class2}");
        // Softmax at X=1 should put most mass on class 2.
        let eta1 = fit.coefficients[2] + fit.coefficients[3];
        let eta2 = fit.coefficients[4] + fit.coefficients[5];
        let m = eta1.max(eta2).max(0.0);
        let p0 = (-m).exp();
        let p1 = (eta1 - m).exp();
        let p2 = (eta2 - m).exp();
        let z = p0 + p1 + p2;
        assert!(p2 / z > p0 / z && p2 / z > p1 / z, "p={:?}", [p0 / z, p1 / z, p2 / z]);
    }

    /// With `max_iter = 1`, IRLS returns after a single coefficient update. Reported
    /// deviance must match a direct recalculation from those returned coefficients
    /// (not the pre-update iterate used inside the working-response pass).
    #[test]
    fn glm_deviance_matches_returned_coefficients_at_max_iter_one() {
        let n = 40usize;
        let mut x = vec![0.0; n * 2];
        let mut y_bin = vec![0.0; n];
        let mut y_count = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y_bin[i] = if i % 7 == 0 { 1.0 - t } else { t };
            y_count[i] = if t < 0.5 { 2.0 } else { 5.0 };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let opts = GlmOptions {
            max_iter: 1,
            tol: 0.0, // force a full update even if delta is tiny
            ridge_on_separation: None,
            ..GlmOptions::default()
        };
        let design_bin = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y_bin };
        let design_count = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y_count };

        let poisson =
            fit_glm(GlmFamily::PoissonLog, design_count, &FaerBackend, &mut ws, &opts).unwrap();
        assert_eq!(poisson.iterations, 1);
        let mut poisson_re = 0.0;
        for r in 0..n {
            let eta = poisson.coefficients[0] + x[n + r] * poisson.coefficients[1];
            let mu = eta.exp().max(1e-12);
            let yi = y_count[r];
            if yi > 0.0 {
                poisson_re += 2.0 * (yi * (yi / mu).ln() - (yi - mu));
            } else {
                poisson_re += 2.0 * mu;
            }
        }
        assert!(
            (poisson.deviance - poisson_re).abs() < 1e-12,
            "poisson deviance {} vs recomputed {}",
            poisson.deviance,
            poisson_re
        );

        let logit =
            fit_glm(GlmFamily::BinomialLogit, design_bin, &FaerBackend, &mut ws, &opts).unwrap();
        assert_eq!(logit.iterations, 1);
        let mut logit_re = 0.0;
        for r in 0..n {
            let eta = logit.coefficients[0] + x[n + r] * logit.coefficients[1];
            let mu = (1.0 / (1.0 + (-eta).exp())).clamp(1e-9, 1.0 - 1e-9);
            if y_bin[r] > 0.0 {
                logit_re += -2.0 * mu.ln();
            } else {
                logit_re += -2.0 * (1.0 - mu).ln();
            }
        }
        assert!(
            (logit.deviance - logit_re).abs() < 1e-12,
            "logit deviance {} vs recomputed {}",
            logit.deviance,
            logit_re
        );

        let probit =
            fit_glm(GlmFamily::BinomialProbit, design_bin, &FaerBackend, &mut ws, &opts).unwrap();
        assert_eq!(probit.iterations, 1);
        let mut probit_re = 0.0;
        for r in 0..n {
            let eta = probit.coefficients[0] + x[n + r] * probit.coefficients[1];
            let mu = norm_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
            if y_bin[r] > 0.0 {
                probit_re += -2.0 * mu.ln();
            } else {
                probit_re += -2.0 * (1.0 - mu).ln();
            }
        }
        assert!(
            (probit.deviance - probit_re).abs() < 1e-12,
            "probit deviance {} vs recomputed {}",
            probit.deviance,
            probit_re
        );

        let nb_opts = GlmOptions { nb_alpha: NbAlphaPolicy::Fixed(0.5), ..opts };
        let nb =
            fit_glm(GlmFamily::NegativeBinomial, design_count, &FaerBackend, &mut ws, &nb_opts)
                .unwrap();
        assert_eq!(nb.iterations, 1);
        let theta = 2.0; // 1 / 0.5
        let mut nb_re = 0.0;
        for r in 0..n {
            let eta = nb.coefficients[0] + x[n + r] * nb.coefficients[1];
            let mu = eta.exp().max(1e-12);
            let yi = y_count[r];
            if yi > 0.0 {
                nb_re +=
                    2.0 * (yi * (yi / mu).ln() - (yi + theta) * ((yi + theta) / (mu + theta)).ln());
            } else {
                nb_re += 2.0 * theta * ((theta + mu) / theta).ln();
            }
        }
        assert!(
            (nb.deviance - nb_re).abs() < 1e-12,
            "negbin deviance {} vs recomputed {}",
            nb.deviance,
            nb_re
        );
    }
    #[test]
    fn weighted_multinomial_matches_fractional_category_counts() {
        for k in [2, 3] {
            let x = vec![1.0; k];
            let y: Vec<u32> = (0..k).map(|i| i as u32).collect();
            let weights: Vec<f64> = (0..k).map(|i| i as f64 + 0.5).collect();
            let fit = fit_multinomial_logit_weighted(
                MultinomialDesignRef {
                    x_colmajor: &x,
                    nrows: k,
                    ncols: 1,
                    y_category: &y,
                    n_categories: k,
                },
                Some(&weights),
                &FaerBackend,
                &mut LeastSquaresWorkspace::default(),
                &GlmOptions::default(),
            )
            .unwrap();
            assert!(fit.converged);
            for j in 1..k {
                assert!((fit.coefficients[j] - (weights[j] / weights[0]).ln()).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn multinomial_collinear_design_errors_at_every_sample_size() {
        // Intercept + x + a duplicate of x: the coefficients on x and x' are not identified.
        // An unconditional 1e-8 ridge used to make this fit "converge" for small n and error
        // only once the Gram scale exceeded 1e4; the verdict is now a property of the design.
        for n in [60usize, 6000] {
            let mut x = vec![0.0; n * 3];
            let mut y = vec![0u32; n];
            for i in 0..n {
                let t = i as f64 / n as f64;
                x[i] = 1.0;
                x[n + i] = t;
                x[2 * n + i] = t;
                y[i] = ((i * 7 + i / 3) % 3) as u32;
            }
            let err = fit_multinomial_logit(
                MultinomialDesignRef {
                    x_colmajor: &x,
                    nrows: n,
                    ncols: 3,
                    y_category: &y,
                    n_categories: 3,
                },
                &FaerBackend,
                &mut LeastSquaresWorkspace::default(),
                &GlmOptions::new(100, 1e-8),
            )
            .unwrap_err();
            assert!(matches!(err, StatsError::RankDeficient { .. }), "n={n} err={err}");
        }
    }

    #[test]
    fn separation_verdict_needs_more_than_saturation() {
        let saturated_only =
            BinomialDiagnostics { deviance: 1.0, saturated: true, perfectly_classified: false };
        // A converged fit that saturates but misclassifies a row has a finite MLE.
        assert!(!saturated_only.separated(true));
        // The same saturation without convergence is divergence: separation.
        assert!(saturated_only.separated(false));
        let perfect =
            BinomialDiagnostics { deviance: 0.0, saturated: true, perfectly_classified: true };
        assert!(perfect.separated(true));
        let interior =
            BinomialDiagnostics { deviance: 0.0, saturated: false, perfectly_classified: true };
        assert!(!interior.separated(false));
    }

    #[test]
    fn require_ok_names_the_actual_defect() {
        let base = GlmFit {
            coefficients: vec![0.0],
            iterations: 1,
            converged: true,
            separated: false,
            boundary_saturated: false,
            penalized: false,
            deviance: 0.0,
            nb_alpha: None,
            diagnostics: FitDiagnostics::new(1, None, "glm-irls", 0),
        };
        assert!(base.require_ok().is_ok());
        let separated = GlmFit { separated: true, boundary_saturated: true, ..base.clone() };
        assert!(separated.require_ok().unwrap_err().to_string().contains("separation"));
        let extreme = GlmFit { boundary_saturated: true, ..base.clone() };
        let msg = extreme.require_ok().unwrap_err().to_string();
        assert!(msg.contains("within 1e-8") && !msg.contains("(quasi-)complete"), "{msg}");
    }

    #[test]
    fn deliberate_ridge_fit_is_penalized_but_not_separated() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = if i % 10 == 0 { 1.0 - t } else { t };
        }
        let fit = fit_glm_ridge(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut LeastSquaresWorkspace::default(),
            &GlmOptions::new(200, 1e-8),
            0.5,
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.penalized && !fit.separated && !fit.boundary_saturated);
        assert!(fit.require_ok().is_ok(), "an ordinary ridge fit must pass require_ok");
    }
}
