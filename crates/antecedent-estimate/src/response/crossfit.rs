//! Cross-fitting machinery for the scalar-curve nuisance regressions.
//!
//! The additive-GAM nuisance fits (raw and design-expanded), the prepared
//! cross-fit fold, the fold-size guard, and the Gaussian treatment residual
//! scales used to build the Kennedy conditional density.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    AdditiveDesign, FaerBackend, GamOptions, GamWorkspace, SmoothSpec, fit_gam,
    fit_gam_weighted_design,
};

use super::CompleteSample;
use crate::EstimationError;

pub(super) fn fit_additive(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    basis: usize,
    lambda: f64,
    workspace: &mut GamWorkspace,
) -> Result<antecedent_stats::GamFit, EstimationError> {
    let specs: Vec<SmoothSpec> =
        (0..ncols).map(|col| SmoothSpec::new(col, basis, lambda)).collect();
    // Response nuisances use a longer backfitting budget than the GAM default: the default
    // 100-iteration / 1e-6 tolerance combination routinely returns converged=false on the
    // cross-fitted Kennedy fixtures while the fit itself is already stable enough to use.
    let fit = fit_gam(
        x,
        nrows,
        ncols,
        y,
        &specs,
        &GamOptions { max_iter: 500, tol: 1e-6 },
        &FaerBackend,
        workspace,
    )?;
    // Observation logistics already call `GlmFit::require_ok`. An unfinished backfit after the
    // extended budget is refused rather than published into a Kennedy curve or ADE.
    require_converged_gam(fit, GAM_NUISANCE_NOT_CONVERGED)
}

/// Refusal of an unfinished additive-GAM nuisance backfit.
pub(super) const GAM_NUISANCE_NOT_CONVERGED: &str =
    "additive GAM nuisance did not converge; refuse rather than publish an unfinished fit";

/// Refusal of an unfinished row-weighted additive-GAM nuisance backfit.
pub(super) const WEIGHTED_GAM_NUISANCE_NOT_CONVERGED: &str =
    "weighted additive GAM nuisance did not converge; refuse rather than publish an unfinished fit";

/// What a caller can change when an additive-GAM backfit does not settle within
/// its budget. Every option named here is a `ContinuousResponseOptions` field
/// (and a `response_options` key in Python).
pub(super) const GAM_NOT_CONVERGED_REMEDY: &str = "standardize the treatment and adjustment \
     columns, drop or coarsen near-collinear or near-constant adjustment covariates, or make \
     the additive fit smoother: raise nuisance_lambda or lower nuisance_basis in the response \
     options";

/// `fit` when its backfit converged; otherwise the refusal `message`, naming
/// [`GAM_NOT_CONVERGED_REMEDY`]. The one owner of the additive-GAM
/// non-convergence refusal, so every such refusal carries the same remedy.
pub(super) fn require_converged_gam(
    fit: antecedent_stats::GamFit,
    message: &'static str,
) -> Result<antecedent_stats::GamFit, EstimationError> {
    if fit.converged {
        Ok(fit)
    } else {
        Err(EstimationError::unsupported_with_remedy(message, GAM_NOT_CONVERGED_REMEDY))
    }
}

/// [`fit_additive`] on a design expanded once ([`AdditiveDesign`]), under
/// optional row weights: the same backfitting budget and the same refusal of
/// an unfinished fit.
pub(super) fn fit_additive_design(
    design: &AdditiveDesign,
    y: &[f64],
    weights: Option<&[f64]>,
    workspace: &mut GamWorkspace,
) -> Result<antecedent_stats::GamFit, EstimationError> {
    let fit = fit_gam_weighted_design(
        design,
        y,
        &GamOptions { max_iter: 500, tol: 1e-6 },
        weights,
        workspace,
    )?;
    require_converged_gam(
        fit,
        if weights.is_some() {
            WEIGHTED_GAM_NUISANCE_NOT_CONVERGED
        } else {
            GAM_NUISANCE_NOT_CONVERGED
        },
    )
}

/// One cross-fitting fold of a [`CompleteSample`], prepared once per
/// execution: its row split, the training rows' outcome and treatment
/// vectors, and the training nuisance designs with their quantile knots and
/// B-spline bases already expanded ([`AdditiveDesign`]).
///
/// A Bayesian-bootstrap draw loop refits the fold's nuisances once per draw
/// under new row weights; the rows, knots and bases are the same for every
/// draw, so re-expanding them per fit (a sort of every training column and a
/// full basis evaluation, twice per fold per draw) was pure repetition. The
/// refit on a prepared fold is bit-identical to the per-fit expansion.
pub(super) struct CrossFitFold {
    /// Training rows (indices into the sample), ascending.
    pub(super) train: Vec<usize>,
    /// Held-out rows, ascending.
    pub(super) valid: Vec<usize>,
    /// Outcome on the training rows.
    pub(super) outcome: Vec<f64>,
    /// Primary treatment on the training rows.
    pub(super) treatment: Vec<f64>,
    /// Outcome nuisance over the training rows' raw predictors.
    pub(super) outcome_design: AdditiveDesign,
    /// Treatment nuisance over the training rows' adjusters; absent without adjusters.
    pub(super) treatment_design: Option<AdditiveDesign>,
}

pub(super) fn treatment_sigma(
    sample: &CompleteSample,
    train: &[usize],
    fit: Option<&antecedent_stats::GamFit>,
) -> Result<f64, EstimationError> {
    let mean = sample.train_treatment_mean(train);
    let (rss, denominator) = if let Some(fit) = fit {
        // The residual scale of a penalized GAM uses effective degrees of freedom, not
        // n−1. Dividing by n−1 understates σ whenever edf > 1, which peaks the Kennedy
        // conditional density and inflates the Gaussian-score Riesz representer.
        let df = (train.len() as f64 - fit.edf_approx).max(1.0);
        (fit.residuals.iter().map(|v| v * v).sum::<f64>(), df)
    } else {
        (
            train.iter().map(|&i| (sample.treatments[i] - mean).powi(2)).sum(),
            train.len().saturating_sub(1).max(1) as f64,
        )
    };
    let sigma = (rss / denominator).sqrt();
    if !sigma.is_finite() || sigma <= f64::EPSILON {
        return Err(EstimationError::unsupported(
            "Gaussian treatment nuisance has degenerate residual variance",
        ));
    }
    Ok(sigma)
}

/// Weighted residual scale of a fold's treatment nuisance: `Σ w r² / (Σ w − edf)`
/// over the training rows, with weights rescaled to sum to the row count (the
/// normalization `fit_gam_weighted` applies to the penalty).
pub(super) fn treatment_sigma_train_weighted(
    sample: &CompleteSample,
    train: &[usize],
    train_weights: &[f64],
    fit: Option<&antecedent_stats::GamFit>,
) -> Result<f64, EstimationError> {
    let total: f64 = train_weights.iter().sum();
    let rows = train.len() as f64;
    if train_weights.len() != train.len() || !total.is_finite() || total <= 0.0 {
        return Err(EstimationError::stats_msg("fold training weights are degenerate"));
    }
    let scaled = |w: f64| w * rows / total;
    let (rss, denominator) = if let Some(fit) = fit {
        let rss = fit
            .residuals
            .iter()
            .zip(train_weights)
            .map(|(residual, &w)| scaled(w) * residual * residual)
            .sum::<f64>();
        (rss, (rows - fit.edf_approx).max(1.0))
    } else {
        let mean =
            train.iter().zip(train_weights).map(|(&i, &w)| w * sample.treatments[i]).sum::<f64>()
                / total;
        let rss = train
            .iter()
            .zip(train_weights)
            .map(|(&i, &w)| scaled(w) * (sample.treatments[i] - mean).powi(2))
            .sum::<f64>();
        (rss, (rows - 1.0).max(1.0))
    };
    let sigma = (rss / denominator).sqrt();
    if !sigma.is_finite() || sigma <= f64::EPSILON {
        return Err(EstimationError::unsupported(
            "weighted Gaussian treatment nuisance has degenerate residual variance",
        ));
    }
    Ok(sigma)
}

pub(super) fn ensure_crossfit_size(
    n: usize,
    folds: usize,
    basis: usize,
) -> Result<(), EstimationError> {
    if folds > n {
        return Err(EstimationError::unsupported(
            "cross-fitting folds cannot exceed complete observations",
        ));
    }
    let smallest_train = n - n.div_ceil(folds);
    if smallest_train <= basis + 2 {
        return Err(EstimationError::unsupported(
            "too few complete rows for requested cross-fitting and nuisance basis",
        ));
    }
    Ok(())
}
