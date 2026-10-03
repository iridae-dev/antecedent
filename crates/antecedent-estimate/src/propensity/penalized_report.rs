//! What a penalized-propensity AIPW fit reports beside its point and score table.
//!
//! The point, the scores and the intervals ride on the ordinary `EffectEstimate` fields
//! (`se_analytic`, `se_bootstrap`); this report carries what those scalars cannot: the failed
//! GLM fit and the fallback that replaced it, the support a lasso kept on each fold, and the
//! penalties the refit bootstrap selected in every replicate (`docs/guides/penalized-aipw.md`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{GlmRefusalKind, StatsError};

use crate::error::EstimationError;
use crate::scores::ScoreTable;

/// `uncertainty_kind` of the refit-bootstrap standard error: a bootstrap SE of the whole
/// pipeline (fold plan, penalty selection and nuisance fits repeated on each resample). Its
/// coverage is what the penalized-AIPW calibration suite measures; the SE itself states no
/// coverage.
pub const REFIT_BOOTSTRAP_UNCERTAINTY_KIND: &str = "refit_bootstrap_se_repeating_penalty_selection";

/// Stage name of a failed propensity fit.
pub const PROPENSITY_FIT_STAGE: &str = "propensity_fit";

/// A GLM propensity fit that failed: where, why, and the failure's own text.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct FailedFit {
    /// Pipeline stage that failed ([`PROPENSITY_FIT_STAGE`]).
    pub stage: &'static str,
    /// Outer cross-fit fold whose training fit failed.
    pub fold: usize,
    /// Failure class: `separated`, `non_converged`, `boundary_saturated`, `rank_deficient` or
    /// `fit_failed`.
    pub reason: &'static str,
    /// The failure's rendered message.
    pub message: String,
}

impl FailedFit {
    /// Classify a propensity-fit `error` of cross-fit fold `fold`.
    #[must_use]
    pub(crate) fn from_error(fold: usize, error: &EstimationError) -> Self {
        let reason = match error {
            EstimationError::Stats(StatsError::GlmRefused { kind, .. }) => match kind {
                GlmRefusalKind::Separated => "separated",
                GlmRefusalKind::NonConverged => "non_converged",
                GlmRefusalKind::BoundarySaturated => "boundary_saturated",
            },
            EstimationError::Stats(StatsError::RankDeficient { .. }) => "rank_deficient",
            _ => "fit_failed",
        };
        Self { stage: PROPENSITY_FIT_STAGE, fold, reason, message: error.to_string() }
    }
}

/// A declared GLM-to-ridge fallback that ran: the failed fit and the selected destination.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct FallbackRecord {
    /// The GLM fit that failed.
    pub failed_fit: FailedFit,
    /// Canonical key of the fallback route that produced the result.
    pub destination: String,
}

/// Covariates a lasso propensity kept on one cross-fit fold's training rows.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct FoldSupport {
    /// Outer cross-fit fold.
    pub fold: usize,
    /// Design columns (1-based; column 0 is the intercept) with a nonzero coefficient.
    pub columns: Vec<usize>,
    /// Covariate names of [`Self::columns`].
    pub names: Vec<String>,
}

/// Penalties the refit bootstrap selected inside one successful replicate, one per fold.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ReplicatePenalties {
    /// Replicate index in `0..requested`.
    pub replicate: u32,
    /// Selected penalty of each outer fold of the replicate's fold plan.
    pub lambdas: Vec<f64>,
}

/// What the refit bootstrap of a penalized cross-fitted AIPW did beside the scalar SE it
/// publishes (`EffectEstimate::se_bootstrap`).
///
/// Every replicate resamples the units (rows) and re-runs the whole pipeline on the resample:
/// the arm-stratified unit fold plan, each fold's inner-CV penalty selection (and a lasso's
/// support selection) and the nuisance fits.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PenalizedVarianceReport {
    /// What [`Self::refit_bootstrap_se`] is ([`REFIT_BOOTSTRAP_UNCERTAINTY_KIND`]).
    pub uncertainty_kind: &'static str,
    /// Sample standard deviation of the successful replicates' contrasts; `None` with fewer
    /// than two successes or more than half of the attempted replicates failing.
    pub refit_bootstrap_se: Option<f64>,
    /// Replicates requested.
    pub replicates_requested: u32,
    /// Replicates that produced a finite contrast.
    pub replicates_ok: u32,
    /// Attempted replicates that failed softly (a fold missing an arm, a failed fit, a
    /// boundary propensity).
    pub replicates_failed: u32,
    /// Cooperative cancellation stopped the loop early.
    pub cancelled: bool,
    /// Per-replicate selected penalties of the successful replicates, in replicate order.
    pub replicate_penalties: Vec<ReplicatePenalties>,
    /// Cross-fitted influence-function standard error of the out-of-fold scores, for
    /// comparison: `sqrt(sum (d_i - mean d)^2 / (n (n - 1)))` with `d_i = phi1_i - phi0_i`.
    /// It treats the out-of-fold nuisances as fixed and is justified only under the remainder
    /// condition the guide states.
    pub influence_se: f64,
}

/// Everything a penalized-propensity fit records beyond the point and the score table.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct PenalizedReport {
    /// The GLM-to-ridge fallback, when the declared fallback ran.
    pub fallback: Option<FallbackRecord>,
    /// Per-fold selected support of a lasso propensity (empty for ridge).
    pub selected_support: Vec<FoldSupport>,
    /// The opt-in refit-bootstrap standard error, when requested.
    pub variance: Option<PenalizedVarianceReport>,
}

impl PenalizedReport {
    /// Whether the report carries nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fallback.is_none() && self.selected_support.is_empty() && self.variance.is_none()
    }
}

/// Cross-fitted influence-function standard error of a binary score table's contrast: the
/// iid sample variance of `d_i = phi1_i - phi0_i` over `n`, `sum (d_i - dbar)^2 / (n (n - 1))`.
///
/// Equal to the table's own contrast standard error; it is the variance the route would
/// publish under the remainder condition (see `docs/guides/penalized-aipw.md`).
///
/// # Errors
///
/// A table without a control and a treated column, or fewer than two rows.
pub fn crossfit_influence_se(table: &ScoreTable) -> Result<f64, EstimationError> {
    let (control, treated) = (table.column(0)?, table.column(1)?);
    let n = control.len();
    if n < 2 {
        return Err(EstimationError::data_msg("an influence standard error needs two rows"));
    }
    let nf = n as f64;
    let contrast: Vec<f64> = control.iter().zip(treated).map(|(c, t)| t - c).collect();
    let mean = contrast.iter().sum::<f64>() / nf;
    let squares: f64 = contrast.iter().map(|d| (d - mean) * (d - mean)).sum();
    Ok((squares / (nf * (nf - 1.0))).sqrt())
}
