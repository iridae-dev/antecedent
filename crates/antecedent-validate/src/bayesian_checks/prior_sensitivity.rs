//! Prior-sensitivity grids for a fitted Bayesian estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, PreparedBayesianProblem,
};
use antecedent_identify::IdentificationStatus;
use antecedent_prob::{
    ExternalPriorSource, PriorSensitivityFamily, PriorSensitivitySummary, PriorSet, PriorSpec,
    compose_external_priors_with_alphas,
};

use crate::common::RefutationReport;
use crate::error::ValidationError;

/// Default max relative range of effect means across the prior-sensitivity grid.
const DEFAULT_MAX_RELATIVE_PRIOR_RANGE: f64 = 0.5;

/// Prior sensitivity grid: isotropic scales, external α multipliers, **or**
/// variance multipliers around the resolved prior in force.
///
/// Exactly one grid is non-empty. [`Self::evaluate`] (isotropic) is only the
/// right grid when no prior was supplied; a staged / transferred prior uses
/// [`Self::evaluate_resolved_prior`] and a prior-bank compose uses
/// [`Self::evaluate_external_alpha`].
#[derive(Clone, Debug)]
pub struct PriorSensitivity {
    /// Prior scales (σ of isotropic Gaussian coefficient prior). Empty in α mode.
    pub scales: Arc<[f64]>,
    /// Multipliers on post-conflict applied alphas. Empty in isotropic scale mode.
    pub alphas: Arc<[f64]>,
    /// Multipliers on the resolved prior's coefficient variances (means kept).
    pub variance_multipliers: Arc<[f64]>,
    /// Fail when `(max−min) / scale` exceeds this, where `scale` is
    /// `max(|means…|, |original_ate|, ε)`.
    pub max_relative_range: f64,
}

/// Inputs for external α-multiplier prior sensitivity.
#[derive(Clone, Copy, Debug)]
pub struct ExternalAlphaSensitivity<'a> {
    /// Hydrated external sources (same order as composition).
    pub sources: &'a [ExternalPriorSource],
    /// Post-conflict applied alphas (length must match `sources`).
    pub alphas_applied: &'a [f64],
}

impl Default for PriorSensitivity {
    fn default() -> Self {
        Self::standard_grid()
    }
}

impl PriorSensitivity {
    /// Standard isotropic grid `{0.5, 1, 2, 5, 10, 20}` with `DEFAULT_MAX_RELATIVE_PRIOR_RANGE`.
    #[must_use]
    pub fn standard_grid() -> Self {
        Self {
            scales: Arc::from(vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
            alphas: Arc::from([]),
            variance_multipliers: Arc::from([]),
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
        }
    }

    /// Standard resolved-prior grid: coefficient variances × `{0.25, 0.5, 1, 2, 4, 10}`.
    ///
    /// Multiplier `1` reproduces the prior in force; smaller multipliers tighten it
    /// around its own means, larger ones weaken it.
    #[must_use]
    pub fn standard_resolved_grid() -> Self {
        Self {
            scales: Arc::from([]),
            alphas: Arc::from([]),
            variance_multipliers: Arc::from(vec![0.25, 0.5, 1.0, 2.0, 4.0, 10.0]),
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
        }
    }

    /// Standard external-α multiplier grid `{0, 0.25, 0.5, 0.75, 1}`.
    ///
    /// Multiplier `0` is baseline-only; `1` uses full post-conflict applied alphas.
    #[must_use]
    pub fn standard_alpha_grid() -> Self {
        Self {
            scales: Arc::from([]),
            alphas: Arc::from(vec![0.0, 0.25, 0.5, 0.75, 1.0]),
            variance_multipliers: Arc::from([]),
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
        }
    }

    /// The grid that perturbs the prior the estimator actually fits with: variance
    /// multipliers around a resolved prior when `estimator.prior` is set (an explicit or
    /// transferred prior), the isotropic scale grid otherwise (the isotropic prior *is* the prior
    /// in force). A prior-bank compose needs [`Self::standard_alpha_grid`] with
    /// [`Self::evaluate_external_alpha`] and the bank's sources instead.
    #[must_use]
    pub fn for_estimator(estimator: &BayesianGComputationAte) -> Self {
        if estimator.prior.is_some() {
            Self::standard_resolved_grid()
        } else {
            Self::standard_grid()
        }
    }

    /// Refit around the prior the estimator fits with, using whichever grid this value holds:
    /// [`Self::evaluate_resolved_prior`] for a variance-multiplier grid, [`Self::evaluate`] for a
    /// scale grid. Pair with [`Self::for_estimator`].
    ///
    /// # Errors
    ///
    /// An α-multiplier grid (it needs the prior bank's sources, see
    /// [`Self::evaluate_external_alpha`]), plus the errors of the dispatched evaluation.
    pub fn evaluate_in_force(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        if !self.variance_multipliers.is_empty() {
            self.evaluate_resolved_prior(estimator, problem, identification, workspace, ctx)
        } else if !self.alphas.is_empty() {
            Err(ValidationError::estimation_msg(
                "an α-multiplier grid needs the prior bank's sources: use evaluate_external_alpha",
            ))
        } else {
            self.evaluate(estimator, problem, identification, workspace, ctx)
        }
    }

    fn grid_len(&self) -> usize {
        if !self.variance_multipliers.is_empty() {
            self.variance_multipliers.len()
        } else if self.alphas.is_empty() {
            self.scales.len()
        } else {
            self.alphas.len()
        }
    }

    /// Refit at each variance multiplier around the resolved prior in force.
    ///
    /// `estimator.prior` must be the prior the reported posterior used. Coefficient
    /// means are kept; every coefficient variance is multiplied by the grid value;
    /// residual-variance and restriction specs are unchanged.
    ///
    /// # Errors
    ///
    /// Empty grid, no resolved prior on `estimator`, invalid multipliers, or fit failures.
    pub fn evaluate_resolved_prior(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        if self.variance_multipliers.is_empty() {
            return Err(ValidationError::estimation_msg(
                "prior sensitivity variance-multiplier grid is empty",
            ));
        }
        let resolved = estimator.prior.as_ref().ok_or_else(|| {
            ValidationError::estimation_msg(
                "resolved-prior sensitivity requires the prior in force (estimator.prior is unset)",
            )
        })?;
        let mut means = Vec::with_capacity(self.variance_multipliers.len());
        let mut sds = Vec::with_capacity(self.variance_multipliers.len());
        let mut posts = Vec::with_capacity(self.variance_multipliers.len());
        for &mult in self.variance_multipliers.iter() {
            if !mult.is_finite() || mult <= 0.0 {
                return Err(ValidationError::estimation_msg(
                    "prior sensitivity variance multiplier must be finite and positive",
                ));
            }
            let est = BayesianGComputationAte {
                n_draws: estimator.n_draws.min(200),
                prior: Some(scale_coefficient_variances(resolved, mult)?),
                ..estimator.clone()
            };
            let post = est.fit(problem, identification, workspace, ctx).map_err(|e| {
                ValidationError::estimation_msg(format!("prior sensitivity fit failed: {e}"))
            })?;
            let eq = post.effect_column().ok_or_else(|| {
                ValidationError::estimation_msg("missing effect column in sensitivity fit")
            })?;
            means.push(post.summaries.mean[eq]);
            sds.push(post.summaries.sd[eq]);
            posts.push(post);
        }
        Ok((
            PriorSensitivitySummary {
                family: PriorSensitivityFamily::ResolvedPriorVariance,
                prior_scales: Arc::from([]),
                alphas: Arc::from([]),
                variance_multipliers: Arc::clone(&self.variance_multipliers),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            },
            posts,
        ))
    }

    /// Refit Bayesian g-comp at each prior scale; return sensitivity summary.
    ///
    /// # Errors
    ///
    /// Fit failures or empty scale grid.
    pub fn evaluate(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        if self.scales.is_empty() {
            return Err(ValidationError::estimation_msg(
                "prior sensitivity scale grid is empty (use evaluate_external_alpha for α mode)",
            ));
        }
        // Every refit below replaces the prior with an isotropic one; against a supplied prior
        // that would report the sensitivity of a prior the analysis did not use.
        if estimator.prior.is_some() {
            return Err(ValidationError::estimation_msg(
                "the estimator fits with a supplied prior: use evaluate_resolved_prior (or \
                 evaluate_in_force), not the isotropic scale grid",
            ));
        }
        let mut means = Vec::with_capacity(self.scales.len());
        let mut sds = Vec::with_capacity(self.scales.len());
        let mut posts = Vec::with_capacity(self.scales.len());
        for &scale in self.scales.iter() {
            let est = BayesianGComputationAte {
                prior_scale: scale,
                n_draws: estimator.n_draws.min(200),
                prior: None,
                ..estimator.clone()
            };
            let post = est.fit(problem, identification, workspace, ctx).map_err(|e| {
                ValidationError::estimation_msg(format!("prior sensitivity fit failed: {e}"))
            })?;
            let eq = post.effect_column().ok_or_else(|| {
                ValidationError::estimation_msg("missing effect column in sensitivity fit")
            })?;
            means.push(post.summaries.mean[eq]);
            sds.push(post.summaries.sd[eq]);
            posts.push(post);
        }
        Ok((
            PriorSensitivitySummary {
                family: PriorSensitivityFamily::IsotropicScale,
                prior_scales: Arc::clone(&self.scales),
                alphas: Arc::from([]),
                variance_multipliers: Arc::from([]),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            },
            posts,
        ))
    }

    /// Refit at each α-multiplier on post-conflict applied alphas (external prior bank).
    ///
    /// For multiplier `m`, composed alphas are `m * alphas_applied[k]` (clamped to `[0, 1]`).
    ///
    /// # Errors
    ///
    /// Empty α grid, length mismatch, compose failures, or fit failures.
    pub fn evaluate_external_alpha(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
        external: ExternalAlphaSensitivity<'_>,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        if self.alphas.is_empty() {
            return Err(ValidationError::estimation_msg("prior sensitivity alpha grid is empty"));
        }
        if external.sources.len() != external.alphas_applied.len() {
            return Err(ValidationError::estimation_msg(
                "evaluate_external_alpha: sources / alphas_applied length mismatch",
            ));
        }
        let n_coef = problem.design.ncols;
        let baseline = PriorSet::weakly_informative(n_coef);
        let requested: Vec<f64> = external.sources.iter().map(|s| s.weight.alpha).collect();
        let mut means = Vec::with_capacity(self.alphas.len());
        let mut sds = Vec::with_capacity(self.alphas.len());
        let mut posts = Vec::with_capacity(self.alphas.len());
        for &mult in self.alphas.iter() {
            if !mult.is_finite() || !(0.0..=1.0).contains(&mult) {
                return Err(ValidationError::estimation_msg(
                    "prior sensitivity alpha multiplier must be finite and in [0, 1]",
                ));
            }
            let scaled: Vec<f64> =
                external.alphas_applied.iter().map(|&a| (a * mult).clamp(0.0, 1.0)).collect();
            let composed = compose_external_priors_with_alphas(
                external.sources,
                &requested,
                &scaled,
                &baseline,
            )
            .map_err(|e| {
                ValidationError::estimation_msg(format!("prior sensitivity compose failed: {e}"))
            })?;
            let est = BayesianGComputationAte {
                n_draws: estimator.n_draws.min(200),
                prior: Some(composed.prior),
                ..estimator.clone()
            };
            let post = est.fit(problem, identification, workspace, ctx).map_err(|e| {
                ValidationError::estimation_msg(format!("prior sensitivity α fit failed: {e}"))
            })?;
            let eq = post.effect_column().ok_or_else(|| {
                ValidationError::estimation_msg("missing effect column in α sensitivity fit")
            })?;
            means.push(post.summaries.mean[eq]);
            sds.push(post.summaries.sd[eq]);
            posts.push(post);
        }
        Ok((
            PriorSensitivitySummary {
                family: PriorSensitivityFamily::ExternalAlpha,
                prior_scales: Arc::from([]),
                alphas: Arc::clone(&self.alphas),
                variance_multipliers: Arc::from([]),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            },
            posts,
        ))
    }

    /// Convert sensitivity range into a refutation-style report.
    ///
    /// Passes when the relative range of effect means is finite and
    /// `≤ max_relative_range`.
    #[must_use]
    pub fn to_report(
        &self,
        summary: &PriorSensitivitySummary,
        original_ate: f64,
    ) -> RefutationReport {
        let min = summary.effect_means.iter().copied().fold(f64::INFINITY, f64::min);
        let max = summary.effect_means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denom =
            summary.effect_means.iter().copied().map(f64::abs).fold(original_ate.abs(), f64::max);
        let informative = !summary.effect_means.is_empty()
            && original_ate.is_finite()
            && summary.effect_means.iter().all(|mean| mean.is_finite());
        let relative = if !informative {
            f64::NAN
        } else if denom == 0.0 {
            0.0
        } else {
            // Divide before subtraction to avoid overflow for opposite signs.
            max / denom - min / denom
        };
        let passed = relative.is_finite() && relative <= self.max_relative_range;
        let kind = match summary.family {
            PriorSensitivityFamily::IsotropicScale => "prior_sensitivity",
            PriorSensitivityFamily::ExternalAlpha => "prior_sensitivity_alpha",
            PriorSensitivityFamily::ResolvedPriorVariance => "prior_sensitivity_resolved_prior",
        };
        RefutationReport {
            refuter: Arc::from(kind),
            original_ate,
            refuted_ate: summary.effect_means.last().copied().unwrap_or(original_ate),
            comparison: relative,
            informative,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "prior sensitivity relative range {relative} exceeds max {}",
                    self.max_relative_range
                )))
            },
            replicates: u32::try_from(self.grid_len()).unwrap_or(u32::MAX),
        }
    }
}

/// Copy of `prior` with every Gaussian coefficient variance multiplied by `mult`.
fn scale_coefficient_variances(prior: &PriorSet, mult: f64) -> Result<PriorSet, ValidationError> {
    let mut scaled = prior.clone();
    let mut found = false;
    for spec in &mut scaled.specs {
        if let PriorSpec::GaussianCoefficients(coef) = spec {
            coef.variance = coef.variance.iter().map(|v| v * mult).collect();
            found = true;
        }
    }
    if !found {
        return Err(ValidationError::estimation_msg(
            "resolved prior has no Gaussian coefficient block to perturb",
        ));
    }
    Ok(scaled)
}

/// Attach prior sensitivity onto a [`CausalPosterior`].
#[must_use]
pub fn with_prior_sensitivity(
    mut posterior: CausalPosterior,
    summary: PriorSensitivitySummary,
) -> CausalPosterior {
    posterior.prior_sensitivity = Some(summary);
    posterior
}
