//! Bayesian mechanisms, g-computation, and posterior functional evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

mod evaluator;
mod gcomp;
mod hydrate;
#[cfg(test)]
mod tests;

pub use evaluator::{CompiledGCompAte, GCompAteEvaluator, PosteriorFunctionalEvaluator};
pub use gcomp::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    LAPLACE_RANK_INTERVAL_BELOW_DRAWS, PreparedBayesianProblem,
};
#[cfg(test)]
pub(crate) use hydrate::HYDRATED_COEFFICIENT_COVARIANCE_ID;
#[cfg(test)]
pub(crate) use hydrate::HYDRATED_COVARIANCE_DROPPED_ID;
#[cfg(test)]
pub(crate) use hydrate::hydrate_prior_from_posterior;
#[cfg(test)]
pub(crate) use hydrate::hydrate_prior_from_quantity_summaries;
pub use hydrate::{
    HydrateMapping, HydrateSource, HydrateTarget, coefficient_covariance_from_draws, hydrate_prior,
    hydrate_prior_from_coefficient_moments,
};

use std::sync::Arc;

use antecedent_core::IdentificationStatus;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ExecutionContext, StreamDomain,
};
use antecedent_prob::{
    ConflictSummary, InferenceDiagnostics, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema,
    PosteriorSummary, PriorSensitivitySummary, PriorSet,
};

use crate::error::EstimationError;

/// Posterior draws of the linear functional `weights' β`, one per retained draw.
/// `weights` is a design-row average after the intervention overlay.
pub(crate) fn linear_response_draws(
    posterior: &CausalPosterior,
    weights: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let mut values = vec![0.0; posterior.draws.n_draws];
    for (index, &weight) in weights.iter().enumerate() {
        let column = posterior
            .draws
            .schema
            .quantities
            .iter()
            .position(|q| {
                matches!(q,
            PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index)
            })
            .ok_or_else(|| EstimationError::stats_msg("response posterior missing coefficient"))?;
        for (value, &coefficient) in values.iter_mut().zip(posterior.draws.column(column)?) {
            *value += weight * coefficient;
        }
    }
    Ok(values)
}

/// `(mean, lower, upper, sd)` of linear-functional draws at an equal-tailed `level`.
pub(crate) fn summarize_linear_response_draws(
    mut values: Vec<f64>,
    level: f64,
) -> Result<(f64, f64, f64, f64), EstimationError> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "response posterior needs at least two finite draws",
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd =
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt();
    values.sort_by(f64::total_cmp);
    // Exchangeable-rank (type-6) ranks so a finite-draw interval covers at its level;
    // the same rule as every other posterior-draw interval in the crate.
    let (lower, upper) = antecedent_stats::equal_tail_interval_sorted(
        &values,
        level,
        antecedent_stats::QuantileRule::ExchangeableRank,
    );
    Ok((mean, lower, upper, sd))
}

/// Minimum kept draws for HMC so the MCMC publication gate (Ř≤1.01, ESS≥100)
/// is reachable on typical Gaussian GLMs.
pub(crate) const HMC_MIN_DRAWS: usize = 3_000;

/// Stable prefix of the inference-diagnostics note recorded when [`HMC_MIN_DRAWS`]
/// raised the requested draw count (`requested=<n> used=<m>`).
pub(crate) const HMC_DRAW_FLOOR_NOTE_PREFIX: &str = "hmc.draw_floor";

/// Diagnostics note when `unit_ids` request random-intercept GLS whitening under a
/// non-Gaussian likelihood. Whitening is a Gaussian linear transform and must not be
/// applied to count or binary outcomes.
pub(crate) const RANDOM_INTERCEPT_WHITEN_SKIP_NOTE: &str = "random_intercept.gls_whiten.skipped: compound-symmetry GLS whitening applies only under \
     GaussianIdentity; unit_ids were ignored for this likelihood";

/// Diagnostics note when `unit_ids` request random-intercept GLS whitening of a
/// Gaussian fit but whitening cannot apply (fewer than two units, no unit with
/// repeated rows, or no positive between-unit variance component), so the rows are
/// fitted as independent.
pub(crate) const RANDOM_INTERCEPT_WHITEN_FALLBACK_NOTE: &str = "random_intercept.gls_whiten.fallback_iid: unit_ids were supplied but compound-symmetry GLS \
     whitening did not apply (degenerate grouping or no positive between-unit variance), so \
     the rows were fitted as independent";

/// `(requested, used)` draw counts when the HMC draw floor raised the request.
#[must_use]
pub fn hmc_draw_floor_from_notes(notes: &[Arc<str>]) -> Option<(usize, usize)> {
    notes.iter().find_map(|note| {
        let rest = note.strip_prefix(HMC_DRAW_FLOOR_NOTE_PREFIX)?;
        let mut requested = None;
        let mut used = None;
        for kv in rest.split_whitespace() {
            if let Some(v) = kv.strip_prefix("requested=") {
                requested = v.parse().ok();
            } else if let Some(v) = kv.strip_prefix("used=") {
                used = v.parse().ok();
            }
        }
        Some((requested?, used?))
    })
}

/// Causal posterior over an identified functional.
#[derive(Clone, Debug)]
pub struct CausalPosterior {
    /// Columnar effect (and optional coefficient) draws.
    pub draws: PosteriorDraws,
    /// Summary of `draws`.
    pub summaries: PosteriorSummary,
    /// Identification status — priors never upgrade this.
    pub identification: IdentificationStatus,
    /// Optional prior-sensitivity grid.
    pub prior_sensitivity: Option<PriorSensitivitySummary>,
    /// Optional external-prior conflict shrink summary.
    pub conflict_summary: Option<ConflictSummary>,
    /// Inference diagnostics.
    pub diagnostics: InferenceDiagnostics,
    /// Assumptions including prior restrictions.
    pub assumptions: AssumptionSet,
    /// Unidentified graph mass retained when aggregating envelopes (0 if single graph).
    ///
    /// Mass of graph atoms that were structurally not identified. Never
    /// includes atoms a latency tier skipped (see [`Self::subsampled_out_mass`])
    /// or atoms that were identified but whose estimation failed (see
    /// [`Self::unevaluable_mass`]) — a refusal to estimate is not a proof of
    /// non-identification.
    pub unidentified_mass: f64,
    /// Identified graph mass the Interactive latency tier left out of the
    /// envelope subsample (0 outside that tier). Those atoms were not evaluated,
    /// so this is neither unidentified mass nor part of the published mixture;
    /// the identified-atom mixture covers `1 − unidentified − unevaluable − subsampled_out`.
    pub subsampled_out_mass: f64,
    /// Identified graph mass excluded from the mixture because estimation
    /// (design preparation, fitting, composition, or draw extraction) failed
    /// on that atom, not because identification failed. Zero unless the
    /// caller separates the two reasons; a path that has not yet been audited
    /// for this distinction may still fold this mass into
    /// [`Self::unidentified_mass`].
    pub unevaluable_mass: f64,
    /// Adaptive draw early-stop (Laplace / conjugate Gaussian redraw path).
    pub early_stopped: bool,
    /// Source treatment contrast `active − control` used to form the effect.
    ///
    /// Recorded so `EffectFunctional` hydrate can map `ATE → β_T` as `ATE/Δ`.
    /// `None` when the posterior is not an identity-link contrast (or the
    /// contrast was not available at fit time).
    pub treatment_contrast: Option<f64>,
}

impl CausalPosterior {
    /// Primary effect column index (first `Effect` quantity), if any.
    #[must_use]
    pub fn effect_column(&self) -> Option<usize> {
        self.draws
            .schema
            .quantities
            .iter()
            .position(|q| matches!(q, PosteriorQuantityKind::Effect { .. }))
    }

    /// Empirical P(effect < threshold) for the primary effect column.
    ///
    /// # Errors
    ///
    /// Missing effect column.
    pub fn probability_below(&self, threshold: f64) -> Result<f64, EstimationError> {
        let q = self
            .effect_column()
            .ok_or_else(|| EstimationError::stats_msg("CausalPosterior has no effect column"))?;
        self.draws.probability_below(q, threshold).map_err(EstimationError::from)
    }
}

/// Bayesian posteriors need at least two draws to form a finite SD / interval.
///
/// This is an API refuse, not a support-matrix cell refuse. Callers must not
/// silently rewrite `0` or `1` to `2`.
///
/// # Errors
///
/// `n_draws < 2`.
pub(crate) fn require_bayesian_n_draws(n_draws: usize) -> Result<usize, EstimationError> {
    if n_draws < 2 {
        return Err(EstimationError::stats_msg(format!(
            "Bayesian inference requires n_draws >= 2; got {n_draws} (refusing silent rewrite)"
        )));
    }
    Ok(n_draws)
}

/// Build a non-identified posterior artifact that still records priors (exit criterion #2).
///
/// Samples prior-predictive draws of the scalar effect from the explicit effect prior
/// `N(effect_mean, effect_sd²)` so Bayesian envelopes can surface uncertainty without
/// inventing identification; `prior` is recorded as assumptions. The effect prior is
/// passed explicitly because a design-shaped [`PriorSet`] has no single effect
/// coefficient (its first entry is the intercept) and its Gaussian variances are
/// σ²-relative, not an effect SD. Status remains [`IdentificationStatus::NotIdentified`].
///
/// # Errors
///
/// A non-finite `effect_mean` or a non-finite / non-positive `effect_sd`.
pub fn nonidentified_with_prior(
    prior: &PriorSet,
    effect_mean: f64,
    effect_sd: f64,
    diagnostics: InferenceDiagnostics,
    n_draws: usize,
    seed: u64,
) -> Result<CausalPosterior, EstimationError> {
    if !effect_mean.is_finite() || !effect_sd.is_finite() || effect_sd <= 0.0 {
        return Err(EstimationError::unsupported(
            "non-identified prior-predictive effect needs a finite mean and a positive finite sd",
        ));
    }
    let mut assumptions = AssumptionSet::new();
    for spec in &prior.specs {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::PriorRestriction(spec.as_assumption()),
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Untestable,
        });
    }
    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate") }]),
    };
    let (mean, scale) = (effect_mean, effect_sd);
    let n = n_draws.max(1);
    let mut values = vec![0.0; n];
    let mut rng =
        ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Bayesian, 0xBA7E_u64);
    for v in &mut values {
        *v = mean + scale * antecedent_kernels::standard_normal(&mut rng);
    }
    let draws = PosteriorDraws::from_column_major(schema, n, Arc::<[f64]>::from(values))
        .unwrap_or_else(|_| PosteriorDraws {
            schema: PosteriorSchema {
                quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate") }]),
            },
            n_draws: 0,
            values: Arc::from([]),
        });
    let summaries = draws.summarize();
    Ok(CausalPosterior {
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        draws,
        summaries,
        identification: IdentificationStatus::NotIdentified,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: 1.0,
        early_stopped: false,
        treatment_contrast: None,
    })
}
