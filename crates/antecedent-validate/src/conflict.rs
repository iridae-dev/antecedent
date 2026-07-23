//! Conflict policy: shrink external prior α from prior-PPC / KL signals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::PreparedBayesianProblem;
use antecedent_prob::{
    ComposedPrior, ConflictSummary, ExternalPriorSource, PriorSet,
    compose_external_priors_with_alphas,
};
use antecedent_stats::gaussian_kl;

use crate::bayesian_checks::PriorPredictiveCheck;
use crate::error::ValidationError;

/// Documented defaults for conflict → α shrink.
///
/// `α' = α · 1{p > p_min} · exp(−kl_scale · kl)`, clipped to `[0, α]` (never
/// increases α). Missing signals contribute a factor of `1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConflictPolicy {
    /// Minimum prior-PPC p-value; at or below this, α is zeroed by the indicator.
    pub p_min: f64,
    /// Scale on Gaussian KL (nats) in the exponential shrink term.
    pub kl_scale: f64,
}

impl Default for ConflictPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictPolicy {
    /// Defaults: `p_min = 0.05`, `kl_scale = 1.0`.
    #[must_use]
    pub const fn new() -> Self {
        Self { p_min: 0.05, kl_scale: 1.0 }
    }

    /// Construct with validation.
    ///
    /// # Errors
    ///
    /// Non-finite or out-of-range parameters (`p_min` must be in `[0, 1]`,
    /// `kl_scale ≥ 0`).
    pub fn try_new(p_min: f64, kl_scale: f64) -> Result<Self, ValidationError> {
        if !p_min.is_finite() || !(0.0..=1.0).contains(&p_min) {
            return Err(ValidationError::estimation_msg(
                "ConflictPolicy p_min must be finite and in [0, 1]",
            ));
        }
        if !kl_scale.is_finite() || kl_scale < 0.0 {
            return Err(ValidationError::estimation_msg(
                "ConflictPolicy kl_scale must be finite and >= 0",
            ));
        }
        Ok(Self { p_min, kl_scale })
    }

    /// Shrink one alpha given optional conflict signals.
    ///
    /// Missing `p` / `kl` skip that factor (treated as no evidence of conflict).
    #[must_use]
    pub fn shrink_alpha(self, alpha: f64, p: Option<f64>, kl: Option<f64>) -> f64 {
        if !(0.0..=1.0).contains(&alpha) || !alpha.is_finite() {
            return 0.0;
        }
        let mut factor = 1.0;
        if let Some(pval) = p {
            if pval.is_finite() && pval <= self.p_min {
                factor = 0.0;
            }
        }
        if let Some(k) = kl {
            if k.is_finite() && k > 0.0 && self.kl_scale > 0.0 {
                factor *= (-self.kl_scale * k).exp();
            }
        }
        (alpha * factor).clamp(0.0, alpha)
    }
}

/// Per-source conflict signals (optional fields).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConflictSignals {
    /// Prior-PPC two-sided p-value.
    pub p_value: Option<f64>,
    /// Gaussian KL (nats) between prior-predictive and observed summaries.
    pub kl: Option<f64>,
}

/// Attach a [`ConflictSummary`] onto a [`antecedent_estimate::CausalPosterior`].
#[must_use]
pub fn with_conflict_summary(
    mut posterior: antecedent_estimate::CausalPosterior,
    summary: ConflictSummary,
) -> antecedent_estimate::CausalPosterior {
    posterior.conflict_summary = Some(summary);
    posterior
}

/// Compose external priors, then shrink α from conflict signals and recompose.
///
/// When `signals` is shorter than `sources`, missing entries are treated as
/// empty (no shrink from that source's signals).
///
/// # Errors
///
/// Composition failures from `antecedent-prob`.
pub fn apply_conflict_and_compose(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
    policy: &ConflictPolicy,
    signals: &[ConflictSignals],
) -> Result<(ComposedPrior, ConflictSummary), ValidationError> {
    let requested: Vec<f64> = sources.iter().map(|s| s.weight.alpha).collect();
    let mut applied = Vec::with_capacity(sources.len());
    let mut p_values = Vec::with_capacity(sources.len());
    let mut kl_values = Vec::with_capacity(sources.len());
    for (i, src) in sources.iter().enumerate() {
        let sig = signals.get(i).copied().unwrap_or_default();
        let a = policy.shrink_alpha(src.weight.alpha, sig.p_value, sig.kl);
        applied.push(a);
        p_values.push(sig.p_value);
        kl_values.push(sig.kl);
    }
    let composed = compose_external_priors_with_alphas(sources, &requested, &applied, baseline)?;
    let summary = ConflictSummary {
        source_ids: Arc::clone(&composed.source_ids),
        alphas_requested: Arc::clone(&composed.alphas_requested),
        alphas_applied: Arc::clone(&composed.alphas_applied),
        p_values: Arc::from(p_values),
        kl_values: Arc::from(kl_values),
    };
    Ok((composed, summary))
}

/// Estimate conflict signals for each source against bound design data.
///
/// Uses a per-source power-prior compose (α as requested, no mixture) for PPC,
/// then Gaussian KL between the predictive mean/sd and a Dirac at the observed
/// outcome mean (`var = predictive_sd²` floored).
///
/// # Errors
///
/// PPC failures.
pub fn estimate_conflict_signals(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
    problem: &PreparedBayesianProblem,
    ctx: &ExecutionContext,
    ppc: &PriorPredictiveCheck,
) -> Result<Vec<ConflictSignals>, ValidationError> {
    let mut out = Vec::with_capacity(sources.len());
    for i in 0..sources.len() {
        // Isolate source i at its requested α (others α=0) for a targeted signal.
        let alphas: Vec<f64> = sources
            .iter()
            .enumerate()
            .map(|(j, s)| if j == i { s.weight.alpha } else { 0.0 })
            .collect();
        let requested: Vec<f64> = sources.iter().map(|s| s.weight.alpha).collect();
        let provisional =
            compose_external_priors_with_alphas(sources, &requested, &alphas, baseline)?;
        let rep = ppc.check_with_prior(problem, &provisional.prior, ctx)?;
        let pred_var = (rep.predictive_sd * rep.predictive_sd).max(1e-12);
        // KL(N(obs, pred_var) ‖ N(pred_mean, pred_var)) = 0.5 (Δμ)² / var
        // Equivalent: distance of observed from predictive under predictive variance.
        let kl = gaussian_kl(rep.observed, pred_var, rep.predictive_mean, pred_var).ok();
        out.push(ConflictSignals { p_value: Some(rep.p_value), kl });
    }
    Ok(out)
}

/// Full conflict path: estimate signals from data, shrink α, recompose.
///
/// # Errors
///
/// PPC or composition failures.
pub fn compose_with_conflict_policy(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
    policy: &ConflictPolicy,
    problem: &PreparedBayesianProblem,
    ctx: &ExecutionContext,
    ppc: &PriorPredictiveCheck,
) -> Result<(ComposedPrior, ConflictSummary), ValidationError> {
    let signals = estimate_conflict_signals(sources, baseline, problem, ctx, ppc)?;
    apply_conflict_and_compose(sources, baseline, policy, &signals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_prob::{ExternalPriorWeight, GaussianCoefficientPrior, PriorSpec};

    fn gauss(mean: f64, var: f64) -> PriorSet {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(
            GaussianCoefficientPrior::shared(1, mean, var).unwrap(),
        ));
        p
    }

    #[test]
    fn shrink_never_increases_alpha() {
        let pol = ConflictPolicy::new();
        assert!((pol.shrink_alpha(0.8, Some(0.5), Some(0.0)) - 0.8).abs() < 1e-12);
        assert!(pol.shrink_alpha(0.8, Some(0.01), None) < 0.8);
        assert!((pol.shrink_alpha(0.8, Some(0.01), None) - 0.0).abs() < 1e-12);
        let shrunk = pol.shrink_alpha(0.8, Some(0.5), Some(1.0));
        assert!(shrunk < 0.8);
        assert!(shrunk > 0.0);
    }

    #[test]
    fn no_conflict_leaves_alpha() {
        let pol = ConflictPolicy::new();
        let a = pol.shrink_alpha(0.7, Some(0.4), Some(0.0));
        assert!((a - 0.7).abs() < 1e-12);
    }

    #[test]
    fn apply_conflict_shrinks_and_records() {
        let baseline = gauss(0.0, 4.0);
        let sources = [ExternalPriorSource {
            id: Arc::from("far"),
            prior: gauss(50.0, 0.25),
            weight: ExternalPriorWeight::power(1.0).unwrap(),
        }];
        let signals = [ConflictSignals { p_value: Some(0.001), kl: Some(2.0) }];
        let (composed, summary) =
            apply_conflict_and_compose(&sources, &baseline, &ConflictPolicy::new(), &signals)
                .unwrap();
        assert!(summary.alphas_applied[0] < summary.alphas_requested[0]);
        assert!((summary.alphas_applied[0] - composed.alphas_applied[0]).abs() < 1e-12);
    }
}
