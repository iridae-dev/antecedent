//! Transport policy for cross-population prior transfer.
//!
//! Population identity is a caller convention (`tags["population"]`). When
//! source and target populations differ, an explicit [`TransportPolicy`] is
//! required; the library never invents identification from transport claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::PriorAssumption;

use crate::error::ProbError;
use crate::external_prior::{
    ComposedPrior, ExternalPriorSource, compose_external_priors_with_alphas,
};
use crate::prior::{GaussianCoefficientPrior, PriorSet, PriorSpec};

/// Stable assumption id recorded for each applied transport claim.
pub const TRANSPORT_ASSUMPTION_ID: &str = "external_transport_prior";

/// Caller-convention tag key for population identity on prior-source meta.
pub const POPULATION_TAG_KEY: &str = "population";

/// Floor on reweighted variance.
const REWEIGHT_VAR_FLOOR: f64 = 1e-12;

/// Explicit invariance claim for cross-population prior transfer.
///
/// Never inferred silently — callers must declare which mechanism is assumed
/// stable across populations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TransportPolicy {
    /// `P(Y | do(T), X)` is invariant across populations.
    InvariantConditionalOutcome,
    /// Effect-modifier relationships are invariant across populations.
    InvariantEffectModifiers,
    /// Propensity mechanism is invariant; requires target-alignment weights
    /// (or α is forced to 0).
    InvariantPropensity,
}

impl TransportPolicy {
    /// Stable string id for assumptions / wire formats.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::InvariantConditionalOutcome => "invariant_conditional_outcome",
            Self::InvariantEffectModifiers => "invariant_effect_modifiers",
            Self::InvariantPropensity => "invariant_propensity",
        }
    }

    /// Parse from a wire / Python string.
    ///
    /// # Errors
    ///
    /// Unknown policy name.
    pub fn parse(s: &str) -> Result<Self, TransportError> {
        match s {
            "invariant_conditional_outcome" | "InvariantConditionalOutcome" => {
                Ok(Self::InvariantConditionalOutcome)
            }
            "invariant_effect_modifiers" | "InvariantEffectModifiers" => {
                Ok(Self::InvariantEffectModifiers)
            }
            "invariant_propensity" | "InvariantPropensity" => Ok(Self::InvariantPropensity),
            other => Err(TransportError::UnknownPolicy { name: Arc::from(other) }),
        }
    }
}

/// Optional unit-level reweight toward the target population.
///
/// `unit_effects` and `target_weights` must share length; weights are
/// non-negative and finite with positive total mass (same spirit as
/// `TargetPopulation::CustomDistribution`).
#[derive(Clone, Debug, PartialEq)]
pub struct TransportAdjustment {
    /// Unit-level scalar effect (or treatment-coef) contributions.
    pub unit_effects: Arc<[f64]>,
    /// Non-negative weights aligning those units to the target population.
    pub target_weights: Arc<[f64]>,
}

impl TransportAdjustment {
    /// Construct with validation.
    ///
    /// # Errors
    ///
    /// Length mismatch, empty, non-finite, negative weights, or zero mass.
    pub fn new(
        unit_effects: impl Into<Arc<[f64]>>,
        target_weights: impl Into<Arc<[f64]>>,
    ) -> Result<Self, TransportError> {
        let unit_effects = unit_effects.into();
        let target_weights = target_weights.into();
        if unit_effects.is_empty() || target_weights.is_empty() {
            return Err(TransportError::InvalidWeights {
                message: "transport adjustment requires non-empty effects and weights",
            });
        }
        if unit_effects.len() != target_weights.len() {
            return Err(TransportError::InvalidWeights {
                message: "unit_effects and target_weights length mismatch",
            });
        }
        let mut mass = 0.0;
        for (&e, &w) in unit_effects.iter().zip(target_weights.iter()) {
            if !e.is_finite() {
                return Err(TransportError::InvalidWeights {
                    message: "unit_effects must be finite",
                });
            }
            if !w.is_finite() || w < 0.0 {
                return Err(TransportError::InvalidWeights {
                    message: "target_weights must be finite and >= 0",
                });
            }
            mass += w;
        }
        if !(mass > 0.0) {
            return Err(TransportError::InvalidWeights {
                message: "target_weights must have positive total mass",
            });
        }
        Ok(Self { unit_effects, target_weights })
    }

    /// Importance-weighted mean and variance of `unit_effects`.
    #[must_use]
    pub fn weighted_moments(&self) -> (f64, f64) {
        let mass: f64 = self.target_weights.iter().sum();
        let mean = self
            .unit_effects
            .iter()
            .zip(self.target_weights.iter())
            .map(|(&e, &w)| w * e)
            .sum::<f64>()
            / mass;
        let var = self
            .unit_effects
            .iter()
            .zip(self.target_weights.iter())
            .map(|(&e, &w)| {
                let d = e - mean;
                w * d * d
            })
            .sum::<f64>()
            / mass;
        (mean, var.max(REWEIGHT_VAR_FLOOR))
    }

    /// Kish effective sample size for diagnostics.
    #[must_use]
    pub fn kish_ess(&self) -> f64 {
        let sum: f64 = self.target_weights.iter().sum();
        let sum_sq: f64 = self.target_weights.iter().map(|w| w * w).sum();
        if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 }
    }
}

/// Inputs for applying transport to one or more external sources.
#[derive(Clone, Debug)]
pub struct TransportContext<'a> {
    /// Per-source population tags (`None` = untaged / unknown).
    pub source_populations: &'a [Option<&'a str>],
    /// Target analysis population tag (`None` = untaged).
    pub target_population: Option<&'a str>,
    /// Declared invariance claim (required when populations differ).
    pub policy: Option<TransportPolicy>,
    /// Optional unit-level reweight (applied to every mismatched source).
    pub adjustment: Option<&'a TransportAdjustment>,
    /// Coefficient index to rewrite under reweight (default: last / treatment).
    pub coef_index: Option<usize>,
}

/// Per-source outcome of [`apply_transport`].
#[derive(Clone, Debug, PartialEq)]
pub struct TransportOutcome {
    /// Source artifact id.
    pub source_id: Arc<str>,
    /// Whether populations differed for this source.
    pub required: bool,
    /// Alpha override when transport forces trust to zero (`Some(0.0)`).
    pub alpha_override: Option<f64>,
    /// Human-readable reason when α was forced to zero.
    pub zero_reason: Option<Arc<str>>,
}

/// Structured transport errors (stable `code()` for conformance).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportError {
    /// Populations differ and no [`TransportPolicy`] was supplied.
    PolicyRequired {
        /// Source population tag (empty if absent).
        source_population: Arc<str>,
        /// Target population tag (empty if absent).
        target_population: Arc<str>,
    },
    /// Source / target population vector length mismatch.
    SourceCountMismatch {
        /// Number of sources.
        n_sources: usize,
        /// Number of population tags.
        n_populations: usize,
    },
    /// Invalid reweight weights / effects.
    InvalidWeights {
        /// Context.
        message: &'static str,
    },
    /// Unknown policy wire name.
    UnknownPolicy {
        /// Provided name.
        name: Arc<str>,
    },
    /// Coefficient index out of range for reweight.
    CoefIndexOutOfRange {
        /// Requested index.
        index: usize,
        /// Prior dimension.
        n_coef: usize,
    },
}

impl TransportError {
    /// Stable machine-readable code (pinned in conformance JSON).
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::PolicyRequired { .. } => "transport_policy_required",
            Self::SourceCountMismatch { .. } => "transport_source_count_mismatch",
            Self::InvalidWeights { .. } => "transport_invalid_weights",
            Self::UnknownPolicy { .. } => "transport_unknown_policy",
            Self::CoefIndexOutOfRange { .. } => "transport_coef_index_out_of_range",
        }
    }
}

impl core::fmt::Display for TransportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PolicyRequired { source_population, target_population } => {
                write!(
                    f,
                    "{}: population mismatch source={source_population:?} target={target_population:?} requires TransportPolicy",
                    self.code()
                )
            }
            Self::SourceCountMismatch { n_sources, n_populations } => {
                write!(
                    f,
                    "{}: source_populations len {n_populations} != sources len {n_sources}",
                    self.code()
                )
            }
            Self::InvalidWeights { message } => {
                write!(f, "{}: {message}", self.code())
            }
            Self::UnknownPolicy { name } => {
                write!(f, "{}: unknown TransportPolicy `{name}`", self.code())
            }
            Self::CoefIndexOutOfRange { index, n_coef } => {
                write!(f, "{}: coef_index {index} out of range for n_coef={n_coef}", self.code())
            }
        }
    }
}

impl std::error::Error for TransportError {}

impl From<TransportError> for ProbError {
    fn from(e: TransportError) -> Self {
        ProbError::Numerical { message: e.to_string() }
    }
}

/// Whether any source population differs from the target (both sides tagged).
///
/// Untagged (`None`) on either side alone does not require transport. Differing
/// concrete tags do. A concrete target with an untagged source (or vice versa)
/// also requires transport — the library will not silently assume equality.
#[must_use]
pub fn populations_require_transport(
    source_population: Option<&str>,
    target_population: Option<&str>,
) -> bool {
    match (source_population, target_population) {
        (None, None) => false,
        (Some(a), Some(b)) => a != b,
        // One side tagged, the other not: treat as a shift that needs a claim.
        (Some(_), None) | (None, Some(_)) => true,
    }
}

fn pop_label(p: Option<&str>) -> Arc<str> {
    Arc::from(p.unwrap_or(""))
}

fn transport_assumption(
    policy: TransportPolicy,
    source_id: &str,
    source_pop: Option<&str>,
    target_pop: Option<&str>,
    extra: &str,
) -> PriorAssumption {
    PriorAssumption {
        id: Arc::from(TRANSPORT_ASSUMPTION_ID),
        description: Arc::from(format!(
            "TransportPolicy {} for source={source_id} source_pop={} target_pop={}{extra}",
            policy.id(),
            source_pop.unwrap_or(""),
            target_pop.unwrap_or(""),
        )),
    }
}

fn replace_coef_moments(
    prior: &mut PriorSet,
    coef_index: usize,
    mean: f64,
    variance: f64,
) -> Result<(), TransportError> {
    let Some(coef) = prior.gaussian_coefficients() else {
        return Err(TransportError::InvalidWeights {
            message: "source prior missing GaussianCoefficients for transport reweight",
        });
    };
    if coef_index >= coef.len() {
        return Err(TransportError::CoefIndexOutOfRange { index: coef_index, n_coef: coef.len() });
    }
    let mut mean_v = coef.mean.to_vec();
    let mut var_v = coef.variance.to_vec();
    mean_v[coef_index] = mean;
    var_v[coef_index] = variance;
    let new_coef = GaussianCoefficientPrior { mean: Arc::from(mean_v), variance: Arc::from(var_v) };
    new_coef.validate().map_err(|_| TransportError::InvalidWeights {
        message: "reweighted coefficient moments invalid",
    })?;
    // Replace GaussianCoefficients entry; keep other specs / restrictions.
    let mut specs = Vec::with_capacity(prior.specs.len());
    let mut replaced = false;
    for s in &prior.specs {
        match s {
            PriorSpec::GaussianCoefficients(_) if !replaced => {
                specs.push(PriorSpec::GaussianCoefficients(new_coef.clone()));
                replaced = true;
            }
            other => specs.push(other.clone()),
        }
    }
    if !replaced {
        specs.push(PriorSpec::GaussianCoefficients(new_coef));
    }
    prior.specs = specs;
    Ok(())
}

/// Apply transport gate / adjustment to cloned sources; return prepared sources
/// and per-source outcomes.
///
/// # Errors
///
/// [`TransportError::PolicyRequired`] when populations differ without a policy,
/// length mismatches, or invalid adjustment weights.
pub fn apply_transport(
    sources: &[ExternalPriorSource],
    ctx: &TransportContext<'_>,
) -> Result<(Vec<ExternalPriorSource>, Vec<TransportOutcome>), TransportError> {
    if ctx.source_populations.len() != sources.len() {
        return Err(TransportError::SourceCountMismatch {
            n_sources: sources.len(),
            n_populations: ctx.source_populations.len(),
        });
    }

    let any_required = sources
        .iter()
        .zip(ctx.source_populations.iter())
        .any(|(_, &sp)| populations_require_transport(sp, ctx.target_population));
    if any_required && ctx.policy.is_none() {
        // Report the first mismatched pair for a stable error payload.
        for &sp in ctx.source_populations {
            if populations_require_transport(sp, ctx.target_population) {
                return Err(TransportError::PolicyRequired {
                    source_population: pop_label(sp),
                    target_population: pop_label(ctx.target_population),
                });
            }
        }
    }

    let policy = ctx.policy;
    let mut out_sources = Vec::with_capacity(sources.len());
    let mut outcomes = Vec::with_capacity(sources.len());

    for (src, &sp) in sources.iter().zip(ctx.source_populations.iter()) {
        let required = populations_require_transport(sp, ctx.target_population);
        let mut prepared = src.clone();
        let mut alpha_override = None;
        let mut zero_reason = None;

        if required {
            let policy = policy.expect("gated above");
            let mut extra = String::new();

            match (policy, ctx.adjustment) {
                (TransportPolicy::InvariantPropensity, None) => {
                    alpha_override = Some(0.0);
                    zero_reason = Some(Arc::from(
                        "invariant_propensity requires target_weights; alpha forced to 0",
                    ));
                    extra.push_str("; alpha_forced=0 reason=missing_propensity_weights");
                }
                (_, Some(adj)) => {
                    let (mean, var) = adj.weighted_moments();
                    let n_coef = prepared
                        .prior
                        .gaussian_coefficients()
                        .ok_or(TransportError::InvalidWeights {
                            message: "source prior missing GaussianCoefficients for transport reweight",
                        })?
                        .len();
                    let idx = ctx.coef_index.unwrap_or(n_coef.saturating_sub(1));
                    replace_coef_moments(&mut prepared.prior, idx, mean, var)?;
                    extra.push_str(&format!(
                        "; reweighted mean={mean:.6} var={var:.6} ess={:.3}",
                        adj.kish_ess()
                    ));
                }
                (
                    TransportPolicy::InvariantConditionalOutcome
                    | TransportPolicy::InvariantEffectModifiers,
                    None,
                ) => {
                    // Claim-only transfer under the invariance declaration.
                }
            }

            prepared.prior.restrictions.push(transport_assumption(
                policy,
                src.id.as_ref(),
                sp,
                ctx.target_population,
                &extra,
            ));

            if let Some(a) = alpha_override {
                prepared.weight.alpha = a;
            }
        }

        outcomes.push(TransportOutcome {
            source_id: Arc::clone(&src.id),
            required,
            alpha_override,
            zero_reason,
        });
        out_sources.push(prepared);
    }

    Ok((out_sources, outcomes))
}

/// Apply transport then compose (power / mixture path).
///
/// When transport forces α overrides, those override the source weights for
/// composition while preserving the original requested alphas.
///
/// # Errors
///
/// Transport gate / adjustment failures or composition failures.
pub fn compose_with_transport(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
    ctx: &TransportContext<'_>,
) -> Result<(ComposedPrior, Vec<TransportOutcome>), ProbError> {
    let (prepared, outcomes) = apply_transport(sources, ctx)?;
    let requested: Vec<f64> = sources.iter().map(|s| s.weight.alpha).collect();
    let applied: Vec<f64> = prepared
        .iter()
        .zip(outcomes.iter())
        .map(|(s, o)| o.alpha_override.unwrap_or(s.weight.alpha))
        .collect();
    // Ensure prepared weights match applied for mixture path consistency.
    let prepared: Vec<ExternalPriorSource> = prepared
        .into_iter()
        .zip(applied.iter())
        .map(|(mut s, &a)| {
            s.weight.alpha = a;
            s
        })
        .collect();
    let composed = compose_external_priors_with_alphas(&prepared, &requested, &applied, baseline)?;
    Ok((composed, outcomes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::external_prior::ExternalPriorWeight;

    fn gauss(mean: f64, var: f64) -> PriorSet {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(
            GaussianCoefficientPrior::shared(1, mean, var).unwrap(),
        ));
        p
    }

    fn source(id: &str, mean: f64, alpha: f64) -> ExternalPriorSource {
        ExternalPriorSource {
            id: Arc::from(id),
            prior: gauss(mean, 1.0),
            weight: ExternalPriorWeight::power(alpha).unwrap(),
        }
    }

    #[test]
    fn same_population_skips_transport() {
        let sources = [source("a", 1.0, 0.8)];
        let ctx = TransportContext {
            source_populations: &[Some("us")],
            target_population: Some("us"),
            policy: None,
            adjustment: None,
            coef_index: None,
        };
        let (out, outcomes) = apply_transport(&sources, &ctx).unwrap();
        assert!(!outcomes[0].required);
        assert!(out[0].prior.restrictions.is_empty());
        assert!((out[0].weight.alpha - 0.8).abs() < 1e-12);
    }

    #[test]
    fn mismatch_without_policy_errors() {
        let sources = [source("a", 1.0, 1.0)];
        let ctx = TransportContext {
            source_populations: &[Some("us")],
            target_population: Some("eu"),
            policy: None,
            adjustment: None,
            coef_index: None,
        };
        let err = apply_transport(&sources, &ctx).unwrap_err();
        assert_eq!(err.code(), "transport_policy_required");
    }

    #[test]
    fn claim_only_records_assumption() {
        let sources = [source("a", 2.0, 1.0)];
        let baseline = gauss(0.0, 4.0);
        let ctx = TransportContext {
            source_populations: &[Some("us")],
            target_population: Some("eu"),
            policy: Some(TransportPolicy::InvariantConditionalOutcome),
            adjustment: None,
            coef_index: None,
        };
        let (composed, outcomes) = compose_with_transport(&sources, &baseline, &ctx).unwrap();
        assert!(outcomes[0].required);
        assert!(outcomes[0].alpha_override.is_none());
        assert!(
            composed.prior.restrictions.iter().any(|r| r.id.as_ref() == TRANSPORT_ASSUMPTION_ID)
        );
        assert!((composed.alphas_applied[0] - 1.0).abs() < 1e-12);
        let coef = composed.prior.gaussian_coefficients().unwrap();
        assert!(coef.mean[0].is_finite());
        assert!(coef.variance[0].is_finite() && coef.variance[0] > 0.0);
    }

    #[test]
    fn propensity_without_weights_forces_alpha_zero() {
        let sources = [source("a", 2.0, 0.9)];
        let baseline = gauss(0.0, 4.0);
        let ctx = TransportContext {
            source_populations: &[Some("us")],
            target_population: Some("eu"),
            policy: Some(TransportPolicy::InvariantPropensity),
            adjustment: None,
            coef_index: None,
        };
        let (composed, outcomes) = compose_with_transport(&sources, &baseline, &ctx).unwrap();
        assert_eq!(outcomes[0].alpha_override, Some(0.0));
        assert!((composed.alphas_requested[0] - 0.9).abs() < 1e-12);
        assert!((composed.alphas_applied[0] - 0.0).abs() < 1e-12);
        assert!(
            composed.prior.restrictions.iter().any(|r| r.id.as_ref() == TRANSPORT_ASSUMPTION_ID)
        );
    }

    #[test]
    fn weighted_moments_shift_mean() {
        // Units: effects 0, 0, 10 with weights concentrating on the last.
        let adj = TransportAdjustment::new([0.0, 0.0, 10.0], [0.0, 0.0, 1.0]).unwrap();
        let (mean, var) = adj.weighted_moments();
        assert!((mean - 10.0).abs() < 1e-12);
        assert!(var >= REWEIGHT_VAR_FLOOR);

        let sources = [source("a", 0.0, 1.0)];
        let baseline = gauss(0.0, 100.0);
        let ctx = TransportContext {
            source_populations: &[Some("us")],
            target_population: Some("eu"),
            policy: Some(TransportPolicy::InvariantConditionalOutcome),
            adjustment: Some(&adj),
            coef_index: Some(0),
        };
        let (composed, _) = compose_with_transport(&sources, &baseline, &ctx).unwrap();
        let coef = composed.prior.gaussian_coefficients().unwrap();
        // Power-add with α=1: prior mean pulled toward 10 from reweighted source.
        assert!(coef.mean[0] > 5.0, "mean {}", coef.mean[0]);
    }

    #[test]
    fn rejects_invalid_weights() {
        assert!(TransportAdjustment::new([1.0], [-0.1]).is_err());
        assert!(TransportAdjustment::new([1.0, 2.0], [1.0]).is_err());
        assert!(TransportAdjustment::new([1.0], [0.0]).is_err());
    }

    #[test]
    fn untagged_both_sides_ok_without_policy() {
        let sources = [source("a", 1.0, 1.0)];
        let ctx = TransportContext {
            source_populations: &[None],
            target_population: None,
            policy: None,
            adjustment: None,
            coef_index: None,
        };
        assert!(apply_transport(&sources, &ctx).is_ok());
    }
}
