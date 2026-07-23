//! Bayesian inference configuration for the facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_estimate::BayesianBackendKind;
use causal_estimate::{HydrateMapping, PreparedBayesianProblem, hydrate_prior};
use causal_io::PosteriorQuantityWire;
use causal_io::PriorMapping;
use causal_io::{decode_posterior_artifact, extract_prior_source_meta, read_and_migrate};
use causal_prob::{
    BayesLikelihood, ComposedPrior, ConflictSummary, ExternalPriorSource, PosteriorQuantityKind,
    PriorSet,
};
use causal_validate::{ConflictPolicy, PriorPredictiveCheck, compose_with_conflict_policy};

use crate::decode_causal_posterior_bytes;
use crate::error::CausalError;

/// Frequentist vs Bayesian inference mode.
#[derive(Clone, Debug, PartialEq)]
pub enum InferenceMode {
    /// Classical point-estimate path (default).
    Frequentist,
    /// Bayesian g-computation / posterior path.
    Bayesian(BayesianConfig),
}

impl Default for InferenceMode {
    fn default() -> Self {
        Self::Frequentist
    }
}

/// External prior bank composition retained for optional conflict re-shrink.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalComposeSpec {
    /// Hydrated sources (same order as in [`ComposedPrior`]).
    pub sources: Arc<[ExternalPriorSource]>,
    /// Precomposed prior (used when conflict policy is absent).
    pub composed: ComposedPrior,
    /// When set, α is re-shrunk after design bind using prior-PPC / KL.
    pub conflict_policy: Option<ConflictPolicy>,
}

/// Bayesian analysis configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianConfig {
    /// Backend kind.
    pub backend: BayesianBackendKind,
    /// Likelihood (Laplace path).
    pub likelihood: BayesLikelihood,
    /// Posterior draws.
    pub n_draws: usize,
    /// Isotropic prior scale (used when [`Self::prior`] and artifact are unset).
    pub prior_scale: f64,
    /// Explicit coefficient prior (e.g. hydrated from a previous posterior artifact).
    /// When set, overrides isotropic [`Self::prior_scale`] and [`Self::prior_artifact`].
    pub prior: Option<PriorSet>,
    /// Posterior artifact bytes for deferred mapped hydrate (after design prepare).
    pub prior_artifact: Option<Arc<[u8]>>,
    /// Mapping for [`Self::prior_artifact`].
    ///
    /// When unset, hydrate picks identical coefficient subspace for matching
    /// designs, or [`PriorMapping::EffectFunctional`] when designs differ and
    /// an effect quantity is present (never silent `coef_i → coef_i` across
    /// heterogeneous layouts).
    pub prior_mapping: Option<PriorMapping>,
    /// External power-prior / mixture compose (optional conflict re-eval).
    pub external_compose: Option<Box<ExternalComposeSpec>>,
}

impl BayesianConfig {
    /// Laplace Gaussian defaults.
    #[must_use]
    pub fn laplace() -> Self {
        Self {
            backend: BayesianBackendKind::Laplace,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            prior_scale: 10.0,
            prior: None,
            prior_artifact: None,
            prior_mapping: None,
            external_compose: None,
        }
    }

    /// Conjugate Gaussian defaults.
    #[must_use]
    pub fn conjugate() -> Self {
        Self {
            backend: BayesianBackendKind::ConjugateGaussian,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            prior_scale: 10.0,
            prior: None,
            prior_artifact: None,
            prior_mapping: None,
            external_compose: None,
        }
    }

    /// Native HMC defaults.
    #[must_use]
    pub fn hmc() -> Self {
        Self {
            backend: BayesianBackendKind::Hmc,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 200,
            prior_scale: 10.0,
            prior: None,
            prior_artifact: None,
            prior_mapping: None,
            external_compose: None,
        }
    }

    /// Weakly informative prior scale.
    #[must_use]
    pub fn prior_scale(mut self, scale: f64) -> Self {
        self.prior_scale = scale;
        self
    }

    /// Draw count.
    #[must_use]
    pub fn n_draws(mut self, n: usize) -> Self {
        self.n_draws = n;
        self
    }

    /// Explicit prior set (sequential Bayes / custom coefficients).
    #[must_use]
    pub fn prior(mut self, prior: PriorSet) -> Self {
        self.prior = Some(prior);
        self.external_compose = None;
        self
    }

    /// Posterior artifact + optional mapping (hydrate deferred until design is known).
    #[must_use]
    pub fn prior_from_artifact(
        mut self,
        bytes: impl Into<Arc<[u8]>>,
        mapping: Option<PriorMapping>,
    ) -> Self {
        self.prior_artifact = Some(bytes.into());
        self.prior_mapping = mapping;
        self.prior = None;
        self.external_compose = None;
        self
    }

    /// Use a composed external prior bank prior (optionally with conflict re-shrink).
    ///
    /// `sources` must be the same inputs used to build `composed` (for conflict
    /// re-evaluation after data bind). When `conflict` is `None`, `composed.prior`
    /// is used as-is.
    #[must_use]
    pub fn prior_from_composed(
        mut self,
        sources: impl Into<Arc<[ExternalPriorSource]>>,
        composed: ComposedPrior,
        conflict: Option<ConflictPolicy>,
    ) -> Self {
        let sources = sources.into();
        self.prior = Some(composed.prior.clone());
        self.prior_artifact = None;
        self.prior_mapping = None;
        self.external_compose =
            Some(Box::new(ExternalComposeSpec { sources, composed, conflict_policy: conflict }));
        self
    }
}

/// Convert IO prior mapping into the estimate hydrate enum.
#[must_use]
pub fn hydrate_mapping_from_io(mapping: &PriorMapping) -> HydrateMapping {
    match mapping {
        PriorMapping::IdenticalCoefficientSubspace => HydrateMapping::IdenticalCoefficientSubspace,
        PriorMapping::EffectFunctional { source_quantity } => {
            HydrateMapping::EffectFunctional { source_quantity: source_quantity.clone() }
        }
        PriorMapping::NamedParameters { pairs } => {
            HydrateMapping::NamedParameters { pairs: pairs.clone() }
        }
    }
}

fn wire_quantities_to_kinds(wire: &[PosteriorQuantityWire]) -> Vec<PosteriorQuantityKind> {
    wire.iter()
        .map(|q| match q {
            PosteriorQuantityWire::Coefficient { index, name } => {
                PosteriorQuantityKind::Coefficient {
                    index: *index as usize,
                    name: name.as_ref().map(|s| Arc::<str>::from(s.as_str())),
                }
            }
            PosteriorQuantityWire::ResidualVariance => PosteriorQuantityKind::ResidualVariance,
            PosteriorQuantityWire::Effect { name } => {
                PosteriorQuantityKind::Effect { name: Arc::from(name.as_str()) }
            }
            PosteriorQuantityWire::Scalar { name } => {
                PosteriorQuantityKind::Scalar { name: Arc::from(name.as_str()) }
            }
        })
        .collect()
}

fn coef_names_for_problem(prep: &PreparedBayesianProblem) -> Arc<[Arc<str>]> {
    if let Some(names) = &prep.coef_names {
        return Arc::clone(names);
    }
    let names: Vec<Arc<str>> =
        (0..prep.design.ncols).map(|i| Arc::<str>::from(format!("coef_{i}"))).collect();
    Arc::from(names)
}

/// True when source coefficients can hydrate by identical subspace into `prep`.
fn designs_compatible(
    source_coef_names: &[Option<&str>],
    target_ncols: usize,
    target_names: &[Arc<str>],
) -> bool {
    if source_coef_names.len() != target_ncols {
        return false;
    }
    // Unnamed source coefficients → index-aligned sequential Bayes.
    if source_coef_names.iter().all(Option::is_none) {
        return true;
    }
    if source_coef_names.len() != target_names.len() {
        return false;
    }
    source_coef_names
        .iter()
        .zip(target_names.iter())
        .all(|(src, tgt)| src.is_some_and(|name| name == tgt.as_ref()))
}

/// Choose hydrate mapping when the caller left `prior_mapping` unset.
///
/// Same-design sequential Bayes keeps identical subspace. Heterogeneous designs
/// default to effect-functional transfer when an `Effect` quantity exists;
/// otherwise the caller must supply an explicit mapping.
fn default_hydrate_mapping(
    bytes: &[u8],
    prep: &PreparedBayesianProblem,
) -> Result<HydrateMapping, CausalError> {
    let artifact = read_and_migrate(bytes)?;
    if let Some(meta) = extract_prior_source_meta(&artifact)? {
        if let Some(mapping) = &meta.declared_mapping {
            return Ok(hydrate_mapping_from_io(mapping));
        }
    }
    let (wire, _) = decode_posterior_artifact(&artifact)?;
    let source_coef_names: Vec<Option<&str>> = wire
        .quantities
        .iter()
        .filter_map(|q| match q {
            PosteriorQuantityWire::Coefficient { name, .. } => Some(name.as_deref()),
            _ => None,
        })
        .collect();
    let target_names = coef_names_for_problem(prep);
    if designs_compatible(&source_coef_names, prep.design.ncols, &target_names) {
        return Ok(HydrateMapping::IdenticalCoefficientSubspace);
    }
    let source_quantity = wire.quantities.iter().find_map(|q| match q {
        PosteriorQuantityWire::Effect { name } => Some(name.clone()),
        _ => None,
    });
    match source_quantity {
        Some(source_quantity) => Ok(HydrateMapping::EffectFunctional { source_quantity }),
        None => Err(CausalError::Compile {
            message: "prior artifact mapping required: designs differ and no Effect \
                      quantity is available for EffectFunctional default"
                .into(),
        }),
    }
}

/// Resolve the coefficient prior for a prepared Bayesian problem.
///
/// Precedence: [`BayesianConfig::external_compose`] (with optional conflict) →
/// explicit [`BayesianConfig::prior`] → mapped [`BayesianConfig::prior_artifact`]
/// → `None` (isotropic `prior_scale` at fit).
///
/// When a conflict policy is set, α is re-shrunk using prior-PPC / KL against
/// the bound design; the returned [`ConflictSummary`] should be attached to the
/// fitted posterior.
///
/// # Errors
///
/// Decode / hydrate / conflict composition failures.
pub fn resolve_bayesian_prior(
    cfg: &BayesianConfig,
    prep: &PreparedBayesianProblem,
) -> Result<Option<PriorSet>, CausalError> {
    let (prior, _) = resolve_bayesian_prior_with_conflict(cfg, prep, None)?;
    Ok(prior)
}

/// Like [`resolve_bayesian_prior`], but runs conflict shrink when `ctx` is
/// provided and a [`ConflictPolicy`] is configured on the external compose.
///
/// # Errors
///
/// Decode / hydrate / conflict composition failures.
pub fn resolve_bayesian_prior_with_conflict(
    cfg: &BayesianConfig,
    prep: &PreparedBayesianProblem,
    ctx: Option<&ExecutionContext>,
) -> Result<(Option<PriorSet>, Option<ConflictSummary>), CausalError> {
    if let Some(ext) = &cfg.external_compose {
        if let (Some(policy), Some(ctx)) = (&ext.conflict_policy, ctx) {
            let baseline = PriorSet::weakly_informative(prep.design.ncols);
            let ppc = PriorPredictiveCheck {
                n_sims: 200,
                seed: ctx.rng.master_seed(),
                ..PriorPredictiveCheck::new()
            };
            let (composed, summary) =
                compose_with_conflict_policy(&ext.sources, &baseline, policy, prep, ctx, &ppc)
                    .map_err(CausalError::from)?;
            return Ok((Some(composed.prior), Some(summary)));
        }
        return Ok((Some(ext.composed.prior.clone()), None));
    }
    if let Some(p) = &cfg.prior {
        return Ok((Some(p.clone()), None));
    }
    let Some(bytes) = cfg.prior_artifact.as_ref() else {
        return Ok((None, None));
    };
    let mapping = match cfg.prior_mapping.as_ref() {
        Some(m) => hydrate_mapping_from_io(m),
        None => default_hydrate_mapping(bytes, prep)?,
    };
    let names = coef_names_for_problem(prep);
    let baseline = PriorSet::weakly_informative(prep.design.ncols);
    let treatment_col = prep.design.treatment_column();
    Ok((
        Some(hydrate_prior_from_posterior_bytes(
            bytes,
            &mapping,
            &baseline,
            &names,
            treatment_col,
        )?),
        None,
    ))
}

/// Hydrate a [`PriorSet`] from posterior artifact bytes under a [`HydrateMapping`].
///
/// # Errors
///
/// Decode failures or hydrate validation errors.
pub fn hydrate_prior_from_posterior_bytes(
    bytes: &[u8],
    mapping: &HydrateMapping,
    baseline: &PriorSet,
    target_coef_names: &[Arc<str>],
    treatment_col: Option<usize>,
) -> Result<PriorSet, CausalError> {
    let (wire, _) = decode_causal_posterior_bytes(bytes)?;
    let quantities = wire_quantities_to_kinds(&wire.quantities);
    hydrate_prior(
        mapping,
        &quantities,
        &wire.mean,
        &wire.sd,
        baseline,
        target_coef_names,
        treatment_col,
    )
    .map_err(CausalError::from)
}
