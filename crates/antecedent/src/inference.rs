//! Inference mode and Bayesian configuration of a study.
//!
//! [`InferenceMode`] chooses frequentist or Bayesian inference; [`BayesianConfig`] declares the
//! backend, likelihood, priors and posterior draws of a Bayesian study, including priors taken
//! from earlier posteriors. The module also re-exports the prior, likelihood and test types
//! these declarations use.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::BayesianBackendKind;
use antecedent_estimate::{
    HydrateMapping, HydrateSource, HydrateTarget, PreparedBayesianProblem,
    coefficient_covariance_from_draws, hydrate_prior, is_temporal_coefficient_name,
};
use antecedent_io::PosteriorQuantityWire;
use antecedent_io::PriorMapping;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionIdentity, DrawAlignment,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_io::{
    CausalPosteriorWire, decode_posterior_artifact, extract_prior_source_meta, read_and_migrate,
};
use antecedent_prob::PosteriorQuantityKind;
use antecedent_validate::{ConflictPolicy, PriorPredictiveCheck, compose_with_conflict_policy};

use crate::error::CausalError;
use crate::io::decode_causal_posterior_bytes;

/// Frequentist vs Bayesian inference mode.
#[allow(
    clippy::large_enum_variant,
    reason = "the Bayesian configuration is declared once per study and moved by value; boxing would change every public pattern match"
)]
#[derive(Clone, Debug, PartialEq, Default)]
pub enum InferenceMode {
    /// Classical point-estimate path (default).
    #[default]
    Frequentist,
    /// Bayesian g-computation / posterior path.
    Bayesian(BayesianConfig),
}

impl InferenceMode {
    /// Whether this is the Bayesian mode.
    #[must_use]
    pub(crate) const fn is_bayesian(&self) -> bool {
        matches!(self, Self::Bayesian(_))
    }

    /// Whether this is the frequentist mode.
    #[must_use]
    pub(crate) const fn is_frequentist(&self) -> bool {
        matches!(self, Self::Frequentist)
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
    /// Original baseline used to compose and re-shrink this prior.
    pub baseline: Option<PriorSet>,
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
    /// Whether the draw count was chosen by the caller (the `n_draws(..)` setter) rather
    /// than left at the backend default. A latency tier resizes only a default count.
    pub n_draws_explicit: bool,
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
    /// Declared premise that the outcome innovation of a single-step temporal linear effect is
    /// serially independent, with the Ljung-Box gate that can refute it. Set with
    /// [`Self::white_innovation`]. Only the temporal pulse, the one-step sustained effect and a
    /// response curve at the single horizon of one step ahead, on a fixed `TemporalDag`, read it;
    /// every other route refuses it rather than ignore it.
    pub white_innovation:
        Option<antecedent_estimate::white_innovation_posterior::InnovationWhitenessGate>,
    /// Declared likelihood-side model of a randomized-design posterior (the hazard partition
    /// and prior of the survival posterior, the dose curve and prior of the dose policy
    /// posterior). Set with [`Self::design_model`]; only those routes read it.
    pub design_model: Option<crate::design_posterior_model::DesignPosteriorModel>,
    /// Multiplier on every Gaussian coefficient variance of the prior in force, applied after
    /// the prior is resolved against the bound design: an explicit prior, a prior hydrated from
    /// a posterior artifact (which exists only once the design is known) or an external
    /// composition. Prior-sensitivity refits set it; it does nothing when the isotropic
    /// `prior_scale` is in force (vary that instead). It is not part of the study identity.
    pub prior_variance_multiplier: Option<f64>,
    /// Declared bounds of the posterior predictive replicates the study retains for
    /// predictive assessment. Set with [`crate::StudyBuilder::bayesian_predictive_replicates`];
    /// only a tabular `AverageEffect` on a fixed `Dag` under Bayesian g-computation reads it.
    pub(crate) predictive_replicates: Option<PredictiveReplicateBounds>,
}

/// Declared bounds of the posterior predictive replicates a Bayesian study retains.
///
/// The replicates are `n_sims` draws over every design row, so they are retained only when
/// asked for, only for a design of at most `max_rows` rows, and only after the Execution
/// memory pool admits them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PredictiveReplicateBounds {
    /// Most posterior draws to replicate (capped by the draws the posterior holds).
    pub(crate) n_sims: u32,
    /// Most design rows a retained replicate matrix may cover.
    pub(crate) max_rows: u32,
}

/// Numerical controls on the posterior draws behind a published interval.
///
/// Requested with [`crate::StudyBuilder::draw_controls`]. The draw count must reach the
/// supported minimum of the backend (2000 draws for the exact conjugate and Laplace
/// samplers, 3000 for HMC, the floor its chain diagnostics need), and a smaller count is
/// refused at build, naming the minimum, never rewritten. The interval is then refused
/// when the Monte Carlo standard error of its larger endpoint is more than a twentieth of
/// its width, where an MCMC sampler's draws are discounted by their tail effective sample
/// size, so a sampler that mixes poorly refuses naming that error; the error is reported
/// beside the interval. The controls apply to a tabular average or conditional effect on one
/// `Dag` estimated by Bayesian g-computation; any other route refuses them rather than
/// ignore them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PosteriorDrawControls {
    /// The supported minimum draw count and the endpoint Monte Carlo error bound.
    Measured,
}

impl PosteriorDrawControls {
    /// Stable name carried in the inference identity.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
        }
    }
}

impl BayesianConfig {
    /// Whether the config carries any coefficient prior beyond the isotropic
    /// scale: an explicit prior, a mapped posterior artifact, or an external
    /// compose.
    #[must_use]
    pub(crate) const fn has_transfer(&self) -> bool {
        self.prior.is_some() || self.has_transferred_source()
    }

    /// Whether the config carries a transferred prior source (posterior
    /// artifact or external compose) rather than an explicit prior.
    #[must_use]
    pub(crate) const fn has_transferred_source(&self) -> bool {
        self.prior_artifact.is_some() || self.external_compose.is_some()
    }

    /// Laplace Gaussian defaults.
    #[must_use]
    pub fn laplace() -> Self {
        Self {
            backend: BayesianBackendKind::Laplace,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            n_draws_explicit: false,
            prior_scale: 10.0,
            prior: None,
            prior_artifact: None,
            prior_mapping: None,
            external_compose: None,
            white_innovation: None,
            design_model: None,
            prior_variance_multiplier: None,
            predictive_replicates: None,
        }
    }

    /// Conjugate Gaussian defaults.
    #[must_use]
    pub fn conjugate() -> Self {
        Self {
            backend: BayesianBackendKind::ConjugateGaussian,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            n_draws_explicit: false,
            prior_scale: 10.0,
            prior: None,
            prior_artifact: None,
            prior_mapping: None,
            external_compose: None,
            white_innovation: None,
            design_model: None,
            prior_variance_multiplier: None,
            predictive_replicates: None,
        }
    }

    /// Native HMC defaults.
    ///
    /// Draw count is sized for the MCMC publication gate (Ř ≤ 1.01, bulk/tail
    /// ESS ≥ 100). Smaller schedules routinely refuse under that bar.
    #[must_use]
    pub fn hmc() -> Self {
        Self {
            backend: BayesianBackendKind::Hmc,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 4_000,
            n_draws_explicit: false,
            prior_scale: 10.0,
            prior: None,
            prior_artifact: None,
            prior_mapping: None,
            external_compose: None,
            white_innovation: None,
            design_model: None,
            prior_variance_multiplier: None,
            predictive_replicates: None,
        }
    }

    /// Declare serially independent innovations for a single-step temporal linear effect.
    ///
    /// The posterior is then the untempered full-likelihood posterior of that effect rather
    /// than the generalized (tempered) posterior. The premise is declared here, never inferred;
    /// the `gate` residual test can refute it and a pass does not prove it. A horizon of more
    /// than one step, a rejecting test, a transferred prior and a non-Gaussian likelihood are
    /// refused with the next attempt named.
    #[must_use]
    pub const fn white_innovation(
        mut self,
        gate: antecedent_estimate::white_innovation_posterior::InnovationWhitenessGate,
    ) -> Self {
        self.white_innovation = Some(gate);
        self
    }

    /// Declare the likelihood-side model of a randomized-design posterior: the hazard
    /// partition and gamma prior of the survival posterior, or the dose support, coefficient
    /// scales and residual prior of the dose-policy posterior.
    #[must_use]
    pub fn design_model(
        mut self,
        model: crate::design_posterior_model::DesignPosteriorModel,
    ) -> Self {
        self.design_model = Some(model);
        self
    }

    /// Weakly informative prior scale.
    #[must_use]
    pub fn prior_scale(mut self, scale: f64) -> Self {
        self.prior_scale = scale;
        self
    }

    /// Outcome likelihood of the g-computation model (Laplace and HMC backends).
    ///
    /// [`BayesLikelihood::GaussianIdentity`] is the default. Bernoulli and
    /// Poisson likelihoods are supported on fixed-DAG tabular mean effects,
    /// conditional effects, and static response levels. Other routes, the
    /// conjugate backend, and transferred priors refuse at build with
    /// `likelihood_not_supported`.
    #[must_use]
    pub const fn likelihood(mut self, likelihood: BayesLikelihood) -> Self {
        self.likelihood = likelihood;
        self
    }

    /// Draw count.
    #[must_use]
    pub fn n_draws(mut self, n: usize) -> Self {
        self.n_draws = n;
        self.n_draws_explicit = true;
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
        self.external_compose = Some(Box::new(ExternalComposeSpec {
            sources,
            composed,
            conflict_policy: conflict,
            baseline: None,
        }));
        self
    }
    /// Use a composed prior and its original baseline for data-bound conflict re-shrink.
    #[must_use]
    pub fn prior_from_composed_with_baseline(
        self,
        sources: impl Into<Arc<[ExternalPriorSource]>>,
        composed: ComposedPrior,
        conflict: Option<ConflictPolicy>,
        baseline: PriorSet,
    ) -> Self {
        let mut config = self.prior_from_composed(sources, composed, conflict);
        if let Some(external) = &mut config.external_compose {
            external.baseline = Some(baseline);
        }
        config
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

/// How [`decode_prior_source`] treats an artifact's declared mapping.
#[derive(Clone, Copy)]
enum MappingDeclaration {
    /// Caller supplied `prior_mapping`; leave `prior_source.meta` unread.
    CallerSupplied,
    /// Caller left `prior_mapping` unset; honor a declaration on the artifact.
    FromArtifact,
}

/// One migrated decode of a posterior artifact for prior hydration.
struct DecodedPriorSource {
    wire: CausalPosteriorWire,
    /// Column-major draws. Empty for a summary-only artifact.
    draws: Vec<f64>,
    /// Mapping from `prior_source.meta` when [`MappingDeclaration::FromArtifact`]
    /// found one. Unset when that section was not read or declared no mapping.
    declared_mapping: Option<HydrateMapping>,
}

impl DecodedPriorSource {
    fn coefficient_names(&self) -> Vec<Option<&str>> {
        self.wire
            .quantities
            .iter()
            .filter_map(|q| match q {
                PosteriorQuantityWire::Coefficient { name, .. } => Some(name.as_deref()),
                _ => None,
            })
            .collect()
    }
}

/// Decode a posterior artifact once for the hydration checks that share it.
///
/// # Errors
///
/// Migration, prior-source metadata, or posterior decode failures.
fn decode_prior_source(
    bytes: &[u8],
    declaration: MappingDeclaration,
) -> Result<DecodedPriorSource, CausalError> {
    let artifact = read_and_migrate(bytes)?;
    let declared_mapping = match declaration {
        MappingDeclaration::CallerSupplied => None,
        MappingDeclaration::FromArtifact => extract_prior_source_meta(&artifact)?
            .and_then(|meta| meta.declared_mapping.as_ref().map(hydrate_mapping_from_io)),
    };
    let (wire, draws) = decode_posterior_artifact(&artifact)?;
    Ok(DecodedPriorSource { wire, draws, declared_mapping })
}

/// Choose hydrate mapping when the caller left `prior_mapping` unset.
///
/// Same-design sequential Bayes keeps identical subspace. Heterogeneous designs
/// default to effect-functional transfer when an `Effect` quantity exists;
/// otherwise the caller must supply an explicit mapping.
fn default_hydrate_mapping(
    source: &DecodedPriorSource,
    prep: &PreparedBayesianProblem,
) -> Result<HydrateMapping, CausalError> {
    if let Some(mapping) = &source.declared_mapping {
        return Ok(mapping.clone());
    }
    let source_coef_names = source.coefficient_names();
    let target_names = coef_names_for_problem(prep);
    if designs_compatible(&source_coef_names, prep.design.ncols, &target_names) {
        return Ok(HydrateMapping::IdenticalCoefficientSubspace);
    }
    let source_quantity = source.wire.quantities.iter().find_map(|q| match q {
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

/// Resolve the coefficient prior for a prepared Bayesian problem, running conflict shrink when `ctx`
/// is provided and a [`ConflictPolicy`] is configured on the external compose.
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
pub fn resolve_bayesian_prior_with_conflict(
    cfg: &BayesianConfig,
    prep: &PreparedBayesianProblem,
    ctx: Option<&ExecutionContext>,
) -> Result<(Option<PriorSet>, Option<ConflictSummary>), CausalError> {
    let (prior, conflict) = resolve_unscaled_prior(cfg, prep, ctx)?;
    let (Some(multiplier), Some(mut prior)) = (cfg.prior_variance_multiplier, prior.clone()) else {
        return Ok((prior, conflict));
    };
    if !(multiplier.is_finite() && multiplier > 0.0) {
        return Err(CausalError::Unsupported {
            message: "the prior variance multiplier must be finite and positive",
        });
    }
    let mut found = false;
    for spec in &mut prior.specs {
        if let antecedent_prob::PriorSpec::GaussianCoefficients(coefficients) = spec {
            coefficients.variance =
                coefficients.variance.iter().map(|variance| variance * multiplier).collect();
            found = true;
        }
    }
    if !found {
        return Err(CausalError::Unsupported {
            message: "the prior in force has no Gaussian coefficient block to scale",
        });
    }
    Ok((Some(prior), conflict))
}

fn resolve_unscaled_prior(
    cfg: &BayesianConfig,
    prep: &PreparedBayesianProblem,
    ctx: Option<&ExecutionContext>,
) -> Result<(Option<PriorSet>, Option<ConflictSummary>), CausalError> {
    if let Some(ext) = &cfg.external_compose {
        if let Some(policy) = ext.conflict_policy {
            policy.validate().map_err(CausalError::from)?;
        }
        if ext.conflict_policy.is_some() && ext.baseline.is_none() {
            return Err(CausalError::Unsupported {
                message: "conflict re-shrink requires the original composition baseline; recompose the prior",
            });
        }
        let target = coef_names_for_problem(prep);
        // A named prior whose coordinates are not the design columns in order is re-indexed by
        // name; the columns it has no coordinate for take the analysis's declared baseline, the
        // isotropic `prior_scale` prior an unnamed fit resolves against.
        let reindex = ext
            .composed
            .coefficient_names
            .as_ref()
            .filter(|names| ***names != *target)
            .map(Arc::clone);
        if let (Some(policy), Some(ctx)) = (&ext.conflict_policy, ctx) {
            let baseline = ext.baseline.as_ref().ok_or(CausalError::Unsupported {
                message: "conflict re-shrink requires the original composition baseline; recompose the prior",
            })?;
            let ppc = PriorPredictiveCheck {
                n_sims: 200,
                seed: ctx.rng.master_seed(),
                ..PriorPredictiveCheck::new()
            };
            let (composed, summary) = if let Some(names) = &reindex {
                // The shrink is per source and every source is evaluated over the whole design,
                // so each source and the baseline are first mapped onto the design by name.
                let fill = declared_design_baseline(cfg, target.len());
                let (sources, baseline) = project_external_sources(ext, names, &target, &fill)?;
                compose_with_conflict_policy(&sources, &baseline, policy, prep, ctx, &ppc)
                    .map_err(CausalError::from)?
            } else {
                compose_with_conflict_policy(&ext.sources, baseline, policy, prep, ctx, &ppc)
                    .map_err(CausalError::from)?
            };
            return Ok((Some(composed.prior), Some(summary)));
        }
        if let Some(names) = &reindex {
            let fill = declared_design_baseline(cfg, target.len());
            return Ok((Some(map_named_composed(&ext.composed, names, &target, &fill)?), None));
        }
        return Ok((Some(ext.composed.prior.clone()), None));
    }
    if let Some(p) = &cfg.prior {
        return Ok((Some(p.clone()), None));
    }
    let Some(bytes) = cfg.prior_artifact.as_ref() else {
        return Ok((None, None));
    };
    let declaration = if cfg.prior_mapping.is_some() {
        MappingDeclaration::CallerSupplied
    } else {
        MappingDeclaration::FromArtifact
    };
    let source = decode_prior_source(bytes, declaration)?;
    let mapping = match cfg.prior_mapping.as_ref() {
        Some(m) => hydrate_mapping_from_io(m),
        None => default_hydrate_mapping(&source, prep)?,
    };
    let names = coef_names_for_problem(prep);
    require_identity_link_static_transfer(&source, &names)?;
    let baseline = PriorSet::weakly_informative(prep.design.ncols);
    let treatment_col = prep.design.treatment_column();
    require_lag_aware_transfer(&source, &mapping, &names, treatment_col)?;
    Ok((
        Some(hydrate_prior_from_decoded(
            &source,
            &mapping,
            &baseline,
            &names,
            treatment_col,
            None,
        )?),
        None,
    ))
}

/// Map a named composed prior onto the design columns of a problem by coordinate name.
///
/// Source coordinates the design lacks are dropped (a marginal of a Gaussian prior), design
/// columns the source lacks take the matching coordinate of `baseline` (the analysis's declared
/// baseline prior over the design columns), and the mapping, with which coordinates came from the
/// source and which from the baseline, is recorded as a prior restriction. Only a composed prior none of whose coordinates names a design
/// column is refused.
///
/// # Errors
///
/// No coordinate of the composed prior names a design column, or the re-indexed prior is
/// invalid.
pub fn map_named_composed(
    composed: &ComposedPrior,
    names: &[Arc<str>],
    target: &[Arc<str>],
    baseline: &PriorSet,
) -> Result<PriorSet, CausalError> {
    let selection =
        antecedent_prob::select_coordinates_by_name(&composed.prior, names, target, baseline)
            .map_err(CausalError::from)?;
    if selection.mapped.is_empty() {
        let list =
            |items: &[Arc<str>]| items.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(", ");
        return Err(CausalError::Estimate(antecedent_estimate::EstimationError::refused(
            antecedent_core::reason_code!("quantity_semantics_mismatch"),
            format!(
                "the composed prior's coordinates [{}] map by name onto none of the design \
                 columns [{}]; name the sources' coefficients as the design columns they inform \
                 (ExternalPriorSourceSpec.coefficient_names)",
                list(names),
                list(target)
            ),
        )));
    }
    Ok(selection.prior)
}

/// The declared baseline prior over `ncols` design columns: the isotropic `prior_scale` prior the
/// fit resolves against when no prior is supplied.
#[must_use]
pub fn declared_design_baseline(cfg: &BayesianConfig, ncols: usize) -> PriorSet {
    let mut baseline = PriorSet::new();
    baseline.push(antecedent_prob::PriorSpec::GaussianCoefficients(
        antecedent_prob::GaussianCoefficientPrior::isotropic(ncols, cfg.prior_scale),
    ));
    baseline
}

/// Map every source of an external composition, and its composition baseline, onto the design
/// columns `target` by coordinate name, so the conflict re-shrink (a per-source, per-coordinate
/// computation over the whole design) applies to the mapped coordinates.
///
/// Source coordinates the design lacks are dropped. A source with no coordinate for some design
/// column cannot be evaluated over the whole design by the prior-predictive check, and is
/// refused with the columns it lacks; the baseline takes `fill` on columns it lacks.
///
/// # Errors
///
/// A source missing a design column, or a mapping failure.
pub fn project_external_sources(
    ext: &ExternalComposeSpec,
    names: &[Arc<str>],
    target: &[Arc<str>],
    fill: &PriorSet,
) -> Result<(Vec<ExternalPriorSource>, PriorSet), CausalError> {
    let missing: Vec<&str> = target
        .iter()
        .filter(|column| !names.iter().any(|name| name == *column))
        .map(AsRef::as_ref)
        .collect();
    if !missing.is_empty() {
        return Err(CausalError::Compile {
            message: format!(
                "conflict re-shrink evaluates each source over the whole design, but the named \
                 sources have no coordinate for the design columns [{}]; name those coordinates \
                 on every source, or compose without a conflict policy",
                missing.join(", ")
            ),
        });
    }
    let sources = ext
        .sources
        .iter()
        .map(|source| {
            antecedent_prob::select_coordinates_by_name(&source.prior, names, target, fill)
                .map(|selection| ExternalPriorSource { prior: selection.prior, ..source.clone() })
                .map_err(CausalError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let original = ext.baseline.as_ref().ok_or(CausalError::Unsupported {
        message: "conflict re-shrink requires the original composition baseline; recompose the prior",
    })?;
    let baseline = antecedent_prob::select_coordinates_by_name(original, names, target, fill)
        .map_err(CausalError::from)?
        .prior;
    Ok((sources, baseline))
}

/// Whether the external composition's sources apply positionally over `prep`'s design: an
/// unnamed composition, or a named one whose coordinates are the design's columns in order.
/// A named composition that is re-indexed onto the design no longer has sources aligned to its
/// coefficients, so a per-source re-composition (the α sensitivity grid) does not apply to it.
#[must_use]
pub(crate) fn external_compose_is_positional(
    ext: &ExternalComposeSpec,
    prep: &PreparedBayesianProblem,
) -> bool {
    ext.composed
        .coefficient_names
        .as_ref()
        .is_none_or(|names| **names == *coef_names_for_problem(prep))
}

/// A static coefficient transfer needs evidence that the source quantity was
/// an identity-link treatment contrast. Non-Gaussian effect draws live on
/// the response scale, so equal coefficient labels alone do not make their
/// slopes interchangeable with a Gaussian outcome model.
///
/// Temporal targets skip this check; [`require_lag_aware_transfer`] covers them.
fn require_identity_link_static_transfer(
    source: &DecodedPriorSource,
    target_names: &[Arc<str>],
) -> Result<(), CausalError> {
    if target_names.iter().any(|name| is_temporal_coefficient_name(name)) {
        return Ok(());
    }
    if source.wire.treatment_contrast.is_none() {
        return Err(CausalError::Unsupported {
            message: "static prior transfer requires an identity-link source contrast and matching coefficient meaning",
        });
    }
    Ok(())
}

/// Fail closed when a transfer onto a temporal (lag-named) design cannot be verified
/// to bind the same lagged coefficients.
///
/// Temporal Pulse / Sustained designs carry lag-aware coefficient names
/// (`coef_<var>@lag<k>`). Identical-subspace transfer binds by position, and
/// effect-functional transfer binds the source effect onto the target treatment
/// coefficient, so both require the source to carry the same lag coordinates:
///
/// - identical subspace: source coefficient names must equal the target's;
/// - effect functional: the source must carry the target's treatment coefficient
///   (same variable at the same lag);
/// - a source without coefficient names is refused;
/// - [`HydrateMapping::NamedParameters`] is the explicit bridge and is always allowed
///   (hydrate validates the names).
///
/// Static (non-temporal) targets keep the legacy rules.
fn require_lag_aware_transfer(
    source: &DecodedPriorSource,
    mapping: &HydrateMapping,
    target_names: &[Arc<str>],
    treatment_col: Option<usize>,
) -> Result<(), CausalError> {
    if !target_names.iter().any(|name| is_temporal_coefficient_name(name)) {
        return Ok(());
    }
    if matches!(mapping, HydrateMapping::NamedParameters { .. }) {
        return Ok(());
    }
    let source_names = source.coefficient_names();
    let target_list = target_names.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(", ");
    if source_names.iter().any(Option::is_none) {
        return Err(CausalError::Compile {
            message: format!(
                "temporal prior transfer refused: the source posterior carries no lag-aware \
                 coefficient names, so its lag structure cannot be checked \
                 against the target [{target_list}]; refit the source or declare \
                 PriorMapping::NamedParameters"
            ),
        });
    }
    let source_names: Vec<&str> = source_names.into_iter().flatten().collect();
    let source_list = source_names.join(", ");
    match mapping {
        HydrateMapping::IdenticalCoefficientSubspace => {
            if source_names.len() != target_names.len()
                || source_names.iter().zip(target_names).any(|(s, t)| *s != t.as_ref())
            {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal prior transfer refused: identical-subspace mapping binds by \
                         position, but source coefficients [{source_list}] differ from target \
                         [{target_list}] (lag structure or covariates); declare \
                         PriorMapping::NamedParameters"
                    ),
                });
            }
        }
        HydrateMapping::EffectFunctional { .. } => {
            let treatment = treatment_col
                .and_then(|col| target_names.get(col))
                .map(AsRef::as_ref)
                .unwrap_or_default();
            if !source_names.contains(&treatment) {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal prior transfer refused: effect-functional mapping onto \
                         `{treatment}` needs a source fitted at the same treatment lag, but the \
                         source coefficients are [{source_list}]; declare \
                         PriorMapping::NamedParameters"
                    ),
                });
            }
        }
        HydrateMapping::NamedParameters { .. } => {}
    }
    Ok(())
}

/// Decoded artifact summaries for mapped-prior hydrate.
#[derive(Clone, Debug)]
pub struct DecodedPriorHydrate {
    /// Mapping declared on the Bayesian config.
    pub mapping: HydrateMapping,
    /// Source posterior quantity kinds.
    pub quantities: Vec<PosteriorQuantityKind>,
    /// Source posterior means.
    pub mean: Vec<f64>,
    /// Source posterior SDs.
    pub sd: Vec<f64>,
    /// Source treatment contrast `active − control` recorded on the artifact.
    pub source_contrast: Option<f64>,
}

/// Decode a mapped prior artifact for mechanism hydrate.
///
/// Returns `None` when no artifact is set. A present artifact without a mapping
/// is the caller's fail-closed case.
///
/// # Errors
///
/// Decode failures.
pub fn decode_prior_hydrate_source(
    cfg: &BayesianConfig,
) -> Result<Option<DecodedPriorHydrate>, CausalError> {
    let Some(bytes) = cfg.prior_artifact.as_ref() else {
        return Ok(None);
    };
    let Some(mapping) = cfg.prior_mapping.as_ref() else {
        return Ok(None);
    };
    let (wire, _) = decode_causal_posterior_bytes(bytes)?;
    Ok(Some(DecodedPriorHydrate {
        mapping: hydrate_mapping_from_io(mapping),
        quantities: wire_quantities_to_kinds(&wire.quantities),
        mean: wire.mean,
        sd: wire.sd,
        source_contrast: wire.treatment_contrast,
    }))
}

/// Decode a posterior artifact into the outcome-nuisance prior of a cross-fitted AIPW
/// estimate ([`antecedent_estimate::DmlAte::with_outcome_prior`]).
///
/// The artifact's coefficient moments (and, when its draws were retained, the dense coefficient
/// covariance) are kept; mapping them onto the design columns of the outcome regression happens
/// when the problem is prepared. `mapping` is the declared mapping; `None` takes the mapping the
/// artifact declares, else the identical coefficient subspace. The source identity is the
/// content digest of the artifact bytes.
///
/// # Errors
///
/// Migration or decode failures, or a mapping that has no coordinate in the outcome regression
/// (an effect-functional mapping).
pub fn outcome_prior_source_from_artifact(
    bytes: &[u8],
    mapping: Option<&PriorMapping>,
) -> Result<antecedent_estimate::OutcomePriorSource, CausalError> {
    let declaration = if mapping.is_some() {
        MappingDeclaration::CallerSupplied
    } else {
        MappingDeclaration::FromArtifact
    };
    let source = decode_prior_source(bytes, declaration)?;
    let mapping = match mapping {
        Some(declared) => hydrate_mapping_from_io(declared),
        None => {
            source.declared_mapping.clone().unwrap_or(HydrateMapping::IdenticalCoefficientSubspace)
        }
    };
    let quantities = wire_quantities_to_kinds(&source.wire.quantities);
    let covariance =
        coefficient_covariance_from_draws(&quantities, &source.draws, source.wire.n_draws as usize)
            .map_err(CausalError::from)?;
    antecedent_estimate::OutcomePriorSource::new(
        quantities,
        source.wire.mean.clone(),
        source.wire.sd.clone(),
        covariance,
        mapping,
        format!("blake3:{}", antecedent_core::identity::blake3_hex(bytes)),
    )
    .map_err(CausalError::from)
}

/// The outcome-nuisance prior of a cross-fitted AIPW estimate
/// ([`antecedent_estimate::DmlAte::with_outcome_prior`]) from a composed external prior.
///
/// The composed prior (its sources, transport weighting and power-prior discount already
/// applied) is a Gaussian coefficient prior on the conjugate scale. Its named coordinates map by
/// name onto the outcome-regression columns when the problem is prepared; an unnamed one is
/// positional over a design of the same width. The identity is a digest of the prior's
/// moments, correlation, coordinate names and provenance restrictions, so a different
/// transport kernel, bandwidth or selected α is a different prior.
///
/// # Errors
///
/// A composed prior without a Gaussian coefficient block, or with absolute-scale variances.
pub fn outcome_prior_source_from_composed(
    composed: &ComposedPrior,
) -> Result<antecedent_estimate::OutcomePriorSource, CausalError> {
    let identity = antecedent_io::prior_set_identity(&composed.prior);
    let names = composed
        .coefficient_names
        .as_ref()
        .map(|names| names.iter().map(AsRef::as_ref).collect::<Vec<&str>>().join(","));
    let canonical = format!("{identity:?};names={names:?}");
    antecedent_estimate::OutcomePriorSource::from_composed(
        composed.coefficient_names.clone(),
        composed.prior.clone(),
        format!("composed:blake3:{}", antecedent_core::identity::blake3_hex(canonical.as_bytes())),
    )
    .map_err(CausalError::from)
}

/// Hydrate a [`PriorSet`] from an already decoded posterior under a [`HydrateMapping`].
///
/// Identical-subspace hydration reads the coefficient covariance from the artifact's draws and
/// carries it as a dense `V0`; an artifact without draws (summary-only) hydrates the diagonal
/// prior.
fn hydrate_prior_from_decoded(
    source: &DecodedPriorSource,
    mapping: &HydrateMapping,
    baseline: &PriorSet,
    target_coef_names: &[Arc<str>],
    treatment_col: Option<usize>,
    fallback_contrast: Option<f64>,
) -> Result<PriorSet, CausalError> {
    let quantities = wire_quantities_to_kinds(&source.wire.quantities);
    let source_contrast = source.wire.treatment_contrast.or(fallback_contrast);
    let covariance =
        coefficient_covariance_from_draws(&quantities, &source.draws, source.wire.n_draws as usize)
            .map_err(CausalError::from)?;
    hydrate_prior(
        mapping,
        &HydrateSource::summaries(&quantities, &source.wire.mean, &source.wire.sd)
            .with_coefficient_covariance(covariance.as_deref())
            .with_source_contrast(source_contrast),
        &HydrateTarget { baseline, coef_names: target_coef_names, treatment_col },
    )
    .map_err(CausalError::from)
}

/// What a mapped posterior transfer targets: the independently supplied source identity, the
/// stable-ID map and the target's quantities, snapshot, contract and baseline prior.
#[derive(Clone, Copy, Debug)]
pub struct MappedTransferSpec<'a> {
    /// Source identity, supplied independently of the artifact bytes.
    pub expected_source: &'a DistributionIdentity,
    /// One stable source-variable-ID to target-variable-ID pair per coefficient.
    pub source_to_target: &'a [(String, String)],
    /// Target quantities in design-column order.
    pub target_quantities: &'a [ScientificQuantityWire],
    /// Snapshot id of the target.
    pub target_snapshot_id: &'a str,
    /// Causal-contract id of the target.
    pub target_contract_id: &'a str,
    /// Baseline prior carrying the target's known residual variance.
    pub baseline: &'a PriorSet,
}

/// Relative tolerance, floored at an absolute `1e-12`, within which a source draw's
/// residual variance equals the target's known variance.
const RESIDUAL_VARIANCE_TOLERANCE: f64 = 1e-12;

fn transfer_refusal(code: &'static str, detail: &str, message: &str) -> CausalError {
    crate::error::io_refused(code, detail, message).into()
}

fn map_refuse(message: &str) -> CausalError {
    transfer_refusal(
        antecedent_core::reason_code!("quantity_semantics_mismatch"),
        antecedent_core::reason_details::mapped_posterior_transfer::COEFFICIENT_MAP,
        message,
    )
}

fn meaning_refuse(message: &str) -> CausalError {
    transfer_refusal(
        antecedent_core::reason_code!("distribution_meaning_mismatch"),
        antecedent_core::reason_details::mapped_posterior_transfer::PARAMETER_POSTERIOR,
        message,
    )
}

fn variance_refuse(message: &str) -> CausalError {
    transfer_refusal(
        antecedent_core::reason_code!("prior_transfer_not_hydrated"),
        antecedent_core::reason_details::mapped_posterior_transfer::RESIDUAL_VARIANCE,
        message,
    )
}

/// One mapped coefficient: its column in the source artifact and in the target, and the stable
/// ids on both sides.
struct MappedCoefficient {
    source_col: usize,
    target_col: usize,
    source_id: String,
    target_id: String,
}

/// The source must be complete, unweighted, joint parameter-posterior draws; returns the
/// target's known residual variance, which must be a finite positive number.
fn check_source_meaning(
    source: &DistributionArtifact,
    spec: &MappedTransferSpec<'_>,
) -> Result<f64, CausalError> {
    if spec.target_snapshot_id.trim().is_empty() || spec.target_contract_id.trim().is_empty() {
        return Err(map_refuse("mapped transfer requires target snapshot and causal contract IDs"));
    }
    if source.semantic() != DistributionMeaningWire::ParameterPosterior
        || source.metadata().identity.alignment != DrawAlignment::Joint
        || source.metadata().weights.is_some()
        || source.metadata().supported.as_ref().is_some_and(|mask| mask.iter().any(|ok| !ok))
    {
        return Err(meaning_refuse(
            "mapped transfer requires complete, unweighted joint parameter-posterior draws",
        ));
    }
    let sigma2 = spec.baseline.known_residual_variance().ok_or_else(|| {
        variance_refuse("mapped transfer requires a target known residual variance")
    })?;
    // The getter returns the stored value unchecked, so the comparison below must not rely on
    // later prior validation to reject a non-finite or non-positive variance.
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return Err(variance_refuse(
            "mapped transfer requires the target known residual variance to be finite and positive",
        ));
    }
    Ok(sigma2)
}

/// The complete bijection between the source's and the target's coefficient coordinates, in the
/// order of `source_to_target`.
fn build_coefficient_map(
    source: &DistributionArtifact,
    spec: &MappedTransferSpec<'_>,
) -> Result<Vec<MappedCoefficient>, CausalError> {
    let target_quantities = spec.target_quantities;
    if spec.source_to_target.len() != target_quantities.len()
        || spec
            .baseline
            .gaussian_coefficients()
            .is_none_or(|prior| prior.len() != target_quantities.len())
        || target_quantities.is_empty()
    {
        return Err(map_refuse("mapped transfer requires a complete coefficient bijection"));
    }
    let mut used_source = std::collections::HashSet::new();
    let mut used_target = std::collections::HashSet::new();
    let mut columns = Vec::with_capacity(target_quantities.len());
    for (source_id, target_id) in spec.source_to_target {
        if !used_source.insert(source_id) || !used_target.insert(target_id) {
            return Err(map_refuse("mapped transfer reuses a coefficient identity"));
        }
        let source_matches: Vec<_> = source
            .quantities()
            .iter()
            .enumerate()
            .filter(|(_, q)| q.variable_id == *source_id && q.functional_id == "coefficient")
            .collect();
        let target_matches: Vec<_> = target_quantities
            .iter()
            .enumerate()
            .filter(|(_, q)| q.variable_id == *target_id && q.functional_id == "coefficient")
            .collect();
        if source_matches.len() != 1 || target_matches.len() != 1 {
            return Err(map_refuse("mapped transfer needs unique stable coefficient IDs"));
        }
        let (source_col, source_q) = source_matches[0];
        let (target_col, target_q) = target_matches[0];
        if source_q.units != target_q.units
            || source_q.population_id != target_q.population_id
            || source_q.regime_id != target_q.regime_id
            || source_q.horizon != target_q.horizon
            || source_q.conditioning != target_q.conditioning
            || source_q.transform_id != target_q.transform_id
            || source_q.role != target_q.role
        {
            return Err(map_refuse("mapped transfer coefficient coordinates differ"));
        }
        columns.push(MappedCoefficient {
            source_col,
            target_col,
            source_id: source_id.clone(),
            target_id: target_id.clone(),
        });
    }
    Ok(columns)
}

/// The source's one residual-variance coordinate must sit at the target's coordinates and equal
/// the target's known variance in every draw (see [`RESIDUAL_VARIANCE_TOLERANCE`]).
fn check_residual_variance(
    source: &DistributionArtifact,
    spec: &MappedTransferSpec<'_>,
    coefficient_count: usize,
    sigma2: f64,
) -> Result<(), CausalError> {
    let residuals: Vec<_> = source
        .quantities()
        .iter()
        .enumerate()
        .filter(|(_, q)| q.functional_id == "residual_variance")
        .collect();
    if residuals.len() != 1 || source.quantities().len() != coefficient_count + 1 {
        return Err(variance_refuse(
            "mapped transfer requires one residual-variance coordinate and no extra parameters",
        ));
    }
    let (residual_col, residual_q) = residuals[0];
    let target = &spec.target_quantities[0];
    if residual_q.population_id != target.population_id
        || residual_q.regime_id != target.regime_id
        || residual_q.horizon != target.horizon
        || residual_q.conditioning != target.conditioning
    {
        return Err(variance_refuse("mapped transfer residual-variance coordinate differs"));
    }
    if source.draws().chunks_exact(source.shape()[1]).any(|row| {
        (row[residual_col] - sigma2).abs() > RESIDUAL_VARIANCE_TOLERANCE * sigma2.max(1.0)
    }) {
        return Err(variance_refuse(
            "mapped transfer residual variance differs from target known variance",
        ));
    }
    Ok(())
}

/// Hydrate the target prior from the source's mapped coefficient moments and full covariance.
fn hydrate_mapped_prior(
    source: &DistributionArtifact,
    spec: &MappedTransferSpec<'_>,
    mut coefficients: Vec<MappedCoefficient>,
    sigma2: f64,
) -> Result<PriorSet, CausalError> {
    coefficients.sort_by_key(|coefficient| coefficient.target_col);
    let mut quantities = Vec::with_capacity(source.quantities().len());
    let mut mean = Vec::with_capacity(source.quantities().len());
    let mut sd = Vec::with_capacity(source.quantities().len());
    for (index, coefficient) in coefficients.iter().enumerate() {
        quantities.push(PosteriorQuantityKind::Coefficient {
            index,
            name: Some(Arc::from(coefficient.source_id.as_str())),
        });
        mean.push(source.mean(coefficient.source_col)?);
        sd.push(source.covariance(coefficient.source_col, coefficient.source_col)?.sqrt());
    }
    quantities.push(PosteriorQuantityKind::ResidualVariance);
    mean.push(sigma2);
    sd.push(0.0);
    let n = coefficients.len();
    let mut covariance = vec![0.0; n * n];
    for (i, row) in coefficients.iter().enumerate() {
        for (j, column) in coefficients.iter().enumerate() {
            covariance[i * n + j] = source.covariance(row.source_col, column.source_col)?;
        }
    }
    let names: Vec<Arc<str>> =
        spec.target_quantities.iter().map(|q| Arc::from(q.variable_id.as_str())).collect();
    let mapping = HydrateMapping::NamedParameters {
        pairs: coefficients
            .iter()
            .map(|coefficient| (coefficient.source_id.clone(), coefficient.target_id.clone()))
            .collect(),
    };
    let mut prior = hydrate_prior(
        &mapping,
        &HydrateSource::summaries(&quantities, &mean, &sd)
            .with_coefficient_covariance(Some(&covariance)),
        &HydrateTarget { baseline: spec.baseline, coef_names: &names, treatment_col: None },
    )
    .map_err(CausalError::from)?;
    prior.restrictions.push(antecedent_core::PriorAssumption {
        id: Arc::from("mapped_posterior_transfer"),
        description: Arc::from(format!(
            "source={} snapshot={} contract={}; target snapshot={} contract={}; complete stable-ID map={:?}; known residual variance={sigma2}",
            spec.expected_source.source_id,
            spec.expected_source.snapshot_id,
            spec.expected_source.causal_contract_id,
            spec.target_snapshot_id,
            spec.target_contract_id,
            spec.source_to_target,
        )),
    });
    Ok(prior)
}

/// Transfer an aligned parameter posterior into a known-variance Gaussian prior.
///
/// The caller supplies the source identity independently of the bytes and one
/// stable source-variable-ID to target-variable-ID pair per coefficient. Target
/// quantities are in design-column order. This route requires a complete
/// bijection, a constant residual-variance coordinate equal to the target's
/// known variance (finite and positive), and identical non-name quantity
/// coordinates. The finite artifact's full covariance is retained under the
/// named permutation.
///
/// # Errors
/// Incompatible meaning, source identity, mapping, scale, or draw alignment.
pub fn mapped_posterior_transfer_with(
    bytes: &[u8],
    spec: &MappedTransferSpec<'_>,
) -> Result<PriorSet, CausalError> {
    let source = DistributionArtifact::from_bytes(bytes, spec.expected_source)?;
    let sigma2 = check_source_meaning(&source, spec)?;
    let coefficients = build_coefficient_map(&source, spec)?;
    check_residual_variance(&source, spec, coefficients.len(), sigma2)?;
    hydrate_mapped_prior(&source, spec, coefficients, sigma2)
}

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Inference backend interface and reusable workspaces.
pub use antecedent_prob::BayesLikelihood;
/// Inference diagnostics for Laplace / conjugate / MCMC backends.
pub use antecedent_prob::ConflictSummary;
/// Probability / inference errors.
pub use antecedent_prob::ProbError;
/// Structured transport refusals, including a compiled transport route's support refusal.
pub use antecedent_prob::TransportError;
/// Power-prior discount chosen from out-of-sample fit, with a selection-aware interval.
pub use antecedent_prob::{
    AlphaCriterion, AlphaSelectionOptions, AlphaSensitivityRow, BetaSource, GammaSource,
    LinearTarget, SelectedAlphaReport, default_alpha_grid,
};
/// Row-weighted target populations and kernel transport weights.
pub use antecedent_prob::{
    BandwidthSensitivityRow, KernelFamily, TARGET_MIN_ESS_FRACTION, TargetOverlap, TransportKernel,
    WeightedTarget, check_weights_overlap,
};
/// Conjugate Beta-Binomial and Gamma-Poisson count posteriors.
pub use antecedent_prob::{BetaBinomialPosterior, GammaPoissonPosterior};
/// Conjugate moment-matching converters: Gaussian summary → Beta / Gamma.
pub use antecedent_prob::{BetaHyperparameters, GammaHyperparameters};
/// External prior bank: power-prior / mixture composition.
pub use antecedent_prob::{
    ComposedPrior, ExternalPriorSource, ExternalPriorWeight, compose_external_priors_with_alphas,
};
/// Coordinate-name mapping of composed priors.
pub use antecedent_prob::{CoordinateMapping, rename_coordinates};
/// Prior specifications.
pub use antecedent_prob::{GaussianCoefficientPrior, PriorSet, PriorSpec};
/// Weighted graph samples for effect envelopes.
pub use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
/// Transport policy for cross-population prior transfer.
pub use antecedent_prob::{TransportAdjustment, TransportContext, TransportPolicy};
/// `antecedent-stats` items at the crate root.
pub use antecedent_stats::ConditionalIndependence;
/// Joint-law propagation arithmetic.
pub use antecedent_stats::DecisionVerdict;
/// Randomization-based kernels for known network assignment designs.
pub use antecedent_stats::ExposureProbabilityMethod;
/// Partial-correlation conditional independence test.
pub use antecedent_stats::PartialCorrelation;
/// CI request / result types and workspace.
pub use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, SignificanceMethod,
};
/// Multiple-testing / false-discovery-rate helpers.
pub use antecedent_stats::{FdrAdjustment, MultipleTestingMethod};
/// GLM fitting: logistic, probit, multinomial logit, Gaussian, Poisson, and NB2 IRLS.
pub use antecedent_stats::{GlmFamily, GlmOptions};
/// Stats-layer errors.
pub use antecedent_stats::{GlmRefusalKind, StatsError};

/// The Gaussian prior composed from `sources` at the discount chosen from the target data, with
/// the selection report. Every source takes the selected `α`; the selection (criterion, `α`,
/// level, interval semantics) is a restriction of the composed prior, so it enters its identity.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::select_linear_prior`].
///
/// # Errors
///
/// As `select_linear_power_prior`, or a composition failure.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled PowerPriorLinearSelection route (Study::compile_*_power_prior_selection), which calls the same kernel function"
)]
pub fn select_linear_prior(
    sources: &[antecedent_prob::ExternalPriorSource],
    baseline: &antecedent_prob::PriorSet,
    target: antecedent_prob::LinearTarget<'_>,
    options: &antecedent_prob::AlphaSelectionOptions,
) -> Result<
    (antecedent_prob::ComposedPrior, antecedent_prob::SelectedAlphaReport),
    antecedent_prob::ProbError,
> {
    antecedent_prob::select_linear_prior(sources, baseline, target, options)
}

/// Choose the power-prior `α` of external Gaussian sources for a Gaussian linear target,
/// and report the target functional's posterior with intervals that stay valid under the selection.
///
/// Every source takes the same `α` from the grid (the power path of
/// `compose_external_priors_with_alphas`); a source with a mixture weight is refused.
/// The residual variance is the baseline's: known, or Inv-Gamma. Rows are unweighted.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::select_linear_power_prior`].
///
/// # Errors
///
/// A design or functional of the wrong shape or with non-finite entries, a mixture-path
/// source, an invalid composition, invalid options, or a numerically singular posterior.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled PowerPriorLinearSelection route (Study::compile_*_power_prior_selection), which calls the same kernel function"
)]
pub fn select_linear_power_prior(
    sources: &[antecedent_prob::ExternalPriorSource],
    baseline: &antecedent_prob::PriorSet,
    target: antecedent_prob::LinearTarget<'_>,
    options: &antecedent_prob::AlphaSelectionOptions,
) -> Result<antecedent_prob::SelectedAlphaReport, antecedent_prob::ProbError> {
    antecedent_prob::select_linear_power_prior(sources, baseline, target, options)
}

/// The Gamma power prior at the discount chosen from the target counts, with the report.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::select_gamma_prior`].
///
/// # Errors
///
/// As `select_gamma_power_prior`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled PowerPriorGammaSelection route (Study::compile_*_power_prior_selection), which calls the same kernel function"
)]
pub fn select_gamma_prior(
    source: antecedent_prob::GammaSource,
    counts: &[u64],
    options: &antecedent_prob::AlphaSelectionOptions,
) -> Result<
    (antecedent_prob::GammaHyperparameters, antecedent_prob::SelectedAlphaReport),
    antecedent_prob::ProbError,
> {
    antecedent_prob::select_gamma_prior(source, counts, options)
}

/// Choose the power-prior `α` of a source rate for a Gamma-Poisson target.
///
/// `counts[i]` is target observation `i`'s count over unit exposure. The reported
/// functional is the rate `λ`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::select_gamma_power_prior`].
///
/// # Errors
///
/// An invalid source or options, or fewer than two target observations.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled PowerPriorGammaSelection route (Study::compile_*_power_prior_selection), which calls the same kernel function"
)]
pub fn select_gamma_power_prior(
    source: antecedent_prob::GammaSource,
    counts: &[u64],
    options: &antecedent_prob::AlphaSelectionOptions,
) -> Result<antecedent_prob::SelectedAlphaReport, antecedent_prob::ProbError> {
    antecedent_prob::select_gamma_power_prior(source, counts, options)
}

/// The Beta power prior at the discount chosen from the target trials, with the selection report.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::select_beta_prior`].
///
/// # Errors
///
/// As `select_beta_power_prior`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled PowerPriorBetaSelection route (Study::compile_*_power_prior_selection), which calls the same kernel function"
)]
pub fn select_beta_prior(
    source: antecedent_prob::BetaSource,
    outcomes: &[bool],
    options: &antecedent_prob::AlphaSelectionOptions,
) -> Result<
    (antecedent_prob::BetaHyperparameters, antecedent_prob::SelectedAlphaReport),
    antecedent_prob::ProbError,
> {
    antecedent_prob::select_beta_prior(source, outcomes, options)
}

/// Choose the power-prior `α` of a source proportion for a Beta-Binomial target.
///
/// `outcomes[i]` is whether target trial `i` was a success (for the exceedance query,
/// whether the outcome exceeded the threshold). The reported functional is the proportion
/// `θ`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::select_beta_power_prior`].
///
/// # Errors
///
/// An invalid source or options, or fewer than two target trials.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled PowerPriorBetaSelection route (Study::compile_*_power_prior_selection), which calls the same kernel function"
)]
pub fn select_beta_power_prior(
    source: antecedent_prob::BetaSource,
    outcomes: &[bool],
    options: &antecedent_prob::AlphaSelectionOptions,
) -> Result<antecedent_prob::SelectedAlphaReport, antecedent_prob::ProbError> {
    antecedent_prob::select_beta_power_prior(source, outcomes, options)
}

/// How the transported estimate moves across a set of declared bandwidths.
///
/// A bandwidth whose weighting fails the overlap check is reported with its
/// refusal message rather than dropped, so the report shows where transport
/// stops being supported.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::kernel_bandwidth_sensitivity`].
///
/// # Errors
///
/// Empty bandwidth set, an invalid bandwidth, or invalid effects / distances.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorTransport route (Study::compile_kernel_bandwidth_sensitivity, CompiledKernelBandwidthSensitivity::sensitivity), which calls the same kernel function"
)]
pub fn kernel_bandwidth_sensitivity(
    family: antecedent_prob::KernelFamily,
    bandwidths: &[f64],
    distances: &[f64],
    unit_effects: &[f64],
    min_ess_fraction: f64,
) -> Result<Vec<antecedent_prob::BandwidthSensitivityRow>, antecedent_prob::TransportError> {
    antecedent_prob::kernel_bandwidth_sensitivity(
        family,
        bandwidths,
        distances,
        unit_effects,
        min_ess_fraction,
    )
}

/// Transport toward a weighted target, then compose.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::compose_with_weighted_target_transport`].
///
/// # Errors
///
/// Overlap refusal, transport failures, or composition failures.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorTransport route (Study::compile_prior_weighted_target_transport, CompiledPriorWeightedTargetTransport::compose_with_weighted_target_transport), which calls the same kernel function"
)]
pub fn compose_with_weighted_target_transport(
    sources: &[antecedent_prob::ExternalPriorSource],
    baseline: &antecedent_prob::PriorSet,
    policy: antecedent_prob::TransportPolicy,
    target: &antecedent_prob::WeightedTarget,
    coef_index: Option<usize>,
) -> Result<
    (antecedent_prob::ComposedPrior, antecedent_prob::TargetOverlap),
    antecedent_prob::ProbError,
> {
    antecedent_prob::compose_with_weighted_target_transport(
        sources, baseline, policy, target, coef_index,
    )
}

/// Apply transport then compose (power / mixture path).
///
/// When transport forces α overrides, those override the source weights for
/// composition while preserving the original requested alphas.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::compose_with_transport`].
///
/// # Errors
///
/// Transport gate / adjustment failures or composition failures.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorTransport route (Study::compile_prior_tag_transport, CompiledPriorTagTransport::compose_with_transport), which calls the same kernel function"
)]
pub fn compose_with_transport(
    sources: &[antecedent_prob::ExternalPriorSource],
    baseline: &antecedent_prob::PriorSet,
    ctx: &antecedent_prob::TransportContext<'_>,
) -> Result<
    (antecedent_prob::ComposedPrior, Vec<antecedent_prob::TransportOutcome>),
    antecedent_prob::ProbError,
> {
    antecedent_prob::compose_with_transport(sources, baseline, ctx)
}

/// Compose Gamma-Poisson count sources `(total count, exposure, α)` into one Gamma prior:
/// `Gamma(s + Σ α S, r + Σ α E)`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::compose_gamma_power_priors`].
///
/// # Errors
///
/// An invalid base, a negative exposure, or `α` outside `[0, 1]`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorComposition route (Study::compile_gamma_power_prior_composition, CompiledGammaPowerPriorComposition::compose), which calls the same kernel function"
)]
pub fn compose_gamma_power_priors(
    base: antecedent_prob::GammaHyperparameters,
    sources: &[(u64, f64, f64)],
) -> Result<antecedent_prob::GammaHyperparameters, antecedent_prob::ProbError> {
    antecedent_prob::compose_gamma_power_priors(base, sources)
}

/// Compose external Gaussian coefficient priors with a baseline.
///
/// * **Power path** (all `mixture_weight` are `None`):\
///   `Λ = Λ₀ + Σ α_k Λ_k`, mean from precision-weighted average.
/// * **Mixture path** (all weights set): moment-match\
///   `Σ w_k N(μ_k, V_k/α_k) + (1−Σw) · baseline`. Sources with `α_k = 0` are
///   dropped and their weight folds into leftover baseline mass.
///
/// Uses `weight.alpha` as both requested and applied. For conflict shrink,
/// mutate source alphas (or call `compose_external_priors_with_alphas`) before
/// composing.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::compose_external_priors`].
///
/// # Errors
///
/// Invalid weights, missing Gaussians, or dimension mismatch.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorComposition route (Study::compile_external_prior_composition, CompiledExternalPriorComposition::compose), which calls the same kernel function"
)]
pub fn compose_external_priors(
    sources: &[antecedent_prob::ExternalPriorSource],
    baseline: &antecedent_prob::PriorSet,
) -> Result<antecedent_prob::ComposedPrior, antecedent_prob::ProbError> {
    antecedent_prob::compose_external_priors(sources, baseline)
}

/// Compose Beta-Binomial count sources `(successes, trials, α)` into one Beta prior:
/// `Beta(a + Σ α k, b + Σ α (n − k))`.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::compose_beta_power_priors`].
///
/// # Errors
///
/// An invalid base, `successes > trials`, or `α` outside `[0, 1]`.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorComposition route (Study::compile_beta_power_prior_composition, CompiledBetaPowerPriorComposition::compose), which calls the same kernel function"
)]
pub fn compose_beta_power_priors(
    base: antecedent_prob::BetaHyperparameters,
    sources: &[(u64, u64, f64)],
) -> Result<antecedent_prob::BetaHyperparameters, antecedent_prob::ProbError> {
    antecedent_prob::compose_beta_power_priors(base, sources)
}

/// Transport sources toward a weighted target.
///
/// Refuses when the weighting fails the overlap check. Otherwise each source's
/// chosen coefficient takes the weighted mean and is widened by the sampling
/// variance of that mean, exactly as the tag path does for caller weights, and
/// the claim (and kernel, when declared) is recorded as assumptions.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::apply_weighted_target_transport`].
///
/// # Errors
///
/// Overlap refusal, or an invalid coefficient index / prior.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorTransport route (Study::compile_prior_weighted_target_transport, CompiledPriorWeightedTargetTransport::apply_weighted_target_transport), which calls the same kernel function"
)]
pub fn apply_weighted_target_transport(
    sources: &[antecedent_prob::ExternalPriorSource],
    policy: antecedent_prob::TransportPolicy,
    target: &antecedent_prob::WeightedTarget,
    coef_index: Option<usize>,
) -> Result<
    (Vec<antecedent_prob::ExternalPriorSource>, antecedent_prob::TargetOverlap),
    antecedent_prob::ProbError,
> {
    antecedent_prob::apply_weighted_target_transport(sources, policy, target, coef_index)
}

/// Apply transport gate / adjustment to cloned sources; return prepared sources
/// and per-source outcomes.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_prob::apply_transport`].
///
/// # Errors
///
/// `TransportError::PolicyRequired` when populations differ without a policy,
/// length mismatches, or invalid adjustment weights.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled ExternalPriorTransport route (Study::compile_prior_tag_transport, CompiledPriorTagTransport::apply_transport), which calls the same kernel function"
)]
pub fn apply_transport(
    sources: &[antecedent_prob::ExternalPriorSource],
    ctx: &antecedent_prob::TransportContext<'_>,
) -> Result<
    (Vec<antecedent_prob::ExternalPriorSource>, Vec<antecedent_prob::TransportOutcome>),
    antecedent_prob::TransportError,
> {
    antecedent_prob::apply_transport(sources, ctx)
}

/// Numerics for inference from adaptively collected data.
pub mod adaptive_inference {
    pub use antecedent_stats::adaptive_inference::{
        AnalysisEstimator, AnytimeSettings, ArmPrior, AssignmentRule, Attribution, BetaPrior,
        BorrowingPlan, ComparisonReading, CorrelationPrior, DecisionRules, Estimand,
        EstimandStatement, EstimatorLaw, ExposureModel, FinalAnalysis, FinalSelection,
        FutilityRule, GaussianMixtureLaw, HistoricalPrior, InversionGrid, LookDecision,
        LookReading, MAX_ADAPTIVE_UNITS, ObservedTerminal, OutcomeModelPrior, OutcomeObservation,
        OutcomeStructure, ParticipantDeclaration, ParticipantHistory, ParticipantSimulationRequest,
        ProcedureDeclaration, RepeatedMeasuresDeclaration, RepeatedMeasuresRequest,
        ResolvedArmPrior, ResourceReceipt, SimulationRequest, StoppingRule, UncertaintyParadigm,
        UnitAssignment, UnitSummary, WeightScheme, WithinUnitCorrelation,
    };
    /// Group-sequential boundaries by alpha spending.
    pub mod group_sequential {
        pub use antecedent_stats::adaptive_inference::group_sequential::{
            Sidedness, StagewiseLook,
        };
    }
}
/// Randomization-based kernels for known network assignment designs.
pub mod interference {
    pub use antecedent_stats::interference::admit_exposure_probability_work_for_shape;
}
/// Pure corrections for uncertain measurements.
pub mod measurement_error {
    pub use antecedent_stats::measurement_error::{
        BinaryErrorRates, CorrectedExposureAssociation, DetectionAdjusted, DetectionAnchor,
        DetectionBounds, DetectionData, DetectionDeclaration, DetectionDirection,
        DetectionFunction, DetectionPoint, ExposureErrorStructure, ExposureOutcomeCounts,
        FalsePositiveRate, LatentClassConstraints, LatentClassData, LatentClassFit,
        LatentClassIdentification, LatentClassSettings, LatentClassStandardErrors,
        ObservedClassSummary, ReliabilityBin, SoftExposureData, SoftExposureReliability,
        TwoByTwoCounts, exposure_rates_from_validation,
    };
}
/// Items of `antecedent-prob::metropolis`.
pub mod metropolis {
    pub use antecedent_prob::metropolis::MetropolisOptions;
}
/// Summary-data synthesis: covariance-preserving pooling.
pub mod synthesis {
    pub use antecedent_stats::synthesis::{BetweenStudyMethod, MeanPrior, TauPrior};
}
