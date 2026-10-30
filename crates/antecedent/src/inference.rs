//! Bayesian inference configuration for the facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
use antecedent_prob::{
    BayesLikelihood, ComposedPrior, ConflictSummary, ExternalPriorSource, PosteriorQuantityKind,
    PriorSet,
};
use antecedent_validate::{ConflictPolicy, PriorPredictiveCheck, compose_with_conflict_policy};

use crate::error::CausalError;
use crate::io::decode_causal_posterior_bytes;

/// Frequentist vs Bayesian inference mode.
#[allow(
    clippy::large_enum_variant,
    reason = "the Bayesian configuration is declared once per study and moved by value; boxing would change every public pattern match"
)]
#[derive(Clone, Debug, PartialEq)]
pub enum InferenceMode {
    /// Classical point-estimate path (default).
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
        if ext.conflict_policy.is_some() && ext.baseline.is_none() {
            return Err(CausalError::Unsupported {
                message: "conflict re-shrink requires the original composition baseline; recompose the prior",
            });
        }
        if let (Some(policy), Some(ctx)) = (&ext.conflict_policy, ctx) {
            let baseline = ext.baseline.as_ref().ok_or(CausalError::Unsupported {
                message: "conflict re-shrink requires the original composition baseline; recompose the prior",
            })?;
            let ppc = PriorPredictiveCheck {
                n_sims: 200,
                seed: ctx.rng.master_seed(),
                ..PriorPredictiveCheck::new()
            };
            let (composed, summary) =
                compose_with_conflict_policy(&ext.sources, baseline, policy, prep, ctx, &ppc)
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

fn transfer_refusal(code: &'static str, scope: &str, message: &str) -> CausalError {
    antecedent_io::IoError::Refused {
        code,
        message: format!("mapped_posterior_transfer.{scope}: {message}"),
    }
    .into()
}

fn map_refuse(message: &str) -> CausalError {
    transfer_refusal(
        antecedent_core::reason_code!("quantity_semantics_mismatch"),
        "coefficient_map",
        message,
    )
}

fn meaning_refuse(message: &str) -> CausalError {
    transfer_refusal(
        antecedent_core::reason_code!("distribution_meaning_mismatch"),
        "parameter_posterior",
        message,
    )
}

fn variance_refuse(message: &str) -> CausalError {
    transfer_refusal(
        antecedent_core::reason_code!("prior_transfer_not_hydrated"),
        "residual_variance",
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
