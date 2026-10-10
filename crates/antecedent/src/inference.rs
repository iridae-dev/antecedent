//! Bayesian inference configuration for the facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::BayesianBackendKind;
use antecedent_estimate::{
    HydrateMapping, PreparedBayesianProblem, coefficient_covariance_from_draws,
    hydrate_prior_with_coefficient_covariance, is_temporal_coefficient_name,
};
use antecedent_io::PosteriorQuantityWire;
use antecedent_io::PriorMapping;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionIdentity, DrawAlignment,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_io::{decode_posterior_artifact, extract_prior_source_meta, read_and_migrate};
use antecedent_prob::{
    BayesLikelihood, ComposedPrior, ConflictSummary, ExternalPriorSource, PosteriorQuantityKind,
    PriorSet,
};
use antecedent_validate::{ConflictPolicy, PriorPredictiveCheck, compose_with_conflict_policy};

use crate::error::CausalError;
use crate::io::decode_causal_posterior_bytes;

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
}

impl BayesianConfig {
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
        }
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
    // A static coefficient transfer needs evidence that the source quantity was
    // an identity-link treatment contrast. Non-Gaussian effect draws live on
    // the response scale, so equal coefficient labels alone do not make their
    // slopes interchangeable with a Gaussian outcome model.
    if !names.iter().any(|name| is_temporal_coefficient_name(name)) {
        let (source, _) = decode_causal_posterior_bytes(bytes)?;
        if source.treatment_contrast.is_none() {
            return Err(CausalError::Unsupported {
                message: "static prior transfer requires an identity-link source contrast and matching coefficient meaning",
            });
        }
    }
    let baseline = PriorSet::weakly_informative(prep.design.ncols);
    let treatment_col = prep.design.treatment_column();
    require_lag_aware_transfer(bytes, &mapping, &names, treatment_col)?;
    Ok((
        Some(hydrate_prior_from_posterior_bytes(
            bytes,
            &mapping,
            &baseline,
            &names,
            treatment_col,
            None,
        )?),
        None,
    ))
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
    bytes: &[u8],
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
    let (wire, _) = decode_causal_posterior_bytes(bytes)?;
    let source_names: Vec<Option<&str>> = wire
        .quantities
        .iter()
        .filter_map(|q| match q {
            PosteriorQuantityWire::Coefficient { name, .. } => Some(name.as_deref()),
            _ => None,
        })
        .collect();
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

/// Hydrate a [`PriorSet`] from posterior artifact bytes under a [`HydrateMapping`].
///
/// Identical-subspace hydration reads the coefficient covariance from the
/// artifact's draws and carries it as a dense `V0`; an artifact without draws
/// (summary-only) hydrates the diagonal prior, as before.
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
    fallback_contrast: Option<f64>,
) -> Result<PriorSet, CausalError> {
    let (wire, draws) = decode_causal_posterior_bytes(bytes)?;
    let quantities = wire_quantities_to_kinds(&wire.quantities);
    let source_contrast = wire.treatment_contrast.or(fallback_contrast);
    let covariance = coefficient_covariance_from_draws(&quantities, &draws, wire.n_draws as usize)
        .map_err(CausalError::from)?;
    hydrate_prior_with_coefficient_covariance(
        mapping,
        &quantities,
        &wire.mean,
        &wire.sd,
        covariance.as_deref(),
        baseline,
        target_coef_names,
        treatment_col,
        source_contrast,
    )
    .map_err(CausalError::from)
}

/// Transfer an aligned parameter posterior into a known-variance Gaussian prior.
///
/// The caller supplies the source identity independently of the bytes and one
/// stable source-variable-ID to target-variable-ID pair per coefficient. Target
/// quantities are in design-column order. This route requires a complete
/// bijection, a constant residual-variance coordinate equal to the target's
/// known variance, and identical non-name quantity coordinates. The finite
/// artifact's full covariance is retained under the named permutation.
///
/// # Errors
/// Incompatible meaning, source identity, mapping, scale, or draw alignment.
#[allow(
    clippy::too_many_lines,
    reason = "keep the complete F19 refusal and transfer boundary together"
)]
pub fn mapped_posterior_transfer(
    bytes: &[u8],
    expected_source: &DistributionIdentity,
    source_to_target: &[(String, String)],
    target_quantities: &[ScientificQuantityWire],
    target_snapshot_id: &str,
    target_contract_id: &str,
    baseline: &PriorSet,
) -> Result<PriorSet, CausalError> {
    use antecedent_io::IoError;
    let map_refuse = |message: &str| IoError::Refused {
        code: antecedent_core::reason_code!("quantity_semantics_mismatch"),
        message: format!("mapped_posterior_transfer.coefficient_map: {message}"),
    };
    let meaning_refuse = |message: &str| IoError::Refused {
        code: antecedent_core::reason_code!("distribution_meaning_mismatch"),
        message: format!("mapped_posterior_transfer.parameter_posterior: {message}"),
    };
    let variance_refuse = |message: &str| IoError::Refused {
        code: antecedent_core::reason_code!("prior_transfer_not_hydrated"),
        message: format!("mapped_posterior_transfer.residual_variance: {message}"),
    };
    let source = DistributionArtifact::from_bytes(bytes, expected_source)?;
    if target_snapshot_id.trim().is_empty() || target_contract_id.trim().is_empty() {
        return Err(
            map_refuse("mapped transfer requires target snapshot and causal contract IDs").into()
        );
    }
    if source.semantic() != DistributionMeaningWire::ParameterPosterior
        || source.metadata().identity.alignment != DrawAlignment::Joint
        || source.metadata().weights.is_some()
        || source.metadata().supported.as_ref().is_some_and(|mask| mask.iter().any(|ok| !ok))
    {
        return Err(meaning_refuse(
            "mapped transfer requires complete, unweighted joint parameter-posterior draws",
        )
        .into());
    }
    let sigma2 = baseline.known_residual_variance().ok_or_else(|| {
        variance_refuse("mapped transfer requires a target known residual variance")
    })?;
    if source_to_target.len() != target_quantities.len()
        || baseline
            .gaussian_coefficients()
            .is_none_or(|prior| prior.len() != target_quantities.len())
        || target_quantities.is_empty()
    {
        return Err(map_refuse("mapped transfer requires a complete coefficient bijection").into());
    }
    let mut used_source = std::collections::HashSet::new();
    let mut used_target = std::collections::HashSet::new();
    let mut coefficient_columns = Vec::with_capacity(target_quantities.len());
    for (source_id, target_id) in source_to_target {
        if !used_source.insert(source_id) || !used_target.insert(target_id) {
            return Err(map_refuse("mapped transfer reuses a coefficient identity").into());
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
            return Err(map_refuse("mapped transfer needs unique stable coefficient IDs").into());
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
            return Err(map_refuse("mapped transfer coefficient coordinates differ").into());
        }
        coefficient_columns.push((source_col, target_col, source_id.clone(), target_id.clone()));
    }
    let residuals: Vec<_> = source
        .quantities()
        .iter()
        .enumerate()
        .filter(|(_, q)| q.functional_id == "residual_variance")
        .collect();
    if residuals.len() != 1 || source.quantities().len() != coefficient_columns.len() + 1 {
        return Err(variance_refuse(
            "mapped transfer requires one residual-variance coordinate and no extra parameters",
        )
        .into());
    }
    let (residual_col, residual_q) = residuals[0];
    if residual_q.population_id != target_quantities[0].population_id
        || residual_q.regime_id != target_quantities[0].regime_id
        || residual_q.horizon != target_quantities[0].horizon
        || residual_q.conditioning != target_quantities[0].conditioning
    {
        return Err(variance_refuse("mapped transfer residual-variance coordinate differs").into());
    }
    if source
        .draws()
        .chunks_exact(source.shape()[1])
        .any(|row| (row[residual_col] - sigma2).abs() > 1e-12 * sigma2.max(1.0))
    {
        return Err(variance_refuse(
            "mapped transfer residual variance differs from target known variance",
        )
        .into());
    }
    coefficient_columns.sort_by_key(|(_, target, _, _)| *target);
    let mut quantities = Vec::with_capacity(source.quantities().len());
    let mut mean = Vec::with_capacity(source.quantities().len());
    let mut sd = Vec::with_capacity(source.quantities().len());
    for (index, (_, _, source_id, _)) in coefficient_columns.iter().enumerate() {
        let source_col = coefficient_columns[index].0;
        quantities.push(PosteriorQuantityKind::Coefficient {
            index,
            name: Some(Arc::from(source_id.as_str())),
        });
        mean.push(source.mean(source_col)?);
        sd.push(source.covariance(source_col, source_col)?.sqrt());
    }
    quantities.push(PosteriorQuantityKind::ResidualVariance);
    mean.push(sigma2);
    sd.push(0.0);
    let n = coefficient_columns.len();
    let mut covariance = vec![0.0; n * n];
    for (i, (source_i, _, _, _)) in coefficient_columns.iter().enumerate() {
        for (j, (source_j, _, _, _)) in coefficient_columns.iter().enumerate() {
            covariance[i * n + j] = source.covariance(*source_i, *source_j)?;
        }
    }
    let names: Vec<Arc<str>> =
        target_quantities.iter().map(|q| Arc::from(q.variable_id.as_str())).collect();
    let mapping = HydrateMapping::NamedParameters {
        pairs: coefficient_columns
            .iter()
            .map(|(_, _, source, target)| (source.clone(), target.clone()))
            .collect(),
    };
    let mut prior = hydrate_prior_with_coefficient_covariance(
        &mapping,
        &quantities,
        &mean,
        &sd,
        Some(&covariance),
        baseline,
        &names,
        None,
        None,
    )
    .map_err(CausalError::from)?;
    prior.restrictions.push(antecedent_core::PriorAssumption {
        id: Arc::from("mapped_posterior_transfer"),
        description: Arc::from(format!(
            "source={} snapshot={} contract={}; target snapshot={} contract={}; complete stable-ID map={:?}; known residual variance={sigma2}",
            expected_source.source_id,
            expected_source.snapshot_id,
            expected_source.causal_contract_id,
            target_snapshot_id,
            target_contract_id,
            source_to_target,
        )),
    });
    Ok(prior)
}
