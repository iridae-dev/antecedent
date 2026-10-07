//! Checked adapter from `PriorCatalog` posterior sources to a future candidate-signal
//! likelihood.
//!
//! A historical posterior is a belief about the decision state; it is not a
//! distribution of what a future study will observe. This adapter pairs the compatible
//! prior sources, after their source-to-target maps, transport policy, weights, prior
//! strength and conflict shrinkage, with an explicit candidate signal family or
//! provider, and returns the [`DecisionPrior`] and [`SignalProvider`] the preposterior
//! analysis consumes. A coefficient prior or a structural unlock list alone is refused,
//! as is any incompatible artifact or any observation shared between a prior source
//! and the candidate study (or between two prior sources).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::ExternalRefusal;
use antecedent_io::{CompatibilityReport, PriorCatalog, TargetDesign};

use crate::preposterior::DecisionPrior;
use crate::signal::{NativeBinomialSignal, NativeGaussianMeanSignal, SignalProvider, make_refusal};

/// Largest number of equally weighted draws the adapter produces.
pub const MAX_PRIOR_RESOLUTION: usize = 1_000_000;

/// Native future-signal family the candidate study follows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NativeSignalFamily {
    /// Gaussian sample mean with known single-observation variance.
    GaussianMean {
        /// Variance `σ²` of one observation.
        noise_variance: f64,
    },
    /// Successes in Bernoulli trials whose probability is the decision state.
    Binomial,
}

/// What the caller supplies as the candidate's future signal.
#[derive(Clone)]
pub enum CandidateSignalSource {
    /// A native signal family.
    Native(NativeSignalFamily),
    /// An exact-request provider (for example an external signal).
    Provider(Arc<dyn SignalProvider>),
    /// Only a coefficient prior; not a signal distribution, always refused.
    CoefficientPriorOnly,
    /// Only a structural unlock list; not a signal distribution, always refused.
    UnlockListOnly(Vec<String>),
}

/// Map from a source posterior draw to the target decision-state coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceToTarget {
    /// The source quantity is the target quantity.
    Identity,
    /// `target = intercept + slope · source` with a nonzero finite slope.
    Affine {
        /// Intercept.
        intercept: f64,
        /// Slope.
        slope: f64,
    },
}

impl SourceToTarget {
    fn apply(self, value: f64) -> f64 {
        match self {
            Self::Identity => value,
            Self::Affine { intercept, slope } => intercept + slope * value,
        }
    }

    fn valid(self) -> bool {
        match self {
            Self::Identity => true,
            Self::Affine { intercept, slope } => {
                intercept.is_finite() && slope.is_finite() && slope.abs() > 0.0
            }
        }
    }
}

/// Declared policy under which source populations transport to the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportPolicyDecl {
    /// Stable policy identity.
    pub policy_id: String,
}

/// One historical posterior source, matched to a [`PriorCatalog`] entry by artifact id.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorSourceInput {
    /// Catalog artifact id.
    pub artifact_id: String,
    /// Population the source posterior describes.
    pub source_population: String,
    /// Equally weighted posterior draws of the source quantity.
    pub draws: Vec<f64>,
    /// Source weight (> 0).
    pub weight: f64,
    /// Prior strength in `(0, 1]` multiplying the weight (a power-prior exponent).
    pub prior_strength: f64,
    /// Conflict shrinkage in `[0, 1]` toward the pooled weighted mean.
    pub conflict_shrinkage: f64,
    /// Lineage identities of the source.
    pub lineage: Vec<String>,
    /// Identities of the observations the source posterior was fit on.
    pub observation_ids: Vec<String>,
    /// Map from the source quantity to the target decision state.
    pub source_to_target: SourceToTarget,
}

/// One adaptation request.
#[derive(Clone)]
pub struct PriorSignalRequest {
    /// Identity of the resulting prior (the signal request's `prior_id`).
    pub prior_id: String,
    /// Sources to pool.
    pub sources: Vec<PriorSourceInput>,
    /// Future candidate signal.
    pub signal: CandidateSignalSource,
    /// Target population.
    pub target_population: String,
    /// Transport policy, required when any source population differs from the target.
    pub transport: Option<TransportPolicyDecl>,
    /// Identities of existing observations the candidate study would reuse.
    pub candidate_observation_ids: Vec<String>,
    /// Number of equally weighted draws in the resulting prior.
    pub resolution: usize,
}

/// What the adapter checked and how the sources were combined.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorSignalDiagnostics {
    /// Source ids in canonical (sorted) order.
    pub source_ids: Vec<String>,
    /// Normalized effective weights (`weight · strength`), same order.
    pub effective_weights: Vec<f64>,
    /// Draws allocated to each source, same order.
    pub draws_allocated: Vec<usize>,
    /// Pooled weighted mean of the mapped source draws.
    pub pooled_mean: f64,
    /// Union of source lineage identities (sorted).
    pub lineage: Vec<String>,
    /// Number of distinct observation identities checked for reuse.
    pub observations_checked: usize,
    /// Observation identities shared with the candidate or between sources; empty
    /// for every successful adaptation.
    pub overlapping_observations: Vec<String>,
    /// Transport policy identity, when one was needed.
    pub transport_policy_id: Option<String>,
    /// BLAKE3 digest of the resulting prior draws and the combination inputs.
    pub source_digest: String,
}

/// A prior and the future-signal provider it is paired with.
#[derive(Clone)]
pub struct CandidateSignalPrior {
    /// Equally weighted draws of the decision state.
    pub prior: DecisionPrior<f64>,
    /// Provider of the future candidate signal likelihood.
    pub provider: Arc<dyn SignalProvider>,
    /// Prior identity.
    pub prior_id: String,
    /// Combination diagnostics.
    pub diagnostics: PriorSignalDiagnostics,
}

impl std::fmt::Debug for CandidateSignalPrior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandidateSignalPrior")
            .field("prior_id", &self.prior_id)
            .field("diagnostics", &self.diagnostics)
            .finish_non_exhaustive()
    }
}

/// Why the adapter refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PriorSignalError {
    /// Only a coefficient prior was supplied.
    CoefficientPriorNotSignal,
    /// Only a structural unlock list was supplied.
    UnlockListNotSignal(usize),
    /// No sources.
    NoSources,
    /// The same artifact id appears twice.
    DuplicateSource(String),
    /// A source id is absent from the catalog.
    ArtifactNotInCatalog(String),
    /// The catalog marks the source incompatible with the target.
    IncompatibleArtifact {
        /// Artifact id.
        artifact_id: String,
        /// Catalog verdict.
        reason: String,
    },
    /// A numeric or identity parameter is invalid.
    InvalidParameter(&'static str),
    /// A source declares no lineage.
    LineageMissing(String),
    /// A source declares no observation identities, so reuse cannot be excluded.
    ObservationLineageMissing(String),
    /// A source population differs from the target and no transport policy was declared.
    TransportPolicyRequired(String),
    /// A prior source and the candidate study share observations.
    ObservationReuse(Vec<String>),
    /// Two prior sources share observations (the evidence would be counted twice).
    PriorSourcesOverlap(Vec<String>),
    /// The requested resolution is zero or above [`MAX_PRIOR_RESOLUTION`].
    ResolutionInvalid,
}

impl PriorSignalError {
    /// Structured refusal under the `prior_signal` namespace.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let signal = antecedent_core::reason_code!("design_signal_invalid");
        let at = |code, detail: &str| make_refusal(code, "declare", detail);
        match self {
            Self::CoefficientPriorNotSignal => ExternalRefusal {
                remedy: Some("supply a future-signal likelihood or provider for the candidate"),
                ..at(signal, "prior_signal.coefficient_prior_not_signal")
            },
            Self::UnlockListNotSignal(count) => ExternalRefusal {
                offending: Some(count.to_string()),
                remedy: Some("supply a future-signal likelihood or provider for the candidate"),
                ..at(signal, "prior_signal.unlock_list_not_signal")
            },
            Self::NoSources => at(signal, "prior_signal.no_sources"),
            Self::DuplicateSource(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..at(signal, "prior_signal.duplicate_source")
            },
            Self::ArtifactNotInCatalog(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..at(signal, "prior_signal.artifact_not_in_catalog")
            },
            Self::IncompatibleArtifact { artifact_id, reason } => ExternalRefusal {
                offending: Some(artifact_id.clone()),
                supplied: Some(reason.clone()),
                ..at(signal, "prior_signal.incompatible_artifact")
            },
            Self::InvalidParameter(name) => ExternalRefusal {
                offending: Some((*name).to_owned()),
                ..at(
                    antecedent_core::reason_code!("invalid_argument"),
                    "prior_signal.invalid_parameter",
                )
            },
            Self::LineageMissing(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..at(signal, "prior_signal.lineage_missing")
            },
            Self::ObservationLineageMissing(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..at(signal, "prior_signal.observation_lineage_missing")
            },
            Self::TransportPolicyRequired(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..at(
                    antecedent_core::reason_code!("transport_policy_required"),
                    "prior_signal.transport_policy_required",
                )
            },
            Self::ObservationReuse(ids) => ExternalRefusal {
                offending: Some(ids.join(",")),
                remedy: Some("use only data not already summarized by the prior"),
                ..at(signal, "prior_signal.observation_reuse")
            },
            Self::PriorSourcesOverlap(ids) => ExternalRefusal {
                offending: Some(ids.join(",")),
                ..at(signal, "prior_signal.prior_sources_overlap")
            },
            Self::ResolutionInvalid => at(
                antecedent_core::reason_code!("invalid_argument"),
                "prior_signal.resolution_invalid",
            ),
        }
    }
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn resolve_signal(
    source: &CandidateSignalSource,
) -> Result<Arc<dyn SignalProvider>, PriorSignalError> {
    match source {
        CandidateSignalSource::CoefficientPriorOnly => {
            Err(PriorSignalError::CoefficientPriorNotSignal)
        }
        CandidateSignalSource::UnlockListOnly(list) => {
            Err(PriorSignalError::UnlockListNotSignal(list.len()))
        }
        CandidateSignalSource::Provider(provider) => Ok(Arc::clone(provider)),
        CandidateSignalSource::Native(NativeSignalFamily::Binomial) => {
            Ok(Arc::new(NativeBinomialSignal))
        }
        CandidateSignalSource::Native(NativeSignalFamily::GaussianMean { noise_variance }) => {
            let signal = NativeGaussianMeanSignal::new(*noise_variance)
                .map_err(|_| PriorSignalError::InvalidParameter("noise_variance"))?;
            Ok(Arc::new(signal))
        }
    }
}

fn check_catalog(
    catalog: &PriorCatalog,
    target: &TargetDesign,
    sources: &[&PriorSourceInput],
) -> Result<(), PriorSignalError> {
    let reports = catalog.filter_compatible(target);
    for source in sources {
        let Some(report) = reports.iter().find(|r| r.artifact_id() == source.artifact_id) else {
            return Err(PriorSignalError::ArtifactNotInCatalog(source.artifact_id.clone()));
        };
        let incompatible = |reason: String| PriorSignalError::IncompatibleArtifact {
            artifact_id: source.artifact_id.clone(),
            reason,
        };
        match report {
            CompatibilityReport::Compatible { .. } => {}
            // Draws are supplied separately and never hydrated from the artifact, so an
            // unnamed coefficient list is the only gap that does not matter here.
            CompatibilityReport::Partial { missing, .. }
                if missing.len() == 1 && missing[0] == "durable_coef_names" => {}
            CompatibilityReport::Partial { missing, .. } => {
                return Err(incompatible(format!("partial: missing {}", missing.join(","))));
            }
            CompatibilityReport::Rejected { reason, .. } => {
                return Err(incompatible(format!("rejected: {reason:?}")));
            }
        }
    }
    Ok(())
}

fn check_source(
    source: &PriorSourceInput,
    request: &PriorSignalRequest,
) -> Result<(), PriorSignalError> {
    let id = &source.artifact_id;
    if !(source.weight.is_finite() && source.weight > 0.0) {
        return Err(PriorSignalError::InvalidParameter("weight"));
    }
    if !(source.prior_strength.is_finite()
        && source.prior_strength > 0.0
        && source.prior_strength <= 1.0)
    {
        return Err(PriorSignalError::InvalidParameter("prior_strength"));
    }
    if !(source.conflict_shrinkage.is_finite() && (0.0..=1.0).contains(&source.conflict_shrinkage))
    {
        return Err(PriorSignalError::InvalidParameter("conflict_shrinkage"));
    }
    if source.draws.is_empty() || !source.draws.iter().all(|d| d.is_finite()) {
        return Err(PriorSignalError::InvalidParameter("draws"));
    }
    if !source.source_to_target.valid() {
        return Err(PriorSignalError::InvalidParameter("source_to_target"));
    }
    if source.lineage.is_empty() || source.lineage.iter().any(|l| blank(l)) {
        return Err(PriorSignalError::LineageMissing(id.clone()));
    }
    if source.observation_ids.is_empty() || source.observation_ids.iter().any(|o| blank(o)) {
        return Err(PriorSignalError::ObservationLineageMissing(id.clone()));
    }
    if blank(&source.source_population) {
        return Err(PriorSignalError::InvalidParameter("source_population"));
    }
    if source.source_population != request.target_population {
        match &request.transport {
            Some(policy) if !blank(&policy.policy_id) => {}
            _ => return Err(PriorSignalError::TransportPolicyRequired(id.clone())),
        }
    }
    Ok(())
}

/// Observation identities shared with the candidate study, then between sources.
fn check_overlap(
    sources: &[&PriorSourceInput],
    candidate_observations: &[String],
) -> Result<usize, PriorSignalError> {
    let candidate: BTreeSet<&str> = candidate_observations.iter().map(String::as_str).collect();
    let mut owners: BTreeMap<&str, usize> = BTreeMap::new();
    let mut reused = BTreeSet::new();
    let mut duplicated = BTreeSet::new();
    for source in sources {
        let own: BTreeSet<&str> = source.observation_ids.iter().map(String::as_str).collect();
        for observation in own {
            if candidate.contains(observation) {
                reused.insert(observation.to_owned());
            }
            *owners.entry(observation).or_insert(0) += 1;
        }
    }
    for (observation, count) in &owners {
        if *count > 1 {
            duplicated.insert((*observation).to_owned());
        }
    }
    if !reused.is_empty() {
        return Err(PriorSignalError::ObservationReuse(reused.into_iter().collect()));
    }
    if !duplicated.is_empty() {
        return Err(PriorSignalError::PriorSourcesOverlap(duplicated.into_iter().collect()));
    }
    Ok(owners.len())
}

/// Largest-remainder allocation of `total` draws in proportion to `weights`
/// (fractional ties go to the lower index).
fn allocate(weights: &[f64], total: usize) -> Vec<usize> {
    let sum: f64 = weights.iter().sum();
    let quotas: Vec<f64> = weights.iter().map(|w| w / sum * total as f64).collect();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "quotas are finite and in [0, total], so the floor is a valid count"
    )]
    let mut counts: Vec<usize> = quotas.iter().map(|q| q.floor() as usize).collect();
    let remaining = total.saturating_sub(counts.iter().sum::<usize>());
    let mut order: Vec<usize> = (0..quotas.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = quotas[a] - quotas[a].floor();
        let fb = quotas[b] - quotas[b].floor();
        fb.total_cmp(&fa).then(a.cmp(&b))
    });
    for &index in order.iter().take(remaining) {
        counts[index] += 1;
    }
    counts
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Adapt compatible prior sources and a future candidate signal into a prior and
/// provider for preposterior analysis.
///
/// Sources are processed in canonical artifact-id order, so the result does not
/// depend on the order they were supplied in. Each source's draws are mapped to the
/// target coordinate, shrunk toward the pooled weighted mean (`weight · strength`)
/// by its conflict shrinkage, and resampled by deterministic stratified quantiles
/// into `resolution` equally weighted draws in proportion to the effective weights.
///
/// # Errors
///
/// A signal that is only a coefficient prior or unlock list, an artifact missing
/// from or incompatible with the catalog, invalid weights, strength, shrinkage or
/// maps, missing lineage, a population change without transport policy, observation
/// reuse between a source and the candidate or between sources, or an invalid
/// resolution.
pub fn adapt_prior_to_signal(
    catalog: &PriorCatalog,
    target: &TargetDesign,
    request: &PriorSignalRequest,
) -> Result<CandidateSignalPrior, PriorSignalError> {
    let provider = resolve_signal(&request.signal)?;
    if blank(&request.prior_id) || blank(&request.target_population) {
        return Err(PriorSignalError::InvalidParameter("prior_id or target_population"));
    }
    if request.resolution == 0 || request.resolution > MAX_PRIOR_RESOLUTION {
        return Err(PriorSignalError::ResolutionInvalid);
    }
    if request.sources.is_empty() {
        return Err(PriorSignalError::NoSources);
    }
    let mut sources: Vec<&PriorSourceInput> = request.sources.iter().collect();
    sources.sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id));
    if let Some(pair) = sources.windows(2).find(|w| w[0].artifact_id == w[1].artifact_id) {
        return Err(PriorSignalError::DuplicateSource(pair[0].artifact_id.clone()));
    }
    check_catalog(catalog, target, &sources)?;
    for source in &sources {
        check_source(source, request)?;
    }
    let observations_checked = check_overlap(&sources, &request.candidate_observation_ids)?;
    let (draws, diagnostics) = combine(&sources, request, observations_checked);
    Ok(CandidateSignalPrior {
        prior: DecisionPrior::Draws(draws),
        provider,
        prior_id: request.prior_id.clone(),
        diagnostics,
    })
}

fn combine(
    sources: &[&PriorSourceInput],
    request: &PriorSignalRequest,
    observations_checked: usize,
) -> (Vec<f64>, PriorSignalDiagnostics) {
    let mapped: Vec<Vec<f64>> = sources
        .iter()
        .map(|s| s.draws.iter().map(|d| s.source_to_target.apply(*d)).collect())
        .collect();
    let raw: Vec<f64> = sources.iter().map(|s| s.weight * s.prior_strength).collect();
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
    let pooled: f64 = weights.iter().zip(&mapped).map(|(w, m)| w * mean(m)).sum();
    let counts = allocate(&weights, request.resolution);
    let mut draws = Vec::with_capacity(request.resolution);
    for ((source, values), &count) in sources.iter().zip(&mapped).zip(&counts) {
        let keep = 1.0 - source.conflict_shrinkage;
        let mut shrunk: Vec<f64> = values.iter().map(|v| pooled + keep * (v - pooled)).collect();
        shrunk.sort_by(f64::total_cmp);
        let len = shrunk.len();
        for j in 0..count {
            let index = ((2 * j + 1) * len / (2 * count)).min(len - 1);
            draws.push(shrunk[index]);
        }
    }
    let mut lineage: BTreeSet<String> = BTreeSet::new();
    for source in sources {
        lineage.extend(source.lineage.iter().cloned());
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"antecedent.prior_signal.v1");
    for (source, weight) in sources.iter().zip(&weights) {
        hasher.update(&(source.artifact_id.len() as u64).to_le_bytes());
        hasher.update(source.artifact_id.as_bytes());
        hasher.update(&weight.to_le_bytes());
    }
    for draw in &draws {
        hasher.update(&draw.to_le_bytes());
    }
    let diagnostics = PriorSignalDiagnostics {
        source_ids: sources.iter().map(|s| s.artifact_id.clone()).collect(),
        effective_weights: weights,
        draws_allocated: counts,
        pooled_mean: pooled,
        lineage: lineage.into_iter().collect(),
        observations_checked,
        overlapping_observations: Vec::new(),
        transport_policy_id: request.transport.as_ref().map(|t| t.policy_id.clone()),
        source_digest: hasher.finalize().to_hex().to_string(),
    };
    (draws, diagnostics)
}
