//! F11/F12/F14 facade: candidate signals, preposterior EVSI with a cost mapping, and the
//! durable `design_ranking_v1` ranking artifact.
//!
//! A caller declares the decision (terminal actions, an affine utility in a scalar state, a
//! finite prior or a conjugate normal belief), the candidate studies and, for each, a signal
//! provider: a native Gaussian-mean or binomial family, or an external provider with attested
//! values (a supplied likelihood updated natively, an externally computed posterior, or
//! externally computed per-branch decision values). [`evaluate`] binds each external
//! declaration to the exact signal request it is evaluated for (so its request fingerprint is
//! Antecedent's, never the supplier's), runs the existing `ReduceDecisionRegret` preposterior
//! path through [`evaluate_evsi`], and seals the result as a [`DesignRankingArtifactWire`].
//!
//! Provider trust is never upgraded here: an external signal is at most
//! `externally_attested` and is never reported as natively replayed. The cost-to-utility
//! mapping must be supplied explicitly; value and cost are otherwise reported separately.
//! Monte Carlo error coverage and rank guarantees are `unmeasured`, and an exact-integration
//! EVSI is a point-only value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CancellationToken, ExternalCapability, ExternalRefusal, ExternalScientificObject,
    ExternalTrustState, ProviderObjectIdentity, ScientificQuantity, SignalProviderContract,
};
use antecedent_design::design_ranking_artifact::{SealInputs, seal};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiError, EvsiReport, EvsiRequest, StudyCostSpec,
    evaluate_evsi,
};
use antecedent_design::ranking::{RankingError, StructuralCandidate};
use antecedent_design::signal::{
    ExternalDecisionValues, ExternalLaw, ExternalSignal, ExternalSignalBody, NativeBinomialSignal,
    NativeGaussianMeanSignal, SignalError, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    SamplingPlan, StudyCost,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::Deserialize;

pub use antecedent_design::design_ranking_artifact::{
    CandidateWire, ConsumeExpectation, ConsumedCandidate, ConsumedDesignRanking,
    DESIGN_RANKING_ARTIFACT_KIND, DESIGN_RANKING_ARTIFACT_VERSION, DESIGN_RANKING_CALIBRATION,
    DesignRankingArtifactError, DesignRankingArtifactWire, ReplayKind, consume, consume_wire,
};
pub use antecedent_design::ranking::{DesignRanking, DesignRankingBasis};
pub use antecedent_design::signal::{SignalTrustLabel, SignalUpdateMode};

fn default_mc_error_tolerance() -> f64 {
    1e-3
}

fn default_tie_tolerance() -> f64 {
    1e-12
}

fn default_max_candidates() -> usize {
    antecedent_design::design_ranking_artifact::MAX_RANKING_CANDIDATES
}

/// The belief the decision is made under.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PriorDeclWire {
    /// Equally weighted draws of the scalar decision state.
    Draws {
        /// Prior draws.
        states: Vec<f64>,
    },
    /// Conjugate normal belief; needs a Gaussian-mean signal and no constraints.
    Normal {
        /// Prior mean.
        mean: f64,
        /// Prior variance.
        variance: f64,
    },
}

/// The decision problem: terminal actions with `U(a, theta) = intercept_a + slope_a * theta`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionDeclWire {
    /// Identity of the decision contract the problem implements.
    pub contract_identity: String,
    /// Utility unit.
    pub utility_unit: String,
    /// Terminal action identities.
    pub action_ids: Vec<String>,
    /// Per-action intercepts.
    pub intercepts: Vec<f64>,
    /// Per-action slopes.
    pub slopes: Vec<f64>,
    /// Belief.
    pub prior: PriorDeclWire,
}

/// Declared study cost.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostDeclWire {
    /// Amount.
    pub amount: f64,
    /// Unit.
    pub unit: String,
}

/// Cost-to-utility mapping.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostMapDeclWire {
    /// Unit study costs are declared in.
    pub cost_unit: String,
    /// Utility unit of the decision.
    pub utility_unit: String,
    /// Utility per unit cost.
    pub utility_per_cost: f64,
}

/// Exact scientific request declared for one candidate's signal.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalDeclWire {
    /// Prior identity the update consumes.
    pub prior_id: String,
    /// State coordinate.
    pub state_quantity: ScientificQuantityWire,
    /// Observation coordinate.
    pub observation_quantity: ScientificQuantityWire,
    /// RNG seed.
    pub rng_seed: u64,
    /// Evidence lineage identities.
    pub evidence_lineage: Vec<String>,
    /// Conditional-independence assumption identity.
    pub conditional_independence: String,
    /// Largest sample size (default: the engine's limit).
    #[serde(default)]
    pub max_sample_size: Option<u64>,
    /// Largest enumerated support (default: the engine's limit).
    #[serde(default)]
    pub max_support: Option<usize>,
}

/// The law an external provider attests.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ExternalLawDeclWire {
    /// Predictive law over the prior draws, updated natively.
    Likelihood {
        /// Prior draws the law is defined on.
        states: Vec<f64>,
        /// Possible observation values.
        statistics: Vec<f64>,
        /// `probabilities[y][k] = P(statistic_y | state_k)`.
        probabilities: Vec<Vec<f64>>,
    },
    /// Externally computed posterior over the equally weighted prior draws.
    Posterior {
        /// Prior draws.
        states: Vec<f64>,
        /// Possible observation values.
        statistics: Vec<f64>,
        /// Predictive probability of each observation.
        predictive: Vec<f64>,
        /// `posterior[y][k] = P(state_k | statistic_y)`.
        posterior: Vec<Vec<f64>>,
    },
    /// Externally computed per-branch decision values.
    DecisionValues {
        /// Probability of each branch.
        branch_probabilities: Vec<f64>,
        /// Action identities naming the value columns.
        action_ids: Vec<String>,
        /// `values[branch][action]`.
        values: Vec<Vec<f64>>,
    },
}

/// The provider declared for a candidate.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderDeclWire {
    /// Native Gaussian sample mean.
    GaussianMean {
        /// Per-observation noise variance.
        noise_variance: f64,
    },
    /// Native binomial.
    Binomial,
    /// External provider with attested values; never native.
    External(Box<ExternalDeclWire>),
}

/// An external provider's attested object. The request fingerprint is bound by Antecedent
/// to the exact request evaluated; the `attested_*` fields state what the supplier's object
/// was produced for (default: the candidate's own), so a supplier answering another
/// candidate, prior or sample size is refused rather than silently rebound.
#[derive(Clone, Debug, Deserialize)]
pub struct ExternalDeclWire {
    /// Provider service or model identity.
    pub provider_id: String,
    /// Object identity within the provider.
    pub object_id: String,
    /// Provider version.
    pub version_id: String,
    /// Data or model snapshot.
    pub snapshot_id: String,
    /// Party attesting the values.
    pub attestor: String,
    /// The attested law.
    pub law: ExternalLawDeclWire,
    /// Candidate the supplier's object was produced for.
    #[serde(default)]
    pub attested_candidate_id: Option<String>,
    /// Prior the supplier's object was produced for.
    #[serde(default)]
    pub attested_prior_id: Option<String>,
    /// Sample size the supplier's law was produced for.
    #[serde(default)]
    pub attested_sample_size: Option<u64>,
}

/// One candidate study.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateDeclWire {
    /// Stable semantic identity.
    pub semantic_id: String,
    /// Observations the candidate collects.
    pub sample_size: u64,
    /// Declared cost.
    pub cost: CostDeclWire,
    /// Exact signal request.
    pub signal: SignalDeclWire,
    /// Signal provider.
    pub provider: ProviderDeclWire,
    /// Identities of existing observations the study would reuse.
    #[serde(default)]
    pub reused_observation_ids: Vec<String>,
}

/// Monte Carlo ranking configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonteCarloDeclWire {
    /// Minimum batches before an early stop.
    pub min_batches: u32,
    /// Maximum batches.
    pub max_batches: u32,
    /// Batch size.
    pub batch_size: u32,
    /// Rank uncertainty threshold.
    pub rank_uncertainty_threshold: f64,
}

/// A declared EVSI/ranking evaluation.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignRankingRequestWire {
    /// Decision problem.
    pub decision: DecisionDeclWire,
    /// Candidates, in any order.
    pub candidates: Vec<CandidateDeclWire>,
    /// Cost-to-utility mapping; absent means value and cost are reported separately.
    #[serde(default)]
    pub cost_map: Option<CostMapDeclWire>,
    /// Refuse instead of reporting separately when no mapping is supplied.
    #[serde(default)]
    pub require_net_value: bool,
    /// Observation identities the prior already summarizes.
    #[serde(default)]
    pub prior_observation_ids: Vec<String>,
    /// Digests of the source distributions behind the prior and the external laws.
    #[serde(default)]
    pub source_digests: Vec<String>,
    /// RNG seed of the evaluation.
    pub rng_seed: u64,
    /// Standard error at or below which a Monte Carlo estimate is converged.
    #[serde(default = "default_mc_error_tolerance")]
    pub mc_error_tolerance: f64,
    /// Gap at or below which two candidates are tied.
    #[serde(default = "default_tie_tolerance")]
    pub tie_tolerance: f64,
    /// Search bound on candidates evaluated.
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,
    /// Monte Carlo configuration (default: the engine's).
    #[serde(default)]
    pub monte_carlo: Option<MonteCarloDeclWire>,
}

/// One candidate of the structural (no probabilistic model) ordering.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralCandidateDeclWire {
    /// Stable semantic identity.
    pub semantic_id: String,
    /// The planner verified the candidate sufficient for the failed query.
    pub verified_sufficient: bool,
    /// Cost units.
    pub cost_units: u64,
    /// Sample budget.
    pub sample_budget: u64,
}

/// Why a declared ranking cannot be evaluated or consumed.
#[derive(Debug)]
#[non_exhaustive]
pub enum DesignRankingError {
    /// The declaration is malformed.
    InvalidRequest(String),
    /// The EVSI evaluation refused.
    Evsi(EvsiError),
    /// The ranking or artifact refused.
    Artifact(DesignRankingArtifactError),
    /// The structural ordering refused.
    Ranking(RankingError),
}

impl From<EvsiError> for DesignRankingError {
    fn from(error: EvsiError) -> Self {
        Self::Evsi(error)
    }
}

impl From<SignalError> for DesignRankingError {
    fn from(error: SignalError) -> Self {
        Self::Evsi(EvsiError::Signal(error))
    }
}

impl From<DesignRankingArtifactError> for DesignRankingError {
    fn from(error: DesignRankingArtifactError) -> Self {
        Self::Artifact(error)
    }
}

impl From<RankingError> for DesignRankingError {
    fn from(error: RankingError) -> Self {
        Self::Ranking(error)
    }
}

impl DesignRankingError {
    /// Structured refusal, or `None` for a serialization failure (corruption, truncation).
    #[must_use]
    pub fn refusal(&self) -> Option<ExternalRefusal> {
        match self {
            Self::InvalidRequest(message) => Some(ExternalRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                stage: "declare",
                detail: "design_ranking.invalid_request".to_owned(),
                offending: Some(message.clone()),
                expected: None,
                supplied: None,
                capability: None,
                remedy: None,
            }),
            Self::Evsi(error) => Some(error.to_refusal()),
            Self::Artifact(error) => error.refusal(),
            Self::Ranking(error) => Some(error.to_refusal()),
        }
    }
}

impl std::fmt::Display for DesignRankingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) => write!(f, "design_ranking.invalid_request: {message}"),
            Self::Evsi(error) => write!(f, "{error:?}"),
            Self::Artifact(error) => write!(f, "{error}"),
            Self::Ranking(error) => write!(f, "{error:?}"),
        }
    }
}

impl std::error::Error for DesignRankingError {}

fn invalid(message: impl Into<String>) -> DesignRankingError {
    DesignRankingError::InvalidRequest(message.into())
}

fn quantity(wire: &ScientificQuantityWire) -> Result<ScientificQuantity, DesignRankingError> {
    ScientificQuantity::try_from(wire.clone()).map_err(invalid)
}

fn signal_request(candidate: &CandidateDeclWire) -> Result<SignalRequest, DesignRankingError> {
    let defaults = SignalLimits::default();
    let decl = &candidate.signal;
    Ok(SignalRequest {
        candidate_id: candidate.semantic_id.clone(),
        prior_id: decl.prior_id.clone(),
        state_quantity: quantity(&decl.state_quantity)?,
        observation_quantity: quantity(&decl.observation_quantity)?,
        sample_size: candidate.sample_size,
        rng_seed: decl.rng_seed,
        evidence_lineage: decl.evidence_lineage.clone(),
        conditional_independence: decl.conditional_independence.clone(),
        limits: SignalLimits {
            max_sample_size: decl.max_sample_size.unwrap_or(defaults.max_sample_size),
            max_support: decl.max_support.unwrap_or(defaults.max_support),
        },
    })
}

fn external_provider(
    request: &SignalRequest,
    declared: &ExternalDeclWire,
) -> Result<Arc<dyn SignalProvider>, DesignRankingError> {
    let (capabilities, law) = match &declared.law {
        ExternalLawDeclWire::Likelihood { states, statistics, probabilities } => (
            vec![
                ExternalCapability::Sample,
                ExternalCapability::Factor,
                ExternalCapability::Update,
            ],
            ExternalLaw::Likelihood {
                states: states.clone(),
                statistics: statistics.clone(),
                probabilities: probabilities.clone(),
            },
        ),
        ExternalLawDeclWire::Posterior { states, statistics, predictive, posterior } => (
            vec![ExternalCapability::Sample, ExternalCapability::Update],
            ExternalLaw::Posterior {
                states: states.clone(),
                statistics: statistics.clone(),
                predictive: predictive.clone(),
                posterior: posterior.clone(),
            },
        ),
        ExternalLawDeclWire::DecisionValues { branch_probabilities, action_ids, values } => (
            vec![ExternalCapability::Sample, ExternalCapability::Update],
            ExternalLaw::DecisionValues(ExternalDecisionValues {
                branch_probabilities: branch_probabilities.clone(),
                action_ids: action_ids.clone(),
                values: values.clone(),
            }),
        ),
    };
    // The attested payload is bound to the exact request it is evaluated for: the request
    // fingerprint is computed here, never supplied by the provider.
    let object = ExternalScientificObject::Signal(SignalProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: declared.provider_id.clone(),
            object_id: declared.object_id.clone(),
            version_id: declared.version_id.clone(),
            snapshot_id: declared.snapshot_id.clone(),
            request_id: request.fingerprint(),
        },
        candidate_id: declared
            .attested_candidate_id
            .clone()
            .unwrap_or_else(|| request.candidate_id.clone()),
        prior_id: declared.attested_prior_id.clone().unwrap_or_else(|| request.prior_id.clone()),
        observation: request.observation_quantity.clone(),
        capabilities,
    });
    let trust = ExternalTrustState::attest(&object, &declared.attestor)
        .map_err(|e| invalid(format!("external provider declaration: {e:?}")))?;
    let body = ExternalSignalBody {
        sample_size: declared.attested_sample_size.unwrap_or(request.sample_size),
        state_quantity: request.state_quantity.clone(),
        observation_quantity: request.observation_quantity.clone(),
        law,
    };
    Ok(Arc::new(ExternalSignal::new(object, trust, body)?))
}

fn provider_for(
    candidate: &CandidateDeclWire,
    request: &SignalRequest,
) -> Result<Arc<dyn SignalProvider>, DesignRankingError> {
    match &candidate.provider {
        ProviderDeclWire::GaussianMean { noise_variance } => Ok(Arc::new(
            NativeGaussianMeanSignal::new(*noise_variance)
                .map_err(|e| invalid(format!("gaussian_mean: {e}")))?,
        )),
        ProviderDeclWire::Binomial => Ok(Arc::new(NativeBinomialSignal)),
        ProviderDeclWire::External(declared) => external_provider(request, declared),
    }
}

/// The prepared problem, belief and request of a declaration.
struct Prepared {
    problem: DecisionProblem<usize, f64>,
    prior: DecisionPrior<f64>,
    request: EvsiRequest,
}

fn prepare(wire: &DesignRankingRequestWire) -> Result<Prepared, DesignRankingError> {
    let decision = &wire.decision;
    let utility = AffineUtility::new(decision.intercepts.clone(), decision.slopes.clone())
        .map_err(|e| invalid(format!("utility: {e}")))?;
    let actions: Vec<usize> = (0..decision.action_ids.len()).collect();
    let problem: DecisionProblem<usize, f64> =
        DecisionProblem::new(actions, Arc::new(utility), vec![]);
    let prior = match &decision.prior {
        PriorDeclWire::Draws { states } => DecisionPrior::Draws(states.clone()),
        PriorDeclWire::Normal { mean, variance } => {
            DecisionPrior::Normal { mean: *mean, variance: *variance }
        }
    };
    let mut candidates = Vec::with_capacity(wire.candidates.len());
    for declared in &wire.candidates {
        let signal_request = signal_request(declared)?;
        let provider = provider_for(declared, &signal_request)?;
        candidates.push(EvsiCandidate {
            semantic_id: declared.semantic_id.clone(),
            design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
                additional_samples: declared.sample_size,
                cost: DesignCost::zero(),
                tag: 0,
            }),
            signal_request,
            provider,
            cost: StudyCostSpec { amount: declared.cost.amount, unit: declared.cost.unit.clone() },
            reused_observation_ids: declared.reused_observation_ids.clone(),
        });
    }
    let rank_config =
        wire.monte_carlo.as_ref().map_or_else(DesignRankConfig::default, |mc| DesignRankConfig {
            min_batches: mc.min_batches,
            max_batches: mc.max_batches,
            batch_size: mc.batch_size,
            rank_uncertainty_threshold: mc.rank_uncertainty_threshold,
        });
    let request = EvsiRequest {
        decision_contract_identity: decision.contract_identity.clone(),
        utility_unit: decision.utility_unit.clone(),
        action_ids: decision.action_ids.clone(),
        candidates,
        cost_map: wire.cost_map.as_ref().map(|m| CostToUtilityMap {
            cost_unit: m.cost_unit.clone(),
            utility_unit: m.utility_unit.clone(),
            utility_per_cost: m.utility_per_cost,
        }),
        require_net_value: wire.require_net_value,
        prior_observation_ids: wire.prior_observation_ids.clone(),
        rank_config,
        rng_seed: wire.rng_seed,
        mc_error_tolerance: wire.mc_error_tolerance,
        tie_tolerance: wire.tie_tolerance,
        max_candidates: wire.max_candidates,
    };
    Ok(Prepared { problem, prior, request })
}

/// A finished, sealed EVSI ranking.
#[derive(Clone, Debug)]
pub struct DesignRankingEvaluation {
    report: EvsiReport,
    ranking: DesignRanking,
    wire: DesignRankingArtifactWire,
}

impl DesignRankingEvaluation {
    /// The preposterior EVSI report (per-candidate values, error, trust, overlap).
    #[must_use]
    pub fn report(&self) -> &EvsiReport {
        &self.report
    }

    /// The order-invariant in-memory ranking value.
    #[must_use]
    pub fn ranking(&self) -> &DesignRanking {
        &self.ranking
    }

    /// The artifact body: everything an independent reader needs to inspect or recompute.
    #[must_use]
    pub fn artifact(&self) -> &DesignRankingArtifactWire {
        &self.wire
    }

    /// Artifact digest; unchanged by the order candidates were declared in.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.wire.digest
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// A blank artifact id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, DesignRankingArtifactError> {
        self.wire.to_bytes(artifact_id).map_err(DesignRankingArtifactError::Io)
    }
}

/// Evaluate every declared candidate's EVSI through the existing decision-regret path and
/// seal the ranking.
///
/// # Errors
/// A malformed declaration, a refused signal or provider (wrong candidate, prior, sample
/// size, request fingerprint or incoherent update), source overlap, incompatible cost units
/// or a changed action set, a bound violation, or an artifact that cannot be sealed.
pub fn evaluate(
    wire: &DesignRankingRequestWire,
) -> Result<DesignRankingEvaluation, DesignRankingError> {
    let prepared = prepare(wire)?;
    let report = evaluate_evsi(
        &prepared.problem,
        &prepared.prior,
        &prepared.request,
        &CancellationToken::new(),
    )?;
    let artifact = seal(&SealInputs {
        problem: &prepared.problem,
        prior: &prepared.prior,
        request: &prepared.request,
        report: &report,
        source_digests: &wire.source_digests,
    })?;
    let ranking = DesignRanking::from_evsi(&report, &wire.source_digests)?;
    Ok(DesignRankingEvaluation { report, ranking, wire: artifact })
}

/// The 2.2 ordering when no probabilistic model is licensed: verified structural
/// sufficiency, then fewer cost units, then smaller sample budget, then semantic id.
///
/// # Errors
/// No candidates, or a repeated semantic id.
pub fn rank_structural(
    candidates: &[StructuralCandidateDeclWire],
) -> Result<DesignRanking, DesignRankingError> {
    let declared: Vec<StructuralCandidate> = candidates
        .iter()
        .map(|c| StructuralCandidate {
            semantic_id: c.semantic_id.clone(),
            verified_sufficient: c.verified_sufficient,
            cost: StudyCost { units: c.cost_units, sample_budget: c.sample_budget },
        })
        .collect();
    Ok(DesignRanking::from_structural(&declared)?)
}
