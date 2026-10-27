//! Design-ranking facade: candidate signals, preposterior EVSI with a cost mapping, and the
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
//! [`evaluate_for_study`] binds the evaluation to a lowered program: the study is compiled
//! for the request through [`Study::compile_for_evsi`] first, which refuses a study whose
//! statistical program publishes no parameter posterior and a Monte Carlo store the
//! admission pool cannot hold, and the evaluation carries that program's identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    ExternalCapability, ExternalRefusal, ExternalScientificObject, ExternalTrustState,
    ProviderObjectIdentity, ScientificQuantity, SignalProviderContract,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::design_ranking_artifact::SealInputs;
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiError, EvsiReport, EvsiRequest, StudyCostSpec,
    evaluate_evsi_with_context,
};
use antecedent_design::ranking::{RankingError, StructuralCandidate};
use antecedent_design::signal::{
    ExternalDecisionValues, ExternalLaw, ExternalSignal, ExternalSignalBody, NativeBinomialSignal,
    NativeGaussianMeanSignal, SignalError, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    GridUtility, SamplingPlan, StudyCost,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use serde::Deserialize;

use crate::planner::PhysicalExecutionPlan;
use crate::{CausalError, CompileCause, Study};

pub use antecedent_design::design_ranking_artifact::{
    ConsumeExpectation, ConsumedCandidate, ConsumedDesignRanking, DESIGN_RANKING_ARTIFACT_KIND,
    DESIGN_RANKING_ARTIFACT_VERSION, DESIGN_RANKING_CALIBRATION, DesignRankingArtifactError,
    DesignRankingArtifactWire, EvsiCandidateWire, ReplayKind, consume, consume_wire,
};
pub use antecedent_design::ranking::{DesignRankingBasis, RankingRecord};
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
    /// Per-action intercepts (empty when `utility_table` is declared).
    #[serde(default)]
    pub intercepts: Vec<f64>,
    /// Per-action slopes (empty when `utility_table` is declared).
    #[serde(default)]
    pub slopes: Vec<f64>,
    /// The utilities as an explicit table over the states the decision was evaluated
    /// at, instead of an affine function of the state. Needs a prior of draws that
    /// all lie on the table's states; nothing is interpolated between them.
    #[serde(default)]
    pub utility_table: Option<UtilityTableDeclWire>,
    /// Belief.
    pub prior: PriorDeclWire,
}

/// Utilities tabulated over the states a decision was evaluated at.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UtilityTableDeclWire {
    /// The evaluated states, distinct.
    pub states: Vec<f64>,
    /// `rows[action][state]`, one row per action in `action_ids` order.
    pub rows: Vec<Vec<f64>>,
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
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DesignRankingError {
    /// The declaration is malformed.
    #[error("design_ranking.invalid_request: {0}")]
    InvalidRequest(String),
    /// The EVSI evaluation refused.
    #[error(transparent)]
    Evsi(#[from] EvsiError),
    /// The ranking or artifact refused.
    #[error(transparent)]
    Artifact(#[from] DesignRankingArtifactError),
    /// The structural ordering refused.
    #[error(transparent)]
    Ranking(#[from] RankingError),
    /// Compiling the study for the request refused.
    #[error(transparent)]
    Compile(CausalError),
}

impl From<SignalError> for DesignRankingError {
    fn from(error: SignalError) -> Self {
        Self::Evsi(EvsiError::Signal(error))
    }
}

impl DesignRankingError {
    /// Structured refusal, or `None` for a serialization failure (corruption, truncation)
    /// and for a compile refusal other than a decision-carrier refusal (read it through
    /// `Display`).
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
            Self::Compile(error) => compile_refusal(error),
        }
    }
}

/// The decision-carrier refusal of a compile, as a bind-stage refusal.
fn compile_refusal(error: &CausalError) -> Option<ExternalRefusal> {
    let CausalError::Rejected(failure) = error.peeled() else {
        return None;
    };
    let CompileCause::Decision(refusal) = failure.cause() else {
        return None;
    };
    Some(
        ExternalRefusal::new(refusal.code, "bind", refusal.detail)
            .with_expected(refusal.demanded.as_str())
            .with_supplied(refusal.realization)
            .with_remedy(refusal.next_attempt.sentence()),
    )
}

fn invalid(message: impl Into<String>) -> DesignRankingError {
    DesignRankingError::InvalidRequest(message.into())
}

fn quantity(wire: ScientificQuantityWire) -> Result<ScientificQuantity, DesignRankingError> {
    ScientificQuantity::try_from(wire).map_err(invalid)
}

impl ExternalLawDeclWire {
    /// The operations the declared law supplies.
    fn capabilities(&self) -> Vec<ExternalCapability> {
        match self {
            Self::Likelihood { .. } => vec![
                ExternalCapability::Sample,
                ExternalCapability::Factor,
                ExternalCapability::Update,
            ],
            Self::Posterior { .. } | Self::DecisionValues { .. } => {
                vec![ExternalCapability::Sample, ExternalCapability::Update]
            }
        }
    }
}

impl From<ExternalLawDeclWire> for ExternalLaw {
    fn from(wire: ExternalLawDeclWire) -> Self {
        match wire {
            ExternalLawDeclWire::Likelihood { states, statistics, probabilities } => {
                Self::Likelihood { states, statistics, probabilities }
            }
            ExternalLawDeclWire::Posterior { states, statistics, predictive, posterior } => {
                Self::Posterior { states, statistics, predictive, posterior }
            }
            ExternalLawDeclWire::DecisionValues { branch_probabilities, action_ids, values } => {
                Self::DecisionValues(ExternalDecisionValues {
                    branch_probabilities,
                    action_ids,
                    values,
                })
            }
        }
    }
}

fn external_provider(
    request: &SignalRequest,
    declared: ExternalDeclWire,
) -> Result<Arc<dyn SignalProvider>, DesignRankingError> {
    let ExternalDeclWire {
        provider_id,
        object_id,
        version_id,
        snapshot_id,
        attestor,
        law,
        attested_candidate_id,
        attested_prior_id,
        attested_sample_size,
    } = declared;
    let capabilities = law.capabilities();
    // The attested payload is bound to the exact request it is evaluated for: the request
    // fingerprint is computed here, never supplied by the provider.
    let object = ExternalScientificObject::Signal(SignalProviderContract {
        identity: ProviderObjectIdentity {
            provider_id,
            object_id,
            version_id,
            snapshot_id,
            request_id: request.fingerprint(),
        },
        candidate_id: attested_candidate_id.unwrap_or_else(|| request.candidate_id.clone()),
        prior_id: attested_prior_id.unwrap_or_else(|| request.prior_id.clone()),
        observation: request.observation_quantity.clone(),
        capabilities,
    });
    let trust = ExternalTrustState::attest(&object, &attestor)
        .map_err(|e| invalid(format!("external provider declaration: {e:?}")))?;
    let body = ExternalSignalBody {
        sample_size: attested_sample_size.unwrap_or(request.sample_size),
        state_quantity: request.state_quantity.clone(),
        observation_quantity: request.observation_quantity.clone(),
        law: ExternalLaw::from(law),
    };
    Ok(Arc::new(ExternalSignal::new(object, trust, body)?))
}

fn provider_for(
    provider: ProviderDeclWire,
    request: &SignalRequest,
) -> Result<Arc<dyn SignalProvider>, DesignRankingError> {
    match provider {
        ProviderDeclWire::GaussianMean { noise_variance } => Ok(Arc::new(
            NativeGaussianMeanSignal::new(noise_variance)
                .map_err(|e| invalid(format!("gaussian_mean: {e}")))?,
        )),
        ProviderDeclWire::Binomial => Ok(Arc::new(NativeBinomialSignal)),
        ProviderDeclWire::External(declared) => external_provider(request, *declared),
    }
}

impl TryFrom<CandidateDeclWire> for EvsiCandidate {
    type Error = DesignRankingError;

    fn try_from(declared: CandidateDeclWire) -> Result<Self, Self::Error> {
        let CandidateDeclWire {
            semantic_id,
            sample_size,
            cost,
            signal,
            provider,
            reused_observation_ids,
        } = declared;
        let defaults = SignalLimits::default();
        let signal_request = SignalRequest {
            candidate_id: semantic_id.clone(),
            prior_id: signal.prior_id,
            state_quantity: quantity(signal.state_quantity)?,
            observation_quantity: quantity(signal.observation_quantity)?,
            sample_size,
            rng_seed: signal.rng_seed,
            evidence_lineage: signal.evidence_lineage,
            conditional_independence: signal.conditional_independence,
            limits: SignalLimits {
                max_sample_size: signal.max_sample_size.unwrap_or(defaults.max_sample_size),
                max_support: signal.max_support.unwrap_or(defaults.max_support),
            },
        };
        let provider = provider_for(provider, &signal_request)?;
        Ok(Self {
            semantic_id,
            design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
                additional_samples: sample_size,
                cost: DesignCost::zero(),
                tag: 0,
            }),
            signal_request,
            provider,
            cost: StudyCostSpec { amount: cost.amount, unit: cost.unit },
            reused_observation_ids,
        })
    }
}

/// The decision problem and belief of a declaration, with the identities the request carries.
struct Decision {
    problem: DecisionProblem<usize, f64>,
    prior: DecisionPrior<f64>,
    contract_identity: String,
    utility_unit: String,
    action_ids: Vec<String>,
}

impl TryFrom<DecisionDeclWire> for Decision {
    type Error = DesignRankingError;

    fn try_from(decision: DecisionDeclWire) -> Result<Self, Self::Error> {
        let DecisionDeclWire {
            contract_identity,
            utility_unit,
            action_ids,
            intercepts,
            slopes,
            utility_table,
            prior,
        } = decision;
        let actions: Vec<usize> = (0..action_ids.len()).collect();
        let problem: DecisionProblem<usize, f64> = if let Some(table) = utility_table {
            if !intercepts.is_empty() || !slopes.is_empty() {
                return Err(invalid(
                    "declare the utility as an affine function or as a table, not both",
                ));
            }
            if !matches!(prior, PriorDeclWire::Draws { .. }) {
                return Err(invalid("a tabulated utility needs a prior of draws on its states"));
            }
            if table.rows.len() != action_ids.len() {
                return Err(invalid("a utility table needs one row per action"));
            }
            let utility = GridUtility::new(table.states, table.rows)
                .map_err(|e| invalid(format!("utility table: {e}")))?;
            DecisionProblem::new(actions, Arc::new(utility), vec![])
        } else {
            let utility = AffineUtility::new(intercepts, slopes)
                .map_err(|e| invalid(format!("utility: {e}")))?;
            DecisionProblem::new(actions, Arc::new(utility), vec![])
        };
        let prior = match prior {
            PriorDeclWire::Draws { states } => DecisionPrior::Draws(states),
            PriorDeclWire::Normal { mean, variance } => DecisionPrior::Normal { mean, variance },
        };
        Ok(Self { problem, prior, contract_identity, utility_unit, action_ids })
    }
}

/// The prepared problem, belief and request of a declaration.
struct Prepared {
    problem: DecisionProblem<usize, f64>,
    prior: DecisionPrior<f64>,
    request: EvsiRequest,
    source_digests: Vec<String>,
}

impl TryFrom<DesignRankingRequestWire> for Prepared {
    type Error = DesignRankingError;

    fn try_from(wire: DesignRankingRequestWire) -> Result<Self, Self::Error> {
        let DesignRankingRequestWire {
            decision,
            candidates,
            cost_map,
            require_net_value,
            prior_observation_ids,
            source_digests,
            rng_seed,
            mc_error_tolerance,
            tie_tolerance,
            max_candidates,
            monte_carlo,
        } = wire;
        let Decision { problem, prior, contract_identity, utility_unit, action_ids } =
            Decision::try_from(decision)?;
        let candidates =
            candidates.into_iter().map(EvsiCandidate::try_from).collect::<Result<Vec<_>, _>>()?;
        let rank_config =
            monte_carlo.map_or_else(DesignRankConfig::default, |mc| DesignRankConfig {
                min_batches: mc.min_batches,
                max_batches: mc.max_batches,
                batch_size: mc.batch_size,
                rank_uncertainty_threshold: mc.rank_uncertainty_threshold,
            });
        let request = EvsiRequest {
            decision_contract_identity: contract_identity,
            utility_unit,
            action_ids,
            candidates,
            cost_map: cost_map.map(|m| CostToUtilityMap {
                cost_unit: m.cost_unit,
                utility_unit: m.utility_unit,
                utility_per_cost: m.utility_per_cost,
            }),
            require_net_value,
            prior_observation_ids,
            rank_config,
            rng_seed,
            mc_error_tolerance,
            tie_tolerance,
            max_candidates,
        };
        Ok(Self { problem, prior, request, source_digests })
    }
}

/// A finished, sealed EVSI ranking.
#[derive(Clone, Debug)]
pub struct DesignRankingEvaluation {
    report: EvsiReport,
    ranking: RankingRecord,
    wire: DesignRankingArtifactWire,
    lowered_program: Option<SemanticDigest>,
}

impl DesignRankingEvaluation {
    /// The preposterior EVSI report (per-candidate values, error, trust, overlap).
    #[must_use]
    pub fn report(&self) -> &EvsiReport {
        &self.report
    }

    /// The order-invariant in-memory ranking value.
    #[must_use]
    pub fn ranking(&self) -> &RankingRecord {
        &self.ranking
    }

    /// The artifact body: everything an independent reader needs to inspect or recompute.
    #[must_use]
    pub fn artifact(&self) -> &DesignRankingArtifactWire {
        &self.wire
    }

    /// Identity of the lowered program the evaluation was bound to; `None` for an
    /// evaluation of a bare declaration ([`evaluate`]).
    #[must_use]
    pub const fn lowered_program_identity(&self) -> Option<SemanticDigest> {
        self.lowered_program
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
    wire: DesignRankingRequestWire,
) -> Result<DesignRankingEvaluation, DesignRankingError> {
    let prepared = Prepared::try_from(wire)?;
    let ctx = ExecutionContext::production(
        prepared.request.rng_seed,
        antecedent_core::default_user_threads(),
    );
    evaluate_prepared(&prepared, &ctx, None)
}

/// Compile `study` as the state belief of the declared EVSI evaluation.
///
/// The plan's lowered program carries the decision and the request (candidates, prior
/// observations, ranking schedule, seed, tolerances and the quoted Monte Carlo store).
/// Nothing is fitted or simulated.
///
/// # Errors
/// A malformed declaration, or [`DesignRankingError::Compile`] when the study publishes no
/// parameter posterior, the Monte Carlo store does not fit the admission pool of `ctx`, or
/// compile otherwise refuses.
pub fn compile_for_study(
    study: &Study,
    wire: DesignRankingRequestWire,
    ctx: &ExecutionContext,
) -> Result<PhysicalExecutionPlan, DesignRankingError> {
    let prepared = Prepared::try_from(wire)?;
    study
        .compile_for_evsi(&prepared.problem, &prepared.request, ctx)
        .map_err(DesignRankingError::Compile)
}

/// [`evaluate`], bound to the lowered program of `study` compiled for the request.
///
/// The compile runs first, so a study that cannot carry the request or a store the pool
/// cannot hold refuses before any candidate is scored. The evaluation observes the
/// cancellation of `ctx` and carries the program's identity
/// ([`DesignRankingEvaluation::lowered_program_identity`]).
///
/// # Errors
/// As [`compile_for_study`] and [`evaluate`].
pub fn evaluate_for_study(
    study: &Study,
    wire: DesignRankingRequestWire,
    ctx: &ExecutionContext,
) -> Result<DesignRankingEvaluation, DesignRankingError> {
    let prepared = Prepared::try_from(wire)?;
    let plan = study
        .compile_for_evsi(&prepared.problem, &prepared.request, ctx)
        .map_err(DesignRankingError::Compile)?;
    let lowered = plan
        .lowered()
        .filter(|lowered| {
            lowered
                .evidence()
                .decision()
                .is_some_and(|decision| decision.information_value().is_some())
        })
        .ok_or_else(|| {
            DesignRankingError::Compile(crate::compile_reason!(
                "route_not_lowered",
                "the plan compiled for an EVSI request carries no value-of-information program"
            ))
        })?;
    evaluate_prepared(&prepared, ctx, Some(lowered.identity()))
}

fn evaluate_prepared(
    prepared: &Prepared,
    ctx: &ExecutionContext,
    lowered_program: Option<SemanticDigest>,
) -> Result<DesignRankingEvaluation, DesignRankingError> {
    let evaluation =
        evaluate_evsi_with_context(&prepared.problem, &prepared.prior, &prepared.request, ctx)?;
    let report = evaluation.report();
    let artifact = antecedent_design::design_ranking_artifact::seal_with_context(
        &SealInputs {
            problem: &prepared.problem,
            prior: &prepared.prior,
            request: &prepared.request,
            report,
            source_digests: &prepared.source_digests,
        },
        &evaluation,
        ctx,
    )?;
    let ranking = RankingRecord::from_evsi(report, &prepared.source_digests)?;
    Ok(DesignRankingEvaluation {
        report: evaluation.into_report(),
        ranking,
        wire: artifact,
        lowered_program,
    })
}

/// The structural ordering when no probabilistic model is licensed: verified structural
/// sufficiency, then fewer cost units, then smaller sample budget, then semantic id.
///
/// # Errors
/// No candidates, or a repeated semantic id.
pub fn rank_structural(
    candidates: &[StructuralCandidateDeclWire],
) -> Result<RankingRecord, DesignRankingError> {
    let declared: Vec<StructuralCandidate> = candidates
        .iter()
        .map(|c| StructuralCandidate {
            semantic_id: c.semantic_id.clone(),
            verified_sufficient: c.verified_sufficient,
            cost: StudyCost { units: c.cost_units, sample_budget: c.sample_budget },
        })
        .collect();
    Ok(RankingRecord::from_structural(&declared)?)
}
