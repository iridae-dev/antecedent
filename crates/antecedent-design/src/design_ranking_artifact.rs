//! Durable design-ranking artifact (`design_ranking_v1`) and its independent consumer.
//!
//! The artifact retains, for one decision contract identity: the terminal action set,
//! the finite prior (or the conjugate normal belief) and the decision table
//! (utilities by action and prior draw, with admissibility), every candidate's semantic
//! id, exact signal request, provider identity, external update mode and trust label,
//! the signal law behind the value, the EVSI/EVPI/net-value numbers with integration
//! method and error, the cost mapping and source-overlap diagnostic, the source
//! distribution digests, the RNG seed and the rank/tie receipts. A digest over the
//! canonical body (candidates in semantic-id order) seals it, so reordering the
//! candidates that were supplied never changes the identity.
//!
//! The consumer ([`consume`]) refuses stored bounds above its own maxima before any
//! work, recomputes the digest, and then recomputes everything it can from the
//! embedded values:
//!
//! * a **native** signal with exact integration (Gaussian mean under the normal belief,
//!   binomial under the finite prior) has its law rebuilt from the family parameters and
//!   its EVSI, EVPI and net value recomputed ([`ReplayKind::NativeExactRecomputed`]);
//! * an **external** likelihood or posterior has the *arithmetic* recomputed from the
//!   retained attested table, never the provider ([`ReplayKind::AttestedLawArithmetic`]);
//! * **external decision values** are only checked for coherence against the prior and
//!   combined ([`ReplayKind::AttestedValuesCombined`]);
//! * a **Monte Carlo** value is bound and bounds-checked but not re-simulated
//!   ([`ReplayKind::MonteCarloNotReplayed`]).
//!
//! Only the first is ever reported as natively replayed. Source overlap, cost-unit
//! incompatibility, a signal request whose fingerprint no longer matches, a trust/mode
//! relabel the law contradicts, an altered table whose recomputed value disagrees with
//! the stored one, and a changed rank refuse even when the digest was recomputed
//! (resealed). Replay does not protect against a producer that states a different prior
//! or decision table and reseals honestly (the artifact is then a correct ranking of
//! those stated inputs); the consumer's [`ConsumeExpectation`] binds the contract,
//! signal identities, source digests and cost mapping it retained independently.
//!
//! Calibration of Monte Carlo error coverage and rank guarantees is `not_measured`; an
//! exact-integration EVSI is a point value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::too_many_lines,
    reason = "each receipt check is one linear pass kept in a single audited place"
)]

use std::collections::{BTreeMap, BTreeSet};

use antecedent_contract::{ExternalRefusal, ProviderObjectIdentity, ScientificQuantity};
use antecedent_core::{CanonicalWriter, is_blank};
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::error::IoError;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use crate::error::DesignError;
use crate::evsi::{
    CostToUtilityMap, DeclaredObservations, EvsiCandidate, EvsiCandidateRow, IntegrationMethod,
    ObservationOverlap, SearchReceipt, declared_overlap,
};
use crate::rank_rows::{AdjacentRank, evsi_bound_ok, rank_rows};
use crate::ranking::{DesignRankingBasis, EvsiRankRow, RankingError, RankingRecord};
use crate::signal::{
    RefusalBuilder, SignalError, SignalLimits, SignalReceipt, SignalRequest, SignalTrustLabel,
    SignalUpdateMode,
};

mod consuming;
mod replay;
mod sealing;
use consuming::rebuild_ranking;
pub use consuming::{
    consume, consume_wire, consume_wire_with_limits, consume_wire_with_receipts,
    consume_with_limits,
};
use replay::{Baseline, baseline, is_native, recompute, rectangular};
pub use sealing::{SealInputs, seal, seal_with_context};
use sealing::{provider_wire, request_wire};

/// Wire schema version.
pub const DESIGN_RANKING_ARTIFACT_VERSION: u32 = 1;
/// Artifact kind.
pub const DESIGN_RANKING_ARTIFACT_KIND: &str = "design_ranking_v1";
/// Calibration label: Monte Carlo error coverage and rank guarantees are not measured.
pub const DESIGN_RANKING_CALIBRATION: &str = "not_measured";
/// Identity of the reported ranking within
/// [`DesignRankingArtifactWire::provenance_chain`].
pub const DESIGN_RANKING_CLAIM_LINK_ID: &str = "design_ranking_result";
/// Maximum artifact bytes accepted before any decode allocation.
pub const MAX_DESIGN_RANKING_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Largest candidate catalog an artifact may retain (the frozen coordinate bound).
pub const MAX_RANKING_CANDIDATES: usize = 1024;
/// Largest number of prior draws or actions an artifact may retain.
pub(crate) const MAX_RANKING_STATES: usize = 65_536;
/// Largest number of table cells (states times actions or observation values).
pub(crate) const MAX_RANKING_CELLS: usize = 4 * 1024 * 1024;

/// Maxima a consumer accepts of a stored ranking, refused before any table is replayed.
///
/// The defaults are the `MAX_*` constants of this module; a consumer replaying larger catalogs or
/// tables raises them through [`consume_with_limits`] and its siblings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DesignRankingReadLimits {
    /// Largest container, in bytes, decoded.
    pub max_bytes: usize,
    /// Most candidates the artifact may retain, and the largest `max_candidates` it may declare.
    pub max_candidates: usize,
    /// Most prior draws, actions or signal support values a retained table may have.
    pub max_states: usize,
    /// Most cells (states times actions or observation values) of a retained table.
    pub max_cells: usize,
}

impl Default for DesignRankingReadLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_DESIGN_RANKING_ARTIFACT_BYTES,
            max_candidates: MAX_RANKING_CANDIDATES,
            max_states: MAX_RANKING_STATES,
            max_cells: MAX_RANKING_CELLS,
        }
    }
}

const BODY_SECTION: &str = "design_ranking_body";
/// Relative tolerance of a replayed number against the stored one.
const REPLAY_TOLERANCE: f64 = 1e-8;
/// Tolerance of probability-mass and coherence checks on retained tables.
const MASS_TOLERANCE: f64 = 1e-6;
// -- wire -------------------------------------------------------------------------------

/// Cost-to-utility mapping on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostMapWire {
    /// Unit the study costs are declared in.
    pub cost_unit: String,
    /// Utility unit of the decision.
    pub utility_unit: String,
    /// Utility per unit cost.
    pub utility_per_cost: f64,
}

/// The belief the decision is made under.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorWire {
    /// Equally weighted draws of the scalar decision state.
    Draws {
        /// Prior draws.
        states: Vec<f64>,
    },
    /// Conjugate normal belief on the scalar decision state.
    Normal {
        /// Prior mean.
        mean: f64,
        /// Prior variance.
        variance: f64,
    },
}

/// The decision table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtilityWire {
    /// `rows[action][draw]` utilities under a draws prior.
    Table {
        /// Utilities, one row per action.
        rows: Vec<Vec<f64>>,
    },
    /// `(intercept, slope)` per action under a normal prior.
    Affine {
        /// Coefficients, one pair per action.
        coefficients: Vec<(f64, f64)>,
    },
}

/// The decision problem the ranking values information for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionWire {
    /// Identity of the decision contract the problem implements.
    pub contract_identity: String,
    /// Utility unit.
    pub utility_unit: String,
    /// Terminal action identities in problem order.
    pub action_ids: Vec<String>,
    /// Belief.
    pub prior: PriorWire,
    /// Decision table.
    pub utility: UtilityWire,
    /// Admissible actions in problem order (all `true` under a normal prior).
    pub admissible: Vec<bool>,
}

/// Monte Carlo ranking configuration.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankConfigWire {
    /// Minimum batches before an early stop.
    pub min_batches: u32,
    /// Maximum batches.
    pub max_batches: u32,
    /// Batch size.
    pub batch_size: u32,
    /// Rank uncertainty threshold.
    pub rank_uncertainty_threshold: f64,
}

/// The exact signal request a provider answered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalRequestWire {
    /// Candidate identity the request names.
    pub candidate_id: String,
    /// Prior identity the update consumes.
    pub prior_id: String,
    /// State coordinate.
    pub state_quantity: ScientificQuantityWire,
    /// Observation coordinate.
    pub observation_quantity: ScientificQuantityWire,
    /// RNG seed.
    pub rng_seed: u64,
    /// Evidence lineage.
    pub evidence_lineage: Vec<String>,
    /// Conditional-independence assumption identity.
    pub conditional_independence: String,
    /// Largest sample size the request allows.
    pub max_sample_size: u64,
    /// Largest enumerated support the request allows.
    pub max_support: u64,
}

/// Exact external provider object identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWire {
    /// Provider service or model identity.
    pub provider_id: String,
    /// Object identity within the provider.
    pub object_id: String,
    /// Provider implementation version.
    pub version_id: String,
    /// Input data or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint.
    pub request_id: String,
}

/// The observation law behind a candidate's value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LawWire {
    /// Native Gaussian sample mean with the declared per-observation variance.
    GaussianMean {
        /// Per-observation noise variance.
        noise_variance: f64,
    },
    /// Native binomial (successes in `n` trials).
    Binomial,
    /// Attested likelihood `probabilities[y][k] = P(statistic_y | state_k)` over the prior
    /// draws, updated natively.
    FiniteLikelihood {
        /// Observation values.
        statistics: Vec<f64>,
        /// Likelihood table.
        probabilities: Vec<Vec<f64>>,
    },
    /// Attested posterior over the equally weighted prior draws, computed externally.
    FinitePosterior {
        /// Observation values.
        statistics: Vec<f64>,
        /// Predictive probability of each observation.
        predictive: Vec<f64>,
        /// `posterior[y][k] = P(state_k | statistic_y)`.
        posterior: Vec<Vec<f64>>,
    },
    /// Attested per-branch decision values, computed externally.
    DecisionValues {
        /// Probability of each branch.
        branch_probabilities: Vec<f64>,
        /// Action identities naming the value columns.
        action_ids: Vec<String>,
        /// `values[branch][action]`.
        values: Vec<Vec<f64>>,
    },
}

/// One ranked candidate with everything needed to inspect or recompute its value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvsiCandidateWire {
    /// Stable semantic identity.
    pub semantic_id: String,
    /// Sample size.
    pub sample_size: u64,
    /// Declared study cost amount.
    pub cost_amount: f64,
    /// Declared study cost unit.
    pub cost_unit: String,
    /// Observation identities the study would reuse.
    pub reused_observation_ids: Vec<String>,
    /// Exact signal request.
    pub request: SignalRequestWire,
    /// Signal family label.
    pub family: String,
    /// Update mode label.
    pub update_mode: String,
    /// Trust label.
    pub trust: String,
    /// Attestor of an externally attested signal.
    pub attestor: Option<String>,
    /// External provider object identity.
    pub provider: Option<ProviderWire>,
    /// Provider-supplied law digest.
    pub law_digest: String,
    /// Exact request fingerprint.
    pub request_fingerprint: String,
    /// Identity of the signal receipt (provider, update mode, trust, request, law).
    pub signal_identity: String,
    /// Signal law.
    pub law: LawWire,
    /// EVSI.
    pub evsi: f64,
    /// EVPI.
    pub evpi: f64,
    /// `exact`, `monte_carlo` or `externally_computed`.
    pub integration: String,
    /// Monte Carlo replicates.
    pub replicates: u64,
    /// Standard error of the EVSI.
    pub stderr: f64,
    /// Effective sample size.
    pub ess: Option<f64>,
    /// Integration converged within the declared tolerance.
    pub converged: bool,
    /// The ranker stopped before its batch cap.
    pub early_stopped: bool,
    /// Study cost in utility units, only with a cost mapping.
    pub study_cost_utility: Option<f64>,
    /// Net value, only with a cost mapping.
    pub net_value: Option<f64>,
    /// Rank (0 best).
    pub rank: u64,
    /// The gap to a neighbour is within error or the tie tolerance.
    pub rank_uncertain: bool,
    /// Distinct observation identities compared for overlap.
    pub overlap_checked: u64,
    /// Overlapping observation identities (empty for every sealed artifact).
    pub overlapping: Vec<String>,
    /// Assumptions the value rests on.
    pub assumptions: Vec<String>,
}

impl EvsiCandidateWire {
    /// Scored columns come from [`EvsiRankRow::from`]; the request, law and
    /// provider fields are the seal inputs that are not on the rank row.
    fn from_scored(row: &EvsiCandidateRow, declared: &EvsiCandidate, law: LawWire) -> Self {
        let scored = EvsiRankRow::from(row);
        let receipt = &row.signal_receipt;
        Self {
            semantic_id: scored.semantic_id,
            sample_size: scored.sample_size,
            cost_amount: scored.study_cost_amount,
            cost_unit: scored.study_cost_unit,
            reused_observation_ids: declared.reused_observation_ids.clone(),
            request: request_wire(&declared.signal_request),
            family: receipt.family.clone(),
            update_mode: scored.update_mode.as_str().to_owned(),
            trust: scored.provider_trust.as_str().to_owned(),
            attestor: receipt.attestor.clone(),
            provider: receipt.provider_identity.as_ref().map(provider_wire),
            law_digest: receipt.law_digest.clone(),
            request_fingerprint: scored.request_fingerprint,
            signal_identity: scored.signal_identity,
            law,
            evsi: scored.evsi,
            evpi: scored.evpi,
            integration: scored.integration.as_str().to_owned(),
            replicates: scored.replicates,
            stderr: scored.mc_stderr,
            ess: row.integration.ess,
            converged: row.integration.converged,
            early_stopped: row.integration.early_stopped,
            study_cost_utility: row.study_cost_utility,
            net_value: scored.net_value,
            rank: scored.rank as u64,
            rank_uncertain: scored.rank_uncertain,
            overlap_checked: row.source_overlap.observations_checked as u64,
            overlapping: row.source_overlap.overlapping.clone(),
            assumptions: row.assumptions.clone(),
        }
    }
}

impl TryFrom<&EvsiCandidateWire> for EvsiRankRow {
    type Error = DesignRankingArtifactError;

    fn try_from(wire: &EvsiCandidateWire) -> std::result::Result<Self, Self::Error> {
        let invalid = DesignRankingArtifactError::InvalidWire;
        Ok(Self {
            semantic_id: wire.semantic_id.clone(),
            rank: usize::try_from(wire.rank).map_err(|_| invalid("rank"))?,
            evsi: wire.evsi,
            mc_stderr: wire.stderr,
            replicates: wire.replicates,
            integration: parse_method(&wire.integration).ok_or_else(|| invalid("integration"))?,
            evpi: wire.evpi,
            sample_size: wire.sample_size,
            study_cost_amount: wire.cost_amount,
            study_cost_unit: wire.cost_unit.clone(),
            net_value: wire.net_value,
            rank_uncertain: wire.rank_uncertain,
            signal_identity: wire.signal_identity.clone(),
            request_fingerprint: wire.request_fingerprint.clone(),
            update_mode: parse_mode(&wire.update_mode).ok_or_else(|| invalid("update_mode"))?,
            provider_trust: parse_trust(&wire.trust).ok_or_else(|| invalid("trust"))?,
        })
    }
}

/// Search receipt on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchWire {
    /// Candidates supplied.
    pub supplied: u64,
    /// Candidates evaluated.
    pub evaluated: u64,
    /// The bound cut the catalog.
    pub truncated: bool,
    /// Semantic ids not evaluated.
    pub unevaluated_ids: Vec<String>,
}

/// A durable, independently recomputable design ranking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignRankingArtifactWire {
    /// Schema version.
    pub version: u32,
    /// `design_ranking_v1`.
    pub kind: String,
    /// `not_measured`.
    pub calibration: String,
    /// `evsi` or `net_value`.
    pub basis: String,
    /// Decision problem.
    pub decision: DecisionWire,
    /// Cost mapping, if one is supported.
    pub cost_map: Option<CostMapWire>,
    /// RNG seed of the evaluation.
    pub rng_seed: u64,
    /// Standard error at or below which a Monte Carlo estimate is converged.
    pub mc_error_tolerance: f64,
    /// Gap at or below which two candidates are tied.
    pub tie_tolerance: f64,
    /// Search bound.
    pub max_candidates: u64,
    /// Monte Carlo configuration.
    pub rank_config: RankConfigWire,
    /// Identities of observations already summarized by the prior.
    pub prior_observation_ids: Vec<String>,
    /// Source distribution digests.
    pub source_digests: Vec<String>,
    /// Bayes action index under the current belief.
    pub bayes_action: u64,
    /// Expected utility of the Bayes action under the current belief.
    pub prior_expected_utility: f64,
    /// Expected value of perfect information.
    pub evpi: f64,
    /// Candidates (any order; the digest uses semantic-id order).
    pub candidates: Vec<EvsiCandidateWire>,
    /// Adjacent pairs tied within the tie tolerance.
    pub ties: Vec<(String, String)>,
    /// Search receipt.
    pub search: SearchWire,
    /// Identity of the in-memory [`RankingRecord`] value.
    pub ranking_identity: String,
    /// Digest over the canonical body.
    pub digest: String,
}

// -- errors -----------------------------------------------------------------------------

/// Why a ranking artifact cannot be sealed or consumed.
#[derive(Debug)]
#[non_exhaustive]
pub enum DesignRankingArtifactError {
    /// Container, checksum or CBOR failure (corruption, truncation, oversize).
    Io(IoError),
    /// The artifact has an unknown major version.
    UnsupportedVersion(u32),
    /// A stored bound exceeds the consumer's maxima.
    Bounds(&'static str),
    /// The body is structurally invalid.
    InvalidWire(&'static str),
    /// The digest does not match the body.
    DigestMismatch,
    /// The digest differs from the identity the consumer retained.
    ExpectedIdentityMismatch,
    /// The ranking value disagrees with a retained expectation or with itself.
    Ranking(RankingError),
    /// A candidate's signal request, receipt, family, mode, trust or law is inconsistent.
    SignalInconsistent {
        /// Candidate.
        candidate: String,
        /// Which slot disagrees.
        slot: &'static str,
    },
    /// A recomputed number differs from the stored one.
    ReplayMismatch {
        /// Candidate (empty for a decision-level number).
        candidate: String,
        /// Which quantity.
        quantity: &'static str,
    },
    /// A study reuses observations the prior already summarizes.
    SourceOverlap {
        /// Candidate.
        candidate: String,
        /// Shared observation identities.
        ids: Vec<String>,
    },
    /// Whether a study reuses the prior's observations is not computable; unknown
    /// dependence is never independence.
    SourceOverlapUnknown {
        /// Candidate.
        candidate: String,
        /// Why.
        reason: String,
    },
    /// A cost unit is incompatible with the cost mapping or the mapping with the decision.
    CostUnitsMismatch(String),
    /// The stored order, tie or search receipt does not follow from the stored values.
    RankMismatch(&'static str),
    /// A producer-side signal failure.
    Signal(SignalError),
    /// A producer-side decision-analysis failure.
    Design(DesignError),
}

impl From<IoError> for DesignRankingArtifactError {
    fn from(error: IoError) -> Self {
        Self::Io(error)
    }
}

impl From<RankingError> for DesignRankingArtifactError {
    fn from(error: RankingError) -> Self {
        Self::Ranking(error)
    }
}

impl From<SignalError> for DesignRankingArtifactError {
    fn from(error: SignalError) -> Self {
        Self::Signal(error)
    }
}

impl From<DesignError> for DesignRankingArtifactError {
    fn from(error: DesignError) -> Self {
        Self::Design(error)
    }
}

impl DesignRankingArtifactError {
    /// Structured refusal under the `design_ranking` namespace; `None` for corruption
    /// and truncation, which are serialization failures rather than refusals.
    #[must_use]
    pub fn refusal(&self) -> Option<ExternalRefusal> {
        let signal = antecedent_core::reason_code!("design_signal_invalid");
        let at = |code, detail: &str| RefusalBuilder::new(code, "consume", detail);
        Some(match self {
            Self::Io(_) => return None,
            Self::UnsupportedVersion(version) => at(
                antecedent_core::reason_code!("route_not_supported"),
                antecedent_core::reason_details::design_ranking::UNSUPPORTED_VERSION,
            )
            .supplied(version.to_string())
            .expected(DESIGN_RANKING_ARTIFACT_VERSION.to_string())
            .build(),
            Self::Bounds(what) => at(
                antecedent_core::reason_code!("route_not_supported"),
                antecedent_core::reason_details::design_ranking::BOUNDS_EXCEEDED,
            )
            .offending(*what)
            .build(),
            Self::InvalidWire(what) => at(
                antecedent_core::reason_code!("invalid_argument"),
                antecedent_core::reason_details::design_ranking::INVALID_ARTIFACT,
            )
            .offending(*what)
            .build(),
            Self::DigestMismatch => {
                at(signal, antecedent_core::reason_details::design_ranking::DIGEST_MISMATCH).build()
            }
            Self::ExpectedIdentityMismatch => {
                at(signal, antecedent_core::reason_details::design_ranking::IDENTITY_MISMATCH)
                    .build()
            }
            Self::Ranking(error) => error.to_refusal(),
            Self::SignalInconsistent { candidate, slot } => {
                at(signal, antecedent_core::reason_details::design_ranking::SIGNAL_INCONSISTENT)
                    .offending(candidate.clone())
                    .expected(*slot)
                    .build()
            }
            Self::ReplayMismatch { candidate, quantity } => {
                at(signal, antecedent_core::reason_details::design_ranking::REPLAY_MISMATCH)
                    .offending(candidate.clone())
                    .expected(*quantity)
                    .build()
            }
            Self::SourceOverlap { candidate, ids } => {
                at(signal, antecedent_core::reason_details::design_ranking::SOURCE_OVERLAP)
                    .offending(candidate.clone())
                    .expected(ids.join(","))
                    .remedy("use only data not already summarized by the prior")
                    .build()
            }
            Self::SourceOverlapUnknown { candidate, reason } => at(
                antecedent_core::reason_code!("evidence_overlap_unknown"),
                antecedent_core::reason_details::design_ranking::SOURCE_OVERLAP_UNKNOWN,
            )
            .offending(candidate.clone())
            .expected(reason.clone())
            .remedy("declare the observations behind the prior and each study")
            .build(),
            Self::CostUnitsMismatch(what) => at(
                antecedent_core::reason_code!("design_cost_units_mismatch"),
                antecedent_core::reason_details::design_ranking::COST_UNITS_MISMATCH,
            )
            .offending(what.clone())
            .build(),
            Self::RankMismatch(what) => {
                at(signal, antecedent_core::reason_details::design_ranking::RANK_MISMATCH)
                    .offending(*what)
                    .build()
            }
            Self::Signal(error) => error.to_refusal(),
            Self::Design(error) => {
                at(signal, antecedent_core::reason_details::design_ranking::DESIGN_ERROR)
                    .offending("decision")
                    .expected(error.to_string())
                    .build()
            }
        })
    }
}

impl std::fmt::Display for DesignRankingArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.refusal() {
            Some(refusal) => write!(
                f,
                "{}{}",
                refusal.detail,
                refusal.offending.map_or_else(String::new, |o| format!(" at {o}"))
            ),
            None => match self {
                Self::Io(error) => write!(f, "design ranking artifact: {error}"),
                _ => write!(f, "design ranking artifact"),
            },
        }
    }
}

impl std::error::Error for DesignRankingArtifactError {}

type Result<T> = std::result::Result<T, DesignRankingArtifactError>;

// -- small helpers ----------------------------------------------------------------------

fn close(stored: f64, recomputed: f64) -> bool {
    stored.is_finite() && (stored - recomputed).abs() <= REPLAY_TOLERANCE * (1.0 + recomputed.abs())
}

fn parse_mode(label: &str) -> Option<SignalUpdateMode> {
    [
        SignalUpdateMode::NativeUpdate,
        SignalUpdateMode::ExternalPosterior,
        SignalUpdateMode::ExternalDecisionValues,
    ]
    .into_iter()
    .find(|mode| mode.as_str() == label)
}

fn parse_trust(label: &str) -> Option<SignalTrustLabel> {
    [
        SignalTrustLabel::NativeSupported,
        SignalTrustLabel::ExternallyAttested,
        SignalTrustLabel::ExactRequestVerified,
    ]
    .into_iter()
    .find(|trust| trust.as_str() == label)
}

fn parse_method(label: &str) -> Option<IntegrationMethod> {
    [IntegrationMethod::Exact, IntegrationMethod::MonteCarlo, IntegrationMethod::ExternallyComputed]
        .into_iter()
        .find(|method| method.as_str() == label)
}

fn rank_basis(label: &str) -> Option<DesignRankingBasis> {
    match label {
        "evsi" => Some(DesignRankingBasis::Evsi),
        "net_value" => Some(DesignRankingBasis::NetValue),
        _ => None,
    }
}

// -- recomputation ----------------------------------------------------------------------

// -- replay kinds and consumed value -----------------------------------------------------

/// How much of a candidate's value the consumer recomputed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayKind {
    /// Native law rebuilt from its family parameters; EVSI, EVPI and net value recomputed.
    NativeExactRecomputed,
    /// Arithmetic recomputed from the retained attested table; the provider was not run
    /// and is not verified.
    AttestedLawArithmetic,
    /// Attested decision values checked for coherence against the prior and combined.
    AttestedValuesCombined,
    /// Native Monte Carlo value bound and bounds-checked, not re-simulated.
    MonteCarloNotReplayed,
}

impl ReplayKind {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeExactRecomputed => "native_exact_recomputed",
            Self::AttestedLawArithmetic => "attested_law_arithmetic",
            Self::AttestedValuesCombined => "attested_values_combined",
            Self::MonteCarloNotReplayed => "monte_carlo_not_replayed",
        }
    }
}

/// One consumed candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedCandidate {
    /// Semantic id.
    pub semantic_id: String,
    /// How the value was checked.
    pub replay: ReplayKind,
    /// Provider trust label (never upgraded by consuming; a stored
    /// `exact_request_verified` is reported `externally_attested` unless the
    /// consumer supplied the receipt).
    pub trust: SignalTrustLabel,
    /// Update mode.
    pub update_mode: SignalUpdateMode,
    /// What the trust label does and does not establish.
    pub trust_limit: String,
}

impl ConsumedCandidate {
    /// Whether Antecedent recomputed the value from a natively reconstructed law. Only a
    /// native exact-integration signal is ever natively replayed.
    #[must_use]
    pub const fn natively_replayed(&self) -> bool {
        matches!(self.replay, ReplayKind::NativeExactRecomputed)
    }
}

/// A consumed ranking.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedDesignRanking {
    /// The ranking value rebuilt from the artifact.
    pub ranking: RankingRecord,
    /// The artifact digest.
    pub identity: String,
    /// Candidates in rank order.
    pub candidates: Vec<ConsumedCandidate>,
    /// `not_measured`.
    pub calibration: &'static str,
    /// Checked derivation chain retained after independent artifact consumption.
    pub provenance_chain: antecedent_contract::ProvenanceChain,
}

/// Identities the consumer retained independently of the bytes. A `None` field is not
/// checked against an external value.
#[derive(Clone, Debug, Default)]
pub struct ConsumeExpectation {
    /// Artifact digest.
    pub artifact_identity: Option<String>,
    /// Decision contract identity.
    pub decision_contract_identity: Option<String>,
    /// Signal receipt identity per candidate semantic id.
    pub signal_identities: Option<BTreeMap<String, String>>,
    /// Source distribution digests.
    pub source_digests: Option<Vec<String>>,
    /// Cost mapping (`Some(None)` asserts that none was used).
    pub cost_mapping: Option<Option<CostToUtilityMap>>,
}

// -- digest, encode, decode -------------------------------------------------------------

impl DesignRankingArtifactWire {
    fn canonical(&self) -> Self {
        let mut body = self.clone();
        body.digest = String::new();
        body.candidates.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
        for candidate in &mut body.candidates {
            candidate.request.evidence_lineage.sort();
            candidate.reused_observation_ids.sort();
            candidate.overlapping.sort();
        }
        body.prior_observation_ids.sort();
        body.source_digests.sort();
        body.ties.sort();
        body.search.unevaluated_ids.sort();
        body
    }

    /// Queryable lineage of the ranking: the decision contract, each source
    /// distribution digest, each external signal provider object, each candidate's
    /// study-ranking signal (named by its signal identity, so a changed provider,
    /// update mode, trust, request or law changes its digest), and the
    /// `design_ranking_result` claim derived from all of them. Parents are listed in
    /// a fixed order (contract, sorted distinct source digests, signals in semantic-id
    /// order), so the chain does not depend on candidate order. It is derived from the
    /// retained identities and is not stored in the artifact bytes.
    ///
    /// # Errors
    /// A blank contract, source digest, semantic id or signal identity refuses.
    pub fn provenance_chain(
        &self,
    ) -> std::result::Result<
        antecedent_contract::ProvenanceChain,
        antecedent_contract::ProvenanceChainError,
    > {
        use antecedent_contract::{CompositionLink, CompositionStage, ProvenanceChain};
        let link = |id: String, stage: CompositionStage, parents: Vec<String>| CompositionLink {
            id,
            stage,
            parents,
            declared_parent_digests: None,
        };
        let decision = format!("decision:{}", self.decision.contract_identity);
        let mut links = vec![link(decision.clone(), CompositionStage::DecisionContract, vec![])];
        let mut result_parents = vec![decision.clone()];
        let sources: BTreeSet<&String> = self.source_digests.iter().collect();
        for digest in sources {
            let id = format!("distribution:{digest}");
            links.push(link(id.clone(), CompositionStage::DistributionArtifact, vec![]));
            result_parents.push(id);
        }
        let mut candidates: Vec<&EvsiCandidateWire> = self.candidates.iter().collect();
        candidates.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
        let mut seen_providers = BTreeSet::new();
        for candidate in candidates {
            let mut parents = vec![decision.clone()];
            if let Some(provider) = &candidate.provider {
                let id = format!(
                    "provider:signal:{}/{}@{}#{}",
                    provider.provider_id,
                    provider.object_id,
                    provider.version_id,
                    provider.snapshot_id
                );
                if seen_providers.insert(id.clone()) {
                    links.push(link(id.clone(), CompositionStage::ExternalProvider, vec![]));
                }
                parents.push(id);
            }
            let id = format!("signal:{}:{}", candidate.semantic_id, candidate.signal_identity);
            links.push(link(id.clone(), CompositionStage::StudyRankingProvider, parents));
            result_parents.push(id);
        }
        links.push(link(
            DESIGN_RANKING_CLAIM_LINK_ID.to_owned(),
            CompositionStage::Claim,
            result_parents,
        ));
        ProvenanceChain::new(links)
    }

    /// Digest of the canonical body; unchanged by candidate order.
    ///
    /// # Errors
    ///
    /// A CBOR encoding failure.
    pub fn compute_digest(&self) -> std::result::Result<String, IoError> {
        let bytes = to_cbor(&self.canonical())?;
        Ok(CanonicalWriter::new("antecedent.design_ranking_artifact.v1").bytes(&bytes).finish_hex())
    }

    /// Recompute and store the digest (used to seal; a consumer-independent check is
    /// still made against the numbers).
    ///
    /// # Errors
    ///
    /// A CBOR encoding failure.
    pub fn reseal(&mut self) -> std::result::Result<(), IoError> {
        self.digest = self.compute_digest()?;
        Ok(())
    }

    /// Recompute the ranking identity from the stored candidates, then the
    /// artifact digest. For a producer that edits a candidate's signal or trust
    /// label; consumption still re-derives every number and the signal identity.
    ///
    /// # Errors
    ///
    /// A malformed ranking body or a CBOR encoding failure.
    pub fn reseal_ranking(&mut self) -> Result<()> {
        self.ranking_identity = rebuild_ranking(self)?.identity();
        self.reseal()?;
        Ok(())
    }

    /// Serialize through the checksummed container, candidates in rank order.
    ///
    /// # Errors
    ///
    /// A blank artifact id or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> std::result::Result<Vec<u8>, IoError> {
        if is_blank(artifact_id) {
            return Err(IoError::Convert("missing artifact id".into()));
        }
        let mut ordered = self.clone();
        ordered
            .candidates
            .sort_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.semantic_id.cmp(&b.semantic_id)));
        let body = to_cbor(&ordered)?;
        antecedent_io_core::encode_container(
            &antecedent_io_core::ArtifactContainer {
                kind: DESIGN_RANKING_ARTIFACT_KIND,
                note: "design_ranking",
                max_bytes: MAX_DESIGN_RANKING_ARTIFACT_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[antecedent_io_core::OutSection {
                id: BODY_SECTION,
                content_type: antecedent_io_core::CONTENT_CBOR,
                bytes: &body,
            }],
        )
        .map_err(|error| error.into_io("unsupported design ranking artifact layout"))
    }

    /// Decode the container and body, refusing oversize input before any allocation.
    ///
    /// # Errors
    ///
    /// Corruption, truncation, oversize, an unknown layout, or an unknown major version.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with_limits(bytes, &DesignRankingReadLimits::default())
    }

    /// [`Self::from_bytes`] under the caller's byte maximum instead of the default.
    ///
    /// # Errors
    ///
    /// Those of [`Self::from_bytes`], with `limits.max_bytes` in place of the default cap.
    pub fn from_bytes_with_limits(bytes: &[u8], limits: &DesignRankingReadLimits) -> Result<Self> {
        let parts = antecedent_io_core::decode_container(
            DESIGN_RANKING_ARTIFACT_KIND,
            limits.max_bytes,
            &[BODY_SECTION],
            bytes,
        )
        .map_err(|error| error.into_io("unsupported design ranking artifact layout"))?;
        let Ok([body]) = <[Vec<u8>; 1]>::try_from(parts) else {
            return Err(
                IoError::Convert("unsupported design ranking artifact layout".into()).into()
            );
        };
        // The version word is read from the untyped item before the typed
        // body, so a newer layout is refused as an unsupported version.
        let wire: Self = antecedent_io_core::decode_with_version(&body, |version: u32| {
            if version == DESIGN_RANKING_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(DesignRankingArtifactError::UnsupportedVersion(version))
            }
        })?;
        Ok(wire)
    }
}

// -- sealing ----------------------------------------------------------------------------

// -- consuming --------------------------------------------------------------------------
