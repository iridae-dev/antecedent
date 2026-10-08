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
//! Calibration of Monte Carlo error coverage and rank guarantees is `unmeasured`; an
//! exact-integration EVSI is a point value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::too_many_lines,
    reason = "each receipt check is one linear pass kept in a single audited place"
)]

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{ExternalRefusal, ProviderObjectIdentity, ScientificQuantity};
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};
use serde::{Deserialize, Serialize};

use crate::decision::{DecisionProblem, DecisionTable};
use crate::error::DesignError;
use crate::evsi::{CostToUtilityMap, EvsiReport, EvsiRequest, IntegrationMethod, SearchReceipt};
use crate::preposterior::{DecisionPrior, DecisionSignal, expected_max_affine};
use crate::ranking::{DesignRanking, DesignRankingBasis, RankingEntry, RankingError};
use crate::signal::{
    PreparedLaw, SignalError, SignalLimits, SignalReceipt, SignalRequest, SignalTrustLabel,
    SignalUpdateMode, make_refusal,
};

/// Wire schema version.
pub const DESIGN_RANKING_ARTIFACT_VERSION: u32 = 1;
/// Artifact kind.
pub const DESIGN_RANKING_ARTIFACT_KIND: &str = "design_ranking_v1";
/// Calibration label: Monte Carlo error coverage and rank guarantees are not measured.
pub const DESIGN_RANKING_CALIBRATION: &str = "unmeasured";
/// Identity of the reported ranking within
/// [`DesignRankingArtifactWire::provenance_chain`].
pub const DESIGN_RANKING_CLAIM_LINK_ID: &str = "design_ranking_result";
/// Maximum artifact bytes accepted before any decode allocation.
pub const MAX_DESIGN_RANKING_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Largest candidate catalog an artifact may retain (the frozen coordinate bound).
pub const MAX_RANKING_CANDIDATES: usize = 1024;
/// Largest number of prior draws or actions an artifact may retain.
pub const MAX_RANKING_STATES: usize = 65_536;
/// Largest number of table cells (states times actions or observation values).
pub const MAX_RANKING_CELLS: usize = 4 * 1024 * 1024;

const BODY_SECTION: &str = "design_ranking_body";
/// Relative tolerance of a replayed number against the stored one.
const REPLAY_TOLERANCE: f64 = 1e-8;
/// Tolerance of probability-mass and coherence checks on retained tables.
const MASS_TOLERANCE: f64 = 1e-6;
/// Critical value of the rank-uncertainty comparison (matches the evaluator).
const RANK_Z: f64 = 1.96;

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
pub struct CandidateWire {
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
    /// `unmeasured`.
    pub calibration: String,
    /// `evsi` or `net_value`.
    pub basis: String,
    /// Decision problem.
    pub decision: DecisionWire,
    /// Cost mapping, if one is licensed.
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
    pub candidates: Vec<CandidateWire>,
    /// Adjacent pairs tied within the tie tolerance.
    pub ties: Vec<(String, String)>,
    /// Search receipt.
    pub search: SearchWire,
    /// Identity of the in-memory [`DesignRanking`] value.
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
        let at = |code, detail: &str| make_refusal(code, "consume", detail);
        let with =
            |code, detail: &str, offending: String, expected: Option<String>| ExternalRefusal {
                offending: Some(offending),
                expected,
                ..make_refusal(code, "consume", detail)
            };
        Some(match self {
            Self::Io(_) => return None,
            Self::UnsupportedVersion(version) => ExternalRefusal {
                supplied: Some(version.to_string()),
                expected: Some(DESIGN_RANKING_ARTIFACT_VERSION.to_string()),
                ..at(
                    antecedent_core::reason_code!("route_not_supported"),
                    "design_ranking.unsupported_version",
                )
            },
            Self::Bounds(what) => with(
                antecedent_core::reason_code!("route_not_supported"),
                "design_ranking.bounds_exceeded",
                (*what).to_owned(),
                None,
            ),
            Self::InvalidWire(what) => with(
                antecedent_core::reason_code!("invalid_argument"),
                "design_ranking.invalid_artifact",
                (*what).to_owned(),
                None,
            ),
            Self::DigestMismatch => at(signal, "design_ranking.digest_mismatch"),
            Self::ExpectedIdentityMismatch => at(signal, "design_ranking.identity_mismatch"),
            Self::Ranking(error) => error.to_refusal(),
            Self::SignalInconsistent { candidate, slot } => with(
                signal,
                "design_ranking.signal_inconsistent",
                candidate.clone(),
                Some((*slot).to_owned()),
            ),
            Self::ReplayMismatch { candidate, quantity } => with(
                signal,
                "design_ranking.replay_mismatch",
                candidate.clone(),
                Some((*quantity).to_owned()),
            ),
            Self::SourceOverlap { candidate, ids } => ExternalRefusal {
                remedy: Some("use only data not already summarized by the prior"),
                ..with(
                    signal,
                    "design_ranking.source_overlap",
                    candidate.clone(),
                    Some(ids.join(",")),
                )
            },
            Self::CostUnitsMismatch(what) => with(
                antecedent_core::reason_code!("design_cost_units_mismatch"),
                "design_ranking.cost_units_mismatch",
                what.clone(),
                None,
            ),
            Self::RankMismatch(what) => {
                with(signal, "design_ranking.rank_mismatch", (*what).to_owned(), None)
            }
            Self::Signal(error) => error.to_refusal(),
            Self::Design(error) => with(
                signal,
                "design_ranking.design_error",
                "decision".to_owned(),
                Some(error.to_string()),
            ),
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

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn put_f64s(hasher: &mut blake3::Hasher, values: &[f64]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

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
        SignalTrustLabel::NativeLicensed,
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

fn native_digest(family: &str, parameters: &[f64]) -> String {
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, family);
    if family == "gaussian_mean" {
        put_f64s(&mut hasher, parameters);
    }
    hasher.finalize().to_hex().to_string()
}

fn rank_basis(label: &str) -> Option<DesignRankingBasis> {
    match label {
        "evsi" => Some(DesignRankingBasis::Evsi),
        "net_value" => Some(DesignRankingBasis::NetValue),
        _ => None,
    }
}

// -- recomputation ----------------------------------------------------------------------

/// What the stored decision implies before any information.
struct Baseline {
    bayes: usize,
    prior_best: f64,
    evpi: f64,
    /// Lines `(c_a, d_a)` of the conjugate-normal analysis (normal belief only).
    lines: Vec<(f64, f64)>,
}

fn finite_rows(decision: &DecisionWire) -> Option<(&[Vec<f64>], usize)> {
    match (&decision.prior, &decision.utility) {
        (PriorWire::Draws { states }, UtilityWire::Table { rows }) => {
            Some((rows.as_slice(), states.len()))
        }
        _ => None,
    }
}

fn admissible_indices(decision: &DecisionWire) -> Vec<usize> {
    decision.admissible.iter().enumerate().filter(|(_, ok)| **ok).map(|(a, _)| a).collect()
}

fn baseline(decision: &DecisionWire) -> Baseline {
    if let Some((rows, k)) = finite_rows(decision) {
        let kf = k as f64;
        let adm = admissible_indices(decision);
        let mut bayes = adm[0];
        let mut prior_best = f64::NEG_INFINITY;
        for &a in &adm {
            let eu = rows[a].iter().sum::<f64>() / kf;
            if eu > prior_best {
                prior_best = eu;
                bayes = a;
            }
        }
        let evpi = (0..k)
            .map(|draw| adm.iter().map(|&a| rows[a][draw]).fold(f64::NEG_INFINITY, f64::max))
            .sum::<f64>()
            / kf
            - prior_best;
        return Baseline { bayes, prior_best, evpi, lines: Vec::new() };
    }
    let (PriorWire::Normal { mean, variance }, UtilityWire::Affine { coefficients }) =
        (&decision.prior, &decision.utility)
    else {
        return Baseline { bayes: 0, prior_best: 0.0, evpi: 0.0, lines: Vec::new() };
    };
    let mut bayes = 0;
    let mut prior_best = f64::NEG_INFINITY;
    for (a, (alpha, beta)) in coefficients.iter().enumerate() {
        let eu = alpha + beta * mean;
        if eu > prior_best {
            prior_best = eu;
            bayes = a;
        }
    }
    let (alpha_star, beta_star) = coefficients[bayes];
    let lines: Vec<(f64, f64)> = coefficients
        .iter()
        .map(|&(alpha, beta)| {
            let slope = beta - beta_star;
            ((alpha - alpha_star + slope * mean).min(0.0), slope)
        })
        .collect();
    let evpi = expected_max_affine(&lines, variance.sqrt());
    Baseline { bayes, prior_best, evpi, lines }
}

/// Binomial likelihood `[y][k]` over the prior draws.
fn binomial_table(states: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut ln_choose = 0.0_f64;
    let mut rows = Vec::with_capacity(n + 1);
    for y in 0..=n {
        if y > 0 {
            ln_choose += ((n - y + 1) as f64 / y as f64).ln();
        }
        rows.push(
            states
                .iter()
                .map(|&theta| {
                    let success = if y > 0 { y as f64 * theta.ln() } else { 0.0 };
                    let failure = if y < n { (n - y) as f64 * (1.0 - theta).ln() } else { 0.0 };
                    (ln_choose + success + failure).exp()
                })
                .collect(),
        );
    }
    rows
}

fn rectangular(rows: &[Vec<f64>], n_rows: usize, n_cols: usize) -> bool {
    rows.len() == n_rows && rows.iter().all(|r| r.len() == n_cols)
}

fn probabilities(values: &[f64]) -> bool {
    values.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
}

/// Expected value after information, `Σ_y max_{a ∈ F} E[P(y | θ) U(a, θ)]`, with the
/// likelihood mass checked.
fn value_after(decision: &DecisionWire, likelihood: &[Vec<f64>]) -> Option<f64> {
    let (rows, k) = finite_rows(decision)?;
    let kf = k as f64;
    let adm = admissible_indices(decision);
    let mut total = 0.0;
    let mut mass = 0.0;
    for prow in likelihood {
        mass += prow.iter().sum::<f64>() / kf;
        total += adm
            .iter()
            .map(|&a| rows[a].iter().zip(prow).map(|(u, p)| u * p).sum::<f64>() / kf)
            .fold(f64::NEG_INFINITY, f64::max);
    }
    ((mass - 1.0).abs() <= MASS_TOLERANCE).then_some(total)
}

fn column_masses_ok(likelihood: &[Vec<f64>], k: usize) -> bool {
    (0..k).all(|col| {
        let mass: f64 = likelihood.iter().map(|row| row[col]).sum();
        (mass - 1.0).abs() <= MASS_TOLERANCE
    })
}

/// Likelihood `[y][k]` implied by an attested posterior, with coherence checked.
fn posterior_to_likelihood(
    k: usize,
    predictive: &[f64],
    posterior: &[Vec<f64>],
) -> Option<Vec<Vec<f64>>> {
    let kf = k as f64;
    if !probabilities(predictive)
        || !rectangular(posterior, predictive.len(), k)
        || !posterior.iter().all(|row| probabilities(row))
        || (predictive.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE
        || !posterior.iter().all(|row| (row.iter().sum::<f64>() - 1.0).abs() <= MASS_TOLERANCE)
    {
        return None;
    }
    for col in 0..k {
        let recovered: f64 = predictive.iter().zip(posterior).map(|(p, row)| p * row[col]).sum();
        if (recovered - 1.0 / kf).abs() > MASS_TOLERANCE {
            return None;
        }
    }
    Some(
        predictive
            .iter()
            .zip(posterior)
            .map(|(p, row)| row.iter().map(|q| kf * p * q).collect())
            .collect(),
    )
}

/// What the consumer could recompute for one candidate.
struct Recomputed {
    /// Recomputed EVSI, when the law is replayable.
    evsi: Option<f64>,
    evpi: f64,
    replay: ReplayKind,
}

fn is_native(law: &LawWire) -> bool {
    matches!(law, LawWire::GaussianMean { .. } | LawWire::Binomial)
}

fn recompute(
    candidate: &CandidateWire,
    decision: &DecisionWire,
    base: &Baseline,
    method: IntegrationMethod,
) -> std::result::Result<Recomputed, &'static str> {
    let n = candidate.sample_size;
    let k = finite_rows(decision).map_or(0, |(_, k)| k);
    match (&candidate.law, method) {
        (LawWire::GaussianMean { noise_variance }, IntegrationMethod::Exact) => {
            let PriorWire::Normal { variance, .. } = &decision.prior else {
                return Err("law");
            };
            if !(noise_variance.is_finite() && *noise_variance > 0.0) {
                return Err("law");
            }
            let nf = n as f64;
            let preposterior = variance * (nf * variance) / (nf * variance + noise_variance);
            let after = expected_max_affine(&base.lines, preposterior.sqrt());
            Ok(Recomputed {
                evsi: Some(after),
                evpi: base.evpi,
                replay: ReplayKind::NativeExactRecomputed,
            })
        }
        (LawWire::GaussianMean { noise_variance }, IntegrationMethod::MonteCarlo) => {
            if !(noise_variance.is_finite() && *noise_variance > 0.0) || k == 0 {
                return Err("law");
            }
            Ok(Recomputed {
                evsi: None,
                evpi: base.evpi,
                replay: ReplayKind::MonteCarloNotReplayed,
            })
        }
        (LawWire::Binomial, IntegrationMethod::Exact) => {
            let PriorWire::Draws { states } = &decision.prior else {
                return Err("law");
            };
            let Ok(n_us) = usize::try_from(n) else {
                return Err("law");
            };
            if !states.iter().all(|s| (0.0..=1.0).contains(s)) || n_us >= MAX_RANKING_STATES {
                return Err("law");
            }
            let table = binomial_table(states, n_us);
            let after = value_after(decision, &table).ok_or("law")?;
            Ok(Recomputed {
                evsi: Some(after - base.prior_best),
                evpi: base.evpi,
                replay: ReplayKind::NativeExactRecomputed,
            })
        }
        (
            LawWire::FiniteLikelihood { statistics, probabilities: table },
            IntegrationMethod::Exact,
        ) => {
            if statistics.is_empty()
                || !rectangular(table, statistics.len(), k)
                || !table.iter().all(|row| probabilities(row))
                || !column_masses_ok(table, k)
            {
                return Err("law");
            }
            let after = value_after(decision, table).ok_or("law")?;
            Ok(Recomputed {
                evsi: Some(after - base.prior_best),
                evpi: base.evpi,
                replay: ReplayKind::AttestedLawArithmetic,
            })
        }
        (
            LawWire::FinitePosterior { statistics, predictive, posterior },
            IntegrationMethod::Exact,
        ) => {
            if statistics.len() != predictive.len() || statistics.is_empty() {
                return Err("law");
            }
            let table = posterior_to_likelihood(k, predictive, posterior).ok_or("law")?;
            let after = value_after(decision, &table).ok_or("law")?;
            Ok(Recomputed {
                evsi: Some(after - base.prior_best),
                evpi: base.evpi,
                replay: ReplayKind::AttestedLawArithmetic,
            })
        }
        (
            LawWire::DecisionValues { branch_probabilities, action_ids, values },
            IntegrationMethod::ExternallyComputed,
        ) => {
            let (rows, kk) = finite_rows(decision).ok_or("law")?;
            let n_actions = decision.action_ids.len();
            let mut left: Vec<&String> = action_ids.iter().collect();
            let mut right: Vec<&String> = decision.action_ids.iter().collect();
            left.sort();
            right.sort();
            if left != right {
                return Err("law");
            }
            if branch_probabilities.is_empty()
                || !probabilities(branch_probabilities)
                || !rectangular(values, branch_probabilities.len(), n_actions)
                || !values.iter().all(|row| row.iter().all(|v| v.is_finite()))
                || (branch_probabilities.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE
            {
                return Err("law");
            }
            let column: Vec<usize> = decision
                .action_ids
                .iter()
                .filter_map(|id| action_ids.iter().position(|v| v == id))
                .collect();
            for action in 0..n_actions {
                let averaged: f64 = branch_probabilities
                    .iter()
                    .enumerate()
                    .map(|(y, p)| p * values[y][column[action]])
                    .sum();
                let prior_eu = rows[action].iter().sum::<f64>() / kk as f64;
                if (averaged - prior_eu).abs() > MASS_TOLERANCE * (1.0 + prior_eu.abs()) {
                    return Err("law");
                }
            }
            let adm = admissible_indices(decision);
            let after: f64 = branch_probabilities
                .iter()
                .enumerate()
                .map(|(y, p)| {
                    p * adm.iter().map(|&a| values[y][column[a]]).fold(f64::NEG_INFINITY, f64::max)
                })
                .sum();
            Ok(Recomputed {
                evsi: Some(after - base.prior_best),
                evpi: base.evpi,
                replay: ReplayKind::AttestedValuesCombined,
            })
        }
        _ => Err("integration"),
    }
}

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
    /// Provider trust label (never upgraded by consuming).
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
    pub ranking: DesignRanking,
    /// The artifact digest.
    pub identity: String,
    /// Candidates in rank order.
    pub candidates: Vec<ConsumedCandidate>,
    /// `unmeasured`.
    pub calibration: &'static str,
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
    ) -> std::result::Result<antecedent_core::ProvenanceChain, antecedent_core::ProvenanceChainError>
    {
        use antecedent_core::{CompositionLink, CompositionStage, ProvenanceChain};
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
        let mut candidates: Vec<&CandidateWire> = self.candidates.iter().collect();
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
        let mut hasher = blake3::Hasher::new();
        put_str(&mut hasher, "antecedent.design_ranking_artifact.v1");
        hasher.update(&bytes);
        Ok(hasher.finalize().to_hex().to_string())
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

    /// Serialize through the checksummed container, candidates in rank order.
    ///
    /// # Errors
    ///
    /// A blank artifact id or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> std::result::Result<Vec<u8>, IoError> {
        if blank(artifact_id) {
            return Err(IoError::Convert("missing artifact id".into()));
        }
        let mut ordered = self.clone();
        ordered
            .candidates
            .sort_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.semantic_id.cmp(&b.semantic_id)));
        let body = to_cbor(&ordered)?;
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: antecedent_io::migrate::STABLE_FORMAT,
                minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(DESIGN_RANKING_ARTIFACT_KIND.into()),
                library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
                artifact_id: artifact_id.into(),
                sections: vec![section_descriptor(BODY_SECTION, "application/cbor", &body)],
                provenance: ProvenanceWire { note: "design_ranking".into() },
            },
            sections: vec![SectionBytes::new(BODY_SECTION, body)],
        };
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes)?;
        if bytes.len() > MAX_DESIGN_RANKING_ARTIFACT_BYTES {
            return Err(IoError::TooLarge);
        }
        Ok(bytes)
    }

    /// Decode the container and body, refusing oversize input before any allocation.
    ///
    /// # Errors
    ///
    /// Corruption, truncation, oversize, an unknown layout, or an unknown major version.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_DESIGN_RANKING_ARTIFACT_BYTES {
            return Err(IoError::TooLarge.into());
        }
        let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
        let manifest = reader.manifest();
        if manifest.artifact_kind != ArtifactKind::Other(DESIGN_RANKING_ARTIFACT_KIND.into())
            || manifest.sections.len() != 1
            || manifest.sections[0].id != BODY_SECTION
        {
            return Err(
                IoError::Convert("unsupported design ranking artifact layout".into()).into()
            );
        }
        if manifest.sections[0].uncompressed_size > MAX_DESIGN_RANKING_ARTIFACT_BYTES as u64 {
            return Err(IoError::TooLarge.into());
        }
        let section = reader.load_section(BODY_SECTION)?;
        let wire: Self = from_cbor(section.as_bytes())?;
        if wire.version != DESIGN_RANKING_ARTIFACT_VERSION {
            return Err(DesignRankingArtifactError::UnsupportedVersion(wire.version));
        }
        Ok(wire)
    }
}

// -- sealing ----------------------------------------------------------------------------

/// Inputs of a ranking evaluation, retained by [`seal`].
pub struct SealInputs<'a, A> {
    /// Decision problem the EVSI values information for.
    pub problem: &'a DecisionProblem<A, f64>,
    /// Belief the decision is made under.
    pub prior: &'a DecisionPrior<f64>,
    /// The evaluated request (candidates and their providers).
    pub request: &'a EvsiRequest,
    /// The evaluation result.
    pub report: &'a EvsiReport,
    /// Digests of the source distributions (prior sources and external laws).
    pub source_digests: &'a [String],
}

fn decision_wire<A: Clone>(inputs: &SealInputs<'_, A>) -> Result<DecisionWire> {
    let problem = inputs.problem;
    let (prior, utility, admissible) = match inputs.prior {
        DecisionPrior::Draws(states) => {
            let table = DecisionTable::build(problem, states)?;
            let rows: Vec<Vec<f64>> = (0..problem.actions.len())
                .map(|a| (0..table.n_outcomes).map(|k| table.utility(a, k)).collect())
                .collect();
            let admissible =
                (0..problem.actions.len()).map(|a| table.admissible.contains(&a)).collect();
            (PriorWire::Draws { states: states.clone() }, UtilityWire::Table { rows }, admissible)
        }
        DecisionPrior::Normal { mean, variance } => {
            let coefficients = problem
                .utility
                .affine_in_state(&problem.actions)
                .filter(|c| c.len() == problem.actions.len())
                .ok_or_else(|| {
                    DesignError::Config(
                        "a normal belief needs a utility that declares affine coefficients".into(),
                    )
                })?;
            (
                PriorWire::Normal { mean: *mean, variance: *variance },
                UtilityWire::Affine { coefficients },
                vec![true; problem.actions.len()],
            )
        }
    };
    Ok(DecisionWire {
        contract_identity: inputs.request.decision_contract_identity.clone(),
        utility_unit: inputs.request.utility_unit.clone(),
        action_ids: inputs.request.action_ids.clone(),
        prior,
        utility,
        admissible,
    })
}

fn likelihood_law(
    family: &str,
    mode: SignalUpdateMode,
    signal: &dyn DecisionSignal<f64>,
    states: Option<&[f64]>,
    n: u64,
) -> Result<LawWire> {
    let mismatch = || DesignRankingArtifactError::InvalidWire("signal law is not retainable");
    match family {
        "gaussian_mean" => Ok(LawWire::GaussianMean {
            noise_variance: signal.gaussian_noise_variance().ok_or_else(mismatch)?,
        }),
        "binomial" => Ok(LawWire::Binomial),
        _ => {
            let states = states.ok_or_else(mismatch)?;
            let statistics = signal.finite_support(n).ok_or_else(mismatch)?;
            let mut table = Vec::with_capacity(statistics.len());
            for &y in &statistics {
                let mut log_lik = vec![0.0; states.len()];
                signal.log_likelihood(y, n, states, &mut log_lik)?;
                table.push(log_lik.iter().map(|ll| ll.exp()).collect::<Vec<f64>>());
            }
            if mode == SignalUpdateMode::ExternalPosterior {
                let k = states.len() as f64;
                let predictive: Vec<f64> =
                    table.iter().map(|row| row.iter().sum::<f64>() / k).collect();
                let posterior = table
                    .iter()
                    .zip(&predictive)
                    .map(|(row, p)| {
                        row.iter().map(|l| if *p > 0.0 { l / (k * p) } else { 1.0 / k }).collect()
                    })
                    .collect();
                Ok(LawWire::FinitePosterior { statistics, predictive, posterior })
            } else {
                Ok(LawWire::FiniteLikelihood { statistics, probabilities: table })
            }
        }
    }
}

fn request_wire(request: &SignalRequest) -> SignalRequestWire {
    let mut lineage = request.evidence_lineage.clone();
    lineage.sort();
    SignalRequestWire {
        candidate_id: request.candidate_id.clone(),
        prior_id: request.prior_id.clone(),
        state_quantity: ScientificQuantityWire::from(&request.state_quantity),
        observation_quantity: ScientificQuantityWire::from(&request.observation_quantity),
        rng_seed: request.rng_seed,
        evidence_lineage: lineage,
        conditional_independence: request.conditional_independence.clone(),
        max_sample_size: request.limits.max_sample_size,
        max_support: request.limits.max_support as u64,
    }
}

fn provider_wire(identity: &ProviderObjectIdentity) -> ProviderWire {
    ProviderWire {
        provider_id: identity.provider_id.clone(),
        object_id: identity.object_id.clone(),
        version_id: identity.version_id.clone(),
        snapshot_id: identity.snapshot_id.clone(),
        request_id: identity.request_id.clone(),
    }
}

/// Seal an evaluated ranking as a durable artifact body.
///
/// Each candidate's provider is asked to answer its request again (providers are
/// deterministic functions of the exact request) so that the signal law behind the
/// value can be retained; the answer's receipt identity must equal the evaluated one.
///
/// # Errors
///
/// A provider that no longer answers its request identically, a law that cannot be
/// retained, or an empty report.
pub fn seal<A: Clone>(inputs: &SealInputs<'_, A>) -> Result<DesignRankingArtifactWire> {
    let report = inputs.report;
    let ranking = DesignRanking::from_evsi(report, inputs.source_digests)?;
    let decision = decision_wire(inputs)?;
    let mut candidates = Vec::with_capacity(report.candidates.len());
    for row in &report.candidates {
        let declared =
            inputs.request.candidates.iter().find(|c| c.semantic_id == row.semantic_id).ok_or(
                DesignRankingArtifactError::InvalidWire("report names an unknown candidate"),
            )?;
        let prepared = declared.provider.prepare(&declared.signal_request)?;
        if prepared.receipt.identity() != row.signal_receipt.identity() {
            return Err(DesignRankingArtifactError::SignalInconsistent {
                candidate: row.semantic_id.clone(),
                slot: "provider_not_reproducible",
            });
        }
        let receipt = &row.signal_receipt;
        let law = match &prepared.law {
            PreparedLaw::Likelihood(signal) => likelihood_law(
                &receipt.family,
                receipt.update_mode,
                signal.as_ref(),
                prepared.bound_states.as_deref(),
                row.sample_size,
            )?,
            PreparedLaw::DecisionValues(values) => LawWire::DecisionValues {
                branch_probabilities: values.branch_probabilities.clone(),
                action_ids: values.action_ids.clone(),
                values: values.values.clone(),
            },
        };
        let mut lineage = receipt.evidence_lineage.clone();
        lineage.sort();
        candidates.push(CandidateWire {
            semantic_id: row.semantic_id.clone(),
            sample_size: row.sample_size,
            cost_amount: row.study_cost.amount,
            cost_unit: row.study_cost.unit.clone(),
            reused_observation_ids: declared.reused_observation_ids.clone(),
            request: request_wire(&declared.signal_request),
            family: receipt.family.clone(),
            update_mode: receipt.update_mode.as_str().to_owned(),
            trust: receipt.trust.as_str().to_owned(),
            attestor: receipt.attestor.clone(),
            provider: receipt.provider_identity.as_ref().map(provider_wire),
            law_digest: receipt.law_digest.clone(),
            request_fingerprint: receipt.request_fingerprint.clone(),
            signal_identity: receipt.identity(),
            law,
            evsi: row.evsi,
            evpi: row.evpi,
            integration: row.integration.method.as_str().to_owned(),
            replicates: row.integration.replicates,
            stderr: row.integration.stderr,
            ess: row.integration.ess,
            converged: row.integration.converged,
            early_stopped: row.integration.early_stopped,
            study_cost_utility: row.study_cost_utility,
            net_value: row.net_value,
            rank: row.rank as u64,
            rank_uncertain: row.rank_uncertain,
            overlap_checked: row.source_overlap.observations_checked as u64,
            overlapping: row.source_overlap.overlapping.clone(),
            assumptions: row.assumptions.clone(),
        });
    }
    let config = &inputs.request.rank_config;
    let mut wire = DesignRankingArtifactWire {
        version: DESIGN_RANKING_ARTIFACT_VERSION,
        kind: DESIGN_RANKING_ARTIFACT_KIND.to_owned(),
        calibration: DESIGN_RANKING_CALIBRATION.to_owned(),
        basis: ranking.basis.as_str().to_owned(),
        decision,
        cost_map: report.cost_map.as_ref().map(|m| CostMapWire {
            cost_unit: m.cost_unit.clone(),
            utility_unit: m.utility_unit.clone(),
            utility_per_cost: m.utility_per_cost,
        }),
        rng_seed: report.rng_seed,
        mc_error_tolerance: inputs.request.mc_error_tolerance,
        tie_tolerance: inputs.request.tie_tolerance,
        max_candidates: inputs.request.max_candidates as u64,
        rank_config: RankConfigWire {
            min_batches: config.min_batches,
            max_batches: config.max_batches,
            batch_size: config.batch_size,
            rank_uncertainty_threshold: config.rank_uncertainty_threshold,
        },
        prior_observation_ids: inputs.request.prior_observation_ids.clone(),
        source_digests: ranking.source_digests.clone(),
        bayes_action: report.bayes_action as u64,
        prior_expected_utility: report.prior_expected_utility,
        evpi: report.evpi,
        candidates,
        ties: report.ties.clone(),
        search: SearchWire {
            supplied: report.search.supplied as u64,
            evaluated: report.search.evaluated as u64,
            truncated: report.search.truncated,
            unevaluated_ids: report.search.unevaluated_ids.clone(),
        },
        ranking_identity: ranking.identity(),
        digest: String::new(),
    };
    wire.reseal()?;
    Ok(wire)
}

// -- consuming --------------------------------------------------------------------------

fn check_bounds(wire: &DesignRankingArtifactWire) -> Result<()> {
    if wire.candidates.is_empty() {
        return Err(DesignRankingArtifactError::InvalidWire("no candidates"));
    }
    if wire.candidates.len() > MAX_RANKING_CANDIDATES
        || !usize::try_from(wire.max_candidates).is_ok_and(|m| m <= MAX_RANKING_CANDIDATES)
    {
        return Err(DesignRankingArtifactError::Bounds("candidates"));
    }
    let actions = wire.decision.action_ids.len();
    if actions == 0 || actions > MAX_RANKING_STATES {
        return Err(DesignRankingArtifactError::Bounds("actions"));
    }
    let states = match &wire.decision.prior {
        PriorWire::Draws { states } => states.len(),
        PriorWire::Normal { .. } => 1,
    };
    if states == 0
        || states > MAX_RANKING_STATES
        || states.saturating_mul(actions) > MAX_RANKING_CELLS
    {
        return Err(DesignRankingArtifactError::Bounds("decision_table"));
    }
    for candidate in &wire.candidates {
        let support = match &candidate.law {
            LawWire::GaussianMean { .. } => 0,
            LawWire::Binomial => usize::try_from(candidate.sample_size)
                .ok()
                .and_then(|n| n.checked_add(1))
                .ok_or(DesignRankingArtifactError::Bounds("support"))?,
            LawWire::FiniteLikelihood { statistics, .. }
            | LawWire::FinitePosterior { statistics, .. } => statistics.len(),
            LawWire::DecisionValues { branch_probabilities, .. } => branch_probabilities.len(),
        };
        if support > MAX_RANKING_STATES || support.saturating_mul(states) > MAX_RANKING_CELLS {
            return Err(DesignRankingArtifactError::Bounds("signal_table"));
        }
    }
    Ok(())
}

fn check_decision(decision: &DecisionWire) -> Result<()> {
    let bad = DesignRankingArtifactError::InvalidWire;
    if blank(&decision.contract_identity) || blank(&decision.utility_unit) {
        return Err(bad("decision identity and utility unit"));
    }
    let n_actions = decision.action_ids.len();
    let unique: BTreeSet<&String> = decision.action_ids.iter().collect();
    if unique.len() != n_actions || decision.action_ids.iter().any(|id| blank(id)) {
        return Err(bad("action ids"));
    }
    if decision.admissible.len() != n_actions || !decision.admissible.iter().any(|ok| *ok) {
        return Err(bad("admissible actions"));
    }
    match (&decision.prior, &decision.utility) {
        (PriorWire::Draws { states }, UtilityWire::Table { rows }) => {
            if !states.iter().all(|s| s.is_finite())
                || !rectangular(rows, n_actions, states.len())
                || !rows.iter().all(|row| row.iter().all(|u| u.is_finite()))
            {
                return Err(bad("decision table"));
            }
        }
        (PriorWire::Normal { mean, variance }, UtilityWire::Affine { coefficients }) => {
            if !(mean.is_finite() && variance.is_finite() && *variance > 0.0)
                || coefficients.len() != n_actions
                || !coefficients.iter().all(|(a, b)| a.is_finite() && b.is_finite())
                || decision.admissible.iter().any(|ok| !ok)
            {
                return Err(bad("normal decision"));
            }
        }
        _ => return Err(bad("prior and utility representation disagree")),
    }
    Ok(())
}

fn check_costs(wire: &DesignRankingArtifactWire) -> Result<()> {
    let utility_unit = &wire.decision.utility_unit;
    if let Some(map) = &wire.cost_map {
        if blank(&map.cost_unit)
            || blank(&map.utility_unit)
            || !(map.utility_per_cost.is_finite() && map.utility_per_cost > 0.0)
        {
            return Err(DesignRankingArtifactError::InvalidWire("cost mapping"));
        }
        if &map.utility_unit != utility_unit {
            return Err(DesignRankingArtifactError::CostUnitsMismatch(format!(
                "map utility unit {} differs from decision utility unit {utility_unit}",
                map.utility_unit
            )));
        }
    }
    for candidate in &wire.candidates {
        if !(candidate.cost_amount.is_finite() && candidate.cost_amount >= 0.0)
            || blank(&candidate.cost_unit)
        {
            return Err(DesignRankingArtifactError::InvalidWire("study cost"));
        }
        if let Some(map) = &wire.cost_map {
            if candidate.cost_unit != map.cost_unit {
                return Err(DesignRankingArtifactError::CostUnitsMismatch(format!(
                    "candidate {} cost unit {} differs from map cost unit {}",
                    candidate.semantic_id, candidate.cost_unit, map.cost_unit
                )));
            }
        }
    }
    Ok(())
}

fn signal_request_of(candidate: &CandidateWire) -> Option<SignalRequest> {
    let wire = &candidate.request;
    let state = ScientificQuantity::try_from(wire.state_quantity.clone()).ok()?;
    let observation = ScientificQuantity::try_from(wire.observation_quantity.clone()).ok()?;
    Some(SignalRequest {
        candidate_id: wire.candidate_id.clone(),
        prior_id: wire.prior_id.clone(),
        state_quantity: state,
        observation_quantity: observation,
        sample_size: candidate.sample_size,
        rng_seed: wire.rng_seed,
        evidence_lineage: wire.evidence_lineage.clone(),
        conditional_independence: wire.conditional_independence.clone(),
        limits: SignalLimits {
            max_sample_size: wire.max_sample_size,
            max_support: usize::try_from(wire.max_support).ok()?,
        },
    })
}

/// Trust-limit sentence for the consumed candidate.
fn trust_limit(trust: SignalTrustLabel, attestor: Option<&str>) -> String {
    match trust {
        SignalTrustLabel::NativeLicensed => {
            "native_licensed: implemented and licensed natively".to_owned()
        }
        SignalTrustLabel::ExternallyAttested => format!(
            "externally_attested by {}: supplier assertion only, not verified or natively replayed by Antecedent",
            attestor.unwrap_or("unnamed attestor")
        ),
        SignalTrustLabel::ExactRequestVerified => "exact_request_verified: the verification receipt is not retained in this artifact; the signal is still not native and was not natively replayed".to_owned(),
    }
}

fn check_signal(candidate: &CandidateWire) -> Result<(SignalTrustLabel, SignalUpdateMode)> {
    let id = &candidate.semantic_id;
    let inconsistent = |slot: &'static str| DesignRankingArtifactError::SignalInconsistent {
        candidate: id.clone(),
        slot,
    };
    let request = signal_request_of(candidate).ok_or_else(|| inconsistent("request"))?;
    request.validate().map_err(|_| inconsistent("request"))?;
    if request.fingerprint() != candidate.request_fingerprint {
        return Err(inconsistent("request_fingerprint"));
    }
    let mode = parse_mode(&candidate.update_mode).ok_or_else(|| inconsistent("update_mode"))?;
    let trust = parse_trust(&candidate.trust).ok_or_else(|| inconsistent("trust"))?;
    // Mode, trust, provider and law must tell one story.
    let law_mode = match &candidate.law {
        LawWire::GaussianMean { .. } | LawWire::Binomial | LawWire::FiniteLikelihood { .. } => {
            SignalUpdateMode::NativeUpdate
        }
        LawWire::FinitePosterior { .. } => SignalUpdateMode::ExternalPosterior,
        LawWire::DecisionValues { .. } => SignalUpdateMode::ExternalDecisionValues,
    };
    if mode != law_mode {
        return Err(inconsistent("update_mode"));
    }
    if is_native(&candidate.law) {
        let family = match &candidate.law {
            LawWire::GaussianMean { .. } => "gaussian_mean",
            _ => "binomial",
        };
        let parameters: Vec<f64> = match &candidate.law {
            LawWire::GaussianMean { noise_variance } => vec![*noise_variance],
            _ => Vec::new(),
        };
        if trust != SignalTrustLabel::NativeLicensed
            || candidate.provider.is_some()
            || candidate.attestor.is_some()
            || candidate.family != family
        {
            return Err(inconsistent("trust"));
        }
        if candidate.law_digest != native_digest(family, &parameters) {
            return Err(inconsistent("law_digest"));
        }
    } else {
        let Some(provider) = &candidate.provider else {
            return Err(inconsistent("provider"));
        };
        if trust == SignalTrustLabel::NativeLicensed || candidate.family != "external_finite_law" {
            return Err(inconsistent("trust"));
        }
        if trust == SignalTrustLabel::ExternallyAttested
            && candidate.attestor.as_deref().is_none_or(blank)
        {
            return Err(inconsistent("attestor"));
        }
        if provider.request_id != candidate.request_fingerprint {
            return Err(inconsistent("provider_request"));
        }
        if blank(&provider.provider_id) || blank(&provider.object_id) {
            return Err(inconsistent("provider"));
        }
    }
    let mut lineage = candidate.request.evidence_lineage.clone();
    lineage.sort();
    let receipt = SignalReceipt {
        request_fingerprint: candidate.request_fingerprint.clone(),
        candidate_id: candidate.request.candidate_id.clone(),
        prior_id: candidate.request.prior_id.clone(),
        sample_size: candidate.sample_size,
        rng_seed: candidate.request.rng_seed,
        family: candidate.family.clone(),
        update_mode: mode,
        trust,
        attestor: candidate.attestor.clone(),
        provider_identity: candidate.provider.as_ref().map(|p| ProviderObjectIdentity {
            provider_id: p.provider_id.clone(),
            object_id: p.object_id.clone(),
            version_id: p.version_id.clone(),
            snapshot_id: p.snapshot_id.clone(),
            request_id: p.request_id.clone(),
        }),
        evidence_lineage: lineage,
        conditional_independence: candidate.request.conditional_independence.clone(),
        limits: request.limits,
        law_digest: candidate.law_digest.clone(),
    };
    if receipt.identity() != candidate.signal_identity {
        return Err(inconsistent("signal_identity"));
    }
    Ok((trust, mode))
}

fn verify_candidate(
    wire: &DesignRankingArtifactWire,
    candidate: &CandidateWire,
    base: &Baseline,
) -> Result<ConsumedCandidate> {
    let id = &candidate.semantic_id;
    let mismatch = |quantity: &'static str| DesignRankingArtifactError::ReplayMismatch {
        candidate: id.clone(),
        quantity,
    };
    if blank(id) {
        return Err(DesignRankingArtifactError::InvalidWire("blank candidate id"));
    }
    let (trust, mode) = check_signal(candidate)?;
    // Source overlap: recomputed, never trusted from the stored diagnostic.
    let prior: BTreeSet<&str> = wire.prior_observation_ids.iter().map(String::as_str).collect();
    let reused: BTreeSet<&str> =
        candidate.reused_observation_ids.iter().map(String::as_str).collect();
    let overlapping: Vec<String> = prior.intersection(&reused).map(|s| (*s).to_owned()).collect();
    if !overlapping.is_empty() {
        return Err(DesignRankingArtifactError::SourceOverlap {
            candidate: id.clone(),
            ids: overlapping,
        });
    }
    if !candidate.overlapping.is_empty()
        || candidate.overlap_checked != prior.union(&reused).count() as u64
    {
        return Err(mismatch("source_overlap"));
    }
    let method = parse_method(&candidate.integration)
        .ok_or(DesignRankingArtifactError::InvalidWire("integration method"))?;
    let recomputed = recompute(candidate, &wire.decision, base, method).map_err(|slot| {
        DesignRankingArtifactError::SignalInconsistent { candidate: id.clone(), slot }
    })?;
    if !close(candidate.evpi, recomputed.evpi) {
        return Err(mismatch("evpi"));
    }
    if let Some(evsi) = recomputed.evsi {
        if !close(candidate.evsi, evsi) {
            return Err(mismatch("evsi"));
        }
        if candidate.stderr.abs() > 0.0 || candidate.replicates != 0 || candidate.ess.is_some() {
            return Err(mismatch("integration_error"));
        }
    } else {
        // Monte Carlo: bound only, error and replicates retained.
        let tolerance = 1e-9 + 4.0 * candidate.stderr;
        if !(candidate.stderr.is_finite() && candidate.stderr >= 0.0 && candidate.replicates > 0)
            || candidate.ess.is_none()
        {
            return Err(mismatch("integration_error"));
        }
        if candidate.evsi < -tolerance || candidate.evsi > candidate.evpi + tolerance {
            return Err(mismatch("evsi_bounds"));
        }
        if candidate.converged != (candidate.stderr <= wire.mc_error_tolerance) {
            return Err(mismatch("converged"));
        }
    }
    let tolerance = 1e-9 + 4.0 * candidate.stderr;
    if candidate.evsi < -tolerance || candidate.evsi > candidate.evpi + tolerance {
        return Err(mismatch("evsi_bounds"));
    }
    match &wire.cost_map {
        Some(map) => {
            let cost_utility = map.utility_per_cost * candidate.cost_amount;
            match (candidate.study_cost_utility, candidate.net_value) {
                (Some(stored_cost), Some(stored_net))
                    if close(stored_cost, cost_utility)
                        && close(stored_net, candidate.evsi - cost_utility) => {}
                _ => return Err(mismatch("net_value")),
            }
        }
        None => {
            if candidate.study_cost_utility.is_some() || candidate.net_value.is_some() {
                return Err(mismatch("net_value"));
            }
        }
    }
    Ok(ConsumedCandidate {
        semantic_id: id.clone(),
        replay: recomputed.replay,
        trust,
        update_mode: mode,
        trust_limit: trust_limit(trust, candidate.attestor.as_deref()),
    })
}

fn rank_key(candidate: &CandidateWire) -> f64 {
    candidate.net_value.unwrap_or(candidate.evsi)
}

/// Recompute the order, rank-uncertainty flags and ties from the stored values.
fn verify_order(wire: &DesignRankingArtifactWire) -> Result<()> {
    let mut rows: Vec<&CandidateWire> = wire.candidates.iter().collect();
    rows.sort_by(|a, b| {
        rank_key(b).total_cmp(&rank_key(a)).then_with(|| a.semantic_id.cmp(&b.semantic_id))
    });
    let mut ties = Vec::new();
    let mut near = vec![false; rows.len()];
    for i in 0..rows.len().saturating_sub(1) {
        let gap = (rank_key(rows[i]) - rank_key(rows[i + 1])).abs();
        let se = rows[i].stderr.hypot(rows[i + 1].stderr);
        if gap <= wire.tie_tolerance {
            ties.push((rows[i].semantic_id.clone(), rows[i + 1].semantic_id.clone()));
        }
        if gap <= wire.tie_tolerance.max(RANK_Z * se) {
            near[i] = true;
            near[i + 1] = true;
        }
    }
    for (i, row) in rows.iter().enumerate() {
        if row.rank != i as u64 {
            return Err(DesignRankingArtifactError::RankMismatch("rank"));
        }
        if row.rank_uncertain != near[i] {
            return Err(DesignRankingArtifactError::RankMismatch("rank_uncertain"));
        }
    }
    let mut stored_ties = wire.ties.clone();
    stored_ties.sort();
    ties.sort();
    if stored_ties != ties {
        return Err(DesignRankingArtifactError::RankMismatch("ties"));
    }
    Ok(())
}

fn verify_search(wire: &DesignRankingArtifactWire) -> Result<()> {
    let search = &wire.search;
    let ids: BTreeSet<&str> = wire.candidates.iter().map(|c| c.semantic_id.as_str()).collect();
    if ids.len() != wire.candidates.len() {
        return Err(DesignRankingArtifactError::InvalidWire("duplicate candidate id"));
    }
    if search.evaluated != wire.candidates.len() as u64
        || search.truncated == search.unevaluated_ids.is_empty()
        || search.supplied != search.evaluated + search.unevaluated_ids.len() as u64
        || search.evaluated > wire.max_candidates
        || search.unevaluated_ids.iter().any(|id| ids.contains(id.as_str()))
    {
        return Err(DesignRankingArtifactError::RankMismatch("search_receipt"));
    }
    Ok(())
}

fn rebuild_ranking(wire: &DesignRankingArtifactWire) -> Result<DesignRanking> {
    let basis =
        rank_basis(&wire.basis).ok_or(DesignRankingArtifactError::InvalidWire("ranking basis"))?;
    if (basis == DesignRankingBasis::NetValue) != wire.cost_map.is_some() {
        return Err(DesignRankingArtifactError::RankMismatch("basis"));
    }
    let mut entries = Vec::with_capacity(wire.candidates.len());
    for c in &wire.candidates {
        let invalid = DesignRankingArtifactError::InvalidWire;
        entries.push(RankingEntry {
            semantic_id: c.semantic_id.clone(),
            rank: usize::try_from(c.rank).map_err(|_| invalid("rank"))?,
            evsi: c.evsi,
            mc_stderr: c.stderr,
            replicates: c.replicates,
            integration: parse_method(&c.integration).ok_or_else(|| invalid("integration"))?,
            evpi: c.evpi,
            sample_size: c.sample_size,
            study_cost_amount: c.cost_amount,
            study_cost_unit: c.cost_unit.clone(),
            net_value: c.net_value,
            rank_uncertain: c.rank_uncertain,
            signal_identity: c.signal_identity.clone(),
            request_fingerprint: c.request_fingerprint.clone(),
            update_mode: parse_mode(&c.update_mode).ok_or_else(|| invalid("update_mode"))?,
            provider_trust: parse_trust(&c.trust).ok_or_else(|| invalid("trust"))?,
        });
    }
    entries.sort_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.semantic_id.cmp(&b.semantic_id)));
    let mut digests = wire.source_digests.clone();
    digests.sort();
    digests.dedup();
    Ok(DesignRanking {
        basis,
        decision_contract_identity: Some(wire.decision.contract_identity.clone()),
        utility_unit: Some(wire.decision.utility_unit.clone()),
        cost_mapping: wire.cost_map.as_ref().map(|m| CostToUtilityMap {
            cost_unit: m.cost_unit.clone(),
            utility_unit: m.utility_unit.clone(),
            utility_per_cost: m.utility_per_cost,
        }),
        rng_seed: Some(wire.rng_seed),
        source_digests: digests,
        entries,
        structural: Vec::new(),
        search: SearchReceipt {
            supplied: usize::try_from(wire.search.supplied)
                .map_err(|_| DesignRankingArtifactError::InvalidWire("search receipt"))?,
            evaluated: usize::try_from(wire.search.evaluated)
                .map_err(|_| DesignRankingArtifactError::InvalidWire("search receipt"))?,
            truncated: wire.search.truncated,
            unevaluated_ids: wire.search.unevaluated_ids.clone(),
        },
    })
}

/// Verify a decoded artifact body: digest, bounds, decision, costs, every candidate's
/// signal and recomputable value, order, search receipt and the retained expectations.
///
/// # Errors
///
/// The first inconsistency found; see [`DesignRankingArtifactError`].
pub fn consume_wire(
    wire: &DesignRankingArtifactWire,
    expectation: &ConsumeExpectation,
) -> Result<ConsumedDesignRanking> {
    if wire.version != DESIGN_RANKING_ARTIFACT_VERSION {
        return Err(DesignRankingArtifactError::UnsupportedVersion(wire.version));
    }
    if wire.kind != DESIGN_RANKING_ARTIFACT_KIND {
        return Err(DesignRankingArtifactError::InvalidWire("kind"));
    }
    check_bounds(wire)?;
    let digest = wire.compute_digest()?;
    if digest != wire.digest {
        return Err(DesignRankingArtifactError::DigestMismatch);
    }
    if let Some(expected) = &expectation.artifact_identity {
        if *expected != digest {
            return Err(DesignRankingArtifactError::ExpectedIdentityMismatch);
        }
    }
    if wire.calibration != DESIGN_RANKING_CALIBRATION {
        return Err(DesignRankingArtifactError::InvalidWire("calibration label"));
    }
    if !(wire.mc_error_tolerance.is_finite()
        && wire.mc_error_tolerance >= 0.0
        && wire.tie_tolerance.is_finite()
        && wire.tie_tolerance >= 0.0)
    {
        return Err(DesignRankingArtifactError::InvalidWire("tolerances"));
    }
    check_decision(&wire.decision)?;
    check_costs(wire)?;
    verify_search(wire)?;
    let base = baseline(&wire.decision);
    if wire.bayes_action != base.bayes as u64
        || !close(wire.prior_expected_utility, base.prior_best)
        || !close(wire.evpi, base.evpi)
    {
        return Err(DesignRankingArtifactError::ReplayMismatch {
            candidate: String::new(),
            quantity: "decision_baseline",
        });
    }
    let mut consumed = Vec::with_capacity(wire.candidates.len());
    for candidate in &wire.candidates {
        consumed.push(verify_candidate(wire, candidate, &base)?);
    }
    verify_order(wire)?;
    let ranking = rebuild_ranking(wire)?;
    if ranking.identity() != wire.ranking_identity {
        return Err(DesignRankingArtifactError::RankMismatch("ranking_identity"));
    }
    let signals: BTreeMap<String, String> =
        expectation.signal_identities.clone().unwrap_or_else(|| {
            ranking
                .entries
                .iter()
                .map(|e| (e.semantic_id.clone(), e.signal_identity.clone()))
                .collect()
        });
    let digests =
        expectation.source_digests.clone().unwrap_or_else(|| ranking.source_digests.clone());
    let contract = expectation
        .decision_contract_identity
        .clone()
        .unwrap_or_else(|| wire.decision.contract_identity.clone());
    let cost_mapping =
        expectation.cost_mapping.clone().unwrap_or_else(|| ranking.cost_mapping.clone());
    ranking.verify_replay(&contract, &signals, &digests, cost_mapping.as_ref())?;
    if let Some(retained) = &expectation.signal_identities {
        let stored: BTreeSet<&String> = ranking.entries.iter().map(|e| &e.semantic_id).collect();
        if let Some(extra) = retained.keys().find(|id| !stored.contains(id)) {
            return Err(RankingError::SignalMismatch(extra.clone()).into());
        }
    }
    consumed.sort_by(|a, b| {
        let rank = |id: &str| ranking.entries.iter().position(|e| e.semantic_id == id);
        rank(&a.semantic_id).cmp(&rank(&b.semantic_id))
    });
    Ok(ConsumedDesignRanking {
        ranking,
        identity: digest,
        candidates: consumed,
        calibration: DESIGN_RANKING_CALIBRATION,
    })
}

/// Decode and verify exported bytes.
///
/// # Errors
///
/// Corruption, oversize, unknown version, or any inconsistency of
/// [`consume_wire`].
pub fn consume(bytes: &[u8], expectation: &ConsumeExpectation) -> Result<ConsumedDesignRanking> {
    let wire = DesignRankingArtifactWire::from_bytes(bytes)?;
    consume_wire(&wire, expectation)
}
