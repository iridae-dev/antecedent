//! Per-candidate expected value of sample information with declared integration,
//! cost mapping, rank uncertainty, provider trust and source-overlap diagnostics.
//!
//! Every candidate's signal comes from its [`SignalProvider`]. A provider that supplies
//! an observation law (native or external likelihood) is scored through the existing
//! [`DesignObjective::ReduceDecisionRegret`] path of [`DesignRanker`] (one
//! preposterior analysis per candidate, scheduled with `map_indexed`), with EVPI and the current
//! Bayes action read from [`PreposteriorAnalysis`]. A provider that supplies externally
//! computed per-branch decision values is evaluated as
//! `EVSI = Σ_y P(y)·max_{a∈F} V_y(a) − max_{a∈F} E[U(a,θ)]`, after the supplied values
//! are checked against the prior (`Σ_y P(y)·V_y(a) = E[U(a,θ)]` for every action) and
//! its action set is checked against the decision's. The terminal action set `F` is the
//! same before and after the information; a changed action set is a different problem
//! and refuses.
//!
//! `NetValueOfInformation = EVSI − StudyCostUtility` is reported only when the caller
//! supplies a [`CostToUtilityMap`] whose utility unit equals the decision's and whose
//! cost unit equals every candidate's; otherwise value and cost are reported
//! separately and incompatible units refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_contract::ExternalRefusal;
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, Overlap, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_core::{CancellationToken, ExecutionContext, is_blank};

use crate::candidate::CandidateDesign;
use crate::decision::{DecisionProblem, DecisionTable, evaluate_decision};
use crate::error::DesignError;
use crate::preposterior::{DecisionPrior, DecisionSignal, EvsiMonteCarlo, PreposteriorAnalysis};
use crate::rank_rows::{AdjacentRank, evsi_bound_violation, rank_rows};
use crate::ranker::{DesignRankConfig, DesignRanker};
use crate::signal::{
    ExternalDecisionValues, PreparedLaw, RefusalBuilder, SignalError, SignalProvider,
    SignalReceipt, SignalTrustLabel, SignalUpdateMode, remember_prepared, state_key,
};

/// Positive linear map from the caller's cost unit to the decision's utility unit:
/// `StudyCostUtility = utility_per_cost · cost`.
#[derive(Clone, Debug, PartialEq)]
pub struct CostToUtilityMap {
    /// Unit the study costs are declared in.
    pub cost_unit: String,
    /// Utility unit of the decision this maps into.
    pub utility_unit: String,
    /// Utility per unit cost (finite, > 0).
    pub utility_per_cost: f64,
}

/// Declared cost of a candidate study.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyCostSpec {
    /// Amount (finite, >= 0).
    pub amount: f64,
    /// Unit of the amount.
    pub unit: String,
}

/// One candidate study and the provider of its signal.
#[derive(Clone)]
pub struct EvsiCandidate {
    /// Stable semantic identity of the candidate.
    pub semantic_id: String,
    /// Candidate design (its declared sample size must equal the signal request's).
    pub design: CandidateDesign,
    /// Exact signal request for this candidate.
    pub signal_request: crate::signal::SignalRequest,
    /// Provider answering the request.
    pub provider: Arc<dyn SignalProvider>,
    /// Declared study cost.
    pub cost: StudyCostSpec,
    /// Identities of existing observations the study would reuse.
    pub reused_observation_ids: Vec<String>,
}

/// One EVSI evaluation request over a finite candidate catalog.
#[derive(Clone)]
pub struct EvsiRequest {
    /// Identity of the decision contract the problem implements.
    pub decision_contract_identity: String,
    /// Utility unit of the decision.
    pub utility_unit: String,
    /// Action identities aligned with the problem's actions.
    pub action_ids: Vec<String>,
    /// Candidates (any order; evaluation is canonical by semantic id).
    pub candidates: Vec<EvsiCandidate>,
    /// Cost-to-utility mapping, if one is licensed.
    pub cost_map: Option<CostToUtilityMap>,
    /// Refuse (instead of reporting value and cost separately) when no map is supplied.
    pub require_net_value: bool,
    /// Identities of observations already summarized by the prior.
    pub prior_observation_ids: Vec<String>,
    /// Monte Carlo configuration of the ranker.
    pub rank_config: DesignRankConfig,
    /// RNG seed of the evaluation.
    pub rng_seed: u64,
    /// Standard error at or below which a Monte Carlo estimate is called converged.
    pub mc_error_tolerance: f64,
    /// Absolute gap at or below which two candidates are tied.
    pub tie_tolerance: f64,
    /// Search bound: at most this many candidates (canonical order) are evaluated.
    pub max_candidates: usize,
}

/// How the EVSI integral was evaluated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegrationMethod {
    /// Finite enumeration or conjugate closed form.
    Exact,
    /// Monte Carlo over states and data.
    MonteCarlo,
    /// Computed by the provider; Antecedent only combined the supplied branches.
    ExternallyComputed,
}

impl IntegrationMethod {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::MonteCarlo => "monte_carlo",
            Self::ExternallyComputed => "externally_computed",
        }
    }
}

/// Integration error and convergence of one candidate's EVSI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrationReport {
    /// Method.
    pub method: IntegrationMethod,
    /// Monte Carlo replicates (zero when not Monte Carlo).
    pub replicates: u64,
    /// Standard error of the EVSI estimate (zero when not Monte Carlo).
    pub stderr: f64,
    /// Effective sample size; replicates are independent draws, so it equals the
    /// replicate count. `None` when not Monte Carlo.
    pub ess: Option<f64>,
    /// Exact methods always; Monte Carlo when `stderr` is within the tolerance.
    pub converged: bool,
    /// The ranker stopped before its batch cap.
    pub early_stopped: bool,
}

/// Overlap between the observations behind the prior and those a study reuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceOverlapDiagnostics {
    /// Distinct observation identities compared.
    pub observations_checked: usize,
    /// Shared identities; empty for every successful evaluation.
    pub overlapping: Vec<String>,
}

/// Report for one candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiCandidateRow {
    /// Semantic identity.
    pub semantic_id: String,
    /// Expected value of sample information.
    pub evsi: f64,
    /// Integration report.
    pub integration: IntegrationReport,
    /// Expected value of perfect information (an upper bound on EVSI).
    pub evpi: f64,
    /// Sample size.
    pub sample_size: u64,
    /// Declared study cost.
    pub study_cost: StudyCostSpec,
    /// Study cost in utility units, only with a valid cost map.
    pub study_cost_utility: Option<f64>,
    /// `EVSI − StudyCostUtility`, only with a valid cost map.
    pub net_value: Option<f64>,
    /// Rank position (0 best); ties share the canonical semantic-id order.
    pub rank: usize,
    /// The gap to a neighbouring candidate is within integration error or the tie
    /// tolerance.
    pub rank_uncertain: bool,
    /// Provider trust label.
    pub provider_trust: SignalTrustLabel,
    /// Update mode.
    pub update_mode: SignalUpdateMode,
    /// Provider receipt for the exact request.
    pub signal_receipt: SignalReceipt,
    /// Source-overlap diagnostics.
    pub source_overlap: SourceOverlapDiagnostics,
    /// Assumptions the value rests on.
    pub assumptions: Vec<String>,
}

/// What the ranking is sorted by.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankingBasis {
    /// EVSI (no cost mapping supplied).
    Evsi,
    /// Net value of information (valid cost mapping supplied).
    NetValue,
}

/// How much of the catalog was evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchReceipt {
    /// Candidates supplied.
    pub supplied: usize,
    /// Candidates evaluated.
    pub evaluated: usize,
    /// The bound cut the catalog; the ranking covers only the evaluated candidates.
    pub truncated: bool,
    /// Semantic ids not evaluated.
    pub unevaluated_ids: Vec<String>,
}

/// EVSI/preposterior receipt over the evaluated candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiReport {
    /// Decision contract identity.
    pub decision_contract_identity: String,
    /// Utility unit.
    pub utility_unit: String,
    /// Terminal action identities (sorted); identical before and after information.
    pub action_ids: Vec<String>,
    /// Index of the Bayes action under the current belief, in problem order.
    pub bayes_action: usize,
    /// Expected utility of the Bayes action under the current belief.
    pub prior_expected_utility: f64,
    /// Expected value of perfect information.
    pub evpi: f64,
    /// RNG seed.
    pub rng_seed: u64,
    /// Cost mapping used, if any.
    pub cost_map: Option<CostToUtilityMap>,
    /// What the candidates are sorted by.
    pub basis: RankingBasis,
    /// Candidates, best first.
    pub candidates: Vec<EvsiCandidateRow>,
    /// Adjacent pairs tied within the tie tolerance.
    pub ties: Vec<(String, String)>,
    /// Search receipt.
    pub search: SearchReceipt,
}

/// Why an EVSI evaluation refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvsiError {
    /// A signal request or provider answer was refused.
    Signal(SignalError),
    /// The underlying decision analysis failed.
    Design(DesignError),
    /// A request field is invalid.
    InvalidRequest(&'static str),
    /// Two candidates share a semantic id.
    DuplicateCandidate(String),
    /// A study reuses observations the prior already summarizes.
    SourceOverlap(Vec<String>),
    /// Whether a study reuses observations the prior summarizes cannot be computed:
    /// an undeclared observation set, an unbound candidate or an unknown source.
    /// Unknown dependence is never treated as independence.
    SourceOverlapUnknown {
        /// Candidate.
        candidate: String,
        /// Why the overlap is not computable.
        reason: String,
    },
    /// A candidate's declared sample size differs from its signal request's.
    SampleSizeMismatch(String),
    /// A provider's receipt answers a different request.
    ReceiptMismatch(String),
    /// An external law is defined on different prior draws than the decision's.
    PriorStatesMismatch(String),
    /// Externally computed decision values name a different terminal action set.
    ActionSetChanged,
    /// Externally computed decision values need a prior of draws.
    DecisionValuesNeedDraws,
    /// Externally computed decision values do not average to the prior values.
    DecisionValuesIncoherent(String),
    /// Net value was required without a cost-to-utility map.
    CostMapRequired,
    /// A cost unit is incompatible with the cost map or the map with the decision.
    CostUnitsMismatch(String),
    /// A declared study cost or map is not finite and nonnegative.
    InvalidCost(String),
    /// EVSI violates `0 <= EVSI <= EVPI` beyond its error.
    BoundViolation(&'static str),
    /// Cancellation was requested.
    Cancelled,
    /// The ranker produced no score for a candidate.
    Unscored(String),
}

impl From<DesignError> for EvsiError {
    fn from(error: DesignError) -> Self {
        match error {
            DesignError::Cancelled => Self::Cancelled,
            other => Self::Design(other),
        }
    }
}

impl From<SignalError> for EvsiError {
    fn from(error: SignalError) -> Self {
        Self::Signal(error)
    }
}

impl EvsiError {
    /// Structured refusal under the `evsi` namespace (signal refusals keep
    /// `signal_provider`).
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let signal = antecedent_core::reason_code!("design_signal_invalid");
        let units = antecedent_core::reason_code!("design_cost_units_mismatch");
        let at = |code, detail: &str| RefusalBuilder::new(code, "evaluate", detail);
        match self {
            Self::Signal(error) => error.to_refusal(),
            Self::Design(error) => {
                at(signal, "evsi.design_error").expected(error.to_string()).build()
            }
            Self::InvalidRequest(what) => at(
                antecedent_core::reason_code!("invalid_argument"),
                "evsi.invalid_request",
            )
            .offending(*what)
            .build(),
            Self::DuplicateCandidate(id) => {
                at(signal, "evsi.duplicate_candidate").offending(id.clone()).build()
            }
            Self::SourceOverlap(ids) => at(signal, "evsi.source_overlap")
                .offending(ids.join(","))
                .remedy("use only data not already summarized by the prior")
                .build(),
            Self::SourceOverlapUnknown { candidate, reason } => at(
                antecedent_core::reason_code!("evidence_overlap_unknown"),
                "evsi.source_overlap_unknown",
            )
            .offending(candidate.clone())
            .expected(reason.clone())
            .remedy("declare the observations behind the prior and the study in the evidence ledger")
            .build(),
            Self::SampleSizeMismatch(id) => {
                at(signal, "evsi.sample_size_mismatch").offending(id.clone()).build()
            }
            Self::ReceiptMismatch(id) => {
                at(signal, "evsi.receipt_mismatch").offending(id.clone()).build()
            }
            Self::PriorStatesMismatch(id) => {
                at(signal, "evsi.prior_states_mismatch").offending(id.clone()).build()
            }
            Self::ActionSetChanged => at(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                "evsi.action_set_changed",
            )
            .remedy("an action-set change is a separate decision problem")
            .build(),
            Self::DecisionValuesNeedDraws => at(signal, "evsi.decision_values_need_draws").build(),
            Self::DecisionValuesIncoherent(id) => {
                at(signal, "evsi.decision_values_incoherent").offending(id.clone()).build()
            }
            Self::CostMapRequired => at(units, "evsi.cost_map_required")
                .remedy("supply a cost-to-utility mapping or report value and cost separately")
                .build(),
            Self::CostUnitsMismatch(what) => at(units, "evsi.cost_units_mismatch")
                .offending(what.clone())
                .remedy(
                    "declare each study cost in the cost map's cost unit and map it into the decision's utility unit",
                )
                .build(),
            Self::InvalidCost(what) => at(
                antecedent_core::reason_code!("invalid_argument"),
                "evsi.invalid_cost",
            )
            .offending(what.clone())
            .build(),
            Self::BoundViolation(which) => {
                at(signal, "evsi.bound_violation").offending(*which).build()
            }
            Self::Cancelled => {
                at(antecedent_core::reason_code!("cancelled_no_claim"), "evsi.cancelled").build()
            }
            Self::Unscored(id) => at(signal, "evsi.unscored_candidate").offending(id.clone()).build(),
        }
    }
}

fn validate_costs(request: &EvsiRequest) -> Result<(), EvsiError> {
    if let Some(map) = &request.cost_map {
        if is_blank(&map.cost_unit)
            || is_blank(&map.utility_unit)
            || !(map.utility_per_cost.is_finite() && map.utility_per_cost > 0.0)
        {
            return Err(EvsiError::InvalidCost("cost_map".into()));
        }
        if map.utility_unit != request.utility_unit {
            return Err(EvsiError::CostUnitsMismatch(format!(
                "map utility unit {} differs from decision utility unit {}",
                map.utility_unit, request.utility_unit
            )));
        }
    } else if request.require_net_value {
        return Err(EvsiError::CostMapRequired);
    }
    for candidate in &request.candidates {
        let cost = &candidate.cost;
        if !(cost.amount.is_finite() && cost.amount >= 0.0) || is_blank(&cost.unit) {
            return Err(EvsiError::InvalidCost(candidate.semantic_id.clone()));
        }
        if let Some(map) = &request.cost_map {
            if cost.unit != map.cost_unit {
                return Err(EvsiError::CostUnitsMismatch(format!(
                    "candidate {} cost unit {} differs from map cost unit {}",
                    candidate.semantic_id, cost.unit, map.cost_unit
                )));
            }
        }
    }
    Ok(())
}

fn validate_request<A>(
    problem: &DecisionProblem<A, f64>,
    request: &EvsiRequest,
) -> Result<(), EvsiError> {
    if is_blank(&request.decision_contract_identity) || is_blank(&request.utility_unit) {
        return Err(EvsiError::InvalidRequest("decision identity and utility unit"));
    }
    if request.action_ids.len() != problem.actions.len()
        || request.action_ids.iter().any(|id| is_blank(id))
    {
        return Err(EvsiError::InvalidRequest("action_ids must align with the problem actions"));
    }
    let mut ids: Vec<&String> = request.action_ids.iter().collect();
    ids.sort();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(EvsiError::InvalidRequest("duplicate action id"));
    }
    if request.candidates.is_empty() || request.max_candidates == 0 {
        return Err(EvsiError::InvalidRequest("candidates and max_candidates"));
    }
    if !(request.mc_error_tolerance.is_finite()
        && request.mc_error_tolerance >= 0.0
        && request.tie_tolerance.is_finite()
        && request.tie_tolerance >= 0.0)
    {
        return Err(EvsiError::InvalidRequest("tolerances"));
    }
    let mut seen: Vec<&String> = request.candidates.iter().map(|c| &c.semantic_id).collect();
    seen.sort();
    if let Some(pair) = seen.windows(2).find(|w| w[0] == w[1]) {
        return Err(EvsiError::DuplicateCandidate(pair[0].clone()));
    }
    if request.candidates.iter().any(|c| is_blank(&c.semantic_id)) {
        return Err(EvsiError::InvalidRequest("blank candidate id"));
    }
    validate_costs(request)
}

/// Domain under which declared observation identities become ledger observations.
const OBSERVATION_DOMAIN: &str = "antecedent.design.declared_observation";

/// One side of an overlap check: the observation identities the producer declared.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DeclaredObservations<'a> {
    /// Identities the source rests on (an empty set declares no data origin).
    Declared(&'a [String]),
}

/// Result of resolving the overlap of two declared observation sets through the
/// evidence ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ObservationOverlap {
    /// Known and disjoint.
    Independent {
        /// Distinct identities compared.
        checked: usize,
    },
    /// Known and sharing identities.
    Shared {
        /// Distinct identities compared.
        checked: usize,
        /// Shared identities, sorted.
        ids: Vec<String>,
    },
    /// Not computable. Never independence.
    Unknown {
        /// Why.
        reason: String,
    },
}

pub(crate) fn observation_id(id: &str) -> ObservationId {
    ObservationId::from_key(OBSERVATION_DOMAIN, id.as_bytes())
}

/// Resolve the overlap of a prior's observations and a study's reused observations
/// through one [`EvidenceLedger`] query. Duplicate identities within a side collapse
/// (a set); any ledger refusal is `Unknown`.
pub(crate) fn declared_overlap(
    prior: DeclaredObservations<'_>,
    reused: DeclaredObservations<'_>,
) -> ObservationOverlap {
    let mut ledger = EvidenceLedger::new();
    let mut names: BTreeMap<ObservationId, String> = BTreeMap::new();
    let mut register = |label: &str,
                        kind: SourceKind,
                        side: DeclaredObservations<'_>|
     -> Result<SourceId, String> {
        let mut declaration = SourceDeclaration::new(label, kind);
        let DeclaredObservations::Declared(ids) = side;
        let unique: BTreeSet<&str> = ids.iter().map(String::as_str).collect();
        for id in &unique {
            names.insert(observation_id(id), (*id).to_owned());
        }
        declaration = declaration.with_observations(unique.iter().map(|id| observation_id(id)));
        ledger.add_source(declaration).map_err(|error| error.message)
    };
    let ids = match (
        register("prior", SourceKind::Prior, prior),
        register("study", SourceKind::ObservationSet, reused),
    ) {
        (Ok(p), Ok(s)) => (p, s),
        (Err(reason), _) | (_, Err(reason)) => return ObservationOverlap::Unknown { reason },
    };
    match ledger.overlap(ids.0, ids.1) {
        Ok(Overlap::Independent) => ObservationOverlap::Independent { checked: names.len() },
        Ok(Overlap::SharedObservations { .. }) => {
            let shared = ledger.shared_observations(ids.0, ids.1).unwrap_or_default();
            ObservationOverlap::Shared {
                checked: names.len(),
                ids: shared
                    .iter()
                    .map(|id| names.get(id).map_or_else(|| id.to_hex(), Clone::clone))
                    .collect(),
            }
        }
        Ok(Overlap::Unknown { reason }) => {
            ObservationOverlap::Unknown { reason: reason.to_owned() }
        }
        Ok(_) => ObservationOverlap::Unknown { reason: "overlap_state_not_recognised".to_owned() },
        Err(error) => ObservationOverlap::Unknown { reason: error.message },
    }
}

/// Evidence-ledger bindings for an EVSI evaluation: the ledger that holds the prior's
/// and each study's observations (lineage resolved), the prior's source, and the
/// source of each candidate by semantic id.
///
/// With bindings, overlap is the ledger's: a study derived from, a subset of, or a
/// duplicate of data the prior rests on is `Shared`, and an undeclared source or
/// unbound candidate is `Unknown`; both refuse. The request's observation-id lists
/// are not consulted and must be empty.
#[derive(Clone, Copy)]
pub struct EvsiEvidence<'a> {
    ledger: &'a EvidenceLedger,
    prior: SourceId,
    candidates: &'a BTreeMap<String, SourceId>,
}

impl<'a> EvsiEvidence<'a> {
    /// Bind a ledger, the prior's source and each candidate's source.
    #[must_use]
    pub const fn new(
        ledger: &'a EvidenceLedger,
        prior: SourceId,
        candidates: &'a BTreeMap<String, SourceId>,
    ) -> Self {
        Self { ledger, prior, candidates }
    }

    fn overlap(&self, candidate: &str) -> ObservationOverlap {
        let Some(&study) = self.candidates.get(candidate) else {
            return ObservationOverlap::Unknown { reason: "candidate_source_unbound".to_owned() };
        };
        let size = |id: SourceId| -> Option<usize> {
            self.ledger.source(id).and_then(|s| s.effective.known()).map(<[_]>::len)
        };
        match self.ledger.overlap(self.prior, study) {
            Ok(Overlap::Independent) => ObservationOverlap::Independent {
                checked: size(self.prior).unwrap_or(0) + size(study).unwrap_or(0),
            },
            Ok(Overlap::SharedObservations { .. }) => ObservationOverlap::Shared {
                checked: size(self.prior).unwrap_or(0) + size(study).unwrap_or(0),
                ids: self
                    .ledger
                    .shared_observations(self.prior, study)
                    .unwrap_or_default()
                    .iter()
                    .map(ObservationId::to_hex)
                    .collect(),
            },
            Ok(Overlap::Unknown { reason }) => {
                ObservationOverlap::Unknown { reason: reason.to_owned() }
            }
            Ok(_) => {
                ObservationOverlap::Unknown { reason: "overlap_state_not_recognised".to_owned() }
            }
            Err(error) => ObservationOverlap::Unknown { reason: error.message },
        }
    }
}

/// Overlap of the prior's observations and one candidate's, from the ledger when
/// bound and from the request's declared identities otherwise (one ledger query
/// either way).
fn candidate_overlap(
    request: &EvsiRequest,
    candidate: &EvsiCandidate,
    evidence: Option<&EvsiEvidence<'_>>,
) -> Result<SourceOverlapDiagnostics, EvsiError> {
    let found = match evidence {
        Some(bound) => bound.overlap(&candidate.semantic_id),
        None => declared_overlap(
            DeclaredObservations::Declared(&request.prior_observation_ids),
            DeclaredObservations::Declared(&candidate.reused_observation_ids),
        ),
    };
    match found {
        ObservationOverlap::Independent { checked } => {
            Ok(SourceOverlapDiagnostics { observations_checked: checked, overlapping: Vec::new() })
        }
        ObservationOverlap::Shared { ids, .. } => Err(EvsiError::SourceOverlap(ids)),
        ObservationOverlap::Unknown { reason } => Err(EvsiError::SourceOverlapUnknown {
            candidate: candidate.semantic_id.clone(),
            reason,
        }),
    }
}

/// Numbers one candidate's evaluation produced.
struct Evaluated {
    evsi: f64,
    integration: IntegrationReport,
    evpi: f64,
    prior_expected_utility: f64,
    bayes_action: usize,
    receipt: SignalReceipt,
    overlap: SourceOverlapDiagnostics,
    /// Digest of the Monte Carlo replicate-stream coordinates, when sampled.
    stream_digest: Option<String>,
}

/// A candidate's value and the numbers it is compared against.
struct Valued {
    evsi: f64,
    integration: IntegrationReport,
    evpi: f64,
    prior_expected_utility: f64,
    bayes_action: usize,
    stream_digest: Option<String>,
}

fn likelihood_value<A>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    signal: &Arc<dyn DecisionSignal<f64>>,
    candidate: &EvsiCandidate,
    request: &EvsiRequest,
    ctx: &ExecutionContext,
) -> Result<Valued, EvsiError> {
    let ranker = DesignRanker::new().with_config(request.rank_config.clone());
    ranker.validate_config()?;
    let analysis = PreposteriorAnalysis::new(problem, prior, signal.as_ref())?;
    let n = signal
        .sample_size(&candidate.design)
        .ok_or_else(|| EvsiError::Unscored(candidate.semantic_id.clone()))?;
    if ctx.cancellation.is_cancelled() {
        return Err(EvsiError::Cancelled);
    }
    // One candidate has no pair to resolve, so the adaptive loop always stops on its own
    // minimum-batch test: `min_batches` batches of `batch_size` replicates. Replicates are
    // keyed by (seed, candidate, replicate ordinal), never by how many draws came before.
    let (evsi, integration, stream_digest) = if let Some(exact) = analysis.exact_evsi(n)? {
        let exact_report = IntegrationReport {
            method: IntegrationMethod::Exact,
            replicates: 0,
            stderr: 0.0,
            ess: None,
            converged: true,
            early_stopped: true,
        };
        (exact, exact_report, None)
    } else {
        let batches = request.rank_config.min_batches.max(1);
        let EvsiMonteCarlo { mean, stderr, replicates, stream_digest } = analysis
            .monte_carlo_evsi(
                n,
                &candidate.semantic_id,
                batches,
                request.rank_config.batch_size,
                ctx,
            )?;
        let report = IntegrationReport {
            method: IntegrationMethod::MonteCarlo,
            replicates,
            stderr,
            ess: Some(replicates as f64),
            converged: stderr <= request.mc_error_tolerance,
            early_stopped: true,
        };
        (mean, report, Some(stream_digest))
    };
    Ok(Valued {
        evsi,
        integration,
        evpi: analysis.expected_value_of_perfect_information(),
        prior_expected_utility: analysis.prior_expected_utility(),
        bayes_action: analysis.bayes_action(),
        stream_digest,
    })
}

fn values_value<A>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    values: &ExternalDecisionValues,
    request: &EvsiRequest,
    id: &str,
) -> Result<Valued, EvsiError> {
    let DecisionPrior::Draws(states) = prior else {
        return Err(EvsiError::DecisionValuesNeedDraws);
    };
    let columns: Vec<usize> = request
        .action_ids
        .iter()
        .map(|action| values.action_ids.iter().position(|v| v == action))
        .collect::<Option<Vec<_>>>()
        .ok_or(EvsiError::ActionSetChanged)?;
    if values.action_ids.len() != request.action_ids.len() {
        return Err(EvsiError::ActionSetChanged);
    }
    let table = DecisionTable::build(problem, states)?;
    let n_actions = problem.actions.len();
    let prior_eu: Vec<f64> = (0..n_actions)
        .map(|a| {
            (0..table.n_outcomes).map(|k| table.utility(a, k)).sum::<f64>()
                / table.n_outcomes as f64
        })
        .collect();
    let branch_value = |branch: usize, action: usize| values.values[branch][columns[action]];
    for (action, &prior) in prior_eu.iter().enumerate() {
        let averaged: f64 = values
            .branch_probabilities
            .iter()
            .enumerate()
            .map(|(y, p)| p * branch_value(y, action))
            .sum();
        if (averaged - prior).abs() > 1e-8 * (1.0 + prior.abs()) {
            return Err(EvsiError::DecisionValuesIncoherent(id.to_owned()));
        }
    }
    let after: f64 = values
        .branch_probabilities
        .iter()
        .enumerate()
        .map(|(y, p)| {
            let best = table
                .admissible
                .iter()
                .map(|&a| branch_value(y, a))
                .fold(f64::NEG_INFINITY, f64::max);
            p * best
        })
        .sum();
    let (bayes_action, prior_best) = table.bayes_action();
    let integration = IntegrationReport {
        method: IntegrationMethod::ExternallyComputed,
        replicates: 0,
        stderr: 0.0,
        ess: None,
        converged: true,
        early_stopped: false,
    };
    let evpi = evaluate_decision(problem, states)?.posterior_regret;
    Ok(Valued {
        evsi: after - prior_best,
        integration,
        evpi,
        prior_expected_utility: prior_best,
        bayes_action,
        stream_digest: None,
    })
}

fn evaluate_candidate<A: Clone>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    candidate: &EvsiCandidate,
    evidence: Option<&EvsiEvidence<'_>>,
    ctx: &ExecutionContext,
) -> Result<Evaluated, EvsiError> {
    let id = &candidate.semantic_id;
    let overlap = candidate_overlap(request, candidate, evidence)?;
    let prepared = candidate.provider.prepare(&candidate.signal_request)?;
    if prepared.receipt.request_fingerprint != candidate.signal_request.fingerprint() {
        return Err(EvsiError::ReceiptMismatch(id.clone()));
    }
    if let Some(bound) = &prepared.bound_states {
        let aligned = match prior {
            DecisionPrior::Draws(states) => {
                states.len() == bound.len()
                    && states.iter().zip(bound).all(|(a, b)| state_key(*a) == state_key(*b))
            }
            DecisionPrior::Normal { .. } => false,
        };
        if !aligned {
            return Err(EvsiError::PriorStatesMismatch(id.clone()));
        }
    }
    let valued = match &prepared.law {
        PreparedLaw::Likelihood(signal) => {
            // The prepared signal's own sample-size rule is authoritative. A copy of
            // the default design-field rule refused an overriding signal (one that
            // answers the request's `n`) whenever the design's row count differed.
            if signal.sample_size(&candidate.design) != Some(candidate.signal_request.sample_size) {
                return Err(EvsiError::SampleSizeMismatch(id.clone()));
            }
            likelihood_value(problem, prior, signal, candidate, request, ctx)?
        }
        PreparedLaw::DecisionValues(values) => values_value(problem, prior, values, request, id)?,
    };
    let Valued { evsi, integration, evpi, prior_expected_utility, bayes_action, stream_digest } =
        valued;
    if let Some(which) = evsi_bound_violation(evsi, evpi, integration.stderr) {
        return Err(EvsiError::BoundViolation(which));
    }
    // Seal reuses this law when the receipt is still the one on the row. The
    // artifact is built from the law, so a miss falls back to preparing again.
    remember_prepared(&prepared);
    Ok(Evaluated {
        evsi,
        integration,
        evpi,
        prior_expected_utility,
        bayes_action,
        receipt: prepared.receipt,
        overlap,
        stream_digest,
    })
}

fn assumptions(
    request: &EvsiRequest,
    prior: &DecisionPrior<f64>,
    evaluated: &Evaluated,
) -> Vec<String> {
    let mut out = vec![
        format!("conditional_independence:{}", evaluated.receipt.conditional_independence),
        "terminal_action_set_identical_before_and_after_information".to_owned(),
        format!("utility_units:{}", request.utility_unit),
        format!("provider_trust:{}", evaluated.receipt.trust.as_str()),
        format!("update_mode:{}", evaluated.receipt.update_mode.as_str()),
        format!("integration:{}", evaluated.integration.method.as_str()),
        match prior {
            DecisionPrior::Draws(_) => "prior:equally_weighted_draws".to_owned(),
            DecisionPrior::Normal { .. } => "prior:normal_conjugate".to_owned(),
        },
    ];
    if evaluated.receipt.update_mode.externally_computed() {
        out.push("update_computed_externally_not_verified_by_antecedent".to_owned());
    }
    if let Some(digest) = &evaluated.stream_digest {
        out.push(format!("monte_carlo_streams:{digest}"));
    }
    out
}

/// Evaluate EVSI for every candidate (canonical semantic-id order, bounded by
/// `max_candidates`) and rank them by net value when a valid cost mapping is
/// supplied, otherwise by EVSI.
///
/// The ranking depends only on the candidate set, not the order it was supplied in.
///
/// # Errors
///
/// Invalid requests, signal refusals, source overlap, sample-size or action-set
/// mismatches, incoherent external values, cost-unit mismatches, bound violations,
/// cancellation, and decision-analysis failures.
pub fn evaluate_evsi<A: Clone + Sync>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    cancellation: &CancellationToken,
) -> Result<EvsiReport, EvsiError> {
    evaluate(problem, prior, request, None, cancellation)
}

/// [`evaluate_evsi`] with the prior's and each study's observations resolved through
/// an [`EvidenceLedger`] (lineage included), so derived, subset or duplicate data is
/// refused as shared and an undeclared source or unbound candidate is refused as
/// unknown rather than treated as independent.
///
/// # Errors
///
/// As [`evaluate_evsi`], plus [`EvsiError::SourceOverlapUnknown`], and an invalid
/// request when the request's own observation-id lists are non-empty (the ledger is
/// the only overlap model on this path).
pub fn evaluate_evsi_with_evidence<A: Clone + Sync>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    evidence: &EvsiEvidence<'_>,
    cancellation: &CancellationToken,
) -> Result<EvsiReport, EvsiError> {
    if !request.prior_observation_ids.is_empty()
        || request.candidates.iter().any(|c| !c.reused_observation_ids.is_empty())
    {
        return Err(EvsiError::InvalidRequest(
            "observation id lists must be empty when overlap is read from the ledger",
        ));
    }
    evaluate(problem, prior, request, Some(evidence), cancellation)
}

fn evaluate<A: Clone + Sync>(
    problem: &DecisionProblem<A, f64>,
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    evidence: Option<&EvsiEvidence<'_>>,
    cancellation: &CancellationToken,
) -> Result<EvsiReport, EvsiError> {
    validate_request(problem, request)?;
    let mut ordered: Vec<&EvsiCandidate> = request.candidates.iter().collect();
    ordered.sort_by(|a, b| a.semantic_id.cmp(&b.semantic_id));
    let supplied = ordered.len();
    let unevaluated_ids: Vec<String> =
        ordered.iter().skip(request.max_candidates).map(|c| c.semantic_id.clone()).collect();
    ordered.truncate(request.max_candidates);
    let mut ctx =
        ExecutionContext::production(request.rng_seed, antecedent_core::default_user_threads());
    ctx.cancellation = cancellation.clone();
    // Each candidate's replicate streams are keyed by (seed, semantic id, replicate), so
    // neither the index order of `map_indexed` nor the other candidates move them.
    let evaluated = ctx.map_indexed(ordered.len(), |index, ctx| {
        if ctx.cancellation.is_cancelled() {
            return Err(EvsiError::Cancelled);
        }
        evaluate_candidate(problem, prior, request, ordered[index], evidence, ctx)
    })?;
    let search = SearchReceipt {
        supplied,
        evaluated: ordered.len(),
        truncated: !unevaluated_ids.is_empty(),
        unevaluated_ids,
    };
    Ok(assemble(prior, request, &ordered, &evaluated, search))
}

fn assemble(
    prior: &DecisionPrior<f64>,
    request: &EvsiRequest,
    ordered: &[&EvsiCandidate],
    evaluated: &[Evaluated],
    search: SearchReceipt,
) -> EvsiReport {
    let map = request.cost_map.as_ref();
    let mut rows: Vec<EvsiCandidateRow> = ordered
        .iter()
        .zip(evaluated)
        .map(|(candidate, e)| {
            let cost_utility = map.map(|m| m.utility_per_cost * candidate.cost.amount);
            EvsiCandidateRow {
                semantic_id: candidate.semantic_id.clone(),
                evsi: e.evsi,
                integration: e.integration,
                evpi: e.evpi,
                sample_size: candidate.signal_request.sample_size,
                study_cost: candidate.cost.clone(),
                study_cost_utility: cost_utility,
                net_value: cost_utility.map(|c| e.evsi - c),
                rank: 0,
                rank_uncertain: false,
                provider_trust: e.receipt.trust,
                update_mode: e.receipt.update_mode,
                signal_receipt: e.receipt.clone(),
                source_overlap: e.overlap.clone(),
                assumptions: assumptions(request, prior, e),
            }
        })
        .collect();
    let key = |row: &EvsiCandidateRow| row.net_value.unwrap_or(row.evsi);
    rows.sort_by(|a, b| key(b).total_cmp(&key(a)).then_with(|| a.semantic_id.cmp(&b.semantic_id)));
    let ranked = rank_rows(
        &rows
            .iter()
            .map(|row| AdjacentRank {
                key: key(row),
                stderr: row.integration.stderr,
                id: row.semantic_id.as_str(),
            })
            .collect::<Vec<_>>(),
        request.tie_tolerance,
    );
    let ties = ranked.ties;
    for (i, row) in rows.iter_mut().enumerate() {
        row.rank = i;
        row.rank_uncertain = ranked.uncertain[i];
    }
    let first = evaluated.first();
    EvsiReport {
        decision_contract_identity: request.decision_contract_identity.clone(),
        utility_unit: request.utility_unit.clone(),
        action_ids: {
            let mut ids = request.action_ids.clone();
            ids.sort();
            ids
        },
        bayes_action: first.map_or(0, |e| e.bayes_action),
        prior_expected_utility: first.map_or(0.0, |e| e.prior_expected_utility),
        evpi: first.map_or(0.0, |e| e.evpi),
        rng_seed: request.rng_seed,
        cost_map: request.cost_map.clone(),
        basis: if map.is_some() { RankingBasis::NetValue } else { RankingBasis::Evsi },
        candidates: rows,
        ties,
        search,
    }
}
