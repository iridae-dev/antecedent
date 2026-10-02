//! Study planning over the X1 (mz) and X9 (mixed-source) catalogs (X6).
//!
//! A planner receives a failed X1 or X9 decision and a declared universe of at
//! most [`STUDY_PLAN_MAX_CANDIDATES`] candidate studies, each already compiled
//! to a typed [`EvidenceCatalogDelta`] with an integer cost. It enumerates every
//! feasible subset of at most [`STUDY_PLAN_MAX_SUBSET`] candidates in cost
//! order and re-runs the same route's decision on each subset's preview
//! catalog, all under ONE shared [`SearchBudget`]: the base decision, every
//! subset entry and every step of every re-run decision charge it, with the
//! live bytes of the proposals already kept retained under every later charge.
//!
//! A subset is **sufficient** only when the re-run decision returns
//! `Identified` and its derivation re-checks against the preview catalog (an mz
//! derivation re-binds; an X9 derivation replays through the independent
//! checker; an X9 named route is re-run on the shared budget and must bind),
//! and the formula cites at least one proposed regime. The plan claims no
//! probability of success and no numeric result: a preview is never available
//! evidence, and arrival re-identifies from the real catalog.
//!
//! What the plan does and does not claim:
//!
//! - "No sufficient subset" is `study_plan.none_certified` within this universe
//!   and rule set, never an impossibility: X9 is sound but incomplete and X1's
//!   obstruction decider is partial. The only theorem-limited refusal is a base
//!   X1 decision that is a replayed checked obstruction, which concerns the
//!   declared controllable sets, so no study inside them repairs it; a new
//!   population or a wider controllable set is outside this universe.
//! - Minimality is claimed only when every feasible subset strictly cheaper
//!   than the top proposal was evaluated to a conclusive outcome without a stop:
//!   "cost-minimal among the verified-derivable subsets of at most three
//!   candidates of this declared universe, under this search and rule set",
//!   never "minimal sufficient evidence". A cheaper subset of four or more
//!   candidates is never examined, and a cheaper subset the route refuses (for
//!   example over the X9 usable-distribution bound) or leaves underivable keeps
//!   the claim: it is only not sufficient under this search.
//! - Supersets of a sufficient subset are not evaluated and are listed
//!   `dominated`, for ranking only; sufficiency is not claimed monotone.
//! - A decision that spends more than its route's public per-decision
//!   operation cap (4096 on Mz, 20000 on Mixed) is inconclusive for that
//!   subset, never sufficient, so an arrival re-identified through the public
//!   route is decided under the same caps.
//!
//! Every cost is a positive integer, so every proper subset of a subset is
//! strictly cheaper and is evaluated first; ties rank by sample budget, subset
//! size and sorted candidate ids ([`STUDY_PLAN_RANKING`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    DEFAULT_SEARCH_MEMORY_BYTES, DistributionAvailability, EvidenceCatalog, EvidenceCatalogDelta,
    EvidenceKind, ExecutionContext, LawOrigin, RegimeId, RegimeKind, SamplingSelection,
    SearchBudget, SearchLimits, SearchReceipt, SearchStop, VariableId, reason_code,
};
use antecedent_expr::{CausalExprArena, ExprId, ExprNode};
use antecedent_graph::{Admg, SelectionDiagram};

use super::meta::{MetaSource, MetaTransportQuery, identify_meta_catalog_metered};
use super::mixed_source::decide_mixed_source_shared;
use super::mz_transport::decide_mz_transport_shared;
use super::z_transport::{ZTransportDecision, decide_z_transport_reporting};
use super::{
    CatalogTransportResult, ClassicalTransportQuery, IdentificationError,
    MIXED_SOURCE_DEFAULT_LIMITS, MZ_TRANSPORT_DEFAULT_LIMITS, MixedSourceDecision,
    MixedSourceDerivationRecord, MixedSourceQuery, MzTransportDecision,
    MzTransportDerivationRecord, MzTransportQuery, MzTransportRoute, SearchCharge, SharedSearch,
    ZTransportQuery, ZTransportSourceSpec, bind_mixed_source_catalog, bind_mz_transport_catalog,
    bind_z_transport_catalog, identify_catalog_transport_metered,
};

/// Candidate studies in one declared universe.
pub const STUDY_PLAN_MAX_CANDIDATES: usize = 16;
/// Regimes one candidate study may propose (one per declared level combination).
pub const STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE: usize = 8;
/// Largest subset of candidates evaluated together.
pub const STUDY_PLAN_MAX_SUBSET: usize = 3;
/// Sufficient proposals one plan retains; evaluation ends after the last one.
pub const STUDY_PLAN_MAX_PROPOSALS: usize = 32;
/// Plan limits: operations charged across the base decision and every subset
/// decision, and depth. They are the defaults and the maxima; the route's own
/// depth maximum also applies (24 on Mz, 16 on Mixed).
pub const STUDY_PLAN_DEFAULT_LIMITS: SearchLimits = SearchLimits { operations: 200_000, depth: 24 };
/// Memory cap of the plan budget: the default and the maximum.
pub const STUDY_PLAN_MEMORY_BYTES: u64 = DEFAULT_SEARCH_MEMORY_BYTES;
/// Largest declared cost of one candidate, so a subset's total never overflows.
pub const STUDY_PLAN_MAX_COST_UNITS: u64 = u64::MAX / 4;
/// Version of the ranking key: cost units, sample budget, subset size, sorted ids.
pub const STUDY_PLAN_RANKING: &str = "x6.ranking.v1";

/// Candidates, regimes, subset size or limits beyond a declared bound.
const BOUNDS_EXCEEDED: &str = "study_plan.bounds_exceeded";
/// A candidate declaration the route cannot accept.
const INVALID_CANDIDATE: &str = "study_plan.invalid_candidate";
/// The route refuses the base query or catalog.
const INVALID_QUERY: &str = "study_plan.invalid_query";
/// The base decision already identifies.
const NO_FAILURE: &str = "study_plan.no_failure_to_repair";
/// A replayed X1 obstruction over the declared controllable sets.
const THEOREM_LIMITED: &str = "study_plan.theorem_limited";
/// No subset of the universe is sufficient; not an impossibility claim.
const NONE_CERTIFIED: &str = "study_plan.none_certified";
/// A plan budget stop.
const BUDGET: &str = "study_plan.budget";

/// A refused plan: a registered top-level reason code, a `study_plan.*` detail
/// and, for a budget stop before any subset was decided, the receipt.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct StudyPlanRefusal {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `study_plan.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
    /// The plan budget's receipt, for a budget stop.
    pub receipt: Option<Box<SearchReceipt>>,
}

impl StudyPlanRefusal {
    /// A request beyond a declared bound (`route_not_supported`,
    /// `study_plan.bounds_exceeded`).
    #[must_use]
    pub fn bounds_exceeded(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("route_not_supported"),
            detail: BOUNDS_EXCEEDED,
            message: message.into(),
            receipt: None,
        }
    }

    /// A candidate declaration the route cannot accept (`invalid_argument`,
    /// `study_plan.invalid_candidate`).
    #[must_use]
    pub fn invalid_candidate(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: INVALID_CANDIDATE,
            message: message.into(),
            receipt: None,
        }
    }

    fn invalid_query(message: impl Into<String>) -> Self {
        Self {
            code: reason_code!("invalid_argument"),
            detail: INVALID_QUERY,
            message: message.into(),
            receipt: None,
        }
    }

    fn budget(receipt: SearchReceipt) -> Self {
        Self {
            code: reason_code!("transport_budget_cancel"),
            detail: BUDGET,
            message: format!(
                "the plan budget stopped ({}) before any subset was decided",
                receipt.stop.code()
            ),
            receipt: Some(Box::new(receipt)),
        }
    }
}

/// The failed route a plan repairs, with its frozen query.
#[derive(Clone, Debug, PartialEq)]
pub enum StudyPlanRoute {
    /// The X1 multi-source limited-experiment route.
    Mz(MzTransportQuery),
    /// The X9 mixed-source proof search.
    Mixed(MixedSourceQuery),
}

impl StudyPlanRoute {
    /// `mz` or `mixed`.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Mz(_) => "mz",
            Self::Mixed(_) => "mixed",
        }
    }

    /// The route in canonical form (sources by population, sorted variable sets).
    #[must_use]
    pub fn canonical(&self) -> Self {
        match self {
            Self::Mz(query) => Self::Mz(query.canonical()),
            Self::Mixed(query) => Self::Mixed(query.canonical()),
        }
    }

    fn target(&self) -> &str {
        match self {
            Self::Mz(query) => &query.target,
            Self::Mixed(query) => &query.target,
        }
    }

    fn sources(&self) -> &[ZTransportSourceSpec] {
        match self {
            Self::Mz(query) => &query.sources,
            Self::Mixed(query) => &query.sources,
        }
    }

    /// The route's depth maximum: 24 on Mz, 16 on Mixed.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        match self {
            Self::Mz(_) => MZ_TRANSPORT_DEFAULT_LIMITS.depth,
            Self::Mixed(_) => MIXED_SOURCE_DEFAULT_LIMITS.depth,
        }
    }

    /// The route's public per-decision operation cap: 4096 on Mz, 20000 on Mixed.
    #[must_use]
    pub const fn decision_operations(&self) -> usize {
        match self {
            Self::Mz(_) => MZ_TRANSPORT_DEFAULT_LIMITS.operations,
            Self::Mixed(_) => MIXED_SOURCE_DEFAULT_LIMITS.operations,
        }
    }
}

/// One candidate study, compiled to the neutral form the planner searches: a
/// hypothetical catalog delta, an integer cost and its subset constraints.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyPlanCandidate {
    /// Stable identity: 1-64 characters of `[A-Za-z0-9_.:-]`.
    pub id: Arc<str>,
    /// The proposed regimes; never available evidence by themselves.
    pub delta: EvidenceCatalogDelta,
    /// Declared cost, 1 to [`STUDY_PLAN_MAX_COST_UNITS`].
    pub cost_units: u64,
    /// Declared sample budget; a ranking tie-breaker, never a sufficiency input.
    pub sample_budget: u64,
    /// Candidates that must be run together with this one.
    pub requires: Arc<[Arc<str>]>,
    /// Candidates that cannot be run together with this one.
    pub conflicts: Arc<[Arc<str>]>,
}

/// Plan budget limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StudyPlanLimits {
    /// Operation and depth limits of the one plan budget.
    pub search: SearchLimits,
    /// Memory cap, at most [`STUDY_PLAN_MEMORY_BYTES`]; the context's hard limit
    /// lowers it further.
    pub memory_limit_bytes: u64,
}

impl Default for StudyPlanLimits {
    fn default() -> Self {
        Self { search: STUDY_PLAN_DEFAULT_LIMITS, memory_limit_bytes: STUDY_PLAN_MEMORY_BYTES }
    }
}

impl StudyPlanLimits {
    /// The maximal limits a route accepts: [`STUDY_PLAN_DEFAULT_LIMITS`] with
    /// the depth lowered to the route's own maximum (16 on Mixed).
    #[must_use]
    pub const fn for_route(route: &StudyPlanRoute) -> Self {
        Self {
            search: SearchLimits {
                operations: STUDY_PLAN_DEFAULT_LIMITS.operations,
                depth: route.max_depth(),
            },
            memory_limit_bytes: STUDY_PLAN_MEMORY_BYTES,
        }
    }

    /// Check a request against the declared bounds before any work: these
    /// limits against the route's maxima (at most
    /// [`STUDY_PLAN_DEFAULT_LIMITS`] operations, the route's depth and
    /// [`STUDY_PLAN_MEMORY_BYTES`]), at most [`STUDY_PLAN_MAX_CANDIDATES`]
    /// candidates (one entry of `regimes_per_candidate` each) and at most
    /// [`STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE`] regimes in any one.
    ///
    /// # Errors
    /// `study_plan.bounds_exceeded` naming the first bound exceeded.
    pub fn check_bounds(
        &self,
        route: &StudyPlanRoute,
        regimes_per_candidate: &[usize],
    ) -> Result<(), StudyPlanRefusal> {
        if self.search.operations > STUDY_PLAN_DEFAULT_LIMITS.operations
            || self.search.depth > route.max_depth()
            || self.memory_limit_bytes > STUDY_PLAN_MEMORY_BYTES
        {
            return Err(StudyPlanRefusal::bounds_exceeded(format!(
                "plan limits must be at most {} operations, depth {} and \
                 {STUDY_PLAN_MEMORY_BYTES} bytes",
                STUDY_PLAN_DEFAULT_LIMITS.operations,
                route.max_depth()
            )));
        }
        if regimes_per_candidate.len() > STUDY_PLAN_MAX_CANDIDATES {
            return Err(StudyPlanRefusal::bounds_exceeded(format!(
                "{} candidates exceed the bound of {STUDY_PLAN_MAX_CANDIDATES}",
                regimes_per_candidate.len()
            )));
        }
        if let Some(count) =
            regimes_per_candidate.iter().find(|&&n| n > STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE)
        {
            return Err(StudyPlanRefusal::bounds_exceeded(format!(
                "a candidate proposes {count} regimes, above the bound of \
                 {STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE}"
            )));
        }
        Ok(())
    }
}

/// The frozen identity of the failure being repaired.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StudyBaseFailure {
    /// The base decision's registered reason code.
    pub code: &'static str,
    /// The base decision's route detail (`mz_transport.*` or `mixed_search.*`).
    pub detail: &'static str,
    /// Stable facts of the failure: missing-factor text, missing joints and
    /// the stages evaluated.
    pub facts: Vec<String>,
}

/// How one subset ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StudySubsetOutcome {
    /// Sufficient: the index of its proposal in [`StudyPlan::proposals`].
    Sufficient {
        /// Proposal index.
        proposal: usize,
    },
    /// The re-run decision did not identify: its reason code and route detail.
    Insufficient {
        /// Route reason code.
        code: &'static str,
        /// Route detail.
        detail: &'static str,
    },
    /// The route refused the preview catalog (not sufficient).
    Refused {
        /// The route's refusal.
        message: String,
    },
    /// The decision spent more than the route's public per-decision cap:
    /// inconclusive, never sufficient.
    OverDecisionCap {
        /// Operations the decision charged.
        operations: usize,
    },
    /// A superset of a sufficient subset; not evaluated (ranking only).
    Dominated {
        /// The sufficient subset it contains.
        by: Vec<Arc<str>>,
    },
    /// Violates a declared `requires` or `conflicts` constraint; outside the
    /// feasible universe.
    Infeasible {
        /// `requires` or `conflicts`.
        reason: &'static str,
    },
    /// Left unevaluated by a budget stop or the proposal cap.
    Unevaluated,
}

impl StudySubsetOutcome {
    /// Whether the outcome decides the subset (inconclusive and unevaluated do not).
    #[must_use]
    pub const fn conclusive(&self) -> bool {
        !matches!(self, Self::OverDecisionCap { .. } | Self::Unevaluated)
    }

    /// Stable status name.
    #[must_use]
    pub const fn status(&self) -> &'static str {
        match self {
            Self::Sufficient { .. } => "sufficient",
            Self::Insufficient { .. } => "insufficient",
            Self::Refused { .. } => "refused",
            Self::OverDecisionCap { .. } => "over_decision_cap",
            Self::Dominated { .. } => "dominated",
            Self::Infeasible { .. } => "infeasible",
            Self::Unevaluated => "unevaluated",
        }
    }
}

/// One subset in evaluation (ranking) order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StudySubsetRecord {
    /// Candidate ids, sorted.
    pub candidates: Vec<Arc<str>>,
    /// Total cost units.
    pub cost_units: u64,
    /// Total sample budget.
    pub sample_budget: u64,
    /// How it ended.
    pub outcome: StudySubsetOutcome,
}

impl StudySubsetRecord {
    /// `subset:[a,b]`, the receipt region of this subset.
    #[must_use]
    pub fn label(&self) -> String {
        subset_label(&self.candidates)
    }
}

/// One factor of the formula a proposed regime supplies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StudyFactor {
    /// Variables of the factor.
    pub variables: Vec<VariableId>,
    /// Conditioning variables of the factor.
    pub conditioned_on: Vec<VariableId>,
}

/// What one proposed regime repairs in the formula.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StudyRepair {
    /// The candidate proposing it.
    pub candidate: Arc<str>,
    /// The proposed regime.
    pub regime: RegimeId,
    /// Its population.
    pub population: Arc<str>,
    /// Its intervention set.
    pub intervened: Vec<VariableId>,
    /// Every formula factor that reads it.
    pub factors: Vec<StudyFactor>,
    /// The margin the proof actually reads from it: the union of its factors'
    /// variables and conditioning variables, or the whole declared margin when
    /// the proof reads the regime only as an X9 input step (no factor of its
    /// own). A smaller declared margin containing it is not verified unless
    /// declared as its own candidate.
    pub required_margin: Vec<VariableId>,
    /// The margin the candidate declared.
    pub declared_margin: Vec<VariableId>,
    /// X9 proof steps (input steps) that cite it; empty on other routes.
    pub proof_steps: Vec<usize>,
}

/// The checked derivation behind a proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StudyProposalDerivation {
    /// An X1 derivation (its record, including the plan-budget receipt).
    Mz(Box<MzTransportDerivationRecord>),
    /// An X9 rule-search derivation.
    Mixed(Box<MixedSourceDerivationRecord>),
    /// An X9 named route, re-run on the shared budget and bound.
    Named {
        /// `target_first_sid`, `z_transport`, `mz_transport` or `meta_transport`.
        route: &'static str,
    },
}

/// A sufficient subset: what to run, what it repairs, and the checked proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StudyProposal {
    /// Candidate ids, sorted.
    pub candidates: Vec<Arc<str>>,
    /// Total cost units.
    pub cost_units: u64,
    /// Total sample budget.
    pub sample_budget: u64,
    /// The route stage that identified (`mz_transport:combined`,
    /// `mixed_source:rule_search`, `mixed_source:named:z_transport`, ...).
    pub stage: String,
    /// Every catalog regime the checked formula cites, sorted.
    pub cited_regimes: Vec<RegimeId>,
    /// What each cited proposed regime repairs.
    pub repairs: Vec<StudyRepair>,
    /// Candidates of the subset none of whose regimes the formula cites.
    pub uncited_candidates: Vec<Arc<str>>,
    /// The checked derivation.
    pub derivation: StudyProposalDerivation,
    /// Operations the re-run decision charged (the named-route re-run excluded).
    pub decision_operations: usize,
}

/// Why a plan ended before every feasible subset was decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StudyPlanStop {
    /// Operations, depth, memory or cancellation stopped the plan budget.
    Budget(SearchReceipt),
    /// [`STUDY_PLAN_MAX_PROPOSALS`] sufficient subsets were found.
    ProposalCap,
}

/// A finished plan.
#[derive(Clone, Debug, PartialEq)]
pub struct StudyPlan {
    /// The canonical route and query.
    pub route: StudyPlanRoute,
    /// The limits in force (memory is the effective cap).
    pub limits: StudyPlanLimits,
    /// The frozen base failure.
    pub failure: StudyBaseFailure,
    /// Every subset, in evaluation order.
    pub subsets: Vec<StudySubsetRecord>,
    /// Sufficient subsets, ranked.
    pub proposals: Vec<StudyProposal>,
    /// Why the plan ended early, if it did.
    pub stop: Option<StudyPlanStop>,
    /// Whether the top proposal is certified cost-minimal among the feasible
    /// subsets of at most [`STUDY_PLAN_MAX_SUBSET`] candidates (every strictly
    /// cheaper one decided without a stop or an over-cap decision); a cheaper
    /// subset of four or more candidates is never examined.
    pub minimal: bool,
    /// Operations the plan budget charged in total.
    pub operations_consumed: usize,
}

impl StudyPlan {
    /// The `(reason code, study_plan.*)` status of a plan with no proposal:
    /// `transport_budget_cancel` / `study_plan.budget` when the budget stopped
    /// it, else `transport_not_certified` / `study_plan.none_certified`. `None`
    /// when a proposal exists.
    #[must_use]
    pub fn status(&self) -> Option<(&'static str, &'static str)> {
        if !self.proposals.is_empty() {
            return None;
        }
        Some(match self.stop {
            Some(StudyPlanStop::Budget(_)) => (reason_code!("transport_budget_cancel"), BUDGET),
            _ => (reason_code!("transport_not_certified"), NONE_CERTIFIED),
        })
    }

    /// The plan budget's receipt, when a budget stop ended the plan.
    #[must_use]
    pub const fn receipt(&self) -> Option<&SearchReceipt> {
        match &self.stop {
            Some(StudyPlanStop::Budget(receipt)) => Some(receipt),
            _ => None,
        }
    }
}

fn subset_label(ids: &[Arc<str>]) -> String {
    format!("subset:[{}]", ids.iter().map(AsRef::as_ref).collect::<Vec<_>>().join(","))
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-'))
}

/// Check a candidate universe against the bounds, the base catalog and the
/// route's population semantics. Candidates are returned sorted by id.
#[allow(clippy::too_many_lines)] // One linear pass over every declared candidate rule.
fn validate_candidates(
    route: &StudyPlanRoute,
    catalog: &EvidenceCatalog,
    candidates: &[StudyPlanCandidate],
) -> Result<Vec<StudyPlanCandidate>, StudyPlanRefusal> {
    use StudyPlanRefusal as R;
    if candidates.is_empty() {
        return Err(R::invalid_candidate("a plan needs at least one candidate study"));
    }
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let ids = sorted.iter().map(|c| Arc::clone(&c.id)).collect::<BTreeSet<_>>();
    if ids.len() != sorted.len() {
        return Err(R::invalid_candidate("candidate ids must be unique"));
    }
    let mut regimes = catalog.regimes.iter().map(|r| r.id).collect::<BTreeSet<_>>();
    // Labels are unique per catalog: a label two candidates share would make
    // their joint preview fail after the plan started, so it is refused here.
    let mut labels =
        catalog.regimes.iter().filter_map(|r| r.label.clone()).collect::<BTreeSet<_>>();
    let target = route.target();
    for candidate in &sorted {
        let who = &candidate.id;
        if !valid_id(who) {
            return Err(R::invalid_candidate(format!(
                "candidate id {who:?} must be 1-64 characters of [A-Za-z0-9_.:-]"
            )));
        }
        if candidate.cost_units == 0 || candidate.cost_units > STUDY_PLAN_MAX_COST_UNITS {
            return Err(R::invalid_candidate(format!(
                "candidate {who} must cost 1 to {STUDY_PLAN_MAX_COST_UNITS} units"
            )));
        }
        if candidate.sample_budget > STUDY_PLAN_MAX_COST_UNITS {
            return Err(R::invalid_candidate(format!(
                "candidate {who} sample budget is too large"
            )));
        }
        if candidate.delta.proposed_regimes.is_empty() {
            return Err(R::invalid_candidate(format!("candidate {who} proposes no regime")));
        }
        for other in candidate.requires.iter().chain(candidate.conflicts.iter()) {
            if !ids.contains(other) || other == who {
                return Err(R::invalid_candidate(format!(
                    "candidate {who} names unknown or self constraint {other}"
                )));
            }
        }
        if candidate.requires.iter().any(|r| candidate.conflicts.contains(r)) {
            return Err(R::invalid_candidate(format!(
                "candidate {who} both requires and conflicts with a candidate"
            )));
        }
        candidate.delta.preview_catalog(catalog).map_err(|error| {
            R::invalid_candidate(format!("candidate {who} does not preview on the base: {error}"))
        })?;
        for regime in candidate.delta.proposed_regimes.iter() {
            if !regimes.insert(regime.id) {
                return Err(R::invalid_candidate(format!(
                    "regime {} of candidate {who} repeats a base or candidate regime id",
                    regime.id.raw()
                )));
            }
            if let Some(label) = &regime.label {
                if !labels.insert(Arc::clone(label)) {
                    return Err(R::invalid_candidate(format!(
                        "regime {} of candidate {who} repeats the label {label:?} of a base or \
                         candidate regime",
                        regime.id.raw()
                    )));
                }
            }
            if regime.measured.is_empty()
                || !regime.conditioned_on.is_empty()
                || regime.distribution != DistributionAvailability::Joint
                || regime.selection != SamplingSelection::Population
                || regime.origin != LawOrigin::Measured
            {
                return Err(R::invalid_candidate(format!(
                    "regime {} of candidate {who} must be a measured, unconditioned, \
                     whole-population joint law with a non-empty margin",
                    regime.id.raw()
                )));
            }
            if regime.population.as_ref() == target {
                if matches!(route, StudyPlanRoute::Mz(_)) && regime.kind == RegimeKind::Experimental
                {
                    return Err(R::invalid_candidate(format!(
                        "candidate {who} experiments in the target; the mz route's target \
                         supplies observational evidence only"
                    )));
                }
            } else {
                let Some(source) =
                    route.sources().iter().find(|s| s.population == regime.population)
                else {
                    return Err(R::invalid_candidate(format!(
                        "candidate {who} studies population {} that is neither the target nor a \
                         declared source (new populations are not planned)",
                        regime.population
                    )));
                };
                if regime.interventions.iter().any(|v| !source.controllable.contains(v)) {
                    return Err(R::invalid_candidate(format!(
                        "candidate {who} intervenes outside source {}'s declared controllable set",
                        source.population
                    )));
                }
            }
            if matches!(route, StudyPlanRoute::Mixed(_)) && !regime.intervention_values.is_empty() {
                return Err(R::invalid_candidate(format!(
                    "candidate {who} restricts its levels; the mixed route states \
                     identification for every level and never cites a value-restricted study"
                )));
            }
        }
    }
    Ok(sorted)
}

/// Every non-empty subset of at most [`STUDY_PLAN_MAX_SUBSET`] candidates, as
/// sorted index lists, in ranking order.
fn subsets(candidates: &[StudyPlanCandidate]) -> Vec<(Vec<usize>, u64, u64)> {
    let n = candidates.len();
    let mut out = Vec::new();
    for a in 0..n {
        out.push(vec![a]);
        for b in a + 1..n {
            out.push(vec![a, b]);
            for c in b + 1..n {
                out.push(vec![a, b, c]);
            }
        }
    }
    let mut keyed = out
        .into_iter()
        .map(|members| {
            let cost = members.iter().map(|&i| candidates[i].cost_units).sum::<u64>();
            let samples = members.iter().map(|&i| candidates[i].sample_budget).sum::<u64>();
            (members, cost, samples)
        })
        .collect::<Vec<_>>();
    keyed.sort_by(|a, b| {
        a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.len().cmp(&b.0.len())).then_with(|| {
            let ids =
                |m: &[usize]| m.iter().map(|&i| Arc::clone(&candidates[i].id)).collect::<Vec<_>>();
            ids(&a.0).cmp(&ids(&b.0))
        })
    });
    keyed
}

fn infeasible(candidates: &[StudyPlanCandidate], members: &[usize]) -> Option<&'static str> {
    let ids = members.iter().map(|&i| &candidates[i].id).collect::<BTreeSet<_>>();
    for &i in members {
        if candidates[i].requires.iter().any(|r| !ids.contains(r)) {
            return Some("requires");
        }
        if candidates[i].conflicts.iter().any(|c| ids.contains(c)) {
            return Some("conflicts");
        }
    }
    None
}

/// The snapshot namespace of the planner's placeholder bindings
/// ([`EvidenceCatalogDelta::with_placeholder_bindings`]).
const PLACEHOLDER_SNAPSHOT_PREFIX: &str = "hypothetical:";

/// Every available base regime carries source/snapshot lineage: a provider
/// binding whose snapshot is non-empty and outside the placeholder namespace.
/// A plan (and so an artifact replay) never runs on a base without it, so an
/// artifact cannot be re-sealed with its lineage stripped.
fn require_base_lineage(catalog: &EvidenceCatalog) -> Result<(), StudyPlanRefusal> {
    for regime in catalog.regimes.iter().filter(|r| r.evidence_kind == EvidenceKind::Available) {
        let traced = catalog.bindings.iter().any(|binding| {
            binding.regime == regime.id
                && !binding.snapshot_identity.trim().is_empty()
                && !binding.snapshot_identity.starts_with(PLACEHOLDER_SNAPSHOT_PREFIX)
        });
        if !traced {
            return Err(StudyPlanRefusal::invalid_query(format!(
                "available base regime {} has no provider binding outside the \
                 {PLACEHOLDER_SNAPSHOT_PREFIX} namespace: a plan needs its source/snapshot lineage",
                regime.id.raw()
            )));
        }
    }
    Ok(())
}

/// Live bytes of one subset's preview catalog, charged on entry.
fn preview_bytes(catalog: &EvidenceCatalog) -> u64 {
    let regimes = catalog.regimes.len() as u64;
    regimes.saturating_mul(512).saturating_add(1024)
}

/// Plan the cheapest study additions that would make the failed X1 or X9
/// decision identify.
///
/// The base decision, every subset's entry (one operation at depth one with
/// its preview bytes) and every step of every re-run decision charge one
/// [`SearchBudget`] under `limits`; the live bytes of retained proposals stay
/// under every later charge. Subsets are evaluated in ranking order (cost
/// units, sample budget, size, sorted ids); a budget stop ends the plan with
/// a receipt naming the explored and unevaluated subsets and keeps every
/// proposal verified so far. Nothing here is a non-identification claim.
///
/// # Errors
///
/// A [`StudyPlanRefusal`]: `study_plan.bounds_exceeded` for a bound,
/// `study_plan.invalid_candidate`, `study_plan.invalid_query` when the route
/// refuses the base, `study_plan.no_failure_to_repair` when the base already
/// identifies, `study_plan.theorem_limited` for a replayed X1 obstruction, and
/// `study_plan.budget` (with the receipt) when the budget stops the base
/// decision.
pub fn plan_study_additions(
    graph: &Admg,
    route: &StudyPlanRoute,
    catalog: &EvidenceCatalog,
    candidates: &[StudyPlanCandidate],
    limits: StudyPlanLimits,
    ctx: &ExecutionContext,
) -> Result<StudyPlan, StudyPlanRefusal> {
    let caps = PlanCaps { decision_cap: |_, cap| cap, max_proposals: STUDY_PLAN_MAX_PROPOSALS };
    plan_with_caps(graph, route, catalog, candidates, limits, ctx, caps)
}

/// The per-decision operation cap and the proposal cap of one plan. Public
/// plans use the route's public cap for every subset and
/// [`STUDY_PLAN_MAX_PROPOSALS`]; unit tests lower them (per subset label) to
/// reach both caps on small fixtures.
#[derive(Clone, Copy, Debug)]
struct PlanCaps {
    /// The cap of the subset labelled by the first argument, given the route's
    /// public cap.
    decision_cap: fn(&str, usize) -> usize,
    max_proposals: usize,
}

#[allow(clippy::too_many_lines)] // One pass: validate, freeze the base, then every subset.
fn plan_with_caps(
    graph: &Admg,
    route: &StudyPlanRoute,
    catalog: &EvidenceCatalog,
    candidates: &[StudyPlanCandidate],
    limits: StudyPlanLimits,
    ctx: &ExecutionContext,
    caps: PlanCaps,
) -> Result<StudyPlan, StudyPlanRefusal> {
    let counts = candidates.iter().map(|c| c.delta.proposed_regimes.len()).collect::<Vec<_>>();
    limits.check_bounds(route, &counts)?;
    catalog.validate().map_err(|e| StudyPlanRefusal::invalid_query(e.to_string()))?;
    require_base_lineage(catalog)?;
    let route = route.canonical();
    let candidates = validate_candidates(&route, catalog, candidates)?;
    let order = subsets(&candidates);
    let ids_of = |members: &[usize]| {
        members.iter().map(|&i| Arc::clone(&candidates[i].id)).collect::<Vec<_>>()
    };
    let all_labels = || order.iter().map(|(m, _, _)| subset_label(&ids_of(m))).collect::<Vec<_>>();
    let mut search = match SearchBudget::with_memory(limits.search, limits.memory_limit_bytes, ctx)
    {
        Ok(budget) => SharedSearch::new(budget),
        Err(mut receipt) => {
            receipt.unevaluated =
                std::iter::once("stage:base".to_owned()).chain(all_labels()).collect();
            return Err(StudyPlanRefusal::budget(receipt));
        }
    };
    let effective =
        StudyPlanLimits { search: limits.search, memory_limit_bytes: search.memory_limit_bytes() };

    // The base decision, charged to the plan budget.
    search.begin(0);
    let failure = match base_failure(graph, &route, catalog, &mut search, ctx) {
        Ok(failure) => failure,
        Err(BaseEnd::Refused(refusal)) => return Err(refusal),
        Err(BaseEnd::Stopped(stop)) => {
            let unevaluated =
                std::iter::once("stage:base".to_owned()).chain(all_labels()).collect();
            return Err(StudyPlanRefusal::budget(search.receipt(stop, Vec::new(), unevaluated)));
        }
    };

    let mut records = Vec::with_capacity(order.len());
    let mut proposals: Vec<StudyProposal> = Vec::new();
    let mut sufficient: Vec<BTreeSet<usize>> = Vec::new();
    let mut explored = vec!["stage:base".to_owned()];
    let mut stop = None;
    let mut retained = search.peak_bytes();
    let total = order.len();
    for (index, (members, cost_units, sample_budget)) in order.iter().enumerate() {
        let ids = ids_of(members);
        let record = |outcome| StudySubsetRecord {
            candidates: ids.clone(),
            cost_units: *cost_units,
            sample_budget: *sample_budget,
            outcome,
        };
        if stop.is_some() {
            records.push(record(StudySubsetOutcome::Unevaluated));
            continue;
        }
        if let Some(reason) = infeasible(&candidates, members) {
            records.push(record(StudySubsetOutcome::Infeasible { reason }));
            continue;
        }
        let set = members.iter().copied().collect::<BTreeSet<_>>();
        if let Some(by) = sufficient.iter().find(|s| s.is_subset(&set)) {
            let by = by.iter().map(|&i| Arc::clone(&candidates[i].id)).collect();
            records.push(record(StudySubsetOutcome::Dominated { by }));
            continue;
        }
        if proposals.len() == caps.max_proposals {
            stop = Some(StudyPlanStop::ProposalCap);
            records.push(record(StudySubsetOutcome::Unevaluated));
            continue;
        }
        let pending =
            || order[index..].iter().map(|(m, _, _)| subset_label(&ids_of(m))).collect::<Vec<_>>();
        let label = subset_label(&ids);
        // Every candidate previews alone and no two share a regime id or label
        // (validated above), so the subset's preview validates its union once.
        let delta = EvidenceCatalogDelta {
            proposed_regimes: members
                .iter()
                .flat_map(|&i| candidates[i].delta.proposed_regimes.iter().cloned())
                .collect(),
        };
        let preview = delta
            .preview_catalog(catalog)
            .and_then(|preview| delta.with_placeholder_bindings(&preview))
            .map_err(|e| StudyPlanRefusal::invalid_candidate(e.to_string()))?;
        search.begin(retained);
        if let Err(stop_kind) = search.charge(1, preview_bytes(&preview)) {
            stop =
                Some(StudyPlanStop::Budget(search.receipt(stop_kind, explored.clone(), pending())));
            records.push(record(StudySubsetOutcome::Unevaluated));
            continue;
        }
        let decision_cap = (caps.decision_cap)(&label, route.decision_operations());
        let before = search.operations();
        let outcome = match decide_preview(graph, &route, &preview, &mut search, ctx, before) {
            Ok(Preview::Identified(found)) => {
                let operations = found.operations;
                let proposed = delta.proposed_regimes.iter().map(|r| r.id).collect::<BTreeSet<_>>();
                match classify_identified(operations, decision_cap, &found.cited, &proposed) {
                    Identified::OverCap => StudySubsetOutcome::OverDecisionCap { operations },
                    Identified::Sufficient => {
                        proposals.push(proposal(
                            &candidates,
                            members,
                            &delta,
                            *cost_units,
                            *sample_budget,
                            found,
                        ));
                        sufficient.push(set);
                        retained = retained.saturating_add(search.peak_bytes());
                        StudySubsetOutcome::Sufficient { proposal: proposals.len() - 1 }
                    }
                    Identified::CitesNoProposed => StudySubsetOutcome::Refused {
                        message: "the identified formula cites no proposed regime".into(),
                    },
                }
            }
            Ok(Preview::NotIdentified { code, detail, operations }) => {
                if operations > decision_cap {
                    StudySubsetOutcome::OverDecisionCap { operations }
                } else {
                    StudySubsetOutcome::Insufficient { code, detail }
                }
            }
            Ok(Preview::Exhausted(inner)) => {
                stop = Some(StudyPlanStop::Budget(search.receipt(
                    inner.stop,
                    explored.clone(),
                    pending(),
                )));
                StudySubsetOutcome::Unevaluated
            }
            Err(error) if error.is_budget_or_cancel() => {
                let kind = search.stop_of(&error);
                stop =
                    Some(StudyPlanStop::Budget(search.receipt(kind, explored.clone(), pending())));
                StudySubsetOutcome::Unevaluated
            }
            Err(error) => StudySubsetOutcome::Refused { message: error.to_string() },
        };
        if outcome.conclusive() || matches!(outcome, StudySubsetOutcome::OverDecisionCap { .. }) {
            explored.push(label);
        }
        records.push(record(outcome));
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)] // At most 696 subsets.
            progress.report((index + 1) as f64 / total as f64, "study planning");
        }
    }
    let minimal = certified_minimal(&records, proposals.first());
    Ok(StudyPlan {
        route,
        limits: effective,
        failure,
        subsets: records,
        proposals,
        stop,
        minimal,
        operations_consumed: search.operations(),
    })
}

/// How an identified re-run decision classifies a subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Identified {
    /// The decision charged more than the per-decision cap: an arrival decided
    /// by the public route could stop where the plan did not, so the subset is
    /// inconclusive, never sufficient.
    OverCap,
    /// Within the cap, and the formula cites at least one proposed regime.
    Sufficient,
    /// The formula cites no proposed regime. The base failed without these
    /// regimes, so this is not a repair the subset supplies. No integration
    /// input is known to reach it (a re-run that identifies without a proposed
    /// regime would have identified the base); it is kept as a guard and
    /// unit-tested on synthetic citations.
    CitesNoProposed,
}

/// Classify an identified re-run decision that charged `operations` under the
/// per-decision cap `cap`, citing `cited`, against the subset's `proposed`
/// regimes. The cap is checked first: an over-cap decision is never sufficient.
fn classify_identified(
    operations: usize,
    cap: usize,
    cited: &[RegimeId],
    proposed: &BTreeSet<RegimeId>,
) -> Identified {
    if operations > cap {
        Identified::OverCap
    } else if cited.iter().any(|r| proposed.contains(r)) {
        Identified::Sufficient
    } else {
        Identified::CitesNoProposed
    }
}

/// The re-bound mz derivation must cite exactly the regimes the decision
/// reported; anything else is an invalid derivation, never a proposal.
fn recheck_mz_citations(bound: &[RegimeId], cited: &[RegimeId]) -> Result<(), IdentificationError> {
    if bound == cited {
        Ok(())
    } else {
        Err(IdentificationError::invalid_derivation("mz_transport.invalid_derivation"))
    }
}

/// The minimality claim of the top proposal: every feasible subset strictly
/// cheaper than it was decided to a conclusive outcome (no stop, no decision
/// over its cap). Infeasible subsets are outside the universe.
fn certified_minimal(records: &[StudySubsetRecord], top: Option<&StudyProposal>) -> bool {
    top.is_some_and(|top| {
        records.iter().filter(|r| r.cost_units < top.cost_units).all(|r| r.outcome.conclusive())
    })
}

enum BaseEnd {
    Refused(StudyPlanRefusal),
    Stopped(SearchStop),
}

fn route_refusal(error: &IdentificationError) -> BaseEnd {
    match error {
        IdentificationError::UnsupportedInput { code } if code.ends_with("bounds_exceeded") => {
            BaseEnd::Refused(StudyPlanRefusal::bounds_exceeded(format!(
                "the route refuses the query as outside its bounds ({code})"
            )))
        }
        other => BaseEnd::Refused(StudyPlanRefusal::invalid_query(other.to_string())),
    }
}

fn no_failure() -> BaseEnd {
    BaseEnd::Refused(StudyPlanRefusal {
        code: reason_code!("invalid_argument"),
        detail: NO_FAILURE,
        message: "the base decision already identifies the query; there is no failure to repair"
            .into(),
        receipt: None,
    })
}

/// Decide the base catalog on the plan budget and freeze its failure.
fn base_failure(
    graph: &Admg,
    route: &StudyPlanRoute,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<StudyBaseFailure, BaseEnd> {
    match route {
        StudyPlanRoute::Mz(query) => {
            let decision = match decide_mz_transport_shared(graph, query, catalog, search, ctx) {
                Ok(decision) => decision,
                Err(error) if error.is_budget_or_cancel() => {
                    return Err(BaseEnd::Stopped(search.stop_of(&error)));
                }
                Err(error) => return Err(route_refusal(&error)),
            };
            let (code, detail) = (decision.reason_code(), decision.detail_code());
            let facts = match &decision {
                MzTransportDecision::Identified { .. } => return Err(no_failure()),
                MzTransportDecision::Exhausted(receipt) => {
                    return Err(BaseEnd::Stopped(receipt.stop));
                }
                MzTransportDecision::ProvenNonTransportable(obstruction) => {
                    return Err(BaseEnd::Refused(StudyPlanRefusal {
                        code: reason_code!("transport_proven_non_transportable"),
                        detail: THEOREM_LIMITED,
                        message: format!(
                            "the base decision is a replayed checked obstruction at C0 = {:?} \
                             over the declared controllable sets; no study inside them repairs it (a \
                             new population or a wider controllable set is outside this universe)",
                            obstruction.c0()
                        ),
                        receipt: None,
                    }));
                }
                MzTransportDecision::MissingEvidence { detail, .. } => vec![detail.clone()],
                MzTransportDecision::NotCertified(inspection) => {
                    stage_facts(inspection.stages.iter().map(|s| (s.stage.as_str(), s.outcome)))
                }
            };
            Ok(StudyBaseFailure {
                code: code.unwrap_or_default(),
                detail: detail.unwrap_or_default(),
                facts,
            })
        }
        StudyPlanRoute::Mixed(query) => {
            let decision = match decide_mixed_source_shared(graph, query, catalog, search, ctx) {
                Ok(decision) => decision,
                Err(error) if error.is_budget_or_cancel() => {
                    return Err(BaseEnd::Stopped(search.stop_of(&error)));
                }
                Err(error) => return Err(route_refusal(&error)),
            };
            let (code, detail) = (decision.reason_code(), decision.detail_code());
            let facts = match &decision {
                MixedSourceDecision::Identified { .. } | MixedSourceDecision::NamedRoute { .. } => {
                    return Err(no_failure());
                }
                MixedSourceDecision::Exhausted(receipt) => {
                    return Err(BaseEnd::Stopped(receipt.stop));
                }
                MixedSourceDecision::MissingEvidence(missing) => missing
                    .leaves
                    .iter()
                    .map(|leaf| {
                        format!(
                            "missing_joint:regime={}:variables={:?}:intervened={:?}",
                            leaf.regime.raw(),
                            raw(&leaf.variables),
                            raw(&leaf.intervened)
                        )
                    })
                    .chain(stage_facts(
                        missing.stages.iter().map(|s| (s.stage.as_str(), s.outcome)),
                    ))
                    .collect(),
                MixedSourceDecision::NotCertified(inspection) => {
                    stage_facts(inspection.stages.iter().map(|s| (s.stage.as_str(), s.outcome)))
                }
            };
            Ok(StudyBaseFailure {
                code: code.unwrap_or_default(),
                detail: detail.unwrap_or_default(),
                facts,
            })
        }
    }
}

fn raw(variables: &[VariableId]) -> Vec<u32> {
    variables.iter().map(|v| v.raw()).collect()
}

/// `stage:<stage>=<outcome>` for each stage of a base decision, in order.
fn stage_facts<'a>(stages: impl IntoIterator<Item = (&'a str, &'a str)>) -> Vec<String> {
    stages.into_iter().map(|(stage, outcome)| format!("stage:{stage}={outcome}")).collect()
}

/// A re-run decision on a preview catalog that identified and re-checked.
struct Found {
    stage: String,
    arena: CausalExprArena,
    root: ExprId,
    cited: Vec<RegimeId>,
    /// `(proof step, regime)` of each X9 input step.
    steps: Vec<(usize, RegimeId)>,
    derivation: StudyProposalDerivation,
    operations: usize,
}

enum Preview {
    Identified(Box<Found>),
    NotIdentified { code: &'static str, detail: &'static str, operations: usize },
    Exhausted(SearchReceipt),
}

/// Re-run the route's decision on a preview catalog and re-check a derivation
/// against it.
fn decide_preview(
    graph: &Admg,
    route: &StudyPlanRoute,
    preview: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
    before: usize,
) -> Result<Preview, IdentificationError> {
    match route {
        StudyPlanRoute::Mz(query) => {
            let decision = decide_mz_transport_shared(graph, query, preview, search, ctx)?;
            let operations = search.operations() - before;
            match decision {
                MzTransportDecision::Identified { derivation, cited } => {
                    let bound = bind_mz_transport_catalog(graph, &derivation, preview)?;
                    recheck_mz_citations(bound.cited_regimes(), &cited)?;
                    let stage = match derivation.route() {
                        MzTransportRoute::TargetOnly => "mz_transport:target_only".to_owned(),
                        MzTransportRoute::SingleSource { population } => {
                            format!("mz_transport:single_source:{population}")
                        }
                        MzTransportRoute::Combined { .. } => "mz_transport:combined".to_owned(),
                    };
                    Ok(Preview::Identified(Box::new(Found {
                        stage,
                        arena: bound.arena().clone(),
                        root: bound.root(),
                        cited: cited.to_vec(),
                        steps: Vec::new(),
                        derivation: StudyProposalDerivation::Mz(Box::new(derivation.to_record())),
                        operations,
                    })))
                }
                MzTransportDecision::Exhausted(receipt) => Ok(Preview::Exhausted(receipt)),
                other => Ok(Preview::NotIdentified {
                    code: other.reason_code().unwrap_or_default(),
                    detail: other.detail_code().unwrap_or_default(),
                    operations,
                }),
            }
        }
        StudyPlanRoute::Mixed(query) => {
            let decision = decide_mixed_source_shared(graph, query, preview, search, ctx)?;
            let operations = search.operations() - before;
            match decision {
                MixedSourceDecision::Identified { derivation, cited, .. } => {
                    // Binding replays every step through the independent checker.
                    let bound = bind_mixed_source_catalog(graph, &derivation, preview)?;
                    Ok(Preview::Identified(Box::new(Found {
                        stage: "mixed_source:rule_search".to_owned(),
                        arena: bound.arena().clone(),
                        root: bound.root(),
                        cited: cited.to_vec(),
                        steps: derivation.leaves().iter().map(|(i, l)| (*i, l.regime)).collect(),
                        derivation: StudyProposalDerivation::Mixed(Box::new(
                            derivation.to_record(),
                        )),
                        operations,
                    })))
                }
                MixedSourceDecision::NamedRoute { route: named, .. } => {
                    match rerun_named(graph, query, named, preview, search, ctx)? {
                        Some((arena, root)) => {
                            let cited = leaf_regimes(&arena, root);
                            Ok(Preview::Identified(Box::new(Found {
                                stage: format!("mixed_source:named:{named}"),
                                arena,
                                root,
                                cited,
                                steps: Vec::new(),
                                derivation: StudyProposalDerivation::Named { route: named },
                                operations,
                            })))
                        }
                        None => Ok(Preview::NotIdentified {
                            code: reason_code!("transport_not_certified"),
                            detail: "mixed_search.invalid_derivation",
                            operations,
                        }),
                    }
                }
                MixedSourceDecision::Exhausted(receipt) => Ok(Preview::Exhausted(receipt)),
                other => Ok(Preview::NotIdentified {
                    code: other.reason_code().unwrap_or_default(),
                    detail: other.detail_code().unwrap_or_default(),
                    operations,
                }),
            }
        }
    }
}

/// Re-run the X9 named route that identified the preview on the shared budget
/// and bind its derivation to the preview (`NamedRoute` carries no derivation).
/// `None` when the re-run does not identify.
fn rerun_named(
    graph: &Admg,
    query: &MixedSourceQuery,
    named: &'static str,
    preview: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<Option<(CausalExprArena, ExprId)>, IdentificationError> {
    // `query` is the plan's canonical route (sources by population), the form
    // the mixed decision ran its named stage on.
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
    let bound = |result: CatalogTransportResult| match result {
        CatalogTransportResult::Identified(bound) => Some((bound.arena().clone(), bound.root())),
        _ => None,
    };
    Ok(match named {
        "target_first_sid" => {
            let classical = ClassicalTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                source: Arc::from("<no source>"),
                target: Arc::clone(&query.target),
            };
            bound(identify_catalog_transport_metered(
                &shared,
                &classical,
                preview,
                search.meter(),
                ctx,
            )?)
        }
        "z_transport" => {
            let Some(source) = query.sources.first() else { return Ok(None) };
            let diagram =
                SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                    .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
            let single = ZTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                controllable: Arc::clone(&source.controllable),
                experiment_assignment: Arc::clone(&source.experiment_assignment),
                source: Arc::clone(&source.population),
                target: Arc::clone(&query.target),
            };
            match decide_z_transport_reporting(
                &diagram,
                &single,
                preview,
                search.meter(),
                ctx,
                &mut None,
            )? {
                ZTransportDecision::Identified(z) => {
                    let bound = bind_z_transport_catalog(&diagram, &single, &z, preview)?;
                    Some((bound.arena().clone(), bound.root()))
                }
                _ => None,
            }
        }
        "mz_transport" => {
            let mz = MzTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                target: Arc::clone(&query.target),
                sources: Arc::clone(&query.sources),
            };
            match decide_mz_transport_shared(graph, &mz, preview, search, ctx)? {
                MzTransportDecision::Identified { derivation, .. } => {
                    let bound = bind_mz_transport_catalog(graph, &derivation, preview)?;
                    Some((bound.arena().clone(), bound.root()))
                }
                MzTransportDecision::Exhausted(receipt) => {
                    return Err(stop_error(receipt.stop));
                }
                _ => None,
            }
        }
        "meta_transport" => {
            let meta = MetaTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                target: Arc::clone(&query.target),
                sources: query
                    .sources
                    .iter()
                    .map(|source| MetaSource {
                        population: source.population.to_string(),
                        selections: source.selection_targets.iter().map(|v| v.raw()).collect(),
                    })
                    .collect(),
            };
            bound(identify_meta_catalog_metered(graph, &meta, preview, search.meter(), ctx)?)
        }
        _ => None,
    })
}

const fn stop_error(stop: SearchStop) -> IdentificationError {
    match stop {
        SearchStop::Cancelled => IdentificationError::Cancelled,
        SearchStop::Memory => IdentificationError::budget(crate::IdentificationBudget::Memory),
        SearchStop::Operations | SearchStop::Depth => {
            IdentificationError::budget(crate::IdentificationBudget::Steps)
        }
    }
}

/// One distribution leaf of a formula: its regime when the binding tagged it
/// (a leaf bound to several per-level regimes stays untagged), population and
/// intervention set.
struct BoundLeaf {
    regime: Option<RegimeId>,
    population: Arc<str>,
    intervened: Vec<VariableId>,
    variables: Vec<VariableId>,
    conditioned: Vec<VariableId>,
}

impl BoundLeaf {
    /// Whether this leaf reads `regime`: tagged with it, or untagged with the
    /// regime's population and intervention set while the formula cites it.
    fn reads(&self, regime: &antecedent_core::EvidenceRegime, cited: &[RegimeId]) -> bool {
        if let Some(id) = self.regime {
            return id == regime.id;
        }
        let mut intervened = regime.interventions.to_vec();
        intervened.sort_unstable();
        cited.contains(&regime.id)
            && self.population == regime.population
            && self.intervened == intervened
    }
}

/// Every distribution leaf reachable from `root`.
fn bound_leaves(arena: &CausalExprArena, root: ExprId) -> Vec<BoundLeaf> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                population,
                regime,
                ..
            } => {
                let sorted = |v: &[VariableId]| {
                    let mut v = v.to_vec();
                    v.sort_unstable();
                    v
                };
                out.push(BoundLeaf {
                    regime: *regime,
                    population: Arc::from(arena.population(*population)),
                    intervened: sorted(
                        &arena
                            .intervention_assignments(*intervention)
                            .iter()
                            .map(|a| a.variable)
                            .collect::<Vec<_>>(),
                    ),
                    variables: sorted(arena.var_set(*variables)),
                    conditioned: sorted(arena.var_set(*conditioned_on)),
                });
            }
            ExprNode::Kernel { body, .. } => pending.push(*body),
            ExprNode::Product(list) => pending.extend(arena.list(*list).iter().copied()),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                pending.push(*expr);
            }
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator]);
            }
            ExprNode::Expectation { distribution, .. } => pending.push(*distribution),
            ExprNode::Contrast { left, right, .. } => pending.extend([*left, *right]),
        }
    }
    out
}

fn leaf_regimes(arena: &CausalExprArena, root: ExprId) -> Vec<RegimeId> {
    let mut cited =
        bound_leaves(arena, root).into_iter().filter_map(|leaf| leaf.regime).collect::<Vec<_>>();
    cited.sort_unstable();
    cited.dedup();
    cited
}

/// Package a sufficient subset: every cited proposed regime with the factors
/// that read it and the margin the proof actually needs from it.
fn proposal(
    candidates: &[StudyPlanCandidate],
    members: &[usize],
    delta: &EvidenceCatalogDelta,
    cost_units: u64,
    sample_budget: u64,
    found: Box<Found>,
) -> StudyProposal {
    let leaves = bound_leaves(&found.arena, found.root);
    let mut cited = found.cited.clone();
    cited.sort_unstable();
    cited.dedup();
    let mut repairs = Vec::new();
    let mut used = BTreeSet::new();
    for regime in delta.proposed_regimes.iter() {
        let cites_leaf = leaves.iter().any(|leaf| leaf.regime == Some(regime.id));
        if !cited.contains(&regime.id) && !cites_leaf {
            continue;
        }
        let Some(&owner) = members
            .iter()
            .find(|&&i| candidates[i].delta.proposed_regimes.iter().any(|r| r.id == regime.id))
        else {
            continue;
        };
        used.insert(owner);
        let mut factors = Vec::new();
        let mut required = BTreeSet::new();
        for leaf in leaves.iter().filter(|leaf| leaf.reads(regime, &cited)) {
            required.extend(leaf.variables.iter().copied());
            required.extend(leaf.conditioned.iter().copied());
            let factor = StudyFactor {
                variables: leaf.variables.clone(),
                conditioned_on: leaf.conditioned.clone(),
            };
            if !factors.contains(&factor) {
                factors.push(factor);
            }
        }
        let proof_steps = found
            .steps
            .iter()
            .filter(|(_, r)| *r == regime.id)
            .map(|(step, _)| *step)
            .collect::<Vec<_>>();
        if factors.is_empty() {
            // An X9 input step reads the whole declared joint.
            required.extend(regime.measured.iter().copied());
        }
        let mut intervened = regime.interventions.to_vec();
        intervened.sort_unstable();
        let mut declared = regime.measured.to_vec();
        declared.sort_unstable();
        repairs.push(StudyRepair {
            candidate: Arc::clone(&candidates[owner].id),
            regime: regime.id,
            population: Arc::clone(&regime.population),
            intervened,
            factors,
            required_margin: required.into_iter().collect(),
            declared_margin: declared,
            proof_steps,
        });
    }
    StudyProposal {
        candidates: members.iter().map(|&i| Arc::clone(&candidates[i].id)).collect(),
        cost_units,
        sample_budget,
        stage: found.stage,
        cited_regimes: cited,
        repairs,
        uncited_candidates: members
            .iter()
            .filter(|i| !used.contains(i))
            .map(|&i| Arc::clone(&candidates[i].id))
            .collect(),
        derivation: found.derivation,
        decision_operations: found.operations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, cost: u64, samples: u64) -> StudyPlanCandidate {
        StudyPlanCandidate {
            id: Arc::from(id),
            delta: EvidenceCatalogDelta { proposed_regimes: Arc::from([]) },
            cost_units: cost,
            sample_budget: samples,
            requires: Arc::from([]),
            conflicts: Arc::from([]),
        }
    }

    #[test]
    fn subsets_are_ranked_by_cost_then_samples_then_size_then_ids() {
        let candidates = vec![candidate("a", 2, 0), candidate("b", 1, 5), candidate("c", 1, 0)];
        let order = subsets(&candidates)
            .into_iter()
            .map(|(m, cost, samples)| {
                (m.iter().map(|&i| candidates[i].id.to_string()).collect::<Vec<_>>(), cost, samples)
            })
            .collect::<Vec<_>>();
        assert_eq!(order.len(), 7);
        let labels = order.iter().map(|(ids, c, s)| (ids.join("+"), *c, *s)).collect::<Vec<_>>();
        assert_eq!(
            labels,
            [
                ("c".into(), 1, 0),
                ("b".into(), 1, 5),
                ("a".into(), 2, 0),
                ("b+c".into(), 2, 5),
                ("a+c".into(), 3, 0),
                ("a+b".into(), 3, 5),
                ("a+b+c".into(), 4, 5),
            ]
        );
        // Every proper subset of a subset comes strictly earlier (positive costs).
        for (index, (ids, _, _)) in order.iter().enumerate() {
            for (earlier, (other, _, _)) in order.iter().enumerate() {
                if other.len() < ids.len() && other.iter().all(|x| ids.contains(x)) {
                    assert!(earlier < index, "{other:?} must precede {ids:?}");
                }
            }
        }
    }

    #[test]
    fn a_universe_of_sixteen_has_six_hundred_ninety_six_subsets_of_at_most_three() {
        let candidates = (0..16).map(|i| candidate(&format!("c{i:02}"), 1, 0)).collect::<Vec<_>>();
        assert_eq!(subsets(&candidates).len(), 16 + 120 + 560);
    }

    fn record(cost: u64, outcome: StudySubsetOutcome) -> StudySubsetRecord {
        StudySubsetRecord {
            candidates: vec![Arc::from(format!("c{cost}"))],
            cost_units: cost,
            sample_budget: 0,
            outcome,
        }
    }

    fn top(cost: u64) -> StudyProposal {
        StudyProposal {
            candidates: Vec::new(),
            cost_units: cost,
            sample_budget: 0,
            stage: String::new(),
            cited_regimes: Vec::new(),
            repairs: Vec::new(),
            uncited_candidates: Vec::new(),
            derivation: StudyProposalDerivation::Named { route: "z_transport" },
            decision_operations: 0,
        }
    }

    #[test]
    fn minimality_needs_every_strictly_cheaper_subset_decided() {
        let insufficient = || StudySubsetOutcome::Insufficient { code: "c", detail: "d" };
        let decided = [
            record(1, insufficient()),
            record(1, StudySubsetOutcome::Infeasible { reason: "requires" }),
            record(2, StudySubsetOutcome::Sufficient { proposal: 0 }),
            record(3, StudySubsetOutcome::Unevaluated),
        ];
        assert!(certified_minimal(&decided, Some(&top(2))));
        assert!(!certified_minimal(&decided, None));
        // An inconclusive decision below the top withdraws the claim ...
        let mut over = decided.clone();
        over[0].outcome = StudySubsetOutcome::OverDecisionCap { operations: 4097 };
        assert!(!certified_minimal(&over, Some(&top(2))));
        // ... as does an unevaluated cheaper subset; an equal-cost one does not.
        let mut left = decided.clone();
        left[0].outcome = StudySubsetOutcome::Unevaluated;
        assert!(!certified_minimal(&left, Some(&top(2))));
        let mut tie = decided.to_vec();
        tie.push(record(2, StudySubsetOutcome::Unevaluated));
        assert!(certified_minimal(&tie, Some(&top(2))));
    }

    #[test]
    fn a_decision_above_the_public_cap_is_over_it_and_at_the_cap_is_not() {
        let mz = StudyPlanRoute::Mz(MzTransportQuery {
            outcomes: Arc::from([]),
            treatments: Arc::from([]),
            target: Arc::from("t"),
            sources: Arc::from([]),
        });
        let mixed = StudyPlanRoute::Mixed(MixedSourceQuery {
            outcomes: Arc::from([]),
            treatments: Arc::from([]),
            target: Arc::from("t"),
            sources: Arc::from([]),
        });
        let proposed = [RegimeId::from_raw(1)].into_iter().collect::<BTreeSet<_>>();
        let cites = [RegimeId::from_raw(1)];
        for (route, cap) in [(&mz, 4096), (&mixed, 20_000)] {
            assert_eq!(route.decision_operations(), cap);
            let at = classify_identified(cap, route.decision_operations(), &cites, &proposed);
            let over = classify_identified(cap + 1, route.decision_operations(), &cites, &proposed);
            assert_eq!((at, over), (Identified::Sufficient, Identified::OverCap));
        }
        assert_eq!((mz.max_depth(), mixed.max_depth()), (24, 16));
        assert!(!StudySubsetOutcome::OverDecisionCap { operations: 1 }.conclusive());
    }

    #[test]
    fn requires_and_conflicts_mark_subsets_infeasible() {
        let mut a = candidate("a", 1, 0);
        a.requires = Arc::from([Arc::from("b")]);
        let mut c = candidate("c", 1, 0);
        c.conflicts = Arc::from([Arc::from("b")]);
        let candidates = vec![a, candidate("b", 1, 0), c];
        assert_eq!(infeasible(&candidates, &[0]), Some("requires"));
        assert_eq!(infeasible(&candidates, &[0, 1]), None);
        assert_eq!(infeasible(&candidates, &[1, 2]), Some("conflicts"));
    }
    #[test]
    fn classification_needs_the_cap_first_and_a_cited_proposed_regime() {
        // Synthetic citations: no integration input reaches the "cites no
        // proposed regime" branch (a re-run that identifies without a proposed
        // regime would have identified the base), so it is tested here.
        let r = RegimeId::from_raw;
        let proposed = [r(5), r(6)].into_iter().collect::<BTreeSet<_>>();
        assert_eq!(classify_identified(10, 10, &[r(0), r(6)], &proposed), Identified::Sufficient);
        assert_eq!(
            classify_identified(10, 10, &[r(0), r(1)], &proposed),
            Identified::CitesNoProposed
        );
        assert_eq!(classify_identified(10, 10, &[], &proposed), Identified::CitesNoProposed);
        // Over the cap wins even when a proposed regime is cited.
        assert_eq!(classify_identified(11, 10, &[r(5)], &proposed), Identified::OverCap);
    }

    #[test]
    fn a_rebound_mz_derivation_must_cite_exactly_the_reported_regimes() {
        let r = RegimeId::from_raw;
        assert!(recheck_mz_citations(&[r(0), r(1)], &[r(0), r(1)]).is_ok());
        for (bound, cited) in [(&[r(0)][..], &[r(0), r(1)][..]), (&[r(0), r(2)], &[r(0), r(1)])] {
            let error = recheck_mz_citations(bound, cited).unwrap_err();
            assert!(error.to_string().contains("invalid_derivation"), "{error}");
        }
    }

    // ------------------------------------------------------------------
    // End-to-end plans on R-443 Fig. 1(c,d) with test-lowered caps: the
    // public caps (4096 operations per mz decision, 32 proposals) are not
    // reachable on a small fixture, so these run the whole planner with the
    // caps lowered to reach `OverDecisionCap` and `ProposalCap`.

    mod fig1 {
        use super::super::*;
        use antecedent_core::{
            DependenceGroup, Environment, EvidenceKind, EvidenceRegime, InterventionAssignment,
            RegimeBinding, SamplingDesign, Value, VariableCoordinate, VariableDomain,
        };
        use antecedent_graph::DenseNodeId;

        pub(super) const Z1: u32 = 0;
        pub(super) const X: u32 = 1;
        pub(super) const Z2: u32 = 2;
        pub(super) const Y: u32 = 3;

        fn v(i: u32) -> VariableId {
            VariableId::from_raw(i)
        }

        pub(super) fn graph() -> Admg {
            let mut g = Admg::with_variables(4);
            for (a, b) in [(Z1, X), (X, Z2), (Z2, Y)] {
                g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
            for (a, b) in [(Z1, X), (Z1, Z2), (Z1, Y)] {
                g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
            g
        }

        pub(super) fn route() -> StudyPlanRoute {
            let spec =
                |population: &str, controllable: u32, selection: [u32; 2]| ZTransportSourceSpec {
                    population: Arc::from(population),
                    controllable: Arc::from([v(controllable)]),
                    experiment_assignment: Arc::from([]),
                    selection_targets: selection.into_iter().map(v).collect::<Vec<_>>().into(),
                };
            let mut b = spec("b", Z1, [Z1, Y]);
            b.experiment_assignment =
                Arc::from([InterventionAssignment { variable: v(Z1), value: Value::Bool(false) }]);
            StudyPlanRoute::Mz(MzTransportQuery {
                outcomes: Arc::from([v(Y)]),
                treatments: Arc::from([v(X)]),
                target: Arc::from("target"),
                sources: vec![spec("a", Z2, [Z1, Z2]), b].into(),
            })
        }

        pub(super) fn regime(
            id: u32,
            kind: EvidenceKind,
            population: &str,
            on: &[(u32, bool)],
        ) -> EvidenceRegime {
            let intervened = on.iter().map(|(x, _)| *x).collect::<Vec<_>>();
            EvidenceRegime::try_new(
                RegimeId::from_raw(id),
                if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
                kind,
                intervened.iter().copied().map(v).collect::<Vec<_>>(),
                on.iter()
                    .map(|(x, l)| InterventionAssignment {
                        variable: v(*x),
                        value: Value::Bool(*l),
                    })
                    .collect::<Vec<_>>(),
                (0..4).filter(|i| !intervened.contains(i)).map(v).collect::<Vec<_>>(),
                population,
                DistributionAvailability::Joint,
            )
            .unwrap()
        }

        pub(super) fn base() -> EvidenceCatalog {
            let regimes = vec![regime(0, EvidenceKind::Available, "target", &[])];
            let bindings = vec![RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(0),
                snapshot_identity: Arc::from("snapshot-0"),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            }];
            let coordinates = (0..4)
                .map(|i| VariableCoordinate {
                    variable: v(i),
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>();
            let environments = ["target", "a", "b"]
                .into_iter()
                .map(|p| Environment::try_new(p, coordinates.clone(), []).unwrap())
                .collect::<Vec<_>>();
            EvidenceCatalog::try_new(environments, regimes, bindings, None).unwrap()
        }

        pub(super) fn candidate(
            base: &EvidenceCatalog,
            id: &str,
            cost: u64,
            regimes: Vec<EvidenceRegime>,
        ) -> StudyPlanCandidate {
            StudyPlanCandidate {
                id: Arc::from(id),
                delta: EvidenceCatalogDelta::try_new(base, regimes).unwrap(),
                cost_units: cost,
                sample_budget: 0,
                requires: Arc::from([]),
                conflicts: Arc::from([]),
            }
        }

        pub(super) fn proposed(id: u32, population: &str, on: &[(u32, bool)]) -> EvidenceRegime {
            regime(id, EvidenceKind::Proposed, population, on)
        }

        /// a's do(Z2) at both levels, b's do(Z1 = 0), an observational b study,
        /// and a second b trial at the same cost (two sufficient pairs).
        pub(super) fn candidates(base: &EvidenceCatalog) -> Vec<StudyPlanCandidate> {
            vec![
                candidate(
                    base,
                    "a_do_z2",
                    3,
                    vec![proposed(1, "a", &[(Z2, false)]), proposed(2, "a", &[(Z2, true)])],
                ),
                candidate(base, "b_do_z1", 2, vec![proposed(3, "b", &[(Z1, false)])]),
                candidate(base, "b_do_z1_again", 2, vec![proposed(5, "b", &[(Z1, false)])]),
                candidate(base, "b_observe", 1, vec![proposed(4, "b", &[])]),
            ]
        }
    }

    fn fig1_plan(caps: PlanCaps) -> StudyPlan {
        let base = fig1::base();
        plan_with_caps(
            &fig1::graph(),
            &fig1::route(),
            &base,
            &fig1::candidates(&base),
            StudyPlanLimits::default(),
            &ExecutionContext::for_tests(1),
            caps,
        )
        .unwrap()
    }

    const PUBLIC: PlanCaps = PlanCaps { decision_cap: |_, cap| cap, max_proposals: 32 };

    #[test]
    fn a_lowered_decision_cap_makes_decisions_inconclusive_and_withdraws_minimality() {
        let public = fig1_plan(PUBLIC);
        assert_eq!(public.proposals.len(), 2);
        assert!(public.minimal);
        assert!(public.subsets.iter().all(|s| s.outcome.status() != "over_decision_cap"));
        let top = &public.proposals[0];
        assert!(top.decision_operations > 0);
        // Cap zero everywhere: every decided subset is over the cap, never sufficient.
        let zero = fig1_plan(PlanCaps { decision_cap: |_, _| 0, ..PUBLIC });
        assert!(zero.proposals.is_empty() && !zero.minimal);
        assert_eq!(zero.status(), Some(("transport_not_certified", "study_plan.none_certified")));
        assert!(zero.subsets.iter().all(|s| matches!(
            s.outcome,
            StudySubsetOutcome::OverDecisionCap { operations } if operations > 0
        ) || matches!(
            s.outcome,
            StudySubsetOutcome::Infeasible { .. }
        )));
        // The top pair's own decision one operation over its cap: never sufficient.
        let pair = fig1_plan(PlanCaps {
            decision_cap: |label, cap| if label == "subset:[a_do_z2,b_do_z1]" { 0 } else { cap },
            ..PUBLIC
        });
        let record = pair.subsets.iter().find(|s| s.label() == "subset:[a_do_z2,b_do_z1]").unwrap();
        assert!(matches!(record.outcome, StudySubsetOutcome::OverDecisionCap { .. }));
        assert_eq!(
            pair.proposals[0].candidates,
            [Arc::from("a_do_z2"), Arc::from("b_do_z1_again")]
        );
        // An insufficient, strictly cheaper subset over its cap: the proposal
        // stands, the cheaper subset is inconclusive, and minimality is withdrawn.
        let cheaper = fig1_plan(PlanCaps {
            decision_cap: |label, cap| if label == "subset:[b_observe]" { 0 } else { cap },
            ..PUBLIC
        });
        assert_eq!(cheaper.proposals[0].candidates, top.candidates);
        let record = cheaper.subsets.iter().find(|s| s.label() == "subset:[b_observe]").unwrap();
        assert!(matches!(record.outcome, StudySubsetOutcome::OverDecisionCap { .. }));
        assert!(!cheaper.minimal, "an inconclusive cheaper subset withdraws minimality");
        // An over-cap subset of the top's own cost leaves the claim standing.
        let tie = fig1_plan(PlanCaps {
            decision_cap: |label, cap| {
                if label == "subset:[b_do_z1,b_do_z1_again,b_observe]" { 0 } else { cap }
            },
            ..PUBLIC
        });
        let record = tie
            .subsets
            .iter()
            .find(|s| s.label() == "subset:[b_do_z1,b_do_z1_again,b_observe]")
            .unwrap();
        assert_eq!(record.cost_units, top.cost_units);
        assert!(matches!(record.outcome, StudySubsetOutcome::OverDecisionCap { .. }));
        assert!(tie.minimal);
    }

    #[test]
    fn the_proposal_cap_stops_the_plan_and_leaves_the_rest_unevaluated() {
        let one = fig1_plan(PlanCaps { max_proposals: 1, ..PUBLIC });
        assert_eq!(one.stop, Some(StudyPlanStop::ProposalCap));
        assert_eq!(one.proposals.len(), 1);
        assert_eq!(one.status(), None);
        assert!(one.receipt().is_none());
        // The second sufficient pair is left unevaluated, never decided.
        let second = subset_label(&[Arc::from("a_do_z2"), Arc::from("b_do_z1_again")]);
        let record = one.subsets.iter().find(|s| s.label() == second).unwrap();
        assert_eq!(record.outcome, StudySubsetOutcome::Unevaluated);
        // The top proposal was found before the cap: it stays minimal.
        assert!(one.minimal);
        assert_eq!(fig1_plan(PUBLIC).stop, None);
    }
}
