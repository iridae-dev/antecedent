//! Study planning over the mz-transport and mixed-source catalogs.
//!
//! A planner receives a failed mz-transport or mixed-source decision and a declared universe of
//! candidate studies, each already compiled
//! to a typed [`EvidenceCatalogDelta`] with an integer cost. It enumerates the
//! feasible subsets in cost order, by increasing subset size as far as the plan's operation
//! budget can pay for (all of them when the budget covers the whole power set, otherwise the
//! plan ends with a budget receipt naming the sizes left unexamined) and re-runs the same route's decision on each subset's preview
//! catalog, all under ONE shared [`SearchBudget`]: the base decision, every
//! subset entry and every step of every re-run decision charge it, with the
//! live bytes of the proposals already kept retained under every later charge.
//!
//! A subset is **sufficient** only when the re-run decision returns
//! `Identified` and its derivation re-checks against the preview catalog (an mz
//! derivation re-binds; a mixed-source derivation replays through the independent
//! checker; a mixed-source named route is re-run on the shared budget and must bind),
//! and the formula cites at least one proposed regime. The plan claims no
//! probability of success and no numeric result: a preview is never available
//! evidence, and arrival re-identifies from the real catalog.
//!
//! What the plan does and does not claim:
//!
//! - "No sufficient subset" is `study_plan.none_certified` within this universe
//!   and rule set, never an impossibility: the mixed-source search is sound but incomplete and the mz route's
//!   obstruction decider is partial. The only theorem-limited refusal is a base
//!   mz-transport decision that is a replayed checked obstruction, which concerns the
//!   declared controllable sets, so no study inside them repairs it; a new
//!   population or a wider controllable set is outside this universe.
//! - Minimality is claimed only when every feasible subset strictly cheaper
//!   than the top proposal was evaluated to a conclusive outcome without a stop:
//!   "cost-minimal among the verified-derivable subsets of this declared
//!   universe, under this search and rule set", never "minimal sufficient
//!   evidence". When the subset sizes were truncated by the budget, minimality is
//!   claimed only if no unexamined subset can be cheaper than the top proposal,
//!   and a cheaper subset the route refuses (for
//!   example over the mixed-source usable-distribution bound) or leaves underivable keeps
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

mod preview;
mod subsets;
use preview::{Preview, decide_preview, proposal};
use subsets::{infeasible, subset_label, subset_size_limit, subsets, validate_candidates};

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    DEFAULT_SEARCH_MEMORY_BYTES, DistributionAvailability, EvidenceCatalog, EvidenceCatalogDelta,
    EvidenceKind, ExecutionContext, LawOrigin, RegimeId, RegimeKind, SamplingSelection,
    SearchBudget, SearchLimits, SearchReceipt, SearchStop, VariableId, reason_code,
};
use antecedent_expr::{CausalExprArena, ExprId, ExprNode};
use antecedent_graph::{Admg, BitSet, DenseNodeId, SelectionDiagram};

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

/// Candidate studies a plan accepts whatever its operation budget (the floor of
/// [`StudyPlanLimits::max_candidates`]); a larger budget admits as many candidates as it has
/// operations.
pub const STUDY_PLAN_MAX_CANDIDATES: usize = 16;
/// Regimes one candidate study could propose under the earlier fixed ceiling. Not enforced.
pub const STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE: usize = 8;
/// Largest subset size under the earlier fixed ceiling. Not enforced: subsets are enumerated
/// by increasing size as far as the operation budget pays for them, and a truncation is a
/// budget stop with a receipt.
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
pub(crate) const STUDY_PLAN_MAX_COST_UNITS: u64 = u64::MAX / 4;
/// Version of the ranking key: cost units, sample budget, subset size, sorted ids.
pub const STUDY_PLAN_RANKING: &str = "study_plan_ranking.v1";

/// Candidates, regimes, subset size or limits beyond a declared bound.
const BOUNDS_EXCEEDED: &str = antecedent_core::reason_details::study_plan::BOUNDS_EXCEEDED;
/// A candidate declaration the route cannot accept.
const INVALID_CANDIDATE: &str = antecedent_core::reason_details::study_plan::INVALID_CANDIDATE;
/// The route refuses the base query or catalog.
const INVALID_QUERY: &str = antecedent_core::reason_details::study_plan::INVALID_QUERY;
/// The base decision already identifies.
const NO_FAILURE: &str = antecedent_core::reason_details::study_plan::NO_FAILURE_TO_REPAIR;
/// A replayed mz-transport obstruction over the declared controllable sets.
const THEOREM_LIMITED: &str = antecedent_core::reason_details::study_plan::THEOREM_LIMITED;
/// No subset of the universe is sufficient; not an impossibility claim.
const NONE_CERTIFIED: &str = antecedent_core::reason_details::study_plan::NONE_CERTIFIED;
/// A plan budget stop.
const BUDGET: &str = antecedent_core::reason_details::study_plan::BUDGET;

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
    /// The mz-transport multi-source limited-experiment route.
    Mz(MzTransportQuery),
    /// The mixed-source proof search.
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
    /// [`STUDY_PLAN_MEMORY_BYTES`]), and the number of candidates against
    /// [`Self::max_candidates`], the universe the operation budget can enter. The regimes per
    /// candidate are not bounded here: the plan charges their preview bytes to its budget.
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
        if regimes_per_candidate.len() > self.max_candidates() {
            return Err(StudyPlanRefusal::bounds_exceeded(format!(
                "a plan under {} operations admits at most {} candidates, got {}: every subset \
                 entered charges an operation, and validation and enumeration run before the \
                 budget is entered",
                self.search.operations,
                self.max_candidates(),
                regimes_per_candidate.len()
            )));
        }
        Ok(())
    }

    /// The most candidates a plan under these limits accepts: as many as its operations (each
    /// singleton subset charges one), and never fewer than [`STUDY_PLAN_MAX_CANDIDATES`], so a
    /// small budget still names the subsets it could not afford in its receipt.
    #[must_use]
    pub const fn max_candidates(&self) -> usize {
        if self.search.operations > STUDY_PLAN_MAX_CANDIDATES {
            self.search.operations
        } else {
            STUDY_PLAN_MAX_CANDIDATES
        }
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
    /// The decision spent more than the route's public per-decision cap and was cut off
    /// at the charge that crossed it: inconclusive, never sufficient.
    OverDecisionCap {
        /// Operations the decision charged, the crossing charge included.
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
    /// the proof reads the regime only as a mixed-source input step (no factor of its
    /// own). A smaller declared margin containing it is not verified unless
    /// declared as its own candidate.
    pub required_margin: Vec<VariableId>,
    /// The margin the candidate declared.
    pub declared_margin: Vec<VariableId>,
    /// mixed-source proof steps (input steps) that cite it; empty on other routes.
    pub proof_steps: Vec<usize>,
}

/// The checked derivation behind a proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StudyProposalDerivation {
    /// A mz-transport derivation (its record, including the plan-budget receipt).
    Mz(Box<MzTransportDerivationRecord>),
    /// A mixed-source rule-search derivation.
    Mixed(Box<MixedSourceDerivationRecord>),
    /// A mixed-source named route, re-run on the shared budget and bound.
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
    /// subsets (every strictly cheaper one decided without a stop or an over-cap
    /// decision, and no subset left unexamined by a size truncation can be cheaper).
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

/// Plan the cheapest study additions that would make the failed mz-transport or mixed-source
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
/// identifies, `study_plan.theorem_limited` for a replayed mz-transport obstruction, and
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

#[allow(clippy::too_many_lines, reason = "One pass: validate, freeze the base, then every subset")]
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
    let max_size = subset_size_limit(candidates.len(), limits.search.operations);
    let order = subsets(&candidates, max_size)?;
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
    // Each sufficient subset as a word-wise mask over the candidates, with its members.
    let mut sufficient: Vec<(BitSet, Vec<usize>)> = Vec::new();
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
        let mut mask = BitSet::with_len(candidates.len());
        for &i in members {
            mask.insert(DenseNodeId::from_index(i));
        }
        if let Some((_, by)) = sufficient.iter().find(|(prev, _)| prev.is_subset_of(&mask)) {
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
        let outcome = match decide_preview(graph, &route, &preview, &mut search, ctx, decision_cap)
        {
            Ok(Preview::OverCap { operations }) => {
                StudySubsetOutcome::OverDecisionCap { operations }
            }
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
                        sufficient.push((mask, members.clone()));
                        retained = retained.saturating_add(search.peak_bytes());
                        StudySubsetOutcome::Sufficient { proposal: proposals.len() - 1 }
                    }
                    Identified::CitesNoProposed => StudySubsetOutcome::Refused {
                        message: "the identified formula cites no proposed regime".into(),
                    },
                }
            }
            Ok(Preview::NotIdentified { code, detail, .. }) => {
                StudySubsetOutcome::Insufficient { code, detail }
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
            #[allow(clippy::cast_precision_loss, reason = "At most 696 subsets")]
            progress.report((index + 1) as f64 / total as f64, "study planning");
        }
    }
    // Subsets larger than `max_size` were not enumerated: say so as a budget stop, and
    // withhold minimality unless even the cheapest unexamined subset cannot undercut the top.
    let unexamined_floor = (max_size < candidates.len()).then(|| {
        let mut costs = candidates.iter().map(|c| c.cost_units).collect::<Vec<_>>();
        costs.sort_unstable();
        costs.iter().take(max_size + 1).fold(0u64, |acc, c| acc.saturating_add(*c))
    });
    if unexamined_floor.is_some() && stop.is_none() {
        stop = Some(StudyPlanStop::Budget(search.receipt(
            SearchStop::Operations,
            explored.clone(),
            vec![format!("subsets:size>{max_size}")],
        )));
    }
    let minimal = certified_minimal(&records, proposals.first(), unexamined_floor);
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
        Err(IdentificationError::invalid_derivation(
            antecedent_core::reason_details::mz_transport::INVALID_DERIVATION,
        ))
    }
}

/// The minimality claim of the top proposal: every feasible subset strictly
/// cheaper than it was decided to a conclusive outcome (no stop, no decision
/// over its cap). Infeasible subsets are outside the universe.
fn certified_minimal(
    records: &[StudySubsetRecord],
    top: Option<&StudyProposal>,
    unexamined_floor: Option<u64>,
) -> bool {
    top.is_some_and(|top| {
        unexamined_floor.is_none_or(|floor| floor >= top.cost_units)
            && records
                .iter()
                .filter(|r| r.cost_units < top.cost_units)
                .all(|r| r.outcome.conclusive())
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
    fn subset_totals_overflow_is_refused_before_ranking() {
        let cost = STUDY_PLAN_MAX_COST_UNITS;
        let costs: Vec<_> = (0..5).map(|i| candidate(&format!("c{i}"), cost, 0)).collect();
        let error = subsets(&costs, 5).unwrap_err();
        assert_eq!(error.detail, INVALID_CANDIDATE);
        assert!(error.message.contains("total cost"));
        let samples: Vec<_> = (0..5).map(|i| candidate(&format!("s{i}"), 1, cost)).collect();
        let error = subsets(&samples, 5).unwrap_err();
        assert_eq!(error.detail, INVALID_CANDIDATE);
        assert!(error.message.contains("sample budget"));
    }

    #[test]
    fn subsets_are_ranked_by_cost_then_samples_then_size_then_ids() {
        let candidates = vec![candidate("a", 2, 0), candidate("b", 1, 5), candidate("c", 1, 0)];
        let order = subsets(&candidates, usize::MAX)
            .unwrap()
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
        assert_eq!(subsets(&candidates, 3).unwrap().len(), 16 + 120 + 560);
        assert_eq!(subsets(&candidates, usize::MAX).unwrap().len(), (1 << 16) - 1);
    }

    #[test]
    fn the_subset_size_limit_follows_the_operation_budget() {
        // 16 + 120 + 560 = 696 subsets of size at most three; 4: 2516.
        assert_eq!(subset_size_limit(16, 696), 3);
        assert_eq!(subset_size_limit(16, 695), 2);
        assert_eq!(subset_size_limit(16, 200_000), 16);
        assert_eq!(subset_size_limit(16, 0), 1);
        assert_eq!(subset_size_limit(500, 200_000), 2);
        assert_eq!(subset_size_limit(0, 10), 0);
    }

    #[test]
    fn a_size_truncation_withholds_minimality_unless_the_floor_clears_the_top() {
        let decided = [record(1, StudySubsetOutcome::Insufficient { code: "c", detail: "d" })];
        assert!(!certified_minimal(&decided, Some(&top(2)), Some(1)));
        assert!(certified_minimal(&decided, Some(&top(2)), Some(2)));
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
        assert!(certified_minimal(&decided, Some(&top(2)), None));
        assert!(!certified_minimal(&decided, None, None));
        // An inconclusive decision below the top withdraws the claim ...
        let mut over = decided.clone();
        over[0].outcome = StudySubsetOutcome::OverDecisionCap { operations: 4097 };
        assert!(!certified_minimal(&over, Some(&top(2)), None));
        // ... as does an unevaluated cheaper subset; an equal-cost one does not.
        let mut left = decided.clone();
        left[0].outcome = StudySubsetOutcome::Unevaluated;
        assert!(!certified_minimal(&left, Some(&top(2)), None));
        let mut tie = decided.to_vec();
        tie.push(record(2, StudySubsetOutcome::Unevaluated));
        assert!(certified_minimal(&tie, Some(&top(2)), None));
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

        use crate::test_support::v;

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
        assert_eq!(
            zero.status(),
            Some((
                "transport_not_certified",
                antecedent_core::reason_details::study_plan::NONE_CERTIFIED
            ))
        );
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
    fn a_decision_is_cut_off_at_its_cap_instead_of_running_on_the_plan_budget() {
        let public = fig1_plan(PUBLIC);
        let capped = fig1_plan(PlanCaps { decision_cap: |_, _| 1, ..PUBLIC });
        assert!(capped.proposals.is_empty());
        assert!(capped.stop.is_none(), "an over-cap decision is not a plan budget stop");
        let cut = capped
            .subsets
            .iter()
            .filter_map(|s| match s.outcome {
                StudySubsetOutcome::OverDecisionCap { operations } => Some(operations),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!cut.is_empty());
        assert!(cut.iter().all(|&operations| operations == 2), "{cut:?}");
        assert!(
            capped.operations_consumed < public.operations_consumed,
            "{} < {}",
            capped.operations_consumed,
            public.operations_consumed
        );
    }

    #[test]
    fn the_candidate_universe_is_bounded_by_the_operation_budget() {
        let route = fig1::route();
        let limits = |operations| StudyPlanLimits {
            search: SearchLimits { operations, depth: 24 },
            ..StudyPlanLimits::default()
        };
        let counts = |n: usize| vec![1; n];
        assert!(limits(1).check_bounds(&route, &counts(STUDY_PLAN_MAX_CANDIDATES)).is_ok());
        assert!(limits(1).check_bounds(&route, &counts(STUDY_PLAN_MAX_CANDIDATES + 1)).is_err());
        assert!(limits(100).check_bounds(&route, &counts(100)).is_ok());
        let refused = limits(100).check_bounds(&route, &counts(101)).unwrap_err();
        assert_eq!(refused.detail, BOUNDS_EXCEEDED);
        assert_eq!(limits(100).max_candidates(), 100);
        assert_eq!(limits(1).max_candidates(), STUDY_PLAN_MAX_CANDIDATES);
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
