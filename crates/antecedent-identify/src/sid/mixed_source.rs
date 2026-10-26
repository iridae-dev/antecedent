//! Bounded mixed-source proof search: identification from complementary
//! distributions of several studies of one population.
//!
//! The theorem-scoped routes run first, in a fixed order, under the same
//! [`SearchBudget`] as everything after them: target-first sID over the
//! catalog's target observational evidence, then the declared z / mz source
//! routes, then classical meta-transportability (mu-sID, Bareinboim and Pearl
//! 2013, Figure 5). Meta is tried only inside its own contract: two or more
//! declared sources that can each experiment on every observed variable (the
//! unrestricted source-experiment family that theorem assumes; a source with a
//! restricted controllable set is the mz route's, one source the z route's).
//! Outside that contract the stage is absent from the plan and never
//! approximated. Its inputs are the declared sources' populations and selection
//! targets and the catalog's source regimes, not the target-population regimes
//! the generic search reads. Only when none of them identifies the query does
//! the generic search
//! run: forward chaining over the supplied distributions with a frozen rule set
//! (`mixed_source_proof_rules.v1`): the probability rules (marginalize, condition, product) and
//! Pearl's three do-calculus rules on the graph's mutilated ADMGs. It is sound
//! and incomplete: a formula is returned only after an independent checker has
//! replayed every step; running out of rules or budget is `NotCertified` or
//! `Exhausted`, never a non-identification claim.
//!
//! # How incomplete
//!
//! Measured against the complete Shpitser-Pearl ID algorithm on a single
//! observational study of the whole joint (the generic rule search run alone, the
//! named routes bypassed), over every query with any non-empty disjoint outcome and
//! treatment sets (module `tests::reference`; the figures are reproduced by its
//! ignored `measure` test). Soundness held on all 230,468 queries measured: the
//! search never identified a query ID refutes.
//!
//! | ADMG nodes | queries | ID-identifiable | search identifies |
//! |---|---|---|---|
//! | 3, every ADMG | 768 | 612 | 612 (100%) |
//! | 4, every ADMG | 204,800 | 142,827 | 142,826 (99.9993%) |
//! | 5, 8,000 random ADMGs x 3 queries | 24,000 | 16,676 | 16,238 (97.4%) |
//! | 6, 300 random ADMGs x 3 queries (outside the claim) | 900 | 534 | 190 (35.6%) |
//!
//! The known gap classes, one minimal pinned example each
//! (`known_incomplete_queries_stay_not_certified_until_deliberately_fixed`):
//!
//! 1. **The napkin family** (ID's ratio of a marginalised C-factor). The four-node
//!    napkin `W -> Z -> X -> Y`, `W <-> X`, `W <-> Y` with `do(X)` on `Y` is the
//!    only four-node miss; 108 of the 109 four-node queries whose ID formula is a
//!    ratio are found, so the frozen rules do reach ratio forms, just not this
//!    one. Five-node napkin embeddings account for 1 of 16,239 five-node queries
//!    with at most three treatments (99.994%). `NotCertified`.
//! 2. **The intervention and move bounds** [`MIXED_SOURCE_MAX_DO`] = 3 actions and
//!    [`MIXED_SOURCE_MAX_MOVE`] = 2 variables per rule application are the *initial*
//!    bounds of the search, not limits of it. A first pass that reaches its fixpoint
//!    having skipped an application only because of a bound is repeated with both
//!    bounds raised by one, up to the number of observed variables, on the same
//!    shared budget; the search ends `NotCertified` only at a fixpoint that no bound
//!    cut short, and `Exhausted` (a receipt naming the deeper bounds left
//!    unevaluated) when the budget stops it first. The measured misses below
//!    (437 of the 438 five-node misses had four treatments) are those of the first
//!    pass alone.
//! 3. **The operation budget**: at six nodes the default 20,000 operations run out
//!    before the closure does in about two thirds of the ID-identifiable queries
//!    (344 of 534 in the sample), even for trivial effects. `Exhausted`, a receipt.
//!    Up to five nodes no measured query exhausted the budget.
//!
//! So "sound, incomplete" here means: up to five nodes and three treatments the
//! search is complete on the single-study reference except for the napkin family;
//! it is not a decision procedure, and a `NotCertified` or `Exhausted` result on a
//! single fully observed study is not evidence of non-identification (the named
//! sID route, which runs first, decides those exactly). No complete oracle exists
//! for the several-study case in this repository (the generalized identification
//! algorithm of Lee, Correa and Bareinboim is not implemented), so there is no
//! multi-study completeness figure; the multi-study claim stays soundness against
//! enumerated structural models only.
//!
//! The two deletion rules (rule 1 and rule 3 deletion) belong to the frozen set
//! and the independent checker accepts a proof that uses one, but the search
//! does not attempt them: within this rule set they never contribute to
//! identification. A deletion may not drop a variable the expression still
//! mentions, so it can only undo an insertion, landing on a quantity the search
//! already holds; over 1,479 random closures (156,339 quantities) none derived
//! a new quantity, and every attempt only spent budget. The unit test
//! `deletions_are_checked_but_add_nothing_to_the_closure` keeps that evidence
//! honest.
//!
//! A quantity `Q(Y | do(D), Z)` is a checked expression whose free variables
//! are `Y`, `Z` and the (symbolic) levels of `D`. A rule that would delete a
//! variable the expression still mentions is not applied, so a proof never
//! leaves a hidden free variable and the root binds only outcomes and request
//! levels. Inputs are the joint laws of the target population's available,
//! measured, whole-population regimes; each proof leaf names the study and
//! regime that supplies it. A distribution that exists only as separate
//! marginals is never a leaf: a second, relaxed pass reports the exact leaf that
//! would need the joint.
//!
//! The A6.1 catalog descriptor ([`antecedent_core::CatalogDistribution`]) is the
//! one place the input classification reads a regime's origin, selection, kind,
//! measured set, interventions and levels, availability, identity, study and
//! snapshot. The pairwise `shared_data` relation is not consulted here: this
//! route evaluates exact laws and claims a point, so shared units matter only to
//! the mz statistical path, which is the only consumer of `shared_data`.
//!
//! Every attempted rule application charges one operation of the shared budget
//! at its derivation generation, with the cumulative live-state bytes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, ExecutionContext, LawOrigin, RegimeId, RegimeKind,
    SamplingSelection, SearchLimits, VariableId,
};
use antecedent_expr::{
    CausalExprArena, DistributionLeaf, DomainRef, ExprId, ExprNode, LeafBinding,
};
use antecedent_graph::Admg;

use super::meta::{MetaSource, MetaTransportQuery, identify_meta_catalog_metered};
use super::z_transport::{
    ZTransportDecision, ZTransportLimitsReceipt, ZTransportQuery, ZTransportSourceSpec,
    decide_z_transport_reporting,
};
use super::{IdentificationError, MzTransportQuery};

/// Observed variables in the graph under the earlier fixed ceiling. Not enforced: the
/// search is bounded by its budget (the mask representation holds 63 variables).
pub const MIXED_SOURCE_MAX_OBSERVED: usize = 10;
/// Usable input distributions in one query under the earlier fixed ceiling. Not enforced:
/// every input is a step charged to the search budget.
pub const MIXED_SOURCE_MAX_DISTRIBUTIONS: usize = 16;
/// Search limits of one whole decision: rule applications charged across every
/// stage, and derivation generations. They are the defaults and also the
/// maxima; larger limits refuse. Memory and cancellation come from the context.
pub const MIXED_SOURCE_DEFAULT_LIMITS: SearchLimits =
    SearchLimits { operations: 20_000, depth: 16 };
/// Version of the frozen rule set. It is bound into every proof record.
pub const MIXED_SOURCE_RULE_SET: &str = "mixed_source_proof_rules.v1";
/// Variables one rule application moves (marginalize, condition, insert, delete or
/// exchange) in the search's first pass; deeper passes raise it on demand.
pub const MIXED_SOURCE_MAX_MOVE: usize = 2;
/// Hard-intervention set size of a derived quantity in the search's first pass; deeper
/// passes raise it on demand.
pub const MIXED_SOURCE_MAX_DO: usize = 3;
/// Alternative last steps recorded for the target, when found.
const MAX_ALTERNATIVES: usize = 4;
/// Frontier entries an inspection keeps.
const FRONTIER_SHOWN: usize = 24;

/// A request outside a declared bound: observed variables, distributions,
/// sources, controllables, or search limits.
const BOUNDS_EXCEEDED: &str = "mixed_search.bounds_exceeded";
/// A malformed query, graph or catalog.
const INVALID_QUERY: &str = "mixed_search.invalid_query";
/// A posterior or other model artifact is offered as an experimental law.
const POSTERIOR_AS_LAW: &str = "mixed_search.posterior_as_law";
/// A theorem-scoped route identifies the query: use that route.
const NAMED_ROUTE: &str = "mixed_search.named_route";
/// The frozen rule set reached its fixpoint without the target.
const NOT_CERTIFIED: &str = "mixed_search.not_certified";
/// The derivation needs a joint that exists only as separate marginals.
const MISSING_JOINT: &str = "mixed_search.missing_joint";
/// A limit or cancellation stopped the search.
const BUDGET: &str = "mixed_search.budget";
/// A derivation or its record does not check.
const INVALID_DERIVATION: &str = "mixed_search.invalid_derivation";

const fn bounds_exceeded() -> IdentificationError {
    IdentificationError::UnsupportedInput { code: BOUNDS_EXCEEDED }
}

fn invalid_query(detail: &dyn std::fmt::Display) -> IdentificationError {
    IdentificationError::invalid_input(format!("{INVALID_QUERY}: {detail}"))
}

const fn invalid_derivation() -> IdentificationError {
    IdentificationError::invalid_derivation(INVALID_DERIVATION)
}

// ---------------------------------------------------------------- rule set

/// One rule of the frozen rule set `mixed_source_proof_rules.v1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MixedRule {
    /// A supplied joint law of one catalog regime.
    Input,
    /// Sum variables out of the jointly distributed set.
    Marginalize,
    /// Move jointly distributed variables behind the conditioning bar.
    Condition,
    /// `P(A | B, Z) P(B | Z) = P(A, B | Z)`.
    Product,
    /// Do-calculus rule 1: insert an observation.
    Rule1Insert,
    /// Do-calculus rule 1: delete an observation (checker-accepted; the search
    /// does not attempt it because it cannot add a quantity here).
    Rule1Delete,
    /// Do-calculus rule 2: exchange an observation for an action.
    Rule2ToDo,
    /// Do-calculus rule 2: exchange an action for an observation.
    Rule2ToObservation,
    /// Do-calculus rule 3: insert an action.
    Rule3Insert,
    /// Do-calculus rule 3: delete an action (checker-accepted; the search does
    /// not attempt it because it cannot add a quantity here).
    Rule3Delete,
}

impl MixedRule {
    /// Every rule of the frozen set, in application order.
    pub const ALL: [Self; 10] = [
        Self::Input,
        Self::Marginalize,
        Self::Condition,
        Self::Rule1Insert,
        Self::Rule1Delete,
        Self::Rule2ToDo,
        Self::Rule2ToObservation,
        Self::Rule3Insert,
        Self::Rule3Delete,
        Self::Product,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Marginalize => "marginalize",
            Self::Condition => "condition",
            Self::Product => "product",
            Self::Rule1Insert => "rule1_insert",
            Self::Rule1Delete => "rule1_delete",
            Self::Rule2ToDo => "rule2_to_do",
            Self::Rule2ToObservation => "rule2_to_observation",
            Self::Rule3Insert => "rule3_insert",
            Self::Rule3Delete => "rule3_delete",
        }
    }
}

/// Names of the frozen rule set, in application order.
#[must_use]
#[doc(hidden)]
pub fn mixed_source_rule_names() -> Vec<&'static str> {
    MixedRule::ALL.iter().map(|rule| rule.as_str()).collect()
}

// -------------------------------------------------------------------- masks

type Mask = u64;

const fn bit(index: usize) -> Mask {
    assert!(index < Mask::BITS as usize, "mixed-source mask index exceeds the mask width");
    1 << index
}

fn members(mask: Mask) -> impl Iterator<Item = usize> {
    (0..Mask::BITS as usize).filter(move |i| (mask >> i) & 1 == 1)
}

fn size(mask: Mask) -> usize {
    mask.count_ones() as usize
}

/// Non-empty submasks of `pool` with at most `max` members, smallest first.
fn subsets(pool: Mask, max: usize) -> Vec<Mask> {
    subsets_within(pool, max, usize::MAX, |_| true)
}

/// The first `cap` non-empty submasks of `pool` with at most `max` members that satisfy
/// `keep`, in the order of [`subsets`] (size, then mask value). They are produced one size
/// at a time (Gosper's hack over the pool's positions, which preserves the numeric order of
/// the masks), so a caller that can only pay for `cap` of them never builds the rest.
fn subsets_within(
    pool: Mask,
    max: usize,
    cap: usize,
    keep: impl Fn(Mask) -> bool,
) -> Vec<Mask> {
    let positions = members(pool).collect::<Vec<_>>();
    let width = positions.len();
    let mut out = Vec::new();
    for count in 1..=max.min(width) {
        let limit = 1u128 << width;
        let mut combination = (1u128 << count) - 1;
        while combination < limit {
            let mut mask: Mask = 0;
            let mut rest = combination;
            while rest != 0 {
                mask |= bit(positions[rest.trailing_zeros() as usize]);
                rest &= rest - 1;
            }
            if keep(mask) {
                if out.len() >= cap {
                    return out;
                }
                out.push(mask);
            }
            let low = combination & combination.wrapping_neg();
            let ripple = combination + low;
            combination = ripple | (((combination ^ ripple) >> 2) / low);
        }
    }
    out
}

// -------------------------------------------------------------------- query

/// One interventional query answered from the target population's studies.
///
/// Declared `sources` are optional: they only feed the theorem-scoped z and mz
/// routes that run first. The generic search uses the target population's
/// regimes.
#[derive(Clone, Debug, PartialEq)]
pub struct MixedSourceQuery {
    /// Joint outcomes.
    pub outcomes: Arc<[VariableId]>,
    /// Treatments whose interventional response is queried.
    pub treatments: Arc<[VariableId]>,
    /// Population whose interventional law is queried; its studies are the inputs.
    pub target: Arc<str>,
    /// Zero or more source populations for the theorem-scoped routes.
    pub sources: Arc<[ZTransportSourceSpec]>,
}

impl MixedSourceQuery {
    /// Canonical form: sources ordered by population and every variable set sorted.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let mz = self.mz_query().canonical();
        Self {
            outcomes: mz.outcomes,
            treatments: mz.treatments,
            target: mz.target,
            sources: mz.sources,
        }
    }

    fn mz_query(&self) -> MzTransportQuery {
        MzTransportQuery {
            outcomes: Arc::clone(&self.outcomes),
            treatments: Arc::clone(&self.treatments),
            target: Arc::clone(&self.target),
            sources: Arc::clone(&self.sources),
        }
    }
}

/// Why a catalog regime is not a search input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedExclusion {
    /// The excluded regime.
    pub regime: RegimeId,
    /// `other_population`, `not_available`, `recovered_law`, `model_artifact`,
    /// `selected_sample`, `conditioned_projection`, `separate_marginals` or
    /// `restricted_levels`.
    pub reason: &'static str,
}

/// One distribution the search may cite: a joint law of a target-population
/// regime over its measured variables, with its intervention set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedInput {
    /// The supplying catalog regime.
    pub regime: RegimeId,
    /// Study identity (the population when none was declared).
    pub study: Arc<str>,
    /// Jointly measured variables of the law (the regime's, without its interventions).
    pub measured: Vec<VariableId>,
    /// Hard-intervention set of the regime.
    pub intervened: Vec<VariableId>,
    /// Canonical identity of the catalog distribution.
    pub identity: String,
    /// Bound snapshot, if any.
    pub snapshot: Option<Arc<str>>,
    /// Variables of a separate-marginal regime; empty for a joint law.
    pub marginals: Vec<VariableId>,
}

/// A query that passed the contract.
#[derive(Clone, Debug)]
pub struct ValidatedMixedSourceQuery {
    /// Canonical query.
    pub query: MixedSourceQuery,
    /// Inputs of the strict search, in regime-id order.
    pub inputs: Vec<MixedInput>,
    /// Separate-marginal regimes, used only by the relaxed missing-joint pass.
    pub marginal_inputs: Vec<MixedInput>,
    /// Every regime that is not an input, with its reason.
    pub exclusions: Vec<MixedExclusion>,
    /// Whether the classical meta-transport route applies: two or more declared
    /// sources, each able to experiment on every observed variable.
    pub meta_scope: bool,
}

impl ValidatedMixedSourceQuery {
    /// The stages of the decision, in order.
    fn stage_plan(&self) -> Vec<&'static str> {
        stage_names(&self.query, self.meta_scope)
    }
}

fn dense_variables(graph: &Admg) -> Result<Vec<VariableId>, IdentificationError> {
    super::static_nodes::static_variables(graph)
        .map_err(|()| invalid_query(&"the graph must consist of static variables"))
}

/// Validate a query against the graph and catalog and classify every regime.
///
/// There is no count bound on observed variables, usable distributions, sources or
/// controllables: the search charges its budget for everything it builds. A regime is
/// an input when it is an available,
/// measured, whole-population, unconditioned joint law of the target population
/// whose interventions, if any, cover every level (a regime restricted to named
/// levels is excluded as `restricted_levels`); every other regime is excluded
/// with its reason.
///
/// # Errors
///
/// `mixed_search.bounds_exceeded`; `mixed_search.invalid_query` for a malformed
/// query, graph or catalog; `mixed_search.posterior_as_law` when a model
/// artifact is offered as an experimental law.
#[allow(
    clippy::too_many_lines,
    reason = "One linear classification of every regime, in exclusion order"
)]
pub fn validate_mixed_source_query(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
) -> Result<ValidatedMixedSourceQuery, IdentificationError> {
    let query = query.canonical();
    let variables = dense_variables(graph)?;
    let known = |v: &VariableId| variables.contains(v);
    let distinct = |vs: &[VariableId]| vs.iter().collect::<BTreeSet<_>>().len() == vs.len();
    if query.outcomes.is_empty()
        || query.treatments.is_empty()
        || !distinct(&query.outcomes)
        || !distinct(&query.treatments)
        || !query.outcomes.iter().chain(query.treatments.iter()).all(known)
        || query.outcomes.iter().any(|v| query.treatments.contains(v))
        || query.target.trim().is_empty()
    {
        return Err(invalid_query(&"outcomes and treatments must be disjoint graph variables"));
    }
    let populations = query.sources.iter().map(|s| s.population.as_ref()).collect::<BTreeSet<_>>();
    if populations.len() != query.sources.len() || populations.contains(query.target.as_ref()) {
        return Err(invalid_query(&"source populations must be distinct and not the target"));
    }
    let mut inputs = Vec::new();
    let mut marginal_inputs = Vec::new();
    let mut exclusions = Vec::new();
    // The catalog's descriptor is the one owner of every regime's semantics (origin,
    // selection, kind, measured set, interventions and levels, availability).
    for distribution in catalog.distributions() {
        let id = distribution.source_regime;
        let exclude = |reason| MixedExclusion { regime: id, reason };
        if distribution.population != query.target {
            exclusions.push(exclude("other_population"));
            continue;
        }
        if !distribution.evidence_kind.can_satisfy_factor() {
            exclusions.push(exclude("not_available"));
            continue;
        }
        // A recovered law is a derived law, neither a
        // model artifact nor a posterior; it is excluded under its own reason in
        // either kind (fail-closed, no new refusal detail).
        if matches!(distribution.origin, LawOrigin::Recovered { .. }) {
            exclusions.push(exclude("recovered_law"));
            continue;
        }
        if distribution.origin != LawOrigin::Measured {
            if distribution.kind == RegimeKind::Experimental {
                return Err(IdentificationError::UnsupportedInput { code: POSTERIOR_AS_LAW });
            }
            exclusions.push(exclude("model_artifact"));
            continue;
        }
        if distribution.selection != SamplingSelection::Population {
            exclusions.push(exclude("selected_sample"));
            continue;
        }
        if !distribution.measured.iter().chain(distribution.interventions.iter()).all(known) {
            return Err(invalid_query(&format!(
                "regime {} names a variable outside the graph",
                id.raw()
            )));
        }
        if !distribution.conditioned_on.is_empty() {
            exclusions.push(exclude("conditioned_projection"));
            continue;
        }
        // A value-restricted regime supplies only the levels it names; the symbolic
        // search states its identification for every level of the treatments, so
        // it can never cite one (`EvidenceRegime::satisfies` refuses the same for
        // the named routes).
        if !distribution.intervention_values.is_empty() {
            exclusions.push(exclude("restricted_levels"));
            continue;
        }
        let mut measured = distribution
            .measured
            .iter()
            .filter(|v| !distribution.interventions.contains(v))
            .copied()
            .collect::<Vec<_>>();
        measured.sort_unstable();
        if measured.is_empty() {
            exclusions.push(exclude("not_available"));
            continue;
        }
        let mut intervened = distribution.interventions.to_vec();
        intervened.sort_unstable();
        let mut input = MixedInput {
            regime: id,
            study: Arc::from(distribution.study_or_population()),
            measured,
            intervened,
            identity: distribution.canonical_identity(),
            snapshot: distribution.snapshot.clone(),
            marginals: Vec::new(),
        };
        if let DistributionAvailability::SeparateMarginals { variables } =
            &distribution.availability
        {
            input.marginals = variables.to_vec();
            input.marginals.sort_unstable();
            exclusions.push(exclude("separate_marginals"));
            marginal_inputs.push(input);
        } else {
            inputs.push(input);
        }
    }
    let meta_scope = meta_applies(&query, &variables);
    Ok(ValidatedMixedSourceQuery { query, inputs, marginal_inputs, exclusions, meta_scope })
}

// -------------------------------------------------------------- proof steps

/// A source distribution a proof leaf cites.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedSourceLeaf {
    /// The supplying regime.
    pub regime: RegimeId,
    /// Study identity of the supplying distribution.
    pub study: Arc<str>,
    /// Population of the supplying distribution.
    pub population: Arc<str>,
    /// Bound snapshot, if any.
    pub snapshot: Option<Arc<str>>,
    /// Canonical identity of the supplying catalog distribution.
    pub identity: String,
}

/// One checked step of a mixed-source proof: the quantity `P(y | do(intervened),
/// conditioned)`, the rule that produced it and the steps it used.
#[derive(Clone, Debug, PartialEq)]
pub struct MixedStep {
    /// Rule applied.
    pub rule: MixedRule,
    /// Earlier steps used (indices into the proof).
    pub premises: Vec<usize>,
    /// The variables the rule moved (dropped, conditioned, inserted, deleted, exchanged).
    pub params: Vec<VariableId>,
    /// Jointly distributed variables.
    pub y: Vec<VariableId>,
    /// Hard-intervention variables (levels are the request's or summed out).
    pub intervened: Vec<VariableId>,
    /// Conditioning variables.
    pub conditioned: Vec<VariableId>,
    /// The supplying regime, for an input step.
    pub source: Option<MixedSourceLeaf>,
    /// Whether an input step stands for a joint the catalog holds only as
    /// separate marginals. Never present in a checked derivation.
    pub relaxed: bool,
    /// The step's expression in the derivation's arena.
    pub expression: ExprId,
}

/// One quantity of the search, in variable coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedQuantity {
    /// Jointly distributed variables.
    pub y: Vec<VariableId>,
    /// Hard-intervention variables.
    pub intervened: Vec<VariableId>,
    /// Conditioning variables.
    pub conditioned: Vec<VariableId>,
}

/// What the search charged before it identified the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixedSearchSummary {
    /// Rule applications charged by the rule search up to the proof's last step.
    pub operations_to_proof: usize,
    /// Derivation generation at which the target first appeared (0 for an input).
    pub generation: usize,
}

/// One stage of the decision and how it ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedStageRecord {
    /// `target_first_sid`, `z_transport`, `mz_transport`, `meta_transport` or
    /// `rule_search`.
    pub stage: String,
    /// `identified`, `missing_evidence`, `not_certified`, `obstruction` or `exhausted`.
    pub outcome: &'static str,
}

/// A checked mixed-source derivation.
#[derive(Clone, Debug)]
pub struct MixedSourceDerivation {
    query: MixedSourceQuery,
    graph_signature: String,
    steps: Vec<MixedStep>,
    arena: CausalExprArena,
    root_step: usize,
    stages: Vec<MixedStageRecord>,
    summary: MixedSearchSummary,
    /// The spelling of the rule set the proof was sealed under.
    rule_set: &'static str,
}

/// Portable premises of a derivation; the arena travels separately.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixedStepRecord {
    /// Rule name.
    pub rule: String,
    /// Premise step indices.
    pub premises: Vec<usize>,
    /// Moved variables.
    pub params: Vec<u32>,
    /// Jointly distributed variables.
    pub y: Vec<u32>,
    /// Hard-intervention variables.
    pub intervened: Vec<u32>,
    /// Conditioning variables.
    pub conditioned: Vec<u32>,
    /// Supplying regime of an input step.
    pub regime: Option<u32>,
    /// Study of the supplying distribution.
    pub study: Option<String>,
    /// Population of the supplying distribution.
    pub population: Option<String>,
    /// Snapshot of the supplying distribution.
    pub snapshot: Option<String>,
    /// Canonical identity of the supplying distribution.
    pub source: Option<String>,
}

/// Portable premises of a mixed-source derivation: the frozen rule-set version,
/// every step with its premises and source distributions, and what the search
/// charged. The expression arena travels separately;
/// [`MixedSourceDerivation::from_record_checked`] re-decides the query against
/// its catalog and accepts only an identical derivation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MixedSourceDerivationRecord {
    /// Shared causal-graph identity.
    pub graph_signature: String,
    /// Frozen rule-set version the proof was searched and checked under.
    pub rule_set: String,
    /// Every step of the proof, in order.
    pub steps: Vec<MixedStepRecord>,
    /// The step that answers the query.
    pub root_step: usize,
    /// Rule applications the rule search charged up to the proof's last step.
    pub operations_to_proof: usize,
    /// Generation at which the target first appeared.
    pub generation: usize,
    /// `(stage, outcome)` of every stage evaluated before the identifying one.
    pub stages: Vec<(String, String)>,
}

fn raw(variables: &[VariableId]) -> Vec<u32> {
    variables.iter().map(|v| v.raw()).collect()
}

impl MixedSourceDerivation {
    /// Canonical query this derivation answers.
    #[must_use]
    pub const fn query(&self) -> &MixedSourceQuery {
        &self.query
    }
    /// Graph identity the derivation was checked against.
    #[must_use]
    pub fn graph_signature(&self) -> &str {
        &self.graph_signature
    }
    /// Every step, in proof order.
    #[must_use]
    pub fn steps(&self) -> &[MixedStep] {
        &self.steps
    }
    /// Expression arena of the derivation.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// The step that answers the query.
    #[must_use]
    pub const fn root_step(&self) -> usize {
        self.root_step
    }
    /// Formula root.
    #[must_use]
    pub fn root(&self) -> ExprId {
        self.steps[self.root_step].expression
    }
    /// Stages evaluated before the rule search, in order.
    #[must_use]
    pub fn stages(&self) -> &[MixedStageRecord] {
        &self.stages
    }
    /// What the rule search charged.
    #[must_use]
    pub const fn summary(&self) -> MixedSearchSummary {
        self.summary
    }
    /// Frozen rule-set version.
    #[must_use]
    pub const fn rule_set(&self) -> &'static str {
        self.rule_set
    }

    /// Every distribution the proof cites: `(step, source)` of each input step.
    #[must_use]
    pub fn leaves(&self) -> Vec<(usize, &MixedSourceLeaf)> {
        self.steps
            .iter()
            .enumerate()
            .filter_map(|(i, step)| step.source.as_ref().map(|leaf| (i, leaf)))
            .collect()
    }

    /// Catalog regimes the proof cites, sorted and without repeats.
    #[must_use]
    pub fn cited_regimes(&self) -> Vec<RegimeId> {
        let mut cited = self.leaves().into_iter().map(|(_, leaf)| leaf.regime).collect::<Vec<_>>();
        cited.sort_unstable();
        cited.dedup();
        cited
    }

    /// Export the proof premises for an independently checked artifact.
    #[must_use]
    pub fn to_record(&self) -> MixedSourceDerivationRecord {
        MixedSourceDerivationRecord {
            graph_signature: self.graph_signature.clone(),
            rule_set: self.rule_set.into(),
            steps: self.steps.iter().map(step_record).collect(),
            root_step: self.root_step,
            operations_to_proof: self.summary.operations_to_proof,
            generation: self.summary.generation,
            stages: self.stages.iter().map(|s| (s.stage.clone(), s.outcome.to_owned())).collect(),
        }
    }

    /// Reconstruct a derivation from an untrusted record: the bounded decision is
    /// re-run on `graph`, `query` and `catalog` under the caller's limits, and the
    /// record and arena must equal the derivation it produces.
    ///
    /// # Errors
    /// The decision does not run the generic search to an identification, or any
    /// recorded premise or expression differs.
    pub fn from_record_checked(
        graph: &Admg,
        query: &MixedSourceQuery,
        catalog: &EvidenceCatalog,
        record: &MixedSourceDerivationRecord,
        arena: &CausalExprArena,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let MixedSourceDecision::Identified { derivation, .. } =
            decide_mixed_source(graph, query, catalog, limits, ctx)?
        else {
            return Err(invalid_derivation());
        };
        if derivation.to_record() != *record || derivation.arena != *arena {
            return Err(invalid_derivation());
        }
        Ok(*derivation)
    }

    /// Render the proof as a compact graph, one line per step. `names` maps
    /// variable ids to names; missing entries render as `v<id>`.
    #[must_use]
    pub fn proof_graph(&self, names: &[String]) -> Vec<String> {
        render_steps(&self.steps, names, self.root_step)
    }
}

fn step_record(step: &MixedStep) -> MixedStepRecord {
    MixedStepRecord {
        rule: step.rule.as_str().into(),
        premises: step.premises.clone(),
        params: raw(&step.params),
        y: raw(&step.y),
        intervened: raw(&step.intervened),
        conditioned: raw(&step.conditioned),
        regime: step.source.as_ref().map(|s| s.regime.raw()),
        study: step.source.as_ref().map(|s| s.study.to_string()),
        population: step.source.as_ref().map(|s| s.population.to_string()),
        snapshot: step.source.as_ref().and_then(|s| s.snapshot.as_ref().map(ToString::to_string)),
        source: step.source.as_ref().map(|s| s.identity.clone()),
    }
}

fn name_of(names: &[String], variable: VariableId) -> String {
    names.get(variable.as_usize()).cloned().unwrap_or_else(|| format!("v{}", variable.raw()))
}

/// `P(y1, y2 | do(x), z)` in names.
#[must_use]
#[doc(hidden)]
pub fn render_quantity(quantity: &MixedQuantity, names: &[String]) -> String {
    let join = |vs: &[VariableId]| vs.iter().map(|v| name_of(names, *v)).collect::<Vec<_>>();
    let mut bar = Vec::new();
    if !quantity.intervened.is_empty() {
        bar.push(format!("do({})", join(&quantity.intervened).join(", ")));
    }
    bar.extend(join(&quantity.conditioned));
    if bar.is_empty() {
        format!("P({})", join(&quantity.y).join(", "))
    } else {
        format!("P({} | {})", join(&quantity.y).join(", "), bar.join(", "))
    }
}

fn render_steps(steps: &[MixedStep], names: &[String], root: usize) -> Vec<String> {
    steps
        .iter()
        .enumerate()
        .map(|(i, step)| {
            let quantity = render_quantity(
                &MixedQuantity {
                    y: step.y.clone(),
                    intervened: step.intervened.clone(),
                    conditioned: step.conditioned.clone(),
                },
                names,
            );
            let moved = step.params.iter().map(|v| name_of(names, *v)).collect::<Vec<_>>();
            let from = step.premises.iter().map(|p| format!("s{p}")).collect::<Vec<_>>();
            let how = match (&step.source, step.rule) {
                (Some(leaf), _) if step.relaxed => format!(
                    "input study \"{}\" regime {} MISSING: supplied only as separate marginals",
                    leaf.study,
                    leaf.regime.raw()
                ),
                (Some(leaf), _) => {
                    format!("input study \"{}\" regime {}", leaf.study, leaf.regime.raw())
                }
                (None, MixedRule::Product) => format!("product({})", from.join(", ")),
                (None, rule) => {
                    format!("{}[{}]({})", rule.as_str(), moved.join(", "), from.join(", "))
                }
            };
            let mark = if i == root { "  <- target" } else { "" };
            format!("s{i} = {quantity}  via {how}{mark}")
        })
        .collect()
}

// ------------------------------------------------------- expression building

/// The one owner of how a step's expression is built, shared by extraction and
/// the independent checker so an arena is a deterministic function of the steps.
fn build_expression(
    arena: &mut CausalExprArena,
    step: &MixedStep,
    premises: &[ExprId],
    target: &str,
) -> Result<ExprId, IdentificationError> {
    let expression = match step.rule {
        MixedRule::Input => {
            let leaf = step.source.as_ref().ok_or_else(invalid_derivation)?;
            let variables = arena.intern_var_set(step.y.iter().copied());
            let conditioned_on = arena.empty_var_set();
            let intervention = arena.intern_intervention_set(step.intervened.iter().copied());
            let domain = if step.intervened.is_empty() {
                DomainRef::Observational
            } else {
                DomainRef::Interventional
            };
            arena
                .intern_distribution_tagged(
                    DistributionLeaf { variables, conditioned_on, intervention, domain },
                    target,
                    Some(leaf.regime),
                    Some(domain),
                )
                .map_err(|_| invalid_derivation())?
        }
        MixedRule::Marginalize => {
            let variables = arena.intern_var_set(step.params.iter().copied());
            arena.intern(ExprNode::SumOut { variables, expr: premises[0] })
        }
        MixedRule::Condition => {
            let variables = arena.intern_var_set(step.y.iter().copied());
            let denominator = arena.intern(ExprNode::SumOut { variables, expr: premises[0] });
            arena.intern(ExprNode::Ratio { numerator: premises[0], denominator })
        }
        MixedRule::Product => {
            let list = arena.intern_list([premises[0], premises[1]]);
            arena.intern(ExprNode::Product(list))
        }
        _ => premises[0],
    };
    Ok(expression)
}

mod search;
pub use search::{
    BoundMixedSourceFunctional, MixedMissingEvidence, MixedMissingLeaf, MixedSearchInspection,
    MixedSourceDecision, bind_mixed_source_catalog, decide_mixed_source,
    verify_mixed_source_derivation,
};
pub(crate) use search::{Checker, Set, decide_mixed_source_shared};
use search::{meta_applies, stage_names};
