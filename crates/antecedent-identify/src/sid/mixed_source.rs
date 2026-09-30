//! Bounded mixed-source proof search (X9): identification from complementary
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
//! (`x9.rules.v1`): the probability rules (marginalize, condition, product) and
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
//! 2. **The intervention bound** [`MIXED_SOURCE_MAX_DO`] = 3: a derived quantity
//!    carries at most three actions, so a query with four or more treatments is
//!    unreachable even when its effect is trivially `P(y)`. Five nodes: 437 of the
//!    438 misses, every one of the 437 ID-identifiable queries with four
//!    treatments. `NotCertified`.
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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, ExecutionContext, LawOrigin, RegimeId, RegimeKind,
    SamplingSelection, SearchBudget, SearchLimits, SearchReceipt, SearchStop, VariableId,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode, LeafBinding};
use antecedent_graph::{Admg, DSeparationWorkspace, DenseNodeId, NodeRef, SelectionDiagram};

use super::meta::{MetaSource, MetaTransportQuery, identify_meta_catalog_metered};
use super::z_transport::{
    ZTransportDecision, ZTransportLimitsReceipt, ZTransportQuery, ZTransportSourceSpec,
    decide_z_transport_reporting,
};
use super::{
    CatalogTransportResult, ClassicalTransportQuery, IdentificationError, MzTransportDecision,
    MzTransportQuery, SearchCharge, SharedSearch, identify_catalog_transport_metered,
};

/// Observed variables in the graph.
pub const MIXED_SOURCE_MAX_OBSERVED: usize = 10;
/// Usable input distributions (regimes of the target population) in one query.
pub const MIXED_SOURCE_MAX_DISTRIBUTIONS: usize = 16;
/// Search limits of one whole decision: rule applications charged across every
/// stage, and derivation generations. They are the defaults and also the
/// maxima; larger limits refuse. Memory and cancellation come from the context.
pub const MIXED_SOURCE_DEFAULT_LIMITS: SearchLimits =
    SearchLimits { operations: 20_000, depth: 16 };
/// Version of the frozen rule set. It is bound into every proof record.
pub const MIXED_SOURCE_RULE_SET: &str = "x9.rules.v1";
/// Most variables one rule application moves (marginalize, condition, insert,
/// delete or exchange). A larger move is reached by repeated applications.
pub const MIXED_SOURCE_MAX_MOVE: usize = 2;
/// Largest hard-intervention set a derived quantity may carry.
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

/// One rule of the frozen rule set `x9.rules.v1`.
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

type Mask = u16;

const fn bit(index: usize) -> Mask {
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
    let mut out = Vec::new();
    let mut sub = pool;
    while sub != 0 {
        if size(sub) <= max {
            out.push(sub);
        }
        sub = (sub - 1) & pool;
    }
    out.sort_unstable_by_key(|s| (size(*s), *s));
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
    graph
        .nodes()
        .iter()
        .map(|node| match node {
            NodeRef::Static(variable) => Ok(*variable),
            _ => Err(invalid_query(&"the graph must consist of static variables")),
        })
        .collect()
}

/// Validate a query against the graph and catalog and classify every regime.
///
/// Bounds: at most [`MIXED_SOURCE_MAX_OBSERVED`] observed variables and
/// [`MIXED_SOURCE_MAX_DISTRIBUTIONS`] usable distributions, and for declared
/// sources the z / mz bounds. A regime is an input when it is an available,
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
#[allow(clippy::too_many_lines)] // One linear classification of every regime, in exclusion order.
pub fn validate_mixed_source_query(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
) -> Result<ValidatedMixedSourceQuery, IdentificationError> {
    let query = query.canonical();
    if graph.node_count() > MIXED_SOURCE_MAX_OBSERVED
        || query.sources.len() > super::MZ_TRANSPORT_MAX_SOURCES
        || query
            .sources
            .iter()
            .any(|s| s.controllable.len() > super::MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE)
    {
        return Err(bounds_exceeded());
    }
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
        // 2.2B X10 (B6 audit D3): a recovered law is a derived law, neither a
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
    if inputs.len() > MIXED_SOURCE_MAX_DISTRIBUTIONS {
        return Err(bounds_exceeded());
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
    /// `target_first_sid`, `z_transport`, `mz_transport` or `rule_search`.
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
        MIXED_SOURCE_RULE_SET
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
            rule_set: MIXED_SOURCE_RULE_SET.into(),
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
                    variables,
                    conditioned_on,
                    intervention,
                    domain,
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

// ------------------------------------------------------------------ engine

/// A step of the working search, in masks over dense graph indices.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Raw {
    rule: MixedRule,
    premises: Vec<usize>,
    params: Mask,
    y: Mask,
    d: Mask,
    z: Mask,
    /// Variables the step's expression mentions freely.
    free: Mask,
    /// Input steps: index into the engine's input list.
    input: Option<usize>,
    relaxed: bool,
}

type Key = (Mask, Mask, Mask);

impl Raw {
    const fn key(&self) -> Key {
        (self.y, self.d, self.z)
    }
}

enum RunEnd {
    Found,
    Fixpoint,
    Stopped(SearchStop),
}

struct Engine<'g> {
    graph: &'g Admg,
    n: usize,
    universe: Mask,
    variables: Vec<VariableId>,
    inputs: Vec<MixedInput>,
    target: Key,
    ws: DSeparationWorkspace,
    steps: Vec<Raw>,
    known: HashMap<Key, usize>,
    by_dz: HashMap<(Mask, Mask), Vec<usize>>,
    alternatives: Vec<Raw>,
    found: Option<usize>,
    operations: usize,
    operations_to_proof: usize,
    generation: usize,
    completed: usize,
    unexpanded: usize,
    rule_counts: BTreeMap<&'static str, usize>,
}

impl<'g> Engine<'g> {
    fn new(
        graph: &'g Admg,
        variables: Vec<VariableId>,
        inputs: Vec<MixedInput>,
        query: &MixedSourceQuery,
    ) -> Result<Self, IdentificationError> {
        let n = variables.len();
        let mask_of = |vs: &[VariableId]| {
            vs.iter().try_fold(0, |acc: Mask, v| {
                variables
                    .iter()
                    .position(|w| w == v)
                    .map(|i| acc | bit(i))
                    .ok_or_else(|| invalid_query(&"variable outside the graph"))
            })
        };
        let target = (mask_of(&query.outcomes)?, mask_of(&query.treatments)?, 0);
        let masks = inputs
            .iter()
            .map(|input| Ok((mask_of(&input.measured)?, mask_of(&input.intervened)?)))
            .collect::<Result<Vec<_>, IdentificationError>>()?;
        let mut engine = Self {
            graph,
            n,
            universe: (1 << n) - 1,
            variables,
            inputs: Vec::new(),
            target,
            ws: DSeparationWorkspace::default(),
            steps: Vec::new(),
            known: HashMap::new(),
            by_dz: HashMap::new(),
            alternatives: Vec::new(),
            found: None,
            operations: 0,
            operations_to_proof: 0,
            generation: 0,
            completed: 0,
            unexpanded: 0,
            rule_counts: BTreeMap::new(),
        };
        for (index, (input, (y, d))) in inputs.iter().zip(masks).enumerate() {
            engine.add(Raw {
                rule: MixedRule::Input,
                premises: Vec::new(),
                params: 0,
                y,
                d,
                z: 0,
                free: y | d,
                input: Some(index),
                relaxed: !input.marginals.is_empty(),
            });
        }
        engine.inputs = inputs;
        Ok(engine)
    }

    fn variables_of(&self, mask: Mask) -> Vec<VariableId> {
        let mut out = members(mask).map(|i| self.variables[i]).collect::<Vec<_>>();
        out.sort_unstable();
        out
    }

    fn live_bytes(&self) -> u64 {
        let per_step = std::mem::size_of::<Raw>() + 96;
        u64::try_from(self.steps.len() * per_step + self.alternatives.len() * per_step)
            .unwrap_or(u64::MAX)
    }

    fn charge(
        &mut self,
        search: &mut SharedSearch<'_>,
        generation: usize,
    ) -> Result<(), SearchStop> {
        search.charge(generation, self.live_bytes())?;
        self.operations += 1;
        Ok(())
    }

    /// Add a candidate; only the first derivation of a quantity is kept, and
    /// another last step of the target is remembered as an alternative.
    fn add(&mut self, raw: Raw) {
        let key = raw.key();
        if self.known.contains_key(&key) {
            let duplicate = |other: &Raw| {
                other.rule == raw.rule
                    && other.premises == raw.premises
                    && other.params == raw.params
                    && other.input == raw.input
            };
            if key == self.target
                && self.alternatives.len() < MAX_ALTERNATIVES
                && !self.found.is_some_and(|f| duplicate(&self.steps[f]))
                && !self.alternatives.iter().any(duplicate)
            {
                self.alternatives.push(raw);
            }
            return;
        }
        let index = self.steps.len();
        self.known.insert(key, index);
        self.by_dz.entry((raw.d, raw.z)).or_default().push(index);
        *self.rule_counts.entry(raw.rule.as_str()).or_default() += 1;
        if key == self.target && self.found.is_none() {
            self.found = Some(index);
            self.operations_to_proof = self.operations;
        }
        self.steps.push(raw);
    }

    /// `G` with every edge into `over` and every directed edge out of `under` removed
    /// (bidirected edges at an `over` node are edges into it).
    fn mutilated(&self, over: Mask, under: Mask) -> Admg {
        let mut graph = Admg::with_variables(u32::try_from(self.n).unwrap_or(0));
        let dense = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap_or(0));
        for from in 0..self.n {
            for to in self.graph.children(dense(from)) {
                let to = to.as_usize();
                if over & bit(to) == 0 && under & bit(from) == 0 {
                    let _ = graph.insert_directed(dense(from), dense(to));
                }
            }
            for to in self.graph.bidirected_neighbors(dense(from)) {
                let to = to.as_usize();
                if from < to && over & (bit(from) | bit(to)) == 0 {
                    let _ = graph.insert_bidirected(dense(from), dense(to));
                }
            }
        }
        graph
    }

    /// Ancestors of `seeds` (inclusive) within `G` without the edges into `over`.
    fn ancestors_without_into(&self, seeds: Mask, over: Mask) -> Mask {
        let dense = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap_or(0));
        let mut ancestors = seeds;
        loop {
            let mut next = ancestors;
            for node in members(ancestors) {
                if over & bit(node) == 0 {
                    for parent in self.graph.parents(dense(node)) {
                        next |= bit(parent.as_usize());
                    }
                }
            }
            if next == ancestors {
                return ancestors;
            }
            ancestors = next;
        }
    }

    /// `a` is m-separated from `b` given `cond` in `G` without the edges into
    /// `over` and out of `under`.
    fn separated(&mut self, over: Mask, under: Mask, a: Mask, b: Mask, cond: Mask) -> bool {
        let graph = self.mutilated(over, under);
        let dense = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap_or(0));
        let conditions = members(cond).map(dense).collect::<Vec<_>>();
        members(a).all(|x| {
            members(b).all(|y| {
                graph.is_m_separated(dense(x), dense(y), &conditions, &mut self.ws).unwrap_or(false)
            })
        })
    }

    /// Apply every searched rule to step `q`: marginalize, condition, rules 1 and
    /// 3 insertion, rule 2 both ways, and product. The two deletion rules are
    /// checker-only (see the module docs). Each attempted application charges
    /// the budget.
    #[allow(clippy::too_many_lines)] // One linear pass over the frozen rule set, in order.
    fn expand(
        &mut self,
        search: &mut SharedSearch<'_>,
        q: usize,
        generation: usize,
    ) -> Result<(), SearchStop> {
        self.charge(search, generation)?;
        let base = self.steps[q].clone();
        let (y, d, z, free) = (base.y, base.d, base.z, base.free);
        let unary = |rule, params, y, d, z, free| Raw {
            rule,
            premises: vec![q],
            params,
            y,
            d,
            z,
            free,
            input: None,
            relaxed: false,
        };
        if size(y) > 1 {
            for w in subsets(y, MIXED_SOURCE_MAX_MOVE).into_iter().filter(|w| *w != y) {
                self.charge(search, generation)?;
                self.add(unary(MixedRule::Marginalize, w, y & !w, d, z, free & !w));
            }
            for w in subsets(y, MIXED_SOURCE_MAX_MOVE).into_iter().filter(|w| *w != y) {
                self.charge(search, generation)?;
                self.add(unary(MixedRule::Condition, w, y & !w, d, z | w, free));
            }
        }
        let outside = self.universe & !(y | d | z);
        for w in subsets(outside, MIXED_SOURCE_MAX_MOVE) {
            self.charge(search, generation)?;
            if self.separated(d, 0, y, w, d | z) {
                self.add(unary(MixedRule::Rule1Insert, w, y, d, z | w, free));
            }
        }
        for w in subsets(z, MIXED_SOURCE_MAX_MOVE) {
            if size(d) + size(w) > MIXED_SOURCE_MAX_DO {
                continue;
            }
            self.charge(search, generation)?;
            if self.separated(d, w, y, w, d | (z & !w)) {
                self.add(unary(MixedRule::Rule2ToDo, w, y, d | w, z & !w, free));
            }
        }
        for w in subsets(d, MIXED_SOURCE_MAX_MOVE) {
            self.charge(search, generation)?;
            if self.separated(d & !w, w, y, w, (d & !w) | z) {
                self.add(unary(MixedRule::Rule2ToObservation, w, y, d & !w, z | w, free));
            }
        }
        for w in subsets(outside, MIXED_SOURCE_MAX_MOVE) {
            if size(d) + size(w) > MIXED_SOURCE_MAX_DO {
                continue;
            }
            self.charge(search, generation)?;
            let past = self.ancestors_without_into(z, d);
            if self.separated(d | (w & !past), 0, y, w, d | z) {
                self.add(unary(MixedRule::Rule3Insert, w, y, d | w, z, free));
            }
        }
        // Product: `q` as the conditional `P(A | B, Z')` of a known `P(B | Z')`.
        for b in subsets(z, size(z)) {
            if let Some(&j) = self.known.get(&(b, d, z & !b)) {
                self.charge(search, generation)?;
                let marginal = self.steps[j].clone();
                self.add(Raw {
                    rule: MixedRule::Product,
                    premises: vec![q, j],
                    params: 0,
                    y: y | b,
                    d,
                    z: z & !b,
                    free: free | marginal.free,
                    input: None,
                    relaxed: false,
                });
            }
        }
        // Product: `q` as the marginal `P(B | Z)` of a known conditional `P(A | B, Z)`.
        let partners = self.by_dz.get(&(d, z | y)).cloned().unwrap_or_default();
        for j in partners {
            self.charge(search, generation)?;
            let conditional = self.steps[j].clone();
            self.add(Raw {
                rule: MixedRule::Product,
                premises: vec![j, q],
                params: 0,
                y: conditional.y | y,
                d,
                z,
                free: conditional.free | free,
                input: None,
                relaxed: false,
            });
        }
        Ok(())
    }

    /// Chain generation by generation until the target is found, no new quantity
    /// appears, or the shared budget stops. A found target keeps its generation
    /// running so alternative last steps can be collected; a stop there is not
    /// a failure.
    fn run(&mut self, search: &mut SharedSearch<'_>) -> RunEnd {
        if self.found.is_some() {
            return RunEnd::Found;
        }
        let mut start = 0;
        let mut generation = 0;
        loop {
            let end = self.steps.len();
            if start == end {
                return RunEnd::Fixpoint;
            }
            generation += 1;
            self.generation = generation;
            for q in start..end {
                if let Err(stop) = self.expand(search, q, generation) {
                    self.unexpanded = end - q;
                    return if self.found.is_some() {
                        RunEnd::Found
                    } else {
                        RunEnd::Stopped(stop)
                    };
                }
            }
            self.completed = generation;
            if self.found.is_some() {
                return RunEnd::Found;
            }
            start = end;
        }
    }

    fn generation_of(&self, index: usize) -> usize {
        let mut memo = vec![0usize; index + 1];
        for i in 0..=index {
            memo[i] = self.steps[i].premises.iter().map(|p| memo[*p] + 1).max().unwrap_or(0);
        }
        memo[index]
    }

    fn quantity(&self, raw: &Raw) -> MixedQuantity {
        MixedQuantity {
            y: self.variables_of(raw.y),
            intervened: self.variables_of(raw.d),
            conditioned: self.variables_of(raw.z),
        }
    }

    /// The proof of step `last` (an existing step, or `replace` standing for it):
    /// its premise closure renumbered in order, as portable [`MixedStep`]s whose
    /// expressions are built in a fresh arena.
    fn extract(
        &self,
        last: &Raw,
        catalog: &EvidenceCatalog,
        target: &str,
    ) -> Result<(Vec<MixedStep>, CausalExprArena), IdentificationError> {
        let mut needed = BTreeSet::new();
        let mut pending = last.premises.clone();
        while let Some(index) = pending.pop() {
            if needed.insert(index) {
                pending.extend(self.steps[index].premises.iter().copied());
            }
        }
        let order = needed.into_iter().collect::<Vec<_>>();
        let position = |old: usize| order.iter().position(|o| *o == old);
        let mut arena = CausalExprArena::new();
        let mut steps: Vec<MixedStep> = Vec::with_capacity(order.len() + 1);
        for raw in order.iter().map(|i| &self.steps[*i]).chain([last]) {
            let premises = raw
                .premises
                .iter()
                .map(|p| position(*p).ok_or_else(invalid_derivation))
                .collect::<Result<Vec<_>, _>>()?;
            let source = raw
                .input
                .map(|index| {
                    let input = &self.inputs[index];
                    let distribution =
                        catalog.distribution(input.regime).ok_or_else(invalid_derivation)?;
                    Ok::<_, IdentificationError>(MixedSourceLeaf {
                        regime: input.regime,
                        study: Arc::clone(&input.study),
                        population: Arc::clone(&distribution.population),
                        snapshot: input.snapshot.clone(),
                        identity: input.identity.clone(),
                    })
                })
                .transpose()?;
            let mut step = MixedStep {
                rule: raw.rule,
                premises,
                params: self.variables_of(raw.params),
                y: self.variables_of(raw.y),
                intervened: self.variables_of(raw.d),
                conditioned: self.variables_of(raw.z),
                source,
                relaxed: raw.relaxed,
                expression: ExprId::from_raw(0),
            };
            let expressions =
                step.premises.iter().map(|p| steps[*p].expression).collect::<Vec<_>>();
            step.expression = build_expression(&mut arena, &step, &expressions, target)?;
            steps.push(step);
        }
        Ok((steps, arena))
    }
}

// ----------------------------------------------------------- independent checker

/// The proof checker. It reads the graph's adjacency directly and decides every
/// do-calculus side condition with Richardson's augmented-graph criterion, sharing
/// no code with the search's m-separation oracle, and rebuilds every expression in
/// a fresh arena; the shared expression layer then compiles the root and binds
/// its leaves ([`check_with_shared_expression_layer`]).
///
/// The conditional transport checker (`sid::conditional`) reuses its separation
/// test for rule-2 premises, so it is visible to the parent module.
pub(super) struct Checker {
    variables: Vec<VariableId>,
    parents: Vec<Vec<usize>>,
    siblings: Vec<Vec<usize>>,
}

pub(super) type Set = BTreeSet<usize>;

impl Checker {
    pub(super) fn new(graph: &Admg) -> Result<Self, IdentificationError> {
        let variables = dense_variables(graph)?;
        let n = variables.len();
        let dense = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap_or(0));
        let mut parents = vec![Vec::new(); n];
        let mut siblings = vec![Vec::new(); n];
        for (from, from_siblings) in siblings.iter_mut().enumerate() {
            for to in graph.children(dense(from)) {
                parents[to.as_usize()].push(from);
            }
            for to in graph.bidirected_neighbors(dense(from)) {
                from_siblings.push(to.as_usize());
            }
        }
        Ok(Self { variables, parents, siblings })
    }

    fn set(&self, variables: &[VariableId]) -> Result<Set, IdentificationError> {
        let set = variables
            .iter()
            .map(|v| self.variables.iter().position(|w| w == v).ok_or_else(invalid_derivation))
            .collect::<Result<Set, _>>()?;
        if set.len() == variables.len() { Ok(set) } else { Err(invalid_derivation()) }
    }

    /// Ancestors of `seeds` (inclusive) along directed edges not entering `over`.
    fn ancestors(&self, seeds: &Set, over: &Set) -> Set {
        let mut out = seeds.clone();
        let mut pending = seeds.iter().copied().collect::<Vec<_>>();
        while let Some(node) = pending.pop() {
            if over.contains(&node) {
                continue;
            }
            for parent in &self.parents[node] {
                if out.insert(*parent) {
                    pending.push(*parent);
                }
            }
        }
        out
    }

    /// `a` and `b` are m-separated given `cond` after every edge into `over` and
    /// every directed edge out of `under` is removed.
    pub(super) fn separated(&self, over: &Set, under: &Set, a: &Set, b: &Set, cond: &Set) -> bool {
        let n = self.variables.len();
        let mut parents = vec![Vec::new(); n];
        let mut siblings = vec![Vec::new(); n];
        for node in 0..n {
            if over.contains(&node) {
                continue;
            }
            parents[node] =
                self.parents[node].iter().copied().filter(|p| !under.contains(p)).collect();
            siblings[node] =
                self.siblings[node].iter().copied().filter(|s| !over.contains(s)).collect();
        }
        a.iter().all(|x| {
            b.iter().all(|y| {
                !cond.contains(x) && !cond.contains(y) && x != y && {
                    !connected(&parents, &siblings, cond, *x, *y)
                }
            })
        })
    }
}

/// Whether `a` and `b` are connected in the augmented graph of the ancestors of
/// `{a, b} ∪ cond` once the conditioned nodes are removed: adjacent nodes, and
/// every pair inside a bidirected district together with that district's parents.
fn connected(
    parents: &[Vec<usize>],
    siblings: &[Vec<usize>],
    cond: &Set,
    a: usize,
    b: usize,
) -> bool {
    let n = parents.len();
    let mut ancestral = vec![false; n];
    let mut pending = cond.iter().copied().chain([a, b]).collect::<Vec<_>>();
    for &i in &pending {
        ancestral[i] = true;
    }
    while let Some(node) = pending.pop() {
        for &parent in &parents[node] {
            if !ancestral[parent] {
                ancestral[parent] = true;
                pending.push(parent);
            }
        }
    }
    let mut adjacent = vec![vec![false; n]; n];
    let mut link = |p: usize, q: usize| {
        if p != q {
            adjacent[p][q] = true;
            adjacent[q][p] = true;
        }
    };
    for node in (0..n).filter(|&i| ancestral[i]) {
        for &parent in &parents[node] {
            link(parent, node);
        }
        for &sibling in &siblings[node] {
            if ancestral[sibling] {
                link(node, sibling);
            }
        }
    }
    let mut assigned = vec![false; n];
    for seed in (0..n).filter(|&i| ancestral[i]) {
        if assigned[seed] {
            continue;
        }
        let mut district = vec![seed];
        assigned[seed] = true;
        let mut cursor = 0;
        while cursor < district.len() {
            let node = district[cursor];
            cursor += 1;
            for &sibling in &siblings[node] {
                if ancestral[sibling] && !assigned[sibling] {
                    assigned[sibling] = true;
                    district.push(sibling);
                }
            }
        }
        let mut clique = district.clone();
        for &member in &district {
            clique.extend(parents[member].iter().copied());
        }
        clique.sort_unstable();
        clique.dedup();
        for (i, &p) in clique.iter().enumerate() {
            for &q in &clique[i + 1..] {
                link(p, q);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut stack = vec![a];
    seen[a] = true;
    while let Some(node) = stack.pop() {
        if node == b {
            return true;
        }
        for (next, visited) in seen.iter_mut().enumerate() {
            if adjacent[node][next] && !*visited && (!cond.contains(&next) || next == b) {
                *visited = true;
                stack.push(next);
            }
        }
    }
    false
}

/// Whether the do-calculus rule `rule` moves the premise quantity
/// `P(y | do(d), z)` to `conclusion` through the variables `w`, deciding every
/// side condition with [`Checker`]. `free` are the variables the premise's
/// expression still mentions: a deletion may not drop one. Rules that are not
/// one-premise do-calculus rules never hold here.
fn unary_rule_holds(
    checker: &Checker,
    universe: &Set,
    rule: MixedRule,
    (py, pd, pz): (&Set, &Set, &Set),
    (y, d, z): (&Set, &Set, &Set),
    w: &Set,
    free: &Set,
) -> bool {
    let union = |sets: &[&Set]| sets.iter().flat_map(|s| s.iter().copied()).collect::<Set>();
    let minus = |a: &Set, b: &Set| a.difference(b).copied().collect::<Set>();
    if w.is_empty() || py != y {
        return false;
    }
    match rule {
        MixedRule::Rule1Insert => {
            w.is_subset(&outside_of(universe, py, pd, pz))
                && pd == d
                && *z == union(&[pz, w])
                && checker.separated(pd, &Set::new(), py, w, &union(&[pd, pz]))
        }
        MixedRule::Rule1Delete => {
            w.is_subset(pz)
                && w.is_disjoint(free)
                && pd == d
                && *z == minus(pz, w)
                && checker.separated(pd, &Set::new(), py, w, &union(&[pd, z]))
        }
        MixedRule::Rule2ToDo => {
            w.is_subset(pz)
                && *d == union(&[pd, w])
                && *z == minus(pz, w)
                && checker.separated(pd, w, py, w, &union(&[pd, z]))
        }
        MixedRule::Rule2ToObservation => {
            w.is_subset(pd)
                && *d == minus(pd, w)
                && *z == union(&[pz, w])
                && checker.separated(d, w, py, w, &union(&[d, pz]))
        }
        MixedRule::Rule3Insert => {
            // The inserted actions that are ancestors of the observed set in the
            // graph without the edges into the existing actions keep their edges
            // (Pearl's `W(Z)` exception): only the other ones are cut.
            let past = checker.ancestors(pz, pd);
            let exposed = union(&[pd, &minus(w, &past)]);
            w.is_subset(&outside_of(universe, py, pd, pz))
                && *d == union(&[pd, w])
                && z == pz
                && checker.separated(&exposed, &Set::new(), py, w, &union(&[pd, pz]))
        }
        MixedRule::Rule3Delete => {
            let past = checker.ancestors(pz, d);
            let exposed = union(&[d, &minus(w, &past)]);
            w.is_subset(pd)
                && w.is_disjoint(free)
                && *d == minus(pd, w)
                && z == pz
                && checker.separated(&exposed, &Set::new(), py, w, &union(&[d, pz]))
        }
        _ => false,
    }
}

/// Check every step of a proof, independently of the search that produced it.
///
/// Each step's quantity is recomputed from its rule and premises, every
/// do-calculus side condition is decided by [`Checker`], a deletion may not drop a
/// variable the premise's expression still mentions, each expression is rebuilt in
/// a fresh arena and must equal the stored one, each expression's free variables
/// stay inside the quantity, and every input step must name an eligible catalog
/// distribution with its recorded source identity. The last step must answer the
/// query. A relaxed (missing-joint) input is accepted only when `allow_relaxed`.
#[allow(clippy::too_many_lines)] // One pass over the ten rules of the frozen set.
fn check_steps(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
    steps: &[MixedStep],
    arena: Option<&CausalExprArena>,
    allow_relaxed: bool,
) -> Result<(), IdentificationError> {
    let checker = Checker::new(graph)?;
    let validated = validate_mixed_source_query(graph, query, catalog)?;
    let target = query.target.as_ref();
    let universe = (0..checker.variables.len()).collect::<Set>();
    let mut fresh = CausalExprArena::new();
    let mut quantities: Vec<(Set, Set, Set)> = Vec::with_capacity(steps.len());
    let mut expressions: Vec<ExprId> = Vec::with_capacity(steps.len());
    for (index, step) in steps.iter().enumerate() {
        let (y, d, z) = (
            checker.set(&step.y)?,
            checker.set(&step.intervened)?,
            checker.set(&step.conditioned)?,
        );
        let w = checker.set(&step.params)?;
        let disjoint = y.is_disjoint(&d) && y.is_disjoint(&z) && d.is_disjoint(&z);
        if !disjoint
            || y.is_empty()
            || d.len() > MIXED_SOURCE_MAX_DO
            || w.len() > MIXED_SOURCE_MAX_MOVE
            || step.premises.iter().any(|p| *p >= index)
            || (step.rule != MixedRule::Input && (step.source.is_some() || step.relaxed))
            || (step.rule == MixedRule::Product && !w.is_empty())
        {
            return Err(invalid_derivation());
        }
        let premise = |k: usize| step.premises.get(k).map(|p| &quantities[*p]);
        let free_of = |k: usize| -> Result<Set, IdentificationError> {
            let expression = expressions[*step.premises.get(k).ok_or_else(invalid_derivation)?];
            Ok(fresh
                .free_variables(expression)
                .iter()
                .filter_map(|v| checker.variables.iter().position(|x| x == v))
                .collect())
        };
        let union = |sets: &[&Set]| sets.iter().flat_map(|s| s.iter().copied()).collect::<Set>();
        let expected: Option<(Set, Set, Set)> = match (step.rule, step.premises.len()) {
            (MixedRule::Input, 0) => {
                let leaf = step.source.as_ref().ok_or_else(invalid_derivation)?;
                let input = validated
                    .inputs
                    .iter()
                    .chain(validated.marginal_inputs.iter())
                    .find(|input| input.regime == leaf.regime)
                    .ok_or_else(invalid_derivation)?;
                let relaxed = !input.marginals.is_empty();
                let same = checker.set(&input.measured)? == y
                    && checker.set(&input.intervened)? == d
                    && z.is_empty()
                    && w.is_empty()
                    && leaf.study == input.study
                    && leaf.identity == input.identity
                    && leaf.snapshot == input.snapshot
                    && leaf.population.as_ref() == target
                    && step.relaxed == relaxed
                    && (allow_relaxed || !relaxed);
                same.then(|| (y.clone(), d.clone(), z.clone()))
            }
            (MixedRule::Marginalize, 1) => {
                let (py, pd, pz) = premise(0).ok_or_else(invalid_derivation)?;
                (!w.is_empty()
                    && w.is_subset(py)
                    && &union(&[&y, &w]) == py
                    && pd == &d
                    && pz == &z)
                    .then(|| (y.clone(), d.clone(), z.clone()))
            }
            (MixedRule::Condition, 1) => {
                let (py, pd, pz) = premise(0).ok_or_else(invalid_derivation)?;
                (!w.is_empty()
                    && w.is_subset(py)
                    && &union(&[&y, &w]) == py
                    && pd == &d
                    && z == union(&[pz, &w]))
                .then(|| (y.clone(), d.clone(), z.clone()))
            }
            (MixedRule::Product, 2) => {
                let (ay, ad, az) = premise(0).ok_or_else(invalid_derivation)?;
                let (by, bd, bz) = premise(1).ok_or_else(invalid_derivation)?;
                (ad == bd
                    && ad == &d
                    && az == &union(&[bz, by])
                    && ay.is_disjoint(by)
                    && bz == &z
                    && y == union(&[ay, by]))
                .then(|| (y.clone(), d.clone(), z.clone()))
            }
            (rule, 1) => {
                let (py, pd, pz) = premise(0).ok_or_else(invalid_derivation)?;
                if w.is_empty() || py != &y {
                    return Err(invalid_derivation());
                }
                let free = free_of(0)?;
                unary_rule_holds(&checker, &universe, rule, (py, pd, pz), (&y, &d, &z), &w, &free)
                    .then(|| (y.clone(), d.clone(), z.clone()))
            }
            _ => None,
        };
        if expected.is_none() {
            return Err(invalid_derivation());
        }
        let premise_expressions = step.premises.iter().map(|p| expressions[*p]).collect::<Vec<_>>();
        let expression = build_expression(&mut fresh, step, &premise_expressions, target)?;
        if arena.is_some() && expression != step.expression {
            return Err(invalid_derivation());
        }
        let inside = union(&[&y, &d, &z]);
        let free = fresh
            .free_variables(expression)
            .iter()
            .map(|v| checker.variables.iter().position(|x| x == v).ok_or_else(invalid_derivation))
            .collect::<Result<Set, _>>()?;
        if !free.is_subset(&inside) {
            return Err(invalid_derivation());
        }
        quantities.push((y, d, z));
        expressions.push(expression);
    }
    let last = quantities.last().ok_or_else(invalid_derivation)?;
    let goal = (checker.set(&query.outcomes)?, checker.set(&query.treatments)?, Set::new());
    if *last != goal || arena.is_some_and(|stored| *stored != fresh) {
        return Err(invalid_derivation());
    }
    check_with_shared_expression_layer(&fresh, steps, &expressions)
}

/// The shared expression layer's own checks of the rebuilt root: the shared
/// compiler must accept it (well-formed nodes, resolvable references), and its
/// distribution leaves, read by the shared scope analysis, must be exactly the
/// `(population, regime)` pairs of the input steps the proof reaches from its
/// last step. A leaf the proof does not cite, or a cited leaf the expression
/// dropped, fails here even though the step-by-step replay passed.
fn check_with_shared_expression_layer(
    fresh: &CausalExprArena,
    steps: &[MixedStep],
    expressions: &[ExprId],
) -> Result<(), IdentificationError> {
    let root = *expressions.last().ok_or_else(invalid_derivation)?;
    fresh.compile(root).map_err(|_| invalid_derivation())?;
    let mut reached = BTreeSet::new();
    let mut pending = vec![steps.len() - 1];
    while let Some(index) = pending.pop() {
        if reached.insert(index) {
            pending.extend(steps[index].premises.iter().copied());
        }
    }
    let expected = reached
        .into_iter()
        .filter_map(|index| steps[index].source.as_ref())
        .map(|leaf| LeafBinding {
            population: Arc::clone(&leaf.population),
            regime: Some(leaf.regime),
        })
        .collect::<Vec<_>>();
    fresh.bind_certificate(root, &expected).map_err(|_| invalid_derivation())
}

fn outside_of(universe: &Set, y: &Set, d: &Set, z: &Set) -> Set {
    universe
        .iter()
        .copied()
        .filter(|v| !y.contains(v) && !d.contains(v) && !z.contains(v))
        .collect()
}

/// Independently check a derivation against a graph and catalog.
///
/// # Errors
/// `mixed_search.invalid_derivation` when any step, expression or source binding
/// fails its check; `mixed_search.invalid_query` for a malformed query or catalog.
pub fn verify_mixed_source_derivation(
    graph: &Admg,
    derivation: &MixedSourceDerivation,
    catalog: &EvidenceCatalog,
) -> Result<(), IdentificationError> {
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| invalid_query(&error))?;
    if super::graph_signature(&shared) != derivation.graph_signature
        || derivation.query != derivation.query.canonical()
        || derivation.root_step + 1 != derivation.steps.len()
    {
        return Err(invalid_derivation());
    }
    check_steps(
        graph,
        &derivation.query,
        catalog,
        &derivation.steps,
        Some(&derivation.arena),
        false,
    )
}

// ---------------------------------------------------------------- decision

/// The exact leaf a missing-joint refusal needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedMissingLeaf {
    /// The step of the relaxed proof that cites it.
    pub step: usize,
    /// The regime that holds the variables only as separate marginals.
    pub regime: RegimeId,
    /// Its study.
    pub study: Arc<str>,
    /// The jointly measured variables the proof needs as one joint law.
    pub variables: Vec<VariableId>,
    /// The regime's hard interventions.
    pub intervened: Vec<VariableId>,
    /// The variables the regime supplies as separate marginals.
    pub marginals: Vec<VariableId>,
}

/// A derivation that would identify the query if the catalog held a joint it holds
/// only as separate marginals. The proof is checked, with its relaxed leaves marked.
#[derive(Clone, Debug)]
pub struct MixedMissingEvidence {
    /// Every joint the proof cites that the catalog lacks.
    pub leaves: Vec<MixedMissingLeaf>,
    /// The relaxed proof, in step order; its relaxed leaves are marked.
    pub proof: Vec<MixedStep>,
    /// Stages evaluated before the rule search, in order.
    pub stages: Vec<MixedStageRecord>,
}

impl MixedMissingEvidence {
    /// The relaxed proof rendered as a compact graph.
    #[must_use]
    pub fn proof_graph(&self, names: &[String]) -> Vec<String> {
        render_steps(&self.proof, names, self.proof.len().saturating_sub(1))
    }
}

/// What a search that stopped short of a formula explored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedSearchInspection {
    /// Every stage evaluated, in order.
    pub stages: Vec<MixedStageRecord>,
    /// The target quantity nothing derived.
    pub goal: MixedQuantity,
    /// The most recently derived quantities.
    pub frontier: Vec<MixedQuantity>,
    /// Quantities derived.
    pub quantities: usize,
    /// Generations completed.
    pub generations: usize,
    /// Rule applications charged by the rule search.
    pub operations: usize,
    /// Quantities derived per rule.
    pub rule_counts: Vec<(&'static str, usize)>,
    /// Every regime that was not an input, with its reason.
    pub exclusions: Vec<MixedExclusion>,
}

/// Outcome of a bounded mixed-source decision. None of them is a
/// non-identification claim.
#[derive(Clone, Debug)]
pub enum MixedSourceDecision {
    /// A theorem-scoped route identifies the query; use it.
    NamedRoute {
        /// `target_first_sid`, `z_transport` or `mz_transport`.
        route: &'static str,
        /// Stages evaluated before it and including it.
        stages: Vec<MixedStageRecord>,
    },
    /// A checked derivation from the supplied distributions.
    Identified {
        /// The checked derivation.
        derivation: Box<MixedSourceDerivation>,
        /// Catalog regimes the proof cites.
        cited: Arc<[RegimeId]>,
        /// Other derivations of the target found in the same generation, each
        /// independently checked. Empty when none was found.
        alternatives: Vec<MixedSourceDerivation>,
    },
    /// The query is identified only if the catalog held a joint it holds as
    /// separate marginals.
    MissingEvidence(Box<MixedMissingEvidence>),
    /// The frozen rule set reached its fixpoint without the target.
    NotCertified(Box<MixedSearchInspection>),
    /// A limit or cancellation stopped the search; nothing is claimed.
    Exhausted(SearchReceipt),
}

impl MixedSourceDecision {
    /// Top-level reason code of a non-identified outcome; `None` when identified.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified { .. } => None,
            Self::NamedRoute { .. } => Some("route_not_supported"),
            Self::MissingEvidence(_) => Some("transport_missing_evidence"),
            Self::NotCertified(_) => Some("transport_not_certified"),
            Self::Exhausted(_) => Some("transport_budget_cancel"),
        }
    }

    /// Stable `mixed_search.*` detail of a non-identified outcome.
    #[must_use]
    pub const fn detail_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified { .. } => None,
            Self::NamedRoute { .. } => Some(NAMED_ROUTE),
            Self::MissingEvidence(_) => Some(MISSING_JOINT),
            Self::NotCertified(_) => Some(NOT_CERTIFIED),
            Self::Exhausted(_) => Some(BUDGET),
        }
    }
}

const STAGE_SID: &str = "target_first_sid";
const STAGE_Z: &str = "z_transport";
const STAGE_MZ: &str = "mz_transport";
const STAGE_META: &str = "meta_transport";
const STAGE_RULES: &str = "rule_search";

/// Whether the classical meta-transport contract applies to `query`: two or more
/// declared sources, each able to experiment on every observed variable (the
/// unrestricted source-experiment family that theorem assumes). A source with a
/// restricted controllable set is the mz route's business, and one source is the
/// z route's; meta is then not tried and never approximated.
fn meta_applies(query: &MixedSourceQuery, variables: &[VariableId]) -> bool {
    query.sources.len() >= 2
        && query
            .sources
            .iter()
            .all(|source| variables.iter().all(|v| source.controllable.contains(v)))
}

fn stage_names(query: &MixedSourceQuery, meta: bool) -> Vec<&'static str> {
    let named = match query.sources.len() {
        0 => None,
        1 => Some(STAGE_Z),
        _ => Some(STAGE_MZ),
    };
    std::iter::once(STAGE_SID)
        .chain(named)
        .chain(meta.then_some(STAGE_META))
        .chain(std::iter::once(STAGE_RULES))
        .collect()
}

/// Decide a bounded mixed-source query.
///
/// Stages run in a fixed order, each only when the earlier ones do not identify:
/// target-first sID over the catalog's target evidence, the declared z (one
/// source) or mz (two or more) route, then the generic rule search over the
/// target population's distributions and, when it fails, the relaxed pass that
/// names the exact missing joint. One [`SearchBudget`] under `limits` is charged
/// by every step of every stage, so the limits bound the whole decision. A stop
/// anywhere is [`MixedSourceDecision::Exhausted`], never a verdict.
///
/// # Errors
///
/// Limits above [`MIXED_SOURCE_DEFAULT_LIMITS`] or a query outside its bounds
/// (`mixed_search.bounds_exceeded`), a malformed query or catalog
/// (`mixed_search.invalid_query`), a model artifact offered as an experimental
/// law (`mixed_search.posterior_as_law`), or a derivation that fails its own check.
pub fn decide_mixed_source(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<MixedSourceDecision, IdentificationError> {
    if limits.operations > MIXED_SOURCE_DEFAULT_LIMITS.operations
        || limits.depth > MIXED_SOURCE_DEFAULT_LIMITS.depth
    {
        return Err(bounds_exceeded());
    }
    let validated = validate_mixed_source_query(graph, query, catalog)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("{INVALID_QUERY}: {error}"))
    })?;
    match SearchBudget::new(limits, ctx) {
        Ok(budget) => {
            decide_charged(graph, &validated, catalog, &mut SharedSearch::new(budget), ctx)
        }
        Err(receipt) => Ok(MixedSourceDecision::Exhausted(SearchReceipt {
            unevaluated: validated.stage_plan().iter().map(|s| format!("stage:{s}")).collect(),
            ..receipt
        })),
    }
}

/// [`decide_mixed_source`] charging a budget an enclosing plan already shares
/// (the X6 study planner re-decides each preview catalog on it). Validation,
/// stages and outcomes are exactly those of the public route; the caller owns
/// the limits.
///
/// # Errors
///
/// As [`decide_mixed_source`], except the limits.
pub(super) fn decide_mixed_source_shared(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<MixedSourceDecision, IdentificationError> {
    let validated = validate_mixed_source_query(graph, query, catalog)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("{INVALID_QUERY}: {error}"))
    })?;
    decide_charged(graph, &validated, catalog, search, ctx)
}

/// The receipt of a stop: finished stages and completed generations are explored;
/// the stopped and later stages, then `pending`, are unevaluated.
fn stopped_receipt(
    search: &SharedSearch<'_>,
    stop: SearchStop,
    names: &[&'static str],
    stages: &[MixedStageRecord],
    explored: Vec<String>,
    pending: Vec<String>,
) -> SearchReceipt {
    let explored =
        stages.iter().map(|s| format!("stage:{}", s.stage)).chain(explored).collect::<Vec<_>>();
    let unevaluated = names[stages.len().min(names.len())..]
        .iter()
        .map(|s| format!("stage:{s}"))
        .chain(pending)
        .collect::<Vec<_>>();
    search.receipt(stop, explored, unevaluated)
}

fn named_route_error(error: &IdentificationError) -> IdentificationError {
    invalid_query(&format_args!("a theorem-scoped route refused the query: {error}"))
}

/// Run the theorem-scoped routes. `Some` is a final decision (a route identifies,
/// or the budget stopped); `None` continues to the rule search.
#[allow(clippy::too_many_lines)] // One linear pass over the named stages and their outcomes.
fn named_routes(
    graph: &Admg,
    query: &MixedSourceQuery,
    plan: &[&'static str],
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
    stages: &mut Vec<MixedStageRecord>,
) -> Result<Option<MixedSourceDecision>, IdentificationError> {
    let solved = |stages: &mut Vec<MixedStageRecord>, route: &'static str| {
        stages.push(MixedStageRecord { stage: route.into(), outcome: "identified" });
        Ok(Some(MixedSourceDecision::NamedRoute { route, stages: stages.clone() }))
    };
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| invalid_query(&error))?;
    let classical = ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::from("<no source>"),
        target: Arc::clone(&query.target),
    };
    match identify_catalog_transport_metered(&shared, &classical, catalog, search.meter(), ctx) {
        Ok(CatalogTransportResult::Identified(_)) => return solved(stages, STAGE_SID),
        Ok(CatalogTransportResult::MissingEvidence { .. }) => {
            stages.push(MixedStageRecord { stage: STAGE_SID.into(), outcome: "missing_evidence" });
        }
        Ok(CatalogTransportResult::NotCertified { .. }) => {
            stages.push(MixedStageRecord { stage: STAGE_SID.into(), outcome: "not_certified" });
        }
        Err(error) if error.is_budget_or_cancel() => {
            let stop = search.stop_of(&error);
            let receipt = stopped_receipt(search, stop, plan, stages, Vec::new(), Vec::new());
            return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
        }
        Err(error) => return Err(named_route_error(&error)),
    }
    match query.sources.len() {
        0 => {}
        1 => {
            let source = &query.sources[0];
            let diagram =
                SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                    .map_err(|error| invalid_query(&error))?;
            let single = ZTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                controllable: Arc::clone(&source.controllable),
                experiment_assignment: Arc::clone(&source.experiment_assignment),
                source: Arc::clone(&source.population),
                target: Arc::clone(&query.target),
            };
            let mut stopped: Option<ZTransportLimitsReceipt> = None;
            let outcome = match decide_z_transport_reporting(
                &diagram,
                &single,
                catalog,
                search.meter(),
                ctx,
                &mut stopped,
            ) {
                Ok(ZTransportDecision::Identified(_)) => return solved(stages, STAGE_Z),
                Ok(ZTransportDecision::MissingEvidence { .. }) => "missing_evidence",
                Ok(ZTransportDecision::ProvenNonTransportable(_)) => "obstruction",
                Ok(ZTransportDecision::NotCertified { .. }) => "not_certified",
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    let explored = stopped
                        .into_iter()
                        .flat_map(|r| r.explored_rules)
                        .map(|r| format!("rule:{r}"))
                        .collect();
                    let receipt = stopped_receipt(search, stop, plan, stages, explored, Vec::new());
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) => return Err(named_route_error(&error)),
            };
            stages.push(MixedStageRecord { stage: STAGE_Z.into(), outcome });
        }
        _ => {
            let outcome = match super::mz_transport::decide_mz_transport_shared(
                graph,
                &query.mz_query(),
                catalog,
                search,
                ctx,
            ) {
                Ok(MzTransportDecision::Identified { .. }) => return solved(stages, STAGE_MZ),
                Ok(MzTransportDecision::MissingEvidence { .. }) => "missing_evidence",
                Ok(MzTransportDecision::ProvenNonTransportable(_)) => "obstruction",
                Ok(MzTransportDecision::NotCertified(_)) => "not_certified",
                Ok(MzTransportDecision::Exhausted(inner)) => {
                    let receipt = stopped_receipt(
                        search,
                        inner.stop,
                        plan,
                        stages,
                        inner.explored,
                        inner.unevaluated,
                    );
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    let receipt =
                        stopped_receipt(search, stop, plan, stages, Vec::new(), Vec::new());
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) => return Err(named_route_error(&error)),
            };
            stages.push(MixedStageRecord { stage: STAGE_MZ.into(), outcome });
        }
    }
    if plan.contains(&STAGE_META) {
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
        let outcome =
            match identify_meta_catalog_metered(graph, &meta, catalog, search.meter(), ctx) {
                Ok(CatalogTransportResult::Identified(_)) => return solved(stages, STAGE_META),
                Ok(CatalogTransportResult::MissingEvidence { .. }) => "missing_evidence",
                Ok(CatalogTransportResult::NotCertified { .. }) => "not_certified",
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    let receipt =
                        stopped_receipt(search, stop, plan, stages, Vec::new(), Vec::new());
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) => return Err(named_route_error(&error)),
            };
        stages.push(MixedStageRecord { stage: STAGE_META.into(), outcome });
    }
    Ok(None)
}

fn decide_charged(
    graph: &Admg,
    validated: &ValidatedMixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<MixedSourceDecision, IdentificationError> {
    let mut stages = Vec::new();
    if let Some(decision) = named_routes(
        graph,
        &validated.query,
        &validated.stage_plan(),
        catalog,
        search,
        ctx,
        &mut stages,
    )? {
        return Ok(decision);
    }
    rule_search(graph, validated, catalog, search, stages)
}

/// The receipt of a stop inside the rule search: completed generations are
/// explored; the interrupted generation's unexpanded quantities and the relaxed
/// pass are unevaluated.
fn rule_stop_receipt(
    search: &SharedSearch<'_>,
    stop: SearchStop,
    validated: &ValidatedMixedSourceQuery,
    stages: &[MixedStageRecord],
    engine: &Engine<'_>,
    relaxed_pending: bool,
) -> SearchReceipt {
    let explored = (1..=engine.completed).map(|g| format!("rule_search:generation:{g}")).collect();
    let mut pending = Vec::new();
    if engine.unexpanded > 0 {
        pending.push(format!(
            "rule_search:generation:{}:unexpanded={}",
            engine.generation, engine.unexpanded
        ));
    }
    if relaxed_pending {
        pending.push("rule_search:relaxed_missing_joint".to_owned());
    }
    stopped_receipt(search, stop, &validated.stage_plan(), stages, explored, pending)
}

/// Search, check and package the proof of `found` and its alternatives.
fn finish_found(
    graph: &Admg,
    validated: &ValidatedMixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    stages: &[MixedStageRecord],
    engine: &mut Engine<'_>,
) -> Result<MixedSourceDecision, IdentificationError> {
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| invalid_query(&error))?;
    let signature = super::graph_signature(&shared);
    let found = engine.found.ok_or_else(invalid_derivation)?;
    let summary = MixedSearchSummary {
        operations_to_proof: engine.operations_to_proof,
        generation: engine.generation_of(found),
    };
    let mut last_steps = vec![engine.steps[found].clone()];
    last_steps.extend(engine.alternatives.clone());
    let mut derivations = Vec::new();
    for last in &last_steps {
        let (steps, arena) = engine.extract(last, catalog, &validated.query.target)?;
        // Every candidate proof is checked, one charged operation per step.
        for _ in &steps {
            let generation = engine.generation;
            if let Err(stop) = engine.charge(search, generation) {
                let receipt = rule_stop_receipt(search, stop, validated, stages, engine, false);
                return Ok(MixedSourceDecision::Exhausted(receipt));
            }
        }
        check_steps(graph, &validated.query, catalog, &steps, Some(&arena), false)?;
        derivations.push(MixedSourceDerivation {
            query: validated.query.clone(),
            graph_signature: signature.clone(),
            root_step: steps.len() - 1,
            steps,
            arena,
            stages: stages.to_vec(),
            summary,
        });
    }
    let derivation = derivations.remove(0);
    let cited = Arc::<[RegimeId]>::from(derivation.cited_regimes());
    Ok(MixedSourceDecision::Identified {
        derivation: Box::new(derivation),
        cited,
        alternatives: derivations,
    })
}

fn rule_search(
    graph: &Admg,
    validated: &ValidatedMixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    mut stages: Vec<MixedStageRecord>,
) -> Result<MixedSourceDecision, IdentificationError> {
    let variables = dense_variables(graph)?;
    let relaxed_pending = !validated.marginal_inputs.is_empty();
    let mut engine =
        Engine::new(graph, variables.clone(), validated.inputs.clone(), &validated.query)?;
    match engine.run(search) {
        RunEnd::Found => {
            return finish_found(graph, validated, catalog, search, &stages, &mut engine);
        }
        RunEnd::Stopped(stop) => {
            let receipt =
                rule_stop_receipt(search, stop, validated, &stages, &engine, relaxed_pending);
            return Ok(MixedSourceDecision::Exhausted(receipt));
        }
        RunEnd::Fixpoint => {}
    }
    stages.push(MixedStageRecord { stage: STAGE_RULES.into(), outcome: "not_certified" });
    if relaxed_pending {
        let mut inputs = validated.inputs.clone();
        inputs.extend(validated.marginal_inputs.iter().cloned());
        inputs.sort_by_key(|input| input.regime);
        let mut relaxed = Engine::new(graph, variables, inputs, &validated.query)?;
        stages.pop();
        match relaxed.run(search) {
            RunEnd::Found => {
                let found = relaxed.found.ok_or_else(invalid_derivation)?;
                let last = relaxed.steps[found].clone();
                let (proof, _) = relaxed.extract(&last, catalog, &validated.query.target)?;
                check_steps(graph, &validated.query, catalog, &proof, None, true)?;
                let leaves = proof
                    .iter()
                    .enumerate()
                    .filter(|(_, step)| step.relaxed)
                    .filter_map(|(index, step)| {
                        let leaf = step.source.as_ref()?;
                        let input = validated
                            .marginal_inputs
                            .iter()
                            .find(|input| input.regime == leaf.regime)?;
                        Some(MixedMissingLeaf {
                            step: index,
                            regime: leaf.regime,
                            study: Arc::clone(&leaf.study),
                            variables: step.y.clone(),
                            intervened: step.intervened.clone(),
                            marginals: input.marginals.clone(),
                        })
                    })
                    .collect::<Vec<_>>();
                if leaves.is_empty() {
                    return Err(invalid_derivation());
                }
                stages.push(MixedStageRecord {
                    stage: STAGE_RULES.into(),
                    outcome: "missing_evidence",
                });
                return Ok(MixedSourceDecision::MissingEvidence(Box::new(MixedMissingEvidence {
                    leaves,
                    proof,
                    stages,
                })));
            }
            RunEnd::Stopped(stop) => {
                let receipt = rule_stop_receipt(search, stop, validated, &stages, &relaxed, false);
                return Ok(MixedSourceDecision::Exhausted(receipt));
            }
            RunEnd::Fixpoint => {
                stages
                    .push(MixedStageRecord { stage: STAGE_RULES.into(), outcome: "not_certified" });
            }
        }
    }
    let goal = MixedQuantity {
        y: validated.query.outcomes.to_vec(),
        intervened: validated.query.treatments.to_vec(),
        conditioned: Vec::new(),
    };
    let frontier = engine
        .steps
        .iter()
        .rev()
        .take(FRONTIER_SHOWN)
        .map(|raw| engine.quantity(raw))
        .collect::<Vec<_>>();
    Ok(MixedSourceDecision::NotCertified(Box::new(MixedSearchInspection {
        stages,
        goal,
        frontier,
        quantities: engine.steps.len(),
        generations: engine.completed,
        operations: engine.operations,
        rule_counts: engine.rule_counts.iter().map(|(rule, count)| (*rule, *count)).collect(),
        exclusions: validated.exclusions.clone(),
    })))
}

// ----------------------------------------------------------------- binding

/// A checked mixed-source derivation bound to the catalog it cites. Every leaf
/// names its regime, so binding re-checks the derivation against the catalog and
/// records the cited regimes; one provider then evaluates the formula.
#[derive(Clone, Debug)]
pub struct BoundMixedSourceFunctional {
    derivation: MixedSourceDerivation,
    catalog: EvidenceCatalog,
    cited: Arc<[RegimeId]>,
}

impl BoundMixedSourceFunctional {
    /// Checked symbolic derivation.
    #[must_use]
    pub const fn derivation(&self) -> &MixedSourceDerivation {
        &self.derivation
    }
    /// Formula arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.derivation.arena
    }
    /// Formula root.
    #[must_use]
    pub fn root(&self) -> ExprId {
        self.derivation.root()
    }
    /// Frozen catalog the leaves are bound to.
    #[must_use]
    pub const fn catalog(&self) -> &EvidenceCatalog {
        &self.catalog
    }
    /// Every catalog regime a leaf cites, sorted and without repeats.
    #[must_use]
    pub fn cited_regimes(&self) -> &[RegimeId] {
        &self.cited
    }
    /// The study and population of each cited regime.
    #[must_use]
    pub fn cited_sources(&self) -> Vec<(RegimeId, Arc<str>, Arc<str>)> {
        self.derivation
            .leaves()
            .into_iter()
            .map(|(_, leaf)| (leaf.regime, Arc::clone(&leaf.study), Arc::clone(&leaf.population)))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Bind a checked derivation to a catalog: the derivation is re-checked against
/// the catalog, so a leaf always reads the regime whose study supplied it.
///
/// # Errors
/// `mixed_search.invalid_derivation` when the derivation does not check against
/// `graph` and `catalog`; `mixed_search.invalid_query` for a malformed catalog.
pub fn bind_mixed_source_catalog(
    graph: &Admg,
    derivation: &MixedSourceDerivation,
    catalog: &EvidenceCatalog,
) -> Result<BoundMixedSourceFunctional, IdentificationError> {
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("{INVALID_QUERY}: {error}"))
    })?;
    verify_mixed_source_derivation(graph, derivation, catalog)?;
    Ok(BoundMixedSourceFunctional {
        derivation: derivation.clone(),
        catalog: catalog.clone(),
        cited: Arc::from(derivation.cited_regimes()),
    })
}

/// Write a quantity list for a stopped search, one rendered quantity per line.
#[must_use]
#[doc(hidden)]
pub fn render_frontier(frontier: &[MixedQuantity], names: &[String]) -> String {
    let mut out = String::new();
    for quantity in frontier {
        let _ = writeln!(out, "{}", render_quantity(quantity, names));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, EvidenceKind, EvidenceRegime, RegimeBinding, RegimeKind, SamplingDesign,
    };

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    fn v_of(i: usize) -> VariableId {
        VariableId::from_raw(u32::try_from(i).unwrap())
    }

    fn admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
        let mut graph = Admg::with_variables(n);
        for (a, b) in directed {
            graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        for (a, b) in bidirected {
            graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
        }
        graph
    }

    fn catalog(regimes: &[(u32, &[u32])]) -> EvidenceCatalog {
        let regimes = regimes
            .iter()
            .map(|(id, measured)| {
                EvidenceRegime::try_new(
                    RegimeId::from_raw(*id),
                    RegimeKind::Observational,
                    EvidenceKind::Available,
                    [],
                    [],
                    measured.iter().copied().map(v).collect::<Vec<_>>(),
                    "target",
                    DistributionAvailability::Joint,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let bindings = regimes
            .iter()
            .map(|r| RegimeBinding {
                dataset_identity: None,
                regime: r.id,
                snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            })
            .collect::<Vec<_>>();
        EvidenceCatalog::try_new([], regimes, bindings, None).unwrap()
    }

    fn query() -> MixedSourceQuery {
        MixedSourceQuery {
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
            target: Arc::from("target"),
            sources: Arc::from([]),
        }
    }

    /// The search's oracle (the graph crate's m-separation on a mutilated copy)
    /// and the checker's independent augmented-graph criterion agree on every
    /// sampled ADMG over four nodes, mutilation and conditioning set.
    #[test]
    fn oracle_and_independent_checker_agree_on_mutilated_separation() {
        let n = 4usize;
        let pairs: Vec<(u32, u32)> = (0..4).flat_map(|a| (a + 1..4).map(move |b| (a, b))).collect();
        let singles = std::iter::once(0).chain((0..n).map(bit)).collect::<Vec<Mask>>();
        let mut checked = 0usize;
        for code in (0..3usize.pow(6)).step_by(11) {
            let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
            let mut rest = code;
            for &(a, b) in &pairs {
                match rest % 3 {
                    1 => directed.push((a, b)),
                    2 => bidirected.push((a, b)),
                    _ => {}
                }
                rest /= 3;
            }
            let graph = admg(4, &directed, &bidirected);
            let checker = Checker::new(&graph).unwrap();
            let variables = (0..4).map(v).collect::<Vec<_>>();
            let mut engine = Engine::new(&graph, variables, Vec::new(), &{
                let mut q = query();
                q.outcomes = Arc::from([v(1)]);
                q
            })
            .unwrap();
            let set = |mask: Mask| members(mask).collect::<Set>();
            for &over in &singles {
                for &under in &singles {
                    for a in (0..n).map(bit) {
                        for b in (0..n).map(bit).filter(|b| *b != a) {
                            for cond in 0..16 as Mask {
                                if cond & (a | b) != 0 {
                                    continue;
                                }
                                let search = engine.separated(over, under, a, b, cond);
                                let check = checker.separated(
                                    &set(over),
                                    &set(under),
                                    &set(a),
                                    &set(b),
                                    &set(cond),
                                );
                                assert_eq!(
                                    search, check,
                                    "{directed:?} {bidirected:?} over {over:#b} under {under:#b} \
                                     {a:#b} vs {b:#b} | {cond:#b}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 50_000, "{checked}");
    }

    /// The checker decides side conditions itself: a hand-built proof that
    /// exchanges an observation for an action across an unblocked confounding
    /// path is rejected even though its expressions are well formed.
    #[test]
    fn the_checker_rejects_an_unsound_exchange_the_search_never_proposes() {
        let bow = admg(2, &[(0, 1)], &[(0, 1)]);
        let c = catalog(&[(1, &[0, 1])]);
        let q = query();
        let leaf = MixedSourceLeaf {
            regime: RegimeId::from_raw(1),
            study: Arc::from("target"),
            population: Arc::from("target"),
            snapshot: Some(Arc::from("snapshot-1")),
            identity: c.distribution(RegimeId::from_raw(1)).unwrap().canonical_identity(),
        };
        let step = |rule,
                    premises: Vec<usize>,
                    params: Vec<u32>,
                    y: Vec<u32>,
                    d: Vec<u32>,
                    z: Vec<u32>| {
            MixedStep {
                rule,
                premises,
                params: params.into_iter().map(v).collect(),
                y: y.into_iter().map(v).collect(),
                intervened: d.into_iter().map(v).collect(),
                conditioned: z.into_iter().map(v).collect(),
                source: (rule == MixedRule::Input).then(|| leaf.clone()),
                relaxed: false,
                expression: ExprId::from_raw(0),
            }
        };
        let mut steps = vec![
            step(MixedRule::Input, vec![], vec![], vec![0, 1], vec![], vec![]),
            step(MixedRule::Condition, vec![0], vec![0], vec![1], vec![], vec![0]),
            step(MixedRule::Rule2ToDo, vec![1], vec![0], vec![1], vec![0], vec![]),
        ];
        let mut arena = CausalExprArena::new();
        for i in 0..steps.len() {
            let premises =
                steps[i].premises.iter().map(|p| steps[*p].expression).collect::<Vec<_>>();
            steps[i].expression =
                build_expression(&mut arena, &steps[i], &premises, "target").unwrap();
        }
        assert_eq!(
            check_steps(&bow, &q, &c, &steps, Some(&arena), false).unwrap_err(),
            invalid_derivation()
        );
        // Without the confounder the same steps are a valid derivation.
        let clean = admg(2, &[(0, 1)], &[]);
        check_steps(&clean, &q, &c, &steps, Some(&arena), false).unwrap();
        // ... and the search finds exactly that on the clean graph, and nothing on the bow.
        let ctx = ExecutionContext::for_tests(1);
        let found = |graph: &Admg| {
            matches!(
                decide_mixed_source(graph, &q, &c, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap(),
                MixedSourceDecision::NamedRoute { .. } | MixedSourceDecision::Identified { .. }
            )
        };
        assert!(found(&clean) && !found(&bow));
    }

    /// A binary structural model consistent with an ADMG: every node is a random
    /// function of its parents and the latent bits of its bidirected edges, flipped
    /// by independent noise so every configuration has positive mass.
    type Mechanism = Box<dyn Fn(&[u8], &[u8]) -> u8>;

    struct Model {
        n: usize,
        exo_p: Vec<f64>,
        f: Vec<Mechanism>,
    }

    impl Model {
        fn random(
            state: &mut u64,
            n: usize,
            directed: &[(usize, usize)],
            bidirected: &[(usize, usize)],
        ) -> Self {
            let mut next = || {
                *state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (*state >> 33) as u32
            };
            let mut exo_p =
                (0..n).map(|_| 0.15 + 0.2 * f64::from(next() % 1000) / 1000.0).collect::<Vec<_>>();
            exo_p.extend(bidirected.iter().map(|_| 0.3 + 0.4 * f64::from(next() % 1000) / 1000.0));
            let f = (0..n)
                .map(|i| {
                    let parents = directed
                        .iter()
                        .filter(|(_, to)| *to == i)
                        .map(|(from, _)| *from)
                        .collect::<Vec<_>>();
                    let latents = bidirected
                        .iter()
                        .enumerate()
                        .filter(|(_, (a, b))| *a == i || *b == i)
                        .map(|(k, _)| n + k)
                        .collect::<Vec<_>>();
                    let width = parents.len() + latents.len();
                    let table =
                        (0..1usize << width).map(|_| u8::from(next() % 2 == 1)).collect::<Vec<_>>();
                    Box::new(move |values: &[u8], exo: &[u8]| {
                        let key = parents
                            .iter()
                            .map(|p| values[*p])
                            .chain(latents.iter().map(|l| exo[*l]))
                            .fold(0usize, |acc, bit| (acc << 1) | usize::from(bit));
                        table[key] ^ exo[i]
                    }) as Mechanism
                })
                .collect();
            Self { n, exo_p, f }
        }

        /// Exact joint law over `measured` (first variable most significant) under `do_`.
        fn law(&self, do_: &[(usize, u8)], measured: &[usize]) -> Vec<f64> {
            let mut out = vec![0.0; 1 << measured.len()];
            let m = self.exo_p.len();
            for mask in 0..(1usize << m) {
                let exo = (0..m).map(|bit| u8::from((mask >> bit) & 1 == 1)).collect::<Vec<_>>();
                let mut weight = 1.0;
                for (bit, p) in self.exo_p.iter().enumerate() {
                    weight *= if exo[bit] == 1 { *p } else { 1.0 - *p };
                }
                let mut values = vec![0u8; self.n];
                for i in 0..self.n {
                    values[i] = match do_.iter().find(|(v, _)| *v == i) {
                        Some((_, level)) => *level,
                        None => (self.f[i])(&values, &exo),
                    };
                }
                let index =
                    measured.iter().fold(0usize, |acc, v| (acc << 1) | usize::from(values[*v]));
                out[index] += weight;
            }
            out
        }
    }

    /// Every quantity the frozen rule set derives, deletions included, computes
    /// exactly the enumerated model's `P(y | z, do(d))`: each rule is applied to
    /// every quantity of the closure over an observational joint and a trial, and
    /// the resulting expression is evaluated by the exact provider and compared
    /// with the truth. This covers rules the identified-query sweeps rarely reach.
    #[test]
    #[allow(clippy::too_many_lines)] // One fixture, one closure, one comparison per quantity.
    fn every_derived_quantity_matches_the_enumerated_model() {
        use antecedent_expr::{
            Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactEvaluationPlan,
            ExactTransportData, InterventionAssignment as ExprAssignment, LawTolerance,
        };
        let ctx = ExecutionContext::for_tests(1);
        let mut state = 0x5EED_u64;
        let (mut checked, mut deletions) = (0usize, 0usize);
        for case in 0..24 {
            let n = 4;
            let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
            for a in 0..n {
                for b in a + 1..n {
                    state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    match (state >> 40) % 5 {
                        0 | 1 => directed.push((a, b)),
                        2 => bidirected.push((a, b)),
                        _ => {}
                    }
                }
            }
            let model = Model::random(&mut state, n, &directed, &bidirected);
            let graph = admg(
                4,
                &directed
                    .iter()
                    .map(|(a, b)| (u32::try_from(*a).unwrap(), u32::try_from(*b).unwrap()))
                    .collect::<Vec<_>>(),
                &bidirected
                    .iter()
                    .map(|(a, b)| (u32::try_from(*a).unwrap(), u32::try_from(*b).unwrap()))
                    .collect::<Vec<_>>(),
            );
            // Regime 1: observational joint of every variable; regime 2: a trial of one
            // variable measuring the rest.
            let trial_on = case % n;
            let rest = (0..n).filter(|v| *v != trial_on).collect::<Vec<_>>();
            let studies = if case % 2 == 0 {
                vec![
                    (1u32, Vec::<usize>::new(), (0..n).collect::<Vec<_>>()),
                    (2, vec![trial_on], rest),
                ]
            } else {
                let partial = rest.iter().copied().take(2).collect::<Vec<_>>();
                vec![
                    (1u32, Vec::<usize>::new(), vec![0, 1, 2]),
                    (2, Vec::new(), vec![1, 2, 3]),
                    (3, vec![trial_on], partial),
                ]
            };
            let mut regimes = Vec::new();
            let mut laws = Vec::new();
            for (id, on, measured) in &studies {
                let measured: &Vec<usize> = measured;
                regimes.push(
                    EvidenceRegime::try_new(
                        RegimeId::from_raw(*id),
                        if on.is_empty() {
                            RegimeKind::Observational
                        } else {
                            RegimeKind::Experimental
                        },
                        EvidenceKind::Available,
                        on.iter().map(|v| v_of(*v)).collect::<Vec<_>>(),
                        [],
                        measured.iter().map(|v| v_of(*v)).collect::<Vec<_>>(),
                        "target",
                        DistributionAvailability::Joint,
                    )
                    .unwrap(),
                );
                let axes = measured
                    .iter()
                    .map(|i| DiscreteAxis {
                        variable: v_of(*i),
                        values: Arc::from([
                            antecedent_core::Value::Bool(false),
                            antecedent_core::Value::Bool(true),
                        ]),
                    })
                    .collect::<Vec<_>>();
                for levels in 0..(1usize << on.len()) {
                    let assignments = on
                        .iter()
                        .enumerate()
                        .map(|(b, v)| (*v, u8::from((levels >> b) & 1 == 1)))
                        .collect::<Vec<_>>();
                    laws.push(
                        ExactDiscreteLaw::try_new(
                            "target",
                            RegimeId::from_raw(*id),
                            assignments
                                .iter()
                                .map(|(v, l)| {
                                    ExprAssignment::concrete(
                                        v_of(*v),
                                        antecedent_core::Value::Bool(*l == 1),
                                    )
                                })
                                .collect::<Vec<_>>(),
                            axes.clone(),
                            model.law(&assignments, measured),
                            format!("snapshot-{id}"),
                            LawTolerance::default(),
                        )
                        .unwrap(),
                    );
                }
            }
            // Every study is cited: each leaf selects its law by world among them.
            let cited = regimes.iter().map(|regime| regime.id).collect::<Vec<_>>();
            let data =
                ExactTransportData::try_new(laws, 4096).unwrap().with_world_bound_leaves(cited);
            let catalog = EvidenceCatalog::try_new(
                [],
                regimes.clone(),
                regimes
                    .iter()
                    .map(|r| RegimeBinding {
                        dataset_identity: None,
                        regime: r.id,
                        snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
                        schema_names: Arc::from([]),
                        sampling: SamplingDesign::Independent,
                        weights: None,
                        dependence: DependenceGroup::IndependentStudies,
                    })
                    .collect::<Vec<_>>(),
                None,
            )
            .unwrap();
            let q = MixedSourceQuery {
                outcomes: Arc::from([v_of(1)]),
                treatments: Arc::from([v_of(0)]),
                target: Arc::from("target"),
                sources: Arc::from([]),
            };
            let validated = validate_mixed_source_query(&graph, &q, &catalog).unwrap();
            let variables = (0..n).map(v_of).collect::<Vec<_>>();
            let mut engine =
                Engine::new(&graph, variables, validated.inputs.clone(), &validated.query).unwrap();
            // Close the rule set: no target, a generous budget.
            engine.target = (0, 0, 0);
            engine.found = None;
            let budget =
                SearchBudget::new(SearchLimits { operations: 4_000_000, depth: 64 }, &ctx).unwrap();
            let mut search = SharedSearch::new(budget);
            assert!(matches!(engine.run(&mut search), RunEnd::Fixpoint));
            let mut verify = |steps: &[MixedStep], arena: &CausalExprArena, label: &str| {
                let step = &steps[steps.len() - 1];
                let (y, d, z) = (step.y.clone(), step.intervened.clone(), step.conditioned.clone());
                let sorted =
                    |vs: &[VariableId]| vs.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
                let bound = z.iter().chain(d.iter()).copied().collect::<Vec<_>>();
                // Every assignment of the conditioning values and the intervention levels.
                for levels in 0..(1usize << bound.len()) {
                    let assign = bound
                        .iter()
                        .enumerate()
                        .map(|(b, v)| (*v, u8::from((levels >> b) & 1 == 1)))
                        .collect::<Vec<_>>();
                    let request = Assignment::from_pairs(
                        assign.iter().map(|(v, l)| (*v, antecedent_core::Value::Bool(*l == 1))),
                    );
                    let plan = ExactEvaluationPlan::compile(
                        arena,
                        step.expression,
                        data.clone(),
                        y.clone(),
                        request,
                        ExactEvaluationLimits::default(),
                        LawTolerance::default(),
                        &ctx,
                    )
                    .unwrap_or_else(|e| panic!("{directed:?} {bidirected:?} {label}: {e:?}"));
                    let got = plan.evaluate(&ctx).unwrap_or_else(|e| {
                        panic!("{directed:?} {bidirected:?} {label} {y:?}|do{d:?},{z:?}: {e:?}")
                    });
                    let levels_of = |vs: &[VariableId]| {
                        vs.iter()
                            .map(|v| assign.iter().find(|(w, _)| w == v).map(|(_, l)| *l).unwrap())
                            .collect::<Vec<_>>()
                    };
                    let do_ = sorted(&d).into_iter().zip(levels_of(&d)).collect::<Vec<_>>();
                    let measured = sorted(&y).into_iter().chain(sorted(&z)).collect::<Vec<_>>();
                    let joint = model.law(&do_, &measured);
                    let z_levels = levels_of(&z);
                    for (atom, p) in got.atoms.iter().zip(got.probabilities.iter()) {
                        let y_levels = atom
                            .iter()
                            .map(|value| u8::from(*value == antecedent_core::Value::Bool(true)))
                            .collect::<Vec<_>>();
                        let key = |ys: &[u8]| {
                            ys.iter()
                                .chain(z_levels.iter())
                                .fold(0usize, |acc, bit| (acc << 1) | usize::from(*bit))
                        };
                        let denominator: f64 = (0..1usize << y.len())
                            .map(|combo| {
                                let ys = (0..y.len())
                                    .map(|b| u8::from((combo >> (y.len() - 1 - b)) & 1 == 1))
                                    .collect::<Vec<_>>();
                                joint[key(&ys)]
                            })
                            .sum();
                        let truth = joint[key(&y_levels)] / denominator;
                        assert!(
                            (p - truth).abs() < 1e-9,
                            "{directed:?} {bidirected:?} {label}: P({y:?} | do{d:?}, {z:?}) = {p}, truth {truth}"
                        );
                        checked += 1;
                    }
                }
            };
            for index in 0..engine.steps.len() {
                let last = engine.steps[index].clone();
                let (steps, arena) = engine.extract(&last, &catalog, "target").unwrap();
                verify(&steps, &arena, &format!("step {index} {:?}", last.rule));
            }
            // Deletions never derive a new quantity here, so their candidates are checked
            // directly: whenever the independent checker accepts one, its number is the truth.
            for index in 0..engine.steps.len() {
                let base = engine.steps[index].clone();
                for (rule, pool) in
                    [(MixedRule::Rule1Delete, base.z), (MixedRule::Rule3Delete, base.d)]
                {
                    for w in subsets(pool, MIXED_SOURCE_MAX_MOVE) {
                        let mut candidate = base.clone();
                        candidate.rule = rule;
                        candidate.premises = vec![index];
                        candidate.params = w;
                        candidate.input = None;
                        candidate.relaxed = false;
                        if rule == MixedRule::Rule1Delete {
                            candidate.z = base.z & !w;
                        } else {
                            candidate.d = base.d & !w;
                        }
                        let (steps, arena) =
                            engine.extract(&candidate, &catalog, "target").unwrap();
                        if check_steps(&graph, &q, &catalog, &steps, Some(&arena), false).is_ok() {
                            deletions += 1;
                            verify(&steps, &arena, &format!("{rule:?} of step {index}"));
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000, "{checked}");
        assert!(deletions > 20, "{deletions}");
        eprintln!("checked {checked} cells, {deletions} valid deletion applications");
    }

    type LawCache = HashMap<(Vec<(usize, u8)>, Vec<usize>), Vec<f64>>;
    type Edges = Vec<(usize, usize)>;

    fn u32_edges(edges: &[(usize, usize)]) -> Vec<(u32, u32)> {
        edges
            .iter()
            .map(|(a, b)| (u32::try_from(*a).unwrap(), u32::try_from(*b).unwrap()))
            .collect()
    }

    /// `P(y | do(d), z)` of the enumerated model at the given levels, over the
    /// assignments of `y` (first variable most significant).
    fn truth(
        model: &Model,
        cache: &mut LawCache,
        do_: &[(usize, u8)],
        y: &[usize],
        z: &[(usize, u8)],
    ) -> Vec<f64> {
        let measured = y.iter().copied().chain(z.iter().map(|(v, _)| *v)).collect::<Vec<_>>();
        let joint = cache
            .entry((do_.to_vec(), measured.clone()))
            .or_insert_with(|| model.law(do_, &measured))
            .clone();
        let key = z.iter().fold(0usize, |acc, (_, level)| (acc << 1) | usize::from(*level));
        let cells =
            (0..1usize << y.len()).map(|ys| joint[(ys << z.len()) | key]).collect::<Vec<_>>();
        let total = cells.iter().sum::<f64>();
        cells.iter().map(|cell| cell / total).collect()
    }

    /// The unary do-calculus rules the search attempts (deletions are checked but
    /// not searched), with the quantity each moves a premise to.
    const SEARCHED_UNARY: [MixedRule; 4] = [
        MixedRule::Rule1Insert,
        MixedRule::Rule2ToDo,
        MixedRule::Rule2ToObservation,
        MixedRule::Rule3Insert,
    ];

    fn conclusion_of(
        rule: MixedRule,
        (y, d, z): (Mask, Mask, Mask),
        w: Mask,
    ) -> (Mask, Mask, Mask) {
        match rule {
            MixedRule::Rule1Insert => (y, d, z | w),
            MixedRule::Rule1Delete => (y, d, z & !w),
            MixedRule::Rule2ToDo => (y, d | w, z & !w),
            MixedRule::Rule2ToObservation => (y, d & !w, z | w),
            MixedRule::Rule3Insert => (y, d | w, z),
            MixedRule::Rule3Delete => (y, d & !w, z),
            _ => unreachable!("only the one-premise do-calculus rules"),
        }
    }

    /// Every do-calculus rule instance the independent checker accepts asserts an
    /// equality; the enumerated structural model must satisfy it. Both the search's
    /// oracle and the checker share the side-condition definitions, so a rule that
    /// is unsound in both (say rule 3 without Pearl's `W(Z)` exception) is caught
    /// here and nowhere else: this is the semantic test of each rule's side
    /// condition. For every random ADMG, random model and every premise quantity
    /// `P(y | do(d), z)` the test also requires the search's accepted applications
    /// to equal the checker's for the searched rules.
    #[test]
    #[allow(clippy::too_many_lines)] // One enumeration of graphs, premises, rules and levels.
    fn every_accepted_rule_instance_holds_in_the_enumerated_model() {
        let ctx = ExecutionContext::for_tests(1);
        let mut state = 0x0BAD_5EED_u64;
        let mut next = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 40) as usize
        };
        let mut graphs: Vec<(usize, Edges, Edges)> = Vec::new();
        // Every three-node ADMG (each pair: none, directed, bidirected, both).
        for code in 0..4usize.pow(3) {
            let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
            for (k, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                let kind = (code >> (2 * k)) & 3;
                if kind & 1 == 1 {
                    directed.push((a, b));
                }
                if kind & 2 == 2 {
                    bidirected.push((a, b));
                }
            }
            graphs.push((3, directed, bidirected));
        }
        // Random four- and five-node ADMGs.
        for size in [4usize, 4, 5] {
            for _ in 0..if size == 4 { 30 } else { 8 } {
                let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
                for a in 0..size {
                    for b in a + 1..size {
                        match next() % 6 {
                            0 | 1 => directed.push((a, b)),
                            2 => bidirected.push((a, b)),
                            3 => {
                                directed.push((a, b));
                                bidirected.push((a, b));
                            }
                            _ => {}
                        }
                    }
                }
                bidirected.truncate(if size == 4 { 4 } else { 3 });
                graphs.push((size, directed, bidirected));
            }
        }
        let mut accepted = BTreeMap::<&str, usize>::new();
        let mut multi_w_with_exception = 0usize;
        let mut exception_only_insert = 0usize;
        let mut ancestral_and_non_ancestral = 0usize;
        let mut compared = 0usize;
        for (index, (n, directed, bidirected)) in graphs.iter().enumerate() {
            let (n, cap) = (*n, Mask::try_from(1usize << *n).unwrap());
            let graph =
                admg(u32::try_from(n).unwrap(), &u32_edges(directed), &u32_edges(bidirected));
            let mut model_state = 0x1000 + index as u64;
            let model = Model::random(&mut model_state, n, directed, bidirected);
            let mut cache = HashMap::new();
            let checker = Checker::new(&graph).unwrap();
            let universe = (0..n).collect::<Set>();
            let variables = (0..n).map(v_of).collect::<Vec<_>>();
            let stride = if n == 5 { 3 } else { 1 };
            let set = |mask: Mask| members(mask).collect::<Set>();
            for code in (0..4usize.pow(u32::try_from(n).unwrap())).step_by(stride) {
                let role = |i: usize| (code >> (2 * i)) & 3;
                let mask_of =
                    |r: usize| (0..n).filter(|i| role(*i) == r).fold(0, |m, i| m | bit(i));
                let (y, d, z) = (mask_of(0), mask_of(1), mask_of(2));
                if y == 0 || size(d) > MIXED_SOURCE_MAX_DO {
                    continue;
                }
                let mut engine =
                    Engine::new(&graph, variables.clone(), Vec::new(), &query()).unwrap();
                engine.add(Raw {
                    rule: MixedRule::Input,
                    premises: Vec::new(),
                    params: 0,
                    y,
                    d,
                    z,
                    free: y | d,
                    input: None,
                    relaxed: false,
                });
                let budget =
                    SearchBudget::new(SearchLimits { operations: 4_000_000, depth: 64 }, &ctx)
                        .unwrap();
                let mut search = SharedSearch::new(budget);
                engine.expand(&mut search, 0, 1).unwrap();
                let found = engine
                    .steps
                    .iter()
                    .filter(|s| SEARCHED_UNARY.contains(&s.rule) && s.premises == [0])
                    .map(|s| (s.rule.as_str(), s.params))
                    .collect::<BTreeSet<_>>();
                let mut by_checker = BTreeSet::new();
                for w in 1..cap {
                    if size(w) > MIXED_SOURCE_MAX_MOVE {
                        continue;
                    }
                    for rule in [
                        MixedRule::Rule1Insert,
                        MixedRule::Rule1Delete,
                        MixedRule::Rule2ToDo,
                        MixedRule::Rule2ToObservation,
                        MixedRule::Rule3Insert,
                        MixedRule::Rule3Delete,
                    ] {
                        let (cy, cd, cz) = conclusion_of(rule, (y, d, z), w);
                        if size(cd) > MIXED_SOURCE_MAX_DO {
                            continue;
                        }
                        let holds = unary_rule_holds(
                            &checker,
                            &universe,
                            rule,
                            (&set(y), &set(d), &set(z)),
                            (&set(cy), &set(cd), &set(cz)),
                            &set(w),
                            &set(y),
                        );
                        if !holds {
                            continue;
                        }
                        if SEARCHED_UNARY.contains(&rule) {
                            by_checker.insert((rule.as_str(), w));
                        }
                        *accepted.entry(rule.as_str()).or_default() += 1;
                        if rule == MixedRule::Rule3Insert {
                            let past = checker.ancestors(&set(z), &set(d));
                            let inside = set(w).intersection(&past).count();
                            if inside > 0 && inside < size(w) {
                                ancestral_and_non_ancestral += 1;
                            }
                            if size(w) == 2 && inside > 0 {
                                multi_w_with_exception += 1;
                            }
                            if inside == 0 {
                                exception_only_insert += 1;
                            }
                        }
                        // The equality the rule asserts, at every level.
                        let bound = set(d | z | w | cd | cz).into_iter().collect::<Vec<_>>();
                        let ys = members(y).collect::<Vec<_>>();
                        for levels in 0..1usize << bound.len() {
                            let level = |v: usize| {
                                let position = bound.iter().position(|b| *b == v).unwrap();
                                u8::from((levels >> position) & 1 == 1)
                            };
                            let at = |mask: Mask| {
                                members(mask).map(|v| (v, level(v))).collect::<Vec<_>>()
                            };
                            let before = truth(&model, &mut cache, &at(d), &ys, &at(z));
                            let after = truth(&model, &mut cache, &at(cd), &ys, &at(cz));
                            for (a, b) in before.iter().zip(&after) {
                                assert!(
                                    (a - b).abs() < 1e-9,
                                    "{rule:?} through {w:#b} is unsound: {directed:?} \
                                     {bidirected:?} P({ys:?} | do({d:#b}), {z:#b}) = {a}, but \
                                     the conclusion (do {cd:#b}, given {cz:#b}) is {b}"
                                );
                            }
                        }
                    }
                }
                compared += 1;
                assert_eq!(
                    found, by_checker,
                    "search and checker disagree on {directed:?} {bidirected:?} \
                     y {y:#b} do {d:#b} z {z:#b}"
                );
            }
        }
        eprintln!(
            "{compared} premises, accepted {accepted:?}, rule 3 insertions: \
             {multi_w_with_exception} multi-W with an ancestral member, \
             {exception_only_insert} with no ancestral member, \
             {ancestral_and_non_ancestral} mixing both"
        );
        for rule in ["rule1_insert", "rule1_delete", "rule2_to_do", "rule2_to_observation"] {
            assert!(accepted.get(rule).copied().unwrap_or(0) > 100, "{rule}: {accepted:?}");
        }
        for rule in ["rule3_insert", "rule3_delete"] {
            assert!(accepted.get(rule).copied().unwrap_or(0) > 100, "{rule}: {accepted:?}");
        }
        assert!(multi_w_with_exception > 20, "{multi_w_with_exception}");
        assert!(exception_only_insert > 20, "{exception_only_insert}");
        assert!(ancestral_and_non_ancestral > 20, "{ancestral_and_non_ancestral}");
    }

    /// The counterexample rule 3 needs its `W(Z)` exception for: `W -> Z` and
    /// `W <-> Y`. Deleting `do(W)` from `P(y | do(w), z)` (or inserting it into
    /// `P(y | z)`) is invalid because `W` is an ancestor of the observed `Z`, so
    /// conditioning on `Z` re-opens the confounded path; the enumerated model
    /// proves the two quantities differ. The same rules are valid once `W` is not
    /// an ancestor of `Z`.
    #[test]
    fn rule_three_refuses_the_ancestral_counterexample() {
        let (w, zed, y) = (0usize, 1usize, 2usize);
        let one = |i: usize| Set::from([i]);
        let both = |a: usize, b: usize| Set::from([a, b]);
        let none = Set::new();
        let counterexample = admg(3, &[(0, 1)], &[(0, 2)]);
        let checker = Checker::new(&counterexample).unwrap();
        let universe = Set::from([0, 1, 2]);
        // P(y | do(w), z) -> P(y | z): refused.
        assert!(!unary_rule_holds(
            &checker,
            &universe,
            MixedRule::Rule3Delete,
            (&one(y), &one(w), &one(zed)),
            (&one(y), &none, &one(zed)),
            &one(w),
            &none,
        ));
        // P(y | z) -> P(y | do(w), z): refused.
        assert!(!unary_rule_holds(
            &checker,
            &universe,
            MixedRule::Rule3Insert,
            (&one(y), &none, &one(zed)),
            (&one(y), &one(w), &one(zed)),
            &one(w),
            &none,
        ));
        // ... and the model agrees that they differ.
        let differs = (0..20u64).any(|seed| {
            let mut state = 0xC0DE + seed;
            let model = Model::random(&mut state, 3, &[(0, 1)], &[(0, 2)]);
            let mut cache = HashMap::new();
            [0u8, 1].into_iter().any(|z_level| {
                [0u8, 1].into_iter().any(|w_level| {
                    let with = truth(&model, &mut cache, &[(w, w_level)], &[y], &[(zed, z_level)]);
                    let without = truth(&model, &mut cache, &[], &[y], &[(zed, z_level)]);
                    (with[1] - without[1]).abs() > 1e-3
                })
            })
        });
        assert!(differs, "the counterexample must separate the two quantities");
        // Without the confounding edge both rules hold.
        let plain = Checker::new(&admg(3, &[(0, 1)], &[])).unwrap();
        assert!(unary_rule_holds(
            &plain,
            &universe,
            MixedRule::Rule3Delete,
            (&one(y), &one(w), &one(zed)),
            (&one(y), &none, &one(zed)),
            &one(w),
            &none,
        ));
        // With `Z -> W` instead (`W` is not an ancestor of `Z`) the exception cuts the
        // confounding edge and both rules hold even though `W <-> Y` remains.
        let reversed = Checker::new(&admg(3, &[(1, 0)], &[(0, 2)])).unwrap();
        for (rule, premise, conclusion) in [
            (MixedRule::Rule3Delete, (one(w), one(zed)), (none.clone(), one(zed))),
            (MixedRule::Rule3Insert, (none.clone(), one(zed)), (one(w), one(zed))),
        ] {
            assert!(unary_rule_holds(
                &reversed,
                &universe,
                rule,
                (&one(y), &premise.0, &premise.1),
                (&one(y), &conclusion.0, &conclusion.1),
                &one(w),
                &none,
            ));
        }
        // Two inserted actions, one ancestral (`W1 -> Z`) and one not (`Z -> W2`): only
        // the ancestral one keeps its confounding edge, so inserting both is refused
        // while inserting the non-ancestral one alone is allowed.
        let mixed = admg(4, &[(0, 1), (1, 2)], &[(0, 3), (2, 3)]);
        let checker = Checker::new(&mixed).unwrap();
        let universe = Set::from([0, 1, 2, 3]);
        let holds = |inserted: Set| {
            unary_rule_holds(
                &checker,
                &universe,
                MixedRule::Rule3Insert,
                (&one(3), &none, &one(1)),
                (&one(3), &inserted, &one(1)),
                &inserted,
                &none,
            )
        };
        assert!(!holds(one(0)), "an ancestral action keeps its edges");
        assert!(!holds(both(0, 2)));
        assert!(holds(one(2)), "a non-ancestral action is cut from its edges");
    }

    /// The deletion rules are accepted by the checker but never attempted by the
    /// search, because they contribute nothing in this rule set: over random
    /// ADMGs and catalogs (single and double trials), every deletion the checker
    /// accepts on a quantity of the closure (its expression still free of the
    /// deleted variable) lands on a quantity the other rules already derived, and
    /// the closure never holds a quantity first derived by a deletion.
    #[test]
    fn deletions_are_checked_but_add_nothing_to_the_closure() {
        let ctx = ExecutionContext::for_tests(1);
        let mut state = 0xFACE_u64;
        let mut next = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 40) as usize
        };
        let (mut closures, mut accepted, mut quantities) = (0usize, 0usize, 0usize);
        for _ in 0..250 {
            let n = 4 + next() % 2;
            let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
            for a in 0..n {
                for b in a + 1..n {
                    match next() % 6 {
                        0 | 1 => directed.push((a, b)),
                        2 => bidirected.push((a, b)),
                        3 => {
                            directed.push((a, b));
                            bidirected.push((a, b));
                        }
                        _ => {}
                    }
                }
            }
            let graph =
                admg(u32::try_from(n).unwrap(), &u32_edges(&directed), &u32_edges(&bidirected));
            let mut inputs = Vec::new();
            for k in 0..2 + next() % 3 {
                let mut measured = (0..n).collect::<Vec<_>>();
                while measured.len() > 1 + next() % 3 {
                    measured.remove(next() % measured.len());
                }
                let mut on = Vec::new();
                let outside = (0..n).filter(|v| !measured.contains(v)).collect::<Vec<_>>();
                if next() % 2 == 0 && !outside.is_empty() {
                    on.push(outside[next() % outside.len()]);
                    let more =
                        outside.iter().copied().filter(|v| !on.contains(v)).collect::<Vec<_>>();
                    if next() % 3 == 0 && !more.is_empty() {
                        on.push(more[next() % more.len()]);
                    }
                }
                inputs.push(MixedInput {
                    regime: RegimeId::from_raw(u32::try_from(k).unwrap() + 1),
                    study: Arc::from(format!("s{k}")),
                    measured: measured.iter().map(|i| v_of(*i)).collect(),
                    intervened: on.iter().map(|i| v_of(*i)).collect(),
                    identity: format!("i{k}"),
                    snapshot: None,
                    marginals: Vec::new(),
                });
            }
            let variables = (0..n).map(v_of).collect::<Vec<_>>();
            let mut engine = Engine::new(&graph, variables, inputs, &query()).unwrap();
            engine.target = (0, 0, 0);
            let budget =
                SearchBudget::new(SearchLimits { operations: 40_000_000, depth: 64 }, &ctx)
                    .unwrap();
            let mut search = SharedSearch::new(budget);
            if !matches!(engine.run(&mut search), RunEnd::Fixpoint) {
                continue;
            }
            closures += 1;
            quantities += engine.steps.len();
            assert!(
                !engine.rule_counts.contains_key("rule1_delete")
                    && !engine.rule_counts.contains_key("rule3_delete")
            );
            let checker = Checker::new(&graph).unwrap();
            let universe = (0..n).collect::<Set>();
            let set = |mask: Mask| members(mask).collect::<Set>();
            for step in &engine.steps {
                for (rule, pool) in
                    [(MixedRule::Rule1Delete, step.z), (MixedRule::Rule3Delete, step.d)]
                {
                    for w in subsets(pool, MIXED_SOURCE_MAX_MOVE) {
                        let (cy, cd, cz) = conclusion_of(rule, (step.y, step.d, step.z), w);
                        if unary_rule_holds(
                            &checker,
                            &universe,
                            rule,
                            (&set(step.y), &set(step.d), &set(step.z)),
                            (&set(cy), &set(cd), &set(cz)),
                            &set(w),
                            &set(step.free),
                        ) {
                            accepted += 1;
                            assert!(
                                engine.known.contains_key(&(cy, cd, cz)),
                                "{directed:?} {bidirected:?}: a deletion derives a new quantity"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("{closures} closures, {quantities} quantities, {accepted} accepted deletions");
        assert!(closures > 200 && accepted > 20, "{closures} {accepted}");
    }

    /// Each field of a checked derivation is load-bearing.
    #[test]
    fn every_field_of_a_checked_derivation_is_load_bearing() {
        let graph = admg(3, &[(0, 1), (1, 2)], &[]);
        let c = catalog(&[(1, &[0, 1]), (2, &[1, 2])]);
        let q = MixedSourceQuery { outcomes: Arc::from([v(2)]), ..query() };
        let ctx = ExecutionContext::for_tests(1);
        let MixedSourceDecision::Identified { derivation, .. } =
            decide_mixed_source(&graph, &q, &c, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
        else {
            panic!("identified");
        };
        verify_mixed_source_derivation(&graph, &derivation, &c).unwrap();
        let bad = invalid_derivation();
        let mutate = |edit: &dyn Fn(&mut MixedSourceDerivation)| {
            let mut copy = (*derivation).clone();
            edit(&mut copy);
            verify_mixed_source_derivation(&graph, &copy, &c).unwrap_err()
        };
        let last = derivation.steps().len() - 1;
        assert_eq!(mutate(&|d| d.steps[last].premises = vec![0]), bad);
        assert_eq!(mutate(&|d| d.steps[last].params.clear()), bad);
        assert_eq!(mutate(&|d| d.steps[0].source.as_mut().unwrap().identity.push('x')), bad);
        assert_eq!(
            mutate(&|d| d.steps[0].source.as_mut().unwrap().regime = RegimeId::from_raw(2)),
            bad
        );
        assert_eq!(mutate(&|d| d.steps[0].relaxed = true), bad);
        assert_eq!(mutate(&|d| d.steps[last].intervened.clear()), bad);
        assert_eq!(mutate(&|d| d.steps[last].expression = ExprId::from_raw(0)), bad);
        assert_eq!(mutate(&|d| d.arena = CausalExprArena::new()), bad);
        assert_eq!(mutate(&|d| d.steps.truncate(last)), bad);
        assert_eq!(mutate(&|d| d.query.target = Arc::from("other")), bad);
        assert_eq!(mutate(&|d| d.graph_signature.push('x')), bad);
        // A rule the step's quantity does not follow from.
        assert_eq!(mutate(&|d| d.steps[last].rule = MixedRule::Rule3Delete), bad);
    }

    #[test]
    fn the_shared_expression_layer_compiles_the_root_and_binds_its_leaves() {
        let graph = admg(3, &[(0, 1), (1, 2)], &[]);
        let c = catalog(&[(1, &[0, 1]), (2, &[1, 2])]);
        let q = MixedSourceQuery { outcomes: Arc::from([v(2)]), ..query() };
        let ctx = ExecutionContext::for_tests(1);
        let MixedSourceDecision::Identified { derivation, .. } =
            decide_mixed_source(&graph, &q, &c, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap()
        else {
            panic!("identified");
        };
        let expressions = derivation.steps().iter().map(|step| step.expression).collect::<Vec<_>>();
        check_with_shared_expression_layer(&derivation.arena, &derivation.steps, &expressions)
            .unwrap();
        // A proof that cites a leaf the root expression does not read, or drops
        // one it reads, no longer binds.
        let mut extra = derivation.steps.clone();
        let input = extra.iter().position(|s| s.source.is_some()).unwrap();
        extra[input].source.as_mut().unwrap().regime = RegimeId::from_raw(2);
        assert_eq!(
            check_with_shared_expression_layer(&derivation.arena, &extra, &expressions)
                .unwrap_err(),
            invalid_derivation()
        );
        // A root outside the arena is refused by the shared compiler.
        let mut outside = expressions.clone();
        *outside.last_mut().unwrap() = ExprId::from_raw(u32::MAX);
        assert_eq!(
            check_with_shared_expression_layer(&derivation.arena, &derivation.steps, &outside)
                .unwrap_err(),
            invalid_derivation()
        );
    }

    /// Completeness against the complete ID algorithm (single observational study).
    mod reference {
        use super::*;
        use crate::{IdIdentifier, IdentificationWorkspace, result::IdentificationStatus};
        use antecedent_core::{
            CausalQuery, Intervention, InterventionalDistributionQuery, SearchBudget, Value,
        };
        use antecedent_expr::ExprNode;

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(super) enum Verdict {
            Identified,
            NotCertified,
            Exhausted,
        }

        /// Whether ID's formula contains a ratio (a conditional of a marginalised
        /// C-factor: the napkin family).
        fn has_ratio(result: &crate::IdentificationResult) -> bool {
            fn walk(arena: &antecedent_expr::CausalExprArena, id: ExprId) -> bool {
                match arena.node(id) {
                    ExprNode::Ratio { .. } => true,
                    ExprNode::Product(list) => arena.list(*list).iter().any(|&e| walk(arena, e)),
                    ExprNode::SumOut { expr, .. } => walk(arena, *expr),
                    _ => false,
                }
            }
            result.estimands.first().is_some_and(|e| walk(&result.arena, e.functional))
        }

        /// Shpitser-Pearl ID on `P(outcomes | do(treatments))`: `None` when not
        /// identifiable, else whether the formula needs a ratio.
        pub(super) fn id_identifiable(
            graph: &Admg,
            outcomes: &[u32],
            treatments: &[u32],
        ) -> Option<bool> {
            let id = IdIdentifier::new();
            let prepared = id.prepare(graph).unwrap();
            let query = CausalQuery::Distribution(
                InterventionalDistributionQuery::new(
                    v(outcomes[0]),
                    treatments
                        .iter()
                        .map(|t| Intervention::set(v(*t), Value::f64(1.0)))
                        .collect::<Vec<_>>(),
                )
                .with_outcomes(outcomes.iter().map(|o| v(*o)).collect::<Vec<_>>()),
            );
            let result =
                id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
            match result.status {
                IdentificationStatus::NonparametricallyIdentified => Some(has_ratio(&result)),
                IdentificationStatus::NotIdentified => None,
                other => panic!("unexpected ID status {other:?}"),
            }
        }

        /// The generic rule search alone over one observational study of every
        /// variable (the named routes are not run).
        pub(super) fn search_only(graph: &Admg, outcomes: &[u32], treatments: &[u32]) -> Verdict {
            let n = u32::try_from(graph.node_count()).unwrap();
            let all = (0..n).collect::<Vec<_>>();
            let evidence = catalog(&[(1, &all)]);
            let q = MixedSourceQuery {
                outcomes: outcomes.iter().map(|o| v(*o)).collect(),
                treatments: treatments.iter().map(|t| v(*t)).collect(),
                target: Arc::from("target"),
                sources: Arc::from([]),
            };
            let validated = validate_mixed_source_query(graph, &q, &evidence).unwrap();
            let ctx = ExecutionContext::for_tests(1);
            let budget = SearchBudget::new(MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap();
            match rule_search(
                graph,
                &validated,
                &evidence,
                &mut SharedSearch::new(budget),
                Vec::new(),
            )
            .unwrap()
            {
                MixedSourceDecision::Identified { .. } => Verdict::Identified,
                MixedSourceDecision::NotCertified(_) => Verdict::NotCertified,
                MixedSourceDecision::Exhausted(_) => Verdict::Exhausted,
                other => panic!("unexpected {other:?}"),
            }
        }

        type Edges = Vec<(u32, u32)>;

        /// Outcome counts of a sweep, by query shape `(|Y|, |X|)`.
        #[derive(Default, Debug)]
        pub(super) struct Tally {
            pub(super) queries: usize,
            /// ID-identifiable queries whose ID formula is a ratio, and how many the search identified.
            pub(super) ratio: [usize; 2],
            /// Per shape: (ID-identifiable, identified, not certified, exhausted, non-identifiable).
            pub(super) shapes: BTreeMap<(usize, usize), [usize; 5]>,
            /// Search identified a query ID refutes (must stay empty).
            pub(super) unsound: Vec<String>,
            /// ID-identifiable queries the search did not identify: (edge count, |X|, description).
            pub(super) gaps: Vec<(usize, usize, String)>,
        }

        impl Tally {
            pub(super) fn totals(&self) -> [usize; 5] {
                let mut t = [0; 5];
                for c in self.shapes.values() {
                    for (a, b) in t.iter_mut().zip(c) {
                        *a += b;
                    }
                }
                t
            }
        }

        /// Run both procedures on one query and record the outcome.
        pub(super) fn record(
            tally: &mut Tally,
            graph: &Admg,
            d: &[(u32, u32)],
            b: &[(u32, u32)],
            ys: &[u32],
            xs: &[u32],
        ) {
            let ratio = id_identifiable(graph, ys, xs);
            let verdict = search_only(graph, ys, xs);
            let cell = tally.shapes.entry((ys.len(), xs.len())).or_default();
            tally.queries += 1;
            let text = format!("directed={d:?} bidirected={b:?} y={ys:?} do={xs:?}");
            let Some(ratio) = ratio else {
                cell[4] += 1;
                if verdict == Verdict::Identified {
                    tally.unsound.push(text);
                }
                return;
            };
            cell[0] += 1;
            if ratio {
                tally.ratio[0] += 1;
                tally.ratio[1] += usize::from(verdict == Verdict::Identified);
            }
            match verdict {
                Verdict::Identified => cell[1] += 1,
                Verdict::NotCertified => cell[2] += 1,
                Verdict::Exhausted => cell[3] += 1,
            }
            if verdict != Verdict::Identified {
                tally.gaps.push((d.len() + b.len(), xs.len(), format!("{verdict:?} {text}")));
            }
        }

        /// Every non-empty disjoint (outcomes, treatments) pair over `n` nodes.
        pub(super) fn all_queries(n: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
            let mut out = Vec::new();
            for code in 0..3usize.pow(u32::try_from(n).unwrap()) {
                let (mut ys, mut xs) = (Vec::new(), Vec::new());
                let mut rest = code;
                for i in 0..n {
                    match rest % 3 {
                        1 => ys.push(u32::try_from(i).unwrap()),
                        2 => xs.push(u32::try_from(i).unwrap()),
                        _ => {}
                    }
                    rest /= 3;
                }
                if !ys.is_empty() && !xs.is_empty() {
                    out.push((ys, xs));
                }
            }
            out
        }

        /// The ADMG of `code` over the `n(n-1)/2` ordered pairs: two bits per pair
        /// (bit 0 directed `a -> b`, bit 1 bidirected `a <-> b`), so `code`
        /// ranges over every ADMG up to relabelling.
        pub(super) fn graph_of(n: usize, code: usize) -> (Admg, Edges, Edges) {
            let (mut d, mut b) = (Vec::new(), Vec::new());
            let mut k = 0;
            for x in 0..u32::try_from(n).unwrap() {
                for y in x + 1..u32::try_from(n).unwrap() {
                    if (code >> (2 * k)) & 1 == 1 {
                        d.push((x, y));
                    }
                    if (code >> (2 * k)) & 2 == 2 {
                        b.push((x, y));
                    }
                    k += 1;
                }
            }
            (admg(u32::try_from(n).unwrap(), &d, &b), d, b)
        }

        pub(super) fn pairs(n: usize) -> u32 {
            u32::try_from(n * (n - 1) / 2).unwrap()
        }

        pub(super) fn report(name: &str, tally: &Tally) {
            let [idn, found, nc, ex, non] = tally.totals();
            eprintln!(
                "{name}: queries={} id_identifiable={idn} identified={found} not_certified={nc} exhausted={ex} id_not_identifiable={non} ratio_form={:?} unsound={}",
                tally.queries,
                tally.ratio,
                tally.unsound.len()
            );
            for ((y, x), c) in &tally.shapes {
                eprintln!(
                    "  |Y|={y} |X|={x}: id={} found={} nc={} ex={} non={}",
                    c[0], c[1], c[2], c[3], c[4]
                );
            }
        }

        /// Seeded generator of the sampled sweeps (a 64-bit LCG; deterministic).
        pub(super) struct Lcg(pub(super) u64);

        impl Lcg {
            pub(super) fn next(&mut self) -> u64 {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                self.0 >> 33
            }
        }

        /// `graphs` random ADMGs over `n` nodes (edge density varied per graph so the
        /// sample holds sparse and dense graphs), `per_graph` random queries each.
        pub(super) fn sample(n: usize, graphs: usize, per_graph: usize, seed: u64) -> Tally {
            let mut rng = Lcg(seed);
            let mut tally = Tally::default();
            let queries = all_queries(n);
            let nodes = u32::try_from(n).unwrap();
            for _ in 0..graphs {
                let density = 1 + rng.next() % 3;
                let (mut d, mut b) = (Vec::new(), Vec::new());
                for x in 0..nodes {
                    for y in x + 1..nodes {
                        if rng.next() % 6 < density + 1 {
                            d.push((x, y));
                        }
                        if rng.next() % 6 < density {
                            b.push((x, y));
                        }
                    }
                }
                let g = admg(nodes, &d, &b);
                for _ in 0..per_graph {
                    let (ys, xs) = &queries[usize::try_from(rng.next()).unwrap() % queries.len()];
                    record(&mut tally, &g, &d, &b, ys, xs);
                }
            }
            tally
        }

        /// Figures behind the documented completeness rates. Run with
        /// `N=<nodes> CASES=<graphs> cargo test -p antecedent-identify --lib --release
        /// measure -- --ignored --nocapture`; `CASES=0` is the exhaustive sweep over every
        /// ADMG (n=3: 768 queries; n=4: 204,800 queries, about 80 s in release).
        #[test]
        #[ignore = "measurement: minutes in release"]
        fn measure() {
            let var = |name: &str, default: usize| {
                std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
            };
            let (n, cases) = (var("N", 3), var("CASES", 0));
            let start = std::time::Instant::now();
            let tally = if cases == 0 {
                let mut tally = Tally::default();
                let queries = all_queries(n);
                for code in 0..4usize.pow(pairs(n)) {
                    let (g, d, b) = graph_of(n, code);
                    for (ys, xs) in &queries {
                        record(&mut tally, &g, &d, &b, ys, xs);
                    }
                }
                tally
            } else {
                sample(n, cases, 3, 0xC0FF_EE00 + n as u64)
            };
            report(&format!("n={n} cases={cases} t={:?}", start.elapsed()), &tally);
            let mut gaps = tally.gaps.clone();
            gaps.sort();
            for (_, _, g) in gaps.iter().filter(|g| g.1 <= MIXED_SOURCE_MAX_DO).take(40) {
                eprintln!("  GAP {g}");
            }
            for u in &tally.unsound {
                eprintln!("  UNSOUND {u}");
            }
        }

        /// Every ADMG on three nodes, every query: the rule search identifies
        /// exactly what ID identifies (612 of 768 queries), and nothing ID refutes.
        #[test]
        fn three_node_sweep_matches_id_exactly() {
            let mut tally = Tally::default();
            let queries = all_queries(3);
            for code in 0..4usize.pow(pairs(3)) {
                let (g, d, b) = graph_of(3, code);
                for (ys, xs) in &queries {
                    record(&mut tally, &g, &d, &b, ys, xs);
                }
            }
            report("three-node sweep", &tally);
            assert!(tally.unsound.is_empty(), "{:?}", tally.unsound);
            assert_eq!(tally.queries, 64 * 12);
            // [ID-identifiable, identified, not certified, exhausted, not identifiable]
            assert_eq!(tally.totals(), [612, 612, 0, 0, 156]);
        }

        /// Four- and five-node samples: never identified where ID refutes; every
        /// ID-identifiable query with at most `MIXED_SOURCE_MAX_DO` treatments is
        /// identified except the pinned napkin-family gaps; a larger treatment set is
        /// `NotCertified`, never exhausted. The rate floors are the measured rates
        /// (exhaustive four-node sweep 142,826 of 142,827; five-node sample 97.3%)
        /// rounded down.
        #[test]
        fn four_and_five_node_samples_are_sound_and_meet_the_completeness_floor() {
            let four = {
                // A stride coprime to 4^6 visits 57 graphs spread over the whole code space.
                let mut tally = Tally::default();
                let queries = all_queries(4);
                for code in (0..4usize.pow(pairs(4))).step_by(73) {
                    let (g, d, b) = graph_of(4, code);
                    for (ys, xs) in &queries {
                        record(&mut tally, &g, &d, &b, ys, xs);
                    }
                }
                tally
            };
            let five = sample(5, 100, 3, 0x5EED_0005);
            for (name, tally) in [("four-node stride sample", &four), ("five-node sample", &five)] {
                report(name, tally);
                assert!(tally.unsound.is_empty(), "{name}: {:?}", tally.unsound);
                let [identifiable, identified, _, exhausted, _] = tally.totals();
                assert_eq!(exhausted, 0, "{name}: budget never runs out up to five nodes");
                let within_do = tally.gaps.iter().filter(|g| g.1 <= MIXED_SOURCE_MAX_DO).count();
                let beyond_do = tally.gaps.iter().filter(|g| g.1 > MIXED_SOURCE_MAX_DO).count();
                assert_eq!(identifiable - identified, within_do + beyond_do);
                // Gaps within the intervention bound are the napkin family: rare.
                assert!(within_do <= 1, "{name}: {:?}", tally.gaps);
                assert!(
                    beyond_do == 0 || tally.gaps.iter().all(|g| g.2.starts_with("NotCertified"))
                );
            }
            let [identifiable, identified, ..] = four.totals();
            assert!(identifiable > 1_500, "{identifiable}");
            assert!(identified * 1000 >= identifiable * 999, "{identified}/{identifiable}");
            let [identifiable, identified, ..] = five.totals();
            assert!(identifiable > 150, "{identifiable}");
            assert!(identified * 100 >= identifiable * 95, "{identified}/{identifiable}");
        }

        /// One known-incomplete query: ID identifies it, the rule search does not.
        struct Gap {
            name: &'static str,
            nodes: u32,
            directed: &'static [(u32, u32)],
            bidirected: &'static [(u32, u32)],
            outcomes: &'static [u32],
            treatments: &'static [u32],
            verdict: Verdict,
        }

        /// The known gaps of the rule search against ID, one minimal example per class.
        /// A future improvement flips an entry deliberately: update the entry, the
        /// class list in the module docs and the record's `inference_notes` together.
        const KNOWN_GAPS: &[Gap] = &[
            // Class 1, the napkin: `W -> Z -> X -> Y` with `W <-> X` and `W <-> Y`.
            // ID identifies it through a ratio of a marginalised C-factor
            // (`P(y | do(x)) = sum_w P(y, x | z, w) P(w) / sum_w P(x | z, w) P(w)`),
            // a form the frozen rules reach from the joint only by a chain the
            // search's two-variable moves and forward closure do not find.
            Gap {
                name: "napkin",
                nodes: 4,
                directed: &[(0, 1), (1, 2), (2, 3)],
                bidirected: &[(0, 2), (0, 3)],
                outcomes: &[3],
                treatments: &[2],
                verdict: Verdict::NotCertified,
            },
            // Class 1 in a five-node graph (the only such gap in the seeded sample).
            Gap {
                name: "napkin_family_five_nodes",
                nodes: 5,
                directed: &[(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)],
                bidirected: &[(0, 2), (0, 4)],
                outcomes: &[4],
                treatments: &[2, 3],
                verdict: Verdict::NotCertified,
            },
            // Class 1 with two outcomes: the napkin `0 -> 1 -> 2 -> 4`, `0 <-> 2`, `0 <-> 4`
            // plus a second outcome `3` confounded with `4`.
            Gap {
                name: "napkin_family_two_outcomes",
                nodes: 5,
                directed: &[(0, 1), (1, 2), (2, 4), (3, 4)],
                bidirected: &[(0, 2), (0, 4), (3, 4)],
                outcomes: &[3, 4],
                treatments: &[2],
                verdict: Verdict::NotCertified,
            },
            // Class 2, the intervention bound: a derived quantity carries at most
            // `MIXED_SOURCE_MAX_DO` = 3 actions, so a fourth treatment is unreachable
            // even when the effect is trivially `P(y)`.
            Gap {
                name: "four_treatments_edgeless",
                nodes: 5,
                directed: &[],
                bidirected: &[],
                outcomes: &[0],
                treatments: &[1, 2, 3, 4],
                verdict: Verdict::NotCertified,
            },
            Gap {
                name: "four_treatments_confounded_chain",
                nodes: 5,
                directed: &[(0, 1), (1, 2)],
                bidirected: &[(1, 3)],
                outcomes: &[4],
                treatments: &[0, 1, 2, 3],
                verdict: Verdict::NotCertified,
            },
            // Class 3, the budget: with six nodes the default 20,000 operations run out
            // before the closure does, even for a trivial effect. A receipt, not a verdict.
            Gap {
                name: "six_nodes_exhaust_the_default_budget",
                nodes: 6,
                directed: &[(2, 3)],
                bidirected: &[],
                outcomes: &[5],
                treatments: &[0, 1, 2],
                verdict: Verdict::Exhausted,
            },
        ];

        #[test]
        fn known_incomplete_queries_stay_not_certified_until_deliberately_fixed() {
            for gap in KNOWN_GAPS {
                let g = admg(gap.nodes, gap.directed, gap.bidirected);
                assert!(
                    id_identifiable(&g, gap.outcomes, gap.treatments).is_some(),
                    "{}: ID must identify the pinned gap",
                    gap.name
                );
                assert_eq!(
                    search_only(&g, gap.outcomes, gap.treatments),
                    gap.verdict,
                    "{}: the rule search's verdict on a known gap changed",
                    gap.name
                );
            }
            // The one-treatment-fewer twins are identified: the bound is the whole reason.
            let three = admg(4, &[], &[]);
            assert_eq!(search_only(&three, &[0], &[1, 2, 3]), Verdict::Identified);
            let chain = admg(5, &[(0, 1), (1, 2)], &[(1, 3)]);
            assert_eq!(search_only(&chain, &[4], &[0, 1, 2]), Verdict::Identified);
        }
    }
}
