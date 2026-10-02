//! The bounded multi-source limited-experiment (mz-transportability) contract.
//!
//! The theorem is `TR^mz` (Bareinboim, Lee, Honavar & Pearl, `NeurIPS` 2013), whose
//! completeness Bareinboim & Pearl prove in "Transportability from Multiple
//! Environments with Limited Experiments: Completeness Results" (`NeurIPS` 2014,
//! UCLA R-443, Theorems 4–5). It is `TR^z` with a loop over source domains at
//! line 10: once a subcall exchanges into one source's experiment, the rest of
//! that subcall stays in that source. Separate c-factors of the line-4
//! factorization may therefore come from different sources; no single c-factor
//! ever mixes sources, so a joint over several sources' interventions is never
//! fabricated.
//!
//! Checked against R-443: Theorem 4 (a `TR^mz` failure at line 12 yields
//! C-forests spanning an mz*-shedge), Theorem 5 (`TR^mz` is complete) and
//! Corollary 1 (`P_x(y)` is mz-transportable iff no mz*-shedge exists), with
//! Theorem 3 (a shedge precludes transportability). Completeness holds for the
//! paper's own information family (Def. 2): for every source, experiments on
//! every subset of its declared controllable set with all observed variables
//! measured, a passive target. A formula bound to a supplied catalog is sound
//! but incomplete; a failure is an obstruction only when it is evaluated
//! structurally over the declared controllable sets, never because a catalog
//! lacks a regime. The obstruction here is a strict subset of the paper's
//! FAILs: only a line-11 terminal reached with no active experiment where no
//! source can exchange. A failure after an exchange (e.g. R-443 Fig. 1(e,f)
//! split across two sources, which the paper calls not transportable) is
//! reported `not_certified`. Fig. 3 line 10 fires only with no active
//! experiment (one exchange per branch). The recursion also permits the active
//! source to exchange its remaining controllables again, but that path is
//! structurally unreachable: at a line-10 state `V \ X` is one c-component inside
//! `An(Y)` in `D_Xbar`, an exchange removes only treatments, so lines 3 and 4
//! cannot later add a controllable of the active source to `X`. The search
//! therefore equals the literal one-exchange `TR^mz` (tested differentially in
//! `tests/mz_transport_paper_reference.rs`), and a cited joint regime is always
//! the one joint of a single exchange. The obstruction is unaffected either way
//! because it is certified only with no active experiment. When several sources certify the same factor, the paper
//! returns a weighted combination; for exact laws each is a valid formula, so
//! this contract returns the first certifying source in canonical source order
//! and does not retain the others, which makes the result invariant to source
//! declaration order.
//!
//! Every stage of one decision charges a single [`SearchBudget`], so the
//! declared limits bound the whole decision, not each stage. The limits are
//! maxima as well as defaults, and each contract bound refuses by the one
//! detail `mz_transport.bounds_exceeded`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    DEFAULT_SEARCH_MEMORY_BYTES, EvidenceCatalog, ExecutionContext, RegimeId, RegimeKind,
    SearchBudget, SearchLimits, SearchReceipt, VariableId,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::SelectionDiagram;

use super::z_transport::{
    TrzCall, TrzDomain, TrzTerminalFailure, Z_TRANSPORT_MAX_CONTROLLABLE, Z_TRANSPORT_MAX_OBSERVED,
    ZTransportDecision, ZTransportDerivation, ZTransportLimitsReceipt, ZTransportMissingEvidence,
    ZTransportQuery, ZTransportSourceSpec, bind_recursive_expression, bind_z_transport_catalog,
    cited_regimes, decide_z_transport_reporting, search_trz_call, validate_z_transport_query,
};
use super::{IdentificationError, SearchCharge, SharedSearch, SidLimits, SidMeter};

/// Observed variables in the shared graph.
pub const MZ_TRANSPORT_MAX_OBSERVED: usize = Z_TRANSPORT_MAX_OBSERVED;
/// Source populations in one query. One source is the single-source z route.
pub const MZ_TRANSPORT_MAX_SOURCES: usize = 4;
/// Controllable variables declared per source.
pub const MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE: usize = Z_TRANSPORT_MAX_CONTROLLABLE;
/// Available source regimes the search may consider as candidate evidence.
pub const MZ_TRANSPORT_MAX_CANDIDATE_REGIMES: usize = 64;
/// Search limits for one whole decision: states charged across every stage, and
/// recursion depth. They are the defaults and also the maxima; larger limits
/// refuse. Memory and cancellation come from the execution context on every charge.
pub const MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits { operations: 4096, depth: 24 };
/// Memory cap (estimated live-state bytes, cumulative across every stage of one
/// decision) of the search: the default and the maximum. The effective cap is the
/// smaller of this and the execution context's hard limit, so it is never absent.
pub const MZ_TRANSPORT_MEMORY_BYTES: u64 = DEFAULT_SEARCH_MEMORY_BYTES;

/// A request outside a declared bound: source count, observed variables,
/// controllables per source, candidate regimes, or search limits.
const BOUNDS_EXCEEDED: &str = "mz_transport.bounds_exceeded";
/// The bounded search ended without a formula or a checked obstruction.
const SEARCH_INCOMPLETE: &str = "mz_transport.search_incomplete";
/// Completing a factor needs a joint over two sources' interventions.
const FABRICATED_JOINT: &str = "mz_transport.fabricated_joint";
/// A certified formula cites a joint regime (or level) the catalog lacks.
const MISSING_JOINT_REGIME: &str = "mz_transport.missing_joint_regime";
/// A checked structural obstruction for the declared controllable sets.
const CHECKED_OBSTRUCTION: &str = "mz_transport.checked_obstruction";
/// A limit or cancellation stopped the search.
const BUDGET: &str = "mz_transport.budget";
/// A derivation or its record does not check.
const INVALID_DERIVATION: &str = "mz_transport.invalid_derivation";
/// An obstruction does not check.
const INVALID_OBSTRUCTION: &str = "mz_transport.invalid_obstruction";

/// The frozen `(reason code, mz_transport.* detail)` pair of an error raised by
/// the mz-transport route, as the X1 promotion record declares them:
/// `route_not_supported` for a bound, `invalid_argument` for an invalid query or
/// catalog, `transport_not_certified` for a derivation, leaf or obstruction that
/// does not check, `transport_missing_evidence` for an unbound cited regime and
/// `transport_budget_cancel` for a stop. `None` for an error the route does not own.
#[must_use]
#[doc(hidden)]
pub fn mz_transport_refusal(error: &IdentificationError) -> Option<(&'static str, &'static str)> {
    use antecedent_core::reason_code;
    match error {
        IdentificationError::UnsupportedInput { code } if *code == BOUNDS_EXCEEDED => {
            Some((reason_code!("route_not_supported"), BOUNDS_EXCEEDED))
        }
        IdentificationError::InvalidInput { message }
            if message.starts_with("mz_transport.invalid_query") =>
        {
            Some((reason_code!("invalid_argument"), "mz_transport.invalid_query"))
        }
        IdentificationError::InvalidCatalog { message }
            if message.starts_with("mz_transport.invalid_catalog") =>
        {
            Some((reason_code!("invalid_argument"), "mz_transport.invalid_catalog"))
        }
        IdentificationError::InvalidDerivation { code }
            if [INVALID_DERIVATION, INVALID_OBSTRUCTION, FABRICATED_JOINT].contains(code) =>
        {
            Some((reason_code!("transport_not_certified"), code))
        }
        IdentificationError::MissingEvidence { .. } => {
            Some((reason_code!("transport_missing_evidence"), MISSING_JOINT_REGIME))
        }
        IdentificationError::Cancelled | IdentificationError::Budget { .. } => {
            Some((reason_code!("transport_budget_cancel"), BUDGET))
        }
        _ => None,
    }
}

const fn bounds_exceeded() -> IdentificationError {
    IdentificationError::UnsupportedInput { code: BOUNDS_EXCEEDED }
}

/// One target interventional query answered from several limited-experiment sources.
///
/// Each source keeps its own population, selection targets on the shared causal
/// graph, declared controllable set (theoretical experimental availability) and
/// concrete experiment assignment. The regimes actually supplied live in the
/// evidence catalog; declaring a controllable set never implies results exist.
#[derive(Clone, Debug, PartialEq)]
pub struct MzTransportQuery {
    /// Joint outcomes in the target population.
    pub outcomes: Arc<[VariableId]>,
    /// Treatments whose target interventional response is queried.
    pub treatments: Arc<[VariableId]>,
    /// Target population; it supplies observational evidence only.
    pub target: Arc<str>,
    /// Two to [`MZ_TRANSPORT_MAX_SOURCES`] sources.
    pub sources: Arc<[ZTransportSourceSpec]>,
}

impl MzTransportQuery {
    /// Canonical form: sources ordered by population identity and every
    /// variable set sorted. Two queries that differ only in declaration order
    /// have equal canonical forms.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let sorted = |variables: &[VariableId]| {
            let mut out = variables.to_vec();
            out.sort_unstable();
            Arc::<[VariableId]>::from(out)
        };
        let mut sources = self
            .sources
            .iter()
            .map(|source| {
                let mut assignment = source.experiment_assignment.to_vec();
                assignment.sort_by_key(|a| a.variable);
                ZTransportSourceSpec {
                    population: Arc::clone(&source.population),
                    controllable: sorted(&source.controllable),
                    experiment_assignment: assignment.into(),
                    selection_targets: sorted(&source.selection_targets),
                }
            })
            .collect::<Vec<_>>();
        sources.sort_by(|a, b| a.population.cmp(&b.population));
        Self {
            outcomes: sorted(&self.outcomes),
            treatments: sorted(&self.treatments),
            target: Arc::clone(&self.target),
            sources: sources.into(),
        }
    }

    /// The single-source z-transport query of one source, sharing the target question.
    #[must_use]
    pub fn source_query(&self, source: &ZTransportSourceSpec) -> ZTransportQuery {
        ZTransportQuery {
            outcomes: Arc::clone(&self.outcomes),
            treatments: Arc::clone(&self.treatments),
            controllable: Arc::clone(&source.controllable),
            experiment_assignment: Arc::clone(&source.experiment_assignment),
            source: Arc::clone(&source.population),
            target: Arc::clone(&self.target),
        }
    }
}

/// A query that passed the contract: canonical, with one selection diagram per
/// source in canonical source order.
#[derive(Clone, Debug)]
pub struct ValidatedMzTransportQuery {
    /// Canonical query.
    pub query: MzTransportQuery,
    /// Each source's selection diagram, aligned with `query.sources`.
    pub diagrams: Vec<SelectionDiagram>,
    /// Available source regimes the search may cite, counted against
    /// [`MZ_TRANSPORT_MAX_CANDIDATE_REGIMES`].
    pub candidate_regimes: usize,
}

/// Validate a multi-source query against the shared graph and supplied catalog.
///
/// Bounds: two to [`MZ_TRANSPORT_MAX_SOURCES`] sources (one source is the
/// single-source z route), at most [`MZ_TRANSPORT_MAX_OBSERVED`] observed
/// variables, one to [`MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE`] controllables
/// per source, and at most [`MZ_TRANSPORT_MAX_CANDIDATE_REGIMES`] available
/// source regimes. Each source must otherwise pass the single-source
/// z-transport contract on its own selection diagram; populations are distinct
/// and none is the target; every available source experiment intervenes only
/// on a subset of that source's declared controllable set; and the target
/// supplies no experiments.
///
/// # Errors
///
/// [`IdentificationError::UnsupportedInput`] `mz_transport.bounds_exceeded` for
/// any exceeded bound; [`IdentificationError::InvalidInput`]
/// `mz_transport.invalid_query` for a malformed query; and
/// [`IdentificationError::InvalidCatalog`] `mz_transport.invalid_catalog` for a
/// catalog that contradicts the declarations. The message names the violation.
#[doc(hidden)]
pub fn validate_mz_transport_query(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<ValidatedMzTransportQuery, IdentificationError> {
    let query = query.canonical();
    if !(2..=MZ_TRANSPORT_MAX_SOURCES).contains(&query.sources.len())
        || graph.node_count() > MZ_TRANSPORT_MAX_OBSERVED
        || query.sources.iter().any(|s| {
            s.controllable.is_empty()
                || s.controllable.len() > MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE
        })
    {
        return Err(bounds_exceeded());
    }
    let invalid_query = |detail: &dyn std::fmt::Display| {
        IdentificationError::invalid_input(format!("mz_transport.invalid_query: {detail}"))
    };
    let populations = query.sources.iter().map(|s| s.population.as_ref()).collect::<BTreeSet<_>>();
    if populations.len() != query.sources.len() || populations.contains(query.target.as_ref()) {
        return Err(invalid_query(&"source populations must be distinct and not the target"));
    }
    let mut diagrams = Vec::with_capacity(query.sources.len());
    for source in query.sources.iter() {
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                .map_err(|error| invalid_query(&error))?;
        validate_z_transport_query(&diagram, &query.source_query(source)).map_err(|error| {
            match error {
                IdentificationError::UnsupportedInput { .. } => bounds_exceeded(),
                other => invalid_query(&format_args!("source {}: {other}", source.population)),
            }
        })?;
        diagrams.push(diagram);
    }
    let invalid_catalog = |detail: String| {
        IdentificationError::invalid_catalog(format!("mz_transport.invalid_catalog: {detail}"))
    };
    let mut candidate_regimes = 0usize;
    for regime in catalog.regimes.iter().filter(|r| r.supplies_population_law()) {
        if regime.population == query.target {
            if regime.kind == RegimeKind::Experimental {
                return Err(invalid_catalog(format!(
                    "target regime {} is experimental; the target supplies observational evidence only",
                    regime.id.raw()
                )));
            }
            continue;
        }
        let Some(source) = query.sources.iter().find(|s| s.population == regime.population) else {
            continue;
        };
        if regime.interventions.iter().any(|v| !source.controllable.contains(v)) {
            return Err(invalid_catalog(format!(
                "regime {} of source {} intervenes outside its declared controllable set",
                regime.id.raw(),
                source.population
            )));
        }
        candidate_regimes += 1;
    }
    if candidate_regimes > MZ_TRANSPORT_MAX_CANDIDATE_REGIMES {
        return Err(bounds_exceeded());
    }
    Ok(ValidatedMzTransportQuery { query, diagrams, candidate_regimes })
}

/// Which search stage identified the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MzTransportRoute {
    /// The target's observational law alone identifies the effect.
    TargetOnly,
    /// One source's single-source z route identifies it.
    SingleSource {
        /// The identifying source.
        population: Arc<str>,
    },
    /// The combined search, where each line-10 exchange may use any source.
    /// Factors certified by different sources combine through the c-component
    /// factorization; each factor comes from exactly one source.
    Combined {
        /// Sources the formula cites, in canonical order.
        populations: Arc<[Arc<str>]>,
    },
}

/// A checked positive derivation of an mz-transport query.
#[derive(Clone, Debug)]
pub struct MzTransportDerivation {
    query: MzTransportQuery,
    graph_signature: String,
    route: MzTransportRoute,
    arena: CausalExprArena,
    root: ExprId,
    rules: Vec<String>,
    single: Option<Box<ZTransportDerivation>>,
    /// Stages evaluated before the identifying one, in order.
    stages: Vec<MzStageRecord>,
    /// The success-path search receipt of the decision that produced it.
    search: MzSearchRecord,
}

/// The search receipt of a decision that identified: the limits it ran under
/// and what it consumed. Operations and depth are deterministic for a given
/// graph, query, catalog and limits, so a consumer replays the decision under
/// these stored limits (never above its own maxima) and compares the record
/// exactly.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MzSearchRecord {
    /// Operation limit in force.
    pub operations_limit: usize,
    /// Depth limit in force.
    pub depth_limit: usize,
    /// Effective memory cap in force, in bytes; never absent.
    pub memory_limit_bytes: u64,
    /// Operations this decision charged, across every stage it ran.
    pub operations_consumed: usize,
    /// Deepest recursion level any stage of this decision reached.
    pub depth_reached: usize,
    /// `stage:<name>` of every stage run, up to and including the identifying one.
    pub explored: Vec<String>,
    /// `stage:<name>` of every later stage the identification made unnecessary.
    pub unevaluated: Vec<String>,
}

/// Portable premises of an mz derivation. The expression arena travels
/// separately; [`MzTransportDerivation::from_record_checked`] re-decides the
/// query against its catalog and accepts only an identical derivation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MzTransportDerivationRecord {
    /// Shared causal-graph identity.
    pub graph_signature: String,
    /// `target_only`, `single_source`, or `combined`.
    pub route: String,
    /// The identifying source (`single_source`) or cited sources (`combined`).
    pub populations: Vec<String>,
    /// Root expression id in the accompanying arena.
    pub root: u32,
    /// Checked rule trace.
    pub rules: Vec<String>,
    /// `(stage, outcome)` of every stage evaluated before identification.
    pub stages: Vec<(String, String)>,
    /// Search limits in force, what was consumed and the stages explored.
    pub search: MzSearchRecord,
}

impl MzTransportDerivation {
    /// Canonical query this derivation answers.
    #[must_use]
    pub const fn query(&self) -> &MzTransportQuery {
        &self.query
    }
    /// Stage that identified the target.
    #[must_use]
    pub const fn route(&self) -> &MzTransportRoute {
        &self.route
    }
    /// Population- and regime-tagged formula arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Formula root.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
    /// Graph identity the derivation was checked against.
    #[must_use]
    pub fn graph_signature(&self) -> &str {
        &self.graph_signature
    }
    /// Checked rule trace, including explored branches, in order.
    #[must_use]
    pub fn rules(&self) -> &[String] {
        &self.rules
    }
    /// The single-source derivation when the route is [`MzTransportRoute::SingleSource`].
    #[must_use]
    pub fn single_source(&self) -> Option<&ZTransportDerivation> {
        self.single.as_deref()
    }
    /// Every stage evaluated before the identifying one.
    #[must_use]
    pub fn stages(&self) -> &[MzStageRecord] {
        &self.stages
    }
    /// Success-path search receipt: limits in force, operations and depth
    /// consumed, and the stages explored and left unevaluated.
    #[must_use]
    pub const fn search_record(&self) -> &MzSearchRecord {
        &self.search
    }

    /// Export the proof premises for an independently checked artifact.
    #[must_use]
    pub fn to_record(&self) -> MzTransportDerivationRecord {
        let (route, populations) = match &self.route {
            MzTransportRoute::TargetOnly => ("target_only", Vec::new()),
            MzTransportRoute::SingleSource { population } => {
                ("single_source", vec![population.to_string()])
            }
            MzTransportRoute::Combined { populations } => {
                ("combined", populations.iter().map(ToString::to_string).collect())
            }
        };
        MzTransportDerivationRecord {
            graph_signature: self.graph_signature.clone(),
            route: route.into(),
            populations,
            root: self.root.raw(),
            rules: self.rules.clone(),
            stages: self.stages.iter().map(|s| (s.stage.clone(), s.outcome.to_owned())).collect(),
            search: self.search.clone(),
        }
    }

    /// Reconstruct a derivation from an untrusted record: the bounded decision is
    /// re-run on `graph`, `query` and `catalog` under the limits the record
    /// stores, which must lie within the caller's `limits` and memory cap (the
    /// smaller of [`MZ_TRANSPORT_MEMORY_BYTES`] and the context's hard limit);
    /// the record, including its search receipt, and the arena must equal the
    /// derivation the replay produces.
    ///
    /// # Errors
    /// `mz_transport.bounds_exceeded` when the stored limits exceed the caller's;
    /// the decision does not identify; or any recorded premise, receipt field or
    /// expression differs.
    pub fn from_record_checked(
        graph: &antecedent_graph::Admg,
        query: &MzTransportQuery,
        catalog: &EvidenceCatalog,
        record: &MzTransportDerivationRecord,
        arena: &CausalExprArena,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let stored = SearchLimits {
            operations: record.search.operations_limit,
            depth: record.search.depth_limit,
        };
        let memory_cap = ctx
            .memory
            .hard_limit_bytes
            .map_or(MZ_TRANSPORT_MEMORY_BYTES, |h| h.min(MZ_TRANSPORT_MEMORY_BYTES));
        if stored.operations > limits.operations
            || stored.depth > limits.depth
            || record.search.memory_limit_bytes > memory_cap
        {
            return Err(bounds_exceeded());
        }
        let MzTransportDecision::Identified { derivation, .. } =
            decide_bounded(graph, query, catalog, stored, record.search.memory_limit_bytes, ctx)?
        else {
            return Err(IdentificationError::invalid_derivation(INVALID_DERIVATION));
        };
        if derivation.to_record() != *record || derivation.arena != *arena {
            return Err(IdentificationError::invalid_derivation(INVALID_DERIVATION));
        }
        Ok(*derivation)
    }
}

/// A checked mz obstruction: the search reached a line-11 state with no active
/// experiment, so every search path reaches it, and no source could exchange
/// there. By `TR^mz` completeness (R-443, Theorems 4–5) the declared
/// controllable sets cannot transport the query; no catalog is consulted.
#[derive(Clone, Debug)]
pub struct MzTransportObstruction {
    query: MzTransportQuery,
    graph_signature: String,
    terminal: TrzTerminalFailure,
}

impl MzTransportObstruction {
    /// Canonical query the obstruction is about.
    #[must_use]
    pub const fn query(&self) -> &MzTransportQuery {
        &self.query
    }
    /// Variables of the failing c-component `C0`.
    #[must_use]
    pub fn c0(&self) -> &[u32] {
        &self.terminal.c0
    }
    /// Each source's terminal premises: its remaining controllables, the ones
    /// active at the terminal, and whether its selection nodes were separated.
    #[must_use]
    pub fn sources(&self) -> Vec<(&str, &[u32], bool)> {
        self.terminal
            .sources
            .iter()
            .map(|s| (s.population.as_str(), s.candidate_active.as_slice(), s.selection_separated))
            .collect()
    }
}

/// One evaluated search stage and how it ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MzStageRecord {
    /// `target_only`, `source:<population>`, or `multi_source`.
    pub stage: String,
    /// `identified`, `missing_evidence`, `not_certified`, `obstruction` or `exhausted`.
    pub outcome: &'static str,
}

/// What a search that stopped short of a formula or an obstruction explored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MzSearchInspection {
    /// `mz_transport.fabricated_joint` when the combined search reached a
    /// line-11 state another source could exchange only jointly with the
    /// active source's experiment; otherwise `mz_transport.search_incomplete`.
    pub detail: &'static str,
    /// Every evaluated stage, in order.
    pub stages: Vec<MzStageRecord>,
    /// Rules the combined search applied, in order.
    pub explored_rules: Vec<String>,
    /// Recursive states the combined search charged.
    pub steps_explored: usize,
    /// Deepest recursion level the combined search reached.
    pub depth_reached: usize,
}

/// Outcome of a bounded mz-transport decision. Only [`Self::ProvenNonTransportable`]
/// is an impossibility claim.
#[derive(Clone, Debug)]
pub enum MzTransportDecision {
    /// A checked formula whose every cited factor binds to supplied evidence.
    Identified {
        /// The checked derivation.
        derivation: Box<MzTransportDerivation>,
        /// Catalog regimes the bound formula cites.
        cited: Arc<[RegimeId]>,
    },
    /// A checked structural obstruction for the declared controllable sets.
    ProvenNonTransportable(Box<MzTransportObstruction>),
    /// A formula is certified, or would be with a declared level, but the catalog
    /// lacks a regime it needs. Supplying that evidence can resolve it.
    MissingEvidence {
        /// The certified derivation, when one was found.
        derivation: Option<Box<MzTransportDerivation>>,
        /// Stable description of the missing factor or level.
        detail: String,
    },
    /// The bounded search finished without a formula or a checked obstruction.
    /// This is not a non-transportability claim.
    NotCertified(MzSearchInspection),
    /// A limit or cancellation stopped the search; nothing is claimed.
    Exhausted(SearchReceipt),
}

impl MzTransportDecision {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    ///
    /// Its legacy Python `outcome` spelling `exhausted` reads as `budget_cancel`.
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified { .. } => antecedent_core::TransportOutcomeKind::Identified,
            Self::ProvenNonTransportable(_) => {
                antecedent_core::TransportOutcomeKind::ProvenNonTransportable
            }
            Self::MissingEvidence { .. } => antecedent_core::TransportOutcomeKind::MissingEvidence,
            Self::NotCertified(_) => antecedent_core::TransportOutcomeKind::NotCertified,
            Self::Exhausted(_) => antecedent_core::TransportOutcomeKind::BudgetCancel,
        }
    }
}

impl MzTransportDecision {
    /// Top-level reason code of a non-identified outcome; `None` when identified.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified { .. } => None,
            Self::ProvenNonTransportable(_) => Some("transport_proven_non_transportable"),
            Self::MissingEvidence { .. } => Some("transport_missing_evidence"),
            Self::NotCertified(_) => Some("transport_not_certified"),
            Self::Exhausted(_) => Some("transport_budget_cancel"),
        }
    }

    /// Stable `mz_transport.*` detail of a non-identified outcome; `None` when
    /// identified. Specifics stay in the variant's structured fields.
    #[must_use]
    pub const fn detail_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified { .. } => None,
            Self::ProvenNonTransportable(_) => Some(CHECKED_OBSTRUCTION),
            Self::MissingEvidence { .. } => Some(MISSING_JOINT_REGIME),
            Self::NotCertified(inspection) => Some(inspection.detail),
            Self::Exhausted(_) => Some(BUDGET),
        }
    }
}

const STAGE_TARGET: &str = "target_only";
const STAGE_MULTI: &str = "multi_source";
/// Receipt region of the obstruction replay that follows the combined search.
const STAGE_OBSTRUCTION_CHECK: &str = "stage:obstruction_check";

/// Decide a bounded multi-source limited-experiment query.
///
/// Stages run in a fixed order, each only when the earlier ones do not
/// identify: the target alone; each source's own z route, in canonical source
/// order; then the combined `TR^mz` search and, when it reaches a forced
/// line-11 state, the replay that certifies it as an obstruction. One
/// [`SearchBudget`] under `limits` is charged by every engine step of every
/// stage, with the engine's live-state bytes, so the limits bound the whole
/// decision. A formula that does not bind is kept as missing evidence and the
/// next stage still runs; anything else is not certified. A stop anywhere is
/// [`MzTransportDecision::Exhausted`], never a verdict.
///
/// # Errors
///
/// Limits above [`MZ_TRANSPORT_DEFAULT_LIMITS`] or a query outside its bounds
/// (`mz_transport.bounds_exceeded`), invalid input
/// ([`validate_mz_transport_query`]), an invalid catalog, or a derivation that
/// fails its own check.
pub fn decide_mz_transport(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<MzTransportDecision, IdentificationError> {
    decide_bounded(graph, query, catalog, limits, MZ_TRANSPORT_MEMORY_BYTES, ctx)
}

/// [`decide_mz_transport`] under an explicit memory cap (at most
/// [`MZ_TRANSPORT_MEMORY_BYTES`]).
fn decide_bounded(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    memory_limit_bytes: u64,
    ctx: &ExecutionContext,
) -> Result<MzTransportDecision, IdentificationError> {
    if limits.operations > MZ_TRANSPORT_DEFAULT_LIMITS.operations
        || limits.depth > MZ_TRANSPORT_DEFAULT_LIMITS.depth
        || memory_limit_bytes > MZ_TRANSPORT_MEMORY_BYTES
    {
        return Err(bounds_exceeded());
    }
    let validated = validated_input(graph, query, catalog)?;
    match SearchBudget::with_memory(limits, memory_limit_bytes, ctx) {
        Ok(budget) => {
            decide_charged(graph, validated, catalog, &mut SharedSearch::new(budget), ctx)
        }
        Err(receipt) => Ok(MzTransportDecision::Exhausted(SearchReceipt {
            unevaluated: stage_names(&validated.query)
                .iter()
                .map(|s| format!("stage:{s}"))
                .collect(),
            ..receipt
        })),
    }
}

/// [`decide_mz_transport`] charging a budget an enclosing decision already
/// shares (the mixed-source search runs this route as one of its named stages).
/// Bounds, validation and the outcomes are exactly those of the public route;
/// the caller owns the limits.
///
/// # Errors
///
/// As [`decide_mz_transport`], except the limits.
pub(super) fn decide_mz_transport_shared(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<MzTransportDecision, IdentificationError> {
    decide_charged(graph, validated_input(graph, query, catalog)?, catalog, search, ctx)
}

/// The query validated against the graph and catalog, and the catalog validated
/// on its own, as every decision entry requires.
fn validated_input(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<ValidatedMzTransportQuery, IdentificationError> {
    let validated = validate_mz_transport_query(graph, query, catalog)?;
    catalog.validate().map_err(invalid_catalog)?;
    Ok(validated)
}

/// A catalog that fails its own validation, reported under this route's detail.
fn invalid_catalog(error: impl std::fmt::Display) -> IdentificationError {
    IdentificationError::invalid_catalog(format!("mz_transport.invalid_catalog: {error}"))
}

/// Stage names in evaluation order.
fn stage_names(query: &MzTransportQuery) -> Vec<String> {
    std::iter::once(STAGE_TARGET.to_owned())
        .chain(query.sources.iter().map(|s| format!("source:{}", s.population)))
        .chain(std::iter::once(STAGE_MULTI.to_owned()))
        .collect()
}

/// Every stage of a validated decision, charged to `search`.
#[allow(clippy::too_many_lines)] // One linear pass over the stages and their outcomes.
fn decide_charged(
    graph: &antecedent_graph::Admg,
    validated: ValidatedMzTransportQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<MzTransportDecision, IdentificationError> {
    let query = validated.query;
    let stage_names = stage_names(&query);
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
    let signature = super::graph_signature(&shared);
    let mut stages = Vec::new();
    let mut missing: Option<(Option<Box<MzTransportDerivation>>, String)> = None;
    // The success-path receipt covers this decision only, even when it charges a
    // budget an enclosing decision shares.
    let start_operations = search.operations();
    search.mark_decision();
    let limits_in_force = search.limits();
    let memory_limit_bytes = search.memory_limit_bytes();
    let receipt = |search: &SharedSearch<'_>, identifying: usize| MzSearchRecord {
        operations_limit: limits_in_force.operations,
        depth_limit: limits_in_force.depth,
        memory_limit_bytes,
        operations_consumed: search.operations() - start_operations,
        depth_reached: search.decision_depth(),
        explored: stage_names[..=identifying].iter().map(|s| format!("stage:{s}")).collect(),
        unevaluated: stage_names[identifying + 1..].iter().map(|s| format!("stage:{s}")).collect(),
    };
    // Explored regions are `stage:<name>` for each finished stage, then
    // `rule:<rule>` for the rules of the stopped one; unevaluated regions are
    // the stopped and later stages, then every untried line-10 source branch.
    let exhausted = |search: &SharedSearch<'_>,
                     stages: &[MzStageRecord],
                     error: &IdentificationError,
                     stopped: Option<ZTransportLimitsReceipt>,
                     after: &[&str]| {
        let explored = stages
            .iter()
            .map(|s| format!("stage:{}", s.stage))
            .chain(stopped.into_iter().flat_map(|r| r.explored_rules).map(|r| format!("rule:{r}")))
            .collect();
        let unevaluated = stage_names[stages.len().min(stage_names.len())..]
            .iter()
            .map(|s| format!("stage:{s}"))
            .chain(after.iter().map(|s| (*s).to_owned()))
            .collect();
        MzTransportDecision::Exhausted(search.receipt(search.stop_of(error), explored, unevaluated))
    };

    // Stage 1: the target alone, before any source experiment is consumed.
    let target_call = recursion_call(&query, &validated.diagrams, false);
    search.begin_stage();
    let mut stopped = None;
    match search_trz_call(&shared, &target_call, search.meter(), ctx, &mut stopped) {
        Ok(result) => {
            if let Some((arena, root, trace)) = result.identified {
                let derivation = recursive_derivation(
                    &query,
                    &signature,
                    MzTransportRoute::TargetOnly,
                    arena,
                    root,
                    trace,
                    &stages,
                    receipt(search, 0),
                )?;
                match bind(&derivation, catalog, &query) {
                    Ok(cited) => {
                        return Ok(MzTransportDecision::Identified {
                            derivation: Box::new(derivation),
                            cited,
                        });
                    }
                    Err(detail) => {
                        stages.push(MzStageRecord {
                            stage: STAGE_TARGET.into(),
                            outcome: "missing_evidence",
                        });
                        missing.get_or_insert((Some(Box::new(derivation)), detail?));
                    }
                }
            } else {
                stages.push(MzStageRecord { stage: STAGE_TARGET.into(), outcome: "not_certified" });
            }
        }
        Err(error) if error.is_budget_or_cancel() => {
            return Ok(exhausted(search, &stages, &error, stopped, &[]));
        }
        Err(error) => return Err(error),
    }
    search.end_stage();

    // Stage 2: each source's own z route, in canonical source order.
    for (index, (source, diagram)) in query.sources.iter().zip(&validated.diagrams).enumerate() {
        search.begin_stage();
        let stage = format!("source:{}", source.population);
        let single = query.source_query(source);
        let mut stopped = None;
        let decided = match decide_z_transport_reporting(
            diagram,
            &single,
            catalog,
            search.meter(),
            ctx,
            &mut stopped,
        ) {
            Ok(decided) => decided,
            Err(error) if error.is_budget_or_cancel() => {
                return Ok(exhausted(search, &stages, &error, stopped, &[]));
            }
            Err(error) => return Err(error),
        };
        match decided {
            ZTransportDecision::Identified(z) => {
                let bound = bind_z_transport_catalog(diagram, &single, &z, catalog)?;
                let derivation = MzTransportDerivation {
                    query: query.clone(),
                    graph_signature: signature.clone(),
                    route: MzTransportRoute::SingleSource {
                        population: Arc::clone(&source.population),
                    },
                    arena: bound.derivation().arena().clone(),
                    root: bound.derivation().root(),
                    rules: z.to_record().rules,
                    single: Some(z),
                    stages: stages.clone(),
                    search: receipt(search, 1 + index),
                };
                check_leaves(&derivation)?;
                return Ok(MzTransportDecision::Identified {
                    derivation: Box::new(derivation),
                    cited: Arc::from(bound.cited_regimes()),
                });
            }
            ZTransportDecision::MissingEvidence { missing: gap } => {
                stages.push(MzStageRecord { stage, outcome: "missing_evidence" });
                let detail = match gap {
                    ZTransportMissingEvidence::UnassignedControllable { variable } => format!(
                        "{MISSING_JOINT_REGIME}: source {} names no experiment level for controllable {}",
                        source.population,
                        variable.raw()
                    ),
                    ZTransportMissingEvidence::CitedFactor { detail } => detail,
                };
                missing.get_or_insert((None, detail));
            }
            // A single-source obstruction does not bind the combined search:
            // other sources may supply the factors this one cannot.
            ZTransportDecision::ProvenNonTransportable(_) => {
                stages.push(MzStageRecord { stage, outcome: "obstruction" });
            }
            ZTransportDecision::NotCertified { .. } => {
                stages.push(MzStageRecord { stage, outcome: "not_certified" });
            }
        }
        search.end_stage();
    }

    // Stage 3: the combined search, where line 10 may exchange into any source.
    let call = recursion_call(&query, &validated.diagrams, true);
    search.begin_stage();
    let mut stopped = None;
    let result = match search_trz_call(&shared, &call, search.meter(), ctx, &mut stopped) {
        Ok(result) => result,
        Err(error) if error.is_budget_or_cancel() => {
            return Ok(exhausted(search, &stages, &error, stopped, &[]));
        }
        Err(error) => return Err(error),
    };
    if let Some((arena, root, trace)) = result.identified {
        let populations = leaf_sources(&arena, root, &query)?;
        let route = MzTransportRoute::Combined { populations: populations.into() };
        let derivation = recursive_derivation(
            &query,
            &signature,
            route,
            arena,
            root,
            trace,
            &stages,
            receipt(search, stage_names.len() - 1),
        )?;
        return match bind(&derivation, catalog, &query) {
            Ok(cited) => {
                Ok(MzTransportDecision::Identified { derivation: Box::new(derivation), cited })
            }
            Err(detail) => Ok(MzTransportDecision::MissingEvidence {
                derivation: Some(Box::new(derivation)),
                detail: detail?,
            }),
        };
    }
    if let Some((derivation, detail)) = missing {
        return Ok(MzTransportDecision::MissingEvidence { derivation, detail });
    }
    if let Some(terminal) = result.terminal_failure.filter(forced_terminal) {
        // Only a replayed line-11 terminal with no active experiment certifies;
        // a stop during the replay is exhaustion, never a verdict.
        stages.push(MzStageRecord { stage: STAGE_MULTI.into(), outcome: "obstruction" });
        let obstruction = MzTransportObstruction { query, graph_signature: signature, terminal };
        return match verify_mz_obstruction_metered(graph, &obstruction, search.meter(), ctx) {
            Ok(()) => Ok(MzTransportDecision::ProvenNonTransportable(Box::new(obstruction))),
            Err(error) if error.is_budget_or_cancel() => {
                Ok(exhausted(search, &stages, &error, None, &[STAGE_OBSTRUCTION_CHECK]))
            }
            Err(error) => Err(error),
        };
    }
    stages.push(MzStageRecord { stage: STAGE_MULTI.into(), outcome: "not_certified" });
    Ok(MzTransportDecision::NotCertified(MzSearchInspection {
        detail: if result.cross_source_joint { FABRICATED_JOINT } else { SEARCH_INCOMPLETE },
        stages,
        explored_rules: result.explored_rules,
        steps_explored: result.steps_explored,
        depth_reached: result.depth_reached,
    }))
}

/// The recursion inputs: no sources for the target-only stage, every source for
/// the combined stage. Each source keeps its own selection targets.
fn recursion_call<'q>(
    query: &'q MzTransportQuery,
    diagrams: &'q [SelectionDiagram],
    with_sources: bool,
) -> TrzCall<'q> {
    TrzCall {
        outcomes: &query.outcomes,
        treatments: &query.treatments,
        target: &query.target,
        anchor: &query.sources[0].population,
        domains: if with_sources {
            query
                .sources
                .iter()
                .zip(diagrams)
                .map(|(source, diagram)| TrzDomain {
                    population: &source.population,
                    controllable: &source.controllable,
                    experiment_assignment: &source.experiment_assignment,
                    selection_targets: diagram.selection_targets(),
                })
                .collect()
        } else {
            Vec::new()
        },
    }
}

#[allow(clippy::too_many_arguments)] // The derivation's premises, each passed once.
fn recursive_derivation(
    query: &MzTransportQuery,
    signature: &str,
    route: MzTransportRoute,
    arena: CausalExprArena,
    root: ExprId,
    trace: Vec<String>,
    stages: &[MzStageRecord],
    search: MzSearchRecord,
) -> Result<MzTransportDerivation, IdentificationError> {
    let derivation = MzTransportDerivation {
        query: query.clone(),
        graph_signature: signature.to_owned(),
        route,
        arena,
        root,
        rules: std::iter::once("mztr.recursive_reduction".to_owned()).chain(trace).collect(),
        single: None,
        stages: stages.to_vec(),
        search,
    };
    check_leaves(&derivation)?;
    Ok(derivation)
}

/// Every distribution leaf is either the target's observational law or an
/// interventional law of one declared source whose every intervened coordinate
/// is one of that source's own declared controllables. A coordinate the formula
/// holds fixed is such a controllable at its declared level, so no other
/// coordinate is allowed. A target-only formula cites no source.
///
/// A source leaf intervening outside its own controllables would be a joint
/// over another source's interventions: it refuses as
/// `mz_transport.fabricated_joint`. Every other failure is
/// `mz_transport.invalid_derivation`.
fn check_leaves(derivation: &MzTransportDerivation) -> Result<(), IdentificationError> {
    let invalid = || IdentificationError::invalid_derivation(INVALID_DERIVATION);
    let query = &derivation.query;
    let target_only = matches!(derivation.route, MzTransportRoute::TargetOnly);
    for (population, intervened, observational) in leaves(&derivation.arena, derivation.root) {
        if population == query.target.as_ref() {
            if !observational {
                return Err(invalid());
            }
            continue;
        }
        let Some(source) = query.sources.iter().find(|s| s.population.as_ref() == population)
        else {
            return Err(invalid());
        };
        if target_only || observational || intervened.is_empty() {
            return Err(invalid());
        }
        if intervened.iter().any(|v| !source.controllable.contains(v)) {
            return Err(IdentificationError::invalid_derivation(FABRICATED_JOINT));
        }
    }
    Ok(())
}

/// `(population, intervened variables, observational)` of each reachable leaf.
fn leaves(arena: &CausalExprArena, root: ExprId) -> Vec<(&str, Vec<VariableId>, bool)> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution { intervention, domain, population, .. } => {
                let intervened = arena
                    .intervention_assignments(*intervention)
                    .iter()
                    .map(|a| a.variable)
                    .collect();
                out.push((
                    arena.population(*population),
                    intervened,
                    *domain == DomainRef::Observational,
                ));
            }
            ExprNode::Product(list) => pending.extend(arena.list(*list).iter().copied()),
            ExprNode::SumOut { expr, .. } => pending.push(*expr),
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator])
            }
            _ => {}
        }
    }
    out
}

fn leaf_sources(
    arena: &CausalExprArena,
    root: ExprId,
    query: &MzTransportQuery,
) -> Result<Vec<Arc<str>>, IdentificationError> {
    let cited =
        leaves(arena, root).into_iter().map(|(p, _, _)| p.to_owned()).collect::<BTreeSet<_>>();
    let populations = query
        .sources
        .iter()
        .filter(|s| cited.contains(s.population.as_ref()))
        .map(|s| Arc::clone(&s.population))
        .collect::<Vec<_>>();
    if populations.is_empty() {
        return Err(IdentificationError::invalid_derivation(INVALID_DERIVATION));
    }
    Ok(populations)
}

/// Bind every leaf to the catalog. `Err(Ok(detail))` is missing evidence;
/// `Err(Err(error))` is any other failure.
fn bind(
    derivation: &MzTransportDerivation,
    catalog: &EvidenceCatalog,
    query: &MzTransportQuery,
) -> Result<Arc<[RegimeId]>, Result<String, IdentificationError>> {
    match bind_recursive(derivation, catalog, query) {
        Ok((_, _, cited)) => Ok(cited),
        Err(error @ IdentificationError::MissingEvidence { .. }) => Err(Ok(error.to_string())),
        Err(error) => Err(Err(error)),
    }
}

/// Bind the leaves of a recursive (target-only or combined) derivation to the
/// catalog: the bound arena, its root and the cited regimes.
fn bind_recursive(
    derivation: &MzTransportDerivation,
    catalog: &EvidenceCatalog,
    query: &MzTransportQuery,
) -> Result<(CausalExprArena, ExprId, Arc<[RegimeId]>), IdentificationError> {
    let mut arena = derivation.arena.clone();
    let mut memo = std::collections::HashMap::new();
    let mut cited = Vec::new();
    let root = bind_recursive_expression(
        derivation.root,
        &mut arena,
        catalog,
        &query.treatments,
        &mut memo,
        &mut cited,
    )?;
    Ok((arena, root, cited_regimes(cited)))
}

/// A checked mz derivation whose every factor leaf is bound to the catalog
/// regime that supplies it. Every route (target-only, single-source, combined)
/// has the same shape, so one provider evaluates them all.
#[derive(Clone, Debug)]
pub struct BoundMzTransportFunctional {
    derivation: MzTransportDerivation,
    arena: CausalExprArena,
    root: ExprId,
    catalog: EvidenceCatalog,
    cited: Arc<[RegimeId]>,
}

impl BoundMzTransportFunctional {
    /// Checked symbolic derivation.
    #[must_use]
    pub const fn derivation(&self) -> &MzTransportDerivation {
        &self.derivation
    }
    /// Regime-bound formula arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Regime-bound formula root.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
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
    /// The source (or target) population of each cited regime.
    #[must_use]
    pub fn cited_populations(&self) -> Vec<(RegimeId, Arc<str>)> {
        self.cited
            .iter()
            .filter_map(|id| {
                self.catalog
                    .regimes
                    .iter()
                    .find(|r| r.id == *id)
                    .map(|r| (*id, Arc::clone(&r.population)))
            })
            .collect()
    }
}

/// Bind a checked derivation to a catalog. Each leaf binds only to a regime of
/// its own population, so a factor is always evaluated from the source whose
/// premises certified it.
///
/// Duplicate evidence binds by the single-source z-transport rule on every
/// route: a family regime (no declared levels) before a concrete one, otherwise
/// the lowest regime id, independent of catalog order; supplying a factor more
/// than once is never refused. Exact evaluation selects world-bound laws among
/// the cited regimes only.
///
/// # Errors
///
/// [`IdentificationError::MissingEvidence`] when a cited regime is absent; an
/// invalid catalog; or a derivation checked against another graph.
pub fn bind_mz_transport_catalog(
    graph: &antecedent_graph::Admg,
    derivation: &MzTransportDerivation,
    catalog: &EvidenceCatalog,
) -> Result<BoundMzTransportFunctional, IdentificationError> {
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
    if super::graph_signature(&shared) != derivation.graph_signature {
        return Err(IdentificationError::invalid_derivation(INVALID_DERIVATION));
    }
    catalog.validate().map_err(invalid_catalog)?;
    let query = &derivation.query;
    let (arena, root, cited) = if let Some(single) = &derivation.single {
        let MzTransportRoute::SingleSource { population } = &derivation.route else {
            return Err(IdentificationError::invalid_derivation(INVALID_DERIVATION));
        };
        let source = query
            .sources
            .iter()
            .find(|s| s.population == *population)
            .ok_or_else(|| IdentificationError::invalid_derivation(INVALID_DERIVATION))?;
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
        let bound =
            bind_z_transport_catalog(&diagram, &query.source_query(source), single, catalog)?;
        (bound.arena().clone(), bound.root(), Arc::from(bound.cited_regimes()))
    } else {
        bind_recursive(derivation, catalog, query)?
    };
    Ok(BoundMzTransportFunctional {
        derivation: derivation.clone(),
        arena,
        root,
        catalog: catalog.clone(),
        cited,
    })
}

/// A terminal reached with no active experiment is forced on every search
/// path; only then does a failure of every source mean the search failed.
fn forced_terminal(terminal: &TrzTerminalFailure) -> bool {
    terminal.active_interventions.is_empty()
        && !terminal.sources.is_empty()
        && terminal.sources.iter().all(|s| s.candidate_active.is_empty() || !s.selection_separated)
}

/// Independently replay an mz obstruction: the combined search must reach the
/// same forced terminal, and each source's line-11 premises are rechecked with
/// the independent separation checker, not the search's own.
///
/// # Errors
/// `mz_transport.invalid_obstruction` when the graph, query or terminal differ,
/// or a source could exchange at `C0`; a budget or cancellation of the replay.
pub fn verify_mz_transport_obstruction(
    graph: &antecedent_graph::Admg,
    obstruction: &MzTransportObstruction,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    verify_mz_obstruction_metered(graph, obstruction, SidMeter::Limits(limits), ctx)
}

/// [`verify_mz_transport_obstruction`] with its replay charged to `meter`.
fn verify_mz_obstruction_metered(
    graph: &antecedent_graph::Admg,
    obstruction: &MzTransportObstruction,
    meter: SidMeter<'_>,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    let bad = || IdentificationError::invalid_derivation(INVALID_OBSTRUCTION);
    let query = &obstruction.query;
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
    if super::graph_signature(&shared) != obstruction.graph_signature || *query != query.canonical()
    {
        return Err(bad());
    }
    let terminal = &obstruction.terminal;
    if !forced_terminal(terminal) || terminal.sources.len() != query.sources.len() {
        return Err(bad());
    }
    let classical = super::ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::clone(&query.sources[0].population),
        target: Arc::clone(&query.target),
    };
    let engine = super::Engine::new(&shared, &classical, SidLimits::default(), ctx)?;
    let set = |raws: &[u32]| {
        engine.set(&raws.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>())
    };
    let (y, x, v) =
        (set(&terminal.outcomes)?, set(&terminal.treatments)?, set(&terminal.vertices)?);
    let c0 = super::difference(&v, &x);
    let districts = engine.prepared.c_components(&c0);
    if !y.any() || !x.any() || districts.len() != 1 || engine.prepared.c_components(&v).len() != 1 {
        return Err(bad());
    }
    let mut c0_vars = engine.vars(&districts[0])?.iter().map(|v| v.raw()).collect::<Vec<_>>();
    c0_vars.sort_unstable();
    if c0_vars != terminal.c0 {
        return Err(bad());
    }
    let state = super::State { y, x: x.clone(), v, kernel: ExprId::from_raw(0) };
    for (source, recorded) in query.sources.iter().zip(&terminal.sources) {
        let mut active = source
            .controllable
            .iter()
            .filter(|variable| {
                engine.prepared.var_to_dense(**variable).is_ok_and(|d| x.contains(d))
            })
            .map(|variable| variable.raw())
            .collect::<Vec<_>>();
        active.sort_unstable();
        let separated = engine.independently_admissible(&state, &[], &source.selection_targets)?;
        if recorded.population != source.population.as_ref()
            || recorded.candidate_active != active
            || recorded.selection_separated != separated
            || (!active.is_empty() && separated)
        {
            return Err(bad());
        }
    }
    let diagrams = query
        .sources
        .iter()
        .map(|s| SelectionDiagram::try_new(graph.clone(), Arc::clone(&s.selection_targets)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
    let replay =
        search_trz_call(&shared, &recursion_call(query, &diagrams, true), meter, ctx, &mut None)?;
    if replay.identified.is_some() || replay.terminal_failure.as_ref() != Some(terminal) {
        return Err(bad());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DistributionAvailability, EvidenceKind, EvidenceRegime, InterventionAssignment, RegimeId,
        Value,
    };
    use antecedent_graph::{Admg, DenseNodeId};

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    /// A four-node chain Z1 -> X -> Z2 -> Y with Z1 <-> X and X <-> Y.
    fn graph() -> Admg {
        let mut g = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 1), (1, 3)] {
            g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        g
    }

    fn source(population: &str, controllable: &[u32], selection: &[u32]) -> ZTransportSourceSpec {
        ZTransportSourceSpec {
            population: Arc::from(population),
            controllable: controllable.iter().copied().map(v).collect::<Vec<_>>().into(),
            experiment_assignment: Arc::from([]),
            selection_targets: selection.iter().copied().map(v).collect::<Vec<_>>().into(),
        }
    }

    fn query(sources: Vec<ZTransportSourceSpec>) -> MzTransportQuery {
        MzTransportQuery {
            outcomes: Arc::from([v(3)]),
            treatments: Arc::from([v(1)]),
            target: Arc::from("target"),
            sources: sources.into(),
        }
    }

    fn experiment(id: u32, population: &str, on: &[u32]) -> EvidenceRegime {
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            on.iter().copied().map(v).collect::<Vec<_>>(),
            [],
            [v(0), v(1), v(2), v(3)]
                .into_iter()
                .filter(|x| !on.contains(&x.raw()))
                .collect::<Vec<_>>(),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    fn catalog(regimes: Vec<EvidenceRegime>) -> EvidenceCatalog {
        EvidenceCatalog::try_new([], regimes, [], None).unwrap()
    }

    #[test]
    fn source_order_does_not_change_the_validated_query() {
        let a = source("a", &[2], &[0]);
        let b = source("b", &[0], &[2]);
        let evidence = catalog(vec![experiment(1, "a", &[2]), experiment(2, "b", &[0])]);
        let forward =
            validate_mz_transport_query(&graph(), &query(vec![a.clone(), b.clone()]), &evidence)
                .unwrap();
        let reverse = validate_mz_transport_query(&graph(), &query(vec![b, a]), &evidence).unwrap();
        assert_eq!(forward.query, reverse.query);
        assert_eq!(&*forward.query.sources[0].population, "a");
        assert_eq!(forward.candidate_regimes, 2);
        assert_eq!(forward.diagrams.len(), 2);
    }

    const EXCEEDED: IdentificationError =
        IdentificationError::UnsupportedInput { code: "mz_transport.bounds_exceeded" };

    fn invalid_query(result: Result<ValidatedMzTransportQuery, IdentificationError>) -> bool {
        matches!(result, Err(IdentificationError::InvalidInput { message }) if message.starts_with("mz_transport.invalid_query: "))
    }

    #[test]
    fn source_count_and_population_bounds_refuse_by_code() {
        let evidence = catalog(vec![]);
        let one =
            validate_mz_transport_query(&graph(), &query(vec![source("a", &[2], &[])]), &evidence);
        assert_eq!(one.unwrap_err(), EXCEEDED);
        let five = (0..5).map(|i| source(&format!("s{i}"), &[2], &[])).collect();
        assert_eq!(
            validate_mz_transport_query(&graph(), &query(five), &evidence).unwrap_err(),
            EXCEEDED
        );
        let duplicate = query(vec![source("a", &[2], &[]), source("a", &[0], &[])]);
        assert!(invalid_query(validate_mz_transport_query(&graph(), &duplicate, &evidence)));
        let as_target = query(vec![source("a", &[2], &[]), source("target", &[0], &[])]);
        assert!(invalid_query(validate_mz_transport_query(&graph(), &as_target, &evidence)));
        // Every source declares one to four controllables.
        let no_controllable = query(vec![source("a", &[], &[]), source("b", &[0], &[])]);
        assert_eq!(
            validate_mz_transport_query(&graph(), &no_controllable, &evidence).unwrap_err(),
            EXCEEDED
        );
        // Each source still passes the single-source contract, reported as mz.
        let overlap = query(vec![source("a", &[2, 2], &[]), source("b", &[0], &[])]);
        assert!(invalid_query(validate_mz_transport_query(&graph(), &overlap, &evidence)));
    }

    #[test]
    fn supplied_regimes_must_respect_declared_availability() {
        let sources = query(vec![source("a", &[2], &[0]), source("b", &[0], &[2])]);
        // Source a declared only {Z2} controllable but supplies do(Z1).
        let invalid_catalog = |evidence: &EvidenceCatalog| matches!(validate_mz_transport_query(&graph(), &sources, evidence), Err(IdentificationError::InvalidCatalog { message }) if message.starts_with("mz_transport.invalid_catalog: "));
        let outside = catalog(vec![experiment(1, "a", &[0])]);
        assert!(invalid_catalog(&outside));
        let target_experiment = catalog(vec![experiment(1, "target", &[1])]);
        assert!(invalid_catalog(&target_experiment));
        // A proposed regime is not supplied evidence and is not counted.
        let mut proposed = experiment(1, "a", &[0]);
        proposed.evidence_kind = EvidenceKind::Proposed;
        let validated =
            validate_mz_transport_query(&graph(), &sources, &catalog(vec![proposed])).unwrap();
        assert_eq!(validated.candidate_regimes, 0);
    }

    #[test]
    fn candidate_regime_bound_refuses_rather_than_truncating() {
        let sources = query(vec![source("a", &[2], &[]), source("b", &[0], &[])]);
        let regimes = (0..=u32::try_from(MZ_TRANSPORT_MAX_CANDIDATE_REGIMES).unwrap())
            .map(|id| {
                let mut regime = experiment(id, "a", &[2]);
                regime.intervention_values = Arc::from([InterventionAssignment {
                    variable: v(2),
                    value: Value::f64(f64::from(id)),
                }]);
                regime
            })
            .collect();
        assert_eq!(
            validate_mz_transport_query(&graph(), &sources, &catalog(regimes)).unwrap_err(),
            EXCEEDED
        );
    }

    /// A derivation citing one leaf of `population` intervening on `on`.
    fn one_leaf(population: &str, on: &[u32], route: MzTransportRoute) -> MzTransportDerivation {
        let mut arena = CausalExprArena::new();
        let variables = arena.intern_var_set([v(3)]);
        let conditioned_on = arena.empty_var_set();
        let intervention = arena.intern_intervention_assignments(
            on.iter().map(|i| antecedent_expr::InterventionAssignment::symbolic(v(*i))),
        );
        let population = arena.intern_population(Arc::from(population));
        let root = arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: if on.is_empty() {
                DomainRef::Observational
            } else {
                DomainRef::Interventional
            },
            population,
            regime: None,
        });
        MzTransportDerivation {
            query: query(vec![source("a", &[2], &[0]), source("b", &[0], &[2])]).canonical(),
            graph_signature: String::new(),
            route,
            arena,
            root,
            rules: Vec::new(),
            single: None,
            stages: Vec::new(),
            search: MzSearchRecord {
                operations_limit: 1,
                depth_limit: 1,
                memory_limit_bytes: 1,
                operations_consumed: 0,
                depth_reached: 0,
                explored: Vec::new(),
                unevaluated: Vec::new(),
            },
        }
    }

    #[test]
    fn a_source_leaf_intervenes_only_on_its_own_controllables() {
        let combined = || MzTransportRoute::Combined { populations: Arc::from([Arc::from("a")]) };
        check_leaves(&one_leaf("a", &[2], combined())).unwrap();
        check_leaves(&one_leaf("target", &[], MzTransportRoute::TargetOnly)).unwrap();
        // do(Z1) is b's experiment: a leaf of a under do(Z1, Z2) would be a
        // joint over both sources' interventions.
        for on in [&[0][..], &[0, 2][..]] {
            assert_eq!(
                check_leaves(&one_leaf("a", on, combined())).unwrap_err(),
                IdentificationError::invalid_derivation("mz_transport.fabricated_joint")
            );
        }
        let invalid = IdentificationError::invalid_derivation("mz_transport.invalid_derivation");
        for derivation in [
            one_leaf("a", &[], combined()),
            one_leaf("target", &[1], combined()),
            one_leaf("elsewhere", &[2], combined()),
            one_leaf("a", &[2], MzTransportRoute::TargetOnly),
        ] {
            assert_eq!(check_leaves(&derivation).unwrap_err(), invalid);
        }
    }

    #[test]
    fn cancellation_in_the_middle_of_a_decision_is_a_receipt() {
        // Figure 1(c,d) with source a alone and an unhelpful source runs every
        // stage; cancel once the shared budget has charged a few operations.
        let mut g = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 1), (0, 2), (0, 3)] {
            g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let q = query(vec![source("a", &[2], &[0, 2]), source("unhelpful", &[1], &[0, 1, 2, 3])]);
        let evidence = catalog(vec![]);
        for after in [1, 4, 12] {
            let ctx = ExecutionContext::for_tests(1);
            let validated = validate_mz_transport_query(&g, &q, &evidence).unwrap();
            let budget = SearchBudget::new(MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap();
            let mut search = SharedSearch::new(budget);
            search.cancel_after = Some((after, ctx.cancellation.clone()));
            let decision = decide_charged(&g, validated, &evidence, &mut search, &ctx).unwrap();
            let MzTransportDecision::Exhausted(receipt) = decision else {
                panic!("cancellation after {after} operations must not be a verdict: {decision:?}");
            };
            assert_eq!(receipt.stop, antecedent_core::SearchStop::Cancelled);
            assert_eq!(receipt.operations_consumed, Some(after));
            assert!(!receipt.unevaluated.is_empty());
        }
    }

    #[test]
    fn a_tampered_obstruction_fails_independent_verification() {
        // R-443 Figure 1(c,d) graph; source "a" (selection {Z1, Z2}, do(Z2)) alone.
        let mut g = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 1), (0, 2), (0, 3)] {
            g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let q = MzTransportQuery {
            outcomes: Arc::from([v(3)]),
            treatments: Arc::from([v(1)]),
            target: Arc::from("target"),
            sources: vec![source("a", &[2], &[0, 2]), source("unhelpful", &[1], &[0, 1, 2, 3])]
                .into(),
        };
        let ctx = ExecutionContext::for_tests(1);
        let MzTransportDecision::ProvenNonTransportable(obstruction) =
            decide_mz_transport(&g, &q, &catalog(vec![]), MZ_TRANSPORT_DEFAULT_LIMITS, &ctx)
                .unwrap()
        else {
            panic!("expected an obstruction");
        };
        verify_mz_transport_obstruction(&g, &obstruction, SidLimits::default(), &ctx).unwrap();
        let mut wrong_c0 = (*obstruction).clone();
        wrong_c0.terminal.c0 = vec![3];
        let mut exchangeable = (*obstruction).clone();
        exchangeable.terminal.sources[0].selection_separated = true;
        exchangeable.terminal.sources[0].candidate_active = vec![2];
        let mut other_graph = (*obstruction).clone();
        other_graph.graph_signature.push('x');
        for tampered in [wrong_c0, exchangeable, other_graph] {
            assert!(
                verify_mz_transport_obstruction(&g, &tampered, SidLimits::default(), &ctx).is_err()
            );
        }
    }
}
