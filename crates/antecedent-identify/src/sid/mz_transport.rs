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
//! Completeness holds for the theorem's own information family: for every
//! source, experiments on every subset of its declared controllable set with all
//! observed variables measured. A formula bound to a supplied catalog is sound
//! but incomplete; a failure is an obstruction only when it is evaluated
//! structurally over the declared controllable sets, never because a catalog
//! lacks a regime. When several sources certify the same factor, the paper
//! returns a weighted combination; for exact laws each is a valid formula, so
//! this contract selects one in canonical source order and keeps the others as
//! alternatives, which makes the result invariant to source declaration order.
//!
//! This module fixes the query, its bounds, and input validation. Identification
//! and execution are separate stages built on it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    EvidenceCatalog, ExecutionContext, RegimeId, RegimeKind, SearchBudget, SearchLimits,
    SearchReceipt, SearchStop, VariableId,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::SelectionDiagram;

use super::z_transport::{
    TrzCall, TrzDomain, TrzTerminalFailure, Z_TRANSPORT_MAX_CONTROLLABLE, Z_TRANSPORT_MAX_OBSERVED,
    ZTransportBudgetKind, ZTransportDecision, ZTransportDerivation, ZTransportLimitsReceipt,
    ZTransportMissingEvidence, ZTransportOutcome, ZTransportQuery, ZTransportSourceSpec,
    bind_recursive_expression, bind_z_transport_catalog, cited_regimes,
    decide_z_transport_inspecting, search_trz_call, validate_z_transport_query,
};
use super::{IdentificationError, SidLimits};

/// Observed variables in the shared graph.
pub const MZ_TRANSPORT_MAX_OBSERVED: usize = Z_TRANSPORT_MAX_OBSERVED;
/// Source populations in one query. One source is the single-source z route.
pub const MZ_TRANSPORT_MAX_SOURCES: usize = 4;
/// Controllable variables declared per source.
pub const MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE: usize = Z_TRANSPORT_MAX_CONTROLLABLE;
/// Available source regimes the search may consider as candidate evidence.
pub const MZ_TRANSPORT_MAX_CANDIDATE_REGIMES: usize = 64;
/// Default search limits: states charged and recursion depth. Memory and
/// cancellation come from the execution context on every charge.
pub const MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits { operations: 4096, depth: 24 };

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
/// Each source must pass the single-source z-transport contract on its own
/// selection diagram. Across sources: two to four distinct populations, none the
/// target; every available source experiment intervenes only on a subset of that
/// source's declared controllable set; the target supplies no experiments; and
/// at most [`MZ_TRANSPORT_MAX_CANDIDATE_REGIMES`] available source regimes.
///
/// # Errors
///
/// [`IdentificationError::UnsupportedInput`] with an `mz_transport.*` code for an
/// exceeded bound; [`IdentificationError::InvalidInput`] for a malformed query or
/// a catalog that contradicts its declarations.
pub fn validate_mz_transport_query(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<ValidatedMzTransportQuery, IdentificationError> {
    let query = query.canonical();
    if query.sources.len() < 2 {
        return Err(IdentificationError::invalid_input(
            "mz_transport.single_source_uses_z_transport",
        ));
    }
    if query.sources.len() > MZ_TRANSPORT_MAX_SOURCES {
        return Err(IdentificationError::UnsupportedInput { code: "mz_transport.source_count" });
    }
    let populations = query.sources.iter().map(|s| s.population.as_ref()).collect::<BTreeSet<_>>();
    if populations.len() != query.sources.len() || populations.contains(query.target.as_ref()) {
        return Err(IdentificationError::invalid_input("mz_transport.population_collision"));
    }
    let mut diagrams = Vec::with_capacity(query.sources.len());
    for source in query.sources.iter() {
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
        validate_z_transport_query(&diagram, &query.source_query(source))?;
        diagrams.push(diagram);
    }
    let mut candidate_regimes = 0usize;
    for regime in catalog.regimes.iter().filter(|r| r.supplies_population_law()) {
        if regime.population == query.target {
            if regime.kind == RegimeKind::Experimental {
                return Err(IdentificationError::invalid_input(
                    "mz_transport.target_experiments_unsupported",
                ));
            }
            continue;
        }
        let Some(source) = query.sources.iter().find(|s| s.population == regime.population) else {
            continue;
        };
        if regime.interventions.iter().any(|v| !source.controllable.contains(v)) {
            return Err(IdentificationError::invalid_input(
                "mz_transport.regime_outside_controllable",
            ));
        }
        candidate_regimes += 1;
    }
    if candidate_regimes > MZ_TRANSPORT_MAX_CANDIDATE_REGIMES {
        return Err(IdentificationError::UnsupportedInput {
            code: "mz_transport.candidate_regime_count",
        });
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
    /// Search receipt: every stage evaluated before the identifying one.
    #[must_use]
    pub fn stages(&self) -> &[MzStageRecord] {
        &self.stages
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
        }
    }

    /// Reconstruct a derivation from an untrusted record: the bounded decision is
    /// re-run on `graph`, `query` and `catalog` under the caller's limits, and the
    /// record and arena must equal the derivation it produces.
    ///
    /// # Errors
    /// The decision does not identify, or any recorded premise or expression differs.
    pub fn from_record_checked(
        graph: &antecedent_graph::Admg,
        query: &MzTransportQuery,
        catalog: &EvidenceCatalog,
        record: &MzTransportDerivationRecord,
        arena: &CausalExprArena,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let MzTransportDecision::Identified { derivation, .. } =
            decide_mz_transport(graph, query, catalog, limits, ctx)?
        else {
            return Err(IdentificationError::invalid_derivation(
                "mz_transport.proof_not_reproduced",
            ));
        };
        if derivation.to_record() != *record || derivation.arena != *arena {
            return Err(IdentificationError::invalid_derivation(
                "mz_transport.proof_record_mismatch",
            ));
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

const STAGE_TARGET: &str = "target_only";
const STAGE_MULTI: &str = "multi_source";

/// Decide a bounded multi-source limited-experiment query.
///
/// Stages run in a fixed order, each only when the earlier ones do not
/// identify: the target alone; each source's own z route, in canonical source
/// order; then the combined `TR^mz` search. The limits apply to each stage.
/// A formula that does not bind is kept as missing evidence and the next
/// stage still runs; a forced line-11 state of the combined search is certified
/// as an obstruction; anything else is not certified.
///
/// # Errors
///
/// Invalid or out-of-bound input ([`validate_mz_transport_query`]), an invalid
/// catalog, or a derivation that fails its own check.
#[allow(clippy::too_many_lines)] // One linear pass over the stages and their outcomes.
pub fn decide_mz_transport(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<MzTransportDecision, IdentificationError> {
    let validated = validate_mz_transport_query(graph, query, catalog)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("mz_transport.invalid_catalog: {error}"))
    })?;
    let query = validated.query;
    let stage_names = std::iter::once(STAGE_TARGET.to_owned())
        .chain(query.sources.iter().map(|s| format!("source:{}", s.population)))
        .chain(std::iter::once(STAGE_MULTI.to_owned()))
        .collect::<Vec<_>>();
    if let Err(receipt) = SearchBudget::new(limits, ctx) {
        return Ok(MzTransportDecision::Exhausted(SearchReceipt {
            unevaluated: stage_names,
            ..receipt
        }));
    }
    let sid_limits = SidLimits { steps: limits.operations, depth: limits.depth };
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
    let signature = super::graph_signature(&shared);
    let mut stages = Vec::new();
    let mut missing: Option<(Option<Box<MzTransportDerivation>>, String)> = None;
    let exhausted = |stages: &[MzStageRecord], receipt: ZTransportLimitsReceipt| {
        let done = stages.len();
        MzTransportDecision::Exhausted(search_receipt(
            receipt,
            limits,
            ctx,
            stages.iter().map(|s| s.stage.clone()).collect(),
            stage_names[done..].to_vec(),
        ))
    };

    // Stage 1: the target alone, before any source experiment is consumed.
    let target_call = recursion_call(&query, &validated.diagrams, false);
    let mut receipt = None;
    match search_trz_call(&shared, &target_call, sid_limits, ctx, &mut receipt) {
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
            return Ok(exhausted(
                &stages,
                receipt.unwrap_or_else(|| pre_search_receipt(&error, sid_limits)),
            ));
        }
        Err(error) => return Err(error),
    }

    // Stage 2: each source's own z route, in canonical source order.
    for (source, diagram) in query.sources.iter().zip(&validated.diagrams) {
        let stage = format!("source:{}", source.population);
        let single = query.source_query(source);
        match decide_z_transport_inspecting(diagram, &single, catalog, sid_limits, ctx)? {
            ZTransportOutcome::Exhausted(receipt) => return Ok(exhausted(&stages, receipt)),
            ZTransportOutcome::Decided(ZTransportDecision::Identified(z)) => {
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
                };
                return Ok(MzTransportDecision::Identified {
                    derivation: Box::new(derivation),
                    cited: Arc::from(bound.cited_regimes()),
                });
            }
            ZTransportOutcome::Decided(ZTransportDecision::MissingEvidence { missing: gap }) => {
                stages.push(MzStageRecord { stage, outcome: "missing_evidence" });
                let detail = match gap {
                    ZTransportMissingEvidence::UnassignedControllable { variable } => format!(
                        "mz_transport.unassigned_controllable: {} {}",
                        source.population,
                        variable.raw()
                    ),
                    ZTransportMissingEvidence::CitedFactor { detail } => detail,
                };
                missing.get_or_insert((None, detail));
            }
            // A single-source obstruction does not bind the combined search:
            // other sources may supply the factors this one cannot.
            ZTransportOutcome::Decided(ZTransportDecision::ProvenNonTransportable(_)) => {
                stages.push(MzStageRecord { stage, outcome: "obstruction" });
            }
            ZTransportOutcome::Decided(ZTransportDecision::NotCertified { .. }) => {
                stages.push(MzStageRecord { stage, outcome: "not_certified" });
            }
        }
    }

    // Stage 3: the combined search, where line 10 may exchange into any source.
    let call = recursion_call(&query, &validated.diagrams, true);
    let mut receipt = None;
    let result = match search_trz_call(&shared, &call, sid_limits, ctx, &mut receipt) {
        Ok(result) => result,
        Err(error) if error.is_budget_or_cancel() => {
            return Ok(exhausted(
                &stages,
                receipt.unwrap_or_else(|| pre_search_receipt(&error, sid_limits)),
            ));
        }
        Err(error) => return Err(error),
    };
    if let Some((arena, root, trace)) = result.identified {
        let populations = leaf_sources(&arena, root, &query)?;
        let route = MzTransportRoute::Combined { populations: populations.into() };
        let derivation =
            recursive_derivation(&query, &signature, route, arena, root, trace, &stages)?;
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
        let obstruction = MzTransportObstruction { query, graph_signature: signature, terminal };
        verify_mz_transport_obstruction(graph, &obstruction, sid_limits, ctx)?;
        return Ok(MzTransportDecision::ProvenNonTransportable(Box::new(obstruction)));
    }
    stages.push(MzStageRecord { stage: STAGE_MULTI.into(), outcome: "not_certified" });
    Ok(MzTransportDecision::NotCertified(MzSearchInspection {
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

fn recursive_derivation(
    query: &MzTransportQuery,
    signature: &str,
    route: MzTransportRoute,
    arena: CausalExprArena,
    root: ExprId,
    trace: Vec<String>,
    stages: &[MzStageRecord],
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
    };
    check_leaves(&derivation)?;
    Ok(derivation)
}

/// Every distribution leaf is either the target's observational law or an
/// interventional law of one declared source, intervening only on that source's
/// declared controllables plus coordinates the formula holds fixed.
fn check_leaves(derivation: &MzTransportDerivation) -> Result<(), IdentificationError> {
    let bad = |reason: &'static str| IdentificationError::invalid_derivation(reason);
    let query = &derivation.query;
    for (population, intervened, observational) in leaves(&derivation.arena, derivation.root) {
        if population == query.target.as_ref() {
            if !observational {
                return Err(bad("mz_transport.target_leaf_not_observational"));
            }
            continue;
        }
        let Some(source) = query.sources.iter().find(|s| s.population.as_ref() == population)
        else {
            return Err(bad("mz_transport.leaf_population_undeclared"));
        };
        if observational || !intervened.iter().any(|v| source.controllable.contains(v)) {
            return Err(bad("mz_transport.source_leaf_without_exchange"));
        }
    }
    if matches!(derivation.route, MzTransportRoute::TargetOnly)
        && leaves(&derivation.arena, derivation.root)
            .iter()
            .any(|(p, _, _)| *p != query.target.as_ref())
    {
        return Err(bad("mz_transport.target_route_cites_source"));
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
        return Err(IdentificationError::InvariantViolated {
            message: "mz_transport.combined_formula_cites_no_source",
        });
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
    let mut arena = derivation.arena.clone();
    let mut memo = std::collections::HashMap::new();
    let mut cited = Vec::new();
    match bind_recursive_expression(
        derivation.root,
        &mut arena,
        catalog,
        &query.treatments,
        &mut memo,
        &mut cited,
    ) {
        Ok(_) => Ok(cited_regimes(cited)),
        Err(error @ IdentificationError::MissingEvidence { .. }) => Err(Ok(error.to_string())),
        Err(error) => Err(Err(error)),
    }
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
        return Err(IdentificationError::invalid_derivation("mz_transport.proof_input_mismatch"));
    }
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("mz_transport.invalid_catalog: {error}"))
    })?;
    let query = &derivation.query;
    let (arena, root, cited) = if let Some(single) = &derivation.single {
        let MzTransportRoute::SingleSource { population } = &derivation.route else {
            return Err(IdentificationError::invalid_derivation("mz_transport.route_mismatch"));
        };
        let source =
            query.sources.iter().find(|s| s.population == *population).ok_or_else(|| {
                IdentificationError::invalid_derivation("mz_transport.route_mismatch")
            })?;
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
        let bound =
            bind_z_transport_catalog(&diagram, &query.source_query(source), single, catalog)?;
        (bound.arena().clone(), bound.root(), Arc::from(bound.cited_regimes()))
    } else {
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
        (arena, root, cited_regimes(cited))
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
/// The graph, query or terminal differ, or a source could exchange at `C0`.
pub fn verify_mz_transport_obstruction(
    graph: &antecedent_graph::Admg,
    obstruction: &MzTransportObstruction,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    let bad = || IdentificationError::invalid_derivation("mz_transport.invalid_obstruction");
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
        search_trz_call(&shared, &recursion_call(query, &diagrams, true), limits, ctx, &mut None)?;
    if replay.identified.is_some() || replay.terminal_failure.as_ref() != Some(terminal) {
        return Err(IdentificationError::invalid_derivation(
            "mz_transport.obstruction_replay_mismatch",
        ));
    }
    Ok(())
}

fn pre_search_receipt(error: &IdentificationError, limits: SidLimits) -> ZTransportLimitsReceipt {
    ZTransportLimitsReceipt {
        budget: match error {
            IdentificationError::Cancelled => ZTransportBudgetKind::Cancelled,
            IdentificationError::Budget { budget: super::IdentificationBudget::Memory } => {
                ZTransportBudgetKind::Memory
            }
            _ => ZTransportBudgetKind::Steps,
        },
        steps_limit: limits.steps,
        depth_limit: limits.depth,
        steps_consumed: None,
        depth_reached: None,
        explored_rules: Vec::new(),
    }
}

/// The shared bounded-search receipt of a stopped stage.
fn search_receipt(
    receipt: ZTransportLimitsReceipt,
    limits: SearchLimits,
    ctx: &ExecutionContext,
    completed_stages: Vec<String>,
    unevaluated: Vec<String>,
) -> SearchReceipt {
    SearchReceipt {
        stop: match receipt.budget {
            ZTransportBudgetKind::Steps => SearchStop::Operations,
            ZTransportBudgetKind::Depth => SearchStop::Depth,
            ZTransportBudgetKind::Memory => SearchStop::Memory,
            ZTransportBudgetKind::Cancelled => SearchStop::Cancelled,
        },
        operations_limit: limits.operations,
        depth_limit: limits.depth,
        memory_limit_bytes: ctx.memory.hard_limit_bytes,
        operations_consumed: receipt.steps_consumed,
        depth_reached: receipt.depth_reached,
        explored: completed_stages.into_iter().chain(receipt.explored_rules).collect(),
        unevaluated,
    }
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

    #[test]
    fn source_count_and_population_bounds_refuse_by_code() {
        let evidence = catalog(vec![]);
        let one =
            validate_mz_transport_query(&graph(), &query(vec![source("a", &[2], &[])]), &evidence);
        assert!(
            matches!(one, Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.single_source_uses_z_transport")
        );
        let five = (0..5).map(|i| source(&format!("s{i}"), &[2], &[])).collect();
        assert!(matches!(
            validate_mz_transport_query(&graph(), &query(five), &evidence),
            Err(IdentificationError::UnsupportedInput { code: "mz_transport.source_count" })
        ));
        let duplicate = query(vec![source("a", &[2], &[]), source("a", &[0], &[])]);
        assert!(
            matches!(validate_mz_transport_query(&graph(), &duplicate, &evidence), Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.population_collision")
        );
        let as_target = query(vec![source("a", &[2], &[]), source("target", &[0], &[])]);
        assert!(validate_mz_transport_query(&graph(), &as_target, &evidence).is_err());
        // Each source still passes the single-source contract.
        let no_controllable = query(vec![source("a", &[], &[]), source("b", &[0], &[])]);
        assert!(validate_mz_transport_query(&graph(), &no_controllable, &evidence).is_err());
    }

    #[test]
    fn supplied_regimes_must_respect_declared_availability() {
        let sources = query(vec![source("a", &[2], &[0]), source("b", &[0], &[2])]);
        // Source a declared only {Z2} controllable but supplies do(Z1).
        let outside = catalog(vec![experiment(1, "a", &[0])]);
        assert!(
            matches!(validate_mz_transport_query(&graph(), &sources, &outside), Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.regime_outside_controllable")
        );
        let target_experiment = catalog(vec![experiment(1, "target", &[1])]);
        assert!(
            matches!(validate_mz_transport_query(&graph(), &sources, &target_experiment), Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.target_experiments_unsupported")
        );
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
        assert!(matches!(
            validate_mz_transport_query(&graph(), &sources, &catalog(regimes)),
            Err(IdentificationError::UnsupportedInput {
                code: "mz_transport.candidate_regime_count"
            })
        ));
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
