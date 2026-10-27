//! Classical single-source sID, scoped to access to all source experiments.
//!
//! Rules follow Bareinboim and Pearl (2013), Figure 5. Catalog binding is a
//! separate operation: failure to bind a leaf is not a nontransportability proof.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::prepared::AncestorMemo;

use crate::selection_separation::{
    MutilatedSelection, SubsetSearchEnd, for_each_admissible_subset_preferring,
    independently_separated,
};

use crate::{IdentificationBudget, IdentificationError, PreparedAdmg};

use antecedent_core::{
    ExecutionContext, SearchBudget, SearchLimits, SearchReceipt, SearchStop, VariableId,
};

use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};

use antecedent_graph::algo::components_by;

use antecedent_graph::{
    Admg, BitSet, DSeparationWorkspace, DenseNodeId, GraphWorkspace, SelectionDiagram,
};

use std::cell::RefCell;

use std::collections::HashMap;

use std::sync::Arc;

mod conditional;

mod conditional_witness;

mod counterfactual_transport;

pub mod cpdag_completion;

mod meta;

mod mixed_source;

mod mz_transport;

pub mod scenario_invariance;

pub mod scenarios;

mod static_nodes;

mod study_planning;

pub mod temporal_initial_shift;

pub mod temporal_sequence;

mod z_transport;

pub use conditional::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, ADMG_CONDITIONAL_MAX_CONDITIONED,
    ADMG_CONDITIONAL_MAX_OBSERVED, ADMG_CONDITIONAL_MAX_TREATMENTS,
    BoundConditionalTransportFunctional, ConditionalObstructionCandidate,
    ConditionalObstructionRecord, ConditionalStageRecord, ConditionalTransportDecision,
    ConditionalTransportDerivation, ConditionalTransportInspection, ConditionalTransportQuery,
    ConditionalTransportRecord, admg_conditional_refusal, decide_admg_conditional_transport,
};

pub use conditional::{ConditionalNonTransportabilityProof, ConditionalNonTransportabilityRecord};

pub use conditional_witness::{
    CONDITIONAL_WITNESS_MAX_LATENT_LEVELS, ConditionalWitnessCheck, ConditionalWitnessRecord,
    WITNESS_SEARCH_DEFAULT_OPERATIONS, WITNESS_VERIFY_DEFAULT_OPERATIONS,
    WITNESS_WORK_PER_OPERATION, WitnessKernelRecord, WitnessLatentRecord, WitnessMeter,
    WitnessModelRecord, WitnessSearch, search_conditional_witness,
    search_conditional_witness_within, verify_conditional_witness,
    verify_conditional_witness_within,
};

pub use counterfactual_transport::{
    COUNTERFACTUAL_TRANSPORT_CONTRACT, COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
    COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES, COUNTERFACTUAL_TRANSPORT_MAX_ATOMS,
    COUNTERFACTUAL_TRANSPORT_MAX_CONTROLLABLE, COUNTERFACTUAL_TRANSPORT_MAX_LIMITS,
    COUNTERFACTUAL_TRANSPORT_MAX_NODES, COUNTERFACTUAL_TRANSPORT_MAX_REGIMES,
    COUNTERFACTUAL_TRANSPORT_MAX_SOURCES, COUNTERFACTUAL_TRANSPORT_MAX_SUM_CELLS,
    COUNTERFACTUAL_TRANSPORT_MAX_WORLDS, COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
    CounterfactualCitedRegime, CounterfactualFactor, CounterfactualFunctional, CounterfactualItem,
    CounterfactualItemSource, CounterfactualLevel, CounterfactualLevelIndex,
    CounterfactualMissingRegime, CounterfactualMovedAtom, CounterfactualSum, CounterfactualSymbol,
    CounterfactualSymbolId, CounterfactualTransportDecision, CounterfactualTransportDerivation,
    CounterfactualTransportInspection, CounterfactualTransportMissing,
    CounterfactualTransportObstruction, CounterfactualTransportProblem,
    CounterfactualTransportQuery, CounterfactualTransportSearchRecord,
    CounterfactualTransportSource, CounterfactualTransportUnsupported,
    CounterfactualTransportWitness, CounterfactualTransportWitnessCheck,
    CounterfactualVariableIndex, counterfactual_transport_refusal,
    identify_counterfactual_transport, replay_counterfactual_transport,
    verify_counterfactual_transport_witness,
};

use meta::{CLASSICAL_SETTING, META_SETTING, validate_meta_sources};

pub use meta::{
    MetaSource, MetaTransportQuery, identify_meta_catalog, identify_meta_transport,
    verify_meta_s_hedge, verify_meta_transport,
};

pub use mixed_source::{
    BoundMixedSourceFunctional, MIXED_SOURCE_DEFAULT_LIMITS, MIXED_SOURCE_MAX_DISTRIBUTIONS,
    MIXED_SOURCE_MAX_DO, MIXED_SOURCE_MAX_MOVE, MIXED_SOURCE_MAX_OBSERVED, MIXED_SOURCE_RULE_SET,
    MixedExclusion, MixedInput, MixedMissingEvidence, MixedMissingLeaf, MixedQuantity, MixedRule,
    MixedSearchInspection, MixedSearchSummary, MixedSourceDecision, MixedSourceDerivation,
    MixedSourceDerivationRecord, MixedSourceLeaf, MixedSourceQuery, MixedStageRecord, MixedStep,
    MixedStepRecord, ValidatedMixedSourceQuery, bind_mixed_source_catalog, decide_mixed_source,
    mixed_source_rule_names, render_quantity, validate_mixed_source_query,
    verify_mixed_source_derivation,
};

#[allow(deprecated, reason = "the retained released name stays reachable")]
pub use mz_transport::MZ_TRANSPORT_MAX_OBSERVED;

pub use mz_transport::{
    BoundMzTransportFunctional, MZ_TRANSPORT_DEFAULT_LIMITS, MZ_TRANSPORT_MAX_CANDIDATE_REGIMES,
    MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE, MZ_TRANSPORT_MAX_SOURCES, MZ_TRANSPORT_MEMORY_BYTES,
    MzSearchInspection, MzSearchRecord, MzStageRecord, MzTransportDecision, MzTransportDerivation,
    MzTransportDerivationRecord, MzTransportObstruction, MzTransportQuery, MzTransportRoute,
    ValidatedMzTransportQuery, bind_mz_transport_catalog, decide_mz_transport,
    mz_transport_refusal, validate_mz_transport_query, verify_mz_transport_obstruction,
};

pub use study_planning::{
    STUDY_PLAN_DEFAULT_LIMITS, STUDY_PLAN_MAX_CANDIDATES, STUDY_PLAN_MAX_PROPOSALS,
    STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE, STUDY_PLAN_MAX_SUBSET, STUDY_PLAN_MEMORY_BYTES,
    STUDY_PLAN_RANKING, StudyBaseFailure, StudyFactor, StudyPlan, StudyPlanCandidate,
    StudyPlanLimits, StudyPlanRefusal, StudyPlanRoute, StudyPlanStop, StudyProposal,
    StudyProposalDerivation, StudyRepair, StudySubsetOutcome, StudySubsetRecord,
    plan_study_additions,
};

pub use z_transport::{
    BoundZTransportFunctional, ComponentFactorization, TwoSourceZTransportComponent,
    TwoSourceZTransportDecision, TwoSourceZTransportQuery, ZExperimentFamilyError,
    ZFactorObligation, ZProofOperation, ZTransportBudgetKind, ZTransportDecision,
    ZTransportDerivation, ZTransportDerivationRecord, ZTransportLimitsReceipt,
    ZTransportMissingEvidence, ZTransportNotCertifiedInspection, ZTransportNotCertifiedKind,
    ZTransportObstruction, ZTransportObstructionRecord, ZTransportOutcome,
    ZTransportProofInspection, ZTransportQuery, ZTransportResult, ZTransportSourceSpec,
    ZTransportTerminalRecord, bind_z_transport_catalog, decide_two_source_z_transport,
    decide_z_transport_inspecting, decide_z_transport_with_catalog, identify_z_transport,
    identify_z_transport_reporting, intervention_ancestral_treatments,
    intervention_levels_conflict, intervention_mutilated_admg, validate_z_experiment_family,
    validate_z_transport_query, verify_z_transport_derivation,
};

/// Theoretical query under the classical family of all source experiments.
/// This contract makes no claim about availability in a finite supplied catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassicalTransportQuery {
    /// Joint outcomes.
    pub outcomes: Arc<[VariableId]>,
    /// Hard-intervention coordinates (symbolic values).
    pub treatments: Arc<[VariableId]>,
    /// Source population possessing the complete experimental family.
    pub source: Arc<str>,
    /// Target population possessing its observational law.
    pub target: Arc<str>,
}

/// Search and verification limits, in addition to execution memory/cancellation.
#[derive(Clone, Copy, Debug)]
pub struct SidLimits {
    /// Maximum recursive subproblems across target ID and sID. It is also the number
    /// of candidate subsets the catalog pretreatment-standardization search may
    /// separation-test; that search keeps its own count, and running out of it is
    /// reported as an unmet obligation rather than an error.
    pub steps: usize,
    /// Maximum recursion depth.
    pub depth: usize,
}

impl Default for SidLimits {
    fn default() -> Self {
        Self { steps: 100_000, depth: 256 }
    }
}

/// A checked successful classical derivation. Fields are private execution authority.
#[derive(Clone, Debug)]
pub struct ClassicalTransportDerivation {
    query: ClassicalTransportQuery,
    graph_signature: String,
    selection_targets: Arc<[VariableId]>,
    arena: CausalExprArena,
    root: ExprId,
    proof: Vec<ProofStep>,
    root_step: usize,
    sources: Vec<MetaSource>,
}

impl ClassicalTransportDerivation {
    /// Frozen theoretical evidence/query scope.
    #[must_use]
    pub const fn query(&self) -> &ClassicalTransportQuery {
        &self.query
    }
    /// Authoritative population-tagged functional.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Distribution root (not a scalar mean).
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
    /// Consumed algorithm branches, for audit and conformance fixtures.
    #[must_use]
    pub fn rules(&self) -> Vec<&'static str> {
        let mut used = vec![false; self.proof.len()];
        let mut pending = vec![self.root_step];
        while let Some(step) = pending.pop() {
            if used[step] {
                continue;
            }
            used[step] = true;
            pending.extend(self.proof[step].children.iter().copied());
        }
        self.proof
            .iter()
            .enumerate()
            .filter(|(index, _)| used[*index])
            .map(|(_, step)| step.rule.name())
            .collect()
    }
}

/// Classical recursion result; an unchecked obstruction is explicitly inconclusive.
#[derive(Clone, Debug)]
pub enum ClassicalTransportResult {
    /// Symbolic distribution whose every local premise was re-derived by the checker.
    /// S-admissibility is decided a second time by an independent implementation;
    /// ancestor and c-component structure is shared with the search, and the
    /// end-to-end evidence is the latent-SCM enumeration suites.
    Identified(Box<ClassicalTransportDerivation>),
    /// Recursion found an obstruction; no negative claim without a checked forest witness.
    NotCertified,
    /// Independently verified single-source s-hedge obstruction.
    ProvenNonTransportable(SHedgeCertificate),
}

impl ClassicalTransportResult {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified(_) => antecedent_core::TransportOutcomeKind::Identified,
            Self::NotCertified => antecedent_core::TransportOutcomeKind::NotCertified,
            Self::ProvenNonTransportable(_) => {
                antecedent_core::TransportOutcomeKind::ProvenNonTransportable
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    Marginal,
    Ancestors,
    Enlarge,
    Districts,
    Factor,
    Recurse,
    /// Figure 5 line 10: `C(D)={D}` and the source experiment answers the query.
    Source,
    /// Definition 6 direct transport at any state whose outcomes are S-admissible:
    /// the source experiment on the state's intervention set is the answer.
    DirectTransport,
    Standardize,
}

impl Rule {
    fn name(self) -> &'static str {
        match self {
            Self::Marginal => "sid.line1",
            Self::Ancestors => "sid.line2",
            Self::Enlarge => "sid.line3",
            Self::Districts => "sid.line4",
            Self::Factor => "sid.line7",
            Self::Recurse => "sid.line8",
            Self::Source => "sid.line10",
            Self::DirectTransport => "transport.direct",
            Self::Standardize => "transport.pretreatment_standardize",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    y: BitSet,
    x: BitSet,
    v: BitSet,
    kernel: ExprId,
}

#[derive(Clone, Debug)]
struct ProofStep {
    state: State,
    rule: Rule,
    children: Vec<usize>,
    output: ExprId,
    parameters: Vec<VariableId>,
}

/// Untrusted portable local derivation premise. Dense coordinates are interpreted
/// only under the recorded graph identity and are checked before construction.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
pub struct SidStepRecord {
    pub outcomes: Vec<u32>,
    pub interventions: Vec<u32>,
    pub vertices: Vec<u32>,
    pub kernel: u32,
    pub rule: Rule,
    pub children: Vec<usize>,
    pub output: u32,
    pub parameters: Vec<u32>,
}

/// Untrusted portable proof, paired with its expression arena by the IO layer.
/// It has no causal authority until `from_record_checked` succeeds.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
pub struct SidDerivationRecord {
    pub evidence_setting: String,
    pub outcomes: Vec<u32>,
    pub treatments: Vec<u32>,
    pub source: String,
    pub target: String,
    pub graph_signature: String,
    pub selections: Vec<u32>,
    pub root: u32,
    pub steps: Vec<SidStepRecord>,
    pub root_step: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<MetaSource>,
}

impl ClassicalTransportDerivation {
    /// Export premises, without converting an untrusted record into authority.
    #[must_use]
    pub fn to_record(&self) -> SidDerivationRecord {
        let ids = |set: &BitSet| set.to_dense_ids().iter().map(|v| v.raw()).collect();
        SidDerivationRecord {
            evidence_setting: self.evidence_setting().into(),
            outcomes: self.query.outcomes.iter().map(|v| v.raw()).collect(),
            treatments: self.query.treatments.iter().map(|v| v.raw()).collect(),
            source: self.query.source.to_string(),
            target: self.query.target.to_string(),
            graph_signature: self.graph_signature.clone(),
            selections: {
                let mut selections =
                    self.selection_targets.iter().map(|v| v.raw()).collect::<Vec<_>>();
                selections.sort_unstable();
                selections
            },
            root: self.root.raw(),
            root_step: self.root_step,
            sources: self.sources.clone(),
            steps: self
                .proof
                .iter()
                .map(|step| SidStepRecord {
                    outcomes: ids(&step.state.y),
                    interventions: ids(&step.state.x),
                    vertices: ids(&step.state.v),
                    kernel: step.state.kernel.raw(),
                    rule: step.rule,
                    children: step.children.clone(),
                    output: step.output.raw(),
                    parameters: step.parameters.iter().map(|v| v.raw()).collect(),
                })
                .collect(),
        }
    }

    /// Re-derive and check portable premises against externally supplied inputs.
    /// No identification search or provider access occurs.
    ///
    /// # Errors
    /// Altered evidence scope, input identities, malformed coordinates or proof.
    pub fn from_record_checked(
        record: SidDerivationRecord,
        arena: CausalExprArena,
        diagram: &SelectionDiagram,
        query: &ClassicalTransportQuery,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let bad = || IdentificationError::invalid_derivation("transport.invalid_derivation_record");
        if record.evidence_setting
            != if record.sources.is_empty() { CLASSICAL_SETTING } else { META_SETTING }
            || record.steps.len() > limits.steps
        {
            return Err(bad());
        }
        check_sid_memory(diagram, record.steps.len(), ctx)?;
        let set = |values: Vec<u32>| -> Result<BitSet, IdentificationError> {
            let mut result = BitSet::with_len(diagram.causal_graph().node_count());
            for value in values {
                let id = DenseNodeId::from_raw(value);
                if id.as_usize() >= diagram.causal_graph().node_count() || result.contains(id) {
                    return Err(bad());
                }
                result.insert(id);
            }
            Ok(result)
        };
        let proof = record
            .steps
            .into_iter()
            .map(|step| {
                Ok(ProofStep {
                    state: State {
                        y: set(step.outcomes)?,
                        x: set(step.interventions)?,
                        v: set(step.vertices)?,
                        kernel: ExprId::from_raw(step.kernel),
                    },
                    rule: step.rule,
                    children: step.children,
                    output: ExprId::from_raw(step.output),
                    parameters: step.parameters.into_iter().map(VariableId::from_raw).collect(),
                })
            })
            .collect::<Result<Vec<_>, IdentificationError>>()?;
        let result = Self {
            query: ClassicalTransportQuery {
                outcomes: record.outcomes.into_iter().map(VariableId::from_raw).collect(),
                treatments: record.treatments.into_iter().map(VariableId::from_raw).collect(),
                source: record.source.into(),
                target: record.target.into(),
            },
            graph_signature: record.graph_signature,
            selection_targets: record.selections.into_iter().map(VariableId::from_raw).collect(),
            arena,
            root: ExprId::from_raw(record.root),
            proof,
            root_step: record.root_step,
            sources: record.sources,
        };
        verify_classical_transport(diagram, query, &result, limits, ctx)?;
        Ok(result)
    }
}

/// Identify under the explicitly declared classical experimental information family.
/// Ordinary target ID is tried first. No finite-catalog completeness is implied.
///
/// # Errors
/// Invalid coordinates, resource exhaustion, cancellation, or failed verification.
pub fn identify_classical_transport(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<ClassicalTransportResult, IdentificationError> {
    identify_classical_transport_metered(diagram, query, SidMeter::Limits(limits), ctx)
}

/// [`identify_classical_transport`] charging `meter`: its own [`SidLimits`], or
/// a shared budget that the search and its verification replay both charge.
///
/// # Errors
/// As [`identify_classical_transport`]; under a shared budget, its stop.
pub(crate) fn identify_classical_transport_metered(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    meter: SidMeter<'_>,
    ctx: &ExecutionContext,
) -> Result<ClassicalTransportResult, IdentificationError> {
    let mut engine = Engine::new_metered(diagram, query, meter, ctx)?;
    let state = engine.initial()?;
    let mut result = engine.solve(state.clone(), false, 0)?;
    if result.is_none() {
        result = engine.solve(state, true, 0)?;
    }
    let Some(root_step) = result else {
        if let Some(state) = engine.obstruction.clone() {
            // Under a shared budget the witness construction (which checks its
            // own candidate) and the independent verification of the result are
            // each one charged operation at the engine's depth: both are linear
            // in the graph, so one operation apiece, but never free.
            engine.charge_shared(1)?;
            if let Some(witness) = engine.negative_witness(&state)? {
                engine.charge_shared(1)?;
                verify_s_hedge(diagram, query, &witness, ctx)?;
                return Ok(ClassicalTransportResult::ProvenNonTransportable(witness));
            }
        }
        return Ok(ClassicalTransportResult::NotCertified);
    };
    let derivation = engine.solved_derivation(root_step, Vec::new())?;
    Ok(ClassicalTransportResult::Identified(Box::new(derivation)))
}

struct Engine<'a> {
    diagram: &'a SelectionDiagram,
    query: &'a ClassicalTransportQuery,
    prepared: PreparedAdmg,
    /// Ancestral closures for this engine. The graph does not store them.
    ancestors: AncestorMemo,
    /// Topological order, shared so a factorization step borrows no table.
    topo: Arc<[DenseNodeId]>,
    /// Every node, for the meta-source relevance restriction.
    all: BitSet,
    arena: CausalExprArena,
    proof: Vec<ProofStep>,
    memo: HashMap<(State, bool), Option<usize>>,
    limits: SidLimits,
    /// The shared transport-search budget this engine charges, when it runs under one;
    /// `None` keeps the engine's own [`SidLimits`] accounting.
    shared: Option<&'a mut (dyn SearchCharge + 'a)>,
    /// Depth of this engine's root within the enclosing search, so a nested
    /// engine charges the shared budget at its true depth.
    depth_offset: usize,
    /// Live-state bytes of the enclosing engines a nested engine runs inside.
    base_bytes: u64,
    ctx: &'a ExecutionContext,
    steps: usize,
    /// Deepest recursion level `charge` has observed, for a limits receipt when
    /// a bounded search stops. It is the engine's own accounting, never a claim.
    max_depth: usize,
    /// Set when a multi-source line-11 state had another source able to
    /// exchange: completing the factor would need a joint over two sources'
    /// interventions, which the search refuses to fabricate.
    cross_source_joint: bool,
    obstruction: Option<State>,
    source_catalog: Option<&'a antecedent_core::EvidenceCatalog>,
    sources: Vec<MetaSource>,
}

/// Durable identity of a selection diagram, stored in every portable proof.
///
/// The encoding is a fixed text grammar over raw ids, never a `Debug`
/// rendering: `sig.v2;nodes=<raw ids in dense order>;directed=<from>-<to>,…;
/// bidirected=<a>-<b>,…;selections=<sorted raw ids>`, with every edge list
/// sorted. A proof recorded under the earlier `Debug`-based signature no
/// longer verifies and must be re-derived.
pub(crate) fn graph_signature(diagram: &SelectionDiagram) -> String {
    let graph = diagram.causal_graph();
    let nodes = graph
        .nodes()
        .iter()
        .map(|node| match node {
            antecedent_graph::NodeRef::Static(variable) => variable.raw().to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>();
    let directed = graph.directed_edge_pairs_sorted();
    let bidirected = graph.bidirected_edge_pairs_sorted();
    let mut selections = diagram.selection_targets().iter().map(|v| v.raw()).collect::<Vec<_>>();
    selections.sort_unstable();
    let edges = |list: &[(u32, u32)]| {
        list.iter().map(|(a, b)| format!("{a}-{b}")).collect::<Vec<_>>().join(",")
    };
    format!(
        "sig.v2;nodes={};directed={};bidirected={};selections={}",
        nodes.join(","),
        edges(&directed),
        edges(&bidirected),
        selections.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
    )
}

/// Certified functional with supplied-catalog leaf bindings. Original derivation
/// remains separate from the provider-bound expression.
#[derive(Clone, Debug)]
pub struct BoundTransportFunctional {
    derivation: ClassicalTransportDerivation,
    arena: CausalExprArena,
    root: ExprId,
    catalog: antecedent_core::EvidenceCatalog,
    searched: Arc<[Arc<str>]>,
}

impl BoundTransportFunctional {
    /// Original checked theorem derivation.
    #[must_use]
    pub const fn derivation(&self) -> &ClassicalTransportDerivation {
        &self.derivation
    }
    /// Provider-bound expression arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Bound distribution root.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
    /// Deterministic alternative strategies attempted before this binding.
    #[must_use]
    pub fn searched_alternatives(&self) -> &[Arc<str>] {
        &self.searched
    }
    /// Frozen evidence contract.
    #[must_use]
    pub const fn catalog(&self) -> &antecedent_core::EvidenceCatalog {
        &self.catalog
    }
    /// The distinct distribution leaves of the bound formula, each with its
    /// expression id and signature, in traversal order. Every consumer that
    /// walks the formula for its factors reads this one view.
    #[must_use]
    pub fn leaf_factors(&self) -> Vec<(ExprId, antecedent_expr::LeafSignature)> {
        self.arena
            .distribution_leaves(self.root)
            .into_iter()
            .zip(self.arena.leaf_signatures(self.root))
            .collect()
    }
}

impl ClassicalTransportDerivation {
    fn validate_catalog_contract(
        &self,
        catalog: &antecedent_core::EvidenceCatalog,
    ) -> Result<(), IdentificationError> {
        for environment in catalog.environments.iter().filter(|e| e.identity == self.query.source) {
            let mut declared = environment.selection_targets.to_vec();
            declared.sort_unstable();
            let mut identified = self.selection_targets.to_vec();
            identified.sort_unstable();
            if declared != identified {
                return Err(IdentificationError::invalid_catalog(
                    "transport.invalid_input: catalog mechanism selections disagree with the checked diagram",
                ));
            }
        }
        if !self.sources.is_empty() {
            let target =
                catalog.environments.iter().find(|e| e.identity == self.query.target).ok_or_else(
                    || {
                        IdentificationError::invalid_catalog(
                            "transport.invalid_input: missing meta target environment",
                        )
                    },
                )?;
            if !target.selection_targets.is_empty() {
                return Err(IdentificationError::invalid_catalog(
                    "transport.invalid_input: target environment cannot declare selection differences from itself",
                ));
            }
            let ExprNode::Distribution { variables, .. } =
                self.arena.node(self.proof[self.root_step].state.kernel)
            else {
                return Err(IdentificationError::InvariantViolated {
                    message: "invalid initial transport kernel",
                });
            };
            let coordinates = self.arena.var_set(*variables);
            if catalog.environments.iter().any(|e| {
                e.variables.iter().any(|v| !coordinates.contains(&v.variable))
                    || e.selection_targets.iter().any(|v| !coordinates.contains(v))
            }) {
                return Err(IdentificationError::invalid_catalog(
                    "transport.invalid_input: catalog coordinate outside shared causal graph",
                ));
            }
        }
        for source in &self.sources {
            let environment = catalog
                .environments
                .iter()
                .find(|e| e.identity.as_ref() == source.population)
                .ok_or_else(|| {
                    IdentificationError::invalid_catalog(
                        "transport.invalid_input: missing meta source environment",
                    )
                })?;
            let mut declared: Vec<_> =
                environment.selection_targets.iter().map(|v| v.raw()).collect();
            declared.sort_unstable();
            if declared != source.selections {
                return Err(IdentificationError::invalid_catalog(
                    "transport.invalid_input: meta source selections disagree with checked proof",
                ));
            }
        }
        Ok(())
    }

    /// Bind this derivation to available, joint, unrestricted catalog laws.
    /// Failure means this particular formula cannot be bound, not that no
    /// alternative catalog-supported formula exists.
    ///
    /// # Errors
    /// Invalid catalog or an unavailable population/regime/measurement obligation.
    pub fn bind_catalog(
        &self,
        catalog: &antecedent_core::EvidenceCatalog,
    ) -> Result<BoundTransportFunctional, IdentificationError> {
        catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
        self.bind_catalog_validated(catalog)
    }

    /// [`Self::bind_catalog`] for a catalog the caller has already validated,
    /// so a search that binds many candidate derivations validates once.
    pub(crate) fn bind_catalog_validated(
        &self,
        catalog: &antecedent_core::EvidenceCatalog,
    ) -> Result<BoundTransportFunctional, IdentificationError> {
        fn bind(
            id: ExprId,
            source: &CausalExprArena,
            arena: &mut CausalExprArena,
            catalog: &antecedent_core::EvidenceCatalog,
            target: &str,
            memo: &mut HashMap<ExprId, ExprId>,
        ) -> Result<ExprId, IdentificationError> {
            if let Some(hit) = memo.get(&id) {
                return Ok(*hit);
            }
            let node = match source.node(id).clone() {
                ExprNode::Distribution { .. } => bind_distribution(id, source, catalog, target)?,
                ExprNode::Product(list) => {
                    let children = source
                        .list(list)
                        .iter()
                        .map(|id| bind(*id, source, arena, catalog, target, memo))
                        .collect::<Result<Vec<_>, _>>()?;
                    ExprNode::Product(arena.intern_list(children))
                }
                ExprNode::SumOut { variables, expr } => ExprNode::SumOut {
                    variables,
                    expr: bind(expr, source, arena, catalog, target, memo)?,
                },
                ExprNode::IntegralOut { variables, expr } => ExprNode::IntegralOut {
                    variables,
                    expr: bind(expr, source, arena, catalog, target, memo)?,
                },
                ExprNode::Ratio { numerator, denominator } => ExprNode::Ratio {
                    numerator: bind(numerator, source, arena, catalog, target, memo)?,
                    denominator: bind(denominator, source, arena, catalog, target, memo)?,
                },
                ExprNode::Kernel { body, bound, population, regime } => ExprNode::Kernel {
                    body: bind(body, source, arena, catalog, target, memo)?,
                    bound,
                    population,
                    regime,
                },
                ExprNode::Expectation { function, distribution } => ExprNode::Expectation {
                    function,
                    distribution: bind(distribution, source, arena, catalog, target, memo)?,
                },
                ExprNode::Contrast { left, right, op } => ExprNode::Contrast {
                    left: bind(left, source, arena, catalog, target, memo)?,
                    right: bind(right, source, arena, catalog, target, memo)?,
                    op,
                },
            };
            let result = arena.intern(node);
            memo.insert(id, result);
            Ok(result)
        }
        self.validate_catalog_contract(catalog)?;
        let mut arena = self.arena.clone();
        let mut memo = HashMap::new();
        let root =
            bind(self.root, &self.arena, &mut arena, catalog, &self.query.target, &mut memo)?;
        Ok(BoundTransportFunctional {
            derivation: self.clone(),
            arena,
            root,
            catalog: catalog.clone(),
            searched: Arc::from([Arc::from("bind_selected_derivation")]),
        })
    }
}

/// A directed/bidirected edge subgraph used in a negative witness.
#[derive(Clone, Debug)]
pub struct SelectionForest {
    /// Original graph coordinates.
    pub nodes: Arc<[VariableId]>,
    /// Directed edges; each node has at most one child.
    pub directed: Arc<[(VariableId, VariableId)]>,
    /// Bidirected edges forming one connected district.
    pub bidirected: Arc<[(VariableId, VariableId)]>,
}

/// Classical single-source obstruction, bound to its exact immutable inputs.
#[derive(Clone, Debug)]
pub struct SHedgeCertificate {
    query: ClassicalTransportQuery,
    graph_signature: String,
    sources: Vec<MetaSource>,
    /// Larger selected forest intersecting the intervention set.
    pub larger: SelectionForest,
    /// Nested selected forest disjoint from interventions, with the same roots.
    pub smaller: SelectionForest,
}

/// Untrusted portable forest edge subgraph, in original variable coordinates.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
pub struct SelectionForestRecord {
    pub nodes: Vec<u32>,
    pub directed: Vec<(u32, u32)>,
    pub bidirected: Vec<(u32, u32)>,
}

impl SelectionForestRecord {
    fn encode(forest: &SelectionForest) -> Self {
        let edges = |edges: &[(VariableId, VariableId)]| {
            edges.iter().map(|(a, b)| (a.raw(), b.raw())).collect()
        };
        Self {
            nodes: forest.nodes.iter().map(|v| v.raw()).collect(),
            directed: edges(&forest.directed),
            bidirected: edges(&forest.bidirected),
        }
    }
    fn decode(self) -> SelectionForest {
        let edges = |edges: Vec<(u32, u32)>| {
            edges
                .into_iter()
                .map(|(a, b)| (VariableId::from_raw(a), VariableId::from_raw(b)))
                .collect()
        };
        SelectionForest {
            nodes: self.nodes.into_iter().map(VariableId::from_raw).collect(),
            directed: edges(self.directed),
            bidirected: edges(self.bidirected),
        }
    }
}

/// Portable negative claim. Deserialization conveys no authority; its forest,
/// query, graph and full-source-experimental evidence scope must all check.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    missing_docs,
    reason = "wire record whose field names are the format and are documented on the domain types"
)]
pub struct SHedgeRecord {
    pub evidence_setting: String,
    pub source: String,
    pub target: String,
    pub outcomes: Vec<u32>,
    pub treatments: Vec<u32>,
    pub graph_signature: String,
    pub larger: SelectionForestRecord,
    pub smaller: SelectionForestRecord,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<MetaSource>,
}

impl SHedgeCertificate {
    /// Export a checked obstruction with its exact evidence and query scope.
    #[must_use]
    pub fn to_record(&self) -> SHedgeRecord {
        SHedgeRecord {
            evidence_setting: if self.sources.is_empty() {
                CLASSICAL_SETTING
            } else {
                META_SETTING
            }
            .into(),
            source: self.query.source.to_string(),
            target: self.query.target.to_string(),
            outcomes: self.query.outcomes.iter().map(|v| v.raw()).collect(),
            treatments: self.query.treatments.iter().map(|v| v.raw()).collect(),
            graph_signature: self.graph_signature.clone(),
            larger: SelectionForestRecord::encode(&self.larger),
            smaller: SelectionForestRecord::encode(&self.smaller),
            sources: self.sources.clone(),
        }
    }
    /// Independently check an untrusted forest record against immutable inputs.
    ///
    /// # Errors
    /// Altered premises, unsupported evidence setting, or budget/cancellation.
    pub fn from_record_checked(
        record: SHedgeRecord,
        diagram: &SelectionDiagram,
        query: &ClassicalTransportQuery,
        ctx: &ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        if record.evidence_setting
            != if record.sources.is_empty() { CLASSICAL_SETTING } else { META_SETTING }
        {
            return Err(IdentificationError::invalid_derivation(
                "transport.invalid_s_hedge_evidence",
            ));
        }
        check_sid_memory(diagram, 1, ctx)?;
        let witness = Self {
            query: ClassicalTransportQuery {
                outcomes: record.outcomes.into_iter().map(VariableId::from_raw).collect(),
                treatments: record.treatments.into_iter().map(VariableId::from_raw).collect(),
                source: record.source.into(),
                target: record.target.into(),
            },
            graph_signature: record.graph_signature,
            larger: record.larger.decode(),
            smaller: record.smaller.decode(),
            sources: record.sources,
        };
        verify_s_hedge(diagram, query, &witness, ctx)?;
        Ok(witness)
    }
}

mod catalog;

use catalog::bind_distribution;

mod engine;
mod verify;
use engine::{difference, intersection, standardization_binds};
use verify::verify_charged;
pub use verify::{verify_classical_transport, verify_s_hedge};

pub(crate) use catalog::identify_catalog_transport_metered;

pub use catalog::{
    CatalogTransportResult, identify_catalog_transport,
    identify_catalog_transport_selection_standardization_first,
    identify_catalog_transport_standardization_first,
};

fn check_sid_memory(
    diagram: &SelectionDiagram,
    steps: usize,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    refuse_over_budget(sid_memory_bytes(diagram, steps), ctx, IdentificationBudget::Memory)
}

/// Live-state estimate of an engine that has charged `steps` subproblems.
fn sid_memory_bytes(diagram: &SelectionDiagram, steps: usize) -> usize {
    // Carried chain-rule kernels have quadratic coordinate storage; a linear
    // node-count charge would substantially understate recursive factorization.
    let n = diagram.causal_graph().node_count();
    n.saturating_mul(n).saturating_mul(64).saturating_add(512).saturating_mul(steps)
}

/// A shared bounded-search budget that engines charge instead of their own
/// [`SidLimits`]. It is a trait object so nested engines, whose graphs live
/// shorter than the budget's context, can reborrow the one budget.
pub(crate) trait SearchCharge {
    /// Charge one operation at `depth` whose live state needs `bytes`.
    fn charge(&mut self, depth: usize, bytes: u64) -> Result<(), SearchStop>;
    /// Limits in force.
    fn limits(&self) -> SearchLimits;
    /// Record a region a stop left unevaluated.
    fn defer(&mut self, region: String);
}

/// One [`SearchBudget`] charged by every search stage of a decision, with the
/// bound that stopped it and the regions the stop left unevaluated.
///
/// A decision creates one, runs each stage's engines on
/// [`SidMeter::Shared`] reborrows of it, and on a budget or cancellation error
/// builds its receipt with [`Self::receipt`]. The receipt then reports the
/// operations and depth consumed across all stages.
pub(crate) struct SharedSearch<'c> {
    budget: SearchBudget<'c>,
    stop: Option<SearchStop>,
    unevaluated: Vec<String>,
    /// Live bytes retained by work already finished under this budget (for a
    /// scenario set, the earlier scenarios' results): added to every charge so
    /// memory is cumulative across the decision, not restarted per engine.
    base_bytes: u64,
    /// Largest live-state estimate charged since the last [`Self::begin`]: the
    /// charge plus the stages retained since then, excluding the `retained`
    /// bytes `begin` was given.
    peak_bytes: u64,
    /// Largest live-state estimate charged since [`Self::begin_stage`], for
    /// stage-to-stage retention.
    stage_peak: u64,
    /// Bytes retained by the finished stages of the current unit of work.
    stages_retained: u64,
    /// Deepest level charged since [`Self::mark_decision`].
    decision_depth: usize,
    /// Test hook: cancel the token once this many operations were charged,
    /// so a test observes cancellation in the middle of a decision.
    #[cfg(test)]
    pub(crate) cancel_after: Option<(usize, antecedent_core::CancellationToken)>,
}

impl<'c> SharedSearch<'c> {
    /// Share `budget` across every stage of one decision.
    pub(crate) const fn new(budget: SearchBudget<'c>) -> Self {
        Self {
            budget,
            stop: None,
            unevaluated: Vec::new(),
            base_bytes: 0,
            peak_bytes: 0,
            stage_peak: 0,
            stages_retained: 0,
            decision_depth: 0,
            #[cfg(test)]
            cancel_after: None,
        }
    }
    /// Start a unit of work on top of `retained` live bytes of finished work.
    pub(crate) fn begin(&mut self, retained: u64) {
        self.base_bytes = retained;
        self.peak_bytes = 0;
        self.stages_retained = 0;
    }
    /// The largest live-state estimate the unit charged since [`Self::begin`]:
    /// what finished work keeps holding, as an upper bound.
    pub(crate) const fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }
    /// Start one search stage of a multi-stage decision.
    pub(crate) const fn begin_stage(&mut self) {
        self.stage_peak = 0;
    }
    /// Finish a stage whose derivations the decision keeps: its largest
    /// live-state estimate is retained by every later charge, so live bytes
    /// accumulate across the stages of one decision instead of restarting.
    pub(crate) const fn end_stage(&mut self) {
        self.base_bytes = self.base_bytes.saturating_add(self.stage_peak);
        self.stages_retained = self.stages_retained.saturating_add(self.stage_peak);
        self.stage_peak = 0;
    }
    /// Charge one operation like [`SearchCharge::charge`], then lease `lease_bytes` of storage
    /// (when non-zero) from the context's admission pool before the caller allocates it. A
    /// refused lease is a memory (or cancellation) stop and is recorded like any other.
    pub(crate) fn charge_leasing(
        &mut self,
        depth: usize,
        bytes: u64,
        lease_bytes: u64,
    ) -> Result<Option<antecedent_core::execution::AdmissionLease>, SearchStop> {
        SearchCharge::charge(self, depth, bytes)?;
        if lease_bytes == 0 {
            return Ok(None);
        }
        self.budget.lease_storage(lease_bytes).map(Some).inspect_err(|stop| {
            self.stop.get_or_insert(*stop);
        })
    }
    /// Operations the budget still allows.
    pub(crate) fn operations_remaining(&self) -> u64 {
        u64::try_from(self.budget.operations_remaining()).unwrap_or(u64::MAX)
    }
    /// Operations charged so far, across every decision on this budget.
    pub(crate) const fn operations(&self) -> usize {
        self.budget.operations()
    }
    /// Effective memory cap of the budget.
    pub(crate) const fn memory_limit_bytes(&self) -> u64 {
        self.budget.memory_limit_bytes()
    }
    /// Start measuring the depth one decision reaches.
    pub(crate) const fn mark_decision(&mut self) {
        self.decision_depth = 0;
    }
    /// Deepest level charged since [`Self::mark_decision`].
    pub(crate) const fn decision_depth(&self) -> usize {
        self.decision_depth
    }
    /// A meter charging this budget.
    pub(crate) fn meter(&mut self) -> SidMeter<'_> {
        SidMeter::Shared(self)
    }
    /// The bound that stopped the search: the first stop the budget reported,
    /// else the one `error` names (a check made outside a charge).
    pub(crate) fn stop_of(&self, error: &IdentificationError) -> SearchStop {
        self.stop.unwrap_or(match error {
            IdentificationError::Cancelled => SearchStop::Cancelled,
            IdentificationError::Budget {
                budget: IdentificationBudget::Memory | IdentificationBudget::BindingMemory,
            } => SearchStop::Memory,
            _ => SearchStop::Operations,
        })
    }
    /// The receipt of a stop: `explored` regions, then `unevaluated` ones
    /// followed by every region the engines deferred.
    pub(crate) fn receipt(
        &self,
        stop: SearchStop,
        explored: Vec<String>,
        mut unevaluated: Vec<String>,
    ) -> SearchReceipt {
        unevaluated.extend(self.unevaluated.iter().cloned());
        self.budget.receipt(stop, explored, unevaluated)
    }
}

impl SearchCharge for SharedSearch<'_> {
    fn charge(&mut self, depth: usize, bytes: u64) -> Result<(), SearchStop> {
        #[cfg(test)]
        if let Some((after, token)) = &self.cancel_after {
            if self.budget.operations() >= *after {
                token.cancel();
            }
        }
        self.peak_bytes = self.peak_bytes.max(bytes.saturating_add(self.stages_retained));
        self.stage_peak = self.stage_peak.max(bytes);
        self.decision_depth = self.decision_depth.max(depth);
        self.budget.charge(depth, bytes.saturating_add(self.base_bytes)).inspect_err(|stop| {
            self.stop.get_or_insert(*stop);
        })
    }
    fn limits(&self) -> SearchLimits {
        self.budget.limits()
    }
    fn defer(&mut self, region: String) {
        self.unevaluated.push(region);
    }
}

/// What an engine charges: its own [`SidLimits`] (every legacy caller, with
/// unchanged semantics and errors), or a shared [`SearchCharge`] budget.
pub(crate) enum SidMeter<'m> {
    /// The engine's own step and depth limits; `depth >= limit` stops.
    Limits(SidLimits),
    /// A budget shared across engines; `depth > limit` stops.
    Shared(&'m mut (dyn SearchCharge + 'm)),
}

impl SidMeter<'_> {
    /// A shorter-lived meter charging the same limits or budget.
    pub(crate) fn reborrow(&mut self) -> SidMeter<'_> {
        match self {
            Self::Limits(limits) => SidMeter::Limits(*limits),
            Self::Shared(shared) => SidMeter::Shared(&mut **shared),
        }
    }
    /// Limits in force, as step and depth limits.
    pub(crate) fn limits(&self) -> SidLimits {
        match self {
            Self::Limits(limits) => *limits,
            Self::Shared(shared) => {
                let limits = shared.limits();
                SidLimits { steps: limits.operations, depth: limits.depth }
            }
        }
    }
}

/// The identification error a shared-budget stop surfaces as.
const fn stop_error(stop: SearchStop) -> IdentificationError {
    match stop {
        SearchStop::Cancelled => IdentificationError::Cancelled,
        SearchStop::Memory => IdentificationError::budget(IdentificationBudget::Memory),
        SearchStop::Operations | SearchStop::Depth => {
            IdentificationError::budget(IdentificationBudget::Steps)
        }
    }
}

/// The one memory-budget check of the transport identifiers: cancellation
/// first, then `bytes` against the context's hard limit, refusing with `budget`.
pub(crate) fn refuse_over_budget(
    bytes: usize,
    ctx: &ExecutionContext,
    budget: IdentificationBudget,
) -> Result<(), IdentificationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    if ctx
        .memory_limit_bytes()
        .is_some_and(|limit| u64::try_from(bytes).map_or(true, |bytes| bytes > limit))
    {
        return Err(IdentificationError::budget(budget));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
