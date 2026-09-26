//! Classical single-source sID, scoped to access to all source experiments.
//!
//! Rules follow Bareinboim and Pearl (2013), Figure 5. Catalog binding is a
//! separate operation: failure to bind a leaf is not a nontransportability proof.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::selection_separation::{
    MutilatedSelection, SubsetSearchEnd, for_each_admissible_subset, independently_separated,
};
use crate::{IdentificationBudget, IdentificationError, PreparedAdmg};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{
    Admg, BitSet, DSeparationWorkspace, DenseNodeId, GraphWorkspace, SelectionDiagram,
};
use std::collections::HashMap;
use std::sync::Arc;

mod meta;
mod z_transport;
use meta::{CLASSICAL_SETTING, META_SETTING, validate_meta_sources};
pub use meta::{
    CheckedTransportDerivation, MetaSource, MetaTransportQuery, identify_meta_catalog,
    identify_meta_transport, verify_meta_s_hedge, verify_meta_transport,
};
pub use z_transport::{
    BoundZTransportFunctional, TwoSourceZTransportComponent, TwoSourceZTransportDecision,
    TwoSourceZTransportQuery, Z_TRANSPORT_MAX_CONTROLLABLE, Z_TRANSPORT_MAX_FAMILY_REGIMES,
    Z_TRANSPORT_MAX_OBSERVED, ZExperimentFamilyError, ZFactorObligation, ZProofOperation,
    ZTransportDecision, ZTransportDerivation, ZTransportDerivationRecord,
    ZTransportMissingEvidence, ZTransportObstruction, ZTransportObstructionRecord,
    ZTransportProofInspection, ZTransportQuery, ZTransportResult, ZTransportSourceSpec,
    ZTransportTerminalRecord, bind_z_transport_catalog, decide_two_source_z_transport,
    decide_z_transport_with_catalog, identify_z_transport, validate_z_experiment_family,
    validate_z_transport_query, verify_z_transport_derivation, verify_z_transport_obstruction,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
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
#[allow(missing_docs)]
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
#[allow(missing_docs)]
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
    let mut engine = Engine::new(diagram, query, limits, ctx)?;
    let state = engine.initial()?;
    let mut result = engine.solve(state.clone(), false, 0)?;
    if result.is_none() {
        result = engine.solve(state, true, 0)?;
    }
    let Some(root_step) = result else {
        if let Some(state) = engine.obstruction.clone() {
            if let Some(witness) = engine.negative_witness(&state)? {
                verify_s_hedge(diagram, query, &witness, ctx)?;
                return Ok(ClassicalTransportResult::ProvenNonTransportable(witness));
            }
        }
        return Ok(ClassicalTransportResult::NotCertified);
    };
    let derivation = engine.solved_derivation(root_step, Vec::new())?;
    Ok(ClassicalTransportResult::Identified(Box::new(derivation)))
}

impl Engine<'_> {
    /// The verified derivation rooted at `root_step`: the reachable proof steps
    /// with the engine's arena, exported for the sources the search declared.
    /// Every identifier builds its positive result through this one path.
    pub(crate) fn solved_derivation(
        &self,
        root_step: usize,
        sources: Vec<MetaSource>,
    ) -> Result<ClassicalTransportDerivation, IdentificationError> {
        let (proof, root_step) = reachable_proof(&self.proof, root_step);
        let derivation = ClassicalTransportDerivation {
            query: self.query.clone(),
            graph_signature: graph_signature(self.diagram),
            selection_targets: Arc::from(self.diagram.selection_targets()),
            root: proof[root_step].output,
            arena: self.arena.clone(),
            proof,
            root_step,
            sources,
        };
        verify_classical_transport(self.diagram, self.query, &derivation, self.limits, self.ctx)?;
        Ok(derivation)
    }
}

/// The proof steps reachable from `root_step`, in their original order, with
/// child indexes remapped. Steps a cleared memo left behind are not exported.
fn reachable_proof(proof: &[ProofStep], root_step: usize) -> (Vec<ProofStep>, usize) {
    let mut used = vec![false; proof.len()];
    let mut pending = vec![root_step];
    while let Some(step) = pending.pop() {
        if step >= used.len() || used[step] {
            continue;
        }
        used[step] = true;
        pending.extend(proof[step].children.iter().copied());
    }
    let mut remap = vec![usize::MAX; proof.len()];
    let mut compact = Vec::new();
    for (index, step) in proof.iter().enumerate() {
        if used[index] {
            remap[index] = compact.len();
            let mut step = step.clone();
            step.children = step.children.iter().map(|child| remap[*child]).collect();
            compact.push(step);
        }
    }
    (compact, remap[root_step])
}

/// Whether the two leaves of a standardization over `over` bind in `catalog`:
/// the source conditional `P^s(y | over, do(x))` and the target law `P*(over)`.
fn standardization_binds(
    catalog: &antecedent_core::EvidenceCatalog,
    query: &ClassicalTransportQuery,
    outcomes: &[VariableId],
    treatments: &[VariableId],
    over: &[VariableId],
) -> bool {
    let source = antecedent_core::FactorNeed {
        population: &query.source,
        variables: outcomes,
        conditioned_on: over,
        interventions: treatments,
    };
    if catalog.satisfying_regime(&source).is_none() {
        return false;
    }
    over.is_empty()
        || (catalog
            .target_sampling
            .is_none_or(antecedent_core::TargetSampling::represents_target_law)
            && catalog
                .satisfying_regime(&antecedent_core::FactorNeed {
                    population: &query.target,
                    variables: over,
                    conditioned_on: &[],
                    interventions: &[],
                })
                .is_some())
}

struct Engine<'a> {
    diagram: &'a SelectionDiagram,
    query: &'a ClassicalTransportQuery,
    prepared: PreparedAdmg,
    /// Topological order, shared so a factorization step borrows no table.
    topo: Arc<[DenseNodeId]>,
    /// Every node, for the meta-source relevance restriction.
    all: BitSet,
    arena: CausalExprArena,
    proof: Vec<ProofStep>,
    memo: HashMap<(State, bool), Option<usize>>,
    limits: SidLimits,
    ctx: &'a ExecutionContext,
    steps: usize,
    obstruction: Option<State>,
    source_catalog: Option<&'a antecedent_core::EvidenceCatalog>,
    sources: Vec<MetaSource>,
}
impl<'a> Engine<'a> {
    fn new(
        diagram: &'a SelectionDiagram,
        query: &'a ClassicalTransportQuery,
        limits: SidLimits,
        ctx: &'a ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        check_sid_memory(diagram, 1, ctx)?;
        let prepared = PreparedAdmg::new(diagram.causal_graph().clone())?;
        if query.outcomes.is_empty()
            || query.source.is_empty()
            || query.target.is_empty()
            || query.source == query.target
        {
            return Err(IdentificationError::invalid_input(
                "transport.invalid_query: invalid classical transport query",
            ));
        }
        for variables in [&query.outcomes, &query.treatments] {
            for (i, v) in variables.iter().enumerate() {
                prepared.var_to_dense(*v)?;
                if variables[..i].contains(v) {
                    return Err(IdentificationError::invalid_input(
                        "transport.invalid_query: duplicate classical query coordinate",
                    ));
                }
            }
        }
        if query.outcomes.iter().any(|v| query.treatments.contains(v)) {
            return Err(IdentificationError::invalid_input(
                "transport.invalid_query: outcomes overlap interventions",
            ));
        }
        let topo: Arc<[DenseNodeId]> = Arc::from(prepared.topo());
        let mut all = BitSet::with_len(diagram.causal_graph().node_count());
        for node in topo.iter() {
            all.insert(*node);
        }
        Ok(Self {
            diagram,
            query,
            prepared,
            topo,
            all,
            arena: CausalExprArena::new(),
            proof: Vec::new(),
            memo: HashMap::new(),
            limits,
            ctx,
            steps: 0,
            obstruction: None,
            source_catalog: None,
            sources: Vec::new(),
        })
    }
    fn charge(&mut self, depth: usize) -> Result<(), IdentificationError> {
        self.steps = self.steps.saturating_add(1);
        if self.ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        if self.steps > self.limits.steps || depth >= self.limits.depth {
            return Err(IdentificationError::budget(IdentificationBudget::Steps));
        }
        check_sid_memory(self.diagram, self.steps, self.ctx)?;
        Ok(())
    }
    fn set(&self, variables: &[VariableId]) -> Result<BitSet, IdentificationError> {
        let mut set = BitSet::with_len(self.diagram.causal_graph().node_count());
        for v in variables {
            set.insert(self.prepared.var_to_dense(*v)?);
        }
        Ok(set)
    }
    fn vars(&self, set: &BitSet) -> Result<Vec<VariableId>, IdentificationError> {
        set.to_dense_ids().into_iter().map(|v| self.prepared.dense_to_var(v)).collect()
    }
    fn initial(&mut self) -> Result<State, IdentificationError> {
        let mut v = BitSet::with_len(self.diagram.causal_graph().node_count());
        for d in self.prepared.topo() {
            v.insert(*d);
        }
        let variables = self.arena.intern_var_set(self.vars(&v)?);
        let conditioned_on = self.arena.empty_var_set();
        let intervention = self.arena.intern_intervention_set([]);
        let population = self.arena.intern_population(self.query.target.clone());
        let kernel = self.arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: DomainRef::Observational,
            population,
            regime: None,
        });
        Ok(State {
            y: self.set(&self.query.outcomes)?,
            x: self.set(&self.query.treatments)?,
            v,
            kernel,
        })
    }
    /// Initial state whose kernel is a source law under the concrete or
    /// symbolic interventions `assignments`, over every node of the diagram.
    /// The z-transport exchange identifies a source c-factor from it.
    pub(super) fn initial_source_kernel(
        &mut self,
        population: &Arc<str>,
        assignments: &[antecedent_expr::InterventionAssignment],
    ) -> Result<State, IdentificationError> {
        let mut v = BitSet::with_len(self.diagram.causal_graph().node_count());
        for d in self.prepared.topo() {
            v.insert(*d);
        }
        let variables = self.arena.intern_var_set(self.vars(&v)?);
        let conditioned_on = self.arena.empty_var_set();
        let intervention = self.arena.intern_intervention_assignments(assignments.iter().cloned());
        let population = self.arena.intern_population(Arc::clone(population));
        let kernel = self.arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: DomainRef::Interventional,
            population,
            regime: None,
        });
        Ok(State {
            y: self.set(&self.query.outcomes)?,
            x: self.set(&self.query.treatments)?,
            v,
            kernel,
        })
    }
    /// Rule names of the proof steps reachable from `root_step`, in step order.
    pub(super) fn reachable_rules(&self, root_step: usize) -> Vec<&'static str> {
        let mut used = vec![false; self.proof.len()];
        let mut pending = vec![root_step];
        while let Some(step) = pending.pop() {
            if step >= used.len() || used[step] {
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
    fn marginal(
        &mut self,
        expression: ExprId,
        remove: &BitSet,
    ) -> Result<ExprId, IdentificationError> {
        if !remove.any() {
            return Ok(expression);
        }
        let variables = self.arena.intern_var_set(self.vars(remove)?);
        Ok(self.arena.intern(ExprNode::SumOut { variables, expr: expression }))
    }
    fn product(&mut self, expressions: Vec<ExprId>) -> ExprId {
        let list = self.arena.intern_list(expressions);
        self.arena.intern(ExprNode::Product(list))
    }
    fn factor(&mut self, state: &State, district: &BitSet) -> Result<ExprId, IdentificationError> {
        let mut suffix = state.v.clone();
        let mut factors = Vec::new();
        let topological_order = Arc::clone(&self.topo);
        for v in topological_order.iter().copied() {
            if !state.v.contains(v) {
                continue;
            }
            suffix.remove(v);
            if district.contains(v) {
                // P(v | predecessors) = N / Σ_v N. The denominator literally
                // marginalizes the numerator, which is what lets the exact
                // evaluator treat a null-event conditional as a bounded
                // extension rather than an undefined 0/0.
                let numerator = self.marginal(state.kernel, &suffix)?;
                let mut summed = BitSet::with_len(self.diagram.causal_graph().node_count());
                summed.insert(v);
                let denominator = self.marginal(numerator, &summed)?;
                factors.push(self.arena.intern(ExprNode::Ratio { numerator, denominator }));
            }
        }
        Ok(self.product(factors))
    }
    /// Line 3 result. Rule 3 makes the child functional constant in `w`, so any
    /// normalized weight over `w` returns it without leaving a free parameter.
    /// The weight is `P(w | x)` from the carried kernel: it is zero exactly where
    /// the child is a conditional on a null event, so the answer never requires
    /// the child to be defined at a level of `w` that the data cannot reach
    /// under the intervention values being asked about.
    fn enlarge_output(
        &mut self,
        state: &State,
        w: &BitSet,
        child: ExprId,
    ) -> Result<ExprId, IdentificationError> {
        let joint = self.marginal(state.kernel, &difference(&difference(&state.v, &state.x), w))?;
        let denominator = self.marginal(joint, w)?;
        let weight = self.arena.intern(ExprNode::Ratio { numerator: joint, denominator });
        let product = self.product(vec![child, weight]);
        self.marginal(product, w)
    }
    fn source(&mut self, state: &State) -> Result<ExprId, IdentificationError> {
        let population = self.query.source.clone();
        self.source_from(state, population)
    }
    fn source_from(
        &mut self,
        state: &State,
        population: Arc<str>,
    ) -> Result<ExprId, IdentificationError> {
        // A district kernel fixes its external parents. Nonparents in the
        // original complement cannot affect it after those parents are fixed;
        // retaining them would demand irrelevant experimental coordinates.
        let mut interventions = state.x.clone();
        for node in state.v.to_dense_ids() {
            for parent in self.diagram.causal_graph().parents(node) {
                if !state.v.contains(*parent) {
                    interventions.insert(*parent);
                }
            }
        }
        if !self.sources.is_empty() {
            let mut ws = GraphWorkspace::default();
            let relevant =
                self.prepared.ancestors_bar_x(&state.y, &self.all, &interventions, &mut ws);
            interventions.intersect_with(&relevant);
        }
        let variables = self.arena.intern_var_set(self.vars(&state.y)?);
        let domain =
            if interventions.any() { DomainRef::Interventional } else { DomainRef::Observational };
        let intervention = self.arena.intern_intervention_set(self.vars(&interventions)?);
        let conditioned_on = self.arena.empty_var_set();
        let population = self.arena.intern_population(population);
        Ok(self.arena.intern(ExprNode::Distribution {
            variables,
            intervention,
            conditioned_on,
            domain,
            population,
            regime: None,
        }))
    }
    fn source_admissible(&self, state: &State) -> Result<bool, IdentificationError> {
        self.source_admissible_given(state, &[])
    }
    fn source_admissible_given(
        &self,
        state: &State,
        given: &[VariableId],
    ) -> Result<bool, IdentificationError> {
        self.admissible_selections(state, given, self.diagram.selection_targets())
    }
    fn admissible_selections(
        &self,
        state: &State,
        given: &[VariableId],
        selections: &[VariableId],
    ) -> Result<bool, IdentificationError> {
        let targets = self.dense_all(selections)?;
        let selection =
            MutilatedSelection::build(self.diagram.causal_graph(), &state.v, &state.x, &targets)?;
        let mut conditions = state.x.to_dense_ids();
        conditions.extend(self.dense_all(given)?);
        selection.separates(
            &state.y.to_dense_ids(),
            &conditions,
            &mut DSeparationWorkspace::default(),
        )
    }
    /// Second implementation of [`Self::admissible_selections`] for the checker:
    /// it builds no graph and calls no m-separation, so a defect in the search's
    /// separation code cannot also pass verification.
    fn independently_admissible(
        &self,
        state: &State,
        given: &[VariableId],
        selections: &[VariableId],
    ) -> Result<bool, IdentificationError> {
        Ok(independently_separated(
            self.diagram.causal_graph(),
            &state.v,
            &state.x,
            &self.dense_all(selections)?,
            &state.y.to_dense_ids(),
            &self.dense_all(given)?,
        ))
    }
    fn dense_all(&self, variables: &[VariableId]) -> Result<Vec<DenseNodeId>, IdentificationError> {
        variables.iter().map(|v| self.prepared.var_to_dense(*v)).collect()
    }
    /// The standardization formula over `over`, when `over` is a valid
    /// pretreatment set. `admissible_known` skips the separation test a caller
    /// has already run for exactly this subset.
    fn standardized(
        &mut self,
        state: &State,
        over: &[VariableId],
        admissible_known: bool,
    ) -> Result<Option<ExprId>, IdentificationError> {
        let mut ws = GraphWorkspace::default();
        let x_nodes = state.x.to_dense_ids();
        for (i, z) in over.iter().enumerate() {
            let dense = self.prepared.var_to_dense(*z)?;
            if over[..i].contains(z)
                || state.y.contains(dense)
                || state.x.contains(dense)
                || x_nodes
                    .iter()
                    .any(|x| self.diagram.causal_graph().reaches_with(*x, dense, &mut ws))
            {
                return Ok(None);
            }
        }
        if !admissible_known && !self.source_admissible_given(state, over)? {
            return Ok(None);
        }
        let variables = self.arena.intern_var_set(self.vars(&state.y)?);
        let conditioned_on = self.arena.intern_var_set(over.iter().copied());
        let intervention = self.arena.intern_intervention_set(self.vars(&state.x)?);
        let population = self.arena.intern_population(self.query.source.clone());
        let source = self.arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: if state.x.any() {
                DomainRef::Interventional
            } else {
                DomainRef::Observational
            },
            population,
            regime: None,
        });
        if over.is_empty() {
            return Ok(Some(source));
        }
        let variables = conditioned_on;
        let conditioned_on = self.arena.empty_var_set();
        let intervention = self.arena.intern_intervention_set([]);
        let population = self.arena.intern_population(self.query.target.clone());
        let target = self.arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: DomainRef::Observational,
            population,
            regime: None,
        });
        let list = self.arena.intern_list([source, target]);
        let expr = self.arena.intern(ExprNode::Product(list));
        Ok(Some(self.arena.intern(ExprNode::SumOut { variables, expr })))
    }
    fn record(&mut self, state: State, rule: Rule, children: Vec<usize>, output: ExprId) -> usize {
        let index = self.proof.len();
        self.proof.push(ProofStep { state, rule, children, output, parameters: Vec::new() });
        index
    }
    fn available_source(&mut self, state: &State) -> Result<Option<ExprId>, IdentificationError> {
        let sources = if self.sources.is_empty() {
            vec![MetaSource {
                population: self.query.source.to_string(),
                selections: self.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
            }]
        } else {
            self.sources.clone()
        };
        for source in sources {
            self.charge(0)?;
            let selections: Vec<_> =
                source.selections.iter().copied().map(VariableId::from_raw).collect();
            if self.admissible_selections(state, &[], &selections)? {
                let output = self.source_from(state, Arc::from(source.population))?;
                if self.source_catalog.is_none_or(|catalog| {
                    bind_distribution(output, &self.arena, catalog, &self.query.target).is_ok()
                }) {
                    return Ok(Some(output));
                }
            }
        }
        Ok(None)
    }
    fn solve(
        &mut self,
        state: State,
        source: bool,
        depth: usize,
    ) -> Result<Option<usize>, IdentificationError> {
        self.charge(depth)?;
        if let Some(hit) = self.memo.get(&(state.clone(), source)) {
            return Ok(*hit);
        }
        if source && self.source_catalog.is_some() {
            if let Some(output) = self.available_source(&state)? {
                let step = self.record(state.clone(), Rule::DirectTransport, Vec::new(), output);
                self.memo.insert((state, source), Some(step));
                return Ok(Some(step));
            }
        }
        let result = self.solve_body(&state, source, depth)?;
        self.memo.insert((state, source), result);
        Ok(result)
    }
    #[allow(clippy::if_not_else)] // Keep the pinned pseudocode branch order.
    #[allow(clippy::too_many_lines)] // One match implementing the pinned recursive algorithm.
    fn solve_body(
        &mut self,
        state: &State,
        source: bool,
        depth: usize,
    ) -> Result<Option<usize>, IdentificationError> {
        let mut ws = GraphWorkspace::default();
        let (rule, children, output) = if !state.x.any() {
            (Rule::Marginal, vec![], self.marginal(state.kernel, &difference(&state.v, &state.y))?)
        } else {
            let ancestors = self.prepared.ancestors_within(&state.y, &state.v, &mut ws);
            let mut non_x = difference(&state.v, &state.x);
            let bar = self.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
            non_x.difference_with(&bar);
            if !ancestors.equal_set(&state.v) {
                let next = State {
                    y: state.y.clone(),
                    x: intersection(&state.x, &ancestors),
                    v: ancestors.clone(),
                    kernel: self.marginal(state.kernel, &difference(&state.v, &ancestors))?,
                };
                let Some(child) = self.solve(next, source, depth + 1)? else {
                    return Ok(None);
                };
                (Rule::Ancestors, vec![child], self.proof[child].output)
            } else if non_x.any() {
                let mut next = state.clone();
                next.x.union_with(&non_x);
                // Direct transport needs no enlargement: it answers the query
                // as posed, without demanding experiments or laws at every
                // level of the irrelevant variables.
                if source {
                    if let Some(output) = self.available_source(state)? {
                        return Ok(Some(self.record(
                            state.clone(),
                            Rule::DirectTransport,
                            vec![],
                            output,
                        )));
                    }
                }
                let Some(child) = self.solve(next, source, depth + 1)? else {
                    return Ok(None);
                };
                let output = self.enlarge_output(state, &non_x, self.proof[child].output)?;
                (Rule::Enlarge, vec![child], output)
            } else {
                let districts = self.prepared.c_components(&difference(&state.v, &state.x));
                if districts.len() > 1 {
                    let mut children = Vec::new();
                    for district in districts {
                        let next = State {
                            y: district.clone(),
                            x: difference(&state.v, &district),
                            ..state.clone()
                        };
                        let Some(child) = self.solve(next, source, depth + 1)? else {
                            return Ok(None);
                        };
                        children.push(child);
                    }
                    let product =
                        self.product(children.iter().map(|i| self.proof[*i].output).collect());
                    let output = self.marginal(
                        product,
                        &difference(&difference(&state.v, &state.x), &state.y),
                    )?;
                    (Rule::Districts, children, output)
                } else {
                    let district =
                        districts.first().ok_or(IdentificationError::InvariantViolated {
                            message: "empty sID district",
                        })?;
                    if source && !self.sources.is_empty() {
                        if let Some(output) = self.available_source(state)? {
                            return Ok(Some(self.record(
                                state.clone(),
                                Rule::DirectTransport,
                                vec![],
                                output,
                            )));
                        }
                    }
                    let containing = self.prepared.c_components(&state.v);
                    if containing.len() == 1 {
                        if !source || !self.sources.is_empty() || !self.source_admissible(state)? {
                            if source {
                                self.obstruction = Some(state.clone());
                            }
                            return Ok(None);
                        }
                        (Rule::Source, vec![], self.source(state)?)
                    } else if containing.iter().any(|d| d.equal_set(district)) {
                        let kernel = self.factor(state, district)?;
                        (
                            Rule::Factor,
                            vec![],
                            self.marginal(kernel, &difference(district, &state.y))?,
                        )
                    } else {
                        let larger = containing.iter().find(|d| district.is_subset_of(d)).ok_or(
                            IdentificationError::InvariantViolated {
                                message: "missing containing district",
                            },
                        )?;
                        let next = State {
                            y: state.y.clone(),
                            x: intersection(&state.x, larger),
                            v: larger.clone(),
                            kernel: self.factor(state, larger)?,
                        };
                        let Some(child) = self.solve(next, source, depth + 1)? else {
                            return Ok(None);
                        };
                        (Rule::Recurse, vec![child], self.proof[child].output)
                    }
                }
            }
        };
        Ok(Some(self.record(state.clone(), rule, children, output)))
    }
}
fn difference(a: &BitSet, b: &BitSet) -> BitSet {
    let mut c = a.clone();
    c.difference_with(b);
    c
}
fn intersection(a: &BitSet, b: &BitSet) -> BitSet {
    let mut c = a.clone();
    c.intersect_with(b);
    c
}

/// Verify recorded local premises and expression transformations without rerunning sID.
///
/// # Errors
/// A changed query, graph premise, child state, expression, or exceeded budget.
#[allow(clippy::too_many_lines)] // One auditable premise dispatch for the seven pinned rules.
pub fn verify_classical_transport(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    derivation: &ClassicalTransportDerivation,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    let bad = || IdentificationError::invalid_derivation("transport.invalid_derivation");
    let mut recorded_selections = derivation.selection_targets.to_vec();
    recorded_selections.sort_unstable();
    let mut input_selections = diagram.selection_targets().to_vec();
    input_selections.sort_unstable();
    if recorded_selections != input_selections {
        return Err(bad());
    }
    if &derivation.query != query
        || derivation.graph_signature != graph_signature(diagram)
        || derivation.root_step >= derivation.proof.len()
    {
        return Err(bad());
    }
    meta::check_meta_resources(diagram.causal_graph(), &derivation.sources, limits.steps, ctx)?;
    validate_meta_sources(diagram.causal_graph(), query, &derivation.sources)?;
    if let Some(first) = derivation.sources.first() {
        let mut selections: Vec<_> = diagram.selection_targets().iter().map(|v| v.raw()).collect();
        selections.sort_unstable();
        if first.selections != selections {
            return Err(bad());
        }
    }
    let mut checker = Engine::new(diagram, query, limits, ctx)?;
    checker.sources.clone_from(&derivation.sources);
    checker.arena = derivation.arena.clone();
    let initial = checker.initial()?;
    let root_step = &derivation.proof[derivation.root_step];
    if root_step.state != initial || root_step.output != derivation.root {
        return Err(bad());
    }
    let mut depths = Vec::<usize>::new();
    for (index, step) in derivation.proof.iter().enumerate() {
        if step.children.iter().any(|i| *i >= index) {
            return Err(bad());
        }
        let depth = step.children.iter().map(|i| depths[*i]).max().unwrap_or(0) + 1;
        checker.charge(depth)?;
        depths.push(depth);
        let state = &step.state;
        if state.v.bit_len() != initial.v.bit_len()
            || state.x.bit_len() != initial.v.bit_len()
            || state.y.bit_len() != initial.v.bit_len()
            || !state.y.is_subset_of(&state.v)
            || !state.x.is_subset_of(&state.v)
            || intersection(&state.x, &state.y).any()
            || state.kernel.raw() as usize >= checker.arena.len()
            || step.output.raw() as usize >= checker.arena.len()
        {
            return Err(bad());
        }
        let children = step.children.iter().map(|i| &derivation.proof[*i]).collect::<Vec<_>>();
        let mut ws = GraphWorkspace::default();
        if step.rule != Rule::Standardize && !step.parameters.is_empty() {
            return Err(bad());
        }
        let expected = match step.rule {
            Rule::Standardize => {
                if !children.is_empty() || state != &initial || !derivation.sources.is_empty() {
                    return Err(bad());
                }
                let selections: Vec<_> = diagram.selection_targets().to_vec();
                if !checker.independently_admissible(state, &step.parameters, &selections)? {
                    return Err(bad());
                }
                checker.standardized(state, &step.parameters, false)?.ok_or_else(bad)?
            }
            Rule::Marginal => {
                if state.x.any() || !children.is_empty() {
                    return Err(bad());
                }
                checker.marginal(state.kernel, &difference(&state.v, &state.y))?
            }
            Rule::Ancestors => {
                let ancestors = checker.prepared.ancestors_within(&state.y, &state.v, &mut ws);
                let expected = State {
                    y: state.y.clone(),
                    x: intersection(&state.x, &ancestors),
                    v: ancestors.clone(),
                    kernel: checker.marginal(state.kernel, &difference(&state.v, &ancestors))?,
                };
                if ancestors.equal_set(&state.v)
                    || children.len() != 1
                    || children[0].state != expected
                {
                    return Err(bad());
                }
                children[0].output
            }
            Rule::Enlarge => {
                let ancestors =
                    checker.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
                let w = difference(&difference(&state.v, &state.x), &ancestors);
                let mut expected = state.clone();
                expected.x.union_with(&w);
                if !w.any() || children.len() != 1 || children[0].state != expected {
                    return Err(bad());
                }
                checker.enlarge_output(state, &w, children[0].output)?
            }
            Rule::Districts => {
                let districts = checker.prepared.c_components(&difference(&state.v, &state.x));
                if districts.len() < 2 || children.len() != districts.len() {
                    return Err(bad());
                }
                for (child, district) in children.iter().zip(districts.iter()) {
                    let expected = State {
                        y: district.clone(),
                        x: difference(&state.v, district),
                        ..state.clone()
                    };
                    if child.state != expected {
                        return Err(bad());
                    }
                }
                let product = checker.product(children.iter().map(|child| child.output).collect());
                checker.marginal(product, &difference(&difference(&state.v, &state.x), &state.y))?
            }
            Rule::Factor | Rule::Recurse => {
                let districts = checker.prepared.c_components(&difference(&state.v, &state.x));
                if districts.len() != 1 {
                    return Err(bad());
                }
                let district = &districts[0];
                let containing = checker.prepared.c_components(&state.v);
                if containing.len() <= 1 {
                    return Err(bad());
                }
                if step.rule == Rule::Factor {
                    if !containing.iter().any(|d| d.equal_set(district)) || !children.is_empty() {
                        return Err(bad());
                    }
                    let kernel = checker.factor(state, district)?;
                    checker.marginal(kernel, &difference(district, &state.y))?
                } else {
                    let larger = containing
                        .iter()
                        .find(|d| district.is_subset_of(d) && !district.equal_set(d))
                        .ok_or_else(bad)?;
                    let expected = State {
                        y: state.y.clone(),
                        x: intersection(&state.x, larger),
                        v: larger.clone(),
                        kernel: checker.factor(state, larger)?,
                    };
                    if children.len() != 1 || children[0].state != expected {
                        return Err(bad());
                    }
                    children[0].output
                }
            }
            Rule::Source | Rule::DirectTransport => {
                if !children.is_empty() {
                    return Err(bad());
                }
                if step.rule == Rule::Source {
                    // Figure 5 line 10 fires only after lines 2-4 declined: the
                    // vertices are their own ancestors, nothing is enlargeable,
                    // and both C(D minus X) and C(D) are single components.
                    let ancestors = checker.prepared.ancestors_within(&state.y, &state.v, &mut ws);
                    let bar =
                        checker.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
                    let mut non_x = difference(&state.v, &state.x);
                    non_x.difference_with(&bar);
                    if !state.x.any()
                        || !ancestors.equal_set(&state.v)
                        || non_x.any()
                        || checker.prepared.c_components(&difference(&state.v, &state.x)).len() != 1
                        || checker.prepared.c_components(&state.v).len() != 1
                    {
                        return Err(bad());
                    }
                }
                let population = match checker.arena.node(step.output) {
                    ExprNode::Distribution { population, .. } => {
                        checker.arena.population(*population).to_string()
                    }
                    _ => return Err(bad()),
                };
                let selections: Vec<_> = if derivation.sources.is_empty() {
                    if population != query.source.as_ref() {
                        return Err(bad());
                    }
                    diagram.selection_targets().to_vec()
                } else {
                    derivation
                        .sources
                        .iter()
                        .find(|s| s.population == population)
                        .ok_or_else(bad)?
                        .selections
                        .iter()
                        .copied()
                        .map(VariableId::from_raw)
                        .collect()
                };
                if !checker.admissible_selections(state, &[], &selections)?
                    || !checker.independently_admissible(state, &[], &selections)?
                {
                    return Err(bad());
                }
                // Compare the source leaf explicitly: symbolic values use the
                // dedicated intervention marker, not a NaN float payload.
                let expected = checker.source_from(state, Arc::from(population))?;
                let (
                    ExprNode::Distribution {
                        variables: av,
                        conditioned_on: ac,
                        intervention: ai,
                        domain: ad,
                        population: ap,
                        regime: ar,
                    },
                    ExprNode::Distribution {
                        variables: bv,
                        conditioned_on: bc,
                        intervention: bi,
                        domain: bd,
                        population: bp,
                        regime: br,
                    },
                ) = (checker.arena.node(expected), checker.arena.node(step.output))
                else {
                    return Err(bad());
                };
                if av != bv
                    || ac != bc
                    || ad != bd
                    || ap != bp
                    || ar != br
                    || checker.arena.intervention_set(*ai) != checker.arena.intervention_set(*bi)
                    || checker.arena.intervention_assignments(*bi).iter().any(|a| !a.is_symbolic())
                {
                    return Err(bad());
                }
                step.output
            }
        };
        if expected != step.output
            && !(step.rule == Rule::Standardize
                && standardization_equal(&checker.arena, expected, step.output))
        {
            return Err(bad());
        }
    }
    Ok(())
}

// Standardization has at most a sum, a product and two distribution leaves.
// Compare symbolic assignment values via the explicit marker, not NaN equality.
fn standardization_equal(arena: &CausalExprArena, a: ExprId, b: ExprId) -> bool {
    match (arena.node(a), arena.node(b)) {
        (
            ExprNode::SumOut { variables: av, expr: ae },
            ExprNode::SumOut { variables: bv, expr: be },
        ) => av == bv && standardization_equal(arena, *ae, *be),
        (ExprNode::Product(al), ExprNode::Product(bl)) => {
            arena.list(*al).len() == arena.list(*bl).len()
                && arena
                    .list(*al)
                    .iter()
                    .zip(arena.list(*bl))
                    .all(|(a, b)| standardization_equal(arena, *a, *b))
        }
        (
            ExprNode::Distribution {
                variables: av,
                conditioned_on: ac,
                intervention: ai,
                domain: ad,
                population: ap,
                regime: ar,
            },
            ExprNode::Distribution {
                variables: bv,
                conditioned_on: bc,
                intervention: bi,
                domain: bd,
                population: bp,
                regime: br,
            },
        ) => {
            let aa = arena.intervention_assignments(*ai);
            let ba = arena.intervention_assignments(*bi);
            av == bv
                && ac == bc
                && ad == bd
                && ap == bp
                && ar == br
                && aa.len() == ba.len()
                && aa.iter().zip(ba).all(|(a, b)| {
                    a.variable == b.variable
                        && (a.value == b.value || (a.is_symbolic() && b.is_symbolic()))
                })
        }
        _ => false,
    }
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
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for i in 0..graph.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).expect("graph capacity"));
        directed.extend(graph.children(from).iter().map(|to| (i, to.as_usize())));
        bidirected.extend(
            graph
                .bidirected_neighbors(from)
                .iter()
                .filter(|to| to.as_usize() > i)
                .map(|to| (i, to.as_usize())),
        );
    }
    directed.sort_unstable();
    bidirected.sort_unstable();
    let mut selections = diagram.selection_targets().iter().map(|v| v.raw()).collect::<Vec<_>>();
    selections.sort_unstable();
    let edges = |list: &[(usize, usize)]| {
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
#[allow(missing_docs)]
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
#[allow(missing_docs)]
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
impl Engine<'_> {
    fn negative_witness(
        &self,
        state: &State,
    ) -> Result<Option<SHedgeCertificate>, IdentificationError> {
        let graph = self.diagram.causal_graph();
        let small = difference(&state.v, &state.x);
        let distances = |within: &BitSet| {
            let mut distances = vec![usize::MAX; graph.node_count()];
            let mut queue = std::collections::VecDeque::new();
            for root in state.y.to_dense_ids() {
                distances[root.as_usize()] = 0;
                queue.push_back(root);
            }
            while let Some(node) = queue.pop_front() {
                for parent in graph.parents(node) {
                    if within.contains(*parent) && distances[parent.as_usize()] == usize::MAX {
                        distances[parent.as_usize()] = distances[node.as_usize()] + 1;
                        queue.push_back(*parent);
                    }
                }
            }
            distances
        };
        let small_distance = distances(&small);
        let large_distance = distances(&state.v);
        let mut directed = Vec::new();
        let mut bidirected = Vec::new();
        for node in state.v.to_dense_ids() {
            if self.ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let distance = if small.contains(node) { &small_distance } else { &large_distance };
            if distance[node.as_usize()] == usize::MAX {
                return Ok(None);
            }
            let child = if state.y.contains(node) {
                None
            } else {
                graph
                    .children(node)
                    .iter()
                    .filter(|child| {
                        state.v.contains(**child)
                            && (!small.contains(node) || small.contains(**child))
                            && distance[child.as_usize()] < distance[node.as_usize()]
                    })
                    .min_by_key(|child| child.raw())
            };
            if let Some(child) = child {
                directed
                    .push((self.prepared.dense_to_var(node)?, self.prepared.dense_to_var(*child)?));
            }
            for other in graph.bidirected_neighbors(node) {
                if node.raw() < other.raw() && state.v.contains(*other) {
                    bidirected.push((
                        self.prepared.dense_to_var(node)?,
                        self.prepared.dense_to_var(*other)?,
                    ));
                }
            }
        }
        let small_nodes = self.vars(&small)?;
        let smaller = SelectionForest {
            nodes: small_nodes.clone().into(),
            directed: directed
                .iter()
                .copied()
                .filter(|(a, b)| small_nodes.contains(a) && small_nodes.contains(b))
                .collect(),
            bidirected: bidirected
                .iter()
                .copied()
                .filter(|(a, b)| small_nodes.contains(a) && small_nodes.contains(b))
                .collect(),
        };
        let witness = SHedgeCertificate {
            query: self.query.clone(),
            graph_signature: graph_signature(self.diagram),
            larger: SelectionForest {
                nodes: self.vars(&state.v)?.into(),
                directed: directed.into(),
                bidirected: bidirected.into(),
            },
            smaller,
            sources: Vec::new(),
        };
        match verify_s_hedge(self.diagram, self.query, &witness, self.ctx) {
            Ok(()) => Ok(Some(witness)),
            Err(error) if self.ctx.cancellation.is_cancelled() => Err(error),
            Err(_) => Ok(None),
        }
    }
}

/// Check an s-hedge's forest, selection, nesting, root, and query conditions.
/// No identifier is invoked and no success flag is trusted.
///
/// # Errors
/// Invalid witness, different evidence/query/graph scope, or cancellation.
#[allow(clippy::too_many_lines)] // Keep the independent forest conditions together.
pub fn verify_s_hedge(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    witness: &SHedgeCertificate,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    let bad = || IdentificationError::invalid_derivation("transport.invalid_s_hedge");
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    if &witness.query != query || witness.graph_signature != graph_signature(diagram) {
        return Err(bad());
    }
    check_sid_memory(diagram, 1, ctx)?;
    if let Some(first) = witness.sources.first() {
        meta::check_meta_resources(diagram.causal_graph(), &witness.sources, usize::MAX, ctx)?;
        validate_meta_sources(diagram.causal_graph(), query, &witness.sources)?;
        let mut selections: Vec<_> = diagram.selection_targets().iter().map(|v| v.raw()).collect();
        selections.sort_unstable();
        if first.selections != selections {
            return Err(bad());
        }
        for source in &witness.sources {
            let source_diagram = SelectionDiagram::try_new(
                diagram.causal_graph().clone(),
                source.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            )?;
            let mut local = witness.clone();
            local.sources.clear();
            local.query.source = Arc::from(source.population.as_str());
            local.graph_signature = graph_signature(&source_diagram);
            verify_s_hedge(&source_diagram, &local.query, &local, ctx)?;
        }
        return Ok(());
    }
    let n = diagram.causal_graph().node_count();
    for forest in [&witness.larger, &witness.smaller] {
        if forest.nodes.len() > n
            || forest.directed.len() > n
            || forest.bidirected.len() > n.saturating_mul(n.saturating_sub(1)) / 2
        {
            return Err(bad());
        }
    }
    let prepared = PreparedAdmg::new(diagram.causal_graph().clone())?;
    let check_forest = |forest: &SelectionForest| -> Result<Vec<VariableId>, IdentificationError> {
        let nodes = &forest.nodes;
        let position: HashMap<VariableId, usize> =
            nodes.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        if nodes.is_empty()
            || position.len() != nodes.len()
            || !nodes.iter().any(|v| diagram.selection_targets().contains(v))
        {
            return Err(bad());
        }
        for v in nodes.iter() {
            prepared.var_to_dense(*v)?;
        }
        // At most one child per node: a forest edge list has distinct sources.
        let mut has_child = vec![false; nodes.len()];
        for (a, b) in forest.directed.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            let (Some(&from), Some(_)) = (position.get(a), position.get(b)) else {
                return Err(bad());
            };
            if std::mem::replace(&mut has_child[from], true)
                || !diagram
                    .causal_graph()
                    .children(prepared.var_to_dense(*a)?)
                    .contains(&prepared.var_to_dense(*b)?)
            {
                return Err(bad());
            }
        }
        let mut unique_edges = std::collections::BTreeSet::new();
        let mut adjacency = vec![Vec::new(); nodes.len()];
        for (a, b) in forest.bidirected.iter() {
            let (Some(&i), Some(&j)) = (position.get(a), position.get(b)) else {
                return Err(bad());
            };
            if !unique_edges.insert(((*a).min(*b), (*a).max(*b)))
                || !diagram
                    .causal_graph()
                    .bidirected_neighbors(prepared.var_to_dense(*a)?)
                    .contains(&prepared.var_to_dense(*b)?)
            {
                return Err(bad());
            }
            adjacency[i].push(j);
            adjacency[j].push(i);
        }
        // One bidirected component must span every node.
        let mut reached = vec![false; nodes.len()];
        reached[0] = true;
        let mut pending = vec![0usize];
        let mut count = 1usize;
        while let Some(node) = pending.pop() {
            if ctx.cancellation.is_cancelled() {
                return Err(IdentificationError::Cancelled);
            }
            for &next in &adjacency[node] {
                if !reached[next] {
                    reached[next] = true;
                    count += 1;
                    pending.push(next);
                }
            }
        }
        if count != nodes.len() {
            return Err(bad());
        }
        let mut roots = nodes
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| !has_child[*i])
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        roots.sort_unstable();
        Ok(roots)
    };
    let roots = check_forest(&witness.larger)?;
    let larger_nodes: std::collections::HashSet<_> = witness.larger.nodes.iter().collect();
    let larger_directed: std::collections::HashSet<_> = witness.larger.directed.iter().collect();
    let larger_bidirected: std::collections::HashSet<_> =
        witness.larger.bidirected.iter().collect();
    if roots != check_forest(&witness.smaller)?
        || roots.is_empty()
        || !witness.larger.nodes.iter().any(|v| query.treatments.contains(v))
        || witness.smaller.nodes.iter().any(|v| query.treatments.contains(v))
        || !witness.smaller.nodes.iter().all(|v| larger_nodes.contains(v))
        || !witness.smaller.directed.iter().all(|e| larger_directed.contains(e))
        || !witness.smaller.bidirected.iter().all(|e| larger_bidirected.contains(e))
    {
        return Err(bad());
    }
    let mut all = BitSet::with_len(diagram.causal_graph().node_count());
    let mut y = all.clone();
    let mut x = all.clone();
    for d in prepared.topo() {
        all.insert(*d);
    }
    for v in query.outcomes.iter() {
        y.insert(prepared.var_to_dense(*v)?);
    }
    for v in query.treatments.iter() {
        x.insert(prepared.var_to_dense(*v)?);
    }
    let ancestors = prepared.ancestors_bar_x(&y, &all, &x, &mut GraphWorkspace::default());
    if roots.iter().any(|v| prepared.var_to_dense(*v).map_or(true, |v| !ancestors.contains(v))) {
        return Err(bad());
    }
    Ok(())
}

/// Result of explicitly bounded catalog-aware alternative search.
#[derive(Clone, Debug)]
pub enum CatalogTransportResult {
    /// A checked derivation with available supplied-catalog leaves.
    Identified(Box<BoundTransportFunctional>),
    /// The searched strategies did not bind. This is not an impossibility proof.
    MissingEvidence {
        /// Strategies attempted, in deterministic order.
        searched: Arc<[Arc<str>]>,
        /// Per-strategy unmet obligations; no claim to exhaustive formula search.
        obligations: Arc<[Arc<str>]>,
    },
    /// Neither bounded strategy produced a derivation; no catalog impossibility claim.
    NotCertified {
        /// Strategies attempted, in deterministic order.
        searched: Arc<[Arc<str>]>,
        /// Scope notes for each attempted strategy.
        obligations: Arc<[Arc<str>]>,
    },
}

/// Search stages of catalog-aware transport identification, in the order they are tried.
const SEARCH_STAGES: [&str; 3] =
    ["target_first_sid", "pretreatment_standardization", "catalog_available_source_first_sid"];

/// The first `count` search stages, as recorded on a result.
fn searched_stages(count: usize) -> Arc<[Arc<str>]> {
    SEARCH_STAGES.iter().take(count).map(|&stage| Arc::from(stage)).collect()
}

/// Pretreatment candidates: non-descendants of the treatments among the remaining nodes.
fn pretreatment_candidates(
    diagram: &SelectionDiagram,
    state: &State,
    x_nodes: &[DenseNodeId],
) -> Vec<DenseNodeId> {
    let mut reach = GraphWorkspace::default();
    difference(&difference(&state.v, &state.x), &state.y)
        .to_dense_ids()
        .into_iter()
        .filter(|z| {
            !x_nodes.iter().any(|x| diagram.causal_graph().reaches_with(*x, *z, &mut reach))
        })
        .collect()
}

/// The obligation recorded when the pretreatment-standardization search finds no derivation.
fn pretreatment_obligation(end: SubsetSearchEnd, steps: usize, n_candidates: usize) -> String {
    if end == SubsetSearchEnd::Capped {
        format!(
            "pretreatment standardization search stopped after {steps} candidate subsets of {n_candidates} pretreatment covariates; inconclusive"
        )
    } else {
        "searched pretreatment standardizations: required supplied source conditional or target marginal is missing".to_owned()
    }
}

/// One-step derivation that standardizes over `over`, or transports directly when it is empty.
fn standardization_derivation(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    arena: CausalExprArena,
    state: &State,
    output: ExprId,
    over: Vec<VariableId>,
) -> ClassicalTransportDerivation {
    ClassicalTransportDerivation {
        query: query.clone(),
        graph_signature: graph_signature(diagram),
        selection_targets: diagram.selection_targets().into(),
        arena,
        root: output,
        root_step: 0,
        sources: Vec::new(),
        proof: vec![ProofStep {
            state: state.clone(),
            rule: if over.is_empty() { Rule::DirectTransport } else { Rule::Standardize },
            children: Vec::new(),
            output,
            parameters: over,
        }],
    }
}

/// Try target-first sID, then catalog-aware source/target recursive choices.
/// This bounded two-strategy search does not inherit classical completeness.
///
/// # Errors
/// Invalid input, exhausted shared search budget, or cancellation. Never a negative theorem claim.
#[allow(clippy::too_many_lines)] // one linear pipeline: source pass, then enlargement, then catalog search
pub fn identify_catalog_transport(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<CatalogTransportResult, IdentificationError> {
    catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
    let mut engine = Engine::new(diagram, query, limits, ctx)?;
    let state = engine.initial()?;
    let mut obligations = Vec::<Arc<str>>::new();
    let mut had_derivation = false;
    let mut result = engine.solve(state.clone(), false, 0)?;
    if result.is_none() {
        result = engine.solve(state.clone(), true, 0)?;
    }
    for strategy in 0..2 {
        if let Some(root_step) = result {
            had_derivation = true;
            let derivation = engine.solved_derivation(root_step, Vec::new())?;
            match derivation.bind_catalog_validated(catalog) {
                Ok(mut bound) => {
                    bound.searched = searched_stages(if strategy == 0 { 1 } else { 3 });
                    return Ok(CatalogTransportResult::Identified(Box::new(bound)));
                }
                Err(error) => obligations.push(Arc::from(error.to_string())),
            }
        } else {
            obligations.push(Arc::from("strategy produced no bindable derivation"));
        }
        if strategy == 0 {
            // Independent sufficient-rule alternatives use only target marginals
            // and source conditionals; they need not bind the target joint law.
            // Pretreatment candidates are the non-descendants of the treatments;
            // subsets are tried smallest first under the search's own budget, so
            // its exhaustion is an obligation, never a failure of the whole call
            // and never a reason to skip the cheaper source-first strategy.
            let x_nodes = state.x.to_dense_ids();
            let candidates = pretreatment_candidates(diagram, &state, &x_nodes);
            let state_outcomes = engine.vars(&state.y)?;
            let state_treatments = engine.vars(&state.x)?;
            let targets = engine.dense_all(diagram.selection_targets())?;
            let selection =
                MutilatedSelection::build(diagram.causal_graph(), &state.v, &state.x, &targets)?;
            let mut found = None;
            let end = for_each_admissible_subset(
                &selection,
                &state.y.to_dense_ids(),
                &x_nodes,
                &candidates,
                limits.steps,
                || {
                    if ctx.cancellation.is_cancelled() {
                        Err(IdentificationError::Cancelled)
                    } else {
                        Ok(())
                    }
                },
                |subset| {
                    let over = subset
                        .iter()
                        .map(|z| engine.prepared.dense_to_var(*z))
                        .collect::<Result<Vec<_>, _>>()?;
                    // The subset passed the separation test that produced it,
                    // and its two leaves are probed against the catalog before
                    // anything is interned or cloned.
                    if !standardization_binds(
                        catalog,
                        query,
                        &state_outcomes,
                        &state_treatments,
                        &over,
                    ) {
                        return Ok(false);
                    }
                    let Some(output) = engine.standardized(&state, &over, true)? else {
                        return Ok(false);
                    };
                    had_derivation = true;
                    let derivation = standardization_derivation(
                        diagram,
                        query,
                        engine.arena.clone(),
                        &state,
                        output,
                        over,
                    );
                    let mut bound = derivation.bind_catalog_validated(catalog)?;
                    verify_classical_transport(diagram, query, &derivation, limits, ctx)?;
                    bound.searched = searched_stages(2);
                    found = Some(bound);
                    Ok(true)
                },
            )?;
            if let Some(bound) = found {
                return Ok(CatalogTransportResult::Identified(Box::new(bound)));
            }
            obligations.push(Arc::from(pretreatment_obligation(
                end,
                limits.steps,
                candidates.len(),
            )));
            // This cache is scoped to immutable evidence and strategy. Changing
            // the evidence-selection policy invalidates it, not just its root.
            engine.memo.clear();
            engine.source_catalog = Some(catalog);
            result = engine.solve(state.clone(), true, 0)?;
        }
    }
    let searched = searched_stages(3);
    let obligations = obligations.into();
    Ok(if had_derivation {
        CatalogTransportResult::MissingEvidence { searched, obligations }
    } else {
        CatalogTransportResult::NotCertified { searched, obligations }
    })
}

fn bind_distribution(
    id: ExprId,
    source: &CausalExprArena,
    catalog: &antecedent_core::EvidenceCatalog,
    target: &str,
) -> Result<ExprNode, IdentificationError> {
    let ExprNode::Distribution {
        variables, conditioned_on, intervention, domain, population, ..
    } = source.node(id).clone()
    else {
        return Err(IdentificationError::InvariantViolated {
            message: "expected a distribution leaf",
        });
    };
    let name = source.population(population);
    let vars = source.var_set(variables);
    let conditions = source.var_set(conditioned_on);
    let interventions = source.intervention_set(intervention);
    if name == target && catalog.target_sampling.is_some_and(|s| !s.represents_target_law()) {
        return Err(IdentificationError::missing_evidence(
            "transport.missing_evidence",
            "target sampling does not represent target law",
        ));
    }
    // One predicate decides regime satisfaction for every transport route:
    // `EvidenceRegime::satisfies`, which the structural identifier also uses.
    let need = antecedent_core::FactorNeed {
        population: name,
        variables: vars,
        conditioned_on: conditions,
        interventions: &interventions,
    };
    let regime = catalog.satisfying_regime(&need).ok_or_else(|| {
        IdentificationError::missing_evidence(
            "transport.missing_evidence",
            format!(
                "this derivation needs joint {name} law over {vars:?} given {conditions:?} under do({interventions:?}); alternative search not exhausted"
            ),
        )
    })?;
    Ok(ExprNode::Distribution {
        variables,
        conditioned_on,
        intervention,
        domain,
        population,
        regime: Some(regime.id),
    })
}

fn check_sid_memory(
    diagram: &SelectionDiagram,
    steps: usize,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    // Carried chain-rule kernels have quadratic coordinate storage; a linear
    // node-count charge would substantially understate recursive factorization.
    let n = diagram.causal_graph().node_count();
    let bytes = n.saturating_mul(n).saturating_mul(64).saturating_add(512).saturating_mul(steps);
    refuse_over_budget(bytes, ctx, IdentificationBudget::Memory)
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
        .memory
        .hard_limit_bytes
        .is_some_and(|limit| u64::try_from(bytes).map_or(true, |bytes| bytes > limit))
    {
        return Err(IdentificationError::budget(budget));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }
    fn d(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }
    fn query() -> ClassicalTransportQuery {
        ClassicalTransportQuery {
            outcomes: Arc::from([v(2)]),
            treatments: Arc::from([v(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        }
    }
    #[test]
    fn target_frontdoor_uses_recursive_multinode_district() {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_directed(d(1), d(2)).unwrap();
        graph.insert_bidirected(d(0), d(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let ClassicalTransportResult::Identified(proof) =
            identify_classical_transport(&diagram, &query(), SidLimits::default(), &ctx).unwrap()
        else {
            panic!("frontdoor must identify");
        };
        assert!(proof.rules().contains(&"sid.line8"));
        assert!(
            proof
                .arena()
                .leaf_bindings(proof.root())
                .iter()
                .all(|b| b.population.as_ref() == "target")
        );
        verify_classical_transport(&diagram, &query(), &proof, SidLimits::default(), &ctx).unwrap();
        let mut corrupted = proof.clone();
        corrupted.proof[corrupted.root_step].output = corrupted.proof[0].output;
        assert!(
            verify_classical_transport(&diagram, &query(), &corrupted, SidLimits::default(), &ctx)
                .is_err()
        );
    }
    #[test]
    fn source_rescues_unconfounded_selection_mechanism_and_rejects_changed_premise() {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(d(0), d(2)).unwrap();
        graph.insert_bidirected(d(0), d(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph.clone(), []).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let ClassicalTransportResult::Identified(proof) =
            identify_classical_transport(&diagram, &query(), SidLimits::default(), &ctx).unwrap()
        else {
            panic!("direct transport");
        };
        assert!(proof.rules().contains(&"sid.line10"));
        let changed = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
        assert!(
            verify_classical_transport(&changed, &query(), &proof, SidLimits::default(), &ctx)
                .is_err()
        );
        assert!(
            identify_classical_transport(
                &diagram,
                &query(),
                SidLimits { steps: 1, depth: 1 },
                &ctx
            )
            .is_err()
        );
        ctx.cancellation.cancel();
        assert!(
            identify_classical_transport(&diagram, &query(), SidLimits::default(), &ctx).is_err()
        );
    }
}
