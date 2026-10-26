//! Engine items of the parent module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::prepared::AncestorMemo;
use crate::selection_separation::{MutilatedSelection, independently_separated};
use crate::{IdentificationBudget, IdentificationError, PreparedAdmg};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{BitSet, DSeparationWorkspace, DenseNodeId, GraphWorkspace, SelectionDiagram};
use std::collections::HashMap;
use std::sync::Arc;
use super::catalog::bind_distribution;

use super::*;

impl Engine<'_> {
    /// The verified derivation rooted at `root_step`: the reachable proof steps
    /// with the engine's arena, exported for the sources the search declared.
    /// Every identifier builds its positive result through this one path.
    pub(crate) fn solved_derivation(
        &mut self,
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
        self.verify(&derivation)?;
        Ok(derivation)
    }

    /// Replay `derivation`'s checker against this engine's meter: its own
    /// limits (unchanged legacy accounting), or the shared budget, charged at
    /// this engine's depth on top of its live bytes.
    pub(crate) fn verify(
        &mut self,
        derivation: &ClassicalTransportDerivation,
    ) -> Result<(), IdentificationError> {
        let (diagram, query, ctx, limits) = (self.diagram, self.query, self.ctx, self.limits);
        let (offset, bytes) = (self.depth_offset, self.live_bytes());
        match self.shared.as_deref_mut() {
            Some(shared) => verify_charged(
                diagram,
                query,
                derivation,
                SidMeter::Shared(shared),
                (offset, bytes),
                ctx,
            ),
            None => verify_classical_transport(diagram, query, derivation, limits, ctx),
        }
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
pub(super) fn standardization_binds(
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

impl<'a> Engine<'a> {
    pub(super) fn new(
        diagram: &'a SelectionDiagram,
        query: &'a ClassicalTransportQuery,
        limits: SidLimits,
        ctx: &'a ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        Self::new_metered(diagram, query, SidMeter::Limits(limits), ctx)
    }
    /// An engine charging `meter`: its own [`SidLimits`], or a shared budget
    /// that every engine of one decision charges.
    pub(super) fn new_metered<'m: 'a>(
        diagram: &'a SelectionDiagram,
        query: &'a ClassicalTransportQuery,
        meter: SidMeter<'m>,
        ctx: &'a ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let limits = meter.limits();
        let shared = match meter {
            SidMeter::Limits(_) => None,
            SidMeter::Shared(shared) => Some(shared as &'a mut (dyn SearchCharge + 'a)),
        };
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
            ancestors: AncestorMemo::default(),
            topo,
            all,
            arena: CausalExprArena::new(),
            proof: Vec::new(),
            memo: HashMap::new(),
            limits,
            shared,
            depth_offset: 0,
            base_bytes: 0,
            ctx,
            steps: 0,
            max_depth: 0,
            cross_source_joint: false,
            obstruction: None,
            source_catalog: None,
            sources: Vec::new(),
        })
    }
    pub(super) fn cached_ancestors(
        &mut self,
        seeds: &BitSet,
        active: &BitSet,
        ws: &mut GraphWorkspace,
    ) -> BitSet {
        self.prepared.ancestors_cached(seeds, active, ws, &mut self.ancestors)
    }
    pub(super) fn charge(&mut self, depth: usize) -> Result<(), IdentificationError> {
        self.steps = self.steps.saturating_add(1);
        self.max_depth = self.max_depth.max(depth);
        // Under a shared budget the operation, depth, memory and cancellation
        // bounds are the budget's, with its `depth > limit` semantics, and the
        // bytes are this engine's live state on top of its enclosing engines'.
        let (bytes, at) = (self.live_bytes(), self.depth_offset.saturating_add(depth));
        if let Some(shared) = self.shared.as_deref_mut() {
            return shared.charge(at, bytes).map_err(stop_error);
        }
        if self.ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        if self.steps > self.limits.steps || depth >= self.limits.depth {
            return Err(IdentificationError::budget(IdentificationBudget::Steps));
        }
        check_sid_memory(self.diagram, self.steps, self.ctx)?;
        Ok(())
    }
    /// Charge one operation at `depth` when this engine runs under a shared
    /// budget; an engine on its own [`SidLimits`] keeps its unchanged accounting.
    pub(super) fn charge_shared(&mut self, depth: usize) -> Result<(), IdentificationError> {
        if self.shared.is_some() { self.charge(depth) } else { Ok(()) }
    }
    /// Estimated live bytes of this engine and the engines it runs inside.
    pub(super) fn live_bytes(&self) -> u64 {
        let own = u64::try_from(sid_memory_bytes(self.diagram, self.steps)).unwrap_or(u64::MAX);
        self.base_bytes.saturating_add(own)
    }
    /// Record a region a stop left unevaluated with the shared budget, if any.
    pub(super) fn defer(&mut self, region: impl FnOnce() -> String) {
        if let Some(shared) = self.shared.as_deref_mut() {
            shared.defer(region());
        }
    }
    /// The meter a nested engine rooted at `depth` of this one charges: the
    /// same shared budget, or limits reduced by what this engine consumed.
    pub(super) fn nested_meter(&mut self, depth: usize) -> (SidMeter<'_>, usize, u64) {
        let (offset, bytes) = (self.depth_offset.saturating_add(depth), self.live_bytes());
        let meter = match self.shared.as_deref_mut() {
            Some(shared) => SidMeter::Shared(shared),
            None => SidMeter::Limits(SidLimits {
                steps: self.limits.steps.saturating_sub(self.steps).max(1),
                depth: self.limits.depth.saturating_sub(depth).max(1),
            }),
        };
        (meter, offset, bytes)
    }
    pub(super) fn set(&self, variables: &[VariableId]) -> Result<BitSet, IdentificationError> {
        let mut set = BitSet::with_len(self.diagram.causal_graph().node_count());
        for v in variables {
            set.insert(self.prepared.var_to_dense(*v)?);
        }
        Ok(set)
    }
    pub(super) fn vars(&self, set: &BitSet) -> Result<Vec<VariableId>, IdentificationError> {
        set.iter().map(|v| self.prepared.dense_to_var(v)).collect()
    }
    pub(super) fn initial(&mut self) -> Result<State, IdentificationError> {
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
        let intervention = crate::intervention_support::intern_assignments(
            &mut self.arena,
            assignments.iter().cloned(),
        )?;
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
    pub(super) fn marginal(
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
    pub(super) fn product(&mut self, expressions: Vec<ExprId>) -> ExprId {
        let list = self.arena.intern_list(expressions);
        self.arena.intern(ExprNode::Product(list))
    }
    pub(super) fn factor(&mut self, state: &State, district: &BitSet) -> Result<ExprId, IdentificationError> {
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
    pub(super) fn enlarge_output(
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
    pub(super) fn source(&mut self, state: &State) -> Result<ExprId, IdentificationError> {
        let population = self.query.source.clone();
        self.source_from(state, population)
    }
    pub(super) fn source_from(
        &mut self,
        state: &State,
        population: Arc<str>,
    ) -> Result<ExprId, IdentificationError> {
        // A district kernel fixes its external parents. Nonparents in the
        // original complement cannot affect it after those parents are fixed;
        // retaining them would demand irrelevant experimental coordinates.
        let mut interventions = state.x.clone();
        for node in state.v.iter() {
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
    pub(super) fn source_admissible(&self, state: &State) -> Result<bool, IdentificationError> {
        self.source_admissible_given(state, &[])
    }
    pub(super) fn source_admissible_given(
        &self,
        state: &State,
        given: &[VariableId],
    ) -> Result<bool, IdentificationError> {
        self.admissible_selections(state, given, self.diagram.selection_targets())
    }
    pub(super) fn admissible_selections(
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
    pub(super) fn independently_admissible(
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
    pub(super) fn dense_all(&self, variables: &[VariableId]) -> Result<Vec<DenseNodeId>, IdentificationError> {
        variables.iter().map(|v| self.prepared.var_to_dense(*v)).collect()
    }
    /// The standardization formula over `over`, when `over` is a valid
    /// pretreatment set. `admissible_known` skips the separation test a caller
    /// has already run for exactly this subset.
    pub(super) fn standardized(
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
    pub(super) fn record(&mut self, state: State, rule: Rule, children: Vec<usize>, output: ExprId) -> usize {
        let index = self.proof.len();
        self.proof.push(ProofStep { state, rule, children, output, parameters: Vec::new() });
        index
    }
    pub(super) fn available_source(&mut self, state: &State) -> Result<Option<ExprId>, IdentificationError> {
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
    pub(super) fn solve(
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
    #[allow(clippy::if_not_else, reason = "Keep the pinned pseudocode branch order")]
    #[allow(
        clippy::too_many_lines,
        reason = "One match implementing the pinned recursive algorithm"
    )]
    pub(super) fn solve_body(
        &mut self,
        state: &State,
        source: bool,
        depth: usize,
    ) -> Result<Option<usize>, IdentificationError> {
        let mut ws = GraphWorkspace::default();
        let (rule, children, output) = if !state.x.any() {
            (Rule::Marginal, vec![], self.marginal(state.kernel, &difference(&state.v, &state.y))?)
        } else {
            let ancestors = self.cached_ancestors(&state.y, &state.v, &mut ws);
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

pub(super) fn difference(a: &BitSet, b: &BitSet) -> BitSet {
    let mut c = a.clone();
    c.difference_with(b);
    c
}

pub(super) fn intersection(a: &BitSet, b: &BitSet) -> BitSet {
    let mut c = a.clone();
    c.intersect_with(b);
    c
}

impl Engine<'_> {
    pub(super) fn negative_witness(
        &self,
        state: &State,
    ) -> Result<Option<SHedgeCertificate>, IdentificationError> {
        let graph = self.diagram.causal_graph();
        let small = difference(&state.v, &state.x);
        // Shortest directed distance to `Y` through parents inside `within`: a layered BFS
        // that needs the hop count of every node, so it is not a closure (which keeps
        // only membership) and stays here.
        let distances = |within: &BitSet| {
            let mut distances = vec![usize::MAX; graph.node_count()];
            let mut queue = std::collections::VecDeque::new();
            for root in state.y.iter() {
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
        for node in state.v.iter() {
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
