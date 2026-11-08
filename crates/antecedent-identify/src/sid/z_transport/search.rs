//! Recursive z-transport search and catalog binding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    Arc, BitSet, CausalExprArena, DenseNodeId, DomainRef, ExprId, ExprNode, IdentificationBudget,
    IdentificationError, SelectionDiagram, VariableId,
};
use super::{
    BoundZTransportFunctional, CatalogInterventionAssignment, DistributionAvailability,
    EvidenceCatalog, NodeRef, RegimeKind, VariableDomain, ZFactorObligation, ZFormulaKind,
    ZProofOperation, ZTransportBudgetKind, ZTransportDerivation, ZTransportLimitsReceipt,
    ZTransportProofInspection, ZTransportQuery, index_u32,
};

mod binding;
mod inspection;
pub use binding::bind_z_transport_catalog;
pub(crate) use binding::{bind_recursive_expression, cited_regimes};
pub(super) use inspection::{
    inspect_z_expression, matches_registered_surrogate_graph, same_variable_set, surrogate_formula,
};

/// Run the `TRz` search, reporting an exhausted engine budget as the
/// z-transport budget. This is the one place that remap happens.
pub(super) fn z_search(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    search_trz_detailed(diagram, query, meter, ctx, receipt).map_err(|error| match error {
        IdentificationError::Budget { budget: IdentificationBudget::Steps } => {
            IdentificationError::budget(IdentificationBudget::ZTransport)
        }
        other => other,
    })
}

// Bareinboim–Pearl (AAAI 2013, Figure 4) rules 1–4 map to the marginal,
// ancestor, enlargement, and district branches below; rules 5–8 map to the
// C-component, factor, and subdistrict branches. Rule 10 exchanges the active
// controllables in X and recurses on the reduced graph; rule 11 is recorded
// only when that exchange's separation premise fails or Z∩X is empty.
#[cfg(test)]
pub(super) fn search_trz(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<(CausalExprArena, ExprId, Vec<String>)>, IdentificationError> {
    Ok(search_trz_detailed(diagram, query, super::super::SidMeter::Limits(limits), ctx, &mut None)?
        .identified)
}

/// One source a `TRz` line-10 exchange may consult.
pub(crate) struct TrzDomain<'q> {
    pub(crate) population: &'q Arc<str>,
    pub(crate) controllable: &'q [VariableId],
    pub(crate) experiment_assignment: &'q [CatalogInterventionAssignment],
    pub(crate) selection_targets: &'q [VariableId],
}

/// Shared inputs of one recursive search: the queried treatments, the target,
/// and the sources line 10 may exchange into (one for `TRz`, several for `TR^mz`).
pub(crate) struct TrzCall<'q> {
    pub(crate) outcomes: &'q Arc<[VariableId]>,
    pub(crate) treatments: &'q Arc<[VariableId]>,
    pub(crate) target: &'q Arc<str>,
    /// A source population distinct from the target. The engine's classical
    /// query needs one; this recursion never builds a kernel from it, since every
    /// exchanged kernel carries its own source's population.
    pub(crate) anchor: &'q Arc<str>,
    pub(crate) domains: Vec<TrzDomain<'q>>,
}

impl<'q> TrzCall<'q> {
    fn single(diagram: &'q SelectionDiagram, query: &'q ZTransportQuery) -> Self {
        Self {
            outcomes: &query.outcomes,
            treatments: &query.treatments,
            target: &query.target,
            anchor: &query.source,
            domains: vec![TrzDomain {
                population: &query.source,
                controllable: &query.controllable,
                experiment_assignment: &query.experiment_assignment,
                selection_targets: diagram.selection_targets(),
            }],
        }
    }
}

/// Whether a recorded line-11 terminal settles the search: with one source every recorded
/// terminal is the first on a path with no alternative, and with several (or none) only a
/// terminal reached with no active experiment is forced on every path.
pub(super) fn terminal_is_settled(sources: usize, terminal: Option<&TrzTerminalFailure>) -> bool {
    terminal.is_some_and(|recorded| sources == 1 || recorded.active_interventions.is_empty())
}

/// The districts' children of `TRz` rule 4. A district the search could not identify does not
/// excuse the others: a later district may still reach a forced line-11 terminal, which proves
/// the product non-transportable, so only a failure that settles the search ends the loop.
pub(super) struct DistrictFactors<T> {
    children: Vec<T>,
    failed: bool,
}

impl<T> DistrictFactors<T> {
    pub(super) fn with_capacity(districts: usize) -> Self {
        Self { children: Vec::with_capacity(districts), failed: false }
    }

    /// Record one district's result; `settled` says whether the search is settled after it.
    /// `true` when the loop must stop.
    pub(super) fn record(&mut self, child: Option<T>, settled: bool) -> bool {
        if let Some(child) = child {
            self.children.push(child);
            false
        } else {
            self.failed = true;
            settled
        }
    }

    /// Every district's child, or `None` when any district failed.
    pub(super) fn finish(self) -> Option<Vec<T>> {
        (!self.failed).then_some(self.children)
    }
}

/// One source's line-11 premises at a multi-source terminal.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TrzSourceTerminal {
    pub(crate) population: String,
    pub(crate) remaining_controllable: Vec<u32>,
    pub(crate) candidate_active: Vec<u32>,
    pub(crate) selection_separated: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TrzTerminalFailure {
    pub(crate) outcomes: Vec<u32>,
    pub(crate) treatments: Vec<u32>,
    pub(crate) vertices: Vec<u32>,
    pub(crate) c0: Vec<u32>,
    pub(super) remaining_controllable: Vec<u32>,
    pub(super) candidate_active: Vec<u32>,
    pub(crate) active_interventions: Vec<(u32, Option<f64>)>,
    pub(super) selection_separated: bool,
    pub(crate) rules: Vec<String>,
    /// Per-source premises when several sources were searched; empty for `TRz`,
    /// whose single source uses the fields above.
    pub(crate) sources: Vec<TrzSourceTerminal>,
}

/// A coordinate a source line-10 exchange must fix at a declared level that the source's
/// experiment assignment does not name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TrzUnassigned {
    pub(crate) population: Arc<str>,
    pub(crate) variable: VariableId,
}

#[derive(Debug)]
pub(crate) struct TrzSearchResult {
    pub(crate) identified: Option<(CausalExprArena, ExprId, Vec<String>)>,
    pub(crate) terminal_failure: Option<TrzTerminalFailure>,
    pub(crate) unassigned: Option<TrzUnassigned>,
    /// Every recursive rule the search applied, in order, whatever the outcome.
    pub(crate) explored_rules: Vec<String>,
    /// Recursive subproblems the search charged.
    pub(crate) steps_explored: usize,
    /// Deepest recursion level the search reached.
    pub(crate) depth_reached: usize,
    /// A multi-source line-11 state needed a joint over two sources'
    /// interventions, which the search refused to fabricate.
    pub(crate) cross_source_joint: bool,
}

fn search_trz_detailed(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    search_trz_call(diagram, &TrzCall::single(diagram, query), meter, ctx, receipt)
}

/// Run the recursive search for `call` on the shared causal graph of `diagram`,
/// charging `meter`. Selection targets are read per source from `call`, never
/// from `diagram`.
pub(crate) fn search_trz_call(
    diagram: &SelectionDiagram,
    call: &TrzCall<'_>,
    meter: super::super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    let limits = meter.limits();
    let classical = super::super::ClassicalTransportQuery {
        outcomes: Arc::clone(call.outcomes),
        treatments: Arc::clone(call.treatments),
        source: Arc::clone(call.anchor),
        target: Arc::clone(call.target),
    };
    // A budget or cancellation observed at engine construction stops the search
    // before any recursive accounting is taken, so the receipt reports the
    // limits in force with no consumed counts rather than fabricating them.
    let mut engine = match super::super::Engine::new_metered(diagram, &classical, meter, ctx) {
        Ok(engine) => engine,
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(ZTransportLimitsReceipt {
                budget: pre_search_budget_kind(&error),
                steps_limit: limits.steps,
                depth_limit: limits.depth,
                steps_consumed: None,
                depth_reached: None,
                explored_rules: Vec::new(),
            });
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let initial = match engine.initial() {
        Ok(initial) => initial,
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(search_limits_receipt(&error, &engine, limits, Vec::new()));
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let mut trace = Vec::new();
    let mut terminal_failure = None;
    let mut unassigned = None;
    let irrelevant = BitSet::with_len(diagram.causal_graph().node_count());
    let outcome = search_trz_state(
        &mut engine,
        &initial,
        call,
        None,
        &[],
        &[],
        &irrelevant,
        0,
        &mut trace,
        &mut terminal_failure,
        &mut unassigned,
    );
    match outcome {
        Ok(result) => {
            let steps_explored = engine.steps;
            let depth_reached = engine.max_depth;
            Ok(TrzSearchResult {
                cross_source_joint: engine.cross_source_joint,
                identified: result.map(|root| (engine.arena, root, trace.clone())),
                terminal_failure,
                unassigned,
                explored_rules: trace,
                steps_explored,
                depth_reached,
            })
        }
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(search_limits_receipt(&error, &engine, limits, trace));
            Err(error)
        }
        Err(error) => Err(error),
    }
}

/// Classify a budget or cancellation observed before the recursive search runs.
pub(super) fn pre_search_budget_kind(error: &IdentificationError) -> ZTransportBudgetKind {
    match error {
        IdentificationError::Cancelled => ZTransportBudgetKind::Cancelled,
        IdentificationError::Budget { budget: IdentificationBudget::Memory } => {
            ZTransportBudgetKind::Memory
        }
        _ => ZTransportBudgetKind::Steps,
    }
}

/// Build a receipt from the engine's own accounting when a budget or
/// cancellation stopped the recursive search. The step-versus-depth
/// distinction is read from the accounting, never guessed.
fn search_limits_receipt(
    error: &IdentificationError,
    engine: &super::super::Engine<'_>,
    limits: super::super::SidLimits,
    explored_rules: Vec<String>,
) -> ZTransportLimitsReceipt {
    let budget = match error {
        IdentificationError::Cancelled => ZTransportBudgetKind::Cancelled,
        IdentificationError::Budget { budget: IdentificationBudget::Memory } => {
            ZTransportBudgetKind::Memory
        }
        _ if engine.max_depth >= limits.depth => ZTransportBudgetKind::Depth,
        _ => ZTransportBudgetKind::Steps,
    };
    ZTransportLimitsReceipt {
        budget,
        steps_limit: limits.steps,
        depth_limit: limits.depth,
        steps_consumed: Some(engine.steps),
        depth_reached: Some(engine.max_depth),
        explored_rules,
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "Mirrors the paper rule state explicitly"
)]
fn search_trz_state(
    engine: &mut super::super::Engine<'_>,
    state: &super::super::State,
    call: &TrzCall<'_>,
    active: Option<usize>,
    remaining_controllable: &[VariableId],
    active_interventions: &[antecedent_expr::InterventionAssignment],
    irrelevant: &BitSet,
    depth: usize,
    trace: &mut Vec<String>,
    terminal_failure: &mut Option<TrzTerminalFailure>,
    unassigned: &mut Option<TrzUnassigned>,
) -> Result<Option<ExprId>, IdentificationError> {
    engine.charge(depth)?;
    let mut ws = antecedent_graph::GraphWorkspace::default();
    // `TRz` rule 1: no active treatment coordinates remain.
    if !state.x.any() {
        trace.push("ztr.line1.marginal".into());
        return Ok(Some(
            engine.marginal(state.kernel, &super::super::difference(&state.v, &state.y))?,
        ));
    }
    let ancestors = engine.cached_ancestors(&state.y, &state.v, &mut ws);
    // `TRz` rule 2: restrict to ancestors of the current outcomes.
    if !ancestors.equal_set(&state.v) {
        trace.push("ztr.line2.ancestors".into());
        let next = super::super::State {
            y: state.y.clone(),
            x: super::super::intersection(&state.x, &ancestors),
            v: ancestors.clone(),
            kernel: engine
                .marginal(state.kernel, &super::super::difference(&state.v, &ancestors))?,
        };
        return search_trz_state(
            engine,
            &next,
            call,
            active,
            remaining_controllable,
            active_interventions,
            irrelevant,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        );
    }
    let mut enlarged = super::super::difference(&state.v, &state.x);
    let bar = engine.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
    enlarged.difference_with(&bar);
    // `TRz` rule 3: add non-treatment vertices outside An(Y) in D_X. The child
    // is constant in them, so a later exchange may fix them at a declared level.
    if enlarged.any() {
        trace.push("ztr.line3.enlarge".into());
        let mut next = state.clone();
        next.x.union_with(&enlarged);
        let mut fixed = irrelevant.clone();
        fixed.union_with(&enlarged);
        let Some(child) = search_trz_state(
            engine,
            &next,
            call,
            active,
            remaining_controllable,
            active_interventions,
            &fixed,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        )?
        else {
            return Ok(None);
        };
        // Rule 3 is the identity: P_x(y) = P_{x,w}(y). The child names a
        // coordinate of `w` only where a later exchange fixed it at its
        // declared level, so it is returned unchanged; a child that still
        // reads `w` is averaged over the kernel's own law of `w` given `x`,
        // which is zero exactly where the child conditions on a null event.
        let enlarged_vars = engine.vars(&enlarged)?;
        let free = engine.arena.free_variables(child);
        if enlarged_vars.iter().any(|variable| free.contains(variable)) {
            trace.push("ztr.line3.kernel_weighted".into());
            return Ok(Some(engine.enlarge_output(state, &enlarged, child)?));
        }
        return Ok(Some(child));
    }
    let districts = engine.prepared.c_components(&super::super::difference(&state.v, &state.x));
    // `TRz` rule 4: factor over multiple districts in D \ X.
    if districts.len() > 1 {
        trace.push("ztr.line4.districts".into());
        let mut factors = DistrictFactors::with_capacity(districts.len());
        for district in districts {
            let next = super::super::State {
                y: district.clone(),
                x: super::super::difference(&state.v, &district),
                ..state.clone()
            };
            let child = search_trz_state(
                engine,
                &next,
                call,
                active,
                remaining_controllable,
                active_interventions,
                irrelevant,
                depth + 1,
                trace,
                terminal_failure,
                unassigned,
            )?;
            let settled = terminal_is_settled(call.domains.len(), terminal_failure.as_ref());
            if factors.record(child, settled) {
                break;
            }
        }
        let Some(expressions) = factors.finish() else {
            return Ok(None);
        };
        let product = engine.product(expressions);
        return Ok(Some(engine.marginal(
            product,
            &super::super::difference(&super::super::difference(&state.v, &state.x), &state.y),
        )?));
    }
    let district = districts
        .first()
        .ok_or(IdentificationError::InvariantViolated { message: "z_transport.empty_district" })?;
    let containing = engine.prepared.c_components(&state.v);
    // `TRz` rules 5–6 establish C0 and whether D is a single c-component.
    if containing.len() > 1 {
        if containing.iter().any(|d| d.equal_set(district)) {
            // `TRz` rule 7: C0 is a c-component of D, so its kernel is available.
            trace.push("ztr.line7.factor".into());
            let kernel = engine.factor(state, district)?;
            return Ok(Some(
                engine.marginal(kernel, &super::super::difference(district, &state.y))?,
            ));
        }
        let larger = containing.iter().find(|d| district.is_subset_of(d)).ok_or(
            IdentificationError::InvariantViolated {
                message: "z_transport.missing_containing_district",
            },
        )?;
        let next = super::super::State {
            y: state.y.clone(),
            x: super::super::intersection(&state.x, larger),
            v: larger.clone(),
            kernel: engine.factor(state, larger)?,
        };
        // `TRz` rule 8: recurse into the containing c-component of D.
        trace.push("ztr.line8.recurse".into());
        return search_trz_state(
            engine,
            &next,
            call,
            active,
            remaining_controllable,
            active_interventions,
            irrelevant,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        );
    }

    // `TRz` line 10 (`TR^mz` line 10 with several sources): with no active
    // experiment every source is a candidate; once a subcall exchanged into one
    // source, only that source's remaining controllables may be exchanged, so a
    // c-factor never mixes sources.
    let candidates = active.map_or_else(|| (0..call.domains.len()).collect(), |i| vec![i]);
    let mut eligibility = Vec::with_capacity(candidates.len());
    for index in candidates {
        let domain = &call.domains[index];
        let controllable =
            if active == Some(index) { remaining_controllable } else { domain.controllable };
        let mut activated = BitSet::with_len(engine.diagram.causal_graph().node_count());
        for variable in controllable.iter().copied() {
            let dense = engine.prepared.var_to_dense(variable)?;
            if state.x.contains(dense) {
                activated.insert(dense);
            }
        }
        let separated = engine.admissible_selections(state, &[], domain.selection_targets)?;
        eligibility.push((index, controllable, activated, separated));
    }
    if !eligibility.iter().any(|(_, _, activated, separated)| activated.any() && *separated) {
        // `TRz` rule 11: rule 10 cannot exchange an active experiment.
        let single = call.domains.len() == 1;
        if let (false, Some(current)) = (single, active) {
            // Another source could exchange here, but only jointly with the
            // active source's experiment: the joint `TR^mz` never fabricates.
            for (index, domain) in call.domains.iter().enumerate() {
                if index == current || engine.cross_source_joint {
                    continue;
                }
                let mut activated = false;
                for variable in domain.controllable.iter().copied() {
                    activated |= state.x.contains(engine.prepared.var_to_dense(variable)?);
                }
                engine.cross_source_joint = activated
                    && engine.admissible_selections(state, &[], domain.selection_targets)?;
            }
        }
        let replace = terminal_failure.as_ref().is_none_or(|recorded| {
            // With several sources, prefer a terminal reached with no active
            // experiment: only that state is forced on every search path.
            !single && !recorded.active_interventions.is_empty() && active_interventions.is_empty()
        });
        if replace {
            let mut c0 = engine.vars(district)?.iter().map(|v| v.raw()).collect::<Vec<_>>();
            c0.sort_unstable();
            let raw_sorted = |engine: &super::super::Engine<'_>, set: &BitSet| {
                engine.vars(set).map(|vars| {
                    let mut raw = vars.iter().map(|v| v.raw()).collect::<Vec<_>>();
                    raw.sort_unstable();
                    raw
                })
            };
            trace.push("ztr.line11.fail".into());
            let (remaining, candidate_active, selection_separated, sources) = if single {
                let (_, controllable, activated, separated) = &eligibility[0];
                (
                    controllable.iter().map(|v| v.raw()).collect(),
                    raw_sorted(engine, activated)?,
                    *separated,
                    Vec::new(),
                )
            } else {
                let mut sources = Vec::with_capacity(eligibility.len());
                for (index, controllable, activated, separated) in &eligibility {
                    sources.push(TrzSourceTerminal {
                        population: call.domains[*index].population.to_string(),
                        remaining_controllable: controllable.iter().map(|v| v.raw()).collect(),
                        candidate_active: raw_sorted(engine, activated)?,
                        selection_separated: *separated,
                    });
                }
                (Vec::new(), Vec::new(), false, sources)
            };
            *terminal_failure = Some(TrzTerminalFailure {
                outcomes: engine.vars(&state.y)?.iter().map(|v| v.raw()).collect(),
                treatments: engine.vars(&state.x)?.iter().map(|v| v.raw()).collect(),
                vertices: engine.vars(&state.v)?.iter().map(|v| v.raw()).collect(),
                c0,
                remaining_controllable: remaining,
                candidate_active,
                active_interventions: active_interventions
                    .iter()
                    .map(|a| (a.variable.raw(), a.value.as_f64()))
                    .collect(),
                selection_separated,
                rules: trace.clone(),
                sources,
            });
        }
        return Ok(None);
    }
    let eligible = eligibility
        .into_iter()
        .filter(|(_, _, activated, separated)| activated.any() && *separated)
        .collect::<Vec<_>>();
    for (position, (index, controllable, activated, _)) in eligible.iter().enumerate() {
        match exchange_trz_source(
            engine,
            state,
            call,
            *index,
            controllable,
            activated,
            active_interventions,
            irrelevant,
            depth,
            trace,
            terminal_failure,
            unassigned,
        ) {
            Ok(Some(result)) => return Ok(Some(result)),
            Ok(None) => {}
            Err(error) => {
                // A stop leaves the later line-10 source branches untried;
                // name them by population so the record is order-invariant.
                if call.domains.len() > 1 && error.is_budget_or_cancel() {
                    for (later, ..) in &eligible[position + 1..] {
                        let population = call.domains[*later].population;
                        engine.defer(|| format!("multi:line10:source={population}@{depth}"));
                    }
                }
                return Err(error);
            }
        }
    }
    Ok(None)
}

/// `TRz` rule 10 for one source: exchange its active controllables for that
/// source's experiment and recurse with the source active.
#[allow(clippy::too_many_arguments, reason = "The recursion state of `search_trz_state`")]
fn exchange_trz_source(
    engine: &mut super::super::Engine<'_>,
    state: &super::super::State,
    call: &TrzCall<'_>,
    index: usize,
    controllable: &[VariableId],
    activated: &BitSet,
    active_interventions: &[antecedent_expr::InterventionAssignment],
    irrelevant: &BitSet,
    depth: usize,
    trace: &mut Vec<String>,
    terminal_failure: &mut Option<TrzTerminalFailure>,
    unassigned: &mut Option<TrzUnassigned>,
) -> Result<Option<ExprId>, IdentificationError> {
    let domain = &call.domains[index];
    let active_vars = engine.vars(activated)?;
    // The level each exchanged coordinate takes in the cited source experiment.
    // A queried treatment and a coordinate that rule 4 handed to this district
    // stay symbolic: the evaluation request binds the former and the enclosing
    // summation binds the latter, so one formula answers every requested level
    // and every summand at its own world. A coordinate that rule 3 proved
    // irrelevant to the child may be fixed at its declared level, since the
    // child is constant in it.
    let mut exchanged = Vec::with_capacity(active_vars.len());
    for variable in active_vars.iter().copied() {
        let dense = engine.prepared.var_to_dense(variable)?;
        if irrelevant.contains(dense) && !call.treatments.contains(&variable) {
            let Some(assignment) = domain
                .experiment_assignment
                .iter()
                .find(|assignment| assignment.variable == variable)
            else {
                *unassigned =
                    Some(TrzUnassigned { population: Arc::clone(domain.population), variable });
                return Ok(None);
            };
            exchanged.push(antecedent_expr::InterventionAssignment::concrete(
                variable,
                assignment.value.clone(),
            ));
        } else {
            exchanged.push(antecedent_expr::InterventionAssignment::symbolic(variable));
        }
    }
    let exchanged_levels =
        exchanged.iter().map(|a| (a.variable.raw(), a.value.as_f64())).collect::<Vec<_>>();
    if call.domains.len() == 1 {
        trace.push(format!("ztr.line10.source_exchange:{exchanged_levels:?}"));
    } else {
        trace
            .push(format!("ztr.line10.source_exchange:{}:{exchanged_levels:?}", domain.population));
    }
    let mut cumulative = active_interventions.to_vec();
    for assignment in exchanged {
        if !cumulative.iter().any(|a| a.variable == assignment.variable) {
            cumulative.push(assignment);
        }
    }
    // `TRz` rule 10: exchange Z∩X and recurse with Z\X and active set I=Z∩X.
    // The new kernel is the source c-factor Q^s_z[V\Z] = P^s_{z ∪ (V_all\V)}(V\Z):
    // every coordinate outside the current vertex set is intervened, not
    // conditioned on. When the state still spans the whole graph that is the
    // cited source law itself; otherwise the c-factor is identified from that
    // law by the interventional-distribution recursion on the graph with the
    // exchanged coordinates removed (sID^z line 7).
    let remaining = super::super::difference(&state.v, activated);
    let Some(kernel) = source_c_factor(
        engine,
        &remaining,
        &cumulative,
        domain.population,
        call.target,
        depth,
        trace,
    )?
    else {
        return Ok(None);
    };
    let next = super::super::State {
        y: state.y.clone(),
        x: super::super::difference(&state.x, activated),
        v: remaining,
        kernel,
    };
    // Fig. 4 line 10 recurses into `TRz` on the reduced graph and retains
    // experiments over the remaining controllable variables.
    let remaining = controllable
        .iter()
        .copied()
        .filter(|variable| !active_vars.contains(variable))
        .collect::<Vec<_>>();
    search_trz_state(
        engine,
        &next,
        call,
        Some(index),
        &remaining,
        &cumulative,
        irrelevant,
        depth + 1,
        trace,
        terminal_failure,
        unassigned,
    )
}

/// The source c-factor `Q^s_z[remaining]` after exchanging `cumulative`, in the
/// z engine's arena. `None` when the interventional-distribution recursion cannot
/// identify it from the cited source law; that is not an obstruction claim.
fn source_c_factor(
    engine: &mut super::super::Engine<'_>,
    remaining: &BitSet,
    cumulative: &[antecedent_expr::InterventionAssignment],
    source: &Arc<str>,
    target: &Arc<str>,
    depth: usize,
    trace: &mut Vec<String>,
) -> Result<Option<ExprId>, IdentificationError> {
    let graph = engine.diagram.causal_graph();
    let exchanged = cumulative.iter().map(|a| a.variable).collect::<Vec<_>>();
    let remaining_vars = engine.vars(remaining)?;
    // An outside coordinate that is not an ancestor of the remaining set in the
    // graph without the exchanged coordinates neither changes the c-factor
    // when intervened on nor needs to be measured: the source law over the
    // ancestral closure is the marginal the theorem reads. Only the ancestral
    // outside coordinates are intervened on and identified away.
    let mut exchanged_nodes = BitSet::with_len(graph.node_count());
    for variable in &exchanged {
        // A variable outside the graph cannot be anyone's parent.
        if let Ok(dense) = engine.prepared.var_to_dense(*variable) {
            exchanged_nodes.insert(dense);
        }
    }
    let mut closure = BitSet::default();
    antecedent_graph::algo::closure_into(
        graph.node_count(),
        |u| graph.parents(u),
        remaining.iter(),
        &mut closure,
        &mut antecedent_graph::GraphWorkspace::default(),
        |_, parent| !exchanged_nodes.contains(parent),
    );
    let kept = engine.vars(&closure)?;
    let outside = kept.iter().copied().filter(|v| !remaining_vars.contains(v)).collect::<Vec<_>>();
    let population = engine.arena.intern_population(Arc::clone(source));
    if outside.is_empty() {
        let variables = engine.arena.intern_var_set(remaining_vars);
        let conditioned_on = engine.arena.empty_var_set();
        let intervention = crate::intervention_support::intern_assignments(
            &mut engine.arena,
            cumulative.iter().cloned(),
        )?;
        return Ok(Some(engine.arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: DomainRef::Interventional,
            population,
            regime: None,
        })));
    }
    // Identify P^s_{z ∪ outside}(remaining) from P^s_z(kept) on the graph
    // without the exchanged coordinates.
    let mut reduced = antecedent_graph::Admg::empty();
    for variable in &kept {
        reduced.add_node(NodeRef::Static(*variable))?;
    }
    let dense = |variable: VariableId| {
        kept.iter().position(|v| *v == variable).map(|i| DenseNodeId::from_raw(index_u32(i)))
    };
    for (from_index, node) in graph.nodes().iter().enumerate() {
        let NodeRef::Static(from) = node else { continue };
        let Some(reduced_from) = dense(*from) else { continue };
        let from_dense = DenseNodeId::from_raw(index_u32(from_index));
        for to in graph.children(from_dense) {
            if let Some(reduced_to) = engine.prepared.dense_to_var(*to).ok().and_then(dense) {
                reduced.insert_directed(reduced_from, reduced_to)?;
            }
        }
        for other in graph.bidirected_neighbors(from_dense) {
            if other.as_usize() > from_index
                && let Some(reduced_other) =
                    engine.prepared.dense_to_var(*other).ok().and_then(dense)
            {
                reduced.insert_bidirected(reduced_from, reduced_other)?;
            }
        }
    }
    let diagram = SelectionDiagram::try_new(reduced, Arc::<[VariableId]>::from([]))?;
    let classical = super::super::ClassicalTransportQuery {
        outcomes: remaining_vars.clone().into(),
        treatments: outside.into(),
        source: Arc::clone(source),
        target: Arc::clone(target),
    };
    // The nested engine charges the enclosing search's budget: the shared one
    // at its true depth, or the steps and depth this engine has left.
    let ctx = engine.ctx;
    let (sub_steps, solved) = {
        let (meter, depth_offset, base_bytes) = engine.nested_meter(depth);
        let mut sub = super::super::Engine::new_metered(&diagram, &classical, meter, ctx)?;
        sub.depth_offset = depth_offset;
        sub.base_bytes = base_bytes;
        let initial = sub.initial_source_kernel(source, cumulative)?;
        let solved = sub.solve(initial, false, 0).map(|root| {
            root.map(|root_step| {
                let rules = format!("ztr.line10.c_factor:{:?}", sub.reachable_rules(root_step));
                (rules, sub.proof[root_step].output, std::mem::take(&mut sub.arena))
            })
        });
        (sub.steps, solved)
    };
    engine.steps = engine.steps.saturating_add(sub_steps);
    let Some((rules, output, arena)) = solved? else {
        trace.push("ztr.line10.c_factor_not_identified".into());
        return Ok(None);
    };
    trace.push(rules);
    let mut memo = std::collections::HashMap::new();
    Ok(Some(transplant(&arena, output, &mut engine.arena, &mut memo)?))
}

/// Copy the expression `id` of `from` into `into`, re-interning every table.
fn transplant(
    from: &CausalExprArena,
    id: ExprId,
    into: &mut CausalExprArena,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> Result<ExprId, IdentificationError> {
    if let Some(copied) = memo.get(&id) {
        return Ok(*copied);
    }
    let copied = match from.node(id).clone() {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            let variables = into.intern_var_set(from.var_set(variables).iter().copied());
            let conditioned_on = into.intern_var_set(from.var_set(conditioned_on).iter().copied());
            let intervention = crate::intervention_support::intern_assignments(
                into,
                from.intervention_assignments(intervention).iter().cloned(),
            )?;
            let population = into.intern_population(Arc::from(from.population(population)));
            into.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            })
        }
        ExprNode::Product(list) => {
            let children = from
                .list(list)
                .iter()
                .map(|child| transplant(from, *child, into, memo))
                .collect::<Result<Vec<_>, _>>()?;
            let list = into.intern_list(children);
            into.intern(ExprNode::Product(list))
        }
        ExprNode::SumOut { variables, expr } => {
            let expr = transplant(from, expr, into, memo)?;
            let variables = into.intern_var_set(from.var_set(variables).iter().copied());
            into.intern(ExprNode::SumOut { variables, expr })
        }
        ExprNode::Ratio { numerator, denominator } => {
            let numerator = transplant(from, numerator, into, memo)?;
            let denominator = transplant(from, denominator, into, memo)?;
            into.intern(ExprNode::Ratio { numerator, denominator })
        }
        _ => {
            return Err(IdentificationError::InvariantViolated {
                message: "source c-factor identification produced an unexpected node",
            });
        }
    };
    memo.insert(id, copied);
    Ok(copied)
}
