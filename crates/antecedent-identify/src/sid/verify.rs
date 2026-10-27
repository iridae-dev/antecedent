//! Verify items of the parent module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::meta::validate_meta_sources;
use crate::{IdentificationError, PreparedAdmg};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_expr::{CausalExprArena, ExprId, ExprNode};
use antecedent_graph::algo::components_by;
use antecedent_graph::{BitSet, GraphWorkspace, SelectionDiagram};
use std::collections::HashMap;
use std::sync::Arc;

use super::*;

/// Verify recorded local premises and expression transformations without rerunning sID.
///
/// # Errors
/// A changed query, graph premise, child state, expression, or exceeded budget.
pub fn verify_classical_transport(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    derivation: &ClassicalTransportDerivation,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    verify_charged(diagram, query, derivation, SidMeter::Limits(limits), (0, 0), ctx)
}

/// [`verify_classical_transport`] charging `meter`, with the checker rooted at
/// `(depth offset, live bytes)` of the search it verifies for.
#[allow(
    clippy::too_many_lines,
    reason = "One auditable premise dispatch for the seven pinned rules"
)]
pub(super) fn verify_charged(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    derivation: &ClassicalTransportDerivation,
    meter: SidMeter<'_>,
    (depth_offset, base_bytes): (usize, u64),
    ctx: &ExecutionContext,
) -> Result<(), IdentificationError> {
    let limits = meter.limits();
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
    let mut checker = Engine::new_metered(diagram, query, meter, ctx)?;
    checker.depth_offset = depth_offset;
    checker.base_bytes = base_bytes;
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
                let ancestors = checker.cached_ancestors(&state.y, &state.v, &mut ws);
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
                    let ancestors = checker.cached_ancestors(&state.y, &state.v, &mut ws);
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

/// Check an s-hedge's forest, selection, nesting, root, and query conditions.
/// No identifier is invoked and no success flag is trusted.
///
/// # Errors
/// Invalid witness, different evidence/query/graph scope, or cancellation.
#[allow(clippy::too_many_lines, reason = "Keep the independent forest conditions together")]
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
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        if components_by(nodes.len(), |i| adjacency[i].iter().copied()).len() != 1 {
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
