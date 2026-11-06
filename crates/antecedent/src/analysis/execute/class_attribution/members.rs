//! Mixed-graph members and temporal posterior atom decoding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::TableView;

use super::{
    AcceptedGraph, Admg, Arc, CausalError, DenseNodeId, Endpoint, ExecutionContext, GraphPosterior,
    GraphPosteriorAtomKind, Lag, LagAlignedScm, LagMask, MemberKind, NodeRef,
    NoiseBlockIdentification, Pag, TemporalAdmg, TemporalDag, TemporalIndexer, TemporalPag,
    TimeSeriesData, VariableId, plan_noise_blocks,
};

/// A member of a mixed graph: the DAG attribution when no latent confounding reaches the
/// ancestors of the targets, the noise-block attribution when it does and every mechanism is
/// identified, the hedge otherwise.
pub(super) fn mixed_member(admg: &Admg, targets: &[VariableId]) -> Result<MemberKind, CausalError> {
    if !admg.has_bidirected() {
        return Ok(MemberKind::Directed(AcceptedGraph::from(
            admg.directed_part().map_err(CausalError::rejected)?,
        )));
    }
    Ok(match plan_noise_blocks(admg, targets).map_err(CausalError::rejected)? {
        NoiseBlockIdentification::Identified(plan)
            if plan.blocks.iter().all(|block| block.len() == 1) =>
        {
            MemberKind::Directed(AcceptedGraph::from(plan.directed))
        }
        NoiseBlockIdentification::Identified(plan) => MemberKind::Blocks { plan, unfolding: None },
        NoiseBlockIdentification::Hedged(hedge) => MemberKind::Hedged(hedge),
    })
}

/// A MAG completion as the most-confounded mixed graph it allows, labelled as the PAG is.
pub(super) fn labelled_confounded_admg(pag: &Pag) -> Result<Admg, CausalError> {
    let confounded =
        antecedent_identify::mag_to_confounded_admg(pag).ok_or_else(|| CausalError::Compile {
            message: "a PAG completion carries a mark that no mixed graph reads".into(),
        })?;
    let mut labelled = Admg::empty();
    for node in pag.nodes() {
        labelled.add_node(*node).map_err(CausalError::rejected)?;
    }
    for from in confounded.node_ids() {
        for &to in confounded.children(from) {
            labelled.insert_directed(from, to).map_err(CausalError::rejected)?;
        }
        for &other in confounded.bidirected_neighbors(from) {
            if from.raw() < other.raw() {
                labelled.insert_bidirected(from, other).map_err(CausalError::rejected)?;
            }
        }
    }
    Ok(labelled)
}

/// The longest lag of a template.
fn template_history(nodes: &[NodeRef]) -> u32 {
    nodes
        .iter()
        .filter_map(|node| match node {
            NodeRef::Lagged { lag, .. } => Some(lag.raw()),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// A member of a temporal ADMG: the temporal DAG attribution when the template carries no
/// bidirected edge, otherwise the noise-block attribution on the template unrolled over the
/// periods the longest lag spans (the causes before the window are latent common causes of their
/// in-window effects).
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "the bounded operation keeps its execution inputs and retained graphs together"
)]
pub(super) fn temporal_mixed_member(
    template: &TemporalAdmg,
    series: &TimeSeriesData,
    targets: &[VariableId],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(MemberKind, Option<antecedent_core::execution::AdmissionLease>), CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal attribution" });
    }
    budget
        .charge_work(template.node_count() as u64)
        .map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let mut has_bidirected = false;
    for i in 0..template.node_count() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        if !template.bidirected_neighbors(DenseNodeId::from_index(i)).is_empty() {
            has_bidirected = true;
            break;
        }
    }
    if !has_bidirected {
        let nodes = template.node_count() as u64;
        let mut edges = 0u64;
        for i in 0..template.node_count() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            edges = edges
                .checked_add(template.children(DenseNodeId::from_index(i)).len() as u64)
                .ok_or_else(|| CausalError::Resource {
                message: "temporal DAG edge overflow".into(),
            })?;
        }
        if nodes > budget.limits().max_graph_nodes || edges > budget.limits().max_graph_edges {
            return Err(CausalError::Resource { message: "temporal DAG graph limit".into() });
        }
        let work = nodes
            .checked_add(edges)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_mul(edges.saturating_add(1)))
            .ok_or_else(|| CausalError::Resource {
                message: "temporal DAG work overflow".into(),
            })?;
        budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = crate::analysis::identification_cache::temporal_atom_bytes(
            nodes,
            edges,
            "temporal DAG bytes overflow",
        )?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "temporal DAG byte limit".into() });
        }
        let lease = crate::analysis::identification_cache::temporal_atom_lease(
            ctx,
            bytes,
            "temporal attribution",
        )?;
        let mut dag = TemporalDag::empty();
        for node in template.nodes() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal attribution" });
            }
            dag.add_node(*node).map_err(CausalError::rejected)?;
        }
        for index in 0..template.node_count() {
            let from = DenseNodeId::from_index(index);
            for &to in template.children(from) {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: "temporal attribution" });
                }
                dag.insert_directed(from, to).map_err(CausalError::rejected)?;
            }
        }
        return Ok((MemberKind::Directed(AcceptedGraph::from(dag)), Some(lease)));
    }
    let history = template_history(template.nodes());
    let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
        CausalError::Compile { message: "series schema exceeds the variable id range".into() }
    })?;
    let periods = history
        .checked_add(1)
        .ok_or_else(|| CausalError::Resource { message: "attribution history overflow".into() })?;
    let window = template.unroll_window_over_within(periods, variable_count, budget, ctx)?;
    let nodes = window.admg().node_count() as u64;
    budget.charge_work(nodes).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let mut edges = 0u64;
    for from in window.admg().node_ids() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        edges = edges
            .checked_add(window.admg().children(from).len() as u64)
            .and_then(|n| n.checked_add(window.admg().bidirected_neighbors(from).len() as u64))
            .ok_or_else(|| CausalError::Resource {
                message: "attribution edge count overflow".into(),
            })?;
    }
    let work =
        edges
            .checked_mul(nodes.checked_add(edges).and_then(|n| n.checked_add(1)).ok_or_else(
                || CausalError::Resource { message: "attribution work overflow".into() },
            )?)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| CausalError::Resource { message: "attribution work overflow".into() })?;
    budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let bytes = nodes
        .checked_mul(1024)
        .and_then(|n| edges.checked_mul(512).and_then(|e| n.checked_add(e)))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| CausalError::Resource { message: "attribution bytes overflow".into() })?;
    if bytes > budget.limits().max_bytes {
        return Err(CausalError::Resource { message: "attribution byte limit".into() });
    }
    let lease = Arc::new(ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal attribution" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?);
    let (unfolding, admg) =
        LagAlignedScm::unfold_mixed(series, history, window.admg(), lease, ctx)?;
    Ok((
        match plan_noise_blocks(&admg, targets).map_err(CausalError::rejected)? {
            NoiseBlockIdentification::Identified(plan) => {
                MemberKind::Blocks { plan, unfolding: Some(unfolding) }
            }
            NoiseBlockIdentification::Hedged(hedge) => MemberKind::Hedged(hedge),
        },
        None,
    ))
}

/// A temporal MAG completion as a temporal ADMG template: its directed and bidirected edges,
/// and a bidirected edge beside every directed edge that is invisible in the unfolded window
/// (an invisible edge may carry a latent common cause, Zhang 2008).
pub(super) fn temporal_mag_template(
    mag: &TemporalPag,
    series: &TimeSeriesData,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(TemporalAdmg, antecedent_core::execution::AdmissionLease), CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal attribution" });
    }
    let nodes = mag.nodes().len() as u64;
    let mut edges = 0u64;
    for _ in mag.edges_lazy() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        budget.charge_work(1).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        edges = edges.checked_add(1).ok_or_else(|| CausalError::Resource {
            message: "temporal template edge overflow".into(),
        })?;
    }
    let work = nodes
        .checked_add(edges)
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_mul(edges.saturating_add(1)))
        .and_then(|v| v.checked_mul(16))
        .ok_or_else(|| CausalError::Resource {
            message: "temporal template work overflow".into(),
        })?;
    budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let bytes = nodes
        .checked_mul(1024)
        .and_then(|n| edges.checked_mul(1024).and_then(|e| n.checked_add(e)))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| CausalError::Resource {
            message: "temporal template byte overflow".into(),
        })?;
    if bytes > budget.limits().max_bytes {
        return Err(CausalError::Resource { message: "temporal template byte limit".into() });
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal attribution" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?;
    let mut template = TemporalAdmg::empty();
    for node in mag.nodes() {
        template.add_node(*node).map_err(CausalError::rejected)?;
    }
    let tolerate = |inserted: Result<(), antecedent_graph::GraphError>| match inserted {
        Ok(()) | Err(antecedent_graph::GraphError::DuplicateEdge { .. }) => Ok(()),
        Err(error) => Err(CausalError::rejected(error)),
    };
    for edge in mag.edges() {
        if let Some((parent, child)) = edge.parent_child() {
            tolerate(template.insert_directed(parent, child))?;
        } else if edge.at_a == Endpoint::Arrow && edge.at_b == Endpoint::Arrow {
            tolerate(template.insert_bidirected(edge.a, edge.b))?;
        } else {
            return Err(CausalError::Compile {
                message: "a temporal PAG completion carries a mark that no mixed graph reads"
                    .into(),
            });
        }
    }
    let variable_count = u32::try_from(series.schema().len()).map_err(|_| {
        CausalError::Compile { message: "series schema exceeds the variable id range".into() }
    })?;
    let history = template_history(mag.nodes());
    let indexer = TemporalIndexer::new(variable_count, history, 1)?;
    let unfolded = mag.unfold_within(indexer, budget, ctx).map_err(CausalError::from)?;
    let node_of = |template: &mut TemporalAdmg,
                   variable: VariableId,
                   lag: u32|
     -> Result<DenseNodeId, CausalError> {
        let lag = Lag::from_raw(lag);
        let held = template.nodes().iter().position(|node| {
            matches!(node, NodeRef::Lagged { variable: v, lag: l } if *v == variable && *l == lag)
        });
        match held {
            Some(index) => Ok(DenseNodeId::from_index(index)),
            None => template.add_lagged(variable, lag).map_err(CausalError::rejected),
        }
    };
    for (tail, head) in antecedent_identify::invisible_directed_edges(&unfolded.pag) {
        let tail_key = unfolded.indexer.key_of(tail.raw())?;
        let head_key = unfolded.indexer.key_of(head.raw())?;
        let delay = u32::try_from(i64::from(head_key.offset) - i64::from(tail_key.offset))
            .map_err(|_| CausalError::Compile {
                message: "an invisible directed edge points backwards in time".into(),
            })?;
        let tail_node = node_of(&mut template, tail_key.variable, delay)?;
        let head_node = node_of(&mut template, head_key.variable, 0)?;
        if tail_node != head_node {
            tolerate(template.insert_bidirected(tail_node, head_node))?;
        }
    }
    Ok((template, lease))
}

/// A decoded posterior atom is either one mechanism model or a class to enumerate.
pub(super) enum SeriesAtomGraph {
    Member(MemberKind, Option<antecedent_core::execution::AdmissionLease>),
    Class(AcceptedGraph, antecedent_core::execution::AdmissionLease),
}

#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "the bounded operation keeps its execution inputs and retained graphs together"
)]
pub(super) fn series_posterior_atom(
    series: &TimeSeriesData,
    gp: &GraphPosterior,
    index: usize,
    lags: (&[LagMask], u32),
    variables: &[VariableId],
    targets: &[VariableId],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Option<SeriesAtomGraph>, CausalError> {
    let (_, max_lag) = lags;
    let graph = match gp.atom_kind() {
        GraphPosteriorAtomKind::Dag => {
            let Some((mut dag, lease)) =
                crate::analysis::identification_cache::admitted_temporal_dag_atom(
                    gp, index, variables, budget, ctx,
                )?
            else {
                return Ok(None);
            };
            // Empty templates still carry isolated target mechanisms.
            if dag.nodes().is_empty() {
                for &target in targets {
                    antecedent_graph::unfold::ensure_lagged(
                        &mut dag,
                        target,
                        antecedent_core::Lag::CONTEMPORANEOUS,
                    )
                    .map_err(CausalError::rejected)?;
                }
            }
            SeriesAtomGraph::Member(
                MemberKind::Directed(AcceptedGraph::temporal_dag(dag)),
                Some(lease),
            )
        }
        GraphPosteriorAtomKind::TemporalAdmg => {
            let Some(atom) = gp.temporal_admg_atom(index) else {
                return Ok(None);
            };
            let _template_lease =
                crate::analysis::temporal_admg_contrast::admit_temporal_atom_template(
                    &atom,
                    gp.n_vars(),
                    budget,
                    ctx,
                )?;
            let template = atom.to_template(gp.n_vars(), max_lag, variables);
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: "temporal atom" });
            }
            let Ok(template) = template else {
                return Ok(None);
            };
            let (kind, lease) = temporal_mixed_member(&template, series, targets, budget, ctx)?;
            SeriesAtomGraph::Member(kind, lease)
        }
        GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag => {
            use crate::analysis::identification_cache::AdmittedTemporalClassAtom;
            match crate::analysis::identification_cache::admitted_temporal_class_atom(
                gp, index, variables, budget, ctx,
            )? {
                Some(AdmittedTemporalClassAtom::Cpdag(graph, lease)) => {
                    let Ok(graph) = AcceptedGraph::temporal_cpdag(graph) else { return Ok(None) };
                    SeriesAtomGraph::Class(graph, lease)
                }
                Some(AdmittedTemporalClassAtom::Pag(graph, lease)) => {
                    SeriesAtomGraph::Class(AcceptedGraph::temporal_pag(graph), lease)
                }
                None => return Ok(None),
            }
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "a series attribution over a graph posterior reads Dag, \
                      TemporalAdmg, Cpdag or Pag atoms with lag masks",
            });
        }
    };
    Ok(Some(graph))
}
