//! An accepted discovery is the evidence a structural excitation reading may cite.
//!
//! A handwritten label is not that evidence. The compiled search must have
//! been accepted, bound to the same variable names, and oriented a directed
//! edge from the exposure to the outcome. A reverse edge, a symmetric pair, a
//! bidirected edge, or an unoriented mark refuses before any bridge is built,
//! each with its registered `events` detail.

use antecedent_core::NodeRef;
use antecedent_core::reason_code;
use antecedent_core::reason_details::events;
use antecedent_graph::{DenseNodeId, MarkedEdge};

use super::EventError;
use crate::AcceptedGraph;

/// Provenance digest of a discovery that orients `source` into `target`.
///
/// # Errors
/// The structure was not accepted from a compiled search, or it does not
/// orient this pair.
pub(super) fn structural_evidence(
    graph: &AcceptedGraph,
    source: &str,
    target: &str,
) -> Result<String, EventError> {
    let provenance = graph.provenance().ok_or_else(|| {
        EventError::bridge(
            events::DISCOVERY_MISSING,
            reason_code!("bridge_assumption_missing"),
            "a structural reading of the excitation mass needs an accepted discovery",
        )
    })?;
    let names = graph.variable_names().ok_or_else(|| {
        EventError::bridge(
            events::DISCOVERY_UNBOUND,
            reason_code!("schema_mismatch"),
            "the accepted discovery is not bound to its variable names",
        )
    })?;
    if source == target {
        return Err(EventError::bridge(
            events::SAME_VARIABLE,
            reason_code!("invalid_argument"),
            "a structural reading compares the exposure with a different outcome",
        ));
    }
    let source_index = index_of(names, source)?;
    let target_index = index_of(names, target)?;
    if let Some(refusal) = refusal(orient(graph, source_index, target_index)) {
        return Err(refusal);
    }
    Ok(crate::accepted::digest_hex(&provenance.digest()))
}

fn refusal((forward, reverse, confound, partial): (bool, bool, bool, bool)) -> Option<EventError> {
    let (slot, code, message) = if confound {
        (
            events::DISCOVERY_CONFOUNDED,
            reason_code!("effect_not_identified"),
            "the discovery edge is confounded",
        )
    } else if forward && reverse {
        (
            events::DISCOVERY_SYMMETRIC,
            reason_code!("effect_not_identified"),
            "the discovery edge is symmetric",
        )
    } else if reverse {
        (
            events::DISCOVERY_REVERSED,
            reason_code!("effect_not_identified"),
            "the discovery edge is a lead of the other variable",
        )
    } else if partial || !forward {
        (
            events::DISCOVERY_UNORIENTED,
            reason_code!("discovered_structure_unresolved"),
            "discovery did not orient a directed edge from the exposure to the outcome",
        )
    } else {
        return None;
    };
    Some(EventError::bridge(slot, code, message))
}

fn index_of(names: &[std::sync::Arc<str>], name: &str) -> Result<usize, EventError> {
    names.iter().position(|item| item.as_ref() == name).ok_or_else(|| {
        EventError::bridge(
            events::NAMES_NOT_IN_DISCOVERY,
            reason_code!("unknown_variable"),
            "the excitation names are not variables of the accepted discovery",
        )
    })
}

fn orient(graph: &AcceptedGraph, source: usize, target: usize) -> (bool, bool, bool, bool) {
    let unoriented = (false, false, false, true);
    if let Some(dag) = graph.as_dag() {
        return static_parents(
            dag.parents(id(target)),
            dag.parents(id(source)),
            source,
            target,
            false,
        );
    }
    if let Some(admg) = graph.as_admg() {
        let confound =
            admg.bidirected_neighbors(id(source)).iter().any(|node| node.as_usize() == target);
        return static_parents(
            admg.parents(id(target)),
            admg.parents(id(source)),
            source,
            target,
            confound,
        );
    }
    if let Some(pag) = graph.as_pag() {
        return one_edge(pag.edge_between(id(source), id(target)), source, target);
    }
    if let Some(cpdag) = graph.as_cpdag() {
        return cpdag.try_into_dag().map_or(unoriented, |dag| {
            static_parents(dag.parents(id(target)), dag.parents(id(source)), source, target, false)
        });
    }
    if let Some(dag) = graph.as_temporal_dag() {
        return scan(dag.nodes(), dag.edges(), source, target);
    }
    if let Some(pag) = graph.as_temporal_pag() {
        return scan(pag.nodes(), pag.edges(), source, target);
    }
    if let Some(admg) = graph.as_temporal_admg() {
        return scan(admg.nodes(), admg.edges(), source, target);
    }
    if let Some(cpdag) = graph.as_temporal_cpdag() {
        return cpdag
            .try_into_temporal_dag()
            .map_or(unoriented, |dag| scan(dag.nodes(), dag.edges(), source, target));
    }
    unoriented
}

fn id(index: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(index).unwrap_or(u32::MAX))
}

fn static_parents(
    into_target: &[DenseNodeId],
    into_source: &[DenseNodeId],
    source: usize,
    target: usize,
    confound: bool,
) -> (bool, bool, bool, bool) {
    (
        into_target.iter().any(|node| node.as_usize() == source),
        into_source.iter().any(|node| node.as_usize() == target),
        confound,
        false,
    )
}

fn one_edge(edge: Option<MarkedEdge>, source: usize, target: usize) -> (bool, bool, bool, bool) {
    let Some(edge) = edge else {
        return (false, false, false, false);
    };
    let mut forward = false;
    let mut reverse = false;
    let confound = edge.is_bidirected();
    let mut partial = false;
    match edge.parent_child() {
        Some((parent, child)) if parent.as_usize() == source && child.as_usize() == target => {
            forward = true;
        }
        Some((parent, child)) if parent.as_usize() == target && child.as_usize() == source => {
            reverse = true;
        }
        Some(_) => {}
        None => partial = !edge.is_bidirected(),
    }
    (forward, reverse, confound, partial)
}

fn scan(
    nodes: &[NodeRef],
    edges: impl IntoIterator<Item = MarkedEdge>,
    source: usize,
    target: usize,
) -> (bool, bool, bool, bool) {
    let mut forward = false;
    let mut reverse = false;
    let mut confound = false;
    let mut partial = false;
    for edge in edges {
        let Some(from) = variable(nodes, edge.a) else { continue };
        let Some(to) = variable(nodes, edge.b) else { continue };
        let pair = (from == source && to == target) || (from == target && to == source);
        if !pair {
            continue;
        }
        if edge.is_bidirected() {
            confound = true;
        }
        match edge.parent_child() {
            Some((parent, child)) => {
                let parent = variable(nodes, parent);
                let child = variable(nodes, child);
                forward |= parent == Some(source) && child == Some(target);
                reverse |= parent == Some(target) && child == Some(source);
            }
            None => partial = true,
        }
    }
    (forward, reverse, confound, partial)
}

fn variable(nodes: &[NodeRef], id: DenseNodeId) -> Option<usize> {
    let node = nodes.get(id.as_usize())?;
    let variable = match node {
        NodeRef::Static(variable)
        | NodeRef::Lagged { variable, .. }
        | NodeRef::Unfolded { variable, .. }
        | NodeRef::Context { variable, .. } => variable,
    };
    usize::try_from(variable.raw()).ok()
}
