//! Typed stationary templates packed through the existing discovery codecs.

use super::{AtomInputs, PyGraphPosterior, require_graph_names};
use antecedent_discovery::adjacency_mask::{
    adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
};
use antecedent_discovery::temporal_admg_atoms::with_lagged_edge;
use antecedent_discovery::{AdjacencyMask, GraphPosteriorAtomKind, LagMask};
use antecedent_graph::{DenseNodeId, Endpoint, MarkedEdge, NodeRef};
use pyo3::prelude::*;

struct Template {
    kind: GraphPosteriorAtomKind,
    names: Vec<String>,
    nodes: Vec<NodeRef>,
    edges: Vec<MarkedEdge>,
}

impl Template {
    fn extract(graph: &Bound<'_, PyAny>) -> Option<PyResult<Self>> {
        macro_rules! template {
            ($class:ty, $field:ident, $kind:ident) => {
                if let Ok(graph) = graph.extract::<$class>() {
                    return Some(Ok(Self {
                        kind: GraphPosteriorAtomKind::$kind,
                        names: graph.names,
                        nodes: graph.$field.nodes().to_vec(),
                        edges: graph.$field.edges().into_iter().collect(),
                    }));
                }
            };
        }
        template!(crate::graphs::TemporalDag, dag, Dag);
        template!(crate::graphs::TemporalCpdag, cpdag, Cpdag);
        template!(crate::graphs::TemporalPag, pag, Pag);
        template!(crate::graphs::TemporalAdmg, graph, TemporalAdmg);
        None
    }

    fn coordinate(&self, id: DenseNodeId) -> PyResult<(usize, u32)> {
        match self.nodes.get(id.as_usize()) {
            Some(NodeRef::Lagged { variable, lag }) if variable.as_usize() < self.names.len() => {
                Ok((variable.as_usize(), lag.raw()))
            }
            _ => Err(crate::value_err("temporal posterior atoms require named lagged nodes")),
        }
    }
}

struct Packed {
    adjacency: AdjacencyMask,
    marks: AdjacencyMask,
    lagged: LagMask,
    confounders: LagMask,
}

/// Return `None` only for a static first atom; mixed families are refused.
pub(super) fn from_graphs(
    names: &[String],
    weights: &[f64],
    graphs: &[Bound<'_, PyAny>],
    ess: Option<f64>,
) -> Option<PyResult<PyGraphPosterior>> {
    let first = Template::extract(&graphs[0])?;
    Some(first.and_then(|first| {
        let kind = first.kind;
        let templates = graphs
            .iter()
            .map(|graph| {
                let template = Template::extract(graph).ok_or_else(|| {
                    crate::value_err("all posterior atoms must share one graph family")
                })??;
                require_graph_names(&template.names, names, "Temporal graph")?;
                if template.kind != kind {
                    return Err(crate::value_err(
                        "all posterior atoms must share one graph family",
                    ));
                }
                Ok(template)
            })
            .collect::<PyResult<Vec<_>>>()?;
        pack_posterior(names, weights, &templates, kind, ess)
    }))
}

fn pack_posterior(
    names: &[String],
    weights: &[f64],
    templates: &[Template],
    kind: GraphPosteriorAtomKind,
    ess: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    let n = names.len();
    if n == 0 {
        return Err(crate::value_err("temporal posterior variable count exceeds the atom codec"));
    }
    let max_lag = templates
        .iter()
        .flat_map(|template| &template.nodes)
        .map(|node| match node {
            NodeRef::Lagged { lag, .. } => lag.raw(),
            _ => 0,
        })
        .max()
        .unwrap_or(0);
    let cells = n.checked_mul(n).ok_or_else(|| crate::value_err("lag space overflow"))?;
    let lag_bits = usize::try_from(max_lag)
        .map_err(crate::py_msg)?
        .checked_mul(cells)
        .ok_or_else(|| crate::value_err("lag space overflow"))?;
    if lag_bits > u64::BITS as usize {
        return Err(crate::value_err("temporal posterior lag space exceeds its 64-bit codec"));
    }
    let packed = templates.iter().map(pack_template).collect::<PyResult<Vec<_>>>()?;
    let mut lagged = vec![0.0; lag_bits];
    for (atom, weight) in packed.iter().zip(weights) {
        for (bit, probability) in lagged.iter_mut().enumerate() {
            if atom.lagged.bit(u32::try_from(bit).unwrap_or(u32::MAX)) {
                *probability += weight;
            }
        }
    }
    let mut inputs = AtomInputs {
        names: names.to_vec(),
        weights: weights.to_vec(),
        adjacency: packed.iter().map(|atom| atom.adjacency.clone()).collect(),
        edge_marginals: None,
        orientation_marginals: None,
        lagged_edge_marginals: Some(lagged),
        lag_masks: Some(
            packed.iter().map(|atom| atom.lagged.to_word().unwrap_or_default()).collect(),
        ),
        lag_bidirected_masks: (kind == GraphPosteriorAtomKind::TemporalAdmg)
            .then(|| {
                packed.iter().map(|atom| atom.confounders.to_word().unwrap_or_default()).collect()
            }),
        max_lag: Some(max_lag),
        ess,
        atom_kind: kind,
        mark_masks: (kind == GraphPosteriorAtomKind::Pag)
            .then(|| packed.iter().map(|atom| atom.marks.clone()).collect()),
    };
    // A purely contemporaneous temporal ADMG has a distinct core atom kind and
    // needs no lag words. Other temporal families retain max_lag=0 explicitly.
    if max_lag == 0 && kind == GraphPosteriorAtomKind::TemporalAdmg {
        inputs.lagged_edge_marginals = None;
        inputs.lag_masks = None;
        inputs.lag_bidirected_masks = None;
        inputs.max_lag = None;
    }
    PyGraphPosterior::from_atom_inputs(inputs)
}

fn pack_template(template: &Template) -> PyResult<Packed> {
    let n = template.names.len();
    let mut cpdag = antecedent_graph::Cpdag::empty();
    let mut pag = antecedent_graph::Pag::empty();
    let mut admg = antecedent_graph::Admg::empty();
    for index in 0..n {
        let variable =
            antecedent_core::VariableId::from_raw(u32::try_from(index).map_err(crate::py_msg)?);
        let node = NodeRef::Static(variable);
        cpdag.add_node(node).map_err(crate::py_err)?;
        pag.add_node(node).map_err(crate::py_err)?;
        admg.add_node(node).map_err(crate::py_err)?;
    }
    let mut packed = Packed {
        adjacency: AdjacencyMask::EMPTY,
        marks: AdjacencyMask::EMPTY,
        lagged: LagMask::EMPTY,
        confounders: LagMask::EMPTY,
    };
    for edge in &template.edges {
        let (a, al) = template.coordinate(edge.a)?;
        let (b, bl) = template.coordinate(edge.b)?;
        if al != 0 || bl != 0 {
            pack_lagged(template.kind, &mut packed, n, (a, al), (b, bl), edge)?;
            continue;
        }
        let edge = MarkedEdge {
            a: DenseNodeId::try_from_usize(a).map_err(crate::py_err)?,
            b: DenseNodeId::try_from_usize(b).map_err(crate::py_err)?,
            ..*edge
        };
        match template.kind {
            GraphPosteriorAtomKind::Dag | GraphPosteriorAtomKind::Cpdag => {
                cpdag.insert_marked(edge)
            }
            GraphPosteriorAtomKind::Pag => pag.insert_marked(edge),
            GraphPosteriorAtomKind::TemporalAdmg => admg.insert_marked(edge),
            _ => return Err(crate::value_err("unsupported temporal atom family")),
        }
        .map_err(crate::py_err)?;
    }
    match template.kind {
        GraphPosteriorAtomKind::Dag | GraphPosteriorAtomKind::Cpdag => {
            packed.adjacency = adjacency_mask_from_cpdag(&cpdag).map_err(crate::py_msg)?;
        }
        GraphPosteriorAtomKind::Pag => {
            let (adjacency, marks) = adjacency_masks_from_pag(&pag).map_err(crate::py_msg)?;
            packed.adjacency = adjacency;
            packed.marks = marks;
        }
        GraphPosteriorAtomKind::TemporalAdmg => {
            packed.adjacency = adjacency_mask_from_admg(&admg).map_err(crate::py_msg)?;
        }
        _ => return Err(crate::value_err("unsupported temporal atom family")),
    }
    Ok(packed)
}

fn pack_lagged(
    kind: GraphPosteriorAtomKind,
    packed: &mut Packed,
    n: usize,
    a: (usize, u32),
    b: (usize, u32),
    edge: &MarkedEdge,
) -> PyResult<()> {
    let (past, now, past_mark, now_mark) =
        if a.1 > b.1 { (a, b, edge.at_a, edge.at_b) } else { (b, a, edge.at_b, edge.at_a) };
    if now.1 != 0 || past.1 == 0 || edge.middle != antecedent_graph::MiddleMark::Empty {
        return Err(crate::value_err(
            "posterior templates require lagged edges into lag zero without middle marks",
        ));
    }
    let mask = match (past_mark, now_mark) {
        (Endpoint::Tail, Endpoint::Arrow) => &mut packed.lagged,
        (Endpoint::Arrow, Endpoint::Arrow) if kind == GraphPosteriorAtomKind::TemporalAdmg => {
            &mut packed.confounders
        }
        _ => {
            return Err(crate::value_err(
                "the posterior codec cannot retain these lagged endpoint marks",
            ));
        }
    };
    *mask = with_lagged_edge(mask.clone(), n, past.1, past.0, now.0, true).map_err(crate::py_msg)?;
    Ok(())
}
