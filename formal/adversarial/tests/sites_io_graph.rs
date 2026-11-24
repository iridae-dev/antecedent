//! Identity sites of antecedent-io-core's graph, inference, snapshot, transport and
//! propagation-chain identities (formal/identity_sites.toml, group io_graph), checked on the
//! production encoders with the category laws of `common`.
//!
//! Graph sites: the identity wire of a labelled graph does not depend on the dense order of
//! its nodes or the order its edges were inserted in (set), and two graphs share a wire only
//! when their variables and labelled edges agree (injective). The wire is what the
//! identification digest hashes, so no property of the hash is used.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_contract::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::{CausalSchemaBuilder, Lag, NodeRef, VariableId};
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Pag, SelectionDiagram, TemporalAdmg, TemporalCpdag, TemporalDag,
};
use antecedent_io_core::identity::{
    GraphIdentityWire, PosteriorAtomGraphWire, PosteriorAtomIdentityWire, admg_identity,
    canonical_dag_wire, canonical_temporal_dag_wire, cpdag_identity, dag_identity, pag_identity,
    temporal_admg_identity, temporal_cpdag_identity, transport_identity,
};
use antecedent_io_core::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire, DagWire, PriorMapping,
    PriorMappingIdentityWire, PropagationChainRecordWire, observation_identity_wire,
    propagation_chain_identity,
};
use common::{config, distinct_permuted, injective, multiset_laws, ordered_laws, set_laws};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Generated labelled graphs.
// ---------------------------------------------------------------------------

/// A labelled graph: distinct variables per slot, and at most one edge per slot pair
/// `(i, j)` with `i < j`, of kind 1 or 2 (the class decides what the kinds mean).
#[derive(Clone, Debug)]
struct Graph {
    labels: Vec<u32>,
    edges: Vec<(usize, usize, u8)>,
}

fn graph() -> impl Strategy<Value = Graph> {
    prop::collection::btree_set(0u32..6, 2..5)
        .prop_flat_map(|set| {
            let labels: Vec<u32> = set.into_iter().collect();
            let n = labels.len();
            let pairs: Vec<(usize, usize)> =
                (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
            let count = pairs.len();
            (Just(labels).prop_shuffle(), Just(pairs), prop::collection::vec(0u8..3, count))
        })
        .prop_map(|(labels, pairs, kinds)| Graph {
            labels,
            edges: pairs
                .into_iter()
                .zip(kinds)
                .filter(|(_, kind)| *kind != 0)
                .map(|((i, j), kind)| (i, j, kind))
                .collect(),
        })
}

/// A graph with a permutation of its slots (an insertion order) and of its edges.
fn graph_with_orders() -> impl Strategy<Value = (Graph, Vec<usize>, Vec<(usize, usize, u8)>)> {
    graph().prop_flat_map(|graph| {
        let slots: Vec<usize> = (0..graph.labels.len()).collect();
        let edges = graph.edges.clone();
        (Just(graph), Just(slots).prop_shuffle(), Just(edges).prop_shuffle())
    })
}

/// The labelled graph as the identity rule reads it. `symmetric` names the kind whose edges
/// have no direction.
fn key(graph: &Graph, symmetric: Option<u8>) -> (Vec<u32>, Vec<(u32, u32, u8)>) {
    let mut variables = graph.labels.clone();
    variables.sort_unstable();
    let mut edges: Vec<(u32, u32, u8)> = graph
        .edges
        .iter()
        .map(|&(i, j, kind)| {
            let (a, b) = (graph.labels[i], graph.labels[j]);
            if Some(kind) == symmetric { (a.min(b), a.max(b), kind) } else { (a, b, kind) }
        })
        .collect();
    edges.sort_unstable();
    (variables, edges)
}

fn natural(graph: &Graph) -> Vec<usize> {
    (0..graph.labels.len()).collect()
}

/// Dense id of each slot when slots are added in `order`.
fn dense_ids(order: &[usize]) -> Vec<DenseNodeId> {
    let mut dense = vec![DenseNodeId::from_raw(0); order.len()];
    for (position, &slot) in order.iter().enumerate() {
        dense[slot] = DenseNodeId::from_raw(u32::try_from(position).unwrap());
    }
    dense
}

fn variable(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn lag0(raw: u32) -> NodeRef {
    NodeRef::Lagged { variable: variable(raw), lag: Lag::CONTEMPORANEOUS }
}

/// One builder per graph class: add the slots in `order`, insert `edges` in the given order.
macro_rules! builder {
    ($name:ident, $graph:ty, $node:expr, $insert:expr) => {
        fn $name(labels: &[u32], order: &[usize], edges: &[(usize, usize, u8)]) -> Option<$graph> {
            let mut graph = <$graph>::empty();
            for &slot in order {
                graph.add_node($node(labels[slot])).ok()?;
            }
            let dense = dense_ids(order);
            for &(i, j, kind) in edges {
                $insert(&mut graph, dense[i], dense[j], kind).ok()?;
            }
            Some(graph)
        }
    };
}

builder!(dag, Dag, |raw| NodeRef::Static(variable(raw)), |g: &mut Dag, a, b, _| g
    .insert_directed(a, b));
builder!(admg, Admg, |raw| NodeRef::Static(variable(raw)), |g: &mut Admg, a, b, kind| {
    if kind == 1 { g.insert_directed(a, b) } else { g.insert_bidirected(a, b) }
});
builder!(cpdag, Cpdag, |raw| NodeRef::Static(variable(raw)), |g: &mut Cpdag, a, b, kind| {
    if kind == 1 { g.insert_directed(a, b) } else { g.insert_undirected(a, b) }
});
builder!(pag, Pag, |raw| NodeRef::Static(variable(raw)), |g: &mut Pag, a, b, kind| {
    if kind == 1 { g.insert_directed(a, b) } else { g.insert_circle_arrow(a, b) }
});
builder!(temporal_dag, TemporalDag, lag0, |g: &mut TemporalDag, a, b, _| g.insert_directed(a, b));
builder!(temporal_admg, TemporalAdmg, lag0, |g: &mut TemporalAdmg, a, b, kind| {
    if kind == 1 { g.insert_directed(a, b) } else { g.insert_bidirected(a, b) }
});
builder!(temporal_cpdag, TemporalCpdag, lag0, |g: &mut TemporalCpdag, a, b, kind| {
    if kind == 1 { g.insert_directed(a, b) } else { g.insert_undirected(a, b) }
});

/// Laws of one graph class: the wire is independent of node and edge insertion order, and
/// injective on the labelled graph. `directed_only` keeps only kind-1 edges (DAG classes).
fn graph_laws<W: PartialEq + std::fmt::Debug>(
    (graph, order, shuffled_edges): &(Graph, Vec<usize>, Vec<(usize, usize, u8)>),
    other: &Graph,
    symmetric: Option<u8>,
    directed_only: bool,
    variable_space: bool,
    identity: impl Fn(&[u32], &[usize], &[(usize, usize, u8)]) -> Option<W>,
) -> Result<(), TestCaseError> {
    let restrict = |graph: &Graph| Graph {
        labels: graph.labels.clone(),
        edges: graph.edges.iter().copied().filter(|e| !directed_only || e.2 == 1).collect(),
    };
    let (graph, other) = (restrict(graph), restrict(other));
    let shuffled: Vec<_> =
        shuffled_edges.iter().copied().filter(|e| !directed_only || e.2 == 1).collect();
    let base = identity(&graph.labels, &natural(&graph), &graph.edges);
    prop_assume!(base.is_some());
    prop_assert_eq!(
        &base,
        &identity(&graph.labels, order, &shuffled),
        "insertion order changed it"
    );
    set_laws(&graph.edges, &shuffled, |edges| identity(&graph.labels, &natural(&graph), edges))?;
    // In variable space a graph names every variable up to its largest; a smaller one it
    // does not name is an isolated node.
    let canonical = |g: &Graph| {
        let (variables, edges) = key(g, symmetric);
        let named = if variable_space { variables.last().map_or(0, |max| max + 1) } else { 0 };
        (if variable_space { Vec::new() } else { variables }, named, edges)
    };
    injective(&graph, &other, canonical, |g| identity(&g.labels, &natural(g), &g.edges))
}

// ---------------------------------------------------------------------------
// Other generated parts.
// ---------------------------------------------------------------------------

fn atom() -> impl Strategy<Value = PosteriorAtomIdentityWire> {
    (prop::sample::select(vec![0.25f64, 0.5, 0.75]), 0u32..3, any::<u64>()).prop_map(
        |(weight, edge, execution_key)| PosteriorAtomIdentityWire {
            format: antecedent_core::IDENTITY_FORMAT,
            weight_bits: weight.to_bits(),
            graph: PosteriorAtomGraphWire::Static(DagWire {
                node_count: 3,
                edges: if edge == 0 { vec![] } else { vec![(0, edge)] },
            }),
            execution_key,
        },
    )
}

fn posterior(atoms: &[PosteriorAtomIdentityWire]) -> GraphIdentityWire {
    GraphIdentityWire::GraphPosterior {
        graph_class: "dag".into(),
        n_atoms: atoms.len() as u64,
        atoms: atoms.to_vec(),
    }
}

/// The posterior premises without the atom count, so a multiset law compares the atoms only.
fn premises_atoms(atoms: &[PosteriorAtomIdentityWire]) -> Option<Vec<PosteriorAtomIdentityWire>> {
    match posterior(atoms).premises() {
        GraphIdentityWire::GraphPosterior { atoms, .. } => Some(atoms),
        _ => None,
    }
}

fn chain(
    sources: &[(String, Vec<(u8, String)>)],
    stage_inputs: &[usize],
) -> Option<PropagationChainRecordWire> {
    let kind = |tag: u8| match tag {
        0 => SharedFactorKind::Row,
        1 => SharedFactorKind::Cluster,
        _ => SharedFactorKind::Prior,
    };
    let wires: Vec<ChainSourceWire> = sources
        .iter()
        .map(|(id, factors)| {
            factors.iter().fold(
                ChainSourceWire::draws(id, LawMeaning::Posterior, "table", &[0.1, 0.2, 0.3]),
                |source, (tag, factor)| source.with_factor(kind(*tag), factor),
            )
        })
        .collect();
    let inputs: Vec<&str> = stage_inputs
        .iter()
        .filter_map(|&index| sources.get(index).map(|(id, _)| id.as_str()))
        .collect();
    let stage = ChainStageWire::new(
        "out",
        StageKind::DerivedQuantity,
        &inputs,
        ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
    );
    Some(PropagationChainRecordWire::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "out",
        wires,
        vec![stage],
    ))
}

proptest! {
    #![proptest_config(config())]

    /// `GraphIdentityWire::premises`: posterior atoms are a weighted multiset; the execution
    /// key is audit data and never reaches the premises.
    #[test]
    fn site_graph_posterior_premises(
        (atoms, shuffled) in common::permuted(atom(), 1..5),
        relabel in any::<u64>(),
    ) {
        multiset_laws(&atoms, &shuffled, premises_atoms)?;
        let mut renumbered = atoms.clone();
        renumbered[0].execution_key = relabel;
        prop_assert_eq!(premises_atoms(&atoms), premises_atoms(&renumbered));
    }

    /// `dag_identity`.
    #[test]
    fn site_dag_identity(parts in graph_with_orders(), other in graph()) {
        graph_laws(&parts, &other, None, true, false, |labels, order, edges| {
            dag_identity(&dag(labels, order, edges)?).ok()
        })?;
    }

    /// `admg_identity`.
    #[test]
    fn site_admg_identity(parts in graph_with_orders(), other in graph()) {
        graph_laws(&parts, &other, Some(2), false, false, |labels, order, edges| {
            admg_identity(&admg(labels, order, edges)?).ok()
        })?;
    }

    /// `canonical_dag_wire`: the unlabelled DAG wire of a schema-aligned DAG is independent of
    /// edge insertion order (labels are the dense indexes, so only aligned graphs are given).
    #[test]
    fn site_canonical_dag_wire((parts, _, shuffled) in graph_with_orders()) {
        let n = u32::try_from(parts.labels.len()).unwrap();
        let aligned: Vec<u32> = (0..n).collect();
        let order: Vec<usize> = (0..parts.labels.len()).collect();
        let directed = |edges: &[(usize, usize, u8)]| -> Vec<(usize, usize, u8)> {
            edges.iter().copied().filter(|e| e.2 == 1).collect()
        };
        set_laws(&directed(&parts.edges), &directed(&shuffled), |edges| {
            canonical_dag_wire(&dag(&aligned, &order, edges)?).ok()
        })?;
    }

    /// `canonical_admg_wire` and `transport_identity`: the selection graph is in variable
    /// space (independent of insertion order, injective on the labelled graph) and the
    /// selection targets are a set.
    #[test]
    fn site_transport_identity(parts in graph_with_orders(), other in graph()) {
        let selection = |labels: &[u32], order: &[usize], edges: &[(usize, usize, u8)]| {
            let graph = admg(labels, order, edges)?;
            // The largest variable: a target fixed by the variable-space graph, whatever the
            // slot order.
            let target = variable(*labels.iter().max()?);
            let diagram = SelectionDiagram::try_new(graph, [target]).ok()?;
            transport_identity(Some(&diagram), None).ok()
        };
        graph_laws(&parts, &other, Some(2), false, true, selection)?;
        let (graph, _, _) = &parts;
        let targets: Vec<VariableId> = graph.labels.iter().copied().map(variable).collect();
        let mut reversed = targets.clone();
        reversed.reverse();
        set_laws(&targets, &reversed, |targets| {
            let built = admg(&graph.labels, &natural(graph), &[])?;
            let diagram = SelectionDiagram::try_new(built, targets.to_vec()).ok()?;
            transport_identity(Some(&diagram), None).ok()
        })?;
    }

    /// `cpdag_identity`.
    #[test]
    fn site_cpdag_identity(parts in graph_with_orders(), other in graph()) {
        graph_laws(&parts, &other, Some(2), false, false, |labels, order, edges| {
            cpdag_identity(&cpdag(labels, order, edges)?).ok()
        })?;
    }

    /// `pag_identity`.
    #[test]
    fn site_pag_identity(parts in graph_with_orders(), other in graph()) {
        graph_laws(&parts, &other, None, false, false, |labels, order, edges| {
            pag_identity(&pag(labels, order, edges)?).ok()
        })?;
    }

    /// `canonical_temporal_dag_wire`.
    #[test]
    fn site_canonical_temporal_dag_wire(parts in graph_with_orders(), other in graph()) {
        graph_laws(&parts, &other, None, true, false, |labels, order, edges| {
            canonical_temporal_dag_wire(&temporal_dag(labels, order, edges)?).ok()
        })?;
    }

    /// `canonical_temporal_admg_wire` (through `temporal_admg_identity`).
    #[test]
    fn site_canonical_temporal_admg_wire(parts in graph_with_orders(), other in graph()) {
        graph_laws(&parts, &other, Some(2), false, false, |labels, order, edges| {
            temporal_admg_identity(&temporal_admg(labels, order, edges)?).ok()
        })?;
    }

    /// `temporal_class_identity` (through `temporal_cpdag_identity`).
    #[test]
    fn site_temporal_class_identity(parts in graph_with_orders(), other in graph()) {
        graph_laws(&parts, &other, Some(2), false, false, |labels, order, edges| {
            temporal_cpdag_identity(&temporal_cpdag(labels, order, edges)?).ok()
        })?;
    }

    /// `PriorMappingIdentityWire::from_mapping`: named parameter pairs are a set.
    #[test]
    fn site_prior_mapping_from_mapping(
        (pairs, shuffled) in distinct_permuted(("[a-c:]{0,2}", "[a-c:]{0,2}"), 1..5),
    ) {
        set_laws(&pairs, &shuffled, |pairs| {
            Some(PriorMappingIdentityWire::from_mapping(&PriorMapping::NamedParameters {
                pairs: pairs.to_vec(),
            }))
        })?;
    }

    /// `observation_identity_wire`: observation tags are a set.
    #[test]
    fn site_observation_identity_wire(
        (tags, shuffled) in distinct_permuted(common::adversarial_text(), 1..5),
    ) {
        let schema = CausalSchemaBuilder::new()
            .continuous("t")
            .treatment()
            .continuous("y")
            .outcome()
            .build()
            .unwrap();
        set_laws(&tags, &shuffled, |tags| Some(observation_identity_wire(&schema, tags.to_vec())))?;
    }

    /// `propagation_chain_identity`: sources are ordered (their order is the chain's), and
    /// stage inputs are ordered.
    #[test]
    fn site_propagation_chain_identity(
        ids in prop::collection::btree_set("[a-c]{1,2}", 2..4),
        i in any::<usize>(),
        j in any::<usize>(),
    ) {
        let sources: Vec<(String, Vec<(u8, String)>)> =
            ids.into_iter().map(|id| (id, vec![])).collect();
        let all: Vec<usize> = (0..sources.len()).collect();
        ordered_laws(&sources, i, j, |sources| {
            propagation_chain_identity(&chain(sources, &all)?).ok()
        })?;
        ordered_laws(&all, i, j, |inputs| {
            propagation_chain_identity(&chain(&sources, inputs)?).ok()
        })?;
    }

    /// `write_factors`: a source's declared factors are a set of `(kind, id)`.
    #[test]
    fn site_write_factors(
        (factors, shuffled) in distinct_permuted((0u8..3, "[a-c]{1,2}"), 1..5),
    ) {
        set_laws(&factors, &shuffled, |factors| {
            let sources = vec![("s".to_owned(), factors.to_vec())];
            propagation_chain_identity(&chain(&sources, &[0])?).ok()
        })?;
    }
}
