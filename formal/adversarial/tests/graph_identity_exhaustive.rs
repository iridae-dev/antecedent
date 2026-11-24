//! Static graph identity on every graph of three nodes, enumerated.
//!
//! `dag_identity` and `admg_identity` over every labelling of three slots with distinct
//! variables below four, every directed edge set (edges `slot i -> slot j`, `i < j`, acyclic in
//! any dense order), every bidirected set for the ADMG, and every insertion order, on the
//! production encoders. The Kani harnesses of `formal/kani/src/graph_identity.rs` state the
//! same properties over the same bound.
//!
//! The expected outcome is the stated identity rule: the wire depends only on the labelled
//! graph, a schema-aligned graph keeps the unlabelled form, and two graphs share a wire only
//! when their variables, directed edges and bidirected pairs agree.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{NodeRef, VariableId};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_io_core::identity::{GraphIdentityWire, admg_identity, dag_identity};

const NODES: usize = 3;
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
const ORDERS: [[usize; NODES]; 6] =
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

/// Every labelling of the three slots with distinct variables below four.
fn labellings() -> Vec<[u32; NODES]> {
    let mut out = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                if a != b && a != c && b != c {
                    out.push([a, b, c]);
                }
            }
        }
    }
    out
}

/// Every subset of the three slot pairs.
fn edge_sets() -> Vec<[bool; 3]> {
    (0..8_u8).map(|bits| [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0]).collect()
}

fn dense_ids(order: [usize; NODES]) -> [DenseNodeId; NODES] {
    let mut dense = [DenseNodeId::from_raw(0); NODES];
    for (position, &slot) in order.iter().enumerate() {
        dense[slot] = DenseNodeId::from_raw(u32::try_from(position).unwrap());
    }
    dense
}

fn dag(labels: [u32; NODES], edges: [bool; 3], order: [usize; NODES]) -> Dag {
    let mut graph = Dag::empty();
    for slot in order {
        graph.add_node(NodeRef::Static(VariableId::from_raw(labels[slot]))).unwrap();
    }
    let dense = dense_ids(order);
    for (index, &(from, to)) in PAIRS.iter().enumerate() {
        if edges[index] {
            graph.insert_directed(dense[from], dense[to]).unwrap();
        }
    }
    graph
}

fn admg(
    labels: [u32; NODES],
    directed: [bool; 3],
    bidirected: [bool; 3],
    order: [usize; NODES],
) -> Admg {
    let mut graph = Admg::empty();
    for slot in order {
        graph.add_node(NodeRef::Static(VariableId::from_raw(labels[slot]))).unwrap();
    }
    let dense = dense_ids(order);
    for (index, &(from, to)) in PAIRS.iter().enumerate() {
        if directed[index] {
            graph.insert_directed(dense[from], dense[to]).unwrap();
        }
        if bidirected[index] {
            graph.insert_bidirected(dense[to], dense[from]).unwrap();
        }
    }
    graph
}

/// The labelled graph: sorted variables, sorted directed pairs and sorted unordered
/// bidirected pairs of variables.
type Labelled = (Vec<u32>, Vec<(u32, u32)>, Vec<(u32, u32)>);

fn labelled(labels: [u32; NODES], directed: [bool; 3], bidirected: [bool; 3]) -> Labelled {
    let mut variables = labels.to_vec();
    variables.sort_unstable();
    let pick = |edges: [bool; 3], unordered: bool| {
        let mut pairs: Vec<(u32, u32)> = PAIRS
            .iter()
            .enumerate()
            .filter(|&(index, _)| edges[index])
            .map(|(_, &(a, b))| {
                let (a, b) = (labels[a], labels[b]);
                if unordered { (a.min(b), a.max(b)) } else { (a, b) }
            })
            .collect();
        pairs.sort_unstable();
        pairs
    };
    (variables, pick(directed, false), pick(bidirected, true))
}

fn aligned(labels: [u32; NODES]) -> bool {
    labels.iter().all(|&label| label < 3)
}

#[test]
fn graph_identity_exhaustive_three_nodes_dag() {
    let none = [false; 3];
    let mut seen: Vec<(GraphIdentityWire, Labelled)> = Vec::new();
    for labels in labellings() {
        for edges in edge_sets() {
            let reference = dag_identity(&dag(labels, edges, ORDERS[0])).unwrap();
            assert!(reference.is_canonical());
            assert_eq!(matches!(reference, GraphIdentityWire::Dag(_)), aligned(labels));
            assert_eq!(matches!(reference, GraphIdentityWire::LabeledDag(_)), !aligned(labels));
            for order in ORDERS {
                assert_eq!(dag_identity(&dag(labels, edges, order)).unwrap(), reference);
            }
            let graph = labelled(labels, edges, none);
            for (wire, other) in &seen {
                assert_eq!(*wire == reference, *other == graph, "{labels:?} {edges:?}");
            }
            seen.push((reference, graph));
        }
    }
}

#[test]
fn graph_identity_exhaustive_three_nodes_admg() {
    let mut seen: Vec<(GraphIdentityWire, Labelled)> = Vec::new();
    for labels in labellings() {
        for directed in edge_sets() {
            for bidirected in edge_sets() {
                let reference =
                    admg_identity(&admg(labels, directed, bidirected, ORDERS[0])).unwrap();
                assert!(reference.is_canonical());
                for order in ORDERS {
                    assert_eq!(
                        admg_identity(&admg(labels, directed, bidirected, order)).unwrap(),
                        reference
                    );
                }
                let graph = labelled(labels, directed, bidirected);
                for (wire, other) in &seen {
                    assert_eq!(*wire == reference, *other == graph, "{labels:?} {directed:?}");
                }
                seen.push((reference, graph));
            }
        }
    }
}
