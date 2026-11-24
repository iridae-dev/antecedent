//! `antecedent_io_core::identity::dag_identity` and `admg_identity` on every static graph of
//! three nodes labelled with distinct variables below four.
//!
//! The identity wire is what the identification digest hashes, so these proofs need no
//! property of the hash: the wire depends only on the labelled graph, never on the dense
//! order its nodes were added in, and two labelled graphs with different variables or edges
//! never share a wire. The released unlabelled wire failed the second property (the same
//! edges over different variables shared one wire).
//!
//! No stubs: graph construction and both encoders run as in production. Their loops are
//! bounded by the node count (three) or the edge count (at most three per kind), and they run
//! no recursion, hashing or tree code, so an unwind of six covers every loop with margin.
//! The reference side is written with bit masks rather than sorted vectors, so the
//! specification adds no sort of its own.

use antecedent_core::{NodeRef, VariableId};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_io_core::identity::{GraphIdentityWire, admg_identity, dag_identity};

const NODES: usize = 3;
/// The edges `slot i -> slot j` with `i < j`; acyclic in any dense order.
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
/// Every insertion order of the three slots.
const ORDERS: [[usize; NODES]; 6] =
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

fn any_labels() -> [u32; NODES] {
    let labels: [u32; NODES] = [kani::any(), kani::any(), kani::any()];
    kani::assume(labels.iter().all(|&label| label < 4));
    kani::assume(labels[0] != labels[1] && labels[0] != labels[2] && labels[1] != labels[2]);
    labels
}

fn any_edges() -> [bool; 3] {
    [kani::any(), kani::any(), kani::any()]
}

fn any_order() -> [usize; NODES] {
    let index: usize = kani::any();
    kani::assume(index < ORDERS.len());
    ORDERS[index]
}

/// Dense id of each slot when the slots are added in `order`.
fn dense_ids(order: [usize; NODES]) -> [DenseNodeId; NODES] {
    let mut dense = [DenseNodeId::from_raw(0); NODES];
    for (position, &slot) in order.iter().enumerate() {
        dense[slot] = DenseNodeId::from_raw(position as u32);
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

/// The variable set as a mask (bit `v` for variable `v`). Labels are distinct, so two label
/// triples name the same variables exactly when their masks agree.
fn variables(labels: [u32; NODES]) -> u8 {
    labels.iter().fold(0, |mask, &label| mask | (1 << label))
}

/// The directed variable pairs as a mask (bit `4 * from + to`).
fn directed_pairs(labels: [u32; NODES], edges: [bool; 3]) -> u16 {
    let mut mask = 0u16;
    for (index, &(from, to)) in PAIRS.iter().enumerate() {
        if edges[index] {
            mask |= 1 << (labels[from] * 4 + labels[to]);
        }
    }
    mask
}

/// The unordered variable pairs as a mask (bit `4 * min + max`).
fn unordered_pairs(labels: [u32; NODES], edges: [bool; 3]) -> u16 {
    let mut mask = 0u16;
    for (index, &(a, b)) in PAIRS.iter().enumerate() {
        if edges[index] {
            let (low, high) = (labels[a].min(labels[b]), labels[a].max(labels[b]));
            mask |= 1 << (low * 4 + high);
        }
    }
    mask
}

/// The DAG identity does not depend on the dense order of the nodes, and is canonical.
#[kani::proof]
#[kani::unwind(6)]
fn dag_identity_ignores_dense_order() {
    let labels = any_labels();
    let edges = any_edges();
    let first = dag_identity(&dag(labels, edges, any_order())).unwrap();
    let second = dag_identity(&dag(labels, edges, any_order())).unwrap();
    assert!(first == second);
    assert!(first.is_canonical());
}

/// Two DAGs share an identity wire only when they have the same variables and the same
/// directed edges between them.
#[kani::proof]
#[kani::unwind(6)]
fn dag_identity_binds_labels_and_edges() {
    let (left_labels, left_edges) = (any_labels(), any_edges());
    let (right_labels, right_edges) = (any_labels(), any_edges());
    let left = dag_identity(&dag(left_labels, left_edges, any_order())).unwrap();
    let right = dag_identity(&dag(right_labels, right_edges, any_order())).unwrap();
    if left == right {
        assert!(variables(left_labels) == variables(right_labels));
        assert!(
            directed_pairs(left_labels, left_edges) == directed_pairs(right_labels, right_edges)
        );
    }
}

/// A schema-aligned DAG (variables 0, 1, 2) is written in the unlabelled form of earlier
/// releases; any other is labelled.
#[kani::proof]
#[kani::unwind(6)]
fn dag_identity_keeps_the_aligned_form() {
    let labels = any_labels();
    let identity = dag_identity(&dag(labels, any_edges(), any_order())).unwrap();
    let aligned = labels.iter().all(|&label| label < 3);
    assert!(matches!(identity, GraphIdentityWire::Dag(_)) == aligned);
    assert!(matches!(identity, GraphIdentityWire::LabeledDag(_)) == !aligned);
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

/// The ADMG identity does not depend on dense order, and two ADMGs share it only when their
/// variables, directed edges and bidirected pairs agree.
#[kani::proof]
#[kani::unwind(6)]
fn admg_identity_binds_labels_not_dense_order() {
    let (labels, directed, bidirected) = (any_labels(), any_edges(), any_edges());
    let first = admg_identity(&admg(labels, directed, bidirected, any_order())).unwrap();
    let second = admg_identity(&admg(labels, directed, bidirected, any_order())).unwrap();
    assert!(first == second);
    assert!(first.is_canonical());
    let (other_labels, other_directed, other_bidirected) = (any_labels(), any_edges(), any_edges());
    let other =
        admg_identity(&admg(other_labels, other_directed, other_bidirected, any_order())).unwrap();
    if first == other {
        assert!(variables(labels) == variables(other_labels));
        assert!(directed_pairs(labels, directed) == directed_pairs(other_labels, other_directed));
        assert!(
            unordered_pairs(labels, bidirected) == unordered_pairs(other_labels, other_bidirected)
        );
    }
}
