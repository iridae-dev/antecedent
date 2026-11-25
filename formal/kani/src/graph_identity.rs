//! Symbolic checks of the label permutation, endpoint mapping, sorting and aligned-form
//! selection used by the production DAG and ADMG encoders. Graph construction and wire
//! assembly are exercised through the public encoders by graph_identity_exhaustive.

use antecedent_core::{NodeRef, VariableId};
use antecedent_graph::DenseNodeId;
#[cfg(kani)]
use antecedent_io_core::identity::{
    IdentityNodeWire, canonical_label_order, canonical_pair, schema_aligned, sort_identity_pairs,
};

const NODES: usize = 3;
/// The edges `slot i -> slot j` with `i < j`; acyclic in any dense order.
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
/// Every dense order of the three topological slots.
const ORDERS: [[usize; NODES]; 6] =
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
#[cfg(kani)]
fn any_order() -> [usize; NODES] {
    let index: u8 = kani::any();
    kani::assume(index < 6);
    ORDERS[usize::from(index)]
}

#[cfg(kani)]
fn any_labels() -> [u32; NODES] {
    let labels: [u32; NODES] = [kani::any(), kani::any(), kani::any()];
    kani::assume(labels.iter().all(|&label| label < 4));
    kani::assume(labels[0] != labels[1] && labels[0] != labels[2] && labels[1] != labels[2]);
    labels
}

#[cfg(kani)]
fn any_edges() -> [bool; 3] {
    [kani::any(), kani::any(), kani::any()]
}

/// Dense id of each slot when the slots are added in `order`.
fn dense_ids(order: [usize; NODES]) -> [DenseNodeId; NODES] {
    let mut dense = [DenseNodeId::from_raw(0); NODES];
    for (position, &slot) in order.iter().enumerate() {
        dense[slot] = DenseNodeId::from_raw(position as u32);
    }
    dense
}

fn nodes_in(labels: [u32; NODES], order: [usize; NODES]) -> [NodeRef; NODES] {
    let mut nodes = [NodeRef::Static(VariableId::from_raw(0)); NODES];
    for (position, &slot) in order.iter().enumerate() {
        nodes[position] = NodeRef::Static(VariableId::from_raw(labels[slot]));
    }
    nodes
}

/// The directed variable pairs as a mask (bit `4 * from + to`).
#[cfg(kani)]
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
#[cfg(kani)]
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

/// Labels in ascending variable order. The three labels are distinct and below four.
#[cfg(kani)]
fn sorted_variables(labels: [u32; NODES]) -> [u32; NODES] {
    let mut sorted = [0u32; NODES];
    let mut count = 0usize;
    for variable in 0..4u32 {
        for &label in &labels {
            if label == variable {
                sorted[count] = variable;
                count += 1;
            }
        }
    }
    sorted
}

/// Index of `variable` in `sorted`.
#[cfg(kani)]
fn rank(sorted: [u32; NODES], variable: u32) -> u32 {
    let mut index = 0u32;
    for &label in &sorted {
        if label == variable {
            return index;
        }
        index += 1;
    }
    index
}

/// Selected mask bits, scanned in variable order and written as indexes into the sorted labels.
///
/// At most three pairs are selected. The scan order is the ascending endpoint order.
#[cfg(kani)]
fn endpoints(mask: u16, sorted: [u32; NODES], unordered: bool) -> ([(u32, u32); 3], usize) {
    let mut pairs = [(0u32, 0u32); 3];
    let mut count = 0usize;
    for from in 0..4u32 {
        for to in 0..4u32 {
            let selected = if unordered { from <= to } else { true };
            if selected && mask & (1u16 << (from * 4 + to)) != 0 {
                pairs[count] = (rank(sorted, from), rank(sorted, to));
                count += 1;
            }
        }
    }
    (pairs, count)
}

#[cfg(kani)]
fn assert_pair_list(actual: &[(u32, u32)], expected: [(u32, u32); 3], count: usize) {
    assert_eq!(actual.len(), count);
    for index in 0..3 {
        match actual.get(index) {
            Some(&pair) => {
                assert!(index < count);
                assert_eq!(pair.0, expected[index].0);
                assert_eq!(pair.1, expected[index].1);
            }
            None => assert!(index >= count),
        }
    }
}

#[cfg(kani)]
fn assert_sorted_nodes(nodes: &[IdentityNodeWire], sorted: [u32; NODES]) {
    assert_eq!(nodes.len(), NODES);
    for (index, &variable) in sorted.iter().enumerate() {
        match nodes.get(index) {
            Some(IdentityNodeWire::Static { variable: found }) => assert_eq!(*found, variable),
            _ => assert!(false),
        }
    }
}

/// Check the production kernels against a reference in variable space. The dense positions
/// are an independent permutation of topological slots. Forward and bidirected edges vary
/// independently. The edge list may also arrive in any order.
#[cfg(kani)]
fn check_parts(labels: [u32; 3], selected: [bool; 3], bidirected: bool) {
    let dense_order = any_order();
    let dense = dense_ids(dense_order);
    let labels_in_dense_order =
        dense_order.map(|i| IdentityNodeWire::Static { variable: labels[i] });
    let mut order = [0, 1, 2];
    let mut position = [0; 3];
    canonical_label_order(&labels_in_dense_order, &mut order, &mut position).unwrap();
    let sorted = sorted_variables(labels);
    let ordered_nodes = order.map(|i| labels_in_dense_order[i].clone());
    assert_sorted_nodes(&ordered_nodes, sorted);
    assert_eq!(schema_aligned(&ordered_nodes), sorted[0] == 0 && sorted[1] == 1 && sorted[2] == 2);
    let mut pairs = [(0, 0); 3];
    let mut n = 0;
    for index in any_order() {
        if selected[index] {
            let (a, b) = PAIRS[index];
            let (a, b) = (dense[a].raw(), dense[b].raw());
            let pair = if bidirected && b < a { (b, a) } else { (a, b) };
            pairs[n] = canonical_pair(&position, pair, bidirected).unwrap();
            n += 1;
        }
    }
    let mut result = [(0, 0); 3];
    sort_identity_pairs(&pairs[..n], &mut [(0, 0); 3][..n], &mut result[..n], &mut [0; 4]);
    let mask = if bidirected {
        unordered_pairs(labels, selected)
    } else {
        directed_pairs(labels, selected)
    };
    let (expected, count) = endpoints(mask, sorted, bidirected);
    assert_pair_list(&result[..n], expected, count);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn dag_identity_ignores_dense_order() {
    check_parts(any_labels(), any_edges(), false);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn dag_identity_binds_labels_and_edges() {
    let labels = any_labels();
    let dense = any_order();
    let nodes = dense.map(|i| IdentityNodeWire::Static { variable: labels[i] });
    let mut order = [0, 1, 2];
    let mut position = [0; 3];
    canonical_label_order(&nodes, &mut order, &mut position).unwrap();
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    kani::assume(a < 3 && b < 3);
    let (x, y) = canonical_pair(&position, (u32::from(a), u32::from(b)), false).unwrap();
    assert_eq!(order[x as usize], usize::from(a));
    assert_eq!(order[y as usize], usize::from(b));
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn dag_identity_keeps_the_aligned_form() {
    let labels = any_labels();
    let nodes = labels.map(|variable| IdentityNodeWire::Static { variable });
    let mut order = [0, 1, 2];
    let mut position = [0; 3];
    canonical_label_order(&nodes, &mut order, &mut position).unwrap();
    let aligned = schema_aligned(&order.map(|i| nodes[i].clone()));
    assert_eq!(aligned, labels.iter().all(|&v| v < 3));
    if aligned {
        for i in 0..3 {
            assert_eq!(position[i], labels[i]);
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn admg_identity_binds_labels_not_dense_order() {
    check_parts(any_labels(), any_edges(), kani::any());
}

/// `from_nodes` and `add_node` claim the same static labels: a repeated label is refused, and
/// two insertion orders of one labelled graph have the same identity wire.
#[cfg(test)]
mod constructor_agreement {
    use antecedent_core::{NodeRef, VariableId};
    use antecedent_graph::{Admg, Dag};
    use antecedent_io_core::identity::{admg_identity, dag_identity};

    use super::{NODES, dense_ids, nodes_in};

    fn static_node(variable: u32) -> NodeRef {
        NodeRef::Static(VariableId::from_raw(variable))
    }

    fn add_dag(nodes: &[NodeRef; NODES]) -> Result<Dag, antecedent_graph::GraphError> {
        let mut graph = Dag::empty();
        for &node in nodes {
            graph.add_node(node)?;
        }
        Ok(graph)
    }

    fn add_admg(nodes: &[NodeRef; NODES]) -> Result<Admg, antecedent_graph::GraphError> {
        let mut graph = Admg::empty();
        for &node in nodes {
            graph.add_node(node)?;
        }
        Ok(graph)
    }

    /// Both graphs refuse every static label they already hold, and both accept one new label.
    fn dag_claims_agree(nodes: &[NodeRef; NODES], built: &Dag, added: &Dag) {
        let (mut built, mut added) = (built.clone(), added.clone());
        for &node in nodes {
            assert!(built.add_node(node).is_err());
            assert!(added.add_node(node).is_err());
        }
        let fresh = static_node(40);
        assert!(built.add_node(fresh).is_ok());
        assert!(added.add_node(fresh).is_ok());
    }

    fn admg_claims_agree(nodes: &[NodeRef; NODES], built: &Admg, added: &Admg) {
        let (mut built, mut added) = (built.clone(), added.clone());
        for &node in nodes {
            assert!(built.add_node(node).is_err());
            assert!(added.add_node(node).is_err());
        }
        let fresh = static_node(40);
        assert!(built.add_node(fresh).is_ok());
        assert!(added.add_node(fresh).is_ok());
    }

    #[test]
    fn from_nodes_agrees_with_add_node_on_repeats_and_two_orders() {
        let repeated = [static_node(1), static_node(2), static_node(1)];
        assert!(Dag::from_nodes(&repeated).is_err());
        assert!(add_dag(&repeated).is_err());
        assert!(Admg::from_nodes(&repeated).is_err());
        assert!(add_admg(&repeated).is_err());

        let orders = [[0usize, 1, 2], [2, 0, 1]];
        for labels in [[0u32, 1, 2], [0, 1, 3]] {
            let mut dag_wires = Vec::with_capacity(2);
            let mut admg_wires = Vec::with_capacity(2);
            for order in orders {
                let nodes = nodes_in(labels, order);
                let dense = dense_ids(order);
                let (from, to, other) = (dense[0], dense[1], dense[2]);

                let mut built = Dag::from_nodes(&nodes).unwrap();
                let mut added = add_dag(&nodes).unwrap();
                built.insert_directed(from, to).unwrap();
                added.insert_directed(from, to).unwrap();
                assert_eq!(built, added);
                dag_claims_agree(&nodes, &built, &added);
                let wire = dag_identity(&built).unwrap();
                assert_eq!(wire, dag_identity(&added).unwrap());
                dag_wires.push(wire);

                let mut built = Admg::from_nodes(&nodes).unwrap();
                let mut added = add_admg(&nodes).unwrap();
                built.insert_directed(from, to).unwrap();
                added.insert_directed(from, to).unwrap();
                built.insert_bidirected(other, to).unwrap();
                added.insert_bidirected(other, to).unwrap();
                assert_eq!(built, added);
                admg_claims_agree(&nodes, &built, &added);
                let wire = admg_identity(&built).unwrap();
                assert_eq!(wire, admg_identity(&added).unwrap());
                admg_wires.push(wire);
            }
            assert_eq!(dag_wires[0], dag_wires[1]);
            assert_eq!(admg_wires[0], admg_wires[1]);
        }
    }
}
