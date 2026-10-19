//! Directed strongly connected components and their condensation schedule.
//!
//! The partition is Kosaraju's algorithm on directed edges. The schedule is
//! Kahn's algorithm on the condensation: a component is placed once every
//! component with an edge into it has been placed, and eligible components are
//! ordered by minimum dense id. That is sources before sinks along every
//! directed edge. An unrelated source can wait while a smaller-id chain is
//! placed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::mem::take;

use crate::types::DenseNodeId;

/// Directed strongly connected components in condensation schedule order.
///
/// [`Self::components`] is a partition of the dense nodes. Every directed edge
/// runs from a component to itself or to a later component, so the list is a
/// total order of the condensation with sources before sinks. When several
/// components have all of their predecessors already placed, the one whose
/// smallest dense id is least comes next. Members of a component are sorted by
/// dense id.
///
/// An isolated node is a source, but it is not placed ahead of a chain whose
/// ids are smaller. On nodes `0, 1, 2` with only `0 -> 1`, the schedule is
/// `[0], [1], [2]`: `0` wins the tie against `2`, then `1` is smaller than `2`.
///
/// The partition and this order depend only on the directed edges and the
/// dense labeling. Edge insertion order does not change them: adjacency lists
/// stay sorted, and the tie-break reads dense ids. Dense ids follow node
/// insertion, so [`CyclicDigraph::with_variables`] and
/// [`CyclicDigraph::from_named_edges`] are the stable labeling. Inserting the
/// same node labels in another order assigns different dense ids and can
/// change which block is placed first.
///
/// Bidirected edges are latent confounding. They are not directed edges and
/// do not change this value. The acyclic limit is one component per node, in
/// the topological order defined above.
///
/// [`CyclicDigraph::with_variables`]: crate::CyclicDigraph::with_variables
/// [`CyclicDigraph::from_named_edges`]: crate::CyclicDigraph::from_named_edges
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "dropping it drops the solve-block schedule"]
pub struct Condensation {
    components: Vec<Vec<DenseNodeId>>,
    component_of: Vec<usize>,
}

impl Condensation {
    /// Components in schedule order.
    ///
    /// Index `0` is a source block. Each block's members ascend by dense id.
    #[must_use]
    pub fn components(&self) -> &[Vec<DenseNodeId>] {
        &self.components
    }

    /// Schedule index of `id`, or `None` when `id` is outside the graph.
    ///
    /// The index is a position in [`Self::components`].
    #[must_use = "the schedule index is the only way to place a node in its block"]
    pub fn component_index(&self, id: DenseNodeId) -> Option<usize> {
        self.component_of.get(id.as_usize()).copied()
    }

    /// Number of components. An empty graph has zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

/// Kosaraju partition of `children`, scheduled by minimum-id Kahn on the condensation.
///
/// `parents` is the transpose. Both slices are the directed adjacency of one graph.
#[must_use = "the condensation is the directed solve-block schedule"]
pub(crate) fn decompose(
    children: &[Vec<DenseNodeId>],
    parents: &[Vec<DenseNodeId>],
) -> Condensation {
    debug_assert_eq!(children.len(), parents.len());
    let (components, component_of) = kosaraju(children, parents);
    schedule(children, components, component_of)
}

/// Finish order of a directed depth-first search.
///
/// Roots and children are visited in ascending dense id, which is the stored
/// adjacency order. Nodes are marked when they are pushed, so each node finishes once.
fn finish_order(children: &[Vec<DenseNodeId>]) -> Vec<usize> {
    let n = children.len();
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some(&(node, next_child)) = stack.last() {
            if next_child < children[node].len() {
                let child = children[node][next_child].as_usize();
                let frame = stack.last_mut().expect("the node being expanded is on the stack");
                frame.1 = next_child + 1;
                if !seen[child] {
                    seen[child] = true;
                    stack.push((child, 0));
                }
            } else {
                order.push(node);
                stack.pop();
            }
        }
    }
    order
}

/// Strongly connected components in Kosaraju's second-pass order.
///
/// That order discovers sinks of the original graph first. Callers replace it
/// with [`schedule`]; only the partition is kept. Members of each component
/// are sorted by dense id.
fn kosaraju(
    children: &[Vec<DenseNodeId>],
    parents: &[Vec<DenseNodeId>],
) -> (Vec<Vec<DenseNodeId>>, Vec<usize>) {
    let n = children.len();
    let finish = finish_order(children);
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    let mut component_of = vec![0usize; n];
    for &start in finish.iter().rev() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut members = Vec::new();
        while let Some(node) = stack.pop() {
            members.push(DenseNodeId::from_index(node));
            for &parent in &parents[node] {
                let next = parent.as_usize();
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        let id = components.len();
        for &member in &members {
            component_of[member.as_usize()] = id;
        }
        members.sort_unstable();
        components.push(members);
    }
    (components, component_of)
}

/// Total order of `components`: sources before sinks, ties by minimum dense id.
fn schedule(
    children: &[Vec<DenseNodeId>],
    mut components: Vec<Vec<DenseNodeId>>,
    mut component_of: Vec<usize>,
) -> Condensation {
    let k = components.len();
    let successors = condensation_edges(children, &component_of, k);
    let mut indegree = vec![0usize; k];
    for row in &successors {
        for &to in row {
            indegree[to] += 1;
        }
    }
    let mut ready = BTreeSet::new();
    for (index, component) in components.iter().enumerate() {
        if indegree[index] == 0 {
            let min_id = component.first().expect("component is non-empty").raw();
            ready.insert((min_id, index));
        }
    }
    let mut scheduled = Vec::with_capacity(k);
    while let Some((_, index)) = ready.pop_first() {
        scheduled.push(index);
        for &next in &successors[index] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                let min_id = components[next].first().expect("component is non-empty").raw();
                ready.insert((min_id, next));
            }
        }
    }
    debug_assert_eq!(
        scheduled.len(),
        k,
        "condensation of strongly connected components is acyclic",
    );

    let mut remap = vec![0usize; k];
    for (new_index, &old_index) in scheduled.iter().enumerate() {
        remap[old_index] = new_index;
    }
    for old in &mut component_of {
        *old = remap[*old];
    }
    let components =
        scheduled.into_iter().map(|old_index| take(&mut components[old_index])).collect();
    Condensation { components, component_of }
}

/// Unique condensation edges `from -> to` as component indices in the Kosaraju numbering.
fn condensation_edges(
    children: &[Vec<DenseNodeId>],
    component_of: &[usize],
    component_count: usize,
) -> Vec<Vec<usize>> {
    let mut successors = vec![Vec::new(); component_count];
    for (node, kids) in children.iter().enumerate() {
        let from = component_of[node];
        for &kid in kids {
            let to = component_of[kid.as_usize()];
            if from != to {
                successors[from].push(to);
            }
        }
    }
    for row in &mut successors {
        row.sort_unstable();
        row.dedup();
    }
    successors
}

#[cfg(test)]
mod tests {
    use antecedent_core::{CausalSchemaBuilder, VariableId};

    use super::Condensation;
    use crate::cyclic::{CyclicDigraph, CyclicMixedGraph};
    use crate::types::{DenseNodeId, NodeRef};

    fn id(raw: u32) -> DenseNodeId {
        DenseNodeId::from_raw(raw)
    }

    fn digraph(n: u32, edges: &[(u32, u32)]) -> CyclicDigraph {
        let mut graph = CyclicDigraph::with_variables(n);
        for &(from, to) in edges {
            graph.insert_directed(id(from), id(to)).unwrap();
        }
        graph
    }

    fn raw_blocks(condensation: &Condensation) -> Vec<Vec<u32>> {
        condensation
            .components()
            .iter()
            .map(|block| block.iter().map(|node| node.raw()).collect())
            .collect()
    }

    /// The schedule matches `expected`, covers every node once, and every directed
    /// edge points at the same component or a later one.
    fn assert_scheduled(graph: &CyclicDigraph, expected: &[&[u32]]) {
        let found = graph.condensation();
        let expected_blocks: Vec<Vec<u32>> = expected.iter().map(|block| block.to_vec()).collect();
        assert_eq!(raw_blocks(&found), expected_blocks);
        assert_eq!(found.len(), expected.len());
        assert_eq!(found.is_empty(), graph.is_empty());
        let mut seen = vec![false; graph.node_count()];
        for (index, block) in found.components().iter().enumerate() {
            assert!(block.iter().zip(block.iter().skip(1)).all(|(left, right)| left < right));
            for &node in block {
                assert!(!seen[node.as_usize()], "duplicate {}", node.raw());
                seen[node.as_usize()] = true;
                assert_eq!(found.component_index(node), Some(index));
            }
        }
        assert!(seen.into_iter().all(|bit| bit));
        for node in graph.node_ids() {
            let earlier = found.component_index(node).expect("node is in the decomposition");
            for &child in graph.children(node) {
                let later = found.component_index(child).expect("child is in the decomposition");
                assert!(
                    earlier <= later,
                    "component {earlier} of {} points at component {later} of {}",
                    node.raw(),
                    child.raw(),
                );
            }
        }
        assert_eq!(found.component_index(id(u32::MAX)), None);
    }

    fn abc_schema() -> antecedent_core::CausalSchema {
        CausalSchemaBuilder::new()
            .continuous("a")
            .finish()
            .continuous("b")
            .finish()
            .continuous("c")
            .finish()
            .build()
            .unwrap()
    }

    #[test]
    fn disjoint_loops_are_ordered_by_minimum_dense_id() {
        // Two directed cycles and no edge between them. Both are sources;
        // the smaller minimum id is placed first. Kosaraju's own discovery
        // order emits the higher loop first, so this locks the tie-break.
        let graph = digraph(4, &[(0, 1), (1, 0), (2, 3), (3, 2)]);
        assert_scheduled(&graph, &[&[0, 1], &[2, 3]]);
    }

    #[test]
    fn node_in_two_directed_cycles_is_one_component() {
        // 0 -> 1 -> 2 -> 0 and 2 -> 3 -> 2. Node 2 lies on both cycles, so
        // every node reaches every other node by directed edges.
        let graph = digraph(4, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 2)]);
        assert_scheduled(&graph, &[&[0, 1, 2, 3]]);
        assert_eq!(graph.condensation().len(), 1);
    }

    #[test]
    fn multiple_components_schedule_sources_before_sinks() {
        // The loop on {0, 1} feeds the loop on {2, 3}.
        let graph = digraph(4, &[(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]);
        assert_scheduled(&graph, &[&[0, 1], &[2, 3]]);
        let mixed = CyclicMixedGraph::from_cyclic_digraph(graph.clone());
        assert_eq!(mixed.condensation(), graph.condensation());
        assert!(mixed.bidirected_edge_pairs_sorted().is_empty());
    }

    #[test]
    fn isolated_nodes_are_singleton_components() {
        assert_scheduled(&CyclicDigraph::empty(), &[]);
        assert!(CyclicMixedGraph::empty().condensation().is_empty());
        assert_scheduled(&CyclicDigraph::with_variables(3), &[&[0], &[1], &[2]]);
        // Node 0 is isolated beside a loop. Both blocks are sources; id 0 is smaller.
        assert_scheduled(&digraph(3, &[(1, 2), (2, 1)]), &[&[0], &[1, 2]]);
    }

    #[test]
    fn reordered_node_insertion_follows_dense_ids() {
        // Dense ids follow insertion. `with_variables` and schema order keep
        // variable i at dense id i. The same labels inserted in another order
        // are a different labeling, and the schedule is read in that labeling.
        let named = CyclicDigraph::from_named_edges(&abc_schema(), &[("b", "a")]).unwrap();
        let mut schema_order = CyclicDigraph::with_variables(3);
        schema_order.insert_directed(id(1), id(0)).unwrap();
        assert_eq!(named.nodes(), schema_order.nodes());
        assert_eq!(named.condensation(), schema_order.condensation());
        // b -> a is dense 1 -> 0. Eligible sources are 1 and isolated 2;
        // placing 1 releases 0, whose id is smaller than 2.
        assert_scheduled(&schema_order, &[&[1], &[0], &[2]]);

        let mut swapped = CyclicDigraph::empty();
        swapped.add_node(NodeRef::Static(VariableId::from_raw(2))).unwrap();
        swapped.add_node(NodeRef::Static(VariableId::from_raw(1))).unwrap();
        swapped.add_node(NodeRef::Static(VariableId::from_raw(0))).unwrap();
        swapped.insert_directed(id(1), id(2)).unwrap();
        assert_ne!(swapped.nodes(), schema_order.nodes());
        assert_eq!(swapped.nodes()[0], NodeRef::Static(VariableId::from_raw(2)));
        // The same named edge b -> a is now dense 1 -> 2. Sources are dense 0 and 1.
        assert_scheduled(&swapped, &[&[0], &[1], &[2]]);
    }

    #[test]
    fn edge_insertion_order_does_not_change_the_schedule() {
        let forward = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)];
        let reverse = [(3, 2), (2, 3), (1, 2), (1, 0), (0, 1)];
        let shuffled = [(1, 2), (3, 2), (0, 1), (2, 3), (1, 0)];
        let first = digraph(5, &forward);
        let second = digraph(5, &reverse);
        let third = digraph(5, &shuffled);
        assert_eq!(first.condensation(), second.condensation());
        assert_eq!(first.condensation(), third.condensation());
        // Node 4 is isolated. The feeding loop is placed before the fed loop,
        // and both precede 4 because their minimum ids become eligible first.
        assert_scheduled(&first, &[&[0, 1], &[2, 3], &[4]]);
    }

    #[test]
    fn acyclic_limit_matches_minimum_id_topological_order() {
        // Each node is its own component. Eligible components are a priority
        // queue keyed by dense id, so the list is that topological order.
        assert_scheduled(&digraph(3, &[(0, 1), (1, 2)]), &[&[0], &[1], &[2]]);
        // After 0, both 1 and 2 are eligible.
        assert_scheduled(&digraph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]), &[&[0], &[1], &[2], &[3]]);
        // 0, 1, and 2 start eligible; 3 waits on 0 and on 2.
        assert_scheduled(&digraph(4, &[(2, 3), (0, 3)]), &[&[0], &[1], &[2], &[3]]);
        // 2 is a source, and it waits: 0 wins the tie, then 1 is smaller than 2.
        assert_scheduled(&digraph(3, &[(0, 1)]), &[&[0], &[1], &[2]]);
    }

    #[test]
    fn bidirected_edge_yields_singleton_components() {
        let mut confounded = CyclicMixedGraph::with_variables(2);
        confounded.insert_bidirected(id(0), id(1)).unwrap();
        assert!(confounded.directed_edge_pairs_sorted().is_empty());
        assert_eq!(confounded.bidirected_edge_pairs_sorted(), vec![(0, 1)]);
        assert_eq!(confounded.condensation(), CyclicDigraph::with_variables(2).condensation());
        assert_eq!(raw_blocks(&confounded.condensation()), vec![vec![0], vec![1]]);

        let loops = digraph(4, &[(0, 1), (1, 0), (2, 3), (3, 2)]);
        let mut bridged = CyclicMixedGraph::from_cyclic_digraph(loops.clone());
        bridged.insert_bidirected(id(1), id(2)).unwrap();
        assert_eq!(bridged.bidirected_edge_pairs_sorted(), vec![(1, 2)]);
        assert_eq!(bridged.condensation(), loops.condensation());
        assert_eq!(raw_blocks(&bridged.condensation()), vec![vec![0, 1], vec![2, 3]]);

        let mut reciprocal = CyclicMixedGraph::with_variables(2);
        reciprocal.insert_directed(id(0), id(1)).unwrap();
        reciprocal.insert_directed(id(1), id(0)).unwrap();
        assert_eq!(raw_blocks(&reciprocal.condensation()), vec![vec![0, 1]]);
    }
}
