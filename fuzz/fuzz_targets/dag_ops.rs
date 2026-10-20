//! Fuzz DAG insert / cycle rejection / ancestry queries, and the cyclic-graph contract.
//!
//! A [`Dag`] stays acyclic: a refused edge is not inserted, and a topological
//! order respects every surviving edge. A [`CyclicDigraph`] accepts a directed
//! cycle, refuses a self-loop, and its condensation runs sources before sinks.
#![no_main]

use antecedent_graph::{
    CyclicDigraph, CyclicNextAttempt, DSeparationWorkspace, Dag, DagOperation, DenseNodeId,
    GraphError, directed_edges_contain_cycle, refuse_directed_cycle,
};
use libfuzzer_sys::fuzz_target;

fn id(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

/// Two-cycle on a DAG is refused and not inserted. The same edges are a cycle
/// on a [`CyclicDigraph`], and a self-loop is not.
fn graph_contract_regressions() {
    let zero = id(0);
    let one = id(1);
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(zero, one).expect("chain");
    let refused = dag.insert_directed(one, zero).expect_err("two-cycle");
    assert!(matches!(refused, GraphError::Cycle { from: 1, to: 0 }));
    assert!(refused.to_string().contains("CyclicDigraph"));
    assert!(!dag.children(one).contains(&zero));
    assert!(dag.reaches(zero, one));
    assert!(!dag.reaches(one, zero));
    let order = dag.topological_order().expect("the chain is still a DAG");
    assert_eq!(order, vec![zero, one]);

    let chain = [(0u32, 1u32)];
    assert!(!directed_edges_contain_cycle(2, &chain));
    assert!(
        refuse_directed_cycle(
            DagOperation::TopologicalOrder,
            CyclicNextAttempt::Condensation,
            2,
            &chain,
        )
        .is_ok()
    );
    let cycle = [(0u32, 1u32), (1, 0)];
    let refusal = refuse_directed_cycle(
        DagOperation::TopologicalOrder,
        CyclicNextAttempt::Condensation,
        2,
        &cycle,
    )
    .expect_err("two-cycle");
    let text = refusal.to_string();
    assert!(text.contains("topological order"), "{text}");
    assert!(text.contains("condensation"), "{text}");

    let mut cyclic = CyclicDigraph::with_variables(2);
    let self_loop = cyclic.insert_directed(zero, zero).expect_err("self-loop");
    assert!(matches!(self_loop, GraphError::ContemporaneousSelfEdge { .. }));
    cyclic.insert_directed(zero, one).expect("forward");
    cyclic.insert_directed(one, zero).expect("reciprocal edge");
    cyclic.validate().expect("a directed cycle is legal");
    assert_eq!(cyclic.condensation().len(), 1);
    assert!(cyclic.children(zero).contains(&one));
    assert!(cyclic.parents(one).contains(&zero));
}

fuzz_target!(|data: &[u8]| {
    graph_contract_regressions();
    if data.is_empty() {
        return;
    }
    let n = u32::from((data[0] % 12).max(1));
    let mut dag = Dag::with_variables(n);
    let mut cyclic = CyclicDigraph::with_variables(n);
    for w in data[1..].chunks(2) {
        if w.len() < 2 {
            break;
        }
        let from = id(u32::from(w[0]) % n);
        let to = id(u32::from(w[1]) % n);
        let dag_before = dag.children(from).len();
        match dag.insert_directed(from, to) {
            Ok(()) => {
                assert!(dag.children(from).contains(&to));
                assert!(dag.parents(to).contains(&from));
            }
            Err(GraphError::Cycle { .. } | GraphError::DuplicateEdge { .. }) => {
                assert_eq!(dag.children(from).len(), dag_before);
            }
            Err(other) => panic!("dag insert {other}"),
        }
        let cyclic_before = cyclic.children(from).len();
        match cyclic.insert_directed(from, to) {
            Ok(()) => {
                assert!(cyclic.children(from).contains(&to));
                assert!(cyclic.parents(to).contains(&from));
            }
            Err(GraphError::ContemporaneousSelfEdge { .. }) => {
                assert_eq!(from, to);
                assert!(!cyclic.children(from).contains(&from));
            }
            Err(GraphError::DuplicateEdge { .. }) => {
                assert_eq!(cyclic.children(from).len(), cyclic_before);
            }
            Err(other) => panic!("cyclic insert {other}"),
        }
    }
    dag.validate().expect("inserted edges stay acyclic");
    let order = dag.topological_order().expect("a DAG has a topological order");
    assert_eq!(order.len(), usize::try_from(n).expect("node count"));
    let mut position = vec![0usize; usize::try_from(n).expect("node count")];
    for (index, node) in order.iter().enumerate() {
        position[node.as_usize()] = index;
    }
    let mut dag_edges = Vec::new();
    for node in dag.node_ids() {
        for &child in dag.children(node) {
            assert!(position[node.as_usize()] < position[child.as_usize()]);
            dag_edges.push((node.raw(), child.raw()));
        }
    }
    assert!(!directed_edges_contain_cycle(usize::try_from(n).expect("node count"), &dag_edges));
    assert!(
        refuse_directed_cycle(
            DagOperation::TopologicalOrder,
            CyclicNextAttempt::Condensation,
            usize::try_from(n).expect("node count"),
            &dag_edges,
        )
        .is_ok()
    );

    cyclic.validate().expect("cyclic graph");
    let schedule = cyclic.condensation();
    let covered: usize = schedule.components().iter().map(Vec::len).sum();
    assert_eq!(covered, usize::try_from(n).expect("node count"));
    for node in cyclic.node_ids() {
        let source = schedule.component_index(node).expect("partition");
        for &child in cyclic.children(node) {
            let sink = schedule.component_index(child).expect("partition");
            assert!(source <= sink);
        }
    }
    if n >= 2 {
        let mut ws = DSeparationWorkspace::default();
        let _ = dag.is_d_separated(id(0), id(1), &[], &mut ws);
    }
});
