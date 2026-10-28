//! Marked-edge invariants of `TemporalCpdag` against an independent model of the rules.
//!
//! The graph class is a representation: it stores marks and enforces local invariants (no
//! duplicate edge, no self edge between nodes with the same variable and lag at lag zero, no
//! lagged self loop, no directed cycle, no directed edge from a smaller to a larger lag index of
//! a lagged pair, `orient` only on an existing undirected edge, `mark_conflict` only on an
//! existing edge). This test writes those rules out once more as a plain edge map with a
//! depth-first reachability check and drives both the product and the model through every
//! sequence of up to three operations over four lagged nodes, up to four over three, and up to two
//! over six nodes (the four, a context node and a duplicate lag-zero node). After every operation the accept/reject decision and the full edge set must
//! agree, and at the end of every sequence the completion to a `TemporalDag` must succeed
//! exactly when no undirected or conflict edge remains and no context node is present (a
//! `TemporalDag` holds lagged nodes only), and must return exactly the directed edges. Context nodes carry no time order in the model, exactly as the product treats them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{EnvironmentId, Lag, VariableId};
use antecedent_graph::{DenseNodeId, TemporalCpdag};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// `from -> to`
    Directed(usize, usize),
    Undirected,
    Conflict,
}

#[derive(Clone, Copy)]
enum Node {
    Lagged { var: u32, lag: u32 },
    Context,
}

const LAGGED: [Node; 4] = [
    Node::Lagged { var: 0, lag: 1 },
    Node::Lagged { var: 1, lag: 1 },
    Node::Lagged { var: 0, lag: 0 },
    Node::Lagged { var: 1, lag: 0 },
];

/// The four lagged nodes, then one context node and a duplicate of the third node.
fn nodes_with_context() -> Vec<Node> {
    let mut v = LAGGED.to_vec();
    v.push(Node::Context);
    v.push(LAGGED[2]);
    v
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Directed(usize, usize),
    Undirected(usize, usize),
    Orient(usize, usize),
    Conflict(usize, usize),
}

fn key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

#[derive(Clone)]
struct Model {
    nodes: Vec<Node>,
    edges: BTreeMap<(usize, usize), Kind>,
}

impl Model {
    fn reaches(&self, from: usize, to: usize) -> bool {
        if from == to {
            return true;
        }
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![from];
        seen[from] = true;
        while let Some(u) = stack.pop() {
            for kind in self.edges.values() {
                if let Kind::Directed(p, c) = *kind {
                    if p == u && !seen[c] {
                        if c == to {
                            return true;
                        }
                        seen[c] = true;
                        stack.push(c);
                    }
                }
            }
        }
        false
    }

    fn future_to_past(&self, from: usize, to: usize) -> bool {
        matches!((self.nodes[from], self.nodes[to]),
            (Node::Lagged { lag: lf, .. }, Node::Lagged { lag: lt, .. }) if lf < lt)
    }

    fn self_edge(&self, a: usize, b: usize) -> bool {
        a == b
            || matches!((self.nodes[a], self.nodes[b]),
                (Node::Lagged { var: v1, lag: l1 }, Node::Lagged { var: v2, lag: l2 })
                    if v1 == v2 && l1 == l2 && l1 == 0)
    }

    fn apply(&mut self, op: Op) -> bool {
        match op {
            Op::Directed(a, b) => {
                if self.self_edge(a, b)
                    || self.future_to_past(a, b)
                    || self.edges.contains_key(&key(a, b))
                    || self.reaches(b, a)
                {
                    return false;
                }
                self.edges.insert(key(a, b), Kind::Directed(a, b));
            }
            Op::Undirected(a, b) => {
                if self.self_edge(a, b) || self.edges.contains_key(&key(a, b)) {
                    return false;
                }
                self.edges.insert(key(a, b), Kind::Undirected);
            }
            Op::Orient(a, b) => {
                if self.future_to_past(a, b)
                    || self.edges.get(&key(a, b)) != Some(&Kind::Undirected)
                    || self.reaches(b, a)
                {
                    return false;
                }
                self.edges.insert(key(a, b), Kind::Directed(a, b));
            }
            Op::Conflict(a, b) => {
                if !self.edges.contains_key(&key(a, b)) {
                    return false;
                }
                self.edges.insert(key(a, b), Kind::Conflict);
            }
        }
        true
    }
}

fn build(nodes: &[Node]) -> (TemporalCpdag, Vec<DenseNodeId>) {
    let mut g = TemporalCpdag::empty();
    let ids = nodes
        .iter()
        .map(|n| match *n {
            Node::Lagged { var, lag } => {
                g.add_lagged(VariableId::from_raw(var), Lag::from_raw(lag)).unwrap()
            }
            Node::Context => {
                g.add_context(VariableId::from_raw(7), Some(EnvironmentId::from_raw(1))).unwrap()
            }
        })
        .collect();
    (g, ids)
}

fn product_edges(g: &TemporalCpdag, ids: &[DenseNodeId]) -> BTreeMap<(usize, usize), Kind> {
    let index = |d: DenseNodeId| ids.iter().position(|x| *x == d).unwrap();
    g.edges()
        .into_iter()
        .map(|e| {
            let (a, b) = (index(e.a), index(e.b));
            let kind = if let Some((p, c)) = e.parent_child() {
                Kind::Directed(index(p), index(c))
            } else if e.is_undirected() {
                Kind::Undirected
            } else {
                assert!(e.is_conflict(), "only legal marks are stored");
                Kind::Conflict
            };
            (key(a, b), kind)
        })
        .collect()
}

fn check_completion(g: &TemporalCpdag, ids: &[DenseNodeId], model: &Model) {
    let has_context = model.nodes.iter().any(|n| matches!(n, Node::Context));
    let only_directed = model.edges.values().all(|k| matches!(k, Kind::Directed(..)));
    let completed = g.try_into_temporal_dag();
    // A TemporalDag holds lagged nodes only: a graph that carries a context node cannot be
    // converted even when every edge is directed.
    assert_eq!(completed.is_ok(), only_directed && !has_context, "{:?}", model.edges);
    if has_context {
        assert!(g.to_directed_skeleton().is_err());
        return;
    }
    let skeleton = g.to_directed_skeleton().unwrap();
    let mut got: Vec<(usize, usize)> = skeleton
        .edges()
        .filter_map(|e| e.parent_child())
        .map(|(p, c)| {
            (ids.iter().position(|x| *x == p).unwrap(), ids.iter().position(|x| *x == c).unwrap())
        })
        .collect();
    got.sort_unstable();
    let mut want: Vec<(usize, usize)> = model
        .edges
        .values()
        .filter_map(|k| if let Kind::Directed(p, c) = *k { Some((p, c)) } else { None })
        .collect();
    want.sort_unstable();
    assert_eq!(got, want);
}

fn all_ops(n: usize) -> Vec<Op> {
    let mut ops = Vec::new();
    for a in 0..n {
        for b in 0..n {
            ops.push(Op::Directed(a, b));
            ops.push(Op::Orient(a, b));
            if a < b {
                ops.push(Op::Undirected(a, b));
                ops.push(Op::Conflict(a, b));
            }
        }
    }
    ops
}

fn run(op: Op, g: &mut TemporalCpdag, ids: &[DenseNodeId]) -> bool {
    match op {
        Op::Directed(a, b) => g.insert_directed(ids[a], ids[b]).is_ok(),
        Op::Undirected(a, b) => g.insert_undirected(ids[a], ids[b]).is_ok(),
        Op::Orient(a, b) => g.orient_undirected(ids[a], ids[b]).is_ok(),
        Op::Conflict(a, b) => g.mark_conflict(ids[a], ids[b]).is_ok(),
    }
}

fn dfs(
    depth: usize,
    ops: &[Op],
    g: &TemporalCpdag,
    ids: &[DenseNodeId],
    model: &Model,
    trail: &mut Vec<Op>,
    counts: &mut [u64; 2],
) {
    if depth == 0 {
        return;
    }
    for &op in ops {
        let (mut g2, mut m2) = (g.clone(), model.clone());
        let accepted = run(op, &mut g2, ids);
        let expected = m2.apply(op);
        trail.push(op);
        assert_eq!(accepted, expected, "decision differs after {trail:?}");
        assert_eq!(product_edges(&g2, ids), m2.edges, "edges differ after {trail:?}");
        counts[usize::from(accepted)] += 1;
        if depth == 1 {
            check_completion(&g2, ids, &m2);
        }
        dfs(depth - 1, ops, &g2, ids, &m2, trail, counts);
        trail.pop();
    }
}

fn sweep(nodes: Vec<Node>, depth: usize) -> [u64; 2] {
    let (g, ids) = build(&nodes);
    let ops = all_ops(nodes.len());
    let mut counts = [0u64; 2];
    let model = Model { nodes, edges: BTreeMap::new() };
    dfs(depth, &ops, &g, &ids, &model, &mut Vec::new(), &mut counts);
    counts
}

#[test]
fn every_short_operation_sequence_matches_the_independent_model() {
    // four lagged nodes: every sequence of three operations
    let four = sweep(LAGGED.to_vec(), 3);
    // three lagged nodes: every sequence of four operations (longer cycles and re-orientation)
    let three = sweep(LAGGED[..3].to_vec(), 4);
    // plus a context node and a duplicate lag-zero node: every sequence of two operations
    let with_context = sweep(nodes_with_context(), 2);
    for (name, c) in [("four", four), ("three", three), ("context", with_context)] {
        eprintln!("{name}: rejected {} accepted {}", c[0], c[1]);
        assert!(c[0] > 1_000 && c[1] > 1_000, "{name}");
    }
}

#[test]
fn cycles_longer_than_two_are_rejected_after_orientation_too() {
    // 0 -> 1 -> 2 directed, then an undirected 2 - 0 may be stored but not oriented 2 -> 0
    let mut g = TemporalCpdag::empty();
    let n: Vec<DenseNodeId> = (0..3)
        .map(|v| g.add_lagged(VariableId::from_raw(v), Lag::CONTEMPORANEOUS).unwrap())
        .collect();
    g.insert_directed(n[0], n[1]).unwrap();
    g.insert_directed(n[1], n[2]).unwrap();
    g.insert_undirected(n[2], n[0]).unwrap();
    assert!(g.orient_undirected(n[2], n[0]).is_err());
    assert!(g.orient_undirected(n[0], n[2]).is_ok());
    assert!(g.insert_directed(n[2], n[0]).is_err(), "edge already present");
}
