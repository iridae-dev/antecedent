//! d-separation unit and property tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::many_single_char_names)]

use super::*;
use crate::error::GraphError;
use causal_core::CausalRng;

fn chain3() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    g
}

#[test]
fn chain_blocked_by_middle() {
    let g = chain3();
    let mut ws = DSeparationWorkspace::default();
    let x = DenseNodeId::from_raw(0);
    let y = DenseNodeId::from_raw(2);
    let z = [DenseNodeId::from_raw(1)];
    assert!(g.is_d_separated(x, y, &z, &mut ws).unwrap());
    assert!(!g.is_d_separated(x, y, &[], &mut ws).unwrap());
}

#[test]
fn collider_opens_with_conditioning() {
    // x -> z <- y
    let mut g = Dag::with_variables(3);
    let x = DenseNodeId::from_raw(0);
    let z = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    g.insert_directed(x, z).unwrap();
    g.insert_directed(y, z).unwrap();
    let mut ws = DSeparationWorkspace::default();
    assert!(g.is_d_separated(x, y, &[], &mut ws).unwrap());
    assert!(!g.is_d_separated(x, y, &[z], &mut ws).unwrap());
}

#[test]
fn fork_blocked_by_common_cause() {
    // x <- z -> y
    let mut g = Dag::with_variables(3);
    let x = DenseNodeId::from_raw(0);
    let z = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    g.insert_directed(z, x).unwrap();
    g.insert_directed(z, y).unwrap();
    let mut ws = DSeparationWorkspace::default();
    assert!(!g.is_d_separated(x, y, &[], &mut ws).unwrap());
    assert!(g.is_d_separated(x, y, &[z], &mut ws).unwrap());
}

#[test]
fn witness_returns_path_when_connected() {
    let g = chain3();
    let mut ws = DSeparationWorkspace::default();
    let res =
        g.d_separation(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2), &[], &mut ws).unwrap();
    match res {
        SeparationResult::Connected { active_path } => {
            assert!(active_path.len() >= 2);
            assert_eq!(active_path[0].node.raw(), 0);
            assert_eq!(active_path[active_path.len() - 1].node.raw(), 2);
        }
        SeparationResult::Separated { .. } => panic!("expected connected"),
    }
}

/// Tiny path-enumeration oracle for property tests (n ≤ 5).
fn path_oracle_d_separated(g: &Dag, x: DenseNodeId, y: DenseNodeId, z: &[DenseNodeId]) -> bool {
    let n = g.node_count();
    let zset: BitSet = {
        let mut b = BitSet::with_len(n);
        for &v in z {
            b.insert(v);
        }
        b
    };
    // DFS all simple directed... actually d-sep cares about all trails in
    // the moral sense — use same moral algorithm as reference by building
    // all undirected paths on moral ancestral graph (duplicate of main alg
    // would be circular). Instead: enumerate all simple undirected paths
    // on the full moral graph of the whole DAG and check blocking.
    // For tiny DAGs, check every simple path in the skeleton with collider rules.
    !exists_active_path_dfs(g, x, y, &zset, &mut vec![false; n], None)
}

fn exists_active_path_dfs(
    g: &Dag,
    cur: DenseNodeId,
    target: DenseNodeId,
    z: &BitSet,
    visited: &mut [bool],
    prev: Option<(DenseNodeId, EdgeKind)>,
) -> bool {
    if cur == target && prev.is_some() {
        return true;
    }
    visited[cur.as_usize()] = true;
    // Neighbors: parents and children
    let mut neighbors: Vec<(DenseNodeId, EdgeKind)> = Vec::new();
    for &p in g.parents(cur) {
        neighbors.push((p, EdgeKind::FromParent));
    }
    for &c in g.children(cur) {
        neighbors.push((c, EdgeKind::ToChild));
    }
    for (next, kind_leaving_cur) in neighbors {
        if visited[next.as_usize()] {
            continue;
        }
        let active = match prev {
            None => true, // leaving start
            Some((_prev_node, kind_in)) => is_triple_active(kind_in, kind_leaving_cur, cur, z, g),
        };
        // Edge kind relative to `next` is the reverse of the leave label at `cur`.
        let kind_at_next = match kind_leaving_cur {
            EdgeKind::ToChild => EdgeKind::FromParent,
            EdgeKind::FromParent => EdgeKind::ToChild,
        };
        if active && exists_active_path_dfs(g, next, target, z, visited, Some((cur, kind_at_next)))
        {
            visited[cur.as_usize()] = false;
            return true;
        }
    }
    visited[cur.as_usize()] = false;
    false
}

#[derive(Clone, Copy)]
enum EdgeKind {
    /// Arrived at cur from a parent (edge parent -> cur).
    FromParent,
    /// Arrived at cur from a child (traversed child <- cur).
    ToChild,
}

/// Triple (prev --in--> cur --out--> next) activity under Z.
fn is_triple_active(
    kind_in: EdgeKind,
    kind_out: EdgeKind,
    cur: DenseNodeId,
    z: &BitSet,
    g: &Dag,
) -> bool {
    let in_z = z.contains(cur);
    match (kind_in, kind_out) {
        // chain / fork: blocked iff middle in Z
        (EdgeKind::FromParent | EdgeKind::ToChild, EdgeKind::ToChild)
        | (EdgeKind::ToChild, EdgeKind::FromParent) => !in_z,
        // collider: -> cur <-
        (EdgeKind::FromParent, EdgeKind::FromParent) => in_z || has_descendant_in_z(g, cur, z),
    }
}

fn has_descendant_in_z(g: &Dag, cur: DenseNodeId, z: &BitSet) -> bool {
    let mut ws = GraphWorkspace::default();
    let mut desc = BitSet::with_len(g.node_count());
    g.descendants_of(&[cur], &mut desc, &mut ws);
    for i in 0..g.node_count() {
        let id = DenseNodeId::from_raw(u32::try_from(i).unwrap());
        if id != cur && desc.contains(id) && z.contains(id) {
            return true;
        }
    }
    false
}

/// Edge-keep probability as reciprocal of `keep_denom` (1 ⇒ always, 3 ⇒ ~1/3, …).
fn random_dag_with_density(rng: &mut CausalRng, n: u32, keep_denom: u64) -> Dag {
    let mut g = Dag::with_variables(n);
    // Prefer edges i -> j for i < j in a random order of nodes to keep acyclic.
    let mut order: Vec<u32> = (0..n).collect();
    for i in (1..n as usize).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        order.swap(i, j);
    }
    let denom = keep_denom.max(1);
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            if rng.next_u64() % denom == 0 {
                let a = DenseNodeId::from_raw(order[i]);
                let b = DenseNodeId::from_raw(order[j]);
                let _ = g.insert_directed(a, b);
            }
        }
    }
    g
}

fn random_dag(rng: &mut CausalRng, n: u32) -> Dag {
    random_dag_with_density(rng, n, 3)
}

/// Sample a Z-mask excluding x and y (uniform over remaining bits).
fn sample_z_mask(rng: &mut CausalRng, n: u32, x: u32, y: u32) -> u32 {
    let mut mask = 0u32;
    for i in 0..n {
        if i == x || i == y {
            continue;
        }
        if rng.next_u64() % 2 == 0 {
            mask |= 1 << i;
        }
    }
    mask
}

fn z_from_mask(n: u32, mask: u32) -> Vec<DenseNodeId> {
    (0..n).filter(|i| (mask & (1 << i)) != 0).map(DenseNodeId::from_raw).collect()
}

/// Whether a trail (parent/child neighbors only) is active under Z.
fn trail_active_under_z(g: &Dag, path: &[DenseNodeId], z: &[DenseNodeId]) -> bool {
    if path.len() < 2 {
        return false;
    }
    let zset: BitSet = {
        let mut b = BitSet::with_len(g.node_count());
        for &v in z {
            b.insert(v);
        }
        b
    };
    let edge_kind = |a: DenseNodeId, b: DenseNodeId| -> Option<EdgeKind> {
        if g.children(a).contains(&b) {
            Some(EdgeKind::ToChild)
        } else if g.parents(a).contains(&b) {
            Some(EdgeKind::FromParent)
        } else {
            None
        }
    };
    let mut prev_kind_at_cur = match edge_kind(path[0], path[1]) {
        Some(EdgeKind::ToChild) => EdgeKind::FromParent, // arrived at path[1] from parent
        Some(EdgeKind::FromParent) => EdgeKind::ToChild, // arrived via reverse of parent→path[0]
        None => return false,
    };
    // First hop always active when leaving the start.
    for i in 1..path.len() - 1 {
        let cur = path[i];
        let next = path[i + 1];
        let Some(kind_leaving) = edge_kind(cur, next) else {
            return false;
        };
        if !is_triple_active(prev_kind_at_cur, kind_leaving, cur, &zset, g) {
            return false;
        }
        prev_kind_at_cur = match kind_leaving {
            EdgeKind::ToChild => EdgeKind::FromParent,
            EdgeKind::FromParent => EdgeKind::ToChild,
        };
    }
    true
}

#[test]
fn property_matches_path_oracle_on_tiny_dags() {
    let mut rng = CausalRng::from_seed(42);
    let mut ws = DSeparationWorkspace::default();
    // n=4: exhaust Z masks; n=5: sample masks (2^5 full sweep is too slow).
    for &(n, graphs, masks_per_pair) in &[(4u32, 40usize, None), (5u32, 24usize, Some(12usize))] {
        for _ in 0..graphs {
            let g = random_dag(&mut rng, n);
            for x in 0..n {
                for y in 0..n {
                    if x == y {
                        continue;
                    }
                    let masks: Vec<u32> = match masks_per_pair {
                        None => (0..(1u32 << n))
                            .filter(|&mask| (mask & (1 << x)) == 0 && (mask & (1 << y)) == 0)
                            .collect(),
                        Some(k) => (0..k).map(|_| sample_z_mask(&mut rng, n, x, y)).collect(),
                    };
                    for mask in masks {
                        let z = z_from_mask(n, mask);
                        let xi = DenseNodeId::from_raw(x);
                        let yi = DenseNodeId::from_raw(y);
                        let got = g.is_d_separated(xi, yi, &z, &mut ws).unwrap();
                        let exp = path_oracle_d_separated(&g, xi, yi, &z);
                        assert_eq!(
                            got,
                            exp,
                            "mismatch x={x} y={y} z={z:?} on graph edges {:?}",
                            g.edges()
                                .map(|e| {
                                    let (a, b) = e.parent_child().unwrap();
                                    (a.raw(), b.raw())
                                })
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn property_dsep_witness_endpoints_and_activity() {
    let mut rng = CausalRng::from_seed(99);
    let mut ws = DSeparationWorkspace::default();
    for _ in 0..60 {
        let n = 4 + (rng.next_u64() % 2) as u32; // 4..=5
        let g = random_dag(&mut rng, n);
        for _ in 0..8 {
            let x = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
            let mut y = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
            while y == x {
                y = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
            }
            let mask = sample_z_mask(&mut rng, n, x.raw(), y.raw());
            let z = z_from_mask(n, mask);
            match g.d_separation(x, y, &z, &mut ws).unwrap() {
                SeparationResult::Connected { active_path } => {
                    assert!(!active_path.is_empty(), "Connected witness must be non-empty");
                    assert_eq!(active_path[0].node, x);
                    assert_eq!(active_path[active_path.len() - 1].node, y);
                    // Oracle agrees they are d-connected.
                    assert!(
                        !path_oracle_d_separated(&g, x, y, &z),
                        "Connected witness but oracle says separated"
                    );
                    // If the moral witness happens to be a DAG trail, it must be active.
                    let nodes: Vec<_> = active_path.iter().map(|s| s.node).collect();
                    let is_trail = nodes.windows(2).all(|w| {
                        g.children(w[0]).contains(&w[1]) || g.parents(w[0]).contains(&w[1])
                    });
                    if is_trail {
                        assert!(
                            trail_active_under_z(&g, &nodes, &z),
                            "DAG-trail witness must be active under Z"
                        );
                    }
                }
                SeparationResult::Separated { .. } => {
                    assert!(path_oracle_d_separated(&g, x, y, &z));
                }
            }
        }
    }
}

#[test]
fn mutilation_removes_incoming() {
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let m = g.mutilate(&[DenseNodeId::from_raw(1)]).unwrap();
    assert!(m.children(DenseNodeId::from_raw(0)).is_empty());
    assert_eq!(m.children(DenseNodeId::from_raw(1)).len(), 1);
}

#[test]
fn property_topological_order_on_random_dags() {
    let mut rng = CausalRng::from_seed(7);
    // Sparse (keep ~1/5), medium (~1/3), dense (~1/2); n up to 8.
    for &(keep_denom, trials) in &[(5u64, 40usize), (3u64, 50usize), (2u64, 40usize)] {
        for _ in 0..trials {
            let n = 3 + (rng.next_u64() % 6) as u32; // 3..=8
            let g = random_dag_with_density(&mut rng, n, keep_denom);
            let order = g.topological_order().expect("random_dag is acyclic");
            let pos = |id: DenseNodeId| order.iter().position(|&x| x == id).unwrap();
            for e in g.edges() {
                let (u, v) = e.parent_child().unwrap();
                assert!(
                    pos(u) < pos(v),
                    "edge {}→{} violates topo order {:?}",
                    u.raw(),
                    v.raw(),
                    order.iter().map(|x| x.raw()).collect::<Vec<_>>()
                );
            }
            // Legal insert that would close a cycle must fail; unchecked cycle ⇒ None order.
            if n >= 2 && order.len() >= 2 {
                let first = order[0];
                let last = *order.last().unwrap();
                if first != last && g.reaches(first, last) && !g.children(last).contains(&first) {
                    assert!(
                        matches!(g.clone().insert_directed(last, first), Err(GraphError::Cycle { .. })),
                        "back-edge closing a path must be rejected"
                    );
                    let mut cyclic = g.clone();
                    cyclic.insert_directed_unchecked(last, first);
                    assert!(
                        cyclic.topological_order().is_none(),
                        "forced cycle must yield None topological_order"
                    );
                }
            }
        }
    }
}

#[test]
fn property_mutilation_drops_incoming_on_random_dags() {
    let mut rng = CausalRng::from_seed(11);
    for _ in 0..40 {
        let n = 4u32;
        let g = random_dag(&mut rng, n);
        let t = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
        let m = g.mutilate(&[t]).unwrap();
        for i in 0..n {
            let u = DenseNodeId::from_raw(i);
            if u == t {
                continue;
            }
            assert!(
                !m.children(u).contains(&t),
                "mutilate must remove incoming into t={}",
                t.raw()
            );
        }
        // Outgoing from t preserved.
        assert_eq!(m.children(t).len(), g.children(t).len());
    }
}

#[test]
fn property_multi_treatment_mutilation_on_random_dags() {
    let mut rng = CausalRng::from_seed(13);
    for _ in 0..50 {
        let n = 5 + (rng.next_u64() % 3) as u32; // 5..=7
        let g = random_dag(&mut rng, n);
        let k = 1 + (rng.next_u64() as usize % 3); // 1..=3 treatments
        let mut treated = Vec::new();
        while treated.len() < k {
            let t = DenseNodeId::from_raw(rng.next_u64() as u32 % n);
            if !treated.contains(&t) {
                treated.push(t);
            }
        }
        let m = g.mutilate(&treated).unwrap();
        for &t in &treated {
            for i in 0..n {
                let u = DenseNodeId::from_raw(i);
                if treated.contains(&u) {
                    continue;
                }
                assert!(
                    !m.children(u).contains(&t),
                    "multi-mutilate must drop {}→{}",
                    u.raw(),
                    t.raw()
                );
            }
            // Outgoing from t to non-treated nodes preserved; edges into other
            // treatments are removed (incoming to those treatments).
            for &c in g.children(t) {
                if treated.contains(&c) {
                    assert!(!m.children(t).contains(&c));
                } else {
                    assert!(m.children(t).contains(&c));
                }
            }
        }
        // Edges not into any treatment are preserved.
        for e in g.edges() {
            let (u, v) = e.parent_child().unwrap();
            if treated.contains(&v) {
                assert!(!m.children(u).contains(&v));
            } else {
                assert!(m.children(u).contains(&v));
            }
        }
    }
}
