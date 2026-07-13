//! d-separation unit and property tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::many_single_char_names)]

use super::*;
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

fn random_dag(rng: &mut CausalRng, n: u32) -> Dag {
    let mut g = Dag::with_variables(n);
    // Prefer edges i -> j for i < j in a random order of nodes to keep acyclic.
    let mut order: Vec<u32> = (0..n).collect();
    for i in (1..n as usize).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        order.swap(i, j);
    }
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            if rng.next_u64() % 3 == 0 {
                let a = DenseNodeId::from_raw(order[i]);
                let b = DenseNodeId::from_raw(order[j]);
                let _ = g.insert_directed(a, b);
            }
        }
    }
    g
}

#[test]
fn property_matches_path_oracle_on_tiny_dags() {
    let mut rng = CausalRng::from_seed(42);
    let mut ws = DSeparationWorkspace::default();
    for _ in 0..40 {
        let n = 4u32;
        let g = random_dag(&mut rng, n);
        for x in 0..n {
            for y in 0..n {
                if x == y {
                    continue;
                }
                for mask in 0..(1u32 << n) {
                    if (mask & (1 << x)) != 0 || (mask & (1 << y)) != 0 {
                        continue;
                    }
                    let z: Vec<DenseNodeId> = (0..n)
                        .filter(|i| (mask & (1 << i)) != 0)
                        .map(DenseNodeId::from_raw)
                        .collect();
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

#[test]
fn mutilation_removes_incoming() {
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let m = g.mutilate(&[DenseNodeId::from_raw(1)]).unwrap();
    assert!(m.children(DenseNodeId::from_raw(0)).is_empty());
    assert_eq!(m.children(DenseNodeId::from_raw(1)).len(), 1);
}
