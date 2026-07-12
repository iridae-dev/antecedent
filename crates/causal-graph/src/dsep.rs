//! d-separation for DAGs (DESIGN.md §6.5).
//!
//! Boolean batch path allocates no path objects. Witness mode returns an active
//! path certificate when nodes are d-connected given the conditioning set.
//!
//! Algorithm: ancestral subgraph → moralize → remove conditioning → undirected
//! reachability (Lauritzen et al. / Pearl).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)] // x, y, z are standard d-separation names

use crate::dag::Dag;
use crate::error::GraphError;
use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

/// Step on an undirected active path (witness mode).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PathStep {
    /// Node at this step.
    pub node: DenseNodeId,
}

/// Certificate that a conditioning set d-separates two nodes (boolean path).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SeparationCertificate {
    /// Conditioning set used.
    pub conditioning: Vec<DenseNodeId>,
}

/// Result of a d-separation query with optional witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SeparationResult {
    /// d-separated given `conditioning`.
    Separated {
        /// Conditioning set.
        conditioning: Vec<DenseNodeId>,
        /// Certificate (Phase 1: echoes conditioning).
        certificate: SeparationCertificate,
    },
    /// d-connected; `active_path` is an undirected path in the moral graph
    /// after removing the conditioning set.
    Connected {
        /// Active path nodes from x to y.
        active_path: Vec<PathStep>,
    },
}

/// Scratch buffers for repeated d-separation queries.
#[derive(Clone, Debug, Default)]
pub struct DSeparationWorkspace {
    /// Ancestral closure.
    pub ancestral: BitSet,
    /// Conditioning set membership.
    pub conditioning: BitSet,
    /// Undirected adjacency for the moral graph (only ancestral nodes).
    pub undirected: Vec<Vec<DenseNodeId>>,
    /// BFS visited.
    pub visited: BitSet,
    /// BFS frontier / predecessor scratch.
    pub frontier: Vec<DenseNodeId>,
    /// Predecessor for path reconstruction.
    pub pred: Vec<Option<DenseNodeId>>,
    /// Graph traversal workspace.
    pub graph_ws: GraphWorkspace,
}

impl DSeparationWorkspace {
    /// Prepare for a DAG with `n` nodes.
    pub fn prepare(&mut self, n: usize) {
        self.ancestral.resize(n);
        self.conditioning.resize(n);
        self.visited.resize(n);
        self.undirected.resize(n, Vec::new());
        for adj in &mut self.undirected {
            adj.clear();
        }
        self.frontier.clear();
        self.pred.clear();
        self.pred.resize(n, None);
        self.graph_ws.prepare(n);
    }
}

impl Dag {
    /// Whether `x` is d-separated from `y` given `z` (boolean; no path alloc).
    ///
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn is_d_separated(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Result<bool, GraphError> {
        self.validate_node_pub(x)?;
        self.validate_node_pub(y)?;
        for &v in z {
            self.validate_node_pub(v)?;
        }
        if x == y {
            return Ok(false);
        }
        Ok(self.d_sep_bool(x, y, z, ws))
    }

    /// Batch boolean d-separation. `out[i]` corresponds to `queries[i] = (x,y,z)`.
    ///
    /// # Errors
    ///
    /// Unknown nodes; or `out.len() != queries.len()`.
    pub fn is_d_separated_batch(
        &self,
        queries: &[(DenseNodeId, DenseNodeId, &[DenseNodeId])],
        out: &mut [bool],
        ws: &mut DSeparationWorkspace,
    ) -> Result<(), GraphError> {
        if out.len() != queries.len() {
            return Err(GraphError::InvalidEndpoints { message: "batch output length mismatch" });
        }
        for (i, &(x, y, z)) in queries.iter().enumerate() {
            out[i] = self.is_d_separated(x, y, z, ws)?;
        }
        Ok(())
    }

    /// d-separation with witness (active path or separation certificate).
    ///
    /// # Errors
    ///
    /// Unknown nodes.
    pub fn d_separation(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Result<SeparationResult, GraphError> {
        self.validate_node_pub(x)?;
        self.validate_node_pub(y)?;
        for &v in z {
            self.validate_node_pub(v)?;
        }
        if x == y {
            return Ok(SeparationResult::Connected { active_path: vec![PathStep { node: x }] });
        }
        if let Some(path) = self.d_sep_active_path(x, y, z, ws) {
            Ok(SeparationResult::Connected {
                active_path: path.into_iter().map(|node| PathStep { node }).collect(),
            })
        } else {
            Ok(SeparationResult::Separated {
                conditioning: z.to_vec(),
                certificate: SeparationCertificate { conditioning: z.to_vec() },
            })
        }
    }

    fn d_sep_bool(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> bool {
        self.d_sep_active_path(x, y, z, ws).is_none()
    }

    /// Returns an active undirected path if d-connected; `None` if separated.
    fn d_sep_active_path(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Option<Vec<DenseNodeId>> {
        let n = self.node_count();
        ws.prepare(n);

        // Ancestral set of {x,y} ∪ z
        let mut seeds = Vec::with_capacity(2 + z.len());
        seeds.push(x);
        seeds.push(y);
        seeds.extend_from_slice(z);
        self.ancestors_of(&seeds, &mut ws.ancestral, &mut ws.graph_ws);

        ws.conditioning.clear();
        for &v in z {
            ws.conditioning.insert(v);
        }

        // Build moral undirected graph on ancestral nodes.
        for i in 0..n {
            let u = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            if !ws.ancestral.contains(u) {
                continue;
            }
            // Directed edges become undirected (within ancestral set).
            for &c in self.children(u) {
                if ws.ancestral.contains(c) {
                    add_undirected(&mut ws.undirected, u, c);
                }
            }
            // Moral edges: marry parents.
            let parents = self.parents(u);
            for (a_idx, &a) in parents.iter().enumerate() {
                if !ws.ancestral.contains(a) {
                    continue;
                }
                for &b in &parents[a_idx + 1..] {
                    if ws.ancestral.contains(b) {
                        add_undirected(&mut ws.undirected, a, b);
                    }
                }
            }
        }

        // BFS from x to y avoiding conditioning set.
        ws.visited.clear();
        for p in &mut ws.pred {
            *p = None;
        }
        if ws.conditioning.contains(x) || ws.conditioning.contains(y) {
            // If x or y is in Z, they are not d-connected as open endpoints
            // for the classical X⊥Y|Z query (conditioning includes the node).
            // Treat as separated when either endpoint is conditioned.
            return None;
        }
        ws.frontier.clear();
        ws.frontier.push(x);
        ws.visited.insert(x);
        while let Some(u) = ws.frontier.pop() {
            if u == y {
                return Some(reconstruct_path(&ws.pred, x, y));
            }
            for &v in &ws.undirected[u.as_usize()] {
                if ws.conditioning.contains(v) || ws.visited.contains(v) {
                    continue;
                }
                if !ws.ancestral.contains(v) {
                    continue;
                }
                ws.visited.insert(v);
                ws.pred[v.as_usize()] = Some(u);
                ws.frontier.push(v);
            }
        }
        None
    }
}

fn add_undirected(adj: &mut [Vec<DenseNodeId>], a: DenseNodeId, b: DenseNodeId) {
    if a == b {
        return;
    }
    let ai = a.as_usize();
    let bi = b.as_usize();
    if !adj[ai].contains(&b) {
        adj[ai].push(b);
    }
    if !adj[bi].contains(&a) {
        adj[bi].push(a);
    }
}

fn reconstruct_path(
    pred: &[Option<DenseNodeId>],
    start: DenseNodeId,
    end: DenseNodeId,
) -> Vec<DenseNodeId> {
    let mut path = vec![end];
    let mut cur = end;
    while cur != start {
        cur = pred[cur.as_usize()].expect("path predecessor");
        path.push(cur);
    }
    path.reverse();
    path
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
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
        let res = g
            .d_separation(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2), &[], &mut ws)
            .unwrap();
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
                Some((_prev_node, kind_in)) => {
                    is_triple_active(kind_in, kind_leaving_cur, cur, z, g)
                }
            };
            // Edge kind relative to `next` is the reverse of the leave label at `cur`.
            let kind_at_next = match kind_leaving_cur {
                EdgeKind::ToChild => EdgeKind::FromParent,
                EdgeKind::FromParent => EdgeKind::ToChild,
            };
            if active
                && exists_active_path_dfs(g, next, target, z, visited, Some((cur, kind_at_next)))
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
            // chain: -> cur ->  or  <- cur <-
            (EdgeKind::FromParent, EdgeKind::ToChild)
            | (EdgeKind::ToChild, EdgeKind::FromParent) => !in_z,
            // fork: <- cur ->
            (EdgeKind::ToChild, EdgeKind::ToChild) => !in_z,
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
}
