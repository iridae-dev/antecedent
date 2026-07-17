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
use crate::overlay::GraphOverlay;
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
        /// Certificate .
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
        self.is_d_separated_with(x, y, z, ws, None)
    }

    /// d-separation under an optional [`GraphOverlay`].
    pub(crate) fn is_d_separated_with(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
        overlay: Option<&GraphOverlay>,
    ) -> Result<bool, GraphError> {
        self.validate_node_pub(x)?;
        self.validate_node_pub(y)?;
        for &v in z {
            self.validate_node_pub(v)?;
        }
        if x == y {
            return Ok(false);
        }
        Ok(self.d_sep_active_path(x, y, z, ws, overlay).is_none())
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
        self.d_separation_with(x, y, z, ws, None)
    }

    /// Witness d-separation under an optional [`GraphOverlay`].
    pub(crate) fn d_separation_with(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
        overlay: Option<&GraphOverlay>,
    ) -> Result<SeparationResult, GraphError> {
        self.validate_node_pub(x)?;
        self.validate_node_pub(y)?;
        for &v in z {
            self.validate_node_pub(v)?;
        }
        if x == y {
            return Ok(SeparationResult::Connected { active_path: vec![PathStep { node: x }] });
        }
        if let Some(path) = self.d_sep_active_path(x, y, z, ws, overlay) {
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

    /// Returns an active undirected path if d-connected; `None` if separated.
    fn d_sep_active_path(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
        overlay: Option<&GraphOverlay>,
    ) -> Option<Vec<DenseNodeId>> {
        let n = self.node_count();
        ws.prepare(n);

        // Ancestral set of {x,y} ∪ z
        let mut seeds = Vec::with_capacity(2 + z.len());
        seeds.push(x);
        seeds.push(y);
        seeds.extend_from_slice(z);
        self.ancestors_of_with(&seeds, &mut ws.ancestral, &mut ws.graph_ws, overlay);

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
                if overlay.is_some_and(|ov| !ov.edge_visible(u, c)) {
                    continue;
                }
                if ws.ancestral.contains(c) {
                    add_undirected(&mut ws.undirected, u, c);
                }
            }
            // Moral edges: marry parents connected by visible edges into u.
            let parents = self.parents(u);
            for (a_idx, &a) in parents.iter().enumerate() {
                if overlay.is_some_and(|ov| !ov.edge_visible(a, u)) {
                    continue;
                }
                if !ws.ancestral.contains(a) {
                    continue;
                }
                for &b in &parents[a_idx + 1..] {
                    if overlay.is_some_and(|ov| !ov.edge_visible(b, u)) {
                        continue;
                    }
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
#[path = "dsep_tests.rs"]
mod tests;
