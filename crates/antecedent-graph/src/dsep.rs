//! d-separation for DAGs.
//!
//! Boolean batch path allocates no path objects. Witness mode returns an active
//! path certificate when nodes are d-connected given the conditioning set.
//!
//! Boolean queries use Bayes-ball (Shachter 1998), extended across bidirected
//! edges as latent forks. Witness paths still come from the ancestral moral
//! graph (Lauritzen et al. 1990 / Pearl 1988), whose neighbor order the
//! certificates depend on.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names, reason = "x, y, z are standard d-separation names")]

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
    ///
    /// Neighbor order is insertion order. [`Self::undirected_member`] is the same
    /// relation as a bitset so duplicate edges are rejected without scanning the list.
    pub undirected: Vec<Vec<DenseNodeId>>,
    /// Membership of [`Self::undirected`], one bitset per node. Kept in lockstep
    /// by [`Self::prepare`] and [`add_undirected`].
    pub(crate) undirected_member: Vec<BitSet>,
    /// BFS visited.
    pub visited: BitSet,
    /// BFS frontier / predecessor scratch.
    pub frontier: Vec<DenseNodeId>,
    /// Predecessor for path reconstruction.
    pub pred: Vec<Option<DenseNodeId>>,
    /// Graph traversal workspace.
    pub graph_ws: GraphWorkspace,
    /// District label per node (ADMG moralization scratch).
    pub(crate) district_label: Vec<u32>,
    /// District search stack (ADMG moralization scratch).
    pub(crate) district_stack: Vec<DenseNodeId>,
    /// Nodes of each district, ascending (ADMG moralization scratch).
    pub(crate) district_groups: Vec<Vec<DenseNodeId>>,
    /// Clique `C ∪ pa(C)` under construction (ADMG moralization scratch).
    pub(crate) clique: Vec<DenseNodeId>,
    /// Membership marks for [`Self::clique`].
    pub(crate) clique_mark: BitSet,
    /// Bayes-ball nodes already entered from a child.
    pub(crate) bayes_up: BitSet,
    /// Bayes-ball nodes already entered from a parent or a bidirected neighbor.
    pub(crate) bayes_down: BitSet,
    /// Bayes-ball targets for the query in progress.
    pub(crate) bayes_targets: BitSet,
    /// Bayes-ball stack. `true` means the ball arrived from a child.
    pub(crate) bayes_queue: Vec<(DenseNodeId, bool)>,
}

impl DSeparationWorkspace {
    /// Reachability from `x` to `y` over `self.undirected`, avoiding `self.conditioning`.
    ///
    /// The moral graph holds only ancestral nodes, so no membership test is needed.
    /// `record_pred` fills `self.pred` for witness-path reconstruction; the frontier is a
    /// stack, so the first witness found depends on adjacency insertion order.
    pub(crate) fn moral_reaches(
        &mut self,
        x: DenseNodeId,
        y: DenseNodeId,
        record_pred: bool,
    ) -> bool {
        self.visited.clear();
        if record_pred {
            for p in &mut self.pred {
                *p = None;
            }
        }
        self.frontier.clear();
        self.frontier.push(x);
        self.visited.insert(x);
        while let Some(u) = self.frontier.pop() {
            if u == y {
                return true;
            }
            for &v in &self.undirected[u.as_usize()] {
                if self.conditioning.contains(v) || self.visited.contains(v) {
                    continue;
                }
                self.visited.insert(v);
                if record_pred {
                    self.pred[v.as_usize()] = Some(u);
                }
                self.frontier.push(v);
            }
        }
        false
    }

    /// Whether any target is reachable from any source over [`Self::undirected`],
    /// avoiding [`Self::conditioning`].
    ///
    /// The moral graph holds only ancestral nodes. Sources and targets that sit in
    /// the conditioning set are not expanded; the caller decides whether that makes
    /// the query ill-posed.
    #[cfg(test)]
    pub(crate) fn moral_reaches_any(&mut self, sources: &[DenseNodeId], targets: &BitSet) -> bool {
        self.visited.clear();
        self.frontier.clear();
        for &source in sources {
            if self.conditioning.contains(source) || self.visited.contains(source) {
                continue;
            }
            if targets.contains(source) {
                return true;
            }
            self.visited.insert(source);
            self.frontier.push(source);
        }
        while let Some(u) = self.frontier.pop() {
            for &v in &self.undirected[u.as_usize()] {
                if self.conditioning.contains(v) || self.visited.contains(v) {
                    continue;
                }
                if targets.contains(v) {
                    return true;
                }
                self.visited.insert(v);
                self.frontier.push(v);
            }
        }
        false
    }

    /// Prepare for a DAG with `n` nodes.
    pub fn prepare(&mut self, n: usize) {
        self.ancestral.resize(n);
        self.conditioning.resize(n);
        self.visited.resize(n);
        self.undirected.resize(n, Vec::new());
        for adj in &mut self.undirected {
            adj.clear();
        }
        if self.undirected_member.len() < n {
            self.undirected_member.resize_with(n, || BitSet::with_len(n));
        } else {
            self.undirected_member.truncate(n);
        }
        for member in &mut self.undirected_member {
            member.resize(n);
            member.clear();
        }
        self.frontier.clear();
        self.pred.clear();
        self.pred.resize(n, None);
        self.graph_ws.prepare(n);
        self.clique_mark.resize(n);
        self.clique_mark.clear();
        self.bayes_up.resize(n);
        self.bayes_up.clear();
        self.bayes_down.resize(n);
        self.bayes_down.clear();
        self.bayes_targets.resize(n);
        self.bayes_targets.clear();
        self.bayes_queue.clear();
    }
}

/// Whether Bayes-ball reaches any `targets` from `sources` given `conditioning`.
///
/// Directed edges honor `overlay`. A bidirected neighbor is the other child of
/// an unobserved latent parent: the ball crosses it exactly when it would
/// visit a directed parent, and it arrives traveling down.
pub(crate) fn bayes_ball_reaches(
    ws: &mut DSeparationWorkspace,
    parents: &[Vec<DenseNodeId>],
    children: &[Vec<DenseNodeId>],
    bidirected: &[Vec<DenseNodeId>],
    overlay: Option<&crate::overlay::GraphOverlay>,
    sources: &[DenseNodeId],
    targets: &[DenseNodeId],
    conditioning: &[DenseNodeId],
) -> bool {
    let n = parents.len();
    ws.prepare(n);
    ws.conditioning.clear();
    for &node in conditioning {
        if node.as_usize() < n {
            ws.conditioning.insert(node);
        }
    }
    crate::algo::closure_into(
        n,
        |node| parents.get(node.as_usize()).map_or(&[][..], Vec::as_slice),
        conditioning.iter().copied(),
        &mut ws.ancestral,
        &mut ws.graph_ws,
        |child, parent| overlay.is_none_or(|ov| ov.edge_visible(parent, child)),
    );
    for &target in targets {
        if target.as_usize() < n {
            ws.bayes_targets.insert(target);
        }
    }
    let directed_ok = |parent: DenseNodeId, child: DenseNodeId| {
        overlay.is_none_or(|ov| ov.edge_visible(parent, child))
    };
    for &source in sources {
        if source.as_usize() >= n || ws.bayes_up.contains(source) {
            continue;
        }
        ws.bayes_up.insert(source);
        ws.bayes_queue.push((source, true));
    }
    while let Some((node, from_child)) = ws.bayes_queue.pop() {
        if ws.bayes_targets.contains(node) {
            return true;
        }
        let parents_of = parents.get(node.as_usize()).map_or(&[][..], Vec::as_slice);
        let children_of = children.get(node.as_usize()).map_or(&[][..], Vec::as_slice);
        let bidirected_of = bidirected.get(node.as_usize()).map_or(&[][..], Vec::as_slice);
        if from_child {
            if ws.conditioning.contains(node) {
                continue;
            }
            for &parent in parents_of {
                if directed_ok(parent, node) {
                    schedule(ws, parent, true);
                }
            }
            for &child in children_of {
                if directed_ok(node, child) {
                    schedule(ws, child, false);
                }
            }
            for &other in bidirected_of {
                schedule(ws, other, false);
            }
        } else {
            if !ws.conditioning.contains(node) {
                for &child in children_of {
                    if directed_ok(node, child) {
                        schedule(ws, child, false);
                    }
                }
            }
            if ws.ancestral.contains(node) {
                for &parent in parents_of {
                    if directed_ok(parent, node) {
                        schedule(ws, parent, true);
                    }
                }
                for &other in bidirected_of {
                    schedule(ws, other, false);
                }
            }
        }
    }
    false
}

fn schedule(ws: &mut DSeparationWorkspace, node: DenseNodeId, from_child: bool) {
    let seen = if from_child { ws.bayes_up.contains(node) } else { ws.bayes_down.contains(node) };
    if seen {
        return;
    }
    if from_child {
        ws.bayes_up.insert(node);
    } else {
        ws.bayes_down.insert(node);
    }
    ws.bayes_queue.push((node, from_child));
}

impl Dag {
    /// Whether `x` is d-separated from `y` given `z` (boolean; no path alloc).
    ///
    /// If `Z ∩ {x,y} ≠ ∅` the query is ill-posed and this returns `false`
    /// (not separated), matching PAG m-separation / definite-status activity.
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
        if z.iter().any(|&v| v == x || v == y) {
            return Ok(false);
        }
        Ok(!bayes_ball_reaches(
            ws,
            self.parent_rows(),
            self.child_rows(),
            &[],
            overlay,
            &[x],
            &[y],
            z,
        ))
    }

    /// d-separation with witness (active path or separation certificate).
    ///
    /// # Errors
    ///
    /// Unknown nodes, or a conditioning set containing an endpoint (for which
    /// no active-path witness can be issued).
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
        if z.iter().any(|&v| v == x || v == y) {
            return Err(GraphError::InvalidEndpoints {
                message: "d-separation witness requires conditioning disjoint from endpoints",
            });
        }
        if x == y {
            return Ok(SeparationResult::Connected { active_path: vec![PathStep { node: x }] });
        }
        if self.d_sep_connected(x, y, z, ws, overlay, true) {
            Ok(SeparationResult::Connected {
                active_path: reconstruct_path(&ws.pred, x, y)
                    .into_iter()
                    .map(|node| PathStep { node })
                    .collect(),
            })
        } else {
            Ok(SeparationResult::Separated {
                conditioning: z.to_vec(),
                certificate: SeparationCertificate { conditioning: z.to_vec() },
            })
        }
    }

    /// Reachability in the moral ancestral graph; only witness callers record predecessors.
    fn d_sep_connected(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
        overlay: Option<&GraphOverlay>,
        record_path: bool,
    ) -> bool {
        let n = self.node_count();
        ws.prepare(n);

        // Ancestral set of {x,y} ∪ z
        ws.frontier.push(x);
        ws.frontier.push(y);
        ws.frontier.extend_from_slice(z);
        self.ancestors_of_with(&ws.frontier, &mut ws.ancestral, &mut ws.graph_ws, overlay);

        ws.conditioning.clear();
        for &v in z {
            ws.conditioning.insert(v);
        }

        // Build moral undirected graph on ancestral nodes.
        for i in 0..n {
            let u = DenseNodeId::from_index(i);
            if !ws.ancestral.contains(u) {
                continue;
            }
            // Directed edges become undirected (within ancestral set).
            for &c in self.children(u) {
                if overlay.is_some_and(|ov| !ov.edge_visible(u, c)) {
                    continue;
                }
                if ws.ancestral.contains(c) {
                    add_undirected(&mut ws.undirected, &mut ws.undirected_member, u, c);
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
                        add_undirected(&mut ws.undirected, &mut ws.undirected_member, a, b);
                    }
                }
            }
        }

        // BFS from x to y avoiding conditioning set.
        ws.moral_reaches(x, y, record_path)
    }
}

/// Insert the undirected edge `a - b` into `adj` once, preserving insertion order.
///
/// `member[i]` is the set of neighbors already stored in `adj[i]`. The bitset
/// replaces a linear scan of the adjacency list; the list itself stays the
/// iteration order witnesses depend on.
pub(crate) fn add_undirected(
    adj: &mut [Vec<DenseNodeId>],
    member: &mut [BitSet],
    a: DenseNodeId,
    b: DenseNodeId,
) {
    if a == b {
        return;
    }
    let ai = a.as_usize();
    let bi = b.as_usize();
    if !member[ai].contains(b) {
        member[ai].insert(b);
        adj[ai].push(b);
    }
    if !member[bi].contains(a) {
        member[bi].insert(a);
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
