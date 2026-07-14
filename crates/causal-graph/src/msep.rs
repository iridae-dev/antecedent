//! m-separation for ADMGs (DESIGN.md §6.5).
//!
//! Ancestral subgraph → moralize (including bidirected as undirected) →
//! remove conditioning → undirected reachability.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use crate::admg::Admg;
use crate::dsep::{
    DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult,
};
use crate::error::GraphError;
use crate::types::DenseNodeId;

impl Admg {
    /// Whether `x` is m-separated from `y` given `z` (boolean; no path alloc).
    ///
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn is_m_separated(
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
        Ok(self.m_sep_bool(x, y, z, ws))
    }

    /// Batch boolean m-separation.
    ///
    /// # Errors
    ///
    /// Unknown nodes or length mismatch.
    pub fn is_m_separated_batch(
        &self,
        queries: &[(DenseNodeId, DenseNodeId, &[DenseNodeId])],
        out: &mut [bool],
        ws: &mut DSeparationWorkspace,
    ) -> Result<(), GraphError> {
        if out.len() != queries.len() {
            return Err(GraphError::InvalidEndpoints {
                message: "batch output length mismatch",
            });
        }
        for (i, &(x, y, z)) in queries.iter().enumerate() {
            out[i] = self.is_m_separated(x, y, z, ws)?;
        }
        Ok(())
    }

    /// m-separation with witness.
    ///
    /// # Errors
    ///
    /// Unknown nodes.
    pub fn m_separation(
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
            return Ok(SeparationResult::Connected {
                active_path: vec![PathStep { node: x }],
            });
        }
        if let Some(path) = self.m_sep_active_path(x, y, z, ws) {
            Ok(SeparationResult::Connected { active_path: path })
        } else {
            Ok(SeparationResult::Separated {
                conditioning: z.to_vec(),
                certificate: SeparationCertificate { conditioning: z.to_vec() },
            })
        }
    }

    fn m_sep_bool(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> bool {
        self.build_moral_ancestral(x, y, z, ws);
        // Remove conditioning nodes from undirected graph by skipping them in BFS.
        !self.undirected_reaches(x, y, ws, false)
    }

    fn m_sep_active_path(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Option<Vec<PathStep>> {
        self.build_moral_ancestral(x, y, z, ws);
        if !self.undirected_reaches(x, y, ws, true) {
            return None;
        }
        let mut path = Vec::new();
        let mut cur = Some(y);
        while let Some(n) = cur {
            path.push(PathStep { node: n });
            if n == x {
                break;
            }
            cur = ws.pred[n.as_usize()];
        }
        path.reverse();
        Some(path)
    }

    /// Ancestral closure of {x,y}∪z, then moralize directed + bidirected edges.
    fn build_moral_ancestral(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) {
        let n = self.node_count();
        ws.prepare(n);
        ws.ancestral.clear();
        ws.conditioning.clear();
        for &v in z {
            ws.conditioning.insert(v);
        }

        // Seeds: x, y, z
        ws.graph_ws.prepare(n);
        ws.graph_ws.frontier.clear();
        for &s in &[x, y] {
            if !ws.ancestral.contains(s) {
                ws.ancestral.insert(s);
                ws.graph_ws.frontier.push(s);
            }
        }
        for &s in z {
            if !ws.ancestral.contains(s) {
                ws.ancestral.insert(s);
                ws.graph_ws.frontier.push(s);
            }
        }
        // Walk parents (directed ancestors).
        while let Some(u) = ws.graph_ws.frontier.pop() {
            for &p in self.parents(u) {
                if !ws.ancestral.contains(p) {
                    ws.ancestral.insert(p);
                    ws.graph_ws.frontier.push(p);
                }
            }
        }

        // Moralize: undirected edges for directed + bidirected within ancestral set;
        // marry co-parents of each node.
        for i in 0..n {
            let u = DenseNodeId::from_raw(i as u32);
            if !ws.ancestral.contains(u) {
                continue;
            }
            for &c in self.children(u) {
                if ws.ancestral.contains(c) {
                    Self::add_undirected(ws, u, c);
                }
            }
            for &b in self.bidirected_neighbors(u) {
                if b.raw() > u.raw() && ws.ancestral.contains(b) {
                    Self::add_undirected(ws, u, b);
                }
            }
            let parents = self.parents(u);
            for (ai, &a) in parents.iter().enumerate() {
                if !ws.ancestral.contains(a) {
                    continue;
                }
                for &b in &parents[ai + 1..] {
                    if ws.ancestral.contains(b) {
                        Self::add_undirected(ws, a, b);
                    }
                }
            }
        }
    }

    fn add_undirected(ws: &mut DSeparationWorkspace, a: DenseNodeId, b: DenseNodeId) {
        if a == b {
            return;
        }
        let ai = a.as_usize();
        let bi = b.as_usize();
        if !ws.undirected[ai].contains(&b) {
            ws.undirected[ai].push(b);
        }
        if !ws.undirected[bi].contains(&a) {
            ws.undirected[bi].push(a);
        }
    }

    /// Undirected reachability avoiding conditioning nodes.
    fn undirected_reaches(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        ws: &mut DSeparationWorkspace,
        record_pred: bool,
    ) -> bool {
        let _ = self;
        ws.visited.clear();
        ws.frontier.clear();
        if record_pred {
            for p in &mut ws.pred {
                *p = None;
            }
        }
        if ws.conditioning.contains(x) || ws.conditioning.contains(y) {
            // If either endpoint is conditioned, they are separated unless x==y (handled earlier).
            return false;
        }
        ws.frontier.push(x);
        ws.visited.insert(x);
        while let Some(u) = ws.frontier.pop() {
            if u == y {
                return true;
            }
            for &v in &ws.undirected[u.as_usize()] {
                if ws.conditioning.contains(v) || ws.visited.contains(v) {
                    continue;
                }
                ws.visited.insert(v);
                if record_pred {
                    ws.pred[v.as_usize()] = Some(u);
                }
                ws.frontier.push(v);
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admg::Admg;

    #[test]
    fn bidirected_connects_without_conditioning() {
        let mut g = Admg::with_variables(2);
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
            .unwrap();
        let mut ws = DSeparationWorkspace::default();
        assert!(!g
            .is_m_separated(
                DenseNodeId::from_raw(0),
                DenseNodeId::from_raw(1),
                &[],
                &mut ws
            )
            .unwrap());
    }

    #[test]
    fn fork_separated_by_common_cause() {
        // X <- U -> Y with U latent as bidirected X↔Y and no directed edges: not separated.
        // Classic: X <- Z -> Y: Z separates X and Y.
        let mut g = Admg::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let z = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(z, x).unwrap();
        g.insert_directed(z, y).unwrap();
        let mut ws = DSeparationWorkspace::default();
        assert!(!g.is_m_separated(x, y, &[], &mut ws).unwrap());
        assert!(g.is_m_separated(x, y, &[z], &mut ws).unwrap());
    }

    #[test]
    fn collider_opens_with_conditioning() {
        // X -> Z <- Y
        let mut g = Admg::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let z = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(x, z).unwrap();
        g.insert_directed(y, z).unwrap();
        let mut ws = DSeparationWorkspace::default();
        assert!(g.is_m_separated(x, y, &[], &mut ws).unwrap());
        assert!(!g.is_m_separated(x, y, &[z], &mut ws).unwrap());
    }
}
