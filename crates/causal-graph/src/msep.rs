//! m-separation for ADMGs and definite-status m-separation for PAGs (DESIGN.md §6.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use crate::admg::Admg;
use crate::dsep::{DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult};
use crate::error::GraphError;
use crate::pag::Pag;
use crate::types::DenseNodeId;

// --- ADMG m-separation (ancestral moralization) ---

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
            return Err(GraphError::InvalidEndpoints { message: "batch output length mismatch" });
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
            return Ok(SeparationResult::Connected { active_path: vec![PathStep { node: x }] });
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
        // then clique each bidirected district C with pa(C) (Richardson augmentation).
        for i in 0..n {
            let u = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
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
        }

        // Bidirected-connected districts in the ancestral subgraph; clique C ∪ pa(C).
        let mut district = vec![u32::MAX; n];
        let mut n_districts = 0u32;
        let mut stack = Vec::new();
        for i in 0..n {
            let u = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            if !ws.ancestral.contains(u) || district[i] != u32::MAX {
                continue;
            }
            district[i] = n_districts;
            stack.push(u);
            while let Some(v) = stack.pop() {
                for &w in self.bidirected_neighbors(v) {
                    let wi = w.as_usize();
                    if ws.ancestral.contains(w) && district[wi] == u32::MAX {
                        district[wi] = n_districts;
                        stack.push(w);
                    }
                }
            }
            n_districts += 1;
        }
        for d in 0..n_districts {
            let mut clique = Vec::new();
            for i in 0..n {
                if district[i] != d {
                    continue;
                }
                let u = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
                if !clique.iter().any(|&x| x == u) {
                    clique.push(u);
                }
                for &p in self.parents(u) {
                    if ws.ancestral.contains(p) && !clique.iter().any(|&x| x == p) {
                        clique.push(p);
                    }
                }
            }
            for (ai, &a) in clique.iter().enumerate() {
                for &b in &clique[ai + 1..] {
                    Self::add_undirected(ws, a, b);
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
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let mut ws = DSeparationWorkspace::default();
        assert!(
            !g.is_m_separated(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), &[], &mut ws)
                .unwrap()
        );
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

    #[test]
    fn district_clique_opens_collider_connected_chain() {
        // X → A ↔ B ← Y; conditioning on {A,B} opens the collider-connected path.
        // Without district augmentation C={A,B} ∪ pa(C)={X,Y}, X and Y look separated.
        let mut g = Admg::with_variables(4);
        let x = DenseNodeId::from_raw(0);
        let a = DenseNodeId::from_raw(1);
        let b = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        g.insert_directed(x, a).unwrap();
        g.insert_bidirected(a, b).unwrap();
        g.insert_directed(y, b).unwrap();
        let mut ws = DSeparationWorkspace::default();
        assert!(
            !g.is_m_separated(x, y, &[a, b], &mut ws).unwrap(),
            "X and Y must be m-connected given {{A,B}} via district clique"
        );
        // Without conditioning the colliders are closed → separated.
        assert!(g.is_m_separated(x, y, &[], &mut ws).unwrap());
    }
}

// --- PAG definite-status m-separation ---

impl Pag {
    /// Whether `x` is m-separated from `y` given `z` via definite-status paths.
    ///
    /// Separated iff no definite-status path from x to y is active given z.
    /// If the bounded search is truncated and no active path was found, returns
    /// [`GraphError::SearchBudgetExhausted`] rather than claiming separation.
    ///
    /// # Errors
    ///
    /// Unknown nodes, or incomplete search under budget when no active path is known.
    pub fn is_m_separated(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<bool, GraphError> {
        self.validate_node_pub(x)?;
        self.validate_node_pub(y)?;
        for &v in z {
            self.validate_node_pub(v)?;
        }
        if x == y {
            return Ok(false);
        }
        let search = self.definite_status_paths(x, y, max_paths, max_len)?;
        if search.paths.iter().any(|p| self.path_active_given(&p.nodes, z)) {
            return Ok(false);
        }
        if search.truncated {
            return Err(GraphError::SearchBudgetExhausted { max_paths, max_len });
        }
        Ok(true)
    }

    /// m-separation with witness (active definite-status path or certificate).
    ///
    /// # Errors
    ///
    /// Unknown nodes, or incomplete search under budget when no active path is known.
    pub fn m_separation(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<SeparationResult, GraphError> {
        self.validate_node_pub(x)?;
        self.validate_node_pub(y)?;
        for &v in z {
            self.validate_node_pub(v)?;
        }
        if x == y {
            return Ok(SeparationResult::Connected { active_path: vec![PathStep { node: x }] });
        }
        let search = self.definite_status_paths(x, y, max_paths, max_len)?;
        if let Some(p) = search.paths.iter().find(|p| self.path_active_given(&p.nodes, z)) {
            Ok(SeparationResult::Connected {
                active_path: p.nodes.iter().map(|&node| PathStep { node }).collect(),
            })
        } else if search.truncated {
            Err(GraphError::SearchBudgetExhausted { max_paths, max_len })
        } else {
            Ok(SeparationResult::Separated {
                conditioning: z.to_vec(),
                certificate: SeparationCertificate { conditioning: z.to_vec() },
            })
        }
    }

    /// Batch boolean PAG m-separation.
    ///
    /// # Errors
    ///
    /// Length mismatch or unknown nodes.
    pub fn is_m_separated_batch(
        &self,
        queries: &[(DenseNodeId, DenseNodeId, &[DenseNodeId])],
        out: &mut [bool],
        max_paths: usize,
        max_len: usize,
    ) -> Result<(), GraphError> {
        if out.len() != queries.len() {
            return Err(GraphError::InvalidEndpoints { message: "batch output length mismatch" });
        }
        for (i, &(x, y, z)) in queries.iter().enumerate() {
            out[i] = self.is_m_separated(x, y, z, max_paths, max_len)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod pag_msep_tests {
    use super::*;
    use crate::pag::Pag;

    #[test]
    fn directed_chain_msep() {
        let mut g = Pag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        assert!(!g.is_m_separated(a, c, &[], 32, 8).unwrap());
        assert!(g.is_m_separated(a, c, &[b], 32, 8).unwrap());
    }

    #[test]
    fn collider_opens_via_descendant() {
        // X → C ← Y, C → D; Z = {D} opens the collider at C.
        let mut g = Pag::with_variables(4);
        let x = DenseNodeId::from_raw(0);
        let c = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let d = DenseNodeId::from_raw(3);
        g.insert_directed(x, c).unwrap();
        g.insert_directed(y, c).unwrap();
        g.insert_directed(c, d).unwrap();
        assert!(g.is_m_separated(x, y, &[], 32, 8).unwrap());
        assert!(!g.is_m_separated(x, y, &[c], 32, 8).unwrap());
        assert!(
            !g.is_m_separated(x, y, &[d], 32, 8).unwrap(),
            "descendant D in Z must open collider C"
        );
    }

    #[test]
    fn budget_exhaustion_is_error_not_separated() {
        // Long directed chain; max_len too small to reach the other end.
        let mut g = Pag::with_variables(5);
        for i in 0..4 {
            g.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
        }
        let x = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(4);
        // Complete search finds the path.
        assert!(!g.is_m_separated(x, y, &[], 32, 8).unwrap());
        // Truncated search must not claim separation.
        let err = g.is_m_separated(x, y, &[], 32, 2).unwrap_err();
        assert!(matches!(err, GraphError::SearchBudgetExhausted { max_len: 2, .. }));
    }
}
