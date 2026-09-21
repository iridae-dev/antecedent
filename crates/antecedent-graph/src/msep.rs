//! m-separation for ADMGs, and class-wide m-separation for PAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use crate::admg::Admg;
use crate::dsep::{DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult};
use crate::error::GraphError;
use crate::pag::Pag;
use crate::types::DenseNodeId;
use crate::workspace::BitSet;

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
        if z.iter().any(|&v| v == x || v == y) {
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
        if z.iter().any(|&v| v == x || v == y) {
            return Ok(SeparationResult::Connected {
                active_path: vec![PathStep { node: x }, PathStep { node: y }],
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

        self.moralize_ancestral_set(ws);
    }

    /// Moralize the full ADMG (ancestral set = all nodes) for Markov blankets.
    fn build_moral_full(&self, ws: &mut DSeparationWorkspace) {
        let n = self.node_count();
        ws.prepare(n);
        ws.ancestral.clear();
        ws.conditioning.clear();
        for i in 0..n {
            ws.ancestral.insert(DenseNodeId::from_raw(u32::try_from(i).expect("node fit")));
        }
        self.moralize_ancestral_set(ws);
    }

    /// Richardson (2003) moralization on `ws.ancestral`: directed + bidirected undirected
    /// edges, then clique each bidirected district `C ∪ pa(C)`.
    fn moralize_ancestral_set(&self, ws: &mut DSeparationWorkspace) {
        let n = self.node_count();
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
        let n_districts = self.districts_into(
            Some(&ws.ancestral),
            &mut ws.district_label,
            &mut ws.district_stack,
        ) as usize;
        let mut groups = core::mem::take(&mut ws.district_groups);
        groups.iter_mut().for_each(Vec::clear);
        groups.resize_with(n_districts.max(groups.len()), Vec::new);
        for (i, &label) in ws.district_label.iter().enumerate() {
            if label != u32::MAX {
                groups[label as usize]
                    .push(DenseNodeId::from_raw(u32::try_from(i).expect("node fit")));
            }
        }
        let mut clique = core::mem::take(&mut ws.clique);
        for members in &groups[..n_districts] {
            clique.clear();
            for &u in members {
                if !ws.clique_mark.contains(u) {
                    ws.clique_mark.insert(u);
                    clique.push(u);
                }
                for &p in self.parents(u) {
                    if ws.ancestral.contains(p) && !ws.clique_mark.contains(p) {
                        ws.clique_mark.insert(p);
                        clique.push(p);
                    }
                }
            }
            for (ai, &a) in clique.iter().enumerate() {
                for &b in &clique[ai + 1..] {
                    Self::add_undirected(ws, a, b);
                }
            }
            for &u in &clique {
                ws.clique_mark.remove(u);
            }
        }
        ws.clique = clique;
        ws.district_groups = groups;
    }

    /// m-separation Markov blanket of `node`: neighbors in the Richardson-moralized
    /// undirected graph of the full ADMG (inducing-path / district closure).
    /// Does not include `node` itself.
    ///
    /// # Errors
    ///
    /// Unknown node id.
    pub fn markov_blanket(&self, node: DenseNodeId, out: &mut BitSet) -> Result<(), GraphError> {
        self.validate_node_pub(node)?;
        let n = self.node_count();
        out.resize(n);
        out.clear();
        let mut ws = DSeparationWorkspace::default();
        self.build_moral_full(&mut ws);
        for &nbr in &ws.undirected[node.as_usize()] {
            if nbr != node {
                out.insert(nbr);
            }
        }
        Ok(())
    }

    /// Sorted m-separation Markov blanket of `node` (excluding `node`).
    ///
    /// # Errors
    ///
    /// Unknown node id.
    pub fn markov_blanket_nodes(&self, node: DenseNodeId) -> Result<Vec<DenseNodeId>, GraphError> {
        let mut bits = BitSet::with_len(self.node_count());
        self.markov_blanket(node, &mut bits)?;
        Ok((0..self.node_count())
            .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node fit")))
            .filter(|&id| bits.contains(id))
            .collect())
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
    fn endpoint_in_z_is_not_m_separated() {
        let mut g = Admg::with_variables(2);
        let x = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        g.insert_directed(x, y).unwrap();
        let mut ws = DSeparationWorkspace::default();
        assert!(!g.is_m_separated(x, y, &[x], &mut ws).unwrap());
        assert!(!g.is_m_separated(x, y, &[y], &mut ws).unwrap());
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

    #[test]
    fn markov_blanket_includes_bidirected_neighbors() {
        // A → T ↔ U ← B  ⇒  MB(T) = {A, U, B}
        let mut g = Admg::with_variables(4);
        let a = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let u = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        g.insert_directed(a, t).unwrap();
        g.insert_bidirected(t, u).unwrap();
        g.insert_directed(b, u).unwrap();
        assert_eq!(g.markov_blanket_nodes(t).unwrap(), vec![a, u, b]);
    }

    #[test]
    fn markov_blanket_inducing_path_closure() {
        // X → A ↔ B ← Y  ⇒  MB(X) = {A, B, Y} (not merely {A}).
        let mut g = Admg::with_variables(4);
        let x = DenseNodeId::from_raw(0);
        let a = DenseNodeId::from_raw(1);
        let b = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        g.insert_directed(x, a).unwrap();
        g.insert_bidirected(a, b).unwrap();
        g.insert_directed(y, b).unwrap();
        assert_eq!(g.markov_blanket_nodes(x).unwrap(), vec![a, b, y]);
    }

    #[test]
    fn markov_blanket_matches_dag_without_bidirected() {
        // A → T ← B, T → Y ← C  ⇒  MB(T) = {A, B, Y, C}
        let mut admg = Admg::with_variables(5);
        let mut dag = crate::dag::Dag::with_variables(5);
        let a = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let b = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        let c = DenseNodeId::from_raw(4);
        for (from, to) in [(a, t), (b, t), (t, y), (c, y)] {
            admg.insert_directed(from, to).unwrap();
            dag.insert_directed(from, to).unwrap();
        }
        assert_eq!(admg.markov_blanket_nodes(t).unwrap(), dag.markov_blanket_nodes(t).unwrap());
    }

    #[test]
    fn markov_blanket_m_separates_outsiders() {
        // X → A ↔ B ← Y; MB(X) = {A,B,Y} must m-separate X from every outsider
        // (here there are none outside MB∪{X}, so also check MB(A)).
        let mut g = Admg::with_variables(4);
        let x = DenseNodeId::from_raw(0);
        let a = DenseNodeId::from_raw(1);
        let b = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        g.insert_directed(x, a).unwrap();
        g.insert_bidirected(a, b).unwrap();
        g.insert_directed(y, b).unwrap();

        let mut ws = DSeparationWorkspace::default();
        for node in [x, a, b, y] {
            let mb = g.markov_blanket_nodes(node).unwrap();
            for i in 0..g.node_count() {
                let w = DenseNodeId::from_raw(u32::try_from(i).unwrap());
                if w == node || mb.contains(&w) {
                    continue;
                }
                assert!(
                    g.is_m_separated(node, w, &mb, &mut ws).unwrap(),
                    "MB({node:?}) must m-separate from outsider {w:?}"
                );
            }
        }
    }
}

// --- PAG m-separation over the whole equivalence class ---

/// m-separation of two nodes in a PAG, as a statement about every MAG it represents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagSeparation {
    /// m-separated in every member of the class.
    Separated,
    /// m-connected in every member of the class.
    Connected,
    /// Neither holds for the whole class, or the class could not be examined
    /// (conflict marks, more circle endpoints than the completion audit
    /// enumerates, or marks that admit no maximal ancestral graph).
    Undetermined,
}

/// Outcome of the class-wide query, with the path that decided a connection.
enum ClassSeparation {
    Separated,
    Connected(Vec<DenseNodeId>),
    Undetermined,
}

impl Pag {
    /// Decide from the marks alone, without enumerating the class.
    ///
    /// A definite-status path that is active given `z` is m-connecting in every
    /// member: its colliders, non-colliders and the directed paths that open
    /// its colliders are shared by all of them. Conversely every path that is
    /// m-connecting in some member is possibly active here, so when no path is
    /// possibly active the nodes are separated in every member. `Ok(None)`
    /// means the marks settle neither; the second component is then a path
    /// that may be active in some member.
    fn separation_from_marks(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<(Option<bool>, Option<Vec<DenseNodeId>>), GraphError> {
        let mut examined = 0usize;
        let mut capped = max_paths == 0 || max_len == 0;
        let mut definite_active = None;
        let mut possibly_active = None;
        let cut = self.walk_simple_paths(x, y, max_len, |path| {
            if examined >= max_paths {
                capped = true;
                return false;
            }
            examined += 1;
            if self.path_is_definite_status(path) && self.path_active_on_edges(path, z) {
                definite_active = Some(path.to_vec());
                return false;
            }
            // A definite-status path can still be open in some member only: a
            // collider's descendants need not be definite.
            if possibly_active.is_none() && self.path_possibly_active_given(path, z) {
                possibly_active = Some(path.to_vec());
            }
            true
        });
        if definite_active.is_some() {
            return Ok((Some(false), definite_active));
        }
        if capped || cut {
            return Err(GraphError::SearchBudgetExhausted { max_paths, max_len });
        }
        Ok((possibly_active.is_none().then_some(true), possibly_active))
    }

    fn class_separation(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<ClassSeparation, GraphError> {
        self.validate_node_pub(x)?;
        self.validate_node_pub(y)?;
        for &v in z {
            self.validate_node_pub(v)?;
        }
        if x == y {
            return Ok(ClassSeparation::Connected(vec![x]));
        }
        if z.iter().any(|&v| v == x || v == y) {
            return Ok(ClassSeparation::Connected(vec![x, y]));
        }
        let (decided, path) = self.separation_from_marks(x, y, z, max_paths, max_len)?;
        match decided {
            Some(true) => return Ok(ClassSeparation::Separated),
            Some(false) => return Ok(ClassSeparation::Connected(path.unwrap_or_default())),
            None => {}
        }
        // The marks leave it open: ask every member. The audit refuses more
        // circle endpoints than it can enumerate, and yields nothing for marks
        // no maximal ancestral graph satisfies; neither is a separation.
        let Ok(members) = crate::completion::CompletionSampler::new(self.clone(), usize::MAX)
        else {
            return Ok(ClassSeparation::Undetermined);
        };
        let mut ws = DSeparationWorkspace::default();
        let mut verdict = None;
        for member in members {
            let separated =
                crate::completion::as_admg(&member.graph).is_m_separated(x, y, z, &mut ws)?;
            if *verdict.get_or_insert(separated) != separated {
                return Ok(ClassSeparation::Undetermined);
            }
        }
        Ok(match verdict {
            Some(true) => ClassSeparation::Separated,
            Some(false) => ClassSeparation::Connected(path.unwrap_or_default()),
            None => ClassSeparation::Undetermined,
        })
    }

    /// Whether `x` and `y` are m-separated given `z` in every MAG the PAG represents,
    /// m-connected in every one, or neither.
    ///
    /// Definite-status paths decide what they can; a path that is not of
    /// definite status is never discarded — it is checked for whether any
    /// member could have it m-connecting, and if so the members are enumerated
    /// (the same audited completion the identification envelopes use).
    /// `max_paths` bounds the `x`–`y` paths examined and `max_len` their length.
    ///
    /// # Errors
    ///
    /// Unknown nodes, or [`GraphError::SearchBudgetExhausted`] when the budget
    /// ran out before a connection was found: an unexplored path is never
    /// read as a separation.
    pub fn m_separation_status(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<PagSeparation, GraphError> {
        Ok(match self.class_separation(x, y, z, max_paths, max_len)? {
            ClassSeparation::Separated => PagSeparation::Separated,
            ClassSeparation::Connected(_) => PagSeparation::Connected,
            ClassSeparation::Undetermined => PagSeparation::Undetermined,
        })
    }

    /// Whether `x` is m-separated from `y` given `z` in every member of the class.
    ///
    /// `Ok(true)` and `Ok(false)` are both statements about every member; see
    /// [`Self::m_separation_status`] for the three-way outcome.
    ///
    /// # Errors
    ///
    /// Unknown nodes, an exhausted search budget, or
    /// [`GraphError::SeparationUndetermined`] when the class is neither
    /// separated nor connected throughout.
    pub fn is_m_separated(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<bool, GraphError> {
        match self.class_separation(x, y, z, max_paths, max_len)? {
            ClassSeparation::Separated => Ok(true),
            ClassSeparation::Connected(_) => Ok(false),
            ClassSeparation::Undetermined => Err(GraphError::SeparationUndetermined),
        }
    }

    /// m-separation with witness.
    ///
    /// The connecting path is a definite-status active path when one exists.
    /// When the connection was established only by enumerating the class, it
    /// is a path that is m-connecting in some member.
    ///
    /// # Errors
    ///
    /// As [`Self::is_m_separated`].
    pub fn m_separation(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<SeparationResult, GraphError> {
        match self.class_separation(x, y, z, max_paths, max_len)? {
            ClassSeparation::Separated => Ok(SeparationResult::Separated {
                conditioning: z.to_vec(),
                certificate: SeparationCertificate { conditioning: z.to_vec() },
            }),
            ClassSeparation::Connected(path) => Ok(SeparationResult::Connected {
                active_path: path.into_iter().map(|node| PathStep { node }).collect(),
            }),
            ClassSeparation::Undetermined => Err(GraphError::SeparationUndetermined),
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
    fn endpoint_in_z_is_not_m_separated() {
        let mut g = Pag::with_variables(2);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        g.insert_directed(a, b).unwrap();
        assert!(!g.is_m_separated(a, b, &[a], 32, 8).unwrap());
        assert!(!g.is_m_separated(a, b, &[b], 32, 8).unwrap());
        assert!(g.path_active_given(&[a, b], &[a]).unwrap());
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

    #[test]
    fn unshielded_circle_chain_is_connected_in_every_member() {
        // X o-o B o-o Y, X and Y not adjacent: B is a non-collider in every MAG
        // of the class, so X and Y are m-connected given the empty set.
        let mut g = Pag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_circle_circle(x, b).unwrap();
        g.insert_circle_circle(b, y).unwrap();
        assert!(!g.is_m_separated(x, y, &[], 32, 6).unwrap());
        assert!(g.is_m_separated(x, y, &[b], 32, 6).unwrap());
        let search = g.definite_status_paths(x, y, 32, 6).unwrap();
        assert_eq!(search.paths.len(), 1, "an unshielded circle triple is a definite non-collider");
    }

    #[test]
    fn conflict_marks_are_unknown_not_absent() {
        // X -> B x-x Y: the conflict leaves B's status open, so the path is
        // neither discarded (separated) nor asserted (connected).
        let mut g = Pag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(x, b).unwrap();
        g.insert_circle_circle(b, y).unwrap();
        g.mark_conflict(b, y).unwrap();
        assert_eq!(g.m_separation_status(x, y, &[], 32, 6).unwrap(), PagSeparation::Undetermined);
        assert!(matches!(
            g.is_m_separated(x, y, &[], 32, 6),
            Err(GraphError::SeparationUndetermined)
        ));
    }

    #[test]
    fn shielded_circle_path_is_settled_by_the_members() {
        // Complete PAG of X - {A, B} - Y with A - B: given {A, B} the path
        // X o-o A o-o B o-o Y is not of definite status, yet every member
        // separates X and Y there.
        let mut g = Pag::with_variables(4);
        let n = DenseNodeId::from_raw;
        for (a, b) in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)] {
            g.insert_circle_circle(n(a), n(b)).unwrap();
        }
        let z = [n(1), n(2)];
        assert_eq!(g.separation_from_marks(n(0), n(3), &z, 64, 6).unwrap().0, None);
        assert_eq!(g.m_separation_status(n(0), n(3), &z, 64, 6).unwrap(), PagSeparation::Separated);
        assert_eq!(
            g.m_separation_status(n(0), n(3), &[n(1)], 64, 6).unwrap(),
            PagSeparation::Connected
        );
    }

    fn mark(code: usize) -> crate::types::Endpoint {
        [
            crate::types::Endpoint::Tail,
            crate::types::Endpoint::Arrow,
            crate::types::Endpoint::Circle,
        ][code]
    }

    /// Graph number `index` over `n` nodes: each pair is absent or carries one of
    /// the eight mark pairs other than tail–tail. `None` when the marks close a
    /// directed cycle, which a `Pag` refuses to hold.
    fn marked_graph(n: u32, mut index: u64) -> Option<Pag> {
        let mut g = Pag::with_variables(n);
        for a in 0..n {
            for b in (a + 1)..n {
                let code = usize::try_from(index % 9).unwrap();
                index /= 9;
                if code > 0 {
                    let mut edge = crate::types::MarkedEdge::directed(
                        DenseNodeId::from_raw(a),
                        DenseNodeId::from_raw(b),
                    );
                    edge.at_a = mark(code / 3);
                    edge.at_b = mark(code % 3);
                    g.insert_marked(edge).ok()?;
                }
            }
        }
        Some(g)
    }

    #[derive(Debug, Default)]
    struct Tally {
        graphs_with_members: u64,
        queries: u64,
        marks_separated: u64,
        marks_connected: u64,
        settled_by_members: u64,
        undetermined: u64,
    }

    /// Compare every answer with m-separation in each enumerated member MAG.
    fn check_against_members(g: &Pag, tally: &mut Tally) {
        let n = g.node_count();
        let members: Vec<Admg> = crate::completion::CompletionSampler::new(g.clone(), usize::MAX)
            .unwrap()
            .map(|m| crate::completion::as_admg(&m.graph))
            .collect();
        if members.is_empty() {
            return;
        }
        tally.graphs_with_members += 1;
        let mut ws = DSeparationWorkspace::default();
        let node = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap());
        for x in 0..n {
            for y in (x + 1)..n {
                let others: Vec<usize> = (0..n).filter(|&k| k != x && k != y).collect();
                for mask in 0..(1usize << others.len()) {
                    let z: Vec<DenseNodeId> = others
                        .iter()
                        .enumerate()
                        .filter(|(bit, _)| (mask >> bit) & 1 == 1)
                        .map(|(_, &k)| node(k))
                        .collect();
                    let separated_in: Vec<bool> = members
                        .iter()
                        .map(|m| m.is_m_separated(node(x), node(y), &z, &mut ws).unwrap())
                        .collect();
                    let all = separated_in.iter().all(|&s| s);
                    let nowhere = separated_in.iter().all(|&s| !s);
                    tally.queries += 1;

                    let (from_marks, _) =
                        g.separation_from_marks(node(x), node(y), &z, 100_000, n).unwrap();
                    match from_marks {
                        Some(true) => {
                            assert!(all, "marks certify a separation some member lacks: {g:?}");
                            tally.marks_separated += 1;
                        }
                        Some(false) => {
                            assert!(nowhere, "marks certify a connection some member lacks: {g:?}");
                            tally.marks_connected += 1;
                        }
                        None => tally.settled_by_members += 1,
                    }
                    let expected = if all {
                        PagSeparation::Separated
                    } else if nowhere {
                        PagSeparation::Connected
                    } else {
                        tally.undetermined += 1;
                        PagSeparation::Undetermined
                    };
                    assert_eq!(
                        g.m_separation_status(node(x), node(y), &z, 100_000, n).unwrap(),
                        expected,
                        "x={x} y={y} z={z:?} on {g:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn class_wide_answers_match_every_member_on_small_graphs() {
        let mut tally = Tally::default();
        for index in 0..9u64.pow(3) {
            if let Some(g) = marked_graph(3, index) {
                check_against_members(&g, &mut tally);
            }
        }
        // Four nodes: a fixed-stride sample of the 9^6 mark assignments.
        for index in (0..9u64.pow(6)).step_by(211) {
            if let Some(g) = marked_graph(4, index) {
                check_against_members(&g, &mut tally);
            }
        }
        // Five nodes: sparser still; longer paths and shielded circle triples.
        for index in (0..9u64.pow(10)).step_by(1_743_391) {
            if let Some(g) = marked_graph(5, index) {
                check_against_members(&g, &mut tally);
            }
        }
        eprintln!("{tally:?}");
        assert!(tally.graphs_with_members > 300, "{tally:?}");
        assert!(tally.marks_separated > 0 && tally.marks_connected > 0, "{tally:?}");
        assert!(tally.settled_by_members > 0, "{tally:?}");
    }
}
