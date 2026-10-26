//! m-separation for ADMGs, and class-wide m-separation for PAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::admg::Admg;
use crate::dsep::{
    BayesBallGraph, DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult,
    add_undirected,
};
use crate::error::GraphError;
use crate::pag::{ClassMembers, Pag};
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
        self.validate_node(x)?;
        self.validate_node(y)?;
        for &v in z {
            self.validate_node(v)?;
        }
        if x == y {
            return Ok(false);
        }
        if z.iter().any(|&v| v == x || v == y) {
            return Ok(false);
        }
        Ok(!self.bayes_m_connected(&[x], &[y], z, ws))
    }

    /// Whether every node in `sources` is m-separated from every node in `targets`
    /// given `z`.
    ///
    /// One Bayes-ball search, the set form of the same m-separation criterion as
    /// [`Self::is_m_separated`]. Empty `sources` or `targets` is vacuously separated.
    /// A shared endpoint, or a conditioning node that is itself an endpoint, is not
    /// separated, matching the pairwise query.
    ///
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn is_m_separated_sets(
        &self,
        sources: &[DenseNodeId],
        targets: &[DenseNodeId],
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Result<bool, GraphError> {
        for &node in sources.iter().chain(targets).chain(z) {
            self.validate_node(node)?;
        }
        if sources.is_empty() || targets.is_empty() {
            return Ok(true);
        }
        let mut target_set = BitSet::with_len(self.node_count());
        for &target in targets {
            target_set.try_insert(target)?;
        }
        for &source in sources {
            if target_set.contains(source) || z.iter().any(|&node| node == source) {
                return Ok(false);
            }
        }
        for &target in targets {
            if z.iter().any(|&node| node == target) {
                return Ok(false);
            }
        }
        Ok(!self.bayes_m_connected(sources, targets, z, ws))
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
        self.validate_node(x)?;
        self.validate_node(y)?;
        for &v in z {
            self.validate_node(v)?;
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

    fn bayes_m_connected(
        &self,
        sources: &[DenseNodeId],
        targets: &[DenseNodeId],
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> bool {
        crate::dsep::bayes_ball_reaches(
            ws,
            BayesBallGraph {
                parents: self.parent_rows(),
                children: self.child_rows(),
                bidirected: self.bidirected_rows(),
                overlay: None,
            },
            sources,
            targets,
            z,
        )
    }

    /// Moral-graph answer for the same query [`Self::is_m_separated`] answers with Bayes-ball.
    #[cfg(test)]
    fn moral_m_separated(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> bool {
        self.m_sep_bool(x, y, z, ws)
    }

    #[cfg(test)]
    fn m_sep_bool(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> bool {
        self.build_moral_ancestral(x, y, z, ws);
        // Remove conditioning nodes from undirected graph by skipping them in BFS.
        !ws.moral_reaches(x, y, false)
    }

    fn m_sep_active_path(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Option<Vec<PathStep>> {
        self.build_moral_ancestral(x, y, z, ws);
        if !ws.moral_reaches(x, y, true) {
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
        self.build_moral_of_seeds(&[x, y], z, ws);
    }

    /// Ancestral closure of `seeds ∪ z`, then moralize directed + bidirected edges.
    fn build_moral_of_seeds(
        &self,
        seeds: &[DenseNodeId],
        z: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) {
        let n = self.node_count();
        ws.prepare(n);
        ws.conditioning.clear();
        for &v in z {
            ws.conditioning.insert(v);
        }

        // Ancestral set of seeds ∪ z (directed ancestors).
        ws.frontier.clear();
        ws.frontier.extend_from_slice(seeds);
        ws.frontier.extend_from_slice(z);
        self.ancestors_of(&ws.frontier, &mut ws.ancestral, &mut ws.graph_ws);
        ws.frontier.clear();

        self.moralize_ancestral_set(ws);
    }

    /// Richardson (2003) moralization on `ws.ancestral`: directed + bidirected undirected
    /// edges, then clique each bidirected district `C ∪ pa(C)`.
    fn moralize_ancestral_set(&self, ws: &mut DSeparationWorkspace) {
        let n = self.node_count();
        for i in 0..n {
            let u = DenseNodeId::from_index(i);
            if !ws.ancestral.contains(u) {
                continue;
            }
            for &c in self.children(u) {
                if ws.ancestral.contains(c) {
                    add_undirected(&mut ws.undirected, &mut ws.undirected_member, u, c);
                }
            }
            for &b in self.bidirected_neighbors(u) {
                if b.raw() > u.raw() && ws.ancestral.contains(b) {
                    add_undirected(&mut ws.undirected, &mut ws.undirected_member, u, b);
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
                groups[label as usize].push(DenseNodeId::from_index(i));
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
                    add_undirected(&mut ws.undirected, &mut ws.undirected_member, a, b);
                }
            }
            for &u in &clique {
                ws.clique_mark.remove(u);
            }
        }
        ws.clique = clique;
        ws.district_groups = groups;
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
    fn set_separation_matches_every_pair() {
        // Collider 0 -> 2 <- 1, plus 2 -> 3. An extra descendant must not
        // open the collider when the set query is the empty conditioning set.
        let mut g = Admg::with_variables(4);
        let n = |i| DenseNodeId::from_raw(i);
        g.insert_directed(n(0), n(2)).unwrap();
        g.insert_directed(n(1), n(2)).unwrap();
        g.insert_directed(n(2), n(3)).unwrap();
        let mut ws = DSeparationWorkspace::default();
        let cases: &[(&[DenseNodeId], &[DenseNodeId], &[DenseNodeId])] = &[
            (&[n(0)], &[n(1)], &[]),
            (&[n(0)], &[n(1)], &[n(2)]),
            (&[n(0), n(1)], &[n(3)], &[]),
            (&[n(0)], &[n(3)], &[n(2)]),
            (&[n(0)], &[n(0)], &[]),
            (&[], &[n(1)], &[]),
            (&[n(3)], &[], &[n(2)]),
        ];
        for &(sources, targets, z) in cases {
            let sets = g.is_m_separated_sets(sources, targets, z, &mut ws).unwrap();
            let pairwise = sources.iter().all(|&source| {
                targets.iter().all(|&target| g.is_m_separated(source, target, z, &mut ws).unwrap())
            });
            assert_eq!(sets, pairwise, "sources={sources:?} targets={targets:?} z={z:?}");
        }
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
    fn bayes_ball_matches_moral_graph_on_fixtures() {
        let n = |i| DenseNodeId::from_raw(i);
        let mut fork = Admg::with_variables(3);
        fork.insert_directed(n(1), n(0)).unwrap();
        fork.insert_directed(n(1), n(2)).unwrap();
        let mut collider = Admg::with_variables(3);
        collider.insert_directed(n(0), n(1)).unwrap();
        collider.insert_directed(n(2), n(1)).unwrap();
        let mut chain = Admg::with_variables(4);
        chain.insert_directed(n(0), n(2)).unwrap();
        chain.insert_directed(n(1), n(2)).unwrap();
        chain.insert_directed(n(2), n(3)).unwrap();
        let mut district = Admg::with_variables(4);
        district.insert_directed(n(0), n(1)).unwrap();
        district.insert_bidirected(n(1), n(2)).unwrap();
        district.insert_directed(n(3), n(2)).unwrap();
        let mut latent = Admg::with_variables(3);
        latent.insert_bidirected(n(0), n(1)).unwrap();
        latent.insert_bidirected(n(1), n(2)).unwrap();
        let graphs = [&fork, &collider, &chain, &district, &latent];
        let mut ws = DSeparationWorkspace::default();
        for g in graphs {
            let count = u32::try_from(g.node_count()).unwrap();
            for x in 0..count {
                for y in 0..count {
                    if x == y {
                        continue;
                    }
                    let others: Vec<u32> = (0..count).filter(|&k| k != x && k != y).collect();
                    for mask in 0..(1u32 << others.len()) {
                        let z: Vec<DenseNodeId> = others
                            .iter()
                            .enumerate()
                            .filter(|(bit, _)| (mask >> bit) & 1 == 1)
                            .map(|(_, &k)| n(k))
                            .collect();
                        let moral = g.moral_m_separated(n(x), n(y), &z, &mut ws);
                        let bayes = !g.bayes_m_connected(&[n(x)], &[n(y)], &z, &mut ws);
                        assert_eq!(bayes, moral, "x={x} y={y} z={z:?}");
                    }
                }
            }
        }
        let sources = [n(0), n(1)];
        let targets = [n(3)];
        let z = [n(2)];
        chain.build_moral_of_seeds(&[n(0), n(1), n(3)], &z, &mut ws);
        let mut target_set = BitSet::with_len(chain.node_count());
        target_set.insert(n(3));
        let moral_sets = !ws.moral_reaches_any(&sources, &target_set);
        let bayes_sets = !chain.bayes_m_connected(&sources, &targets, &z, &mut ws);
        assert_eq!(bayes_sets, moral_sets);
    }

    #[test]
    fn bayes_ball_matches_moral_graph_on_all_four_node_admgs() {
        // Every four-node DAG has a topological relabeling represented by one of
        // these 64 forward-edge subsets. Bidirected masks cover all six possible
        // latent-fork edges. The moral-graph implementation is a separate
        // ancestral-moralization criterion, so this exhaustively differentially
        // checks Bayes-ball across every pair and disjoint conditioning set.
        let directed_edges = [(0_u32, 1_u32), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let bidirected_edges = [(0_u32, 1_u32), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let id = |raw| DenseNodeId::from_raw(raw);
        let mut ws = DSeparationWorkspace::default();

        for directed_mask in 0_u32..64 {
            for bidirected_mask in 0_u32..64 {
                let mut graph = Admg::with_variables(4);
                for (bit, &(from, to)) in directed_edges.iter().enumerate() {
                    if directed_mask & (1_u32 << bit) != 0 {
                        graph.insert_directed(id(from), id(to)).unwrap();
                    }
                }
                for (bit, &(a, b)) in bidirected_edges.iter().enumerate() {
                    if bidirected_mask & (1_u32 << bit) != 0 {
                        graph.insert_bidirected(id(a), id(b)).unwrap();
                    }
                }

                for x in 0_u32..4 {
                    for y in (x + 1)..4 {
                        let remaining: Vec<_> =
                            (0_u32..4).filter(|&node| node != x && node != y).collect();
                        for conditioning_mask in 0_u32..(1_u32 << remaining.len()) {
                            let conditioning: Vec<_> = remaining
                                .iter()
                                .enumerate()
                                .filter(|(bit, _)| conditioning_mask & (1_u32 << bit) != 0)
                                .map(|(_, &node)| id(node))
                                .collect();
                            let x = id(x);
                            let y = id(y);
                            let moral = graph.moral_m_separated(x, y, &conditioning, &mut ws);
                            let bayes =
                                !graph.bayes_m_connected(&[x], &[y], &conditioning, &mut ws);
                            assert_eq!(
                                bayes,
                                moral,
                                "directed={directed_mask:06b} bidirected={bidirected_mask:06b} \
                                 x={} y={} z={:?}",
                                x.raw(),
                                y.raw(),
                                conditioning.iter().map(|node| node.raw()).collect::<Vec<_>>(),
                            );
                        }
                    }
                }
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
    /// (conflict marks, a completion audit that exhausts its search budget or
    /// cannot afford the equivalence audit, or marks that admit no maximal
    /// ancestral graph).
    Undetermined,
}

/// What the marks alone settle about a PAG separation query.
enum MarkDecision {
    /// No path is possibly active: separated in every member.
    Separated,
    /// A definite-status active path connects the nodes in every member.
    Connected(Vec<DenseNodeId>),
    /// The marks settle neither; the path may be active in some member.
    Open(Vec<DenseNodeId>),
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
    /// possibly active the nodes are separated in every member. [`MarkDecision::Open`]
    /// means the marks settle neither and carries a path that may be active in some member.
    fn separation_from_marks(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<MarkDecision, GraphError> {
        let mut examined = 0usize;
        let mut capped = max_paths == 0 || max_len == 0;
        let mut definite_active = None;
        let mut possibly_active = None;
        let mut memo = crate::pag::DescendantMemo::default();
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
            if possibly_active.is_none() && self.path_possibly_active_given(path, z, &mut memo) {
                possibly_active = Some(path.to_vec());
            }
            true
        });
        if let Some(path) = definite_active {
            return Ok(MarkDecision::Connected(path));
        }
        if capped || cut {
            return Err(GraphError::SearchBudgetExhausted { max_paths, max_len });
        }
        Ok(possibly_active.map_or(MarkDecision::Separated, MarkDecision::Open))
    }

    fn class_separation(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        z: &[DenseNodeId],
        max_paths: usize,
        max_len: usize,
    ) -> Result<ClassSeparation, GraphError> {
        self.validate_node(x)?;
        self.validate_node(y)?;
        for &v in z {
            self.validate_node(v)?;
        }
        if x == y {
            return Ok(ClassSeparation::Connected(vec![x]));
        }
        if z.iter().any(|&v| v == x || v == y) {
            return Ok(ClassSeparation::Connected(vec![x, y]));
        }
        let possible_path = match self.separation_from_marks(x, y, z, max_paths, max_len)? {
            MarkDecision::Separated => return Ok(ClassSeparation::Separated),
            MarkDecision::Connected(path) => return Ok(ClassSeparation::Connected(path)),
            MarkDecision::Open(path) => path,
        };
        // The marks leave it open: ask every member. The audit is refused when its search
        // budget cannot enumerate and certify the whole class, and yields nothing for marks
        // no maximal ancestral graph satisfies; neither is a separation.
        let enumerated;
        let members: &[Admg] = match self.class_members() {
            ClassMembers::Refused => return Ok(ClassSeparation::Undetermined),
            ClassMembers::Cached(members) => {
                enumerated = members;
                &enumerated
            }
            ClassMembers::Uncached => {
                let Some(members) = self.build_class_members() else {
                    return Ok(ClassSeparation::Undetermined);
                };
                enumerated = members.into();
                &enumerated
            }
        };
        let mut ws = DSeparationWorkspace::default();
        let mut verdict = None;
        for member in members {
            let separated = member.is_m_separated(x, y, z, &mut ws)?;
            if *verdict.get_or_insert(separated) != separated {
                return Ok(ClassSeparation::Undetermined);
            }
        }
        Ok(match verdict {
            Some(true) => ClassSeparation::Separated,
            Some(false) => ClassSeparation::Connected(possible_path),
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
        assert!(matches!(g.separation_from_marks(n(0), n(3), &z, 64, 6).unwrap(), MarkDecision::Open(_)));
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
        let node = DenseNodeId::from_index;
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

                    let from_marks =
                        g.separation_from_marks(node(x), node(y), &z, 100_000, n).unwrap();
                    match from_marks {
                        MarkDecision::Separated => {
                            assert!(all, "marks certify a separation some member lacks: {g:?}");
                            tally.marks_separated += 1;
                        }
                        MarkDecision::Connected(_) => {
                            assert!(nowhere, "marks certify a connection some member lacks: {g:?}");
                            tally.marks_connected += 1;
                        }
                        MarkDecision::Open(_) => tally.settled_by_members += 1,
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
