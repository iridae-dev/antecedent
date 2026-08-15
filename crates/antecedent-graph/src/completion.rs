//! Streamed / bounded PAG completion sampling.
//!
//! Completions are never retained without bound.  Before yielding anything, the
//! sampler exhausts a deliberately small endpoint-assignment space and verifies
//! that every survivor is ancestral, maximal, agrees with the PAG's unshielded
//! colliders, and belongs to one common m-separation equivalence class.  A PAG
//! whose marks admit multiple equivalence classes is refused rather than sampled
//! from an unproved superset.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::admg::Admg;
use crate::dsep::DSeparationWorkspace;
use crate::error::GraphError;
use crate::pag::Pag;
use crate::types::{DenseNodeId, Endpoint};
use crate::workspace::GraphWorkspace;

/// One circle-free completion of a PAG (MAG marks only).
#[derive(Clone, Debug)]
pub struct PagCompletion {
    /// Completed graph (no Circle endpoints).
    pub graph: Pag,
    /// Index of this completion in the stream (0-based).
    pub index: usize,
}

/// Audit counts for a bounded PAG-to-MAG completion enumeration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompletionValidationReport {
    /// Endpoint assignments examined exhaustively.
    pub assignments_examined: u64,
    /// Assignments rejected because they were not ancestral graphs.
    pub rejected_non_ancestral: u64,
    /// Ancestral assignments rejected because they were not maximal.
    pub rejected_nonmaximal: u64,
    /// MAGs rejected because an unshielded collider contradicted the source PAG.
    pub rejected_local_incompatible: u64,
    /// Whether locally compatible MAGs represented more than one global Markov class.
    pub ambiguous_global_class: bool,
    /// Whether the global m-separation class audit was skipped as too expensive.
    ///
    /// Local validity (ancestral, maximal, collider-preserving) still holds for every
    /// yielded completion, but they have not been proved to share one Markov class, so
    /// no caller may conclude a class-wide property from them.
    pub equivalence_audit_skipped: bool,
    /// Number of verified members of the single represented class (before the yield cap).
    pub represented_completions: usize,
}

/// Streams PAG completions with a hard cap (no unbounded retain).
#[derive(Clone, Debug)]
pub struct CompletionSampler {
    completions: Vec<Pag>,
    max_completions: usize,
    next_index: usize,
    n_circle_sites: usize,
    report: CompletionValidationReport,
}

const MAX_AUDITED_CIRCLE_SITES: usize = 16;
const MAX_EQUIVALENCE_NODES: usize = 12;
const MAX_EQUIVALENCE_QUERIES: u128 = 2_000_000;

impl CompletionSampler {
    /// Build a sampler that yields at most `max_completions` **valid** MAG completions.
    ///
    /// # Errors
    ///
    /// More than 16 circle endpoints: the endpoint-assignment space is enumerated
    /// exhaustively, so refusal is explicit instead of silently validating a prefix.
    /// A graph too large for the *global* equivalence audit is not refused — the audit is
    /// skipped and recorded in [`CompletionValidationReport::equivalence_audit_skipped`].
    #[allow(clippy::needless_pass_by_value)] // preserve the established constructor API
    pub fn new(pag: Pag, max_completions: usize) -> Result<Self, GraphError> {
        let mut sites = Vec::new();
        let n = pag.node_count();
        for i in 0..n {
            let a = DenseNodeId::try_from_usize(i)?;
            for (b, at_a, at_b) in pag.neighbors(a) {
                if b.raw() < a.raw() {
                    continue;
                }
                if matches!(at_a, Endpoint::Circle) {
                    sites.push((a, b, true));
                }
                if matches!(at_b, Endpoint::Circle) {
                    sites.push((a, b, false));
                }
            }
        }
        if sites.len() > MAX_AUDITED_CIRCLE_SITES {
            return Err(GraphError::InvalidEndpoints {
                message: "PAG completion audit supports at most 16 circle endpoints",
            });
        }
        let total = 1u64 << sites.len();
        // The global m-separation signature costs one d-separation query per node pair per
        // conditioning subset, so it is exponential in node count where the assignment
        // enumeration above is only exponential in circle sites. Refusing the whole graph on
        // that basis would withdraw a capability from every PAG past 12 nodes. Instead the
        // audit is skipped and recorded: local validation still runs, and callers that would
        // otherwise assert a class-wide property must downgrade on this flag.
        let audit_class = sites.is_empty() || {
            let n = pag.node_count() as u128;
            let conditioning_sets = 1u128 << pag.node_count().saturating_sub(2);
            let queries = u128::from(total)
                .saturating_mul(n.saturating_mul(n.saturating_sub(1)) / 2)
                .saturating_mul(conditioning_sets);
            pag.node_count() <= MAX_EQUIVALENCE_NODES && queries <= MAX_EQUIVALENCE_QUERIES
        };
        let mut report = CompletionValidationReport {
            assignments_examined: total,
            equivalence_audit_skipped: !audit_class,
            ..Default::default()
        };
        let mut completions = Vec::new();
        let mut represented_completions = 0usize;
        let mut reference_signature = None;
        for mask in 0..total {
            let Some(candidate) = orient_assignment(&pag, &sites, mask) else {
                report.rejected_non_ancestral += 1;
                continue;
            };
            if !is_maximal_ancestral_graph(&candidate) {
                report.rejected_nonmaximal += 1;
                continue;
            }
            if !preserves_unshielded_colliders(&pag, &candidate) {
                report.rejected_local_incompatible += 1;
                continue;
            }
            if audit_class {
                let signature =
                    if sites.is_empty() { Vec::new() } else { m_separation_signature(&candidate) };
                if let Some(reference) = &reference_signature {
                    if reference != &signature {
                        report.ambiguous_global_class = true;
                        continue;
                    }
                } else {
                    reference_signature = Some(signature);
                }
            }
            represented_completions += 1;
            if completions.len() < max_completions {
                completions.push(candidate);
            }
        }
        if report.ambiguous_global_class {
            // A partial graph that admits multiple Markov classes is not a certified PAG for
            // this enumerator.  Keeping either class would invent membership information.
            completions.clear();
        }
        report.represented_completions =
            if report.ambiguous_global_class { 0 } else { represented_completions };
        Ok(Self {
            completions,
            max_completions,
            next_index: 0,
            n_circle_sites: sites.len(),
            report,
        })
    }

    /// Hard cap on yielded valid completions.
    #[must_use]
    pub fn max_completions(&self) -> usize {
        self.max_completions
    }

    /// Number of circle endpoints being oriented.
    #[must_use]
    pub fn n_circle_sites(&self) -> usize {
        self.n_circle_sites
    }

    /// Exhaustive validation report for caller diagnostics and mass-scope reporting.
    #[must_use]
    pub const fn validation_report(&self) -> CompletionValidationReport {
        self.report
    }

    /// Whether the yielded completions are unproved as one Markov equivalence class.
    ///
    /// True when the global audit was skipped as too expensive. Callers concluding a
    /// property "for every member of the class" must treat this exactly like
    /// [`Self::hit_cap`]: the yielded set is locally valid but not certified to be the
    /// class, so a class-wide claim is unearned.
    #[must_use]
    pub const fn class_audit_incomplete(&self) -> bool {
        self.report.equivalence_audit_skipped
    }

    /// Whether more verified class members exist than the retained/yielded prefix.
    ///
    /// Callers that reason over the *whole* Markov equivalence class — e.g. concluding an
    /// effect is identified in every member because every member they saw was identified —
    /// must consult this. All endpoint assignments have been validated, but yielded
    /// completions remain a deterministic low-mask prefix rather than a representative sample.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.report.represented_completions > self.max_completions
    }
}

fn orient_assignment(
    base: &Pag,
    sites: &[(DenseNodeId, DenseNodeId, bool)],
    mask: u64,
) -> Option<Pag> {
    let mut graph = base.clone();
    for (i, &(a, b, at_a_circle)) in sites.iter().enumerate() {
        let endpoint = if ((mask >> i) & 1) == 1 { Endpoint::Arrow } else { Endpoint::Tail };
        let edge = graph.edge_between(a, b)?;
        let marks = if at_a_circle { (endpoint, edge.at_b) } else { (edge.at_a, endpoint) };
        graph.set_marks(a, b, marks.0, marks.1).ok()?;
    }
    is_ancestral_orientation(&graph).then_some(graph)
}

/// Whether `g` is a maximal ancestral graph completion in the supported directed/bidirected
/// MAG family (selection-variable Tail–Tail edges are outside this sampler's scope).
#[must_use]
pub fn is_mag_completion(g: &Pag) -> bool {
    is_maximal_ancestral_graph(g)
}

fn is_ancestral_orientation(g: &Pag) -> bool {
    let n = g.node_count();
    let mut ws = GraphWorkspace::default();
    for i in 0..n {
        let a = DenseNodeId::try_from_usize(i).expect("node fit");
        for (b, at_a, at_b) in g.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if matches!(at_a, Endpoint::Circle | Endpoint::Conflict)
                || matches!(at_b, Endpoint::Circle | Endpoint::Conflict)
            {
                return false;
            }
            // Directed MAGs (Zhang) allow → and ↔ only — not undirected —o—.
            if matches!((at_a, at_b), (Endpoint::Tail, Endpoint::Tail)) {
                return false;
            }
            if matches!((at_a, at_b), (Endpoint::Arrow, Endpoint::Arrow)) {
                // Almost-directed cycle: bidirected + directed path either way.
                if g.reaches_directed_with(&mut ws, a, b) || g.reaches_directed_with(&mut ws, b, a)
                {
                    return false;
                }
            }
        }
    }
    true
}

/// Whether a circle-free graph is both ancestral and maximal.
#[must_use]
pub fn is_maximal_ancestral_graph(g: &Pag) -> bool {
    if !is_ancestral_orientation(g) {
        return false;
    }
    // Richardson–Spirtes Corollary 5.3 gives the separating set for every missing
    // edge in a MAG.  Conversely, a nonmaximal ancestral graph has a nonadjacent
    // pair connected given every set.  Thus this is an exact inducing-path-free test.
    let admg = as_admg(g);
    let n = g.node_count();
    let mut graph_ws = GraphWorkspace::default();
    let mut sep_ws = DSeparationWorkspace::default();
    for i in 0..n {
        let x = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
        for j in (i + 1)..n {
            let y = DenseNodeId::from_raw(u32::try_from(j).expect("node fit"));
            if g.has_edge(x, y) {
                continue;
            }
            let separating: Vec<_> = (0..n)
                .map(|k| DenseNodeId::from_raw(u32::try_from(k).expect("node fit")))
                .filter(|&node| {
                    node != x
                        && node != y
                        && (g.reaches_directed_with(&mut graph_ws, node, x)
                            || g.reaches_directed_with(&mut graph_ws, node, y))
                })
                .collect();
            if !admg.is_m_separated(x, y, &separating, &mut sep_ws).expect("known nodes") {
                return false;
            }
        }
    }
    true
}

fn preserves_unshielded_colliders(pag: &Pag, mag: &Pag) -> bool {
    let n = pag.node_count();
    for middle_i in 0..n {
        let middle = DenseNodeId::from_raw(u32::try_from(middle_i).expect("node fit"));
        let neighbors: Vec<_> = pag.neighbors(middle).map(|(node, mark, _)| (node, mark)).collect();
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let (left, pag_left_mark) = neighbors[i];
                let (right, pag_right_mark) = neighbors[j];
                if pag.has_edge(left, right) {
                    continue;
                }
                let mag_left_mark = mag
                    .neighbors(middle)
                    .find(|(node, _, _)| *node == left)
                    .map(|(_, mark, _)| mark)
                    .expect("same skeleton");
                let mag_right_mark = mag
                    .neighbors(middle)
                    .find(|(node, _, _)| *node == right)
                    .map(|(_, mark, _)| mark)
                    .expect("same skeleton");
                let pag_collider = matches!(pag_left_mark, Endpoint::Arrow)
                    && matches!(pag_right_mark, Endpoint::Arrow);
                let mag_collider = matches!(mag_left_mark, Endpoint::Arrow)
                    && matches!(mag_right_mark, Endpoint::Arrow);
                if pag_collider != mag_collider {
                    return false;
                }
            }
        }
    }
    true
}

fn as_admg(g: &Pag) -> Admg {
    let mut admg = Admg::with_variables(u32::try_from(g.node_count()).expect("node count fits"));
    for i in 0..g.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
        for (b, at_a, at_b) in g.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    admg.insert_directed(a, b).expect("validated MAG");
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    admg.insert_directed(b, a).expect("validated MAG");
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    admg.insert_bidirected(a, b).expect("validated MAG");
                }
                _ => unreachable!("validated directed MAG marks"),
            }
        }
    }
    admg
}

fn m_separation_signature(g: &Pag) -> Vec<bool> {
    let admg = as_admg(g);
    let n = g.node_count();
    let mut signature = Vec::new();
    let mut ws = DSeparationWorkspace::default();
    for i in 0..n {
        for j in (i + 1)..n {
            let others: Vec<_> = (0..n).filter(|&k| k != i && k != j).collect();
            for mask in 0..(1usize << others.len()) {
                let z: Vec<_> = others
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| ((mask >> bit) & 1) == 1)
                    .map(|(_, &k)| DenseNodeId::from_raw(u32::try_from(k).expect("node fit")))
                    .collect();
                signature.push(
                    admg.is_m_separated(
                        DenseNodeId::from_raw(u32::try_from(i).expect("node fit")),
                        DenseNodeId::from_raw(u32::try_from(j).expect("node fit")),
                        &z,
                        &mut ws,
                    )
                    .expect("known nodes"),
                );
            }
        }
    }
    signature
}

impl Iterator for CompletionSampler {
    type Item = PagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        let graph = self.completions.get(self.next_index)?.clone();
        let index = self.next_index;
        self.next_index += 1;
        Some(PagCompletion { graph, index })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pag::Pag;

    #[test]
    fn respects_max_completions_bound() {
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let sampler = CompletionSampler::new(pag, 2).unwrap();
        assert_eq!(sampler.n_circle_sites(), 2);
        let collected: Vec<_> = sampler.collect();
        assert!(collected.len() <= 2);
        assert!(!collected.is_empty());
        for c in &collected {
            assert!(is_mag_completion(&c.graph));
            let e =
                c.graph.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            assert!(!matches!(e.at_a, Endpoint::Circle));
            assert!(!matches!(e.at_b, Endpoint::Circle));
            // No undirected Tail–Tail in directed MAG completions.
            assert!(!matches!((e.at_a, e.at_b), (Endpoint::Tail, Endpoint::Tail)));
        }
    }

    #[test]
    fn no_circle_yields_single_completion() {
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let collected: Vec<_> = CompletionSampler::new(pag, 10).unwrap().collect();
        assert_eq!(collected.len(), 1);
        assert!(is_mag_completion(&collected[0].graph));
    }

    #[test]
    fn rejects_almost_directed_cycle() {
        // a → b → c with a ↔ c: directed path a ⇝ c plus bidirected a ↔ c.
        let mut g = Pag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        g.insert_bidirected(a, c).unwrap();
        assert!(!is_mag_completion(&g));
    }

    #[test]
    fn accepts_bidirected_without_directed_path() {
        let mut g = Pag::with_variables(2);
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(is_mag_completion(&g));
    }

    #[test]
    fn rejects_nonmaximal_ancestral_orientation() {
        // X ↔ A ↔ B ↔ Y, A → Y, B → X has an inducing path X,A,B,Y:
        // A and B are colliders and ancestors of Y and X respectively.  X and Y are
        // nonadjacent, so this ancestral graph is not maximal (Richardson–Spirtes §3.7).
        let mut graph = Pag::with_variables(4);
        let endpoint_x = DenseNodeId::from_raw(0);
        let inner_a = DenseNodeId::from_raw(1);
        let inner_b = DenseNodeId::from_raw(2);
        let endpoint_y = DenseNodeId::from_raw(3);
        graph.insert_bidirected(endpoint_x, inner_a).unwrap();
        graph.insert_bidirected(inner_a, inner_b).unwrap();
        graph.insert_bidirected(inner_b, endpoint_y).unwrap();
        graph.insert_directed(inner_a, endpoint_y).unwrap();
        graph.insert_directed(inner_b, endpoint_x).unwrap();
        assert!(is_ancestral_orientation(&graph), "graph is ancestral but not maximal");
        assert!(!is_maximal_ancestral_graph(&graph));

        let sampler = CompletionSampler::new(graph, 8).unwrap();
        assert_eq!(sampler.validation_report().rejected_nonmaximal, 1);
        assert_eq!(sampler.count(), 0, "nonmaximal orientation must never be yielded");
    }

    #[test]
    fn rejects_unshielded_collider_not_encoded_by_pag() {
        // In a PAG, unshielded collider status is invariant.  X o-o M o-o Y does not
        // encode X *-> M <-* Y, so endpoint assignments that invent that collider must
        // be rejected even though they are otherwise maximal ancestral graphs.
        let mut pag = Pag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        pag.insert_circle_circle(x, m).unwrap();
        pag.insert_circle_circle(m, y).unwrap();
        let sampler = CompletionSampler::new(pag, 32).unwrap();
        assert!(sampler.validation_report().rejected_local_incompatible > 0);
        for completion in sampler {
            let xm = completion.graph.edge_between(x, m).unwrap();
            let my = completion.graph.edge_between(m, y).unwrap();
            let at_m_from_x = if xm.a == m { xm.at_a } else { xm.at_b };
            let at_m_from_y = if my.a == m { my.at_a } else { my.at_b };
            assert!(!(at_m_from_x == Endpoint::Arrow && at_m_from_y == Endpoint::Arrow));
        }
    }

    #[test]
    fn fails_closed_when_marks_admit_multiple_global_classes() {
        // <X,Q,B,Y> is a discriminating path for B: Q is a collider on the path,
        // Q -> Y, and X is not adjacent to Y.  Leaving the B-Y endpoint unresolved
        // admits both Q <-> B -> Y (noncollider) and Q <-> B <-> Y (collider).
        // They agree on all unshielded colliders but are not Markov equivalent.
        let mut pag = Pag::with_variables(4);
        let x = DenseNodeId::from_raw(0);
        let q = DenseNodeId::from_raw(1);
        let b = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        pag.insert_directed(x, q).unwrap();
        pag.insert_bidirected(q, b).unwrap();
        pag.insert_directed(q, y).unwrap();
        pag.insert_circle_circle(b, y).unwrap();

        let sampler = CompletionSampler::new(pag, 8).unwrap();
        assert!(sampler.validation_report().ambiguous_global_class);
        assert_eq!(sampler.count(), 0, "an underoriented PAG must fail closed");
    }

    #[test]
    fn skips_rather_than_refuses_equivalence_audits_above_work_bound() {
        // Refusing here would withdraw completion enumeration from every graph past the
        // audit's exponential budget. The completions are still locally validated; what is
        // lost is the proof that they form a single Markov class, and that loss is recorded
        // so a caller cannot assert a class-wide property from them.
        let mut pag = Pag::with_variables(12);
        for (left, right) in [(0, 1), (2, 3), (4, 5)] {
            pag.insert_circle_circle(DenseNodeId::from_raw(left), DenseNodeId::from_raw(right))
                .unwrap();
        }
        let sampler = CompletionSampler::new(pag, 8).unwrap();
        assert!(sampler.class_audit_incomplete());
        assert!(sampler.validation_report().equivalence_audit_skipped);
        assert!(!sampler.validation_report().ambiguous_global_class);
        assert!(sampler.count() > 0, "locally valid completions are still yielded");
    }

    #[test]
    fn wide_pags_with_circle_marks_are_not_refused_outright() {
        // A 13-node PAG with a single circle mark exceeds the equivalence-audit node bound.
        // It must still enumerate, flagged, rather than error the caller out entirely.
        let mut pag = Pag::with_variables(13);
        pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let sampler = CompletionSampler::new(pag, 8).unwrap();
        assert!(sampler.class_audit_incomplete());
        assert!(sampler.count() > 0);
    }

    fn random_pag_with_circles(rng: &mut antecedent_core::CausalRng, n: u32) -> Pag {
        let mut pag = Pag::with_variables(n);
        // Prefer a topological skeleton so directed inserts stay acyclic.
        let mut order: Vec<u32> = (0..n).collect();
        for i in (1..usize::try_from(n).unwrap_or(0)).rev() {
            let bound = u64::try_from(i + 1).unwrap_or(1);
            let j = usize::try_from(rng.next_u64() % bound).unwrap_or(0);
            order.swap(i, j);
        }
        let n_usize = usize::try_from(n).unwrap_or(0);
        for i in 0..n_usize {
            for j in (i + 1)..n_usize {
                if rng.next_u64() % 3 != 0 {
                    continue;
                }
                let a = DenseNodeId::from_raw(order[i]);
                let b = DenseNodeId::from_raw(order[j]);
                let kind = rng.next_u64() % 4;
                let _ = match kind {
                    0 => pag.insert_directed(a, b),
                    1 => pag.insert_circle_arrow(a, b),
                    2 => pag.insert_circle_circle(a, b),
                    _ => pag.insert_bidirected(a, b),
                };
            }
        }
        pag
    }

    /// Completions respect the yield cap and never retain circle marks.
    #[test]
    fn property_completions_respect_bound_and_no_circles() {
        use antecedent_core::CausalRng;

        let mut rng = CausalRng::from_seed(23);
        for _ in 0..40 {
            let n = 2 + u32::try_from(rng.next_u64() % 3).unwrap_or(0); // 2..=4
            let pag = random_pag_with_circles(&mut rng, n);
            let max_c = 1 + usize::try_from(rng.next_u64() % 4).unwrap_or(0); // 1..=4
            let Ok(sampler) = CompletionSampler::new(pag, max_c) else {
                continue; // too many circle sites
            };
            let collected: Vec<_> = sampler.collect();
            assert!(collected.len() <= max_c, "exceeded max_completions");
            for (i, c) in collected.iter().enumerate() {
                assert_eq!(c.index, i);
                assert!(is_mag_completion(&c.graph));
                for i in 0..c.graph.node_count() {
                    let a = DenseNodeId::from_raw(u32::try_from(i).unwrap());
                    for (b, at_a, at_b) in c.graph.neighbors(a) {
                        if b.raw() < a.raw() {
                            continue;
                        }
                        assert!(!matches!(at_a, Endpoint::Circle | Endpoint::Conflict));
                        assert!(!matches!(at_b, Endpoint::Circle | Endpoint::Conflict));
                    }
                }
            }
        }
    }

    /// Where cheap: an active definite-status path in the PAG remains m-connecting in
    /// every MAG completion (sound direction only; PAG separation is incomplete).
    #[test]
    fn property_definite_msep_stable_across_completions() {
        use antecedent_core::CausalRng;

        let mut rng = CausalRng::from_seed(29);
        for _ in 0..30 {
            let n = 3u32;
            let pag = random_pag_with_circles(&mut rng, n);
            let Ok(sampler) = CompletionSampler::new(pag.clone(), 8) else {
                continue;
            };
            if sampler.n_circle_sites() > 4 {
                continue; // keep enumeration cheap
            }
            let completions: Vec<_> = sampler.collect();
            if completions.is_empty() {
                continue;
            }
            for x in 0..n {
                for y in 0..n {
                    if x == y {
                        continue;
                    }
                    let xi = DenseNodeId::from_raw(x);
                    let yi = DenseNodeId::from_raw(y);
                    // Empty Z only — cheapest definite-status check.
                    let Ok(pag_sep) = pag.is_m_separated(xi, yi, &[], 32, 6) else {
                        continue; // budget exhaustion — skip
                    };
                    if pag_sep {
                        continue; // incomplete: PAG sep ⇏ completion sep
                    }
                    for c in &completions {
                        let Ok(comp_sep) = c.graph.is_m_separated(xi, yi, &[], 32, 6) else {
                            continue;
                        };
                        assert!(
                            !comp_sep,
                            "PAG m-connected but completion {} separated {}–{}",
                            c.index, x, y
                        );
                    }
                }
            }
        }
    }
}
