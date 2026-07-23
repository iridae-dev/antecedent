//! Streamed / bounded PAG completion sampling.
//!
//! Completions are never retained without bound: the sampler yields at most
//! `max_completions` **valid MAG** completions (circle-free ancestral graphs).
//! Invalid orientations (directed cycles, almost-directed cycles, undirected
//! Tail–Tail marks) are skipped and do not count toward the yield cap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Streams PAG completions with a hard cap (no unbounded retain).
#[derive(Clone, Debug)]
pub struct CompletionSampler {
    base: Pag,
    circle_sites: Vec<(DenseNodeId, DenseNodeId, bool)>, // (a,b, at_a_is_circle)
    max_completions: usize,
    next_index: usize,
    /// Bitmask assignment for circle sites (site i uses bit i); advances as a counter.
    assign: u64,
}

impl CompletionSampler {
    /// Build a sampler that yields at most `max_completions` **valid** MAG completions.
    ///
    /// # Errors
    ///
    /// More than 63 circle endpoints (mask capacity).
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
        if sites.len() > 63 {
            return Err(GraphError::InvalidEndpoints {
                message: "too many circle endpoints for CompletionSampler mask",
            });
        }
        Ok(Self { base: pag, circle_sites: sites, max_completions, next_index: 0, assign: 0 })
    }

    /// Hard cap on yielded valid completions.
    #[must_use]
    pub fn max_completions(&self) -> usize {
        self.max_completions
    }

    /// Number of circle endpoints being oriented.
    #[must_use]
    pub fn n_circle_sites(&self) -> usize {
        self.circle_sites.len()
    }

    fn build_completion(&self, mask: u64) -> Option<Pag> {
        let mut g = self.base.clone();
        for (i, &(a, b, at_a_circle)) in self.circle_sites.iter().enumerate() {
            let choose_arrow = ((mask >> i) & 1) == 1;
            let new_mark = if choose_arrow { Endpoint::Arrow } else { Endpoint::Tail };
            let edge = g.edge_between(a, b)?;
            let (at_a, at_b) =
                if at_a_circle { (new_mark, edge.at_b) } else { (edge.at_a, new_mark) };
            // Skip illegal directed cycles at insertion time.
            if g.set_marks(a, b, at_a, at_b).is_err() {
                return None;
            }
        }
        if is_mag_completion(&g) { Some(g) } else { None }
    }
}

/// Whether `g` is a legal directed MAG completion: no circles, no Tail–Tail undirected
/// edges, no directed cycles (assumed from construction), and no almost-directed cycles
/// (bidirected edge `a ↔ b` with a directed path either way).
#[must_use]
pub fn is_mag_completion(g: &Pag) -> bool {
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

impl Iterator for CompletionSampler {
    type Item = PagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        let n_sites = self.circle_sites.len();
        let total = if n_sites == 0 { 1u64 } else { 1u64 << n_sites };
        while self.assign < total {
            let mask = self.assign;
            self.assign += 1;
            if let Some(graph) = self.build_completion(mask) {
                let index = self.next_index;
                self.next_index += 1;
                return Some(PagCompletion { graph, index });
            }
            // Invalid MAG — skip without counting against the yield cap.
        }
        None
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
