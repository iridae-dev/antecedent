//! Streamed / bounded PAG completion sampling (DESIGN.md §6.5 / Phase 8 exit).
//!
//! Completions are never retained without bound: the sampler yields at most
//! `max_completions` MAGs (circle-free ancestral graphs).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::GraphError;
use crate::pag::Pag;
use crate::types::{DenseNodeId, Endpoint, MarkedEdge};

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
    circle_sites: Vec<(DenseNodeId, DenseNodeId, bool)>, // (a,b, at_a_is_circle) — one site per circle endpoint
    max_completions: usize,
    next_index: usize,
    /// Bitmask assignment for circle sites (site i uses bit i); advances as a counter.
    assign: u64,
}

impl CompletionSampler {
    /// Build a sampler that yields at most `max_completions` completions.
    ///
    /// # Errors
    ///
    /// More than 63 circle endpoints (mask capacity).
    pub fn new(pag: Pag, max_completions: usize) -> Result<Self, GraphError> {
        let mut sites = Vec::new();
        let n = pag.node_count();
        for i in 0..n {
            let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
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

    /// Hard cap.
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
            // Skip illegal directed cycles.
            if g.set_marks(a, b, at_a, at_b).is_err() {
                return None;
            }
        }
        // Verify no circles remain.
        for i in 0..g.node_count() {
            let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            for (b, at_a, at_b) in g.neighbors(a) {
                if b.raw() < a.raw() {
                    continue;
                }
                if matches!(at_a, Endpoint::Circle) || matches!(at_b, Endpoint::Circle) {
                    return None;
                }
                let _ = MarkedEdge { a, b, at_a, at_b };
            }
        }
        Some(g)
    }
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
            // Invalid orientation — try next mask without counting against...
            // Actually invalid ones shouldn't count toward max? Plan says max completions
            // yielded. Skip invalid without incrementing next_index — but still need to
            // avoid infinite loop. assign already advanced.
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
            let e =
                c.graph.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            assert!(!matches!(e.at_a, Endpoint::Circle));
            assert!(!matches!(e.at_b, Endpoint::Circle));
        }
    }

    #[test]
    fn no_circle_yields_single_completion() {
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let collected: Vec<_> = CompletionSampler::new(pag, 10).unwrap().collect();
        assert_eq!(collected.len(), 1);
    }
}
