//! Streamed [`TemporalPag`] → [`TemporalDag`] completions.
//!
//! Circle-arrow edges are compelled to Tail→Arrow. Circle-circle edges are oriented both ways. Tail-Tail
//! edges denote selection structure and cannot be refined to DAG edges. Bidirected or conflict marks refuse the sampler:
//! those graphs cannot complete to a [`TemporalDag`] and stay refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::GraphError;
use crate::temporal::TemporalDag;
use crate::temporal_pag::TemporalPag;
use crate::types::{DenseNodeId, Endpoint};

/// One [`TemporalDag`] completion of a [`TemporalPag`].
#[derive(Clone, Debug)]
pub struct TemporalPagCompletion {
    /// Completed temporal DAG.
    pub graph: TemporalDag,
    /// Index among valid yields.
    pub index: usize,
}

/// Streams [`TemporalPag`] → [`TemporalDag`] completions with a hard cap.
#[derive(Clone, Debug)]
pub struct TemporalPagCompletionSampler {
    base: TemporalPag,
    /// Two-way edges `(a, b)` with `a.raw() <= b.raw()`.
    orientable: Vec<(DenseNodeId, DenseNodeId)>,
    /// Circle-arrow edges forced to `from -> to`.
    forced: Vec<(DenseNodeId, DenseNodeId)>,
    max_completions: usize,
    next_index: usize,
    assign: u64,
}

impl TemporalPagCompletionSampler {
    /// Build a sampler that yields at most `max_completions` [`TemporalDag`]s.
    ///
    /// # Errors
    ///
    /// Bidirected or conflict marks, or more than 63 orientable edges.
    pub fn new(pag: TemporalPag, max_completions: usize) -> Result<Self, GraphError> {
        let mut orientable = Vec::new();
        let mut forced = Vec::new();
        for e in pag.edges() {
            if e.is_conflict() {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalPagCompletionSampler refuses conflict (x-x) edges",
                });
            }
            if e.at_a == Endpoint::Arrow && e.at_b == Endpoint::Arrow {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalPagCompletionSampler refuses bidirected edges; \
                              those graphs cannot complete to a TemporalDag",
                });
            }
            if let Some((from, to)) = e.parent_child() {
                forced.push((from, to));
                continue;
            }
            if e.is_undirected() {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalPagCompletionSampler refuses tail-tail selection edges",
                });
            }
            if e.at_a == Endpoint::Circle && e.at_b == Endpoint::Circle {
                let (a, b) = if e.a.raw() <= e.b.raw() { (e.a, e.b) } else { (e.b, e.a) };
                orientable.push((a, b));
                continue;
            }
            if e.at_a == Endpoint::Circle && e.at_b == Endpoint::Arrow {
                forced.push((e.a, e.b));
            } else if e.at_a == Endpoint::Arrow && e.at_b == Endpoint::Circle {
                forced.push((e.b, e.a));
            } else {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalPagCompletionSampler cannot orient this mark combination",
                });
            }
        }
        orientable.sort_by_key(|(a, b)| (a.raw(), b.raw()));
        orientable.dedup();
        if orientable.len() > 63 {
            return Err(GraphError::InvalidEndpoints {
                message: "too many orientable edges for TemporalPagCompletionSampler mask",
            });
        }
        Ok(Self { base: pag, orientable, forced, max_completions, next_index: 0, assign: 0 })
    }

    /// Hard cap on yielded valid completions.
    #[must_use]
    pub fn max_completions(&self) -> usize {
        self.max_completions
    }

    /// Whether the retention cap stopped the stream before every mask was examined.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.next_index >= self.max_completions && self.assign < self.total_masks()
    }

    fn total_masks(&self) -> u64 {
        let n = self.orientable.len();
        if n == 0 { 1 } else { 1u64 << n }
    }

    fn build_completion(&self, mask: u64) -> Option<TemporalDag> {
        let mut g = self.base.clone();
        for &(from, to) in &self.forced {
            if g.set_marks(from, to, Endpoint::Tail, Endpoint::Arrow).is_err() {
                return None;
            }
        }
        for (i, &(a, b)) in self.orientable.iter().enumerate() {
            let reverse = ((mask >> i) & 1) == 1;
            let (from, to) = if reverse { (b, a) } else { (a, b) };
            if g.set_marks(from, to, Endpoint::Tail, Endpoint::Arrow).is_err() {
                return None;
            }
        }
        g.try_into_temporal_dag().ok()
    }
}

impl Iterator for TemporalPagCompletionSampler {
    type Item = TemporalPagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        let total = self.total_masks();
        while self.assign < total {
            let mask = self.assign;
            self.assign += 1;
            if let Some(graph) = self.build_completion(mask) {
                let index = self.next_index;
                self.next_index += 1;
                return Some(TemporalPagCompletion { graph, index });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{Lag, VariableId};

    use super::*;

    fn lagged(g: &mut TemporalPag, var: u32, lag: u32) -> DenseNodeId {
        g.add_lagged(VariableId::from_raw(var), Lag::from_raw(lag)).unwrap()
    }

    #[test]
    fn circle_arrow_chain_yields_one() {
        let mut g = TemporalPag::empty();
        let a = lagged(&mut g, 0, 1);
        let b = lagged(&mut g, 1, 0);
        g.insert_circle_arrow(a, b).unwrap();
        let collected: Vec<_> = TemporalPagCompletionSampler::new(g, 8).unwrap().collect();
        assert_eq!(collected.len(), 1);
        assert!(collected[0].graph.children(a).contains(&b));
    }

    #[test]
    fn circle_circle_confounder_has_two_dags() {
        let mut g = TemporalPag::empty();
        let z = lagged(&mut g, 0, 1);
        let t = lagged(&mut g, 1, 1);
        let y = lagged(&mut g, 2, 0);
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_circle_circle_with_middle(z, t, crate::types::MiddleMark::Empty).unwrap();
        let collected: Vec<_> = TemporalPagCompletionSampler::new(g, 8).unwrap().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn selection_tails_are_not_replaced_with_arrowheads() {
        let mut g = TemporalPag::empty();
        let a = lagged(&mut g, 0, 0);
        let b = lagged(&mut g, 1, 0);
        g.insert_marked(crate::types::MarkedEdge {
            a,
            b,
            at_a: Endpoint::Tail,
            at_b: Endpoint::Tail,
            middle: crate::types::MiddleMark::Empty,
        })
        .unwrap();
        assert!(TemporalPagCompletionSampler::new(g, 4).is_err());
    }

    #[test]
    fn refuses_bidirected() {
        let mut g = TemporalPag::empty();
        let a = lagged(&mut g, 0, 1);
        let b = lagged(&mut g, 1, 0);
        g.insert_marked(crate::types::MarkedEdge {
            a,
            b,
            at_a: Endpoint::Arrow,
            at_b: Endpoint::Arrow,
            middle: crate::types::MiddleMark::Empty,
        })
        .unwrap();
        assert!(TemporalPagCompletionSampler::new(g, 4).is_err());
    }
}
