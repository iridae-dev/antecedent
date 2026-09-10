//! Temporal PAG endpoint refinements retaining directed and bidirected MAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{CompletionSampler, CompletionValidationReport, GraphError, NodeRef, Pag, TemporalPag};
use antecedent_core::TemporalIndexer;

/// One definite mixed-graph refinement of a temporal PAG.
#[derive(Clone, Debug)]
pub struct TemporalPagCompletion {
    /// Temporal MAG template, preserving the source's lag coordinates.
    pub graph: TemporalPag,
    /// Index among retained, valid yields.
    pub index: usize,
}

/// Bounded temporal MAG completions, constrained before retention and class audit.
#[derive(Clone, Debug)]
pub struct TemporalPagCompletionSampler {
    base: TemporalPag,
    inner: CompletionSampler,
    max_completions: usize,
}
impl TemporalPagCompletionSampler {
    /// Enumerate directed/bidirected completions under temporal and stationary constraints.
    ///
    /// The MAG validity and equivalence audit applies to the stationary unfolding.
    /// Identification must additionally certify and audit its unfolded query window.
    /// Selection-variable tail-tail edges remain outside the directed/bidirected family.
    ///
    /// # Errors
    /// Conflict/selection marks, excessive circle sites, or invalid graph coordinates.
    pub fn new(pag: TemporalPag, max_completions: usize) -> Result<Self, GraphError> {
        if pag.edges().iter().any(|e| e.is_conflict() || e.is_undirected()) {
            return Err(GraphError::InvalidEndpoints {
                message: "temporal MAG completion requires non-conflict directed/bidirected-family marks; selection edges are unsupported",
            });
        }
        let mut variables = 0;
        let mut history = 0;
        for node in pag.nodes() {
            if let NodeRef::Lagged { variable, lag } = node {
                variables =
                    variables.max(variable.raw().checked_add(1).ok_or(GraphError::TooManyNodes)?);
                history = history.max(lag.raw());
            }
        }
        let indexer =
            TemporalIndexer::new(variables, history, 1).map_err(|_| GraphError::TooManyNodes)?;
        Self::for_window(pag, max_completions, &indexer)
    }
    /// Enumerate templates whose stationary copies are valid MAGs in a supplied window.
    ///
    /// # Errors
    /// Invalid source marks, coordinates, or excessive endpoint-assignment space.
    pub fn for_window(
        pag: TemporalPag,
        max_completions: usize,
        indexer: &TemporalIndexer,
    ) -> Result<Self, GraphError> {
        if pag.edges().iter().any(|e| e.is_conflict() || e.is_undirected()) {
            return Err(GraphError::InvalidEndpoints {
                message: "temporal MAG completion requires the non-conflict directed/bidirected family",
            });
        }
        let inner = CompletionSampler::with_projection(
            &pag.as_static_pag_for_alg(),
            max_completions,
            |candidate| {
                refine(&pag, candidate)
                    .and_then(|graph| graph.unfold(indexer.clone()).ok())
                    .map(|unfolded| unfolded.pag)
            },
        )?;
        Ok(Self { base: pag, inner, max_completions })
    }
    /// Maximum retained valid templates.
    #[must_use]
    pub const fn max_completions(&self) -> usize {
        self.max_completions
    }
    /// Whether valid templates were omitted by the retention bound.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.inner.hit_cap()
    }
    /// Full template assignment and equivalence audit, including constraint rejects.
    #[must_use]
    pub const fn validation_report(&self) -> CompletionValidationReport {
        self.inner.validation_report()
    }
}
fn refine(base: &TemporalPag, candidate: &Pag) -> Option<TemporalPag> {
    let mut graph = base.clone();
    for edge in base.edges() {
        let completed = candidate.edge_between(edge.a, edge.b)?;
        graph.set_marks(edge.a, edge.b, completed.at_a, completed.at_b).ok()?;
    }
    Some(graph)
}
impl Iterator for TemporalPagCompletionSampler {
    type Item = TemporalPagCompletion;
    fn next(&mut self) -> Option<Self::Item> {
        let completion = self.inner.next()?;
        Some(TemporalPagCompletion {
            graph: refine(&self.base, &completion.graph)
                .expect("constraints checked before retention"),
            index: completion.index,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark};
    use antecedent_core::{Lag, VariableId};
    fn pair() -> (TemporalPag, DenseNodeId, DenseNodeId) {
        let mut graph = TemporalPag::empty();
        let a = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = graph.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        (graph, a, b)
    }
    #[test]
    fn future_circle_arrow_retains_only_latent_completion() {
        let (mut graph, past, present) = pair();
        graph.insert_circle_arrow(present, past).unwrap();
        let mut sampler = TemporalPagCompletionSampler::new(graph, 1).unwrap();
        let completion = sampler.next().expect("valid latent completion");
        let edge = completion.graph.edge_between(present, past).unwrap();
        assert_eq!((edge.at_a, edge.at_b), (Endpoint::Arrow, Endpoint::Arrow));
        assert!(sampler.next().is_none());
        assert!(!sampler.hit_cap());
    }
    #[test]
    fn circle_arrow_retains_latent_completion() {
        let (mut graph, a, b) = pair();
        graph.insert_circle_arrow(a, b).unwrap();
        let completions: Vec<_> = TemporalPagCompletionSampler::new(graph, 8).unwrap().collect();
        assert_eq!(completions.len(), 2);
        assert!(
            completions.iter().any(|c| c.graph.edge_between(a, b).unwrap().at_a == Endpoint::Arrow)
        );
    }
    #[test]
    fn bidirected_template_is_retained() {
        let (mut graph, a, b) = pair();
        graph
            .insert_marked(MarkedEdge {
                a,
                b,
                at_a: Endpoint::Arrow,
                at_b: Endpoint::Arrow,
                middle: MiddleMark::Empty,
            })
            .unwrap();
        assert_eq!(TemporalPagCompletionSampler::new(graph, 8).unwrap().count(), 1);
    }
    #[test]
    fn future_to_past_refinements_do_not_consume_retention_cap() {
        let (mut graph, a, b) = pair();
        graph.insert_circle_circle_with_middle(a, b, MiddleMark::Empty).unwrap();
        let mut sampler = TemporalPagCompletionSampler::new(graph, 1).unwrap();
        assert!(sampler.validation_report().rejected_constraints > 0);
        assert!(sampler.next().is_some());
        assert!(sampler.hit_cap());
    }
}
