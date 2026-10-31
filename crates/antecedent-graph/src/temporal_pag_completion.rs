//! Temporal PAG endpoint refinements retaining directed and bidirected MAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::completion::default_completion_search;
use crate::{CompletionSampler, CompletionValidationReport, GraphError, NodeRef, Pag, TemporalPag};
use antecedent_core::{ExecutionContext, SearchLimits, SearchReceipt, SearchStop, TemporalIndexer};

/// One definite mixed-graph refinement of a temporal PAG.
#[derive(Clone, Debug)]
pub struct TemporalPagCompletion {
    /// Temporal MAG template, preserving the source's lag coordinates.
    pub graph: TemporalPag,
    /// Index among retained, valid yields.
    pub index: usize,
}

/// Bounded temporal MAG completions, constrained before retention and class audit.
#[derive(Debug)]
pub struct TemporalPagCompletionSampler {
    _template_lease: antecedent_core::execution::AdmissionLease,
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
    /// Conflict/selection marks or invalid graph coordinates.
    pub fn new(pag: TemporalPag, max_completions: usize) -> Result<Self, GraphError> {
        let (limits, ctx) = default_completion_search();
        Self::with_budget(pag, max_completions, limits, &ctx)
    }

    /// Enumerate the same default stationary window under the caller's search budget.
    ///
    /// # Errors
    /// Conflict/selection marks or invalid graph coordinates.
    pub fn with_budget(
        pag: TemporalPag,
        max_completions: usize,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        Self::with_temporal_budget(pag, max_completions, limits, &mut budget, ctx)
    }

    /// Quote a borrowed template copy before constructing its sampler.
    pub fn with_borrowed_temporal_budget(
        pag: &TemporalPag,
        max_completions: usize,
        limits: SearchLimits,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        let _copy = admit_template_copy(pag, budget, ctx)?;
        Self::with_temporal_budget(pag.clone(), max_completions, limits, budget, ctx)
    }

    /// Enumerate under one cumulative temporal materialization budget.
    pub fn with_temporal_budget(
        pag: TemporalPag,
        max_completions: usize,
        limits: SearchLimits,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
        }
        budget.charge_work(pag.nodes().len() as u64)?;
        let mut variables = 0;
        let mut history = 0;
        for node in pag.nodes() {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            if let NodeRef::Lagged { variable, lag } = node {
                variables =
                    variables.max(variable.raw().checked_add(1).ok_or(GraphError::TooManyNodes)?);
                history = history.max(lag.raw());
            }
        }
        let indexer =
            TemporalIndexer::new(variables, history, 1).map_err(|_| GraphError::TooManyNodes)?;
        Self::for_window_with_temporal_budget(pag, max_completions, &indexer, limits, budget, ctx)
    }
    /// Enumerate templates whose stationary copies are valid MAGs in a supplied window.
    ///
    /// The walk is bounded by [`crate::DEFAULT_COMPLETION_OPERATIONS`]; see
    /// [`Self::for_window_with_budget`].
    ///
    /// # Errors
    /// Invalid source marks or coordinates.
    pub fn for_window(
        pag: TemporalPag,
        max_completions: usize,
        indexer: &TemporalIndexer,
    ) -> Result<Self, GraphError> {
        let (limits, ctx) = default_completion_search();
        Self::for_window_with_budget(pag, max_completions, indexer, limits, &ctx)
    }
    /// [`Self::for_window`] under an explicit search budget; a stop is reported by
    /// [`Self::search_stop`] and makes [`Self::class_audit_incomplete`] true.
    ///
    /// # Errors
    /// Invalid source marks or coordinates.
    pub fn for_window_with_budget(
        pag: TemporalPag,
        max_completions: usize,
        indexer: &TemporalIndexer,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        Self::for_window_with_temporal_budget(
            pag,
            max_completions,
            indexer,
            limits,
            &mut budget,
            ctx,
        )
    }

    /// Admit the borrowed source copy before enumerating this window.
    pub fn for_window_with_borrowed_temporal_budget(
        pag: &TemporalPag,
        max_completions: usize,
        indexer: &TemporalIndexer,
        limits: SearchLimits,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        let _copy = admit_template_copy(pag, budget, ctx)?;
        Self::for_window_with_temporal_budget(
            pag.clone(),
            max_completions,
            indexer,
            limits,
            budget,
            ctx,
        )
    }

    /// Enumerate while charging each projected window to the shared temporal budget.
    pub fn for_window_with_temporal_budget(
        pag: TemporalPag,
        max_completions: usize,
        indexer: &TemporalIndexer,
        limits: SearchLimits,
        temporal_budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        use antecedent_core::TemporalMaterializationError as T;
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        temporal_budget.charge_work(pag.nodes().len() as u64)?;
        let mut edges = 0u64;
        for edge in pag.edges_lazy() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            temporal_budget.charge_work(1)?;
            edges = edges.checked_add(1).ok_or(T::Limit { dimension: "graph edges" })?;
            if edge.is_conflict() || edge.is_undirected() {
                return Err(GraphError::InvalidEndpoints {
                    message: "temporal MAG completion requires the non-conflict directed/bidirected family",
                });
            }
        }
        let bytes = (pag.nodes().len() as u64)
            .checked_mul(512)
            .and_then(|n| edges.checked_mul(512).and_then(|e| n.checked_add(e)))
            .and_then(|n| n.checked_add(4096))
            .ok_or(T::Limit { dimension: "bytes" })?;
        if bytes > temporal_budget.limits().max_bytes {
            return Err(T::Limit { dimension: "bytes" }.into());
        }
        let template_lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled,
            other => T::Admission(other),
        })?;
        let projected = pag.as_static_pag_for_alg()?;
        let projection_budget = std::cell::RefCell::new(temporal_budget.clone());
        let projection_error = std::cell::RefCell::new(None);
        // The first lease covers audit_source; later leases cover the current candidate.
        let projection_leases = std::cell::RefCell::new(Vec::new());
        let inner_result = CompletionSampler::with_projection(
            &projected,
            max_completions,
            limits,
            ctx,
            |candidate| {
                if projection_error.borrow().is_some() {
                    return None;
                }
                projection_leases.borrow_mut().truncate(1);
                if let Err(error) = projection_budget
                    .borrow_mut()
                    .charge_work((pag.nodes().len() as u64).saturating_add(edges).saturating_mul(8))
                {
                    *projection_error.borrow_mut() =
                        Some(GraphError::TemporalMaterialization(error));
                    return None;
                }
                let graph = refine(&pag, candidate)?;
                match graph.unfold_within(indexer.clone(), &mut projection_budget.borrow_mut(), ctx)
                {
                    Ok(window) => {
                        let (graph, lease) = window.into_parts();
                        projection_leases.borrow_mut().push(lease);
                        Some(graph)
                    }
                    Err(error @ GraphError::TemporalMaterialization(_)) => {
                        *projection_error.borrow_mut() = Some(error);
                        None
                    }
                    Err(_) => None,
                }
            },
        );
        if let Some(error) = projection_error.into_inner() {
            return Err(error);
        }
        let inner = inner_result?;
        Ok(Self { base: pag, inner, max_completions, _template_lease: template_lease })
    }
    /// Maximum retained valid templates.
    #[must_use]
    pub const fn max_completions(&self) -> usize {
        self.max_completions
    }
    /// Whether valid templates were omitted by the retention bound, or a search bound
    /// stopped the walk before every template was classified.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.inner.hit_cap()
    }
    /// Whether the yielded templates are unproved as one Markov class: the audit was skipped
    /// as too expensive or a search bound stopped the walk.
    #[must_use]
    pub const fn class_audit_incomplete(&self) -> bool {
        self.inner.class_audit_incomplete()
    }
    /// The bound that stopped the walk, if any.
    #[must_use]
    pub const fn search_stop(&self) -> Option<SearchStop> {
        self.inner.search_stop()
    }
    /// The limits in force and the work consumed, once a bound has stopped the walk.
    #[must_use]
    pub const fn search_receipt(&self) -> Option<&SearchReceipt> {
        self.inner.search_receipt()
    }
    /// Full template assignment and equivalence audit, including constraint rejects.
    #[must_use]
    pub const fn validation_report(&self) -> CompletionValidationReport {
        self.inner.validation_report()
    }
}
fn admit_template_copy(
    pag: &TemporalPag,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<antecedent_core::execution::AdmissionLease, GraphError> {
    use antecedent_core::TemporalMaterializationError as T;
    if ctx.cancellation.is_cancelled() {
        return Err(T::Cancelled.into());
    }
    budget.charge_work(pag.nodes().len() as u64)?;
    let mut edges = 0u64;
    for _ in pag.edges_lazy() {
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        budget.charge_work(1)?;
        edges = edges.checked_add(1).ok_or(T::Limit { dimension: "graph edges" })?;
    }
    let bytes = (pag.nodes().len() as u64)
        .checked_mul(256)
        .and_then(|v| edges.checked_mul(256).and_then(|e| v.checked_add(e)))
        .and_then(|v| v.checked_add(4096))
        .ok_or(T::Limit { dimension: "bytes" })?;
    if bytes > budget.limits().max_bytes {
        return Err(T::Limit { dimension: "bytes" }.into());
    }
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => T::Cancelled.into(),
        other => T::Admission(other).into(),
    })
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
    #[test]
    fn default_window_budgeted_constructor_preserves_members_and_honors_context() {
        let (mut graph, past, present) = pair();
        graph.insert_circle_arrow(past, present).unwrap();
        let expected: Vec<_> = TemporalPagCompletionSampler::new(graph.clone(), 32)
            .unwrap()
            .map(|completion| completion.graph)
            .collect();
        let (limits, _) = default_completion_search();
        let ctx = ExecutionContext::for_tests(87);
        let actual: Vec<_> =
            TemporalPagCompletionSampler::with_budget(graph.clone(), 32, limits, &ctx)
                .unwrap()
                .map(|completion| completion.graph)
                .collect();
        assert_eq!(expected, actual);
        let cancelled = ExecutionContext::for_tests(87);
        cancelled.cancellation.cancel();
        assert!(matches!(
            TemporalPagCompletionSampler::with_budget(graph.clone(), 32, limits, &cancelled),
            Err(GraphError::TemporalMaterialization(
                antecedent_core::TemporalMaterializationError::Cancelled
            ))
        ));
        let limited = ExecutionContext::for_tests(87).with_memory_limit(Some(1));
        assert!(matches!(
            TemporalPagCompletionSampler::with_budget(graph, 32, limits, &limited),
            Err(GraphError::TemporalMaterialization(
                antecedent_core::TemporalMaterializationError::Admission(
                    antecedent_core::execution::AdmissionError::ExceedsRemainder { .. }
                )
            ))
        ));
    }
}
