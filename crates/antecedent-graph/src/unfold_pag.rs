//! Finite stationary unfolding of temporal mixed graphs without changing marks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{DenseNodeId, Endpoint, GraphError, MarkedEdge, NodeRef, Pag, TemporalPag};
use antecedent_core::{TemporalIndexer, TemporalNodeKey};

/// A finite PAG whose dense variable IDs retain their temporal coordinates.
#[derive(Clone, Debug)]
pub struct UnfoldedTemporalPag(std::sync::Arc<UnfoldedTemporalPagInner>);
#[derive(Debug)]
struct UnfoldedTemporalPagInner {
    window: UnfoldedTemporalPagData,
    lease: antecedent_core::execution::AdmissionLease,
}
/// Borrowed fields of an admitted temporal PAG window.
#[derive(Debug)]
pub struct UnfoldedTemporalPagData {
    /// Mixed graph; circles and bidirected or selection edges are preserved.
    pub pag: Pag,
    /// Coordinate map used to build the graph.
    pub indexer: TemporalIndexer,
}

impl std::ops::Deref for UnfoldedTemporalPag {
    type Target = UnfoldedTemporalPagData;
    fn deref(&self) -> &Self::Target {
        &self.0.window
    }
}
impl UnfoldedTemporalPag {
    pub(crate) fn into_parts(self) -> (Pag, antecedent_core::execution::AdmissionLease) {
        let inner = std::sync::Arc::try_unwrap(self.0).expect("fresh unfolded PAG owner");
        (inner.window.pag, inner.lease)
    }
}

impl TemporalPag {
    /// Repeat every template edge at each translation contained in `indexer`.
    ///
    /// This materializes a window, not a certificate that its history is sufficient.
    /// Identification must separately certify the query's boundary closure.
    /// Conflicting stationary copies are errors, rather than order-dependent marks.
    ///
    /// # Errors
    /// Invalid template coordinates, incompatible stationary copies, graph errors,
    /// or a variable outside the supplied indexer schema.
    pub fn unfold(&self, indexer: TemporalIndexer) -> Result<UnfoldedTemporalPag, GraphError> {
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        self.unfold_within(indexer, &mut budget, &ctx)
    }

    /// Unfold using one caller-owned budget and a lease retained with the graph.
    pub fn unfold_within(
        &self,
        indexer: TemporalIndexer,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<UnfoldedTemporalPag, GraphError> {
        use antecedent_core::TemporalMaterializationError as T;
        let limit = |dimension| GraphError::TemporalMaterialization(T::Limit { dimension });
        if ctx.cancellation.is_cancelled() {
            return Err(T::Cancelled.into());
        }
        let nodes = indexer.dense_len() as u64;
        if indexer.history() > (i32::MAX as u32) + 1 || indexer.horizon() > (i32::MAX as u32) + 1 {
            return Err(limit("window coordinates"));
        }
        if nodes > budget.limits().max_graph_nodes {
            return Err(limit("graph nodes"));
        }
        let mut templates = 0u64;
        budget.charge_work(self.nodes().len() as u64)?;
        for _ in self.edges_lazy() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            budget.charge_work(1)?;
            templates = templates.checked_add(1).ok_or_else(|| limit("graph edges"))?;
        }
        let slices = u64::from(indexer.history())
            .checked_add(u64::from(indexer.horizon()))
            .ok_or_else(|| limit("window"))?;
        let edges = templates.checked_mul(slices).ok_or_else(|| limit("graph edges"))?;
        if edges > budget.limits().max_graph_edges {
            return Err(limit("graph edges"));
        }
        let work = nodes
            .checked_add(edges)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_mul(edges.saturating_add(1)))
            .and_then(|v| v.checked_mul(8))
            .ok_or_else(|| limit("work"))?;
        budget.charge_work(work)?;
        let bytes = nodes
            .checked_mul(512)
            .and_then(|n| edges.checked_mul(256).and_then(|e| n.checked_add(e)))
            .and_then(|v| v.checked_add(4096))
            .ok_or_else(|| limit("bytes"))?;
        if bytes > budget.limits().max_bytes {
            return Err(limit("bytes"));
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                GraphError::TemporalMaterialization(T::Cancelled)
            }
            other => GraphError::TemporalMaterialization(T::Admission(other)),
        })?;
        for (i, _) in self.nodes().iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            let (variable, _) = template_key(self, DenseNodeId::try_from_usize(i)?)?;
            if variable.raw() >= indexer.variable_count() {
                return Err(GraphError::InvalidEndpoints {
                    message: "temporal PAG variable outside unfold schema",
                });
            }
        }
        let mut pag = Pag::from_unfolding_admitted(&indexer, ctx)?;
        for edge in self.edges_lazy() {
            if ctx.cancellation.is_cancelled() {
                return Err(T::Cancelled.into());
            }
            let (a_variable, a_offset) = template_key(self, edge.a)?;
            let (b_variable, b_offset) = template_key(self, edge.b)?;
            if a_variable.raw() >= indexer.variable_count()
                || b_variable.raw() >= indexer.variable_count()
            {
                return Err(GraphError::InvalidEndpoints {
                    message: "temporal PAG variable outside unfold schema",
                });
            }
            for shift in crate::unfold::stationary_shifts(&indexer, a_offset, b_offset) {
                if ctx.cancellation.is_cancelled() {
                    return Err(T::Cancelled.into());
                }
                let a = unfolded_node(&indexer, a_variable, a_offset + shift)?;
                let b = unfolded_node(&indexer, b_variable, b_offset + shift)?;
                let copy =
                    MarkedEdge { a, b, at_a: edge.at_a, at_b: edge.at_b, middle: edge.middle };
                if let Some(previous) = pag.edge_between(a, b) {
                    if previous.middle != copy.middle {
                        return Err(GraphError::InvalidEndpoints {
                            message: "incompatible stationary temporal PAG middle marks",
                        });
                    }
                    pag.set_marks(
                        a,
                        b,
                        refine(previous.at_a, copy.at_a)?,
                        refine(previous.at_b, copy.at_b)?,
                    )?;
                } else {
                    pag.insert_marked(copy)?;
                }
            }
        }
        Ok(UnfoldedTemporalPag(std::sync::Arc::new(UnfoldedTemporalPagInner {
            window: UnfoldedTemporalPagData { pag, indexer },
            lease,
        })))
    }
}

fn refine(a: Endpoint, b: Endpoint) -> Result<Endpoint, GraphError> {
    if a == b || b == Endpoint::Circle {
        Ok(a)
    } else if a == Endpoint::Circle {
        Ok(b)
    } else {
        Err(GraphError::InvalidEndpoints {
            message: "incompatible stationary temporal PAG edge copies",
        })
    }
}

fn template_key(
    graph: &TemporalPag,
    node: DenseNodeId,
) -> Result<(antecedent_core::VariableId, i64), GraphError> {
    match graph.nodes().get(node.as_usize()) {
        Some(NodeRef::Lagged { variable, lag }) => Ok((*variable, -i64::from(lag.raw()))),
        _ => Err(GraphError::InvalidEndpoints {
            message: "temporal PAG unfold requires lagged nodes",
        }),
    }
}
fn unfolded_node(
    indexer: &TemporalIndexer,
    variable: antecedent_core::VariableId,
    offset: i64,
) -> Result<DenseNodeId, GraphError> {
    let offset = i32::try_from(offset).map_err(|_| GraphError::TooManyNodes)?;
    let id = indexer.dense_id(TemporalNodeKey { variable, offset }).map_err(|_| {
        GraphError::InvalidEndpoints { message: "temporal PAG unfold endpoint outside window" }
    })?;
    Ok(DenseNodeId::from_raw(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Endpoint, MiddleMark};
    use antecedent_core::{Lag, VariableId};
    fn lagged(graph: &mut TemporalPag, variable: u32, lag: u32) -> DenseNodeId {
        graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
    }
    #[test]
    fn admitted_pag_window_shares_lease_and_refuses_limits() {
        let mut graph = TemporalPag::empty();
        let a = lagged(&mut graph, 0, 1);
        let b = lagged(&mut graph, 0, 0);
        graph.insert_directed(a, b).unwrap();
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 100_000);
        let ctx = antecedent_core::ExecutionContext::for_tests(0).with_admission(pool.clone());
        let indexer = TemporalIndexer::new(1, 2, 1).unwrap();
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let window = graph.unfold_within(indexer.clone(), &mut budget, &ctx).unwrap();
        let held = pool.available().bytes();
        let clone = window.clone();
        drop(window);
        assert_eq!(pool.available().bytes(), held);
        drop(clone);
        assert_eq!(pool.available().bytes(), 100_000);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_graph_nodes = 0;
        let mut budget = antecedent_core::TemporalBudget::new(limits);
        assert!(matches!(
            graph.unfold_within(indexer, &mut budget, &ctx),
            Err(GraphError::TemporalMaterialization(_))
        ));
    }

    #[test]
    fn public_pag_constructor_admits_window_and_clones_share_its_lease() {
        use antecedent_core::{TemporalBudget, TemporalMaterializationError};
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 100_000);
        let ctx = antecedent_core::ExecutionContext::for_tests(0).with_admission(pool.clone());
        let indexer = TemporalIndexer::new(2, 1, 2).unwrap();
        let mut budget = TemporalBudget::new(ctx.temporal_materialization_limits());
        let window = Pag::from_unfolding_within(&indexer, &mut budget, &ctx).unwrap();
        assert_eq!(window.indexer, indexer);
        assert_eq!(window.pag.node_count(), 6);
        for dense in 0..6_u32 {
            let key = indexer.key_of(dense).unwrap();
            assert_eq!(
                window.pag.nodes()[dense as usize],
                NodeRef::Unfolded { variable: key.variable, offset: key.offset }
            );
        }
        assert!(window.pag.neighbors(DenseNodeId::from_raw(0)).next().is_none());
        let held = pool.available().bytes();
        assert!(held < 100_000);
        let clone = window.clone();
        drop(window);
        assert_eq!(pool.available().bytes(), held);
        drop(clone);
        assert_eq!(pool.available().bytes(), 100_000);
        let mut limits = ctx.temporal_materialization_limits();
        limits.max_graph_nodes = 5;
        let mut budget = TemporalBudget::new(limits);
        assert!(matches!(
            Pag::from_unfolding_within(&indexer, &mut budget, &ctx),
            Err(GraphError::TemporalMaterialization(TemporalMaterializationError::Limit {
                dimension: "graph nodes"
            }))
        ));
        let cancelled = antecedent_core::ExecutionContext::for_tests(1);
        cancelled.cancellation.cancel();
        let mut budget = TemporalBudget::new(cancelled.temporal_materialization_limits());
        assert!(matches!(
            Pag::from_unfolding_within(&indexer, &mut budget, &cancelled),
            Err(GraphError::TemporalMaterialization(TemporalMaterializationError::Cancelled))
        ));
        let huge = TemporalIndexer::new(1, i32::MAX as u32, 1).unwrap();
        assert!(matches!(
            Pag::from_unfolding(&huge),
            Err(GraphError::TemporalMaterialization(TemporalMaterializationError::Limit { .. }))
        ));
    }

    #[test]
    fn mixed_marks_replicate_even_when_both_template_nodes_are_lagged() {
        for (at_a, at_b) in [
            (Endpoint::Circle, Endpoint::Arrow),
            (Endpoint::Arrow, Endpoint::Arrow),
            (Endpoint::Tail, Endpoint::Tail),
        ] {
            let mut graph = TemporalPag::empty();
            let a = lagged(&mut graph, 0, 2);
            let b = lagged(&mut graph, 1, 1);
            graph
                .insert_marked(MarkedEdge { a, b, at_a, at_b, middle: MiddleMark::Empty })
                .unwrap();
            let indexer = TemporalIndexer::new(2, 2, 1).unwrap();
            let unfolded = graph.unfold(indexer.clone()).unwrap();
            for offset in [-2, -1] {
                let a = unfolded_node(&indexer, VariableId::from_raw(0), offset).unwrap();
                let b = unfolded_node(&indexer, VariableId::from_raw(1), offset + 1).unwrap();
                let edge = unfolded.pag.edge_between(a, b).unwrap();
                assert_eq!((edge.at_a, edge.at_b), (at_a, at_b));
            }
        }
    }
    #[test]
    fn inconsistent_stationary_templates_are_rejected() {
        let mut graph = TemporalPag::empty();
        let a = lagged(&mut graph, 0, 1);
        let b = lagged(&mut graph, 1, 0);
        let c = lagged(&mut graph, 0, 2);
        let d = lagged(&mut graph, 1, 1);
        graph.insert_directed(a, b).unwrap();
        graph
            .insert_marked(MarkedEdge {
                a: c,
                b: d,
                at_a: Endpoint::Arrow,
                at_b: Endpoint::Arrow,
                middle: MiddleMark::Empty,
            })
            .unwrap();
        assert!(graph.unfold(TemporalIndexer::new(2, 2, 1).unwrap()).is_err());
    }
    #[test]
    fn compatible_circles_are_refined_across_stationary_copies() {
        let mut graph = TemporalPag::empty();
        let a = lagged(&mut graph, 0, 1);
        let b = lagged(&mut graph, 1, 0);
        let c = lagged(&mut graph, 0, 2);
        let d = lagged(&mut graph, 1, 1);
        graph.insert_circle_arrow(a, b).unwrap();
        graph.insert_directed(c, d).unwrap();
        let indexer = TemporalIndexer::new(2, 2, 1).unwrap();
        let unfolded = graph.unfold(indexer.clone()).unwrap();
        let edge = unfolded
            .pag
            .edge_between(
                unfolded_node(&indexer, VariableId::from_raw(0), -1).unwrap(),
                unfolded_node(&indexer, VariableId::from_raw(1), 0).unwrap(),
            )
            .unwrap();
        assert_eq!((edge.at_a, edge.at_b), (Endpoint::Tail, Endpoint::Arrow));
    }

    #[test]
    fn agreeing_stationary_duplicates_are_idempotent() {
        let mut graph = TemporalPag::empty();
        let a = lagged(&mut graph, 0, 1);
        let b = lagged(&mut graph, 1, 0);
        let c = lagged(&mut graph, 0, 2);
        let d = lagged(&mut graph, 1, 1);
        graph.insert_directed(a, b).unwrap();
        graph.insert_directed(c, d).unwrap();
        let indexer = TemporalIndexer::new(2, 2, 1).unwrap();
        let unfolded = graph.unfold(indexer.clone()).unwrap();
        // Both templates are the same edge up to a shift: x[s] -> y[s+1] for s in {-2, -1}, once each.
        let mut endpoints = 0usize;
        for i in 0..unfolded.pag.node_count() {
            endpoints += unfolded.pag.neighbors(DenseNodeId::from_index(i)).count();
        }
        assert_eq!(endpoints, 4, "two edges, two adjacency halves each");
        for offset in [-2, -1] {
            let from = unfolded_node(&indexer, VariableId::from_raw(0), offset).unwrap();
            let to = unfolded_node(&indexer, VariableId::from_raw(1), offset + 1).unwrap();
            let edge = unfolded.pag.edge_between(from, to).unwrap();
            assert_eq!((edge.at_a, edge.at_b), (Endpoint::Tail, Endpoint::Arrow));
        }
    }
}
