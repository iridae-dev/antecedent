//! Finite stationary unfolding of temporal mixed graphs without changing marks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{DenseNodeId, Endpoint, GraphError, MarkedEdge, NodeRef, Pag, TemporalPag};
use antecedent_core::{TemporalIndexer, TemporalNodeKey};

/// A finite PAG whose dense variable IDs retain their temporal coordinates.
#[derive(Clone, Debug)]
pub struct UnfoldedTemporalPag {
    /// Mixed graph; circles and bidirected or selection edges are preserved.
    pub pag: Pag,
    /// Coordinate map used to build the graph.
    pub indexer: TemporalIndexer,
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
        for (i, _) in self.nodes().iter().enumerate() {
            let (variable, _) = template_key(self, DenseNodeId::try_from_usize(i)?)?;
            if variable.raw() >= indexer.variable_count() {
                return Err(GraphError::InvalidEndpoints {
                    message: "temporal PAG variable outside unfold schema",
                });
            }
        }
        let count = u32::try_from(indexer.dense_len()).map_err(|_| GraphError::TooManyNodes)?;
        let mut pag = Pag::with_variables(count);
        let lower = -i64::from(indexer.history());
        let upper = i64::from(indexer.horizon()) - 1;
        for edge in self.edges() {
            let (a_variable, a_offset) = template_key(self, edge.a)?;
            let (b_variable, b_offset) = template_key(self, edge.b)?;
            if a_variable.raw() >= indexer.variable_count()
                || b_variable.raw() >= indexer.variable_count()
            {
                return Err(GraphError::InvalidEndpoints {
                    message: "temporal PAG variable outside unfold schema",
                });
            }
            for shift in (lower - a_offset.min(b_offset))..=(upper - a_offset.max(b_offset)) {
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
        Ok(UnfoldedTemporalPag { pag, indexer })
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
        assert!(graph.unfold(TemporalIndexer::new(2, 2, 1).unwrap()).is_ok());
    }
}
