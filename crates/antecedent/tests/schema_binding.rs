//! Schema-binding validation: an `AcceptedGraph`'s node indices must describe the
//! data a `Study` is built against.
//!
//! Static graph nodes are positional (`DenseNodeId(i)` is `VariableId(i)`) with no
//! stored record of which schema those indices meant. `StudyBuilder::build` refuses
//! a graph whose node count does not match the data's variable count, and — when the
//! graph was bound via `AcceptedGraph::with_schema` — refuses one whose bound names
//! do not match the data's, in order.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names, clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent::prelude::*;
use antecedent_core::{Lag, TemporalPolicy};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex, ValidityBitmap,
};

/// Build a continuous schema with the given names, in order.
fn schema_with_names(names: &[&str]) -> CausalSchema {
    let mut b = CausalSchemaBuilder::new();
    for &name in names {
        b = b.continuous(name).context();
    }
    b.build().unwrap()
}

/// A tiny tabular dataset carrying `schema`, with one dummy continuous column per
/// variable (20 rows). Consumes a clone of `schema` into storage; the caller keeps
/// the original for binding / assertions.
fn tabular_data_for(schema: &CausalSchema) -> TabularData {
    let n = 20;
    let cols: Vec<OwnedColumn> = schema
        .variables()
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let id = VariableId::from_raw(i as u32);
            let values: Vec<f64> = (0..n).map(|r| r as f64 + i as f64 * 0.01).collect();
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema.clone(), cols, None, None).unwrap();
    TabularData::new(storage)
}

#[test]
fn node_count_below_schema_length_is_refused() {
    let schema = schema_with_names(&["z", "t", "y"]);
    let data = tabular_data_for(&schema);
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

    let err = Study::tabular(data)
        .graph(g)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .build()
        .unwrap_err();

    match err {
        CausalError::SchemaMismatch { detail } => {
            assert_eq!(
                detail,
                "graph has 2 nodes but data has 3 variables; the \
                 structure does not describe this table"
            );
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[test]
fn node_count_above_schema_length_is_refused() {
    let schema = schema_with_names(&["t", "y"]);
    let data = tabular_data_for(&schema);
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();

    let err = Study::tabular(data)
        .graph(g)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .build()
        .unwrap_err();

    match err {
        CausalError::SchemaMismatch { detail } => {
            assert_eq!(
                detail,
                "graph has 3 nodes but data has 2 variables; the \
                 structure does not describe this table"
            );
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[test]
fn matching_node_count_without_binding_builds() {
    let schema = schema_with_names(&["z", "t", "y"]);
    let data = tabular_data_for(&schema);
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();

    // Binding is opt-in: an unbound graph with the right shape must not be refused.
    let built = Study::tabular(data)
        .graph(g)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2)))
        .build();

    assert!(built.is_ok(), "expected Ok, got {:?}", built.err());
}

#[test]
fn bound_graph_from_a_different_schema_is_refused_even_with_matching_shape() {
    // schema A: z, t, y — the structure below is reviewed and bound against this.
    let schema_a = schema_with_names(&["z", "t", "y"]);
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let bound = AcceptedGraph::dag(g).with_schema(&schema_a);

    // schema B: z, revenue, region — same node count (3), first divergent name at
    // index 1 ("t" vs "revenue"). Same shape, different meaning: the dangerous case.
    let schema_b = schema_with_names(&["z", "revenue", "region"]);
    let data_b = tabular_data_for(&schema_b);

    let err = Study::tabular(data_b)
        .graph(bound)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2)))
        .build()
        .unwrap_err();

    match err {
        CausalError::SchemaMismatch { detail } => {
            assert_eq!(
                detail,
                "graph is bound to variable 1 `t` but data has `revenue` at \
                 that position; the structure was built against a different schema"
            );
        }
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }
}

#[test]
fn bound_to_the_matching_schema_builds() {
    let schema = schema_with_names(&["z", "t", "y"]);
    let data = tabular_data_for(&schema);
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let bound = AcceptedGraph::dag(g).with_schema(&schema);

    let built = Study::tabular(data)
        .graph(bound)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2)))
        .build();

    assert!(built.is_ok(), "expected Ok, got {:?}", built.err());
}

#[test]
fn temporal_study_is_exempt_from_the_node_count_check() {
    let schema = schema_with_names(&["pressure", "defect"]);
    let n_vars = schema.len();
    let n = 20;
    let cols: Vec<OwnedColumn> = schema
        .variables()
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let id = VariableId::from_raw(i as u32);
            let values: Vec<f64> = (0..n).map(|r| r as f64 + i as f64 * 0.01).collect();
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 3_600_000_000_000 },
            length: n,
        },
    )
    .unwrap();

    // Three lagged nodes over two variables: node_count() != the variable count,
    // which is exactly why temporal classes are exempt from the shape check (their
    // nodes are (variable, lag) pairs, not one node per variable).
    let mut g = TemporalDag::empty();
    let pressure_lag1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let pressure_lag2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let defect_now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(pressure_lag1, defect_now).unwrap();
    g.insert_directed(pressure_lag2, defect_now).unwrap();
    assert_ne!(g.node_count(), n_vars);

    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);

    let built = Study::series(series).graph(g).temporal_query(query).build();

    assert!(built.is_ok(), "expected Ok, got {:?}", built.err());
}
