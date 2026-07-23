//! Supplied Cpdag / Admg facade paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::prelude::*;
use causal_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Admg, Cpdag, DenseNodeId};

fn tiny_backdoor_table() -> (TabularData, VariableId, VariableId) {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "t",
        ValueType::Binary,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let n = 40;
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        z[i] = (i as f64) * 0.1;
        t[i] = if z[i] > 2.0 { 1.0 } else { 0.0 };
        y[i] = 0.5 * z[i] + 1.5 * t[i];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TabularData::new(storage);
    (data, VariableId::from_raw(1), VariableId::from_raw(2))
}

#[test]
fn supplied_oriented_cpdag_runs_ate() {
    let (data, t, y) = tiny_backdoor_table();
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let result = CausalAnalysis::builder()
        .data(data)
        .cpdag(cpdag)
        .query(AverageEffectQuery::binary_ate(t, y))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    assert!(result.estimate.ate.is_finite());
}

#[test]
fn supplied_admg_without_bidirected_coerces_to_dag() {
    let (data, t, y) = tiny_backdoor_table();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let result = CausalAnalysis::builder()
        .data(data)
        .admg(admg)
        .query(AverageEffectQuery::binary_ate(t, y))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    assert!(result.estimate.ate.is_finite());
}

#[test]
fn cpdag_json_codec_round_trip_via_facade() {
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let s = antecedent::io::cpdag_to_json(&cpdag, Some(&["a".into(), "b".into()])).unwrap();
    let back = antecedent::io::cpdag_from_json(&s).unwrap();
    assert_eq!(back.undirected_edge_count(), 1);
}
