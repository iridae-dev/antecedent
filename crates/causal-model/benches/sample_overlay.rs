//! Batch interventional sampling benchmark (Phase 7 exit criterion).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::{
    CausalRng, CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
    SmallRoleSet, Value, ValueType, VariableId,
};
use causal_data::column::{Float64Column, ValidityBitmap};
use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use causal_graph::{Dag, DenseNodeId};
use causal_model::{
    CompiledCausalModel, MechanismRegistry, MechanismWorkspace, SelectionPolicy,
    sample_interventional,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn fitted_model() -> CompiledCausalModel {
    let n = 200usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
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
    let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
    let yv: Vec<f64> = xv.iter().map(|x| 0.5 + 1.5 * x).collect();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let compiled = CompiledCausalModel::compile(g).unwrap();
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
        .unwrap();
    compiled.with_mechanisms(store)
}

fn bench_interventional(c: &mut Criterion) {
    let model = fitted_model();
    let ctx = ExecutionContext::for_tests(1);
    c.bench_function("sample_interventional_n1000_overlay", |b| {
        b.iter(|| {
            let mut rng = CausalRng::from_seed(7);
            let mut ws = MechanismWorkspace::default();
            sample_interventional(
                &model,
                &[Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
                1000,
                &mut rng,
                &mut ws,
                &ctx,
            )
            .unwrap()
        });
    });
}

criterion_group!(benches, bench_interventional);
criterion_main!(benches);
