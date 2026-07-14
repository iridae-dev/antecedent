//! Counterfactual batch / streaming equivalence bench (Phase 7 exit).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint, SmallRoleSet,
    Value, ValueType, VariableId,
};
use causal_counterfactual::{CounterfactualEngine, CounterfactualWorld, streaming_matches_retained};
use causal_data::column::{Float64Column, ValidityBitmap};
use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use causal_graph::{Dag, DenseNodeId};
use causal_model::{
    CompiledCausalModel, MechanismRegistry, MechanismWorkspace, SelectionPolicy,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn engine() -> (CounterfactualEngine, TabularData) {
    let n = 100usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
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
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let y: Vec<f64> = t.iter().enumerate().map(|(i, ti)| 2.0 * ti + 0.01 * i as f64).collect();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let compiled = CompiledCausalModel::compile(g).unwrap();
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
        .unwrap();
    (CounterfactualEngine::new(compiled.with_mechanisms(store)), data)
}

fn bench_cf(c: &mut Criterion) {
    let (eng, data) = engine();
    let exo = eng.abduct(&data, false).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    c.bench_function("counterfactual_predict_n100", |b| {
        b.iter(|| {
            let mut ws = MechanismWorkspace::default();
            let worlds = [CounterfactualWorld {
                unit_rows: None,
                interventions: Arc::from([Intervention::set(
                    VariableId::from_raw(0),
                    Value::f64(1.0),
                )]),
            }];
            let res = eng
                .predict(&exo, &worlds, &[VariableId::from_raw(1)], false, &mut ws, &ctx)
                .unwrap();
            assert!(streaming_matches_retained(&res, 0, DenseNodeId::from_raw(1)));
            res
        });
    });
}

criterion_group!(benches, bench_cf);
criterion_main!(benches);
