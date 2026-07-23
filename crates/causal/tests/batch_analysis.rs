//! conformance: batch multi-query shares one table (BACKLOG E).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use causal::{BatchAnalysis, CausalAnalysis, RefuteSuite};
use causal_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId};
use causal_kernels::standard_normal;

fn two_treatment_scm(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xB07C_u64);
    let mut z = vec![0.0; n];
    let mut t1 = vec![0.0; n];
    let mut t2 = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let p1 = 1.0 / (1.0 + (-(-0.3 + 0.8 * zi)).exp());
        let p2 = 1.0 / (1.0 + (-(-0.2 + 0.7 * zi)).exp());
        let a = if rng.next_f64() < p1 { 1.0 } else { 0.0 };
        let b = if rng.next_f64() < p2 { 1.0 } else { 0.0 };
        z[i] = zi;
        t1[i] = a;
        t2[i] = b;
        y[i] = 2.0 * a + 1.5 * b + zi + 0.4 * standard_normal(&mut rng);
    }
    let mut builder = CausalSchemaBuilder::new();
    for (name, role) in [
        ("t1", RoleHint::TreatmentCandidate),
        ("t2", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let cols: Vec<OwnedColumn> = [t1, t2, y, z]
        .into_iter()
        .enumerate()
        .map(|(i, data)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(data),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut dag = Dag::with_variables(4);
    // z→t1, z→t2, z→y, t1→y, t2→y
    for (f, t) in [(3u32, 0), (3, 1), (3, 2), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(f), DenseNodeId::from_raw(t)).unwrap();
    }
    (data, dag)
}

#[test]
fn batch_matches_solo_estimates() {
    let (data, dag) = two_treatment_scm(500, 9);
    let q1 = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2));
    let q2 = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let ctx = ExecutionContext::for_tests(3);

    let batch = BatchAnalysis::new(data.clone(), dag.clone())
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .estimate_many(&[q1.clone(), q2.clone()], &ctx)
        .unwrap();
    assert_eq!(batch.len(), 2);

    let solo1 = CausalAnalysis::builder()
        .data(data.clone())
        .graph(dag.clone())
        .query(q1)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let solo2 = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(q2)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();

    assert!((batch[0].estimate.ate - solo1.estimate.ate).abs() < 1e-12);
    assert!((batch[1].estimate.ate - solo2.estimate.ate).abs() < 1e-12);
    assert!((batch[0].estimate.ate - 2.0).abs() < 0.4);
    assert!((batch[1].estimate.ate - 1.5).abs() < 0.4);
}
