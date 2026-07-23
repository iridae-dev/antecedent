//! conformance: post-ID column projection matches full-column estimate (BACKLOG E).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use causal::CausalAnalysis;
use causal_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId};
use causal_kernels::standard_normal;

fn wide_confounded_scm(
    n: usize,
    noise_cols: usize,
    seed: u64,
) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5051_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.4;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
    }

    let mut vars: Vec<(String, RoleHint, Vec<f64>)> = Vec::with_capacity(3 + noise_cols);
    vars.push(("t".into(), RoleHint::TreatmentCandidate, t));
    vars.push(("y".into(), RoleHint::OutcomeCandidate, y));
    vars.push(("z".into(), RoleHint::Context, z));
    for k in 0..noise_cols {
        let mut col = vec![0.0; n];
        for v in &mut col {
            *v = standard_normal(&mut rng);
        }
        vars.push((format!("noise_{k}"), RoleHint::Context, col));
    }

    let mut b = CausalSchemaBuilder::new();
    for (name, role, _) in &vars {
        b.add_variable(
            name.as_str(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(*role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols: Vec<OwnedColumn> = vars
        .iter()
        .enumerate()
        .map(|(i, (_, _, data))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(data.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());

    let n_vars = u32::try_from(3 + noise_cols).unwrap();
    let mut dag = Dag::with_variables(n_vars);
    // t=0, y=1, z=2
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

#[test]
fn wide_table_projection_matches_full_column_ate() {
    let noise_cols = 200;
    let (data, dag, query) = wide_confounded_scm(500, noise_cols, 42);
    assert_eq!(data.schema().len(), 3 + noise_cols);

    let ctx = ExecutionContext::for_tests(7);
    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(0)
        .refute(causal::RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();

    assert!((result.estimate.ate - 2.0).abs() < 0.35, "ate={}", result.estimate.ate);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "exec.project.columns"),
        "expected projection diagnostic on wide table"
    );
    // Projected to t, y, z only.
    let msg = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "exec.project.columns")
        .unwrap()
        .message
        .as_ref();
    assert!(
        msg.contains(&format!("{} → 3", 3 + noise_cols)),
        "unexpected projection message: {msg}"
    );
}

#[test]
fn thin_table_skips_projection_diagnostic() {
    let (data, dag, query) = wide_confounded_scm(200, 0, 3);
    assert_eq!(data.schema().len(), 3);
    let ctx = ExecutionContext::for_tests(1);
    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(0)
        .refute(causal::RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(
        result
            .diagnostics
            .iter()
            .all(|d| d.code.as_ref() != "exec.project.columns")
    );
}
