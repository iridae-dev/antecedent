//! Temporal mediation sparse/stress benches (Phase 9).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, MediationContrast, MediationQuery,
    RoleHint, SmallRoleSet, Value, ValueType, VariableId,
};
use causal_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use causal_estimate::TemporalMediationEstimator;
use causal_expr::{CausalExprArena, IdentifiedEstimand};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn mediated(n: usize) -> (TimeSeriesData, MediationQuery, IdentifiedEstimand) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 1..n {
        t[i] = 0.3 * t[i - 1] + 0.05 * (i as f64).sin() + 0.01 * ((i * 3) as f64).cos();
        m[i] = 0.7 * t[i - 1] + 0.02 * (i as f64).cos();
        y[i] = 0.5 * m[i] + 0.2 * t[i - 1] + 0.01 * (i as f64).sin();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(m), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let q = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    );
    let mut arena = CausalExprArena::new();
    let functional =
        arena.frontdoor_ate(q.treatment, q.outcome, &q.mediators, Value::f64(1.0), Value::f64(0.0));
    let estimand = IdentifiedEstimand::frontdoor(
        "temporal_mediation.mediated",
        Arc::clone(&q.mediators),
        functional,
    );
    (data, q, estimand)
}

fn bench_mediation(c: &mut Criterion) {
    // Soft budgets (phase9_regime_mediation.md); 2× headroom for gate `--test` noise.
    const SPARSE_BUDGET: Duration = Duration::from_millis(10);
    const STRESS_BUDGET: Duration = Duration::from_millis(40);

    let est = TemporalMediationEstimator::new();

    c.bench_function("mediation_sparse_200", |b| {
        let (data, q, estimand) = mediated(200);
        b.iter(|| {
            let r = est.estimate(&data, &estimand, &q, &ExecutionContext::for_tests(1)).unwrap();
            black_box(r.mediated);
        });
    });
    c.bench_function("mediation_stress_800", |b| {
        let (data, q, estimand) = mediated(800);
        b.iter(|| {
            let r = est.estimate(&data, &estimand, &q, &ExecutionContext::for_tests(2)).unwrap();
            black_box(r.total);
        });
    });

    {
        let (data, q, estimand) = mediated(200);
        let t0 = Instant::now();
        let r = est.estimate(&data, &estimand, &q, &ExecutionContext::for_tests(1)).unwrap();
        let elapsed = t0.elapsed();
        assert!(r.mediated.is_some());
        assert!(
            elapsed < SPARSE_BUDGET,
            "mediation_sparse_200 exceeded soft budget: {elapsed:?} >= {SPARSE_BUDGET:?}"
        );
    }
    {
        let (data, q, estimand) = mediated(800);
        let t0 = Instant::now();
        let r = est.estimate(&data, &estimand, &q, &ExecutionContext::for_tests(2)).unwrap();
        let elapsed = t0.elapsed();
        assert!(r.total.is_some());
        assert!(
            elapsed < STRESS_BUDGET,
            "mediation_stress_800 exceeded soft budget: {elapsed:?} >= {STRESS_BUDGET:?}"
        );
    }
}

criterion_group!(benches, bench_mediation);
criterion_main!(benches);
