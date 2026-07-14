//! Regime discovery sparse/stress benches (Phase 9).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use causal_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use causal_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, PcmciPlus, Rpcmci, TemporalConstraints,
    two_regime_half_split,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn series(n: usize) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y"] {
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
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.4 * x[t - 1] + 0.05 * (t as f64).sin();
        y[t] = 0.6 * x[t] + 0.1 * y[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 1 },
            length: n,
        },
    )
    .unwrap()
}

fn bench_rpcmci(c: &mut Criterion) {
    let plus = PcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(1),
            min_lag: Lag::CONTEMPORANEOUS,
        },
        alpha: 0.2,
        max_cond_size: 1,
        ..DiscoveryConstraints::default()
    });
    let alg = Rpcmci::new().with_min_regime_len(30).with_pcmci_plus(plus);
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];

    // Soft budgets from benches/baselines/phase9_regime_mediation.md (Apple M1 class).
    const SPARSE_BUDGET: Duration = Duration::from_millis(500);
    const STRESS_BUDGET: Duration = Duration::from_secs(2);

    c.bench_function("rpcmci_sparse_120", |b| {
        let data = series(120);
        let assign = two_regime_half_split(120);
        b.iter(|| {
            let mut ws = DiscoveryWorkspace::default();
            let r = alg
                .run(&data, &vars, &assign, &mut ws, &ExecutionContext::for_tests(1))
                .unwrap();
            black_box(r.graphs.len());
        });
    });

    c.bench_function("rpcmci_stress_240", |b| {
        let data = series(240);
        let assign = two_regime_half_split(240);
        b.iter(|| {
            let mut ws = DiscoveryWorkspace::default();
            let r = alg
                .run(&data, &vars, &assign, &mut ws, &ExecutionContext::for_tests(2))
                .unwrap();
            black_box(r.graphs.len());
        });
    });

    // Gate `--test` path: single timed iteration must stay under soft budgets.
    {
        let data = series(120);
        let assign = two_regime_half_split(120);
        let mut ws = DiscoveryWorkspace::default();
        let t0 = Instant::now();
        let r = alg
            .run(&data, &vars, &assign, &mut ws, &ExecutionContext::for_tests(1))
            .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(r.graphs.len(), 2);
        assert!(
            elapsed < SPARSE_BUDGET,
            "rpcmci_sparse_120 exceeded soft budget: {elapsed:?} >= {SPARSE_BUDGET:?}"
        );
    }
    {
        let data = series(240);
        let assign = two_regime_half_split(240);
        let mut ws = DiscoveryWorkspace::default();
        let t0 = Instant::now();
        let r = alg
            .run(&data, &vars, &assign, &mut ws, &ExecutionContext::for_tests(2))
            .unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(r.graphs.len(), 2);
        assert!(
            elapsed < STRESS_BUDGET,
            "rpcmci_stress_240 exceeded soft budget: {elapsed:?} >= {STRESS_BUDGET:?}"
        );
    }
}

criterion_group!(benches, bench_rpcmci);
criterion_main!(benches);
