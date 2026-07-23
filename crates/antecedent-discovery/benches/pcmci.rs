//! PCMCI parent-search / MCI benchmark .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::time::Instant;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, NonZeroThreadCount, Parallelism,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Pcmci, TemporalConstraints};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn synth(n: usize, p: usize) -> (TimeSeriesData, Vec<VariableId>) {
    let mut b = CausalSchemaBuilder::new();
    for i in 0..p {
        b.add_variable(
            format!("v{i}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut cols = Vec::new();
    for v in 0..p {
        let values: Vec<f64> = (0..n)
            .map(|t| ((t + v * 13) as f64 * 0.01).sin() + 0.1 * ((t as f64) * 0.001))
            .collect();
        cols.push(OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(v as u32),
                Arc::from(values),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ));
    }
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let vars: Vec<_> = (0..p).map(|i| VariableId::from_raw(i as u32)).collect();
    (data, vars)
}

fn pcmci_config() -> Pcmci {
    Pcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(2), min_lag: Lag::from_raw(1) },
        max_cond_size: 1,
        alpha: 0.05,
        ..DiscoveryConstraints::default()
    })
}

fn bench_pcmci(c: &mut Criterion) {
    let (data, vars) = synth(500, 4);
    let pcmci = pcmci_config();
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);

    c.bench_function("pcmci_n500_p4_lag2", |b| {
        b.iter(|| {
            let _ = black_box(pcmci.run(
                black_box(&data),
                black_box(&vars),
                black_box(&mut ws),
                black_box(&ctx),
            ));
        });
    });
}

fn bench_pcmci_target_parallel_scaling(c: &mut Criterion) {
    // Larger target count so target-wise parallelism is meaningful.
    let (data, vars) = synth(400, 8);
    let pcmci = pcmci_config();
    let mut group = c.benchmark_group("pcmci_target_parallel");
    for threads in [1u32, 2, 4] {
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter_custom(|iters| {
                let mut ws = DiscoveryWorkspace::default();
                let mut ctx = ExecutionContext::for_tests(1);
                ctx.parallelism =
                    Parallelism::bounded(NonZeroThreadCount::new(threads).expect("threads"));
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = black_box(pcmci.run(&data, &vars, &mut ws, &ctx));
                }
                start.elapsed()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pcmci, bench_pcmci_target_parallel_scaling);
criterion_main!(benches);
