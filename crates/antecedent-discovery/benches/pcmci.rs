//! PCMCI parent-search / MCI benchmark .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

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
            let _ = black_box(
                pcmci
                    .run(black_box(&data), black_box(&vars), black_box(&mut ws), black_box(&ctx))
                    .expect("PCMCI benchmark must return a complete result"),
            );
        });
    });
}

fn bench_pcmci_target_parallel_scaling(c: &mut Criterion) {
    // Larger target count so target-wise parallelism is meaningful.
    let (data, vars) = synth(400, 8);
    let pcmci = pcmci_config();
    let expected = pcmci
        .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ExecutionContext::for_tests(1))
        .expect("serial discovery truth");
    let mut group = c.benchmark_group("pcmci_target_parallel");
    for threads in [1u32, 2, 4] {
        let mut probe = ExecutionContext::for_tests(1);
        probe.parallelism =
            Parallelism::bounded(NonZeroThreadCount::new(threads).expect("threads"));
        probe = probe
            .with_admission(antecedent_core::execution::AdmissionPool::fixed(threads, u64::MAX));
        let result = pcmci
            .run(&data, &vars, &mut DiscoveryWorkspace::default(), &probe)
            .expect("parallel discovery truth");
        assert_eq!(result.performance.ci_tests, expected.performance.ci_tests);
        assert_eq!(result.evidence.links, expected.evidence.links);
        eprintln!(
            "pcmci_target_parallel threads={threads} admitted={} ci_tests={} links={} matches_serial=true",
            probe.admission().thread_budget(),
            result.performance.ci_tests,
            result.evidence.links.len()
        );
        group.bench_function(format!("threads_{threads}"), |b| {
            b.iter_custom(|iters| {
                let mut ws = DiscoveryWorkspace::default();
                let mut ctx = ExecutionContext::for_tests(1);
                ctx.parallelism =
                    Parallelism::bounded(NonZeroThreadCount::new(threads).expect("threads"));
                ctx = ctx.with_admission(antecedent_core::execution::AdmissionPool::fixed(
                    threads,
                    u64::MAX,
                ));
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = black_box(
                        pcmci
                            .run(&data, &vars, &mut ws, &ctx)
                            .expect("PCMCI benchmark must return a complete result"),
                    );
                }
                start.elapsed()
            });
        });
    }
    // Same fixture as the historical requested-4/one-slot context. This measures
    // the scheduler's bounded serial fallback separately from funded parallelism.
    group.bench_function("requested_4_admitted_1", |b| {
        b.iter_custom(|iters| {
            let mut ws = DiscoveryWorkspace::default();
            let mut ctx = ExecutionContext::for_tests(1);
            ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).expect("threads"));
            let result = pcmci.run(&data, &vars, &mut ws, &ctx).expect("capped discovery truth");
            assert_eq!(result.evidence.links, expected.evidence.links);
            assert_eq!(result.performance.ci_tests, expected.performance.ci_tests);
            let start = Instant::now();
            for _ in 0..iters {
                black_box(
                    pcmci.run(&data, &vars, &mut ws, &ctx).expect("complete capped discovery"),
                );
            }
            start.elapsed()
        });
    });
    group.finish();
}

criterion_group!(benches, bench_pcmci, bench_pcmci_target_parallel_scaling);
criterion_main!(benches);
