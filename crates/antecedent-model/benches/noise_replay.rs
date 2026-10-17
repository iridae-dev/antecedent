//! Batch coalition replay against a per-draw model-cloning baseline.
//!
//! Shapley attribution over exogenous noise replays the same chain `a -> b -> c` under
//! thousands of coalitions, each fixing a subset of nodes to their observed noise and
//! resampling the rest. The batch route borrows the model, shares two retained noise tables
//! and reuses one replayer's scratch and one execution context across every coalition. The
//! baseline it replaces clones the compiled model and starts from empty scratch for every
//! draw. Replay must cost no more than the baseline plus the regression margin.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use antecedent_core::ExecutionContext;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ModelView, NoiseReplayer,
    NoiseSource, NoiseTable,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// Coalitions of the conformance configuration.
const COALITIONS: u32 = 4096;
/// Rows of the retained noise tables in the conformance configuration.
const ROWS: usize = 100_000;
/// Rows of the tables the timed Criterion loop replays.
const BENCH_ROWS: usize = 1_000;
/// Most the batch route may cost relative to the cloning baseline.
const MAX_REGRESSION: f64 = 1.1;
const REPEATS: usize = 3;

fn chain() -> CompiledCausalModel {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let linear = |intercept: f64, coeffs: &[f64], sigma: f64| MechanismSlot::LinearGaussian {
        intercept,
        coeffs: Arc::from(coeffs.to_vec()),
        sigma,
    };
    CompiledCausalModel::compile(graph).unwrap().with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            linear(1.0, &[], 1.0),
            linear(0.5, &[2.0], 0.5),
            linear(-1.0, &[3.0], 2.0),
        ]),
    })
}

/// The observed noise, an independent redraw of every node, and the coalition masks.
struct Fixture {
    observed: NoiseTable,
    redraw: NoiseTable,
}

fn fixture(model: &CompiledCausalModel, rows: usize, ctx: &ExecutionContext) -> Fixture {
    let view = ModelView::observational(model);
    let observed = NoiseTable::sample(&view, rows, ctx).unwrap();
    let mut redraw = observed.clone();
    let nodes: Vec<DenseNodeId> = (0..3).map(DenseNodeId::from_raw).collect();
    redraw.randomize(&view, &nodes, 1, ctx).unwrap();
    Fixture { observed, redraw }
}

fn mask(coalition: u32) -> [bool; 3] {
    [coalition & 1 != 0, coalition & 2 != 0, coalition & 4 != 0]
}

/// Every coalition replayed through one borrowed model, one replayer and one context.
fn batch(model: &CompiledCausalModel, f: &Fixture, ctx: &ExecutionContext) -> f64 {
    let view = ModelView::observational(model);
    let mut replayer = NoiseReplayer::new();
    let mut values = vec![0.0; f.observed.n_rows() * f.observed.n_nodes()];
    let mut checksum = 0.0;
    for coalition in 0..COALITIONS {
        let mask = mask(coalition);
        let source = NoiseSource::coalition(&f.observed, &f.redraw, &mask).unwrap();
        replayer.replay_into(&view, source, 0, &mut values, ctx).unwrap();
        checksum += values[values.len() - 1];
    }
    checksum
}

/// The same coalitions, each on a fresh clone of the model with empty scratch.
fn cloning(model: &CompiledCausalModel, f: &Fixture, ctx: &ExecutionContext) -> f64 {
    let mut values = vec![0.0; f.observed.n_rows() * f.observed.n_nodes()];
    let mut checksum = 0.0;
    for coalition in 0..COALITIONS {
        let draw_model = model.clone();
        let view = ModelView::observational(&draw_model);
        let mut replayer = NoiseReplayer::new();
        let mask = mask(coalition);
        let source = NoiseSource::coalition(&f.observed, &f.redraw, &mask).unwrap();
        replayer.replay_into(&view, source, 0, &mut values, ctx).unwrap();
        checksum += values[values.len() - 1];
    }
    checksum
}

fn timed(run: impl FnOnce() -> f64) -> (Duration, f64) {
    let started = Instant::now();
    let checksum = black_box(run());
    (started.elapsed(), checksum)
}

fn bench_noise_replay(c: &mut Criterion) {
    let model = chain();
    let ctx = ExecutionContext::for_tests(31);

    let small = fixture(&model, BENCH_ROWS, &ctx);
    let mut group = c.benchmark_group("noise_replay");
    group.sample_size(10);
    group.bench_function("coalitions_4096_n1000_batch", |b| b.iter(|| batch(&model, &small, &ctx)));
    group.bench_function("coalitions_4096_n1000_cloning_baseline", |b| {
        b.iter(|| cloning(&model, &small, &ctx));
    });
    group.finish();

    // Reuse gate: one replayer's scratch stops growing after the first coalition, and the
    // two routes compute the same values.
    let view = ModelView::observational(&model);
    let mut replayer = NoiseReplayer::new();
    let mut values = vec![0.0; BENCH_ROWS * 3];
    let warm_mask = mask(5);
    let warm = NoiseSource::coalition(&small.observed, &small.redraw, &warm_mask).unwrap();
    replayer.replay_into(&view, warm, 0, &mut values, &ctx).unwrap();
    let footprint = replayer.scratch_footprint();
    for coalition in 0..COALITIONS {
        let mask = mask(coalition);
        let source = NoiseSource::coalition(&small.observed, &small.redraw, &mask).unwrap();
        replayer.replay_into(&view, source, 0, &mut values, &ctx).unwrap();
    }
    assert_eq!(replayer.scratch_footprint(), footprint, "coalitions must reuse one scratch");
    assert_eq!(
        batch(&model, &small, &ctx).to_bits(),
        cloning(&model, &small, &ctx).to_bits(),
        "the batch route and the cloning baseline replay the same values"
    );

    // Regression gate at the configured table size, by the fastest of a few repetitions.
    let large = fixture(&model, ROWS, &ctx);
    let (mut replay, mut baseline) = (Duration::MAX, Duration::MAX);
    for _ in 0..REPEATS {
        replay = replay.min(timed(|| batch(&model, &large, &ctx)).0);
        baseline = baseline.min(timed(|| cloning(&model, &large, &ctx)).0);
    }
    let ratio = replay.as_secs_f64() / baseline.as_secs_f64();
    assert!(
        ratio <= MAX_REGRESSION,
        "batch coalition replay took {ratio:.3}x the cloning baseline ({replay:?} vs {baseline:?})"
    );
}

criterion_group!(benches, bench_noise_replay);
criterion_main!(benches);
