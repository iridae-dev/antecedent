//! Program-executed interventional sampling against plain sampling of the same shape.
//!
//! The pricing chain `price -> demand -> revenue` (and `price -> revenue`) has a lognormal
//! price (`location 3.0`, `scale 0.4`). The policy `price := max(10, 0.85 * price)` is one
//! extra value-program pass over the price column, so a program-executed draw must stay
//! within the regression margin of a plain draw. The margin is relative to a plain draw on
//! the same machine in the same process, so it holds on any runner.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use antecedent_core::{CausalRng, ExecutionContext, Intervention, ValueExpr, VariableId};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, DynamicMechanism, MechanismSlot,
    MechanismWorkspace, ModelError, ParentBatch, sample_interventional,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// Rows of the conformance configuration.
const ROWS: usize = 1_000_000;
/// Most a program-executed draw may cost relative to a plain draw.
const MAX_REGRESSION: f64 = 1.1;
/// Timed repetitions per arm; the minimum is compared so scheduler noise cannot fail the bench.
const REPEATS: usize = 5;
const PRICE_LOCATION: f64 = 3.0;
const PRICE_SCALE: f64 = 0.4;

/// A root price `exp(location + scale z)`.
struct LogNormalPrice;

impl DynamicMechanism for LogNormalPrice {
    fn sample_noise_column(
        &self,
        n_rows: usize,
        rng: &mut CausalRng,
        output: &mut [f64],
    ) -> Result<(), ModelError> {
        for slot in &mut output[..n_rows] {
            *slot = standard_normal(rng);
        }
        Ok(())
    }

    fn evaluate_column(
        &self,
        parents: ParentBatch<'_>,
        noise: &[f64],
        output: &mut [f64],
        _workspace: &mut MechanismWorkspace,
    ) -> Result<(), ModelError> {
        for (out, &z) in output[..parents.n_rows].iter_mut().zip(noise) {
            *out = (PRICE_LOCATION + PRICE_SCALE * z).exp();
        }
        Ok(())
    }
}

fn gaussian(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

fn pricing_chain() -> CompiledCausalModel {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let compiled = CompiledCausalModel::compile(graph).unwrap();
    // Revenue reads price and demand; align the coefficients with the compiled gather order.
    let parents = compiled.gather_for(DenseNodeId::from_raw(2)).unwrap().parents.clone();
    let coeffs: Vec<f64> =
        parents.iter().map(|p| if p.as_usize() == 0 { 0.5 } else { 2.0 }).collect();
    compiled.with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            MechanismSlot::Dynamic { id: Arc::from("price"), mechanism: Arc::new(LogNormalPrice) },
            gaussian(50.0, &[-0.8], 1.0),
            gaussian(3.0, &coeffs, 0.5),
        ]),
    })
}

fn price_floor() -> [Intervention; 1] {
    let floor =
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated()));
    [Intervention::transform(VariableId::from_raw(0), floor)]
}

fn draw(
    model: &CompiledCausalModel,
    policy: &[Intervention],
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Duration {
    let mut rng = CausalRng::from_seed(4_691_094);
    let started = Instant::now();
    let batch = sample_interventional(model, policy, ROWS, &mut rng, ws, ctx).unwrap();
    let elapsed = started.elapsed();
    black_box(batch);
    elapsed
}

fn bench_policy_sampling(c: &mut Criterion) {
    let model = pricing_chain();
    // One context and one workspace serve every draw of every arm.
    let ctx = ExecutionContext::for_tests(1);
    let mut ws = MechanismWorkspace::default();
    let policy = price_floor();

    let mut group = c.benchmark_group("policy_sampling");
    group.sample_size(10);
    group.bench_function("sample_plain_n1000000", |b| {
        b.iter(|| draw(&model, &[], &mut ws, &ctx));
    });
    group.bench_function("sample_price_floor_program_n1000000", |b| {
        b.iter(|| draw(&model, &policy, &mut ws, &ctx));
    });
    group.finish();

    // Reuse gate: after a warm draw the workspace does not grow, with or without a program.
    draw(&model, &policy, &mut ws, &ctx);
    let grown = ws.grow_count;
    for _ in 0..3 {
        draw(&model, &policy, &mut ws, &ctx);
        draw(&model, &[], &mut ws, &ctx);
    }
    assert_eq!(ws.grow_count, grown, "policy sampling must reuse the workspace buffers");

    // Regression gate: the program costs at most `MAX_REGRESSION` of a plain draw. The arms
    // are interleaved and compared by their fastest repetition.
    let (mut plain, mut program) = (Duration::MAX, Duration::MAX);
    for _ in 0..REPEATS {
        plain = plain.min(draw(&model, &[], &mut ws, &ctx));
        program = program.min(draw(&model, &policy, &mut ws, &ctx));
    }
    let ratio = program.as_secs_f64() / plain.as_secs_f64();
    assert!(
        ratio <= MAX_REGRESSION,
        "program-executed sampling took {ratio:.3}x a plain draw ({program:?} vs {plain:?})"
    );
}

criterion_group!(benches, bench_policy_sampling);
criterion_main!(benches);
