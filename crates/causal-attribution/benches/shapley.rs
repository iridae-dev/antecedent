//! Shapley attribution coalition-cache / MC bench (Phase 10).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]

use std::time::{Duration, Instant};

use causal_attribution::{CoalitionPayoff, estimate_shapley};
use causal_core::{CachePolicy, ComponentId, ExecutionContext, Parallelism, ShapleyConfig};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

struct AdditivePayoff {
    weights: Vec<f64>,
}

impl CoalitionPayoff for AdditivePayoff {
    fn value(&mut self, mask: u64) -> Result<f64, causal_attribution::AttributionError> {
        let mut s = 0.0;
        for (i, &w) in self.weights.iter().enumerate() {
            if mask & (1u64 << i) != 0 {
                s += w;
            }
        }
        Ok(s)
    }
}

fn run_mc(n_players: usize, n_samples: usize, cache: bool) {
    let players: Vec<_> = (0..n_players as u32).map(ComponentId::from_raw).collect();
    let mut payoff = AdditivePayoff { weights: (0..n_players).map(|i| (i + 1) as f64).collect() };
    let cfg = ShapleyConfig::monte_carlo(n_samples).with_seed(1);
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.parallelism = Parallelism::serial();
    ctx.cache_policy =
        if cache { CachePolicy::enabled(Some(10_000_000)) } else { CachePolicy::disabled() };
    let est = estimate_shapley(&players, &cfg, &mut payoff, &ctx).unwrap();
    black_box(est.values);
}

fn shapley_benches(c: &mut Criterion) {
    c.bench_function("shapley_mc_8p_200_cached", |b| {
        b.iter(|| run_mc(8, 200, true));
    });
    c.bench_function("shapley_mc_8p_200_uncached", |b| {
        b.iter(|| run_mc(8, 200, false));
    });
    c.bench_function("shapley_exact_10p_cached", |b| {
        b.iter(|| {
            let players: Vec<_> = (0..10u32).map(ComponentId::from_raw).collect();
            let mut payoff =
                AdditivePayoff { weights: (0..10).map(|i| f64::from(i + 1)).collect() };
            let cfg = ShapleyConfig::exact().with_max_exact_components(12);
            let mut ctx = ExecutionContext::for_tests(1);
            ctx.cache_policy = CachePolicy::enabled(Some(10_000_000));
            black_box(estimate_shapley(&players, &cfg, &mut payoff, &ctx).unwrap());
        });
    });
}

fn latency_gate_assert() {
    let t0 = Instant::now();
    run_mc(8, 200, true);
    let elapsed = t0.elapsed();
    assert!(elapsed < Duration::from_millis(500), "shapley_mc_8p_200_cached took {elapsed:?}");
}

criterion_group!(benches, shapley_benches);
criterion_main!(benches);

#[allow(dead_code)]
fn _gate_entry() {
    latency_gate_assert();
}
