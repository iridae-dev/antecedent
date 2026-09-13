//! DBN-posterior Pulse hot path (ADR 0011).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss)]

use std::time::{Duration, Instant};

use antecedent::discovery::GraphPosterior;
use antecedent::{InferenceMode, RefuteSuite, Study};
use antecedent_core::{ExecutionContext, TemporalEffectQuery, TemporalPolicy, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_prob::InferenceDiagnostics;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const PULSE_BUDGET: Duration = Duration::from_millis(250);

fn series(n: usize) -> TimeSeriesData {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.2 * ((t as f64) * 0.07).sin();
        y[t] = 0.8 * x[t - 1];
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn posterior() -> GraphPosterior {
    GraphPosterior::new(
        2,
        vec![0.6, 0.4],
        vec![0, 0],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0 / (0.6 * 0.6 + 0.4 * 0.4),
        InferenceDiagnostics::analytic("dbn_pulse_bench"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![2, 2])
    .unwrap()
}

fn query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn bench_dbn_posterior_pulse(c: &mut Criterion) {
    let data = series(400);
    let ctx = ExecutionContext::for_tests(11);
    let study = Study::series(data.clone())
        .graph_posterior(posterior())
        .temporal_query(query())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(8)
        .build()
        .unwrap();
    c.bench_function("dbn_posterior_pulse_n400_boot8", |b| {
        b.iter(|| black_box(study.clone().run(&ctx).unwrap().estimate.ate));
    });
    let started = Instant::now();
    let _ = study.run(&ctx).unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < PULSE_BUDGET,
        "dbn_posterior_pulse_n400_boot8 exceeded soft budget: {elapsed:?}"
    );
}

criterion_group!(benches, bench_dbn_posterior_pulse);
criterion_main!(benches);
