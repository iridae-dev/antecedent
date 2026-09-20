//! User-path jobs the 1.11 speed program measures.
//!
//! Not a `hot_paths.md` merge blocker and not in `gate_release.sh` Criterion
//! smoke. `--test` uses toy n so CI knows this compiles. Full 10⁴ / 10⁵
//! numbers are local: `USER_PATH_N=10000 cargo bench -p antecedent --bench user_path`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

use std::time::Instant;

use antecedent::discovery::GraphPosterior;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    Lag, ResponseFunctional, ResponseQuery, TemporalEffectQuery, TemporalPolicy, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::set_edge;
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_prob::InferenceDiagnostics;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn job_n() -> usize {
    std::env::var("USER_PATH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(64)
}

fn bootstrap() -> u32 {
    if job_n() >= 10_000 { 199 } else { 8 }
}

fn draws() -> usize {
    if job_n() >= 10_000 { 400 } else { 16 }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::production_default(11)
}

fn columns(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let u = (i as f64 + 0.5) * 0.017;
        z[i] = u.sin();
        t[i] = if (i % 2) == 0 { 1.0 } else { 0.0 };
        y[i] = 2.0 * t[i] + z[i] + 0.1 * (i as f64 * 0.13).cos();
    }
    (z, t, y)
}

fn tabular(n: usize) -> TabularData {
    let (z, t, y) = columns(n);
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn ate_query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ))
}

fn report(name: &str, ctx: &ExecutionContext, result: &antecedent::StudyResult) {
    let identify_hits =
        result.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count();
    eprintln!(
        "user_path {name}: n={} threads={} wall_ns={:?} copies={} bytes_borrowed={:?} identify_cache_hits={identify_hits} ate={}",
        job_n(),
        ctx.parallelism.max_threads.get(),
        result.performance.wall_time_ns,
        result.performance.copy_count,
        result.performance.bytes_borrowed,
        result.estimate.ate
    );
}

fn bench_user_path(c: &mut Criterion) {
    let n = job_n();
    let data = tabular(n);
    let graph = dag();
    let ctx = ctx();
    if ctx.parallelism.max_threads.get() == 1 {
        eprintln!(
            "user_path threads=1 (record this; production default is capped available_parallelism)"
        );
    }
    let ate = ate_query();

    let frequentist = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(ate.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap())
        .build()
        .unwrap();
    c.bench_function("user_path_dag_ate_frequentist", |b| {
        b.iter(|| black_box(frequentist.clone().run(&ctx).unwrap().estimate.ate));
    });
    let started = Instant::now();
    let freq_result = frequentist.run(&ctx).unwrap();
    report("dag_ate_frequentist", &ctx, &freq_result);
    eprintln!("user_path dag_ate_frequentist elapsed={:?}", started.elapsed());

    let prepared = frequentist.prepare(&ctx).unwrap();
    let inspect_start = Instant::now();
    black_box(prepared.inspect().unwrap());
    let inspect_elapsed = inspect_start.elapsed();
    let estimate_start = Instant::now();
    black_box(prepared.estimate(&data, &ctx).unwrap());
    let estimate_elapsed = estimate_start.elapsed();
    assert!(
        inspect_elapsed < estimate_elapsed || n < 128,
        "inspect must stay cheaper than a fit on the user path"
    );

    let bayesian = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(ate.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(draws())))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    c.bench_function("user_path_dag_ate_bayesian_laplace", |b| {
        b.iter(|| black_box(bayesian.clone().run(&ctx).unwrap().estimate.ate));
    });
    report("dag_ate_bayesian_laplace", &ctx, &bayesian.run(&ctx).unwrap());

    let level =
        CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: std::sync::Arc::from([Intervention::set(
                VariableId::from_raw(0),
                Value::f64(1.0),
            )]),
        }));
    let intervention = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(level)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap())
        .build()
        .unwrap();
    c.bench_function("user_path_intervention_level", |b| {
        b.iter(|| black_box(intervention.clone().run(&ctx).unwrap().estimate.ate));
    });
    report("intervention_level", &ctx, &intervention.run(&ctx).unwrap());

    let kn = n.max(128);
    let ka: Vec<f64> = (0..kn).map(|i| (i as f64 * 0.71).sin()).collect();
    let ky: Vec<f64> =
        ka.iter().enumerate().map(|(i, a)| 2.0 * a + 0.1 * (i as f64 * 0.13).cos()).collect();
    let kennedy_data =
        TabularData::from_f64_columns([("t", ka.as_slice()), ("y", ky.as_slice())]).unwrap();
    let mut kennedy_dag = Dag::with_variables(2);
    kennedy_dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let kennedy = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    }));
    let curve = Study::tabular(kennedy_data)
        .graph(kennedy_dag)
        .query(kennedy)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .response_options(antecedent_estimate::ContinuousResponseOptions {
            bandwidth: Some(0.35),
            ..Default::default()
        })
        .build()
        .unwrap();
    c.bench_function("user_path_kennedy_curve", |b| {
        b.iter(|| black_box(curve.clone().run(&ctx).unwrap().estimate.ate));
    });
    report("kennedy_curve", &ctx, &curve.run(&ctx).unwrap());

    let mask_a = set_edge(0, 3, 0, 1, true);
    let mask_b = set_edge(set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true), 3, 0, 1, true);
    let posterior = GraphPosterior::new(
        3,
        vec![0.6, 0.4],
        vec![mask_a, mask_b],
        vec![0.0; 9],
        vec![0.0; 9],
        1.0,
        InferenceDiagnostics::analytic("user_path_gp"),
        0,
    )
    .unwrap();
    let mixture = Study::tabular(data)
        .graph_posterior(posterior)
        .query(ate)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap())
        .build()
        .unwrap();
    c.bench_function("user_path_graph_posterior_mixture", |b| {
        b.iter(|| black_box(mixture.clone().run(&ctx).unwrap().estimate.ate));
    });
    report("graph_posterior_mixture", &ctx, &mixture.run(&ctx).unwrap());

    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.2 * ((t as f64) * 0.07).sin();
        y[t] = 0.8 * x[t - 1];
    }
    let series =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let pulse = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let mut temporal_dag = TemporalDag::empty();
    let x1 = ensure_lagged(&mut temporal_dag, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 =
        ensure_lagged(&mut temporal_dag, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    temporal_dag.insert_directed(x1, y0).unwrap();
    let temporal = Study::series(series)
        .graph(temporal_dag)
        .temporal_query(pulse)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(bootstrap())
        .build()
        .unwrap();
    c.bench_function("user_path_temporal_pulse", |b| {
        b.iter(|| black_box(temporal.clone().run(&ctx).unwrap().estimate.ate));
    });
    report("temporal_pulse", &ctx, &temporal.run(&ctx).unwrap());
}

criterion_group!(benches, bench_user_path);
criterion_main!(benches);
