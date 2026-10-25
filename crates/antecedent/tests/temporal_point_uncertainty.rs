//! Whole-procedure uncertainty of the linear unfolded temporal point routes:
//! cross-world contrasts, derivatives and the one-modifier conditional effect on a
//! `TemporalDag`, under a noisy simulated series with known mechanism coefficients.
//!
//! Frequentist intervals refit every regression of the route on one shared
//! circular-block bootstrap of series times; Bayesian intervals evaluate the route
//! on joint draws of every regression's tempered slope posterior. Neither has a
//! coverage record: these tests pin recovery of the known truth within statistical
//! tolerance, reproducibility across thread budgets, and the admission refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, DerivativeScale, ExecutionContext, IntervalMethod,
    Intervention, InterventionalDistributionQuery, Lag, NestedCounterfactualQuery,
    NonZeroThreadCount, Parallelism, PathSpecificEffectQuery, ResponseFunctional, ResponseQuery,
    ResponseUncertainty, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 400;

/// Deterministic standard normals (xorshift64* and Box–Muller), so the fixture
/// does not depend on any library generator.
struct Normals(u64);

impl Normals {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// `x_t = 0.5 x_{t-1} + u`, `m_t = 0.8 x_{t-1} + v`, `y_t = x_{t-1} + 1.5 m_t + w`,
/// unit-variance Gaussian disturbances: direct 1.0, mediated 0.8 · 1.5 = 1.2,
/// total 2.2, interventional mean at `x = 1` equal to 2.2.
fn mediated_series(seed: u64) -> TimeSeriesData {
    let mut draw = Normals(seed | 1);
    let mut x = vec![0.0; N];
    let mut m = vec![0.0; N];
    let mut y = vec![0.0; N];
    for t in 1..N {
        x[t] = 0.5 * x[t - 1] + draw.next();
        m[t] = 0.8 * x[t - 1] + draw.next();
        y[t] = x[t - 1] + 1.5 * m[t] + draw.next();
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn mediated_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    graph
}

fn cross_world_queries() -> Vec<(&'static str, CausalQuery, f64)> {
    let (x, m, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    vec![
        (
            "nested",
            CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(x, m, y).unwrap()),
            1.0,
        ),
        (
            "path",
            CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(x, y).with_path_nodes([m])),
            1.2,
        ),
        (
            "counterfactual",
            CausalQuery::Counterfactual(
                CounterfactualQuery::new(y, Arc::from([Intervention::set(x, Value::f64(1.0))]))
                    .with_control_level(0.0),
            ),
            2.2,
        ),
        (
            "distribution",
            CausalQuery::Distribution(InterventionalDistributionQuery::new(
                y,
                [Intervention::set(x, Value::f64(1.0))],
            )),
            2.2,
        ),
    ]
}

fn bayesian(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(draws))
}

fn run_series(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: CausalQuery,
    inference: Option<InferenceMode>,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<StudyResult, antecedent::CausalError> {
    let builder = Study::series(data.clone())
        .graph(graph.clone())
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates);
    let builder = match inference {
        Some(inference) => builder.inference(inference),
        None => builder,
    };
    builder.build()?.prepare(ctx)?.estimate_retained(ctx)
}

/// The published frequentist SE of a scalar result.
fn block_se(result: &StudyResult) -> f64 {
    result.estimate.as_effect().unwrap().se_bootstrap().expect("a shared-block SE")
}

#[test]
fn frequentist_cross_world_intervals_refit_every_regression_and_recover_the_mechanism() {
    let data = mediated_series(17);
    let graph = mediated_graph();
    let ctx = ExecutionContext::for_tests(5);
    for (key, query, truth) in cross_world_queries() {
        let result = run_series(&data, &graph, query, None, 199, &ctx).unwrap();
        let binding = result.interval.as_ref().expect("an interval binding");
        assert_eq!(binding.method, IntervalMethod::CircularBlockSe, "{key}");
        let se = block_se(&result);
        let point = result.estimate.as_effect().unwrap().ate;
        assert!(se.is_finite() && se > 0.0 && se < 0.5, "{key} se={se}");
        assert!((point - truth).abs() < 4.0 * se, "{key}: {point} vs {truth} (se {se})");
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.temporal.point.circular_block"),
            "{key}"
        );
    }
}

#[test]
fn bayesian_cross_world_posteriors_are_joint_draws_of_every_regression() {
    let data = mediated_series(17);
    let graph = mediated_graph();
    let ctx = ExecutionContext::for_tests(5);
    for (key, query, truth) in cross_world_queries() {
        let result = run_series(&data, &graph, query, Some(bayesian(400)), 0, &ctx).unwrap();
        let posterior = result.posterior.as_ref().expect("a posterior");
        assert_eq!(
            result.interval.as_ref().map(|binding| binding.method),
            Some(IntervalMethod::PosteriorQuantile),
            "{key}"
        );
        // The effect column, not column 0: a g-computation posterior also carries its
        // coefficient columns (column 0 of the unit counterfactual is the intercept).
        let column = posterior.effect_column().expect("an effect column");
        let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
        assert!(sd.is_finite() && sd > 0.0 && sd < 0.5, "{key} sd={sd}");
        assert!((mean - truth).abs() < 4.0 * sd, "{key}: {mean} vs {truth} (sd {sd})");
        let (low, high) = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
        assert!(low < mean && mean < high, "{key}");
    }
}

#[test]
fn point_only_without_replicates_and_the_same_interval_for_any_thread_budget() {
    let data = mediated_series(23);
    let graph = mediated_graph();
    let (_, query, _) = cross_world_queries().remove(1);
    let serial = ExecutionContext::for_tests(9);
    let mut parallel = ExecutionContext::for_tests(9);
    parallel.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).unwrap());
    let a = run_series(&data, &graph, query.clone(), None, 99, &serial).unwrap();
    let b = run_series(&data, &graph, query.clone(), None, 99, &parallel).unwrap();
    assert_eq!(block_se(&a).to_bits(), block_se(&b).to_bits());
    // No replicates requested: the point is published alone and carries no interval
    // binding, like every point-only route.
    let point = run_series(&data, &graph, query.clone(), None, 0, &serial).unwrap();
    assert!(point.interval.is_none());
    assert!(
        point
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.temporal.point.uncertainty_withheld")
    );
    let draws_a = run_series(&data, &graph, query.clone(), Some(bayesian(64)), 0, &serial).unwrap();
    let draws_b = run_series(&data, &graph, query, Some(bayesian(64)), 0, &parallel).unwrap();
    assert_eq!(draws_a.posterior.unwrap().draws.values, draws_b.posterior.unwrap().draws.values);
}

#[test]
fn a_tiny_admission_pool_refuses_the_block_bootstrap_and_returns_its_lease() {
    let data = mediated_series(29);
    let graph = mediated_graph();
    let (_, query, _) = cross_world_queries().remove(2);
    let study = Study::series(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(20_000)
        .build()
        .unwrap();
    let tiny = ExecutionContext::for_tests(1)
        .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 40_000));
    let before = tiny.admission.available();
    // Compilation accounts for declared retained replicates before execution.
    // The aggregate quote exceeds this pool even though the base design fits.
    let error = study.compile(&tiny).unwrap_err();
    assert!(matches!(error, antecedent::CausalError::Resource { .. }), "{error}");
    assert!(error.to_string().contains("estimated peak memory"), "{error}");
    assert_eq!(tiny.admission.available(), before, "a rejected quote holds no lease");

    // A prepared plan may later run under a smaller caller pool. Exercise the
    // block bootstrap's real allocation refusal and cleanup independently of
    // compile-time admission, retaining the same declared replicate workload.
    let prepared = study.prepare(&ExecutionContext::for_tests(1)).unwrap();
    let error = prepared.estimate_retained(&tiny).unwrap_err();
    assert!(
        error.to_string().contains("admission pool cannot fund the temporal point block bootstrap"),
        "{error}"
    );
    assert_eq!(tiny.admission.available(), before);
}

/// `y_t = 3 + 2 x_{t-1} + 0.5 b_{t-1} + e`, `v_t = 1 + x_{t-1} - b_{t-1} + e'`.
fn derivative_series(seed: u64) -> TimeSeriesData {
    let mut draw = Normals(seed | 1);
    let mut x = vec![0.0; N];
    let mut b = vec![0.0; N];
    let mut y = vec![0.0; N];
    let mut v = vec![0.0; N];
    for t in 0..N {
        x[t] = 1.0 + draw.next();
        b[t] = draw.next();
        if t > 0 {
            y[t] = 3.0 + 2.0 * x[t - 1] + 0.5 * b[t - 1] + draw.next();
            v[t] = 1.0 + x[t - 1] - b[t - 1] + draw.next();
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("b", b.as_slice()), ("y", y.as_slice()), ("v", v.as_slice())],
        1,
    )
    .unwrap()
}

fn derivative_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b1 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let v0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(b1, y0).unwrap();
    graph.insert_directed(x1, v0).unwrap();
    graph.insert_directed(b1, v0).unwrap();
    graph
}

fn jacobian_query() -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            at: Arc::from([1.0, 0.0]),
            scale: DerivativeScale::Identity,
        })
        .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap()),
    )
}

#[test]
fn derivative_bands_cover_every_jacobian_cell_with_a_simultaneous_band() {
    let data = derivative_series(31);
    let graph = derivative_graph();
    let truth = [2.0, 0.5, 1.0, -1.0];
    let ctx = ExecutionContext::for_tests(3);
    for inference in [None, Some(bayesian(400))] {
        let bayes = inference.is_some();
        let replicates = if bayes { 0 } else { 199 };
        let result =
            run_series(&data, &graph, jacobian_query(), inference, replicates, &ctx).unwrap();
        let response = result.response().expect("a derivative response");
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
            panic!("bayes={bayes}: expected a pointwise band, got {:?}", response.uncertainty);
        };
        for (cell, want) in truth.iter().enumerate() {
            let width = upper[cell] - lower[cell];
            assert!(width > 0.0 && width < 1.0, "bayes={bayes} cell {cell} width {width}");
            // Recovery within a widened band: the band is unmeasured, not a coverage claim.
            let slack = width;
            assert!(
                lower[cell] - slack < *want && *want < upper[cell] + slack,
                "bayes={bayes} cell {cell}: [{}, {}] vs {want}",
                lower[cell],
                upper[cell]
            );
        }
        assert!(
            response
                .support
                .diagnostics
                .iter()
                .any(|d| d.id.as_ref() == antecedent_estimate::SIMULTANEOUS_BAND_CRITICAL),
            "bayes={bayes}: a sup-t simultaneous band accompanies the pointwise band"
        );
        let method = result.interval.as_ref().map(|binding| binding.method);
        let expected =
            if bayes { IntervalMethod::PosteriorQuantile } else { IntervalMethod::CircularBlockSe };
        assert_eq!(method, Some(expected), "bayes={bayes}");
    }
}

/// `y_t = 1.5 x_{t-1} + 0.2 w_t + 0.8 x_{t-1} w_t + e`, binary `w`.
fn conditional_series(seed: u64) -> (TimeSeriesData, f64) {
    let mut draw = Normals(seed | 1);
    let mut x = vec![0.0; N];
    let mut w = vec![0.0; N];
    let mut y = vec![0.0; N];
    for t in 0..N {
        x[t] = draw.next();
        w[t] = if draw.uniform() < 0.4 { 1.0 } else { 0.0 };
        if t > 0 {
            y[t] = 1.5 * x[t - 1] + 0.2 * w[t] + 0.8 * x[t - 1] * w[t] + draw.next();
        }
    }
    let mean_w = w[1..].iter().sum::<f64>() / (N - 1) as f64;
    let data = TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap();
    (data, 1.5 + 0.8 * mean_w)
}

#[test]
fn conditional_marginal_effect_interval_refits_the_interaction_and_the_modifier_mean() {
    let (data, truth) = conditional_series(37);
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    let query = CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1))
            .with_effect_modifiers([VariableId::from_raw(2)])
            .with_policy(TemporalPolicy::pulse(-1)),
    );
    let ctx = ExecutionContext::for_tests(7);
    let frequentist = run_series(&data, &graph, query.clone(), None, 199, &ctx).unwrap();
    let se = block_se(&frequentist);
    let point = frequentist.estimate.as_effect().unwrap().ate;
    assert_eq!(
        frequentist.interval.as_ref().map(|binding| binding.method),
        Some(IntervalMethod::CircularBlockSe)
    );
    assert!(se > 0.0 && (point - truth).abs() < 4.0 * se, "{point} vs {truth} (se {se})");
    let posterior = run_series(&data, &graph, query, Some(bayesian(400)), 0, &ctx).unwrap();
    let draws = posterior.posterior.as_ref().expect("a posterior");
    let column = draws.effect_column().expect("an effect column");
    let summaries = &draws.summaries;
    assert!((summaries.mean[column] - truth).abs() < 4.0 * summaries.sd[column]);
}
