//! Known-truth check of `temporal.sequential.gcomp` on a multi-step sustained schedule over
//! `TemporalDag`, `TemporalCpdag` and `TemporalPag`, explicit, accepted and graph-posterior:
//! the Bayesian plain contrast, and (second test) the modifier-averaged effect of a multi-step
//! schedule with a named effect modifier on a fixed `TemporalDag`; the class graphs and graph
//! posteriors refuse a modifier set on a multi-step window. Cases run on scoped threads and every failure is collected by label.
//!
//! The law is `y_t = 1 + 2 x_{t-1} + 3 x_{t-2} + 0.5 w_t`, with the binary treatment cycling
//! `0, 0, 1`, the binary modifier alternating, and `r`, the cause of the treatment that gives a
//! PAG its visible edges, a nuisance cycle. The modifier does not interact with the treatment, so
//! the sustained effect over lags `-2..-1`, marginal over `w`, is `2 + 3 = 5`.
//!
//! * Frequentist: the series is noise-free, ordinary least squares is exact, tolerance `1e-8`.
//! * Bayesian: a deterministic period-5 disturbance of amplitude `0.02` keeps the residual
//!   variance positive. The conjugate posterior has standard deviation near `0.003` on the 120 rows, its
//!   mean differs from the least-squares value by far less, and `100` draws put the Monte Carlo
//!   error of the posterior mean near `3e-4`, so the stated bound `0.02` is several posterior
//!   standard deviations wide.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use antecedent_prob::InferenceDiagnostics;

const TRUTH: f64 = 5.0;
const FREQUENTIST_TOLERANCE: f64 = 1e-8;
const BAYESIAN_BOUND: f64 = 0.02;
const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

fn var(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// Columns `x, y, w, r`.
fn series(disturbance: f64) -> TimeSeriesData {
    let n = 120usize;
    let x = (0..n).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let w = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let r = (0..n).map(|i| [0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0][i % 7]).collect::<Vec<_>>();
    let pattern = [1.0, -1.0, 0.5, -0.5, 0.0];
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| x[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| x[j])
                + 0.5 * w[i]
                + disturbance * pattern[i % 5]
        })
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice()), ("r", r.as_slice())],
        1,
    )
    .unwrap()
}

/// `r_{t-1} -> x_{t-1}`, `r_{t-2} -> x_{t-2}`, `x_{t-1} -> y_t`, `x_{t-2} -> y_t`, `w_t -> y_t`.
fn dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let r1 = ensure_lagged(&mut graph, var(3), Lag::from_raw(1)).unwrap();
    let r2 = ensure_lagged(&mut graph, var(3), Lag::from_raw(2)).unwrap();
    let x1 = ensure_lagged(&mut graph, var(0), Lag::from_raw(1)).unwrap();
    let x2 = ensure_lagged(&mut graph, var(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, var(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = ensure_lagged(&mut graph, var(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(r1, x1).unwrap();
    graph.insert_directed(r2, x2).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(x2, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

/// The same edges; an arrowhead into each treatment lag makes `x -> y` visible.
fn pag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let r1 = graph.add_lagged(var(3), Lag::from_raw(1)).unwrap();
    let r2 = graph.add_lagged(var(3), Lag::from_raw(2)).unwrap();
    let x1 = graph.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let x2 = graph.add_lagged(var(0), Lag::from_raw(2)).unwrap();
    let y0 = graph.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(r1, x1).unwrap();
    graph.insert_directed(r2, x2).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(x2, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

/// One-atom posterior over `(x, y, w, r)`: `w_t -> y_t`, `r_t -> x_t` and the lag-1 and lag-2
/// edges `x -> y`.
fn posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let n = 4;
    let contemporaneous =
        set_edge(set_edge(0, n, 2, 1, true), n, 3, 0, true).to_word().expect("one word");
    // Lag `l` edge `from -> to` is bit `(l - 1) * 16 + from * 4 + to`.
    let mask = (1_u64 << 1) | (1_u64 << 17);
    let mut marginals = vec![0.0; 2 * n * n];
    marginals[1] = 1.0;
    marginals[17] = 1.0;
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![contemporaneous],
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("sequential_multistep_known_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(2, marginals)
    .unwrap()
    .with_lag_masks(vec![mask])
    .unwrap()
}

fn query(conditional: bool) -> CausalQuery {
    let base = TemporalEffectQuery::sustained(var(0), var(1), 0, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1));
    CausalQuery::TemporalEffect(if conditional {
        base.with_effect_modifiers([var(2)])
    } else {
        base
    })
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100).prior_scale(1_000.0))
}

/// Posterior mean for a Bayesian result, else the point estimate, else the identified envelope.
fn ends(result: &antecedent::StudyResult, inference: &InferenceMode) -> (f64, f64) {
    assert!(!result.logical_plan.plan_id.is_empty(), "the result names its logical plan");
    if matches!(inference, InferenceMode::Bayesian(_))
        && let Some(posterior) = &result.posterior
        && let Some(column) = posterior.effect_column()
    {
        let mean = posterior.summaries.mean[column];
        return (mean, mean);
    }
    let ate = result.estimate().as_effect().unwrap().ate;
    if ate.is_finite() {
        return (ate, ate);
    }
    let set = result
        .structural_response
        .as_ref()
        .and_then(|mixture| mixture.identified_set.as_ref())
        .expect("an identified set");
    (set.lower[0], set.upper[0])
}

struct Case {
    conditional: bool,
    bayesian: bool,
    class: &'static str,
    accepted: bool,
    kind: Option<GraphPosteriorAtomKind>,
    suite: RefuteSuite,
}

fn run_case(case: &Case, ctx: &ExecutionContext) -> Option<String> {
    let inference = if case.bayesian { bayesian() } else { InferenceMode::Frequentist };
    let (data, tolerance) = if case.bayesian {
        (series(0.02), BAYESIAN_BOUND)
    } else {
        (series(0.0), FREQUENTIST_TOLERANCE)
    };
    let base = Study::series(data);
    let builder = match (case.kind, case.class, case.accepted) {
        (Some(kind), _, _) => base.graph_posterior(posterior(kind)),
        (None, "TemporalDag", false) => base.graph(dag()),
        (None, "TemporalDag", true) => base.graph(AcceptedGraph::from(dag())),
        (None, "TemporalCpdag", false) => base.graph(TemporalCpdag::from_temporal_dag(&dag())),
        (None, "TemporalCpdag", true) => {
            base.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&dag())))
        }
        (None, _, false) => base.graph(pag()),
        (None, _, true) => base.graph(AcceptedGraph::temporal_pag(pag())),
    };
    let label = format!(
        "conditional={} bayesian={} {} accepted={} posterior={} {:?}",
        case.conditional,
        case.bayesian,
        case.class,
        case.accepted,
        case.kind.is_some(),
        case.suite
    );
    let builder = builder
        .refute(case.suite)
        .query(query(case.conditional))
        .estimator(EstimatorId::TemporalSequentialGcomp)
        .inference(inference.clone())
        .bootstrap_replicates(0);
    let built = builder.clone().build();
    drop(builder);
    let outcome = built.and_then(|study| study.run(ctx));
    match outcome {
        Ok(result) => {
            let (low, high) = ends(&result, &inference);
            ((low - TRUTH).abs() >= tolerance || (high - TRUTH).abs() >= tolerance)
                .then(|| format!("{label}: value [{low}, {high}] vs {TRUTH}"))
        }
        Err(error) => Some(format!("{label}: REFUSED {error}")),
    }
}

#[test]
fn sequential_gcomp_recovers_the_multi_step_effect_for_bayesian_plain_and_conditional_contrasts() {
    let ctx = ExecutionContext::for_tests(31);
    let mut cases = Vec::new();
    for suite in SUITES {
        // The Bayesian plain contrast: every class, structure and the graph posterior.
        for class in ["TemporalDag", "TemporalCpdag", "TemporalPag"] {
            for accepted in [false, true] {
                cases.push(Case {
                    conditional: false,
                    bayesian: true,
                    class,
                    accepted,
                    kind: None,
                    suite,
                });
            }
        }
        for (class, kind) in [
            ("TemporalDag", GraphPosteriorAtomKind::Dag),
            ("TemporalCpdag", GraphPosteriorAtomKind::Cpdag),
            ("TemporalPag", GraphPosteriorAtomKind::Pag),
        ] {
            cases.push(Case {
                conditional: false,
                bayesian: true,
                class,
                accepted: false,
                kind: Some(kind),
                suite,
            });
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failures = std::sync::Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(cases.len());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(case) = cases.get(index) else { break };
                    if let Some(failure) = run_case(case, &ctx) {
                        failures.lock().unwrap().push(failure);
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{} coordinates:\n{}", failures.len(), failures.join("\n"));
}

/// `t` cycles `-2..2`; the modifier `w_i = 0.8 + 0.6 t_{i-1} + c_i` with `c` cycling over a
/// symmetric grid; `y_i = t_{i-1} + 0.5 t_{i-2} + 3 t_{i-1} w_i (+ disturbance)`. Columns
/// `t, y, w`. The modifier is heterogeneous: the effect of the first lag is `1 + 3 w`.
fn modifier_series(disturbance: f64) -> (TimeSeriesData, f64) {
    let n = 250usize;
    let t: Vec<f64> = (0..n).map(|i| [-2.0, -1.0, 0.0, 1.0, 2.0][i % 5]).collect();
    let mut w = vec![0.0; n];
    let mut y = vec![0.0; n];
    let pattern = [1.0, -1.0, 0.5, -0.5, 0.0];
    for i in 2..n {
        let noise = [-1.5, -0.5, 0.5, 1.5][i % 4];
        w[i] = 0.8 + 0.6 * t[i - 1] + noise;
        y[i] = t[i - 1] + 0.5 * t[i - 2] + 3.0 * t[i - 1] * w[i] + disturbance * pattern[i % 5];
    }
    #[allow(clippy::cast_precision_loss, reason = "small exact row count")]
    let mean_w = w[2..].iter().sum::<f64>() / (n - 2) as f64;
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap();
    // The sustained effect over lags -2..-1 averaged over the modifier: 1 + 3 mean(w) + 0.5.
    (data, 1.0 + 3.0 * mean_w + 0.5)
}

fn modifier_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, var(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, var(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, var(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

fn modifier_run(
    data: &TimeSeriesData,
    accepted: bool,
    modified: bool,
    inference: &InferenceMode,
    suite: RefuteSuite,
    ctx: &ExecutionContext,
) -> Result<f64, String> {
    let base = Study::series(data.clone());
    let base = if accepted {
        base.graph(AcceptedGraph::from(modifier_graph()))
    } else {
        base.graph(modifier_graph())
    };
    let plain = TemporalEffectQuery::sustained(var(0), var(1), 0, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1));
    let query = if modified { plain.with_effect_modifiers([var(2)]) } else { plain };
    let builder = base
        .query(CausalQuery::TemporalEffect(query))
        .estimator(EstimatorId::TemporalSequentialGcomp)
        .inference(inference.clone())
        .refute(suite)
        .bootstrap_replicates(0);
    let built = builder.clone().build();
    drop(builder);
    built
        .and_then(|study| study.run(ctx))
        .map(|result| ends(&result, inference).0)
        .map_err(|error| error.to_string())
}

/// `TemporalConditionalSustained` on a fixed `TemporalDag`, explicit and accepted, Frequentist
/// and Bayesian, every suite: the result is the effect averaged over the modifier, which the
/// interaction makes differ from the unmodified sustained effect by more than 0.5.
#[test]
fn sequential_gcomp_returns_the_modifier_averaged_effect_on_a_fixed_dag() {
    let ctx = ExecutionContext::for_tests(32);
    let mut cases = Vec::new();
    for bayesian_inference in [false, true] {
        for accepted in [false, true] {
            for suite in SUITES {
                cases.push((bayesian_inference, accepted, suite));
            }
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failures = std::sync::Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(cases.len());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(is_bayesian, accepted, suite)) = cases.get(index) else { break };
                    let (inference, disturbance, tolerance) = if is_bayesian {
                        (bayesian(), 0.02, BAYESIAN_BOUND)
                    } else {
                        (InferenceMode::Frequentist, 0.0, FREQUENTIST_TOLERANCE)
                    };
                    let (data, expected) = modifier_series(disturbance);
                    let label = format!("bayesian={is_bayesian} accepted={accepted} {suite:?}");
                    let outcome = modifier_run(&data, accepted, true, &inference, suite, &ctx)
                        .and_then(|value| {
                            let plain = modifier_run(
                                &data,
                                accepted,
                                false,
                                &inference,
                                RefuteSuite::None,
                                &ctx,
                            )?;
                            Ok((value, plain))
                        });
                    match outcome {
                        Ok((value, plain)) => {
                            if (value - expected).abs() >= tolerance {
                                failures.lock().unwrap().push(format!(
                                    "{label}: {value} vs modifier-averaged {expected}"
                                ));
                            }
                            if (value - plain).abs() <= 0.5 {
                                failures.lock().unwrap().push(format!(
                                "{label}: {value} does not differ from the plain sustained {plain}"
                            ));
                            }
                        }
                        Err(error) => {
                            failures.lock().unwrap().push(format!("{label}: REFUSED {error}"))
                        }
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{} coordinates:\n{}", failures.len(), failures.join("\n"));
}
