//! The lag-0 `AverageEffect` under a temporal graph against the structural coefficient.
//!
//! The fixture `conformance/estimate/lag_zero_average_effect` states the linear law
//! `x_t = a z_{t-1} + e_t`, `y_t = beta x_t + gamma z_{t-1} + 0.5 u_t`: the contrast
//! `E[y_t | do(x_t = 1)] - E[y_t | do(x_t = 0)]` is `beta` whatever `gamma` is, and omitting the
//! confounder converges to `beta + gamma a / (a^2 + 1)`. The build compiles the query as the
//! pulse at offset 0 read at horizon 1, so the same number is published on a `TemporalDag`,
//! on the temporal backdoor of every completion of a `TemporalCpdag` and on the visibility-aware
//! adjustment of a `TemporalPag`, as explicit, accepted and graph-posterior structures, under
//! Frequentist and Bayesian inference at every validation depth. The refuters the fixture names
//! run, with the placebo treatment and the dummy outcome at the exact null 0 and the random
//! common cause, subset and bootstrap refuters at the original value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag};
use antecedent_prob::InferenceDiagnostics;

const FIXTURE: &str =
    include_str!("../../../conformance/estimate/lag_zero_average_effect/expected.json");
const N: usize = 1500;

struct Truth {
    beta: f64,
    loading: f64,
    gamma: f64,
    effect_tolerance: f64,
    null_tolerance: f64,
    stability_tolerance: f64,
    document: serde_json::Value,
}

fn truth() -> Truth {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let law = |key: &str| document["law"][key].as_f64().unwrap();
    let tolerance = |key: &str| document["tolerance"][key].as_f64().unwrap();
    Truth {
        beta: law("beta"),
        loading: law("treatment_loading"),
        gamma: law("confounder_to_outcome"),
        effect_tolerance: tolerance("effect"),
        null_tolerance: tolerance("null_refuter"),
        stability_tolerance: tolerance("stability_refuter"),
        document,
    }
}

fn x() -> VariableId {
    VariableId::from_raw(0)
}

fn y() -> VariableId {
    VariableId::from_raw(1)
}

fn z() -> VariableId {
    VariableId::from_raw(2)
}

/// A deterministic standard-normal stream (xorshift and Box-Muller).
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        })
        .collect()
}

/// The fixture's law with `z_{t-1} -> y_t` of strength `gamma` (zero is the law of [`pag`]).
fn series(truth: &Truth, gamma: f64) -> TimeSeriesData {
    let zs = normals(N, 1);
    let e = normals(N, 2);
    let u = normals(N, 3);
    let mut xs = vec![0.0; N];
    let mut ys = vec![0.0; N];
    for t in 1..N {
        xs[t] = truth.loading * zs[t - 1] + e[t];
        ys[t] = truth.beta * xs[t] + gamma * zs[t - 1] + 0.5 * u[t];
    }
    TimeSeriesData::from_f64_columns(
        [("x", xs.as_slice()), ("y", ys.as_slice()), ("z", zs.as_slice())],
        1,
    )
    .unwrap()
}

fn dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let z1 = graph.add_lagged(z(), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(x(), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(y(), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x0).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(x0, y0).unwrap();
    graph
}

/// `z[t-1] -> x[t] -> y[t]` with `z[t-1]` not adjacent to `y[t]`.
fn pag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let z1 = graph.add_lagged(z(), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(x(), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(y(), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x0).unwrap();
    graph.insert_directed(x0, y0).unwrap();
    graph
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1u64 << (from * n + to)
}

/// One-atom posterior whose atom is `x_t -> y_t` with the lag-1 edges `lag`.
fn posterior(kind: GraphPosteriorAtomKind, lag: u64) -> GraphPosterior {
    let n = 3;
    let contemporaneous = set_edge(0, n, 0, 1, true).to_word().expect("one word");
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![contemporaneous],
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("lag_zero_average_effect_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag])
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Class {
    TemporalDag,
    TemporalCpdag,
    TemporalPag,
}

#[derive(Clone, Copy, Debug)]
enum Structure {
    Explicit,
    Accepted,
    GraphPosterior,
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200).prior_scale(100.0))
}

fn inference_name(inference: &InferenceMode) -> &'static str {
    if matches!(inference, InferenceMode::Frequentist) { "Frequentist" } else { "Bayesian" }
}

fn suite_name(suite: RefuteSuite) -> &'static str {
    match suite {
        RefuteSuite::Cheap => "cheap",
        RefuteSuite::Full => "full",
        _ => "none",
    }
}

fn query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::with_levels(x(), y(), 0.0, 1.0))
}

fn lag_zero_pulse() -> CausalQuery {
    CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(x(), y(), 1.0))
}

fn run(
    truth: &Truth,
    class: Class,
    structure: Structure,
    inference: &InferenceMode,
    suite: RefuteSuite,
    query: CausalQuery,
) -> StudyResult {
    // The PAG omits z[t-1] -> y[t], so it is fitted on the law that has no such edge.
    let data = match class {
        Class::TemporalPag => series(truth, 0.0),
        _ => series(truth, truth.gamma),
    };
    let builder = Study::series(data);
    let builder = match (class, structure) {
        (Class::TemporalDag, Structure::Explicit) => builder.graph(dag()),
        (Class::TemporalDag, Structure::Accepted) => builder.graph(AcceptedGraph::from(dag())),
        (Class::TemporalCpdag, Structure::Explicit) => {
            builder.graph(TemporalCpdag::from_temporal_dag(&dag()))
        }
        (Class::TemporalCpdag, Structure::Accepted) => {
            builder.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&dag())))
        }
        (Class::TemporalPag, Structure::Explicit) => builder.graph(pag()),
        (Class::TemporalPag, Structure::Accepted) => {
            builder.graph(AcceptedGraph::temporal_pag(pag()))
        }
        (Class::TemporalDag, Structure::GraphPosterior) => builder.graph_posterior(posterior(
            GraphPosteriorAtomKind::Dag,
            lag_bit(3, 2, 0) | lag_bit(3, 2, 1),
        )),
        (Class::TemporalCpdag, Structure::GraphPosterior) => builder.graph_posterior(posterior(
            GraphPosteriorAtomKind::Cpdag,
            lag_bit(3, 2, 0) | lag_bit(3, 2, 1),
        )),
        (Class::TemporalPag, Structure::GraphPosterior) => {
            builder.graph_posterior(posterior(GraphPosteriorAtomKind::Pag, lag_bit(3, 2, 0)))
        }
    };
    builder
        .query(query)
        .inference(inference.clone())
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap_or_else(|error| panic!("{class:?} {structure:?}: {error}"))
}

/// The published effect, or the ends of the identified set when the class publishes a set.
fn ends(result: &StudyResult) -> (f64, f64) {
    let ate = result.estimate().as_effect().unwrap().ate;
    if ate.is_finite() {
        return (ate, ate);
    }
    let set = result
        .structural_response
        .as_ref()
        .and_then(|mixture| mixture.identified_set.as_ref())
        .expect("a class study without a scalar publishes its identified set");
    (set.lower[0], set.upper[0])
}

/// A refuter's name without the completion prefix a class study adds.
fn bare(name: &str) -> &str {
    match name.strip_prefix("completion.") {
        Some(rest) => rest.split_once('.').map_or(rest, |(_, bare)| bare),
        None => name,
    }
}

fn assert_cell(
    result: &StudyResult,
    inference: &InferenceMode,
    suite: RefuteSuite,
    truth: &Truth,
    contract_key: &str,
    no_standard_error: bool,
    label: &str,
) {
    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
    let (low, high) = ends(result);
    assert!(
        (low - truth.beta).abs() < truth.effect_tolerance
            && (high - truth.beta).abs() < truth.effect_tolerance,
        "{label}: effect [{low}, {high}] vs structural coefficient {}",
        truth.beta
    );
    if suite == RefuteSuite::None {
        assert!(result.refutations.is_empty(), "{label}");
        return;
    }
    let contract = &truth.document["refuter_contract"];
    let wanted = contract[contract_key][suite_name(suite)].as_array().unwrap();
    for name in wanted {
        let name = name.as_str().unwrap();
        assert!(
            result.refutations.iter().any(|report| bare(report.refuter.as_ref()) == name),
            "{label}: refuter {name} did not run"
        );
    }
    // The e-value passes only through the confidence limit of the effect: an estimate that
    // carries no standard error (the explicit or accepted TemporalDag Frequentist study with no
    // bootstrap replicates) reports the point e-value as uninformative and cannot pass, while
    // a class study, whose route supplies an analytic standard error, passes it.
    if let Some(evalue) = result
        .refutations
        .iter()
        .find(|report| bare(report.refuter.as_ref()) == "sensitivity.evalue")
    {
        assert_eq!(evalue.informative, !no_standard_error, "{label}: e-value informativeness");
        if no_standard_error {
            assert!(!evalue.passed, "{label}: a point e-value cannot support a pass");
        }
    }
    let listed = |key: &str, name: &str| {
        contract[key].as_array().unwrap().iter().any(|entry| entry.as_str() == Some(name))
    };
    for report in &result.refutations {
        let name = bare(report.refuter.as_ref());
        if listed("null_expected", name) {
            assert!(
                report.refuted_ate.abs() < truth.null_tolerance,
                "{label}: {name} must sit at the null 0, got {}",
                report.refuted_ate
            );
            assert!(report.passed, "{label}: {name} flags the clean law");
        }
        if listed("original_expected", name) {
            assert!(
                (report.refuted_ate - report.original_ate).abs() < truth.stability_tolerance,
                "{label}: {name} moved the effect from {} to {}",
                report.original_ate,
                report.refuted_ate
            );
            assert!(report.passed, "{label}: {name} flags the clean law");
        }
        if name == "posterior_predictive" {
            assert!(report.passed, "{label}: the posterior check flags the clean law");
        }
    }
}

/// `AverageEffect` x `TemporalDag`, `TemporalCpdag` and `TemporalPag` x explicit, accepted and
/// graph_posterior x Frequentist and Bayesian x none, cheap and full: the lag-0 effect is the
/// structural coefficient and the refuters the fixture names run at their expected values.
#[test]
fn lag_zero_average_effect_matches_the_structural_coefficient_on_every_temporal_cell() {
    let truth = truth();
    for class in [Class::TemporalDag, Class::TemporalCpdag, Class::TemporalPag] {
        for structure in [Structure::Explicit, Structure::Accepted, Structure::GraphPosterior] {
            for inference in [InferenceMode::Frequentist, bayesian()] {
                for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label = format!(
                        "{class:?} {structure:?} {} {}",
                        inference_name(&inference),
                        suite_name(suite)
                    );
                    let result = run(&truth, class, structure, &inference, suite, query());
                    // A class graph posterior mixes the completions' frequentist-style
                    // suite under either inference mode; the predictive checks and the
                    // prior sensitivity are not run there.
                    let key = match (class, structure, &inference) {
                        (
                            Class::TemporalCpdag | Class::TemporalPag,
                            Structure::GraphPosterior,
                            InferenceMode::Bayesian(_),
                        ) => "Bayesian_class_graph_posterior",
                        _ => inference_name(&inference),
                    };
                    let no_standard_error = matches!(
                        (class, structure, &inference),
                        (
                            Class::TemporalDag,
                            Structure::Explicit | Structure::Accepted,
                            InferenceMode::Frequentist
                        )
                    );
                    assert_cell(&result, &inference, suite, &truth, key, no_standard_error, &label);
                }
            }
        }
    }
}

/// The alias is exactly the explicit lag-0 pulse, and omitting the confounder would not give
/// the structural coefficient: the pulse on the DAG fitted on the law with the edge reproduces
/// the alias bit for bit, and the unadjusted slope the fixture records lies outside the band.
#[test]
fn lag_zero_average_effect_is_the_explicit_pulse_and_the_adjustment_matters() {
    let truth = truth();
    let unadjusted = truth.document["unadjusted_plim"].as_f64().unwrap();
    assert!((unadjusted - truth.beta).abs() > 5.0 * truth.effect_tolerance);
    for class in [Class::TemporalDag, Class::TemporalCpdag, Class::TemporalPag] {
        for structure in [Structure::Explicit, Structure::GraphPosterior] {
            for inference in [InferenceMode::Frequentist, bayesian()] {
                let alias = run(&truth, class, structure, &inference, RefuteSuite::None, query());
                let direct =
                    run(&truth, class, structure, &inference, RefuteSuite::None, lag_zero_pulse());
                let (a_low, a_high) = ends(&alias);
                let (d_low, d_high) = ends(&direct);
                assert_eq!(a_low.to_bits(), d_low.to_bits(), "{class:?} {structure:?}");
                assert_eq!(a_high.to_bits(), d_high.to_bits(), "{class:?} {structure:?}");
                assert!((a_low - unadjusted).abs() > 3.0 * truth.effect_tolerance);
            }
        }
    }
}

/// The retained plan runs every class and structure with the builder dropped.
#[test]
fn lag_zero_average_effect_cells_execute_from_the_retained_plan_with_the_builder_dropped() {
    let truth = truth();
    let context = ExecutionContext::for_tests(12);
    for class in [Class::TemporalDag, Class::TemporalCpdag, Class::TemporalPag] {
        for structure in [Structure::Explicit, Structure::Accepted, Structure::GraphPosterior] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap] {
                let data = match class {
                    Class::TemporalPag => series(&truth, 0.0),
                    _ => series(&truth, truth.gamma),
                };
                let base = Study::series(data.clone());
                let builder = match (class, structure) {
                    (Class::TemporalDag, Structure::Explicit) => base.graph(dag()),
                    (Class::TemporalDag, Structure::Accepted) => {
                        base.graph(AcceptedGraph::from(dag()))
                    }
                    (Class::TemporalCpdag, Structure::Explicit) => {
                        base.graph(TemporalCpdag::from_temporal_dag(&dag()))
                    }
                    (Class::TemporalCpdag, Structure::Accepted) => {
                        base.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&dag())))
                    }
                    (Class::TemporalPag, Structure::Explicit) => base.graph(pag()),
                    (Class::TemporalPag, Structure::Accepted) => {
                        base.graph(AcceptedGraph::temporal_pag(pag()))
                    }
                    (Class::TemporalDag, Structure::GraphPosterior) => base.graph_posterior(
                        posterior(GraphPosteriorAtomKind::Dag, lag_bit(3, 2, 0) | lag_bit(3, 2, 1)),
                    ),
                    (Class::TemporalCpdag, Structure::GraphPosterior) => {
                        base.graph_posterior(posterior(
                            GraphPosteriorAtomKind::Cpdag,
                            lag_bit(3, 2, 0) | lag_bit(3, 2, 1),
                        ))
                    }
                    (Class::TemporalPag, Structure::GraphPosterior) => base
                        .graph_posterior(posterior(GraphPosteriorAtomKind::Pag, lag_bit(3, 2, 0))),
                };
                let builder = builder
                    .query(query())
                    .inference(InferenceMode::Frequentist)
                    .refute(suite)
                    .bootstrap_replicates(0);
                let study = builder.clone().build().unwrap();
                let prepared = study.prepare(&context).unwrap();
                drop(builder);
                drop(study);
                assert!(prepared.plan().logical.record.estimator.is_some(), "{class:?}");
                let result = prepared
                    .estimate_series(&data, &context)
                    .unwrap_or_else(|error| panic!("{class:?} {structure:?}: {error}"));
                let (low, high) = ends(&result);
                assert!(
                    (low - truth.beta).abs() < truth.effect_tolerance,
                    "{class:?} {structure:?}"
                );
                assert!(
                    (high - truth.beta).abs() < truth.effect_tolerance,
                    "{class:?} {structure:?}"
                );
                assert_eq!(result.refutations.is_empty(), suite == RefuteSuite::None);
            }
        }
    }
}
