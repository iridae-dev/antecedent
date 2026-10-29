//! Cheap and full validation of an `InterventionResponse` level on a `TemporalDag`, `Cpdag`,
//! `Pag`, `TemporalCpdag` and `TemporalPag` against the closed-form level of each completion.
//!
//! The fixture `conformance/estimate/intervention_response_class_refuter_states` states the
//! laws and derives, for each completion, the interventional level (the completion that adjusts
//! for the confounder, 3) or the association (the completion that does not, 3.441 on the static
//! law and 3.209 on the temporal one) and the unit-contrast witness each completion's refuters
//! run on (2 and 2.299). A Cpdag or Pag level runs the plugin-level suite on every completion's
//! own level; a temporal level runs the Pulse-native suite on each completion's own witness.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, Lag, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{
    Cpdag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalCpdag, TemporalDag,
    TemporalPag, ensure_lagged,
};

const FIXTURE: &str = include_str!(
    "../../../conformance/estimate/intervention_response_class_refuter_states/expected.json"
);

struct Truth {
    adjusted_level: f64,
    adjusted_witness: f64,
    static_unadjusted: f64,
    temporal_unadjusted: f64,
    temporal_witness_unadjusted: f64,
    level_tolerance: f64,
    null: f64,
    stability: f64,
    document: serde_json::Value,
}

fn truth() -> Truth {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let closed = |key: &str| document["closed_form"][key].as_f64().unwrap();
    let tolerance = |key: &str| document["tolerance"][key].as_f64().unwrap();
    Truth {
        adjusted_level: closed("adjusted_level"),
        adjusted_witness: closed("adjusted_witness"),
        static_unadjusted: closed("static_unadjusted"),
        temporal_unadjusted: closed("temporal_unadjusted"),
        temporal_witness_unadjusted: closed("temporal_witness_unadjusted"),
        level_tolerance: tolerance("level"),
        null: tolerance("null_refuter"),
        stability: tolerance("stability"),
        document,
    }
}

struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn var(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// Columns `z, t, y`: `t = 0.6 z + e`, `y = 1 + 2 t + z + 0.5 u`.
fn cpdag_table() -> TabularData {
    let mut stream = Stream(13);
    let n = 800;
    let z: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let t: Vec<f64> = z.iter().map(|z| 0.6 * z + stream.normal()).collect();
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + z + 0.5 * stream.normal()).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

/// Columns `t, y, z, r`: `t = 0.5 z + 0.4 r + e`, `y = 1 + 2 t + 0.8 z + u`.
fn pag_table() -> TabularData {
    let mut stream = Stream(5);
    let n = 500;
    let (mut t, mut y, mut z, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = stream.normal();
        r[i] = stream.normal();
        t[i] = 0.5 * z[i] + 0.4 * r[i] + stream.normal();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + stream.normal();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap()
}

/// Series `t, y`: `y_i = 1 + 2 t_{i-1} + 0.5 u`.
fn tdag_series() -> TimeSeriesData {
    let mut stream = Stream(17);
    let n: usize = 400;
    let t: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j]) + 0.5 * stream.normal())
        .collect();
    TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

/// Series `t, y, z, r` (the last only when `with_r`).
fn class_series(with_r: bool) -> TimeSeriesData {
    let mut stream = Stream(1);
    let n = 500;
    let (mut t, mut y, mut z, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        r[i] = stream.normal();
        z[i] = 0.4 * r[i] + stream.normal();
        t[i] = 0.3 + 0.5 * r[i] + 0.2 * z[i] + 0.7 * stream.normal();
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 0.6 * z[i - 1] + 0.3 * stream.normal();
        }
    }
    let mut named: Vec<(&str, &[f64])> =
        vec![("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())];
    if with_r {
        named.push(("r", r.as_slice()));
    }
    TimeSeriesData::from_f64_columns(named, 1).unwrap()
}

/// `z - t` undirected, both into `y`: one completion adjusts `z`, the other adjusts nothing.
fn cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    graph.insert_undirected(node(0), node(1)).unwrap();
    graph.insert_directed(node(0), node(2)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph
}

/// `r o-> t`, `z o-> t`, `z -> y`, `t -> y` over `(t, y, z, r)`.
fn pag() -> Pag {
    let mut graph = Pag::with_variables(4);
    for (a, b, at_a, at_b) in [
        (3, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 1, Endpoint::Tail, Endpoint::Arrow),
        (0, 1, Endpoint::Tail, Endpoint::Arrow),
    ] {
        graph
            .insert_marked(MarkedEdge {
                a: node(a),
                b: node(b),
                at_a,
                at_b,
                middle: MiddleMark::Empty,
            })
            .unwrap();
    }
    graph
}

fn tdag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, var(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, var(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph
}

/// `x@-1 -> y`, `z@-1 -> y`, `z@-1 - x@-1` undirected.
fn tcpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(var(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_undirected(z1, t1).unwrap();
    graph
}

/// Shielded `z@-1 o-o t@-1` with `r@-1 -> {z, t}@-1` and both into `y`: directed MAG completions
/// identify, the latent `z <-> t` completion does not.
fn tpag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let t1 = graph.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(var(2), Lag::from_raw(1)).unwrap();
    let r1 = graph.add_lagged(var(3), Lag::from_raw(1)).unwrap();
    graph.insert_directed(r1, z1).unwrap();
    graph.insert_directed(r1, t1).unwrap();
    graph.insert_circle_circle_with_middle(z1, t1, MiddleMark::Empty).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph
}

#[derive(Clone, Copy, Debug)]
enum Class {
    TemporalDag,
    Cpdag,
    Pag,
    TemporalCpdag,
    TemporalPag,
}

impl Class {
    fn temporal(self) -> bool {
        matches!(self, Self::TemporalDag | Self::TemporalCpdag | Self::TemporalPag)
    }

    fn name(self) -> &'static str {
        match self {
            Self::TemporalDag => "TemporalDag",
            Self::Cpdag => "Cpdag",
            Self::Pag => "Pag",
            Self::TemporalCpdag => "TemporalCpdag",
            Self::TemporalPag => "TemporalPag",
        }
    }
}

fn level_query(class: Class) -> CausalQuery {
    let (outcome, treatment) = match class {
        Class::Cpdag => (2, 1),
        Class::Pag => (1, 0),
        _ => (1, 0),
    };
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: var(outcome),
        interventions: Arc::from([Intervention::set(var(treatment), Value::f64(1.0))]),
    });
    CausalQuery::Response(if class.temporal() {
        query.with_temporal(
            TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap(),
        )
    } else {
        query
    })
}

fn run(class: Class, accepted: bool, inference: &InferenceMode, suite: RefuteSuite) -> StudyResult {
    macro_rules! structure {
        ($builder:expr, $graph:expr) => {
            if accepted {
                $builder.graph(AcceptedGraph::from($graph))
            } else {
                $builder.graph($graph)
            }
        };
    }
    let builder = match class {
        Class::Cpdag => structure!(Study::tabular(cpdag_table()), cpdag()),
        Class::Pag => structure!(Study::tabular(pag_table()), pag()),
        Class::TemporalDag => structure!(Study::series(tdag_series()), tdag()),
        Class::TemporalCpdag => structure!(Study::series(class_series(false)), tcpdag()),
        Class::TemporalPag => structure!(Study::series(class_series(true)), tpag()),
    };
    builder
        .query(level_query(class))
        .inference(inference.clone())
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap_or_else(|error| panic!("{class:?}: {error}"))
        .run(&ExecutionContext::for_tests(13))
        .unwrap_or_else(|error| panic!("{class:?}: {error}"))
}

/// Every level the response publishes: a point, the ends of an identified set, or the levels of
/// the completions a graph-dependent result keeps apart.
fn levels(result: &StudyResult) -> Vec<f64> {
    fn of(value: &ResponseValue) -> Vec<f64> {
        match value {
            ResponseValue::Scalar(value) => vec![*value],
            ResponseValue::Surface { mean, .. } => mean.to_vec(),
            ResponseValue::Envelope(envelope) => {
                envelope.lower.iter().chain(envelope.upper.iter()).copied().collect()
            }
            _ => Vec::new(),
        }
    }
    match &result.response().expect("response").estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => of(value),
        ResponseIdentification::GraphDependent(parts) => {
            parts.iter().flat_map(|(_, value)| of(value)).collect()
        }
        other => panic!("unexpected identification {other:?}"),
    }
}

fn inference_name(inference: &InferenceMode) -> &'static str {
    if matches!(inference, InferenceMode::Frequentist) { "Frequentist" } else { "Bayesian" }
}

fn suite_name(suite: RefuteSuite) -> &'static str {
    if suite == RefuteSuite::Cheap { "cheap" } else { "full" }
}

fn has_diagnostic(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
}

fn assert_cell(
    result: &StudyResult,
    class: Class,
    inference: &InferenceMode,
    suite: RefuteSuite,
    truth: &Truth,
    label: &str,
) {
    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
    // Each completion's level is the adjusted level or the association of the unadjusted one.
    let unadjusted =
        if class.temporal() { truth.temporal_unadjusted } else { truth.static_unadjusted };
    for level in levels(result) {
        assert!(
            (level - truth.adjusted_level).abs() < truth.level_tolerance
                || (matches!(class, Class::Cpdag | Class::TemporalCpdag | Class::TemporalPag)
                    && (level - unadjusted).abs() < truth.level_tolerance),
            "{label}: level {level} is neither {} nor {unadjusted}",
            truth.adjusted_level
        );
    }
    let state = if class.temporal() {
        "refute.temporal_response.pulse_witness_state"
    } else {
        "refute.response.class_state"
    };
    assert!(has_diagnostic(result, state), "{label}: the {state} disclosure");
    let family = if class.temporal() { "temporal" } else { "static" };
    let contract = &truth.document["refuter_contract"];
    for name in contract[inference_name(inference)][family][suite_name(suite)].as_array().unwrap() {
        let name = name.as_str().unwrap();
        assert!(
            result.refutations.iter().any(|report| report.refuter.as_ref() == name),
            "{label}: refuter {name} did not run"
        );
    }
    let listed = |key: &str, name: &str| {
        contract[key].as_array().unwrap().iter().any(|entry| entry.as_str() == Some(name))
    };
    // The refuters read each completion's own level (static) or witness effect (temporal).
    let originals: Vec<f64> = if class.temporal() {
        vec![truth.adjusted_witness, truth.temporal_witness_unadjusted]
    } else {
        vec![truth.adjusted_level, truth.static_unadjusted]
    };
    for report in &result.refutations {
        let name = report.refuter.as_ref();
        if matches!(name, "prior_predictive" | "posterior_predictive" | "prior_sensitivity") {
            continue;
        }
        assert!(
            originals.iter().any(|want| (report.original_ate - want).abs() < truth.level_tolerance),
            "{label}: {name} reads {} which is no completion's closed form {originals:?}",
            report.original_ate
        );
        if !listed("verdict_not_asserted", name) {
            assert!(report.passed, "{label}: {name} flags the clean law");
        }
        if listed("null_expected", name) {
            assert!(report.refuted_ate.abs() < truth.null, "{label}: {name} must sit at 0");
        }
        if listed("original_expected", name) {
            assert!(
                (report.refuted_ate - report.original_ate).abs() < truth.stability,
                "{label}: {name} moved {} to {}",
                report.original_ate,
                report.refuted_ate
            );
        }
    }
    for report in &result.refutations {
        if report.refuter.as_ref() == "posterior_predictive" {
            assert!(report.passed, "{label}: the posterior check flags the clean law");
        }
    }
}

/// `InterventionResponse` x `TemporalDag`, `Cpdag`, `Pag`, `TemporalCpdag` and `TemporalPag` x
/// explicit and accepted x Frequentist and Bayesian x cheap and full.
#[test]
fn intervention_response_class_cheap_and_full_match_each_completions_closed_form() {
    let truth = truth();
    let bayes = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64));
    for class in
        [Class::TemporalDag, Class::Cpdag, Class::Pag, Class::TemporalCpdag, Class::TemporalPag]
    {
        for accepted in [false, true] {
            for inference in [InferenceMode::Frequentist, bayes.clone()] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label = format!(
                        "{} accepted={accepted} {} {}",
                        class.name(),
                        inference_name(&inference),
                        suite_name(suite)
                    );
                    let result = run(class, accepted, &inference, suite);
                    assert_cell(&result, class, &inference, suite, &truth, &label);
                }
            }
        }
    }
}

/// The retained plan of every class runs the refuter state with the builder dropped.
#[test]
fn intervention_response_class_cells_execute_from_the_retained_plan_with_the_builder_dropped() {
    let truth = truth();
    let context = ExecutionContext::for_tests(14);
    for class in
        [Class::TemporalDag, Class::Cpdag, Class::Pag, Class::TemporalCpdag, Class::TemporalPag]
    {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let (builder, data): (_, Result<TabularData, TimeSeriesData>) = match class {
                Class::Cpdag => (Study::tabular(cpdag_table()).graph(cpdag()), Ok(cpdag_table())),
                Class::Pag => (Study::tabular(pag_table()).graph(pag()), Ok(pag_table())),
                Class::TemporalDag => {
                    (Study::series(tdag_series()).graph(tdag()), Err(tdag_series()))
                }
                Class::TemporalCpdag => {
                    (Study::series(class_series(false)).graph(tcpdag()), Err(class_series(false)))
                }
                Class::TemporalPag => {
                    (Study::series(class_series(true)).graph(tpag()), Err(class_series(true)))
                }
            };
            let builder = builder.query(level_query(class)).refute(suite).bootstrap_replicates(0);
            let study = builder.clone().build().unwrap();
            let prepared = study.prepare(&context).unwrap();
            drop(builder);
            drop(study);
            assert!(prepared.plan().logical.record.estimator.is_some(), "{class:?}");
            let result = match &data {
                Ok(table) => prepared.estimate(table, &context),
                Err(series) => prepared.estimate_series(series, &context),
            }
            .unwrap_or_else(|error| panic!("{class:?}: {error}"));
            assert!(!result.refutations.is_empty(), "{class:?} {suite:?}");
            for level in levels(&result) {
                assert!(
                    (level - truth.adjusted_level).abs() < truth.level_tolerance
                        || (level - truth.static_unadjusted).abs() < truth.level_tolerance
                        || (level - truth.temporal_unadjusted).abs() < truth.level_tolerance,
                    "{class:?}: level {level}"
                );
            }
        }
    }
}
