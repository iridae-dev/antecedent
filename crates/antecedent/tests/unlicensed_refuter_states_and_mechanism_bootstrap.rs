//! Refuter states and intervals that execute and stay unlicensed.
//!
//! * The Frequentist mechanism-refit row bootstrap of a static counterfactual, anomaly
//!   attribution and change attribution: a requested replicate count publishes a
//!   bootstrap standard error, keeps failed replicates as failed mass, does not depend
//!   on the thread count, and leases from the admission pool before it allocates.
//! * The class-aware (Cpdag) and `TemporalDag` intervention-level refuter states.
//! * The Bayesian joint intervention level on a `CoDetermined` tier closure.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, CounterfactualQuery,
    ExecutionContext, Intervention, Lag, PopulationSelector, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec,
    Value, VariableId,
};
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use antecedent_graph::{
    Cpdag, Dag, DenseNodeId, TemporalCpdag, TemporalDag, TieredBackground, WithinTier,
    ensure_lagged,
};

/// Deterministic uniform stream (xorshift64*).
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
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Columns `z, t, y`: `t = 0.6 z + N(0, 1)`, `y = 1 + 2 t + z + N(0, 0.5²)`.
fn linear_table(n: usize, seed: u64) -> TabularData {
    let mut stream = Stream(seed.max(1));
    let z: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let t: Vec<f64> = z.iter().map(|z| 0.6 * z + stream.normal()).collect();
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + z + 0.5 * stream.normal()).collect();
    let columns = [("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())];
    TabularData::from_f64_columns(columns).unwrap()
}

/// `z -> t`, `z -> y`, `t -> y` over `(z, t, y)`.
fn linear_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn counterfactual_study(data: TabularData, replicates: u32) -> Study {
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(1), Value::f64(1.0))]),
    )
    .with_control_level(0.0);
    Study::tabular(data)
        .graph(linear_dag())
        .query(CausalQuery::Counterfactual(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

fn bootstrap_se(result: &StudyResult) -> Option<f64> {
    result.estimate.as_effect().and_then(|effect| effect.uncertainty.bootstrap.se)
}

fn has_diagnostic(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
}

#[test]
fn frequentist_counterfactual_bootstrap_is_an_unlicensed_interval() {
    let data = linear_table(600, 3);
    let point = counterfactual_study(data.clone(), 0).run(&ExecutionContext::for_tests(3)).unwrap();
    assert_eq!(point.support_status, Some(CellStatus::Licensed));
    assert!(bootstrap_se(&point).is_none());

    let result =
        counterfactual_study(data.clone(), 40).run(&ExecutionContext::for_tests(3)).unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    let effect = result.estimate.as_effect().unwrap();
    // The linear law's unit contrast is the slope.
    assert!((effect.ate - 2.0).abs() < 0.2, "mean ITE {}", effect.ate);
    let se = bootstrap_se(&result).expect("bootstrap standard error");
    assert!(se > 0.0 && se < 0.5, "bootstrap SE {se}");
    assert_eq!(effect.uncertainty.bootstrap.replicates_ok, Some(40));
    assert_eq!(effect.uncertainty.bootstrap.replicates_failed, Some(0));
    assert!(has_diagnostic(&result, "gcm.frequentist_bootstrap"));
    assert!(!has_diagnostic(&result, "gcm.counterfactual.uncertainty_unavailable"));

    let threaded = |threads: u32| {
        bootstrap_se(
            &counterfactual_study(data.clone(), 20)
                .run(&ExecutionContext::production(5, threads))
                .unwrap(),
        )
        .unwrap()
    };
    assert_eq!(threaded(1).to_bits(), threaded(4).to_bits());
}

#[test]
fn mechanism_bootstrap_leases_before_allocating_and_restores_the_pool() {
    let data = linear_table(400, 5);
    let ctx = ExecutionContext::for_tests(5).with_memory_limit(Some(64));
    let before = ctx.admission().available().bytes();
    let error = counterfactual_study(data, 50).run(&ctx).unwrap_err();
    assert!(matches!(error, antecedent::CausalError::Resource { .. }), "{error}");
    assert_eq!(ctx.admission().available().bytes(), before);
}

#[test]
fn frequentist_attribution_bootstraps_publish_unlicensed_standard_errors() {
    // The anomaly scorer attributes at most `max_units` rows; the fixture fits that cap.
    let data = linear_table(100, 7);
    let anomaly = Study::tabular(data.clone())
        .graph(linear_dag())
        .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(2)],
            100,
        )))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(25)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(7))
        .unwrap();
    assert_eq!(anomaly.support_status, Some(CellStatus::Refused));
    // Known truth: `y` is Gaussian, so against its median / 1.4826·MAD reference the
    // two-sided tail probability is uniform and the score `-ln p` is Exp(1): mean 1.
    let mean_score = anomaly.estimate.as_effect().unwrap().ate;
    assert!((mean_score - 1.0).abs() < 0.35, "mean anomaly score {mean_score}");
    assert!(bootstrap_se(&anomaly).is_some_and(|se| se.is_finite() && se > 0.0));
    assert!(has_diagnostic(&anomaly, "gcm.frequentist_bootstrap"));

    // Two periods: the second shifts the outcome intercept by 3.
    let n = 200;
    let mut stream = Stream(11);
    let x: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let shift = if i < n / 2 { 0.0 } else { 3.0 };
            1.0 + shift + 2.0 * x[i] + 0.5 * stream.normal()
        })
        .collect();
    let table = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let change = Study::tabular(table)
        .graph(graph)
        .query(CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: n / 2 },
            PopulationSelector::TimeRange { start: n / 2, end: n },
        )))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(20)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap();
    assert_eq!(change.support_status, Some(CellStatus::Refused));
    let total = change.estimate.as_effect().unwrap().ate;
    assert!((total - 3.0).abs() < 0.6, "total change {total}");
    assert!(bootstrap_se(&change).is_some_and(|se| se > 0.0));
}

fn level_query(outcome: u32, treatment: u32) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(outcome),
        interventions: Arc::from([Intervention::set(
            VariableId::from_raw(treatment),
            Value::f64(1.0),
        )]),
    })
}

#[test]
fn cpdag_intervention_level_runs_its_per_completion_refuter_state() {
    let data = linear_table(800, 13);
    // `z - t` undirected: one completion adjusts `z`, the other adjusts nothing.
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let result = Study::tabular(data.clone())
            .graph(cpdag.clone())
            .query(CausalQuery::Response(level_query(2, 1)))
            .inference(InferenceMode::Frequentist)
            .refute(suite)
            .build()
            .unwrap_or_else(|error| panic!("{suite:?}: {error}"))
            .run(&ExecutionContext::for_tests(13))
            .unwrap_or_else(|error| panic!("{suite:?}: {error}"));
        assert_eq!(result.support_status, Some(CellStatus::Refused), "{suite:?}");
        assert!(!result.refutations.is_empty(), "{suite:?}");
        assert!(has_diagnostic(&result, "refute.response.class_envelope_disagreement"));
        assert!(has_diagnostic(&result, "refute.response.class_state"));
        assert!(!has_diagnostic(&result, "refute.response.skipped"));
    }
}

#[test]
fn temporal_dag_intervention_level_runs_its_pulse_witness_state() {
    let n: usize = 400;
    let mut stream = Stream(17);
    let t: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j]) + 0.5 * stream.normal())
        .collect();
    let series =
        TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    let query = level_query(1, 0).with_temporal(
        TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap(),
    );
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128)),
    ] {
        let label = format!("{inference:?}");
        let result = Study::series(series.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(query.clone()))
            .inference(inference)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap_or_else(|error| panic!("{label}: {error}"))
            .run(&ExecutionContext::for_tests(17))
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(result.support_status, Some(CellStatus::Refused), "{label}");
        assert!(!result.refutations.is_empty(), "{label}");
        assert!(has_diagnostic(&result, "refute.temporal_response.pulse_witness_state"), "{label}");
    }
}

#[test]
fn codetermined_bayesian_joint_level_executes_unlicensed() {
    let n = 1_200;
    let mut stream = Stream(19);
    let z: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let t1: Vec<f64> = z
        .iter()
        .map(|z| f64::from(u8::from(stream.uniform() < 1.0 / (1.0 + (-z).exp()))))
        .collect();
    let t2: Vec<f64> = z
        .iter()
        .map(|z| f64::from(u8::from(stream.uniform() < 1.0 / (1.0 + (0.5 * z).exp()))))
        .collect();
    let y: Vec<f64> =
        (0..n).map(|i| 1.0 + 2.0 * t1[i] + t2[i] + 1.5 * z[i] + 0.5 * stream.normal()).collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mean_z = z.iter().sum::<f64>() / n as f64;
    let background = TieredBackground::from_named(
        data.schema(),
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(3),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
        ]),
    });
    let result = Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(19))
        .unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    let level = match &result.response().expect("response").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected a scalar joint level, got {other:?}"),
    };
    let truth = 4.0 + 1.5 * mean_z;
    assert!((level - truth).abs() < 0.2, "joint level {level} vs {truth}");
    assert!(has_diagnostic(&result, "estimate.codetermined.bayesian_joint"));
}

/// Series `x, y, z`: `x = 0.4 z + N(0, 0.5²)`, `y = 0.8 x@-1 + N(0, 0.35²)`.
fn lagged_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut stream = Stream(seed.max(1));
    let z: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let x: Vec<f64> = z.iter().map(|z| 0.4 * z + 0.5 * stream.normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.8 * i.checked_sub(1).map_or(0.0, |j| x[j]) + 0.35 * stream.normal())
        .collect();
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

/// `x@-1 -> y`, `z@-1 -> y`, `z@-1 - x@-1` undirected.
fn lagged_cpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let x1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_undirected(z1, x1).unwrap();
    graph
}

#[test]
fn temporal_class_intervention_level_runs_per_completion_witness_state() {
    let series = lagged_series(500, 23);
    let query = level_query(1, 0).with_temporal(
        TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap(),
    );
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128)),
    ] {
        let label = format!("{inference:?}");
        let result = Study::series(series.clone())
            .graph(lagged_cpdag())
            .query(CausalQuery::Response(query.clone()))
            .inference(inference)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap_or_else(|error| panic!("{label}: {error}"))
            .run(&ExecutionContext::for_tests(23))
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(result.support_status, Some(CellStatus::Refused), "{label}");
        assert!(!result.refutations.is_empty(), "{label}");
        assert!(
            has_diagnostic(&result, "refute.temporal_response.class_envelope_disagreement"),
            "{label}"
        );
        assert!(has_diagnostic(&result, "refute.temporal_response.pulse_witness_state"), "{label}");
    }
}

#[test]
fn temporal_derivative_runs_its_refuter_state_on_dag_and_class() {
    let series = lagged_series(500, 29);
    let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(1),
        treatment: VariableId::from_raw(0),
        at: 0.0,
        order: 1,
        scale: antecedent_core::DerivativeScale::Identity,
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());
    let mut dag = TemporalDag::empty();
    let x1 = ensure_lagged(&mut dag, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut dag, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    dag.insert_directed(x1, y0).unwrap();
    for (class, suite) in
        [(false, RefuteSuite::Cheap), (false, RefuteSuite::Full), (true, RefuteSuite::Cheap)]
    {
        let label = format!("class={class} {suite:?}");
        let builder = Study::series(series.clone());
        let builder =
            if class { builder.graph(lagged_cpdag()) } else { builder.graph(dag.clone()) };
        let result = builder
            .query(CausalQuery::Response(query.clone()))
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap_or_else(|error| panic!("{label}: {error}"))
            .run(&ExecutionContext::for_tests(29))
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(result.support_status, Some(CellStatus::Refused), "{label}");
        let fd = result
            .refutations
            .iter()
            .find(|report| report.refuter.as_ref() == "derivative.finite_difference")
            .unwrap_or_else(|| panic!("{label}: finite-difference check"));
        assert!((fd.refuted_ate - 0.8).abs() < 0.15, "{label}: {}", fd.refuted_ate);
        assert!(has_diagnostic(&result, "refute.temporal_response.pulse_witness_state"), "{label}");
    }
}

// ---------------------------------------------------------------------------
// Frequentist/Bayesian symmetry of the counterpart inventory.
// ---------------------------------------------------------------------------

const INVENTORY: &str = include_str!("../../../parity/counterparts.toml");

/// One method row of the inventory.
#[derive(Clone, Debug, Default)]
struct MethodRow {
    id: String,
    method: String,
    design: String,
    inference: String,
    status: String,
    uncertainty: String,
}

fn quoted(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?.trim();
    let inner = rest.strip_prefix('"')?;
    Some(inner[..inner.rfind('"')?].to_owned())
}

fn method_rows() -> Vec<MethodRow> {
    let mut rows = Vec::new();
    let mut current = std::collections::HashMap::<String, String>::new();
    let mut flush = |row: &std::collections::HashMap<String, String>| {
        if row.get("kind").map(String::as_str) == Some("method") {
            let get = |key: &str| row.get(key).cloned().unwrap_or_default();
            rows.push(MethodRow {
                id: get("id"),
                method: get("method"),
                design: get("design"),
                inference: get("inference"),
                status: get("status"),
                uncertainty: get("uncertainty"),
            });
        }
    };
    for line in INVENTORY.lines() {
        if line.starts_with("[[row]]") {
            flush(&current);
            current.clear();
            continue;
        }
        for key in ["id", "kind", "method", "design", "inference", "status", "uncertainty"] {
            if let Some(value) = quoted(line, key) {
                current.insert(key.to_owned(), value);
            }
        }
    }
    flush(&current);
    rows
}

fn executes(status: &str) -> bool {
    matches!(status, "licensed" | "executing_unlicensed")
}

/// Bayesian method rows whose uncertainty is not an ordinary posterior, with what they report
/// instead. A generalized posterior (a Bayesian bootstrap, block weights, a class envelope of
/// per-completion posteriors) reports resampling or an analytic enclosure; a programme
/// (discovery, a decision search, an audit) has no interval of its own.
const NOT_AN_ORDINARY_POSTERIOR: &[(&str, &str)] = &[
    ("P05.static.bayesian", "none"),
    ("P05.temporal.bayesian", "none"),
    ("P06.temporal.bayesian", "design_aware_resampling"),
    ("P07.curve.temporal.bayesian", "analytic"),
    ("P07.derivative.temporal.bayesian", "design_aware_resampling"),
    ("P12.counterfactual.temporal.bayesian", "design_aware_resampling"),
    ("P12.intervention.temporal.bayesian", "analytic"),
    ("P14.temporal.bayesian", "design_aware_resampling"),
    ("P15.conditional.static.bayesian", "design_aware_resampling"),
    ("P15.conditional.temporal.bayesian", "design_aware_resampling"),
    ("P15.projection.temporal.bayesian", "design_aware_resampling"),
    ("P15.pulse.temporal.bayesian", "design_aware_resampling"),
    ("P15.sustained.temporal.bayesian", "design_aware_resampling"),
    ("P19.static.bayesian", "none"),
    ("P19.temporal.bayesian", "none"),
    ("P20.static.bayesian", "none"),
    ("P20.temporal.bayesian", "none"),
    ("X01.temporal.bayesian", "analytic"),
    ("X04.1.static.bayesian", "none"),
    ("X04.1.temporal.bayesian", "none"),
    ("X04.2.static.bayesian", "none"),
    ("X04.2.temporal.bayesian", "none"),
];

/// Licensed frequentist rows that report no interval of their own by default. Their licence
/// covers the point value; the sampling interval is the requested mechanism-refit row bootstrap
/// of this file, which executes and stays unlicensed, and the Bayesian side reports a
/// posterior over mechanisms. The list only shrinks: a licensed row with no interval anywhere
/// else is a geometric support cell, not a licence.
const LICENSED_WITHOUT_DEFAULT_INTERVAL: &[&str] =
    &["P12.counterfactual.static.frequentist", "P13.attribution.static.frequentist"];

/// Quadrants that execute on the Bayesian side only because the target does not exist for a
/// sampling distribution: updating a prior by an experiment is a Bayesian operation and a
/// bootstrap law is not a prior. The frequentist row must record that as a theory boundary.
const BAYESIAN_ONLY_BY_THEORY: &[&str] = &["X05.temporal"];

/// Quadrants whose two inference sides differ in licence: the licensed side has a measured
/// or asserted interval, the other executes without a licence.
const LICENCE_ASYMMETRIES: &[&str] = &["P04.static", "P09.interference.static"];

#[test]
fn every_method_executes_on_both_inference_sides_or_records_why_not_on_both() {
    let rows = method_rows();
    assert!(rows.len() >= 200, "the inventory lists {} method rows", rows.len());
    let mut sides = std::collections::BTreeMap::<(String, String), Vec<&MethodRow>>::new();
    for row in &rows {
        sides.entry((row.method.clone(), row.design.clone())).or_default().push(row);
    }
    let mut asymmetric = Vec::new();
    for ((method, design), pair) in &sides {
        let frequentist = pair.iter().find(|r| r.inference == "frequentist");
        let bayesian = pair.iter().find(|r| r.inference == "bayesian");
        let (Some(frequentist), Some(bayesian)) = (frequentist, bayesian) else {
            panic!("{method} {design} has a single inference side in the inventory");
        };
        // A method either executes on both sides or neither does, and a side the theory rules
        // out is ruled out with a recorded reason on that side.
        if BAYESIAN_ONLY_BY_THEORY.contains(&format!("{method}.{design}").as_str()) {
            assert!(
                !executes(&frequentist.status) && executes(&bayesian.status),
                "{method} {design} is pinned as Bayesian-only"
            );
            assert_eq!(frequentist.status, "not_applicable_with_theory_reason");
            continue;
        }
        assert_eq!(
            executes(&frequentist.status),
            executes(&bayesian.status),
            "{method} {design}: {} vs {}",
            frequentist.status,
            bayesian.status
        );
        if (frequentist.status == "licensed") != (bayesian.status == "licensed") {
            asymmetric.push(format!("{method}.{design}"));
        }
    }
    // A licence is one-sided in exactly the recorded quadrants.
    asymmetric.sort();
    let mut expected: Vec<String> = LICENCE_ASYMMETRIES.iter().map(|s| (*s).to_owned()).collect();
    expected.sort();
    assert_eq!(asymmetric, expected, "one-sided licences in the inventory");
}

#[test]
fn bayesian_rows_that_are_not_ordinary_posteriors_are_exactly_the_classified_ones() {
    let rows = method_rows();
    let mut found: Vec<(String, String)> = rows
        .iter()
        .filter(|row| {
            row.inference == "bayesian" && executes(&row.status) && row.uncertainty != "posterior"
        })
        .map(|row| (row.id.clone(), row.uncertainty.clone()))
        .collect();
    found.sort();
    let mut expected: Vec<(String, String)> = NOT_AN_ORDINARY_POSTERIOR
        .iter()
        .map(|(id, kind)| ((*id).to_owned(), (*kind).to_owned()))
        .collect();
    expected.sort();
    assert_eq!(found, expected, "classify every executing Bayesian row that is not a posterior");
    // A licensed row reports an interval or enclosure of its own, except the pinned rows whose
    // interval is the requested unlicensed bootstrap.
    let mut bare: Vec<&str> = rows
        .iter()
        .filter(|row| row.status == "licensed" && row.uncertainty == "none")
        .map(|row| row.id.as_str())
        .collect();
    bare.sort_unstable();
    assert_eq!(bare, LICENSED_WITHOUT_DEFAULT_INTERVAL, "licensed rows with no uncertainty report");
    // A frequentist row that executes reports an interval of its own wherever the Bayesian
    // side reports a posterior, except the programmes that have none on either side.
    for bayesian in rows.iter().filter(|r| {
        r.inference == "bayesian" && executes(&r.status) && r.uncertainty == "posterior"
    }) {
        if BAYESIAN_ONLY_BY_THEORY.contains(&format!("{}.{}", bayesian.method, bayesian.design).as_str()) {
            continue;
        }
        let frequentist = rows
            .iter()
            .find(|r| {
                r.inference == "frequentist"
                    && r.method == bayesian.method
                    && r.design == bayesian.design
            })
            .unwrap();
        assert!(
            executes(&frequentist.status),
            "{} has a posterior but {} does not execute",
            bayesian.id,
            frequentist.id
        );
    }
}
