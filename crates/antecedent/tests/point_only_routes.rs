//! Routes whose published answer is a point value: each test runs the route the way its
//! retained-plan evidence does and asserts that the result carries no interval, no standard
//! error and no posterior, with the point itself finite.
//!
//! The statements are about the default configuration of the route. A requested replicate
//! count or a Bayesian inference mode is a different procedure with its own record.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(deprecated, reason = "the direct facade path is the oracle this file checks")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test fixtures: small exact indices and one coordinate walk per test"
)]

mod common;

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent::analysis::{RetainedOperation, StudyBuilder};
use antecedent::period_refresh::{PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec};
use antecedent::{
    AcceptedGraph, EstimatorId, IdentifierId, InferenceMode, IntoGraphInput, PreparedStudy,
    RefuteSuite, Study,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, CounterfactualQuery,
    ExecutionContext, IntervalMethod, Intervention, NestedCounterfactualQuery, PopulationSelector,
    Value, VariableId,
};
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    ActionOutcome, MeanEstimate, MeanSource, evaluate_contract_on_means,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, HistoryStep, ObservationChannel,
};
use antecedent_design::transported_policy::{
    MechanismStatus, ModelMechanism, SelectionDiagram, TransportedPolicyEvaluation,
    TransportedPolicyRequest, evaluate_transported_policy,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge};
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
};
use antecedent_graph::{Admg, Dag, DenseNodeId, TemporalAdmg};
use antecedent_prob::InferenceDiagnostics;
use antecedent_state::{ConjugateOlsPrior, OlsRefresh, OlsSnapshot, RefreshRow};
use common::persistent_dgp::{autoregressive_pulse_dag, autoregressive_pulse_series, pulse};

// ---------------------------------------------------------------------------
// What a result publishes as an interval
// ---------------------------------------------------------------------------

/// The interval a result publishes, read from the only quantities it publishes that could form
/// one.
trait PublishedInterval {
    fn interval(&self) -> Option<(f64, f64)>;
}

/// An action outcome forms an interval only from its standard error.
impl PublishedInterval for ActionOutcome {
    fn interval(&self) -> Option<(f64, f64)> {
        let standard_error = self.standard_error?;
        let half = 1.96 * standard_error;
        Some((self.value - half, self.value + half))
    }
}

/// A transported value declares its inferential claim; only a point-only claim forms no interval,
/// and any other claim fails the test that reads it.
impl PublishedInterval for TransportedPolicyEvaluation {
    fn interval(&self) -> Option<(f64, f64)> {
        match self.inference {
            "point_only" => None,
            other => panic!("the evaluation claims `{other}`, not a point value"),
        }
    }
}

/// A least-squares summary names every field it publishes; none is a standard error or an
/// interval, and a field added later fails to compile here until it is read.
impl PublishedInterval for OlsSnapshot {
    fn interval(&self) -> Option<(f64, f64)> {
        let Self {
            source: _,
            n: _,
            xtx: _,
            xty: _,
            yty: _,
            beta: _,
            residual_variance: _,
            source_digest: _,
            ledger: _,
        } = self;
        None
    }
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    antecedent::support::caller_thread().stack_size(64 << 20).spawn(body).unwrap().join().unwrap();
}

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

// ---------------------------------------------------------------------------
// Static DAG attribution
// ---------------------------------------------------------------------------

fn two_node_dag() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph
}

fn anomaly_data(outlier: f64) -> TabularData {
    let x: Vec<_> = (0..20).map(f64::from).collect();
    let y: Vec<_> = (0..20).map(|i| if i == 19 { outlier } else { 2.0 * f64::from(i) }).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn change_data(second_shift: f64) -> TabularData {
    let x: Vec<_> = (0..80).map(|i| f64::from(i % 40) * 0.1).collect();
    let y: Vec<_> = (0..80)
        .map(|i| if i < 40 { 1.0 + 2.0 * x[i] } else { 6.0 + second_shift + 2.0 * x[i] })
        .collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

#[test]
fn the_default_frequentist_dag_anomaly_attribution_publishes_scores_and_no_interval() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(13);
        let data = anomaly_data(200.0);
        let query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(1)], 100));
        // No replicate count is requested: the route runs its point procedure.
        let anomaly_builder = Study::tabular(data.clone())
            .graph(two_node_dag())
            .query(query.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let prepared = anomaly_builder.prepare(&ctx).unwrap();
        let RetainedOperation::Attribution(plan) = prepared.retained_operation() else {
            panic!("sealed anomaly operation")
        };
        assert_eq!(plan.query, query);
        assert_eq!(plan.identifier, IdentifierId::GcmParametric);
        assert_eq!(plan.estimator, EstimatorId::GcmFit);
        drop(anomaly_builder);
        let result = prepared.estimate(&data, &ctx).unwrap();
        let scores = result.anomaly().expect("anomaly result");
        let score = scores.iter().find(|score| score.target == v(1)).unwrap();
        assert!(!score.scores.is_empty());
        assert!(score.scores.iter().all(|value| value.is_finite()), "the scores are points");
        let top = score.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        assert_eq!(score.rows[top.0], 19, "the injected outlier is the top anomaly");
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "a point attribution reports no interval"
        );
        assert!(result.posterior.is_none(), "a frequentist attribution has no posterior");
    });
}

#[test]
fn the_default_frequentist_dag_change_attribution_publishes_the_change_and_no_interval() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(13);
        let data = change_data(0.0);
        let query = CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
            v(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        ));
        // No replicate count is requested: the route runs its point procedure.
        let change_builder = Study::tabular(data.clone())
            .graph(two_node_dag())
            .query(query.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let prepared = change_builder.prepare(&ctx).unwrap();
        let RetainedOperation::Attribution(plan) = prepared.retained_operation() else {
            panic!("sealed change operation")
        };
        assert_eq!(plan.query, query);
        assert_eq!(plan.identifier, IdentifierId::GcmParametric);
        assert_eq!(plan.estimator, EstimatorId::GcmFit);
        drop(change_builder);
        let result = prepared.estimate(&data, &ctx).unwrap();
        let total_change = result.change_attribution().unwrap().total_change;
        assert!(total_change.is_finite());
        assert!((total_change - 5.0).abs() < 1e-8, "known population mean shift: {total_change}");
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "a point attribution reports no interval"
        );
        assert!(result.posterior.is_none(), "a frequentist attribution has no posterior");
    });
}

// ---------------------------------------------------------------------------
// Cross-world effects on a latent-confounded class
// ---------------------------------------------------------------------------

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_class_truth/expected.json"
    ))
    .unwrap()
}

fn count_table(pin: &serde_json::Value, name: &str) -> TabularData {
    let spec = &pin["tables"][name];
    let names: Vec<String> = spec["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .collect();
    let mut data: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for cell in spec["counts"].as_array().unwrap() {
        let levels: Vec<f64> =
            cell["levels"].as_array().unwrap().iter().map(|l| l.as_f64().unwrap()).collect();
        for _ in 0..cell["count"].as_u64().unwrap() {
            for (column, level) in data.iter_mut().zip(&levels) {
                column.push(*level);
            }
        }
    }
    TabularData::from_f64_columns(
        names.iter().zip(&data).map(|(name, column)| (name.as_str(), column.as_slice())),
    )
    .unwrap()
}

fn admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for &(a, b) in directed {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

fn counterfactual(treatment: u32, outcome: u32) -> CausalQuery {
    CausalQuery::Counterfactual(CounterfactualQuery::new(
        v(outcome),
        Arc::from([Intervention::set(v(treatment), Value::f64(1.0))]),
    ))
}

fn nested(treatment: u32, mediator: u32, outcome: u32) -> CausalQuery {
    CausalQuery::NestedCounterfactual(
        NestedCounterfactualQuery::new(v(treatment), v(mediator), v(outcome)).unwrap(),
    )
}

/// A frequentist class study with no replicate count and no validation suite.
fn class_study(
    data: TabularData,
    graph: impl IntoGraphInput + Into<AcceptedGraph>,
    accepted: bool,
    query: CausalQuery,
) -> StudyBuilder {
    let base = Study::tabular(data);
    let base = if accepted { base.graph(graph.into()) } else { base.graph(graph) };
    base.query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

#[test]
fn a_frequentist_admg_counterfactual_publishes_the_point_and_no_interval() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(7);
    let truth = pin["cases"]["admg_counterfactual"]["expected"].as_f64().unwrap();
    for accepted in [false, true] {
        let data = count_table(&pin, "front_door");
        let class_builder = class_study(
            data.clone(),
            admg(3, &[(0, 1), (1, 2)], &[(0, 2)]),
            accepted,
            counterfactual(0, 2),
        );
        let study = class_builder.clone().build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        drop(class_builder);
        drop(study);
        assert!(matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld));
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.effect().is_finite());
        assert!((result.effect() - truth).abs() <= 1e-9, "accepted={accepted}");
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "accepted={accepted}: a point cross-world mean reports no interval"
        );
        assert!(result.posterior.is_none(), "accepted={accepted}");
    }
}

#[test]
fn a_frequentist_temporal_admg_counterfactual_publishes_the_point_and_no_interval() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(7);
    let truth = pin["cases"]["admg_counterfactual"]["expected"].as_f64().unwrap();
    for accepted in [false, true] {
        let data = count_table(&pin, "front_door");
        let temporal =
            TemporalAdmg::from_static_contemporaneous(&admg(3, &[(0, 1), (1, 2)], &[(0, 2)]))
                .unwrap();
        let class_builder = class_study(data.clone(), temporal, accepted, counterfactual(0, 2));
        let study = class_builder.clone().build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        drop(class_builder);
        drop(study);
        assert!(matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld));
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.effect().is_finite());
        assert!((result.effect() - truth).abs() <= 1e-9, "accepted={accepted}");
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "accepted={accepted}: a point cross-world mean reports no interval"
        );
        assert!(result.posterior.is_none(), "accepted={accepted}");
    }
}

#[test]
fn a_frequentist_admg_natural_direct_effect_publishes_the_point_and_no_interval() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(7);
    let truth = pin["cases"]["admg_natural_direct"]["expected"].as_f64().unwrap();
    for accepted in [false, true] {
        let data = count_table(&pin, "front_door");
        let class_builder = class_study(
            data.clone(),
            admg(3, &[(0, 1), (0, 2), (1, 2)], &[]),
            accepted,
            nested(0, 1, 2),
        );
        let study = class_builder.clone().build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        drop(class_builder);
        drop(study);
        assert!(matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld));
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.effect().is_finite());
        assert!((result.effect() - truth).abs() <= 1e-9, "accepted={accepted}");
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "accepted={accepted}: a point natural direct effect reports no interval"
        );
        assert!(result.posterior.is_none(), "accepted={accepted}");
    }
}

#[test]
fn a_frequentist_temporal_admg_natural_direct_effect_publishes_the_point_and_no_interval() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(7);
    let truth = pin["cases"]["admg_natural_direct"]["expected"].as_f64().unwrap();
    for accepted in [false, true] {
        let data = count_table(&pin, "front_door");
        let temporal =
            TemporalAdmg::from_static_contemporaneous(&admg(3, &[(0, 1), (0, 2), (1, 2)], &[]))
                .unwrap();
        let class_builder = class_study(data.clone(), temporal, accepted, nested(0, 1, 2));
        let study = class_builder.clone().build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        drop(class_builder);
        drop(study);
        assert!(matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld));
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.effect().is_finite());
        assert!((result.effect() - truth).abs() <= 1e-9, "accepted={accepted}");
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "accepted={accepted}: a point natural direct effect reports no interval"
        );
        assert!(result.posterior.is_none(), "accepted={accepted}");
    }
}

// ---------------------------------------------------------------------------
// Counterfactual on a supplied DAG
// ---------------------------------------------------------------------------

#[test]
fn a_frequentist_dag_counterfactual_publishes_the_mean_effect_and_no_interval() {
    let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 20.0 - 2.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + 0.01 * (i as f64).sin()).collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let query = CounterfactualQuery::new(v(1), [Intervention::set(v(0), Value::f64(1.0))]);
    let ctx = ExecutionContext::for_tests(59);
    for accepted in [false, true] {
        // No replicate count is requested: the route runs its point procedure.
        let dag_builder = if accepted {
            Study::tabular(data.clone()).graph(AcceptedGraph::from(two_node_dag()))
        } else {
            Study::tabular(data.clone()).graph(two_node_dag())
        }
        .query(CausalQuery::Counterfactual(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
        let prepared = dag_builder.prepare(&ctx).unwrap();
        drop(dag_builder);
        let RetainedOperation::Counterfactual(plan) = prepared.retained_operation() else {
            panic!("retained checked plan")
        };
        assert_eq!(plan.query(), &query);
        let result = prepared.estimate(&data, &ctx).unwrap();
        let effect = result.counterfactual().unwrap();
        assert!(effect.mean_ite.is_finite());
        assert!((effect.mean_ite - 2.0).abs() < 0.001, "mean_ite={}", effect.mean_ite);
        assert!(
            effect.unit_effect_intervals.is_none(),
            "accepted={accepted}: no per-unit interval is formed"
        );
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "accepted={accepted}: a point counterfactual reports no interval"
        );
        assert!(result.posterior.is_none(), "accepted={accepted}");
    }
}

// ---------------------------------------------------------------------------
// Natural direct effect over a posterior of DAGs
// ---------------------------------------------------------------------------

fn dag_mask(n: usize, edges: &[(usize, usize)]) -> u64 {
    edges.iter().fold(0_u64, |m, &(a, b)| set_edge(m, n, a, b, true))
}

fn three_node_posterior() -> GraphPosterior {
    let cyclic = dag_mask(3, &[(0, 1), (1, 2), (2, 0)]);
    let masks =
        vec![dag_mask(3, &[(0, 1), (0, 2), (1, 2)]), dag_mask(3, &[(0, 2), (1, 2)]), cyclic];
    GraphPosterior::new(
        3,
        vec![0.5, 0.3, 0.2],
        masks,
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("x"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag)
}

#[test]
fn a_frequentist_dag_posterior_natural_direct_effect_publishes_the_point_and_no_interval() {
    on_large_stack(|| {
        let pin = pin();
        let expected = pin["posterior"]["expected"].as_f64().unwrap();
        let data = count_table(&pin, "front_door");
        let ctx = ExecutionContext::for_tests(4);
        let posterior_builder = Study::tabular(data.clone())
            .graph_posterior(three_node_posterior())
            .query(nested(0, 1, 2))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        let study = posterior_builder.clone().build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        drop(posterior_builder);
        drop(study);
        let operation = format!("{:?}", prepared.retained_operation());
        assert!(
            !matches!(prepared.retained_operation(), RetainedOperation::FreshRun),
            "{operation}"
        );
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.effect().is_finite());
        assert!((result.effect() - expected).abs() <= 1e-9, "{}", result.effect());
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::None),
            "a frequentist posterior-weighted point reports no interval"
        );
        assert!(result.posterior.is_none(), "a frequentist study publishes no posterior");
    });
}

// ---------------------------------------------------------------------------
// Decision from means
// ---------------------------------------------------------------------------

fn quantity(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// `E[Y | do(a)] = 1 + 2a` on the grid `a = 0, 1, 2`.
fn mean_source() -> MeanSource {
    MeanSource {
        coordinates: ["do(a=0)", "do(a=1)", "do(a=2)"]
            .iter()
            .map(|regime| quantity(regime, "mean"))
            .collect(),
        means: vec![1.0, 3.0, 5.0],
        provider_id: "lab".into(),
        snapshot_id: "snap-9".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
        estimate: MeanEstimate::Frequentist,
    }
}

/// `2 * x0 - 1`.
fn affine() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
        UtilityExpr::Const(1.0),
    )
}

fn decision_contract() -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "wait".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=0)", "mean")],
                utility: affine(),
            },
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=2)", "mean")],
                utility: affine(),
            },
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn a_frequentist_decision_from_means_publishes_the_values_and_no_interval() {
    let result = evaluate_contract_on_means(&decision_contract(), &mean_source()).unwrap();
    assert_eq!(result.actions.len(), 2);
    // 2 * 1 - 1 = 1 and 2 * 5 - 1 = 9, in declaration order.
    assert!((result.actions[0].value - 1.0).abs() < 1e-12);
    assert!((result.actions[1].value - 9.0).abs() < 1e-12);
    for outcome in &result.actions {
        assert!(outcome.value.is_finite());
        assert!(outcome.interval().is_none(), "{}: a point value forms no interval", outcome.id);
        assert_eq!(outcome.standard_error, None);
        assert_eq!(outcome.expected_regret, None);
        assert_eq!(outcome.max_regret, None);
    }
    assert_eq!(result.evpi, None);
    assert_eq!(result.n_draws, 0);
}

// ---------------------------------------------------------------------------
// Transported policy value
// ---------------------------------------------------------------------------

const INSPECT: usize = 0;
const SHIP: usize = 1;
const HOLD: usize = 2;

fn commit(label: &str, immediate: [f64; 2]) -> ActionSpec {
    ActionSpec {
        label: label.into(),
        role: ActionRole::Commit,
        utility_cost: 0.0,
        resource_use: 0.0,
        transition: vec![1.0, 0.0, 0.0, 1.0],
        channel: None,
        immediate_utility: immediate.to_vec(),
        delayed: None,
    }
}

fn inspection_model(
    alarm: antecedent_core::measurement_model::MeasurementModel,
) -> FiniteSequentialModel {
    FiniteSequentialModel {
        label: "inspection".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: vec![0.5, 0.5],
        horizon: 2,
        actions: vec![
            ActionSpec {
                label: "inspect".into(),
                role: ActionRole::Continue,
                utility_cost: 0.0,
                resource_use: 0.0,
                transition: vec![1.0, 0.0, 0.0, 1.0],
                channel: Some(ObservationChannel { measurement: alarm }),
                immediate_utility: vec![0.0, 0.0],
                delayed: None,
            },
            commit("ship", [10.0, -10.0]),
            commit("hold", [0.0, 0.0]),
        ],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: Vec::new(),
        support: ActionSupport::everywhere(3),
    }
}

fn inspection_alarm() -> antecedent_core::measurement_model::MeasurementModel {
    use antecedent_core::measurement_model::{
        ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
        MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
        ParameterEvidence, RepresentationVersion,
    };
    MeasurementModel {
        model_id: "inspection_alarm".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "alarm".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "category".into(),
        },
        construct: MeasurementVariable {
            id: "unit_condition".into(),
            kind: MeasurementKind::LatentScientific,
            units: "category".into(),
        },
        binding: MeasurementBinding {
            population_id: "plant_a".into(),
            regime_id: "operations".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.75,
                specificity: 0.75,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion {
            coordinate_system_id: "alarm_panel".into(),
            version: 1,
        },
        evidence: ParameterEvidence::Declared { declarer: "reliability_engineering".into() },
    }
}

fn inspect_then_ship_on_quiet() -> ContingentPolicy {
    let after =
        |observation: usize| HistoryKey::from_steps([HistoryStep { action: INSPECT, observation }]);
    ContingentPolicy::new("inspect_then_ship_on_quiet")
        .decide(HistoryKey::root(), INSPECT)
        .decide(after(0), SHIP)
        .decide(after(1), HOLD)
}

fn all_invariant_but_initial(
    model: &FiniteSequentialModel,
) -> Vec<(ModelMechanism, MechanismStatus)> {
    let mut rows = vec![
        (ModelMechanism::InitialState, MechanismStatus::Shifted),
        (ModelMechanism::Utility, MechanismStatus::Invariant),
    ];
    for (action, spec) in model.actions.iter().enumerate() {
        rows.push((ModelMechanism::Transition { action }, MechanismStatus::Invariant));
        if spec.channel.is_some() {
            rows.push((ModelMechanism::Observation { action }, MechanismStatus::Invariant));
        }
    }
    rows
}

#[test]
fn a_transported_policy_value_is_a_finite_point_and_no_interval() {
    let source = inspection_model(inspection_alarm());
    let policy = inspect_then_ship_on_quiet();
    let diagram = SelectionDiagram {
        source_population: "plant_a".into(),
        target_population: "plant_b".into(),
        mechanisms: all_invariant_but_initial(&source),
    };
    let ctx = ExecutionContext::for_tests(5);
    for faulty in [0.0, 0.2, 0.5, 0.8, 1.0] {
        let law = [1.0 - faulty, faulty];
        let request = TransportedPolicyRequest {
            source: &source,
            policy: &policy,
            target_population: "plant_b",
            target_initial: Some(&law),
            diagram: Some(&diagram),
            target_utility: None,
            limits: EvaluationLimits::default(),
        };
        let evaluation = evaluate_transported_policy(&request, &ctx).unwrap();
        let hand = 7.5 - 10.0 * faulty;
        assert!(evaluation.target_value.is_finite());
        assert!(
            (evaluation.target_value - hand).abs() < 1e-12,
            "p = {faulty}: {} vs {hand}",
            evaluation.target_value
        );
        assert!(
            evaluation.interval().is_none(),
            "p = {faulty}: a transported value forms no interval"
        );
    }
}

// ---------------------------------------------------------------------------
// Least-squares refresh
// ---------------------------------------------------------------------------

const X: [f64; 12] = [0.5, 1.2, -0.3, 2.0, 1.5, -1.0, 0.8, 2.2, 0.1, 1.7, -0.6, 0.3];
const Y: [f64; 12] = [1.1, 2.4, 0.2, 3.9, 3.1, -0.9, 1.7, 4.6, 0.9, 3.2, -0.1, 1.0];

fn refresh_row(id: usize) -> RefreshRow {
    RefreshRow::new(id as u64, vec![1.0, X[id]], Y[id])
}

fn refresh_rows(range: std::ops::Range<usize>) -> Vec<RefreshRow> {
    range.map(refresh_row).collect()
}

#[test]
fn an_appended_least_squares_refresh_publishes_finite_coefficients_and_no_interval() {
    let mut fit = OlsRefresh::new("series_a", 2).unwrap();
    for id in 0..5 {
        fit.append_row("series_a", &refresh_row(id)).unwrap();
    }
    fit.append_window("series_a", &refresh_rows(5..9)).unwrap();
    fit.append_window("series_a", &refresh_rows(9..12)).unwrap();
    assert_eq!(fit.recomputes(), 0, "appends are not rebuilds");
    let snapshot = fit.snapshot();
    assert_eq!(snapshot.n, 12);
    let beta = snapshot.beta.as_ref().expect("coefficients");
    assert_eq!(beta.len(), 2);
    assert!(beta.iter().all(|value| value.is_finite()));
    assert!(snapshot.interval().is_none(), "the snapshot carries no standard error or interval");
    fit.check_against_fresh().expect("cached result equals the fresh rebuild");
}

// ---------------------------------------------------------------------------
// Period refresh of an autoregressive mechanism
// ---------------------------------------------------------------------------

const N: usize = 1200;
const SPLIT: usize = 600;
const LAGS: usize = 2;

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn window(start: i64, end: i64, snapshot: &str) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "ar_mechanism".into(),
        horizon: 2,
        lag_alignment: vec![("t".into(), 1)],
        intervention_history: vec!["a1=1".into(), "a2=0".into()],
        selection_targets: set(&["s_t1"]),
        regimes: set(&["observational"]),
        period: ObservationPeriod { start, end },
        unit_ids: set(&["u1", "u2"]),
        snapshot_id: snapshot.into(),
        proof_id: "proof_1".into(),
    }
}

fn cadence() -> ObservationCadence {
    ObservationCadence::new(1).unwrap()
}

fn refresh_spec(standing: MechanismStanding) -> PeriodRefreshSpec<'static> {
    PeriodRefreshSpec {
        series_id: "t",
        cadence: cadence(),
        initial_law: "law_1",
        standing,
        lags: LAGS,
    }
}

fn prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(
        vec![0.0; 3],
        vec![0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3],
        3.0,
        2.0,
    )
    .unwrap()
}

fn prepared_series(data: &TimeSeriesData) -> PreparedStudy {
    let ctx = ExecutionContext::for_tests(11);
    Study::series(data.clone())
        .graph(AcceptedGraph::temporal_dag(autoregressive_pulse_dag(true, false)))
        .query(CausalQuery::TemporalEffect(pulse()))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
}

fn column(data: &TimeSeriesData, name: &str) -> Vec<f64> {
    data.float64_values(data.schema().id_of(name).unwrap()).unwrap()
}

/// The rows `[from, to)` of every column, as a series of the prepared study's schema.
fn slice(data: &TimeSeriesData, from: usize, to: usize) -> TimeSeriesData {
    let names = ["t", "y", "z"];
    let columns: Vec<Vec<f64>> =
        names.iter().map(|name| column(data, name)[from..to].to_vec()).collect();
    TimeSeriesData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, values)| (*name, values.as_slice())),
        1,
    )
    .unwrap()
}

#[test]
fn a_declared_invariant_period_refresh_publishes_the_least_squares_fit_and_no_interval() {
    let data = autoregressive_pulse_series(N, 0.6, false, 31);
    let study = prepared_series(&data);
    let first = window(0, i64::try_from(SPLIT).unwrap(), "snap_1");
    let mut refresh: PeriodRefresh = study
        .start_period_refresh(
            &slice(&data, 0, SPLIT),
            "t",
            &refresh_spec(MechanismStanding::DeclaredInvariant),
            &first,
            &prior(),
        )
        .unwrap();
    let second = window(i64::try_from(SPLIT).unwrap(), i64::try_from(N).unwrap(), "snap_2");
    let outcome = study
        .apply_period_refresh(
            &mut refresh,
            &slice(&data, SPLIT, N),
            "t",
            &second,
            cadence(),
            "law_1",
        )
        .unwrap();
    assert_eq!(outcome, PeriodRefreshOutcome::Extended { contiguous: true });
    assert_eq!(refresh.fit().recomputes(), 0, "an append is not a rebuild");
    let snapshot = refresh.fit().snapshot();
    let beta = snapshot.beta.as_ref().expect("coefficients");
    assert_eq!(beta.len(), 3);
    assert!(beta.iter().all(|value| value.is_finite()));
    assert!(
        snapshot.interval().is_none(),
        "the least-squares fit carries no standard error or interval"
    );
}
