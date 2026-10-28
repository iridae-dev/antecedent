//! Observation weights on a table are never silently ignored by a static effect route.
//!
//! A table that carries observation weights (`TabularData::with_weights`) is read by the
//! randomized and panel designs and by a single series. The static effect routes (the average
//! effect, the conditional effect, natural-direct mediation, the counterfactual, the mean
//! curve, the average derivative, the intervention response, and the dose, regression
//! discontinuity and temporal-design routes) read none of them: the unweighted number would
//! describe a population the weights do not declare. Each is therefore refused with
//! `option_not_applicable` and the detail `sampling_weights.route_unsupported`, which names the
//! next attempt. The average effect does honour weights declared on its estimator
//! (`with_sampling_weights`): they equal the same fit on rows replicated by the integer weights.
//! Weights declared on an estimator that a route does not run are refused the same way.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain,
    ContinuousDoseResponseQuery, CounterfactualQuery, DerivativeWeighting, ExecutionContext,
    GridSpec, Intervention, MediationContrast, MediationQuery, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalAssignment,
    TemporalDesignFunctional, TemporalDesignQuery, Value, VariableId,
};
use antecedent::LocalPolynomialRatioQuery;
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{AnalyticSeKind, LinearAdjustmentAte, SamplingWeights};
use antecedent_graph::{Admg, Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

const N: usize = 300;

/// Columns `a, m, y, z` of `y = 3 a + 4 m + z` with `m = 2 a + noise` and `a` driven by `z`.
fn columns() -> Vec<(&'static str, Vec<f64>)> {
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..N).map(|i| (at(i) * 0.37).sin()).collect();
    let a: Vec<f64> = (0..N).map(|i| z[i] + (at(i) * 0.71).cos()).collect();
    let m: Vec<f64> = (0..N).map(|i| 2.0 * a[i] + (at(i) * 1.13).cos()).collect();
    let y: Vec<f64> =
        (0..N).map(|i| 3.0 * a[i] + 4.0 * m[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    vec![("a", a), ("m", m), ("y", y), ("z", z)]
}

/// Integer weights 1 to 4 that vary from row to row.
fn integer_weights(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + ((i * 7) % 4) as f64).collect()
}

fn table(columns: &[(&str, Vec<f64>)]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, c)| (*name, c.as_slice()))).unwrap()
}

fn weighted(data: &TabularData, weights: Vec<f64>) -> TabularData {
    data.with_weights(Arc::from(weights)).unwrap()
}

/// The table with row `i` repeated `weights[i]` times.
fn replicated(columns: &[(&str, Vec<f64>)], weights: &[f64]) -> TabularData {
    let expanded: Vec<(&str, Vec<f64>)> = columns
        .iter()
        .map(|(name, c)| {
            let rows =
                c.iter().zip(weights).flat_map(|(&x, &w)| std::iter::repeat_n(x, w as usize));
            (*name, rows.collect())
        })
        .collect();
    table(&expanded)
}

fn edges() -> [(u32, u32); 5] {
    [(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)]
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (from, to) in edges() {
        graph
            .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .unwrap();
    }
    graph
}

fn admg() -> Admg {
    let mut graph = Admg::with_variables(4);
    for (from, to) in edges() {
        graph
            .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .unwrap();
    }
    graph
}

#[derive(Clone, Copy, PartialEq)]
enum Graph {
    Dag,
    Admg,
    None,
}

struct Route {
    name: &'static str,
    data: TabularData,
    query: CausalQuery,
    graph: Graph,
    /// Bootstrap replicates; the quantile interval needs a whole-procedure bootstrap.
    replicates: u32,
}

fn build(
    route: &Route,
    data: TabularData,
    estimator: Option<LinearAdjustmentAte>,
) -> Result<antecedent::Study, CausalError> {
    let builder = Study::tabular(data);
    let builder = match route.graph {
        Graph::Dag => builder.graph(dag()),
        Graph::Admg => builder.graph(admg()),
        Graph::None => builder,
    };
    let builder = match estimator {
        Some(estimator) => builder.estimator(estimator),
        None => builder.refute(RefuteSuite::None).bootstrap_replicates(route.replicates),
    };
    builder.query(route.query.clone()).build()
}

fn run(
    route: &Route,
    data: TabularData,
    estimator: Option<LinearAdjustmentAte>,
) -> Result<StudyResult, CausalError> {
    build(route, data, estimator)?.run(&ExecutionContext::for_tests(5))
}

fn mediation() -> CausalQuery {
    let mut query =
        MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), MediationContrast::NaturalDirect);
    query.control = Intervention::set(v(0), Value::f64(0.0));
    query.active = Intervention::set(v(0), Value::f64(1.0));
    CausalQuery::Mediation(query)
}

fn response(functional: ResponseFunctional) -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(functional))
}

/// The effect routes over the structural model, plus the graphless dose routes.
fn routes() -> Vec<Route> {
    let ate = || CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(2)));
    let data = || table(&columns());
    let mut routes = vec![
        Route { name: "average effect", data: data(), query: ate(), graph: Graph::Dag, replicates: 0 },
        Route { name: "average effect on an ADMG", data: data(), query: ate(), graph: Graph::Admg, replicates: 0 },
        Route {
            name: "conditional effect",
            data: data(),
            query: CausalQuery::ConditionalEffect(
                ConditionalEffectQuery::try_new(
                    AverageEffectQuery::binary_ate(v(0), v(2)).with_effect_modifiers(vec![v(3)]),
                )
                .unwrap(),
            ),
            graph: Graph::Admg,
            replicates: 0,
        },
        Route { name: "natural-direct mediation", data: data(), query: mediation(), graph: Graph::Dag, replicates: 0 },
        Route {
            name: "counterfactual",
            data: data(),
            query: CausalQuery::Counterfactual(
                CounterfactualQuery::new(
                    v(2),
                    Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
                )
                .with_control_level(0.0),
            ),
            graph: Graph::Dag,
            replicates: 0,
        },
        Route {
            name: "mean curve",
            data: data(),
            query: response(ResponseFunctional::MeanCurve {
                outcome: v(2),
                treatment: ContinuousDomain::new(
                    v(0),
                    GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
                ),
            }),
            graph: Graph::Dag,
            replicates: 0,
        },
        Route {
            name: "average derivative",
            data: data(),
            query: response(ResponseFunctional::AverageDerivative {
                outcome: v(2),
                treatment: v(0),
                weighting: DerivativeWeighting::Observed,
            }),
            graph: Graph::Dag,
            replicates: 0,
        },
        Route {
            name: "intervention response",
            data: data(),
            query: response(ResponseFunctional::InterventionResponse {
                outcome: v(2),
                interventions: Arc::from([Intervention::set(v(0), Value::f64(0.25))]),
            }),
            graph: Graph::Dag,
            replicates: 0,
        },
    ];
    routes.push(continuous_dose());
    routes.push(local_polynomial_ratio());
    routes.push(temporal_design("temporal design mean", TemporalDesignFunctional::Mean));
    routes.push(temporal_design(
        "temporal design quantile",
        TemporalDesignFunctional::Quantile { level: 0.5 },
    ));
    routes
}

fn continuous_dose() -> Route {
    let doses = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let (mut outcome, mut dose) = (Vec::new(), Vec::new());
    for intercept in [1.0, 10.0] {
        for value in doses {
            outcome.push(intercept + 2.0 * value);
            dose.push(value);
        }
    }
    let data = table(&[("y", outcome), ("dose", dose), ("density", vec![0.5; 10])]);
    let groups: Vec<Arc<str>> = ["control", "treated"]
        .into_iter()
        .flat_map(|group| std::iter::repeat_n(Arc::<str>::from(group), 5))
        .collect();
    let query = ContinuousDoseResponseQuery {
        outcome: v(0),
        dose: v(1),
        dose_density: v(2),
        baseline_groups: groups.into(),
        target_doses: Arc::from([0.0, 0.5]),
        bandwidth: 0.6,
        min_local_support: 3,
        density_provenance: Arc::from("known"),
        fixed_policy: None,
    };
    Route {
        name: "continuous dose response",
        data,
        query: CausalQuery::ContinuousDoseResponse(query),
        graph: Graph::None,
        replicates: 0,
    }
}

fn local_polynomial_ratio() -> Route {
    let (mut x, mut treatment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for step in 1..80 {
        for sign in [-1.0, 1.0] {
            let score = sign * f64::from(step) / 80.0;
            for replicate in 0..4 {
                let dose = if sign < 0.0 {
                    f64::from(u8::from(replicate == 0))
                } else {
                    f64::from(u8::from(replicate != 3))
                };
                x.push(score);
                treatment.push(dose);
                outcome.push(
                    1.0 + 2.0 * score
                        + 0.5 * score * score
                        + 3.0 * dose
                        + [-0.02, -0.01, 0.01, 0.02][replicate],
                );
            }
        }
    }
    let data = table(&[("x", x), ("t", treatment), ("y", outcome)]);
    let query = LocalPolynomialRatioQuery {
        outcome: v(2),
        treatment: v(1),
        running: v(0),
        cutoff: 0.0,
        bandwidth: 1.0,
        kink: false,
        options: None,
    };
    Route {
        name: "regression discontinuity",
        data,
        query: CausalQuery::LocalPolynomialRatio(query),
        graph: Graph::None,
        replicates: 0,
    }
}

fn temporal_design(name: &'static str, functional: TemporalDesignFunctional) -> Route {
    let (units, periods) = (60_usize, 5_usize);
    let mut unit_labels: Vec<Arc<str>> = Vec::new();
    let (mut period, mut assignment, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..units {
        for t in 0..periods {
            let a = (i * 3 + t * 5) % 7 < 3;
            unit_labels.push(Arc::from(format!("u{i}")));
            period.push(i64::try_from(t).unwrap());
            assignment.push(a);
            outcome.push(0.01 * ((i * 13 + t * 7) % 11) as f64 + 0.5 * f64::from(u8::from(a)));
        }
    }
    let rows = outcome.len();
    let query_is_quantile = matches!(functional, TemporalDesignFunctional::Quantile { .. });
    let query = TemporalDesignQuery {
        outcome: v(0),
        units: unit_labels.into(),
        periods: period.as_slice().into(),
        lag: 1,
        assignment: TemporalAssignment::TwoArm {
            assignment: assignment.as_slice().into(),
            probability: vec![3.0 / 7.0; rows].into(),
        },
        functional,
        min_probability: 0.05,
        outcome_misclassification: None,
    };
    Route {
        name,
        data: table(&[("y", outcome)]),
        query: CausalQuery::TemporalDesign(query),
        graph: Graph::None,
        replicates: if matches!(query_is_quantile, true) { 99 } else { 0 },
    }
}

fn refusal_text(error: &CausalError) -> String {
    error.to_string()
}

fn assert_typed_weight_refusal(route: &str, error: &CausalError) {
    let text = refusal_text(error);
    assert!(text.starts_with("reason=option_not_applicable"), "{route}: {text}");
    assert!(text.contains("sampling_weights.route_unsupported"), "{route}: {text}");
    assert!(text.contains("Next attempt"), "{route}: {text}");
}

fn headline(result: &StudyResult) -> f64 {
    if let Some(counterfactual) = result.counterfactual() {
        return counterfactual.mean_ite;
    }
    if let Some(response) = result.response() {
        return match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
            ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
                mean[mean.len() - 1] - mean[0]
            }
            other => panic!("a point-identified response, got {other:?}"),
        };
    }
    result.estimate().as_effect().unwrap().ate
}

#[test]
fn table_weights_are_never_silently_ignored_on_any_static_effect_route() {
    for route in routes() {
        // The route works on the unweighted table, so the refusal below is of weights alone.
        run(&route, route.data.clone(), None)
            .unwrap_or_else(|error| panic!("{}: unweighted run failed: {error}", route.name));

        let n = route.data.row_count();
        let non_constant = weighted(&route.data, integer_weights(n));
        let constant = weighted(&route.data, vec![1.0; n]);
        for (label, data) in [("non-constant", non_constant), ("constant", constant)] {
            let error = run(&route, data, None).err().unwrap_or_else(|| {
                panic!("{}: published an estimate for a table with {label} weights", route.name)
            });
            assert_typed_weight_refusal(route.name, &error);
        }
    }
}

#[test]
fn a_prepared_study_refuses_a_rebound_table_that_carries_weights() {
    for route in routes() {
        let ctx = ExecutionContext::for_tests(5);
        let prepared = build(&route, route.data.clone(), None).unwrap().prepare(&ctx).unwrap();
        let n = route.data.row_count();
        let error = prepared
            .estimate(&weighted(&route.data, integer_weights(n)), &ctx)
            .err()
            .unwrap_or_else(|| panic!("{}: a rebound weighted table was estimated", route.name));
        assert_typed_weight_refusal(route.name, &error);
    }
}

fn linear(weights: Option<Vec<f64>>) -> LinearAdjustmentAte {
    let estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(0).with_se_kind(AnalyticSeKind::Hc1);
    match weights {
        Some(weights) => estimator.with_sampling_weights(SamplingWeights::new(weights).unwrap()),
        None => estimator,
    }
}

#[test]
fn estimator_declared_weights_on_the_average_effect_equal_row_replication() {
    let weights = integer_weights(N);
    let replicated = replicated(&columns(), &weights);
    for route in routes().into_iter().filter(|r| r.name.starts_with("average effect")) {
        let honoured = run(&route, route.data.clone(), Some(linear(Some(weights.clone())))).unwrap();
        let copies = run(
            &Route { data: replicated.clone(), ..clone_route(&route) },
            replicated.clone(),
            Some(linear(None)),
        )
        .unwrap();
        let unweighted = run(&route, route.data.clone(), Some(linear(None))).unwrap();
        let (got, want, plain) =
            (headline(&honoured), headline(&copies), headline(&unweighted));
        assert!((got - want).abs() < 1e-9, "{}: weighted {got} vs replicated {want}", route.name);
        assert!((got - plain).abs() > 1e-4, "{}: weights changed nothing ({got})", route.name);
        let record = honoured.estimate().as_effect().unwrap().diagnostics.sampling_weights.clone();
        assert_eq!(record.expect("a weight record").rows_used, N as u64);
    }
}

fn clone_route(route: &Route) -> Route {
    Route { name: route.name, data: route.data.clone(), query: route.query.clone(), graph: route.graph, replicates: route.replicates }
}

#[test]
fn weights_declared_on_an_estimator_a_route_does_not_run_are_refused() {
    let weights = integer_weights(N);
    for route in routes()
        .into_iter()
        .filter(|r| matches!(r.name, "counterfactual"))
    {
        let error = run(&route, route.data.clone(), Some(linear(Some(weights.clone()))))
            .err()
            .unwrap_or_else(|| panic!("{}: declared weights were ignored", route.name));
        assert_typed_weight_refusal(route.name, &error);
    }
}
