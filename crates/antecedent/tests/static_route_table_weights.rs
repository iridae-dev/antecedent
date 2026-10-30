//! Observation weights on a table are never silently ignored by a static effect route.
//!
//! A table that carries observation weights (`TabularData::with_weights`) is read by the
//! randomized and panel designs, by a single series and by the conditional effect (a weighted
//! interaction regression whose error is the weighted sandwich; its known-answer tests are
//! below). The other static effect routes (the average effect, natural-direct mediation, the
//! counterfactual, the mean curve, the average derivative, the intervention response, and the
//! dose, regression discontinuity and temporal-design routes) read none of them: the unweighted number would
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

/// The temporal arm design with the mean functional reads the weights the table carries (weighted
/// arm means and the weighted unit-cluster sandwich; its known-answer test is
/// `unavailable_b_measurement_route_declarations`), so it is not among the routes that refuse them.
fn reads_table_weights(route: &Route) -> bool {
    matches!(route.name, "temporal design mean" | "conditional effect")
}

#[test]
fn the_temporal_arm_design_runs_on_a_weighted_table_and_scale_free_weights_change_nothing() {
    let route = routes().into_iter().find(|r| r.name == "temporal design mean").unwrap();
    let n = route.data.row_count();
    let varying = run(&route, weighted(&route.data, integer_weights(n)), None).unwrap();
    let scaled = run(
        &route,
        weighted(&route.data, integer_weights(n).into_iter().map(|w| 7.0 * w).collect()),
        None,
    )
    .unwrap();
    let constant = run(&route, weighted(&route.data, vec![1.0; n]), None).unwrap();
    let unweighted = run(&route, route.data.clone(), None).unwrap();
    let first = |result: &StudyResult| {
        let section = result.temporal_design().unwrap();
        (section.estimate[0], section.standard_error[0])
    };
    let (point, se) = first(&varying);
    let (scaled_point, scaled_se) = first(&scaled);
    assert!((point - scaled_point).abs() < 1e-12 && (se - scaled_se).abs() < 1e-12);
    let (plain, plain_se) = first(&unweighted);
    let (unit, unit_se) = first(&constant);
    assert!((plain - unit).abs() < 1e-12 && (plain_se - unit_se).abs() < 1e-12);
    assert!((point - plain).abs() > 1e-9, "varying weights move the estimate");
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
    for route in routes().into_iter().filter(|route| !reads_table_weights(route)) {
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
    for route in routes().into_iter().filter(|route| !reads_table_weights(route)) {
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


// ---------------------------------------------------------------------------------------------
// The conditional effect under sampling weights.
// ---------------------------------------------------------------------------------------------

use antecedent_core::OutcomeFunctional;
use antecedent_graph::Cpdag;

/// An exact rational, enough for the weighted stratum contrasts of integer data.
#[derive(Clone, Copy, Debug)]
struct Q(i128, i128);

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

impl Q {
    fn new(n: i128, d: i128) -> Self {
        let g = gcd(n, d).max(1);
        let sign = if d < 0 { -1 } else { 1 };
        Q(sign * n / g, sign * d / g)
    }
    fn int(n: i128) -> Self {
        Q(n, 1)
    }
    fn add(self, o: Q) -> Q {
        Q::new(self.0 * o.1 + o.0 * self.1, self.1 * o.1)
    }
    fn sub(self, o: Q) -> Q {
        Q::new(self.0 * o.1 - o.0 * self.1, self.1 * o.1)
    }
    fn mul(self, o: Q) -> Q {
        Q::new(self.0 * o.0, self.1 * o.1)
    }
    fn f64(self) -> f64 {
        self.0 as f64 / self.1 as f64
    }
}

/// Rows `(w, t, y, weight)` of a saturated design: a binary modifier `w`, a binary treatment `t`
/// and an integer outcome, every `(w, t)` cell holding several rows with unequal weights.
fn saturated_rows() -> Vec<(i128, i128, i128, i128)> {
    (0..48_i128)
        .map(|i| {
            let (w, t) = (i % 2, (i / 2) % 2);
            let y = 3 + 2 * t + 5 * w + 4 * t * w + (i * 7) % 5;
            (w, t, y, 1 + (i * 5) % 4)
        })
        .collect()
}

fn saturated_table(rows: &[(i128, i128, i128, i128)], weights: bool) -> TabularData {
    let col = |f: fn(&(i128, i128, i128, i128)) -> i128| -> Vec<f64> {
        rows.iter().map(|r| f(r) as f64).collect()
    };
    let data = table(&[
        ("t", col(|r| r.1)),
        ("w", col(|r| r.0)),
        ("y", col(|r| r.2)),
    ]);
    if weights { weighted(&data, col(|r| r.3)) } else { data }
}

fn saturated_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn conditional_on(modifier: u32, functional: Option<OutcomeFunctional>) -> CausalQuery {
    let mut inner = AverageEffectQuery::binary_ate(v(0), v(2)).with_effect_modifiers(vec![v(modifier)]);
    if let Some(functional) = functional {
        inner = inner.with_outcome_functional(functional);
    }
    CausalQuery::ConditionalEffect(ConditionalEffectQuery::try_new(inner).unwrap())
}

fn saturated_run(data: TabularData, suite: RefuteSuite) -> Result<StudyResult, CausalError> {
    Study::tabular(data)
        .graph(saturated_dag())
        .refute(suite)
        .bootstrap_replicates(0)
        .query(conditional_on(1, None))
        .build()?
        .run(&ExecutionContext::for_tests(5))
}

/// Weighted cell sums `(Σ s, Σ s y)` of the rows with modifier `w` and treatment `t`.
fn cell(rows: &[(i128, i128, i128, i128)], w: i128, t: i128) -> (i128, i128) {
    rows.iter()
        .filter(|r| r.0 == w && r.1 == t)
        .fold((0, 0), |(s, sy), r| (s + r.3, sy + r.3 * r.2))
}

#[test]
fn a_weighted_conditional_effect_is_the_exact_weighted_stratum_contrast_with_the_weighted_sandwich() {
    let rows = saturated_rows();
    // Exact: tau_w = weighted mean of y at t=1 minus at t=0 in stratum w; the effect averages
    // tau_w over the weighted modifier shares p_w.
    let mean = |w, t| {
        let (s, sy) = cell(&rows, w, t);
        Q::new(sy, s)
    };
    let tau = [mean(0, 1).sub(mean(0, 0)), mean(1, 1).sub(mean(1, 0))];
    let stratum_weight = |w| (0..2).map(|t| cell(&rows, w, t).0).sum::<i128>();
    let total = stratum_weight(0) + stratum_weight(1);
    let p1 = Q::new(stratum_weight(1), total);
    let p = [Q::int(1).sub(p1), p1];
    let exact = tau[0].mul(p[0]).add(tau[1].mul(p[1]));

    let result = saturated_run(saturated_table(&rows, true), RefuteSuite::None).unwrap();
    let effect = result.estimate().as_effect().unwrap();
    assert!((effect.ate - exact.f64()).abs() < 1e-12, "{} vs exact {:?}", effect.ate, exact);
    let plain = saturated_run(saturated_table(&rows, false), RefuteSuite::None).unwrap();
    assert!(
        (plain.estimate().as_effect().unwrap().ate - effect.ate).abs() > 1e-6,
        "the weights must move the contrast"
    );

    // Independent weighted sandwich of the saturated model: row i in cell (w, t) carries
    // s_i * (sign_t p_w e_i / S_cell + (tau_w - tau_bar) / S), HC1 factor n / (n - 4).
    let tau_bar = exact.f64();
    let mut sum_squares = 0.0;
    for &(w, t, y, s) in &rows {
        let (cell_weight, _) = cell(&rows, w, t);
        let e = y as f64 - mean(w, t).f64();
        let sign = if t == 1 { 1.0 } else { -1.0 };
        let phi = sign * p[w as usize].f64() * e / cell_weight as f64
            + (tau[w as usize].f64() - tau_bar) / total as f64;
        sum_squares += (s as f64 * phi).powi(2);
    }
    let n = rows.len() as f64;
    let se = (n / (n - 4.0) * sum_squares).sqrt();
    assert!(
        (effect.se_analytic() - se).abs() < 1e-10 * se.max(1.0),
        "se {} vs sandwich {se}",
        effect.se_analytic()
    );
    let record = effect.diagnostics.sampling_weights.expect("a weight record");
    assert_eq!(record.rows_used, rows.len() as u64);
    assert!((record.weight_sum - total as f64).abs() < 1e-9);
}

#[test]
fn integer_weights_on_the_conditional_effect_equal_the_replicated_rows_and_scale_free_weights_change_nothing()
{
    let route = routes().into_iter().find(|r| r.name == "conditional effect").unwrap();
    let weights = integer_weights(N);
    let honoured = run(&route, weighted(&route.data, weights.clone()), None).unwrap();
    let copies = run(&route, replicated(&columns(), &weights), None).unwrap();
    let plain = run(&route, route.data.clone(), None).unwrap();
    let (got, want) = (headline(&honoured), headline(&copies));
    assert!((got - want).abs() < 1e-9, "weighted {got} vs replicated {want}");
    let se = |r: &StudyResult| r.estimate().as_effect().unwrap().se_analytic();
    assert!((se(&honoured) - se(&plain)).abs() > 1e-6, "weights changed nothing ({})", se(&plain));

    let scaled = run(
        &route,
        weighted(&route.data, weights.iter().map(|w| 7.0 * w).collect()),
        None,
    )
    .unwrap();
    assert!((headline(&scaled) - got).abs() < 1e-12 && (se(&scaled) - se(&honoured)).abs() < 1e-12);
    let unit = run(&route, weighted(&route.data, vec![1.0; N]), None).unwrap();
    assert!((headline(&unit) - headline(&plain)).abs() < 1e-12);
    assert!(se(&honoured).is_finite() && se(&honoured) > 0.0);

    // A zero-weight row stands for no population unit: the fit equals the one without the row.
    let mut with_zeros = weights.clone();
    for i in (0..N).step_by(5) {
        with_zeros[i] = 0.0;
    }
    let kept: Vec<usize> = (0..N).filter(|i| i % 5 != 0).collect();
    let trimmed_columns: Vec<(&str, Vec<f64>)> = columns()
        .into_iter()
        .map(|(name, c)| (name, kept.iter().map(|&i| c[i]).collect()))
        .collect();
    let trimmed_weights: Vec<f64> = kept.iter().map(|&i| weights[i]).collect();
    let zeroed = run(&route, weighted(&route.data, with_zeros), None).unwrap();
    let deleted =
        run(&route, weighted(&table(&trimmed_columns), trimmed_weights), None).unwrap();
    assert!((headline(&zeroed) - headline(&deleted)).abs() < 1e-9);
    assert!((se(&zeroed) - se(&deleted)).abs() < 1e-9);
    let record = zeroed.estimate().as_effect().unwrap().diagnostics.sampling_weights.unwrap();
    assert_eq!(record.zero_weight_rows, (N / 5) as u64);
}

#[test]
fn a_weighted_conditional_effect_is_read_on_a_cpdag_class_and_equals_the_dag() {
    let weights = integer_weights(N);
    let query = conditional_on(3, None);
    let data = weighted(&table(&columns()), weights);
    let mut cpdag = Cpdag::with_variables(4);
    for (from, to) in edges() {
        cpdag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let on_class = Study::tabular(data.clone())
        .graph(cpdag)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .query(query.clone())
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
        .unwrap();
    let on_dag = Study::tabular(data)
        .graph(dag())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .query(query)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
        .unwrap();
    assert!((headline(&on_class) - headline(&on_dag)).abs() < 1e-9);
    assert!(on_class.estimate().as_effect().unwrap().diagnostics.sampling_weights.is_some());
}

#[test]
fn weights_the_conditional_effect_cannot_read_are_refused_with_the_typed_detail() {
    let weights = integer_weights(N);
    let data = weighted(&table(&columns()), weights.clone());
    let build = |query: CausalQuery, suite: RefuteSuite, graph: Graph| {
        let builder = Study::tabular(data.clone());
        let builder = match graph {
            Graph::Admg => builder.graph(admg()),
            _ => builder.graph(dag()),
        };
        builder.refute(suite).bootstrap_replicates(0).query(query).build()?.run(&ExecutionContext::for_tests(5))
    };
    // Refuters refit the unweighted rows.
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let error = build(conditional_on(3, None), suite, Graph::Dag).err().expect("refused");
        assert_typed_weight_refusal("conditional effect with refuters", &error);
    }
    // The quantile grid is built from unweighted cross-fitted scores.
    let error = build(
        conditional_on(3, Some(OutcomeFunctional::quantile(0.5))),
        RefuteSuite::None,
        Graph::Dag,
    )
    .err()
    .expect("refused");
    assert_typed_weight_refusal("conditional quantile", &error);
    // The general-ID functional of a bidirected ADMG fits unweighted rows.
    let mut bidirected = Admg::with_variables(4);
    for (from, to) in edges() {
        bidirected
            .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .unwrap();
    }
    bidirected.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let error = Study::tabular(data.clone())
        .graph(bidirected)
        .refute(RefuteSuite::None)
        .query(conditional_on(3, None))
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
        .err()
        .expect("refused");
    assert_typed_weight_refusal("conditional effect on a bidirected ADMG", &error);
    // Weights declared on an estimator are read by nothing on this route.
    let error = run(
        &routes().into_iter().find(|r| r.name == "conditional effect").unwrap(),
        table(&columns()),
        Some(linear(Some(weights))),
    )
    .err()
    .expect("refused");
    // The conditional route runs no `linear.adjustment.ate` fit, so the declaration is refused
    // by strategy compatibility before any weight is read.
    assert!(error.to_string().starts_with("reason=strategy_incompatible"), "{error}");
}
