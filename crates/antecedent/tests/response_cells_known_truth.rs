//! Known-truth evidence for the static response and derivative estimators and the Bayesian
//! joint IV estimator on the coordinates a latent-free ADMG, a `TemporalAdmg` projection and a
//! supplied DAG give them.
//!
//! * A `TemporalAdmg` whose nodes are all contemporaneous (each variable once) projects to its
//!   ADMG; without a bidirected edge the projection is the back-door DAG, so the static
//!   estimator the caller selects runs on tabular data.
//! * Derivative design: `y = 5 + 2 a - 0.5 b`, `v = 1 + 0.25 a + 1.5 b` with `a` and `b`
//!   deterministic and independent causes of both. The derivative of `y` in `a` is 2 at every
//!   point, so the identity derivative, the elasticity `2 at / E[y | do(a = at)]`, the two
//!   semi-elasticities (`2 at` and `2 / E[y | do(a = at)]`), the average derivative, the
//!   directional derivative and the Jacobian have closed forms evaluated here from the rows
//!   (`E[y | do(a)] = 5 + 2 a - 0.5 mean(b)`). The surface is linear, so the kernel-weighted
//!   local fit, the additive GAM and the Riesz average derivative are unbiased on it and the
//!   tolerances are the ones the committed static route evidence pins.
//! * Level design: the exact table of `conformance/response/intervention_response`
//!   (`z = sin(i / 17)`, `t = z + cos(i / 11)`, `y = 1 + 2 t + 0.8 z`), where
//!   `E[y | do(t = L)] = 1 + 2 L + 0.8 mean(z)`. The frequentist additive g-computation is
//!   exact on the linear law; the conjugate Bayesian response is held to six Monte-Carlo
//!   standard errors of at least 2 000 draws of a posterior whose sd is at most 1e-1.
//! * Joint Bayesian IV: the posterior of `(alpha, pi, mu, beta)` is exactly Gaussian (a stacked
//!   Gaussian regression with prior sd `prior_scale` and unit stage variances); the test solves
//!   its 4x4 normal equations by Gaussian elimination and holds the draw mean to six Monte-Carlo
//!   standard errors of that closed-form posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "test scaffolding builds small deterministic fixtures"
)]

#[path = "known_truth_cells/mod.rs"]
mod support;

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
    StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, DerivativeScale, DerivativeWeighting,
    ExecutionContext, GridSpec, Intervention, Lag, ResponseFunctional as F, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Admg, Dag, DenseNodeId, TemporalAdmg};
use support::{Coordinate, GraphKind, Structure, for_each_matching, on_large_stack};

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn ids(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect()
}

// ------------------------------------------------------------------ shared builders

/// A graph over contemporaneous nodes `0..n` (variable id order) with the directed `edges`.
fn temporal_graph(n: u32, edges: &[(u32, u32)]) -> TemporalAdmg {
    let mut graph = TemporalAdmg::empty();
    let nodes: Vec<_> =
        (0..n).map(|i| graph.add_lagged(v(i), Lag::CONTEMPORANEOUS).unwrap()).collect();
    for &(from, to) in edges {
        graph.insert_directed(nodes[from as usize], nodes[to as usize]).unwrap();
    }
    graph
}

fn admg_graph(n: u32, edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for &(from, to) in edges {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

/// Attach the graph a coordinate names (`Admg` or `TemporalAdmg`, explicit or accepted).
fn with_graph(
    c: &Coordinate,
    study: antecedent::analysis::StudyBuilder,
    n: u32,
    edges: &[(u32, u32)],
) -> antecedent::analysis::StudyBuilder {
    match (c.graph, c.structure) {
        (GraphKind::TemporalAdmg, Structure::Explicit) => study.graph(temporal_graph(n, edges)),
        (GraphKind::TemporalAdmg, Structure::Accepted) => {
            study.graph(AcceptedGraph::from(temporal_graph(n, edges)))
        }
        (GraphKind::Admg, Structure::Explicit) => study.graph(admg_graph(n, edges)),
        (GraphKind::Admg, Structure::Accepted) => {
            study.graph(AcceptedGraph::from(admg_graph(n, edges)))
        }
        other => panic!("{}: unexpected graph/structure {other:?}", c.text),
    }
}

fn scalar_or_values(result: &StudyResult, label: &str) -> Vec<f64> {
    match &result.response().unwrap_or_else(|| panic!("{label}: no response")).estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(x)) => vec![*x],
        ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => values.to_vec(),
        ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) => {
            values.to_vec()
        }
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        other => panic!("{label}: unexpected response identification {other:?}"),
    }
}

/// Validation must execute under cheap/full and must not appear under none.
fn assert_validation(c: &Coordinate, result: &StudyResult) {
    match c.suite {
        RefuteSuite::None => {
            assert!(result.refutations.is_empty(), "{}: validation none emits no refuters", c.text);
        }
        _ if c.bayesian => assert!(
            !result.refutations.is_empty() || !result.predictive_checks.is_empty(),
            "{}: validation (refuters or predictive checks) must execute",
            c.text
        ),
        _ => assert!(!result.refutations.is_empty(), "{}: refuters must execute", c.text),
    }
}

fn assert_estimator(c: &Coordinate, result: &StudyResult, estimator: EstimatorId) {
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some(estimator.as_str()),
        "{}: the explicitly selected estimator ran",
        c.text
    );
}

fn assert_values(c: &Coordinate, got: &[f64], truth: &[f64], tolerance: f64) {
    assert_eq!(got.len(), truth.len(), "{}: value count", c.text);
    for (got, truth) in got.iter().zip(truth) {
        assert!(
            (got - truth).abs() < tolerance,
            "{}: published {got} vs closed form {truth}",
            c.text
        );
    }
}

// ------------------------------------------------------------------ derivative design

const N_DERIVATIVE: usize = 800;
const AT: f64 = 2.0;
/// Edges `a -> y`, `a -> v`, `b -> y`, `b -> v` over the variable ids `a = 0, b = 1, y = 2, v = 3`.
const DERIVATIVE_EDGES: [(u32, u32); 4] = [(0, 2), (0, 3), (1, 2), (1, 3)];

struct DerivativeDesign {
    data: TabularData,
    mean_b: f64,
}

impl DerivativeDesign {
    fn new() -> Self {
        let n = N_DERIVATIVE;
        let a: Vec<f64> =
            (0..n).map(|i| 2.0 + (i as f64 * 0.71).sin() + 0.2 * (i as f64 * 0.13).cos()).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 1.13).cos()).collect();
        let y: Vec<f64> = a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b).collect();
        let w: Vec<f64> = a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 1.5 * b).collect();
        let mean_b = b.iter().sum::<f64>() / n as f64;
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y", y.as_slice()),
            ("v", w.as_slice()),
        ])
        .unwrap();
        Self { data, mean_b }
    }

    /// `E[y | do(a = level)] = 5 + 2 level - 0.5 mean(b)`, by hand from the rows.
    fn level(&self, level: f64) -> f64 {
        5.0 + 2.0 * level - 0.5 * self.mean_b
    }
}

fn derivative_inference(c: &Coordinate) -> InferenceMode {
    if c.bayesian {
        InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(64))
    } else {
        InferenceMode::Frequentist
    }
}

fn run_derivative(
    c: &Coordinate,
    design: &DerivativeDesign,
    estimator: EstimatorId,
    functional: F,
) -> StudyResult {
    let study = with_graph(c, Study::tabular(design.data.clone()), 4, &DERIVATIVE_EDGES)
        .query(CausalQuery::Response(ResponseQuery::new(functional)))
        .estimator(estimator)
        .inference(derivative_inference(c))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.35), ..Default::default() })
        .refute(c.suite);
    let study = if c.suite == RefuteSuite::None { study } else { study.bootstrap_replicates(12) };
    study
        .build()
        .unwrap_or_else(|e| panic!("{}: build: {e}", c.text))
        .run(&ExecutionContext::for_tests(805))
        .unwrap_or_else(|e| panic!("{}: run: {e}", c.text))
}

fn point(scale: DerivativeScale) -> F {
    F::PointDerivative { outcome: v(2), treatment: v(0), at: AT, order: 1, scale }
}

/// Run `body` on the `TemporalAdmg` coordinates of `family` (every claimed coordinate of it).
fn on_temporal_admg(family: &'static str, body: impl Fn(&Coordinate) + Send + 'static) {
    on_large_stack(move || {
        for_each_matching(
            family,
            |c| c.graph == GraphKind::TemporalAdmg && c.structure != Structure::GraphPosterior,
            body,
        );
    });
}

#[test]
fn kennedy_dr_point_derivative_matches_the_closed_form_slope() {
    on_temporal_admg("response.kennedy_dr|PointDerivative", |c| {
        let design = DerivativeDesign::new();
        let result = run_derivative(
            c,
            &design,
            EstimatorId::ResponseKennedyDr,
            point(DerivativeScale::Identity),
        );
        assert_estimator(c, &result, EstimatorId::ResponseKennedyDr);
        assert_values(c, &scalar_or_values(&result, &c.text), &[2.0], 0.01);
        assert_validation(c, &result);
    });
}

#[test]
fn kennedy_dr_elasticity_matches_the_closed_form() {
    on_temporal_admg("response.kennedy_dr|Elasticity", |c| {
        let design = DerivativeDesign::new();
        let truth = 2.0 * AT / design.level(AT);
        let result = run_derivative(
            c,
            &design,
            EstimatorId::ResponseKennedyDr,
            point(DerivativeScale::LogLog),
        );
        assert_estimator(c, &result, EstimatorId::ResponseKennedyDr);
        assert_values(c, &scalar_or_values(&result, &c.text), &[truth], 0.01);
        assert_validation(c, &result);
    });
}

#[test]
fn kennedy_dr_semi_elasticity_matches_the_closed_form_on_both_scales() {
    on_temporal_admg("response.kennedy_dr|SemiElasticity", |c| {
        let design = DerivativeDesign::new();
        let scales = [
            (DerivativeScale::LogTreatment, 2.0 * AT),
            (DerivativeScale::LogOutcome, 2.0 / design.level(AT)),
        ];
        for (scale, truth) in scales {
            let result = run_derivative(c, &design, EstimatorId::ResponseKennedyDr, point(scale));
            assert_estimator(c, &result, EstimatorId::ResponseKennedyDr);
            assert_values(c, &scalar_or_values(&result, &c.text), &[truth], 0.01);
            assert_validation(c, &result);
        }
    });
}

#[test]
fn riesz_average_derivative_matches_the_closed_form_slope() {
    on_temporal_admg("response.riesz_ade|AverageDerivative", |c| {
        let design = DerivativeDesign::new();
        let functional = F::AverageDerivative {
            outcome: v(2),
            treatment: v(0),
            weighting: DerivativeWeighting::Observed,
        };
        let result = run_derivative(c, &design, EstimatorId::ResponseRieszAde, functional);
        assert_estimator(c, &result, EstimatorId::ResponseRieszAde);
        assert_values(c, &scalar_or_values(&result, &c.text), &[2.0], 0.08);
        assert_validation(c, &result);
    });
}

#[test]
fn gam_directional_derivative_matches_the_jacobian_times_direction() {
    on_temporal_admg("response.gam_derivative|DirectionalDerivative", |c| {
        let design = DerivativeDesign::new();
        // Jacobian rows y: (2, -0.5), v: (0.25, 1.5); direction (1, 2).
        let functional = F::DirectionalDerivative {
            outcomes: ids(&[2, 3]),
            treatments: ids(&[0, 1]),
            at: Arc::from([AT, 0.0]),
            direction: Arc::from([1.0, 2.0]),
        };
        let result = run_derivative(c, &design, EstimatorId::ResponseGamDerivative, functional);
        assert_estimator(c, &result, EstimatorId::ResponseGamDerivative);
        let truth = [2.0 * 1.0 - 0.5 * 2.0, 0.25 * 1.0 + 1.5 * 2.0];
        assert_values(c, &scalar_or_values(&result, &c.text), &truth, 1e-6);
        assert_validation(c, &result);
    });
}

#[test]
fn gam_jacobian_matches_the_structural_coefficients() {
    on_temporal_admg("response.gam_derivative|ResponseJacobian", |c| {
        let design = DerivativeDesign::new();
        let functional = F::Jacobian {
            outcomes: ids(&[2, 3]),
            treatments: ids(&[0, 1]),
            at: Arc::from([AT, 0.0]),
            scale: DerivativeScale::Identity,
        };
        let result = run_derivative(c, &design, EstimatorId::ResponseGamDerivative, functional);
        assert_estimator(c, &result, EstimatorId::ResponseGamDerivative);
        assert_values(c, &scalar_or_values(&result, &c.text), &[2.0, -0.5, 0.25, 1.5], 1e-6);
        assert_validation(c, &result);
    });
}

// ------------------------------------------------------------------ response curves

const GRID: [f64; 3] = [1.5, 2.0, 2.5];

fn curve_query() -> ResponseQuery {
    ResponseQuery::new(F::MeanCurve {
        outcome: v(2),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from(GRID))),
    })
}

fn run_curve(c: &Coordinate, design: &DerivativeDesign, estimator: EstimatorId) -> StudyResult {
    let inference = if c.bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4_000).prior_scale(1_000.0))
    } else {
        InferenceMode::Frequentist
    };
    with_graph(c, Study::tabular(design.data.clone()), 4, &DERIVATIVE_EDGES)
        .query(CausalQuery::Response(curve_query()))
        .estimator(estimator)
        .inference(inference)
        .refute(c.suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap_or_else(|e| panic!("{}: build: {e}", c.text))
        .run(&ExecutionContext::for_tests(806))
        .unwrap_or_else(|e| panic!("{}: run: {e}", c.text))
}

fn on_admg_classes(family: &'static str, body: impl Fn(&Coordinate) + Send + 'static) {
    on_large_stack(move || {
        for_each_matching(
            family,
            |c| {
                matches!(c.graph, GraphKind::Admg | GraphKind::TemporalAdmg)
                    && c.structure != Structure::GraphPosterior
            },
            body,
        );
    });
}

#[test]
fn kennedy_dr_response_curve_matches_the_closed_form_levels() {
    on_admg_classes("response.kennedy_dr|ResponseCurve", |c| {
        let design = DerivativeDesign::new();
        let truth: Vec<f64> = GRID.iter().map(|g| design.level(*g)).collect();
        let result = run_curve(c, &design, EstimatorId::ResponseKennedyDr);
        assert_estimator(c, &result, EstimatorId::ResponseKennedyDr);
        assert_values(c, &scalar_or_values(&result, &c.text), &truth, 0.01);
        assert_validation(c, &result);
    });
}

#[test]
fn bayesian_response_curve_matches_the_closed_form_levels() {
    on_admg_classes("response.bayesian|ResponseCurve", |c| {
        let design = DerivativeDesign::new();
        let truth: Vec<f64> = GRID.iter().map(|g| design.level(*g)).collect();
        let result = run_curve(c, &design, EstimatorId::ResponseBayesian);
        assert_estimator(c, &result, EstimatorId::ResponseBayesian);
        // Six Monte-Carlo standard errors of at least 2 000 draws of sd at most 1e-1.
        assert_values(c, &scalar_or_values(&result, &c.text), &truth, 6.0 * 0.1 / 2_000_f64.sqrt());
        assert_validation(c, &result);
    });
}

// ------------------------------------------------------------------ intervention level

const N_LEVEL: usize = 400;
const LEVEL: f64 = 0.5;
/// Edges `z -> t`, `z -> y`, `t -> y` over `t = 0, y = 1, z = 2`.
const LEVEL_EDGES: [(u32, u32); 3] = [(2, 0), (2, 1), (0, 1)];

fn level_table() -> (TabularData, f64) {
    // The treatment smooth of the plug-in outcome GAM is unpenalized, so it reproduces a linear
    // dose effect exactly; the adjustment smooth (`z`) is roughness-penalized and shrinks even a
    // linear effect, so the outcome law here has no `z` term (`z` stays a graph parent and a
    // regressor, whose fitted smooth of a zero residual is exactly zero). `t` is a coprime-stride
    // permutation of the `z` grid so the two are not concurved.
    let n1 = (N_LEVEL - 1) as f64;
    let z: Vec<f64> = (0..N_LEVEL).map(|i| -1.0 + 2.0 * i as f64 / n1).collect();
    let t: Vec<f64> =
        (0..N_LEVEL).map(|i| -1.0 + 2.0 * ((i * 173 + 37) % N_LEVEL) as f64 / n1).collect();
    let y: Vec<f64> = t.iter().map(|t| 1.0 + 2.0 * t).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    (data, 1.0 + 2.0 * LEVEL)
}

fn level_query() -> ResponseQuery {
    ResponseQuery::new(F::InterventionResponse {
        outcome: v(1),
        interventions: Arc::from([Intervention::set(v(0), Value::f64(LEVEL))]),
    })
}

fn check_level(family: &'static str, estimator: EstimatorId, tolerance: f64) {
    let (data, truth) = level_table();
    on_large_stack(move || {
        for_each_matching(
            family,
            |c| c.graph == GraphKind::TemporalAdmg && c.structure != Structure::GraphPosterior,
            |c| {
                let inference = if c.bayesian {
                    InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(4_000).prior_scale(1_000.0),
                    )
                } else {
                    InferenceMode::Frequentist
                };
                let result = with_graph(c, Study::tabular(data.clone()), 3, &LEVEL_EDGES)
                    .query(CausalQuery::Response(level_query()))
                    .estimator(estimator)
                    .inference(inference)
                    .refute(c.suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap_or_else(|e| panic!("{}: build: {e}", c.text))
                    .run(&ExecutionContext::for_tests(52))
                    .unwrap_or_else(|e| panic!("{}: run: {e}", c.text));
                assert_estimator(c, &result, estimator);
                assert_values(c, &scalar_or_values(&result, &c.text), &[truth], tolerance);
                assert_validation(c, &result);
            },
        );
    });
}

#[test]
fn intervention_gcomp_level_matches_the_closed_form_on_temporal_admg_coordinates() {
    check_level(
        "response.intervention_gcomp|InterventionResponse",
        EstimatorId::ResponseInterventionGcomp,
        1e-6,
    );
}

#[test]
fn bayesian_response_level_matches_the_closed_form_on_temporal_admg_coordinates() {
    check_level(
        "response.bayesian|InterventionResponse",
        EstimatorId::ResponseBayesian,
        6.0 * 0.1 / 2_000_f64.sqrt(),
    );
}

// ------------------------------------------------------------------ joint Bayesian IV

const IV_PRIOR_SD: f64 = 1_000.0;
const IV_DRAWS: usize = 4_000;

/// Gaussian elimination with partial pivoting for the 4x4 system `a x = b`.
fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    for col in 0..4 {
        let pivot = (col..4).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..4 {
            let factor = a[row][col] / a[col][col];
            for k in col..4 {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; 4];
    for row in (0..4).rev() {
        let tail: f64 = (row + 1..4).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// Posterior mean and sd of `beta` in `T = alpha + pi Z + v`, `Y = mu + beta T + v + e` with unit
/// stage variances and independent `N(0, IV_PRIOR_SD^2)` priors, from the stacked regression of
/// `T` on `[1, z, 0, 0]` and `Y - T` on `[-1, -z, 1, t]`.
fn iv_posterior(z: &[f64], t: &[f64], y: &[f64]) -> (f64, f64) {
    let mut precision = [[0.0; 4]; 4];
    let mut rhs = [0.0; 4];
    for ((&z, &t), &y) in z.iter().zip(t).zip(y) {
        for (x, target) in [([1.0, z, 0.0, 0.0], t), ([-1.0, -z, 1.0, t], y - t)] {
            for i in 0..4 {
                rhs[i] += x[i] * target;
                for j in 0..4 {
                    precision[i][j] += x[i] * x[j];
                }
            }
        }
    }
    for i in 0..4 {
        precision[i][i] += 1.0 / IV_PRIOR_SD.powi(2);
    }
    let mean = solve4(precision, rhs)[3];
    let variance = solve4(precision, [0.0, 0.0, 0.0, 1.0])[3];
    (mean, variance.sqrt())
}

#[test]
fn bayesian_joint_linear_iv_matches_the_closed_form_posterior() {
    let n = 1200_usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let u: Vec<f64> = (0..n).map(|i| ((i % 17) as f64 - 8.0) / 5.0).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.6 * z[i] + u[i]).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + u[i]).collect();
    let (mean, sd) = iv_posterior(&z, &t, &y);
    // The structural coefficient is 2; the fixed-loading model is the generating one.
    assert!((mean - 2.0).abs() < 0.2, "closed-form posterior mean {mean}");
    assert!(sd < 0.1, "closed-form posterior sd {sd}");
    let tolerance = 6.0 * sd / (IV_DRAWS as f64).sqrt() + 1e-9;
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    on_large_stack(move || {
        for_each_matching(
            "iv.bayesian_joint_linear|AverageEffect",
            |c| {
                c.graph == GraphKind::Dag
                    && c.structure != Structure::GraphPosterior
                    && c.suite == RefuteSuite::None
            },
            |c| {
                let study = Study::tabular(data.clone());
                let study = match c.structure {
                    Structure::Explicit => study.graph(dag.clone()),
                    _ => study.graph(AcceptedGraph::from(dag.clone())),
                };
                let result = study
                    .query(AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0))
                    .identifier(IdentifierId::Iv)
                    .estimator(EstimatorId::BayesianIvJointLinear)
                    .inference(InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(IV_DRAWS).prior_scale(IV_PRIOR_SD),
                    ))
                    .refute(c.suite)
                    .build()
                    .unwrap_or_else(|e| panic!("{}: build: {e}", c.text))
                    .run(&ExecutionContext::for_tests(81))
                    .unwrap_or_else(|e| panic!("{}: run: {e}", c.text));
                assert_estimator(c, &result, EstimatorId::BayesianIvJointLinear);
                let ate = result.estimate().as_effect().unwrap().ate;
                assert!(
                    (ate - mean).abs() < tolerance,
                    "{}: posterior mean {ate} vs closed form {mean} (tolerance {tolerance})",
                    c.text
                );
                assert_validation(c, &result);
            },
        );
    });
}
