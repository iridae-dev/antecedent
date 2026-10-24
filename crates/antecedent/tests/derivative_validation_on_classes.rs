//! Cheap and full validation of derivatives on supplied classes and front-door windows.
//!
//! A derivative on a Cpdag, Pag, Admg or TemporalAdmg runs a refuter state at cheap and full
//! and stays unlicensed (the support status is `refused`, so no coverage record is implied).
//! The fixtures are linear structural laws whose structural slopes are computed here by the
//! normal equations, independently of the library:
//!
//! * adjustment-identified derivatives: the finite difference of the identified intervention
//!   levels must reproduce the regression coefficient of the treatment given the confounder;
//! * front-door-identified derivatives: the slope of the mediator on the treatment times the
//!   mediator's coefficient given the treatment, and the smoother-sensitivity refits must stay
//!   on that product;
//! * a temporal ADMG: single and several horizons, front-door and adjustment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test scaffolding indexes small fixtures and one scenario is kept in one body"
)]

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, Lag,
    ResponseFunctional as F, ResponseQuery, TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag, TemporalAdmg};

const N: usize = 800;

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// Least-squares coefficients of `y` on an intercept and `regressors` (normal equations,
/// Gaussian elimination with partial pivoting).
fn ols(regressors: &[&[f64]], y: &[f64]) -> Vec<f64> {
    let k = regressors.len() + 1;
    let mut a = vec![vec![0.0; k + 1]; k];
    for i in 0..y.len() {
        let row: Vec<f64> =
            std::iter::once(1.0).chain(regressors.iter().map(|column| column[i])).collect();
        for p in 0..k {
            for q in 0..k {
                a[p][q] += row[p] * row[q];
            }
            a[p][k] += row[p] * y[i];
        }
    }
    for col in 0..k {
        let pivot = (col..k).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs())).unwrap();
        a.swap(col, pivot);
        for row in 0..k {
            if row != col {
                let factor = a[row][col] / a[col][col];
                for c in col..=k {
                    let value = a[col][c];
                    a[row][c] -= factor * value;
                }
            }
        }
    }
    (0..k).map(|p| a[p][k] / a[p][p]).collect()
}

fn report<'a>(
    result: &'a StudyResult,
    name: &str,
    label: &str,
) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == name)
        .unwrap_or_else(|| panic!("{label}: no `{name}` report"))
}

fn has(result: &StudyResult, name: &str) -> bool {
    result.refutations.iter().any(|report| report.refuter.as_ref() == name)
}

// ---------------------------------------------------------------------------------------
// Adjustment-identified derivative on a Cpdag, a Pag and a latent-confounded Admg.
// ---------------------------------------------------------------------------------------

/// Columns `a, z, y, r, v, w` as in the class derivative fixture: `z` confounds `a` and `y`,
/// `r` is exogenous to `a`, and a latent cause moves `w` and `y`.
fn class_columns() -> [Vec<f64>; 6] {
    let mut cols: [Vec<f64>; 6] = Default::default();
    for i in 0..N {
        let t = i as f64;
        let latent = (t * 0.77 + 2.0).cos();
        let w = (t * 1.7 + 0.3).sin() + 0.8 * latent;
        let r = (t * 0.29 + 1.0).sin();
        let z = (t * 0.41).cos() + 0.5 * w;
        let a = 2.0 + 0.6 * z + 0.5 * r + (t * 0.71).sin() + 0.2 * (t * 0.13).cos();
        let y = 5.0 + 2.0 * a + 3.0 * z + 0.7 * latent + 0.05 * (t * 3.1).cos();
        let vv = 1.0 + 0.25 * a - 1.5 * z + 0.05 * (t * 2.7).sin();
        for (column, value) in cols.iter_mut().zip([a, z, y, r, vv, w]) {
            column.push(value);
        }
    }
    cols
}

fn class_table(cols: &[Vec<f64>; 6]) -> TabularData {
    TabularData::from_f64_columns([
        ("a", cols[0].as_slice()),
        ("z", cols[1].as_slice()),
        ("y", cols[2].as_slice()),
        ("r", cols[3].as_slice()),
        ("v", cols[4].as_slice()),
        ("w", cols[5].as_slice()),
    ])
    .unwrap()
}

const EDGES: [(u32, u32); 6] = [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4)];

fn cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(6);
    for (from, to) in EDGES {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g
}

fn pag() -> Pag {
    let mut g = Pag::with_variables(6);
    for (from, to) in EDGES {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g
}

fn admg() -> Admg {
    let mut g = Admg::with_variables(6);
    for (from, to) in EDGES.into_iter().chain([(5, 1)]) {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g.insert_bidirected(d(5), d(2)).unwrap();
    g
}

fn average_derivative() -> F {
    F::AverageDerivative {
        outcome: v(2),
        treatment: v(0),
        weighting: DerivativeWeighting::Observed,
    }
}

fn class_study(
    table: &TabularData,
    kind: &str,
    accepted: bool,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let builder = Study::tabular(table.clone());
    let builder = match (kind, accepted) {
        ("cpdag", false) => builder.graph(cpdag()),
        ("cpdag", true) => builder.graph(AcceptedGraph::from(cpdag())),
        ("pag", false) => builder.graph(pag()),
        ("pag", true) => builder.graph(AcceptedGraph::from(pag())),
        ("admg", false) => builder.graph(admg()),
        _ => builder.graph(AcceptedGraph::from(admg())),
    };
    builder
        .query(CausalQuery::Response(ResponseQuery::new(average_derivative())))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn class_derivative_validation_checks_the_finite_difference_against_the_regression_slope() {
    let cols = class_columns();
    let table = class_table(&cols);
    let structural = ols(&[&cols[0], &cols[1]], &cols[2])[1];
    let naive = ols(&[&cols[0]], &cols[2])[1];
    assert!((structural - 2.0).abs() < 0.1, "{structural}");
    assert!((naive - structural).abs() > 0.2, "the confounder must bias the naive slope");
    let ctx = ExecutionContext::for_tests(31);
    for (kind, accepted) in [
        ("cpdag", false),
        ("cpdag", true),
        ("pag", false),
        ("pag", true),
        ("admg", false),
        ("admg", true),
    ] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("{kind} accepted={accepted} {suite:?}");
            let result = class_study(&table, kind, accepted, InferenceMode::Frequentist, suite)
                .run(&ctx)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(result.support_status, Some(CellStatus::Refused), "{label}");
            assert!(report(&result, "derivative.support", &label).passed, "{label}");
            let fd = report(&result, "derivative.finite_difference", &label);
            assert!(
                (fd.refuted_ate - structural).abs() < 0.12 * structural.abs() + 0.1,
                "{label}: finite difference {} vs regression slope {structural}",
                fd.refuted_ate
            );
            assert!((fd.refuted_ate - naive).abs() > 0.1, "{label}: confounded slope");
            assert_eq!(
                has(&result, "derivative.bootstrap_stability"),
                suite == RefuteSuite::Full,
                "{label}"
            );
        }
    }
}

#[test]
fn bayesian_class_derivative_validation_adds_the_predictive_checks() {
    let cols = class_columns();
    let table = class_table(&cols);
    let inference = || InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(48));
    let ctx = ExecutionContext::for_tests(32);
    for kind in ["pag", "admg"] {
        let result = class_study(&table, kind, false, inference(), RefuteSuite::Cheap)
            .run(&ctx)
            .unwrap_or_else(|error| panic!("{kind}: {error}"));
        assert_eq!(result.support_status, Some(CellStatus::Refused), "{kind}");
        assert!(has(&result, "derivative.support"), "{kind}");
        assert!(!result.predictive_checks.is_empty(), "{kind}: posterior predictive checks");
    }
}

// ---------------------------------------------------------------------------------------
// Front-door-identified derivative (static Admg).
// ---------------------------------------------------------------------------------------

fn front_door_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 900;
    let (mut x, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let t = i as f64;
        let latent = (t * 0.77 + 2.0).cos();
        let xi = 1.0 + (t * 0.71).sin() + 0.2 * (t * 0.13).cos() + 0.8 * latent;
        let mi = 0.5 + 0.9 * xi + 0.5 * (t * 1.9).sin();
        x.push(xi);
        m.push(mi);
        y.push(3.0 + 2.0 * mi + 0.7 * latent + 0.05 * (t * 3.1).cos());
    }
    (x, m, y)
}

#[test]
fn a_front_door_derivative_is_checked_by_smoother_sensitivity_and_resampling() {
    let (x, m, y) = front_door_columns();
    let reference = ols(&[&x], &m)[1] * ols(&[&m, &x], &y)[1];
    let naive = ols(&[&x], &y)[1];
    assert!((reference - 1.8).abs() < 0.05, "{reference}");
    assert!((naive - reference).abs() > 0.2);
    let table = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    let ctx = ExecutionContext::for_tests(33);
    for (inference, suite, replicates) in [
        (InferenceMode::Frequentist, RefuteSuite::Cheap, 24),
        (InferenceMode::Frequentist, RefuteSuite::Full, 24),
        (InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24)), RefuteSuite::Full, 0),
    ] {
        let label = format!("{inference:?} {suite:?}");
        let result = Study::tabular(table.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(ResponseQuery::new(average_derivative_front_door())))
            .inference(inference)
            .refute(suite)
            .bootstrap_replicates(replicates)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(result.support_status, Some(CellStatus::Refused), "{label}");
        assert!(report(&result, "derivative.support", &label).passed, "{label}");
        let sensitivity = report(&result, "derivative.front_door_smoothness_sensitivity", &label);
        assert!(sensitivity.passed, "{label}: {:?}", sensitivity.failure_condition);
        assert!(
            (sensitivity.refuted_ate - reference).abs() < 0.15,
            "{label}: refit slope {} vs the front-door product {reference}",
            sensitivity.refuted_ate
        );
        assert_eq!(
            has(&result, "derivative.bootstrap_stability"),
            suite == RefuteSuite::Full,
            "{label}"
        );
        if suite == RefuteSuite::Full {
            let stability = report(&result, "derivative.bootstrap_stability", &label);
            assert!(stability.replicates >= 2, "{label}");
            assert!((stability.refuted_ate - reference).abs() < 0.15, "{label}");
        }
        // The contrast-shaped finite difference does not apply to a front-door plug-in.
        assert!(!has(&result, "derivative.finite_difference"), "{label}");
    }
}

fn average_derivative_front_door() -> F {
    F::AverageDerivative {
        outcome: v(2),
        treatment: v(0),
        weighting: DerivativeWeighting::Observed,
    }
}

// ---------------------------------------------------------------------------------------
// Temporal ADMG: front-door window and several horizons.
// ---------------------------------------------------------------------------------------

const T: usize = 600;

fn temporal_derivative(horizons: Vec<u32>) -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(F::PointDerivative {
            outcome: v(1),
            treatment: v(0),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        })
        .with_temporal(
            TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    )
}

#[test]
fn a_temporal_front_door_slope_runs_the_front_door_state_on_its_window() {
    let (mut x, mut m, mut y) = (vec![0.0; T], vec![0.0; T], vec![0.0; T]);
    let mut latent = vec![0.0; T];
    for t in 0..T {
        let clock = t as f64;
        latent[t] = (clock * 0.77 + 2.0).cos();
        x[t] = 1.0 + (clock * 0.71).sin() + 0.2 * (clock * 0.13).cos() + 0.8 * latent[t];
        m[t] = 0.5 + 0.9 * x[t] + 0.5 * (clock * 1.9).sin();
    }
    for t in 1..T {
        y[t] = 3.0 + 2.0 * m[t - 1] + 0.7 * latent[t - 1] + 0.05 * (t as f64 * 3.1).cos();
    }
    let lagged = |c: &Vec<f64>| c[..T - 1].to_vec();
    let outcome = y[1..].to_vec();
    let reference =
        ols(&[&lagged(&x)], &lagged(&m))[1] * ols(&[&lagged(&m), &lagged(&x)], &outcome)[1];
    let data = TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("m", m.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let m1 = graph.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m1).unwrap();
    graph.insert_directed(m1, y0).unwrap();
    graph.insert_bidirected(x1, y0).unwrap();
    let ctx = ExecutionContext::for_tests(34);
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let label = format!("{suite:?}");
        let result = Study::series(data.clone())
            .graph(graph.clone())
            .query(temporal_derivative(vec![1]))
            .refute(suite)
            .bootstrap_replicates(24)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(result.support_status, Some(CellStatus::Refused), "{label}");
        assert!(report(&result, "derivative.support", &label).passed, "{label}");
        let sensitivity = report(&result, "derivative.front_door_smoothness_sensitivity", &label);
        assert!(
            (sensitivity.refuted_ate - reference).abs() < 0.2,
            "{label}: refit slope {} vs the front-door product {reference}",
            sensitivity.refuted_ate
        );
        assert_eq!(
            has(&result, "derivative.bootstrap_stability"),
            suite == RefuteSuite::Full,
            "{label}"
        );
    }
}

#[test]
fn several_temporal_horizons_are_checked_without_mixing_them() {
    // Template and law of the adjustment fixture: `x(t-1)` reaches `y(t)` through the window
    // adjustment, and horizon 3 has no path.
    let mut cols: [Vec<f64>; 4] = [vec![0.0; T], vec![0.0; T], vec![0.0; T], vec![0.0; T]];
    let mut latent = vec![0.0; T];
    for t in 0..T {
        let clock = t as f64;
        latent[t] = (clock * 0.77 + 2.0).cos();
        cols[3][t] = (clock * 1.7 + 0.3).sin() + 0.8 * latent[t];
        cols[2][t] = (clock * 0.41).cos() + 0.5 * cols[3][t];
        cols[0][t] = 2.0 + 0.6 * cols[2][t] + (clock * 0.71).sin() + 0.2 * (clock * 0.13).cos();
    }
    for t in 1..T {
        cols[1][t] = 5.0
            + 2.0 * cols[0][t - 1]
            + 3.0 * cols[2][t - 1]
            + 0.7 * latent[t - 1]
            + 0.05 * (t as f64 * 3.1).cos();
    }
    let data = TimeSeriesData::from_f64_columns(
        [
            ("x", cols[0].as_slice()),
            ("y", cols[1].as_slice()),
            ("z", cols[2].as_slice()),
            ("w", cols[3].as_slice()),
        ],
        1,
    )
    .unwrap();
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let z1 = graph.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    let w1 = graph.add_lagged(v(3), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x1).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w1, z1).unwrap();
    graph.insert_bidirected(w1, y0).unwrap();
    let ctx = ExecutionContext::for_tests(35);
    let result = Study::series(data)
        .graph(graph)
        .query(temporal_derivative(vec![1, 3]))
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(24)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    assert!(has(&result, "derivative.support"));
    // The support report plus the Pulse-native suite of each of the two horizons' witnesses.
    assert!(result.refutations.len() > 2, "{:?}", result.refutations);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_ref()
                == "refute.temporal_response.pulse_witness_state"),
        "the witness state note"
    );
}
