//! The Bayesian refuter state of a static DAG intervention level.
//!
//! Bayesian `InterventionResponse` cheap and full on an explicit, accepted or
//! graph-posterior DAG execute and stay unlicensed: the support status is
//! `refused`, the refutations carry the plugin-level suite on the posterior-mean
//! level, both predictive checks, and under full a prior-sensitivity report whose
//! grid values are refitted posterior-mean levels (not a dose coefficient).
//!
//! The law is `y = 1 + 2 t + 1.5 z + e` with `t` confounded by `z`, so
//! `E[y | do(t = 1)] = 3 + 1.5 E[z]`; the published level is checked against the
//! sample version `3 + 1.5 mean(z)`.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

const N: usize = 1_500;

/// Deterministic uniform stream (xorshift64*), so the fixture needs no RNG crate.
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

/// Columns `t, y, z`: binary `t` with `P(t = 1 | z) = logistic(0.8 z)`,
/// `y = 1 + 2 t + 1.5 z + N(0, 0.5²)`. Returns the table and `mean(z)`.
fn data(seed: u64) -> (TabularData, f64) {
    let mut stream = Stream(seed.max(1));
    let z: Vec<f64> = (0..N).map(|_| stream.normal()).collect();
    let t: Vec<f64> = z
        .iter()
        .map(|z| f64::from(u8::from(stream.uniform() < 1.0 / (1.0 + (-0.8 * z).exp()))))
        .collect();
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + 1.5 * z + 0.5 * stream.normal()).collect();
    let mean_z = z.iter().sum::<f64>() / N as f64;
    let columns = [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())];
    let table = TabularData::from_f64_columns(columns).unwrap();
    (table, mean_z)
}

/// Continuous treatment: `t = 0.6 z + N(0, 1)`, `y = 1 + 2 t + 1.5 z + N(0, 0.5²)`.
fn continuous_data(seed: u64) -> TabularData {
    let mut stream = Stream(seed.max(1));
    let z: Vec<f64> = (0..N).map(|_| stream.normal()).collect();
    let t: Vec<f64> = z.iter().map(|z| 0.6 * z + stream.normal()).collect();
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + 1.5 * z + 0.5 * stream.normal()).collect();
    let columns = [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())];
    TabularData::from_f64_columns(columns).unwrap()
}

/// `z -> t`, `z -> y`, `t -> y` over `(t, y, z)`.
fn graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

/// The same DAG as a two-atom posterior of weight 0.6 / 0.4 (both atoms identical).
fn posterior() -> GraphPosterior {
    let mask = set_edge(set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true), 3, 0, 1, true)
        .to_word()
        .expect("a three-variable graph fits one word");
    let weights = vec![0.6, 0.4];
    let ess = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    GraphPosterior::from_atoms(
        3,
        weights,
        vec![mask, mask],
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess,
            diagnostics: InferenceDiagnostics::analytic("bayesian_level_refutation_fixture"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag)
    .with_algorithm("bayesian_level_refutation_fixture")
}

fn level_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    })
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256))
}

#[derive(Clone, Copy, Debug)]
enum Structure {
    Explicit,
    Accepted,
    GraphPosterior,
}

fn study(data: TabularData, structure: Structure, suite: Option<RefuteSuite>) -> Study {
    let builder = Study::tabular(data);
    let builder = match structure {
        Structure::Explicit => builder.graph(graph()),
        Structure::Accepted => builder.graph(AcceptedGraph::from(graph())),
        Structure::GraphPosterior => builder.graph_posterior(posterior()),
    };
    let builder = builder
        .query(CausalQuery::Response(level_query()))
        .inference(bayesian())
        .bootstrap_replicates(0);
    let builder = match suite {
        Some(suite) => builder.refute(suite),
        None => builder,
    };
    builder.build().unwrap()
}

fn published_level(result: &StudyResult) -> f64 {
    if let Some(structural) = result.structural_response.as_ref() {
        match structural.conditional_on_identified.as_ref() {
            Some(ResponseValue::Scalar(value)) => return *value,
            other => panic!("expected a scalar pooled level, got {other:?}"),
        }
    }
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected a point-identified scalar level, got {other:?}"),
    }
}

fn has_refuter(result: &StudyResult, prefix: &str) -> bool {
    result.refutations.iter().any(|report| report.refuter.starts_with(prefix))
}

#[test]
fn bayesian_level_cheap_and_full_execute_licensed_on_every_dag_structure() {
    let (table, mean_z) = data(41);
    let truth = 3.0 + 1.5 * mean_z;
    for structure in [Structure::Explicit, Structure::Accepted, Structure::GraphPosterior] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("{structure:?} {suite:?}");
            let result =
                study(table.clone(), structure, Some(suite)).run(&ExecutionContext::for_tests(5));
            let result = result.unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
            let level = published_level(&result);
            assert!((level - truth).abs() < 0.03, "{label}: level {level} vs truth {truth}");
            assert!(has_refuter(&result, "overlap"), "{label}: plugin-level overlap runs");
            assert!(has_refuter(&result, "prior_predictive"), "{label}: prior predictive check");
            assert!(has_refuter(&result, "posterior_predictive"), "{label}: posterior check");
            assert_eq!(result.predictive_checks.len(), 2, "{label}");
            let disclosed = result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code.as_ref() == "refute.response.bayesian_state");
            assert!(disclosed, "{label}: the state is disclosed");
            let sensitivity = result
                .refutations
                .iter()
                .find(|report| report.refuter.starts_with("prior_sensitivity"));
            if suite == RefuteSuite::Full {
                let report = sensitivity.unwrap_or_else(|| panic!("{label}: prior sensitivity"));
                // The grid reads back refitted levels: under a weak isotropic prior the
                // widest grid point reproduces the level, not the dose coefficient (2).
                assert!(
                    (report.refuted_ate - report.original_ate).abs() < 0.1,
                    "{label}: grid level {} vs level {}",
                    report.refuted_ate,
                    report.original_ate
                );
                assert!(report.refuted_ate.is_finite() && (report.refuted_ate - 2.0).abs() > 0.5);
            } else {
                assert!(sensitivity.is_none(), "{label}: prior sensitivity is a full check");
            }
        }
    }
}

#[test]
fn the_default_suite_is_the_licensed_cheap_state() {
    // The Bayesian level's cheap state is licensed, so the default suite is no longer
    // downgraded to validation none: it runs the cheap state.
    let (table, _) = data(43);
    for structure in [Structure::Explicit, Structure::GraphPosterior] {
        let result = study(table.clone(), structure, None).run(&ExecutionContext::for_tests(6));
        let result = result.unwrap();
        assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{structure:?}");
        assert!(has_refuter(&result, "overlap"), "{structure:?}");
        if matches!(structure, Structure::Explicit) {
            assert_eq!(result.predictive_checks.len(), 2, "{structure:?}");
        }
    }
}

#[test]
fn the_bayesian_state_does_not_depend_on_the_thread_count() {
    let (table, _) = data(47);
    let run = |threads: u32| {
        study(table.clone(), Structure::Explicit, Some(RefuteSuite::Full))
            .run(&ExecutionContext::production(11, threads))
            .unwrap()
    };
    let (one, four) = (run(1), run(4));
    assert_eq!(published_level(&one).to_bits(), published_level(&four).to_bits());
    assert_eq!(one.refutations.len(), four.refutations.len());
    for (a, b) in one.refutations.iter().zip(&four.refutations) {
        assert_eq!(a.refuter, b.refuter);
        assert_eq!(a.passed, b.passed, "{}", a.refuter);
        assert!(
            a.refuted_ate.to_bits() == b.refuted_ate.to_bits()
                || (a.refuted_ate.is_nan() && b.refuted_ate.is_nan()),
            "{}: {} vs {}",
            a.refuter,
            a.refuted_ate,
            b.refuted_ate
        );
    }
}

#[test]
fn derivative_refuter_state_executes_unlicensed_and_checks_the_finite_difference() {
    let table = continuous_data(53);
    let derivative = ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: VariableId::from_raw(1),
        treatment: VariableId::from_raw(0),
        weighting: antecedent_core::DerivativeWeighting::Observed,
    });
    for (inference, suite) in [
        (InferenceMode::Frequentist, RefuteSuite::Cheap),
        (InferenceMode::Frequentist, RefuteSuite::Full),
        (bayesian(), RefuteSuite::Cheap),
    ] {
        let label = format!("{inference:?} {suite:?}");
        let result = Study::tabular(table.clone())
            .graph(graph())
            .query(CausalQuery::Response(derivative.clone()))
            .inference(inference)
            .refute(suite)
            .build()
            .unwrap_or_else(|error| panic!("{label}: {error}"))
            .run(&ExecutionContext::for_tests(9))
            .unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(result.support_status(), Some(CellStatus::Refused), "{label}");
        assert!(has_refuter(&result, "derivative.support"), "{label}");
        let fd = result
            .refutations
            .iter()
            .find(|report| report.refuter.as_ref() == "derivative.finite_difference")
            .unwrap_or_else(|| panic!("{label}: finite-difference check"));
        // The law is linear in t with slope 2: the shift difference of the identified
        // levels is the slope.
        assert!((fd.refuted_ate - 2.0).abs() < 0.03, "{label}: {}", fd.refuted_ate);
        assert_eq!(
            has_refuter(&result, "derivative.bootstrap_stability"),
            suite == RefuteSuite::Full,
            "{label}"
        );
    }
}
