//! The Bayesian refuter state of a static DAG `InterventionResponse` against known truth.
//!
//! The fixture `conformance/estimate/bayesian_level_refuter_states` states the law
//! `y = 1 + 2 t + 1.5 z + noise` with the treatment confounded by the wave covariate
//! `z_i = sin(0.37 i)`, so the adjusted level `E[y | do(t = 1)]` is `3 + 1.5 mean(z)`, and the
//! refuters a Bayesian level study runs at each validation depth. On the clean law every
//! refuter passes and the published level is the closed form; a planted positivity violation
//! (the treatment is a deterministic function of the covariate) is detected by the overlap
//! assessment and a planted curvature in the covariate by the posterior predictive check.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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

const FIXTURE: &str =
    include_str!("../../../conformance/estimate/bayesian_level_refuter_states/expected.json");

/// Deterministic uniform stream (xorshift64*), so the noise needs no RNG crate.
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

struct Contract {
    rows: usize,
    level: f64,
    level_tolerance: f64,
    grid_tolerance: f64,
    cheap: Vec<String>,
    full: Vec<String>,
    predictive_checks: usize,
}

fn contract() -> Contract {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let names = |suite: &str| -> Vec<String> {
        document["refuter_contract"][suite]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| name.as_str().unwrap().to_owned())
            .collect()
    };
    Contract {
        rows: document["design"]["rows"].as_u64().unwrap() as usize,
        level: document["level"]["value"].as_f64().unwrap(),
        level_tolerance: document["tolerance"]["level"].as_f64().unwrap(),
        grid_tolerance: document["tolerance"]["prior_grid_level"].as_f64().unwrap(),
        cheap: names("cheap"),
        full: names("full"),
        predictive_checks: document["refuter_contract"]["predictive_checks"].as_u64().unwrap()
            as usize,
    }
}

/// Columns `t, y, z`. `z_i = sin(0.37 i)`. Clean: `P(t = 1 | z) = 0.5 + 0.25 z` through the
/// golden-ratio sequence. `separate`: `t = 1[z > 0]`. `curved`: the outcome gains `3 (z² - 1)`.
fn data(rows: usize, seed: u64, separate: bool, curved: bool) -> TabularData {
    let mut stream = Stream(seed);
    let z: Vec<f64> = (0..rows).map(|i| (0.37 * i as f64).sin()).collect();
    let t: Vec<f64> = z
        .iter()
        .enumerate()
        .map(|(i, z)| {
            if separate {
                f64::from(*z > 0.0)
            } else {
                let u = (i as f64 * 0.618_033_988_749_894_9).fract();
                f64::from(u < 0.5 + 0.25 * z)
            }
        })
        .collect();
    let y: Vec<f64> = t
        .iter()
        .zip(&z)
        .map(|(t, z)| {
            let bend = if curved { 3.0 * (z * z - 1.0) } else { 0.0 };
            1.0 + 2.0 * t + 1.5 * z + bend + 0.5 * stream.normal()
        })
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
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
            diagnostics: InferenceDiagnostics::analytic("bayesian_level_refuter_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Dag)
    .with_algorithm("bayesian_level_refuter_truth")
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

fn study(data: TabularData, structure: Structure, suite: RefuteSuite) -> Study {
    let builder = Study::tabular(data);
    let builder = match structure {
        Structure::Explicit => builder.graph(graph()),
        Structure::Accepted => builder.graph(AcceptedGraph::from(graph())),
        Structure::GraphPosterior => builder.graph_posterior(posterior()),
    };
    builder
        .query(CausalQuery::Response(level_query()))
        .inference(bayesian())
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
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

fn report<'a>(
    result: &'a StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

/// `InterventionResponse` x `Dag` x explicit, accepted and graph_posterior x `Bayesian` x cheap
/// and full: on the clean law the level is the closed form and every refuter the fixture names
/// runs and passes; a full study adds the refitted prior-grid level.
#[test]
fn bayesian_level_cheap_and_full_pass_on_the_clean_law_at_the_closed_form_level() {
    let contract = contract();
    let table = data(contract.rows, 41, false, false);
    for structure in [Structure::Explicit, Structure::Accepted, Structure::GraphPosterior] {
        for (suite, expected) in
            [(RefuteSuite::Cheap, &contract.cheap), (RefuteSuite::Full, &contract.full)]
        {
            let label = format!("{structure:?} {suite:?}");
            let result = study(table.clone(), structure, suite)
                .run(&ExecutionContext::for_tests(5))
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
            let level = published_level(&result);
            assert!(
                (level - contract.level).abs() < contract.level_tolerance,
                "{label}: level {level} vs closed form {}",
                contract.level
            );
            let names: Vec<&str> =
                result.refutations.iter().map(|report| report.refuter.as_ref()).collect();
            assert_eq!(names.len(), expected.len(), "{label}: {names:?}");
            for name in expected {
                let found = report(&result, name)
                    .unwrap_or_else(|| panic!("{label}: refuter {name} did not run: {names:?}"));
                assert!(found.passed, "{label}: {name} flags the clean law");
            }
            assert_eq!(result.predictive_checks.len(), contract.predictive_checks, "{label}");
            if let Some(grid) = report(&result, "prior_sensitivity") {
                // The grid reads back refitted levels, not the dose coefficient (2).
                assert!(
                    (grid.refuted_ate - grid.original_ate).abs() < contract.grid_tolerance,
                    "{label}: grid level {} vs level {}",
                    grid.refuted_ate,
                    grid.original_ate
                );
                assert!((grid.refuted_ate - 2.0).abs() > 0.5, "{label}: the grid is a level");
            }
        }
    }
}

/// A planted violation is detected by the refuter built for it, and only on the planted law.
#[test]
fn bayesian_level_refuters_detect_planted_positivity_and_curvature_violations() {
    let contract = contract();
    let context = ExecutionContext::for_tests(6);
    let run = |table: TabularData| {
        study(table, Structure::Explicit, RefuteSuite::Full).run(&context).unwrap()
    };
    let separated = run(data(contract.rows, 41, true, false));
    assert!(!report(&separated, "overlap.assessment").unwrap().passed, "positivity violation");
    assert!(!report(&separated, "overlap.rule").unwrap().passed, "positivity violation");
    let curved = run(data(contract.rows, 41, false, true));
    assert!(!report(&curved, "posterior_predictive").unwrap().passed, "planted curvature");
    assert!(report(&curved, "overlap.assessment").unwrap().passed, "overlap is not the culprit");
    let clean = run(data(contract.rows, 41, false, false));
    assert!(clean.refutations.iter().all(|report| report.passed), "the clean law passes");
}

/// The default suite keeps the licensed validation `none`, and a requested cheap or full
/// study executes from the retained plan with the builder dropped.
#[test]
fn bayesian_level_refuters_execute_from_the_retained_plan_with_the_builder_dropped() {
    let contract = contract();
    let table = data(contract.rows, 41, false, false);
    let context = ExecutionContext::for_tests(7);
    for structure in [Structure::Explicit, Structure::Accepted, Structure::GraphPosterior] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("{structure:?} {suite:?}");
            let builder = study(table.clone(), structure, suite);
            let prepared =
                builder.prepare(&context).unwrap_or_else(|error| panic!("{label}: {error}"));
            drop(builder);
            assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{label}");
            assert!(prepared.plan().logical.record.estimator.is_some(), "{label}");
            let result = prepared
                .estimate(&table, &context)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert!(!result.refutations.is_empty(), "{label}: the retained plan runs the state");
            assert!((published_level(&result) - contract.level).abs() < contract.level_tolerance);
        }
    }
}
