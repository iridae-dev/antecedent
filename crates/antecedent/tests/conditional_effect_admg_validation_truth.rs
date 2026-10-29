//! `ConditionalEffect` on an ADMG and on a `TemporalAdmg` (explicit, accepted and graph
//! posterior) at every validation depth, against the stratified difference of an exact table.
//!
//! The fixture `conformance/estimate/admg_conditional_strata` counts eight `(z, x, y)` cells
//! with different effects in the two strata of the modifier `z`; the modifier-conditioned
//! contrast is `sum_z P(z) (E[y | x = 1, z] - E[y | x = 0, z]) = 7/27`. A `TemporalAdmg` is read
//! through the contemporaneous projection of its template, so it answers with the same number.
//! A graph posterior mixes atoms that each answer with the executor a supplied ADMG has and
//! states the mass of an atom the identifier refuses. The functional refuter state runs the
//! placebo treatment (exact null 0), subset stability and, under full, the placebo outcome and
//! bootstrap stability of the whole procedure; the Bayesian arm refits at each prior-grid point.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{adjacency_mask_from_admg, set_edge};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Admg, DenseNodeId, TemporalAdmg};
use antecedent_prob::InferenceDiagnostics;

const FIXTURE: &str =
    include_str!("../../../conformance/estimate/admg_conditional_strata/expected.json");

struct Truth {
    cells: Vec<(Vec<f64>, usize)>,
    expected: f64,
    bayes_tolerance: f64,
    null_tolerance: f64,
    stability_tolerance: f64,
    unidentified_mass: f64,
    document: serde_json::Value,
}

fn truth() -> Truth {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let cells = document["table"]
        .as_array()
        .unwrap()
        .iter()
        .map(|cell| {
            let at = |key: &str| cell[key].as_f64().unwrap();
            (vec![at("z"), at("x"), at("y")], cell["count"].as_u64().unwrap() as usize)
        })
        .collect();
    let tolerance = |key: &str| document["tolerance"][key].as_f64().unwrap();
    Truth {
        cells,
        expected: document["expected_ate"].as_f64().unwrap(),
        bayes_tolerance: tolerance("bayesian_posterior_mean"),
        null_tolerance: tolerance("placebo_null"),
        stability_tolerance: tolerance("stability"),
        unidentified_mass: document["graph_posterior"]["unidentified_mass"].as_f64().unwrap(),
        document,
    }
}

fn var(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn table(truth: &Truth) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 3];
    for (levels, count) in &truth.cells {
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(*level, *count));
        }
    }
    TabularData::from_f64_columns(
        ["z", "x", "y"].iter().zip(&columns).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

fn conditional() -> CausalQuery {
    CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(var(1), var(2)).with_effect_modifiers(vec![var(0)]),
        )
        .unwrap(),
    )
}

/// `Z -> X`, `Z -> Y`, `X -> Y`, `Z <-> X`: a latent common cause of the modifier and the
/// treatment. Conditioning on `Z` closes the back door of `X -> Y`.
fn bow_graph() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(0), node(2)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(1)).unwrap();
    graph
}

/// `Z -> X`, `X -> Y`, `Z <-> Y`: latent-confounded without a bow, so an ADMG posterior
/// adjacency can encode it; conditioning on `Z` closes `X <- Z <-> Y`.
fn posterior_graph() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(2)).unwrap();
    graph
}

fn posterior(kind: GraphPosteriorAtomKind, unidentified: bool, truth: &Truth) -> GraphPosterior {
    let weights: Vec<f64> = truth.document["graph_posterior"]["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|weight| weight.as_f64().unwrap())
        .collect();
    let identified = adjacency_mask_from_admg(&posterior_graph()).unwrap();
    // A directed cycle is not an ADMG: the atom stays unidentified.
    let mut cycle = set_edge(0, 3, 0, 1, true);
    cycle = set_edge(cycle, 3, 1, 2, true);
    cycle = set_edge(cycle, 3, 2, 0, true);
    let second = if unidentified { cycle } else { identified.clone() };
    GraphPosterior::from_atoms(
        3,
        weights,
        vec![identified, second],
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("conditional_effect_admg_validation_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind)
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256).prior_scale(1_000.0))
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

fn effect(result: &StudyResult) -> f64 {
    result.estimate().as_effect().unwrap().ate
}

/// The published effect, the status, and the refuters the fixture's contract names for this
/// inference mode and depth, each passing with the null or the original as its expectation.
fn assert_cell(result: &StudyResult, inference: &InferenceMode, suite: RefuteSuite, truth: &Truth) {
    let label = format!("{} {}", inference_name(inference), suite_name(suite));
    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
    let tolerance = if matches!(inference, InferenceMode::Frequentist) {
        truth.document["tolerance"]["frequentist"].as_f64().unwrap()
    } else {
        truth.bayes_tolerance
    };
    assert!(
        (effect(result) - truth.expected).abs() < tolerance,
        "{label}: {} vs stratified difference {}",
        effect(result),
        truth.expected
    );
    if suite == RefuteSuite::None {
        assert!(result.refutations.is_empty(), "{label}");
        return;
    }
    let expected: Vec<&str> = truth.document["refuter_contract"][inference_name(inference)]
        [suite_name(suite)]
    .as_array()
    .unwrap()
    .iter()
    .map(|name| name.as_str().unwrap())
    .collect();
    let mut names: Vec<&str> =
        result.refutations.iter().map(|report| report.refuter.as_ref()).collect();
    names.sort_unstable();
    let mut wanted = expected.clone();
    wanted.sort_unstable();
    assert_eq!(names, wanted, "{label}: the refuters that ran");
    for report in &result.refutations {
        assert!(report.passed, "{label}: {} flags the exact table", report.refuter);
        match report.refuter.as_ref() {
            "target.placebo_treatment" | "target.placebo_outcome" => assert!(
                report.refuted_ate.abs() < truth.null_tolerance,
                "{label}: {} must sit at the null 0, got {}",
                report.refuter,
                report.refuted_ate
            ),
            "target.subset_stability" | "target.bootstrap_stability" => assert!(
                (report.refuted_ate - truth.expected).abs() < truth.stability_tolerance,
                "{label}: {} moved the effect to {}",
                report.refuter,
                report.refuted_ate
            ),
            _ => {}
        }
    }
}

const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

/// `ConditionalEffect` x `Admg` and `TemporalAdmg` x explicit and accepted x Frequentist and
/// Bayesian x none, cheap and full (the `Admg` x none cells are licensed with the cross-check
/// beside this test; here they are read against the fixture too).
#[test]
fn conditional_effect_admg_explicit_and_accepted_match_the_stratified_difference() {
    let truth = truth();
    let data = table(&truth);
    let context = ExecutionContext::for_tests(41);
    let temporal = TemporalAdmg::from_static_contemporaneous(&bow_graph()).unwrap();
    for temporal_graph in [false, true] {
        for accepted in [false, true] {
            for inference in [InferenceMode::Frequentist, bayesian()] {
                for suite in SUITES {
                    let base = Study::tabular(data.clone())
                        .query(conditional())
                        .inference(inference.clone())
                        .refute(suite)
                        .bootstrap_replicates(0);
                    let builder = match (temporal_graph, accepted) {
                        (false, false) => base.graph(bow_graph()),
                        (false, true) => base.graph(AcceptedGraph::from(bow_graph())),
                        (true, false) => base.graph(temporal.clone()),
                        (true, true) => base.graph(AcceptedGraph::from(temporal.clone())),
                    };
                    let result = builder.build().unwrap().run(&context).unwrap_or_else(|error| {
                        panic!("temporal={temporal_graph} accepted={accepted}: {error}")
                    });
                    assert_cell(&result, &inference, suite, &truth);
                    assert!(
                        result.diagnostics.iter().all(|d| d.code.as_ref() != "refute.skipped"),
                        "temporal={temporal_graph} accepted={accepted}: no refuter was skipped"
                    );
                }
            }
        }
    }
}

/// `ConditionalEffect` x `Admg` and `TemporalAdmg` x graph_posterior x Frequentist and Bayesian
/// x none, cheap and full: every identified atom answers with the stratified difference, so
/// the pooled value is the same number; an atom the identifier refuses is stated as
/// unidentified mass and is not redistributed.
#[test]
fn conditional_effect_admg_graph_posterior_states_its_mass_and_matches_the_stratified_difference() {
    let truth = truth();
    let data = table(&truth);
    let context = ExecutionContext::for_tests(42);
    for kind in [GraphPosteriorAtomKind::Admg, GraphPosteriorAtomKind::TemporalAdmg] {
        for unidentified in [false, true] {
            for inference in [InferenceMode::Frequentist, bayesian()] {
                for suite in SUITES {
                    let label = format!("{kind:?} unidentified={unidentified}");
                    let result = Study::tabular(data.clone())
                        .graph_posterior(posterior(kind, unidentified, &truth))
                        .query(conditional())
                        .inference(inference.clone())
                        .refute(suite)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap()
                        .run(&context)
                        .unwrap_or_else(|error| panic!("{label}: {error}"));
                    assert_cell(&result, &inference, suite, &truth);
                    let mixture = result.structural_response.as_ref().expect("atom mixture");
                    let mass = if unidentified { truth.unidentified_mass } else { 0.0 };
                    assert!((mixture.unidentified_mass - mass).abs() < 1e-12, "{label}");
                }
            }
        }
    }
}

/// The retained plan runs every explicit, accepted and graph-posterior cell with the builder
/// dropped; the plan names the IDC identifier and the functional estimator.
#[test]
fn conditional_effect_admg_cells_execute_from_the_retained_plan_with_the_builder_dropped() {
    let truth = truth();
    let data = table(&truth);
    let context = ExecutionContext::for_tests(43);
    let temporal = TemporalAdmg::from_static_contemporaneous(&bow_graph()).unwrap();
    for suite in SUITES {
        for variant in 0..4 {
            let base = Study::tabular(data.clone())
                .query(conditional())
                .inference(InferenceMode::Frequentist)
                .refute(suite)
                .bootstrap_replicates(0);
            let builder = match variant {
                0 => base.graph(bow_graph()),
                1 => base.graph(AcceptedGraph::from(temporal.clone())),
                2 => base.graph_posterior(posterior(GraphPosteriorAtomKind::Admg, false, &truth)),
                _ => base.graph_posterior(posterior(
                    GraphPosteriorAtomKind::TemporalAdmg,
                    true,
                    &truth,
                )),
            };
            let study = builder.clone().build().unwrap();
            let prepared = study.prepare(&context).unwrap();
            drop(builder);
            drop(study);
            let plan = &prepared.plan().logical.record;
            assert_eq!(plan.identifier.as_deref(), Some("general.id"), "variant {variant}");
            assert_eq!(plan.estimator.as_deref(), Some("functional.effect"), "variant {variant}");
            let click = prepared.estimate(&data, &context).unwrap();
            // Frequentist publishes the point only; the Bayesian arm below publishes the
            // posterior-quantile interval of the same contrast.
            assert!(
                click
                    .interval()
                    .as_ref()
                    .map_or(true, |b| b.method == antecedent_core::IntervalMethod::None)
            );
            assert!(
                (effect(&click) - truth.expected).abs() < 1e-9,
                "variant {variant}: {}",
                effect(&click)
            );
            assert_eq!(click.refutations.is_empty(), suite == RefuteSuite::None);
        }
    }
}
