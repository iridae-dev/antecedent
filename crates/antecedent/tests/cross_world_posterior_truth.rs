//! The natural direct effect over a graph posterior of Dag, ADMG, temporal-ADMG, CPDAG and PAG atoms,
//! on both inference sides and under every validation suite, against the posterior-weighted mean
//! of each atom's own closed form (`conformance/estimate/cross_world_class_truth` and
//! `conformance/estimate/cross_world_dag_validation`).
//!
//! An atom the identifier refuses is unidentified posterior mass; it is not redistributed, and
//! the status resolves to graph-dependent. Each atom runs its own refuter suite against its own
//! estimate and the reports mix by posterior weight.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "test fixtures: one coordinate walk per test"
)]
use antecedent::analysis::RetainedOperation;
use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, IdentificationStatus, NestedCounterfactualQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg,
    adjacency_mask_from_cpdag, set_edge,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_class_truth/expected.json"
    ))
    .unwrap()
}
fn node(r: u32) -> DenseNodeId {
    DenseNodeId::from_raw(r)
}
fn v(r: u32) -> VariableId {
    VariableId::from_raw(r)
}
fn table(pin: &serde_json::Value, name: &str) -> TabularData {
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
            for (c, l) in data.iter_mut().zip(&levels) {
                c.push(*l);
            }
        }
    }
    TabularData::from_f64_columns(names.iter().zip(&data).map(|(n, c)| (n.as_str(), c.as_slice())))
        .unwrap()
}
fn dag_mask(n: usize, edges: &[(usize, usize)]) -> u64 {
    edges.iter().fold(0_u64, |m, &(a, b)| set_edge(m, n, a, b, true))
}
fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("x"),
        rejected_invalid: 0,
    }
}
fn gp(n: usize, kind: GraphPosteriorAtomKind, w: &[f64], masks: Vec<u64>) -> GraphPosterior {
    let mut g = GraphPosterior::new(
        n,
        w.to_vec(),
        masks.clone(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling(),
    )
    .unwrap()
    .with_atom_kind(kind);
    if kind == GraphPosteriorAtomKind::Pag {
        g = g.with_mark_masks(vec![0; masks.len()]).unwrap();
    }
    g
}
fn admg(edges: &[(u32, u32)]) -> Admg {
    let mut g = Admg::with_variables(3);
    for &(a, b) in edges {
        g.insert_directed(node(a), node(b)).unwrap();
    }
    g
}
fn dag(edges: &[(u32, u32)]) -> Dag {
    let mut g = Dag::with_variables(3);
    for &(a, b) in edges {
        g.insert_directed(node(a), node(b)).unwrap();
    }
    g
}

struct Posterior {
    kind: &'static str,
    graphs: GraphPosterior,
    data: TabularData,
    query: CausalQuery,
    expected: f64,
    tolerance: f64,
}

fn nested(t: u32, m: u32, y: u32) -> CausalQuery {
    CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(v(t), v(m), v(y)).unwrap())
}

fn posteriors() -> Vec<Posterior> {
    let pin = pin();
    let expected = pin["posterior"]["expected"].as_f64().unwrap();
    let a = [(0, 1), (0, 2), (1, 2)];
    let b = [(0, 2), (1, 2)];
    let cyclic = dag_mask(3, &[(0, 1), (1, 2), (2, 0)]);
    let w = [0.5, 0.3, 0.2];
    let mut out = Vec::new();
    let discrete = |kind, graphs| Posterior {
        kind,
        graphs,
        data: table(&pin, "front_door"),
        query: nested(0, 1, 2),
        expected,
        tolerance: 1e-9,
    };
    out.push(discrete(
        "Admg",
        gp(
            3,
            GraphPosteriorAtomKind::Admg,
            &w,
            vec![
                adjacency_mask_from_admg(&admg(&a)).unwrap(),
                adjacency_mask_from_admg(&admg(&b)).unwrap(),
                cyclic,
            ],
        ),
    ));
    out.push(discrete(
        "TemporalAdmg",
        gp(
            3,
            GraphPosteriorAtomKind::TemporalAdmg,
            &w,
            vec![
                adjacency_mask_from_admg(&admg(&a)).unwrap(),
                adjacency_mask_from_admg(&admg(&b)).unwrap(),
                cyclic,
            ],
        ),
    ));
    out.push(discrete(
        "Cpdag",
        gp(
            3,
            GraphPosteriorAtomKind::Cpdag,
            &w,
            vec![
                adjacency_mask_from_cpdag(&Cpdag::from_dag(&dag(&a))).unwrap(),
                adjacency_mask_from_cpdag(&Cpdag::from_dag(&dag(&b))).unwrap(),
                cyclic,
            ],
        ),
    ));
    // A PAG whose edges are all visible beside a reverse-causal atom that leaves the treatment edge
    // invisible: one identified atom carries the whole conditional value.
    let visible = [(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        .iter()
        .fold(0_u64, |m, &(x, z)| set_edge(m, 5, x, z, true));
    let reverse = [(2, 0), (1, 3), (2, 3), (2, 4), (3, 4)]
        .iter()
        .fold(0_u64, |m, &(x, z)| set_edge(m, 5, x, z, true));
    out.push(Posterior {
        kind: "Pag",
        graphs: gp(5, GraphPosteriorAtomKind::Pag, &[0.8, 0.2], vec![visible, reverse]),
        data: table(&pin, "mediated_two_parents"),
        query: nested(2, 3, 4),
        expected: pin["cases"]["pag_natural_direct"]["expected"].as_f64().unwrap(),
        tolerance: 1e-9,
    });
    out.push(discrete(
        "Dag",
        gp(
            3,
            GraphPosteriorAtomKind::Dag,
            &w,
            vec![dag_mask(3, &[(0, 1), (0, 2), (1, 2)]), dag_mask(3, &[(0, 2), (1, 2)]), cyclic],
        ),
    ));
    out
}

fn run(
    p: &Posterior,
    bayesian: bool,
    suite: RefuteSuite,
) -> Result<(antecedent::analysis::Study, antecedent::StudyResult), antecedent::CausalError> {
    let inference = if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100))
    } else {
        InferenceMode::Frequentist
    };
    let study = Study::tabular(p.data.clone())
        .graph_posterior(p.graphs.clone())
        .query(p.query.clone())
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()?;
    let result = study.run(&ExecutionContext::for_tests(3))?;
    Ok((study, result))
}

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join().unwrap();
}

#[test]
fn nested_graph_posterior_is_the_weighted_mean_of_the_atoms_closed_forms_under_every_suite() {
    on_large_stack(|| {
        let pin = pin();
        let unidentified = pin["posterior"]["unidentified_mass"].as_f64().unwrap();
        for p in posteriors() {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    if p.kind == "Dag" && suite == RefuteSuite::Full {
                        continue;
                    }
                    let label = format!("{} bayesian={bayesian} {suite:?}", p.kind);
                    let (study, result) =
                        run(&p, bayesian, suite).unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(study.support_status(), Some(CellStatus::Licensed), "{label}");
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    let tolerance = if bayesian { p.tolerance.max(0.03) } else { p.tolerance };
                    assert!(
                        (result.effect() - p.expected).abs() <= tolerance,
                        "{label}: {} vs {}",
                        result.effect(),
                        p.expected
                    );
                    // The refused atom is unidentified posterior mass, not redistributed.
                    let mixture = result.structural_response.as_ref().expect(&label);
                    assert!((mixture.unidentified_mass - unidentified).abs() < 1e-9, "{label}");
                    assert_eq!(
                        result.identification.status,
                        IdentificationStatus::GraphDependent,
                        "{label}"
                    );
                    if suite == RefuteSuite::None {
                        assert!(result.refutations.is_empty(), "{label}");
                        continue;
                    }
                    let placebo = result
                        .refutations
                        .iter()
                        .find(|r| r.refuter.as_ref() == "target.placebo_treatment")
                        .unwrap_or_else(|| panic!("{label}: no placebo"));
                    assert!(placebo.passed, "{label}");
                    // A mixed verdict passes only if every contributing atom passes.
                    assert!(
                        result.refutations.iter().all(|r| r.passed),
                        "{label}: {:?}",
                        result
                            .refutations
                            .iter()
                            .filter(|r| !r.passed)
                            .map(|r| r.refuter.to_string())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    });
}

#[test]
fn nested_graph_posterior_executes_from_the_retained_plan_without_the_builder() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(4);
        for p in posteriors() {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    if p.kind == "Dag" && suite == RefuteSuite::Full {
                        continue;
                    }
                    let label = format!("{} bayesian={bayesian} {suite:?}", p.kind);
                    let inference = if bayesian {
                        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100))
                    } else {
                        InferenceMode::Frequentist
                    };
                    let builder = Study::tabular(p.data.clone())
                        .graph_posterior(p.graphs.clone())
                        .query(p.query.clone())
                        .inference(inference)
                        .refute(suite)
                        .bootstrap_replicates(0);
                    let study = builder.clone().build().unwrap();
                    let prepared = study.prepare(&ctx).unwrap();
                    drop(builder);
                    drop(study);
                    let operation = format!("{:?}", prepared.retained_operation());
                    assert!(
                        !matches!(prepared.retained_operation(), RetainedOperation::FreshRun),
                        "{label}: {operation}"
                    );
                    let result = prepared.estimate(&p.data, &ctx).unwrap();
                    let tolerance = if bayesian { p.tolerance.max(0.03) } else { p.tolerance };
                    assert!(
                        (result.effect() - p.expected).abs() <= tolerance,
                        "{label}: {}",
                        result.effect()
                    );
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                }
            }
        }
    });
}

#[test]
fn a_full_suite_over_dag_atoms_is_refused_with_its_reason() {
    let dag = posteriors().into_iter().find(|p| p.kind == "Dag").unwrap();
    for bayesian in [false, true] {
        let inference = if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100))
        } else {
            InferenceMode::Frequentist
        };
        let error = Study::tabular(dag.data.clone())
            .graph_posterior(dag.graphs.clone())
            .query(dag.query.clone())
            .inference(inference)
            .refute(RefuteSuite::Full)
            .bootstrap_replicates(0)
            .build()
            .unwrap_err()
            .to_string();
        assert!(error.starts_with("refused:"), "{error}");
        assert!(error.contains("Dag atoms"), "{error}");
    }
}
