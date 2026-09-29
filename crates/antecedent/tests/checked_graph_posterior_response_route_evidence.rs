//! Builder-independent evidence for DAG, CPDAG and PAG graph-posterior
//! response routes under frequentist and Bayesian inference.
//!
//! The law is `z -> t -> y`, linear, so `E[y | do(t = a)] = 1 + 2a` whether a
//! completion adjusts `{z}` or nothing. An all-identified posterior pins that
//! truth for every atom kind and suite; a posterior with a reverse-causal atom
//! pins that its unidentified mass is retained through prepare, click, and
//! refresh rather than renormalized away.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, set_edge};
use antecedent_io::consume_analysis_result;
use antecedent_prob::{GraphIdentFlag, InferenceDiagnostics};

/// Columns `t, y, z`: `z` and the treatment noise are deterministic waves,
/// `t = 0.5 z + wave`, `y = shift + 1 + 2 t + small noise`.
fn data(shift: f64) -> TabularData {
    let n = 800usize;
    let wave = |i: usize, freq: f64| (i as f64 * freq).sin();
    let z: Vec<f64> = (0..n).map(|i| wave(i, 0.37)).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.5 * z[i] + 0.8 * wave(i, 0.61)).collect();
    let y: Vec<f64> = (0..n).map(|i| shift + 1.0 + 2.0 * t[i] + 0.05 * wave(i, 0.29)).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// `z -> t -> y` as an adjacency mask over `(t, y, z)`.
fn chain_mask() -> u64 {
    set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true)
}

/// `y -> t` (reverse causal): no adjustment identifies `do(t)`.
fn reverse_mask() -> u64 {
    set_edge(0, 3, 1, 0, true)
}

fn posterior(kind: GraphPosteriorAtomKind, masks: Vec<u64>, weights: Vec<f64>) -> GraphPosterior {
    let ess = 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    let n = masks.len();
    let posterior = GraphPosterior::new(
        3,
        weights,
        masks,
        vec![0.0; 9],
        vec![0.0; 9],
        ess,
        InferenceDiagnostics::analytic("graph_posterior_response_known_truth"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_algorithm("known_truth_fixture");
    if kind == GraphPosteriorAtomKind::Pag {
        posterior.with_mark_masks(vec![0; n]).unwrap()
    } else {
        posterior
    }
}

fn all_identified(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    posterior(kind, vec![chain_mask(), chain_mask()], vec![0.6, 0.4])
}

fn with_unidentified_mass(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    posterior(kind, vec![chain_mask(), reverse_mask()], vec![0.8, 0.2])
}

const GRID: [f64; 2] = [-0.5, 0.5];
const LEVEL: f64 = 0.5;

fn intervention_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(LEVEL))]),
    })
}

fn curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(GRID)),
        ),
    })
}

fn truth(shift: f64, dose: f64) -> f64 {
    shift + 1.0 + 2.0 * dose
}

fn build(
    data: TabularData,
    posterior: GraphPosterior,
    query: ResponseQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    Study::tabular(data)
        .graph_posterior(posterior)
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// Levels of every evaluated atom (`None` for an atom that carries no value).
fn atom_levels(result: &antecedent::StudyResult) -> Vec<Option<Vec<f64>>> {
    result
        .structural_response
        .as_ref()
        .expect("graph-posterior response carries its atom mixture")
        .atoms
        .iter()
        .map(|atom| match &atom.value {
            Some(ResponseValue::Scalar(value)) => Some(vec![*value]),
            Some(ResponseValue::Surface { mean, .. }) => Some(mean.to_vec()),
            None => None,
            other => panic!("unexpected atom value {other:?}"),
        })
        .collect()
}

fn expected_levels(shift: f64, curve: bool) -> Vec<f64> {
    if curve {
        GRID.iter().map(|dose| truth(shift, *dose)).collect()
    } else {
        vec![truth(shift, LEVEL)]
    }
}

fn assert_identified_atoms_match_truth(
    result: &antecedent::StudyResult,
    shift: f64,
    curve: bool,
    tolerance: f64,
    label: &str,
) {
    let want = expected_levels(shift, curve);
    let mut evaluated = 0;
    for atom in atom_levels(result).into_iter().flatten() {
        evaluated += 1;
        assert_eq!(atom.len(), want.len(), "{label}");
        for (got, want) in atom.iter().zip(&want) {
            assert!((got - want).abs() < tolerance, "{label}: level {got} vs truth {want}");
        }
    }
    assert!(evaluated > 0, "{label}: at least one identified atom must carry a value");
}

/// `InterventionResponse`/`ResponseCurve` × `Dag`/`Cpdag`/`Pag` ×
/// `graph_posterior` × Frequentist/Bayesian × the licensed suites: the retained
/// plan freezes the atoms, weights, identification flags, procedure and
/// inference; the prepared click reproduces the one-shot run; every identified
/// atom matches the linear truth; unidentified mass survives prepare, click and
/// refresh; schema-changed data is refused; and the exported artifact names the
/// missing checked graph-posterior response operation to an independent
/// consumer.
#[test]
fn graph_posterior_responses_are_sealed_for_every_atom_kind_and_inference() {
    run_on_large_stack(graph_posterior_responses_body);
}

fn graph_posterior_responses_body() {
    let initial = data(0.0);
    let shifted = data(0.5);
    for kind in
        [GraphPosteriorAtomKind::Dag, GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag]
    {
        for bayesian in [false, true] {
            let inference = if bayesian {
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(96).prior_scale(30.0))
            } else {
                InferenceMode::Frequentist
            };
            let tolerance = 0.15;
            for (name, query, key) in [
                (
                    "intervention",
                    intervention_query(),
                    "dependencies.checked_intervention_response_operation",
                ),
                ("curve", curve_query(), "dependencies.checked_response_grid_operation"),
            ] {
                let curve = name == "curve";
                let suites: Vec<RefuteSuite> =
                    if curve || (bayesian && kind == GraphPosteriorAtomKind::Dag) {
                        vec![RefuteSuite::None]
                    } else {
                        vec![RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full]
                    };
                let expected_estimator = if bayesian {
                    EstimatorId::ResponseBayesian
                } else if curve {
                    EstimatorId::ResponseKennedyDr
                } else {
                    EstimatorId::ResponseInterventionGcomp
                };
                let expected_identifier = if kind == GraphPosteriorAtomKind::Dag {
                    IdentifierId::ResponseBackdoor
                } else {
                    IdentifierId::GeneralizedAdjustment
                };
                for suite in suites {
                    let label = format!("{kind:?} {name} bayesian={bayesian} {suite:?}");
                    let context = ExecutionContext::for_tests(7_331);

                    // All-identified posterior: known truth for every atom and suite.
                    let gp = all_identified(kind);
                    let builder =
                        build(initial.clone(), gp.clone(), query.clone(), inference.clone(), suite);
                    let one_shot = builder.run(&context).unwrap();
                    let mut prepared = builder.prepare(&context).unwrap();
                    drop(builder);
                    let plan =
                        prepared.checked_graph_posterior_response_info().unwrap_or_else(|| {
                            panic!("{label}: retained checked posterior response plan")
                        });
                    assert_eq!(plan.query, query, "{label}");
                    assert_eq!(plan.atom_kind, kind, "{label}");
                    assert_eq!(plan.identifier, expected_identifier, "{label}");
                    assert_eq!(plan.estimator, expected_estimator, "{label}");
                    assert_eq!(plan.inference.starts_with("bayesian:"), bayesian, "{label}");
                    assert_eq!(plan.validation, suite, "{label}");
                    assert_eq!(plan.graph_keys.as_ref(), gp.graph_keys.as_ref(), "{label}");
                    assert_eq!(plan.weights.as_ref(), &[0.6, 0.4], "{label}");
                    assert_eq!(
                        plan.identified.as_ref(),
                        &[GraphIdentFlag::Identified; 2],
                        "{label}"
                    );

                    let click = prepared.estimate(&initial, &context).unwrap();
                    for result in [&one_shot, &click] {
                        assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{label}");
                        assert_eq!(
                            result.logical_plan.estimator.as_deref(),
                            Some(expected_estimator.as_str()),
                            "{label}"
                        );
                        let mixture = result.structural_response.as_ref().unwrap();
                        assert!((mixture.identified_mass - 1.0).abs() < 1e-12, "{label}");
                        assert!(mixture.unidentified_mass.abs() < 1e-12, "{label}");
                        assert_identified_atoms_match_truth(result, 0.0, curve, tolerance, &label);
                        assert_eq!(
                            result.refutations.is_empty(),
                            suite == RefuteSuite::None,
                            "{label}"
                        );
                    }
                    assert!(
                        one_shot
                            .diagnostics
                            .iter()
                            .any(|diagnostic| diagnostic.code.as_ref() == "exec.identify.cached"),
                        "{label}: one-shot run must execute its retained prepared plan"
                    );
                    let click_atoms = atom_levels(&click);
                    let one_shot_atoms = atom_levels(&one_shot);
                    assert_eq!(
                        click_atoms, one_shot_atoms,
                        "{label}: click must reproduce one-shot"
                    );

                    let refreshed = prepared.refresh(shifted.clone(), &context).unwrap();
                    assert_identified_atoms_match_truth(&refreshed, 0.5, curve, tolerance, &label);
                    let retained = prepared.checked_graph_posterior_response_info().unwrap();
                    assert_eq!(retained.graph_keys, plan.graph_keys, "{label}");
                    assert_eq!(retained.weights, plan.weights, "{label}");

                    let two_columns = TabularData::from_f64_columns([
                        ("t", vec![0.0; 8].as_slice()),
                        ("y", vec![0.0; 8].as_slice()),
                    ])
                    .unwrap();
                    let error = prepared.refresh(two_columns, &context).unwrap_err();
                    assert!(
                        error.to_string().contains("same schema"),
                        "{label}: schema-changed refresh must be refused: {error}"
                    );

                    let artifact = prepared
                        .encode_contracted_result(
                            &refreshed,
                            "checked-graph-posterior-response",
                            &context,
                        )
                        .unwrap();
                    let consumed = consume_analysis_result(&artifact).unwrap();
                    for expected in ["dependencies.checked_graph_posterior_response_operation", key]
                    {
                        assert!(
                            consumed
                                .acceptance
                                .unresolved
                                .iter()
                                .any(|reason| reason.as_ref() == expected),
                            "{label}: independent consumption must name {expected}: {:?}",
                            consumed.acceptance.unresolved
                        );
                    }
                    assert!(!consumed.acceptance.accepts_as_verified_program(), "{label}");

                    // Posterior with a reverse-causal atom: its mass is retained, not
                    // renormalized away, through prepare, click and refresh.
                    if suite != RefuteSuite::None {
                        continue;
                    }
                    let mixed_label = format!("{label} unidentified-mass");
                    let gp = with_unidentified_mass(kind);
                    let builder = build(
                        initial.clone(),
                        gp.clone(),
                        query.clone(),
                        inference.clone(),
                        RefuteSuite::None,
                    );
                    let mut prepared = builder.prepare(&context).unwrap();
                    drop(builder);
                    let plan = prepared.checked_graph_posterior_response_info().unwrap();
                    assert_eq!(plan.weights.as_ref(), &[0.8, 0.2], "{mixed_label}");
                    assert_eq!(
                        plan.identified.as_ref(),
                        &[GraphIdentFlag::Identified, GraphIdentFlag::Unidentified],
                        "{mixed_label}"
                    );
                    let click = prepared.estimate(&initial, &context).unwrap();
                    let refreshed = prepared.refresh(shifted.clone(), &context).unwrap();
                    for (result, shift) in [(&click, 0.0), (&refreshed, 0.5)] {
                        let mixture = result.structural_response.as_ref().unwrap();
                        assert!((mixture.identified_mass - 0.8).abs() < 1e-12, "{mixed_label}");
                        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12, "{mixed_label}");
                        let atoms = atom_levels(result);
                        assert_eq!(atoms.len(), 2, "{mixed_label}");
                        assert!(atoms[1].is_none(), "{mixed_label}: reverse atom carries no level");
                        assert_identified_atoms_match_truth(
                            result,
                            shift,
                            curve,
                            tolerance,
                            &mixed_label,
                        );
                    }
                }
            }
        }
    }
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-graph-posterior-response-evidence".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
