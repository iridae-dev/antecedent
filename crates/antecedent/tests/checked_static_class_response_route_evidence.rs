//! Builder-independent evidence for explicit/accepted CPDAG and PAG response
//! routes under frequentist and Bayesian inference.
//!
//! Both laws are linear Gaussian, so every completion's intervention level and
//! curve member has a closed form. The CPDAG fixture has two completions that
//! disagree (adjust `{z}` versus adjust nothing) and stays graph-dependent;
//! the PAG fixture has one visible completion and publishes a point.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, GraphClass, IdentifierId, InferenceMode,
    RefuteSuite, Study,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, DenseNodeId, Pag};
use antecedent_io::consume_analysis_result;

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn gaussian(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            sum += (state >> 11) as f64 / (1u64 << 53) as f64;
        }
        sum - 6.0
    }
}

/// `z — t`, `z -> y`, `t -> y`: completion `z -> t` adjusts `{z}`, completion
/// `t -> z` adjusts nothing.
fn two_completion_cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    graph.insert_directed(d(2), d(1)).unwrap();
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_undirected(d(2), d(0)).unwrap();
    graph
}

/// `z -> t -> y` with `z` not adjacent to `y`: `t -> y` is visible and the
/// single completion identifies every level.
fn chain_pag() -> Pag {
    let mut graph = Pag::with_variables(3);
    graph.insert_directed(d(2), d(0)).unwrap();
    graph.insert_directed(d(0), d(1)).unwrap();
    graph
}

/// Columns `t, y, z`: `z ~ N(0, 1)`, `t = 0.8 z + e`, `y = shift + t + z + e`.
///
/// Adjusting `{z}`: `E[y | do(t = a)] = shift + a`. Adjusting nothing:
/// `E[y | t = a] = shift + a + Cov(z, t) / Var(t) * a = shift + (1 + 0.8 / 1.64) a`.
fn cpdag_data(n: usize, seed: u64, shift: f64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = 0.8 * z[i] + g();
        y[i] = shift + t[i] + z[i] + g();
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// Columns `t, y, z`: `z ~ N(0, 1)`, `t = 0.8 z + e`, `y = shift + 1.5 t + e`,
/// consistent with [`chain_pag`]: `E[y | do(t = a)] = shift + 1.5 a`.
fn pag_data(n: usize, seed: u64, shift: f64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = 0.8 * z[i] + g();
        y[i] = shift + 1.5 * t[i] + g();
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

const GRID: [f64; 2] = [0.0, 1.0];
const LEVEL: f64 = 1.0;

fn intervention_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: v(1),
        interventions: Arc::from([Intervention::set(v(0), Value::f64(LEVEL))]),
    })
}

fn curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from(GRID))),
    })
}

/// Explicit structure passes the class graph itself; accepted structure passes
/// the same graph through an acceptance record. Both must seal identically.
fn build(
    data: TabularData,
    class: GraphClass,
    accepted: bool,
    query: ResponseQuery,
    inference: InferenceMode,
) -> Study {
    let builder = Study::tabular(data);
    let builder = match (class, accepted) {
        (GraphClass::Cpdag, false) => builder.graph(two_completion_cpdag()),
        (GraphClass::Cpdag, true) => builder.graph(AcceptedGraph::from(two_completion_cpdag())),
        (GraphClass::Pag, false) => builder.graph(chain_pag()),
        (GraphClass::Pag, true) => builder.graph(AcceptedGraph::from(chain_pag())),
        (other, _) => panic!("unexpected class {other:?}"),
    };
    builder
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// Every atom value as a vector of levels (one per grid member, or one).
fn atom_levels(result: &antecedent::StudyResult) -> Vec<Option<Vec<f64>>> {
    result
        .structural_response
        .as_ref()
        .expect("class response carries its completion atoms")
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

fn point_levels(result: &antecedent::StudyResult) -> Vec<f64> {
    match &result.response.as_ref().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![*value],
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        other => panic!("expected a point-identified response, got {other:?}"),
    }
}

/// `InterventionResponse`/`ResponseCurve` × `Cpdag`/`Pag` × explicit/accepted ×
/// Frequentist/Bayesian × none: the retained plan fixes the class, the
/// completion envelope, the procedure and the inference mode; the prepared
/// click reproduces the one-shot run; every completion level matches its
/// closed form; refresh re-executes the sealed envelope on shifted data;
/// schema-changed data is refused; and the exported artifact names the
/// missing checked class response operation to an independent consumer.
#[test]
fn static_class_responses_are_sealed_for_both_classes_and_inferences() {
    run_on_large_stack(static_class_responses_body);
}

fn static_class_responses_body() {
    let seed = 4_021;
    let n = 600;
    for class in [GraphClass::Cpdag, GraphClass::Pag] {
        let (initial, shifted) = match class {
            GraphClass::Cpdag => (cpdag_data(n, seed, 0.0), cpdag_data(n, seed, 0.5)),
            _ => (pag_data(n, seed, 0.0), pag_data(n, seed, 0.5)),
        };
        // Closed-form levels per completion, in envelope order, for the base law.
        let doses: &[f64] = &GRID;
        let cpdag_truths = |shift: f64| -> Vec<Vec<f64>> {
            let unadjusted = 1.0 + 0.8 / 1.64;
            vec![
                doses.iter().map(|a| shift + a).collect(),
                doses.iter().map(|a| shift + unadjusted * a).collect(),
            ]
        };
        let pag_truth =
            |shift: f64| -> Vec<f64> { doses.iter().map(|a| shift + 1.5 * a).collect() };
        for bayesian in [false, true] {
            let inference = if bayesian {
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(96).prior_scale(30.0))
            } else {
                InferenceMode::Frequentist
            };
            let tolerance = 0.2;
            for (name, query, key) in [
                (
                    "intervention",
                    intervention_query(),
                    "dependencies.checked_intervention_response_operation",
                ),
                ("curve", curve_query(), "dependencies.checked_response_grid_operation"),
            ] {
                for accepted in [false, true] {
                    let label = format!("{class:?} {name} bayesian={bayesian} accepted={accepted}");
                    let context = ExecutionContext::for_tests(seed);
                    let builder =
                        build(initial.clone(), class, accepted, query.clone(), inference.clone());
                    let one_shot = builder.run(&context).unwrap();
                    let mut prepared = builder.prepare(&context).unwrap();
                    drop(builder);

                    let expected_estimator = if bayesian {
                        EstimatorId::ResponseBayesian
                    } else if name == "curve" {
                        EstimatorId::ResponseKennedyDr
                    } else {
                        EstimatorId::ResponseInterventionGcomp
                    };
                    let plan = prepared
                        .checked_static_class_response_info()
                        .unwrap_or_else(|| panic!("{label}: retained checked class response plan"));
                    assert_eq!(plan.query, query, "{label}");
                    assert_eq!(plan.graph_class, class, "{label}");
                    assert_eq!(plan.identifier, IdentifierId::GeneralizedAdjustment, "{label}");
                    assert_eq!(plan.estimator, expected_estimator, "{label}");
                    assert_eq!(plan.validation, RefuteSuite::None, "{label}");
                    assert_eq!(plan.inference.starts_with("bayesian:"), bayesian, "{label}");
                    let expected_completions = if class == GraphClass::Cpdag { 2 } else { 1 };
                    assert_eq!(plan.completion_count, expected_completions, "{label}");
                    assert!(plan.identified_mass > 0.0, "{label}");
                    assert!(plan.unidentified_mass.abs() < 1e-12, "{label}");

                    let click = prepared.estimate(&initial, &context).unwrap();
                    for result in [&one_shot, &click] {
                        assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{label}");
                        assert_eq!(
                            result.logical_plan.estimator.as_deref(),
                            Some(expected_estimator.as_str()),
                            "{label}"
                        );
                        assert!(result.refutations.is_empty(), "{label}: validation none");
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
                    assert_eq!(click_atoms.len(), expected_completions, "{label}");
                    for (click_atom, one_shot_atom) in click_atoms.iter().zip(&one_shot_atoms) {
                        let (Some(click_atom), Some(one_shot_atom)) = (click_atom, one_shot_atom)
                        else {
                            panic!("{label}: every completion is evaluable");
                        };
                        for (a, b) in click_atom.iter().zip(one_shot_atom) {
                            assert!((a - b).abs() < 1e-9, "{label}: click {a} vs one-shot {b}");
                        }
                    }
                    let assert_levels = |result: &antecedent::StudyResult,
                                         shift: f64,
                                         tag: &str| {
                        let atoms = atom_levels(result);
                        let level_count = if name == "curve" { GRID.len() } else { 1 };
                        if class == GraphClass::Cpdag {
                            let truths = cpdag_truths(shift);
                            // Disagreeing completions publish the identified set over
                            // the completions, never a weighted mean of them.
                            assert_eq!(
                                format!("{:?}", result.identification.status),
                                "PartiallyIdentified",
                                "{label} {tag}: disagreeing completions stay set-valued"
                            );
                            let ResponseIdentification::PartiallyIdentified(_) =
                                &result.response.as_ref().unwrap().estimate
                            else {
                                panic!("{label} {tag}: expected a completion identified set");
                            };
                            let mixture = result.structural_response.as_ref().unwrap();
                            let set = mixture
                                .identified_set
                                .as_ref()
                                .unwrap_or_else(|| panic!("{label} {tag}: identified set"));
                            let evaluated: Vec<&Vec<f64>> =
                                atoms.iter().map(|atom| atom.as_ref().unwrap()).collect();
                            let mut spread: f64 = 0.0;
                            for (member, (lower, upper)) in
                                set.lower.iter().zip(set.upper.iter()).enumerate()
                            {
                                let lo = evaluated
                                    .iter()
                                    .map(|atom| atom[member])
                                    .fold(f64::INFINITY, f64::min);
                                let hi = evaluated
                                    .iter()
                                    .map(|atom| atom[member])
                                    .fold(f64::NEG_INFINITY, f64::max);
                                assert!(
                                    (lower - lo).abs() < 1e-9 && (upper - hi).abs() < 1e-9,
                                    "{label} {tag}: set [{lower}, {upper}] vs completions [{lo}, {hi}]"
                                );
                                spread = spread.max(hi - lo);
                            }
                            assert!(spread > 0.1, "{label} {tag}: completions must disagree");
                            let wants: Vec<Vec<f64>> =
                                truths
                                    .iter()
                                    .map(|truth| {
                                        if name == "curve" { truth.clone() } else { vec![truth[1]] }
                                    })
                                    .collect();
                            let mut seen = vec![false; wants.len()];
                            for atom in atoms.iter().map(|atom| atom.as_ref().unwrap()) {
                                assert_eq!(atom.len(), level_count, "{label} {tag}");
                                let matched = wants.iter().position(|want| {
                                    atom.len() == want.len()
                                        && atom
                                            .iter()
                                            .zip(want)
                                            .all(|(got, want)| (got - want).abs() < tolerance)
                                });
                                let Some(index) = matched else {
                                    panic!(
                                        "{label} {tag}: completion levels {atom:?} match no closed form {wants:?}"
                                    );
                                };
                                seen[index] = true;
                            }
                            assert!(seen.iter().all(|seen| *seen), "{label} {tag}: {seen:?}");
                        } else {
                            let truth = pag_truth(shift);
                            let published = point_levels(result);
                            let want: Vec<f64> =
                                if name == "curve" { truth.clone() } else { vec![truth[1]] };
                            assert_eq!(published.len(), want.len(), "{label} {tag}");
                            for (got, want) in published.iter().zip(&want) {
                                assert!(
                                    (got - want).abs() < tolerance,
                                    "{label} {tag}: level {got} vs truth {want}"
                                );
                            }
                            for atom in atoms.iter().map(|atom| atom.as_ref().unwrap()) {
                                assert_eq!(atom, &published, "{label} {tag}");
                            }
                        }
                    };
                    assert_levels(&click, 0.0, "click");

                    let refreshed = prepared.refresh(shifted.clone(), &context).unwrap();
                    assert_levels(&refreshed, 0.5, "refresh");
                    assert!(prepared.checked_static_class_response_info().is_some(), "{label}");

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
                            "checked-static-class-response",
                            &context,
                        )
                        .unwrap();
                    let consumed = consume_analysis_result(&artifact).unwrap();
                    for expected in ["dependencies.checked_static_class_response_operation", key] {
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
                }
            }
        }
    }
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-static-class-response-evidence".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
