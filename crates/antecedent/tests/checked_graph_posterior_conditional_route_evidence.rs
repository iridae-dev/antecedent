//! Builder-independent evidence for conditional effects over static graph posteriors.
//!
//! DAG, CPDAG, and PAG posterior atoms are sealed for frequentist conditional
//! linear adjustment and Bayesian conditional g-computation under every
//! built-in validation suite.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one loop covers every licensed coordinate")]

use std::sync::Arc;

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, SlotAvailability,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, set_edge};
use antecedent_io::consume_analysis_result;
use antecedent_prob::{GraphIdentFlag, InferenceDiagnostics};

/// Balanced confounded blocks with `Y = shift + 2T + slope * Z + eps`.
fn blocked_columns(outcome_shift: f64, confounder_slope: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 320usize;
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut confounder = Vec::with_capacity(n);
    for _ in 0..(n / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(outcome_shift + 2.0 * t + confounder_slope * z + epsilon);
            }
        }
    }
    (treatment, outcome, confounder)
}

/// `Y = 2T + 2Z + eps`; `z` is both the confounder and the effect modifier,
/// so every identified atom's interaction model recovers the structural slope 2.
fn confounded_data(outcome_shift: f64) -> TabularData {
    let (t, y, z) = blocked_columns(outcome_shift, 2.0);
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// `Y = 2T + eps` with `Z -> T`: the modifier carries no outcome effect.
fn unconfounded_data(outcome_shift: f64) -> TabularData {
    let (t, y, z) = blocked_columns(outcome_shift, 0.0);
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// Same columns as [`confounded_data`] in a different order: a different
/// semantic schema that a prepared handle must refuse on refresh.
fn reordered_data() -> TabularData {
    let (t, y, z) = blocked_columns(0.0, 2.0);
    TabularData::from_f64_columns([("z", z.as_slice()), ("y", y.as_slice()), ("t", t.as_slice())])
        .unwrap()
}

/// Known-truth mixture: direct-only, `Z`-adjusted, and reverse-causal atoms
/// with weights 0.5 / 0.3 / 0.2. The reverse atom's mass stays unidentified.
fn mixture_posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    let weights = Arc::<[f64]>::from([0.5, 0.3, 0.2]);
    let mut marginals = vec![0.0; 9];
    marginals[1] = weights[0] + weights[1];
    marginals[3] = weights[2];
    marginals[6] = weights[1];
    marginals[7] = weights[1];
    GraphPosterior::new(
        3,
        weights.clone(),
        vec![direct, adjusted, unidentified],
        marginals.clone(),
        marginals,
        1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_algorithm("known_truth_fixture")
}

/// `Z -> T -> Y` makes `T -> Y` visible in this PAG; a reverse-causal atom
/// keeps a separate unidentified 0.2 sample mass.
fn visible_pag_posterior() -> GraphPosterior {
    let identified = set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true);
    let reverse = set_edge(0, 3, 1, 0, true);
    let weights = Arc::<[f64]>::from([0.8, 0.2]);
    let mut marginals = vec![0.0; 9];
    marginals[1] = weights[0];
    marginals[3] = weights[1];
    marginals[6] = weights[0];
    GraphPosterior::new(
        3,
        weights.clone(),
        vec![identified, reverse],
        marginals.clone(),
        marginals,
        1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
        InferenceDiagnostics::analytic("visible_pag_known_truth"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_masks(vec![0, 0])
    .unwrap()
    .with_algorithm("visible_pag_fixture")
}

struct Fixture {
    graph_class: &'static str,
    posterior: GraphPosterior,
    base: TabularData,
    shifted: TabularData,
    /// Number of identified atoms whose conditional effect must be 2.
    identified_atoms: usize,
    /// Whether the outer scalar is mixable (identical estimands across atoms).
    scalar_mixable: bool,
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            graph_class: "Dag",
            posterior: mixture_posterior(GraphPosteriorAtomKind::Dag),
            base: confounded_data(0.0),
            shifted: confounded_data(4.0),
            identified_atoms: 2,
            scalar_mixable: false,
        },
        Fixture {
            graph_class: "Cpdag",
            posterior: mixture_posterior(GraphPosteriorAtomKind::Cpdag),
            base: confounded_data(0.0),
            shifted: confounded_data(4.0),
            identified_atoms: 2,
            scalar_mixable: false,
        },
        Fixture {
            graph_class: "Pag",
            posterior: visible_pag_posterior(),
            base: unconfounded_data(0.0),
            shifted: unconfounded_data(4.0),
            identified_atoms: 1,
            scalar_mixable: true,
        },
    ]
}

struct RetainedPlan {
    query: AverageEffectQuery,
    conditional: bool,
    estimator: &'static str,
    validation: RefuteSuite,
    graph_keys: Arc<[u64]>,
    weights: Arc<[f64]>,
    identified: Arc<[GraphIdentFlag]>,
}

fn retained_plan(
    prepared: &antecedent::analysis::PreparedStudy,
    graph_class: &str,
    bayesian: bool,
) -> RetainedPlan {
    match (graph_class, bayesian) {
        ("Dag", false) => {
            let info = prepared
                .checked_graph_posterior_effect_info()
                .expect("frequentist DAG graph-posterior operation retained");
            assert!(prepared.checked_bayesian_graph_posterior_ate_info().is_none());
            RetainedPlan {
                query: info.query,
                conditional: info.conditional,
                estimator: info.estimator.as_str(),
                validation: info.validation,
                graph_keys: info.graph_keys,
                weights: info.weights,
                identified: info.identified,
            }
        }
        ("Dag", true) => {
            let info = prepared
                .checked_bayesian_graph_posterior_ate_info()
                .expect("Bayesian DAG graph-posterior operation retained");
            assert!(info.inference.starts_with("bayesian:"));
            assert!(!prepared.has_checked_graph_posterior_effect_operation());
            RetainedPlan {
                query: info.query,
                conditional: info.conditional,
                estimator: info.estimator.as_str(),
                validation: info.validation,
                graph_keys: info.graph_keys,
                weights: info.weights,
                identified: info.identified,
            }
        }
        _ => {
            let info = prepared
                .checked_class_graph_posterior_effect_info()
                .expect("class graph-posterior operation retained");
            assert!(prepared.has_checked_class_graph_posterior_effect_operation());
            assert_eq!(
                info.inference.starts_with("bayesian:"),
                bayesian,
                "retained inference must match the sealed click"
            );
            assert_eq!(info.class_atom_count, if graph_class == "Pag" { 1 } else { 2 });
            RetainedPlan {
                query: info.query,
                conditional: info.conditional,
                estimator: info.estimator.as_str(),
                validation: info.validation,
                graph_keys: info.graph_keys,
                weights: info.weights,
                identified: info.identified,
            }
        }
    }
}

fn atom_values(result: &antecedent::StudyResult) -> Vec<f64> {
    result
        .structural_response
        .as_ref()
        .expect("structural graph mixture")
        .atoms
        .iter()
        .filter_map(|atom| match atom.value {
            Some(antecedent_core::ResponseValue::Scalar(value)) => Some(value),
            _ => None,
        })
        .collect()
}

#[test]
fn graph_posterior_conditional_effect_is_sealed_across_atom_kinds_inferences_and_suites() {
    run_on_large_stack(graph_posterior_conditional_effect_body);
}

fn graph_posterior_conditional_effect_body() {
    let ctx = ExecutionContext::for_tests(2_207);
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    let config = BayesianConfig::conjugate().n_draws(128).prior_scale(100.0);

    for fixture in fixtures() {
        for bayesian in [false, true] {
            for (label, suite) in [
                ("none", RefuteSuite::None),
                ("cheap", RefuteSuite::Cheap),
                ("full", RefuteSuite::Full),
            ] {
                let coordinate = format!(
                    "ConditionalEffect:{}:graph_posterior:{}:{label}",
                    fixture.graph_class,
                    if bayesian { "Bayesian" } else { "Frequentist" }
                );
                let (inference, estimator, dependency, tolerance) = if bayesian {
                    (
                        InferenceMode::Bayesian(config.clone()),
                        "conditional.bayesian",
                        if fixture.graph_class == "Dag" {
                            "dependencies.checked_bayesian_graph_posterior_ate_operation"
                        } else {
                            "dependencies.checked_class_graph_posterior_effect_operation"
                        },
                        0.2,
                    )
                } else {
                    (
                        InferenceMode::Frequentist,
                        "conditional.linear.adjustment",
                        if fixture.graph_class == "Dag" {
                            "dependencies.checked_graph_posterior_effect_operation"
                        } else {
                            "dependencies.checked_class_graph_posterior_effect_operation"
                        },
                        0.06,
                    )
                };
                let builder = Study::tabular(fixture.base.clone())
                    .graph_posterior(fixture.posterior.clone())
                    .query(CausalQuery::ConditionalEffect(query.clone()))
                    .inference(inference)
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                let one_shot = builder.run(&ctx).unwrap();
                let mut prepared = builder.prepare(&ctx).unwrap();
                drop(builder);

                assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{coordinate}");
                match &prepared.contract().unwrap().reasoning.support {
                    SlotAvailability::Available(slot) => {
                        assert_eq!(slot.matrix_coordinate.as_deref(), Some(coordinate.as_str()));
                    }
                    other => panic!("{coordinate}: support was not available: {other:?}"),
                }
                let plan = retained_plan(&prepared, fixture.graph_class, bayesian);
                assert!(plan.conditional, "{coordinate}: plan must record the conditional target");
                assert_eq!(plan.query, query.inner, "{coordinate}");
                assert_eq!(plan.estimator, estimator, "{coordinate}");
                assert_eq!(plan.validation, suite, "{coordinate}");
                assert_eq!(plan.graph_keys.as_ref(), fixture.posterior.graph_keys.as_ref());
                assert_eq!(plan.weights.as_ref(), fixture.posterior.weights.as_ref());
                assert_eq!(plan.identified.len(), fixture.posterior.n_graphs);
                assert_eq!(
                    plan.identified.iter().filter(|f| **f == GraphIdentFlag::Unidentified).count(),
                    1,
                    "{coordinate}: the reverse-causal atom mass stays unidentified"
                );

                let result = prepared.estimate(&fixture.base, &ctx).unwrap();
                assert_eq!(result.logical_plan.estimator.as_deref(), Some(estimator));
                assert_eq!(result.support_status.map(CellStatus::as_str), Some("licensed"));
                let mixture = result.structural_response.as_ref().expect("graph mixture");
                assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9, "{coordinate}");
                assert!((mixture.identified_mass - 0.8).abs() < 1e-9, "{coordinate}");
                let values = atom_values(&result);
                assert_eq!(values.len(), fixture.identified_atoms, "{coordinate}");
                for value in &values {
                    assert!((value - 2.0).abs() < tolerance, "{coordinate}: atom effect {value}");
                }
                if fixture.scalar_mixable {
                    assert!(
                        (result.estimate.ate - 2.0).abs() < tolerance,
                        "{coordinate}: {}",
                        result.estimate.ate
                    );
                } else {
                    assert!(
                        result.estimate.ate.is_nan(),
                        "{coordinate}: distinct adjustment certificates withhold the scalar"
                    );
                }
                assert_eq!(result.refutations.is_empty(), suite == RefuteSuite::None);
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
                    "{coordinate}: the click reuses the retained atom identification"
                );

                // The one-shot facade executes the same retained operation.
                assert_eq!(one_shot.logical_plan.estimator.as_deref(), Some(estimator));
                assert_eq!(one_shot.estimate.ate.is_nan(), result.estimate.ate.is_nan());
                if !one_shot.estimate.ate.is_nan() {
                    assert!((one_shot.estimate.ate - result.estimate.ate).abs() < tolerance);
                }
                assert_eq!(atom_values(&one_shot).len(), values.len(), "{coordinate}");
                assert!(
                    one_shot.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
                    "{coordinate}: one-shot run must execute its retained prepared plan"
                );

                // Compatible data refreshes through the retained atoms; the
                // outcome shift leaves every conditional effect unchanged.
                let refreshed = prepared.refresh(fixture.shifted.clone(), &ctx).unwrap();
                let retained = retained_plan(&prepared, fixture.graph_class, bayesian);
                assert_eq!(retained.graph_keys, plan.graph_keys);
                assert_eq!(retained.weights, plan.weights);
                assert!(retained.conditional);
                let refreshed_values = atom_values(&refreshed);
                assert_eq!(refreshed_values.len(), values.len(), "{coordinate}");
                for value in &refreshed_values {
                    assert!((value - 2.0).abs() < tolerance, "{coordinate}: refreshed {value}");
                }
                assert_eq!(refreshed.estimate.ate.is_nan(), result.estimate.ate.is_nan());

                // A different semantic schema is refused by refresh.
                assert!(
                    prepared.refresh(reordered_data(), &ctx).is_err(),
                    "{coordinate}: reordered columns must be refused"
                );

                let artifact = prepared
                    .encode_contracted_result(
                        &refreshed,
                        &format!("checked-graph-posterior-conditional-{label}"),
                        &ctx,
                    )
                    .unwrap();
                let consumed = consume_analysis_result(&artifact).unwrap();
                assert!(
                    consumed
                        .acceptance
                        .unresolved
                        .iter()
                        .any(|reason| reason.as_ref() == dependency),
                    "{coordinate}: independent consumers must report {dependency}; got {:?}",
                    consumed.acceptance.unresolved
                );
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-graph-posterior-conditional-evidence".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
