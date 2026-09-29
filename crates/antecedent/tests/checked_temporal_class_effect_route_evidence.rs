//! Builder independent evidence for frequentist temporal class effect envelopes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalPag};
use common::fixtures::{
    self, B1, chain_pag, chain_pag_series, confounded_cpdag, confounded_series,
};

enum SourceGraph {
    Cpdag(TemporalCpdag),
    Pag(TemporalPag),
}

fn graphs() -> [SourceGraph; 2] {
    [SourceGraph::Cpdag(confounded_cpdag()), SourceGraph::Pag(chain_pag())]
}

fn series(graph: &SourceGraph) -> TimeSeriesData {
    match graph {
        SourceGraph::Cpdag(_) => confounded_series(640, 0.0, 0.0, 977),
        SourceGraph::Pag(_) => chain_pag_series(640, 0.0, 977),
    }
}

fn accepted(graph: &SourceGraph) -> AcceptedGraph {
    match graph {
        SourceGraph::Cpdag(graph) => AcceptedGraph::from(graph.clone()),
        SourceGraph::Pag(graph) => AcceptedGraph::from(graph.clone()),
    }
}

fn explicit(graph: &SourceGraph) -> AcceptedGraph {
    match graph {
        SourceGraph::Cpdag(graph) => graph.clone().into(),
        SourceGraph::Pag(graph) => graph.clone().into(),
    }
}

fn graph_class(graph: &SourceGraph) -> &'static str {
    match graph {
        SourceGraph::Cpdag(_) => "TemporalCpdag",
        SourceGraph::Pag(_) => "TemporalPag",
    }
}

fn query(multi_step: bool) -> TemporalEffectQuery {
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(1);
    if multi_step {
        TemporalEffectQuery::sustained(treatment, outcome, -2, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1))
            .with_horizon_steps(1)
    } else {
        TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
    }
}

fn truth(graph: &SourceGraph, multi_step: bool) -> f64 {
    match graph {
        SourceGraph::Cpdag(_) if multi_step => B1,
        SourceGraph::Cpdag(_) => {
            let [_, adjusted] = fixtures::cpdag_completion_truths(0.0);
            adjusted
        }
        // The generating directed completion of the seven atom chain PAG has
        // treatment effect B1 for Pulse and this terminal outcome horizon.
        SourceGraph::Pag(_) => B1,
    }
}

#[test]
fn all_frequentist_temporal_class_effect_coordinates_execute_from_retained_source_proofs() {
    let ctx = ExecutionContext::for_tests(977);
    let suites = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];
    for graph in graphs() {
        let data = series(&graph);
        for multi_step in [false, true] {
            let query = query(multi_step);
            for source in ["explicit", "accepted"] {
                for suite in suites {
                    let structure =
                        if source == "accepted" { accepted(&graph) } else { explicit(&graph) };
                    let builder = Study::series(data.clone())
                        .graph(structure)
                        .query(CausalQuery::TemporalEffect(query.clone()))
                        .inference(InferenceMode::Frequentist)
                        .refute(suite)
                        .bootstrap_replicates(16)
                        .build()
                        .unwrap();
                    let mut prepared = builder.prepare(&ctx).unwrap();
                    drop(builder);
                    let plan = prepared
                        .checked_temporal_class_effect_info()
                        .expect("class effect must retain its complete execution plan");
                    assert_eq!(plan.query, query);
                    assert_eq!(plan.graph_class.as_str(), graph_class(&graph));
                    assert_eq!(plan.identifier.as_str(), "generalized.adjustment");
                    assert_eq!(
                        plan.estimator.as_str(),
                        if multi_step {
                            "temporal.sequential.gcomp"
                        } else {
                            "temporal.linear.adjustment"
                        }
                    );
                    assert_eq!(plan.validation, suite);
                    assert_eq!(plan.bootstrap_replicates, 16);
                    assert!(plan.completion_count > 0);
                    assert!(plan.identified_mass > 0.0);

                    let result = prepared.estimate_series(&data, &ctx).unwrap();
                    assert!(result.estimate.ate.is_finite());
                    let true_effect = truth(&graph, multi_step);
                    let mixture = result
                        .structural_response
                        .as_ref()
                        .expect("completion-level effects remain in the result");
                    let envelope = mixture
                        .identified_set
                        .as_ref()
                        .expect("identified completion values define an envelope");
                    assert!(
                        envelope.lower[0] - 0.08 <= true_effect
                            && true_effect <= envelope.upper[0] + 0.08,
                        "class={}, source={source}, multi_step={multi_step}, suite={suite:?}, truth {true_effect} outside [{}, {}]",
                        graph_class(&graph),
                        envelope.lower[0],
                        envelope.upper[0]
                    );
                    if matches!(graph, SourceGraph::Pag(_)) {
                        assert!(
                            mixture.unidentified_mass + mixture.unevaluable_mass > 0.0,
                            "TemporalPag must preserve the unresolved or unevaluable completion mass"
                        );
                    }
                    let standard_error = result
                        .estimate
                        .se_bootstrap
                        .expect("shared circular-block procedure must execute");
                    assert!(standard_error.is_finite() && standard_error >= 0.0);

                    let refreshed_data = series(&graph);
                    let refreshed = prepared.refresh_series(refreshed_data, &ctx).unwrap();
                    let refreshed_truth = true_effect;
                    let refreshed_envelope = refreshed
                        .structural_response
                        .as_ref()
                        .and_then(|mixture| mixture.identified_set.as_ref())
                        .expect("refreshed completion values define an envelope");
                    assert!(
                        refreshed_envelope.lower[0] - 0.08 <= refreshed_truth
                            && refreshed_truth <= refreshed_envelope.upper[0] + 0.08
                    );
                    assert_eq!(prepared.checked_temporal_class_effect_info().unwrap().query, query);
                    let artifact = prepared
                        .encode_contracted_result(&refreshed, "checked-temporal-class-effect", &ctx)
                        .unwrap();
                    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                    assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
                        dependency.as_ref()
                            == "dependencies.checked_temporal_class_effect_operation"
                    }));
                    assert!(!consumed.acceptance.accepts_as_verified_program());
                }
            }
        }
    }
}
