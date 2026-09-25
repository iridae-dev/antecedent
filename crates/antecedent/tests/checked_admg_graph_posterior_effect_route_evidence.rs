//! Builder-independent evidence for ADMG graph-posterior average effects.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{
    adjacency_mask_from_admg, set_edge, GraphPosterior, GraphPosteriorAtomKind,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_io::consume_analysis_result;
use antecedent_prob::InferenceDiagnostics;

fn fixture() -> (TabularData, Admg, AverageEffectQuery, f64) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|value| value.as_str().unwrap()).collect();
    let mut values = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let data = TabularData::from_f64_columns(
        columns.iter().zip(&values).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap();
    let node = |name: &str| {
        DenseNodeId::from_raw(
            u32::try_from(columns.iter().position(|column| *column == name).unwrap()).unwrap(),
        )
    };
    let mut graph = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        graph
            .insert_directed(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        graph
            .insert_bidirected(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    let query = AverageEffectQuery::with_levels(
        VariableId::from_raw(node(pin["query"]["treatment"].as_str().unwrap()).raw()),
        VariableId::from_raw(node(pin["query"]["outcome"].as_str().unwrap()).raw()),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    );
    (data, graph, query, pin["frequentist"]["expected_ate"].as_f64().unwrap())
}

fn posterior(graph: &Admg) -> GraphPosterior {
    let n = graph.node_count();
    let identified = adjacency_mask_from_admg(graph).unwrap();
    let mut cyclic = set_edge(0, n, 0, 1, true);
    cyclic = set_edge(cyclic, n, 1, 2, true);
    cyclic = set_edge(cyclic, n, 2, 0, true);
    GraphPosterior::new(
        n,
        vec![0.8, 0.2],
        vec![identified, cyclic],
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("admg_graph_posterior_checked"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg)
}

#[test]
fn admg_graph_posterior_effect_is_sealed_for_both_inference_modes() {
    let (data, graph, query, truth) = fixture();
    let graphs = posterior(&graph);
    for bayesian in [false, true] {
        let inference = if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32))
        } else {
            InferenceMode::Frequentist
        };
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let builder = Study::tabular(data.clone())
                .graph_posterior(graphs.clone())
                .query(query.clone())
                .inference(inference.clone())
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let context = ExecutionContext::for_tests(84_004);
            let mut prepared = builder.prepare(&context).unwrap();
            drop(builder);
            let plan = prepared
                .checked_graph_posterior_effect_info()
                .expect("prepared ADMG graph-posterior plan");
            assert_eq!(plan.query, query);
            assert_eq!(plan.estimator.as_str(), "functional.effect");
            assert_eq!(plan.validation, suite);
            assert_eq!(plan.weights.as_ref(), &[0.8, 0.2]);
            assert_eq!(plan.graph_keys.as_ref(), graphs.graph_keys.as_ref());
            let result = prepared.estimate(&data, &context).unwrap();
            let tolerance = if bayesian { 0.05 } else { 1e-9 };
            assert!(
                (result.estimate.ate - truth).abs() < tolerance,
                "bayesian={bayesian} suite={suite:?}: {} vs {truth}",
                result.estimate.ate
            );
            let mixture = result.structural_response.as_ref().expect("mixture");
            assert!((mixture.identified_mass - 0.8).abs() < 1e-12);
            assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12);
            assert!(result.diagnostics.iter().any(|diagnostic| {
                diagnostic.code.as_ref() == "estimate.graph_posterior.admg_functional"
            }));
            assert!(result
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.code.as_ref() == "exec.identify.cached" }));
            match suite {
                RefuteSuite::None => assert!(result.refutations.is_empty()),
                RefuteSuite::Cheap | RefuteSuite::Full => {
                    assert!(
                        !result.refutations.is_empty(),
                        "{suite:?} must refute identified atoms"
                    );
                }
                RefuteSuite::PlaceboAndRcc => unreachable!(),
            }
            let refreshed = prepared.refresh(data.clone(), &context).unwrap();
            assert!((refreshed.estimate.ate - result.estimate.ate).abs() < 1e-9);
            let retained = prepared.checked_graph_posterior_effect_info().unwrap();
            assert_eq!(retained.weights, plan.weights);
            assert_eq!(retained.graph_keys, plan.graph_keys);
            let artifact = prepared
                .encode_contracted_result(&refreshed, "admg-graph-posterior-effect", &context)
                .unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(
                consumed.acceptance.accepts_as_verified_program()
                    || !consumed.acceptance.unresolved.is_empty(),
                "independent consumer must verify the effect or explain its dependency refusal"
            );
            assert!(
                consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_graph_posterior_effect_operation"
                }),
                "graph-posterior aggregation is an explicit replay dependency: {:?}",
                consumed.acceptance.unresolved
            );
        }
    }
}
