//! Lifecycle evidence for sealed explicit/accepted ADMG scalar response routes.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, StructureSource, Study,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, SlotAvailability, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, adjacency_mask_from_admg, set_edge,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_io::consume_analysis_result;
use antecedent_prob::InferenceDiagnostics;

fn fixture(outcome_shift: f64) -> (TabularData, Admg) {
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
            let value =
                cell[*name].as_f64().unwrap() + if index == 2 { outcome_shift } else { 0.0 };
            values[index].extend(std::iter::repeat_n(value, count));
        }
    }
    let data = TabularData::from_f64_columns(
        columns.iter().zip(&values).map(|(name, values)| (*name, values.as_slice())),
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
    (data, graph)
}

fn query_with_value(value: Value) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), value)]),
    })
}

fn response_value(result: &antecedent::StudyResult) -> f64 {
    match &result.response.as_ref().expect("response").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected point identified scalar, got {other:?}"),
    }
}

#[test]
fn explicit_and_accepted_admg_scalar_response_is_sealed_for_both_inference_modes() {
    let (data, graph) = fixture(0.0);
    let expected = [(0.0, 0.314), (1.0, 0.596)];
    for accepted in [false, true] {
        for bayesian in [false, true] {
            let inference = if bayesian {
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1024))
            } else {
                InferenceMode::Frequentist
            };
            for (level, truth) in expected {
                // Preserve the query's typed value in the retained member.
                let intervention_value = Value::f64(level);
                let query = query_with_value(intervention_value.clone());
                let base = Study::tabular(data.clone())
                    .query(CausalQuery::Response(query.clone()))
                    .identifier(IdentifierId::GeneralId)
                    .estimator(EstimatorId::FunctionalEffect)
                    .inference(inference.clone())
                    .refute(RefuteSuite::None)
                    .bootstrap_replicates(0);
                let builder = if accepted {
                    base.graph(AcceptedGraph::from(graph.clone()))
                } else {
                    base.graph(graph.clone())
                };
                let context = ExecutionContext::for_tests(84_002);
                let study = builder.clone().build().unwrap();
                let mut prepared = study.prepare(&context).unwrap();
                drop(builder);
                drop(study);
                assert_eq!(prepared.query(), &CausalQuery::Response(query));
                assert_eq!(
                    prepared.structure_source(),
                    if accepted { StructureSource::Accepted } else { StructureSource::Explicit }
                );
                assert_eq!(prepared.support_status(), Some(CellStatus::Licensed));
                let contract = prepared.contract().unwrap();
                assert_eq!(contract.query_kind.as_ref(), "InterventionResponse");
                assert_eq!(contract.identifier.as_deref(), Some("general.id"));
                assert_eq!(contract.estimator.as_deref(), Some("functional.effect"));
                assert_eq!(
                    contract.inference.as_ref(),
                    if bayesian { "bayesian" } else { "frequentist" }
                );
                let coordinate = format!(
                    "InterventionResponse:Admg:{}:{}:none",
                    if accepted { "accepted" } else { "explicit" },
                    if bayesian { "Bayesian" } else { "Frequentist" }
                );
                match &contract.reasoning.support {
                    SlotAvailability::Available(slot) => {
                        assert_eq!(slot.matrix_coordinate.as_deref(), Some(coordinate.as_str()));
                    }
                    other => panic!("{coordinate}: support unavailable: {other:?}"),
                }
                let members = prepared
                    .checked_functional_effect_response_members()
                    .expect("sealed response operation retains its scalar member");
                assert_eq!(members.len(), 1);
                assert_eq!(members[0].grid_value(), level);
                assert_eq!(
                    members[0].query().functional,
                    ResponseFunctional::InterventionResponse {
                        outcome: VariableId::from_raw(2),
                        interventions: Arc::from([Intervention::set(
                            VariableId::from_raw(0),
                            intervention_value,
                        )]),
                    }
                );
                assert_eq!(
                    members[0].identification().status,
                    antecedent_core::IdentificationStatus::NonparametricallyIdentified
                );
                assert_eq!(
                    members[0].program().mapping().source,
                    members[0].program().mapping().executable
                );

                let result = prepared.estimate(&data, &context).unwrap();
                assert!(
                    (response_value(&result) - truth).abs() < if bayesian { 0.06 } else { 1e-12 }
                );
                if bayesian {
                    assert_eq!(result.posterior.as_ref().unwrap().draws.n_draws, 1024);
                    assert_eq!(result.posterior.as_ref().unwrap().draws.schema.n_quantities(), 1);
                } else {
                    assert!(result.posterior.is_none());
                }
                let (refreshed_data, _) = fixture(0.4);
                let refreshed = prepared.refresh(refreshed_data, &context).unwrap();
                assert!(
                    (response_value(&refreshed) - (truth + 0.4)).abs()
                        < if bayesian { 0.06 } else { 1e-12 }
                );
                assert!(prepared.checked_functional_effect_response_members().is_some());
                let artifact =
                    prepared.encode_contracted_result(&refreshed, &coordinate, &context).unwrap();
                let consumed = consume_analysis_result(&artifact).unwrap();
                if bayesian {
                    assert!(
                        consumed.acceptance.unresolved.iter().any(|reason| {
                            reason.as_ref()
                                == "dependencies.functional_effect_response_posterior_draws"
                        }),
                        "{coordinate}: {:?}",
                        consumed.acceptance.unresolved
                    );
                    assert!(!consumed.acceptance.accepts_as_verified_program());
                } else {
                    assert!(
                        consumed.acceptance.unresolved.iter().any(|reason| {
                            reason.as_ref() == "program.functional_effect_response_grid"
                        }),
                        "{coordinate}: {:?}",
                        consumed.acceptance.unresolved
                    );
                    assert!(!consumed.acceptance.accepts_as_verified_program());
                }
            }
        }
    }
}

#[test]
fn explicit_and_accepted_admg_scalar_response_runs_plugin_level_refuters() {
    let (data, graph) = fixture(0.0);
    let truth = 0.596;
    let query = query_with_value(Value::f64(1.0));
    for accepted in [false, true] {
        for bayesian in [false, true] {
            let inference = if bayesian {
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
            } else {
                InferenceMode::Frequentist
            };
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let suite_name = match suite {
                    RefuteSuite::Cheap => "cheap",
                    RefuteSuite::Full => "full",
                    RefuteSuite::None | RefuteSuite::PlaceboAndRcc => unreachable!(),
                };
                let base = Study::tabular(data.clone())
                    .query(CausalQuery::Response(query.clone()))
                    .identifier(IdentifierId::GeneralId)
                    .estimator(EstimatorId::FunctionalEffect)
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(0);
                let builder = if accepted {
                    base.graph(AcceptedGraph::from(graph.clone()))
                } else {
                    base.graph(graph.clone())
                };
                let context = ExecutionContext::for_tests(84_003);
                let study = builder.clone().build().unwrap();
                let mut prepared = study.prepare(&context).unwrap();
                drop(builder);
                drop(study);
                let coordinate = format!(
                    "InterventionResponse:Admg:{}:{}:{suite_name}",
                    if accepted { "accepted" } else { "explicit" },
                    if bayesian { "Bayesian" } else { "Frequentist" }
                );
                let contract = prepared.contract().unwrap();
                match &contract.reasoning.support {
                    SlotAvailability::Available(slot) => {
                        assert_eq!(slot.matrix_coordinate.as_deref(), Some(coordinate.as_str()));
                    }
                    other => panic!("{coordinate}: support unavailable: {other:?}"),
                }
                let members = prepared
                    .checked_functional_effect_response_members()
                    .expect("sealed response operation retains its scalar member");
                assert_eq!(members.len(), 1);
                assert_eq!(members[0].grid_value(), 1.0);
                assert_eq!(
                    members[0].program().mapping().source,
                    members[0].program().mapping().executable
                );
                let result = prepared.estimate(&data, &context).unwrap();
                assert!(
                    (response_value(&result) - truth).abs() < if bayesian { 0.08 } else { 1e-12 },
                    "{coordinate}: {}",
                    response_value(&result)
                );
                assert_plugin_level_validation(&result, suite, &coordinate);
                let refreshed = prepared.refresh(data.clone(), &context).unwrap();
                assert_plugin_level_validation(&refreshed, suite, &coordinate);
            }
        }
    }
}

fn graph_posterior(graph: &Admg, include_unidentified_atom: bool) -> GraphPosterior {
    let n = graph.node_count();
    let identified = adjacency_mask_from_admg(graph).unwrap();
    let mut cyclic = set_edge(0, n, 0, 1, true);
    cyclic = set_edge(cyclic, n, 1, 2, true);
    cyclic = set_edge(cyclic, n, 2, 0, true);
    GraphPosterior::new(
        n,
        vec![0.8, 0.2],
        if include_unidentified_atom {
            vec![identified, cyclic]
        } else {
            vec![identified, identified]
        },
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("admg_graph_posterior_response_checked"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg)
}

#[test]
fn graph_posterior_admg_intervention_response_is_sealed_across_refuters() {
    let (data, graph) = fixture(0.0);
    let query = query_with_value(Value::f64(1.0));
    for bayesian in [false, true] {
        let inference = if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(48))
        } else {
            InferenceMode::Frequentist
        };
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            // Preserve a failed atom in the no-refuter case. Refuters run only
            // for a fully identified same-estimand mixture by contract.
            let posterior = graph_posterior(&graph, suite == RefuteSuite::None);
            let context = ExecutionContext::for_tests(84_005);
            let builder = Study::tabular(data.clone())
                .graph_posterior(posterior.clone())
                .query(CausalQuery::Response(query.clone()))
                .inference(inference.clone())
                .refute(suite)
                .bootstrap_replicates(0);
            let study = builder.clone().build().unwrap();
            let mut prepared = study.prepare(&context).unwrap();
            drop(builder);
            drop(study);
            assert!(prepared.has_checked_admg_graph_posterior_response_operation());
            let plan = prepared.checked_admg_graph_posterior_response_info().unwrap();
            assert_eq!(plan.query, query);
            assert_eq!(plan.weights.as_ref(), &[0.8, 0.2]);
            assert_eq!(plan.identified.len(), 2);
            assert_eq!(plan.identified[0], antecedent_prob::GraphIdentFlag::Identified);
            assert_eq!(
                plan.identified[1],
                if suite == RefuteSuite::None {
                    antecedent_prob::GraphIdentFlag::Unidentified
                } else {
                    antecedent_prob::GraphIdentFlag::Identified
                }
            );
            assert_eq!(plan.validation, suite);
            let result = prepared.estimate(&data, &context).unwrap();
            let mixture = result.structural_response.as_ref().expect("posterior response mass");
            let expected_unidentified = if suite == RefuteSuite::None { 0.2 } else { 0.0 };
            assert!((mixture.identified_mass - (1.0 - expected_unidentified)).abs() < 1e-12);
            assert!((mixture.unidentified_mass - expected_unidentified).abs() < 1e-12);
            match suite {
                RefuteSuite::None => assert!(result.refutations.is_empty()),
                RefuteSuite::Cheap | RefuteSuite::Full => {
                    assert_plugin_level_validation(
                        &result,
                        suite,
                        &format!(
                            "InterventionResponse:Admg:graph_posterior:{}:{suite:?}",
                            if bayesian { "Bayesian" } else { "Frequentist" }
                        ),
                    );
                }
                RefuteSuite::PlaceboAndRcc => unreachable!(),
            }
            let refreshed = prepared.refresh(data.clone(), &context).unwrap();
            let refreshed_mass = refreshed.structural_response.as_ref().unwrap();
            assert!((refreshed_mass.identified_mass - (1.0 - expected_unidentified)).abs() < 1e-12);
            assert!((refreshed_mass.unidentified_mass - expected_unidentified).abs() < 1e-12);
            let artifact = prepared.encode_contracted_result(&refreshed, "admg-gp-response", &context).unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(
                consumed.acceptance.accepts_as_verified_program()
                    || !consumed.acceptance.unresolved.is_empty(),
                "independent consumer must verify the route or explain its dependency refusal"
            );
            assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_admg_graph_posterior_response_operation"
            }));
        }
    }
}

fn assert_plugin_level_validation(
    result: &antecedent::StudyResult,
    suite: RefuteSuite,
    coordinate: &str,
) {
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_ref() == "refute.evalue.not_a_contrast"),
        "{coordinate}: plugin level must refuse a contrast-shaped E-value"
    );
    assert!(
        result.refutations.iter().any(|report| report.refuter.as_ref() == "overlap.assessment"),
        "{coordinate}: missing overlap refuter: {:?}",
        result.refutations.iter().map(|report| report.refuter.as_ref()).collect::<Vec<_>>()
    );
    assert!(
        result.refutations.iter().all(|report| report.refuter.as_ref() != "sensitivity.evalue"),
        "{coordinate}: E-value is not licensed for an intervention level"
    );
    match suite {
        RefuteSuite::Cheap => {
            assert!(
                result.refutations.iter().all(|report| report.refuter.as_ref().contains("overlap")),
                "{coordinate}: cheap is overlap only: {:?}",
                result.refutations.iter().map(|report| report.refuter.as_ref()).collect::<Vec<_>>()
            );
        }
        RefuteSuite::Full => {
            assert!(
                result.refutations.iter().any(|report| report.refuter.as_ref() == "overlap.rule"),
                "{coordinate}: full must add the overlap rule: {:?}",
                result.refutations.iter().map(|report| report.refuter.as_ref()).collect::<Vec<_>>()
            );
            for validator in ["bootstrap", "data_subset", "graph"] {
                assert!(
                    result.diagnostics.iter().any(|diagnostic| {
                        diagnostic.code.as_ref() == "refute.validator.not_applicable"
                            && diagnostic.fields.iter().any(|(key, value)| {
                                key.as_ref() == "validator" && value.as_ref() == validator
                            })
                    }),
                    "{coordinate}: {validator} is not licensed on a functional.effect level"
                );
            }
        }
        RefuteSuite::None | RefuteSuite::PlaceboAndRcc => unreachable!(),
    }
}
