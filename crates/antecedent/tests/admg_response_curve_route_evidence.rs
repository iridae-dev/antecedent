//! Coordinate-specific closure evidence for ADMG functional-effect response curves.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, StructureSource, Study,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    ResponseValue, SlotAvailability, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_io::consume_analysis_result;

fn fixture() -> (TabularData, Admg, ResponseQuery) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(&values).map(|(name, values)| (*name, values.as_slice())).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let node = |name: &str| {
        DenseNodeId::from_raw(
            u32::try_from(columns.iter().position(|col| *col == name).unwrap()).unwrap(),
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
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    (data, graph, query)
}

fn means(result: &antecedent::StudyResult) -> &[f64] {
    match &result.response.as_ref().expect("response result").estimate {
        antecedent_core::ResponseIdentification::PointIdentified(ResponseValue::Surface {
            mean,
            ..
        }) => mean,
        other => panic!("expected a point-identified response surface, got {other:?}"),
    }
}

fn verify_coordinate(coordinate: &str, accepted: bool, bayesian: bool) {
    let (data, graph, query) = fixture();
    let ctx = ExecutionContext::for_tests(42_771);
    let inference = if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1024))
    } else {
        InferenceMode::Frequentist
    };
    let base = Study::tabular(data.clone())
        .query(CausalQuery::Response(query.clone()))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let builder = if accepted {
        base.graph(AcceptedGraph::from(graph.clone()))
    } else {
        base.graph(graph.clone())
    };
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let mut prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &CausalQuery::Response(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} source"
    );
    assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{coordinate}");
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "ResponseCurve", "{coordinate}");
    assert_eq!(contract.identifier.as_deref(), Some("general.id"), "{coordinate}");
    assert_eq!(contract.estimator.as_deref(), Some("functional.effect"), "{coordinate}");
    assert_eq!(contract.inference.as_ref(), if bayesian { "bayesian" } else { "frequentist" });
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => {
            assert_eq!(slot.matrix_coordinate.as_deref(), Some(coordinate))
        }
        other => panic!("{coordinate} unavailable support: {other:?}"),
    }
    assert_eq!(prepared.plan().logical.record.identifier.as_deref(), Some("general.id"));
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("functional.effect"));
    assert_eq!(prepared.plan().logical.record.validation_suite, None);

    let members = prepared
        .checked_functional_effect_response_members()
        .unwrap_or_else(|| panic!("{coordinate} omitted retained checked grid members"));
    assert_eq!(members.len(), 2, "{coordinate} member count");
    for (member, expected_level) in members.iter().zip([0.0_f64, 1.0_f64]) {
        assert_eq!(member.grid_value().to_bits(), expected_level.to_bits(), "{coordinate} order");
        assert_eq!(
            member.identification().status,
            antecedent_core::IdentificationStatus::NonparametricallyIdentified,
            "{coordinate} identification"
        );
        assert_eq!(member.program().mapping().source, member.program().mapping().executable);
        assert_eq!(
            member.query().functional,
            ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(2),
                interventions: Arc::from([antecedent_core::Intervention::set(
                    VariableId::from_raw(0),
                    antecedent_core::Value::f64(expected_level),
                )]),
            },
            "{coordinate} typed member query"
        );
        let _ = member.estimand();
    }

    let result =
        prepared.estimate(&data, &ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let expected = [0.314, 0.596];
    for (actual, expected) in means(&result).iter().zip(expected) {
        assert!(
            (actual - expected).abs() < if bayesian { 0.06 } else { 1e-12 },
            "{coordinate}: {actual} vs {expected}"
        );
    }
    if bayesian {
        let posterior = result.posterior.as_ref().expect("joint response-curve posterior");
        assert_eq!(posterior.draws.n_draws, 1024, "{coordinate} shared draw count");
        assert_eq!(
            posterior.draws.schema.n_quantities(),
            2,
            "{coordinate} ordered grid draw columns"
        );
        assert_eq!(posterior.draws.values.len(), 2048, "{coordinate} joint response draws");
        let repeated = prepared.estimate(&data, &ctx).unwrap();
        assert_eq!(
            repeated.posterior.as_ref().unwrap().draws.values,
            posterior.draws.values,
            "{coordinate} repeat must preserve shared draw identities"
        );
        for (index, member) in members.iter().enumerate() {
            let scalar_base = Study::tabular(data.clone())
                .query(CausalQuery::Response(member.query().clone()))
                .identifier(IdentifierId::GeneralId)
                .estimator(EstimatorId::FunctionalEffect)
                .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1024)))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0);
            let scalar_builder = if accepted {
                scalar_base.graph(AcceptedGraph::from(graph.clone()))
            } else {
                scalar_base.graph(graph.clone())
            };
            let scalar_study = scalar_builder.build().unwrap();
            let scalar_prepared = scalar_study.prepare(&ctx).unwrap();
            let scalar_result = scalar_prepared.estimate(&data, &ctx).unwrap();
            let scalar_draws = scalar_result
                .posterior
                .as_ref()
                .expect("scalar member posterior")
                .draws
                .column(0)
                .unwrap();
            let joint_draws = posterior.draws.column(index).unwrap();
            assert_eq!(
                joint_draws,
                scalar_draws,
                "{coordinate} member {} must use the same ordered row-weight draws as its scalar query",
                member.grid_value()
            );
        }
    } else {
        assert!(result.posterior.is_none(), "{coordinate} unexpected posterior");
    }
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert_eq!(means(&refreshed), means(&result), "{coordinate} refresh");
    assert!(
        prepared.checked_functional_effect_response_members().is_some(),
        "{coordinate} refresh discarded members"
    );

    let artifact = prepared
        .encode_contracted_result(&refreshed, &format!("response-curve-{coordinate}"), &ctx)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    if bayesian {
        assert_eq!(
            consumed.acceptance.unresolved.len(),
            1,
            "{coordinate}: {:?}",
            consumed.acceptance.unresolved
        );
        assert!(
            consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.functional_effect_response_posterior_draws"
            }),
            "{coordinate}: {:?}",
            consumed.acceptance.unresolved
        );
        assert!(!consumed.acceptance.accepts_as_verified_program());
    } else {
        assert!(
            consumed.acceptance.accepts_as_verified_program(),
            "{coordinate}: {:?}",
            consumed.acceptance.unresolved
        );
    }
    let wire_means =
        match &consumed.body.response.as_ref().expect("portable response payload").estimate {
            antecedent_io::ResponseIdentificationWire::PointIdentified(
                antecedent_io::ResponseValueWire::Surface { grid, mean, .. },
            ) => {
                assert_eq!(grid, &[0.0, 1.0], "{coordinate} artifact grid");
                mean
            }
            other => panic!("{coordinate} unexpected artifact response: {other:?}"),
        };
    for (actual, expected) in wire_means.iter().zip([0.314, 0.596]) {
        assert!(
            (actual - expected).abs() < if bayesian { 0.06 } else { 1e-12 },
            "{coordinate} artifact {actual} vs {expected}"
        );
    }
}

#[test]
fn admg_response_curve_accepted_frequentist_none() {
    verify_coordinate("ResponseCurve:Admg:accepted:Frequentist:none", true, false);
}

#[test]
fn admg_response_curve_accepted_bayesian_none() {
    verify_coordinate("ResponseCurve:Admg:accepted:Bayesian:none", true, true);
}

#[test]
fn admg_response_curve_explicit_frequentist_none() {
    verify_coordinate("ResponseCurve:Admg:explicit:Frequentist:none", false, false);
}

#[test]
fn admg_response_curve_explicit_bayesian_none() {
    verify_coordinate("ResponseCurve:Admg:explicit:Bayesian:none", false, true);
}
