//! Public lifecycle evidence for the two licensed explicit-DAG interference routes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    BayesianConfig, EstimatorId, IdentifierId, InferenceMode, InterferenceSpec, RefuteSuite,
    StructureSource, Study,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    InterferenceFunctional, InterferenceQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TableView, TabularData};
use antecedent_graph::Dag;
use antecedent_io::consume_analysis_result;

fn query() -> InterferenceQuery {
    InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 1.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
    )
}

fn design_fixture() -> (TabularData, Vec<bool>, Vec<NetworkEdge>, serde_json::Value) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/randomized_interference/expected.json"
    ))
    .unwrap();
    let outcomes = fixture["outcomes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let assignment = fixture["assignment"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_bool().unwrap())
        .collect::<Vec<_>>();
    let edges = fixture["directed_edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| NetworkEdge {
            from: edge[0].as_u64().unwrap() as u32,
            to: edge[1].as_u64().unwrap() as u32,
            weight: edge[2].as_f64().unwrap(),
        })
        .collect();
    (
        TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap(),
        assignment,
        edges,
        fixture,
    )
}

fn bayesian_fixture() -> (TabularData, Vec<bool>, Vec<NetworkEdge>, serde_json::Value) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_interference/expected.json"
    ))
    .unwrap();
    let outcomes = fixture["outcomes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let assignment = fixture["assignment"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_bool().unwrap())
        .collect::<Vec<_>>();
    let edges = fixture["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| NetworkEdge {
            from: edge[0].as_u64().unwrap() as u32,
            to: edge[1].as_u64().unwrap() as u32,
            weight: 1.0,
        })
        .collect();
    (
        TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap(),
        assignment,
        edges,
        fixture,
    )
}

fn build(
    data: TabularData,
    assignment: Vec<bool>,
    edges: Vec<NetworkEdge>,
    bayesian: bool,
) -> Study {
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let study = Study::tabular(data)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query()))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None);
    let study = if bayesian {
        study.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(20_000)))
    } else {
        study
    };
    study.build().unwrap()
}

fn shifted_outcomes(data: &TabularData, delta: f64) -> TabularData {
    let outcome = data.float64_values(VariableId::from_raw(0)).unwrap();
    let shifted = outcome.iter().map(|value| value + delta).collect::<Vec<_>>();
    TabularData::from_f64_columns([("y", shifted.as_slice())]).unwrap()
}

fn wrong_unit_count() -> TabularData {
    TabularData::from_f64_columns([("y", &[1.0, 2.0, 3.0][..])]).unwrap()
}

#[test]
fn bernoulli_neighbor_count_design_route_is_sealed_refreshable_and_artifact_scoped() {
    let (data, assignment, edges, fixture) = design_fixture();
    let builder = build(data.clone(), assignment, edges, false);
    let context = ExecutionContext::for_tests(6192);
    let mut prepared = builder.prepare(&context).unwrap();
    drop(builder);

    let plan =
        prepared.checked_interference_info().expect("prepared design-based interference operation");
    assert_eq!(prepared.structure_source(), StructureSource::Explicit);
    assert_eq!(plan.query, query());
    assert_eq!(plan.identifier, IdentifierId::InterferenceDesign);
    assert_eq!(plan.estimator, EstimatorId::InterferenceHtHajek);
    assert_eq!(plan.unit_count, 2);
    assert_eq!(plan.network_edge_count, 2);

    let result = prepared.estimate(&data, &context).unwrap();
    assert_eq!(result.support_status, Some(antecedent::CellStatus::Licensed));
    let estimated = result.interference.as_ref().expect("design-based contrast");
    let truth = &fixture["expected"];
    let tolerance = fixture["tolerance"]["atol"].as_f64().unwrap();
    assert!(
        (estimated.contrast.horvitz_thompson
            - truth["horvitz_thompson_contrast"].as_f64().unwrap())
        .abs()
            <= tolerance
    );
    assert!(
        (estimated.contrast.hajek - truth["hajek_contrast"].as_f64().unwrap()).abs() <= tolerance
    );
    assert!(
        (estimated.contrast.conservative_variance
            - truth["conservative_variance"].as_f64().unwrap())
        .abs()
            <= tolerance
    );

    let artifact = prepared
        .encode_contracted_result(&result, "checked-interference-design", &context)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
        dependency.as_ref() == "dependencies.checked_interference_operation"
    }));
    assert!(!consumed.acceptance.accepts_as_verified_program());

    let shifted = shifted_outcomes(&data, 0.5);
    let refreshed = prepared.refresh(shifted, &context).unwrap();
    assert!(
        (refreshed.interference.as_ref().unwrap().contrast.horvitz_thompson
            - estimated.contrast.horvitz_thompson)
            .abs()
            <= tolerance
    );
    assert_eq!(prepared.checked_interference_info().unwrap().network_edge_count, 2);
    assert!(prepared.refresh(wrong_unit_count(), &context).is_err());
}

#[test]
fn conjugate_gaussian_interference_route_is_sealed_refreshable_and_artifact_scoped() {
    let (data, assignment, edges, fixture) = bayesian_fixture();
    let builder = build(data.clone(), assignment, edges, true);
    let context = ExecutionContext::for_tests(6193);
    let mut prepared = builder.prepare(&context).unwrap();
    drop(builder);

    let plan =
        prepared.checked_interference_info().expect("prepared Bayesian interference operation");
    assert_eq!(prepared.structure_source(), StructureSource::Explicit);
    assert_eq!(plan.query, query());
    assert_eq!(plan.identifier, IdentifierId::InterferenceDesign);
    assert_eq!(plan.estimator, EstimatorId::InterferenceBayesianGaussian);
    assert_eq!(plan.unit_count, 4);
    assert_eq!(plan.network_edge_count, 4);

    let result = prepared.estimate(&data, &context).unwrap();
    assert_eq!(result.support_status, Some(antecedent::CellStatus::Licensed));
    assert_eq!(
        result.identification.status,
        antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions
    );
    let truth = fixture["expected_contrast"].as_f64().unwrap();
    let tolerance = fixture["tolerance"].as_f64().unwrap();
    assert!((result.estimate.ate - truth).abs() < tolerance);
    assert_eq!(result.posterior.as_ref().unwrap().draws.n_draws, 20_000);
    assert!(result.posterior.as_ref().unwrap().assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, antecedent_core::Assumption::ParametricRestriction(item)
            if item.id.as_ref() == "interference.fixed_network_gaussian_potential_outcomes")
    }));

    let artifact = prepared
        .encode_contracted_result(&result, "checked-interference-bayesian", &context)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
        dependency.as_ref() == "dependencies.checked_interference_operation"
    }));
    assert!(!consumed.acceptance.accepts_as_verified_program());

    let refreshed = prepared.refresh(shifted_outcomes(&data, 0.5), &context).unwrap();
    assert!((refreshed.estimate.ate - result.estimate.ate).abs() < tolerance);
    assert_eq!(prepared.checked_interference_info().unwrap().unit_count, 4);
    assert!(prepared.refresh(wrong_unit_count(), &context).is_err());
}
