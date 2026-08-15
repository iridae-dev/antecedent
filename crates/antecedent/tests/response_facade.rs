//! End-to-end continuous-response facade coverage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SupportStatus, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn response_curve_runs_through_public_study_facade() {
    let n = 240;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> =
        (0..n).map(|i| z[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i] + (i as f64 / 13.0).sin() * 0.05)
        .collect();
    let data = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("confounder", z.as_slice()),
    ])
    .unwrap();

    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    });

    let study = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(50)).unwrap();
    let response = result.response.as_ref().expect("response payload");

    assert_eq!(result.logical_plan.plan_id.as_ref(), "static_response");
    assert_eq!(result.logical_plan.identifier.as_deref(), Some("response.backdoor"));
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("response.kennedy_dr"));
    assert_eq!(response.support.status, SupportStatus::Supported);
    assert_eq!(response.provenance_id.as_ref(), "estimate.response.kennedy_dr");
    assert!(result.provenance.get("identify.response").is_some());
    assert!(result.provenance.get("estimate.response.kennedy_dr").is_some());
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified response surface");
    };
    assert_eq!(mean.len(), 3);
    assert!(mean.iter().all(|value| value.is_finite()));
}

#[test]
fn simultaneous_curve_band_runs_through_public_study_facade() {
    let n = 240i32;
    let treatment: Vec<f64> = (0..n).map(|i| (f64::from(i) / 17.0).sin()).collect();
    // A noiseless outcome makes every influence contribution exactly zero, and the
    // multiplier band correctly refuses a degenerate standard error. The band is the
    // subject of this test, so the fixture carries residual variation.
    let outcome: Vec<f64> = treatment
        .iter()
        .enumerate()
        .map(|(i, value)| 1.0 + 2.0 * value + 0.15 * (i as f64 / 7.0).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    });
    let options = ContinuousResponseOptions {
        bandwidth: Some(0.3),
        simultaneous_replicates: Some(100),
        multiplier_seed: 17,
        ..ContinuousResponseOptions::default()
    };
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .response_options(options)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(50))
        .unwrap();
    assert!(matches!(
        result.response.as_ref().unwrap().uncertainty,
        ResponseUncertainty::SimultaneousBand { replicates: 100, .. }
    ));
    assert!(result.provenance.get("estimate.response.kennedy_dr_simultaneous").is_some());
}

#[test]
fn selected_outcome_curve_runs_through_facade_without_complete_data_bands() {
    let n = 120;
    let confounder: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> =
        (0..n).map(|i| confounder[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03).collect();
    let outcome: Vec<f64> =
        (0..n).map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * confounder[i]).collect();
    let selected = vec![1.0; n];
    let data = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("confounder", confounder.as_slice()),
        ("selected", selected.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    })
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(3),
        },
        [ObservationAssumption::OutcomeIndependentGiven(Arc::from([
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ]))],
    );
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(50))
        .unwrap();
    let response = result.response.as_ref().unwrap();
    assert_eq!(response.uncertainty, ResponseUncertainty::None);
    assert_eq!(response.provenance_id.as_ref(), "estimate.response.observation_adjusted");
    assert!(result.provenance.get("estimate.response.observation_adjusted").is_some());
}

#[test]
fn intervention_response_runs_through_public_study_facade_with_its_own_strategy() {
    let n = 240;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> = (0..n).map(|i| z[i] + (i as f64 / 11.0).cos()).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i]).collect();
    let data = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("confounder", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.25))]),
    });

    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(52))
        .unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("response.intervention_gcomp"));
    let response = result.response.as_ref().unwrap();
    assert_eq!(response.provenance_id.as_ref(), "estimate.response.intervention_gcomp");
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) = response.estimate
    else {
        panic!("expected scalar intervention response");
    };
    assert!((value - 1.5).abs() < 0.25, "value={value}");
}

#[test]
fn two_point_curve_contrast_conforms_to_average_effect_under_shared_linear_contract() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/two_point_curve_average_effect/expected.json"
    ))
    .unwrap();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> =
        (0..n).map(|i| 0.6 * z[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i] + (i as f64 / 13.0).sin() * 0.02)
        .collect();
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let control = fixture["contract"]["control_level"].as_f64().unwrap();
    let active = fixture["contract"]["active_level"].as_f64().unwrap();
    let curve_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![control, active].into()),
        ),
    });
    let curve = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(curve_query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(51))
        .unwrap();
    let average = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            control,
            active,
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(51))
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &curve.response.as_ref().unwrap().estimate
    else {
        panic!("expected a point-identified two-point response");
    };
    let curve_contrast = mean[1] - mean[0];
    let average_effect = average.effect();
    let contrast_tolerance = fixture["tolerance"]["contrast_absolute"].as_f64().unwrap();
    let truth_tolerance = fixture["tolerance"]["truth_absolute"].as_f64().unwrap();
    let truth = fixture["contract"]["true_contrast"].as_f64().unwrap();
    assert!(
        (curve_contrast - average_effect).abs() <= contrast_tolerance,
        "curve contrast {curve_contrast} and AverageEffect {average_effect} exceed the documented tolerance {contrast_tolerance}"
    );
    assert!((curve_contrast - truth).abs() <= truth_tolerance);
    assert!((average_effect - truth).abs() <= truth_tolerance);
}
