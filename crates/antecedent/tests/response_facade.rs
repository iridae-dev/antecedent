//! End-to-end continuous-response facade coverage.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SupportStatus, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
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
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/observation_primitives/expected.json"
    ))
    .unwrap();
    let sel = &pin["selected_outcome"];
    let nums = |key: &str| -> Vec<f64> {
        sel[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
    };
    let mu = nums("outcome_regression");
    let got = antecedent_stats::selected_outcome_pseudo_values(
        &nums("observed"),
        &nums("indicator"),
        &nums("probabilities"),
        Some(mu.as_slice()),
    )
    .unwrap();
    let expected = nums("expected_aipw_pseudo_values");
    let atol = pin["tolerance"]["atol"].as_f64().unwrap_or(1e-12);
    assert!(
        got.iter().zip(&expected).all(|(a, b)| (a - b).abs() <= atol),
        "facade ObservationSpec::Selected run must consume observation_primitives, got {got:?} expected {expected:?}"
    );

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

/// Known-truth pin for `response.intervention_gcomp`: see
/// `conformance/response/intervention_response/expected.json`. Same deterministic
/// generator as `intervention_response_runs_through_public_study_facade_with_its_own_strategy`
/// above (zero outcome noise), but checked against the fixture's analytically pinned
/// `true_response` rather than a hand-typed literal in the test body.
#[test]
fn intervention_response_conforms_to_known_truth_fixture() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/intervention_response/expected.json"
    ))
    .unwrap();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> = (0..n).map(|i| z[i] + (i as f64 / 11.0).cos()).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i]).collect();
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
    let set_value = fixture["contract"]["intervention"]["value"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(
            VariableId::from_raw(0),
            Value::f64(set_value),
        )]),
    });

    // Exercise both the direct `Study::run` path and the prepared handle (which is
    // what the licensed cell actually uses on the Python `PreparedAnalysis` surface),
    // and pin both against the same fixture truth.
    let study = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let direct = study.run(&ExecutionContext::for_tests(52)).unwrap();
    let prepared = study.prepare(&ExecutionContext::for_tests(52)).unwrap();
    let via_prepared = prepared.estimate(&data, &ExecutionContext::for_tests(52)).unwrap();

    let truth = fixture["contract"]["true_response"].as_f64().unwrap();
    let tolerance = fixture["tolerance"]["truth_absolute"].as_f64().unwrap();
    for (label, result) in [("direct", &direct), ("prepared", &via_prepared)] {
        let response = result.response.as_ref().unwrap();
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.intervention_gcomp");
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        else {
            panic!("{label}: expected scalar intervention response");
        };
        assert!(
            (value - truth).abs() <= tolerance,
            "{label}: value={value} truth={truth} tolerance={tolerance}"
        );
    }
}

#[test]
#[allow(clippy::too_many_lines)]
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

    let bayes_pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/response_surfaces/expected.json"
    ))
    .unwrap();
    for accepted in [false, true] {
        for intervention in [false, true] {
            let query = if intervention {
                ResponseQuery::new(ResponseFunctional::InterventionResponse {
                    outcome: VariableId::from_raw(1),
                    interventions: Arc::from([Intervention::set(
                        VariableId::from_raw(0),
                        Value::f64(active),
                    )]),
                })
            } else {
                curve_query.clone()
            };
            let builder = Study::tabular(data.clone());
            let builder = if accepted {
                builder.graph(antecedent::AcceptedGraph::dag(graph.clone()))
            } else {
                builder.graph(graph.clone())
            };
            let study = builder
                .query(query)
                .inference(antecedent::InferenceMode::Bayesian(
                    antecedent::BayesianConfig::conjugate().n_draws(4096),
                ))
                .refute(RefuteSuite::None)
                .build()
                .unwrap();
            let result = study
                .prepare(&ExecutionContext::for_tests(51))
                .unwrap()
                .estimate(&data, &ExecutionContext::for_tests(51))
                .unwrap();
            let response = result.response.unwrap();
            assert_eq!(response.provenance_id.as_ref(), "estimate.response.bayesian");
            let values = match response.estimate {
                ResponseIdentification::PointIdentified(ResponseValue::Surface {
                    mean, ..
                }) => mean.to_vec(),
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(mean)) => vec![mean],
                _ => panic!("response shape"),
            };
            let truth = if intervention { vec![2.0] } else { vec![0.0, 2.0] };
            for (value, truth) in values.iter().zip(truth) {
                assert!((value - truth).abs() < bayes_pin["static_tolerance"].as_f64().unwrap());
            }
            assert!(!matches!(response.uncertainty, ResponseUncertainty::None));
        }
    }
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

fn mean_curve_study() -> (antecedent_data::TabularData, Dag, ResponseQuery) {
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
    (data, graph, query)
}

#[test]
fn prepared_response_curve_reuses_identification() {
    let (data, graph, query) = mean_curve_study();
    let ctx = ExecutionContext::for_tests(50);
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = study.run(&ctx).unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    assert!(click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    assert!(fresh.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"));
    assert_eq!(click.estimand.adjustment_set, fresh.estimand.adjustment_set);
    let click_mean = match &click.response.as_ref().unwrap().estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => mean,
        other => panic!("expected surface, got {other:?}"),
    };
    let fresh_mean = match &fresh.response.as_ref().unwrap().estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => mean,
        other => panic!("expected surface, got {other:?}"),
    };
    assert_eq!(click_mean.len(), fresh_mean.len());
    for (a, b) in click_mean.iter().zip(fresh_mean.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn graph_posterior_response_retains_probability_atoms_and_mass() {
    let (data, _graph, query) = mean_curve_study();
    let ctx = ExecutionContext::for_tests(1);
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let gp = antecedent::discovery::discover_exact_dag_posterior(
        &data,
        &vars,
        &antecedent::discovery::BayesianDiscoverParams::default(),
        &ctx,
    )
    .unwrap();
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128)),
    ] {
        let study = Study::tabular(data.clone())
            .graph_posterior(gp.clone())
            .query(CausalQuery::Response(query.clone()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let result = study.run(&ctx).unwrap();
        let structural = result.structural_response.as_ref().expect("structural response");
        assert_eq!(
            structural.weight_basis,
            antecedent::result::StructuralWeightBasis::PosteriorProbability
        );
        assert!(structural.conditional_on_identified.is_some());
        assert!((structural.identified_mass + structural.unidentified_mass - 1.0).abs() < 1e-10);
        assert_eq!(structural.atoms.len(), gp.n_graphs);
        assert!(result.response.is_some());
    }
}

#[test]
fn prepared_graph_posterior_response_reuses_identification() {
    let (data, _graph, query) = mean_curve_study();
    let ctx = ExecutionContext::for_tests(1);
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let gp = antecedent::discovery::discover_exact_dag_posterior(
        &data,
        &vars,
        &antecedent::discovery::BayesianDiscoverParams::default(),
        &ctx,
    )
    .unwrap();
    let study = Study::tabular(data.clone())
        .graph_posterior(gp)
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = study.clone().run(&ctx).unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let second = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(
        fresh.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count(),
        0
    );
    assert_eq!(
        first.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count(),
        1
    );
    assert_eq!(
        second.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count(),
        1
    );
    let structural = first.structural_response.as_ref().expect("structural");
    assert!(fresh.diagnostics.iter().any(|d| d.message.contains("unevaluable_mass")
        || d.code.as_ref() == "estimate.response.graph_posterior"));
    assert!(
        first
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.response.graph_posterior.joint_if_se"
                || d.code.as_ref() == "estimate.response.graph_posterior.uncertainty_withheld")
    );
    assert!((structural.identified_mass + structural.unidentified_mass - 1.0).abs() <= 1.0 + 1e-9);
}

#[test]
fn graph_posterior_intervention_response_cheap_and_full_run_plugin_refuters() {
    let (data, _graph, _) = mean_curve_study();
    let ctx = ExecutionContext::for_tests(3);
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let gp = antecedent::discovery::discover_exact_dag_posterior(
        &data,
        &vars,
        &antecedent::discovery::BayesianDiscoverParams::default(),
        &ctx,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.25))]),
    });
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let result = Study::tabular(data.clone())
            .graph_posterior(gp.clone())
            .query(CausalQuery::Response(query.clone()))
            .inference(InferenceMode::Frequentist)
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!(
            !result.refutations.is_empty(),
            "graph-posterior InterventionResponse {suite:?} must run plugin-level refuters"
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.code.as_ref() == "refute.evalue.not_a_contrast" })
        );
        assert!(result.structural_response.is_some());
    }
}

#[test]
fn prepared_response_refute_is_refused() {
    let (data, graph, query) = mean_curve_study();
    let ctx = ExecutionContext::for_tests(50);
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let prior = prepared.estimate(&data, &ctx).unwrap();
    let err = prepared.refute(&prior, &data, RefuteSuite::Cheap, &ctx).unwrap_err();
    let msg = err.to_string();
    assert!(msg.starts_with("refused:"), "{msg}");
    assert!(msg.contains("AverageEffect and scalar InterventionResponse"), "{msg}");
}

#[test]
fn curve_joint_influence_has_sample_mean_scaling() {
    let (data, _, query) = mean_curve_study();
    let mut est =
        antecedent_estimate::ContinuousResponseEstimator::new(
            std::sync::Arc::<[VariableId]>::from([VariableId::from_raw(2)]),
        );
    est.options.export_row_diagnostics = true;
    let (response, scores) = est
        .estimate_identified_scored(
            &data,
            &query,
            antecedent_core::IdentificationStatus::NonparametricallyIdentified,
            antecedent_core::AssumptionSet::default(),
        )
        .unwrap();
    let scores = scores.unwrap();
    let n = scores.row_index.len();
    let exported = &response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.row_influence")
        .unwrap()
        .values;
    for (g, col) in scores.columns.iter().enumerate() {
        for (i, value) in col.iter().enumerate() {
            assert!((value - n as f64 * exported[g * n + i]).abs() < 1e-12);
        }
    }
    let refs: Vec<_> = scores.columns.iter().map(Vec::as_slice).collect();
    let cov = antecedent_estimate::joint_influence_covariance(&refs, None).unwrap();
    if let ResponseUncertainty::PointwiseBand { lower, upper, .. } = response.uncertainty {
        for g in 0..lower.len() {
            let se = (upper[g] - lower[g]) / (2.0 * 1.959_963_984_540_054);
            let joint_se = cov.values[g * cov.dim + g].sqrt();
            assert!((joint_se / se - (n as f64 / (n - 1) as f64).sqrt()).abs() < 1e-6);
        }
    } else {
        panic!("expected pointwise curve band");
    }
}

/// Frozen continuous-treatment law of `known_truth_mixtures.static_response`:
/// every treatment level carries four rows `Z = slope·T ± offset`,
/// `Y = 1 + 2T + 1.5Z ± noise`, so within-level deviations cancel exactly and
/// both atoms' regressions are exact.
fn known_truth_response_data(pin: &serde_json::Value) -> TabularData {
    let levels = usize::try_from(pin["treatment_levels"].as_u64().unwrap()).unwrap();
    let span = pin["treatment_span"].as_f64().unwrap();
    let slope = pin["z_slope"].as_f64().unwrap();
    let offset = pin["z_offset"].as_f64().unwrap();
    let noise = pin["outcome_noise"].as_f64().unwrap();
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..levels {
        let tv = -span + 2.0 * span * k as f64 / (levels - 1) as f64;
        for (z_sign, e_sign) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let zv = slope * tv + z_sign * offset;
            t.push(tv);
            z.push(zv);
            y.push(1.0 + 2.0 * tv + 1.5 * zv + e_sign * noise);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn known_truth_mixture_posterior(weights: &[f64]) -> antecedent_discovery::GraphPosterior {
    use antecedent_discovery::set_edge;
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    antecedent_discovery::GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![direct, adjusted, unidentified],
        vec![0.0; 9],
        vec![0.0; 9],
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        antecedent_prob::InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
}

fn response_values(value: &ResponseValue) -> Vec<f64> {
    match value {
        ResponseValue::Scalar(v) => vec![*v],
        ResponseValue::Surface { mean, .. } => mean.to_vec(),
        other => panic!("expected a scalar or surface response, got {other:?}"),
    }
}

/// D-2 (1.9 cell review): numeric known-truth pin of the static graph-posterior
/// response `conditional_on_identified` mean for `InterventionResponse` and
/// `ResponseCurve`, Frequentist and Bayesian. Multi-atom response uncertainty is
/// declared unavailable, so these cells carry numeric pins, not coverage.
#[test]
#[allow(clippy::too_many_lines)]
fn graph_posterior_response_known_truth_conditional_on_identified() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["static_response"];
    let floats = |key: &str| -> Vec<f64> {
        pin[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    };
    let weights = floats("posterior_weights");
    let grid = floats("treatment_grid");
    let atom_curves =
        [floats("identified_atom_curve_direct"), floats("identified_atom_curve_adjusted")];
    let expected_curve = floats("expected_conditional_on_identified_curve");
    let identified_mass = weights[0] + weights[1];
    for (g, value) in expected_curve.iter().enumerate() {
        let recomputed =
            (weights[0] * atom_curves[0][g] + weights[1] * atom_curves[1][g]) / identified_mass;
        assert!((recomputed - value).abs() < 1e-12, "fixture arithmetic at grid point {g}");
    }
    let data = known_truth_response_data(pin);
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(t, GridSpec::Values(grid.clone().into())),
    });
    let active = pin["intervention_level"].as_f64().unwrap();
    let level_index = grid.iter().position(|&v| (v - active).abs() < 1e-12).unwrap();
    let level = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(t, Value::f64(active))]),
    });
    let draws = usize::try_from(pin["bayesian_n_draws"].as_u64().unwrap()).unwrap();
    for (mode, inference, tolerance) in [
        (
            "frequentist",
            InferenceMode::Frequentist,
            pin["frequentist_abs_tolerance"].as_f64().unwrap(),
        ),
        (
            "bayesian",
            InferenceMode::Bayesian(
                BayesianConfig::conjugate()
                    .n_draws(draws)
                    .prior_scale(pin["bayesian_prior_scale"].as_f64().unwrap()),
            ),
            pin["bayesian_abs_tolerance"].as_f64().unwrap(),
        ),
    ] {
        for (label, query, expected_values) in [
            ("ResponseCurve", curve.clone(), expected_curve.clone()),
            ("InterventionResponse", level.clone(), vec![expected_curve[level_index]]),
        ] {
            let result = Study::tabular(data.clone())
                .graph_posterior(known_truth_mixture_posterior(&weights))
                .query(CausalQuery::Response(query))
                .inference(inference.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(pin["seed"].as_u64().unwrap()))
                .unwrap();
            let structural = result.structural_response.as_ref().expect("structural response");
            assert!(
                (structural.unidentified_mass
                    - pin["expected_unidentified_mass"].as_f64().unwrap())
                .abs()
                    < 1e-12,
                "{mode} {label}: unidentified mass must stay out of the mean"
            );
            let got = response_values(
                structural.conditional_on_identified.as_ref().expect("conditional mean"),
            );
            // Each identified atom keeps its own graph-specific value.
            let atom_values: Vec<Vec<f64>> = structural
                .atoms
                .iter()
                .filter_map(|atom| atom.value.as_ref().map(response_values))
                .collect();
            assert_eq!(atom_values.len(), 2, "{mode} {label}: two identified atoms");
            for (atom, pinned) in atom_values.iter().zip(&atom_curves) {
                let pinned: Vec<f64> = if label == "ResponseCurve" {
                    pinned.clone()
                } else {
                    vec![pinned[level_index]]
                };
                for (a, b) in atom.iter().zip(&pinned) {
                    assert!((a - b).abs() < tolerance, "{mode} {label} atom value {a} vs {b}");
                }
            }
            assert_eq!(got.len(), expected_values.len(), "{mode} {label}");
            for (g, (a, b)) in got.iter().zip(&expected_values).enumerate() {
                assert!(
                    (a - b).abs() < tolerance,
                    "{mode} {label} conditional_on_identified[{g}]={a}, pinned {b}"
                );
            }
            let uncertainty = &result.response.as_ref().unwrap().uncertainty;
            let has_diagnostic = |code: &str| {
                result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
            };
            if mode == "frequentist" && label == "InterventionResponse" {
                // A scalar frozen-weight aggregate takes the joint-IF SE of
                // the identified atoms on the shared sample.
                let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } = uncertainty
                else {
                    panic!("{mode} {label}: joint-IF aggregate SE expected, got {uncertainty:?}");
                };
                assert!(standard_error.is_finite() && *standard_error > 0.0);
                assert!(lower < &got[0] && &got[0] < upper, "{mode} {label}: interval brackets");
                assert!(has_diagnostic("estimate.response.graph_posterior.joint_if_se"));
            } else {
                assert!(
                    matches!(uncertainty, ResponseUncertainty::None),
                    "{mode} {label}: multi-atom aggregate uncertainty is withheld"
                );
                assert!(
                    has_diagnostic("estimate.response.graph_posterior.uncertainty_withheld"),
                    "{mode} {label}: withheld aggregate interval is disclosed"
                );
            }
        }
    }
    // Frequentist InterventionResponse cheap/full: the plugin-level refuters run
    // on the same frozen law and the pinned conditional-on-identified level holds.
    let tolerance = pin["frequentist_abs_tolerance"].as_f64().unwrap();
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let result = Study::tabular(data.clone())
            .graph_posterior(known_truth_mixture_posterior(&weights))
            .query(CausalQuery::Response(level.clone()))
            .inference(InferenceMode::Frequentist)
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(pin["seed"].as_u64().unwrap()))
            .unwrap();
        let structural = result.structural_response.as_ref().expect("structural response");
        let got = response_values(structural.conditional_on_identified.as_ref().unwrap());
        assert!(
            (got[0] - expected_curve[level_index]).abs() < tolerance,
            "{suite:?} InterventionResponse conditional_on_identified={}, pinned {}",
            got[0],
            expected_curve[level_index]
        );
        assert!(!result.refutations.is_empty(), "{suite:?} must run plugin-level refuters");
    }
}

/// E-4 (1.9 cell review): the static class-envelope response discloses
/// `estimate.envelope.response_posterior_not_mixed` only when per-completion
/// posterior uncertainty was actually dropped, not for a single completion.
#[test]
fn class_response_posterior_not_mixed_fires_only_when_uncertainty_is_dropped() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let data = known_truth_response_data(&expected["static_response"]);
    let (t, y, z) = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    });
    let run = |undirected: bool| {
        let mut cpdag = antecedent_graph::Cpdag::with_variables(3);
        cpdag.insert_directed(z, y).unwrap();
        cpdag.insert_directed(t, y).unwrap();
        if undirected {
            cpdag.insert_undirected(z, t).unwrap();
        } else {
            cpdag.insert_directed(z, t).unwrap();
        }
        Study::tabular(data.clone())
            .graph(cpdag)
            .query(CausalQuery::Response(query.clone()))
            .inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(128).prior_scale(1_000.0),
            ))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(4))
            .unwrap()
    };
    let fires = |result: &antecedent::StudyResult| {
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.envelope.response_posterior_not_mixed")
    };
    let single = run(false);
    assert!(
        !matches!(single.response.as_ref().unwrap().uncertainty, ResponseUncertainty::None),
        "a single completion keeps its own posterior band"
    );
    assert!(!fires(&single), "nothing is omitted for a single completion");
    let multi = run(true);
    assert!(matches!(multi.response.as_ref().unwrap().uncertainty, ResponseUncertainty::None));
    assert!(fires(&multi), "dropped per-completion bands must be disclosed");
}
