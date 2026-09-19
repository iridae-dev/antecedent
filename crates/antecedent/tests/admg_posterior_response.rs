//! ADMG graph-posterior response surfaces: functional.effect per atom, frozen mix.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, SemanticApplicability,
    StructuralAggregationPolicy, StructuralWeightBasis, Study,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, adjacency_mask_from_admg, admg_from_adjacency_mask,
    dag_from_adjacency_mask, mask_is_dag, set_edge,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap()
}

fn expand_contingency(pin: &serde_json::Value) -> TabularData {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn node(columns: &[&str], name: &str) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap())
}

fn admg_from_pin(pin: &serde_json::Value) -> Admg {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut admg = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        admg.insert_directed(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        admg.insert_bidirected(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    admg
}

fn cyclic_mask(n: usize) -> u64 {
    let mut mask = 0_u64;
    mask = set_edge(mask, n, 0, 1, true);
    if n >= 3 {
        mask = set_edge(mask, n, 1, 2, true);
        mask = set_edge(mask, n, 2, 0, true);
    } else {
        mask = set_edge(mask, n, 1, 0, true);
    }
    mask
}

fn admg_posterior(n: usize, weights: &[f64], masks: &[u64]) -> GraphPosterior {
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks.to_vec(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("admg_posterior_response"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg)
}

fn intervention_response(t: VariableId, y: VariableId, level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: y,
        interventions: Arc::from([Intervention::set(t, Value::f64(level))]),
    })
}

fn response_curve(t: VariableId, y: VariableId) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from([0.0, 1.0]))),
    })
}

fn response_values(result: &antecedent::StudyResult) -> Vec<f64> {
    let response = result.response.as_ref().expect("response payload");
    match &response.estimate {
        antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Surface { mean, .. },
        ) => mean.to_vec(),
        antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Scalar(value),
        ) => vec![*value],
        other => panic!("unexpected response estimate {other:?}"),
    }
}

fn run_explicit_admg_response(
    data: &TabularData,
    admg: Admg,
    query: ResponseQuery,
    inference: InferenceMode,
) -> antecedent::StudyResult {
    Study::tabular(data.clone())
        .graph(admg)
        .query(CausalQuery::Response(query))
        .identifier(antecedent::IdentifierId::GeneralId)
        .estimator(antecedent::EstimatorId::FunctionalEffect)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap()
}

fn has_policy(result: &antecedent::StudyResult, policy: StructuralAggregationPolicy) -> bool {
    result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
            && d.message.contains(policy.as_str())
    })
}

#[test]
fn admg_graph_posterior_intervention_response_matches_identified_atom() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let n = admg.node_count();
    let identified_mask = adjacency_mask_from_admg(&admg).unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(2);
    let response = intervention_response(t, y, 1.0);
    let weights = [1.0];
    let gp = admg_posterior(n, &weights, &[identified_mask]);

    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        let explicit =
            run_explicit_admg_response(&data, admg.clone(), response.clone(), inference.clone());
        let mixed = Study::tabular(data.clone())
            .graph_posterior(gp.clone())
            .query(CausalQuery::Response(response.clone()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(3))
            .unwrap();
        assert_eq!(mixed.support_status.unwrap().as_str(), "licensed");
        assert!(mixed.diagnostics.iter().any(|d| {
            d.code.as_ref() == "identify.response.general_id"
                || d.code.as_ref() == "estimate.response.admg_graph_posterior"
        }));
        assert!(has_policy(&mixed, StructuralAggregationPolicy::SameEstimandWeightedMean));
        let atom = response_values(&explicit);
        let got = response_values(&mixed);
        assert_eq!(got.len(), atom.len());
        for (value, want) in got.iter().zip(atom.iter()) {
            assert!((value - want).abs() < 1e-6, "IR mix {value} != atom {want}");
        }
    }
}

#[test]
fn admg_graph_posterior_response_curve_matches_identified_atom() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let n = admg.node_count();
    let identified_mask = adjacency_mask_from_admg(&admg).unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(2);
    let response = response_curve(t, y);
    let gp = admg_posterior(n, &[1.0], &[identified_mask]);

    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        let explicit =
            run_explicit_admg_response(&data, admg.clone(), response.clone(), inference.clone());
        let mixed = Study::tabular(data.clone())
            .graph_posterior(gp.clone())
            .query(CausalQuery::Response(response.clone()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(3))
            .unwrap();
        assert_eq!(mixed.support_status.unwrap().as_str(), "licensed");
        let atom = response_values(&explicit);
        let got = response_values(&mixed);
        assert_eq!(got.len(), atom.len());
        for (value, want) in got.iter().zip(atom.iter()) {
            assert!((value - want).abs() < 1e-6, "curve mix {value} != atom {want}");
        }
    }
}

#[test]
fn admg_graph_posterior_response_retains_unidentified_mass() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let n = admg.node_count();
    let identified_mask = adjacency_mask_from_admg(&admg).unwrap();
    assert!(admg_from_adjacency_mask(cyclic_mask(n), n).is_err());
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(2);
    let response = intervention_response(t, y, 0.0);
    let gp = admg_posterior(n, &[0.8, 0.2], &[identified_mask, cyclic_mask(n)]);

    let result = Study::tabular(data)
        .graph_posterior(gp)
        .query(CausalQuery::Response(response))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let mixture = result.structural_response.as_ref().expect("mixture");
    assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
    assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12);
    assert!(result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.response.admg_graph_posterior"
            && d.message.contains("functional.effect")
    }));
}

fn assert_plugin_level_ir(result: &antecedent::StudyResult, inference: &InferenceMode, suite: RefuteSuite) {
    assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{inference:?} {suite:?}");
    assert!(
        !result.refutations.is_empty(),
        "{inference:?} {suite:?} must run plugin-level refuters"
    );
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "refute.evalue.not_a_contrast"),
        "{inference:?} {suite:?} missing plugin-level not-a-contrast diagnostic"
    );
    assert!(
        result.refutations.iter().any(|r| r.refuter.contains("overlap")),
        "{inference:?} {suite:?} cheap/full must report overlap on the intervention level"
    );
    if suite == RefuteSuite::Full {
        let stability_report = result.refutations.iter().any(|r| {
            r.refuter.contains("bootstrap")
                || r.refuter.contains("data_subset")
                || r.refuter.contains("graph")
                || r.refuter.contains("stability")
        });
        let stability_requested = result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "refute.validator.not_applicable"
                && d.fields.iter().any(|(k, v)| {
                    k.as_ref() == "validator"
                        && matches!(v.as_ref(), "bootstrap" | "data_subset" | "graph")
                })
        });
        assert!(
            stability_report || stability_requested,
            "{inference:?} {suite:?} full must request sampling-stability \
             (report or not_applicable skip): reports={:?} diags={:?}",
            result.refutations.iter().map(|r| r.refuter.as_ref()).collect::<Vec<_>>(),
            result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn admg_graph_posterior_intervention_response_cheap_and_full_run() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let n = admg.node_count();
    let identified_mask = adjacency_mask_from_admg(&admg).unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(2);
    let response = intervention_response(t, y, 1.0);
    let gp = admg_posterior(n, &[1.0], &[identified_mask]);

    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = Study::tabular(data.clone())
                .graph_posterior(gp.clone())
                .query(CausalQuery::Response(response.clone()))
                .inference(inference.clone())
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(3))
                .unwrap();
            assert_plugin_level_ir(&result, &inference, suite);
        }
    }
}

#[test]
fn admg_explicit_and_accepted_intervention_response_cheap_and_full_run() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(2);
    let response = intervention_response(t, y, 1.0);

    for accepted in [false, true] {
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
        ] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let mut builder = Study::tabular(data.clone());
                builder = if accepted {
                    builder.graph(AcceptedGraph::from(admg.clone()))
                } else {
                    builder.graph(admg.clone())
                };
                let result = builder
                    .query(CausalQuery::Response(response.clone()))
                    .identifier(antecedent::IdentifierId::GeneralId)
                    .estimator(antecedent::EstimatorId::FunctionalEffect)
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(3))
                    .unwrap();
                assert_plugin_level_ir(&result, &inference, suite);
            }
        }
    }
}

#[test]
fn admg_graph_posterior_response_inspect_and_execute_agree() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let n = admg.node_count();
    let identified_mask = adjacency_mask_from_admg(&admg).unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(2);
    let response = intervention_response(t, y, 1.0);
    let gp = admg_posterior(n, &[1.0], &[identified_mask]);
    let ctx = ExecutionContext::for_tests(3);

    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    ] {
        let builder = Study::tabular(data.clone())
            .graph_posterior(gp.clone())
            .query(CausalQuery::Response(response.clone()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        let inspected = builder.clone().inspect().unwrap();
        assert_eq!(inspected.support_status, Some(CellStatus::Licensed));
        assert_eq!(
            builder.clone().capability().unwrap().applicability,
            SemanticApplicability::Licensed
        );
        let result = builder.build().unwrap().run(&ctx).unwrap();
        assert_eq!(result.support_status, inspected.support_status);
    }
}

#[test]
fn admg_frontdoor_mask_is_not_a_dag_for_response() {
    let pin = pin();
    let admg = admg_from_pin(&pin);
    let mask = adjacency_mask_from_admg(&admg).unwrap();
    assert!(!mask_is_dag(mask, admg.node_count()));
    assert!(dag_from_adjacency_mask(mask, admg.node_count()).is_err());
}
