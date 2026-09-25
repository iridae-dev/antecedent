//! Builder independent evidence for Bayesian DAG graph-posterior ATE execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, set_edge};
use antecedent_io::consume_analysis_result;
use antecedent_prob::InferenceDiagnostics;

fn data() -> TabularData {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut confounder = Vec::new();
    for _ in 0..20 {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(1.0 + 2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

fn adjusted_graph() -> u64 {
    let graph = set_edge(0, 3, 0, 1, true);
    let graph = set_edge(graph, 3, 2, 0, true);
    set_edge(graph, 3, 2, 1, true)
}

fn posterior() -> GraphPosterior {
    let graph = adjusted_graph();
    let masks = vec![graph, graph];
    let weights = vec![0.5, 0.5];
    GraphPosterior::new(
        3,
        weights.clone(),
        masks,
        vec![0.0; 9],
        vec![0.0; 9],
        1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>(),
        InferenceDiagnostics::analytic("bayesian_dag_ate_known_truth"),
        0,
    )
    .unwrap()
    .with_algorithm("known_truth_fixture")
}

#[test]
fn bayesian_dag_graph_posterior_ate_is_sealed_across_validation_lifecycle() {
    let data = data();
    let graph_posterior = posterior();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let context = ExecutionContext::for_tests(29_087);
    let inference =
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(96).prior_scale(30.0));

    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
        let builder = Study::tabular(data.clone())
            .graph_posterior(graph_posterior.clone())
            .query(query.clone())
            .inference(inference.clone())
            .refute(suite)
            .build()
            .unwrap();
        let mut prepared = builder.prepare(&context).unwrap();
        drop(builder);

        let plan = prepared
            .checked_bayesian_graph_posterior_ate_info()
            .expect("sealed Bayesian graph-posterior operation");
        assert_eq!(plan.query, query);
        assert_eq!(plan.estimator.as_str(), "bayesian.gcomp");
        assert_eq!(plan.validation, suite);
        assert_eq!(plan.graph_keys.as_ref(), graph_posterior.graph_keys.as_ref());
        assert_eq!(plan.weights.as_ref(), &[0.5, 0.5]);
        assert!(plan.inference.starts_with("bayesian:"));

        let result = prepared.estimate(&data, &context).unwrap();
        assert!((result.estimate.ate - 2.0).abs() < 0.12, "ATE={}", result.estimate.ate);
        let posterior_result = result.posterior.as_ref().expect("joint posterior retained");
        assert!(posterior_result.draws.n_draws >= 96);
        let mixture = result.structural_response.as_ref().expect("graph mass summary");
        assert!((mixture.identified_mass - 1.0).abs() < 1e-12);
        assert!(mixture.unidentified_mass.abs() < 1e-12);
        match suite {
            RefuteSuite::None => assert!(result.refutations.is_empty()),
            RefuteSuite::Cheap | RefuteSuite::Full => assert!(!result.refutations.is_empty()),
            RefuteSuite::PlaceboAndRcc => unreachable!(),
        }

        let refreshed = prepared.refresh(data.clone(), &context).unwrap();
        assert!((refreshed.estimate.ate - result.estimate.ate).abs() < 1e-10);
        let retained = prepared.checked_bayesian_graph_posterior_ate_info().unwrap();
        assert_eq!(retained.graph_keys, plan.graph_keys);
        assert_eq!(retained.weights, plan.weights);

        let artifact = prepared
            .encode_contracted_result(&refreshed, "bayesian-dag-graph-posterior-ate", &context)
            .unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert!(consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_bayesian_graph_posterior_ate_operation"
        }));
        assert!(!consumed.acceptance.accepts_as_verified_program());
    }
}
