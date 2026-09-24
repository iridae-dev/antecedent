//! Bayesian trial-to-target IPW through the public staged route.

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, TransportTrialSpec};
use antecedent_core::{
    CausalQuery, ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId};

#[test]
fn bayesian_trial_transport_matches_two_source_row_dirichlet_moments() {
    // Equal selection/treatment probabilities make the posterior draw
    // 6U - 2(1-U) = 8U - 2, with U ~ Beta(1,1). Its mean is 2 and
    // variance is 16/3, independently calculated from the uniform law.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_trial_transport/expected.json"
    ))
    .unwrap();
    let data = TabularData::from_f64_columns([
        ("a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("y", &[3.0, 1.0, 0.0, 0.0][..]),
        ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
        ("s", &[0.5, 0.5, 0.5, 0.5][..]),
        ("e", &[0.5, 0.5, 0.5, 0.5][..]),
    ])
    .unwrap();
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    let query = TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)]);
    let study = Study::tabular(data.clone())
        .graph(admg)
        .query(CausalQuery::Transport(query))
        .selection_targets(Arc::from([]))
        .transport_trial(TransportTrialSpec {
            trial: VariableId::from_raw(2),
            selection_probability: VariableId::from_raw(3),
            treatment_probability: VariableId::from_raw(4),
        })
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let context = antecedent_core::ExecutionContext::for_tests(41);
    let result = study.prepare(&context).unwrap().estimate(&data, &context).unwrap();
    let posterior = result.posterior.as_ref().expect("transport posterior");
    assert_eq!(posterior.draws.n_draws, 2_000);
    assert!(
        (posterior.summaries.mean[0] - fixture["expected_mean"].as_f64().unwrap()).abs()
            < fixture["mean_tolerance"].as_f64().unwrap()
    );
    assert!(
        (posterior.summaries.sd[0].powi(2) - fixture["expected_variance"].as_f64().unwrap()).abs()
            < fixture["variance_tolerance"].as_f64().unwrap()
    );
    assert!((result.estimate.ate - posterior.summaries.mean[0]).abs() < 1e-12);
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("transport.trial_bayesian_bootstrap")
    );
}
