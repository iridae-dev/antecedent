//! Builder independent execution evidence for the certified trial-to-target transport routes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
    TransportTrialSpec,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_io::consume_analysis_result;

const NAMES: [&str; 5] = ["a", "y", "trial", "s", "e"];

fn table(y: &[f64]) -> TabularData {
    TabularData::from_f64_columns([
        (NAMES[0], &[1.0, 0.0, 0.0, 0.0][..]),
        (NAMES[1], y),
        (NAMES[2], &[1.0, 1.0, 0.0, 0.0][..]),
        (NAMES[3], &[0.5, 0.5, 0.5, 0.5][..]),
        (NAMES[4], &[0.5, 0.5, 0.5, 0.5][..]),
    ])
    .unwrap()
}

fn widened() -> TabularData {
    TabularData::from_f64_columns([
        (NAMES[0], &[1.0, 0.0, 0.0, 0.0][..]),
        (NAMES[1], &[3.0, 1.0, 0.0, 0.0][..]),
        (NAMES[2], &[1.0, 1.0, 0.0, 0.0][..]),
        (NAMES[3], &[0.5, 0.5, 0.5, 0.5][..]),
        (NAMES[4], &[0.5, 0.5, 0.5, 0.5][..]),
        ("extra", &[0.0, 0.0, 0.0, 0.0][..]),
    ])
    .unwrap()
}

fn query() -> TransportQuery {
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)])
}

fn build(data: &TabularData, bayesian: bool) -> Study {
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let study = Study::tabular(data.clone())
        .graph(admg)
        .query(CausalQuery::Transport(query()))
        .selection_targets(Arc::from([]))
        .transport_trial(TransportTrialSpec {
            trial: VariableId::from_raw(2),
            selection_probability: VariableId::from_raw(3),
            treatment_probability: VariableId::from_raw(4),
        })
        .refute(RefuteSuite::None);
    let study = if bayesian {
        study.inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000)))
    } else {
        study
    };
    study.build().unwrap()
}

#[test]
fn transport_trial_executes_from_retained_sid_proof_after_builder_drop() {
    run_on_large_stack(transport_trial_executes_from_retained_sid_proof_after_builder_drop_body);
}

fn transport_trial_executes_from_retained_sid_proof_after_builder_drop_body() {
    let ipw_pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_transport/expected.json"
    ))
    .unwrap();
    let bayesian_pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_trial_transport/expected.json"
    ))
    .unwrap();
    let ipw = ipw_pin["ipw"].as_f64().unwrap();
    let ipw_tolerance = ipw_pin["tolerance"].as_f64().unwrap();
    let data = table(&[3.0, 1.0, 0.0, 0.0]);
    // Shifting every outcome leaves the IPW contrast of the two trial rows unchanged.
    let shifted = table(&[3.5, 1.5, 0.5, 0.5]);

    for bayesian in [false, true] {
        let ctx = ExecutionContext::for_tests(41);
        let builder = build(&data, bayesian);
        let one_shot = builder.run(&ctx).unwrap();
        let mut prepared = builder.prepare(&ctx).unwrap();
        drop(builder);

        let expected_estimator = if bayesian {
            EstimatorId::TransportTrialBayesianBootstrap
        } else {
            EstimatorId::TransportTrialIpw
        };
        let plan = prepared.checked_transport_trial_info().expect("retained transport plan");
        assert_eq!(plan.query, query());
        assert_eq!(plan.identifier, IdentifierId::TransportSid);
        assert_eq!(plan.estimator, expected_estimator);
        assert_eq!(plan.posterior_draws, bayesian.then_some(2_000));
        assert!(!plan.rule.is_empty());
        assert_eq!(prepared.plan().record.plan_id, plan.plan_id);

        let result = prepared.estimate(&data, &ctx).unwrap();
        assert!(result.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached"));
        assert_eq!(result.estimand.method.as_ref(), "transport.sid.direct");
        assert_eq!(result.logical_plan.estimator.as_deref(), Some(expected_estimator.as_str()));
        let transported = result.transport.as_ref().expect("transport estimate");
        if bayesian {
            let posterior = result.posterior.as_ref().expect("transport posterior");
            assert_eq!(posterior.draws.n_draws, 2_000);
            assert!(
                (posterior.summaries.mean[0] - bayesian_pin["expected_mean"].as_f64().unwrap())
                    .abs()
                    < bayesian_pin["mean_tolerance"].as_f64().unwrap()
            );
            assert!(
                (posterior.summaries.sd[0].powi(2)
                    - bayesian_pin["expected_variance"].as_f64().unwrap())
                .abs()
                    < bayesian_pin["variance_tolerance"].as_f64().unwrap()
            );
            assert!((result.estimate.ate - posterior.summaries.mean[0]).abs() < 1e-12);
            // The one-shot facade executes the same retained row-law bootstrap and seed.
            assert_eq!(
                one_shot.posterior.as_ref().unwrap().draws.values.as_ref(),
                posterior.draws.values.as_ref()
            );
        } else {
            assert!((transported.ipw - ipw).abs() <= ipw_tolerance);
            assert!((result.estimate.ate - ipw).abs() <= ipw_tolerance);
        }
        assert!((one_shot.estimate.ate - result.estimate.ate).abs() < 1e-12);

        let bytes =
            prepared.encode_contracted_result(&result, "checked-transport-trial", &ctx).unwrap();
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert!(
            consumed.acceptance.unresolved.iter().any(|dependency| {
                dependency.as_ref() == "dependencies.checked_transport_trial_operation"
            }),
            "{:?}",
            consumed.acceptance.unresolved
        );
        assert!(!consumed.acceptance.accepts_as_verified_program());

        let refreshed = prepared.refresh(shifted.clone(), &ctx).unwrap();
        assert!(
            refreshed.diagnostics.iter().any(|item| item.code.as_ref() == "exec.identify.cached")
        );
        assert_eq!(refreshed.estimand.method.as_ref(), "transport.sid.direct");
        if !bayesian {
            assert!((refreshed.transport.as_ref().unwrap().ipw - ipw).abs() <= ipw_tolerance);
        }
        assert!(prepared.has_checked_transport_trial_operation());
        assert!(prepared.refresh(widened(), &ctx).is_err());
    }
}

/// Debug builds of the shared prepared dispatcher carry frames close to the
/// default test-thread stack, so the evidence body runs on its own thread.
fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-route-evidence".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}
