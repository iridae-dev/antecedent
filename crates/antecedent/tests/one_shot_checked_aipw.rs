//! One-shot AIPW uses its complete prepared operation on the licensed slice.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{EstimatorId, EstimatorSpec, RefuteSuite, Study};
use antecedent_core::ExecutionContext;

mod common;

use common::fixtures::confounded_scm;

#[test]
fn one_shot_aipw_matches_prepared_and_nearby_linear_stays_supported() {
    let ctx = ExecutionContext::for_tests(29);
    let (data, dag, query) = confounded_scm(512, 73);
    let aipw = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();

    let one_shot = aipw.run(&ctx).unwrap();
    let prepared = aipw.prepare(&ctx).unwrap();
    let clicked = prepared.estimate(&data, &ctx).unwrap();
    assert!((one_shot.effect() - clicked.effect()).abs() < 1e-10);

    let linear = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert!((linear.run(&ctx).unwrap().effect() - 2.0).abs() < 0.2);
}

#[test]
fn one_shot_aipw_fails_closed_when_preparation_has_no_checked_operation() {
    let ctx = ExecutionContext::for_tests(31);
    let (data, dag, query) = confounded_scm(512, 73);
    let mut fitter = antecedent_estimate::AipwAte::new();
    fitter.overlap = antecedent_estimate::OverlapPolicy::RequireDiagnostics {
        clip: Some(0.01),
        trim: Some(0.02),
    };
    let study = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorSpec::Aipw(Box::new(fitter)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();

    let error = study.run(&ctx).unwrap_err();
    assert!(error.to_string().contains("did not retain its complete checked operation"));
}
