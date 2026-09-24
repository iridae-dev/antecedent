//! Compiler-surface consistency: an estimator never runs under an inference
//! mode it does not implement, not-identified questions refuse with their
//! identification outcome, inspection reports no program it has not compiled,
//! sharp RD's contract names its per-execution identification, and a
//! transformation preview reports the refusal its apply raises.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    BayesianConfig, CausalError, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
};
use antecedent_core::{ExecutionContext, IdentificationStatus, SlotAvailability, TransformIntent};
use antecedent_graph::{Admg, DenseNodeId};

mod common;

use common::fixtures::confounded_scm;

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16))
}

#[test]
fn checked_adjustment_executes_after_builder_discard_and_refresh() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(512, 73);
    let builder = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .bootstrap_replicates(0);
    let study = builder.clone().build().unwrap();
    let one_shot = study.run(&ctx).unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    drop(builder);
    drop(study);

    let plan = prepared.checked_linear_adjustment().expect("checked lowering retained");
    assert_eq!(plan.source_functional(), plan.program().mapping().source);
    assert_eq!(plan.executable_functional(), plan.program().mapping().executable);
    assert_eq!(plan.lowering().population, antecedent_core::TargetPopulation::AllObserved);
    let first = prepared.estimate(&data, &ctx).unwrap();
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert!(prepared.checked_linear_adjustment().is_some());
    let second_refresh = prepared.refresh(data, &ctx).unwrap();
    assert!((first.estimate.ate - 2.0).abs() < 0.2);
    assert!((first.estimate.ate - one_shot.estimate.ate).abs() < 1e-9);
    assert!((first.estimate.ate - refreshed.estimate.ate).abs() < 1e-9);
    assert!((first.estimate.ate - second_refresh.estimate.ate).abs() < 1e-9);
}

#[test]
fn checked_aipw_executes_after_builder_discard_and_rebinds_rows() {
    let ctx = ExecutionContext::for_tests(11);
    let (data, dag, query) = confounded_scm(512, 73);
    let aipw_builder = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let study = aipw_builder.clone().build().unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    drop(aipw_builder);
    drop(study);

    let lowering = prepared.checked_aipw_ate().expect("checked AIPW lowering retained");
    assert_eq!(lowering.program().mapping().source, lowering.target().functional);
    assert_eq!(lowering.lowering().population, antecedent_core::TargetPopulation::AllObserved);
    assert_eq!(lowering.lowering().rows.len(), 512);
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert!((first.estimate.ate - 2.0).abs() < 0.3);
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert!((refreshed.estimate.ate - first.estimate.ate).abs() < 1e-10);
    assert_eq!(prepared.checked_aipw_ate().unwrap().lowering().rows.len(), 512);
    let bytes = prepared.encode_contracted_result(&refreshed, "checked-aipw", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()));

    let (data, dag, query) = confounded_scm(512, 73);
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(8)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert_eq!(prepared.checked_aipw_ate().unwrap().lowering().bootstrap_replicates, 8);
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!(result.estimate.se_bootstrap.is_some());
    let bytes = prepared.encode_contracted_result(&result, "checked-aipw-bootstrap", &ctx).unwrap();
    assert!(
        antecedent_io::consume_analysis_result(&bytes)
            .unwrap()
            .acceptance
            .accepts_as_verified_program()
    );
}

/// Estimators that never run under the named inference mode on a static
/// `AverageEffect`. Every other estimator either implements the requested
/// mode or is refused for a different reason (identifier, query, data).
const REFUSED_UNDER_BAYESIAN: &[&str] = &[
    "linear.adjustment.ate",
    "propensity.weighting",
    "propensity.matching",
    "propensity.stratification",
    "distance.matching",
    "aipw",
    "glm.adjustment",
    "frontdoor.linear_two_stage",
    "frontdoor.functional",
    "iv.wald",
    "iv.2sls",
    "rd.sharp",
    "conditional.linear.adjustment",
    "cell.aipw",
    "transport.trial_ipw",
    "interference.ht_hajek",
    "dml",
    "dr.learner",
    "causal.forest",
];
const REFUSED_UNDER_FREQUENTIST: &[&str] = &[
    "bayesian.gcomp",
    "bayesian.basis.gcomp",
    "bayesian.robust_ate",
    "iv.bayesian_joint_linear",
    "rd.bayesian_local_linear",
    "transport.trial_bayesian_bootstrap",
    "conditional.bayesian",
    "bayesian.temporal.gcomp",
    "response.temporal.bayesian",
    "response.bayesian",
    "temporal.mediation.bayesian",
];

#[test]
fn estimator_inference_mismatch_is_refused_in_both_directions() {
    let (data, dag, query) = confounded_scm(64, 3);
    for &estimator in EstimatorId::ALL {
        for (label, inference, refused) in [
            ("Bayesian", bayesian(), REFUSED_UNDER_BAYESIAN),
            ("Frequentist", InferenceMode::Frequentist, REFUSED_UNDER_FREQUENTIST),
        ] {
            // The builder is order-independent: selecting the estimator before
            // or after the inference mode resolves to the same study.
            let orders = [
                Study::tabular(data.clone())
                    .graph(dag.clone())
                    .query(query.clone())
                    .estimator(estimator)
                    .inference(inference.clone()),
                Study::tabular(data.clone())
                    .graph(dag.clone())
                    .query(query.clone())
                    .inference(inference.clone())
                    .estimator(estimator),
            ];
            for builder in orders {
                let code = builder
                    .bootstrap_replicates(0)
                    .build()
                    .err()
                    .and_then(|err| err.reason_code().map(str::to_string));
                let expected = refused.contains(&estimator.as_str());
                assert_eq!(
                    code.as_deref() == Some("estimator_inference_mismatch"),
                    expected,
                    "{} under {label}: refusal code {code:?}",
                    estimator.as_str()
                );
            }
        }
    }
}

#[test]
fn frequentist_estimator_under_bayesian_never_executes() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(96, 5);
    for estimator in [EstimatorId::Aipw, EstimatorId::LinearAdjustmentAte] {
        let built = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query.clone())
            .inference(bayesian())
            .estimator(estimator)
            .bootstrap_replicates(0)
            .build();
        let Err(err) = built else {
            panic!("{} under Bayesian inference compiled a study", estimator.as_str());
        };
        assert_eq!(err.reason_code(), Some("estimator_inference_mismatch"), "{err}");
    }
    // The Bayesian estimator itself still runs under Bayesian inference.
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .inference(bayesian())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert!(prepared.estimate(&data, &ctx).unwrap().posterior.is_some());
}

#[test]
fn not_identified_refusal_carries_its_identification_outcome() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, _, query) = confounded_scm(96, 43);
    // Treatment and outcome share a latent confounder and nothing else
    // identifies the effect.
    let mut bow = Admg::with_variables(3);
    bow.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    bow.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let outcome = Study::tabular(data.clone())
        .graph(bow)
        .query(query)
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .and_then(|prepared| prepared.estimate(&data, &ctx));
    let Err(err) = outcome else { panic!("an unidentified effect produced a result") };
    match &err {
        CausalError::NotIdentified { status, search_capped, .. } => {
            assert_eq!(*status, IdentificationStatus::NotIdentified);
            assert!(!search_capped, "general ID completes on a two-node ADMG");
        }
        other => panic!("expected a typed not-identified refusal, got {other:?}"),
    }
    assert_eq!(err.reason_code(), Some("effect_not_identified"));
    assert!(err.to_string().contains("the search completed"), "{err}");

    let capped = CausalError::not_identified(IdentificationStatus::NotIdentified, true, "capped");
    assert_eq!(capped.reason_code(), Some("effect_not_identified"));
    assert!(capped.to_string().contains("not a proof of non-identification"), "{capped}");
}

#[test]
fn inspection_reports_no_program_before_identification() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(64, 7);
    let built =
        Study::tabular(data).graph(dag).query(query).bootstrap_replicates(0).build().unwrap();
    assert_eq!(built.inspect().unwrap().identities.program, None);
    let prepared = built.prepare(&ctx).unwrap();
    assert_eq!(prepared.inspect().unwrap().identities.program, None);
    let contract = prepared.contract().unwrap();
    let program = contract.identities.program.expect("a prepared contract compiles its program");
    assert!(contract.identities.refs().iter().any(|id| id.digest == program));
}

#[test]
fn sharp_rd_contract_names_per_execution_identification() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(96, 10);
    let prepared = Study::tabular(data)
        .graph(dag)
        .query(query)
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(antecedent_core::VariableId::from_raw(2), 0.0, 1.0)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    match &prepared.contract().unwrap().reasoning.identification {
        SlotAvailability::Unavailable { reason } => {
            assert_eq!(&**reason, "identified_per_execution");
        }
        SlotAvailability::Available(slot) => {
            panic!("sharp RD caches no identification product, got {slot:?}")
        }
        other => panic!("unexpected slot {other:?}"),
    }
}

#[test]
fn retarget_preview_reports_the_refusal_the_apply_raises() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(96, 12);
    let linear = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let preview = linear.preview_transform(TransformIntent::Retarget).unwrap();
    assert!(preview.refused, "a study without a score table cannot retarget");
    let weights = vec![1.0; 96];
    let applied = linear.retarget(&weights, &[], &ctx).unwrap_err();
    assert_eq!(applied.reason_code(), Some("score_table_unavailable"));
    assert_eq!(preview.refusal.as_deref(), Some(applied.to_string().as_str()));
    let refused_apply = linear.apply_retarget(&preview, &weights, &[], &ctx).unwrap_err();
    assert_eq!(refused_apply.reason_code(), Some("score_table_unavailable"));

    let aipw = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let preview = aipw.preview_transform(TransformIntent::Retarget).unwrap();
    assert!(!preview.refused);
    assert_eq!(preview.refusal, None);
    assert!(aipw.apply_retarget(&preview, &weights, &[], &ctx).is_ok());
    let refresh = aipw.preview_transform(TransformIntent::CompatibleDataReplace).unwrap();
    assert!(!refresh.refused, "a prepared study can always attempt a compatible refresh");
}
