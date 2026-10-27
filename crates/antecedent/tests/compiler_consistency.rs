//! Compiler-surface consistency: an estimator never runs under an inference
//! mode it does not implement, not-identified questions refuse with their
//! identification outcome, inspection reports no program it has not compiled,
//! a sharp RD design that does not identify names its per-execution identification, and a
//! transformation preview reports the refusal its apply raises.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{
    BayesianConfig, CausalError, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
};
use antecedent_contract::{SlotAvailability, TransformIntent};
use antecedent_core::{ExecutionContext, IdentificationStatus};
use antecedent_graph::{Admg, DenseNodeId};

#[path = "fixtures_common/mod.rs"]
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

    let RetainedOperation::CheckedLinear(plan) = prepared.retained_operation() else {
        panic!("checked lowering retained")
    };
    assert_eq!(plan.source_functional(), plan.program().mapping().source);
    assert_eq!(plan.executable_functional(), plan.program().mapping().executable);
    assert_eq!(plan.lowering().population, antecedent_core::TargetPopulation::AllObserved);
    let first = prepared.estimate(&data, &ctx).unwrap();
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert!(matches!(prepared.retained_operation(), RetainedOperation::CheckedLinear(..)));
    let second_refresh = rebind_estimate(&mut prepared, data, &ctx).unwrap();
    assert!((first.estimate().as_effect().unwrap().ate - 2.0).abs() < 0.2);
    assert!(
        (first.estimate().as_effect().unwrap().ate - one_shot.estimate().as_effect().unwrap().ate)
            .abs()
            < 1e-9
    );
    assert!(
        (first.estimate().as_effect().unwrap().ate - refreshed.estimate().as_effect().unwrap().ate)
            .abs()
            < 1e-9
    );
    assert!(
        (first.estimate().as_effect().unwrap().ate
            - second_refresh.estimate().as_effect().unwrap().ate)
            .abs()
            < 1e-9
    );
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

    let RetainedOperation::CheckedAipw(lowering) = prepared.retained_operation() else {
        panic!("checked AIPW lowering retained")
    };
    assert_eq!(lowering.program().mapping().source, lowering.target().functional);
    assert_eq!(lowering.lowering().population, antecedent_core::TargetPopulation::AllObserved);
    assert_eq!(lowering.lowering().rows.len(), 512);
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert!((first.estimate().as_effect().unwrap().ate - 2.0).abs() < 0.3);
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert!(
        (refreshed.estimate().as_effect().unwrap().ate - first.estimate().as_effect().unwrap().ate)
            .abs()
            < 1e-10
    );
    let RetainedOperation::CheckedAipw(retained) = prepared.retained_operation() else {
        panic!("checked AIPW lowering retained after refresh")
    };
    assert_eq!(retained.lowering().rows.len(), 512);
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
    let RetainedOperation::CheckedAipw(retained) = prepared.retained_operation() else {
        panic!("checked AIPW lowering retained")
    };
    assert_eq!(retained.lowering().bootstrap_replicates, 8);
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!(result.estimate().as_effect().unwrap().se_bootstrap().is_some());
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
/// mode or is refused for a different reason (identifier, query, data). The
/// matching estimators and propensity weighting are refused under Bayesian inference with their
/// typed posterior boundary (`route_not_supported`), not as a mismatch: matching has no
/// likelihood and weighting has no outcome law to put a posterior on.
const REFUSED_UNDER_BAYESIAN: &[&str] = &[
    "linear.adjustment.ate",
    "propensity.stratification",
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
    "interference.cluster_neyman",
    "interference.observational_ipw",
    "interference.saturation_exact",
    "dml",
    "dr.learner",
    "causal.forest",
    "policy.triangular_kernel_inverse_density",
    "randomized.survival_product_limit",
    "temporal.glm_link_contrast",
    "temporal.lagged_instrument",
    "temporal_design.excursion",
    "temporal_design.regime_survival",
    "iv_bounds.balke_pearl",
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
    "bayesian.temporal.white_innovation_linear",
    "bayesian.panel_did",
    "bayesian.synthetic_control",
    "bayesian.local_ratio",
    "bayesian.piecewise_survival",
    "bayesian.randomized_quantile",
    "bayesian.continuous_dose_policy",
    "bayesian.randomized_finite_population",
    "bayesian.randomized_superpopulation",
    "bayesian.complier_effect",
    "bayesian.switchback_effect",
    "bayesian.binary_policy_value",
    "bayesian.sequential_regime_value",
    "bayesian.sequential_crossfit_contrast",
    "bayesian.period_history_outcome_model",
    "bayesian.temporal.glm_link_contrast",
    "bayesian.temporal.lagged_instrument",
    "bayesian.temporal_design.excursion",
    "bayesian.temporal_design.regime_survival",
    "bayesian.iv_bounds.endpoint_bootstrap",
];

#[test]
fn estimator_inference_mismatch_is_refused_in_both_directions() {
    let (data, dag, query) = confounded_scm(64, 3);
    let mut disagreements = Vec::new();
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
                if (code.as_deref() == Some("estimator_inference_mismatch")) != expected {
                    disagreements.push(format!(
                        "{} under {label}: refusal code {code:?}, expected mismatch {expected}",
                        estimator.as_str()
                    ));
                }
            }
        }
    }
    assert!(disagreements.is_empty(), "{disagreements:#?}");
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

// Sharp RD identifies at compile and prepare: a design that identifies retains its product
// (the next test), and one that does not (the running variable is not a parent of the
// treatment, so no threshold rule assigns it) is refused at prepare rather than deferred to
// each execution, so no prepared sharp RD names identification as per-execution.
#[test]
fn sharp_rd_that_cannot_identify_is_refused_at_prepare_not_deferred_to_execution() {
    use antecedent_graph::Dag;

    let ctx = ExecutionContext::for_tests(1);
    let (data, _, query) = confounded_scm(96, 10);
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(0, 1), (2, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let refusal = Study::tabular(data)
        .graph(dag)
        .query(query)
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(antecedent_core::VariableId::from_raw(2), 0.0, 1.0)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap_err();
    assert_eq!(refusal.reason_code(), Some("effect_not_identified"), "{refusal}");
    assert!(refusal.to_string().contains("threshold assignment"), "{refusal}");
}

#[test]
fn sharp_rd_that_identifies_retains_its_design_from_compile() {
    use antecedent_core::{AverageEffectQuery, VariableId};
    use antecedent_graph::Dag;

    let ctx = ExecutionContext::for_tests(1);
    let n = 200_usize;
    let noise = |i: usize| ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5;
    let r: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect();
    let t: Vec<f64> = r.iter().map(|value| f64::from(*value >= 0.0)).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + 0.5 * r[i] + 0.1 * noise(i)).collect();
    let data = antecedent_data::TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap();
    // The running variable is the treatment's only parent, as a threshold rule requires.
    let mut dag = Dag::with_variables(3);
    for (from, to) in [(2, 0), (0, 1), (2, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::RdSharp)
        .rd_config(VariableId::from_raw(2), 0.0, 1.0)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    // Compile identifies the design and retains it; execution reads it from the program.
    let plan = study.compile(&ctx).unwrap();
    assert!(plan.lowered().unwrap().identification_retained());
    let prepared = study.prepare(&ctx).unwrap();
    assert!(
        matches!(
            &prepared.contract().unwrap().reasoning.identification,
            SlotAvailability::Available(_)
        ),
        "an identified sharp design retains its identification product"
    );
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!((result.estimate().as_effect().unwrap().ate - 2.0).abs() < 0.5);
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

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}
