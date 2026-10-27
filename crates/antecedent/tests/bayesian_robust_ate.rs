//! Staged, licensed Bayesian robust ATE cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, CausalQuery as CoreCausalQuery, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_robust_ate/expected.json"
    ))
    .unwrap()
}

fn synthetic() -> (TabularData, Dag) {
    let spec = fixture();
    let n = usize::try_from(spec["n"].as_u64().unwrap()).expect("row count fits usize");
    let x: Vec<f64> = (0..n).map(|i| ((i * 7919 % n) as f64 / n as f64) * 2.0 - 1.0).collect();
    let treatment: Vec<f64> = {
        let mut rng = antecedent_core::ExecutionContext::for_tests(611)
            .rng
            .stream_for(antecedent_core::StreamDomain::Estimate, 91);
        x.iter()
            .map(|v| f64::from(rng.next_f64() < 1.0 / (1.0 + (-(-0.1 + 0.8 * v)).exp())))
            .collect()
    };
    let outcome: Vec<f64> = x.iter().zip(&treatment).map(|(v, t)| 1.4 * v * v + 2.0 * t).collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    (data, dag)
}

#[test]
fn bayesian_robust_ate_all_observed_staged_known_truth() {
    let spec = fixture();
    let (data, dag) = synthetic();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let builder = Study::tabular(data.clone())
        .graph(dag)
        .query(CoreCausalQuery::AverageEffect(query.clone()))
        .estimator(antecedent::EstimatorId::BayesianRobustAte)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(80)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    assert_eq!(builder.support_status(), Some(CellStatus::Licensed));
    let ctx = antecedent_core::ExecutionContext::for_tests(22);
    let one_shot = builder.run(&ctx).unwrap();
    let mut prepared = builder.prepare(&ctx).unwrap();
    let RetainedOperation::BayesianRobustAte(plan) = prepared.retained_operation() else {
        panic!("checked robust ATE plan survives builder disposal")
    };
    assert_eq!(plan.query, query);
    assert_eq!(plan.posterior_draws, 80);
    assert_eq!(plan.adjustment_set.as_ref(), &[VariableId::from_raw(0)]);
    assert_eq!(plan.row_ids.len(), data.row_count());
    assert_eq!(plan.fold_ids.len(), data.row_count());
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("bayesian.robust_ate"));
    drop(builder);
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!(
        (one_shot.estimate().as_effect().unwrap().ate - result.estimate().as_effect().unwrap().ate)
            .abs()
            < 1e-12
    );
    assert!(result.posterior.is_some());
    assert!(
        (result.estimate().as_effect().unwrap().ate - spec["truth"].as_f64().unwrap()).abs()
            < spec["tolerance"].as_f64().unwrap()
    );
    assert!(
        result.diagnostics.iter().any(
            |d| d.code.as_ref() == "estimate.bayesian.robust_ate.modular_bootstrap_pushforward"
        )
    );
    assert!(result.posterior.as_ref().unwrap().assumptions.entries.iter().any(|a| matches!(&a.assumption, antecedent_core::Assumption::ParametricRestriction(p) if p.id.as_ref() == "bayesian.robust_ate.modular_bootstrap_pushforward")));
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert_eq!(
        refreshed.estimate().as_effect().unwrap().ate.to_bits(),
        result.estimate().as_effect().unwrap().ate.to_bits()
    );
    let RetainedOperation::BayesianRobustAte(retained) = prepared.retained_operation() else {
        panic!("checked robust ATE plan survives refresh")
    };
    assert_eq!(retained, plan);
    let x = data.float64_values(VariableId::from_raw(0)).unwrap();
    let treatment = data.float64_values(VariableId::from_raw(1)).unwrap();
    let outcome = data.float64_values(VariableId::from_raw(2)).unwrap();
    let shortened = antecedent_data::TabularData::from_f64_columns([
        ("x", &x[..x.len() - 1]),
        ("t", &treatment[..treatment.len() - 1]),
        ("y", &outcome[..outcome.len() - 1]),
    ])
    .unwrap();
    // A branch on other rows carries the row and fold identities of those rows; the handle it
    // was bound from keeps its own.
    let shorter = prepared.rebind_data(shortened, &ctx).unwrap().materialize(&ctx).unwrap();
    let RetainedOperation::BayesianRobustAte(rebound) = shorter.retained_operation() else {
        panic!("a rebound branch keeps the checked robust ATE operation")
    };
    assert_eq!(rebound.row_ids.len(), data.row_count() - 1);
    assert_eq!(rebound.fold_ids.len(), data.row_count() - 1);
    assert_eq!(rebound.query, query);
    assert_eq!(rebound.posterior_draws, 80);
    let RetainedOperation::BayesianRobustAte(unchanged) = prepared.retained_operation() else {
        panic!("the source handle keeps the checked robust ATE operation")
    };
    assert_eq!(unchanged.row_ids.len(), data.row_count());
    assert_eq!(unchanged.fold_ids.len(), data.row_count());
    let encoded = prepared.encode_contracted_result(&refreshed, "robust-stage", &ctx).unwrap();
    let (_, _, artifact) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert!(artifact.estimate.is_some());
    let bytes = artifact.posterior_artifact.unwrap();
    let (wire, draws) = antecedent_io::decode_causal_posterior_bytes(&bytes).unwrap();
    assert_eq!(wire.n_draws, 80);
    assert_eq!(draws.as_slice(), refreshed.posterior.as_ref().unwrap().draws.values.as_ref());
    let consumed = antecedent_io::consume_analysis_result(&encoded).unwrap();
    assert!(
        consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_bayesian_robust_ate_operation"
        })
    );
    assert!(!consumed.acceptance.accepts_as_verified_program());
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
