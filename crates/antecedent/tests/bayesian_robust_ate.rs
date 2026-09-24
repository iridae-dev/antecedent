//! Staged, licensed Bayesian robust ATE cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, CausalQuery as CoreCausalQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_robust_ate/expected.json"
    ))
    .unwrap()
}

fn synthetic() -> (TabularData, Dag) {
    let spec = fixture();
    let n = spec["n"].as_u64().unwrap() as usize;
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
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(CoreCausalQuery::AverageEffect(query))
        .estimator(antecedent::EstimatorId::BayesianRobustAte)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(80)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
    let ctx = antecedent_core::ExecutionContext::for_tests(22);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!(result.posterior.is_some());
    assert!(
        (result.estimate.ate - spec["truth"].as_f64().unwrap()).abs()
            < spec["tolerance"].as_f64().unwrap()
    );
    assert!(
        result.diagnostics.iter().any(
            |d| d.code.as_ref() == "estimate.bayesian.robust_ate.modular_bootstrap_pushforward"
        )
    );
    assert!(result.posterior.as_ref().unwrap().assumptions.entries.iter().any(|a| matches!(&a.assumption, antecedent_core::Assumption::ParametricRestriction(p) if p.id.as_ref() == "bayesian.robust_ate.modular_bootstrap_pushforward")));
    let encoded = prepared.encode_contracted_result(&result, "robust-stage", &ctx).unwrap();
    let (_, _, artifact) = antecedent_io::decode_analysis_result_artifact(&encoded).unwrap();
    assert!(artifact.estimate.is_some());
    let bytes = artifact.posterior_artifact.unwrap();
    let (wire, draws) = antecedent_io::decode_causal_posterior_bytes(&bytes).unwrap();
    assert_eq!(wire.n_draws, 80);
    assert_eq!(draws.as_slice(), result.posterior.as_ref().unwrap().draws.values.as_ref());
}
