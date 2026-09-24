//! Staged Bayesian IV and sharp-RD model known-truth checks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, StreamDomain, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

fn table(vars: &[(&str, RoleHint, Vec<f64>)]) -> TabularData {
    let n = vars[0].2.len();
    let mut b = CausalSchemaBuilder::new();
    for (name, role, _) in vars {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(*role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols = vars
        .iter()
        .enumerate()
        .map(|(i, (_, _, values))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(values.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}
fn rd_graph() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    g
}

#[test]
fn bayesian_iv_joint_fixed_loading_executes_staged_with_known_truth() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_iv/expected.json"
    ))
    .unwrap();
    let truth = expected["true_structural_effect"].as_f64().unwrap();
    let n = 1200;
    let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let u: Vec<f64> = (0..n).map(|i| ((i % 17) as f64 - 8.0) / 5.0).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.6 * z[i] + u[i]).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + u[i]).collect();
    let data = table(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let refused = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        ))
        .identifier(IdentifierId::Iv)
        .estimator(EstimatorId::BayesianIvJointLinear)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512)))
        .refute(antecedent::RefuteSuite::Cheap)
        .build();
    assert!(refused.is_err(), "IV cheap validation has no support license");
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        ))
        .identifier(IdentifierId::Iv)
        .estimator(EstimatorId::BayesianIvJointLinear)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512)))
        .refute(antecedent::RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(81);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.support_status, Some(antecedent::CellStatus::Licensed));
    assert!((result.estimate.ate - truth).abs() < 0.2);
    let posterior = result.posterior.as_ref().unwrap();
    assert_eq!(posterior.draws.n_draws, 512);
    assert!(posterior.summaries.q025[0] < truth && posterior.summaries.q975[0] > truth);
    let bytes = prepared.encode_contracted_result(&result, "iv-joint-stage", &ctx).unwrap();
    let (_, _, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let posterior_bytes = artifact.posterior_artifact.unwrap();
    let (wire, consumed) = antecedent_io::decode_causal_posterior_bytes(&posterior_bytes).unwrap();
    assert_eq!(wire.n_draws, 512);
    assert_eq!(consumed.as_slice(), posterior.draws.values.as_ref());
}

#[test]
fn bayesian_rd_executes_staged_and_checks_bandwidth_conditioning() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/bayesian_rd/expected.json"
    ))
    .unwrap();
    let truth = expected["true_jump"].as_f64().unwrap();
    let bandwidth = expected["fixed_bandwidth"].as_f64().unwrap();
    let n = 400;
    let r: Vec<f64> = (0..n).map(|i| (i as f64 - 200.0) / 100.0).collect();
    let t: Vec<f64> = r.iter().map(|x| f64::from(*x >= 0.0)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.5 * t[i] + 0.8 * r[i] + 1.2 * t[i] * r[i] + ((i % 5) as f64 - 2.0) * 0.01)
        .collect();
    let data = table(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("r", RoleHint::Context, r),
    ]);
    let study = Study::tabular(data.clone())
        .graph(rd_graph())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .identifier(IdentifierId::RdSharp)
        .estimator(EstimatorId::BayesianRdLocalLinear)
        .rd_config(VariableId::from_raw(2), 0.0, bandwidth)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512)))
        .refute(antecedent::RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(9);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!((result.estimate.ate - truth).abs() < 0.08, "{}", result.estimate.ate);
    let post = result.posterior.as_ref().unwrap();
    assert_eq!(post.draws.n_draws, 512);
    assert!(post.diagnostics.notes.iter().any(|n| n.contains("fixed_bandwidth=0.8")));
    let bytes = prepared.encode_contracted_result(&result, "rd-local-stage", &ctx).unwrap();
    let (_, _, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let posterior_bytes = artifact.posterior_artifact.unwrap();
    let (wire, consumed) = antecedent_io::decode_causal_posterior_bytes(&posterior_bytes).unwrap();
    assert_eq!(wire.n_draws, 512);
    assert_eq!(consumed.as_slice(), post.draws.values.as_ref());
}

#[test]
fn rd_posterior_intervals_calibrate_under_declared_local_linear_law() {
    const REPS: usize = 300;
    const DRAWS: usize = 400;
    let mut rd_covered = 0;
    for rep in 0..REPS {
        let mut rng = ExecutionContext::for_tests(4_000 + rep as u64)
            .rng
            .stream_for(StreamDomain::Test, 0x3C4D);
        let n = 300;
        let mut r = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            r[i] = 2.0 * rng.next_f64() - 1.0;
            t[i] = f64::from(r[i] >= 0.0);
            y[i] =
                1.0 + 2.5 * t[i] + 0.8 * r[i] + 1.2 * t[i] * r[i] + 0.5 * standard_normal(&mut rng);
        }
        let rd_data = table(&[
            ("t", RoleHint::TreatmentCandidate, t),
            ("y", RoleHint::OutcomeCandidate, y),
            ("r", RoleHint::Context, r),
        ]);
        let rd_study = Study::tabular(rd_data)
            .graph(rd_graph())
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .identifier(IdentifierId::RdSharp)
            .estimator(EstimatorId::BayesianRdLocalLinear)
            .rd_config(VariableId::from_raw(2), 0.0, 0.8)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
            .refute(antecedent::RefuteSuite::None)
            .build()
            .unwrap();
        if let Ok(result) = rd_study.run(&ExecutionContext::for_tests(5_000 + rep as u64)) {
            let p = result.posterior.unwrap();
            if p.summaries.q025[0] <= 2.5 && 2.5 <= p.summaries.q975[0] {
                rd_covered += 1;
            }
        }
    }
    let rd_rate = rd_covered as f64 / REPS as f64;
    let mcse = (0.95 * 0.05 / REPS as f64).sqrt();
    eprintln!(
        "declared-model 95% posterior interval coverage: RD={rd_rate:.3}; 3MCSE={:.3}",
        3.0 * mcse
    );
    assert!((rd_rate - 0.95).abs() <= 3.0 * mcse, "RD coverage {rd_rate}");
}

#[test]
fn iv_joint_fixed_loading_public_intervals_calibrate_under_declared_law() {
    const REPS: usize = 300;
    const DRAWS: usize = 400;
    let mut covered = 0;
    for rep in 0..REPS {
        let mut rng = ExecutionContext::for_tests(8_000 + rep as u64)
            .rng
            .stream_for(StreamDomain::Test, 0x5A6B);
        let n = 180;
        let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let v = standard_normal(&mut rng);
            t[i] = z[i] + v;
            y[i] = 1.0 + 2.0 * t[i] + v + standard_normal(&mut rng);
        }
        let data = table(&[
            ("t", RoleHint::TreatmentCandidate, t),
            ("y", RoleHint::OutcomeCandidate, y),
            ("z", RoleHint::Context, z),
        ]);
        let mut graph = Dag::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let study = Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::with_levels(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                0.0,
                1.0,
            ))
            .identifier(IdentifierId::Iv)
            .estimator(EstimatorId::BayesianIvJointLinear)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
            .refute(antecedent::RefuteSuite::None)
            .build()
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(9_000 + rep as u64)).unwrap();
        let posterior = result.posterior.unwrap();
        covered +=
            usize::from(posterior.summaries.q025[0] <= 2.0 && 2.0 <= posterior.summaries.q975[0]);
    }
    let rate = covered as f64 / REPS as f64;
    let mcse = (0.95 * 0.05 / REPS as f64).sqrt();
    eprintln!("public joint IV 95% interval coverage: {covered}/{REPS} ({rate:.3})");
    assert!((rate - 0.95).abs() <= 3.0 * mcse, "IV coverage {rate}");
}
