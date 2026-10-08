//! Portable CATE prediction: executed fit counts, SCM anchors and process isolation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::DrLearner;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_learn::fit_counts::count_resolved_fits;

fn fixture(effect: f64) -> (TabularData, Dag) {
    let n = 1600_u32;
    let mut rng = antecedent_core::CausalRng::from_seed(19);
    let mut z = Vec::new();
    let mut t = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let x = f64::from(i % 21) / 10.0 - 1.0;
        let a = f64::from(rng.next_f64() < 0.5);
        z.push(x);
        t.push(a);
        y.push(a * (effect + 0.4 * x) + x);
    }
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    (data, graph)
}

fn fitted_bytes(effect: f64) -> Vec<u8> {
    let (data, graph) = fixture(effect);
    let ctx = ExecutionContext::for_tests(3);
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2)))
        .estimator(DrLearner::new())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    prepared.encode_contracted_result(&result, "portable-cate", &ctx).unwrap()
}

fn fit_bytes(effect: f64) -> (Vec<u8>, u64) {
    count_resolved_fits(|| fitted_bytes(effect))
}

fn predict_bytes(bytes: &[u8]) -> (Vec<f64>, String) {
    let consumed = antecedent_io::consume_analysis_result(bytes).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
    assert!(consumed.acceptance.verified_references);
    let claim = consumed.contract.as_ref().unwrap().claim.as_ref().unwrap();
    let parent = antecedent_io::digest_hex(&claim.claim_id);
    let model = consumed.body.fitted_effect.unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let values = model.predict(&[VariableId::from_raw(0)], &[&[-1.0, 0.0, 1.0]], 3, &ctx).unwrap();
    // Independent SCM truth E[Y(1)-Y(0)|z] = 2 + 0.4 z.
    for (value, truth) in values.iter().zip([1.6, 2.0, 2.4]) {
        assert!((value - truth).abs() < 0.12, "CATE SCM anchor: {value} vs {truth}");
    }
    assert!(model.predict(&[VariableId::from_raw(1)], &[&[0.0]], 1, &ctx).is_err());
    assert!(model.predict(&[VariableId::from_raw(0)], &[&[f64::NAN]], 1, &ctx).is_err());
    (values, parent)
}

#[test]
fn c2_portable_predictor_zero_refits_and_fresh_process_match_full_run() {
    let (bytes, fits) = fit_bytes(2.0);
    assert!(fits > 1, "resolved nuisance and final-stage models actually fit");
    assert_eq!(bytes, fitted_bytes(2.0), "instrumentation preserves the complete claim bytes");
    let (one, resume_fits) = count_resolved_fits(|| predict_bytes(&bytes));
    assert_eq!(resume_fits, 0);
    let (rerun, rerun_fits) = fit_bytes(2.0);
    assert_eq!(rerun_fits, fits);
    assert_eq!(one, predict_bytes(&rerun));
    let directory =
        std::env::temp_dir().join(format!("antecedent-portable-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("model.ant");
    std::fs::write(&path, &bytes).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "c2_portable_predictor_child", "--nocapture"])
        .env("ANTECEDENT_PORTABLE_TEST_ARTIFACT", &path)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // Refit requires compatible supplied raw data; changing outcomes cannot reuse this parent.
    let (changed, changed_fits) = fit_bytes(3.0);
    assert!(changed_fits > 1);
    let changed = antecedent_io::consume_analysis_result(&changed).unwrap();
    let changed_parent =
        antecedent_io::digest_hex(&changed.contract.unwrap().claim.unwrap().claim_id);
    assert_ne!(changed_parent, one.1);
}

#[test]
fn c2_portable_predictor_child() {
    let Some(path) = std::env::var_os("ANTECEDENT_PORTABLE_TEST_ARTIFACT") else { return };
    let bytes = std::fs::read(path).unwrap();
    let (_, fits) = count_resolved_fits(|| predict_bytes(&bytes));
    assert_eq!(fits, 0);
}
