//! A release build refuses a `not_supported` route with the reason code `not_supported`.
//! Compiled only without `--features development`; `development_build.rs` runs the same cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(not(feature = "development"))]

use antecedent::{InferenceMode, RefuteSuite, Study};
use antecedent_core::support_gate::{BUILD_FLAVOR, require_supported};
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

/// Columns `z, t, y` from a deterministic stream: `y = 1 + 2 t + z + noise`.
fn linear_table(n: usize) -> TabularData {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    let z: Vec<f64> = (0..n).map(|_| uniform() - 0.5).collect();
    let t: Vec<f64> = z.iter().map(|z| f64::from(0.6 * z + uniform() - 0.5 > 0.0)).collect();
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + z + 0.1 * (uniform() - 0.5)).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

/// `z -> t`, `z -> y`, `t -> y`.
fn linear_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

/// A synthetic route the support gate is asked about; no facade route is unsupported today.
const DETAIL: &str = "synthetic route without a supported evidence record";
const MISSING: &str = "estimator_evidence:synthetic.estimator";

/// The default frequentist average effect on the confounded DAG: a supported route that runs.
fn supported_route() -> antecedent::StudyBuilder {
    Study::tabular(linear_table(300))
        .graph(linear_dag())
        .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        )))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

#[test]
fn the_gate_refuses_a_not_supported_route_with_its_detail_and_missing_evidence() {
    assert_eq!(BUILD_FLAVOR, "release");
    let ctx = ExecutionContext::for_tests(3);
    let refusal = require_supported(&ctx, DETAIL, &[MISSING]).unwrap_err();
    assert_eq!(refusal.code(), "not_supported");
    assert_eq!(refusal.detail, DETAIL);
    assert_eq!(refusal.missing, [MISSING]);
    let message = refusal.to_string();
    assert!(message.starts_with("reason=not_supported: "), "{message}");
    assert!(message.contains(DETAIL) && message.contains(MISSING), "{message}");
    // A release build records nothing.
    assert!(ctx.would_refuse().is_empty());
}

#[test]
fn a_supported_route_builds_and_runs() {
    let ctx = ExecutionContext::for_tests(3);
    let result = supported_route().build().unwrap().run(&ctx).unwrap();
    assert_eq!(result.support_status(), Some(antecedent::CellStatus::Supported));
    assert!(result.would_refuse().is_empty());
}
