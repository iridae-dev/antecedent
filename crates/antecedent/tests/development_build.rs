//! A development build runs a `not_supported` route and says what a release build would have
//! refused. Compiled only under `--features development`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "development")]

use antecedent::{InferenceMode, RefuteSuite, Study};
use antecedent_core::support_gate::{BUILD_FLAVOR, WouldRefuse, require_supported};
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
fn the_gate_runs_a_not_supported_route_and_records_what_a_release_build_would_refuse() {
    assert_eq!(BUILD_FLAVOR, "development");
    let ctx = ExecutionContext::for_tests(3);
    assert!(require_supported(&ctx, DETAIL, &[MISSING]).is_ok());
    let expected = WouldRefuse::not_supported(DETAIL, &[MISSING]);
    assert_eq!(ctx.would_refuse(), [expected.clone()]);
    // The log is shared by every clone of the context.
    assert_eq!(ctx.clone().would_refuse(), [expected.clone()]);

    // A supported facade route run with the same context carries the entry on its result.
    let result = supported_route().build().unwrap().run(&ctx).unwrap();
    assert_eq!(result.support_status(), Some(antecedent::CellStatus::Supported));
    assert!(result.would_refuse().contains(&expected), "{:?}", result.would_refuse());
    assert!(result.would_refuse().iter().all(|entry| entry.code == "not_supported"));
}

#[test]
fn a_supported_route_lists_nothing() {
    let ctx = ExecutionContext::for_tests(3);
    let result = supported_route().build().unwrap().run(&ctx).unwrap();
    assert_eq!(result.support_status(), Some(antecedent::CellStatus::Supported));
    assert!(result.would_refuse().is_empty());
}
