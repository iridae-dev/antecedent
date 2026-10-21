//! Writes the flagship decision artifact that `antecedent-io` consumes without
//! this crate.
//!
//! The decision combines the cluster margin, the temporal effect, and the cost
//! contrast under a declared cross-covariance (correlations 0.3, -0.2, 0.1 of
//! the routes' published standard deviations). The consumer in
//! `antecedent-io/tests/flagship_decision_artifact.rs` reads only the committed
//! bytes, so a missing file fails there until this test is run:
//! `ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent --test
//! flagship_decision_fixture -- --ignored regenerate_rust_fixtures`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::{FlagshipDependence, FlagshipWorkflow};
use antecedent_core::ExecutionContext;
use antecedent_test_support::conformance_dir;

/// Writes `rust_flagship_decision.bin` and its identity for the io consumer.
#[test]
#[ignore = "writes conformance/cross_surface; set ANTECEDENT_WRITE_FIXTURES=1"]
fn regenerate_rust_fixtures() {
    assert_eq!(
        std::env::var("ANTECEDENT_WRITE_FIXTURES").as_deref(),
        Ok("1"),
        "set ANTECEDENT_WRITE_FIXTURES=1 to write fixtures"
    );
    let ctx = ExecutionContext::for_tests(0xF1A6);
    let base = FlagshipWorkflow::execute(&ctx, FlagshipDependence::Independent).expect("base");
    let decision = base.decision().expect("current decision");
    let (margin, temporal, cost) =
        (decision.margin.variance, decision.temporal_variance, decision.cost_variance);
    let dependence = FlagshipDependence::Covariance {
        id: "flagship-cross".to_owned(),
        margin_temporal: 0.3 * (margin * temporal).sqrt(),
        margin_cost: -0.2 * (margin * cost).sqrt(),
        temporal_cost: 0.1 * (temporal * cost).sqrt(),
    };
    let workflow = FlagshipWorkflow::execute(&ctx, dependence).expect("declared workflow");
    let artifact = workflow.export_historical_replay().expect("export");
    let dir = conformance_dir("cross_surface");
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join("rust_flagship_decision.bin"), &artifact.bytes).expect("write bytes");
    let identity = serde_json::json!({ "identity": artifact.identity });
    std::fs::write(
        dir.join("rust_flagship_decision.identity.json"),
        serde_json::to_vec_pretty(&identity).expect("encode identity"),
    )
    .expect("write identity");
}
