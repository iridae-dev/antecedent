//! Runs every `examples/rust` example as a test.
//!
//! Each example keeps a `pub fn run()` beside its `fn main()`; the test includes the file
//! and calls it, so an example that panics, or whose own assertion against the
//! simulation's truth fails, fails the suite instead of surviving until a reader runs it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

#[allow(dead_code)]
#[path = "../../../examples/rust/ate_quickstart.rs"]
mod ate_quickstart;

#[allow(dead_code)]
#[path = "../../../examples/rust/causal_state_workflow.rs"]
mod causal_state_workflow;

#[allow(dead_code)]
#[path = "../../../examples/rust/class_preserving_cpdag.rs"]
mod class_preserving_cpdag;

#[allow(dead_code)]
#[path = "../../../examples/rust/discover_then_estimate.rs"]
mod discover_then_estimate;

#[allow(dead_code)]
#[path = "../../../examples/rust/gcm_do.rs"]
mod gcm_do;

#[allow(dead_code)]
#[path = "../../../examples/rust/identify_only.rs"]
mod identify_only;

#[allow(dead_code)]
#[path = "../../../examples/rust/manufacturing_temporal.rs"]
mod manufacturing_temporal;

#[allow(dead_code)]
#[path = "../../../examples/rust/prior_bank_surveys.rs"]
mod prior_bank_surveys;

#[allow(dead_code)]
#[path = "../../../examples/rust/propensity_weighting.rs"]
mod propensity_weighting;

#[allow(dead_code)]
#[path = "../../../examples/rust/rank_designs.rs"]
mod rank_designs;

#[allow(dead_code)]
#[path = "../../../examples/rust/sales_spreadsheet_e2e.rs"]
mod sales_spreadsheet_e2e;

#[allow(dead_code)]
#[path = "../../../examples/rust/sequential_bayes.rs"]
mod sequential_bayes;

#[allow(dead_code)]
#[path = "../../../examples/rust/staged_static_kinds.rs"]
mod staged_static_kinds;

#[allow(dead_code)]
#[path = "../../../examples/rust/temporal_response_curve.rs"]
mod temporal_response_curve;

#[allow(dead_code)]
#[path = "../../../examples/rust/transport_exact.rs"]
mod transport_exact;

#[allow(dead_code)]
#[path = "../../../examples/rust/transport_meta_grid.rs"]
mod transport_meta_grid;

#[allow(dead_code)]
#[path = "../../../examples/rust/transport_statistical.rs"]
mod transport_statistical;

#[test]
fn example_ate_quickstart_runs() {
    ate_quickstart::run().unwrap();
}

#[test]
fn example_causal_state_workflow_runs() {
    causal_state_workflow::run().unwrap();
}

#[test]
fn example_class_preserving_cpdag_runs() {
    class_preserving_cpdag::run().unwrap();
}

#[test]
fn example_discover_then_estimate_runs() {
    discover_then_estimate::run().unwrap();
}

#[test]
fn example_gcm_do_runs() {
    gcm_do::run().unwrap();
}

#[test]
fn example_identify_only_runs() {
    identify_only::run().unwrap();
}

#[test]
fn example_manufacturing_temporal_runs() {
    manufacturing_temporal::run().unwrap();
}

#[test]
fn example_prior_bank_surveys_runs() {
    prior_bank_surveys::run().unwrap();
}

#[test]
fn example_propensity_weighting_runs() {
    propensity_weighting::run().unwrap();
}

#[test]
fn example_rank_designs_runs() {
    rank_designs::run().unwrap();
}

#[test]
fn example_sales_spreadsheet_e2e_runs() {
    sales_spreadsheet_e2e::run().unwrap();
}

#[test]
fn example_sequential_bayes_runs() {
    sequential_bayes::run().unwrap();
}

#[test]
fn example_staged_static_kinds_runs() {
    staged_static_kinds::run().unwrap();
}

#[test]
fn example_temporal_response_curve_runs() {
    temporal_response_curve::run().unwrap();
}

#[test]
fn example_transport_exact_runs() {
    transport_exact::run().unwrap();
}

#[test]
fn example_transport_meta_grid_runs() {
    transport_meta_grid::run().unwrap();
}

#[test]
fn example_transport_statistical_runs() {
    transport_statistical::run().unwrap();
}
