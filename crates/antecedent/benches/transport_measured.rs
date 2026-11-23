//! Native measurement of the transport control: allocator bytes and calls, live-byte peak,
//! peak resident set and the engine's counts across complete runs of the example.
//!
//! Runs the `transport_statistical` example's `run()` unchanged under
//! `antecedent_core::measurement_record::measure`. Writes
//! `$ANTECEDENT_MEASUREMENT_OUT/transport/measurement.json`. `ANTECEDENT_LANE_ROWS` sets the
//! row tier exactly as it does for the example.
#![allow(
    missing_docs,
    dead_code,
    reason = "benchmark crate root: the example's own main and helpers go unused here"
)]

use antecedent_core::ExecutionContext;
use antecedent_kernels::native_probe::{CountingAllocator, measured_pass, set_counting};

#[path = "../../../examples/rust/transport_statistical.rs"]
mod transport;

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator::new();

fn main() {
    set_counting(false);
    let ctx = ExecutionContext::for_tests(7);
    measured_pass("transport", 5, &ctx, transport::run);
}
