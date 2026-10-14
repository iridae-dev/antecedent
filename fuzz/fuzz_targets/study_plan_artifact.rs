//! Fuzz the study plan artifact: CBOR decode of the wire, then replay under the default limits.
#![no_main]

use antecedent_core::ExecutionContext;
use antecedent_design::{StudyPlanArtifactWire, StudyPlanConsumeLimits};
use antecedent_io_core::from_cbor;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let ctx = ExecutionContext::production_default(0);
    if let Ok(wire) = from_cbor::<StudyPlanArtifactWire>(data) {
        let _ = wire.consume_with_limits(StudyPlanConsumeLimits::default(), &ctx);
    }
});
