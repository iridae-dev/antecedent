//! Fuzz the repair report artifact: bounded decode, then replay under the default limits.
#![no_main]

use antecedent_core::ExecutionContext;
use antecedent_design::{RepairConsumeLimits, RepairReportArtifact};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let ctx = ExecutionContext::production_default(0);
    if let Ok(artifact) = RepairReportArtifact::from_bytes(data) {
        let _ = artifact.consume(RepairConsumeLimits::default(), &ctx);
    }
});
