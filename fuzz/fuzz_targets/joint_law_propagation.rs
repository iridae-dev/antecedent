//! Fuzz the joint-law propagation artifact: wire decode (version gate, bounds, values), then
//! replay of whatever decodes so the consume cost is exercised.
#![no_main]

use antecedent_core::ExecutionContext;
use antecedent_io_core::joint_law_propagation_artifact::JointLawPropagationArtifactWire;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(wire) = JointLawPropagationArtifactWire::decode(data) {
        let ctx = ExecutionContext::production_default(0);
        let _ = wire.consume(None, &ctx);
    }
});
