//! Fuzz the measurement model artifact; the first 32 bytes seed the expected model identity.
#![no_main]

use antecedent_io_core::measurement_model_wire::MeasurementModelArtifact;
use libfuzzer_sys::fuzz_target;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fuzz_target!(|data: &[u8]| {
    let (head, body) = data.split_at(data.len().min(32));
    let _ = MeasurementModelArtifact::from_bytes(body, &hex(head));
    let _ = MeasurementModelArtifact::from_bytes(data, &"0".repeat(64));
});
