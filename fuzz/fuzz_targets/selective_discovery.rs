//! Fuzz the selective discovery artifact (inspect and consume); the first 32 bytes seed the identity.
#![no_main]

use antecedent_io_core::selective_discovery_artifact::SelectiveDiscoveryArtifact;
use libfuzzer_sys::fuzz_target;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fuzz_target!(|data: &[u8]| {
    let (head, body) = data.split_at(data.len().min(32));
    let _ = SelectiveDiscoveryArtifact::inspect(data);
    let _ = SelectiveDiscoveryArtifact::from_bytes(data, None);
    let _ = SelectiveDiscoveryArtifact::from_bytes(body, Some(&hex(head)));
});
