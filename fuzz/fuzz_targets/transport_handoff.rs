//! Fuzz the transport handoff artifact: verified decode and metadata-only inspect.
#![no_main]

use antecedent_io_core::transport_handoff_artifact::TransportHandoffArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = TransportHandoffArtifact::inspect(data);
    let _ = TransportHandoffArtifact::from_bytes(data, None);
});
