//! Fuzz source evidence and source projection consumption (nested-artifact readers); the first
//! 32 bytes seed the projection's expected identity.
#![no_main]

use antecedent_design::source_evidence::SourceEvidence;
use antecedent_design::source_projection_artifact::SourceProjectionArtifact;
use libfuzzer_sys::fuzz_target;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fuzz_target!(|data: &[u8]| {
    let (head, body) = data.split_at(data.len().min(32));
    let _ = SourceEvidence::consume(data);
    let _ = SourceEvidence::from_result(data);
    let _ = SourceProjectionArtifact::inspect(data);
    let _ = SourceProjectionArtifact::consume(body, &hex(head));
});
