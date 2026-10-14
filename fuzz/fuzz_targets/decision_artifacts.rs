//! Fuzz the decision contract and result artifacts; the first 32 bytes seed the expected identity.
#![no_main]

use antecedent_design::decision_artifact::{DecisionContractArtifact, DecisionResultArtifact};
use antecedent_design::decision_robust_artifact::RobustResultArtifact;
use antecedent_design::decision_structural_artifact::StructuralResultArtifact;
use libfuzzer_sys::fuzz_target;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fuzz_target!(|data: &[u8]| {
    let (head, body) = data.split_at(data.len().min(32));
    let identity = hex(head);
    let zero = "0".repeat(64);
    let _ = DecisionContractArtifact::from_bytes(body, &identity);
    let _ = DecisionResultArtifact::from_bytes(body, &identity, &zero);
    let _ = RobustResultArtifact::from_bytes(body, &identity, &[]);
    let _ = StructuralResultArtifact::from_bytes(body, &identity, &[]);
});
