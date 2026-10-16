//! Fuzz the rollout artifact. The first byte is the length of a CBOR expectation the consumer
//! retained (an undecodable one is skipped); the remainder is the artifact.
#![no_main]

use antecedent_design::rollout_artifact::{RolloutArtifact, RolloutExpectation, consume};
use antecedent_io_core::from_cbor;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = RolloutArtifact::from_bytes(data);
    let Some((&len, rest)) = data.split_first() else {
        return;
    };
    let (expectation, body) = rest.split_at(rest.len().min(usize::from(len)));
    if let Ok(expected) = from_cbor::<RolloutExpectation>(expectation) {
        let _ = consume(body, &expected);
    }
});
