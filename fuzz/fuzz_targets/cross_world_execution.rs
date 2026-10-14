//! Fuzz the cross-world execution artifact. The first byte is the length of a CBOR identity the
//! consumer retained (an undecodable one is skipped); the remainder is the artifact.
#![no_main]

use antecedent_core::ExecutionContext;
use antecedent_io_core::cross_world_execution_artifact::{
    CrossWorldExecutionArtifact, CrossWorldExecutionIdentity,
};
use antecedent_io_core::from_cbor;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let ctx = ExecutionContext::production_default(0);
    let _ = CrossWorldExecutionArtifact::from_bytes(data, None, &ctx);
    let Some((&len, rest)) = data.split_first() else {
        return;
    };
    let (identity, body) = rest.split_at(rest.len().min(usize::from(len)));
    if let Ok(expected) = from_cbor::<CrossWorldExecutionIdentity>(identity) {
        let _ = CrossWorldExecutionArtifact::from_bytes(body, Some(&expected), &ctx);
    }
});
