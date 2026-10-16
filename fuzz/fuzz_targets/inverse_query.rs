//! Fuzz the inverse query artifact: container parts, then full decode with no expected identity.
#![no_main]

use antecedent_design::inverse_query_artifact::{InverseQueryArtifact, decode_parts};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = decode_parts(data);
    let _ = InverseQueryArtifact::from_bytes(data, None);
});
