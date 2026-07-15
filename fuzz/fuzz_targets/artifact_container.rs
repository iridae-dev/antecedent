//! Fuzz artifact container deserialization (length-cap / no panic).
#![no_main]

use causal_io::EncodedArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = EncodedArtifact::read_from(data);
});
