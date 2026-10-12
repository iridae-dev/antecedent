//! Fuzz the causal payload decoder.
//!
//! One call dispatches the payload kinds (query, static result, response,
//! transport identification, transport estimate, interference). The per-artifact
//! `decode` methods are separate public functions and are not behind one dispatcher.
#![no_main]

use antecedent_io::decode_causal_payload_artifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = decode_causal_payload_artifact(data);
});
