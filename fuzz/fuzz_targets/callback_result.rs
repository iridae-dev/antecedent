//! Fuzz the callback result artifact (declaration, recorded output, replay contract).
#![no_main]

use antecedent_io_core::callback_result_artifact::CallbackResultArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = CallbackResultArtifact::from_bytes(data, None);
});
