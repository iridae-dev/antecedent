//! Fuzz the external callback artifact and provider envelope readers (declared inspect paths).
#![no_main]

use antecedent_io_core::external_callback_artifact::ExternalCallbackArtifact;
use antecedent_io_core::provider_envelope::{inspect_provider_envelope, open_provider_envelope};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ExternalCallbackArtifact::inspect(data);
    let _ = inspect_provider_envelope(data);
    let _ = open_provider_envelope(data);
});
