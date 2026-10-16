//! Fuzz the interface contract artifact decode (declaration schema, bounds, identity).
#![no_main]

use antecedent_io_core::interface_contract_artifact::InterfaceContractArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = InterfaceContractArtifact::from_bytes(data, None);
});
