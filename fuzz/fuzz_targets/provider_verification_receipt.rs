//! Fuzz the provider verification receipt reader; the first 32 bytes seed the retained digest.
#![no_main]

use antecedent_io_core::from_cbor;
use antecedent_io_core::provider_verification_artifact::{
    ProviderVerificationReceiptWire, receipt_from_bytes,
};
use libfuzzer_sys::fuzz_target;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fuzz_target!(|data: &[u8]| {
    let (head, body) = data.split_at(data.len().min(32));
    let _ = receipt_from_bytes(body, &hex(head));
    let _ = receipt_from_bytes(data, &"0".repeat(64));
    if let Ok(wire) = from_cbor::<ProviderVerificationReceiptWire>(data) {
        let _ = wire.into_receipt();
    }
});
