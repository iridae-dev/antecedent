//! Fuzz the external claim and column claim artifacts. The first byte is the length of a CBOR
//! identity the consumer retained; the remainder is the artifact.
#![no_main]

use antecedent_io_core::external_claim_artifact::{ExternalClaimArtifact, ExternalClaimIdentity};
use antecedent_io_core::external_column_claim_artifact::{
    ExternalColumnClaimArtifact, ExternalColumnIdentity,
};
use antecedent_io_core::from_cbor;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&len, rest)) = data.split_first() else {
        return;
    };
    let (identity, body) = rest.split_at(rest.len().min(usize::from(len)));
    if let Ok(expected) = from_cbor::<ExternalClaimIdentity>(identity) {
        let _ = ExternalClaimArtifact::from_bytes(body, &expected);
    }
    if let Ok(expected) = from_cbor::<ExternalColumnIdentity>(identity) {
        let _ = ExternalColumnClaimArtifact::from_bytes(body, &expected);
    }
});
