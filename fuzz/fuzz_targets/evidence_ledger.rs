//! Fuzz the evidence ledger export artifact decode (bounds, source graph, identity re-derivation).
#![no_main]

use antecedent_io_core::evidence_ledger_artifact::EvidenceLedgerArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = EvidenceLedgerArtifact::from_bytes(data, None);
});
