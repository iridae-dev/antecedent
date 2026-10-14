//! Fuzz the cohort decision source artifact; the first 32 bytes seed the expected identity.
#![no_main]

use antecedent_io_core::cohort_decision::{CohortDecisionSource, CohortExpectation};
use libfuzzer_sys::fuzz_target;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fuzz_target!(|data: &[u8]| {
    let (head, body) = data.split_at(data.len().min(32));
    let expectation = CohortExpectation::for_identity(hex(head));
    let _ = CohortDecisionSource::consume(body, &expectation);
    let _ = CohortDecisionSource::consume(data, &CohortExpectation::default());
});
