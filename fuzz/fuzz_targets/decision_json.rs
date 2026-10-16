//! Fuzz the decision contract JSON declaration entry points (length cap, unknown fields,
//! stored identity ignored).
#![no_main]

use antecedent_design::decision_artifact::{contract_from_json, contract_from_json_refusal};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(json) = std::str::from_utf8(data) else {
        return;
    };
    let _ = contract_from_json(json);
    let _ = contract_from_json_refusal(json);
});
