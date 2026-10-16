//! Fuzz analysis-result contract section decode.
#![no_main]

use antecedent_io::{
    AnalysisResultContractWire, EncodedArtifact, decode_analysis_result_contract, from_cbor,
    validate_contract_section,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(artifact) = EncodedArtifact::read_from(data) {
        if let Ok(Some(contract)) = decode_analysis_result_contract(&artifact) {
            let _ = validate_contract_section(&contract);
        }
    }
    if let Ok(contract) = from_cbor::<AnalysisResultContractWire>(data) {
        let _ = validate_contract_section(&contract);
    }
});
