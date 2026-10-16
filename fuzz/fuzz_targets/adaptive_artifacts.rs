//! Fuzz the adaptive inference result and monitoring contract artifacts (metadata and checked decode).
#![no_main]

use antecedent_core::ExecutionContext;
use antecedent_io::adaptive_inference_result_artifact::AdaptiveInferenceResultArtifact;
use antecedent_io_core::adaptive_monitoring_contract_artifact::AdaptiveMonitoringContractArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let ctx = ExecutionContext::production_default(0);
    let _ = AdaptiveInferenceResultArtifact::decode_unchecked(data);
    let _ = AdaptiveInferenceResultArtifact::from_bytes(data, None, &ctx);
    let _ = AdaptiveMonitoringContractArtifact::decode_unchecked(data);
    let _ = AdaptiveMonitoringContractArtifact::from_bytes(data, None);
});
