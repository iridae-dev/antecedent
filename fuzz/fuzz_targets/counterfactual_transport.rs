//! Fuzz the counterfactual transport artifacts: wire decode, transported-counterfactual consume, sampled transport decode.
#![no_main]

use antecedent_io::sampled_counterfactual_transport_artifact::SampledCounterfactualTransportArtifactWire;
use antecedent_io_core::counterfactual_transport_wire::CounterfactualTransportArtifactWire;
use antecedent_io_core::transported_counterfactual_artifact::TransportedCounterfactualArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = CounterfactualTransportArtifactWire::decode(data);
    let _ = TransportedCounterfactualArtifact::from_bytes(data, None);
    let _ = SampledCounterfactualTransportArtifactWire::decode(data);
});
