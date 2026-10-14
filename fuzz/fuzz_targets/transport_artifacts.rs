//! Fuzz the estimator-bound transport artifacts: joint Bayesian, ADMG conditional (and its
//! obstruction), mz and z transport. Each decodes, then replays under the default limits.
#![no_main]

use antecedent_core::ExecutionContext;
use antecedent_io::admg_conditional_transport_artifact::{
    AdmgConditionalArtifactWire, AdmgConditionalConsumeLimits, AdmgConditionalObstructionWire,
};
use antecedent_io::joint_bayesian_transport_artifact::{
    JointBayesianArtifactWire, JointBayesianConsumeLimits,
};
use antecedent_io::mz_transport_artifact::{MzTransportArtifactWire, MzTransportConsumeLimits};
use antecedent_io::z_transport_artifact::ZTransportArtifactWire;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let ctx = ExecutionContext::production_default(0);
    let _ = JointBayesianArtifactWire::decode(data);
    let _ = JointBayesianArtifactWire::consume_with_limits(
        data,
        JointBayesianConsumeLimits::default(),
        &ctx,
    );
    let _ = AdmgConditionalArtifactWire::decode(data);
    let _ = AdmgConditionalArtifactWire::consume_with_limits(
        data,
        AdmgConditionalConsumeLimits::default(),
        &ctx,
    );
    let _ = AdmgConditionalObstructionWire::decode(data);
    let _ = AdmgConditionalObstructionWire::consume(data, &ctx);
    let _ = MzTransportArtifactWire::decode(data);
    let _ = MzTransportArtifactWire::consume_with_limits(
        data,
        MzTransportConsumeLimits::default(),
        &ctx,
    );
    let _ = ZTransportArtifactWire::decode(data);
    let _ = ZTransportArtifactWire::consume(data, &ctx);
});
