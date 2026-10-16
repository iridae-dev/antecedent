//! Fuzz the sequential evaluation artifact: bounded decode, then replay without an expectation.
#![no_main]

use antecedent_core::ExecutionContext;
use antecedent_design::sequential_evaluation_artifact::{
    SequentialEvaluationArtifact, consume,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let ctx = ExecutionContext::production_default(0);
    let _ = SequentialEvaluationArtifact::from_bytes(data);
    let _ = consume(data, None, &ctx);
});
