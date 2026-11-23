use super::mcmc_summary_parallel;
use antecedent_core::{ExecutionContext, execution::AdmissionError};

#[test]
fn summary_propagates_cancellation_instead_of_falling_back() {
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    let result = mcmc_summary_parallel(&[1.0; 16], 2, 8, 1, &ctx);
    assert!(matches!(result, Err(AdmissionError::Cancelled)));
}
