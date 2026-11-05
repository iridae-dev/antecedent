//! Runs a study from its retained plan after the builder that made it is gone.

use antecedent::StudyResult;
use antecedent::analysis::StudyBuilder;
use antecedent_core::ExecutionContext;

/// Build the study, discard the builder, and run the built study: the result comes from the
/// retained plan, never from a builder that is still alive.
pub fn run_retained(builder: StudyBuilder, label: &str, ctx: &ExecutionContext) -> StudyResult {
    let study = builder.clone().build().unwrap_or_else(|e| panic!("{label}: build: {e}"));
    drop(builder);
    let result = study.run(ctx).unwrap_or_else(|e| panic!("{label}: run: {e}"));
    assert!(!result.logical_plan.plan_id.is_empty(), "{label}: the result names its logical plan");
    result
}
