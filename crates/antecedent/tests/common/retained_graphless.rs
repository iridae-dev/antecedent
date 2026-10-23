//! Retained-program assertions shared by the graphless design-route targets.
//!
//! A graphless design route executes through the same compiled path as a
//! matrix-supported route: the prepared handle retains a sealed operation, its plan
//! view reads the lowered program the handle executes, and the executed result's
//! contract is accepted as a verified program by the artifact consumer.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::planner::PlanStage;
use antecedent::{PreparedStudy, StudyResult};
use antecedent_core::ExecutionContext;

/// Assert that `prepared` and `result` came from a retained compiled program and that its
/// claim survives the contracted artifact path.
pub fn assert_retained_graphless_program(
    label: &str,
    prepared: &PreparedStudy,
    result: &StudyResult,
    ctx: &ExecutionContext,
) {
    let operation = prepared.retained_operation();
    assert!(
        !matches!(operation, RetainedOperation::OrdinaryDispatch),
        "{label}: a graphless route must retain a sealed operation, not ordinary dispatch",
    );
    let view = prepared.plan_view().unwrap_or_else(|error| panic!("{label}: {error}"));
    assert_eq!(view.stage(), PlanStage::Prepared, "{label}");
    assert_eq!(view.operation(), Some(operation.name()), "{label}: view names the operation");
    assert_eq!(
        Some(view.identity()),
        prepared.plan().lowered_program_identity(),
        "{label}: the view reads the program the handle executes",
    );
    assert_eq!(view.support().status(), prepared.support_status(), "{label}: support status");
    let bytes = prepared
        .encode_contracted_result(result, label, ctx)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    // The independent consumer recognizes the program and its claim. What it cannot replay from
    // the artifact alone (the fitted operation, the posterior draws) is named as a precise
    // dependency, never left as an opaque unresolved reference.
    let acceptance = &consumed.acceptance;
    assert!(
        acceptance.recognized
            && acceptance.claim_present
            && acceptance.unresolved.iter().all(|reason| reason.starts_with("dependencies.")),
        "{label}: the executed claim must be recognized with only named dependencies \
         (recognized {}, claim {}, unresolved: {:?})",
        acceptance.recognized,
        acceptance.claim_present,
        acceptance.unresolved,
    );
}
