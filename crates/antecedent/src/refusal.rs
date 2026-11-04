//! Registered scientific refusals a host raises itself: typed constructors that carry the
//! reason code, structured detail and message the library would have produced.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::EstimationError;

/// The registered refusal when the context is cancelled, checked at `stage`.
///
/// # Errors
/// `transport_budget_cancel` once the cancellation token has fired.
pub fn ensure_not_cancelled(
    ctx: &antecedent_core::ExecutionContext,
    stage: &str,
) -> Result<(), EstimationError> {
    antecedent_estimate::refuse_cancelled(ctx, stage)
}

/// The registered refusal for an exact-evaluation failure of a transport program.
#[must_use]
pub fn evaluation_failure(error: &antecedent_expr::EvalError) -> EstimationError {
    antecedent_estimate::refuse_eval(error)
}

/// The refusal for an interval or joint covariance over a penalized joint-cell family.
#[must_use]
pub fn joint_interval_unsupported() -> EstimationError {
    antecedent_estimate::refuse_joint_inference()
}

/// The refusal for an interval or joint covariance over learner-supplied joint cells.
#[must_use]
pub fn joint_learner_interval_unsupported() -> EstimationError {
    antecedent_estimate::refuse_joint_learner_inference()
}

/// The refusal of any attribution of the raw-versus-adjusted gap to adjustment columns.
#[must_use]
pub fn column_attribution_unsupported() -> EstimationError {
    antecedent_estimate::refuse_column_attribution()
}

/// The refusal of a cluster or dependence `option` declared on a flexible-learner estimator.
#[must_use]
pub fn flexible_learner_dependence(estimator_id: &str, option: &str) -> EstimationError {
    antecedent_estimate::flexible_learner_dependence_refusal(estimator_id, option)
}

/// The refusal for a declaration that is not a valid set of sampling weights.
#[must_use]
pub fn sampling_weights_invalid(message: &str) -> EstimationError {
    antecedent_estimate::sampling_weights::invalid(message)
}

/// The refusal for a route that has no construction with sampling weights.
#[must_use]
pub fn sampling_weights_unsupported_route(message: &str) -> EstimationError {
    antecedent_estimate::sampling_weights::unsupported_route(message)
}
