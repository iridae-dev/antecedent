//! Licensed evidence-to-decision lifecycle for repeated categorical choices.
//!
//! The qualified prior, update, sequential look and study ranking live in
//! [`antecedent_design::choice_evidence`]; the inference family is
//! [`antecedent_stats::adaptive_inference::DirichletPosterior`]. This module is the facade
//! entry through which hosts reach them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_design::choice_evidence::*;
pub use antecedent_stats::adaptive_inference::{DirichletPosterior, MAX_CHOICE_OPTIONS};

/// Run the lifecycle through the stages the request carries: qualified prior, supported
/// observations, licensed inference, sequential look and next-study ranking.
///
/// # Errors
///
/// The refusal of the first stage that cannot be licensed.
pub fn evaluate_lifecycle(
    request: &LifecycleRequest,
    cancelled: &dyn Fn() -> bool,
) -> Result<LifecycleResult, ChoiceRefusal> {
    antecedent_design::choice_evidence::evaluate_lifecycle(request, cancelled)
}
