//! incremental causal-state facade helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::CacheBudget;
use causal_state::{CausalState, StateError, StateEvent};

use crate::error::AnalysisError;

/// Construct a fresh [`CausalState`] with the given cache budget.
#[must_use]
pub fn new_causal_state(budget: CacheBudget) -> CausalState {
    CausalState::new(budget)
}

/// Apply a state event without auto-rerunning analyses.
///
/// # Errors
///
/// Propagates state update failures.
pub fn apply_state_event(
    state: &mut CausalState,
    event: StateEvent,
) -> Result<causal_core::StateVersion, AnalysisError> {
    state.apply(event).map_err(map_state)
}

fn map_state(err: StateError) -> AnalysisError {
    match err {
        StateError::CacheBudget { need, remaining } => AnalysisError::Resource {
            message: format!("cache budget exceeded (need {need}, remaining {remaining})"),
        },
        other => AnalysisError::Compile { message: other.to_string() },
    }
}
