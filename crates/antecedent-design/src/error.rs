//! Design-crate errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Errors from experiment / measurement design evaluation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DesignError {
    /// Empty candidate list.
    #[error("no candidate designs to rank")]
    EmptyCandidates,
    /// Empty graph / model posterior draws.
    #[error("empty posterior ensemble for design evaluation")]
    EmptyPosterior,
    /// Shape / length mismatch in inputs.
    #[error("design shape error: {0}")]
    Shape(String),
    /// Invalid configuration (budget, threshold, etc.).
    #[error("invalid design config: {0}")]
    Config(String),
    /// Numerical failure (singular Gram, non-finite score).
    #[error("design numerical failure: {0}")]
    Numerical(String),
    /// Probability / stats layer failure.
    #[error("probability error: {0}")]
    Prob(String),
    /// No action of a decision problem meets its chance threshold, so there is no
    /// decision to value.
    #[error("no admissible action: {0}")]
    NoAdmissibleAction(String),
    /// In-process utility / constraint callback failed.
    #[error("callback {name}: {message}")]
    Callback {
        /// Callback name (`utility` or a constraint name).
        name: String,
        /// Failure detail.
        message: String,
    },
    /// A foreign utility / constraint callback raised. The failure is structured data, so a
    /// host binding finds the exception it retained from the error chain; the text is the
    /// same as the [`Self::Callback`] it replaces.
    #[error("callback {name}: {failure}")]
    CallbackFailed {
        /// Callback name (`utility` or a constraint name).
        name: String,
        /// The structured failure.
        #[source]
        failure: antecedent_core::CallbackFailure,
    },
    /// The caller cancelled the evaluation. A partial ranking is not a result.
    #[error("design evaluation cancelled")]
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::DesignError;
    use antecedent_core::CallbackFailure;

    #[test]
    fn a_callback_failure_round_trips_through_the_design_error() {
        let failure = CallbackFailure::new(Some(5), "utility failed: boom", None);
        let error =
            DesignError::CallbackFailed { name: "utility".into(), failure: failure.clone() };
        assert_eq!(error.to_string(), "callback utility: utility failed: boom");
        assert_eq!(CallbackFailure::in_chain(&error), Some(&failure));
        assert!(CallbackFailure::in_chain(&DesignError::Cancelled).is_none());
    }
}
