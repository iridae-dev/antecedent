//! Identification errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;
use antecedent_graph::GraphError;
use thiserror::Error;

/// Identification failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum IdentificationError {
    /// Treatment/outcome missing from graph.
    #[error("unknown variable {id}")]
    UnknownVariable {
        /// Variable.
        id: VariableId,
    },
    /// Query type not supported.
    #[error("unsupported query: {message}")]
    UnsupportedQuery {
        /// Explanation.
        message: &'static str,
    },
    /// Temporal backdoor is Pulse-only when used as single-node backdoor; Sustained
    /// is handled by sequential / g-formula ID on the unfolded graph.
    #[error(
        "temporal backdoor identification supports Pulse policies only; \
     sustained interventions require sequential (g-formula) identification"
    )]
    SustainedPolicyUnsupported,
    /// Identification could not be certified (e.g. a temporal history-cap truncation cut the
    /// search short before it could prove or disprove identifiability). This is distinct from
    /// [`antecedent_core::IdentificationStatus::NotIdentified`], which is an `Ok` status meaning
    /// identification *proved* non-identifiability (e.g. via a hedge); this variant means the
    /// algorithm could not tell either way. Named `NotCertified` (not `NotIdentified`) precisely
    /// to avoid colliding with that status name.
    #[error("not certified: {message}")]
    NotCertified {
        /// Explanation.
        message: &'static str,
    },
    /// Result limit exceeded during enumeration.
    #[error("adjustment enumeration exceeded limit {limit}")]
    ResultLimitExceeded {
        /// Configured limit.
        limit: usize,
    },
    /// Graph error.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Index / configuration message that is not a raw [`GraphError`].
    #[error("{0}")]
    Message(String),
}

impl IdentificationError {
    /// Ad-hoc message helper.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    /// Fixed unsupported query.
    #[must_use]
    pub const fn unsupported(message: &'static str) -> Self {
        Self::UnsupportedQuery { message }
    }
}
