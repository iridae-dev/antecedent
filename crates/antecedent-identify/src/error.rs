//! Identification errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{TransportOutcomeKind, VariableId};
use antecedent_graph::GraphError;
use thiserror::Error;

/// Which budget an identification search or catalog binding exhausted.
///
/// Exhaustion is never an impossibility claim; each kind keeps the stable code
/// callers already read from the message.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum IdentificationBudget {
    /// The step or depth limit of the transport identification recursion.
    Steps,
    /// The execution memory budget charged per identification step.
    Memory,
    /// The operation budget of catalog binding.
    Binding,
    /// The memory budget of catalog binding or meta-source checking.
    BindingMemory,
    /// The bounded z-transport search.
    ZTransport,
}

impl IdentificationBudget {
    /// Stable code, also the error's display.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Steps => "transport.identification_budget",
            Self::Memory => "transport.identification_memory_budget",
            Self::Binding => "transport.binding_budget",
            Self::BindingMemory => "transport.memory_budget",
            Self::ZTransport => "z_transport.exhausted_computation",
        }
    }
}

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
    /// The query is malformed (as opposed to well-formed but unsupported).
    #[error("invalid query: {message}")]
    InvalidQuery {
        /// What the query's own validation reported.
        message: String,
    },
    /// An internal invariant of an identification algorithm did not hold. This is a defect
    /// in the library, never a statement about identifiability.
    #[error("identification invariant violated: {message}")]
    InvariantViolated {
        /// Which invariant.
        message: &'static str,
    },
    /// Graph error.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Cooperative cancellation observed during identification or binding.
    #[error("transport.cancelled")]
    Cancelled,
    /// A search, verification, or binding budget was exhausted. Not an
    /// impossibility claim.
    #[error("{}", budget.code())]
    Budget {
        /// Which budget ran out.
        budget: IdentificationBudget,
    },
    /// A transport query, diagram, or coordinate is malformed for the
    /// requested route.
    #[error("{message}")]
    InvalidInput {
        /// Stable code, optionally followed by detail.
        message: String,
    },
    /// The request is well formed but outside the route's declared bound.
    #[error("{code}")]
    UnsupportedInput {
        /// Stable code naming the exceeded bound.
        code: &'static str,
    },
    /// The evidence catalog itself is invalid or disagrees with the checked
    /// inputs.
    #[error("{message}")]
    InvalidCatalog {
        /// Stable code, optionally followed by detail.
        message: String,
    },
    /// A required available regime or law is absent from the catalog. Distinct
    /// from not-certified and from a verified impossibility witness.
    #[error("{code}: {detail}")]
    MissingEvidence {
        /// Stable code of the route that found the gap.
        code: &'static str,
        /// Which factor or law is missing.
        detail: String,
    },
    /// A portable derivation, obstruction, or hedge record does not check
    /// against the supplied inputs.
    #[error("{code}")]
    InvalidDerivation {
        /// Stable code naming the failed check.
        code: &'static str,
    },
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

    /// An exhausted budget.
    #[must_use]
    pub const fn budget(budget: IdentificationBudget) -> Self {
        Self::Budget { budget }
    }

    /// A malformed transport input.
    #[must_use]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput { message: message.into() }
    }

    /// An invalid or disagreeing evidence catalog.
    #[must_use]
    pub fn invalid_catalog(message: impl Into<String>) -> Self {
        Self::InvalidCatalog { message: message.into() }
    }

    /// A failed derivation, obstruction, or hedge check.
    #[must_use]
    pub const fn invalid_derivation(code: &'static str) -> Self {
        Self::InvalidDerivation { code }
    }

    /// A missing available regime or law.
    #[must_use]
    pub fn missing_evidence(code: &'static str, detail: impl Into<String>) -> Self {
        Self::MissingEvidence { code, detail: detail.into() }
    }

    /// Whether this error is a budget or cancellation outcome rather than a
    /// scientific one.
    #[must_use]
    pub const fn is_budget_or_cancel(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Budget { .. })
    }

    /// The stable transport outcome kind this error maps to. Callers match the
    /// kind; they never parse the message.
    #[must_use]
    pub const fn transport_outcome_kind(&self) -> TransportOutcomeKind {
        match self {
            Self::Cancelled | Self::Budget { .. } => TransportOutcomeKind::BudgetCancel,
            Self::MissingEvidence { .. } => TransportOutcomeKind::MissingEvidence,
            Self::UnknownVariable { .. }
            | Self::InvalidQuery { .. }
            | Self::Graph(_)
            | Self::InvalidInput { .. }
            | Self::UnsupportedInput { .. }
            | Self::InvalidCatalog { .. } => TransportOutcomeKind::InvalidInput,
            Self::UnsupportedQuery { .. }
            | Self::SustainedPolicyUnsupported
            | Self::NotCertified { .. }
            | Self::ResultLimitExceeded { .. }
            | Self::InvariantViolated { .. }
            | Self::InvalidDerivation { .. }
            | Self::Message(_) => TransportOutcomeKind::NotCertified,
        }
    }
}
