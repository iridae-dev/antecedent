//! Execution context: request identity and receipts.
//! SPDX-License-Identifier: MIT OR Apache-2.0


/// Host request identity: scoped to the complete scientific + execution inputs.
///
/// A key reused for different inputs is a conflict. The request binds every
/// contract layer (target and population, premises, products, program,
/// inference binding, observation, data snapshot) plus execution lineage, so
/// a changed estimand, prior, or numeric knob under a reused key is detected.
/// The host owns scheduling; this record only names what the engine
/// considered one logical request.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct RequestIdentity {
    /// Caller-supplied idempotency key.
    pub idempotency_key: crate::identity::SemanticDigest,
    /// Every contract identity layer the request was bound to.
    pub contract: crate::identity::ContractIdentities,
    /// Execution-lineage identity (seed, threads, backend, budgets, version).
    pub execution: crate::identity::SemanticDigest,
}

impl RequestIdentity {
    /// Construct a complete request identity.
    #[must_use]
    pub const fn new(
        idempotency_key: crate::identity::SemanticDigest,
        contract: crate::identity::ContractIdentities,
        execution: crate::identity::SemanticDigest,
    ) -> Self {
        Self { idempotency_key, contract, execution }
    }

    /// Whether `other` reuses this key for a different scientific request.
    #[must_use]
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.idempotency_key == other.idempotency_key
            && (self.contract != other.contract || self.execution != other.execution)
    }
}

/// Lifecycle of an external execution request. Distinct from scientific result status.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ExecutionRequestState {
    /// Accepted but not started.
    Pending,
    /// Currently executing.
    Running,
    /// Finished with a published scientific result.
    Completed,
    /// Refused before or during execution (scientific or license).
    Refused,
    /// Failed without publishing a completed claim.
    Failed,
    /// Cancelled; diagnostics may be retained, a claim must not be published.
    Cancelled,
}

impl ExecutionRequestState {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Refused => "refused",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether this state may publish a completed claim.
    #[must_use]
    pub const fn publishes_claim(self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// Receipt for one host request. Cancellation or failure cannot mark a claim current.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReceipt {
    /// Request identity this receipt answers.
    pub request: RequestIdentity,
    /// Lifecycle state.
    pub state: ExecutionRequestState,
}

impl ExecutionReceipt {
    /// Construct a receipt.
    #[must_use]
    pub const fn new(request: RequestIdentity, state: ExecutionRequestState) -> Self {
        Self { request, state }
    }
}
