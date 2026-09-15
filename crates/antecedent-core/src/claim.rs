//! Portable claim envelope over a compiled contract and one result.
//!
//! A program defines an analysis. A claim records what a particular
//! execution or licensed derivation concludes. Receiving bytes,
//! understanding semantics, verifying dependencies, and being able to
//! execute remain separate states.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::identity::{ContractIdentities, SemanticDigest};
use crate::reasoning::ReasoningView;

/// Kind of value a claim carries. Refusals and bounds are first-class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ClaimKind {
    /// Scalar point estimate.
    Point,
    /// Identified-set bounds.
    Bounds,
    /// Probability-weighted mixture.
    Mixture,
    /// Function-valued response.
    Response,
    /// Structured refusal.
    Refusal,
    /// Incomplete contract with explicit obligations.
    Incomplete,
}

impl ClaimKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Bounds => "bounds",
            Self::Mixture => "mixture",
            Self::Response => "response",
            Self::Refusal => "refusal",
            Self::Incomplete => "incomplete",
        }
    }
}

/// Status of a requested coordinate relative to a claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DomainStatus {
    /// Identified at this coordinate.
    Identified,
    /// Empirically supported at this coordinate.
    Supported,
    /// Actually evaluated at this coordinate.
    Evaluated,
    /// Outside the declared domain.
    OutsideScope,
    /// Unknown; not a failed support and not an executable guarantee.
    Unknown,
}

impl DomainStatus {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identified => "identified",
            Self::Supported => "supported",
            Self::Evaluated => "evaluated",
            Self::OutsideScope => "outside_scope",
            Self::Unknown => "unknown",
        }
    }
}

/// Separate identification / support / evaluation domains.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ClaimDomains {
    /// Identification domain status.
    pub identification: DomainStatus,
    /// Empirical support domain status.
    pub support: DomainStatus,
    /// Actually evaluated domain status.
    pub evaluated: DomainStatus,
}

impl ClaimDomains {
    /// Construct the three domain statuses.
    #[must_use]
    pub const fn new(
        identification: DomainStatus,
        support: DomainStatus,
        evaluated: DomainStatus,
    ) -> Self {
        Self { identification, support, evaluated }
    }
}

/// Versioned claim envelope. Scientific payload encodings stay in `antecedent-io`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ClaimEnvelope {
    /// Claim id (content-addressed from the envelope fields).
    pub claim_id: SemanticDigest,
    /// Program / target / inference identities this claim references.
    pub identities: ContractIdentities,
    /// Result kind.
    pub kind: ClaimKind,
    /// Scalar value when [`ClaimKind::Point`]; `None` is explicit absence.
    pub value: Option<f64>,
    /// Outcome scale / units, when known.
    pub outcome_units: Option<Arc<str>>,
    /// Four reasoning slots.
    pub reasoning: ReasoningView,
    /// Domain statuses.
    pub domains: ClaimDomains,
    /// Producer / execution lineage digest, when an execution exists.
    pub execution: Option<SemanticDigest>,
    /// Evidence / dependency references (artifact ids, fixture names).
    pub evidence: Arc<[Arc<str>]>,
}

impl ClaimEnvelope {
    /// Construct a claim envelope.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        claim_id: SemanticDigest,
        identities: ContractIdentities,
        kind: ClaimKind,
        value: Option<f64>,
        outcome_units: Option<Arc<str>>,
        reasoning: ReasoningView,
        domains: ClaimDomains,
        execution: Option<SemanticDigest>,
        evidence: impl Into<Arc<[Arc<str>]>>,
    ) -> Self {
        Self {
            claim_id,
            identities,
            kind,
            value,
            outcome_units,
            reasoning,
            domains,
            execution,
            evidence: evidence.into(),
        }
    }
}

/// Receiving-system acceptance report. Distinct from byte storage.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct AcceptanceReport {
    /// Whether required semantic features were recognized.
    pub recognized: bool,
    /// Whether referenced identities / sections verified.
    pub verified_references: bool,
    /// Unresolved dependencies.
    pub unresolved: Arc<[Arc<str>]>,
    /// Operations the receiver can perform.
    pub supported_operations: Arc<[Arc<str>]>,
    /// Why acceptance is restricted or refused.
    pub restriction: Option<Arc<str>>,
}

impl AcceptanceReport {
    /// Construct a receiving-system report.
    #[must_use]
    pub fn new(
        recognized: bool,
        verified_references: bool,
        unresolved: impl Into<Arc<[Arc<str>]>>,
        supported_operations: impl Into<Arc<[Arc<str>]>>,
        restriction: Option<Arc<str>>,
    ) -> Self {
        Self {
            recognized,
            verified_references,
            unresolved: unresolved.into(),
            supported_operations: supported_operations.into(),
            restriction,
        }
    }

    /// Opaque storage/forwarding without interpretation.
    #[must_use]
    pub fn opaque_storage() -> Self {
        Self {
            recognized: false,
            verified_references: false,
            unresolved: Arc::from([Arc::from("required_semantics")]),
            supported_operations: Arc::from([Arc::from("store"), Arc::from("forward")]),
            restriction: Some(Arc::from("opaque_envelope")),
        }
    }

    /// Whether this report accepts the claim as a usable causal claim.
    #[must_use]
    pub fn accepts_as_claim(&self) -> bool {
        self.recognized && self.verified_references && self.unresolved.is_empty()
    }

    /// Whether the artifact is a fully verified program or claim.
    ///
    /// Distinct from storage/forwarding. Old artifacts without a contract
    /// section remain readable but do not pass this check.
    #[must_use]
    pub fn accepts_as_verified_program(&self) -> bool {
        self.accepts_as_claim()
    }
}

/// Handoff / loss receipt. Chained receipts accumulate unresolved losses.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct HandoffReceipt {
    /// Input claim id.
    pub input: SemanticDigest,
    /// Output claim id, when a claim was produced.
    pub output: Option<SemanticDigest>,
    /// Consumer capability profile id.
    pub consumer: Arc<str>,
    /// Transformation rule applied.
    pub rule: Arc<str>,
    /// Fields / sections retained.
    pub retained: Arc<[Arc<str>]>,
    /// Fields / sections omitted.
    pub omitted: Arc<[Arc<str>]>,
    /// Unresolved references after the handoff.
    pub unresolved: Arc<[Arc<str>]>,
    /// Operations no longer available.
    pub unavailable_operations: Arc<[Arc<str>]>,
}

impl HandoffReceipt {
    /// Whether omitted required meaning prevents equivalent-claim acceptance.
    #[must_use]
    pub fn equivalent_claim(&self) -> bool {
        self.omitted.is_empty() && self.unresolved.is_empty() && self.output.is_some()
    }

    /// Accumulate unresolved losses from `self` then `next`.
    #[must_use]
    pub fn chain(&self, next: &Self) -> Self {
        let mut unresolved: Vec<Arc<str>> = self.unresolved.iter().cloned().collect();
        for item in next.unresolved.iter() {
            if !unresolved.iter().any(|existing| existing == item) {
                unresolved.push(item.clone());
            }
        }
        let mut omitted: Vec<Arc<str>> = self.omitted.iter().cloned().collect();
        for item in next.omitted.iter() {
            if !omitted.iter().any(|existing| existing == item) {
                omitted.push(item.clone());
            }
        }
        Self {
            input: self.input,
            output: next.output,
            consumer: next.consumer.clone(),
            rule: next.rule.clone(),
            retained: next.retained.clone(),
            omitted: Arc::from(omitted),
            unresolved: Arc::from(unresolved),
            unavailable_operations: next.unavailable_operations.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        let mut bytes = [0u8; 32];
        bytes[0] = byte;
        SemanticDigest::from_bytes(bytes)
    }

    #[test]
    fn opaque_storage_is_not_claim_acceptance() {
        let report = AcceptanceReport::opaque_storage();
        assert!(!report.accepts_as_claim());
        assert!(report.supported_operations.iter().any(|op| &**op == "forward"));
    }

    #[test]
    fn chained_receipts_accumulate_losses() {
        let first = HandoffReceipt {
            input: digest(1),
            output: Some(digest(2)),
            consumer: Arc::from("restricted"),
            rule: Arc::from("scalar_view"),
            retained: Arc::from([Arc::from("value")]),
            omitted: Arc::from([Arc::from("unidentified_mass")]),
            unresolved: Arc::from([]),
            unavailable_operations: Arc::from([Arc::from("re_estimate")]),
        };
        let second = HandoffReceipt {
            input: digest(2),
            output: Some(digest(3)),
            consumer: Arc::from("forward"),
            rule: Arc::from("reserialize"),
            retained: Arc::from([Arc::from("value")]),
            omitted: Arc::from([]),
            unresolved: Arc::from([Arc::from("covariance")]),
            unavailable_operations: Arc::from([]),
        };
        let chained = first.chain(&second);
        assert!(!chained.equivalent_claim());
        assert!(chained.omitted.iter().any(|f| &**f == "unidentified_mass"));
        assert!(chained.unresolved.iter().any(|f| &**f == "covariance"));
    }
}
