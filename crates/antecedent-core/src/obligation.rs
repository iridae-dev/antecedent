//! Typed obligation references that extend [`crate::AssumptionRecord`].
//!
//! This is not a parallel assumption taxonomy. Source, status, and the
//! assumption itself reuse the existing records. Obligations add a stable
//! id, a finer scope, a required check, and how the obligation entered.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::assumption::{Assumption, AssumptionSource, AssumptionStatus};

/// Scope at which an obligation applies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ObligationScope {
    /// Entire compiled program.
    Program,
    /// One identified factor / functional factor.
    Factor,
    /// One structural atom (graph / completion).
    Atom,
    /// One temporal horizon.
    Horizon {
        /// Horizon step.
        horizon: u32,
    },
}

impl ObligationScope {
    /// Stable `snake_case` name (horizon includes the step).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Program => "program",
            Self::Factor => "factor",
            Self::Atom => "atom",
            Self::Horizon { .. } => "horizon",
        }
    }
}

/// How an obligation was introduced or last evaluated.
///
/// Distinct from [`AssumptionStatus`]: a passed refuter is an empirical
/// diagnostic, not a proof of an observational assumption.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ObligationKind {
    /// Caller-declared premise.
    UserAssertion,
    /// Implied by the accepted graph / identifier.
    GraphicalImplication,
    /// Empirical diagnostic that was run.
    EmpiricalDiagnostic,
    /// Empirical diagnostic that failed.
    FailedCheck,
    /// Not empirically testable from available evidence.
    Uncheckable,
    /// Required check has not been run.
    CheckNotRun,
}

impl ObligationKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserAssertion => "user_assertion",
            Self::GraphicalImplication => "graphical_implication",
            Self::EmpiricalDiagnostic => "empirical_diagnostic",
            Self::FailedCheck => "failed_check",
            Self::Uncheckable => "uncheckable",
            Self::CheckNotRun => "check_not_run",
        }
    }

    /// Whether this obligation is still unresolved.
    #[must_use]
    pub const fn is_unresolved(self) -> bool {
        matches!(self, Self::CheckNotRun | Self::FailedCheck)
    }
}

/// One scoped obligation referenced by a contract or claim.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ObligationRecord {
    /// Stable obligation id.
    pub id: Arc<str>,
    /// Program / factor / atom / horizon.
    pub scope: ObligationScope,
    /// How the underlying assumption entered, when one exists.
    pub source: AssumptionSource,
    /// How the obligation was introduced or last evaluated.
    pub kind: ObligationKind,
    /// Validation status of the underlying assumption.
    pub status: AssumptionStatus,
    /// Required check or declaration id, when one exists.
    pub required_check: Option<Arc<str>>,
    /// Evidence reference (artifact id, diagnostic code, or fixture).
    pub evidence: Option<Arc<str>>,
    /// Underlying assumption, when this obligation wraps one.
    pub assumption: Option<Assumption>,
    /// Human-readable description.
    pub description: Arc<str>,
}

impl ObligationRecord {
    /// Construct a fully specified obligation.
    #[must_use]
    pub fn new(
        id: impl Into<Arc<str>>,
        scope: ObligationScope,
        source: AssumptionSource,
        kind: ObligationKind,
        status: AssumptionStatus,
        description: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            id: id.into(),
            scope,
            source,
            kind,
            status,
            required_check: None,
            evidence: None,
            assumption: None,
            description: description.into(),
        }
    }

    /// Attach a required check id.
    #[must_use]
    pub fn with_required_check(mut self, check: impl Into<Arc<str>>) -> Self {
        self.required_check = Some(check.into());
        self
    }

    /// Attach an evidence reference.
    #[must_use]
    pub fn with_evidence(mut self, evidence: impl Into<Arc<str>>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }

    /// Attach the underlying assumption.
    #[must_use]
    pub fn with_assumption(mut self, assumption: Assumption) -> Self {
        self.assumption = Some(assumption);
        self
    }

    /// Whether this obligation still blocks a stronger claim.
    #[must_use]
    pub const fn is_unresolved(&self) -> bool {
        self.kind.is_unresolved()
            || matches!(self.status, AssumptionStatus::Declared | AssumptionStatus::Contradicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_kinds_are_explicit() {
        assert!(ObligationKind::CheckNotRun.is_unresolved());
        assert!(ObligationKind::FailedCheck.is_unresolved());
        assert!(!ObligationKind::GraphicalImplication.is_unresolved());
        assert!(!ObligationKind::EmpiricalDiagnostic.is_unresolved());
    }
}
