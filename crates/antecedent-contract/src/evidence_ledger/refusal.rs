//! Typed refusals of the evidence ledger.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use super::ids::SourceId;

/// Why a ledger request was refused. Each kind maps to one registered reason
/// code; the namespaced `detail` names the specific rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RefusalKind {
    /// Observations would be counted more than once.
    DoubleCounted,
    /// Overlap is unknown and independence is not assumed.
    OverlapUnknown,
    /// A source or result the request needs is withdrawn, superseded or stale.
    SourceRetired,
    /// The request is malformed or inconsistent with the ledger.
    InvalidArgument,
    /// The requested statement has no license (an interval, a posterior).
    NotLicensed,
    /// A ledger bound was exceeded.
    LimitExceeded,
}

/// A typed, actionable ledger refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerError {
    /// Which rule family refused.
    pub kind: RefusalKind,
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail, `evidence_ledger.<rule>`.
    pub detail: &'static str,
    /// What was refused and what would license it.
    pub message: String,
    /// Sources the refusal names.
    pub sources: Vec<SourceId>,
}

impl LedgerError {
    /// The refusal as `(code, detail, explanation)`.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str, &str) {
        (self.code, self.detail, &self.message)
    }

    /// Attach the sources the refusal is about.
    #[must_use]
    pub fn with_sources(mut self, sources: impl IntoIterator<Item = SourceId>) -> Self {
        self.sources = sources.into_iter().collect();
        self
    }
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for LedgerError {}
