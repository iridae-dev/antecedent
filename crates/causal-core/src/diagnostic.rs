//! Structured diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// Severity of a diagnostic condition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DiagnosticSeverity {
    /// Informational note.
    Info,
    /// Non-fatal scientific or operational warning.
    Warning,
    /// Severe condition that may invalidate interpretation.
    Error,
}

/// Stable diagnostic category.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DiagnosticKind {
    /// Scientific / statistical condition.
    Scientific,
    /// Execution / performance path choice.
    Execution,
}

/// Machine-readable diagnostic attached to analysis artifacts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Stable diagnostic code (e.g. `exec.arrow_copied`).
    pub code: Arc<str>,
    /// Kind of diagnostic.
    pub kind: DiagnosticKind,
    /// Severity.
    pub severity: DiagnosticSeverity,
    /// Human-readable message.
    pub message: Arc<str>,
    /// Optional affected artifact identifier.
    pub artifact_id: Option<Arc<str>>,
    /// Optional structured fields as `key=value` pairs.
    pub fields: Arc<[(Arc<str>, Arc<str>)]>,
}

impl Diagnostic {
    /// Construct a diagnostic with no fields.
    #[must_use]
    pub fn new(
        code: impl Into<Arc<str>>,
        kind: DiagnosticKind,
        severity: DiagnosticSeverity,
        message: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            code: code.into(),
            kind,
            severity,
            message: message.into(),
            artifact_id: None,
            fields: Arc::from([]),
        }
    }
}

/// Collection of diagnostics.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiagnosticSet {
    /// Ordered diagnostics.
    pub entries: Vec<Diagnostic>,
}

impl DiagnosticSet {
    /// Empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a diagnostic.
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.entries.push(diagnostic);
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
