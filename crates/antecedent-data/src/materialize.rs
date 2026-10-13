//! Explicit materialization / copy diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "arrow")]
use antecedent_core::{Diagnostic, DiagnosticKind, DiagnosticSeverity};

/// Why a buffer was materialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg(feature = "arrow")]
pub(crate) enum MaterializationReason {
    /// Arrow / foreign buffer could not be borrowed zero-copy.
    ForeignBufferIncompatible,
    /// A validity bitmap was built from Arrow null slots beside a zero-copy value borrow.
    ValidityBitmap,
}

/// Record a materialization as an execution diagnostic.
#[must_use]
#[cfg(feature = "arrow")]
pub(crate) fn materialization_diagnostic(reason: MaterializationReason, bytes: u64) -> Diagnostic {
    let code = match reason {
        MaterializationReason::ForeignBufferIncompatible => "exec.materialize.foreign",
        MaterializationReason::ValidityBitmap => "exec.materialize.validity",
    };
    let mut d = Diagnostic::new(
        code,
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        format!("materialized {bytes} bytes ({reason:?})"),
    );
    d.fields = std::sync::Arc::from([(
        std::sync::Arc::<str>::from("bytes"),
        std::sync::Arc::<str>::from(bytes.to_string()),
    )]);
    d
}
