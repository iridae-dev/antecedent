//! Explicit materialization / copy diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Diagnostic, DiagnosticKind, DiagnosticSeverity};

/// Why a buffer was materialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MaterializationReason {
    /// Caller requested an owned contiguous copy.
    ExplicitCopy,
    /// Layout incompatible with the requested kernel (e.g. non-contiguous).
    LayoutIncompatible,
    /// Arrow / foreign buffer could not be borrowed zero-copy.
    ForeignBufferIncompatible,
}

/// Record a materialization as an execution diagnostic.
#[must_use]
pub fn materialization_diagnostic(reason: MaterializationReason, bytes: u64) -> Diagnostic {
    let code = match reason {
        MaterializationReason::ExplicitCopy => "exec.materialize.explicit_copy",
        MaterializationReason::LayoutIncompatible => "exec.materialize.layout",
        MaterializationReason::ForeignBufferIncompatible => "exec.materialize.foreign",
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
