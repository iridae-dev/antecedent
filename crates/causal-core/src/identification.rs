//! Shared identification status vocabulary (DESIGN.md §10.1 / §14.4).
//!
//! Lives in `causal-core` so both `causal-identify` and `causal-estimate` can
//! reference the same enum without a layering edge estimate → identify
//! (DESIGN.md §3.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Status of an identification attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationStatus {
    /// Nonparametrically identified.
    NonparametricallyIdentified,
    /// Identified only under a proper subset of the model class (partial ID).
    PartiallyIdentified,
    /// Identification depends on which graph in an equivalence class / ensemble.
    GraphDependent,
    /// Not identified.
    NotIdentified,
}
