//! Shared identification status vocabulary (DESIGN.md §10.1 / §14.4).
//!
//! Lives in `causal-core` so both `causal-identify` and `causal-estimate` can
//! reference the same enum without a layering edge estimate → identify
//! (DESIGN.md §3.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Status of an identification attempt.
///
/// [`Self::IdentifiedUnderParametricRestrictions`] and
/// [`Self::IdentifiedUnderPriorRestrictions`] are vocabulary for assumption-restricted
/// identification (DESIGN §10.1). They are **not** emitted by current algorithms and must
/// **not** be confused with “Bayesian estimation ran with a prior” — priors alone must not
/// flip [`Self::NotIdentified`] to an identified status. Estimation gates remain
/// nonparametric-only until parametric / prior-restricted ID algorithms ship.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationStatus {
    /// Nonparametrically identified.
    NonparametricallyIdentified,
    /// Identified under parametric restrictions on the model class (DESIGN §10.1).
    ///
    /// Estimation gates do not yet accept this status; reserved for future ID producers.
    IdentifiedUnderParametricRestrictions,
    /// Identified under prior / substantive restrictions treated as identifying assumptions
    /// (DESIGN §10.1). Distinct from attaching a prior to a non-identified estimand.
    ///
    /// Estimation gates do not yet accept this status; reserved for future ID producers.
    IdentifiedUnderPriorRestrictions,
    /// Identified only under a proper subset of the model class (partial ID).
    PartiallyIdentified,
    /// Identification depends on which graph in an equivalence class / ensemble.
    GraphDependent,
    /// Not identified.
    NotIdentified,
}
