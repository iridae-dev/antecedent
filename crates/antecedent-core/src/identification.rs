//! Shared identification status vocabulary.
//!
//! Lives in `antecedent-core` so both `antecedent-identify` and `antecedent-estimate` can
//! reference the same enum without a layering edge estimate → identify
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Status of an identification attempt.
///
/// [`Self::IdentifiedUnderParametricRestrictions`] and
/// [`Self::IdentifiedUnderPriorRestrictions`] are vocabulary for assumption-restricted
/// identification. They must **not** be confused with "Bayesian estimation ran with a
/// prior" — priors alone must not flip [`Self::NotIdentified`] to an identified status.
/// [`Self::IdentifiedUnderParametricRestrictions`] is emitted by the GCM / parametric-SCM
/// path (`parametric_scm_identification` in the `antecedent` crate, covering counterfactual,
/// anomaly-attribution, change-attribution, mechanism-change, and unit-change queries), and
/// both variants are whitelisted by estimation gates (see
/// `identification_status_acceptable()` in the `antecedent` crate's `strategy_table`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentificationStatus {
    /// Nonparametrically identified.
    NonparametricallyIdentified,
    /// Identified under parametric restrictions on the model class.
    ///
    /// Emitted by the GCM / parametric-SCM path; accepted by estimation gates.
    IdentifiedUnderParametricRestrictions,
    /// Identified under prior / substantive restrictions treated as identifying assumptions.
    /// Distinct from attaching a prior to a non-identified estimand.
    ///
    /// Accepted by estimation gates; not yet emitted by any current algorithm (reserved for
    /// future ID producers).
    IdentifiedUnderPriorRestrictions,
    /// Identified only under a proper subset of the model class (partial ID).
    PartiallyIdentified,
    /// Identification depends on which graph in an equivalence class / ensemble.
    GraphDependent,
    /// Not identified.
    NotIdentified,
}
