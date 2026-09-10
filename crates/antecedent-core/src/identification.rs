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
/// anomaly-attribution, change-attribution, mechanism-change, and unit-change queries)
/// and by IV Wald identification (linearity, or LATE under monotonicity).
/// [`Self::IdentifiedUnderPriorRestrictions`] is reserved and is **not** accepted by
/// estimation gates or prior-bank hydration until an in-tree identifier emits it.
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
    /// Reserved: not emitted by any current algorithm, and not accepted by estimation
    /// gates or prior-bank hydration.
    IdentifiedUnderPriorRestrictions,
    /// Identified only under a proper subset of the model class (partial ID).
    PartiallyIdentified,
    /// Identification depends on which graph in an equivalence class / ensemble.
    GraphDependent,
    /// Search or budget could not determine identifiability.
    ///
    /// Distinct from [`Self::NotIdentified`]: the algorithm did not prove a
    /// non-identifiable completion (for example a candidate-family cap fired
    /// before enumeration). A budget miss must not be reported as a scientific
    /// open-back-door.
    Undetermined,
    /// Proven not identified (search completed, or a certificate of non-ID).
    NotIdentified,
}
