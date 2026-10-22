//! Shared identification status vocabulary.
//!
//! Lives in `antecedent-core` so both `antecedent-identify` and `antecedent-estimate` can
//! reference the same enum without a layering edge estimate → identify
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::named_variants;

/// Unidentified, unevaluable, subsampled, or failed posterior mass at or below
/// this value is numerical dust. Every posterior status ladder uses this
/// cutoff, so `1e-13` is not graph-dependent on one route and point-identified
/// on another.
pub const POSTERIOR_MASS_TOLERANCE: f64 = 1e-12;

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
/// Licensed `TransportQuery` / `InterferenceQuery` execute paths do **not** use that
/// helper: they emit [`Self::NonparametricallyIdentified`] under `transport.sid` or
/// `interference.design`.
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
    /// Not identified.
    ///
    /// Exhaustive search with no set, or a certificate of non-ID, is the
    /// scientific case. A search that could not finish (candidate-family cap,
    /// history bound) keeps this status — there is no third variant — and is
    /// marked by an Execution diagnostic, not a scientific open-back-door.
    NotIdentified,
}

named_variants! {
    IdentificationStatus {
        NonparametricallyIdentified => "nonparametrically_identified",
        IdentifiedUnderParametricRestrictions => "identified_under_parametric_restrictions",
        IdentifiedUnderPriorRestrictions => "identified_under_prior_restrictions",
        PartiallyIdentified => "partially_identified",
        GraphDependent => "graph_dependent",
        NotIdentified => "not_identified",
    }
}

impl IdentificationStatus {
    /// Evidential strength of the identification claim, larger is stronger.
    ///
    /// A derivation may keep or lower this rank but never raise it: a derived
    /// claim cannot assert a stronger identification than the parents it came from.
    #[must_use]
    pub const fn strength_rank(self) -> u8 {
        match self {
            Self::NonparametricallyIdentified => 5,
            Self::IdentifiedUnderParametricRestrictions => 4,
            Self::IdentifiedUnderPriorRestrictions => 3,
            Self::PartiallyIdentified => 2,
            Self::GraphDependent => 1,
            Self::NotIdentified => 0,
        }
    }
}
