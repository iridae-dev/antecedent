//! Sample construction policies.
//!
//! Distinct from `antecedent_counterfactual::AbductionMissingPolicy` (abduction).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// How missing values (validity bits) affect row selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum MissingPolicy {
    /// Drop any row where a requested column is invalid.
    #[default]
    CompleteCase,
    /// Fail if any requested column has an invalid value in a candidate row.
    ErrorOnMissing,
}

/// How the optional analysis mask is applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum MaskPolicy {
    /// Intersect the analysis mask with complete-case / validity filtering.
    #[default]
    Honor,
    /// Ignore the analysis mask (still apply missingness policy).
    Ignore,
}

/// How observation weights are exposed on the prepared sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum WeightPolicy {
    /// Use storage weights when present; otherwise unit weights.
    #[default]
    Honor,
    /// Always use unit weights.
    Unit,
    /// Ignore weights entirely (`PreparedSample.weights` is `None`).
    Ignore,
}
