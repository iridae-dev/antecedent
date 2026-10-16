//! Sample construction policies.
//!
//! Distinct from `antecedent_counterfactual::AbductionMissingPolicy` (abduction).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

/// How missing values (validity bits) affect row selection.
///
/// Each policy is a declaration about the missingness mechanism as well as a row rule; the
/// route that executes it states the assumption it needs (see the missing-data guide).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum MissingPolicy {
    /// Drop any row where a requested column is invalid.
    ///
    /// Unbiased when the dropped rows are a random subset (missing completely at random) or,
    /// for a regression of the outcome on the retained covariates, when missingness depends
    /// only on those covariates. The dropped rows are always counted.
    #[default]
    CompleteCase,
    /// Fail if any requested column has an invalid value in a candidate row.
    ErrorOnMissing,
    /// Randomized outcomes missing at random given the arm and declared baseline covariates:
    /// each responder is weighted by the inverse of its estimated probability of being
    /// observed within its arm. The assignment of a non-responder is never dropped from
    /// its arm's denominator.
    InverseProbabilityOfObservation,
    /// Panel designs: drop every unit that has a missing value in any period, so the
    /// remaining panel is balanced. Whole units are dropped, never single unit-periods.
    BalancedPanel,
    /// Panel designs: each contrast uses the units observed in the periods that contrast
    /// needs. Unit and period identities are preserved; nothing is relabelled.
    AvailableCase,
    /// Covariance-based discovery: every pair of variables uses the rows observed on both,
    /// and the pairwise covariance is repaired to positive semidefinite before scoring.
    /// Valid when missingness is completely at random; the repair is reported.
    PairwiseAvailable,
}

impl MissingPolicy {
    /// Stable lower-case name used in receipts, identities and the Python option.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CompleteCase => "complete_case",
            Self::ErrorOnMissing => "error",
            Self::InverseProbabilityOfObservation => "inverse_probability",
            Self::BalancedPanel => "balanced_panel",
            Self::AvailableCase => "available_case",
            Self::PairwiseAvailable => "pairwise_available",
        }
    }

    /// The policy named by `name`, if any.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [
            Self::CompleteCase,
            Self::ErrorOnMissing,
            Self::InverseProbabilityOfObservation,
            Self::BalancedPanel,
            Self::AvailableCase,
            Self::PairwiseAvailable,
        ]
        .into_iter()
        .find(|policy| policy.name() == name)
    }

    /// Every policy name, in documentation order.
    pub const NAMES: [&'static str; 6] = [
        "error",
        "complete_case",
        "inverse_probability",
        "balanced_panel",
        "available_case",
        "pairwise_available",
    ];
}

/// A caller's declaration of how missing values are handled: the policy and, for
/// [`MissingPolicy::InverseProbabilityOfObservation`], the covariates the observation
/// probability may depend on.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MissingDeclaration {
    /// Declared policy.
    pub policy: MissingPolicy,
    /// Covariates of the observation model; empty declares missing completely at random
    /// within arm.
    pub covariates: Arc<[VariableId]>,
}

impl MissingDeclaration {
    /// A declaration with no covariates.
    #[must_use]
    pub fn new(policy: MissingPolicy) -> Self {
        Self { policy, covariates: Arc::from([]) }
    }

    /// Declare the covariates the observation probability depends on.
    #[must_use]
    pub fn with_covariates(mut self, covariates: impl Into<Arc<[VariableId]>>) -> Self {
        self.covariates = covariates.into();
        self
    }
}

/// How the optional analysis mask is applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
#[non_exhaustive]
pub enum MaskPolicy {
    /// Intersect the analysis mask with complete-case / validity filtering.
    #[default]
    Honor,
    /// Ignore the analysis mask (still apply missingness policy).
    Ignore,
}
