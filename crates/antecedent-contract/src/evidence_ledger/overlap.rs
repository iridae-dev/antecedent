//! Observation overlap, dependence labels and mass accounting.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use crate::claim::EvidenceDependence;

use super::ids::{ObservationId, SourceId};

/// Overlap of the observations behind two sources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Overlap {
    /// Both observation sets are known and disjoint.
    Independent,
    /// Both are known and `shared` observations appear in both.
    SharedObservations {
        /// Number of shared observations.
        shared: u64,
    },
    /// At least one set is not computable. This is never independence.
    Unknown {
        /// Why the observations are unknown.
        reason: &'static str,
    },
}

impl Overlap {
    /// The coarse dependence label used by the claim contract.
    #[must_use]
    pub const fn dependence(self) -> EvidenceDependence {
        match self {
            Self::Independent => EvidenceDependence::Independent,
            Self::SharedObservations { .. } => EvidenceDependence::Shared,
            Self::Unknown { .. } => EvidenceDependence::Unknown,
        }
    }
}

/// How two evidence sources relate, in the vocabulary the composition checks
/// use (`shared_data`, `shared_prior`, `shared_fitted_model`,
/// `unknown_dependence`, `independent`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RelationKind {
    /// Both rest on the same observations.
    SharedData,
    /// Both rest on the same prior.
    SharedPrior,
    /// Both rest on the same fitted model.
    SharedFittedModel,
    /// Dependence is not known.
    UnknownDependence,
    /// Observation sets are known and disjoint.
    Independent,
}

impl RelationKind {
    /// Stable `snake_case` name, equal to the composition vocabulary's.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SharedData => "shared_data",
            Self::SharedPrior => "shared_prior",
            Self::SharedFittedModel => "shared_fitted_model",
            Self::UnknownDependence => "unknown_dependence",
            Self::Independent => "independent",
        }
    }
}

/// How shared observations re-entered.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ReentryKind {
    /// An exact copy of an earlier source.
    Duplicate,
    /// A transformation of the same observations.
    Transform,
    /// A prior formed from, or paired with, the likelihood's own data.
    PriorPlusLikelihood,
    /// A model or external claim built from data already used.
    DerivedFromUsedData,
    /// The observations coincide with no declared derivation.
    SharedObservations,
}

impl ReentryKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Duplicate => "duplicate",
            Self::Transform => "transform",
            Self::PriorPlusLikelihood => "prior_plus_likelihood",
            Self::DerivedFromUsedData => "derived_from_used_data",
            Self::SharedObservations => "shared_observations",
        }
    }
}

/// What a result counted: each observation once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MassAccount {
    /// Contributing sources.
    pub sources: u64,
    /// Observation slots declared across contributing sources (a shared
    /// observation fills one slot per source that carries it).
    pub declared_slots: u64,
    /// Distinct observations: what the result counts, each once.
    pub distinct_observations: u64,
    /// `declared_slots - distinct_observations`: slots a naive count would
    /// double count.
    pub repeated_slots: u64,
    /// Contributing sources whose observations are not computable.
    pub unknown_sources: Vec<SourceId>,
}

/// Number of elements two sorted, distinct id slices share.
pub(crate) fn intersection_count(a: &[ObservationId], b: &[ObservationId]) -> u64 {
    let (mut i, mut j, mut shared) = (0, 0, 0_u64);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                shared += 1;
                i += 1;
                j += 1;
            }
        }
    }
    shared
}

/// Elements two sorted, distinct id slices share, in order.
pub(crate) fn intersection(a: &[ObservationId], b: &[ObservationId]) -> Vec<ObservationId> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Sorted union of sorted, distinct id slices.
pub(crate) fn union_sorted<'a>(sets: impl IntoIterator<Item = &'a [ObservationId]>) -> Vec<ObservationId> {
    let mut all: Vec<ObservationId> = sets.into_iter().flat_map(<[ObservationId]>::to_vec).collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// Whether sorted `sub` is contained in sorted `sup`.
pub(crate) fn is_subset(sub: &[ObservationId], sup: &[ObservationId]) -> bool {
    intersection_count(sub, sup) == sub.len() as u64
}
