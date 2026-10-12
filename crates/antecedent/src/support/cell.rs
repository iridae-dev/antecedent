//! Support-matrix cell types: the coordinate, its classification and the
//! structure-source axis.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{
    Admg, Cpdag, Dag, Pag, TemporalAdmg, TemporalCpdag, TemporalDag, TemporalPag,
};

use crate::accepted::AcceptedGraph;

/// Stable support-matrix refusal id.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SupportRefusal {
    /// Cell is typed-impossible (`parity/support_n_a.toml`).
    NotApplicable,
    /// Cell is in the cartesian product and is not licensed (default).
    Refused,
}

impl SupportRefusal {
    /// Wire id (`not_applicable`, `refused`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::Refused => "refused",
        }
    }
}

impl std::fmt::Display for SupportRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the caller supplied causal structure.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StructureSource {
    /// Caller passed a graph of the given [`GraphClass`](crate::GraphClass).
    Explicit,
    /// Caller passed an [`crate::AcceptedGraph`] produced by discovery review.
    Accepted,
    /// Caller passed a graph posterior (mixture over structures).
    GraphPosterior,
    /// Randomization itself supplies identification; no graph was supplied.
    RandomizedTrial,
}

/// Convert a caller-supplied structure into a graph plus its matrix axis.
///
/// [`AcceptedGraph`] is `accepted`. Bare graph types (`Dag`, `Admg`, …) are
/// `explicit`. Graph posteriors use [`StudyBuilder::graph_posterior`](crate::StudyBuilder::graph_posterior),
/// not this trait.
pub trait IntoGraphInput {
    /// Graph object and the structure-source axis value it represents.
    fn into_graph_input(self) -> (AcceptedGraph, StructureSource);
}

impl IntoGraphInput for AcceptedGraph {
    fn into_graph_input(self) -> (AcceptedGraph, StructureSource) {
        (self, StructureSource::Accepted)
    }
}

macro_rules! explicit_graph_input {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl IntoGraphInput for $ty {
                fn into_graph_input(self) -> (AcceptedGraph, StructureSource) {
                    (AcceptedGraph::from(self), StructureSource::Explicit)
                }
            }
        )+
    };
}

explicit_graph_input!(Dag, Admg, Pag, Cpdag, TemporalDag, TemporalCpdag, TemporalPag, TemporalAdmg);

impl StructureSource {
    /// Matrix axis value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Accepted => "accepted",
            Self::GraphPosterior => "graph_posterior",
            Self::RandomizedTrial => "randomized_trial",
        }
    }
}

/// One support-matrix coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SupportCell {
    /// Public query name.
    pub query: &'static str,
    /// [`GraphClass::as_str`](crate::GraphClass::as_str).
    pub graph_class: &'static str,
    /// [`StructureSource::as_str`].
    pub structure: &'static str,
    /// `Frequentist` or `Bayesian`.
    pub inference: &'static str,
    /// `none`, `cheap`, or `full`.
    pub validation: &'static str,
}

/// Classification of a [`SupportCell`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CellStatus {
    /// Listed in `support_licensed.toml`.
    Licensed,
    /// Matches an n/a predicate. `reason` is the rule text.
    NotApplicable {
        /// Why the cell is typed-impossible.
        reason: &'static str,
    },
    /// Retained `allowed_unlicensed` compatibility status.
    ///
    /// No matrix cell produces this status; the variant remains for wire
    /// compatibility with older artifacts and clients.
    Allowlisted {
        /// Historical reason this cell ran without a license.
        reason: &'static str,
        /// Historical licensed or keep-running family this row rode.
        parent: &'static str,
    },
    /// Default: in the product, not licensed, and not n/a.
    Refused,
}

impl CellStatus {
    /// Wire / Python id (`licensed`, `allowed_unlicensed`, `not_applicable`, `refused`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Licensed => "licensed",
            Self::Allowlisted { .. } => "allowed_unlicensed",
            Self::NotApplicable { .. } => "not_applicable",
            Self::Refused => "refused",
        }
    }

    /// Historical compatibility `reason`, when this status is [`Self::Allowlisted`].
    #[must_use]
    pub const fn allowlist_reason(self) -> Option<&'static str> {
        match self {
            Self::Allowlisted { reason, .. } => Some(reason),
            _ => None,
        }
    }

    /// Historical compatibility `parent`, when this status is [`Self::Allowlisted`].
    #[must_use]
    pub const fn allowlist_parent(self) -> Option<&'static str> {
        match self {
            Self::Allowlisted { parent, .. } => Some(parent),
            _ => None,
        }
    }
}
