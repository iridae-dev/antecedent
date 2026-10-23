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
    /// The study searches for structure; no graph was supplied and the
    /// discovered structure is the answer, not an input.
    Discovered,
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
            Self::Discovered => "discovered",
        }
    }
}

/// Matrix query axis. The names are the public support-matrix query labels.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum SupportQuery {
    /// Average effect.
    AverageEffect,
    /// Conditional effect.
    ConditionalEffect,
    /// Counterfactual.
    Counterfactual,
    /// Nested counterfactual effect.
    NestedCounterfactualEffect,
    /// Interventional distribution.
    InterventionalDistribution,
    /// Path-specific effect.
    PathSpecificEffect,
    /// Mediation effect.
    MediationEffect,
    /// Temporal mediation effect.
    TemporalMediationEffect,
    /// Pulse effect.
    PulseEffect,
    /// Sustained effect.
    SustainedEffect,
    /// Temporal conditional pulse.
    TemporalConditionalPulse,
    /// Temporal conditional sustained.
    TemporalConditionalSustained,
    /// Response curve.
    ResponseCurve,
    /// Average derivative.
    AverageDerivative,
    /// Point derivative.
    PointDerivative,
    /// Semi-elasticity.
    SemiElasticity,
    /// Elasticity.
    Elasticity,
    /// Directional derivative.
    DirectionalDerivative,
    /// Response Jacobian.
    ResponseJacobian,
    /// Intervention response.
    InterventionResponse,
    /// Transport query.
    TransportQuery,
    /// Interference query.
    InterferenceQuery,
    /// Anomaly attribution.
    AnomalyAttribution,
    /// Change attribution.
    ChangeAttribution,
}

impl SupportQuery {
    /// Wire name (`parity/support_axes.toml`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AverageEffect => "AverageEffect",
            Self::ConditionalEffect => "ConditionalEffect",
            Self::Counterfactual => "Counterfactual",
            Self::NestedCounterfactualEffect => "NestedCounterfactualEffect",
            Self::InterventionalDistribution => "InterventionalDistribution",
            Self::PathSpecificEffect => "PathSpecificEffect",
            Self::MediationEffect => "MediationEffect",
            Self::TemporalMediationEffect => "TemporalMediationEffect",
            Self::PulseEffect => "PulseEffect",
            Self::SustainedEffect => "SustainedEffect",
            Self::TemporalConditionalPulse => "TemporalConditionalPulse",
            Self::TemporalConditionalSustained => "TemporalConditionalSustained",
            Self::ResponseCurve => "ResponseCurve",
            Self::AverageDerivative => "AverageDerivative",
            Self::PointDerivative => "PointDerivative",
            Self::SemiElasticity => "SemiElasticity",
            Self::Elasticity => "Elasticity",
            Self::DirectionalDerivative => "DirectionalDerivative",
            Self::ResponseJacobian => "ResponseJacobian",
            Self::InterventionResponse => "InterventionResponse",
            Self::TransportQuery => "TransportQuery",
            Self::InterferenceQuery => "InterferenceQuery",
            Self::AnomalyAttribution => "AnomalyAttribution",
            Self::ChangeAttribution => "ChangeAttribution",
        }
    }

    /// Parse a matrix query label.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "AverageEffect" => Self::AverageEffect,
            "ConditionalEffect" => Self::ConditionalEffect,
            "Counterfactual" => Self::Counterfactual,
            "NestedCounterfactualEffect" => Self::NestedCounterfactualEffect,
            "InterventionalDistribution" => Self::InterventionalDistribution,
            "PathSpecificEffect" => Self::PathSpecificEffect,
            "MediationEffect" => Self::MediationEffect,
            "TemporalMediationEffect" => Self::TemporalMediationEffect,
            "PulseEffect" => Self::PulseEffect,
            "SustainedEffect" => Self::SustainedEffect,
            "TemporalConditionalPulse" => Self::TemporalConditionalPulse,
            "TemporalConditionalSustained" => Self::TemporalConditionalSustained,
            "ResponseCurve" => Self::ResponseCurve,
            "AverageDerivative" => Self::AverageDerivative,
            "PointDerivative" => Self::PointDerivative,
            "SemiElasticity" => Self::SemiElasticity,
            "Elasticity" => Self::Elasticity,
            "DirectionalDerivative" => Self::DirectionalDerivative,
            "ResponseJacobian" => Self::ResponseJacobian,
            "InterventionResponse" => Self::InterventionResponse,
            "TransportQuery" => Self::TransportQuery,
            "InterferenceQuery" => Self::InterferenceQuery,
            "AnomalyAttribution" => Self::AnomalyAttribution,
            "ChangeAttribution" => Self::ChangeAttribution,
            _ => return None,
        })
    }
}

/// Graph-class axis of a support cell.
///
/// [`Self::CoDetermined`] and [`Self::Unknown`] are classification-only labels.
/// They are not [`GraphClass`](crate::GraphClass) variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum SupportGraphClass {
    /// Directed acyclic graph.
    Dag,
    /// Acyclic directed mixed graph.
    Admg,
    /// Completed partial DAG.
    Cpdag,
    /// Partial ancestral graph.
    Pag,
    /// Temporal DAG.
    TemporalDag,
    /// Temporal CPDAG.
    TemporalCpdag,
    /// Temporal PAG.
    TemporalPag,
    /// Temporal ADMG.
    TemporalAdmg,
    /// Known closure ADMG. Not a [`GraphClass`](crate::GraphClass).
    CoDetermined,
    /// Two-scenario envelope. Not a [`GraphClass`](crate::GraphClass).
    Unknown,
}

impl SupportGraphClass {
    /// Wire name (`parity/support_axes.toml`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dag => "Dag",
            Self::Admg => "Admg",
            Self::Cpdag => "Cpdag",
            Self::Pag => "Pag",
            Self::TemporalDag => "TemporalDag",
            Self::TemporalCpdag => "TemporalCpdag",
            Self::TemporalPag => "TemporalPag",
            Self::TemporalAdmg => "TemporalAdmg",
            Self::CoDetermined => "CoDetermined",
            Self::Unknown => "Unknown",
        }
    }

    /// Parse a matrix graph-class label, including the two extras.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "Dag" => Self::Dag,
            "Admg" => Self::Admg,
            "Cpdag" => Self::Cpdag,
            "Pag" => Self::Pag,
            "TemporalDag" => Self::TemporalDag,
            "TemporalCpdag" => Self::TemporalCpdag,
            "TemporalPag" => Self::TemporalPag,
            "TemporalAdmg" => Self::TemporalAdmg,
            "CoDetermined" => Self::CoDetermined,
            "Unknown" => Self::Unknown,
            _ => return None,
        })
    }

    /// The [`GraphClass`](crate::GraphClass) this axis names, when it is one.
    #[must_use]
    pub const fn graph_class(self) -> Option<crate::GraphClass> {
        Some(match self {
            Self::Dag => crate::GraphClass::Dag,
            Self::Admg => crate::GraphClass::Admg,
            Self::Cpdag => crate::GraphClass::Cpdag,
            Self::Pag => crate::GraphClass::Pag,
            Self::TemporalDag => crate::GraphClass::TemporalDag,
            Self::TemporalCpdag => crate::GraphClass::TemporalCpdag,
            Self::TemporalPag => crate::GraphClass::TemporalPag,
            Self::TemporalAdmg => crate::GraphClass::TemporalAdmg,
            Self::CoDetermined | Self::Unknown => return None,
        })
    }
}

/// Structure axis stored on a cell. [`StructureSource::RandomizedTrial`] is
/// included so a trial cell can be named even though no geometric row uses it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum SupportStructure {
    /// Caller-supplied graph.
    Explicit,
    /// Reviewed graph.
    Accepted,
    /// Graph posterior.
    GraphPosterior,
    /// Randomized trial, off the geometric graph axis.
    RandomizedTrial,
    /// Structure search, off the geometric graph axis.
    Discovered,
}

impl SupportStructure {
    /// Wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Accepted => "accepted",
            Self::GraphPosterior => "graph_posterior",
            Self::RandomizedTrial => "randomized_trial",
            Self::Discovered => "discovered",
        }
    }

    /// Parse a structure label.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "explicit" => Self::Explicit,
            "accepted" => Self::Accepted,
            "graph_posterior" => Self::GraphPosterior,
            "randomized_trial" => Self::RandomizedTrial,
            "discovered" => Self::Discovered,
            _ => return None,
        })
    }
}

impl From<StructureSource> for SupportStructure {
    fn from(source: StructureSource) -> Self {
        match source {
            StructureSource::Explicit => Self::Explicit,
            StructureSource::Accepted => Self::Accepted,
            StructureSource::GraphPosterior => Self::GraphPosterior,
            StructureSource::RandomizedTrial => Self::RandomizedTrial,
            StructureSource::Discovered => Self::Discovered,
        }
    }
}

/// Inference axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum SupportInference {
    /// Frequentist.
    Frequentist,
    /// Bayesian.
    Bayesian,
}

impl SupportInference {
    /// Wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Frequentist => "Frequentist",
            Self::Bayesian => "Bayesian",
        }
    }

    /// Parse an inference label.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "Frequentist" => Some(Self::Frequentist),
            "Bayesian" => Some(Self::Bayesian),
            _ => None,
        }
    }
}

/// Validation axis. Placebo-and-RCC shares the `full` label with the full suite.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum SupportValidation {
    /// No refuters.
    None,
    /// Cheap suite.
    Cheap,
    /// Full suite, including placebo-and-RCC.
    Full,
}

impl SupportValidation {
    /// Wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Cheap => "cheap",
            Self::Full => "full",
        }
    }

    /// Parse a validation label.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "cheap" => Some(Self::Cheap),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
}

macro_rules! axis_display {
    ($($ty:ty),* $(,)?) => {$(
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }
    )*};
}

axis_display!(
    SupportQuery,
    SupportGraphClass,
    SupportStructure,
    SupportInference,
    SupportValidation,
);

/// One support-matrix coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SupportCell {
    /// Public query name.
    pub query: SupportQuery,
    /// Graph class, including the classification-only extras.
    pub graph_class: SupportGraphClass,
    /// How structure was supplied.
    pub structure: SupportStructure,
    /// Frequentist or Bayesian.
    pub inference: SupportInference,
    /// Validation suite collapsed onto `none` / `cheap` / `full`.
    pub validation: SupportValidation,
}

impl SupportCell {
    /// Parse the five axis labels. Unknown text returns `None`.
    #[must_use]
    pub fn from_labels(
        query: &str,
        graph_class: &str,
        structure: &str,
        inference: &str,
        validation: &str,
    ) -> Option<Self> {
        Some(Self {
            query: SupportQuery::parse(query)?,
            graph_class: SupportGraphClass::parse(graph_class)?,
            structure: SupportStructure::parse(structure)?,
            inference: SupportInference::parse(inference)?,
            validation: SupportValidation::parse(validation)?,
        })
    }
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
