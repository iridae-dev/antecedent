//! Public support-matrix lookup.
//!
//! Axes, n/a predicates, and licensed cells are generated from
//! `parity/support_*.toml`. Dispatch refuses [`CellStatus::NotApplicable`]
//! now. Cells in `parity/support_closed.toml` raise [`SupportRefusal::Refused`].
//! Remaining default-refused cells still run until licensed or closed.

use antecedent_core::{CausalQuery, DerivativeScale, ResponseFunctional, TemporalPolicy};

use antecedent_graph::{Admg, Dag, Pag, TemporalDag};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::analysis::RefuteSuite;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::support_matrix_data::{CLOSED_RULES, LICENSED, NA_RULES};

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
    /// Caller passed a graph of the given [`GraphClass`].
    Explicit,
    /// Caller passed an [`crate::AcceptedGraph`] produced by discovery review.
    Accepted,
    /// Caller passed a graph posterior (mixture over structures).
    GraphPosterior,
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

explicit_graph_input!(Dag, Admg, Pag, TemporalDag);

impl StructureSource {
    /// Matrix axis value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Accepted => "accepted",
            Self::GraphPosterior => "graph_posterior",
        }
    }
}

/// One support-matrix coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SupportCell {
    /// Public query name.
    pub query: &'static str,
    /// [`GraphClass::as_str`].
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
    /// Default: in the product, not licensed, not n/a.
    Refused,
}

fn axis_in(allowed: Option<&[&str]>, value: &str) -> bool {
    allowed.is_none_or(|xs| xs.contains(&value))
}

/// Classify `cell` against the generated n/a rules and licensed set.
#[must_use]
pub fn classify(cell: SupportCell) -> CellStatus {
    for rule in NA_RULES {
        if axis_in(rule.queries, cell.query)
            && axis_in(rule.graph_classes, cell.graph_class)
            && axis_in(rule.structures, cell.structure)
            && axis_in(rule.inferences, cell.inference)
            && axis_in(rule.validations, cell.validation)
        {
            return CellStatus::NotApplicable { reason: rule.reason };
        }
    }
    if LICENSED.iter().any(|row| {
        row.query == cell.query
            && row.graph_class == cell.graph_class
            && row.structure == cell.structure
            && row.inference == cell.inference
            && row.validation == cell.validation
    }) {
        return CellStatus::Licensed;
    }
    CellStatus::Refused
}

/// Matrix query name for `query` on `graph_class`, if the query is on the public axis.
#[must_use]
pub fn query_axis_name(query: &CausalQuery, graph_class: GraphClass) -> Option<&'static str> {
    match query {
        CausalQuery::AverageEffect(_) => Some("AverageEffect"),
        CausalQuery::ConditionalEffect(_) => Some("ConditionalEffect"),
        CausalQuery::Counterfactual(_) => Some("Counterfactual"),
        CausalQuery::Distribution(_) => Some("InterventionalDistribution"),
        CausalQuery::PathSpecific(_) => Some("PathSpecificEffect"),
        CausalQuery::Mediation(_) => match graph_class {
            GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag => {
                Some("TemporalMediationEffect")
            }
            GraphClass::Dag | GraphClass::Admg | GraphClass::Cpdag | GraphClass::Pag => {
                Some("MediationEffect")
            }
        },
        CausalQuery::TemporalEffect(q) => match q.policy {
            TemporalPolicy::Pulse { .. } => Some("PulseEffect"),
            TemporalPolicy::Sustained { .. } => Some("SustainedEffect"),
            _ => None,
        },
        CausalQuery::Response(q) => match &q.functional {
            ResponseFunctional::MeanCurve { .. } => Some("ResponseCurve"),
            ResponseFunctional::AverageDerivative { .. } => Some("AverageDerivative"),
            ResponseFunctional::PointDerivative { scale, .. } => match scale {
                DerivativeScale::Identity => Some("PointDerivative"),
                DerivativeScale::LogTreatment | DerivativeScale::LogOutcome => {
                    Some("SemiElasticity")
                }
                DerivativeScale::LogLog => Some("Elasticity"),
            },
            ResponseFunctional::DirectionalDerivative { .. } => Some("DirectionalDerivative"),
            ResponseFunctional::Jacobian { .. } => Some("ResponseJacobian"),
            ResponseFunctional::InterventionResponse { .. } => Some("InterventionResponse"),
        },
        CausalQuery::Transport(_) => Some("TransportQuery"),
        CausalQuery::Interference(_) => Some("InterferenceQuery"),
        // Attribution queries and any later `CausalQuery` variant stay off the axis.
        _ => None,
    }
}

fn inference_axis(mode: &InferenceMode) -> &'static str {
    match mode {
        InferenceMode::Frequentist => "Frequentist",
        InferenceMode::Bayesian(_) => "Bayesian",
    }
}

fn validation_axis(suite: RefuteSuite) -> &'static str {
    match suite {
        RefuteSuite::None => "none",
        RefuteSuite::Cheap => "cheap",
        RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => "full",
    }
}

/// Build a [`SupportCell`] when `query` is on the public axis.
#[must_use]
pub fn support_cell(
    query: &CausalQuery,
    graph_class: GraphClass,
    structure: StructureSource,
    inference: &InferenceMode,
    refute: RefuteSuite,
) -> Option<SupportCell> {
    Some(SupportCell {
        query: query_axis_name(query, graph_class)?,
        graph_class: graph_class.as_str(),
        structure: structure.as_str(),
        inference: inference_axis(inference),
        validation: validation_axis(refute),
    })
}

fn closed_reason(cell: SupportCell) -> Option<&'static str> {
    for rule in CLOSED_RULES {
        if axis_in(rule.queries, cell.query)
            && axis_in(rule.graph_classes, cell.graph_class)
            && axis_in(rule.structures, cell.structure)
            && axis_in(rule.inferences, cell.inference)
            && axis_in(rule.validations, cell.validation)
        {
            return Some(rule.reason);
        }
    }
    None
}

/// Refuse n/a cells and enforced closed holes. Licensed and remaining
/// default-refused cells pass.
///
/// # Errors
///
/// [`CausalError::Support`] when the cell is n/a or matches
/// `parity/support_closed.toml`.
pub fn refuse_if_not_applicable(cell: SupportCell) -> Result<CellStatus, CausalError> {
    match classify(cell) {
        CellStatus::NotApplicable { reason } => {
            Err(CausalError::Support { id: SupportRefusal::NotApplicable, message: reason })
        }
        CellStatus::Refused => match closed_reason(cell) {
            Some(reason) => {
                Err(CausalError::Support { id: SupportRefusal::Refused, message: reason })
            }
            None => Ok(CellStatus::Refused),
        },
        CellStatus::Licensed => Ok(CellStatus::Licensed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(
        query: &'static str,
        graph: &'static str,
        structure: &'static str,
        inference: &'static str,
        validation: &'static str,
    ) -> SupportCell {
        SupportCell { query, graph_class: graph, structure, inference, validation }
    }

    #[test]
    fn pulse_on_static_dag_is_not_applicable() {
        let status = classify(cell("PulseEffect", "Dag", "explicit", "Frequentist", "none"));
        assert!(matches!(status, CellStatus::NotApplicable { .. }));
    }

    #[test]
    fn average_effect_on_temporal_dag_is_not_applicable() {
        let status =
            classify(cell("AverageEffect", "TemporalDag", "explicit", "Frequentist", "none"));
        assert!(matches!(status, CellStatus::NotApplicable { .. }));
    }

    #[test]
    fn temporal_mediation_on_temporal_dag_is_not_n_a() {
        let status = classify(cell(
            "TemporalMediationEffect",
            "TemporalDag",
            "explicit",
            "Frequentist",
            "none",
        ));
        assert!(!matches!(status, CellStatus::NotApplicable { .. }));
    }

    #[test]
    fn dag_average_effect_frequentist_none_is_licensed() {
        assert_eq!(
            classify(cell("AverageEffect", "Dag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("AverageEffect", "Dag", "accepted", "Frequentist", "none")),
            CellStatus::Licensed
        );
    }

    #[test]
    fn dag_response_curve_frequentist_none_is_licensed() {
        assert_eq!(
            classify(cell("ResponseCurve", "Dag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("ResponseCurve", "Dag", "accepted", "Frequentist", "none")),
            CellStatus::Licensed
        );
    }

    #[test]
    fn response_curve_graph_posterior_is_not_applicable() {
        for graph in ["Dag", "Cpdag", "Pag", "Admg", "TemporalDag"] {
            let status =
                classify(cell("ResponseCurve", graph, "graph_posterior", "Frequentist", "none"));
            assert!(matches!(status, CellStatus::NotApplicable { .. }), "{graph}: {status:?}");
        }
    }

    #[test]
    fn pag_average_effect_is_recorded_refused_not_n_a() {
        assert_eq!(
            classify(cell("AverageEffect", "Pag", "explicit", "Frequentist", "none")),
            CellStatus::Refused
        );
        assert!(
            refuse_if_not_applicable(cell(
                "AverageEffect",
                "Pag",
                "explicit",
                "Frequentist",
                "none"
            ))
            .is_ok()
        );
    }

    #[test]
    fn closed_derivative_and_counterfactual_are_enforced() {
        let err =
            refuse_if_not_applicable(cell("Elasticity", "Dag", "explicit", "Frequentist", "none"))
                .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
        let err = refuse_if_not_applicable(cell(
            "Counterfactual",
            "Dag",
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
        let err = refuse_if_not_applicable(cell(
            "ResponseCurve",
            "Pag",
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
    }

    #[test]
    fn closed_path_and_distribution_on_accepted_are_enforced() {
        for query in ["PathSpecificEffect", "InterventionalDistribution"] {
            let status = classify(cell(query, "Dag", "accepted", "Frequentist", "none"));
            assert_eq!(status, CellStatus::Refused, "{query}");
            let err =
                refuse_if_not_applicable(cell(query, "Dag", "accepted", "Frequentist", "none"))
                    .unwrap_err();
            assert!(
                err.to_string().starts_with(
                    "refused: Path and distribution queries are licensed only as explicit"
                ),
                "{query}: {err}"
            );
        }
    }

    #[test]
    fn closed_mediation_and_sustained_are_enforced() {
        let err = refuse_if_not_applicable(cell(
            "MediationEffect",
            "Dag",
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
        let err = refuse_if_not_applicable(cell(
            "SustainedEffect",
            "TemporalDag",
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
        for query in ["TransportQuery", "InterferenceQuery"] {
            let err = refuse_if_not_applicable(cell(
                query,
                "Dag",
                "explicit",
                "Frequentist",
                "none",
            ))
            .unwrap_err();
            assert!(err.to_string().starts_with("refused:"), "{query}: {err}");
        }
    }
}
