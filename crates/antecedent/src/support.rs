//! Public support-matrix lookup.
//!
//! Axes, n/a predicates, and licensed cells are generated from
//! `parity/support_*.toml`. Dispatch refuses [`CellStatus::NotApplicable`]
//! now. Cells in `parity/support_closed.toml` raise [`SupportRefusal::Refused`].
//! Remaining default-refused cells still run until licensed or closed.

use antecedent_core::{CausalQuery, DerivativeScale, ResponseFunctional, TemporalPolicy};

use antecedent_graph::{Admg, Dag, DenseNodeId, Pag, TemporalDag};

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
        CausalQuery::TemporalEffect(q) => match &q.policy {
            TemporalPolicy::Pulse { .. } => Some("PulseEffect"),
            TemporalPolicy::Sustained { .. } => Some("SustainedEffect"),
            // A dynamic rule is classified by its evaluated schedule shape so it
            // cannot dodge the matrix: one active step rides the Pulse cell
            // (the engine estimates it as a rule-tagged pulse), any longer
            // schedule is a sustained intervention and hits the Sustained
            // closure. Mirrors `refuse_multi_step_schedule` in
            // antecedent-estimate, which already refuses multi-step Dynamic.
            TemporalPolicy::Dynamic { active_at, .. } => {
                if active_at.len() == 1 {
                    Some("PulseEffect")
                } else {
                    Some("SustainedEffect")
                }
            }
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

/// Whether an ADMG carries no bidirected edges — the ADMG cell would then be a
/// fiction, since the engine has nothing left for the ADMG-specific path to do.
/// Mirrors `admg_has_bidirected` (`analysis/execute/support.rs`), which the engine
/// itself consults at `dispatch.rs` (`GraphClass::Admg` arm) and `compile.rs`
/// (`GraphClass::Admg` arm) to choose between `execute_admg`/the static-DAG
/// completion. That helper lives in a private module unreachable from here, so this
/// re-derives the same check directly off `Admg`'s public API rather than reaching
/// across the module boundary.
fn admg_is_bidirected_free(admg: &Admg) -> bool {
    (0..admg.node_count()).all(|i| {
        let id = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        admg.bidirected_neighbors(id).is_empty()
    })
}

/// The graph class the engine actually dispatches on, for support-matrix
/// classification. Collapsing a non-Dag class to `Dag`/`TemporalDag` here is
/// classification only — it never changes what `Study::compile`/`execute` or
/// `identify_with` run; those still match on `AcceptedGraph::class()` directly.
/// This exists so the cell an honest caller sees matches the engine's real
/// behavior instead of the raw class tag, per each collapse's one-line
/// justification below. Each collapse is scoped to exactly the query family
/// `analysis/execute/compile.rs` wires it for — applying it more broadly would
/// license cells the engine cannot actually run (see the `ResponseCurve` note).
///
/// - **ADMG with no bidirected edges, under `AverageEffect`**: `compile.rs`'s
///   `(AverageEffect, GraphClass::Admg)` arm and `dispatch.rs`'s
///   `GraphClass::Admg` arm both branch on `admg_has_bidirected` and run the
///   *static DAG* path when it is false — the Dag cell's license is the
///   honest claim. `compile.rs` wires no other query against
///   `GraphClass::Admg`: `CausalQuery::Response` on an ADMG (bidirected or
///   not) hits compile.rs's wildcard `Err("unsupported data/graph/query
///   combination")` because `dispatch.rs`'s Response arm requires a literal
///   `self.graph.as_dag()`, which an ADMG-shaped `AcceptedGraph` never
///   satisfies — so the collapse must not (and does not) fire there.
/// - **Cpdag, under `AverageEffect`**: `AcceptedGraph::cpdag` /
///   `CpdagReview::into_accepted` already refuse any Cpdag carrying
///   undirected or conflict marks at *construction* time (`accepted.rs`), so
///   every `GraphClass::Cpdag` this function ever sees is already fully
///   oriented; `compile.rs`'s lone `GraphClass::Cpdag` arm always completes
///   it via `try_into_dag`. This re-derives that fact structurally (rather
///   than trusting the invariant) so a future relaxation of the accept gate
///   cannot silently misclassify a partially-oriented Cpdag as Dag. Scoped to
///   `AverageEffect` for the same reason as the ADMG case: it is the only
///   query `compile.rs` wires for `GraphClass::Cpdag` — `CausalQuery::Response`
///   on a Cpdag hits the same compile-time wildcard refusal as above, which is
///   exactly why `ResponseCurve` stays closed on Cpdag (`support_closed.toml`)
///   even though a fully-oriented Cpdag would otherwise look Dag-shaped.
/// - **`TemporalCpdag` / `TemporalPag`, under `TemporalEffect`**: the identical
///   accept-time invariant (`accepted.rs`) guarantees these are always
///   complete; `compile.rs`'s `TemporalEffect × {TemporalCpdag, TemporalPag}`
///   arms always complete them to a `TemporalDag` via `try_into_temporal_dag`.
///   Scoped to `TemporalEffect` because that is the only query `compile.rs`
///   wires for these two temporal classes — e.g. temporal
///   `CausalQuery::Mediation` only has a `GraphClass::TemporalDag` arm, so a
///   TemporalCpdag/TemporalPag stays uncollapsed (and thus not misclassified)
///   for that query.
///
/// Not collapsed: a static [`Pag`]'s circle marks are information the
/// class-aware generalized-adjustment identifier is built to consume, never
/// incompleteness ([`AcceptedGraph::pag`] is infallible for exactly this
/// reason), so there is no Dag-equivalent Pag case.
#[must_use]
pub(crate) fn effective_graph_class(graph: &AcceptedGraph, query: &CausalQuery) -> GraphClass {
    match (graph.class(), query) {
        (GraphClass::Admg, CausalQuery::AverageEffect(_)) => {
            let admg = graph.as_admg().expect("class() == Admg implies as_admg() is Some");
            if admg_is_bidirected_free(admg) { GraphClass::Dag } else { GraphClass::Admg }
        }
        (GraphClass::Cpdag, CausalQuery::AverageEffect(_)) => {
            let cpdag = graph.as_cpdag().expect("class() == Cpdag implies as_cpdag() is Some");
            if cpdag.try_into_dag().is_ok() { GraphClass::Dag } else { GraphClass::Cpdag }
        }
        (GraphClass::TemporalCpdag, CausalQuery::TemporalEffect(_)) => {
            let cpdag = graph
                .as_temporal_cpdag()
                .expect("class() == TemporalCpdag implies as_temporal_cpdag() is Some");
            if cpdag.try_into_temporal_dag().is_ok() {
                GraphClass::TemporalDag
            } else {
                GraphClass::TemporalCpdag
            }
        }
        (GraphClass::TemporalPag, CausalQuery::TemporalEffect(_)) => {
            let pag = graph
                .as_temporal_pag()
                .expect("class() == TemporalPag implies as_temporal_pag() is Some");
            if pag.try_into_temporal_dag().is_ok() {
                GraphClass::TemporalDag
            } else {
                GraphClass::TemporalPag
            }
        }
        (class, _) => class,
    }
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
    fn dynamic_policy_classifies_by_schedule_shape() {
        use antecedent_core::{DynamicRuleId, TemporalEffectQuery, TemporalPolicy, VariableId};
        let make = |active_at: &[i32]| {
            let mut q =
                TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
            q.policy = TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), active_at.to_vec());
            CausalQuery::TemporalEffect(q)
        };
        assert_eq!(query_axis_name(&make(&[0]), GraphClass::TemporalDag), Some("PulseEffect"));
        // A multi-step dynamic schedule is a sustained intervention; it must
        // hit the Sustained closure, not bypass the matrix.
        assert_eq!(
            query_axis_name(&make(&[0, 1]), GraphClass::TemporalDag),
            Some("SustainedEffect")
        );
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
            let err =
                refuse_if_not_applicable(cell(query, "Dag", "explicit", "Frequentist", "none"))
                    .unwrap_err();
            assert!(err.to_string().starts_with("refused:"), "{query}: {err}");
        }
    }

    // -- `effective_graph_class` --------------------------------------------

    use antecedent_core::{AverageEffectQuery, TemporalEffectQuery, VariableId};
    use antecedent_graph::{Cpdag, TemporalCpdag, TemporalPag};

    fn ate_query() -> CausalQuery {
        CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ))
    }

    fn pulse_query() -> CausalQuery {
        CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            1.0,
        ))
    }

    fn distribution_query() -> CausalQuery {
        use antecedent_core::{Intervention, InterventionalDistributionQuery, Value};
        CausalQuery::Distribution(InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        ))
    }

    /// (a) An ADMG with no bidirected edges collapses to Dag under `AverageEffect`,
    /// and the resulting cell is Licensed — the same Dag/explicit/Frequentist/none
    /// row every supplied-DAG study licenses.
    #[test]
    fn admg_without_bidirected_collapses_to_dag_under_average_effect() {
        let mut admg = Admg::with_variables(2);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::from(admg);
        assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Dag);

        let sc = support_cell(
            &ate_query(),
            effective_graph_class(&graph, &ate_query()),
            StructureSource::Explicit,
            &InferenceMode::Frequentist,
            RefuteSuite::None,
        )
        .unwrap();
        assert_eq!(classify(sc), CellStatus::Licensed);
    }

    /// (b) An ADMG *with* a bidirected edge does not collapse: the ADMG path is
    /// still live (dispatch runs `execute_admg`, not the static-DAG completion),
    /// so the cell stays the Admg cell — default-refused, not licensed.
    #[test]
    fn admg_with_bidirected_edge_does_not_collapse() {
        let mut admg = Admg::with_variables(2);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::from(admg);
        assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Admg);

        let sc = support_cell(
            &ate_query(),
            effective_graph_class(&graph, &ate_query()),
            StructureSource::Explicit,
            &InferenceMode::Frequentist,
            RefuteSuite::None,
        )
        .unwrap();
        assert_eq!(classify(sc), CellStatus::Refused);
    }

    /// The ADMG collapse is scoped to `AverageEffect`: `compile.rs` wires no other
    /// query against `GraphClass::Admg` (`CausalQuery::Distribution` requires a
    /// literal `as_dag()` and errors for any ADMG, bidirected or not), so applying
    /// the collapse there would license a cell the engine cannot run.
    #[test]
    fn admg_collapse_does_not_apply_outside_average_effect() {
        let mut admg = Admg::with_variables(2);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::from(admg);
        assert_eq!(effective_graph_class(&graph, &distribution_query()), GraphClass::Admg);
    }

    /// (c) A fully-oriented Cpdag collapses to Dag under `AverageEffect`, and the
    /// resulting cell is Licensed.
    #[test]
    fn fully_oriented_cpdag_collapses_to_dag_under_average_effect() {
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::cpdag(cpdag).unwrap();
        assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Dag);

        let sc = support_cell(
            &ate_query(),
            effective_graph_class(&graph, &ate_query()),
            StructureSource::Explicit,
            &InferenceMode::Frequentist,
            RefuteSuite::None,
        )
        .unwrap();
        assert_eq!(classify(sc), CellStatus::Licensed);
    }

    /// (c, other half) A partially-oriented Cpdag can never reach
    /// `effective_graph_class` in the first place: `AcceptedGraph::cpdag` is the
    /// review gate, and it structurally refuses any Cpdag carrying undirected or
    /// conflict marks *at construction*, before there is a `GraphClass::Cpdag`
    /// value to classify. This is the invariant `effective_graph_class`'s Cpdag
    /// arm documents and re-derives rather than assumes.
    #[test]
    fn partially_oriented_cpdag_is_refused_at_the_accept_gate_not_at_classification() {
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let err = AcceptedGraph::cpdag(cpdag).unwrap_err();
        assert!(matches!(err, CausalError::ReviewRequired { .. }), "{err}");
    }

    /// The Cpdag collapse is scoped to `AverageEffect` for the same reason as the
    /// ADMG collapse: `compile.rs` wires no other query against `GraphClass::Cpdag`.
    /// This is exactly why `ResponseCurve` stays closed on Cpdag even for a
    /// fully-oriented one (`support_closed.toml`'s Pag/Cpdag/Admg rule) — the
    /// collapse must not un-close it.
    #[test]
    fn cpdag_collapse_does_not_apply_outside_average_effect() {
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::cpdag(cpdag).unwrap();
        assert_eq!(effective_graph_class(&graph, &distribution_query()), GraphClass::Cpdag);
    }

    /// (d) A complete `TemporalCpdag` collapses to `TemporalDag` under `TemporalEffect`.
    #[test]
    fn complete_temporal_cpdag_collapses_to_temporal_dag_under_temporal_effect() {
        let mut cpdag = TemporalCpdag::empty();
        let a =
            cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b = cpdag
            .add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
        cpdag.insert_directed(a, b).unwrap();
        let graph = AcceptedGraph::temporal_cpdag(cpdag).unwrap();
        assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalDag);
    }

    /// An incomplete `TemporalCpdag` is refused at `AcceptedGraph::temporal_cpdag`
    /// itself, mirroring the static Cpdag invariant.
    #[test]
    fn incomplete_temporal_cpdag_is_refused_at_the_accept_gate() {
        let mut cpdag = TemporalCpdag::empty();
        let a =
            cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b = cpdag
            .add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
        cpdag.insert_undirected(a, b).unwrap();
        assert!(AcceptedGraph::temporal_cpdag(cpdag).is_err());
    }

    /// (d) A complete `TemporalPag` (no circle marks) collapses to `TemporalDag` under
    /// `TemporalEffect`.
    #[test]
    fn complete_temporal_pag_collapses_to_temporal_dag_under_temporal_effect() {
        use antecedent_graph::TemporalPag;
        let mut pag = TemporalPag::empty();
        let a = pag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b =
            pag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        pag.insert_directed(a, b).unwrap();
        let graph = AcceptedGraph::temporal_pag(pag).unwrap();
        assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalDag);
    }

    /// A `TemporalPag` with an unresolved circle mark is refused at
    /// `AcceptedGraph::temporal_pag` itself.
    #[test]
    fn temporal_pag_with_circle_mark_is_refused_at_the_accept_gate() {
        let mut pag = TemporalPag::empty();
        let a = pag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b =
            pag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        pag.insert_circle_arrow(a, b).unwrap();
        assert!(AcceptedGraph::temporal_pag(pag).is_err());
    }

    /// The temporal collapse is scoped to `TemporalEffect`: `compile.rs` wires no
    /// other query against `GraphClass::TemporalCpdag`/`TemporalPag` (temporal
    /// `Mediation` only has a `GraphClass::TemporalDag` arm), so it must not fire
    /// for e.g. a static query landing on a temporal class.
    #[test]
    fn temporal_collapse_does_not_apply_outside_temporal_effect() {
        let mut cpdag = TemporalCpdag::empty();
        let a =
            cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b = cpdag
            .add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
        cpdag.insert_directed(a, b).unwrap();
        let graph = AcceptedGraph::temporal_cpdag(cpdag).unwrap();
        // AverageEffect on a temporal class is typed-impossible regardless of
        // collapse, but the collapse itself must still not fire here.
        assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::TemporalCpdag);
    }

    /// A static Pag never collapses: circle marks are information the
    /// class-aware generalized-adjustment identifier consumes, not incompleteness.
    #[test]
    fn pag_never_collapses() {
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::pag(pag);
        assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Pag);
    }
}
