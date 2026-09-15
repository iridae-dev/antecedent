//! Public support-matrix lookup.
//!
//! Axes, n/a predicates, licensed cells, refusal-reason rules, and the legacy
//! allowlist source are generated from `parity/support_*.toml`. Runtime states
//! are three: licensed
//! (`parity/support_licensed.toml`), n/a (`parity/support_n_a.toml`, typed
//! impossibility, [`SupportRefusal::NotApplicable`]), or not licensed
//! ([`SupportRefusal::Refused`]). `parity/support_closed.toml` is the reason
//! table for refused cells, not a fourth state; its filename is retained for
//! compatibility. `allowed_unlicensed` is also retained as a wire value for
//! older artifacts and clients, but the 0.9 gate requires
//! `parity/support_allowlist.toml` to have zero active entries. Any refused
//! cell without a reason uses the shared default-refusal message.

use antecedent_core::{
    CausalQuery, DerivativeScale, LicensedNeighbor, PremiseChange, ResponseFunctional,
    TemporalPolicy,
};

use antecedent_graph::{Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::analysis::RefuteSuite;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::support_matrix_data::{ALLOWED_RULES, CLOSED_RULES, LICENSED, NA_RULES};

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

explicit_graph_input!(Dag, Admg, Pag, Cpdag, TemporalDag, TemporalCpdag, TemporalPag);

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
    /// Retained `allowed_unlicensed` compatibility status.
    ///
    /// No 0.9 matrix cell may produce this status; the variant remains for wire
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
            // refusal. Mirrors `refuse_multi_step_schedule` in
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
        CausalQuery::Response(q) => {
            let temporal_graph = matches!(
                graph_class,
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            );
            // Static response on a temporal graph (and temporal attachment on a
            // static graph) are typed impossibilities, not matrix cells.
            if q.is_temporal() != temporal_graph {
                return None;
            }
            match &q.functional {
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
            }
        }
        CausalQuery::Transport(_) => Some("TransportQuery"),
        CausalQuery::Interference(_) => Some("InterferenceQuery"),
        CausalQuery::AnomalyAttribution(_) => Some("AnomalyAttribution"),
        CausalQuery::ChangeAttribution(_) => Some("ChangeAttribution"),
        // Mechanism/unit-change and any later CausalQuery variant stay off the axis.
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
    support_cell_named(query, graph_class.as_str(), structure, inference, refute)
}

/// Like [`support_cell`], but `graph_class` may be a classification-only axis
/// (`CoDetermined`, `Unknown`) that is not a [`GraphClass`] variant.
#[must_use]
pub fn support_cell_named(
    query: &CausalQuery,
    graph_class: &'static str,
    structure: StructureSource,
    inference: &InferenceMode,
    refute: RefuteSuite,
) -> Option<SupportCell> {
    Some(SupportCell {
        query: query_axis_name(query, naming_class(graph_class)?)?,
        graph_class,
        structure: structure.as_str(),
        inference: inference_axis(inference),
        validation: validation_axis(refute),
    })
}

fn naming_class(matrix: &str) -> Option<GraphClass> {
    Some(match matrix {
        "Dag" | "CoDetermined" | "Unknown" => GraphClass::Dag,
        "Admg" => GraphClass::Admg,
        "Cpdag" => GraphClass::Cpdag,
        "Pag" => GraphClass::Pag,
        "TemporalDag" => GraphClass::TemporalDag,
        "TemporalCpdag" => GraphClass::TemporalCpdag,
        "TemporalPag" => GraphClass::TemporalPag,
        _ => return None,
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
///   `GraphClass::Admg` arm both branch on [`Admg::has_bidirected`] and run the
///   *static DAG* path when it is false — the Dag cell's license is the
///   honest claim. `compile.rs` wires no other query against
///   `GraphClass::Admg` except `CoDetermined` joint cells: a bare ADMG
///   `CausalQuery::Response` still hits compile.rs's wildcard because
///   `dispatch.rs` requires Dag/Cpdag/Pag, or a `CoDetermined` tier closure.
///   The collapse must not fire for a supplied Admg response. `CoDetermined`
///   / `Unknown` are classification-only matrix extras ([`matrix_graph_class`]),
///   not [`GraphClass`] variants.
/// - **Cpdag, under `AverageEffect`**: undirected marks are MEC information.
///   Completing a CPDAG to a DAG is a `Dag` cell only when the *caller*
///   supplies a `Dag`. A supplied `Cpdag` stays `Cpdag` even when it has a
///   unique orientation.
/// - **`TemporalCpdag` / `TemporalPag`**: supplied classes remain class-shaped,
///   including fully oriented inputs. A directed MAG edge is not a DAG edge:
///   visibility must still be certified before adjustment.
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
            if admg.has_bidirected() { GraphClass::Admg } else { GraphClass::Dag }
        }
        (GraphClass::Cpdag, CausalQuery::AverageEffect(_)) => GraphClass::Cpdag,
        (class, _) => class,
    }
}

/// Matrix graph axis, including classification-only tier-rule extras.
///
/// A `TieredBackground` is not a supplied `Admg`/`Pag`. `CoDetermined` is a known
/// closure; Unknown is a two-scenario envelope. Classification only — execute
/// still materializes the closure ADMG or PAG.
#[must_use]
pub(crate) fn matrix_graph_class(
    graph: &AcceptedGraph,
    query: &CausalQuery,
    tiered: Option<&antecedent_graph::TieredBackground>,
) -> &'static str {
    match tiered.map(|b| b.within_tier) {
        Some(antecedent_graph::WithinTier::CoDetermined) => "CoDetermined",
        Some(antecedent_graph::WithinTier::Unknown) => "Unknown",
        None => effective_graph_class(graph, query).as_str(),
    }
}

/// Closed-rule or default refusal text for a refused cell.
#[must_use]
pub fn refused_message(cell: SupportCell) -> &'static str {
    refusal_reason(cell).unwrap_or(UNLICENSED)
}

fn refusal_reason(cell: SupportCell) -> Option<&'static str> {
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

/// Historical reason for an `allowed_unlicensed` compatibility entry, if any.
///
/// The 0.9 gate requires `parity/support_allowlist.toml` to remain empty.
#[must_use]
pub fn allowed_reason(cell: SupportCell) -> Option<&'static str> {
    allowed_rule(cell).map(|rule| rule.reason)
}

/// Historical parent family for an `allowed_unlicensed` compatibility entry.
///
/// The 0.9 gate requires `parity/support_allowlist.toml` to remain empty.
#[must_use]
pub fn allowed_parent(cell: SupportCell) -> Option<&'static str> {
    allowed_rule(cell).map(|rule| rule.parent)
}

fn allowed_rule(cell: SupportCell) -> Option<&'static crate::support_matrix_data::AllowedRule> {
    ALLOWED_RULES.iter().find(|rule| {
        axis_in(rule.queries, cell.query)
            && axis_in(rule.graph_classes, cell.graph_class)
            && axis_in(rule.structures, cell.structure)
            && axis_in(rule.inferences, cell.inference)
            && axis_in(rule.validations, cell.validation)
    })
}

/// The single static message every unmatched `Refused` cell reports. Cell
/// coordinates vary at runtime — the caller's [`CausalError::Support`] already
/// carries the offending cell in its own context — so this stays a fixed,
/// shared string rather than one formatted per cell.
const UNLICENSED: &str =
    "cell is not licensed (parity/support_licensed.toml) and is not n/a; it is refused.";

/// Format a matrix coordinate as `query:graph:structure:inference:validation`.
#[must_use]
pub fn cell_coordinate(cell: SupportCell) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        cell.query, cell.graph_class, cell.structure, cell.inference, cell.validation
    )
}

/// Reconstruct a cell from [`cell_coordinate`] by interning against licensed axes.
#[must_use]
pub fn support_cell_from_coordinate(coordinate: &str) -> Option<SupportCell> {
    let mut parts = coordinate.split(':');
    let query = intern_licensed_axis(parts.next()?, |cell| cell.query)?;
    let graph_class = intern_licensed_axis(parts.next()?, |cell| cell.graph_class)?;
    let structure = intern_licensed_axis(parts.next()?, |cell| cell.structure)?;
    let inference = intern_licensed_axis(parts.next()?, |cell| cell.inference)?;
    let validation = intern_licensed_axis(parts.next()?, |cell| cell.validation)?;
    if parts.next().is_some() {
        return None;
    }
    Some(SupportCell { query, graph_class, structure, inference, validation })
}

fn intern_licensed_axis(
    value: &str,
    pick: impl Fn(&crate::support_matrix_data::LicensedCell) -> &'static str,
) -> Option<&'static str> {
    LICENSED.iter().map(pick).find(|&axis| axis == value)
}

const MAX_LICENSED_NEIGHBORS: usize = 8;

/// Licensed alternatives that keep the requested graph class.
///
/// Neighbors are explicit changed-premise suggestions, never automatic
/// fallbacks. Relabeling a class (`Cpdag` → `Dag`) is not a neighbor. An
/// already-licensed cell has no neighbors.
#[must_use]
pub fn licensed_neighbors(cell: SupportCell) -> Vec<LicensedNeighbor> {
    if matches!(classify(cell), CellStatus::Licensed) {
        return Vec::new();
    }
    let mut scored: Vec<(Vec<PremiseChange>, crate::support_matrix_data::LicensedCell)> = LICENSED
        .iter()
        .copied()
        .filter_map(|row| {
            if row.graph_class != cell.graph_class {
                return None;
            }
            let changed = neighbor_changes(cell, &row);
            (!changed.is_empty()).then_some((changed, row))
        })
        .collect();
    scored.sort_by(|left, right| {
        left.0
            .len()
            .cmp(&right.0.len())
            .then_with(|| left.1.query.cmp(right.1.query))
            .then_with(|| left.1.structure.cmp(right.1.structure))
            .then_with(|| left.1.inference.cmp(right.1.inference))
            .then_with(|| left.1.validation.cmp(right.1.validation))
    });
    scored.dedup_by(|left, right| {
        left.1.query == right.1.query
            && left.1.structure == right.1.structure
            && left.1.inference == right.1.inference
            && left.1.validation == right.1.validation
    });
    scored
        .into_iter()
        .take(MAX_LICENSED_NEIGHBORS)
        .map(|(changed, row)| {
            let required_action = neighbor_action(&changed);
            LicensedNeighbor::new(
                cell_coordinate(support_cell_from_licensed(&row)),
                changed,
                required_action,
            )
        })
        .collect()
}

fn support_cell_from_licensed(row: &crate::support_matrix_data::LicensedCell) -> SupportCell {
    SupportCell {
        query: row.query,
        graph_class: row.graph_class,
        structure: row.structure,
        inference: row.inference,
        validation: row.validation,
    }
}

fn neighbor_changes(
    cell: SupportCell,
    row: &crate::support_matrix_data::LicensedCell,
) -> Vec<PremiseChange> {
    [
        ("query", cell.query, row.query),
        ("structure", cell.structure, row.structure),
        ("inference", cell.inference, row.inference),
        ("validation", cell.validation, row.validation),
    ]
    .into_iter()
    .filter(|(_, from, to)| from != to)
    .map(|(axis, from, to)| PremiseChange::new(axis, from, to))
    .collect()
}

fn neighbor_action(changed: &[PremiseChange]) -> &'static str {
    match changed {
        [change] if change.axis.as_ref() == "query" => {
            "change the query; do not relabel the graph class"
        }
        [change] if change.axis.as_ref() == "inference" => "change the inference contract",
        [change] if change.axis.as_ref() == "validation" => "change the validation suite",
        [change] if change.axis.as_ref() == "structure" => {
            "change how structure is supplied; this is not a graph-class relabel"
        }
        _ => "change the listed premises; do not relabel the graph class",
    }
}

/// Refuse n/a cells and every unlicensed meaningful cell. Licensed cells pass.
///
/// The `allowed_unlicensed` branch is retained for wire compatibility, but the
/// 0.9 gate requires zero active allowlist entries.
///
/// # Errors
///
/// [`CausalError::Support`] when the cell is n/a or refused. A refusal uses the
/// reason from legacy-named `parity/support_closed.toml` when one matches,
/// otherwise it uses the shared default-refusal message.
pub fn refuse_if_not_applicable(cell: SupportCell) -> Result<CellStatus, CausalError> {
    match classify(cell) {
        CellStatus::NotApplicable { reason } => {
            Err(CausalError::Support { id: SupportRefusal::NotApplicable, message: reason })
        }
        CellStatus::Refused => {
            if let Some(reason) = refusal_reason(cell) {
                return Err(CausalError::Support { id: SupportRefusal::Refused, message: reason });
            }
            if let Some(rule) = allowed_rule(cell) {
                return Ok(CellStatus::Allowlisted { reason: rule.reason, parent: rule.parent });
            }
            Err(CausalError::Support { id: SupportRefusal::Refused, message: UNLICENSED })
        }
        CellStatus::Licensed => Ok(CellStatus::Licensed),
        CellStatus::Allowlisted { .. } => {
            unreachable!(
                "classify never returns Allowlisted; refuse_if_not_applicable constructs it"
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::DenseNodeId;

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
    fn licensed_neighbors_keep_graph_class_and_never_relabel() {
        let pulse = cell("PulseEffect", "Dag", "explicit", "Frequentist", "none");
        let neighbors = licensed_neighbors(pulse);
        assert!(!neighbors.is_empty(), "Pulse on Dag should name a static licensed neighbor");
        assert!(
            neighbors.iter().any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Dag:")),
            "{neighbors:?}"
        );
        assert!(
            neighbors.iter().all(|n| n.coordinate.as_ref().split(':').nth(1) == Some("Dag")),
            "graph-class relabel is not a neighbor: {neighbors:?}"
        );

        let cpdag = cell("PulseEffect", "Cpdag", "explicit", "Frequentist", "none");
        let neighbors = licensed_neighbors(cpdag);
        assert!(
            neighbors.iter().all(|n| n.coordinate.as_ref().split(':').nth(1) == Some("Cpdag")),
            "Cpdag must not recommend a Dag relabel: {neighbors:?}"
        );
        assert!(
            neighbors.iter().any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Cpdag:"))
        );

        let refused = cell("ConditionalEffect", "Admg", "explicit", "Frequentist", "none");
        let neighbors = licensed_neighbors(refused);
        assert!(neighbors.iter().any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Admg:")));
        assert!(neighbors.iter().all(|n| !n.coordinate.as_ref().contains(":Dag:")));

        assert!(
            licensed_neighbors(cell("AverageEffect", "Dag", "explicit", "Frequentist", "none"))
                .is_empty()
        );
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
    fn response_curve_graph_posterior_is_licensed_only_for_static_dag_atoms() {
        for graph in ["Dag", "Cpdag", "Pag", "Admg", "TemporalDag"] {
            let status =
                classify(cell("ResponseCurve", graph, "graph_posterior", "Frequentist", "none"));
            if graph == "Dag" {
                assert_eq!(status, CellStatus::Licensed);
            } else {
                assert_eq!(status, CellStatus::Refused, "{graph}: {status:?}");
            }
        }
    }

    #[test]
    fn response_curve_cheap_on_dag_is_not_applicable() {
        let status = classify(cell("ResponseCurve", "Dag", "explicit", "Frequentist", "cheap"));
        assert!(matches!(status, CellStatus::NotApplicable { .. }), "{status:?}");
    }

    #[test]
    fn dag_intervention_response_cheap_and_full_are_licensed() {
        for structure in ["explicit", "accepted"] {
            for validation in ["cheap", "full"] {
                let status = classify(cell(
                    "InterventionResponse",
                    "Dag",
                    structure,
                    "Frequentist",
                    validation,
                ));
                assert_eq!(status, CellStatus::Licensed, "{structure}/{validation}");
                refuse_if_not_applicable(cell(
                    "InterventionResponse",
                    "Dag",
                    structure,
                    "Frequentist",
                    validation,
                ))
                .unwrap();
            }
        }
        for graph in ["TemporalDag", "Cpdag", "Pag", "TemporalCpdag", "TemporalPag"] {
            let status =
                classify(cell("InterventionResponse", graph, "explicit", "Frequentist", "cheap"));
            assert!(matches!(status, CellStatus::NotApplicable { .. }), "{graph}: {status:?}");
        }
    }

    #[test]
    fn function_valued_and_class_response_cheap_are_not_applicable() {
        for graph in ["TemporalCpdag", "TemporalPag"] {
            let status = classify(cell("ResponseCurve", graph, "explicit", "Frequentist", "cheap"));
            assert!(matches!(status, CellStatus::NotApplicable { .. }), "{graph}: {status:?}");
        }
        let status =
            classify(cell("ResponseCurve", "Dag", "graph_posterior", "Frequentist", "full"));
        assert!(matches!(status, CellStatus::NotApplicable { .. }), "{status:?}");
        let status =
            classify(cell("InterventionResponse", "Dag", "graph_posterior", "Bayesian", "cheap"));
        assert!(matches!(status, CellStatus::NotApplicable { .. }), "{status:?}");
    }

    #[test]
    fn frequentist_dbn_and_cpdag_mediation_suites_are_licensed() {
        for validation in ["cheap", "full"] {
            for query in ["PulseEffect", "SustainedEffect"] {
                let c = cell(query, "TemporalDag", "graph_posterior", "Frequentist", validation);
                assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
            }
            for structure in ["explicit", "accepted"] {
                let c = cell(
                    "TemporalMediationEffect",
                    "TemporalCpdag",
                    structure,
                    "Frequentist",
                    validation,
                );
                assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
            }
            let c =
                cell("InterventionResponse", "Dag", "graph_posterior", "Frequentist", validation);
            assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
        }
        let status = classify(cell(
            "PulseEffect",
            "TemporalCpdag",
            "graph_posterior",
            "Frequentist",
            "none",
        ));
        assert_eq!(status, CellStatus::Refused);
        assert!(
            refusal_reason(cell(
                "PulseEffect",
                "TemporalCpdag",
                "graph_posterior",
                "Frequentist",
                "none"
            ))
            .unwrap()
            .contains("class-aware combiner")
        );
    }

    #[test]
    fn pag_average_effect_is_licensed() {
        assert_eq!(
            classify(cell("AverageEffect", "Pag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("AverageEffect", "Cpdag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("AverageEffect", "Cpdag", "accepted", "Bayesian", "full")),
            CellStatus::Licensed
        );
        assert_eq!(
            refuse_if_not_applicable(cell(
                "AverageEffect",
                "Pag",
                "explicit",
                "Frequentist",
                "none"
            ))
            .unwrap(),
            CellStatus::Licensed
        );
    }

    #[test]
    fn closed_derivative_and_counterfactual_are_enforced() {
        let err =
            refuse_if_not_applicable(cell("Elasticity", "Pag", "explicit", "Frequentist", "none"))
                .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
        let err = refuse_if_not_applicable(cell(
            "Counterfactual",
            "Pag",
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
        assert_eq!(
            classify(cell("ResponseCurve", "Pag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
    }

    #[test]
    fn remaining_closed_dag_cells_are_named() {
        let cases = [
            (
                "AverageDerivative",
                "graph_posterior",
                "Bayesian",
                "none",
                "Graph-posterior derivative mixtures are not staged",
            ),
            ("Counterfactual", "graph_posterior", "Frequentist", "none", "graph-posterior"),
            (
                "Counterfactual",
                "explicit",
                "Frequentist",
                "cheap",
                "Counterfactual cheap/full are not licensed",
            ),
        ];
        for (query, structure, inference, validation, needle) in cases {
            let err =
                refuse_if_not_applicable(cell(query, "Dag", structure, inference, validation))
                    .unwrap_err();
            let text = err.to_string();
            assert!(text.starts_with("refused:"), "{query}: {text}");
            assert!(text.contains(needle), "{query}: {text}");
        }
        assert_eq!(
            classify(cell("PointDerivative", "Dag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("MediationEffect", "Dag", "accepted", "Frequentist", "full")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("Counterfactual", "Dag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("Counterfactual", "Dag", "accepted", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("Counterfactual", "Dag", "accepted", "Bayesian", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("AnomalyAttribution", "Dag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("ChangeAttribution", "Dag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("TransportQuery", "Admg", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("InterferenceQuery", "Dag", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
    }

    #[test]
    fn accepted_path_and_distribution_are_licensed() {
        for query in ["PathSpecificEffect", "InterventionalDistribution"] {
            for validation in ["none", "cheap", "full"] {
                let c = cell(query, "Dag", "accepted", "Frequentist", validation);
                assert_eq!(classify(c), CellStatus::Licensed);
                refuse_if_not_applicable(c).unwrap();
            }
            assert_eq!(
                classify(cell(query, "Dag", "graph_posterior", "Frequentist", "none")),
                CellStatus::Refused
            );
        }
    }

    #[test]
    fn closed_mediation_is_enforced() {
        let err = refuse_if_not_applicable(cell(
            "MediationEffect",
            "Pag",
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(err.to_string().starts_with("refused:"), "{err}");
        for query in ["TransportQuery", "InterferenceQuery"] {
            let err =
                refuse_if_not_applicable(cell(query, "Pag", "explicit", "Frequentist", "none"))
                    .unwrap_err();
            assert!(err.to_string().starts_with("refused:"), "{query}: {err}");
        }
    }

    #[test]
    fn licensed_pulse_temporal_class_is_open() {
        for query in ["PulseEffect", "SustainedEffect"] {
            for graph in ["TemporalCpdag", "TemporalPag"] {
                for structure in ["explicit", "accepted"] {
                    let status = classify(cell(query, graph, structure, "Frequentist", "none"));
                    assert_eq!(status, CellStatus::Licensed, "{query}/{graph}/{structure}");
                    refuse_if_not_applicable(cell(query, graph, structure, "Frequentist", "none"))
                        .unwrap();
                    for validation in ["none", "cheap", "full"] {
                        assert_eq!(
                            refuse_if_not_applicable(cell(
                                query, graph, structure, "Bayesian", validation
                            ))
                            .unwrap(),
                            CellStatus::Licensed,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn licensed_pulse_and_sustained_temporal_dag_are_open() {
        for query in ["PulseEffect", "SustainedEffect"] {
            for structure in ["explicit", "accepted"] {
                for inference in ["Frequentist", "Bayesian"] {
                    let status = classify(cell(query, "TemporalDag", structure, inference, "none"));
                    assert_eq!(status, CellStatus::Licensed, "{query}/{structure}/{inference}");
                    refuse_if_not_applicable(cell(
                        query,
                        "TemporalDag",
                        structure,
                        inference,
                        "none",
                    ))
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn closed_conditional_effect_off_dag_is_enforced() {
        let status = classify(cell("ConditionalEffect", "Admg", "accepted", "Frequentist", "none"));
        assert_eq!(status, CellStatus::Refused, "Admg");
        let err = refuse_if_not_applicable(cell(
            "ConditionalEffect",
            "Admg",
            "accepted",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(
            err.to_string().starts_with("refused: ConditionalEffect on Admg has no compile arm"),
            "Admg: {err}"
        );
        for graph in ["Cpdag", "Pag"] {
            for inference in ["Frequentist", "Bayesian"] {
                for validation in ["none", "cheap", "full"] {
                    let status = classify(cell(
                        "ConditionalEffect",
                        graph,
                        "accepted",
                        inference,
                        validation,
                    ));
                    assert_eq!(status, CellStatus::Licensed, "{graph}/{inference}/{validation}");
                    refuse_if_not_applicable(cell(
                        "ConditionalEffect",
                        graph,
                        "accepted",
                        inference,
                        validation,
                    ))
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn closed_path_and_distribution_on_explicit_admg_pag_is_enforced() {
        for graph in ["Admg", "Pag"] {
            let status =
                classify(cell("PathSpecificEffect", graph, "explicit", "Frequentist", "none"));
            assert_eq!(status, CellStatus::Refused, "PathSpecificEffect/{graph}");
            let err = refuse_if_not_applicable(cell(
                "PathSpecificEffect",
                graph,
                "explicit",
                "Frequentist",
                "none",
            ))
            .unwrap_err();
            assert!(
                err.to_string().starts_with(
                    "refused: Path-specific queries execute only on a supplied static Dag"
                ),
                "PathSpecificEffect/{graph}: {err}"
            );
        }
        assert_eq!(
            classify(cell("InterventionalDistribution", "Admg", "explicit", "Frequentist", "none")),
            CellStatus::Licensed
        );
        assert_eq!(
            classify(cell("InterventionalDistribution", "Pag", "explicit", "Frequentist", "none")),
            CellStatus::Refused
        );
        for validation in ["cheap", "full"] {
            assert_eq!(
                classify(cell(
                    "InterventionalDistribution",
                    "Admg",
                    "explicit",
                    "Frequentist",
                    validation
                )),
                CellStatus::Refused,
                "Admg distribution {validation}"
            );
        }
    }

    #[test]
    fn closed_intervention_response_off_dag_is_enforced() {
        let err = refuse_if_not_applicable(cell(
            "InterventionResponse",
            "Admg",
            "explicit",
            "Frequentist",
            "none",
        ))
        .unwrap_err();
        assert!(
            err.to_string().starts_with("refused: Admg response has no functional plug-in"),
            "{err}"
        );
        for graph in ["Cpdag", "Pag"] {
            for inference in ["Frequentist", "Bayesian"] {
                assert_eq!(
                    classify(cell("InterventionResponse", graph, "explicit", inference, "none")),
                    CellStatus::Licensed,
                    "{graph}/{inference}"
                );
                refuse_if_not_applicable(cell(
                    "InterventionResponse",
                    graph,
                    "explicit",
                    inference,
                    "none",
                ))
                .unwrap();
            }
        }
    }

    #[test]
    fn temporal_mediation_class_boundary_is_enforced() {
        for graph in ["TemporalCpdag", "TemporalPag"] {
            let status =
                classify(cell("TemporalMediationEffect", graph, "accepted", "Frequentist", "none"));
            assert_eq!(
                status,
                if graph == "TemporalCpdag" { CellStatus::Licensed } else { CellStatus::Refused },
                "{graph}"
            );
        }
    }

    #[test]
    fn licensed_graph_posterior_frequentist_ate_is_open() {
        for validation in ["none", "cheap", "full"] {
            let c = cell("AverageEffect", "Dag", "graph_posterior", "Frequentist", validation);
            assert_eq!(classify(c), CellStatus::Licensed, "{validation}");
            refuse_if_not_applicable(c).unwrap();
        }
        for inference in ["Frequentist", "Bayesian"] {
            for validation in ["none", "cheap", "full"] {
                let c = cell("ConditionalEffect", "Dag", "graph_posterior", inference, validation);
                assert_eq!(classify(c), CellStatus::Licensed);
                refuse_if_not_applicable(c).unwrap();
            }
        }
    }

    /// End-to-end: `Study::build` accepts the licensed Frequentist graph-posterior ATE cell.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn build_accepts_graph_posterior_under_frequentist() {
        use antecedent_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use antecedent_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
        };
        use std::sync::Arc;

        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [("t", RoleHint::TreatmentCandidate), ("y", RoleHint::OutcomeCandidate)]
        {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let n = 10;
        let t = vec![0.0; n];
        let y = vec![0.0; n];
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let gp = crate::discovery::discover_exact_dag_posterior(
            &data,
            &[VariableId::from_raw(0), VariableId::from_raw(1)],
            &crate::discovery::BayesianDiscoverParams::default(),
            &ctx,
        )
        .unwrap();

        crate::analysis::Study::tabular(data)
            .graph_posterior(gp)
            .query(ate_query())
            .inference(crate::inference::InferenceMode::Frequentist)
            .build()
            .expect("AverageEffect × graph_posterior × Frequentist is licensed");
    }

    /// End-to-end class-aware static case: `Study::build` accepts a `ConditionalEffect`
    /// query on a supplied Cpdag once the cell is licensed.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn build_accepts_conditional_effect_on_cpdag() {
        use antecedent_core::{
            AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet,
            ValueType,
        };
        use antecedent_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
        };
        use std::sync::Arc;

        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("z", RoleHint::Context),
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let n = 10;
        let z = vec![0.0; n];
        let t = vec![0.0; n];
        let y = vec![0.0; n];
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);

        let mut cpdag = Cpdag::with_variables(3);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        cpdag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();

        let inner =
            AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2))
                .with_effect_modifiers(vec![VariableId::from_raw(0)]);
        let query = antecedent_core::ConditionalEffectQuery::try_new(inner).unwrap();

        crate::analysis::Study::tabular(data)
            .graph(AcceptedGraph::cpdag(cpdag).unwrap())
            .query(query)
            .build()
            .expect("ConditionalEffect × Cpdag is licensed");
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
    ///         so the cell stays the Admg cell — licensed Frequentist ATE, not collapsed to Dag.
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
        assert_eq!(classify(sc), CellStatus::Licensed);
    }

    #[test]
    fn codetermined_joint_ir_is_a_licensed_matrix_cell() {
        let sc = cell("InterventionResponse", "CoDetermined", "explicit", "Frequentist", "none");
        assert_eq!(classify(sc), CellStatus::Licensed);
        let unknown = cell("AverageEffect", "Unknown", "explicit", "Frequentist", "none");
        assert_eq!(classify(unknown), CellStatus::Licensed);
        let off = cell("ConditionalEffect", "Unknown", "explicit", "Frequentist", "none");
        assert!(matches!(classify(off), CellStatus::NotApplicable { .. }));
    }

    /// The ADMG collapse is scoped to `AverageEffect`. Distribution on Admg is a
    /// separate licensed cell (general ID, no DAG coercion); collapse must not
    /// rewrite that class to Dag.
    #[test]
    fn admg_collapse_does_not_apply_outside_average_effect() {
        let mut admg = Admg::with_variables(2);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::from(admg);
        assert_eq!(effective_graph_class(&graph, &distribution_query()), GraphClass::Admg);
    }

    /// A supplied Cpdag stays Cpdag under `AverageEffect`, including when it
    /// has a unique orientation. Completing it to a `Dag` is a different cell.
    #[test]
    fn cpdag_does_not_collapse_under_average_effect() {
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::from(cpdag);
        assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Cpdag);
    }

    /// Undirected marks are MEC information: `AcceptedGraph::cpdag` accepts them.
    #[test]
    fn undirected_cpdag_is_accepted_and_stays_cpdag() {
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::cpdag(cpdag).unwrap();
        assert_eq!(effective_graph_class(&graph, &ate_query()), GraphClass::Cpdag);
    }

    /// Response / distribution queries on a Cpdag stay Cpdag (and refused):
    /// there is no compile arm that would license them via collapse.
    #[test]
    fn cpdag_does_not_collapse_outside_average_effect() {
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let graph = AcceptedGraph::from(cpdag);
        assert_eq!(effective_graph_class(&graph, &distribution_query()), GraphClass::Cpdag);
    }

    /// Fully oriented temporal CPDAGs retain their supplied class.
    #[test]
    fn complete_temporal_cpdag_retains_class_under_temporal_effect() {
        let mut cpdag = TemporalCpdag::empty();
        let a =
            cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b = cpdag
            .add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
        cpdag.insert_directed(a, b).unwrap();
        let graph = AcceptedGraph::temporal_cpdag(cpdag).unwrap();
        assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalCpdag);
    }

    /// An incomplete `TemporalCpdag` is accepted: undirected marks are the MEC.
    #[test]
    fn incomplete_temporal_cpdag_is_accepted_and_stays_temporal_cpdag() {
        let mut cpdag = TemporalCpdag::empty();
        let a =
            cpdag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b = cpdag
            .add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS)
            .unwrap();
        cpdag.insert_undirected(a, b).unwrap();
        let graph = AcceptedGraph::temporal_cpdag(cpdag).unwrap();
        assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalCpdag);
    }

    /// Fully oriented temporal PAGs still require MAG adjustment certification.
    #[test]
    fn complete_temporal_pag_retains_class_under_temporal_effect() {
        use antecedent_graph::TemporalPag;
        let mut pag = TemporalPag::empty();
        let a = pag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b =
            pag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        pag.insert_directed(a, b).unwrap();
        let graph = AcceptedGraph::temporal_pag(pag);
        assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalPag);
    }

    /// A `TemporalPag` with a circle mark is accepted: circles are the incomplete class.
    #[test]
    fn temporal_pag_with_circle_mark_is_accepted_and_stays_temporal_pag() {
        let mut pag = TemporalPag::empty();
        let a = pag.add_lagged(VariableId::from_raw(0), antecedent_core::Lag::from_raw(1)).unwrap();
        let b =
            pag.add_lagged(VariableId::from_raw(1), antecedent_core::Lag::CONTEMPORANEOUS).unwrap();
        pag.insert_circle_arrow(a, b).unwrap();
        let graph = AcceptedGraph::temporal_pag(pag);
        assert_eq!(effective_graph_class(&graph, &pulse_query()), GraphClass::TemporalPag);
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

    // -- retained allowlist compatibility ----------------------------------

    /// One concrete [`SupportCell`] that satisfies `rule`, using the rule's own
    /// first listed value on each constrained axis and a harmless default
    /// (`Dag` / `explicit` / `Frequentist` / `none`) on every unconstrained one.
    /// Every `ALLOWED_RULES` entry constrains `queries`, so this always picks a
    /// real query; disjointness from licensed / n/a / reason-backed refused cells is enforced by
    /// `scripts/gate_support_matrix.sh`, not re-derived here.
    fn representative_cell(rule: &crate::support_matrix_data::AllowedRule) -> SupportCell {
        SupportCell {
            query: rule
                .queries
                .and_then(|xs| xs.first())
                .copied()
                .expect("every allowed rule constrains queries"),
            graph_class: rule.graph_classes.and_then(|xs| xs.first()).copied().unwrap_or("Dag"),
            structure: rule.structures.and_then(|xs| xs.first()).copied().unwrap_or("explicit"),
            inference: rule.inferences.and_then(|xs| xs.first()).copied().unwrap_or("Frequentist"),
            validation: rule.validations.and_then(|xs| xs.first()).copied().unwrap_or("none"),
        }
    }

    /// Every retained `parity/support_allowlist.toml` compatibility row would
    /// fire. The 0.9 gate requires this loop to have zero entries.
    #[test]
    fn every_allowlist_rule_fires_on_its_representative_cell() {
        use crate::support_matrix_data::ALLOWED_RULES;
        for rule in ALLOWED_RULES {
            let cell = representative_cell(rule);
            assert_eq!(
                classify(cell),
                CellStatus::Refused,
                "{cell:?} (rule reason: {})",
                rule.reason
            );
            assert_eq!(allowed_reason(cell), Some(rule.reason), "{cell:?}");
            assert_eq!(allowed_parent(cell), Some(rule.parent), "{cell:?}");
            assert!(!rule.parent.is_empty(), "rule parent must be non-empty: {}", rule.reason);
            // A retained compatibility entry would still pass through.
            let passed = refuse_if_not_applicable(cell).unwrap();
            assert_eq!(passed.as_str(), "allowed_unlicensed", "{cell:?}");
            assert_eq!(passed.allowlist_reason(), Some(rule.reason), "{cell:?}");
            assert_eq!(passed.allowlist_parent(), Some(rule.parent), "{cell:?}");
        }
    }

    /// End-to-end: a PAG ATE study is licensed (generalized-adjustment envelope),
    /// under Frequentist none. Prepared-cache reuse is covered separately.
    #[test]
    fn build_pag_ate_is_licensed() {
        let mut pag = Pag::with_variables(3);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        let data = antecedent_data::TabularData::from_f64_columns([
            ("t", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0].as_slice()),
            ("y", [0.0, 2.0, 0.1, 2.1, 0.0, 2.0, 0.1, 2.1].as_slice()),
            ("r", [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0].as_slice()),
        ])
        .unwrap();
        let result = crate::analysis::Study::tabular(data)
            .graph(pag)
            .query(ate_query())
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .run(&antecedent_core::ExecutionContext::for_tests(1))
            .unwrap();
        assert!(result.estimate.ate.is_finite());
        assert_eq!(result.support_status.unwrap().as_str(), "licensed");
        let trace = result.analysis_trace_wire();
        assert_eq!(trace.support_status.as_deref(), Some("licensed"));
    }

    /// A licensed ADMG Bayesian cell still requires identification of its graph.
    #[test]
    fn admg_bayesian_bow_requires_identification() {
        let mut admg = Admg::with_variables(2);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let data = antecedent_data::TabularData::from_f64_columns([
            ("t", [0.0, 1.0, 0.0, 1.0].as_slice()),
            ("y", [0.0, 2.0, 0.1, 2.1].as_slice()),
        ])
        .unwrap();
        let err = crate::analysis::Study::tabular(data)
            .graph(admg)
            .query(ate_query())
            .inference(InferenceMode::Bayesian(crate::inference::BayesianConfig::conjugate()))
            .refute(RefuteSuite::None)
            .build()
            .expect("ADMG Bayesian ATE is licensed")
            .run(&antecedent_core::ExecutionContext::for_tests(1))
            .unwrap_err();
        assert!(!matches!(err, CausalError::Support { .. }), "{err}");
        assert!(err.to_string().to_lowercase().contains("identif"), "{err}");
    }

    #[test]
    fn sustained_dbn_posterior_all_suites_are_licensed() {
        assert_eq!(
            classify(cell("SustainedEffect", "TemporalDag", "graph_posterior", "Bayesian", "none")),
            CellStatus::Licensed
        );
        for v in ["cheap", "full"] {
            let c = cell("SustainedEffect", "TemporalDag", "graph_posterior", "Bayesian", v);
            assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
            assert!(refusal_reason(c).is_none());
        }
    }

    #[test]
    fn temporal_mediation_dbn_posterior_all_suites_are_licensed() {
        for v in ["none", "cheap", "full"] {
            let c =
                cell("TemporalMediationEffect", "TemporalDag", "graph_posterior", "Bayesian", v);
            assert_eq!(classify(c), CellStatus::Licensed, "{c:?}");
            assert!(refusal_reason(c).is_none());
        }
    }
}
