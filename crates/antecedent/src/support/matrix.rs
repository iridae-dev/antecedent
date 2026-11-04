//! Matrix lookup: classification of a cell against the generated rules and
//! supported rows, refusal reasons, coordinates and supported neighbors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{PremiseChange, SupportedNeighbor};
use antecedent_core::ExecutionContext;

use crate::error::CausalError;
use crate::strategy_table::EstimatorId;

use super::cell::{
    CellStatus, SupportCell, SupportGraphClass, SupportInference, SupportQuery, SupportRefusal,
    SupportStructure, SupportValidation,
};
use super::matrix_data::{self, Axis, CLOSED_RULES, NA_RULES, NaRule, SUPPORTED, SupportedCell};

fn axis_in(allowed: Option<&[&str]>, value: &str) -> bool {
    allowed.is_none_or(|xs| xs.contains(&value))
}

/// Whether every axis of `cell` is admitted by `rule` (an absent axis admits all).
fn rule_matches(rule: &NaRule, cell: SupportCell) -> bool {
    axis_in(rule.queries, cell.query.as_str())
        && axis_in(rule.graph_classes, cell.graph_class.as_str())
        && axis_in(rule.structures, cell.structure.as_str())
        && axis_in(rule.inferences, cell.inference.as_str())
        && axis_in(rule.validations, cell.validation.as_str())
}

/// Classify `cell` against the generated n/a rules and supported set.
#[must_use]
pub fn classify(cell: SupportCell) -> CellStatus {
    if let Some(rule) = NA_RULES.iter().find(|rule| rule_matches(rule, cell)) {
        return CellStatus::NotApplicable { reason: rule.reason };
    }
    if supported_row(cell).is_some() {
        return CellStatus::Supported;
    }
    CellStatus::NotSupported
}

/// Estimator wire-ids whose evidence ran for this geometric cell.
///
/// Empty when the cell is not supported, or when neither the compiler route nor
/// a calibration-record token named an estimator (do not guess one).
#[must_use]
pub fn supported_estimators(cell: SupportCell) -> &'static [&'static str] {
    supported_row(cell).map_or(&[], |row| row.estimators)
}

/// Compiler-plan estimator for this geometric cell (`parity/supported_routes.toml`).
///
/// This is the default unbound compile path; it is independent of how many
/// additional estimators the cell's calibration records support.
#[must_use]
pub fn supported_route_estimator(cell: SupportCell) -> Option<EstimatorId> {
    let wire = supported_row(cell)?.route_estimator?;
    wire.parse().ok()
}

fn supported_row(cell: SupportCell) -> Option<&'static SupportedCell> {
    matrix_data::supported_cell([
        cell.query.as_str(),
        cell.graph_class.as_str(),
        cell.structure.as_str(),
        cell.inference.as_str(),
        cell.validation.as_str(),
    ])
}

/// Classify `cell` for a concrete estimator.
///
/// [`CellStatus::Supported`] only when the geometric cell is supported and the
/// row's evidence list contains `estimator`. Otherwise [`CellStatus::NotSupported`]
/// (or the geometric n/a status)..
#[must_use]
pub fn classify_estimator(cell: SupportCell, estimator: EstimatorId) -> CellStatus {
    match classify(cell) {
        CellStatus::NotApplicable { reason } => CellStatus::NotApplicable { reason },
        CellStatus::Supported => {
            if supported_estimators(cell).contains(&estimator.as_str()) {
                CellStatus::Supported
            } else {
                CellStatus::NotSupported
            }
        }
        CellStatus::NotSupported => CellStatus::NotSupported,
    }
}

/// Whether a score-table retarget may run on a prepared handle of `status` on `cell`.
///
/// A retarget reweights the scores an estimator already kept and reports the handle's own
/// status; it makes no claim of its own. So it runs wherever the cell itself is supported and
/// where no matrix cell is declared (`status` is `None`), and its result carries what the
/// handle was built under: a supported cell whose estimator or construction has no coverage
/// record (a `NotSupported` status on a supported cell) retargets and reports `not_supported`. It is
/// blocked where the cell is not applicable or itself refused. The operation
/// report and the retarget gate both read this, so they cannot disagree.
#[must_use]
pub(crate) fn retarget_permitted(status: Option<CellStatus>, cell: Option<SupportCell>) -> bool {
    match status {
        None | Some(CellStatus::Supported) => true,
        Some(CellStatus::NotApplicable { .. }) => false,
        Some(CellStatus::NotSupported) => {
            cell.is_some_and(|cell| classify(cell) == CellStatus::Supported)
        }
    }
}

/// Closed-rule or default refusal text for a refused cell.
#[must_use]
pub fn refused_message(cell: SupportCell) -> &'static str {
    refusal_reason(cell).unwrap_or(UNSUPPORTED)
}

pub(super) fn refusal_reason(cell: SupportCell) -> Option<&'static str> {
    CLOSED_RULES.iter().find(|rule| rule_matches(rule, cell)).map(|rule| rule.reason)
}

/// The single static message every unmatched `Refused` cell reports. Cell
/// coordinates vary at runtime — the caller's [`CausalError::Support`] already
/// carries the offending cell in its own context — so this stays a fixed,
/// shared string rather than one formatted per cell.
const UNSUPPORTED: &str =
    "cell is not supported (parity/support_cells.toml) and is not n/a; it is refused.";

/// Format a matrix coordinate as `query:graph:structure:inference:validation`.
#[must_use]
pub fn cell_coordinate(cell: SupportCell) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        cell.query.as_str(),
        cell.graph_class.as_str(),
        cell.structure.as_str(),
        cell.inference.as_str(),
        cell.validation.as_str()
    )
}

/// Reconstruct a cell from [`cell_coordinate`] by interning against supported axes.
#[must_use]
pub fn support_cell_from_coordinate(coordinate: &str) -> Option<SupportCell> {
    let mut parts = coordinate.split(':');
    let query = SupportQuery::parse(matrix_data::intern_axis(Axis::Query, parts.next()?)?)?;
    let graph_class =
        SupportGraphClass::parse(matrix_data::intern_axis(Axis::GraphClass, parts.next()?)?)?;
    let structure =
        SupportStructure::parse(matrix_data::intern_axis(Axis::Structure, parts.next()?)?)?;
    let inference =
        SupportInference::parse(matrix_data::intern_axis(Axis::Inference, parts.next()?)?)?;
    let validation =
        SupportValidation::parse(matrix_data::intern_axis(Axis::Validation, parts.next()?)?)?;
    if parts.next().is_some() {
        return None;
    }
    Some(SupportCell { query, graph_class, structure, inference, validation })
}

/// Supported alternatives that keep the requested graph class.
///
/// Neighbors are explicit changed-premise suggestions, never automatic
/// fallbacks. Relabeling a class (`Cpdag` → `Dag`) is not a neighbor. An
/// already-supported cell has no neighbors. Every alternative is returned, nearest
/// first (fewest changed premises), in a deterministic order.
#[must_use]
pub fn supported_neighbors(cell: SupportCell) -> Vec<SupportedNeighbor> {
    if matches!(classify(cell), CellStatus::Supported) {
        return Vec::new();
    }
    let mut scored: Vec<(Vec<PremiseChange>, &'static SupportedCell)> = SUPPORTED
        .iter()
        .filter_map(|row| {
            if row.graph_class != cell.graph_class.as_str() {
                return None;
            }
            let changed = neighbor_changes(cell, row);
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
        .map(|(changed, row)| {
            let required_action = neighbor_action(&changed);
            SupportedNeighbor::new(
                cell_coordinate(support_cell_from_supported(row)),
                changed,
                required_action,
            )
        })
        .collect()
}

/// Every supported support-matrix coordinate.
pub fn supported_cells() -> impl Iterator<Item = SupportCell> {
    SUPPORTED.iter().map(support_cell_from_supported)
}

fn support_cell_from_supported(row: &SupportedCell) -> SupportCell {
    SupportCell {
        query: SupportQuery::parse(row.query).expect("supported query is a support axis"),
        graph_class: SupportGraphClass::parse(row.graph_class)
            .expect("supported graph class is a support axis"),
        structure: SupportStructure::parse(row.structure)
            .expect("supported structure is a support axis"),
        inference: SupportInference::parse(row.inference)
            .expect("supported inference is a support axis"),
        validation: SupportValidation::parse(row.validation)
            .expect("supported validation is a support axis"),
    }
}

fn neighbor_changes(cell: SupportCell, row: &SupportedCell) -> Vec<PremiseChange> {
    [
        ("query", cell.query.as_str(), row.query),
        ("structure", cell.structure.as_str(), row.structure),
        ("inference", cell.inference.as_str(), row.inference),
        ("validation", cell.validation.as_str(), row.validation),
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

/// Refuse n/a cells and every unsupported meaningful cell. Supported cells pass.
///
/// This is the context-free form: a development build records what a release build would
/// refuse in the calling thread's log ([`antecedent_core::support_gate::drain_unscoped`]).
/// Prefer [`refuse_if_not_applicable_in`] where an [`ExecutionContext`] exists.
///
/// # Errors
///
/// [`CausalError::Support`] when the cell is n/a (every build) or not supported (a release
/// build; a development build returns `Ok(CellStatus::NotSupported)` and records the entry).
pub fn refuse_if_not_applicable(cell: SupportCell) -> Result<CellStatus, CausalError> {
    refuse_cell(cell, antecedent_core::support_gate::require_supported_unscoped)
}

/// [`refuse_if_not_applicable`] recording a development build's would-refuse entry in `ctx`.
///
/// # Errors
///
/// As [`refuse_if_not_applicable`].
pub fn refuse_if_not_applicable_in(
    cell: SupportCell,
    ctx: &ExecutionContext,
) -> Result<CellStatus, CausalError> {
    refuse_cell(cell, |detail, missing| {
        antecedent_core::support_gate::require_supported(ctx, detail, missing)
    })
}

fn refuse_cell(
    cell: SupportCell,
    gate: impl FnOnce(&'static str, &[&str]) -> Result<(), antecedent_core::NotSupported>,
) -> Result<CellStatus, CausalError> {
    match classify(cell) {
        CellStatus::NotApplicable { reason } => {
            Err(CausalError::Support { id: SupportRefusal::NotApplicable, message: reason })
        }
        CellStatus::NotSupported => {
            let coordinate = format!("support_cell:{}", cell_coordinate(cell));
            gate(refused_message(cell), &[coordinate.as_str()])?;
            Ok(CellStatus::NotSupported)
        }
        CellStatus::Supported => Ok(CellStatus::Supported),
    }
}
