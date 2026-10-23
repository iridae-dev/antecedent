//! Matrix lookup: classification of a cell against the generated rules and
//! licensed rows, refusal reasons, coordinates and licensed neighbors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{LicensedNeighbor, PremiseChange};

use crate::error::CausalError;
use crate::strategy_table::EstimatorId;

use super::cell::{
    CellStatus, SupportCell, SupportGraphClass, SupportInference, SupportQuery, SupportRefusal,
    SupportStructure, SupportValidation,
};
use super::matrix_data::{self, Axis, CLOSED_RULES, LICENSED, LicensedCell, NA_RULES};

fn axis_in(allowed: Option<&[&str]>, value: &str) -> bool {
    allowed.is_none_or(|xs| xs.contains(&value))
}

/// Classify `cell` against the generated n/a rules and licensed set.
#[must_use]
pub fn classify(cell: SupportCell) -> CellStatus {
    for rule in NA_RULES {
        if axis_in(rule.queries, cell.query.as_str())
            && axis_in(rule.graph_classes, cell.graph_class.as_str())
            && axis_in(rule.structures, cell.structure.as_str())
            && axis_in(rule.inferences, cell.inference.as_str())
            && axis_in(rule.validations, cell.validation.as_str())
        {
            return CellStatus::NotApplicable { reason: rule.reason };
        }
    }
    if licensed_row(cell).is_some() {
        return CellStatus::Licensed;
    }
    CellStatus::Refused
}

/// Estimator wire-ids whose evidence ran for this geometric cell.
///
/// Empty when the cell is not licensed, or when neither the compiler route nor
/// a calibration-record token named an estimator (do not guess one).
#[must_use]
pub fn licensed_estimators(cell: SupportCell) -> &'static [&'static str] {
    licensed_row(cell).map_or(&[], |row| row.estimators)
}

/// Compiler-plan estimator for this geometric cell (`parity/licensed_routes.toml`).
///
/// This is the default unbound compile path; it is independent of how many
/// additional estimators the cell's calibration records license.
#[must_use]
pub fn licensed_route_estimator(cell: SupportCell) -> Option<EstimatorId> {
    let wire = licensed_row(cell)?.route_estimator?;
    wire.parse().ok()
}

fn licensed_row(cell: SupportCell) -> Option<&'static LicensedCell> {
    matrix_data::licensed_cell([
        cell.query.as_str(),
        cell.graph_class.as_str(),
        cell.structure.as_str(),
        cell.inference.as_str(),
        cell.validation.as_str(),
    ])
}

/// Classify `cell` for a concrete estimator.
///
/// [`CellStatus::Licensed`] only when the geometric cell is licensed and the
/// row's evidence list contains `estimator`. Otherwise [`CellStatus::Refused`]
/// (or the geometric n/a status). Never returns [`CellStatus::Allowlisted`]:
/// that status is historical compatibility only, and unmeasured estimators
/// must not ride it to stay runnable.
#[must_use]
pub fn classify_estimator(cell: SupportCell, estimator: EstimatorId) -> CellStatus {
    match classify(cell) {
        CellStatus::NotApplicable { reason } => CellStatus::NotApplicable { reason },
        CellStatus::Licensed => {
            if licensed_estimators(cell).contains(&estimator.as_str()) {
                CellStatus::Licensed
            } else {
                CellStatus::Refused
            }
        }
        CellStatus::Refused | CellStatus::Allowlisted { .. } => CellStatus::Refused,
    }
}

/// Closed-rule or default refusal text for a refused cell.
#[must_use]
pub fn refused_message(cell: SupportCell) -> &'static str {
    refusal_reason(cell).unwrap_or(UNLICENSED)
}

pub(super) fn refusal_reason(cell: SupportCell) -> Option<&'static str> {
    for rule in CLOSED_RULES {
        if axis_in(rule.queries, cell.query.as_str())
            && axis_in(rule.graph_classes, cell.graph_class.as_str())
            && axis_in(rule.structures, cell.structure.as_str())
            && axis_in(rule.inferences, cell.inference.as_str())
            && axis_in(rule.validations, cell.validation.as_str())
        {
            return Some(rule.reason);
        }
    }
    None
}

/// Whether a refused cell executes and stays unlicensed.
///
/// True only when the cell is refused (not licensed, not n/a) and the first
/// `parity/support_closed.toml` rule that names it carries
/// `executes_unlicensed = true`. Such a cell builds and runs; its result reports
/// the support status `refused`, so no coverage or evidence row is implied. The
/// default refuter suite is still downgraded on it (an implicit request keeps the
/// licensed `none` cell); only an explicit request runs it.
#[must_use]
pub(crate) fn executes_unlicensed(cell: SupportCell) -> bool {
    if !matches!(classify(cell), CellStatus::Refused) {
        return false;
    }
    CLOSED_RULES
        .iter()
        .find(|rule| {
            axis_in(rule.queries, cell.query.as_str())
                && axis_in(rule.graph_classes, cell.graph_class.as_str())
                && axis_in(rule.structures, cell.structure.as_str())
                && axis_in(rule.inferences, cell.inference.as_str())
                && axis_in(rule.validations, cell.validation.as_str())
        })
        .is_some_and(|rule| rule.executes_unlicensed)
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
        cell.query.as_str(),
        cell.graph_class.as_str(),
        cell.structure.as_str(),
        cell.inference.as_str(),
        cell.validation.as_str()
    )
}

/// Reconstruct a cell from [`cell_coordinate`] by interning against licensed axes.
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
    let mut scored: Vec<(Vec<PremiseChange>, LicensedCell)> = LICENSED
        .iter()
        .copied()
        .filter_map(|row| {
            if row.graph_class != cell.graph_class.as_str() {
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

/// Every licensed support-matrix coordinate.
pub fn licensed_support_cells() -> impl Iterator<Item = SupportCell> {
    LICENSED.iter().map(support_cell_from_licensed)
}

fn support_cell_from_licensed(row: &LicensedCell) -> SupportCell {
    SupportCell {
        query: SupportQuery::parse(row.query).expect("licensed query is a support axis"),
        graph_class: SupportGraphClass::parse(row.graph_class)
            .expect("licensed graph class is a support axis"),
        structure: SupportStructure::parse(row.structure)
            .expect("licensed structure is a support axis"),
        inference: SupportInference::parse(row.inference)
            .expect("licensed inference is a support axis"),
        validation: SupportValidation::parse(row.validation)
            .expect("licensed validation is a support axis"),
    }
}

fn neighbor_changes(cell: SupportCell, row: &LicensedCell) -> Vec<PremiseChange> {
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

/// Refuse n/a cells and every unlicensed meaningful cell. Licensed cells pass.
///
/// Historical [`CellStatus::Allowlisted`] is refused the same way as
/// [`CellStatus::Refused`]. [`classify`] does not produce it.
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
        CellStatus::Refused | CellStatus::Allowlisted { .. } => {
            if let Some(reason) = refusal_reason(cell) {
                return Err(CausalError::Support { id: SupportRefusal::Refused, message: reason });
            }
            Err(CausalError::Support { id: SupportRefusal::Refused, message: UNLICENSED })
        }
        CellStatus::Licensed => Ok(CellStatus::Licensed),
    }
}
