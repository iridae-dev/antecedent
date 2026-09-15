//! Operation capability reports projected from inspect / classify.
//!
//! This is a companion record, not a second compiler or error hierarchy.
//! Applicability is the support-matrix coordinate. Readiness is snapshot
//! state. Budget and cancel are runtime, not scientific impossibility.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::obligation::ObligationRecord;
use crate::transform::SemanticLayer;

/// Support-matrix applicability for one requested operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SemanticApplicability {
    /// Listed in the licensed matrix.
    Licensed,
    /// Meaningful on the axes, but not licensed.
    Unlicensed,
    /// Typed-impossible (`not_applicable`).
    Impossible,
    /// Off-axis or unclassified. Not failed support and not a license.
    Unknown,
}

impl SemanticApplicability {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Licensed => "licensed",
            Self::Unlicensed => "unlicensed",
            Self::Impossible => "impossible",
            Self::Unknown => "unknown",
        }
    }
}

/// Snapshot readiness for the inspected operation.
///
/// `Executable` requires a licensed cell and met bindings. Unknown support is
/// not failed support and is not an executable guarantee.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum OperationReadiness {
    /// Licensed cell with met bindings for this snapshot.
    Executable,
    /// Graph / class review is still required.
    ReviewRequired,
    /// A required assumption has not been declared.
    AssumptionMissing,
    /// A required empirical check has not been run.
    EmpiricalCheckPending,
    /// A required empirical check failed.
    EmpiricalSupportFailed,
    /// Artifact, provider, graph, query, or identification product is missing.
    BindingMissing,
    /// Runtime resource or cancellation, not scientific impossibility.
    RuntimeUnavailable,
    /// No matrix cell. Distinct from failed support.
    UnknownSupport,
}

impl OperationReadiness {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Executable => "executable",
            Self::ReviewRequired => "review_required",
            Self::AssumptionMissing => "assumption_missing",
            Self::EmpiricalCheckPending => "empirical_check_pending",
            Self::EmpiricalSupportFailed => "empirical_support_failed",
            Self::BindingMissing => "binding_missing",
            Self::RuntimeUnavailable => "runtime_unavailable",
            Self::UnknownSupport => "unknown_support",
        }
    }
}

/// Named operation the report is about. Not a support-matrix row.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum OperationKind {
    /// Cheap structural inspect / classify.
    Inspect,
    /// Prepare a licensed handle.
    Prepare,
    /// Estimate / execute against the bound snapshot.
    Execute,
    /// Prepared score-table retarget.
    Retarget,
    /// Design ranking composed onto a prepared handle.
    RankDesigns,
    /// Export a contracted artifact.
    Export,
}

impl OperationKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inspect => "inspect",
            Self::Prepare => "prepare",
            Self::Execute => "execute",
            Self::Retarget => "retarget",
            Self::RankDesigns => "rank_designs",
            Self::Export => "export",
        }
    }

    /// Whether this operation is a support-matrix coordinate or a scoped contract.
    #[must_use]
    pub const fn uses_matrix_row(self) -> bool {
        matches!(self, Self::Inspect | Self::Prepare | Self::Execute)
    }
}

/// One stable blocked-operation record.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct BlockedOperation {
    /// Stable id (`support.refused`, `binding.graph`, `runtime.cancelled`, …).
    pub id: Arc<str>,
    /// Human-readable reason; id is authoritative.
    pub reason: Arc<str>,
    /// Scientific impossibility versus budget / cancel / resource.
    pub scientific: bool,
}

impl BlockedOperation {
    /// Construct a blocker.
    #[must_use]
    pub fn new(id: impl Into<Arc<str>>, reason: impl Into<Arc<str>>, scientific: bool) -> Self {
        Self { id: id.into(), reason: reason.into(), scientific }
    }

    /// Support-matrix n/a.
    #[must_use]
    pub fn not_applicable(reason: impl Into<Arc<str>>) -> Self {
        Self::new("support.not_applicable", reason, true)
    }

    /// Support-matrix refusal.
    #[must_use]
    pub fn refused(reason: impl Into<Arc<str>>) -> Self {
        Self::new("support.refused", reason, true)
    }

    /// Off-axis query / graph pairing.
    #[must_use]
    pub fn off_axis() -> Self {
        Self::new("support.off_axis", "query is not on the public support-matrix axis", true)
    }

    /// Missing builder / artifact binding.
    #[must_use]
    pub fn binding(field: &str) -> Self {
        Self::new(format!("binding.{field}"), format!("missing required binding: {field}"), true)
    }

    /// Scoped operation (retarget / rank / export) without a license.
    #[must_use]
    pub fn operation_unlicensed(reason: impl Into<Arc<str>>) -> Self {
        Self::new("operation.unlicensed", reason, true)
    }

    /// Cooperative cancellation. Not scientific impossibility.
    #[must_use]
    pub fn cancelled(stage: impl Into<Arc<str>>) -> Self {
        Self::new("runtime.cancelled", stage, false)
    }

    /// Resource / budget refusal. Not scientific impossibility.
    #[must_use]
    pub fn resource(reason: impl Into<Arc<str>>) -> Self {
        Self::new("runtime.resource", reason, false)
    }
}

/// Suggested next action. Never an automatic fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct NextAction {
    /// Stable action id.
    pub id: Arc<str>,
    /// What the caller must do.
    pub description: Arc<str>,
}

impl NextAction {
    /// Construct a next action.
    #[must_use]
    pub fn new(id: impl Into<Arc<str>>, description: impl Into<Arc<str>>) -> Self {
        Self { id: id.into(), description: description.into() }
    }
}

/// One changed support-matrix premise on a licensed neighbor.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct PremiseChange {
    /// Axis (`query`, `graph_class`, `structure`, `inference`, `validation`).
    pub axis: Arc<str>,
    /// Current value.
    pub from: Arc<str>,
    /// Neighbor value.
    pub to: Arc<str>,
}

impl PremiseChange {
    /// Construct a premise change.
    #[must_use]
    pub fn new(
        axis: impl Into<Arc<str>>,
        from: impl Into<Arc<str>>,
        to: impl Into<Arc<str>>,
    ) -> Self {
        Self { axis: axis.into(), from: from.into(), to: to.into() }
    }
}

/// Licensed alternative with explicit changed premises. Never a fallback.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct LicensedNeighbor {
    /// `query:graph_class:structure:inference:validation`.
    pub coordinate: Arc<str>,
    /// Axes that differ from the requested cell.
    pub changed: Arc<[PremiseChange]>,
    /// Required user action. Graph-class relabel is never a neighbor.
    pub required_action: Arc<str>,
}

impl LicensedNeighbor {
    /// Construct a neighbor suggestion.
    #[must_use]
    pub fn new(
        coordinate: impl Into<Arc<str>>,
        changed: impl Into<Arc<[PremiseChange]>>,
        required_action: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            coordinate: coordinate.into(),
            changed: changed.into(),
            required_action: required_action.into(),
        }
    }
}

/// Additive operation report beside existing errors.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct OperationReport {
    /// Operation this report describes.
    pub operation: OperationKind,
    /// Matrix applicability.
    pub applicability: SemanticApplicability,
    /// Snapshot readiness. `None` when applicability is the only blocker.
    pub readiness: Option<OperationReadiness>,
    /// Actual matrix coordinate, when classified.
    pub coordinate: Option<Arc<str>>,
    /// Stable blockers. Empty when the operation is executable.
    pub blockers: Arc<[BlockedOperation]>,
    /// Unresolved obligations retained from the contract.
    pub unmet_obligations: Arc<[ObligationRecord]>,
    /// Layers the current premises still preserve.
    pub preserved_layers: Arc<[SemanticLayer]>,
    /// Layers the current premises invalidate.
    pub invalidated_layers: Arc<[SemanticLayer]>,
    /// Possible next actions. Never executed automatically.
    pub next_actions: Arc<[NextAction]>,
    /// Licensed neighbors. Empty when already licensed, no graph, or none exist.
    pub neighbors: Arc<[LicensedNeighbor]>,
}

impl OperationReport {
    /// Fully specified report.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operation: OperationKind,
        applicability: SemanticApplicability,
        readiness: Option<OperationReadiness>,
        coordinate: Option<Arc<str>>,
        blockers: impl Into<Arc<[BlockedOperation]>>,
        unmet_obligations: impl Into<Arc<[ObligationRecord]>>,
        preserved_layers: impl Into<Arc<[SemanticLayer]>>,
        invalidated_layers: impl Into<Arc<[SemanticLayer]>>,
        next_actions: impl Into<Arc<[NextAction]>>,
        neighbors: impl Into<Arc<[LicensedNeighbor]>>,
    ) -> Self {
        Self {
            operation,
            applicability,
            readiness,
            coordinate,
            blockers: blockers.into(),
            unmet_obligations: unmet_obligations.into(),
            preserved_layers: preserved_layers.into(),
            invalidated_layers: invalidated_layers.into(),
            next_actions: next_actions.into(),
            neighbors: neighbors.into(),
        }
    }

    /// Missing graph or query: no neighbor recommendations.
    #[must_use]
    pub fn binding_missing(operation: OperationKind, field: &str) -> Self {
        Self::new(
            operation,
            SemanticApplicability::Unknown,
            Some(OperationReadiness::BindingMissing),
            None,
            [BlockedOperation::binding(field)],
            [],
            [],
            [SemanticLayer::Program, SemanticLayer::Support],
            [NextAction::new(
                format!("supply_{field}"),
                format!("supply {field} before requesting licensed neighbors"),
            )],
            [],
        )
    }

    /// Whether this report licenses execution of `operation`.
    #[must_use]
    pub fn is_executable(&self) -> bool {
        self.applicability == SemanticApplicability::Licensed
            && self.readiness == Some(OperationReadiness::Executable)
            && self.blockers.is_empty()
    }

    /// First stable blocker id, when blocked.
    #[must_use]
    pub fn primary_blocker_id(&self) -> Option<&str> {
        self.blockers.first().map(|blocker| blocker.id.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_is_not_scientific_impossibility() {
        let cancel = BlockedOperation::cancelled("estimate");
        let refused = BlockedOperation::refused("cell is not licensed");
        assert!(!cancel.scientific);
        assert!(refused.scientific);
        assert_ne!(cancel.id.as_ref(), refused.id.as_ref());
    }

    #[test]
    fn missing_graph_has_no_neighbors() {
        let report = OperationReport::binding_missing(OperationKind::Inspect, "graph");
        assert!(report.neighbors.is_empty());
        assert_eq!(report.readiness, Some(OperationReadiness::BindingMissing));
        assert!(!report.is_executable());
    }

    #[test]
    fn scoped_operations_are_not_matrix_rows() {
        assert!(!OperationKind::RankDesigns.uses_matrix_row());
        assert!(!OperationKind::Retarget.uses_matrix_row());
        assert!(!OperationKind::Export.uses_matrix_row());
        assert!(OperationKind::Inspect.uses_matrix_row());
    }
}
