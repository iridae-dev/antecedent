//! State events (DESIGN.md §20).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{AssumptionRecord, CausalQuery};

use crate::store::{
    ConstraintId, DataBatchRef, DataVersion, GraphConstraintRecord, GraphEvidenceRecord,
    InterventionRecord,
};

/// Events that update [`crate::CausalState`] without auto-rerunning analyses.
#[derive(Clone, Debug)]
pub enum StateEvent {
    /// Append a data batch reference.
    AppendData(DataBatchRef),
    /// Replace the data catalog version (invalidates dependent caches).
    ReplaceData(DataVersion),
    /// Add graph evidence.
    AddGraphEvidence(GraphEvidenceRecord),
    /// Add a graph constraint.
    AddConstraint(GraphConstraintRecord),
    /// Remove a graph constraint.
    RemoveConstraint(ConstraintId),
    /// Update / insert an assumption.
    UpdateAssumption(AssumptionRecord),
    /// Register a causal query.
    RegisterQuery(CausalQuery),
    /// Record an intervention (does not dispatch external actions).
    RecordIntervention(InterventionRecord),
}
