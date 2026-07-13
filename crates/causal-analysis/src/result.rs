//! Analysis result artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{
    Diagnostic, ExecutionPerformanceRecord, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord,
    ProvenanceGraph, VariableId,
};
use causal_estimate::EffectEstimate;
use causal_identify::{IdentificationResult, IdentifiedEstimand};
use causal_validate::RefutationReport;

/// End-to-end analysis result.
#[derive(Clone, Debug)]
pub struct CausalAnalysisResult {
    /// Logical plan record.
    pub logical_plan: LogicalAnalysisPlanRecord,
    /// Physical plan record.
    pub physical_plan: PhysicalExecutionPlanRecord,
    /// Full identification artifact.
    pub identification: IdentificationResult,
    /// Primary estimand used for estimation.
    pub estimand: IdentifiedEstimand,
    /// Point estimate + uncertainty.
    pub estimate: EffectEstimate,
    /// Refutation reports (may be empty).
    pub refutations: Vec<RefutationReport>,
    /// Diagnostics.
    pub diagnostics: Vec<Diagnostic>,
    /// Provenance.
    pub provenance: ProvenanceGraph,
    /// Performance record.
    pub performance: ExecutionPerformanceRecord,
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
}
