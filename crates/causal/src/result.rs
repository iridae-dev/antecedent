//! Analysis result artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{
    Diagnostic, ExecutionPerformanceRecord, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord,
    ProvenanceGraph, VariableId,
};
use causal_estimate::EffectEstimate;
use causal_identify::{IdentificationResult, IdentifiedEstimand};
use causal_io::{AnalysisTraceWire, DerivationStepWire, assumptions_to_wire};
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

impl CausalAnalysisResult {
    /// Build a durable analysis-trace wire payload (assumptions + derivation).
    #[must_use]
    pub fn analysis_trace_wire(&self) -> AnalysisTraceWire {
        AnalysisTraceWire {
            assumptions: assumptions_to_wire(&self.estimate.assumptions),
            derivation: self
                .identification
                .derivation
                .steps
                .iter()
                .map(|s| DerivationStepWire {
                    rule: s.rule.to_string(),
                    detail: s.detail.to_string(),
                })
                .collect(),
            method: self.estimand.method.to_string(),
            adjustment_set: self.estimand.adjustment_set.iter().map(|id| id.raw()).collect(),
        }
    }
}
