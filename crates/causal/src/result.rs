//! Analysis result artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_attribution::{
    AnomalyScores, ChangeAttributionResult, MechanismChangeDetection, UnitChangeResult,
};
use causal_core::{
    Diagnostic, ExecutionPerformanceRecord, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord,
    ProvenanceGraph, VariableId,
};
use causal_estimate::{CausalPosterior, EffectEstimate, InterventionalDistributionEstimate, TemporalMediationEstimate};
use causal_identify::{IdentificationResult, IdentifiedEstimand};
use causal_io::{AnalysisTraceWire, DerivationStepWire, assumptions_to_wire};
use causal_validate::{PredictiveCheckReport, RefutationReport};

use crate::gcm::IteResult;

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
    /// Point estimate + uncertainty (frequentist, or Bayesian posterior mean summary).
    ///
    /// For [`CausalQuery::Distribution`](causal_core::CausalQuery::Distribution) this holds the
    /// interventional mean of the first numeric outcome when defined (`ate` field), else NaN.
    pub estimate: EffectEstimate,
    /// Full interventional distribution when the query was [`CausalQuery::Distribution`].
    pub distribution: Option<InterventionalDistributionEstimate>,
    /// Bayesian posterior when `InferenceMode::Bayesian` was used.
    pub posterior: Option<CausalPosterior>,
    /// Temporal / static mediation decomposition when the query was mediation.
    pub mediation: Option<TemporalMediationEstimate>,
    /// Unit-level ITE when the query was counterfactual.
    pub counterfactual: Option<IteResult>,
    /// Anomaly scores when the query was anomaly attribution.
    pub anomaly: Option<Vec<AnomalyScores>>,
    /// Change-attribution result.
    pub change_attribution: Option<ChangeAttributionResult>,
    /// Mechanism-change detections.
    pub mechanism_change: Option<Vec<MechanismChangeDetection>>,
    /// Unit-change attribution.
    pub unit_change: Option<UnitChangeResult>,
    /// Refutation reports (may be empty).
    pub refutations: Vec<RefutationReport>,
    /// Prior/posterior predictive check reports (Bayesian path; may be empty).
    pub predictive_checks: Vec<PredictiveCheckReport>,
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
