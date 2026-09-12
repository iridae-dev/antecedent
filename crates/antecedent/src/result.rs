//! Analysis result artifact.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_attribution::{
    AnomalyScores, ChangeAttributionResult, MechanismChangeDetection, UnitChangeResult,
};
use antecedent_core::{
    CausalResponse, Diagnostic, ExecutionPerformanceRecord, IdentificationStatus,
    LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, ProvenanceGraph, ResponseEnvelope,
    ResponseValue, VariableId,
};
use antecedent_estimate::{
    CausalPosterior, EffectEstimate, InterventionalDistributionEstimate, TemporalMediationEstimate,
    TemporalMediationGrid,
};
use antecedent_identify::{IdentificationResult, IdentifiedEstimand};
use antecedent_io::{AnalysisTraceWire, DerivationStepWire, assumptions_to_wire};
use antecedent_validate::{PredictiveCheckReport, RefutationReport};

use crate::gcm::IteResult;

/// Identification certificate retained from the actual execution, including class atoms.
#[derive(Clone, Debug)]
pub struct AnalysisIdentification {
    /// Full point or completion-envelope artifact in its original coordinates.
    pub identification: crate::Identification,
    /// Query whose functional was estimated.
    pub query: antecedent_core::CausalQuery,
    /// Supplied graph class, preserved even when fully oriented.
    pub graph_class: crate::GraphClass,
}

/// Meaning of structural atom weights retained on a response result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum StructuralWeightBasis {
    /// Probabilities supplied by a graph posterior.
    PosteriorProbability,
    /// Enumeration weights over CPDAG/PAG completions; not posterior probabilities.
    CompletionEnumeration,
}

/// One structural atom in a graph-dependent response.
#[derive(Clone, Debug)]
pub struct StructuralResponseAtom {
    /// Stable key within this result.
    pub graph_key: u64,
    /// Raw atom weight in the declared basis.
    pub weight: f64,
    /// Structural identification/evaluation status.
    pub status: IdentificationStatus,
    /// Numerical response when identified and evaluable.
    pub value: Option<ResponseValue>,
}

/// Structural uncertainty retained separately from sampling uncertainty.
#[derive(Clone, Debug)]
pub struct StructuralResponseMixture {
    /// Interpretation of [`StructuralResponseAtom::weight`].
    pub weight_basis: StructuralWeightBasis,
    /// Every examined structural atom, including nonidentified/unevaluable atoms.
    pub atoms: Vec<StructuralResponseAtom>,
    /// Fraction of total weight with an evaluable identified response.
    pub identified_mass: f64,
    /// Fraction proved or conservatively treated as structurally unidentified.
    pub unidentified_mass: f64,
    /// Fraction identified in theory but not evaluable by the selected estimator.
    pub unevaluable_mass: f64,
    /// Pointwise range over identified atom point responses.
    pub identified_set: Option<ResponseEnvelope>,
    /// Probability-weighted summary, only meaningful for posterior probability weights.
    pub conditional_on_identified: Option<ResponseValue>,
    /// Whether reported mass covers the full class rather than a capped subset.
    pub full_mass_scope: bool,
    /// Atoms whose identification search was capped before a determination.
    pub truncated_atoms: usize,
}

/// End-to-end analysis result.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct StudyResult {
    /// Logical plan record.
    pub logical_plan: LogicalAnalysisPlanRecord,
    /// Physical plan record.
    pub physical_plan: PhysicalExecutionPlanRecord,
    /// Full identification artifact.
    pub identification: IdentificationResult,
    /// Complete identification certificate, when supplied by the execution path.
    pub certificate: Option<AnalysisIdentification>,
    /// Primary estimand used for estimation.
    pub estimand: IdentifiedEstimand,
    /// Point estimate + uncertainty (frequentist, or Bayesian posterior mean summary).
    ///
    /// For [`CausalQuery::Distribution`](antecedent_core::CausalQuery::Distribution) this holds the
    /// interventional mean of the first numeric outcome when defined (`ate` field), else NaN.
    pub estimate: EffectEstimate,
    /// Function-valued causal response for [`CausalQuery::Response`](antecedent_core::CausalQuery::Response).
    pub response: Option<CausalResponse>,
    /// Structural atom/mass result for class-aware or graph-posterior responses.
    pub structural_response: Option<StructuralResponseMixture>,
    /// Full interventional distribution when the query was
    /// [`CausalQuery::Distribution`](antecedent_core::CausalQuery::Distribution).
    pub distribution: Option<InterventionalDistributionEstimate>,
    /// Bayesian posterior when `InferenceMode::Bayesian` was used.
    pub posterior: Option<CausalPosterior>,
    /// Temporal / static mediation decomposition when the query was mediation.
    pub mediation: Option<TemporalMediationEstimate>,
    /// Horizon-indexed temporal mediation decomposition. Present for temporal
    /// mediation, including single-horizon results; it does not imply a joint posterior.
    pub mediation_grid: Option<TemporalMediationGrid>,
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
    /// Support-matrix evidence contract that produced this result.
    ///
    /// `licensed` and `allowed_unlicensed` both yield a successful study.
    /// Downstream consumers must not treat a number as licensed unless this is
    /// [`crate::support::CellStatus::Licensed`]. `None` when the query is not on
    /// the public axis.
    pub support_status: Option<crate::support::CellStatus>,
    /// How the caller supplied structure. Graph-posterior mixtures are never a
    /// single adjustment set, even when every identified atom happens to agree.
    pub structure_source: crate::support::StructureSource,
    /// Performance record.
    pub performance: ExecutionPerformanceRecord,
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Candidate-selection screen recorded for a prepared batch family.
    pub candidate_selection: Option<crate::analysis::CandidateSelection>,
}

impl StudyResult {
    /// Primary scalar effect for display and tests.
    ///
    /// Prefer this over reading [`EffectEstimate::ate`] directly when the query may be a
    /// distribution, mediation, or counterfactual: returns the interventional mean,
    /// mediation total, or mean ITE when present, otherwise the estimate's `ate` field.
    #[must_use]
    pub fn effect(&self) -> f64 {
        if let Some(dist) = &self.distribution {
            return dist.mean;
        }
        if let Some(med) = &self.mediation {
            if let Some(total) = med.total {
                return total;
            }
        }
        if let Some(cf) = &self.counterfactual {
            return cf.mean_ite;
        }
        self.estimate.ate
    }

    /// Borrow the logical plan record (semantics).
    #[must_use]
    pub fn logical_plan(&self) -> &LogicalAnalysisPlanRecord {
        &self.logical_plan
    }

    /// Borrow the physical plan record (layouts / kernels / batching).
    #[must_use]
    pub fn physical_plan(&self) -> &PhysicalExecutionPlanRecord {
        &self.physical_plan
    }

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
            support_status: self
                .support_status
                .map(crate::support::CellStatus::as_str)
                .map(str::to_string),
            allowlist_reason: self
                .support_status
                .and_then(crate::support::CellStatus::allowlist_reason)
                .map(str::to_string),
            allowlist_parent: self
                .support_status
                .and_then(crate::support::CellStatus::allowlist_parent)
                .map(str::to_string),
        }
    }
}
