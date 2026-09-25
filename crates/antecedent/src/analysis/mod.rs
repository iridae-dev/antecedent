//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, clippy::doc_markdown, clippy::too_many_arguments)]

mod batch;
mod builder;
mod checked_bayesian_graph_posterior;
mod checked_bayesian_temporal_effect;
mod checked_class_graph_posterior_effect;
mod checked_conditional;
mod checked_graph_posterior;
mod checked_propensity;
mod checked_temporal_class_effect;
mod checked_temporal_effect;
mod checked_temporal_response;
mod contract;
mod contract_identity;
mod exact;
mod execute;
mod learned_trial;
mod statistical;
mod transport_grid;
mod z_transport;
mod z_transport_sensitivity_artifact;
pub(crate) use checked_bayesian_graph_posterior::CheckedBayesianGraphPosteriorAte;
pub(crate) use checked_bayesian_temporal_effect::{
    CheckedBayesianTemporalEffectOperation, CheckedBayesianTemporalTarget,
};
pub(crate) use checked_class_graph_posterior_effect::CheckedClassGraphPosteriorEffect;
pub(crate) use checked_conditional::{CheckedConditionalOperation, ConditionalProcedure};
pub(crate) use checked_graph_posterior::{
    CheckedAdmgGraphPosteriorResponse, CheckedGraphPosteriorEffect, CheckedStaticClassEffect,
    StaticClassGraph, StaticClassIdentification,
};
pub(crate) use checked_temporal_class_effect::CheckedTemporalClassEffectOperation;
pub(crate) use checked_temporal_effect::CheckedTemporalEffectOperation;
pub use exact::{
    ExactFactorRequirement, ExactPreparedState, ExactStudyIdentities, ExactStudyInspection,
    ExactStudyResult,
};
pub use learned_trial::{LearnedTrialResult, LearnedTrialState};
pub use statistical::{
    StatisticalBindingView, StatisticalContrast, StatisticalPreparedState,
    StatisticalStudyInspection, StatisticalStudyResult,
};
pub use transport_grid::{
    TransportGridData, TransportGridFailure, TransportGridPoint, TransportGridQuery,
    TransportGridResult, TransportGridState,
};
pub use z_transport::{PreparedZTransport, ZTransportResult, consume_z_transport_artifact};
pub use z_transport_sensitivity_artifact::ZTransportSensitivityArtifactWire;
mod helpers;
mod latency;
mod prepared;
mod stage;
mod transport_common;

pub use antecedent_core::{
    BlockedOperation, LicensedNeighbor, NextAction, OperationKind, OperationReadiness,
    OperationReport, PremiseChange, SemanticApplicability,
};
pub use batch::{
    BatchQuery, BatchStudy, CandidateProcedure, CandidateScreen, CandidateSelection,
    CellFamilyContrast, PreparedBatch, SharedBatchDesign, SharedCovariateDesign,
};
pub use builder::{InterferenceSpec, RdConfig, RefuteSuite, StudyBuilder, TransportTrialSpec};
pub use contract::CausalContract;
pub use execute::DagResponseOrigin;
pub use execute::Study;
pub use latency::{
    ComputeBudget, INTERACTIVE_BOOTSTRAP, INTERACTIVE_MAX_ENVELOPE_GRAPHS, INTERACTIVE_N_DRAWS,
    LatencyMode, REPORT_BOOTSTRAP, REPORT_N_DRAWS, ResolvedLatencyBudget, STANDARD_BOOTSTRAP,
    STANDARD_N_DRAWS, refuse_non_report_hmc,
};
pub use prepared::{
    CachedTemporalIdentification, CheckedAdmgGraphPosteriorResponseInfo, CheckedAttributionInfo,
    CheckedBayesianGraphPosteriorAteInfo, CheckedCellAipwResponseInfo,
    CheckedClassGraphPosteriorEffectInfo, CheckedConditionalEffectInfo,
    CheckedGraphPosteriorEffectInfo, CheckedInterferenceInfo,
    CheckedBayesianTemporalDagEffectInfo, CheckedStaticMediationInfo,
    CheckedTemporalClassEffectInfo, CheckedTemporalDagEffectInfo,
    CheckedTemporalDagResponseInfo, CheckedTemporalMediationInfo, CheckedUnknownTieredAverageInfo,
    PreparedStudy,
};
pub use stage::{StageEvent, StageResultSink};

pub(crate) use checked_temporal_response::{
    CheckedTemporalResponseOperation, TemporalResponseHorizonEvidence,
};
pub(crate) use execute::{parametric_scm_identification, response_witness_ate};
