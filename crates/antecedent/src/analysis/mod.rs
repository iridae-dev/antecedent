//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]

mod batch;
mod builder;
mod contract;
mod contract_identity;
mod exact;
mod execute;
mod learned_trial;
mod statistical;
mod transport_grid;
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
mod helpers;
mod latency;
mod prepared;
mod stage;

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
pub use execute::Study;
pub use latency::{
    ComputeBudget, INTERACTIVE_BOOTSTRAP, INTERACTIVE_MAX_ENVELOPE_GRAPHS, INTERACTIVE_N_DRAWS,
    LatencyMode, REPORT_BOOTSTRAP, REPORT_N_DRAWS, ResolvedLatencyBudget, STANDARD_BOOTSTRAP,
    STANDARD_N_DRAWS, refuse_non_report_hmc,
};
pub use prepared::{CachedTemporalIdentification, PreparedStudy};
pub use stage::{StageEvent, StageResultSink};

pub(crate) use execute::{parametric_scm_identification, response_witness_ate};
