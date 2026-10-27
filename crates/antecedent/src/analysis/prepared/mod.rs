//! Compile-once / re-estimate-many prepared analysis handle.
//!
//! Rediscover policy: structure is frozen at prepare time. Changing bootstrap,
//! prior scale, treatment levels, or latency never re-runs discovery — only an
//! explicit new discover / review → prepare cycle may replace the graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod cached;
mod estimate;
mod execute_checked;
mod execution;
mod execution_accessors;
mod identification_view;
mod info;
pub use identification_view::PreparedIdentificationView;
mod joint_response;
pub use joint_response::{JointResponseResult, PreparedJointResponse};
mod lifecycle;
mod operations;
mod period_refresh_entry;
mod prepare;
mod prepare_identification;
mod propensity_nuisance_entry;
mod retained_refresh_entry;
mod retarget;
mod seal;
mod sealed;
mod sealers;
mod support;

#[cfg(test)]
mod tests;

pub(in crate::analysis) use cached::conditional_class_proof_is_complete;
pub(crate) use cached::{
    CachedCpdagIdentification, CachedPagIdentification, CachedTemporalClassIdentification,
    IdentificationCache,
};
pub use cached::{
    CachedStaticIdentification, CachedTemporalHorizonIdentification, CachedTemporalIdentification,
};
pub(crate) use execution::*;
pub(crate) use execution_accessors::*;

pub use info::{
    CheckedAdmgGraphPosteriorResponseInfo, CheckedAttributionInfo, CheckedBayesianBasisAteInfo,
    CheckedBayesianClassConditionalInfo, CheckedBayesianGraphPosteriorAteInfo,
    CheckedBayesianRobustAteInfo, CheckedBayesianTemporalClassEffectInfo,
    CheckedBayesianTemporalDagEffectInfo, CheckedCellAipwResponseInfo,
    CheckedClassGraphPosteriorEffectInfo, CheckedConditionalEffectInfo,
    CheckedGraphPosteriorEffectInfo, CheckedGraphPosteriorResponseInfo, CheckedInterferenceInfo,
    CheckedStaticClassEffectInfo, CheckedStaticClassResponseInfo, CheckedStaticMediationInfo,
    CheckedTemporalClassEffectInfo, CheckedTemporalClassMediationInfo,
    CheckedTemporalClassResponseInfo, CheckedTemporalDagEffectInfo, CheckedTemporalDagResponseInfo,
    CheckedTemporalGraphPosteriorEffectInfo, CheckedTemporalGraphPosteriorResponseInfo,
    CheckedTemporalMediationInfo, CheckedTemporalMediationTargetInfo,
};
pub use info::{
    CheckedDerivativeResponseInfo, CheckedPropensityInfo, CheckedStaticDagResponseInfo,
    RetainedOperation,
};
pub(crate) use operations::{
    CheckedAdmgResponseCurveOperation, CheckedAipwOperation, CheckedDistributionOperation,
    CheckedFrontDoorOperation, CheckedFunctionalEffectOperation, CheckedGlmAdjustmentOperation,
    CheckedIvOperation, CheckedLinearOperation, CheckedPathSpecificEffectOperation,
    CheckedRdOperation, CheckedStaticResponseCurve, CheckedUnknownTieredAverageOperation,
};
pub use operations::{
    CheckedBayesianGcompOperation, CheckedFunctionalEffectResponseMember,
    CheckedUnknownTieredAverageInfo,
};
use operations::{CheckedStaticResultMetadata, static_result_metadata};
pub(crate) use prepare_identification::*;
pub(crate) use sealed::Inspect;
pub(crate) use support::*;

use crate::analysis::execute::Study;
pub(crate) use crate::analysis::identification_cache::{
    CachedClassPosteriorAtomIdentification, CachedDbnPosteriorAtomIdentification,
    CachedDbnPosteriorIdentification, CachedGraphPosteriorIdentification,
    CachedTemporalClassPosteriorAtomIdentification, CachedTemporalClassPosteriorIdentification,
    build_admg_graph_posterior_response_identification_cache,
    build_dbn_posterior_identification_cache, build_dbn_posterior_mediation_identification_cache,
    build_dbn_posterior_response_identification_cache, build_graph_posterior_identification_cache,
    build_temporal_class_posterior_identification_cache, temporal_dag_from_dbn_atom,
};
#[cfg(test)]
pub(crate) use crate::analysis::identification_cache::{
    CachedClassPosteriorCase, build_dbn_mediation_cache_with_identifier,
};
use crate::planner::PhysicalExecutionPlan;
use antecedent_core::CausalSchema;

use std::sync::Arc;

/// Test instrument: bind `data` on a new branch, materialize it and estimate, then adopt the
/// branch as the handle. A failure leaves the handle unchanged, as the immutable lifecycle does.
#[cfg(test)]
pub(crate) fn rebind_estimate(
    prepared: &mut PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<crate::result::StudyResult, crate::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    // A native estimator may return a usable partial result after cancellation; a cancelled
    // click must not be adopted as the new handle.
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalError::Cancelled { stage: "refresh" });
    }
    *prepared = branch;
    Ok(result)
}

/// Data modality a [`PreparedStudy`] was compiled for.
///
/// The physical plan is modality-specific, so a handle prepared on one
/// modality must refuse data of another instead of running a different route
/// under the frozen plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedModality {
    /// Tabular rows: [`PreparedStudy::estimate`] / [`PreparedStudy::refresh`].
    Tabular,
    /// One time series (or event data on a regular grid):
    /// [`PreparedStudy::estimate_series`] / [`PreparedStudy::refresh_series`].
    Series,
    /// Multi-unit panel: [`PreparedStudy::estimate_panel`] / [`PreparedStudy::refresh_panel`].
    Panel,
    /// Multi-environment series: [`PreparedStudy::estimate_multi_env`] /
    /// [`PreparedStudy::refresh_multi_env`].
    MultiEnv,
}

impl PreparedModality {
    const fn label(self) -> &'static str {
        match self {
            Self::Tabular => "tabular",
            Self::Series => "series",
            Self::Panel => "panel",
            Self::MultiEnv => "multi_env",
        }
    }

    const fn estimate_entry(self) -> &'static str {
        match self {
            Self::Tabular => "estimate",
            Self::Series => "estimate_series",
            Self::Panel => "estimate_panel",
            Self::MultiEnv => "estimate_multi_env",
        }
    }
}

/// Durable handle: fixed schema, graph, query, and estimator; swap data and re-estimate.
///
/// Created via [`Study::prepare`]. Discovery / review-required graphs are refused —
/// prepare is for the interactive estimate click path on an already-accepted artifact.
///
/// **Frozen at prepare:** schema (names, types, order); graph / `AcceptedGraph`
/// or supplied graph-posterior atoms and weights; query identity; identifier;
/// observation / transport / interference assumptions; target-population
/// bindings.
///
/// **Estimate click:** same-schema data; estimator numeric knobs, latency,
/// seeds, bootstrap; `ExecutionContext` budget / cancellation. Does not
/// re-identify or recompile the logical plan.
///
/// **Refute click:** same frozen identification and estimand; schema-gated
/// data and suite. Currently [`CausalQuery::AverageEffect`] only.
///
/// **Re-prepare required:** any frozen field change, including schema
/// mismatch. Changing a frozen field on [`Self::refresh`] is an error, not a
/// silent recompile. `analyze` / [`Study::run`] is sugar over identify →
/// prepare → estimate.
#[derive(Clone, Debug)]
pub struct PreparedStudy<S = SampledPreparedState> {
    pub(crate) state: S,
}

/// Scientific target and per-atom products compiled together at preparation.
/// Keeping the query beside its products makes an accidental internal retarget
/// a typed refusal instead of publication of a stale atom program.
#[derive(Clone, Debug)]
struct RetainedPosteriorPrograms {
    query: Arc<[u8]>,
    programs: antecedent_io::GraphPosteriorProgramsWire,
}

/// Retained state for sampled-data modalities of the common prepared handle.
#[derive(Clone, Debug)]
pub struct SampledPreparedState {
    /// Frozen analysis config (data slot replaced only by checked refresh).
    analysis: Study,
    /// Whether data-dependent statistical operations have been materialized.
    materialized: bool,
    /// Data-independent contract layers, compiled once per handle state.
    program_cache: std::sync::OnceLock<Arc<crate::analysis::contract::ProgramPayloads>>,
    /// Immutable base contract; execution claims and reasoning are rebound per result.
    contract_cache: std::sync::OnceLock<Arc<crate::analysis::contract::CompiledContract>>,
    /// Bounded metadata sealed before a prepared state becomes visible.
    inspection_header: std::sync::OnceLock<Arc<crate::analysis::contract::PreparedInspection>>,
    /// Per-atom scientific programs compiled under the prepare caller's resource
    /// and cancellation context. Contract inspection only reads these products.
    posterior_programs: Option<Arc<RetainedPosteriorPrograms>>,
    /// Caller-admitted physical plan. A compatible estimator-configuration refresh
    /// re-lowers its numerical work and requotes admission against the new options.
    plan: std::sync::Arc<PhysicalExecutionPlan>,
    /// Schema fingerprint from prepare-time data.
    schema: CausalSchema,
    /// Data modality frozen at prepare; each estimate / refresh entry point
    /// accepts only its own modality.
    modality: PreparedModality,
    /// Sampling regularity frozen for series and panel prepares (`None` = tabular).
    time_regularity: Option<antecedent_data::SamplingRegularity>,
    /// Cross-fitted AIPW scores frozen at prepare when the cell can export them.
    score_table: Option<antecedent_estimate::ScoreTable>,
    /// Sealed route selection and its checked operation, or the ordinary dispatcher
    /// for a configuration no sealed operation admits.
    execution: PreparedExecution,
    /// What each refreshed series period invalidated (see
    /// [`antecedent_state::TemporalProduct`]); empty until a period is applied.
    temporal_log: antecedent_state::InvalidationLog,
}

impl std::ops::Deref for PreparedStudy {
    type Target = SampledPreparedState;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}
impl std::ops::DerefMut for PreparedStudy {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl PreparedStudy {
    /// Retained scientific/identification/operator program of a checked linear operation.
    #[must_use]
    pub fn compiled_linear_program(
        &self,
    ) -> Option<&crate::analysis::compiled_linear::CompiledLinearProgram> {
        self.execution.linear_operation().map(|operation| operation.compiled.as_ref())
    }
}

pub(crate) use joint_response::joint_target_admission;
