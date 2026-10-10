//! DAG completions of a supplied CPDAG as one finite scenario set, and the
//! shared-data covariance of scenario estimates (2.3A X2).
//!
//! Preparation enumerates the completions of the CPDAG (at most six fully
//! observed nodes), binds evidence to each completion, decides each through the
//! classical catalog route on one shared budget and compiles each identified one
//! against supplied exact laws; estimation evaluates the retained plans and never
//! re-identifies. The report keeps every completion whatever its status, an
//! unweighted structural envelope over the identified ones, and the identified,
//! unidentified, unevaluated and never-enumerated counts apart, never
//! renormalized. No graph probability is assigned and no inferential statement
//! across completions is licensed.
//!
//! [`StudyBuilder::scenario_shared_covariance`] is the separate point-only cell:
//! one joint covariance matrix of per-scenario plug-in estimates from the same
//! complete-row sample, never an interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::StudyBuilder;
use antecedent_core::{ExecutionContext, SearchLimits};
use antecedent_estimate::cpdag_scenarios::{CpdagScenarioReport, PreparedCpdagScenarios};
use antecedent_estimate::scenario_covariance::ScenarioCovariance;
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::cpdag_completion::{CpdagCompletionInput, CpdagEvidence};
use antecedent_identify::sid::scenarios::ScenarioCoordinate;
use antecedent_io::IoError;
use antecedent_io::cpdag_completion_artifact::{
    CpdagCompletionArtifactWire, CpdagConsumeLimits, CpdagScenarioPremises,
};
use antecedent_io::scenario_covariance_artifact::{
    CovarianceSpec, ScenarioCovarianceArtifactWire, ScenarioCovarianceConsumeLimits,
};
use std::sync::Arc;

/// A decided, compiled set of CPDAG completions.
#[derive(Clone, Debug)]
pub struct PreparedCpdagCompletionScenarios {
    inner: PreparedCpdagScenarios,
    premises: CpdagScenarioPremises,
}

impl StudyBuilder {
    /// Enumerate every DAG completion of `input`, bind `evidence` to each, decide
    /// each once on the shared `budget` and compile each identified completion
    /// against supplied exact laws.
    ///
    /// # Errors
    /// `route_not_supported` / `cpdag_scenarios.bounds_exceeded` (more than six
    /// nodes), `cpdag_scenarios.selection_or_latent`; `invalid_argument` /
    /// `cpdag_scenarios.not_a_cpdag`, `cpdag_scenarios.evidence_identity_mismatch`;
    /// `schema_mismatch` for a catalog, law or request outside the shared
    /// coordinate schema; invalid query, catalog or laws; or a request that fails
    /// for every completion. A budget or cancellation stop is not an error: it
    /// comes back in the report's receipt.
    #[allow(clippy::too_many_arguments)] // Every premise of the preparation, explicitly.
    pub fn cpdag_completion_scenarios(
        input: CpdagCompletionInput,
        coordinates: Arc<[ScenarioCoordinate]>,
        query: ClassicalTransportQuery,
        evidence: CpdagEvidence,
        budget: SearchLimits,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedCpdagCompletionScenarios, IoError> {
        let premises = CpdagScenarioPremises {
            input,
            coordinates,
            query,
            evidence,
            budget,
            memory_limit_bytes: ctx.memory.hard_limit_bytes,
            data,
            request,
            evaluation: limits,
        };
        let inner = premises.prepare(ctx)?;
        Ok(PreparedCpdagCompletionScenarios { inner, premises })
    }

    /// The joint sampling covariance of the declared scenario estimates over one
    /// shared row table: point only, never an interval.
    ///
    /// # Errors
    /// `route_not_supported` / `scenario_covariance.unknown_dependence` for a
    /// different snapshot, unit list or undeclared dependence; the other
    /// `scenario_covariance.*` refusals of the estimators.
    pub fn scenario_shared_covariance(
        spec: &CovarianceSpec,
        ctx: &ExecutionContext,
    ) -> Result<ScenarioCovariance, IoError> {
        spec.compute(ctx)
    }
}

impl PreparedCpdagCompletionScenarios {
    /// Evaluate every compiled completion and report all of them.
    ///
    /// # Errors
    /// A numerical failure or cancellation.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<CpdagScenarioReport, IoError> {
        Ok(self.inner.evaluate(ctx)?)
    }

    /// The decided completions and compiled plans.
    #[must_use]
    pub const fn prepared(&self) -> &PreparedCpdagScenarios {
        &self.inner
    }

    /// The premises the completions were decided under.
    #[must_use]
    pub const fn premises(&self) -> &CpdagScenarioPremises {
        &self.premises
    }

    /// Export a report with every premise needed for independent replay. A report
    /// truncated by an operation, depth or memory bound exports with its recorded
    /// limits; one truncated by cancellation does not.
    ///
    /// # Errors
    /// The premises do not encode, or the report was truncated by cancellation.
    pub fn export(&self, report: &CpdagScenarioReport) -> Result<Vec<u8>, IoError> {
        CpdagCompletionArtifactWire::checked(&self.premises, &self.inner, report)?.export()
    }
}

/// Independently replay a completion artifact and return the recomputed report.
///
/// # Errors
/// Any limit, digest, reconstruction or report mismatch.
pub fn consume_cpdag_scenarios_artifact(
    bytes: &[u8],
    limits: CpdagConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CpdagScenarioReport, IoError> {
    Ok(CpdagCompletionArtifactWire::consume_with_limits(bytes, limits, ctx)?.1)
}

/// Export a covariance with its row table and scenario declarations.
///
/// # Errors
/// The premises do not encode.
pub fn export_scenario_covariance(
    spec: &CovarianceSpec,
    covariance: &ScenarioCovariance,
) -> Result<Vec<u8>, IoError> {
    ScenarioCovarianceArtifactWire::checked(spec, covariance)?.export()
}

/// Independently recompute an exported covariance and return it.
///
/// # Errors
/// Any limit, digest, reconstruction or result mismatch, or a scenario declared
/// on a different snapshot (`scenario_covariance.unknown_dependence`).
pub fn consume_scenario_covariance_artifact(
    bytes: &[u8],
    limits: ScenarioCovarianceConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<ScenarioCovariance, IoError> {
    Ok(ScenarioCovarianceArtifactWire::consume_with_limits(bytes, limits, ctx)?.1)
}
