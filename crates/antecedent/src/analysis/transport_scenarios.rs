//! Prepared finite graph/selection scenario sets for one transport question.
//!
//! Preparation decides every scenario once and compiles each identified one;
//! estimation evaluates the retained plans and never re-identifies. The report
//! keeps every scenario whatever its status, a structural envelope over the
//! identified ones, and, only for declared weights, a report that never
//! renormalizes. No inferential statement across scenarios is licensed.
use super::StudyBuilder;
use super::transport_common::estimate_err;
use antecedent_core::{EvidenceCatalog, ExecutionContext, SearchLimits};
use antecedent_estimate::transport_scenarios::{
    PreparedScenarioSet, ScenarioSetReport, prepare_transport_scenarios,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    ClassicalTransportQuery, SidLimits,
    sid::scenarios::{TransportScenarioSet, decide_transport_scenarios},
};
use antecedent_io::IoError;
use antecedent_io::transport_scenario_artifact::{
    TransportScenarioArtifactWire, TransportScenarioConsumeLimits,
};

/// A decided, compiled scenario set.
#[derive(Clone, Debug)]
pub struct PreparedTransportScenarios {
    inner: PreparedScenarioSet,
    query: ClassicalTransportQuery,
    catalog: EvidenceCatalog,
    identification: SidLimits,
    scenario_budget: SearchLimits,
}

impl StudyBuilder {
    /// Decide every scenario once against the shared question and catalog, and
    /// compile each identified scenario against `data`.
    ///
    /// # Errors
    /// Invalid query, catalog or laws, or a request that fails for every scenario.
    #[allow(clippy::too_many_arguments)] // Every premise of the preparation, explicitly.
    pub fn transport_scenarios(
        set: &TransportScenarioSet,
        query: ClassicalTransportQuery,
        catalog: EvidenceCatalog,
        identification: SidLimits,
        scenario_budget: SearchLimits,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTransportScenarios, IoError> {
        let decision = decide_transport_scenarios(
            set,
            &query,
            &catalog,
            identification,
            scenario_budget,
            ctx,
        )?;
        let inner = prepare_transport_scenarios(decision, data, request, limits, ctx)
            .map_err(estimate_err)?;
        Ok(PreparedTransportScenarios { inner, query, catalog, identification, scenario_budget })
    }
}

impl PreparedTransportScenarios {
    /// Evaluate every compiled scenario and report all of them.
    ///
    /// # Errors
    /// A numerical failure or cancellation.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<ScenarioSetReport, IoError> {
        self.inner.evaluate(ctx).map_err(estimate_err)
    }

    /// Recompile identified scenarios against new laws; decisions are kept.
    ///
    /// # Errors
    /// Laws that do not match the catalog in a way that fails every scenario.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        Ok(Self { inner: self.inner.refresh(data, ctx).map_err(estimate_err)?, ..self.clone() })
    }

    /// Inference across scenarios that share data is not licensed; this always
    /// refuses with `scenario_aggregate_not_licensed`.
    ///
    /// # Errors
    /// Always.
    pub fn aggregate_interval(&self) -> Result<(), IoError> {
        Err(IoError::Refused {
            code: antecedent_core::reason_code!("scenario_aggregate_not_licensed"),
            message:
                "scenarios.shared_data_aggregate: only per-scenario results and the structural \
                      envelope are licensed"
                    .into(),
        })
    }

    /// The frozen per-scenario decisions and compiled plans.
    #[must_use]
    pub const fn prepared(&self) -> &PreparedScenarioSet {
        &self.inner
    }

    /// Export a report with every premise needed for independent replay.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn export(&self, report: &ScenarioSetReport) -> Result<Vec<u8>, IoError> {
        TransportScenarioArtifactWire::checked(
            &self.inner,
            &self.query,
            &self.catalog,
            self.identification,
            self.scenario_budget,
            report,
        )?
        .export()
    }
}

/// Independently replay a scenario artifact and return the recomputed report.
///
/// # Errors
/// Any limit, digest, reconstruction or report mismatch.
pub fn consume_transport_scenarios_artifact(
    bytes: &[u8],
    limits: TransportScenarioConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<ScenarioSetReport, IoError> {
    Ok(TransportScenarioArtifactWire::consume_with_limits(bytes, limits, ctx)?.1)
}
