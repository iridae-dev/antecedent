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
use antecedent_estimate::StatisticalTransportInput;
use antecedent_estimate::transport_scenarios::{
    PreparedScenarioSet, ScenarioSetReport, prepare_empirical_transport_scenarios,
    prepare_transport_scenarios,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    ClassicalTransportQuery,
    sid::scenarios::{TransportScenarioSet, decide_transport_scenarios},
};
use antecedent_io::IoError;
use antecedent_io::transport_scenario_artifact::{
    TransportScenarioArtifactWire, TransportScenarioConsumeLimits, scenario_refusal,
};

/// A decided, compiled scenario set.
#[derive(Clone, Debug)]
pub struct PreparedTransportScenarios {
    inner: PreparedScenarioSet,
    query: ClassicalTransportQuery,
    catalog: EvidenceCatalog,
}

/// Decide every scenario once. A catalog or question disagreeing with the
/// shared coordinate schema refuses before any search.
fn decide(
    set: &TransportScenarioSet,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<antecedent_identify::sid::scenarios::ScenarioSetDecision, IoError> {
    set.check_catalog(catalog).map_err(scenario_refusal)?;
    set.check_query(query).map_err(scenario_refusal)?;
    Ok(decide_transport_scenarios(set, query, catalog, budget, ctx)?)
}

impl StudyBuilder {
    /// Decide every scenario once against the shared question and catalog, and
    /// compile each identified scenario against supplied exact laws.
    ///
    /// `budget` is the one shared search budget of the set: each scenario
    /// entered is charged at depth one with the decision's live bytes, and each
    /// scenario's search and verification replay charge it too, against the
    /// context's hard memory limit and cancellation.
    ///
    /// # Errors
    /// `schema_mismatch` for a catalog, law or request outside the shared
    /// coordinate schema; invalid query, catalog or laws; or a request that
    /// fails for every scenario.
    #[allow(clippy::too_many_arguments)] // Every premise of the preparation, explicitly.
    pub fn transport_scenarios(
        set: &TransportScenarioSet,
        query: ClassicalTransportQuery,
        catalog: EvidenceCatalog,
        budget: SearchLimits,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTransportScenarios, IoError> {
        let decision = decide(set, &query, &catalog, budget, ctx)?;
        let inner = prepare_transport_scenarios(decision, data, request, limits, ctx)
            .map_err(estimate_err)?;
        Ok(PreparedTransportScenarios { inner, query, catalog })
    }

    /// As [`Self::transport_scenarios`], with empirical plug-in frequency tables
    /// fitted once from finite samples (plus any supplied exact laws) in place
    /// of supplied laws. Per-scenario points only; no interval.
    ///
    /// # Errors
    /// As [`Self::transport_scenarios`], plus samples the plug-in cannot fit.
    #[allow(clippy::too_many_arguments)] // Every premise of the preparation, explicitly.
    pub fn transport_scenarios_empirical(
        set: &TransportScenarioSet,
        query: ClassicalTransportQuery,
        catalog: EvidenceCatalog,
        budget: SearchLimits,
        input: StatisticalTransportInput,
        max_joint_cells: usize,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTransportScenarios, IoError> {
        let mut input = input;
        super::statistical::canonicalize_input(&mut input)?;
        let decision = decide(set, &query, &catalog, budget, ctx)?;
        let inner = prepare_empirical_transport_scenarios(
            decision,
            input,
            max_joint_cells,
            request,
            limits,
            ctx,
        )
        .map_err(estimate_err)?;
        Ok(PreparedTransportScenarios { inner, query, catalog })
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

    /// Refit the empirical plug-in from new samples and recompile; decisions
    /// are kept.
    ///
    /// # Errors
    /// Samples the plug-in cannot fit, or laws outside the shared schema.
    pub fn refresh_empirical(
        &self,
        input: StatisticalTransportInput,
        max_joint_cells: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let mut input = input;
        super::statistical::canonicalize_input(&mut input)?;
        let inner =
            self.inner.refresh_empirical(input, max_joint_cells, ctx).map_err(estimate_err)?;
        Ok(Self { inner, ..self.clone() })
    }

    /// Inference across scenarios that share data is not licensed; this always
    /// refuses with `scenario_aggregate_not_licensed`.
    ///
    /// # Errors
    /// Always.
    pub fn aggregate_interval(&self) -> Result<(), IoError> {
        Err(IoError::Refused {
            code: antecedent_core::reason_code!("scenario_aggregate_not_licensed"),
            message: "scenarios.shared_data_aggregate: only per-scenario results are licensed"
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
        TransportScenarioArtifactWire::checked(&self.inner, &self.query, &self.catalog, report)?
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
