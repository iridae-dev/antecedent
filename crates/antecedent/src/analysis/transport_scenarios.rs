//! Prepared finite graph/selection scenario sets for one transport question.
//!
//! Preparation decides every scenario once and compiles each identified one;
//! estimation evaluates the retained plans and never re-identifies. The report
//! keeps every scenario whatever its status, a structural envelope over the
//! identified ones, and, only for declared weights, a report that never
//! renormalizes. No inferential statement across scenarios is licensed.
//!
//! The question is the classical `P*(y | do(x))` or, through
//! [`StudyBuilder::conditional_transport_scenarios`], the conditional
//! `P*(y | do(x), w)` of the bounded ADMG conditional row (2.2B B1), decided per
//! scenario on the same shared budget and reported in the same envelope.
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
    ClassicalTransportQuery, ConditionalTransportQuery,
    sid::scenarios::{ScenarioQuestion, TransportScenarioSet, decide_scenario_question},
};
use antecedent_io::IoError;
use antecedent_io::transport_scenario_artifact::{
    TransportScenarioArtifactWire, TransportScenarioConsumeLimits, scenario_refusal,
};

/// A decided, compiled scenario set.
#[derive(Clone, Debug)]
pub struct PreparedTransportScenarios {
    inner: PreparedScenarioSet,
    catalog: EvidenceCatalog,
}

/// Decide every scenario once. A catalog or question disagreeing with the
/// shared coordinate schema refuses before any search.
fn decide(
    set: &TransportScenarioSet,
    question: &ScenarioQuestion,
    catalog: &EvidenceCatalog,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<antecedent_identify::sid::scenarios::ScenarioSetDecision, IoError> {
    set.check_catalog(catalog).map_err(scenario_refusal)?;
    set.check_question(question).map_err(scenario_refusal)?;
    decide_scenario_question(set, question, catalog, budget, ctx).map_err(|error| {
        // The conditional route's bound and query refusals keep their
        // `admg_transport.*` reason code and detail.
        if question.is_conditional() {
            antecedent_io::admg_conditional_transport_artifact::admg_conditional_identification_error(
                error,
            )
        } else {
            error.into()
        }
    })
}

impl StudyBuilder {
    /// Decide every scenario once against the shared question and catalog, and
    /// compile each identified scenario against supplied exact laws.
    ///
    /// `budget` is the one shared search budget of the set: each scenario
    /// entered is charged at depth one on top of the memory the scenarios
    /// already decided keep holding, and each scenario's search, s-hedge check
    /// and verification replay charge it too, against the context's hard memory
    /// limit and cancellation.
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
        let decision = decide(set, &query.into(), &catalog, budget, ctx)?;
        let inner = prepare_transport_scenarios(decision, data, request, limits, ctx)
            .map_err(estimate_err)?;
        Ok(PreparedTransportScenarios { inner, catalog })
    }

    /// Decide every scenario once against one conditional question
    /// `P*(y | do(x), w)` by the bounded ADMG conditional route (2.2B B1), and
    /// compile each identified scenario through that route's exact prepared
    /// path. The request binds exactly the treatments and the conditioned
    /// variables. The shared budget, envelope, masses, receipts and artifact
    /// behave as in [`Self::transport_scenarios`]; a not-certified scenario
    /// carries the inspection-only obstruction candidate and is never
    /// structurally unidentified. Exact laws only: counted laws are refused
    /// (`cell_not_licensed`).
    ///
    /// # Errors
    /// As [`Self::transport_scenarios`], plus `route_not_supported` /
    /// `admg_transport.bounds_exceeded` (more than 6 observed variables, 3
    /// treatments or 3 conditioned variables) and `invalid_argument` /
    /// `admg_transport.invalid_query` or `admg_transport.invalid_request`.
    #[allow(clippy::too_many_arguments)] // Every premise of the preparation, explicitly.
    pub fn conditional_transport_scenarios(
        set: &TransportScenarioSet,
        query: ConditionalTransportQuery,
        catalog: EvidenceCatalog,
        budget: SearchLimits,
        data: ExactTransportData,
        request: Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedTransportScenarios, IoError> {
        let decision = decide(set, &query.into(), &catalog, budget, ctx)?;
        let inner = prepare_transport_scenarios(decision, data, request, limits, ctx)
            .map_err(estimate_err)?;
        Ok(PreparedTransportScenarios { inner, catalog })
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
        let decision = decide(set, &query.into(), &catalog, budget, ctx)?;
        let inner = prepare_empirical_transport_scenarios(
            decision,
            input,
            max_joint_cells,
            request,
            limits,
            ctx,
        )
        .map_err(estimate_err)?;
        Ok(PreparedTransportScenarios { inner, catalog })
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

    /// Export a report with every premise needed for independent replay. A
    /// report truncated by an operation, depth or memory bound exports with its
    /// recorded limits; one truncated by cancellation does not.
    ///
    /// # Errors
    /// The premises do not encode, or the report was truncated by cancellation.
    pub fn export(&self, report: &ScenarioSetReport) -> Result<Vec<u8>, IoError> {
        TransportScenarioArtifactWire::checked(&self.inner, &self.catalog, report)?.export()
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
