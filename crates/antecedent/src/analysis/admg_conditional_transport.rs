//! Prepared execution of an ADMG conditional transport formula (2.2B B1).
//!
//! Preparation re-checks one decided, catalog-bound conditional derivation
//! against the selection diagram and compiles one conditional plan per request.
//! Estimation evaluates those plans and never searches again. The route is
//! point-only: exact laws carry no sampling uncertainty, so counted laws are
//! refused (`cell_not_licensed`) and no interval is computed, stored or
//! accepted. Refresh replaces laws only for the snapshots the frozen catalog
//! binds; the proof is unchanged.
use super::StudyBuilder;
use super::transport_common::{compile_plans, estimate_err, evaluate_plans};
use antecedent_core::{ExecutionContext, SearchLimits};
use antecedent_estimate::AdmgConditionalExactPlan;
use antecedent_expr::{Assignment, ExactDistribution, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    BoundConditionalTransportFunctional, ConditionalNonTransportabilityProof,
};
use antecedent_io::IoError;
use antecedent_io::admg_conditional_transport_artifact::{
    AdmgConditionalArtifactInput, AdmgConditionalArtifactWire, AdmgConditionalConsumeLimits,
    AdmgConditionalObstructionWire, admg_conditional_identification_error,
};

/// A prepared conditional formula: frozen proof, bound catalog, laws and plans.
#[derive(Clone, Debug)]
pub struct PreparedAdmgConditionalTransport {
    diagram: SelectionDiagram,
    functional: BoundConditionalTransportFunctional,
    search: SearchLimits,
    data: ExactTransportData,
    requests: Vec<Assignment>,
    limits: ExactEvaluationLimits,
    plans: Vec<AdmgConditionalExactPlan>,
}

/// One execution: the conditional point of every request.
#[derive(Clone, Debug)]
pub struct AdmgConditionalResult {
    distributions: Vec<ExactDistribution>,
}

impl AdmgConditionalResult {
    /// The conditional point distribution of every request, in request order.
    #[must_use]
    pub fn distributions(&self) -> &[ExactDistribution] {
        &self.distributions
    }

    /// Export this result with its checked proof, catalog, laws and requests, for
    /// an artifact consumer that re-derives everything; `variable_names` (or
    /// empty) is bound into the verified identity.
    ///
    /// # Errors
    /// The premises do not encode or exceed the route's bounds.
    pub fn export_named(
        &self,
        prepared: &PreparedAdmgConditionalTransport,
        variable_names: &[String],
    ) -> Result<Vec<u8>, IoError> {
        AdmgConditionalArtifactWire::checked(&AdmgConditionalArtifactInput {
            diagram: &prepared.diagram,
            functional: &prepared.functional,
            search: prepared.search,
            data: &prepared.data,
            requests: &prepared.requests,
            limits: prepared.limits,
            variable_names,
            results: &self.distributions,
        })?
        .export()
    }
}

/// Consume an ADMG conditional transport artifact: re-check the proof with the
/// independent checkers and replay the decision under the producer's stored
/// limits, re-bind the joint and recompute every point bit for bit, without
/// fetching data. The replay re-runs the producer's own search and evaluator, so
/// it is independent of the artifact, not of the implementation.
///
/// # Errors
/// Any reconstruction or replay failure.
pub fn consume_admg_conditional_transport_artifact(
    bytes: &[u8],
    limits: AdmgConditionalConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<AdmgConditionalResult, IoError> {
    let consumed = AdmgConditionalArtifactWire::consume_with_limits(bytes, limits, ctx)?;
    Ok(AdmgConditionalResult { distributions: consumed.distributions })
}

/// Export a proven obstruction (the proof record with its two-model witness)
/// decided on `diagram`; the proof is re-checked before it is written and
/// `variable_names` (or empty) is bound into the verified identity.
///
/// # Errors
/// The proof does not re-check on `diagram`, or the premises do not encode.
pub fn export_admg_conditional_obstruction(
    diagram: &SelectionDiagram,
    proof: &ConditionalNonTransportabilityProof,
    variable_names: &[String],
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, IoError> {
    AdmgConditionalObstructionWire::checked(diagram, proof, variable_names, ctx)?.export()
}

/// Verify an exported proven obstruction: re-verify the two-model witness by
/// exact enumeration and re-check the moves and the s-hedge. Nothing stored is
/// trusted; no data is fetched.
///
/// # Errors
/// Any decoding, bound, digest or verification failure, as a typed refusal.
pub fn consume_admg_conditional_obstruction_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<ConditionalNonTransportabilityProof, IoError> {
    Ok(AdmgConditionalObstructionWire::consume(bytes, ctx)?.proof)
}

impl StudyBuilder {
    /// Prepare exact-law evaluation of a decided conditional functional for
    /// `requests` (each binding exactly the treatments and conditioned
    /// variables). The derivation is re-checked against `diagram` under `search`,
    /// the limits it was decided under, so a functional decided on another graph
    /// refuses.
    ///
    /// # Errors
    /// No request, a derivation that does not check on `diagram`, counted laws,
    /// or laws, requests or resources that do not compile.
    pub fn admg_conditional_transport(
        diagram: SelectionDiagram,
        functional: BoundConditionalTransportFunctional,
        search: SearchLimits,
        data: ExactTransportData,
        requests: Vec<Assignment>,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedAdmgConditionalTransport, IoError> {
        if requests.is_empty() {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("invalid_argument"),
                message: "admg_transport.invalid_request: prepare at least one request".into(),
            });
        }
        functional
            .derivation()
            .recheck(&diagram, search, ctx)
            .map_err(admg_conditional_identification_error)?;
        let plans = compile(&functional, &data, &requests, limits, ctx)?;
        Ok(PreparedAdmgConditionalTransport {
            diagram,
            functional,
            search,
            data,
            requests,
            limits,
            plans,
        })
    }
}

/// One plan per request. Counted laws are refused here by the estimate-level
/// preparation (`cell_not_licensed`, `admg_transport.interval_withheld`) before
/// anything compiles: this route has no sampling theory.
fn compile(
    functional: &BoundConditionalTransportFunctional,
    data: &ExactTransportData,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<Vec<AdmgConditionalExactPlan>, IoError> {
    compile_plans(requests, |request| {
        antecedent_estimate::prepare_exact_admg_conditional_transport(
            functional,
            data.clone(),
            request,
            limits,
            ctx,
        )
        .map_err(estimate_err)
    })
}

impl PreparedAdmgConditionalTransport {
    /// Evaluate every retained plan. Exact laws are point-only.
    ///
    /// # Errors
    /// Cancellation, an evaluation refusal, or a zero-mass conditioning event
    /// (`transport_support_failure`).
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<AdmgConditionalResult, IoError> {
        let distributions =
            evaluate_plans("admg conditional transport estimate", &self.plans, ctx, |plan| {
                plan.evaluate(ctx).map_err(estimate_err)
            })?;
        Ok(AdmgConditionalResult { distributions })
    }

    /// Replace the laws for the snapshots the frozen catalog binds and recompile.
    /// The proof, bindings and requests are unchanged.
    ///
    /// # Errors
    /// Counted laws, a law that does not match the frozen catalog, or cancellation.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "admg conditional transport refresh")
            .map_err(estimate_err)?;
        let plans = compile(&self.functional, &data, &self.requests, self.limits, ctx)?;
        Ok(Self { data, plans, ..self.clone() })
    }

    /// The frozen, catalog-bound functional.
    #[must_use]
    pub const fn functional(&self) -> &BoundConditionalTransportFunctional {
        &self.functional
    }

    /// Retained requests.
    #[must_use]
    pub fn requests(&self) -> &[Assignment] {
        &self.requests
    }

    /// The compiled conditional plan of every request, in request order.
    #[must_use]
    pub fn plans(&self) -> &[AdmgConditionalExactPlan] {
        &self.plans
    }
}
