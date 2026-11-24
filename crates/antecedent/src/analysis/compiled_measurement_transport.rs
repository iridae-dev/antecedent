//! Compiled transport of a measurement-error model to another population.
//!
//! Two routes, one per way the error is related across populations:
//!
//! * [`Study::compile_measurement_transport`] decides, once, whether a model established in a
//!   source population may correct an indicator observed in the analysis population, by the
//!   library's transport decision (`antecedent_estimate::measurement_transport::
//!   transport_measurement_model`): a model established for the target is used as is, a
//!   declared invariance of every error parameter the correction uses re-binds the source
//!   model, a declared population-differential error never borrows another population's
//!   rates, and no declaration refuses naming the premise. The receipt and the re-bound model
//!   are bound into the program; [`CompiledMeasurementTransport::execute`] forms the corrected
//!   functional on the re-bound model through
//!   [`super::transported_measurement::transport_measurement_estimate`], so every prepare check
//!   of the measurement route still runs.
//! * [`Study::compile_pooled_differential_transport`] binds the validation counts of two or
//!   more source populations, the target's optional own validation and its recorded positives;
//!   [`CompiledPooledDifferentialTransport::execute`] pools the sensitivity and specificity
//!   logits by random effects and propagates the between-population variance into the
//!   Rogan-Gladen correction through `antecedent_estimate::measurement_transport_pooled::
//!   transport_pooled_differential_prevalence`, drawing from the context's seed as the direct
//!   call does.
//!
//! Neither route claims calibrated uncertainty: the measurement route's standard error is the
//! established uncalibrated delta-method diagnostic, conditional on the premise, and the pooled
//! route's interval is a simulation interval reported `not_measured`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ExternalScientificObject, ExternalTrustState};
use antecedent_core::measurement_model::MeasurementModel;
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_estimate::measurement_transport::{
    PopulationErrorStructure, TransportedMeasurementModel,
};
use antecedent_estimate::measurement_transport_pooled::{
    PooledDifferentialCorrection, PooledDifferentialRequest, PopulationValidation,
    transport_pooled_differential_prevalence,
};

use super::compiled_point_program::PointProgram;
use super::measurement_model::{
    MeasurementAnalysisBinding, MeasurementInput, MeasurementModelRefusal,
};
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::transported_measurement::{
    TransportedMeasurementEstimate, transport_measurement, transport_measurement_estimate,
};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};
use crate::planner::{PlanStage, PlanView};

/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A measurement model to carry to the analysis population, and the data to correct there.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MeasurementTransportRequest {
    /// The models established so far, each for its own population.
    pub models: Vec<MeasurementModel>,
    /// How the error is declared to relate across populations; `None` declares nothing.
    pub structure: Option<PopulationErrorStructure>,
    /// The population, regime, arm and representation of the analysis.
    pub binding: MeasurementAnalysisBinding,
    /// The outcome-side data the corrected functional reads.
    pub input: MeasurementInput,
    /// The provider object and its trust, for the external latent posterior route.
    pub provider: Option<(ExternalScientificObject, ExternalTrustState)>,
}

impl MeasurementTransportRequest {
    /// A request with no provider object.
    #[must_use]
    pub fn new(
        models: Vec<MeasurementModel>,
        structure: Option<PopulationErrorStructure>,
        binding: MeasurementAnalysisBinding,
        input: MeasurementInput,
    ) -> Self {
        Self { models, structure, binding, input, provider: None }
    }

    /// The same request with the provider object of an external latent posterior.
    #[must_use]
    pub fn with_provider(
        mut self,
        object: ExternalScientificObject,
        trust: ExternalTrustState,
    ) -> Self {
        self.provider = Some((object, trust));
        self
    }
}

/// A decided measurement transport bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledMeasurementTransport {
    request: MeasurementTransportRequest,
    transported: TransportedMeasurementModel,
    lowered: Arc<LoweredProgram>,
}

fn lowering_refusal(error: &CausalError) -> MeasurementModelRefusal {
    MeasurementModelRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: antecedent_core::reason_details::measurement_model::INPUT_INVALID.to_owned(),
        message: format!("the measurement transport program cannot be lowered: {error}"),
        remedy: None,
    }
}

impl Study {
    /// Decide whether a measurement model may correct the analysis population's indicator and
    /// bind the decision, the re-bound model and the data into a compiled program.
    ///
    /// # Errors
    ///
    /// The refusal of the transport decision naming the unmet premise
    /// (`measurement_model.invariance_premise_required`, `.invariance_not_declared`, ...), or
    /// cancellation.
    pub fn compile_measurement_transport(
        request: MeasurementTransportRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledMeasurementTransport, MeasurementModelRefusal> {
        if ctx.cancellation.is_cancelled() {
            return Err(MeasurementModelRefusal {
                code: antecedent_core::reason_code!("cancelled_no_claim"),
                detail: antecedent_core::reason_details::measurement_model::CANCELLED.to_owned(),
                message: "cancelled before the transport was decided; this is not a verdict \
                          on the data"
                    .to_owned(),
                remedy: None,
            });
        }
        let transported =
            transport_measurement(&request.models, request.structure.as_ref(), &request.binding)?;
        let receipt = transported.receipt();
        let mut question = CanonicalWriter::new("antecedent.measurement_transport.question.v1");
        question
            .bytes(receipt.identity.as_bytes())
            .bytes(transported.model().identities().model.as_bytes())
            .text(&format!("{:?}", request.binding))
            .text(&format!("{:?}", request.structure));
        if let Some((object, trust)) = &request.provider {
            question.text(&format!("{object:?}")).text(&format!("{trust:?}"));
        }
        let question = question.finish();
        let source = CanonicalWriter::digest_bytes(
            "antecedent.measurement_transport.input.v1",
            format!("{:?}", request.input).as_bytes(),
        );
        let lowered = PointProgram {
            cell: ProgramCell::MeasurementModelTransport,
            plan_id: "measurement_model_transport.v1",
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            graph_class: "declared_population_error_structure",
            query_variables: Arc::from([]),
            question,
            source,
            source_role: EvidenceRole::Observations,
            evaluator: "antecedent_estimate::measurement_transport::transport_measurement_model",
            supplied: Vec::new(),
            rows: 1,
            columns: 1,
            input_bytes: 0,
            claim: antecedent_contract::ClaimKind::Point,
        }
        .lower()
        .map_err(|error| lowering_refusal(&error))?;
        Ok(CompiledMeasurementTransport { request, transported, lowered })
    }
}

impl CompiledProgramHandle for CompiledMeasurementTransport {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledMeasurementTransport {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without correcting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// The re-bound model and its receipt, as the transport decision issued them.
    #[must_use]
    pub const fn transported(&self) -> &TransportedMeasurementModel {
        &self.transported
    }

    /// Form the corrected functional on the re-bound model.
    ///
    /// # Errors
    ///
    /// Every prepare check and estimator refusal of the measurement route on the re-bound
    /// model, cancellation, or the support gate once it is on.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TransportedMeasurementEstimate, MeasurementModelRefusal> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| MeasurementModelRefusal {
                code: antecedent_core::reason_code!("route_not_supported"),
                detail: antecedent_core::reason_details::measurement_model::INPUT_INVALID
                    .to_owned(),
                message: refusal.to_string(),
                remedy: None,
            })?;
        }
        let request = &self.request;
        transport_measurement_estimate(
            &request.models,
            request.structure.as_ref(),
            request.binding.clone(),
            request.input.clone(),
            request.provider.as_ref().map(|(object, trust)| (object, trust)),
            &scoped,
        )
    }
}

/// Validation counts of the source populations and the target's recorded positives.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PooledDifferentialTransportRequest {
    /// Validated source populations; at least two, distinct from the target.
    pub sources: Vec<PopulationValidation>,
    /// The target's own validation sample, when it has one.
    pub target_validation: Option<PopulationValidation>,
    /// Target population label.
    pub target_population: String,
    /// Recorded positives in the target sample.
    pub observed_positive: u64,
    /// Target sample size.
    pub observed_total: u64,
    /// Propagation draws; at least 1000.
    pub draws: usize,
}

impl PooledDifferentialTransportRequest {
    /// A request without a target validation sample.
    #[must_use]
    pub fn new(
        sources: Vec<PopulationValidation>,
        target_population: impl Into<String>,
        observed_positive: u64,
        observed_total: u64,
        draws: usize,
    ) -> Self {
        Self {
            sources,
            target_validation: None,
            target_population: target_population.into(),
            observed_positive,
            observed_total,
            draws,
        }
    }

    /// The same request with the target's own validation sample.
    #[must_use]
    pub fn with_target_validation(mut self, validation: PopulationValidation) -> Self {
        self.target_validation = Some(validation);
        self
    }
}

/// Bound pooled-differential transport.
#[derive(Clone, Debug)]
pub struct CompiledPooledDifferentialTransport {
    request: PooledDifferentialTransportRequest,
    lowered: Arc<LoweredProgram>,
}

/// The corrected prevalence, bound to the program that produced it.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PooledDifferentialRun {
    /// The correction, its simulation interval and draws, as the kernel returned them.
    pub correction: PooledDifferentialCorrection,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
}

fn write_counts(writer: &mut CanonicalWriter, validation: &PopulationValidation) {
    writer.text(&validation.population).words(&[
        validation.true_positive,
        validation.false_negative,
        validation.true_negative,
        validation.false_positive,
    ]);
}

impl Study {
    /// Bind a pooled-differential measurement transport into a compiled program.
    ///
    /// The request is checked by the kernel when the program executes; binding it refuses
    /// nothing but an unlowerable program.
    ///
    /// # Errors
    ///
    /// A physical operator that cannot be declared.
    pub fn compile_pooled_differential_transport(
        request: PooledDifferentialTransportRequest,
    ) -> Result<CompiledPooledDifferentialTransport, CausalError> {
        let mut question =
            CanonicalWriter::new("antecedent.pooled_differential_transport.question.v1");
        question.text(&request.target_population).word(request.draws as u64);
        let mut source = CanonicalWriter::new("antecedent.pooled_differential_transport.counts.v1");
        source.word(request.sources.len() as u64);
        for validation in &request.sources {
            write_counts(&mut source, validation);
        }
        match &request.target_validation {
            Some(validation) => {
                source.word(1);
                write_counts(&mut source, validation);
            }
            None => {
                source.word(0);
            }
        }
        source.words(&[request.observed_positive, request.observed_total]);
        let (question, source) = (question.finish(), source.finish());
        let lowered = PointProgram {
            cell: ProgramCell::PooledDifferentialTransport,
            plan_id: "pooled_differential_transport.v1",
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            graph_class: "declared_population_error_structure",
            query_variables: Arc::from([]),
            question,
            source,
            source_role: EvidenceRole::Observations,
            evaluator: "antecedent_estimate::measurement_transport_pooled::\
                        transport_pooled_differential_prevalence",
            supplied: Vec::new(),
            rows: request.sources.len() as u64 + 1,
            columns: 4,
            input_bytes: (request.sources.len() as u64 + 2).saturating_mul(32),
            claim: antecedent_contract::ClaimKind::Point,
        }
        .lower()?;
        Ok(CompiledPooledDifferentialTransport { request, lowered })
    }
}

impl CompiledProgramHandle for CompiledPooledDifferentialTransport {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledPooledDifferentialTransport {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without drawing.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Pool the rates and draw the corrected prevalence from the context's seed.
    ///
    /// # Errors
    ///
    /// The kernel's refusals (fewer than two sources, a source labelled as the target, a
    /// repeated source, an empty validation margin, invalid target counts, fewer than 1000
    /// draws, pooled rates that leave the target unidentified), admission, cancellation, or
    /// the support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<PooledDifferentialRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)?;
        }
        let request = &self.request;
        let correction = transport_pooled_differential_prevalence(
            PooledDifferentialRequest {
                sources: &request.sources,
                target_validation: request.target_validation.as_ref(),
                target_population: &request.target_population,
                observed_positive: request.observed_positive,
                observed_total: request.observed_total,
                draws: request.draws,
            },
            &scoped,
        )?;
        Ok(PooledDifferentialRun { correction, lowered_program_identity: self.lowered.identity() })
    }
}
