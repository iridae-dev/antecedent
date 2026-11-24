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
use antecedent_core::measurement_model::{
    ErrorStructure, MappingSemantics, MeasurementModel, RepresentationVersion,
};
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
    CategoricalTarget, MeasurementAnalysisBinding, MeasurementInput, MeasurementModelRefusal,
};
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::variant;
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

fn write_representation(writer: &mut CanonicalWriter, representation: &RepresentationVersion) {
    writer.text(&representation.coordinate_system_id).word(u64::from(representation.version));
}

fn write_error_structure(writer: &mut CanonicalWriter, structure: ErrorStructure) {
    match structure {
        ErrorStructure::NonDifferential => {
            variant(writer, "non_differential");
        }
        ErrorStructure::Differential(axis) => {
            variant(writer, "differential").text(axis.as_str());
        }
        // The structure enum is non-exhaustive; a variant added later is written under its own
        // tag until it gains its own arm here.
        _ => {
            variant(writer, "unrecognized_variant");
        }
    }
}

/// The analysis binding: every field in declaration order.
fn write_binding(writer: &mut CanonicalWriter, binding: &MeasurementAnalysisBinding) {
    writer
        .text(&binding.construct_id)
        .text(&binding.population_id)
        .text(&binding.regime_id)
        .opt_text(binding.arm_id.as_deref());
    write_representation(writer, &binding.representation);
    match &binding.representation_mapping {
        Some(mapping) => {
            writer.word(1);
            write_representation(writer, &mapping.from);
            write_representation(writer, &mapping.to);
            writer.text(&mapping.mapping_id);
            match &mapping.semantics {
                MappingSemantics::CoordinateRelabeling => {
                    variant(writer, "coordinate_relabeling");
                }
                MappingSemantics::InterventionPreserving { evidence_id } => {
                    variant(writer, "intervention_preserving").text(evidence_id);
                }
                // Non-exhaustive: a later variant is written under its own tag until it gains
                // its own arm here.
                _ => {
                    variant(writer, "unrecognized_variant");
                }
            }
        }
        None => {
            writer.word(0);
        }
    }
    write_error_structure(writer, binding.assumed_structure);
}

/// The declared population error structure. The invariant parameters are a set (the transport
/// decision reads them by membership); their `BTreeSet` iterates sorted and without repeats.
fn write_population_structure(
    writer: &mut CanonicalWriter,
    structure: Option<&PopulationErrorStructure>,
) {
    match structure {
        None => {
            variant(writer, "undeclared");
        }
        Some(PopulationErrorStructure::Invariant(declaration)) => {
            variant(writer, "invariant")
                .text(&declaration.source_population)
                .text(&declaration.target_population)
                .text(&declaration.declarer)
                .word(declaration.invariant.len() as u64);
            for parameter in &declaration.invariant {
                writer.text(parameter.as_str());
            }
        }
        Some(PopulationErrorStructure::Differential { declarer }) => {
            variant(writer, "differential").text(declarer);
        }
    }
}

/// The provider object by its canonical contract digest (every declared field, including the
/// exact request) and its trust by kind, attestor or the verification receipt's digest.
fn write_provider(
    writer: &mut CanonicalWriter,
    provider: Option<&(ExternalScientificObject, ExternalTrustState)>,
) {
    let Some((object, trust)) = provider else {
        writer.word(0);
        return;
    };
    writer.word(1).text(&object.contract_digest());
    match trust {
        ExternalTrustState::NativeSupported => {
            variant(writer, "native_supported");
        }
        ExternalTrustState::ExternallyAttested { attestor } => {
            variant(writer, "externally_attested").text(attestor);
        }
        ExternalTrustState::ExactRequestVerified(receipt) => {
            variant(writer, "exact_request_verified").text(&receipt.digest());
        }
    }
}

/// The declared inputs of a transport request the question binds besides the decision's
/// receipt and the re-bound model: the binding, the population structure and the provider.
fn write_declaration(writer: &mut CanonicalWriter, request: &MeasurementTransportRequest) {
    write_binding(writer, &request.binding);
    write_population_structure(writer, request.structure.as_ref());
    write_provider(writer, request.provider.as_ref());
}

/// Digest of the outcome-side data: the variant tag, then every field in declaration order,
/// reals by their bits. Every list is ordered (rows, draws, columns, levels).
fn input_digest(input: &MeasurementInput) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.measurement_transport.input.v2");
    match input {
        MeasurementInput::ClassicalLinear {
            outcome,
            regressor,
            covariates,
            covariate_count,
            replicate_counts,
        } => {
            variant(&mut writer, "classical_linear")
                .reals(outcome)
                .reals(regressor)
                .reals(covariates)
                .word(*covariate_count as u64);
            match replicate_counts {
                Some(counts) => writer.word(1).u32s(counts),
                None => writer.word(0),
            };
        }
        MeasurementInput::RepeatedLinear {
            outcome,
            replicate_values,
            replicate_counts,
            covariates,
            covariate_count,
        } => {
            variant(&mut writer, "repeated_linear")
                .reals(outcome)
                .reals(replicate_values)
                .u32s(replicate_counts)
                .reals(covariates)
                .word(*covariate_count as u64);
        }
        MeasurementInput::BinaryPrevalence { events, total } => {
            variant(&mut writer, "binary_prevalence").word(*events).word(*total);
        }
        MeasurementInput::BinaryRiskDifference { exposed_arm_id, unexposed_arm_id, table } => {
            variant(&mut writer, "binary_risk_difference")
                .text(exposed_arm_id)
                .text(unexposed_arm_id)
                .words(&[
                    table.exposed_events,
                    table.exposed_total,
                    table.unexposed_events,
                    table.unexposed_total,
                ]);
        }
        MeasurementInput::Categorical { classes, target } => {
            variant(&mut writer, "categorical").word(classes.len() as u64);
            for class in classes {
                writer.word(class.count).real(class.mean).real(class.variance);
            }
            match target {
                CategoricalTarget::Proportion(level) => {
                    variant(&mut writer, "proportion").text(level);
                }
                CategoricalTarget::OutcomeMean(level) => {
                    variant(&mut writer, "outcome_mean").text(level);
                }
                CategoricalTarget::OutcomeContrast { first, second } => {
                    variant(&mut writer, "outcome_contrast").text(first).text(second);
                }
            }
        }
        MeasurementInput::LatentPosterior {
            outcome,
            draws,
            draw_count,
            covariates,
            covariate_count,
        } => {
            variant(&mut writer, "latent_posterior")
                .reals(outcome)
                .reals(draws)
                .word(*draw_count as u64)
                .reals(covariates)
                .word(*covariate_count as u64);
        }
    }
    writer.finish()
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
        let mut question = CanonicalWriter::new("antecedent.measurement_transport.question.v2");
        question
            .bytes(receipt.identity.as_bytes())
            .bytes(transported.model().identities().model.as_bytes());
        write_declaration(&mut question, &request);
        let question = question.finish();
        let source = input_digest(&request.input);
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

#[cfg(test)]
mod identity_tests {
    use antecedent_core::CanonicalWriter;
    use antecedent_core::measurement_model::{ErrorStructure, RepresentationVersion};
    use antecedent_estimate::measurement_transport::{
        ErrorParameter, InvarianceDeclaration, PopulationErrorStructure,
    };

    use super::{
        MeasurementAnalysisBinding, MeasurementInput, input_digest, write_binding,
        write_population_structure,
    };

    fn binding(arm_id: Option<&str>) -> MeasurementAnalysisBinding {
        MeasurementAnalysisBinding {
            construct_id: "exposure".to_owned(),
            population_id: "target".to_owned(),
            regime_id: "observational".to_owned(),
            arm_id: arm_id.map(str::to_owned),
            representation: RepresentationVersion {
                coordinate_system_id: "sensor".to_owned(),
                version: 1,
            },
            representation_mapping: None,
            assumed_structure: ErrorStructure::NonDifferential,
        }
    }

    fn binding_digest(binding: &MeasurementAnalysisBinding) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("test");
        write_binding(&mut writer, binding);
        writer.finish()
    }

    fn structure_digest(structure: Option<&PopulationErrorStructure>) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("test");
        write_population_structure(&mut writer, structure);
        writer.finish()
    }

    fn linear(outcome: Vec<f64>) -> MeasurementInput {
        MeasurementInput::ClassicalLinear {
            outcome,
            regressor: vec![0.5, 1.5],
            covariates: Vec::new(),
            covariate_count: 0,
            replicate_counts: None,
        }
    }

    #[test]
    fn the_same_declaration_has_the_same_identity() {
        assert_eq!(binding_digest(&binding(Some("a"))), binding_digest(&binding(Some("a"))));
        assert_eq!(input_digest(&linear(vec![1.0, 2.0])), input_digest(&linear(vec![1.0, 2.0])));
    }

    #[test]
    fn a_changed_field_changes_the_identity() {
        assert_ne!(binding_digest(&binding(Some("a"))), binding_digest(&binding(None)));
        let mut differential = binding(Some("a"));
        differential.assumed_structure = ErrorStructure::Differential(
            antecedent_core::measurement_model::DifferentialAxis::Population,
        );
        assert_ne!(binding_digest(&binding(Some("a"))), binding_digest(&differential));
        assert_ne!(input_digest(&linear(vec![1.0, 2.0])), input_digest(&linear(vec![1.0, 3.0])));
        assert_ne!(
            input_digest(&MeasurementInput::BinaryPrevalence { events: 1, total: 2 }),
            input_digest(&MeasurementInput::BinaryPrevalence { events: 2, total: 2 })
        );
    }

    #[test]
    fn reordering_rows_changes_the_identity() {
        assert_ne!(input_digest(&linear(vec![1.0, 2.0])), input_digest(&linear(vec![2.0, 1.0])));
    }

    #[test]
    fn the_invariant_parameters_are_a_set() {
        let declare = |invariant: &[ErrorParameter]| {
            PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
                "source", "target", "analyst", invariant,
            ))
        };
        let forward = declare(&[ErrorParameter::Sensitivity, ErrorParameter::Specificity]);
        let reversed = declare(&[ErrorParameter::Specificity, ErrorParameter::Sensitivity]);
        let repeated = declare(&[
            ErrorParameter::Specificity,
            ErrorParameter::Sensitivity,
            ErrorParameter::Specificity,
        ]);
        assert_eq!(structure_digest(Some(&forward)), structure_digest(Some(&reversed)));
        assert_eq!(structure_digest(Some(&forward)), structure_digest(Some(&repeated)));
        let narrower = declare(&[ErrorParameter::Sensitivity]);
        assert_ne!(structure_digest(Some(&forward)), structure_digest(Some(&narrower)));
        assert_ne!(structure_digest(Some(&forward)), structure_digest(None));
    }

    #[test]
    fn negative_zero_is_not_zero() {
        assert_ne!(input_digest(&linear(vec![0.0, 2.0])), input_digest(&linear(vec![-0.0, 2.0])));
    }
}
