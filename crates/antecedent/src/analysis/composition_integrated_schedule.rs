//! The six-layer compiler and independent portable consumer for the joint schedule decision.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, ExecutionContext, PartialIdentificationEnvelope, PhysicalEffect,
    PhysicalOperator,
};
use antecedent_design::intervention_schedule_decision::{
    CompiledScheduleEvidence, IntegratedScheduleBundle, IntegratedScheduleDecision,
    ScheduleBundleError, ScheduleDecisionDeclaration, ScheduleDecisionError, ScheduleEvidence,
    ScheduleSourceSubject, compile_program_schedule_bundle, consume_program_schedule_bundle,
    inspect_program_schedule_bundle,
};

use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// Sealed program for the source fit, target transport and joint decision.
pub struct CompiledIntegratedScheduleDecision {
    children: [CompiledScheduleEvidence; 3],
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledIntegratedScheduleDecision {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

fn lowered(
    source_rows: usize,
    target_rows: usize,
    schedule_count: usize,
    children: &[CompiledScheduleEvidence; 3],
) -> Result<LoweredProgram, ScheduleBundleError> {
    let [source, target, declaration] = children.each_ref();
    let source_digest = source.snapshot_identity();
    let problem_digest = declaration.snapshot_identity();
    let design = DesignShape::quoted((source_rows + target_rows) as u64, 5, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query: problem_digest,
        },
        plan_id: Arc::from("integrated_schedule_decision.v1"),
        data: DataClassification::Tabular,
        graph_class: "two_declared_longitudinal_structures",
        query_variables: Arc::from([]),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: problem_digest },
    ));
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Envelope(
            PartialIdentificationEnvelope::new("finite_schedule_optimum.hoeffding_joint_region")
                .map_err(|_| ScheduleDecisionError::InvalidDeclaration)?,
        ),
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef {
                role: EvidenceRole::Observations,
                digest: Some(source.program_identity()),
            },
            EvidenceSourceRef {
                role: EvidenceRole::TransportSource,
                digest: Some(target.program_identity()),
            },
            EvidenceSourceRef {
                role: EvidenceRole::JointScenarios,
                digest: Some(declaration.program_identity()),
            },
        ]),
        composition: Composition::Chained { stages: 4 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::IdentifiedSet,
            utility: DecisionProgram::utility_form(None),
            actions: schedule_count,
            constraints: 0,
            chance_threshold: 0.0,
            utility_unit: Some(Arc::from("bounded_endpoint_minus_declared_cost")),
            contract_identity: None,
            information_value: None,
            scenario_source: None,
        }),
        planning: None,
        chain: None,
    };
    let input_bytes = (source_rows as u64)
        .saturating_mul(128)
        .saturating_add((target_rows as u64).saturating_mul(4));
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(input_bytes))
        .build()
        .map_err(|_| ScheduleDecisionError::InvalidDeclaration)?;
    let mut operators = vec![read];
    for effect in [PhysicalEffect::Fit, PhysicalEffect::Transport, PhysicalEffect::Condition] {
        operators.push(
            PhysicalOperator::declare(effect)
                .build()
                .map_err(|_| ScheduleDecisionError::InvalidDeclaration)?,
        );
    }
    Ok(StandaloneCompositionLoweredParts {
        cell: ProgramCell::IntegratedScheduleDecision,
        scientific,
        identification,
        statistical,
        evidence,
        physical: PhysicalLayer { operators: Arc::from(operators), design },
        claim: ClaimDemand {
            uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds),
        },
    }
    .finish())
}

fn schedule_working_bytes(
    source_rows: usize,
    target_rows: usize,
) -> Result<u64, ScheduleBundleError> {
    let source = u64::try_from(source_rows)
        .ok()
        .and_then(|rows| rows.checked_mul(64))
        .ok_or_else(|| ScheduleBundleError::Resource("schedule source quote overflowed".into()))?;
    let target = u64::try_from(target_rows)
        .map_err(|_| ScheduleBundleError::Resource("schedule target quote overflowed".into()))?;
    source.checked_add(target).ok_or_else(|| {
        ScheduleBundleError::Resource("schedule working-set quote overflowed".into())
    })
}

/// Compile the source, transport and joint decision layers without fitting.
///
/// The source and target rows are admitted against `ctx` by their working-set quote.
/// Overlap floors stay on the fits. There is no fixed subject ceiling.
///
/// # Errors
/// The declaration is incomplete, a sample is empty, or the quote does not fit the execution.
pub fn compile_integrated_schedule_decision(
    source: Vec<ScheduleSourceSubject>,
    target: Vec<bool>,
    declaration: ScheduleDecisionDeclaration,
    ctx: &ExecutionContext,
) -> Result<CompiledIntegratedScheduleDecision, ScheduleBundleError> {
    declaration.check()?;
    if source.is_empty() || target.is_empty() {
        return Err(ScheduleDecisionError::MissingJointEvidence.into());
    }
    if ctx.cancellation.is_cancelled() {
        return Err(ScheduleDecisionError::Cancelled.into());
    }
    let bytes = schedule_working_bytes(source.len(), target.len())?;
    let _lease = ctx.acquire_admission(0, bytes).map_err(|error| {
        if ctx.cancellation.is_cancelled() {
            ScheduleBundleError::Procedure(ScheduleDecisionError::Cancelled)
        } else {
            ScheduleBundleError::Resource(error.to_string())
        }
    })?;
    let (source_rows, target_rows, schedule_count) =
        (source.len(), target.len(), declaration.schedules.len());
    let children = [
        CompiledScheduleEvidence::compile(ScheduleEvidence::Source { rows: source })?,
        CompiledScheduleEvidence::compile(ScheduleEvidence::Target { baseline: target })?,
        CompiledScheduleEvidence::compile(ScheduleEvidence::Declaration { problem: declaration })?,
    ];
    let lowered = Arc::new(lowered(source_rows, target_rows, schedule_count, &children)?);
    Ok(CompiledIntegratedScheduleDecision { children, lowered })
}

impl CompiledIntegratedScheduleDecision {
    /// Inspect the compiled port, evidence roles and support decision.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Inspect the lowered layers and their retained identity.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Source, independent target and finite preference declaration child identities.
    #[must_use]
    pub fn child_program_identities(&self) -> [[u8; 32]; 3] {
        self.children.each_ref().map(CompiledScheduleEvidence::program_identity)
    }

    /// Fit all admitted inputs and seal the evidence with the semantic program identity.
    ///
    /// # Errors
    /// A support, overlap, resource or execution stop.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<IntegratedScheduleBundle, ScheduleBundleError> {
        if ctx.cancellation.is_cancelled() {
            return Err(ScheduleDecisionError::Cancelled.into());
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let publication = gate_program(&scoped, self)
            .map_err(|error| ScheduleBundleError::Unsupported(error.message().to_owned()))?;
        publication.finish(compile_program_schedule_bundle(
            &self.children,
            *self.lowered.identity().as_bytes(),
            &scoped,
        ))
    }
}

/// Compile and execute the integrated schedule decision through the public causal compiler.
///
/// # Errors
/// Invalid source evidence, incomplete declarations, or a support or execution stop.
pub fn compile_schedule_decision_bundle(
    source: Vec<ScheduleSourceSubject>,
    target: Vec<bool>,
    declaration: ScheduleDecisionDeclaration,
    ctx: &ExecutionContext,
) -> Result<IntegratedScheduleBundle, ScheduleBundleError> {
    compile_integrated_schedule_decision(source, target, declaration, ctx)?.execute(ctx)
}

/// Independently recompile and replay an integrated schedule decision under a caller's context.
///
/// # Errors
/// A bundle or program identity differs, a child is missing, or replay fails.
pub fn consume_schedule_decision_bundle_with_context(
    bytes: &[u8],
    expected_identity: &str,
    ctx: &ExecutionContext,
) -> Result<IntegratedScheduleDecision, ScheduleBundleError> {
    if ctx.cancellation.is_cancelled() {
        return Err(ScheduleDecisionError::Cancelled.into());
    }
    let recompile_memory = ctx
        .acquire_admission(0, (bytes.len() as u64).saturating_mul(4))
        .map_err(|error| ScheduleBundleError::Resource(error.to_string()))?;
    let inputs = inspect_program_schedule_bundle(bytes, expected_identity)?;
    let compiled =
        compile_integrated_schedule_decision(inputs.source, inputs.target, inputs.problem, ctx)?;
    let identity = *compiled.lowered.identity().as_bytes();
    if identity != inputs.program_identity {
        return Err(ScheduleBundleError::ChangedProgram);
    }
    let scoped = ctx.enter_lowered_dispatch(compiled.lowered.identity());
    let publication = gate_program(&scoped, &compiled)
        .map_err(|error| ScheduleBundleError::Unsupported(error.message().to_owned()))?;
    drop(compiled);
    drop(recompile_memory);
    publication.finish(consume_program_schedule_bundle(bytes, expected_identity, identity, &scoped))
}

/// Independently consume a portable decision under the default execution context.
///
/// # Errors
/// A bundle or program identity differs, a child is missing, or replay fails.
pub fn consume_schedule_decision_bundle(
    bytes: &[u8],
    expected_identity: &str,
) -> Result<IntegratedScheduleDecision, ScheduleBundleError> {
    consume_schedule_decision_bundle_with_context(
        bytes,
        expected_identity,
        &ExecutionContext::production_default(0),
    )
}
