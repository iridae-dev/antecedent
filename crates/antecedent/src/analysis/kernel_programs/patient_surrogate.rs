//! Causal compiler and independent artifact consumer for patient principal-surrogate bounds.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{ExecutionContext, NotSupported, SemanticDigest};
pub use antecedent_io::patient_surrogate::{
    ObservedSurrogatePatient, PatientSurrogateDesign, PatientSurrogateError,
    PatientSurrogateEvidence, PatientSurrogateReading, PatientSurrogateRequest,
    PatientSurrogateResult, PrincipalSurrogateBounds, PrincipalSurrogateError,
    StoredPatientSurrogate, SurrogateThresholdDecision, SurrogateThresholdReading,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::Study;
use crate::planner::{
    PlanView,
    lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading},
};

/// A refusal of the compiled patient-surrogate identification or its replay.
#[derive(Debug, thiserror::Error)]
pub enum PatientSurrogateProgramError {
    /// The input, identification, decision or retained artifact failed its check.
    #[error(transparent)]
    Scientific(#[from] PatientSurrogateError),
    /// The causal compiler refused to lower the input.
    #[error(transparent)]
    Compile(#[from] crate::CausalError),
    /// The route is unavailable under the release policy.
    #[error(transparent)]
    Support(#[from] NotSupported),
}

/// Bound patient-level evidence and partial-identification program.
#[derive(Clone, Debug)]
pub struct CompiledPatientSurrogate(KernelProgram<PatientSurrogateRequest>);

impl Study {
    /// Compile the monotone binary principal-surrogate question.
    ///
    /// # Errors
    /// Invalid identification premises, compilation refusal or cancellation.
    pub fn compile_patient_surrogate(
        request: PatientSurrogateRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPatientSurrogate, PatientSurrogateProgramError> {
        let identity = request.identity(ctx)?;
        let uncertainty = match &request.evidence {
            PatientSurrogateEvidence::PopulationJointLaw { .. } => UncertaintyReading::Envelope,
            PatientSurrogateEvidence::CompletePatientSample { .. } => UncertaintyReading::None,
        };
        let shape = KernelProgramShape {
            cell: ProgramCell::PatientSurrogateIdentification,
            plan_id: "identify.patient_surrogate",
            functional: FunctionalKind::IdentifiedSet,
            evaluator: "monotone_binary_principal_surrogate_compatibility_bounds",
            realization: KernelRealization::PartialIdentification,
            uncertainty,
            role: EvidenceRole::Observations,
            graph_class: "RandomizedTrial",
            data: DataClassification::Tabular,
            design: "joint binary endpoints from a parallel individually randomized patient trial",
        };
        Ok(CompiledPatientSurrogate(KernelProgram::compile(
            &shape,
            request,
            identity.as_bytes(),
            ctx,
        )?))
    }
}

impl CompiledPatientSurrogate {
    /// Inspect the complete compiled design, evidence and uncertainty layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.0.program_layers()
    }

    /// Portable public view of the causal program.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.0.view()
    }

    /// Identify the population interval or evaluate the empirical plug-in bounds.
    ///
    /// # Errors
    /// Release policy, unsupported arm margins, resource refusal or cancellation.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PatientSurrogateResult, PatientSurrogateProgramError> {
        self.0.gate(ctx)?;
        Ok(self.0.input().evaluate(ctx)?)
    }

    /// Execute and export the result with its original patient-level evidence.
    ///
    /// # Errors
    /// The result cannot be identified, or export exceeds its bounds.
    pub fn export(
        &self,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, PatientSurrogateProgramError> {
        let result = self.execute(ctx)?;
        Ok(self.0.input().export(&result, self.lowered().identity(), artifact_id, ctx)?)
    }
}

/// Independently consume a result under the expected original program and result identities.
///
/// # Errors
/// Invalid retained evidence, wrong identities, revised endpoints or failed evaluation.
pub fn consume_patient_surrogate(
    bytes: &[u8],
    program: SemanticDigest,
    result: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<PatientSurrogateResult, PatientSurrogateProgramError> {
    let _lease = ctx.admit_scratch(1, (bytes.len() as u64).saturating_mul(4)).map_err(|_| {
        if ctx.cancellation.is_cancelled() {
            PatientSurrogateError::Cancelled
        } else {
            PatientSurrogateError::Resource
        }
    })?;
    consume_stored(StoredPatientSurrogate::open(bytes, program, result)?, ctx)
}

pub(crate) fn consume_stored(
    stored: StoredPatientSurrogate,
    ctx: &ExecutionContext,
) -> Result<PatientSurrogateResult, PatientSurrogateProgramError> {
    let compiled = Study::compile_patient_surrogate(stored.request.clone(), ctx)?;
    let result = compiled.execute(ctx)?;
    stored.verify(compiled.lowered().identity(), &result)?;
    Ok(result)
}
