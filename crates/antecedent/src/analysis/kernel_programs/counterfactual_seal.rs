// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compiled finite cross-world execution and general temporal counterfactual programs.
//!
//! Each compiles a declared request (its wire) into a kernel program bound to its support unit,
//! gates the program, then seals the answer with the same kernel function the direct
//! `CrossWorldExecution::seal` and `GeneralTemporalCounterfactual::evaluate` call, so the
//! numbers, states and receipts are those of the direct path.

use antecedent_contract::DataClassification;
use antecedent_core::{ExecutionContext, NotSupported};
use antecedent_counterfactual::selective::{
    CrossWorldExecutionError, CrossWorldRefusal, FiniteExecutionResult, FiniteScmExecutor,
};
use antecedent_io::cross_world_execution_artifact::{
    CrossWorldExecutionArtifact, CrossWorldExecutionArtifactError, CrossWorldExecutionRefusalWire,
    CrossWorldRequestWire, request_of,
};
use antecedent_io::temporal_counterfactual_artifact::TemporalCounterfactualRefusalWire;
use antecedent_io::temporal_counterfactual_general_artifact::{
    GeneralTemporalCounterfactualArtifact, GeneralTemporalCounterfactualArtifactError,
    GeneralTemporalRequestWire,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization, canonical_cbor};
use crate::analysis::Study;
use crate::analysis::cross_world_execution::CrossWorldExecution;
use crate::analysis::temporal_counterfactual::GeneralTemporalCounterfactual;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const CROSS_WORLD_EXECUTION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::CrossWorldFiniteExecution,
    plan_id: "cross_world_execution.finite_markovian_scm",
    functional: FunctionalKind::Counterfactual,
    evaluator: "finite_scm_shared_exogenous_execution",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::Observations,
    graph_class: "finite_markovian_scm",
    data: DataClassification::Tabular,
    design: "declared finite structural causal model over shared exogenous states",
};

const GENERAL_TEMPORAL_COUNTERFACTUAL: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::GeneralTemporalCounterfactual,
    plan_id: "counterfactual.temporal_fixed_population",
    functional: FunctionalKind::Counterfactual,
    evaluator: "temporal_shared_abduction_replay",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::Observations,
    graph_class: "temporal_dag",
    data: DataClassification::Temporal,
    design: "one abduced exogenous history per unit replayed under every named action history",
};

/// A compiled finite cross-world execution: the declared request, bound to the
/// `cross_world_execution.finite_markovian_scm` unit.
#[derive(Clone, Debug)]
pub struct CompiledCrossWorldExecution {
    program: KernelProgram<CrossWorldRequestWire>,
}

/// A compiled general temporal counterfactual: the declared request, bound to the
/// `counterfactual.temporal_fixed_population` unit.
#[derive(Clone, Debug)]
pub struct CompiledGeneralTemporalCounterfactual {
    program: KernelProgram<GeneralTemporalRequestWire>,
}

impl Study {
    /// Compile the finite cross-world execution `request` declares under the
    /// `CrossWorldFiniteExecution` route. Compiling executes nothing.
    /// Pass `(&request).into()` for a domain-level
    /// [`FiniteExecutionRequest`](crate::analysis::cross_world_execution::FiniteExecutionRequest).
    /// Call [`CompiledCrossWorldExecution::seal`] to execute the plan.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a request that does not encode.
    pub fn compile_cross_world_execution(
        request: CrossWorldRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<CompiledCrossWorldExecution, CausalError> {
        let canonical = canonical_cbor(&request)?;
        let program = KernelProgram::compile(&CROSS_WORLD_EXECUTION, request, &canonical, ctx)?;
        Ok(CompiledCrossWorldExecution { program })
    }

    /// Compile the general temporal counterfactual `request` declares under the
    /// `GeneralTemporalCounterfactual` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a request that does not encode.
    pub fn compile_general_temporal_counterfactual(
        request: GeneralTemporalRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<CompiledGeneralTemporalCounterfactual, CausalError> {
        let canonical = canonical_cbor(&request)?;
        let program =
            KernelProgram::compile(&GENERAL_TEMPORAL_COUNTERFACTUAL, request, &canonical, ctx)?;
        Ok(CompiledGeneralTemporalCounterfactual { program })
    }
}

impl CompiledCrossWorldExecution {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The request the program binds.
    #[must_use]
    pub const fn request(&self) -> &CrossWorldRequestWire {
        self.program.input()
    }

    /// Gate the program on its unit, then execute the request with a fresh executor and seal
    /// the answer, as `CrossWorldExecution::seal` does.
    ///
    /// # Errors
    ///
    /// A unit the support model does not support (`not_supported`), and the refusals,
    /// cancellation and format bounds of the execution.
    pub fn seal(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CrossWorldExecution, CrossWorldExecutionArtifactError> {
        let publication = self.program.gate(ctx).map_err(|refusal| {
            CrossWorldExecutionArtifactError::Refused(Box::new(CrossWorldExecutionRefusalWire {
                code: refusal.code().to_owned(),
                stage: "gate".to_owned(),
                detail: refusal.detail.to_owned(),
                offending: None,
                remedy: Some(remedy(&refusal)),
                missing: refusal.missing.clone(),
            }))
        })?;
        let request = request_of(self.program.input())?;
        publication.finish(
            CrossWorldExecutionArtifact::seal(&request, ctx)
                .map(CrossWorldExecution::from_artifact),
        )
    }

    /// Gate the program on its unit, then execute the bound request on `executor`.
    ///
    /// `executor` is the caller's. A fresh executor is not built, so abductions it already
    /// retains stay available to this request. The kernel is [`FiniteScmExecutor::execute`].
    ///
    /// # Errors
    ///
    /// A unit the support model does not support (`not_supported`, stage `gate`), a request
    /// the wire cannot rebuild (`invalid_argument`), and every refusal or cancellation of
    /// the executor.
    pub fn execute_retained(
        &self,
        executor: &mut FiniteScmExecutor,
        ctx: &ExecutionContext,
    ) -> Result<FiniteExecutionResult, CrossWorldExecutionError> {
        let publication = self.program.gate(ctx).map_err(|refusal| {
            CrossWorldExecutionError::Refused(Box::new(CrossWorldRefusal {
                refusal: antecedent_contract::ExternalRefusal::new(
                    refusal.code(),
                    "gate",
                    refusal.detail,
                )
                .with_remedy(
                    "run the route in a development build, which records the missing evidence, \
                     or use a supported route",
                ),
                missing: refusal.missing,
            }))
        })?;
        let request = request_of(self.program.input()).map_err(|_| {
            CrossWorldExecutionError::Refused(Box::new(CrossWorldRefusal {
                refusal: antecedent_contract::ExternalRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    "execute",
                    antecedent_core::reason_details::cross_world_execution::INVALID_REQUEST,
                ),
                missing: Vec::new(),
            }))
        })?;
        publication.finish(executor.execute(&request, ctx))
    }
}

impl CompiledGeneralTemporalCounterfactual {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The request the program binds.
    #[must_use]
    pub const fn request(&self) -> &GeneralTemporalRequestWire {
        self.program.input()
    }

    /// Gate the program on its unit, then abduce each unit once, replay every named action
    /// history and seal the answer, as `GeneralTemporalCounterfactual::evaluate` does.
    ///
    /// # Errors
    ///
    /// A unit the support model does not support (`not_supported`), and the refusals and
    /// cancellation of the evaluation.
    pub fn evaluate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GeneralTemporalCounterfactual, GeneralTemporalCounterfactualArtifactError> {
        let publication = self.program.gate(ctx).map_err(|refusal| {
            GeneralTemporalCounterfactualArtifactError::Refused(Box::new(
                TemporalCounterfactualRefusalWire {
                    code: refusal.code().to_owned(),
                    stage: "gate".to_owned(),
                    detail: refusal.detail.to_owned(),
                    offending: None,
                    remedy: Some(remedy(&refusal)),
                    witness: None,
                },
            ))
        })?;
        publication.finish(
            GeneralTemporalCounterfactualArtifact::seal(self.program.input(), ctx)
                .map(GeneralTemporalCounterfactual::from_artifact),
        )
    }
}

fn remedy(refusal: &NotSupported) -> String {
    format!(
        "{refusal}; run the route in a development build, which records the missing evidence, \
         or use a supported route"
    )
}
