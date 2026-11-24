//! The transported static path-specific counterfactual as a compiled program.
//!
//! [`TransportedCounterfactual::compile`] seals the request exactly as
//! [`TransportedCounterfactual::evaluate`] does (the same
//! `evaluate_transported_path_specific` call on the same canonical input), lowers it to a program
//! of its own (program family `TransportedPathSpecific`, support unit
//! `counterfactual.transported_additive`) keyed by the sealed request's canonical identity, and
//! gates the result on that unit. The answer, artifact and refusals are those of the direct
//! entry; the program adds the contract-facing layers and the support decision.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;

use crate::analysis::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::transported_counterfactual::{
    TransportedCounterfactual, TransportedCounterfactualArtifactError,
    TransportedCounterfactualRequestWire,
};
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;
use crate::planner::{PlanStage, PlanView};

/// Why a compiled transported counterfactual did not return an answer.
#[derive(Debug)]
#[non_exhaustive]
pub enum TransportedCompileError {
    /// The evaluation refused or failed, exactly as the direct entry would: a typed refusal
    /// (with its witness and missing factors) or a bound.
    Evaluation(TransportedCounterfactualArtifactError),
    /// The compiled program refused: a reentrant call, cancellation, a lowering failure or an
    /// unsupported unit in a release build.
    Program(CausalError),
}

impl std::fmt::Display for TransportedCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Evaluation(error) => write!(f, "{error}"),
            Self::Program(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TransportedCompileError {}

impl From<TransportedCompileError> for CausalError {
    fn from(error: TransportedCompileError) -> Self {
        match error {
            TransportedCompileError::Evaluation(error) => Self::from(error),
            TransportedCompileError::Program(error) => error,
        }
    }
}

/// A transported counterfactual run through its compiled program.
#[derive(Clone, Debug)]
pub struct CompiledTransportedCounterfactual {
    counterfactual: TransportedCounterfactual,
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledTransportedCounterfactual {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTransportedCounterfactual {
    /// The sealed answer: contrasts, per-unit contrasts, derivation, identity and artifact.
    #[must_use]
    pub const fn counterfactual(&self) -> &TransportedCounterfactual {
        &self.counterfactual
    }

    /// The sealed answer, taking it out of the run.
    #[must_use]
    pub fn into_counterfactual(self) -> TransportedCounterfactual {
        self.counterfactual
    }

    /// The target contrast `sum_z P_T(z) G(z)`.
    #[must_use]
    pub const fn target_contrast(&self) -> f64 {
        self.counterfactual.target_contrast()
    }

    /// The program the answer ran under.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Every layer of the program, with the support decision of the unit its route runs under.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }
}

impl TransportedCounterfactual {
    /// Evaluate and seal the transported contrast through its compiled program.
    ///
    /// The numbers, artifact and refusals are those of [`Self::evaluate`]; the program is keyed
    /// by the sealed request's canonical identity and gated on the support unit
    /// `counterfactual.transported_additive`.
    ///
    /// # Errors
    ///
    /// [`TransportedCompileError::Evaluation`] for every refusal [`Self::evaluate`] returns;
    /// [`TransportedCompileError::Program`] for a reentrant call, cancellation, a lowering failure
    /// or an unsupported unit in a release build.
    pub fn compile(
        request: &TransportedCounterfactualRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<CompiledTransportedCounterfactual, TransportedCompileError> {
        crate::analysis::lowering::refuse_reentry(ctx).map_err(TransportedCompileError::Program)?;
        if ctx.cancellation.is_cancelled() {
            return Err(TransportedCompileError::Program(CausalError::Cancelled {
                stage: "transported counterfactual",
            }));
        }
        let counterfactual =
            Self::evaluate(request).map_err(TransportedCompileError::Evaluation)?;
        let lowered = crate::analysis::lowering::transported_path_specific::lower(
            counterfactual.request(),
            counterfactual.identity(),
        )
        .map_err(TransportedCompileError::Program)?;
        let compiled =
            CompiledTransportedCounterfactual { counterfactual, lowered: Arc::new(lowered) };
        gate_program(ctx, &compiled)
            .map_err(|refusal| TransportedCompileError::Program(CausalError::from(refusal)))?;
        Ok(compiled)
    }
}
