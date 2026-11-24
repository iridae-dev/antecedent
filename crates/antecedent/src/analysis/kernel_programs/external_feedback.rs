//! A feedback system with registered external mechanisms as a compiled program family.
//!
//! An external cyclic request (`external_cyclic_artifact::ExternalCyclicRequest`) carries the
//! specified system (built-in equations and the recorded identities of the external
//! mechanisms), the optional joint fit, the hard-clamp worlds, the aligned disturbances, the
//! starting state, the solve policy and the optional factual state. Compiling one binds the
//! digest of its canonical bytes into a six-layer program under the
//! `ExternalFeedbackEquilibrium` route; the implementations are never part of it. Evaluating
//! passes the support gate of the route's unit and then calls `ExternalCyclicRequest::execute`
//! with the caller's trusted resolver, the call the direct path makes: the resolver is verified
//! against the recorded identities before anything runs, and the fit, the equilibria and the
//! counterfactuals are the direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::ExecutionContext;
use antecedent_io::external_cyclic_artifact::{
    ExternalCyclicArtifactError, ExternalCyclicReport, ExternalCyclicRequest,
};
use antecedent_model::external_mechanism::TrustedMechanismResolver;

use super::{KernelProgram, KernelProgramShape, KernelRealization, canonical_cbor};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const EXTERNAL_FEEDBACK: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExternalFeedbackEquilibrium,
    plan_id: "cyclic.external_mechanisms",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "external_mechanism_equilibrium_solve",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::Observations,
    graph_class: "cyclic",
    data: DataClassification::Tabular,
    design: "hard-clamp equilibrium worlds over aligned disturbance realizations",
};

/// A compiled external feedback request: the request and the program that binds it.
#[derive(Clone, Debug)]
pub struct CompiledExternalFeedback {
    program: KernelProgram<ExternalCyclicRequest>,
}

impl Study {
    /// Compile an external feedback request under the `ExternalFeedbackEquilibrium` route.
    /// Compiling executes nothing and resolves no mechanism.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a request that does not
    /// encode.
    pub fn compile_external_feedback(
        request: ExternalCyclicRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledExternalFeedback, CausalError> {
        let canonical = canonical_cbor(&request)?;
        let program = KernelProgram::compile(&EXTERNAL_FEEDBACK, request, &canonical, ctx)?;
        Ok(CompiledExternalFeedback { program })
    }
}

impl CompiledExternalFeedback {
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
    pub const fn request(&self) -> &ExternalCyclicRequest {
        self.program.input()
    }

    /// Pass the support gate, verify the external mechanisms `resolver` supplies, run the
    /// retained fit and evaluate the worlds, as `ExternalCyclicRequest::execute` does.
    ///
    /// # Errors
    ///
    /// The support refusal (reason `not_supported`) in a release build when the unit is not
    /// supported, and the refusals of the request: no resolver, an unavailable or changed
    /// implementation, a size limit, admission, cancellation or a failed solve.
    pub fn evaluate(
        &self,
        resolver: Option<&dyn TrustedMechanismResolver>,
        ctx: &ExecutionContext,
    ) -> Result<ExternalCyclicReport, ExternalCyclicArtifactError> {
        self.program
            .gate(ctx)
            .map_err(|refusal| ExternalCyclicArtifactError::Invalid(refusal.to_string()))?;
        self.program.input().execute(resolver, ctx)
    }
}
