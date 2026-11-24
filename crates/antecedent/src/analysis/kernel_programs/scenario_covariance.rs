//! The shared-row covariance of declared scenario estimates as a compiled program family.
//!
//! A covariance specification (`scenario_covariance_artifact::CovarianceSpec`) carries the
//! shared row table, the scenario declarations in matrix order and the method (a seeded
//! whole-row bootstrap or exact enumeration of the resampling distribution). Compiling one
//! binds the digest of its canonical bytes into a six-layer program under the
//! `SharedScenarioCovariance` route. Computing passes the support gate of the route's unit and
//! then calls `CovarianceSpec::compute` / `compute_within`, the calls the direct path makes:
//! the bootstrap draws come from the declared seed, so the matrix, the means, the replicate
//! ids and every digest are the direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::ExecutionContext;
use antecedent_estimate::scenario_covariance::ScenarioCovariance;
use antecedent_io::IoError;
use antecedent_io::scenario_covariance_artifact::{
    CovarianceSpec, ScenarioCovarianceConsumeLimits,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization, canonical_cbor, io_refusal};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const SCENARIO_COVARIANCE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SharedScenarioCovariance,
    plan_id: "transport.shared_scenario_covariance",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "shared_row_resampling_covariance",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::JointScenarios,
    graph_class: "cpdag_completions",
    data: DataClassification::Tabular,
    design: "one replicate identity across scenarios over the shared rows",
};

/// A compiled scenario covariance: the specification and the program that binds it.
#[derive(Clone, Debug)]
pub struct CompiledScenarioCovariance {
    program: KernelProgram<CovarianceSpec>,
}

impl Study {
    /// Compile the shared-row covariance of the scenarios `spec` declares under the
    /// `SharedScenarioCovariance` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a specification that does not
    /// encode.
    pub fn compile_scenario_covariance(
        spec: CovarianceSpec,
        ctx: &ExecutionContext,
    ) -> Result<CompiledScenarioCovariance, CausalError> {
        let canonical = canonical_cbor(&spec)?;
        let program = KernelProgram::compile(&SCENARIO_COVARIANCE, spec, &canonical, ctx)?;
        Ok(CompiledScenarioCovariance { program })
    }
}

impl CompiledScenarioCovariance {
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

    /// The specification the program binds.
    #[must_use]
    pub const fn spec(&self) -> &CovarianceSpec {
        self.program.input()
    }

    /// Pass the support gate and compute the joint covariance, as `CovarianceSpec::compute`
    /// does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the estimators and the method.
    pub fn covariance(&self, ctx: &ExecutionContext) -> Result<ScenarioCovariance, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        self.program.input().compute(ctx)
    }

    /// [`Self::covariance`] under the caller's bounds, as `CovarianceSpec::compute_within` does.
    ///
    /// # Errors
    ///
    /// As [`Self::covariance`]; more rows than `limits.max_rows` is `execution_resource_limit`.
    pub fn covariance_within(
        &self,
        ctx: &ExecutionContext,
        limits: &ScenarioCovarianceConsumeLimits,
    ) -> Result<ScenarioCovariance, IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        self.program.input().compute_within(ctx, limits)
    }
}
