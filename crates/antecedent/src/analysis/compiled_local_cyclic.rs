//! A process-local feedback system of declared callback mechanisms, run as a compiled program.
//!
//! [`compile_local_cyclic`] binds each mechanism's declared [`BatchCallback`] (it must provide
//! [`BatchCallback::value_and_derivatives`]) into a `LocalCyclicModel` through
//! `CallbackCyclicMechanism`, lowers the run to a program of its own (program family
//! `LocalCyclicCallback`, support unit `cyclic.general_systems`) under
//! [`ProgramIdentityScope::RuntimeCallbackDeclarationOnly`](crate::planner::lowered::ProgramIdentityScope):
//! the identity commits to every callback's declaration (name, input labels, output, pinned
//! dependencies, implementation identity), never to its code. The run is gated on the unit and
//! executes the damped Newton solve and the additive abduction of `LocalCyclicModel` unchanged,
//! inside the program's lowered dispatch.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{BatchCallback, ExecutionContext};
use antecedent_model::callback_cyclic::CallbackCyclicMechanism;
use antecedent_model::local_cyclic::{
    LocalCyclicInspection, LocalCyclicMechanism, LocalCyclicModel, LocalCyclicPolicy,
};
use antecedent_model::nonlinear_equilibrium::{Clamp, EquilibriumError, EquilibriumResult};

use crate::analysis::program_handle::{CompiledProgramHandle, gate_program};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::LoweredProgram;

/// One mechanism of a local feedback system: its stable id, the endogenous coordinates it reads
/// (in the callback's declared input order) and the declared callback that evaluates it.
#[derive(Clone, Debug)]
pub struct DeclaredCyclicMechanism {
    /// Stable scientific mechanism identity, unique within the system.
    pub id: String,
    /// Endogenous coordinates read, in callback input order.
    pub reads: Vec<usize>,
    /// The callback; it must provide its derivatives.
    pub callback: Arc<dyn BatchCallback>,
}

/// What a local feedback system is asked: a solve under clamps and, with a factual state, the
/// counterfactual solve on the noise abduced from it.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalCyclicRun {
    /// Additive noise of each mechanism, in variable order.
    pub noise: Vec<f64>,
    /// Hard clamps.
    pub clamps: Vec<Clamp>,
    /// Newton starting point, in variable order.
    pub initial: Vec<f64>,
    /// A factual state whose noise is abduced for the counterfactual solve.
    pub factual: Option<Vec<f64>>,
    /// Numerical policy.
    pub policy: LocalCyclicPolicy,
}

/// Why a compiled local feedback run returned no answer.
#[derive(Debug)]
#[non_exhaustive]
pub enum LocalCyclicCompileError {
    /// The model refused or failed, exactly as `LocalCyclicModel` would (a callback failure is
    /// kept in the error chain).
    Model(EquilibriumError),
    /// The compiled program refused: a reentrant call, a lowering failure or an unsupported unit
    /// in a release build.
    Program(CausalError),
}

impl std::fmt::Display for LocalCyclicCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(error) => write!(f, "{error}"),
            Self::Program(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for LocalCyclicCompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Program(error) => Some(error),
        }
    }
}

/// A local feedback run through its compiled program.
#[derive(Clone, Debug)]
pub struct CompiledLocalCyclic {
    inspection: LocalCyclicInspection,
    result: EquilibriumResult,
    counterfactual: Option<EquilibriumResult>,
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledLocalCyclic {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledLocalCyclic {
    /// The post-clamp schedule, reads and stated limits of the system.
    #[must_use]
    pub const fn inspection(&self) -> &LocalCyclicInspection {
        &self.inspection
    }

    /// The solve under the run's noise, clamps and starting point.
    #[must_use]
    pub const fn result(&self) -> &EquilibriumResult {
        &self.result
    }

    /// The counterfactual solve on the noise abduced from the factual state, when one was given.
    #[must_use]
    pub const fn counterfactual(&self) -> Option<&EquilibriumResult> {
        self.counterfactual.as_ref()
    }

    /// The program the run executed.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Every layer of the program, with the support decision of the unit its route runs under.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }
}

/// Solve a local feedback system of declared callback mechanisms through its compiled program.
///
/// The numbers are those of `LocalCyclicModel::solve` (and, with a factual state,
/// `LocalCyclicModel::abduce` followed by `solve` on the shared noise from the factual state) on
/// mechanisms that evaluate the same functions and derivatives.
///
/// # Errors
///
/// [`LocalCyclicCompileError::Model`] for every refusal or failure of the model (invalid ids,
/// reads or inputs, nonconvergence, a singular Jacobian, a resource limit, a callback that raises
/// or provides no derivatives); [`LocalCyclicCompileError::Program`] for a reentrant call, a
/// lowering failure or an unsupported unit in a release build.
pub fn compile_local_cyclic(
    mechanisms: &[DeclaredCyclicMechanism],
    run: &LocalCyclicRun,
    ctx: &ExecutionContext,
) -> Result<CompiledLocalCyclic, LocalCyclicCompileError> {
    crate::analysis::lowering::refuse_reentry(ctx).map_err(LocalCyclicCompileError::Program)?;
    let model = LocalCyclicModel {
        mechanisms: mechanisms
            .iter()
            .map(|m| {
                CallbackCyclicMechanism::new(m.id.clone(), m.reads.clone(), Arc::clone(&m.callback))
                    .map(|mechanism| Box::new(mechanism) as Box<dyn LocalCyclicMechanism>)
            })
            .collect::<Result<_, _>>()
            .map_err(LocalCyclicCompileError::Model)?,
    };
    model.validate().map_err(LocalCyclicCompileError::Model)?;
    let lowered = Arc::new(
        crate::analysis::lowering::local_cyclic_callback::lower(mechanisms, run)
            .map_err(LocalCyclicCompileError::Program)?,
    );
    let program = ProgramOnly(Arc::clone(&lowered));
    let publication = gate_program(ctx, &program)
        .map_err(|refusal| LocalCyclicCompileError::Program(CausalError::from(refusal)))?;
    let scoped = ctx.enter_lowered_dispatch(lowered.identity());
    let solved = (|| {
        let inspection = model.inspect(&run.clamps)?;
        let result = model.solve(&run.noise, &run.clamps, &run.initial, run.policy, &scoped)?;
        let counterfactual = run
            .factual
            .as_ref()
            .map(|factual| {
                let shared = model.abduce(factual, run.policy, &scoped)?;
                model.solve(&shared, &run.clamps, factual, run.policy, &scoped)
            })
            .transpose()?;
        Ok::<_, EquilibriumError>((inspection, result, counterfactual))
    })()
    .map_err(LocalCyclicCompileError::Model)?;
    let (inspection, result, counterfactual) = solved;
    publication.finish(Ok(CompiledLocalCyclic { inspection, result, counterfactual, lowered }))
}

/// The lowered program of a run, before it has executed.
struct ProgramOnly(Arc<LoweredProgram>);

impl CompiledProgramHandle for ProgramOnly {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.0
    }
}
