//! Fitted-model operations as compiled operations on a prepared study.
//!
//! A prepared study holds the DAG it was declared on, the table it retained and (for a model
//! route) the mechanism program it fits. [`PreparedStudy::model_operation`] fits that program on
//! that graph and table and runs one [`ModelOperation`] on the fitted mechanisms (edge strengths,
//! a population do-contrast, a named do-sampler, the falsification evaluation or a nested
//! counterfactual) through a lowered program of its own: the attribution operations under program
//! family `FittedModelAttribution` (unit `gcm.attribute.mechanism_refit`), the others under
//! `FittedModelSimulation` (unit `gcm.simulate.mechanism_refit`). The run is gated on the unit
//! and executes inside the program's lowered dispatch. Each operation calls the kernel the direct
//! `antecedent::gcm` entry calls, with the inputs the operation declares.
//!
//! The study's query, contract and program identity are untouched: the operation is not a query
//! of the study, and its program is keyed by the study's program, the mechanism program and the
//! operation's canonical encoding.
//!
//! The nested counterfactual and the population do-contrast are causal quantities with query
//! forms of their own, [`antecedent_core::SimulationTarget::NestedCounterfactual`] and
//! [`antecedent_core::AttributeTarget::PopulationDoContrast`], which a study poses to get the
//! query-level contract and claim (routes `GcmSimulateDag` and `GcmAttributeDag`). Their
//! operation forms are kept for existing callers: they run under their own program family as
//! before, and execute through the query route's own executor (`run_simulate` and
//! `run_attribute_with_fit` on the fitted model), so the two forms are one implementation and
//! give the same numbers. New code poses the query.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AttributeQuery, AttributeTarget, ExecutionContext, ModelOperation, QueryError, SimulateQuery,
    SimulationTarget,
};

use crate::analysis::builder::DataInput;
use crate::analysis::lowering::model_operation::OperationSource;
use crate::analysis::prepared::PreparedStudy;
use crate::analysis::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::structure_route::model_operations::{
    run_arrow_strengths, run_do_sampler, run_model_evaluation,
};
use crate::analysis::structure_route::{
    AttributionOutcome, ChangeReading, MechanismProgram, SimulationOutcome, run_attribute_with_fit,
    run_simulate,
};
use crate::error::CausalError;
use crate::gcm::MechanismAssignment;
use crate::planner::lowered::LoweredProgram;
use crate::planner::{PlanStage, PlanView};

/// What a fitted-model operation answered.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ModelOperationAnswer {
    /// Model-implied strength of every edge.
    ArrowStrengths(Vec<crate::gcm::ArrowStrength>),
    /// Population do-contrast with its paired Monte Carlo error.
    PopulationDoContrast(crate::gcm::PopulationDoContrast),
    /// A named do-sampler's draws or weighted sample.
    DoSample(crate::gcm::DoSampleResult),
    /// The fitted model's falsification evaluation.
    ModelEvaluation(antecedent_model::ModelEvaluationReport),
    /// The nested counterfactual mean over the study's rows.
    NestedCounterfactual(f64),
}

/// The lowered program of an operation, before it has run.
struct OperationProgram(Arc<LoweredProgram>);

impl CompiledProgramHandle for OperationProgram {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.0
    }
}

/// A fitted-model operation run through its compiled program.
#[derive(Clone, Debug)]
pub struct CompiledModelOperation {
    answer: ModelOperationAnswer,
    mechanisms: Arc<[MechanismAssignment]>,
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledModelOperation {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledModelOperation {
    /// The engine's answer.
    #[must_use]
    pub const fn answer(&self) -> &ModelOperationAnswer {
        &self.answer
    }

    /// The engine's answer, taking it out of the run.
    #[must_use]
    pub fn into_answer(self) -> ModelOperationAnswer {
        self.answer
    }

    /// Per-node mechanism selection record of the fit the operation ran on.
    #[must_use]
    pub fn mechanisms(&self) -> &[MechanismAssignment] {
        &self.mechanisms
    }

    /// The program the operation ran.
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

fn refuse(message: &str) -> CausalError {
    crate::compile_reason!("option_not_applicable", "a fitted-model operation {message}")
}

fn outcome_mismatch() -> CausalError {
    CausalError::Compile {
        message: "a fitted-model operation's query executor returned another outcome".into(),
    }
}

impl PreparedStudy {
    /// Run `operation` on the mechanisms the study's mechanism program fits on its own DAG and
    /// table (the standard program when the study declares none).
    ///
    /// The numbers are those of the direct `antecedent::gcm` entry on the same fitted model:
    /// [`crate::gcm::arrow_strengths`], [`crate::gcm::population_do_contrast`] under a context
    /// seeded with the operation's seed, the do-samplers started at
    /// `CausalRng::from_seed(seed)`, [`crate::gcm::ModelEvaluator::evaluate`] and
    /// [`crate::gcm::nested_counterfactual`].
    ///
    /// [`ModelOperation::NestedCounterfactual`] and [`ModelOperation::PopulationDoContrast`] are
    /// kept for existing callers; their query forms
    /// [`antecedent_core::SimulationTarget::NestedCounterfactual`] and
    /// [`antecedent_core::AttributeTarget::PopulationDoContrast`] give the same numbers with the
    /// query-level contract and claim.
    ///
    /// # Errors
    ///
    /// An invalid operation; a study that is not a supplied or accepted static DAG over a table;
    /// an unsupported unit in a release build; cancellation; a fit or engine refusal.
    pub fn model_operation(
        &self,
        operation: &ModelOperation,
        ctx: &ExecutionContext,
    ) -> Result<CompiledModelOperation, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        operation.validate().map_err(|fault| CausalError::from(QueryError::from(fault)))?;
        let study = self.study();
        if study.graph_posterior.is_some() {
            return Err(refuse("reads one fitted model; a graph posterior holds many structures"));
        }
        let DataInput::Tabular(data) = &study.data else {
            return Err(refuse(
                "fits its mechanisms on a table; this study holds another modality",
            ));
        };
        let Some(graph) = study.graph.as_dag() else {
            return Err(refuse(
                "fits its mechanisms on a static DAG; this study holds another class",
            ));
        };
        let standard = MechanismProgram::standard();
        let program = study.mechanisms.as_deref().unwrap_or(&standard);
        let label = program.label();
        let source = OperationSource {
            study_program: self.plan().lowered().map(LoweredProgram::identity),
            mechanism_program: &label,
            graph,
            data,
        };
        let lowered =
            Arc::new(crate::analysis::lowering::model_operation::lower(&source, operation)?);
        gate_program(ctx, &OperationProgram(Arc::clone(&lowered)))?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "fitted-model operation" });
        }
        let scoped = ctx.enter_lowered_dispatch(lowered.identity());
        let fitted = program.fit(graph, data)?;
        let answer = match operation {
            ModelOperation::ArrowStrengths(_) => {
                ModelOperationAnswer::ArrowStrengths(run_arrow_strengths(&fitted)?)
            }
            ModelOperation::PopulationDoContrast(target) => {
                // The query form's executor, on this operation's fitted model.
                let query = AttributeQuery::new(AttributeTarget::PopulationDoContrast(*target));
                let outcome = run_attribute_with_fit(
                    &fitted,
                    graph,
                    data,
                    &query,
                    None,
                    ChangeReading::default(),
                    None,
                    None,
                    &scoped,
                )?;
                let AttributionOutcome::PopulationDoContrast(result) = outcome else {
                    return Err(outcome_mismatch());
                };
                ModelOperationAnswer::PopulationDoContrast(result)
            }
            ModelOperation::DoSample(target) => {
                ModelOperationAnswer::DoSample(run_do_sampler(&fitted, data, target, &scoped)?)
            }
            ModelOperation::ModelEvaluation(target) => ModelOperationAnswer::ModelEvaluation(
                run_model_evaluation(&fitted, data, target, &scoped)?,
            ),
            ModelOperation::NestedCounterfactual(target) => {
                // The query form's executor, on this operation's fitted model.
                let query =
                    SimulateQuery::new(SimulationTarget::NestedCounterfactual(target.clone()));
                let SimulationOutcome::NestedCounterfactual(mean) =
                    run_simulate(&fitted, data, &query, &scoped)?
                else {
                    return Err(outcome_mismatch());
                };
                ModelOperationAnswer::NestedCounterfactual(mean)
            }
            other => {
                return Err(refuse(&format!("`{}` is not known to this version", other.as_str())));
            }
        };
        Ok(CompiledModelOperation {
            answer,
            mechanisms: Arc::from(fitted.assignments.clone()),
            lowered,
        })
    }
}
