//! Experiment planning through the compiler: power, sample size and minimum
//! detectable effect of a declared randomized design.
//!
//! [`compile_planning`] lowers a [`PlanningRequest`] to a program whose layer
//! four retains every planning assumption; [`PlanningPlan::execute`] solves it
//! analytically and, when the request declares a simulation, simulates the
//! whole procedure (data generation, the design's analysis estimator, the
//! planned decision) under keyed streams leased from the admission pool.
//! [`plan_experiment`] does both. `Study::compile_for_planning` plans the next
//! trial of a randomized study's analysis. A plan's `experiment_candidate`,
//! `sampling_candidate` and `future_data` carry its sample size into design
//! ranking and value-of-information evaluation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

pub use super::execute::{PlannedExperiment, PlanningPlan, STANDALONE_PLANNING};
pub use antecedent_design::planning::{
    AnalyticCharacteristics, ArmComparison, FactorialContrast, FamilyCharacteristics, FamilyPower,
    FamilySimulation, InterferenceExposure, Multiplicity, NormalPrior, OutcomeModel,
    PLANNING_NEXT_ATTEMPTS, PlannedDesign, PlanningAssumptions, PlanningCriterion, PlanningGoal,
    PlanningRefusal, PlanningRequest, PlanningResult, SequentialCharacteristics,
    SequentialSimulation, SimulatedCharacteristics, SimulationSpec, StratumSpec, SuccessCriterion,
    TestSidedness,
};

use crate::error::CausalError;

/// Compile a planning request for a future trial that binds no data.
///
/// # Errors
///
/// A reason-coded refusal of the request (naming the next attempt where one
/// exists), [`CausalError::Resource`] when the simulation's replicate store
/// exceeds the admission pool, re-entry inside a lowered dispatch, or
/// cancellation.
pub fn compile_planning(
    request: &PlanningRequest,
    ctx: &ExecutionContext,
) -> Result<PlanningPlan, CausalError> {
    super::execute::compile_planning(request, ctx)
}

/// Compile and execute a planning request.
///
/// # Errors
///
/// The refusals of [`compile_planning`] and of [`PlanningPlan::execute`].
pub fn plan_experiment(
    request: &PlanningRequest,
    ctx: &ExecutionContext,
) -> Result<PlannedExperiment, CausalError> {
    compile_planning(request, ctx)?.execute(ctx)
}
