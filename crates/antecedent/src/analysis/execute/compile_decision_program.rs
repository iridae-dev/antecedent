//! A terminal decision and a value-of-information request as programs of their own.
//!
//! [`Study::compile_decision_program`] and [`Study::compile_evsi_program`] compile the same plan
//! the deprecated `compile_for_decision` and `compile_for_evsi` return (layer four of the
//! study's program carries the decision), and compose the decision's own program over it
//! ([`crate::analysis::attached_decision`]): its identity is the plan's program identity and the
//! decision's declaration, and its route is bound to the decision's unit (`decision.contract`
//! for a terminal decision, `design.evsi` for a value-of-information request). The study's plan
//! keeps its own cell.
//!
//! Executing the handle passes the decision's gate first (a release build refuses a unit that
//! is not supported, a development build records it and runs) and then runs the same kernel
//! the direct path runs: [`score_decision`] on what an executed answer published, or
//! [`evaluate_evsi_with_context`] over the decision's prior. Neither changes a number.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::evsi::{EvaluatedEvsi, EvsiError, EvsiRequest, evaluate_evsi_with_context};
use antecedent_design::{DecisionPrior, DecisionProblem};

use super::Study;
use super::compile_decision::{
    DecisionCompileRequest, DecisionDemand, MissingCarrier, PublishedCarriers, ScoredDecision,
    affine_coefficients, score_decision,
};
use crate::analysis::attached_decision::{ParentDecisionProgram, ParentSpec, lower_parent};
use crate::analysis::supplied_identity::Declaration;
use crate::analysis::supplied_terminal::TerminalError;
use crate::error::CausalError;
use crate::planner::lowered::{Composition, DecisionProgram, ProgramCell};
use crate::planner::{PhysicalExecutionPlan, PlanView};

/// Tag of a terminal decision's parent program over the study's plan.
const DECISION_TAG: &str = "antecedent.attached_decision.study_decision.v1";
/// Tag of a value-of-information request's parent program over the study's plan.
const EVSI_TAG: &str = "antecedent.attached_decision.study_evsi.v1";
/// Tag of the declaration either parent covers beside its child.
const DECLARATION_TAG: &str = "antecedent.attached_decision.study_decision.declaration.v1";

/// The parent of `plan`'s decision `program`: one child, the plan's program, with the decision
/// program in the parent's layer four.
fn parent_over(
    plan: &PhysicalExecutionPlan,
    program: DecisionProgram,
    (cell, plan_id, tag, evaluator): (ProgramCell, &'static str, &'static str, &'static str),
) -> Result<ParentDecisionProgram, CausalError> {
    let child = plan.lowered_program_identity().ok_or_else(|| {
        crate::compile_reason!(
            "route_not_lowered",
            "the plan compiled for the decision carries no lowered program"
        )
    })?;
    // The decision program itself is layer four of the parent, which its identity binds; the
    // declaration names what the parent answers and the carrier it reads.
    let decision = Declaration::new(DECLARATION_TAG)
        .field("answers", plan_id)
        .field("carrier", program.carrier().as_str())
        .finish();
    lower_parent(ParentSpec {
        cell,
        plan_id,
        tag,
        children: &[Some(child)],
        decision,
        evaluator,
        composition: Composition::Single,
        program,
        claim: None,
    })
}

/// `problem` is the problem `compiled` was compiled from: the same actions, constraints,
/// chance threshold and utility form.
fn same_problem<A, O>(compiled: &DecisionProgram, problem: &DecisionProblem<A, O>) -> bool {
    compiled.actions == problem.actions.len()
        && compiled.constraints == problem.constraints.len()
        && compiled.chance_threshold.to_bits() == problem.chance_threshold.to_bits()
        && compiled.utility == DecisionProgram::utility_form(affine_coefficients(problem))
}

/// A terminal decision compiled with the study that answers it.
#[derive(Clone, Debug)]
pub struct CompiledDecisionProgram {
    plan: PhysicalExecutionPlan,
    program: DecisionProgram,
    parent: ParentDecisionProgram,
}

impl CompiledDecisionProgram {
    /// The study's plan, with the decision in its lowered program's layer four: the plan the
    /// deprecated `Study::compile_for_decision` returns.
    #[must_use]
    pub const fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// The study's plan, by value.
    #[must_use]
    pub fn into_plan(self) -> PhysicalExecutionPlan {
        self.plan
    }

    /// The plan view of the decision's own program, whose [`PlanView::unit_support`] is the
    /// support decision of `decision.contract`. The study's plan keeps its own cell.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.parent.plan_view()
    }

    /// Identity of the decision's own program: the study plan's program identity and the
    /// decision program in its layer four.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.parent.identity()
    }

    /// Score `demand` from what an executed answer of the plan published, after the decision's
    /// own gate: [`score_decision`], unchanged.
    ///
    /// # Errors
    /// [`TerminalError::Refused`]: a release build's refusal of the decision's unit, or a
    /// demand whose carrier is not the one compiled (`invalid_argument`);
    /// [`TerminalError::Kernel`]: the scorer's [`MissingCarrier`].
    pub fn execute(
        &self,
        demand: DecisionDemand,
        published: PublishedCarriers,
        ctx: &ExecutionContext,
    ) -> Result<ScoredDecision, TerminalError<MissingCarrier>> {
        self.parent.gate(ctx).map_err(CausalError::from)?;
        if demand.carrier() != self.program.carrier {
            return Err(TerminalError::Refused(crate::compile_reason!(
                "invalid_argument",
                "the demand reads the {} carrier and the decision was compiled for {}",
                demand.carrier().as_str(),
                self.program.carrier.as_str()
            )));
        }
        score_decision(demand, published).map_err(TerminalError::Kernel)
    }
}

/// A value-of-information request compiled with the study that carries its state belief.
#[derive(Clone)]
pub struct CompiledEvsiProgram {
    plan: PhysicalExecutionPlan,
    program: DecisionProgram,
    request: EvsiRequest,
    parent: ParentDecisionProgram,
}

impl std::fmt::Debug for CompiledEvsiProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledEvsiProgram")
            .field("plan", &self.plan)
            .field("program", &self.program)
            .field("parent", &self.parent)
            .finish_non_exhaustive()
    }
}

impl CompiledEvsiProgram {
    /// The study's plan, with the decision and the request in its lowered program's layer
    /// four: the plan the deprecated `Study::compile_for_evsi` returns.
    #[must_use]
    pub const fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// The study's plan, by value.
    #[must_use]
    pub fn into_plan(self) -> PhysicalExecutionPlan {
        self.plan
    }

    /// The request compiled.
    #[must_use]
    pub const fn request(&self) -> &EvsiRequest {
        &self.request
    }

    /// The plan view of the decision's own program, whose [`PlanView::unit_support`] is the
    /// support decision of `design.evsi`. The study's plan keeps its own cell.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.parent.plan_view()
    }

    /// Identity of the decision's own program: the study plan's program identity and the
    /// decision program, request included.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.parent.identity()
    }

    /// Evaluate the compiled request for `problem` under `prior`, after the decision's own
    /// gate: [`evaluate_evsi_with_context`], unchanged, on the compiled request.
    ///
    /// # Errors
    /// [`TerminalError::Refused`]: a release build's refusal of the decision's unit, or a
    /// problem other than the one compiled (`invalid_argument`); [`TerminalError::Kernel`]:
    /// the engine's [`EvsiError`].
    pub fn execute<A: Clone + Sync>(
        &self,
        problem: &DecisionProblem<A, f64>,
        prior: &DecisionPrior<f64>,
        ctx: &ExecutionContext,
    ) -> Result<EvaluatedEvsi, TerminalError<EvsiError>> {
        self.parent.gate(ctx).map_err(CausalError::from)?;
        if !same_problem(&self.program, problem) {
            return Err(TerminalError::Refused(crate::compile_reason!(
                "invalid_argument",
                "the decision problem is not the one the request was compiled for"
            )));
        }
        evaluate_evsi_with_context(problem, prior, &self.request, ctx)
            .map_err(TerminalError::Kernel)
    }
}

impl Study {
    /// Compile the study for a terminal decision over its answer, as a program of its own.
    ///
    /// The study's plan is the one the deprecated `Self::compile_for_decision` returns; the
    /// decision's own program is composed over it and bound to `decision.contract`
    /// ([`CompiledDecisionProgram::plan_view`]). Nothing is fitted.
    ///
    /// # Errors
    /// Every refusal of `Self::compile_for_decision`.
    pub fn compile_decision_program<A, O>(
        &self,
        problem: &DecisionProblem<A, O>,
        request: &DecisionCompileRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDecisionProgram, CausalError> {
        let (plan, program) = self.decision_plan(problem, request, ctx)?;
        let parent = parent_over(
            &plan,
            program.clone(),
            (
                ProgramCell::ContractAttachedDecision,
                "attached.study_decision.v1",
                DECISION_TAG,
                "compile_decision::score_decision",
            ),
        )?;
        Ok(CompiledDecisionProgram { plan, program, parent })
    }

    /// Compile the study as the state belief of a value-of-information request, as a program
    /// of its own.
    ///
    /// The study's plan is the one the deprecated `Self::compile_for_evsi` returns; the
    /// request's own program is composed over it and bound to `design.evsi`
    /// ([`CompiledEvsiProgram::plan_view`]). Nothing is fitted or simulated.
    ///
    /// # Errors
    /// Every refusal of `Self::compile_for_evsi`.
    pub fn compile_evsi_program<A, O>(
        &self,
        problem: &DecisionProblem<A, O>,
        request: &EvsiRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledEvsiProgram, CausalError> {
        let (plan, program) = self.evsi_plan(problem, request, ctx)?;
        let parent = parent_over(
            &plan,
            program.clone(),
            (
                ProgramCell::EvsiAttachedDecision,
                "attached.study_evsi.v1",
                EVSI_TAG,
                "evsi::evaluate_evsi_with_context",
            ),
        )?;
        Ok(CompiledEvsiProgram { plan, program, request: request.clone(), parent })
    }
}

#[cfg(test)]
mod site_tests {
    use std::sync::Arc;

    use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder};
    use antecedent_data::TabularData;
    use antecedent_design::AffineUtility;

    use super::*;
    use crate::identity_laws::all_distinct;
    use crate::planner::lowered::CarrierDemand;
    use crate::{BayesianConfig, Dag, InferenceMode, RefuteSuite};

    fn study() -> Study {
        let schema = CausalSchemaBuilder::new()
            .continuous("t")
            .treatment()
            .continuous("y")
            .outcome()
            .build()
            .unwrap();
        let columns: [(&str, &[f64]); 2] = [
            ("t", &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            ("y", &[1.1, 3.0, 0.9, 3.1, 1.0, 2.9, 1.0, 3.0]),
        ];
        let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
        let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        Study::tabular(data)
            .graph(graph)
            .query(query)
            .inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(256).prior_scale(1.0e6),
            ))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    /// `parent_over`: the declaration names what the parent answers and the carrier it reads,
    /// each a framed field beside the child (the plan's program identity) in the parent's
    /// identity; the parent compiled by `compile_decision_program` is the one rebuilt here.
    #[test]
    fn site_compile_decision_program_parent_over_names_the_answer_and_the_carrier() {
        let ctx = ExecutionContext::for_tests(1);
        let utility = AffineUtility::new(vec![0.0, -0.5], vec![0.0, 2.0]).unwrap();
        let problem: DecisionProblem<usize, f64> =
            DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![]);
        let request = DecisionCompileRequest::new(CarrierDemand::ParameterPosterior)
            .with_contract_identity("launch-decision")
            .with_utility_unit("net_margin");
        let compiled = study().compile_decision_program(&problem, &request, &ctx).unwrap();
        let parent = |program: DecisionProgram, answers: &'static str| {
            parent_over(
                &compiled.plan,
                program,
                (
                    ProgramCell::ContractAttachedDecision,
                    answers,
                    DECISION_TAG,
                    "compile_decision::score_decision",
                ),
            )
            .unwrap()
            .identity()
        };
        let base = parent(compiled.program.clone(), "attached.study_decision.v1");
        assert_eq!(base, compiled.program_identity());
        let mut mean = compiled.program.clone();
        mean.carrier = CarrierDemand::Mean;
        all_distinct(&[
            base,
            parent(compiled.program.clone(), "attached.study_evsi.v1"),
            parent(mean, "attached.study_decision.v1"),
        ]);
    }
}
