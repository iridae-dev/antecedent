//! Compiled history operator declarations and execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    ActionTiming, AdmissionError, CensoringAssignmentWeight, ExecutionContext, HistoryConstruction,
    HistoryExecutionError, HistoryOperatorError, InitialState, MAX_ROLLOUT_WORK, PolicyEvaluation,
    PolicyRule, ROLLOUT_POLL_STEPS, ROLLOUT_WORLDS, RolloutBudget, SequentialRegression,
    TickSeries, TrajectoryExecution, TrajectoryReduction, Transition, admit_rollout_work,
    horizon_steps, rollout_bytes, rollout_work,
};

/// The kinds of operator a temporal program lowers to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HistoryOperatorKind {
    /// [`HistoryConstruction`].
    HistoryConstruction,
    /// [`SequentialRegression`].
    SequentialRegression,
    /// [`CensoringAssignmentWeight`].
    Weights,
    /// [`ActionTiming`].
    ActionTiming,
    /// [`PolicyEvaluation`].
    PolicyEvaluation,
    /// [`Transition`].
    Transition,
    /// [`TrajectoryReduction`].
    TrajectoryReduction,
}

impl HistoryOperatorKind {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HistoryConstruction => "history_construction",
            Self::SequentialRegression => "sequential_regression",
            Self::Weights => "censoring_assignment_weight",
            Self::ActionTiming => "action_timing",
            Self::PolicyEvaluation => "policy_evaluation",
            Self::Transition => "transition",
            Self::TrajectoryReduction => "trajectory_reduction",
        }
    }

    /// The typed operator edge this kind lowers to.
    ///
    /// History construction conditions on the past; the regression and the weights are
    /// nuisance fits; timing and policy intervene; the transition is the temporal scan; the
    /// reduction reuses the fitted mechanism and reads no new observation.
    ///
    /// # Errors
    ///
    /// [`antecedent_core::OperatorError`] when the declaration is illegal. The seven
    /// declarations are fixed.
    pub fn operator(self) -> Result<antecedent_core::Operator, antecedent_core::OperatorError> {
        use antecedent_core::{
            EvidenceClaims, FrozenFitReuse, Operator, OperatorDeclaration, OperatorEffect,
            OperatorGraph, OperatorPorts, OperatorRole,
        };
        let (effect, graph, inputs, outputs, claims): (
            OperatorEffect,
            OperatorGraph,
            &[OperatorRole],
            &[OperatorRole],
            EvidenceClaims,
        ) = match self {
            Self::HistoryConstruction => (
                OperatorEffect::Condition,
                OperatorGraph::Execution,
                &[OperatorRole::Outcome],
                &[OperatorRole::Conditioning],
                EvidenceClaims::none(),
            ),
            Self::SequentialRegression => (
                OperatorEffect::Fit,
                OperatorGraph::Execution,
                &[OperatorRole::Conditioning, OperatorRole::Outcome],
                &[OperatorRole::FitHandle],
                EvidenceClaims::nuisance_fit(),
            ),
            Self::Weights => (
                OperatorEffect::Fit,
                OperatorGraph::Execution,
                &[OperatorRole::Population, OperatorRole::Outcome],
                &[OperatorRole::FitHandle],
                EvidenceClaims::nuisance_fit(),
            ),
            Self::ActionTiming => (
                OperatorEffect::Intervene,
                OperatorGraph::Causal,
                &[OperatorRole::Population],
                &[OperatorRole::WorldOrSchedule],
                EvidenceClaims::none(),
            ),
            Self::PolicyEvaluation => (
                OperatorEffect::Intervene,
                OperatorGraph::Causal,
                &[OperatorRole::WorldOrSchedule, OperatorRole::Conditioning],
                &[OperatorRole::WorldOrSchedule],
                EvidenceClaims::none(),
            ),
            Self::Transition => (
                OperatorEffect::Scan,
                OperatorGraph::Causal,
                &[OperatorRole::FitHandle, OperatorRole::WorldOrSchedule],
                &[OperatorRole::WorldOrSchedule],
                EvidenceClaims::none(),
            ),
            Self::TrajectoryReduction => (
                OperatorEffect::ReuseFrozenFit,
                OperatorGraph::Execution,
                &[OperatorRole::FitHandle, OperatorRole::WorldOrSchedule],
                &[OperatorRole::FitHandle],
                EvidenceClaims::none(),
            ),
        };
        let mut declaration = OperatorDeclaration::new(
            effect,
            graph,
            OperatorPorts::new(inputs.iter().copied(), outputs.iter().copied()),
        )
        .with_evidence_claims(claims);
        if effect == OperatorEffect::ReuseFrozenFit {
            declaration = declaration.with_frozen_fit_reuse(FrozenFitReuse::Allowed);
        }
        Operator::new(declaration)
    }
}

/// The compiled temporal program: every operator over one cadence, horizon and initial law.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryProgram {
    pub(super) history: HistoryConstruction,
    timing: ActionTiming,
    policy: PolicyEvaluation,
    transition: Transition,
    pub(super) regression: SequentialRegression,
    pub(super) weights: Option<CensoringAssignmentWeight>,
    reduction: TrajectoryReduction,
}

impl HistoryProgram {
    /// Bind the operators into one program.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::FrameMismatch`] when an operator was compiled over a frame
    /// other than the history's, or [`HistoryExecutionError::UnknownFunctional`].
    pub fn try_new(
        history: HistoryConstruction,
        timing: ActionTiming,
        policy: PolicyEvaluation,
        transition: Transition,
        regression: SequentialRegression,
        weights: Option<CensoringAssignmentWeight>,
        reduction: TrajectoryReduction,
    ) -> Result<Self, HistoryExecutionError> {
        let frame = history.frame();
        let checks = [
            ("action_timing", timing.frame()),
            ("policy_evaluation", policy.frame()),
            ("transition", transition.frame()),
            ("sequential_regression", regression.frame()),
            ("trajectory_reduction", reduction.frame()),
        ];
        if let Some((operator, _)) = checks.iter().find(|(_, other)| *other != frame) {
            return Err(HistoryExecutionError::FrameMismatch { operator });
        }
        if weights.as_ref().is_some_and(|w| w.frame() != frame) {
            return Err(HistoryExecutionError::FrameMismatch {
                operator: "censoring_assignment_weight",
            });
        }
        reduction.evaluated()?;
        Ok(Self { history, timing, policy, transition, regression, weights, reduction })
    }

    /// The operator kinds this program runs, in execution order.
    #[must_use]
    pub fn kinds(&self) -> Vec<HistoryOperatorKind> {
        let mut kinds = vec![
            HistoryOperatorKind::HistoryConstruction,
            HistoryOperatorKind::SequentialRegression,
        ];
        if self.weights.is_some() {
            kinds.push(HistoryOperatorKind::Weights);
        }
        kinds.extend([
            HistoryOperatorKind::ActionTiming,
            HistoryOperatorKind::PolicyEvaluation,
            HistoryOperatorKind::Transition,
            HistoryOperatorKind::TrajectoryReduction,
        ]);
        kinds
    }

    /// Name of the policy the program evaluates.
    #[must_use]
    pub fn policy_name(&self) -> &str {
        self.policy.policy()
    }

    /// Seconds-free transition step of the program: one cadence interval, in nanoseconds.
    #[must_use]
    pub const fn step_ns(&self) -> u64 {
        self.transition.step_ns()
    }

    /// Ticks of the horizon.
    fn horizon_steps(&self) -> Result<u32, HistoryOperatorError> {
        horizon_steps(&self.history)
    }

    /// Run the program on `series`.
    ///
    /// One scan builds the windows; the mechanism is fit once under `row_weights` (one per
    /// window, from [`CensoringAssignmentWeight::row_weights`]; they are required exactly
    /// when the program declares weights); the policy and the reference are then rolled
    /// through that same mechanism from the initial state over the horizon, the first
    /// horizon tick being the tick after the series; and the reduction folds both paths.
    /// The action tick is [`ActionTiming::at_ns`] in ticks.
    ///
    /// A [`InitialLaw::contemporaneous_observed`] law starts the roll at the last retained
    /// ticks of the series; a named law needs its `initial` state.
    ///
    /// Under `ctx` the rollouts of both worlds are admitted against the memory pool before
    /// they allocate (one lease for the call), poll cancellation every few thousand ticks,
    /// and charge one shared work budget of ticks times retained state, capped at a finite
    /// bound. The result is not held against the pool after the call returns.
    ///
    /// # Errors
    ///
    /// Any [`HistoryExecutionError`], including
    /// [`HistoryExecutionError::RolloutMemoryRefused`],
    /// [`HistoryExecutionError::RolloutWorkCapExceeded`] and
    /// [`HistoryExecutionError::Cancelled`].
    pub fn execute(
        &self,
        series: &TickSeries<'_>,
        row_weights: Option<&[f64]>,
        policy: PolicyRule,
        reference: PolicyRule,
        initial: Option<&InitialState>,
        ctx: &ExecutionContext,
    ) -> Result<TrajectoryExecution, HistoryExecutionError> {
        if ctx.cancellation.is_cancelled() {
            return Err(HistoryExecutionError::Cancelled);
        }
        let declared = self.weights.is_some();
        let supplied = row_weights.is_some();
        if declared != supplied {
            return Err(HistoryExecutionError::WeightDeclarationMismatch { declared, supplied });
        }
        let policy = policy.admit()?;
        let reference = reference.admit()?;
        let windows = self.history.scan(series)?;
        let fit = self.regression.fit(&windows, row_weights)?;
        let d = windows.depth() as usize;
        let state = match (self.history.frame().initial_law().name(), initial) {
            (_, Some(state)) => {
                if state.outcomes.len() != d || state.actions.len() != d {
                    return Err(HistoryExecutionError::InitialStateShape {
                        outcomes: state.outcomes.len(),
                        actions: state.actions.len(),
                        depth: windows.depth(),
                    });
                }
                state.clone()
            }
            (None, None) => InitialState {
                outcomes: series.outcome[series.len() - d..].to_vec(),
                actions: series.action[series.len() - d..].to_vec(),
            },
            (Some(name), None) => {
                return Err(HistoryExecutionError::InitialStateRequired { law: name.to_owned() });
            }
        };
        let steps = self.horizon_steps().map_err(|_| HistoryExecutionError::HorizonUncountable)?;
        let start = u32::try_from(self.timing.at_ns() / self.history.frame().interval_ns())
            .unwrap_or(u32::MAX);
        let state_len = state.outcomes.len() + state.actions.len() + 1;
        admit_rollout_work(rollout_work(steps, state_len), MAX_ROLLOUT_WORK)?;
        let required_bytes = rollout_bytes(steps, d).unwrap_or(u64::MAX);
        let _lease = ctx.acquire_admission(0, required_bytes).map_err(|error| match error {
            AdmissionError::Cancelled => HistoryExecutionError::Cancelled,
            _ => HistoryExecutionError::RolloutMemoryRefused {
                required_bytes,
                available_bytes: ctx.admission().available().bytes(),
            },
        })?;
        let mut budget = RolloutBudget { cap: MAX_ROLLOUT_WORK, spent: 0 };
        let mut roll = |rule: PolicyRule| -> Result<Vec<f64>, HistoryExecutionError> {
            let mut outcomes = state.outcomes.clone();
            let mut actions = state.actions.clone();
            let mut path = Vec::with_capacity(steps as usize);
            for step in 0..steps {
                if step % ROLLOUT_POLL_STEPS == 0 {
                    if ctx.cancellation.is_cancelled() {
                        return Err(HistoryExecutionError::Cancelled);
                    }
                    let batch = (steps - step).min(ROLLOUT_POLL_STEPS);
                    budget.charge(rollout_work(batch, state_len) / ROLLOUT_WORLDS)?;
                }
                let previous = outcomes[outcomes.len() - 1];
                let action = rule.action(step, start, previous);
                let next = Transition::step(&fit, &outcomes, &actions, action);
                if !next.is_finite() {
                    return Err(HistoryExecutionError::UnstableRollout { step });
                }
                outcomes.push(next);
                actions.push(action);
                path.push(next);
            }
            Ok(path)
        };
        let policy_path = roll(policy)?;
        let reference_path = roll(reference)?;
        let policy_value = self.reduction.reduce(&policy_path)?;
        let reference_value = self.reduction.reduce(&reference_path)?;
        Ok(TrajectoryExecution {
            fit,
            windows: windows.rows(),
            policy: policy_path,
            reference: reference_path,
            policy_value,
            reference_value,
        })
    }
}
