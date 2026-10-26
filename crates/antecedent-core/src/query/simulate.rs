//! Simulation query on a fitted structural causal model: interventional draws,
//! the model-based value of a policy, and unit-level effects after abduction.
//!
//! Each answer is a property of the fitted model, not an identified estimand of
//! the population; only the Monte Carlo error of the draws is quantified. The
//! route executes and stays unlicensed until a coverage record licenses it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::{InterventionFault, SimulateFault};
use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, count, encode_list};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::{Intervention, InterventionSequence};

use super::error::QueryError;

/// What a simulation query simulates.
#[derive(Clone, Debug, PartialEq)]
pub enum SimulationTarget {
    /// Interventional ancestral draws under the declared interventions.
    InterventionalSample {
        /// Interventions, in declared order.
        interventions: Arc<[Intervention]>,
        /// Rows drawn.
        draws: usize,
        /// Seed of the sampling stream.
        seed: u64,
    },
    /// The value of a policy on one outcome against the unintervened outcome on
    /// the same noise, with the Monte Carlo error of both.
    PolicyValue {
        /// The policy's interventions, in declared order.
        interventions: Arc<[Intervention]>,
        /// Outcome variable.
        outcome: VariableId,
        /// Rows drawn (at least 2).
        draws: usize,
    },
    /// Unit-level effects of two hard levels of `treatment` on `outcome`, by
    /// abduction of each observed unit's noise.
    UnitEffects {
        /// Treatment variable.
        treatment: VariableId,
        /// Outcome variable.
        outcome: VariableId,
        /// Active level.
        active: f64,
        /// Control level.
        control: f64,
    },
    /// Interventional draws together with the exogenous noise that produced
    /// them. The noise is drawn from the execution context's stream.
    NoiseSample {
        /// Interventions, in declared order.
        interventions: Arc<[Intervention]>,
        /// Rows drawn.
        draws: usize,
    },
    /// Retained exogenous noise replayed through the fitted model under the
    /// declared interventions.
    NoiseReplay {
        /// Interventions, in declared order.
        interventions: Arc<[Intervention]>,
        /// Noise in the model's variable order, `n_rows` values per variable
        /// (`noise[variable * n_rows + row]`).
        noise: Arc<[f64]>,
        /// Rows replayed.
        n_rows: usize,
        /// Key of the stream a stochastic policy draws from.
        policy_salt: u64,
    },
    /// Counterfactual values of `outcomes` for every factual row of the
    /// study's data under the declared interventions, after abduction.
    Counterfactual {
        /// Interventions, in declared order.
        interventions: Arc<[Intervention]>,
        /// Outcome variables.
        outcomes: Arc<[VariableId]>,
    },
    /// A rollout of the fitted model over time under a schedule; a lagged
    /// read sees up to `window` earlier steps.
    TemporalRollout {
        /// Scheduled interventions.
        schedule: InterventionSequence,
        /// Rows per step.
        n_rows: usize,
        /// Steps rolled out.
        n_steps: usize,
        /// Earlier steps a lagged read may see.
        window: u32,
        /// Steps before time zero, most recent first, each `n_rows` values per
        /// variable in the model's variable order.
        history: Arc<[f64]>,
        /// Seed of the noise stream.
        seed: u64,
    },
    /// A factual panel replayed under a schedule after abduction at every step.
    TemporalCounterfactual {
        /// Scheduled interventions.
        schedule: InterventionSequence,
        /// Units per step.
        n_rows: usize,
        /// Steps in the panel.
        n_steps: usize,
        /// Earlier steps a lagged read may see.
        window: u32,
        /// Factual panel: step by step, `n_rows` values per variable in the
        /// model's variable order.
        panel: Arc<[f64]>,
        /// Steps before the panel, most recent first, laid out like `panel`.
        history: Arc<[f64]>,
    },
}

fn encode_schedule(writer: &mut CanonicalWriter, schedule: &InterventionSequence) {
    writer.word(count(schedule.steps.len()));
    for step in schedule.steps.iter() {
        step.intervention.encode(writer);
        step.temporal.encode(writer);
    }
}

fn validate_interventions(interventions: &[Intervention]) -> Result<(), QueryError> {
    for intervention in interventions {
        intervention.validate().map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
    }
    Ok(())
}

impl SimulationTarget {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InterventionalSample { .. } => "interventional_sample",
            Self::PolicyValue { .. } => "policy_value",
            Self::UnitEffects { .. } => "unit_effects",
            Self::NoiseSample { .. } => "noise_sample",
            Self::NoiseReplay { .. } => "noise_replay",
            Self::Counterfactual { .. } => "counterfactual",
            Self::TemporalRollout { .. } => "temporal_rollout",
            Self::TemporalCounterfactual { .. } => "temporal_counterfactual",
        }
    }
}

impl CanonicalEncode for SimulationTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::InterventionalSample { interventions, draws, seed } => {
                writer.word(1);
                encode_list(writer, interventions);
                writer.word(count(*draws)).word(*seed);
            }
            Self::PolicyValue { interventions, outcome, draws } => {
                writer.word(2);
                encode_list(writer, interventions);
                outcome.encode(writer);
                writer.word(count(*draws));
            }
            Self::UnitEffects { treatment, outcome, active, control } => {
                writer.word(3);
                treatment.encode(writer);
                outcome.encode(writer);
                writer.real(*active).real(*control);
            }
            Self::NoiseSample { interventions, draws } => {
                writer.word(4);
                encode_list(writer, interventions);
                writer.word(count(*draws));
            }
            Self::NoiseReplay { interventions, noise, n_rows, policy_salt } => {
                writer.word(5);
                encode_list(writer, interventions);
                writer.reals(noise).word(count(*n_rows)).word(*policy_salt);
            }
            Self::Counterfactual { interventions, outcomes } => {
                writer.word(6);
                encode_list(writer, interventions);
                encode_list(writer, outcomes);
            }
            Self::TemporalRollout { schedule, n_rows, n_steps, window, history, seed } => {
                writer.word(7);
                encode_schedule(writer, schedule);
                writer
                    .word(count(*n_rows))
                    .word(count(*n_steps))
                    .word(u64::from(*window))
                    .reals(history)
                    .word(*seed);
            }
            Self::TemporalCounterfactual { schedule, n_rows, n_steps, window, panel, history } => {
                writer.word(8);
                encode_schedule(writer, schedule);
                writer
                    .word(count(*n_rows))
                    .word(count(*n_steps))
                    .word(u64::from(*window))
                    .reals(panel)
                    .reals(history);
            }
        }
    }
}

/// Simulation query on a fitted structural causal model.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulateQuery {
    /// What is simulated.
    pub target: SimulationTarget,
}

impl CanonicalEncode for SimulateQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.target.encode(writer);
    }
}

impl SimulateQuery {
    /// Simulate `target`.
    #[must_use]
    pub const fn new(target: SimulationTarget) -> Self {
        Self { target }
    }

    /// The `(treatment, outcome)` pair the simulation reads: the first
    /// intervened variable (the outcome when there is none) and the outcome,
    /// or every intervened variable's first for a plain sample.
    #[must_use]
    pub fn endpoints(&self) -> (Option<VariableId>, Option<VariableId>) {
        match &self.target {
            SimulationTarget::InterventionalSample { interventions, .. }
            | SimulationTarget::NoiseSample { interventions, .. }
            | SimulationTarget::NoiseReplay { interventions, .. } => {
                (interventions.first().and_then(Intervention::primary_variable), None)
            }
            SimulationTarget::PolicyValue { interventions, outcome, .. } => {
                (interventions.first().and_then(Intervention::primary_variable), Some(*outcome))
            }
            SimulationTarget::UnitEffects { treatment, outcome, .. } => {
                (Some(*treatment), Some(*outcome))
            }
            SimulationTarget::Counterfactual { interventions, outcomes } => (
                interventions.first().and_then(Intervention::primary_variable),
                outcomes.first().copied(),
            ),
            SimulationTarget::TemporalRollout { schedule, .. }
            | SimulationTarget::TemporalCounterfactual { schedule, .. } => {
                (schedule.steps.first().and_then(|step| step.intervention.primary_variable()), None)
            }
        }
    }

    /// Validate the target.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIntervention`] for an invalid intervention, and
    /// [`QueryError::InvalidSimulate`] for no draws, fewer than two policy
    /// draws, non-finite levels, or a treatment that is also the outcome.
    pub fn validate(&self) -> Result<(), QueryError> {
        let invalid = |fault: SimulateFault| Err(QueryError::InvalidSimulate(fault));
        match &self.target {
            SimulationTarget::InterventionalSample { interventions, draws, .. } => {
                for intervention in interventions.iter() {
                    intervention
                        .validate()
                        .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
                }
                if *draws == 0 {
                    return invalid(SimulateFault::SampleDraws);
                }
                Ok(())
            }
            SimulationTarget::PolicyValue { interventions, draws, .. } => {
                for intervention in interventions.iter() {
                    intervention
                        .validate()
                        .map_err(|e| QueryError::InvalidIntervention(InterventionFault::Rejected(e)))?;
                }
                if *draws < 2 {
                    return invalid(SimulateFault::PolicyValueDraws);
                }
                Ok(())
            }
            SimulationTarget::UnitEffects { treatment, outcome, active, control } => {
                if treatment == outcome {
                    return Err(QueryError::TreatmentEqualsOutcome { id: *treatment });
                }
                if !active.is_finite() || !control.is_finite() {
                    return invalid(SimulateFault::UnitEffectLevels);
                }
                Ok(())
            }
            SimulationTarget::NoiseSample { interventions, draws } => {
                validate_interventions(interventions)?;
                if *draws == 0 {
                    return invalid(SimulateFault::NoiseSampleDraws);
                }
                Ok(())
            }
            SimulationTarget::NoiseReplay { interventions, noise, n_rows, .. } => {
                validate_interventions(interventions)?;
                if *n_rows == 0 || noise.is_empty() || noise.len() % *n_rows != 0 {
                    return invalid(SimulateFault::ReplayedNoiseShape);
                }
                if noise.iter().any(|value| !value.is_finite()) {
                    return invalid(SimulateFault::ReplayedNoiseNotFinite);
                }
                Ok(())
            }
            SimulationTarget::Counterfactual { interventions, outcomes } => {
                validate_interventions(interventions)?;
                if outcomes.is_empty() {
                    return Err(QueryError::EmptyCounterfactualOutcomes);
                }
                Ok(())
            }
            SimulationTarget::TemporalRollout { schedule, n_rows, n_steps, history, .. } => {
                let steps: Vec<Intervention> =
                    schedule.steps.iter().map(|step| step.intervention.clone()).collect();
                validate_interventions(&steps)?;
                if *n_rows == 0 || *n_steps == 0 {
                    return invalid(SimulateFault::RolloutShape);
                }
                if history.len() % *n_rows != 0 {
                    return invalid(SimulateFault::RolloutHistoryShape);
                }
                Ok(())
            }
            SimulationTarget::TemporalCounterfactual {
                schedule,
                n_rows,
                n_steps,
                panel,
                history,
                ..
            } => {
                let steps: Vec<Intervention> =
                    schedule.steps.iter().map(|step| step.intervention.clone()).collect();
                validate_interventions(&steps)?;
                if *n_rows == 0 || *n_steps == 0 {
                    return invalid(SimulateFault::PanelReplayShape);
                }
                if panel.len() % (n_rows.saturating_mul(*n_steps)) != 0 || panel.is_empty() {
                    return invalid(SimulateFault::PanelValuesShape);
                }
                if history.len() % *n_rows != 0 {
                    return invalid(SimulateFault::PanelHistoryShape);
                }
                if panel.iter().chain(history.iter()).any(|value| !value.is_finite()) {
                    return invalid(SimulateFault::PanelNotFinite);
                }
                Ok(())
            }
        }
    }
}
