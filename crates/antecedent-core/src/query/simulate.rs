//! Simulation query on a fitted structural causal model: interventional draws,
//! the model-based value of a policy, and unit-level effects after abduction.
//!
//! Each answer is a property of the fitted model, not an identified estimand of
//! the population; only the Monte Carlo error of the draws is quantified. The
//! route executes and stays unlicensed until a coverage record licenses it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, count, encode_list};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::Intervention;

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
}

impl SimulationTarget {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InterventionalSample { .. } => "interventional_sample",
            Self::PolicyValue { .. } => "policy_value",
            Self::UnitEffects { .. } => "unit_effects",
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
            SimulationTarget::InterventionalSample { interventions, .. } => {
                (interventions.first().and_then(Intervention::primary_variable), None)
            }
            SimulationTarget::PolicyValue { interventions, outcome, .. } => {
                (interventions.first().and_then(Intervention::primary_variable), Some(*outcome))
            }
            SimulationTarget::UnitEffects { treatment, outcome, .. } => {
                (Some(*treatment), Some(*outcome))
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
        let invalid = |message: &str| Err(QueryError::InvalidSimulate(message.to_owned()));
        match &self.target {
            SimulationTarget::InterventionalSample { interventions, draws, .. } => {
                for intervention in interventions.iter() {
                    intervention
                        .validate()
                        .map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
                }
                if *draws == 0 {
                    return invalid("an interventional sample needs at least one draw");
                }
                Ok(())
            }
            SimulationTarget::PolicyValue { interventions, draws, .. } => {
                for intervention in interventions.iter() {
                    intervention
                        .validate()
                        .map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
                }
                if *draws < 2 {
                    return invalid(
                        "a policy value needs at least two draws to report its simulation error",
                    );
                }
                Ok(())
            }
            SimulationTarget::UnitEffects { treatment, outcome, active, control } => {
                if treatment == outcome {
                    return Err(QueryError::TreatmentEqualsOutcome { id: *treatment });
                }
                if !active.is_finite() || !control.is_finite() {
                    return invalid("unit effects need finite treatment levels");
                }
                Ok(())
            }
        }
    }
}
