//! Interventions on causal variables (DESIGN.md §8.1).
//!
//! enables hard, shift, stochastic, soft, and sequenced interventions.
//! Estimators that only support hard `Set` continue to reject other variants.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;
use crate::value::Value;

/// Temporal intervention policy over discrete time steps (DESIGN.md §8.1).
///
/// Horizons and offsets are **time steps** relative to the series indexer, not
/// wall-clock durations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TemporalPolicy {
    /// Instantaneous intervention at a single offset.
    Pulse {
        /// Time offset (steps) of the pulse relative to the analysis window origin.
        at: i32,
    },
    /// Intervention held from `from` through `until` inclusive (step offsets).
    Sustained {
        /// First intervened step.
        from: i32,
        /// Last intervened step (inclusive).
        until: i32,
    },
}

impl TemporalPolicy {
    /// One-step pulse at offset `at`.
    #[must_use]
    pub const fn pulse(at: i32) -> Self {
        Self::Pulse { at }
    }

    /// Sustained intervention on `[from, until]`.
    #[must_use]
    pub const fn sustained(from: i32, until: i32) -> Self {
        Self::Sustained { from, until }
    }

    /// Validate policy bounds.
    ///
    /// # Errors
    ///
    /// Returns [`InterventionError`] when the sustained window is empty or inverted.
    pub const fn validate(self) -> Result<(), InterventionError> {
        match self {
            Self::Pulse { .. } => Ok(()),
            Self::Sustained { from, until } => {
                if until < from {
                    return Err(InterventionError::InvalidTemporalWindow { from, until });
                }
                Ok(())
            }
        }
    }
}

/// Opaque mechanism replacement used by soft interventions (DESIGN.md §8.1).
///
/// The model layer resolves `family_id` against its registry; `parameters` are a
/// packed coefficient / noise vector interpreted by that family.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismOverride {
    /// Registry family identifier (e.g. `"linear_gaussian"`, `"constant"`).
    pub family_id: Arc<str>,
    /// Packed parameters for the override family.
    pub parameters: Arc<[f64]>,
}

impl MechanismOverride {
    /// Named family with packed parameters.
    #[must_use]
    pub fn named(family_id: impl Into<Arc<str>>, parameters: impl Into<Arc<[f64]>>) -> Self {
        Self { family_id: family_id.into(), parameters: parameters.into() }
    }

    /// Constant structural assignment (soft form of a hard set).
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self::named("constant", Arc::<[f64]>::from(vec![value]))
    }

    /// Additive shift applied to the structural assignment.
    #[must_use]
    pub fn additive_shift(delta: f64) -> Self {
        Self::named("additive_shift", Arc::<[f64]>::from(vec![delta]))
    }
}

impl Eq for MechanismOverride {}

impl core::hash::Hash for MechanismOverride {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.family_id.hash(state);
        for p in self.parameters.iter() {
            p.to_bits().hash(state);
        }
    }
}

/// Stochastic assignment policy for an intervened variable (DESIGN.md §8.1).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StochasticPolicy {
    /// Bernoulli draw with success probability `p` in `[0, 1]`.
    Bernoulli {
        /// Success probability.
        p: f64,
    },
    /// Independent Gaussian draws.
    Gaussian {
        /// Mean.
        mean: f64,
        /// Variance (must be positive).
        variance: f64,
    },
    /// Categorical over `probs` (non-negative; normalized at sample time).
    Categorical {
        /// Category probabilities (length = support size).
        probs: Arc<[f64]>,
    },
}

impl StochasticPolicy {
    /// Bernoulli policy.
    #[must_use]
    pub const fn bernoulli(p: f64) -> Self {
        Self::Bernoulli { p }
    }

    /// Gaussian policy.
    #[must_use]
    pub const fn gaussian(mean: f64, variance: f64) -> Self {
        Self::Gaussian { mean, variance }
    }

    /// Categorical policy.
    #[must_use]
    pub fn categorical(probs: impl Into<Arc<[f64]>>) -> Self {
        Self::Categorical { probs: probs.into() }
    }

    /// Validate policy parameters.
    ///
    /// # Errors
    ///
    /// Invalid probability, non-positive variance, or empty categorical support.
    pub fn validate(&self) -> Result<(), InterventionError> {
        match self {
            Self::Bernoulli { p } => {
                if !(0.0..=1.0).contains(p) || !p.is_finite() {
                    return Err(InterventionError::InvalidStochasticPolicy {
                        message: "Bernoulli p must be finite and in [0, 1]",
                    });
                }
            }
            Self::Gaussian { variance, .. } => {
                if !(variance.is_finite() && *variance > 0.0) {
                    return Err(InterventionError::InvalidStochasticPolicy {
                        message: "Gaussian variance must be finite and > 0",
                    });
                }
            }
            Self::Categorical { probs } => {
                if probs.is_empty() {
                    return Err(InterventionError::InvalidStochasticPolicy {
                        message: "categorical probs must be non-empty",
                    });
                }
                if probs.iter().any(|p| !(p.is_finite() && *p >= 0.0)) {
                    return Err(InterventionError::InvalidStochasticPolicy {
                        message: "categorical probs must be finite and >= 0",
                    });
                }
                let sum: f64 = probs.iter().sum();
                if sum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    return Err(InterventionError::InvalidStochasticPolicy {
                        message: "categorical probs must sum to a positive value",
                    });
                }
            }
        }
        Ok(())
    }
}

impl Eq for StochasticPolicy {}

impl core::hash::Hash for StochasticPolicy {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Bernoulli { p } => p.to_bits().hash(state),
            Self::Gaussian { mean, variance } => {
                mean.to_bits().hash(state);
                variance.to_bits().hash(state);
            }
            Self::Categorical { probs } => {
                for p in probs.iter() {
                    p.to_bits().hash(state);
                }
            }
        }
    }
}

/// One step in an intervention sequence with temporal policy.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SequencedIntervention {
    /// Intervention applied at this step.
    pub intervention: Intervention,
    /// When the intervention is active.
    pub temporal: TemporalPolicy,
}

impl SequencedIntervention {
    /// Construct a sequenced step.
    #[must_use]
    pub fn new(intervention: Intervention, temporal: TemporalPolicy) -> Self {
        Self { intervention, temporal }
    }
}

/// Ordered list of temporally scoped interventions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InterventionSequence {
    /// Steps in application order.
    pub steps: Arc<[SequencedIntervention]>,
}

impl InterventionSequence {
    /// Construct from steps.
    #[must_use]
    pub fn new(steps: impl Into<Arc<[SequencedIntervention]>>) -> Self {
        Self { steps: steps.into() }
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }
}

/// An intervention applied to one or more variables (DESIGN.md §8.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Intervention {
    /// Hard assignment `do(variable := value)`.
    Set {
        /// Target variable.
        variable: VariableId,
        /// Assigned value.
        value: Value,
    },
    /// Additive shift `do(variable := variable + delta)` on the factual scale.
    Shift {
        /// Target variable.
        variable: VariableId,
        /// Delta added to the structural assignment.
        delta: Value,
    },
    /// Stochastic assignment from a policy.
    Stochastic {
        /// Target variable.
        variable: VariableId,
        /// Sampling policy.
        policy: StochasticPolicy,
    },
    /// Soft intervention replacing the structural mechanism.
    Soft {
        /// Target variable.
        variable: VariableId,
        /// Replacement mechanism description.
        mechanism: MechanismOverride,
    },
    /// Ordered temporal sequence of interventions.
    Sequence(InterventionSequence),
}

impl Intervention {
    /// Hard set intervention.
    #[must_use]
    pub const fn set(variable: VariableId, value: Value) -> Self {
        Self::Set { variable, value }
    }

    /// Additive shift intervention.
    #[must_use]
    pub const fn shift(variable: VariableId, delta: Value) -> Self {
        Self::Shift { variable, delta }
    }

    /// Stochastic intervention.
    #[must_use]
    pub const fn stochastic(variable: VariableId, policy: StochasticPolicy) -> Self {
        Self::Stochastic { variable, policy }
    }

    /// Soft mechanism override.
    #[must_use]
    pub fn soft(variable: VariableId, mechanism: MechanismOverride) -> Self {
        Self::Soft { variable, mechanism }
    }

    /// Temporal sequence.
    #[must_use]
    pub fn sequence(seq: InterventionSequence) -> Self {
        Self::Sequence(seq)
    }

    /// Variable targeted by this intervention, when unique (not a multi-target sequence).
    #[must_use]
    pub fn primary_variable(&self) -> Option<VariableId> {
        match self {
            Self::Set { variable, .. }
            | Self::Shift { variable, .. }
            | Self::Stochastic { variable, .. }
            | Self::Soft { variable, .. } => Some(*variable),
            Self::Sequence(seq) => {
                if seq.steps.is_empty() {
                    return None;
                }
                let first = seq.steps[0].intervention.primary_variable()?;
                if seq.steps.iter().all(|s| s.intervention.primary_variable() == Some(first)) {
                    Some(first)
                } else {
                    None
                }
            }
        }
    }

    /// Validate nested policies and sequences.
    ///
    /// # Errors
    ///
    /// Invalid stochastic policy or empty sequence.
    pub fn validate(&self) -> Result<(), InterventionError> {
        match self {
            Self::Set { .. } | Self::Shift { .. } | Self::Soft { .. } => Ok(()),
            Self::Stochastic { policy, .. } => policy.validate(),
            Self::Sequence(seq) => {
                if seq.is_empty() {
                    return Err(InterventionError::EmptySequence);
                }
                for step in seq.steps.iter() {
                    step.temporal.validate()?;
                    step.intervention.validate()?;
                }
                Ok(())
            }
        }
    }

    /// Whether this is a hard `Set` .
    #[must_use]
    pub const fn is_hard_set(&self) -> bool {
        matches!(self, Self::Set { .. })
    }
}

/// Errors from intervention construction or validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterventionError {
    /// Stochastic policy parameters are invalid.
    InvalidStochasticPolicy {
        /// Context.
        message: &'static str,
    },
    /// Sequence has no steps.
    EmptySequence,
    /// Sustained window has `until < from`.
    InvalidTemporalWindow {
        /// Window start.
        from: i32,
        /// Window end.
        until: i32,
    },
}

impl core::fmt::Display for InterventionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidStochasticPolicy { message } => {
                write!(f, "invalid stochastic policy: {message}")
            }
            Self::EmptySequence => write!(f, "intervention sequence is empty"),
            Self::InvalidTemporalWindow { from, until } => {
                write!(f, "invalid temporal window [{from}, {until}]")
            }
        }
    }
}

impl std::error::Error for InterventionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_set_primary_variable() {
        let v = VariableId::from_raw(3);
        let i = Intervention::set(v, Value::f64(1.0));
        assert_eq!(i.primary_variable(), Some(v));
        assert!(i.is_hard_set());
        i.validate().unwrap();
    }

    #[test]
    fn stochastic_bernoulli_validates() {
        let v = VariableId::from_raw(0);
        let ok = Intervention::stochastic(v, StochasticPolicy::bernoulli(0.4));
        ok.validate().unwrap();
        let bad = Intervention::stochastic(v, StochasticPolicy::bernoulli(1.5));
        assert!(bad.validate().is_err());
    }

    #[test]
    fn sequence_rejects_empty() {
        let seq = Intervention::sequence(InterventionSequence::new(Vec::new()));
        assert!(matches!(seq.validate(), Err(InterventionError::EmptySequence)));
    }

    #[test]
    fn sequence_uniform_primary() {
        let v = VariableId::from_raw(1);
        let seq = Intervention::sequence(InterventionSequence::new(vec![
            SequencedIntervention::new(
                Intervention::set(v, Value::f64(1.0)),
                TemporalPolicy::pulse(0),
            ),
            SequencedIntervention::new(
                Intervention::shift(v, Value::f64(0.1)),
                TemporalPolicy::sustained(1, 3),
            ),
        ]));
        seq.validate().unwrap();
        assert_eq!(seq.primary_variable(), Some(v));
    }

    #[test]
    fn soft_override_helpers() {
        let m = MechanismOverride::constant(2.0);
        assert_eq!(&*m.family_id, "constant");
        assert_eq!(m.parameters.as_ref(), &[2.0]);
    }
}
