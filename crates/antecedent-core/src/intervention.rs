//! Interventions on causal variables.
//!
//! enables hard, shift, stochastic, soft, and sequenced interventions.
//! Estimators that only support hard `Set` continue to reject other variants.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::callback::Portability;
use crate::canonical_encode::{CanonicalEncode, count, i32_bits};
use crate::identity::CanonicalWriter;
use crate::ids::{DynamicRuleId, VariableId};
use crate::value::Value;
use crate::value_expr::{ExpressionFault, PolicyReads, ValueExpr};

/// Temporal intervention policy over discrete time steps.
///
/// Horizons and offsets are **time steps** relative to the series indexer, not
/// wall-clock durations.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
    /// Rule-tagged schedule with explicit active time steps.
    ///
    /// `active_at` is the evaluated schedule (sorted unique offsets). `rule` is
    /// an opaque provenance handle for wire / caller registries.
    Dynamic {
        /// Opaque rule handle.
        rule: DynamicRuleId,
        /// Non-empty sorted unique step offsets where the intervention is active.
        active_at: Arc<[i32]>,
    },
}

impl CanonicalEncode for TemporalPolicy {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Pulse { at } => {
                writer.word(1).word(i32_bits(*at));
            }
            Self::Sustained { from, until } => {
                writer.word(2).word(i32_bits(*from)).word(i32_bits(*until));
            }
            Self::Dynamic { rule, active_at } => {
                writer.word(3).word(u64::from(rule.raw())).word(count(active_at.len()));
                for step in active_at.as_ref() {
                    writer.word(i32_bits(*step));
                }
            }
        }
    }
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

    /// Dynamic schedule with explicit active offsets (deduped + sorted).
    ///
    /// # Panics
    ///
    /// Never panics; empty `active_at` fails [`Self::validate`].
    #[must_use]
    pub fn dynamic(rule: DynamicRuleId, active_at: impl Into<Arc<[i32]>>) -> Self {
        let mut steps: Vec<i32> = active_at.into().as_ref().to_vec();
        steps.sort_unstable();
        steps.dedup();
        Self::Dynamic { rule, active_at: Arc::from(steps) }
    }

    /// Whether this is a pulse or a sustained window rather than a dynamic
    /// schedule.
    #[must_use]
    pub const fn is_pulse_or_sustained(&self) -> bool {
        matches!(self, Self::Pulse { .. } | Self::Sustained { .. })
    }

    /// Whether this policy intervenes at exactly one step: a pulse, or a
    /// sustained window whose `from` and `until` coincide.
    #[must_use]
    pub const fn is_single_step(&self) -> bool {
        match self {
            Self::Pulse { .. } => true,
            Self::Sustained { from, until } => *from == *until,
            Self::Dynamic { .. } => false,
        }
    }

    /// Whether step `t` is an active intervention time under this policy.
    #[must_use]
    pub fn is_active_at(&self, t: i32) -> bool {
        match self {
            Self::Pulse { at } => t == *at,
            Self::Sustained { from, until } => t >= *from && t <= *until,
            Self::Dynamic { active_at, .. } => active_at.binary_search(&t).is_ok(),
        }
    }

    /// Active treatment offsets (Pulse: `[at]`; Sustained: `from..=until`; Dynamic: schedule).
    ///
    /// # Errors
    ///
    /// Empty dynamic schedule or inverted sustained window.
    pub fn active_offsets(&self) -> Result<Arc<[i32]>, InterventionError> {
        match self {
            Self::Pulse { at } => Ok(Arc::from([*at])),
            Self::Sustained { from, until } => {
                if *until < *from {
                    return Err(InterventionError::InvalidTemporalWindow {
                        from: *from,
                        until: *until,
                    });
                }
                let steps: Vec<i32> = (*from..=*until).collect();
                Ok(Arc::from(steps))
            }
            Self::Dynamic { active_at, .. } => {
                if active_at.is_empty() {
                    return Err(InterventionError::EmptyDynamicSchedule);
                }
                Ok(Arc::clone(active_at))
            }
        }
    }

    /// Validate policy bounds.
    ///
    /// # Errors
    ///
    /// Empty/inverted sustained window or empty dynamic schedule.
    pub fn validate(&self) -> Result<(), InterventionError> {
        match self {
            Self::Pulse { .. } => Ok(()),
            Self::Sustained { from, until } => {
                if *until < *from {
                    return Err(InterventionError::InvalidTemporalWindow {
                        from: *from,
                        until: *until,
                    });
                }
                Ok(())
            }
            Self::Dynamic { active_at, .. } => {
                if active_at.is_empty() {
                    return Err(InterventionError::EmptyDynamicSchedule);
                }
                Ok(())
            }
        }
    }
}

/// Opaque mechanism replacement used by soft interventions.
///
/// The model layer resolves `family_id` against its registry; `parameters` are a
/// packed coefficient / noise vector interpreted by that family. Equality and
/// hashing are by parameter bits, like [`Value`], so `Eq` and `Hash` agree and a
/// NaN parameter equals itself.
#[derive(Clone, Debug)]
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

    /// Multiply a temporal structural assignment (including innovation) by `factor`.
    /// Supported by the unfolded sequential mean-mechanism engine.
    #[must_use]
    pub fn multiplicative(factor: f64) -> Self {
        Self::named("multiplicative", vec![factor])
    }

    /// Shift a temporal assignment to target a bounded population mean.
    ///
    /// Defines `f_policy = f + clamp(mu + delta, lower, upper) - mu`, where
    /// `mu = E[f]` under preceding interventions. Parent effects and innovations
    /// remain; realized outcomes need not lie within the bounds. Parameters are
    /// `[delta, lower, upper]`. This is not stochastic outcome clipping.
    #[must_use]
    pub fn truncated_shift(delta: f64, lower: f64, upper: f64) -> Self {
        Self::named("truncated_shift", vec![delta, lower, upper])
    }

    /// Additive shift applied to the structural assignment.
    #[must_use]
    pub fn additive_shift(delta: f64) -> Self {
        Self::named("additive_shift", Arc::<[f64]>::from(vec![delta]))
    }
}

impl PartialEq for MechanismOverride {
    fn eq(&self, other: &Self) -> bool {
        self.family_id == other.family_id && bits_eq(&self.parameters, &other.parameters)
    }
}

impl Eq for MechanismOverride {}

/// Elementwise bit equality: the relation `Hash` (over `to_bits`) is consistent with.
fn bits_eq(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

impl core::hash::Hash for MechanismOverride {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.family_id.hash(state);
        for p in self.parameters.iter() {
            p.to_bits().hash(state);
        }
    }
}

/// Stochastic assignment policy for an intervened variable.
///
/// Equality and hashing are by parameter bits, like [`Value`], so `Eq` and `Hash`
/// agree; [`Self::validate`] rejects the NaN parameters that would make the
/// bit relation differ from numeric equality.
#[derive(Clone, Debug)]
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

impl CanonicalEncode for StochasticPolicy {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Bernoulli { p } => {
                writer.word(1).real(*p);
            }
            Self::Gaussian { mean, variance } => {
                writer.word(2).real(*mean).real(*variance);
            }
            Self::Categorical { probs } => {
                writer.word(3);
                writer.reals(probs);
            }
        }
    }
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
            Self::Gaussian { mean, variance } => {
                if !mean.is_finite() {
                    return Err(InterventionError::InvalidStochasticPolicy {
                        message: "Gaussian mean must be finite",
                    });
                }
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

impl PartialEq for StochasticPolicy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bernoulli { p: a }, Self::Bernoulli { p: b }) => a.to_bits() == b.to_bits(),
            (
                Self::Gaussian { mean: ma, variance: va },
                Self::Gaussian { mean: mb, variance: vb },
            ) => ma.to_bits() == mb.to_bits() && va.to_bits() == vb.to_bits(),
            (Self::Categorical { probs: a }, Self::Categorical { probs: b }) => bits_eq(a, b),
            _ => false,
        }
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

/// An intervention applied to one or more variables.
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
    /// Transform of the node's generated value: the mechanism runs as usual (parents and
    /// noise flow through), then the expression replaces the generated value, so the
    /// effect propagates to descendants while the node keeps its own noise. The
    /// expression may read [`ValueExpr::Generated`].
    Transform {
        /// Target variable.
        variable: VariableId,
        /// Expression of the generated value and of other variables' current values.
        expr: ValueExpr,
        /// Declared reads; must equal the variables `expr` references.
        reads: PolicyReads,
    },
    /// Replacement of the node's assignment by an expression of other variables: the
    /// mechanism is discarded, so incoming edges are severed (a computed `do`). The
    /// expression cannot read [`ValueExpr::Generated`].
    Assign {
        /// Target variable.
        variable: VariableId,
        /// Expression of other variables' current values.
        expr: ValueExpr,
        /// Declared reads; must equal the variables `expr` references.
        reads: PolicyReads,
    },
    /// Ordered temporal sequence of interventions.
    Sequence(InterventionSequence),
}

impl CanonicalEncode for Intervention {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Set { variable, value } => {
                writer.word(1);
                variable.encode(writer);
                value.encode(writer);
            }
            Self::Shift { variable, delta } => {
                writer.word(2);
                variable.encode(writer);
                delta.encode(writer);
            }
            Self::Stochastic { variable, policy } => {
                writer.word(3);
                variable.encode(writer);
                policy.encode(writer);
            }
            Self::Soft { variable, mechanism } => {
                writer.word(4);
                variable.encode(writer);
                writer.text(&mechanism.family_id);
                writer.reals(&mechanism.parameters);
            }
            Self::Transform { variable, expr, reads } => {
                writer.word(5);
                variable.encode(writer);
                expr.encode(writer);
                reads.encode(writer);
            }
            Self::Assign { variable, expr, reads } => {
                writer.word(6);
                variable.encode(writer);
                expr.encode(writer);
                reads.encode(writer);
            }
            Self::Sequence(sequence) => {
                writer.word(7).word(count(sequence.steps.len()));
                for step in sequence.steps.as_ref() {
                    step.intervention.encode(writer);
                    step.temporal.encode(writer);
                }
            }
        }
    }
}

impl Intervention {
    /// Transform of the generated value; the declared reads are the expression's own.
    #[must_use]
    pub fn transform(variable: VariableId, expr: ValueExpr) -> Self {
        let reads = expr.reads();
        Self::Transform { variable, expr, reads }
    }

    /// Computed replacement of the assignment; the declared reads are the expression's own.
    #[must_use]
    pub fn assign(variable: VariableId, expr: ValueExpr) -> Self {
        let reads = expr.reads();
        Self::Assign { variable, expr, reads }
    }

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

    /// Every variable this intervention assigns, including Sequence steps.
    #[must_use]
    pub fn target_variables(&self) -> Vec<VariableId> {
        match self {
            Self::Set { variable, .. }
            | Self::Shift { variable, .. }
            | Self::Stochastic { variable, .. }
            | Self::Soft { variable, .. }
            | Self::Transform { variable, .. }
            | Self::Assign { variable, .. } => vec![*variable],
            Self::Sequence(seq) => {
                let mut variables = Vec::new();
                for step in seq.steps.iter() {
                    for variable in step.intervention.target_variables() {
                        if !variables.contains(&variable) {
                            variables.push(variable);
                        }
                    }
                }
                variables
            }
        }
    }

    /// Variable targeted by this intervention, when unique (not a multi-target sequence).
    #[must_use]
    pub fn primary_variable(&self) -> Option<VariableId> {
        match self {
            Self::Set { variable, .. }
            | Self::Shift { variable, .. }
            | Self::Stochastic { variable, .. }
            | Self::Soft { variable, .. }
            | Self::Transform { variable, .. }
            | Self::Assign { variable, .. } => Some(*variable),
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

    /// Whether the intervention can leave the process: closed-form interventions are
    /// portable, a callback program only as a verifiable recovery record, and a program
    /// with a runtime-bound callback not at all. Encoders for artifacts, identities and
    /// the wire call this and refuse on [`Portability::RuntimeBound`].
    #[must_use]
    pub fn portability(&self) -> Portability {
        match self {
            Self::Transform { expr, .. } | Self::Assign { expr, .. } => expr.portability(),
            Self::Sequence(seq) => seq
                .steps
                .iter()
                .fold(Portability::Portable, |acc, step| acc.and(step.intervention.portability())),
            Self::Set { .. } | Self::Shift { .. } | Self::Stochastic { .. } | Self::Soft { .. } => {
                Portability::Portable
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
            // The symbolic marker is a label, not a level; only a genuine
            // non-finite float is a malformed assignment.
            Self::Set { value: level, .. } | Self::Shift { delta: level, .. } => {
                if matches!(level, Value::Float64(v) if !v.is_finite()) {
                    return Err(InterventionError::NonFiniteLevel);
                }
                Ok(())
            }
            Self::Soft { mechanism, .. } => {
                if mechanism.parameters.iter().any(|p| !p.is_finite()) {
                    return Err(InterventionError::NonFiniteLevel);
                }
                Ok(())
            }
            Self::Stochastic { policy, .. } => policy.validate(),
            Self::Transform { expr, reads, .. } => expr
                .validate_program(true, reads)
                .map_err(|fault| InterventionError::Expression { fault }),
            Self::Assign { expr, reads, .. } => expr
                .validate_program(false, reads)
                .map_err(|fault| InterventionError::Expression { fault }),
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
}

/// Errors from intervention construction or validation.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum InterventionError {
    /// Stochastic policy parameters are invalid.
    #[error("invalid stochastic policy: {message}")]
    InvalidStochasticPolicy {
        /// Context.
        message: &'static str,
    },
    /// A hard level, shift, or soft-mechanism parameter is NaN or infinite.
    #[error("intervention level must be finite")]
    NonFiniteLevel,
    /// Sequence has no steps.
    #[error("intervention sequence is empty")]
    EmptySequence,
    /// Sustained window has `until < from`.
    #[error("invalid temporal window [{from}, {until}]")]
    InvalidTemporalWindow {
        /// Window start.
        from: i32,
        /// Window end.
        until: i32,
    },
    /// Dynamic schedule has no active offsets.
    #[error("TemporalPolicy::Dynamic requires a non-empty active_at schedule")]
    EmptyDynamicSchedule,
    /// A transform or assignment expression is not a valid node program.
    #[error("invalid intervention expression: {fault}")]
    Expression {
        /// What is wrong with the expression.
        fault: ExpressionFault,
    },
    /// A sequence already names a policy on each step.
    #[error("a sequence carries its own temporal policy and cannot take another")]
    TimedSequence,
    /// An initial-state law was not named.
    #[error("initial-state law name is empty")]
    EmptyInitialLaw,
}

/// Which model surface is being asked to execute an intervention.
///
/// The operation comes from [`Intervention`] and, when the action is temporal,
/// [`TemporalPolicy`]. A surface that cannot perform that operation refuses it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExecutionSurface {
    /// One decision time. No pulse, sustained window, or initial-state law.
    Static,
    /// Finite-horizon delayed feedback. Pulse and sustained sets only.
    Temporal,
    /// Counterfactual worlds on a static structural model.
    Counterfactual,
    /// Linear equilibrium of a cyclic model. Untimed [`Intervention::Set`] only.
    Cyclic,
}

impl ExecutionSurface {
    /// Every surface, in a stable order.
    pub const ALL: [Self; 4] = [Self::Static, Self::Temporal, Self::Counterfactual, Self::Cyclic];

    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Temporal => "temporal",
            Self::Counterfactual => "counterfactual",
            Self::Cyclic => "cyclic",
        }
    }

    /// Classify `intervention` under `temporal` and accept it on this surface.
    ///
    /// `temporal` of `None` is the untimed intervention. A pulse and a sustained
    /// window stay those policies, including a sustained window of one step.
    /// The returned operation is the intervention's own meaning.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] when this surface cannot perform the operation.
    /// [`InterventionError`] when the intervention or the policy is malformed.
    pub fn admit(
        self,
        intervention: &Intervention,
        temporal: Option<&TemporalPolicy>,
    ) -> Result<InterventionOperation, AdmitError> {
        let operation = intervention.operation(temporal)?;
        self.admit_operation(&operation)?;
        Ok(operation)
    }

    /// Accept `operation` on this surface.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] naming the operation this surface cannot perform.
    pub fn admit_operation(
        self,
        operation: &InterventionOperation,
    ) -> Result<(), CapabilityRefusal> {
        if self.supports(operation) { Ok(()) } else { Err(self.refusal(operation)) }
    }

    /// Whether this surface performs `operation` without changing it.
    #[must_use]
    pub fn supports(self, operation: &InterventionOperation) -> bool {
        match self {
            Self::Static | Self::Counterfactual => matches!(
                operation,
                InterventionOperation::Set { .. }
                    | InterventionOperation::Shift { .. }
                    | InterventionOperation::Stochastic { .. }
                    | InterventionOperation::Soft { .. }
                    | InterventionOperation::Transform { .. }
                    | InterventionOperation::Assign { .. }
                    | InterventionOperation::Sequence(_)
            ),
            Self::Cyclic => matches!(operation, InterventionOperation::Set { .. }),
            Self::Temporal => match operation {
                InterventionOperation::Pulse { action, .. }
                | InterventionOperation::Sustained { action, .. } => {
                    matches!(action.as_ref(), InterventionOperation::Set { .. })
                }
                _ => false,
            },
        }
    }

    fn refusal(self, operation: &InterventionOperation) -> CapabilityRefusal {
        let name = refusal_name(operation);
        let reason = match (self, name) {
            (Self::Temporal, "shift") => {
                "a shift is an additive change on the factual scale and is not executed as a set"
            }
            (Self::Temporal, "soft") => {
                "a soft intervention replaces a mechanism and is not executed as a set"
            }
            (Self::Temporal, "set") => {
                "an untimed set is a hard clamp and a finite-horizon rollout does not hard-clamp"
            }
            (_, "shift") => {
                "a shift is an additive change on the factual scale and is not a hard clamp"
            }
            (_, "soft") => "a soft intervention replaces a mechanism and is not a hard clamp",
            (_, "pulse") => {
                "a pulse is one intervention time and is not a hard clamp and is not a sustained \
                 action"
            }
            (_, "sustained") => {
                "a sustained action holds its level across its window and is not a pulse and is \
                 not a hard clamp"
            }
            (_, "initial_state_change") => {
                "an initial-state change replaces the initial-state law and is not a hard clamp"
            }
            (_, "stochastic") => "a stochastic assignment is not a hard clamp",
            (_, "transform") => "a transform runs the mechanism and is not a hard clamp",
            (_, "assign") => "an assignment discards the mechanism and is not a hard clamp",
            (_, "sequence") => "a sequence is not a hard clamp",
            (_, "dynamic") => {
                "a dynamic schedule is not a pulse or a sustained action and is not a hard clamp"
            }
            (_, "set") => "an untimed set is a hard clamp and this surface does not hard-clamp",
            _ => "this surface does not perform that operation",
        };
        CapabilityRefusal { surface: self, operation: name, reason }
    }
}

impl std::fmt::Display for ExecutionSurface {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Refusal to perform an intervention, or a malformed intervention at admission.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AdmitError {
    /// The intervention or its policy is malformed.
    #[error(transparent)]
    Invalid(#[from] InterventionError),
    /// The surface cannot perform the operation.
    #[error(transparent)]
    Refused(#[from] CapabilityRefusal),
}

/// A surface cannot perform an operation and did not rewrite it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{surface} refuses {operation}: {reason}")]
pub struct CapabilityRefusal {
    /// Surface that refused.
    pub surface: ExecutionSurface,
    /// Operation that was asked for.
    pub operation: &'static str,
    /// Why that operation was not rewritten into one this surface supports.
    pub reason: &'static str,
}

impl CapabilityRefusal {
    /// Operation name.
    #[must_use]
    pub const fn operation(self) -> &'static str {
        self.operation
    }

    /// Why the operation was refused.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        self.reason
    }
}

/// Meaning of an [`Intervention`], timed by [`TemporalPolicy`] when the caller supplied one.
///
/// Equality is the execution identity. A shift is not a set. A soft replacement
/// is not a set. A pulse is not a sustained window. An initial-state change is
/// not a set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InterventionOperation {
    /// Hard assignment `do(variable := value)`.
    Set {
        /// Target variable.
        variable: VariableId,
        /// Assigned value.
        value: Value,
    },
    /// Additive `do(variable := variable + delta)` on the factual scale.
    Shift {
        /// Target variable.
        variable: VariableId,
        /// Delta added to the factual assignment.
        delta: Value,
    },
    /// Stochastic assignment.
    Stochastic {
        /// Target variable.
        variable: VariableId,
        /// Sampling policy.
        policy: StochasticPolicy,
    },
    /// Mechanism replacement.
    Soft {
        /// Target variable.
        variable: VariableId,
        /// Replacement mechanism.
        mechanism: MechanismOverride,
    },
    /// Mechanism runs, then the expression replaces the generated value.
    Transform {
        /// Target variable.
        variable: VariableId,
        /// Expression of the generated value.
        expr: ValueExpr,
        /// Declared reads.
        reads: PolicyReads,
    },
    /// Mechanism discarded; the expression assigns the variable.
    Assign {
        /// Target variable.
        variable: VariableId,
        /// Expression of other variables.
        expr: ValueExpr,
        /// Declared reads.
        reads: PolicyReads,
    },
    /// One intervention time. Not a sustained window and not an untimed set.
    Pulse {
        /// Time offset of the pulse.
        at: i32,
        /// Operation applied at that time.
        action: Box<InterventionOperation>,
    },
    /// Level held on an inclusive window. A window of one step is still sustained.
    Sustained {
        /// First intervened step.
        from: i32,
        /// Last intervened step, inclusive.
        until: i32,
        /// Operation held on the window.
        action: Box<InterventionOperation>,
    },
    /// Rule-tagged schedule. Not a pulse and not a sustained window.
    Dynamic {
        /// Opaque rule handle.
        rule: DynamicRuleId,
        /// Active offsets.
        active_at: Arc<[i32]>,
        /// Operation applied at those offsets.
        action: Box<InterventionOperation>,
    },
    /// Ordered steps. Each step keeps the policy it was declared with.
    Sequence(Arc<[InterventionOperation]>),
    /// Replacement of an initial-state law. Not a clamp of an equilibrium variable.
    InitialState {
        /// Name of the law being replaced.
        law_name: Arc<str>,
        /// Replacement level.
        level: Value,
    },
}

impl InterventionOperation {
    /// Replace the initial-state law `law_name`.
    ///
    /// This is not [`Intervention::Set`].
    ///
    /// # Errors
    ///
    /// [`InterventionError::EmptyInitialLaw`] when `law_name` is empty.
    /// [`InterventionError::NonFiniteLevel`] when `level` is a non-finite float.
    #[must_use = "a refused initial-state change is not an operation"]
    pub fn initial_state(
        law_name: impl Into<Arc<str>>,
        level: Value,
    ) -> Result<Self, InterventionError> {
        let law_name = law_name.into();
        if law_name.is_empty() {
            return Err(InterventionError::EmptyInitialLaw);
        }
        if matches!(level, Value::Float64(value) if !value.is_finite()) {
            return Err(InterventionError::NonFiniteLevel);
        }
        Ok(Self::InitialState { law_name, level })
    }

    /// Stable name of this operation.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Set { .. } => "set",
            Self::Shift { .. } => "shift",
            Self::Stochastic { .. } => "stochastic",
            Self::Soft { .. } => "soft",
            Self::Transform { .. } => "transform",
            Self::Assign { .. } => "assign",
            Self::Pulse { .. } => "pulse",
            Self::Sustained { .. } => "sustained",
            Self::Dynamic { .. } => "dynamic",
            Self::Sequence(_) => "sequence",
            Self::InitialState { .. } => "initial_state_change",
        }
    }
}

impl Intervention {
    /// Operation this intervention performs.
    ///
    /// `temporal` of `None` leaves the intervention untimed. A pulse and a
    /// sustained window wrap the intervention and do not rewrite it: a shift
    /// stays a shift, and a sustained window of one step stays sustained.
    ///
    /// # Errors
    ///
    /// [`InterventionError::TimedSequence`] when `temporal` is supplied for a
    /// sequence. Nested policy and expression errors of [`Self::validate`].
    pub fn operation(
        &self,
        temporal: Option<&TemporalPolicy>,
    ) -> Result<InterventionOperation, InterventionError> {
        if let Some(policy) = temporal {
            policy.validate()?;
            if matches!(self, Self::Sequence(_)) {
                return Err(InterventionError::TimedSequence);
            }
        }
        let untimed = match self {
            Self::Set { variable, value } => {
                InterventionOperation::Set { variable: *variable, value: value.clone() }
            }
            Self::Shift { variable, delta } => {
                InterventionOperation::Shift { variable: *variable, delta: delta.clone() }
            }
            Self::Stochastic { variable, policy } => {
                InterventionOperation::Stochastic { variable: *variable, policy: policy.clone() }
            }
            Self::Soft { variable, mechanism } => {
                InterventionOperation::Soft { variable: *variable, mechanism: mechanism.clone() }
            }
            Self::Transform { variable, expr, reads } => InterventionOperation::Transform {
                variable: *variable,
                expr: expr.clone(),
                reads: reads.clone(),
            },
            Self::Assign { variable, expr, reads } => InterventionOperation::Assign {
                variable: *variable,
                expr: expr.clone(),
                reads: reads.clone(),
            },
            Self::Sequence(seq) => {
                let mut steps = Vec::with_capacity(seq.steps.len());
                for step in seq.steps.iter() {
                    steps.push(step.intervention.operation(Some(&step.temporal))?);
                }
                InterventionOperation::Sequence(Arc::from(steps))
            }
        };
        Ok(match temporal {
            None => untimed,
            Some(TemporalPolicy::Pulse { at }) => {
                InterventionOperation::Pulse { at: *at, action: Box::new(untimed) }
            }
            Some(TemporalPolicy::Sustained { from, until }) => InterventionOperation::Sustained {
                from: *from,
                until: *until,
                action: Box::new(untimed),
            },
            Some(TemporalPolicy::Dynamic { rule, active_at }) => InterventionOperation::Dynamic {
                rule: *rule,
                active_at: Arc::clone(active_at),
                action: Box::new(untimed),
            },
        })
    }
}

/// Name the inner operation when a timed wrapper would hide a shift, a soft
/// replacement, or another action that is not a set.
fn refusal_name(operation: &InterventionOperation) -> &'static str {
    match operation {
        InterventionOperation::Pulse { action, .. }
        | InterventionOperation::Sustained { action, .. }
        | InterventionOperation::Dynamic { action, .. } => match action.as_ref() {
            InterventionOperation::Set { .. } => operation.name(),
            inner => inner.name(),
        },
        other => other.name(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_set_primary_variable() {
        let v = VariableId::from_raw(3);
        let i = Intervention::set(v, Value::f64(1.0));
        assert_eq!(i.primary_variable(), Some(v));
        assert!(matches!(i, Intervention::Set { .. }));
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
    fn non_finite_levels_fail_validation() {
        let v = VariableId::from_raw(0);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                Intervention::set(v, Value::f64(bad)).validate(),
                Err(InterventionError::NonFiniteLevel)
            );
            assert_eq!(
                Intervention::shift(v, Value::f64(bad)).validate(),
                Err(InterventionError::NonFiniteLevel)
            );
            assert_eq!(
                Intervention::soft(v, MechanismOverride::additive_shift(bad)).validate(),
                Err(InterventionError::NonFiniteLevel)
            );
        }
        assert!(
            Intervention::stochastic(v, StochasticPolicy::gaussian(f64::NAN, 1.0))
                .validate()
                .is_err()
        );
        // The symbolic marker is not a numeric level and stays valid.
        assert!(Intervention::set(v, Value::symbolic_intervention()).validate().is_ok());
        assert!(Intervention::set(v, Value::f64(-0.0)).validate().is_ok());
    }

    #[test]
    fn policy_eq_and_hash_agree_on_signed_zero_and_nan() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash = |x: &dyn Fn(&mut DefaultHasher)| {
            let mut h = DefaultHasher::new();
            x(&mut h);
            h.finish()
        };
        let pos = StochasticPolicy::gaussian(0.0, 1.0);
        let neg = StochasticPolicy::gaussian(-0.0, 1.0);
        // Whatever Eq decides, Hash must decide the same way.
        assert_eq!(pos == neg, hash(&|h| pos.hash(h)) == hash(&|h| neg.hash(h)));
        let nan = StochasticPolicy::bernoulli(f64::NAN);
        assert_eq!(nan, nan.clone());
        assert_eq!(hash(&|h| nan.hash(h)), hash(&|h| nan.clone().hash(h)));
        let soft = MechanismOverride::additive_shift(-0.0);
        let soft_pos = MechanismOverride::additive_shift(0.0);
        assert_eq!(soft == soft_pos, hash(&|h| soft.hash(h)) == hash(&|h| soft_pos.hash(h)));
        let nan_soft = MechanismOverride::additive_shift(f64::NAN);
        assert_eq!(nan_soft, nan_soft.clone());
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

    #[test]
    fn transform_and_assign_validate_their_programs() {
        let x = VariableId::from_raw(0);
        let z = VariableId::from_raw(1);
        let floor =
            ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated()));
        let transform = Intervention::transform(x, floor.clone());
        transform.validate().unwrap();
        assert_eq!(transform.primary_variable(), Some(x));
        // An assignment replaces the generated value, so it cannot read it.
        assert_eq!(
            Intervention::assign(x, floor).validate(),
            Err(InterventionError::Expression { fault: ExpressionFault::GeneratedInAssignment })
        );
        let from_z = ValueExpr::constant(2.0).mul(ValueExpr::var(z));
        Intervention::assign(x, from_z.clone()).validate().unwrap();
        let undeclared =
            Intervention::Assign { variable: x, expr: from_z, reads: PolicyReads::default() };
        assert_eq!(
            undeclared.validate(),
            Err(InterventionError::Expression { fault: ExpressionFault::DeclaredReadsMismatch })
        );
        let boolean = Intervention::transform(x, ValueExpr::var(z).gt(ValueExpr::constant(0.0)));
        assert!(matches!(
            boolean.validate(),
            Err(InterventionError::Expression { fault: ExpressionFault::OutputType { .. } })
        ));
    }

    #[test]
    fn dynamic_policy_validates() {
        use crate::ids::DynamicRuleId;
        let p = TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), [0, 2, 5]);
        p.validate().unwrap();
        assert!(p.is_active_at(2));
        assert!(!p.is_active_at(1));
        assert_eq!(p.active_offsets().unwrap().as_ref(), &[0, 2, 5]);
        let empty = TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), []);
        assert!(matches!(empty.validate(), Err(InterventionError::EmptyDynamicSchedule)));
    }

    fn refusal_operation(error: AdmitError) -> &'static str {
        match error {
            AdmitError::Refused(refusal) => refusal.operation(),
            AdmitError::Invalid(error) => panic!("malformed intervention: {error}"),
        }
    }

    #[test]
    fn the_same_set_is_one_operation_on_every_surface_that_supports_it() {
        let set = Intervention::set(VariableId::from_raw(0), Value::f64(1.0));
        let static_op = ExecutionSurface::Static.admit(&set, None).unwrap();
        let counterfactual_op = ExecutionSurface::Counterfactual.admit(&set, None).unwrap();
        let cyclic_op = ExecutionSurface::Cyclic.admit(&set, None).unwrap();
        assert_eq!(static_op, counterfactual_op);
        assert_eq!(static_op, cyclic_op);
        assert!(matches!(static_op, InterventionOperation::Set { .. }));
        let refused = ExecutionSurface::Temporal.admit(&set, None).unwrap_err();
        let AdmitError::Refused(refusal) = refused else {
            panic!("an untimed set is not a finite-horizon action");
        };
        assert_eq!(refusal.operation(), "set");
    }

    #[test]
    fn shift_soft_pulse_sustained_and_initial_state_are_not_a_set() {
        let variable = VariableId::from_raw(0);
        let set = Intervention::set(variable, Value::f64(1.0));
        let shift = Intervention::shift(variable, Value::f64(1.0));
        let soft = Intervention::soft(variable, MechanismOverride::constant(1.0));
        let shift_op = ExecutionSurface::Static.admit(&shift, None).unwrap();
        let soft_op = ExecutionSurface::Static.admit(&soft, None).unwrap();
        assert!(matches!(shift_op, InterventionOperation::Shift { .. }));
        assert!(matches!(soft_op, InterventionOperation::Soft { .. }));
        assert_ne!(shift_op, ExecutionSurface::Static.admit(&set, None).unwrap());
        assert_ne!(soft_op, ExecutionSurface::Static.admit(&set, None).unwrap());

        let pulse =
            ExecutionSurface::Temporal.admit(&set, Some(&TemporalPolicy::pulse(0))).unwrap();
        let sustained =
            ExecutionSurface::Temporal.admit(&set, Some(&TemporalPolicy::sustained(0, 0))).unwrap();
        assert!(matches!(pulse, InterventionOperation::Pulse { .. }));
        assert!(matches!(sustained, InterventionOperation::Sustained { .. }));
        assert_ne!(pulse, sustained);
        assert_ne!(pulse.name(), "set");
        assert_ne!(sustained.name(), "set");
        assert_ne!(pulse.name(), sustained.name());

        for surface in
            [ExecutionSurface::Static, ExecutionSurface::Counterfactual, ExecutionSurface::Cyclic]
        {
            let admitted_shift = surface.supports(&shift_op);
            let admitted_soft = surface.supports(&soft_op);
            assert_eq!(admitted_shift, surface != ExecutionSurface::Cyclic);
            assert_eq!(admitted_soft, surface != ExecutionSurface::Cyclic);
            if surface == ExecutionSurface::Cyclic {
                assert_eq!(refusal_operation(surface.admit(&shift, None).unwrap_err()), "shift");
                assert_eq!(refusal_operation(surface.admit(&soft, None).unwrap_err()), "soft");
            }
            assert_eq!(
                refusal_operation(
                    surface.admit(&set, Some(&TemporalPolicy::pulse(0))).unwrap_err()
                ),
                "pulse"
            );
            assert_eq!(
                refusal_operation(
                    surface.admit(&set, Some(&TemporalPolicy::sustained(0, 0))).unwrap_err()
                ),
                "sustained"
            );
        }

        let initial = InterventionOperation::initial_state("x0", Value::f64(2.0)).unwrap();
        assert_eq!(initial.name(), "initial_state_change");
        for surface in ExecutionSurface::ALL {
            let refusal = surface.admit_operation(&initial).unwrap_err();
            assert_eq!(refusal.operation(), "initial_state_change");
            assert!(refusal.reason().contains("not a hard clamp"));
        }
        assert_eq!(
            InterventionOperation::initial_state("", Value::f64(2.0)).unwrap_err(),
            InterventionError::EmptyInitialLaw
        );
    }
}
