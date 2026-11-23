//! Intervention operation classification and surface capability admission.

use super::{Intervention, InterventionError, MechanismOverride, StochasticPolicy, TemporalPolicy};
use crate::ids::{DynamicRuleId, VariableId};
use crate::refusal::NextAttempt;
use crate::value::Value;
use crate::value_expr::{PolicyReads, ValueExpr};
use std::sync::Arc;

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
    /// Cyclic model. Without a named family this surface admits the operation
    /// every cyclic family performs: an untimed [`Intervention::Set`].
    /// [`CyclicFamily`] admits the further operations of one family.
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

/// Model family behind [`ExecutionSurface::Cyclic`].
///
/// Each family admits its own operations. A refusal is a [`CapabilityRefusal`]
/// on the cyclic surface, and [`Self::next_attempt`] names what to declare
/// instead. No operation is rewritten into another one:
///
/// - Both equilibrium families admit an untimed hard clamp
///   ([`InterventionOperation::Set`]), an additive shock on the structural
///   assignment ([`InterventionOperation::Shift`]) and a mechanism replacement
///   ([`InterventionOperation::Soft`]).
/// - The continuous-time family admits an untimed hard clamp held from the
///   initial time, a drift-mechanism replacement, and a change of an
///   initial-state law ([`InterventionOperation::InitialState`]).
///
/// The bare [`ExecutionSurface::Cyclic`] keeps admitting only the untimed set.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum CyclicFamily {
    /// Linear equilibrium `X = KX + c + ε` with a zero structural diagonal.
    LinearEquilibrium,
    /// Nonlinear additive-noise equilibrium.
    NonlinearEquilibrium,
    /// Deterministic continuous-time initial-value dynamics.
    ContinuousTime,
}

/// Declare the system on the continuous-time family, or run the schedule on
/// the temporal surface.
pub const CYCLIC_CONTINUOUS_TIME_ATTEMPT: NextAttempt = NextAttempt::new(
    "continuous_time_family",
    "declare the system on the continuous-time family, where an untimed set clamps from the \
     initial time and the initial state is its own law, or run the step schedule on the \
     temporal surface",
);

/// Evaluate each level of a stochastic assignment as its own hard-clamp world.
pub const CYCLIC_PER_LEVEL_CLAMP_ATTEMPT: NextAttempt = NextAttempt::new(
    "per_level_hard_clamp_worlds",
    "evaluate each level as its own hard-clamp world and mix the world results under the \
     declared assignment law",
);

/// Declare an expression intervention as a soft mechanism replacement.
pub const CYCLIC_SOFT_REPLACEMENT_ATTEMPT: NextAttempt = NextAttempt::new(
    "soft_mechanism_replacement",
    "declare the replacement equation as a soft intervention of family `constant`, \
     `linear_mechanism`, or `additive_shift`",
);

/// Declare a drift input or an initial-state change instead of a trajectory shift.
pub const CYCLIC_DRIFT_INPUT_ATTEMPT: NextAttempt = NextAttempt::new(
    "drift_input_or_initial_state",
    "declare a soft `additive_shift` drift input, or an initial-state change of the shifted \
     variable",
);

/// Declare an untimed clamp, or run the step schedule on the temporal surface.
pub const CYCLIC_UNTIMED_CLAMP_ATTEMPT: NextAttempt = NextAttempt::new(
    "untimed_clamp_or_temporal_surface",
    "declare an untimed set, which clamps from the initial time over the whole horizon, or run \
     the step schedule on the temporal surface",
);

/// Declare one of the operations the family performs.
pub const CYCLIC_SUPPORTED_OPERATION_ATTEMPT: NextAttempt = NextAttempt::new(
    "supported_cyclic_operation",
    "declare a hard clamp, a soft mechanism replacement, an additive shock, or on the \
     continuous-time family an initial-state change",
);

impl CyclicFamily {
    /// Every family, in a stable order.
    pub const ALL: [Self; 3] =
        [Self::LinearEquilibrium, Self::NonlinearEquilibrium, Self::ContinuousTime];

    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LinearEquilibrium => "linear_equilibrium",
            Self::NonlinearEquilibrium => "nonlinear_equilibrium",
            Self::ContinuousTime => "continuous_time",
        }
    }

    /// Whether this family performs `operation` without changing it.
    #[must_use]
    pub fn supports(self, operation: &InterventionOperation) -> bool {
        match self {
            Self::LinearEquilibrium | Self::NonlinearEquilibrium => matches!(
                operation,
                InterventionOperation::Set { .. }
                    | InterventionOperation::Shift { .. }
                    | InterventionOperation::Soft { .. }
            ),
            Self::ContinuousTime => matches!(
                operation,
                InterventionOperation::Set { .. }
                    | InterventionOperation::Soft { .. }
                    | InterventionOperation::InitialState { .. }
            ),
        }
    }

    /// Classify `intervention` under `temporal` and accept it on this family.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] on [`ExecutionSurface::Cyclic`] when this family
    /// cannot perform the operation; [`InterventionError`] when the intervention
    /// or the policy is malformed.
    pub fn admit(
        self,
        intervention: &Intervention,
        temporal: Option<&TemporalPolicy>,
    ) -> Result<InterventionOperation, AdmitError> {
        let operation = intervention.operation(temporal)?;
        self.admit_operation(&operation)?;
        Ok(operation)
    }

    /// Accept `operation` on this family.
    ///
    /// # Errors
    ///
    /// [`CapabilityRefusal`] on [`ExecutionSurface::Cyclic`] naming the operation.
    pub fn admit_operation(
        self,
        operation: &InterventionOperation,
    ) -> Result<(), CapabilityRefusal> {
        if self.supports(operation) {
            return Ok(());
        }
        let name = self.refused_name(operation);
        Err(CapabilityRefusal {
            surface: ExecutionSurface::Cyclic,
            operation: name,
            reason: self.refusal_parts(name).0,
        })
    }

    /// What to declare instead of the refused operation `operation`.
    ///
    /// `operation` is the name a [`CapabilityRefusal`] from this family carries.
    #[must_use]
    pub fn next_attempt(self, operation: &str) -> NextAttempt {
        self.refusal_parts(operation).1
    }

    /// The timed wrapper is named when its action is one this family performs
    /// untimed; otherwise the refused inner action is named.
    fn refused_name(self, operation: &InterventionOperation) -> &'static str {
        match operation {
            InterventionOperation::Pulse { action, .. }
            | InterventionOperation::Sustained { action, .. }
            | InterventionOperation::Dynamic { action, .. } => {
                if self.supports(action) {
                    operation.name()
                } else {
                    self.refused_name(action)
                }
            }
            other => other.name(),
        }
    }

    fn refusal_parts(self, operation: &str) -> (&'static str, NextAttempt) {
        let equilibrium = !matches!(self, Self::ContinuousTime);
        match (equilibrium, operation) {
            (true, "pulse" | "sustained" | "dynamic" | "sequence") => (
                "a static equilibrium has no time axis, so a timed action is not an equilibrium \
                 intervention",
                CYCLIC_CONTINUOUS_TIME_ATTEMPT,
            ),
            (true, "initial_state_change") => (
                "an equilibrium has no initial-state law; solver initialization is numerical and \
                 is not a causal initial state",
                CYCLIC_CONTINUOUS_TIME_ATTEMPT,
            ),
            (false, "shift") => (
                "a shift on a trajectory could be a state jump or a drift input, and the \
                 continuous-time family does not choose between them",
                CYCLIC_DRIFT_INPUT_ATTEMPT,
            ),
            (false, "pulse" | "sustained" | "dynamic" | "sequence") => (
                "a step-indexed schedule is not a real-time action of the continuous-time \
                 family, whose clamps hold from the initial time",
                CYCLIC_UNTIMED_CLAMP_ATTEMPT,
            ),
            (_, "stochastic") => (
                "a stochastic assignment needs its law joined with the model's disturbance, \
                 which this family does not declare",
                CYCLIC_PER_LEVEL_CLAMP_ATTEMPT,
            ),
            (_, "transform" | "assign") => (
                "an expression intervention inside a feedback loop is a new simultaneous \
                 equation that this family does not compile from an expression",
                CYCLIC_SOFT_REPLACEMENT_ATTEMPT,
            ),
            _ => (
                "this cyclic family does not perform that operation",
                CYCLIC_SUPPORTED_OPERATION_ATTEMPT,
            ),
        }
    }
}

impl std::fmt::Display for CyclicFamily {
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
