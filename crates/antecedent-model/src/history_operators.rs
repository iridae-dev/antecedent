// SPDX-License-Identifier: MIT OR Apache-2.0

//! History, timing, policy, transition, regression, weight, and reduction operators.
//!
//! Each operator is a value over a declared cadence, horizon, and initial law.
//! The cadence is [`TimeIndex`]: a regular interval in nanoseconds and the
//! number of ticks that interval declares. [`Horizon`] is the exclusive end of
//! the operator in those same nanoseconds. A step count such as
//! [`TimeFrame::horizon_steps`](antecedent_core::TimeFrame::horizon_steps)
//! cannot state an end that falls between ticks, so the check lives here.
//!
//! The initial law is [`InitialLaw`]. Absence is refused. A time column is not
//! a law.
//!
//! A horizon that does not end on a tick, or that runs past the last tick the
//! cadence declares, is refused. A censoring or assignment weight that is not
//! finite is refused.
//!
//! Delayed feedback binds that same frame to a history window, an initial law,
//! and one transition step. A finite-horizon response of that binding is a
//! different target from a static equilibrium, a stationary law, and a
//! continuous-time dynamic. A pulse is a different action from a sustained
//! window, including a window of one step. A history shorter than the delay is
//! refused unless a truncation certificate names that same short history; the
//! history is not extended. A search cap that does not cover the ancestry is
//! refused, and the cap is not raised.
//!
//! The operators are values until they run: [`HistoryProgram`] binds them over one frame
//! and executes them on a [`TickSeries`], with one scan, one weighted fit, one rolled
//! transition under the policy and its reference, and one reduction. Every consumer of a
//! temporal program reads those operators instead of its own scan.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::sync::Arc;

use antecedent_core::intervention::{AdmitError, ExecutionSurface, InterventionOperation};
use antecedent_core::{
    HistoryWindow, InitialLaw, Intervention, InterventionError, TemporalPolicy, Value, is_blank,
};
use antecedent_data::{SamplingRegularity, TimeIndex};
use thiserror::Error;

use crate::information_time::{
    AvailableOutcome, CadenceTick, ForwardOnly, InformationRefusal, InformationUse,
    ObservedCadence, RetrospectivePermission, admit_delayed_outcome,
};

/// Exclusive end of an operator, in the cadence's nanoseconds.
///
/// The origin is the cadence's first tick. [`Self::ending_at`] does not check
/// the cadence; [`OperatorFrame::try_new`] does.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Horizon {
    end_ns: u64,
}

impl Horizon {
    /// Horizon ending at `end_ns`, exclusive, measured from the cadence origin.
    #[must_use]
    pub const fn ending_at(end_ns: u64) -> Self {
        Self { end_ns }
    }

    /// Exclusive end, in nanoseconds from the cadence origin.
    #[must_use]
    pub const fn end_ns(self) -> u64 {
        self.end_ns
    }
}

/// Cadence, horizon, and initial law shared by every history operator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorFrame {
    cadence: TimeIndex,
    horizon: Horizon,
    initial_law: InitialLaw,
    interval_ns: u64,
}

impl OperatorFrame {
    /// Bind `cadence`, `horizon`, and `initial_law`.
    ///
    /// # Errors
    ///
    /// [`HistoryOperatorError::HorizonCrossesCadence`] when the horizon does
    /// not end on a tick of `cadence` or runs past the ticks `cadence` declares.
    /// [`HistoryOperatorError::CadenceSpanOverflow`] when the declared tick count
    /// times the interval does not fit in `u64`.
    /// [`HistoryOperatorError::CadenceUncountable`] when the tick count does not
    /// fit in `u32`, the same grid [`crate::information_time::ObservedCadence`]
    /// reads from [`TimeIndex`].
    #[must_use = "a refused frame is not a cadence, horizon, and initial law"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
    ) -> Result<Self, HistoryOperatorError> {
        let interval_ns = admit_horizon(&cadence, horizon)?;
        Ok(Self { cadence, horizon, initial_law, interval_ns })
    }

    /// Declared observation cadence.
    #[must_use]
    pub const fn cadence(&self) -> &TimeIndex {
        &self.cadence
    }

    /// Horizon bound to this cadence.
    #[must_use]
    pub const fn horizon(&self) -> Horizon {
        self.horizon
    }

    /// Initial-state law. Construction refuses a missing law.
    #[must_use]
    pub const fn initial_law(&self) -> &InitialLaw {
        &self.initial_law
    }

    /// Positive regular interval of [`Self::cadence`], in nanoseconds.
    #[must_use]
    pub const fn interval_ns(&self) -> u64 {
        self.interval_ns
    }
}

/// Which weight was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WeightRole {
    /// Censoring weight.
    Censoring,
    /// Treatment-assignment weight.
    Assignment,
}

impl WeightRole {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Censoring => "censoring",
            Self::Assignment => "assignment",
        }
    }
}

impl std::fmt::Display for WeightRole {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Refusal while compiling a history operator.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum HistoryOperatorError {
    /// The horizon misses a tick or runs past the cadence.
    #[error("horizon {end_ns} ns crosses the cadence; next attempt: end on a tick inside it")]
    HorizonCrossesCadence {
        /// Exclusive horizon end, in nanoseconds.
        end_ns: u64,
        /// Regular interval, or [`None`] when the cadence is irregular.
        interval_ns: Option<u64>,
        /// Declared tick count, when it fits in `u64`.
        declared_ticks: Option<u64>,
    },
    /// Tick count times the interval does not fit in `u64`.
    #[error("cadence of {ticks} ticks at {interval_ns} ns overflows; next attempt: shorten it")]
    CadenceSpanOverflow {
        /// Regular interval, in nanoseconds.
        interval_ns: u64,
        /// Declared tick count.
        ticks: u64,
    },
    /// The tick count does not fit in `u32`.
    ///
    /// This is the same limit as [`crate::information_time::ObservedCadence`].
    #[error("cadence tick count does not fit in u32; next attempt: shorten the cadence")]
    CadenceUncountable,
    /// A censoring or assignment weight was `NaN` or infinite.
    #[error("{role} weight {value} is not finite; next attempt: supply a finite weight")]
    NonFiniteWeight {
        /// Which weight.
        role: WeightRole,
        /// The refused value.
        value: f64,
    },
    /// A censoring or assignment weight was finite but zero or negative.
    ///
    /// These are inverse-probability weights. A zero weight is a probability of
    /// infinity, which is a positivity violation, and a negative weight is not a
    /// reciprocal of a probability.
    #[error(
        "{role} weight {value} is not positive; next attempt: supply a finite weight above zero, \
         or declare the positivity violation"
    )]
    NonPositiveWeight {
        /// Which weight.
        role: WeightRole,
        /// The refused value.
        value: f64,
    },
    /// An action time is off the cadence or outside the horizon.
    #[error(
        "action at {at_ns} ns misses the cadence; next attempt: use a tick before the horizon end"
    )]
    ActionMissesCadence {
        /// Requested action time, in nanoseconds from the origin.
        at_ns: u64,
        /// Cadence interval, in nanoseconds.
        interval_ns: u64,
    },
    /// A policy, outcome, or functional name is blank.
    #[error("{field} name is empty; next attempt: name the {field}")]
    EmptyName {
        /// `policy`, `outcome`, or `functional`.
        field: &'static str,
    },
    /// A delay of zero is contemporaneous transmission, not delayed feedback.
    #[error(
        "feedback delay is 0 ticks; next attempt: use a positive delay, or a finite horizon \
         without delay"
    )]
    ContemporaneousFeedback,
    /// An action step lies outside the half-open horizon.
    #[error(
        "action step {step} is outside a horizon of {horizon_steps} steps; next attempt: place \
         it before the horizon end"
    )]
    ActionOutsideHorizon {
        /// Offending step, relative to the origin.
        step: i32,
        /// Exclusive horizon length, in cadence steps.
        horizon_steps: u32,
    },
    /// A dynamic schedule is neither a pulse nor a sustained window.
    #[error("action is a dynamic schedule; next attempt: declare a pulse or a sustained window")]
    ActionNotDelayable,
    /// A level or a dynamic coefficient was `NaN` or infinite.
    #[error("level {value} is not finite; next attempt: supply a finite level")]
    NonFiniteLevel {
        /// The refused value.
        value: f64,
    },
    /// A sustained window ends before it starts.
    #[error(
        "sustained window {from_step}..={until_step} is inverted; next attempt: order the window"
    )]
    InvertedWindow {
        /// First step.
        from_step: i32,
        /// Last step.
        until_step: i32,
    },
    /// History is shorter than the delay and no certificate names that history.
    ///
    /// The retained depth is the one that was declared. It is not rewritten to
    /// the required depth.
    #[error(
        "history retains {retained_ticks} ticks but delayed feedback needs {required_ticks}; \
         next attempt: certify that truncation, or declare a history that covers the delay"
    )]
    UncertifiedTruncation {
        /// Explicit history cap, in ticks. Not increased.
        retained_ticks: u32,
        /// Delay or action lag that the history does not cover.
        required_ticks: u32,
    },
    /// Ancestry does not fit in the declared search cap.
    ///
    /// `cap` is the cap that was given. It is not replaced by `ancestry_steps`.
    #[error(
        "search cap {cap} does not cover ancestry of {ancestry_steps} steps; next attempt: \
         report it uncertified without raising the cap"
    )]
    SearchNotIdentified {
        /// Declared cap, unchanged.
        cap: u32,
        /// Ancestry depth that was asked about.
        ancestry_steps: u32,
    },
    /// The delayed outcome is not yet available at the decision.
    #[error("delayed feedback is not available: {refusal}")]
    InformationRefused {
        /// Information-time refusal. The row is not dropped.
        refusal: InformationRefusal,
    },
    /// The exclusive horizon length does not fit in `u32` steps.
    #[error("horizon length does not fit in u32 steps; next attempt: shorten the horizon")]
    HorizonStepsUncountable,
    /// `steps` times the interval does not fit in `u64`.
    #[error(
        "horizon of {steps} steps at {interval_ns} ns overflows; next attempt: shorten the horizon"
    )]
    HorizonEndOverflow {
        /// Regular interval, in nanoseconds.
        interval_ns: u64,
        /// Requested step count.
        steps: u32,
    },
    /// `a` is one, so the map has no finite fixed point.
    #[error(
        "autoregression has no finite fixed point; next attempt: ask for the finite-horizon \
         response"
    )]
    NoFixedPoint,
    /// `|a| >= 1` and not one, so the map has no stable fixed point or stationary law.
    #[error(
        "autoregression {autoregression} is not stable (|a| >= 1); next attempt: ask for the \
         finite-horizon response"
    )]
    UnstableAutoregression {
        /// The refused coefficient.
        autoregression: f64,
    },
    /// The innovation variance is negative.
    #[error(
        "innovation variance {value} is negative; next attempt: supply a variance of zero or more"
    )]
    NegativeInnovationVariance {
        /// The refused value.
        value: f64,
    },
    /// The coefficients were declared for a different initial law than the frame's.
    #[error(
        "dynamics declare initial law {dynamics:?} but the frame binds {frame:?}; next attempt: \
         declare the state of the frame's law"
    )]
    InitialLawMismatch {
        /// Initial law bound to the delayed frame.
        frame: InitialLaw,
        /// Initial law the dynamics were declared for.
        dynamics: InitialLaw,
    },
    /// The intervention is not a pulse or a sustained set, and was not rewritten.
    #[error("{operation} is not a delayed-feedback action: {reason}")]
    UnsupportedOperation {
        /// Operation that was asked for.
        operation: &'static str,
        /// Why it was not executed as a different action.
        reason: &'static str,
    },
    /// The cadence has no regular positive interval to convert steps into nanoseconds.
    #[error(
        "a horizon of {steps} steps has no end on a cadence with interval {interval_ns:?} ns; \
         next attempt: declare a regular positive cadence"
    )]
    CadenceWithoutInterval {
        /// Regular interval, or [`None`] when the cadence is irregular.
        interval_ns: Option<u64>,
        /// Requested step count.
        steps: u32,
    },
}

/// History retained on the frame. This does not read rows or fill lags.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HistoryConstruction {
    frame: OperatorFrame,
    history: HistoryWindow,
}

impl HistoryConstruction {
    /// Compile history construction.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`].
    #[must_use = "a refused history construction is not an operator"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
        history: HistoryWindow,
    ) -> Result<Self, HistoryOperatorError> {
        let frame = OperatorFrame::try_new(cadence, horizon, initial_law)?;
        Ok(Self { frame, history })
    }

    /// Cadence, horizon, and initial law.
    #[must_use]
    pub const fn frame(&self) -> &OperatorFrame {
        &self.frame
    }

    /// History window the operator retains. It is not inferred from a column.
    #[must_use]
    pub const fn history(&self) -> HistoryWindow {
        self.history
    }
}

/// Time at which an action is applied. This does not apply the action.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionTiming {
    frame: OperatorFrame,
    at_ns: u64,
}

impl ActionTiming {
    /// Compile action timing at `at_ns`.
    ///
    /// `at_ns` is nanoseconds from the cadence origin. It must be a tick and
    /// must lie in the half-open horizon.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`], or
    /// [`HistoryOperatorError::ActionMissesCadence`].
    #[must_use = "a refused action time is not an operator"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
        at_ns: u64,
    ) -> Result<Self, HistoryOperatorError> {
        let frame = OperatorFrame::try_new(cadence, horizon, initial_law)?;
        admit_action(&frame, at_ns)?;
        Ok(Self { frame, at_ns })
    }

    /// Cadence, horizon, and initial law.
    #[must_use]
    pub const fn frame(&self) -> &OperatorFrame {
        &self.frame
    }

    /// Action time, in nanoseconds from the cadence origin.
    #[must_use]
    pub const fn at_ns(&self) -> u64 {
        self.at_ns
    }
}

/// A named policy evaluated on the frame. This does not evaluate the policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyEvaluation {
    frame: OperatorFrame,
    policy: Arc<str>,
}

impl PolicyEvaluation {
    /// Compile policy evaluation of `policy`.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`], or
    /// [`HistoryOperatorError::EmptyName`] when `policy` is blank.
    #[must_use = "a refused policy evaluation is not an operator"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
        policy: impl Into<Arc<str>>,
    ) -> Result<Self, HistoryOperatorError> {
        let frame = OperatorFrame::try_new(cadence, horizon, initial_law)?;
        let policy = admit_name("policy", policy.into())?;
        Ok(Self { frame, policy })
    }

    /// Cadence, horizon, and initial law.
    #[must_use]
    pub const fn frame(&self) -> &OperatorFrame {
        &self.frame
    }

    /// Policy name. The name is not a fitted policy.
    #[must_use]
    pub fn policy(&self) -> &str {
        &self.policy
    }
}

/// One transition of the cadence interval. This does not fit a transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transition {
    frame: OperatorFrame,
}

impl Transition {
    /// Compile the transition across one cadence interval.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`].
    #[must_use = "a refused transition is not an operator"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
    ) -> Result<Self, HistoryOperatorError> {
        Ok(Self { frame: OperatorFrame::try_new(cadence, horizon, initial_law)? })
    }

    /// Cadence, horizon, and initial law.
    #[must_use]
    pub const fn frame(&self) -> &OperatorFrame {
        &self.frame
    }

    /// Transition step, which is the cadence interval in nanoseconds.
    #[must_use]
    pub const fn step_ns(&self) -> u64 {
        self.frame.interval_ns()
    }
}

/// Sequential regression and integration over the horizon.
///
/// Regression and integration are this one operator. It does not fit the
/// regression and it does not integrate a column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequentialRegression {
    frame: OperatorFrame,
    outcome: Arc<str>,
}

impl SequentialRegression {
    /// Compile sequential regression and integration of `outcome`.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`], or
    /// [`HistoryOperatorError::EmptyName`] when `outcome` is blank.
    #[must_use = "a refused sequential regression is not an operator"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
        outcome: impl Into<Arc<str>>,
    ) -> Result<Self, HistoryOperatorError> {
        let frame = OperatorFrame::try_new(cadence, horizon, initial_law)?;
        let outcome = admit_name("outcome", outcome.into())?;
        Ok(Self { frame, outcome })
    }

    /// Cadence, horizon, and initial law.
    #[must_use]
    pub const fn frame(&self) -> &OperatorFrame {
        &self.frame
    }

    /// Outcome the regression and the integration name.
    #[must_use]
    pub fn outcome(&self) -> &str {
        &self.outcome
    }
}

/// Censoring and assignment weights on the frame. This does not weight rows.
#[derive(Clone, Debug, PartialEq)]
pub struct CensoringAssignmentWeight {
    frame: OperatorFrame,
    censoring: f64,
    assignment: f64,
}

impl CensoringAssignmentWeight {
    /// Compile `censoring` and `assignment` weights.
    ///
    /// Both values are inverse-probability weights and must be finite and above
    /// zero. A zero weight is a positivity violation, and a negative weight is
    /// not the reciprocal of a probability.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`],
    /// [`HistoryOperatorError::NonFiniteWeight`], or
    /// [`HistoryOperatorError::NonPositiveWeight`].
    #[must_use = "a refused weight is not an operator"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
        censoring: f64,
        assignment: f64,
    ) -> Result<Self, HistoryOperatorError> {
        let frame = OperatorFrame::try_new(cadence, horizon, initial_law)?;
        admit_weight(WeightRole::Censoring, censoring)?;
        admit_weight(WeightRole::Assignment, assignment)?;
        Ok(Self { frame, censoring, assignment })
    }

    /// Cadence, horizon, and initial law.
    #[must_use]
    pub const fn frame(&self) -> &OperatorFrame {
        &self.frame
    }

    /// Censoring weight as declared.
    #[must_use]
    pub const fn censoring(&self) -> f64 {
        self.censoring
    }

    /// Assignment weight as declared.
    #[must_use]
    pub const fn assignment(&self) -> f64 {
        self.assignment
    }
}

/// Reduction of a trajectory to a named functional. This does not reduce rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrajectoryReduction {
    frame: OperatorFrame,
    functional: Arc<str>,
}

impl TrajectoryReduction {
    /// Compile trajectory reduction to `functional`.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`], or
    /// [`HistoryOperatorError::EmptyName`] when `functional` is blank.
    #[must_use = "a refused trajectory reduction is not an operator"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: InitialLaw,
        functional: impl Into<Arc<str>>,
    ) -> Result<Self, HistoryOperatorError> {
        let frame = OperatorFrame::try_new(cadence, horizon, initial_law)?;
        let functional = admit_name("functional", functional.into())?;
        Ok(Self { frame, functional })
    }

    /// Cadence, horizon, and initial law.
    #[must_use]
    pub const fn frame(&self) -> &OperatorFrame {
        &self.frame
    }

    /// Functional name. The name is not an evaluated reduction.
    #[must_use]
    pub fn functional(&self) -> &str {
        &self.functional
    }
}

/// Ticks before a feedback edge re-enters the state.
///
/// Zero is contemporaneous transmission and is refused here. A finite horizon
/// with no delay uses [`Option::None`] on [`DelayedFeedbackRequest::delay`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FeedbackDelay {
    ticks: u32,
}

impl FeedbackDelay {
    /// Delay of `ticks` cadence steps.
    ///
    /// # Errors
    ///
    /// [`HistoryOperatorError::ContemporaneousFeedback`] when `ticks` is zero.
    #[must_use = "a zero delay is not delayed feedback"]
    pub const fn delayed(ticks: u32) -> Result<Self, HistoryOperatorError> {
        if ticks == 0 {
            Err(HistoryOperatorError::ContemporaneousFeedback)
        } else {
            Ok(Self { ticks })
        }
    }

    /// Delay in cadence steps. At least one.
    #[must_use = "the delay is the lag the history must cover"]
    pub const fn ticks(self) -> u32 {
        self.ticks
    }
}

/// Caller-supplied replacement of an initial-state law.
///
/// Construction of [`DelayedFeedback`] refuses this. It does not clamp a
/// variable and it does not change the declared dynamics.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialStateChange {
    /// Name of the law being replaced.
    pub law_name: String,
    /// Replacement level.
    pub level: f64,
}

/// Cap on a feedback-ancestry search, in steps.
///
/// The stored count is the cap that was declared. Certifying a deeper ancestry
/// does not replace it with that depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FeedbackSearchCap {
    steps: u32,
}

impl FeedbackSearchCap {
    /// Cap of `steps`.
    #[must_use = "the cap is the bound the search was given"]
    pub const fn declared(steps: u32) -> Self {
        Self { steps }
    }

    /// Declared step count. Not a wider bound discovered during search.
    #[must_use = "the cap is the bound the search was given"]
    pub const fn steps(self) -> u32 {
        self.steps
    }
}

/// Named permission to use a history shorter than the delay.
///
/// The certificate names the retained depth. It does not lengthen the history
/// window to the delay.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TruncationCertificate {
    retained_ticks: u32,
}

impl TruncationCertificate {
    /// Certify truncation at exactly `retained_ticks`.
    #[must_use = "a certificate names the history it truncates"]
    pub const fn retaining(retained_ticks: u32) -> Self {
        Self { retained_ticks }
    }

    /// Retained depth this certificate names.
    #[must_use = "the retained depth is the history the certificate allows"]
    pub const fn retained_ticks(self) -> u32 {
        self.retained_ticks
    }
}

/// Build the delayed action the caller declared.
///
/// A pulse of a set stays a pulse. A sustained set stays sustained, including a
/// window of one step. A shift, a soft replacement, and an untimed set are
/// refused and are not rewritten as a hard assignment.
///
/// # Errors
///
/// [`HistoryOperatorError::ActionNotDelayable`] for a dynamic schedule.
/// [`HistoryOperatorError::UnsupportedOperation`] when the intervention is not
/// a pulse or a sustained set. [`HistoryOperatorError::NonFiniteLevel`] when
/// the set level is not finite. [`HistoryOperatorError::InvertedWindow`] when
/// the sustained window ends before it starts.
#[must_use = "a refused action is not the caller's intervention"]
pub fn delayed_action_of(
    intervention: &Intervention,
    temporal: &TemporalPolicy,
) -> Result<DelayedAction, HistoryOperatorError> {
    if matches!(temporal, TemporalPolicy::Dynamic { .. }) {
        return Err(HistoryOperatorError::ActionNotDelayable);
    }
    let operation = ExecutionSurface::Temporal
        .admit(intervention, Some(temporal))
        .map_err(|error| history_refusal(&error))?;
    match operation {
        InterventionOperation::Pulse { at, action } => {
            DelayedAction::pulse(at, finite_set_level(action.as_ref())?)
        }
        InterventionOperation::Sustained { from, until, action } => {
            DelayedAction::sustained(from, until, finite_set_level(action.as_ref())?)
        }
        other => Err(HistoryOperatorError::UnsupportedOperation {
            operation: other.name(),
            reason: "a delayed-feedback action is a pulse or a sustained set",
        }),
    }
}

/// Refuse a replacement of the initial-state law.
///
/// The rollout keeps the law the dynamics declared. The change is not applied
/// as a clamp and is not applied as the action level.
///
/// # Errors
///
/// [`HistoryOperatorError::UnsupportedOperation`] naming `initial_state_change`,
/// or a malformed law name or level.
#[must_use = "a refused initial-state change is not a rollout"]
pub fn refuse_initial_state_change(law_name: &str, level: f64) -> Result<(), HistoryOperatorError> {
    let operation =
        InterventionOperation::initial_state(law_name, Value::f64(level)).map_err(|error| {
            match error {
                InterventionError::EmptyInitialLaw => {
                    HistoryOperatorError::EmptyName { field: "initial law" }
                }
                InterventionError::NonFiniteLevel => {
                    HistoryOperatorError::NonFiniteLevel { value: level }
                }
                _ => HistoryOperatorError::UnsupportedOperation {
                    operation: "initial_state_change",
                    reason: "an initial-state change replaces the initial-state law and is not a \
                             hard clamp",
                },
            }
        })?;
    match ExecutionSurface::Temporal.admit_operation(&operation) {
        Err(refusal) => Err(history_operation(refusal)),
        Ok(()) => Err(HistoryOperatorError::UnsupportedOperation {
            operation: "initial_state_change",
            reason: "an initial-state change replaces the initial-state law and is not a \
                     hard clamp",
        }),
    }
}

fn history_refusal(error: &AdmitError) -> HistoryOperatorError {
    match error {
        AdmitError::Refused(refusal) => history_operation(*refusal),
        AdmitError::Invalid(InterventionError::InvalidTemporalWindow { from, until }) => {
            HistoryOperatorError::InvertedWindow { from_step: *from, until_step: *until }
        }
        AdmitError::Invalid(InterventionError::NonFiniteLevel) => {
            HistoryOperatorError::NonFiniteLevel { value: f64::NAN }
        }
        AdmitError::Invalid(_) => HistoryOperatorError::UnsupportedOperation {
            operation: "intervention",
            reason: "the intervention is malformed",
        },
        _ => HistoryOperatorError::UnsupportedOperation {
            operation: "intervention",
            reason: "the intervention was not admitted",
        },
    }
}

fn history_operation(
    refusal: antecedent_core::intervention::CapabilityRefusal,
) -> HistoryOperatorError {
    HistoryOperatorError::UnsupportedOperation {
        operation: refusal.operation(),
        reason: refusal.reason(),
    }
}

fn finite_set_level(action: &InterventionOperation) -> Result<f64, HistoryOperatorError> {
    match action {
        InterventionOperation::Set { value: Value::Float64(level), .. } if level.is_finite() => {
            Ok(*level)
        }
        InterventionOperation::Set { value: Value::Float64(level), .. } => {
            Err(HistoryOperatorError::NonFiniteLevel { value: *level })
        }
        InterventionOperation::Set { .. } => Err(HistoryOperatorError::UnsupportedOperation {
            operation: "set",
            reason: "a delayed-feedback set holds one finite float and is not coerced",
        }),
        other => Err(HistoryOperatorError::UnsupportedOperation {
            operation: other.name(),
            reason: "a delayed-feedback action holds a set and does not rewrite it",
        }),
    }
}

/// Pulse or sustained action on the delayed frame.
///
/// A sustained window with `from_step == until_step` is still sustained. It is
/// not a pulse.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DelayedAction {
    /// One step.
    Pulse {
        /// Step relative to the origin. Negative steps sit in the history.
        at_step: i32,
        /// Bit pattern of the level. `-0.0` is stored as `+0.0`.
        level_bits: u64,
    },
    /// Inclusive step window.
    Sustained {
        /// First intervened step.
        from_step: i32,
        /// Last intervened step, inclusive.
        until_step: i32,
        /// Bit pattern of the held level. `-0.0` is stored as `+0.0`.
        level_bits: u64,
    },
}

impl DelayedAction {
    /// Pulse at `at_step` with finite `level`.
    ///
    /// # Errors
    ///
    /// [`HistoryOperatorError::NonFiniteLevel`] when `level` is not finite.
    #[must_use = "a non-finite pulse level is not an action"]
    pub fn pulse(at_step: i32, level: f64) -> Result<Self, HistoryOperatorError> {
        Ok(Self::Pulse { at_step, level_bits: finite_bits(level)? })
    }

    /// Sustained action on `from_step..=until_step` with finite `level`.
    ///
    /// # Errors
    ///
    /// [`HistoryOperatorError::InvertedWindow`] when `until_step < from_step`.
    /// [`HistoryOperatorError::NonFiniteLevel`] when `level` is not finite.
    #[must_use = "a refused sustained window is not an action"]
    pub fn sustained(
        from_step: i32,
        until_step: i32,
        level: f64,
    ) -> Result<Self, HistoryOperatorError> {
        if until_step < from_step {
            return Err(HistoryOperatorError::InvertedWindow { from_step, until_step });
        }
        Ok(Self::Sustained { from_step, until_step, level_bits: finite_bits(level)? })
    }

    /// Level at `step`, or zero when the action is inactive there.
    #[must_use = "an inactive step contributes no action level"]
    pub const fn level_at(self, step: i32) -> f64 {
        let bits = match self {
            Self::Pulse { at_step, level_bits } if step == at_step => level_bits,
            Self::Sustained { from_step, until_step, level_bits }
                if step >= from_step && step <= until_step =>
            {
                level_bits
            }
            Self::Pulse { .. } | Self::Sustained { .. } => 0.0f64.to_bits(),
        };
        f64::from_bits(bits)
    }

    /// Level held forever by an equilibrium or stationary question.
    ///
    /// A pulse returns to zero. A sustained action keeps its level.
    #[must_use = "the permanent level is the constant input of an equilibrium"]
    pub const fn permanent_level(self) -> f64 {
        match self {
            Self::Pulse { .. } => 0.0,
            Self::Sustained { level_bits, .. } => f64::from_bits(level_bits),
        }
    }
}

/// Which response a delayed system was asked for.
///
/// These are different targets. A finite-horizon rollout is not a static
/// equilibrium, not a stationary law, and not a continuous-time dynamic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResponseTarget {
    /// State after the admitted number of transition steps.
    FiniteHorizon,
    /// Fixed point of the one-step map under the action's permanent level.
    StaticEquilibrium,
    /// Invariant mean and innovation-driven variance of that map.
    Stationary,
    /// Continuous-time convolution over the same horizon.
    ContinuousTime,
}

impl ResponseTarget {
    /// Stable name.
    #[must_use = "the name distinguishes this target from the other three"]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FiniteHorizon => "finite_horizon",
            Self::StaticEquilibrium => "static_equilibrium",
            Self::Stationary => "stationary",
            Self::ContinuousTime => "continuous_time",
        }
    }
}

impl std::fmt::Display for ResponseTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Coefficients of `s <- a s + b u + e`, the innovation variance of `e`, and
/// the state at the origin.
///
/// The initial state is the level of one [`InitialLaw`]. The dynamics carry
/// that law, and [`DelayedFeedback::response`] refuses dynamics declared for a
/// different law than the frame binds.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DelayedDynamics {
    autoregression_bits: u64,
    input_gain_bits: u64,
    innovation_variance_bits: u64,
    initial_law: InitialLaw,
    initial_state_bits: u64,
}

impl DelayedDynamics {
    /// Finite autoregression `a`, input gain `b`, innovation variance, and the
    /// level of `initial_law` at the origin.
    ///
    /// # Errors
    ///
    /// [`HistoryOperatorError::NonFiniteLevel`] when any number is not finite.
    /// [`HistoryOperatorError::NegativeInnovationVariance`] when the variance
    /// is below zero.
    #[must_use = "a non-finite coefficient is not a delayed system"]
    pub fn try_new(
        autoregression: f64,
        input_gain: f64,
        innovation_variance: f64,
        initial_law: InitialLaw,
        initial_state: f64,
    ) -> Result<Self, HistoryOperatorError> {
        let innovation_variance_bits = finite_bits(innovation_variance)?;
        if innovation_variance < 0.0 {
            return Err(HistoryOperatorError::NegativeInnovationVariance {
                value: innovation_variance,
            });
        }
        Ok(Self {
            autoregression_bits: finite_bits(autoregression)?,
            input_gain_bits: finite_bits(input_gain)?,
            innovation_variance_bits,
            initial_law,
            initial_state_bits: finite_bits(initial_state)?,
        })
    }

    /// Autoregressive coefficient.
    #[must_use = "the autoregression is the transition's state coefficient"]
    pub const fn autoregression(&self) -> f64 {
        f64::from_bits(self.autoregression_bits)
    }

    /// Coefficient on the delayed action.
    #[must_use = "the input gain scales the delayed action"]
    pub const fn input_gain(&self) -> f64 {
        f64::from_bits(self.input_gain_bits)
    }

    /// Variance of the transition's innovation.
    #[must_use = "the innovation variance sets the stationary spread"]
    pub const fn innovation_variance(&self) -> f64 {
        f64::from_bits(self.innovation_variance_bits)
    }

    /// Law whose level at the origin is [`Self::initial_state`].
    #[must_use = "the law names the state the rollout starts from"]
    pub const fn initial_law(&self) -> &InitialLaw {
        &self.initial_law
    }

    /// Level of [`Self::initial_law`] at the origin.
    #[must_use = "the initial state is the law's level at the origin"]
    pub const fn initial_state(&self) -> f64 {
        f64::from_bits(self.initial_state_bits)
    }
}

/// Level of one requested target.
///
/// `target` is the question that was asked. Two responses with different
/// targets are not interchangeable, even when a coefficient choice makes a
/// level coincide.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TargetResponse {
    target: ResponseTarget,
    level_bits: u64,
    spread_bits: u64,
    action: DelayedAction,
}

impl TargetResponse {
    /// Target that was evaluated.
    #[must_use = "the target distinguishes a finite horizon from the other laws"]
    pub const fn target(self) -> ResponseTarget {
        self.target
    }

    /// Bit pattern of the response level.
    #[must_use = "the level is the response of this target"]
    pub const fn level_bits(self) -> u64 {
        self.level_bits
    }

    /// Response level.
    #[must_use = "the level is the response of this target"]
    pub const fn level(self) -> f64 {
        f64::from_bits(self.level_bits)
    }

    /// Bit pattern of the second-moment spread.
    ///
    /// Zero for a finite horizon, a static equilibrium, and a continuous-time
    /// path. The stationary target carries the variance implied by the
    /// dynamics' innovation variance.
    #[must_use = "the spread separates a stationary law from a fixed point"]
    pub const fn spread_bits(self) -> u64 {
        self.spread_bits
    }

    /// Action the response was evaluated under.
    #[must_use = "the action separates a pulse from a sustained window"]
    pub const fn action(self) -> DelayedAction {
        self.action
    }
}

/// How the ancestry search was bounded.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FeedbackSearch {
    /// Ancestry fit in the declared cap. The cap is that declaration.
    Identified {
        /// Cap that was given.
        cap: FeedbackSearchCap,
    },
    /// No caller cap was declared. Nothing was widened to a larger cap.
    PlannerDefault,
}

/// History, initial law, and one transition, bound to a delayed action.
///
/// Construction admits only a finite-horizon frame. [`Self::response`] can
/// evaluate another target of the same coefficients; that result names its
/// own target and is not this frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DelayedFeedback {
    history: HistoryConstruction,
    transition: Transition,
    delay: Option<FeedbackDelay>,
    action: DelayedAction,
    search: FeedbackSearch,
    information: Option<AvailableOutcome>,
    steps: u32,
}

/// Inputs of [`DelayedFeedback::try_new`].
///
/// `search_cap` and `ancestry_steps` are the declared bound and the depth the
/// caller walked. When the depth exceeds the cap, construction refuses and
/// leaves the cap unchanged.
#[derive(Clone, Debug)]
pub struct DelayedFeedbackRequest {
    /// Observation cadence.
    pub cadence: TimeIndex,
    /// Exclusive horizon end.
    pub horizon: Horizon,
    /// Initial-state law bound to the frame.
    pub initial_law: InitialLaw,
    /// History window. An explicit cap shorter than the delay needs a certificate.
    pub history: HistoryWindow,
    /// Delay. [`None`] is a finite horizon with contemporaneous transmission.
    pub delay: Option<FeedbackDelay>,
    /// Intervention applied on [`Self::temporal`]. A set is not a shift.
    pub intervention: Intervention,
    /// Pulse or sustained timing of [`Self::intervention`].
    pub temporal: TemporalPolicy,
    /// Replacement of the initial-state law. Absence does not add one.
    /// Presence is refused and is not executed as the action.
    pub initial_state_change: Option<InitialStateChange>,
    /// Certificate for a history shorter than the delay. Absence refuses that history.
    pub truncation: Option<TruncationCertificate>,
    /// Declared ancestry cap. [`None`] is the planner default, not a raised cap.
    pub search_cap: Option<FeedbackSearchCap>,
    /// Ancestry depth the search walked. Compared with [`Self::search_cap`].
    pub ancestry_steps: u32,
}

impl DelayedFeedback {
    /// Bind history, the initial law, and one transition on a finite horizon.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`].
    /// [`HistoryOperatorError::UncertifiedTruncation`] when an explicit history
    /// is shorter than the delay or the action lag and `truncation` does not
    /// name that retained depth. The history window is not lengthened.
    /// [`HistoryOperatorError::SearchNotIdentified`] when `ancestry_steps`
    /// exceeds `search_cap`. The cap is not replaced by the ancestry depth.
    /// [`HistoryOperatorError::ActionOutsideHorizon`] when an action step is
    /// at or past the horizon end.
    /// [`HistoryOperatorError::InformationRefused`] when the delayed outcome of
    /// the last intervened step arrives after the last included tick. Arrival
    /// is `until_step + delay`; a window that arrives before the origin is
    /// admitted at the origin.
    /// [`HistoryOperatorError::HorizonStepsUncountable`] when the horizon
    /// length does not fit in `u32` steps.
    #[must_use = "a refused delayed feedback is not a finite-horizon frame"]
    pub fn try_new(request: DelayedFeedbackRequest) -> Result<Self, HistoryOperatorError> {
        let DelayedFeedbackRequest {
            cadence,
            horizon,
            initial_law,
            history,
            delay,
            intervention,
            temporal,
            initial_state_change,
            truncation,
            search_cap,
            ancestry_steps,
        } = request;
        if let Some(change) = &initial_state_change {
            refuse_initial_state_change(&change.law_name, change.level)?;
        }
        let action = delayed_action_of(&intervention, &temporal)?;
        let history =
            HistoryConstruction::try_new(cadence.clone(), horizon, initial_law.clone(), history)?;
        let transition = Transition::try_new(cadence, horizon, initial_law)?;
        let steps = horizon_steps(&history)?;
        admit_action_window(action, steps)?;
        admit_truncation(history.history(), delay, action, truncation)?;
        let search = admit_search(search_cap, ancestry_steps)?;
        let information = admit_feedback_information(
            history.frame().cadence(),
            steps,
            delay.map_or(0, FeedbackDelay::ticks),
            action,
        )?;
        Ok(Self { history, transition, delay, action, search, information, steps })
    }

    /// History operator. Its window is the one that was declared.
    #[must_use = "the history is the window that was declared"]
    pub const fn history(&self) -> &HistoryConstruction {
        &self.history
    }

    /// Transition operator. Its step is one cadence interval.
    #[must_use = "the transition is one cadence step of the delayed system"]
    pub const fn transition(&self) -> &Transition {
        &self.transition
    }

    /// Initial-state law on the shared frame.
    #[must_use = "the initial law is the state before the first transition"]
    pub const fn initial_law(&self) -> &InitialLaw {
        self.history.frame().initial_law()
    }

    /// Delay, or [`None`] when transmission is contemporaneous.
    #[must_use = "the delay is the lag between the action and the state"]
    pub const fn delay(&self) -> Option<FeedbackDelay> {
        self.delay
    }

    /// Action on this frame.
    #[must_use = "the action separates a pulse from a sustained window"]
    pub const fn action(&self) -> DelayedAction {
        self.action
    }

    /// Exclusive horizon length, in cadence steps.
    #[must_use = "the step count is the finite horizon"]
    pub const fn steps(&self) -> u32 {
        self.steps
    }

    /// Search bound. [`FeedbackSearch::Identified`] carries the declared cap.
    #[must_use = "the search records the cap that was given"]
    pub const fn search(&self) -> FeedbackSearch {
        self.search
    }

    /// This frame is a finite-horizon target.
    #[must_use = "a delayed frame is a finite horizon, not an equilibrium"]
    pub const fn target(&self) -> ResponseTarget {
        ResponseTarget::FiniteHorizon
    }

    /// Information-time admission of the delayed outcome, when the action step
    /// is inside the observed cadence and the outcome has arrived.
    #[must_use = "availability is the information-time admission of the delay"]
    pub const fn information(&self) -> Option<AvailableOutcome> {
        self.information
    }

    /// Evaluate `target` on these coefficients.
    ///
    /// The returned [`TargetResponse::target`] is `target`. A finite-horizon
    /// result applies [`Self::transition`] once per cadence step, starting from
    /// [`DelayedDynamics::initial_state`], which is the level of the frame's
    /// [`Self::initial_law`]. The other three targets use different maps and
    /// are not that rollout.
    ///
    /// # Errors
    ///
    /// [`HistoryOperatorError::InitialLawMismatch`] when the dynamics were
    /// declared for a different initial law than the frame binds.
    /// [`HistoryOperatorError::NoFixedPoint`] for a static equilibrium or a
    /// stationary mean when the autoregression is one.
    /// [`HistoryOperatorError::UnstableAutoregression`] when `|a| >= 1` and
    /// not one, for those same two targets.
    /// [`HistoryOperatorError::NonFiniteLevel`] when a computed level is not
    /// finite.
    #[must_use = "a refused target is not a response of the delayed system"]
    pub fn response(
        &self,
        target: ResponseTarget,
        dynamics: &DelayedDynamics,
    ) -> Result<TargetResponse, HistoryOperatorError> {
        if dynamics.initial_law() != self.initial_law() {
            return Err(HistoryOperatorError::InitialLawMismatch {
                frame: self.initial_law().clone(),
                dynamics: dynamics.initial_law().clone(),
            });
        }
        let LevelSpread { level, spread } = match target {
            ResponseTarget::FiniteHorizon => {
                LevelSpread { level: finite_horizon_level(self, dynamics)?, spread: 0.0 }
            }
            ResponseTarget::StaticEquilibrium => LevelSpread {
                level: fixed_point(dynamics, self.action.permanent_level())?,
                spread: 0.0,
            },
            ResponseTarget::Stationary => stationary_law(dynamics, self.action.permanent_level())?,
            ResponseTarget::ContinuousTime => {
                LevelSpread { level: continuous_level(self, dynamics)?, spread: 0.0 }
            }
        };
        Ok(TargetResponse {
            target,
            level_bits: finite_bits(level)?,
            spread_bits: finite_bits(spread)?,
            action: self.action,
        })
    }
}

/// Horizon end, in cadence steps, for a response whose step count is `steps`.
///
/// An irregular cadence and a zero interval have no step length, so they are
/// refused here rather than extended onto a tick.
///
/// # Errors
///
/// [`HistoryOperatorError::CadenceWithoutInterval`] when the cadence is
/// irregular or its interval is zero.
/// [`HistoryOperatorError::HorizonEndOverflow`] when `steps` times the regular
/// interval does not fit in `u64`.
#[must_use = "an overflowing horizon is not a tick the frame can admit"]
pub fn horizon_ending_after(
    cadence: &TimeIndex,
    steps: u32,
) -> Result<Horizon, HistoryOperatorError> {
    let SamplingRegularity::Regular { interval_ns } = cadence.regularity else {
        return Err(HistoryOperatorError::CadenceWithoutInterval { interval_ns: None, steps });
    };
    if interval_ns == 0 {
        return Err(HistoryOperatorError::CadenceWithoutInterval { interval_ns: Some(0), steps });
    }
    let Some(end_ns) = u64::from(steps).checked_mul(interval_ns) else {
        return Err(HistoryOperatorError::HorizonEndOverflow { interval_ns, steps });
    };
    Ok(Horizon::ending_at(end_ns))
}

fn admit_horizon(cadence: &TimeIndex, horizon: Horizon) -> Result<u64, HistoryOperatorError> {
    let end_ns = horizon.end_ns();
    let grid = match ObservedCadence::from_observed(cadence) {
        Ok(grid) => grid,
        Err(InformationRefusal::IrregularCadence) => return Err(crosses(end_ns, None, None)),
        Err(InformationRefusal::ZeroInterval) => {
            return Err(crosses(end_ns, Some(0), u64::try_from(cadence.length).ok()));
        }
        Err(InformationRefusal::EmptyCadence) => {
            let interval_ns = match cadence.regularity {
                SamplingRegularity::Regular { interval_ns } => Some(interval_ns),
                SamplingRegularity::Irregular => None,
            };
            return Err(crosses(end_ns, interval_ns, Some(0)));
        }
        Err(InformationRefusal::CadenceUncountable) => {
            return Err(HistoryOperatorError::CadenceUncountable);
        }
        Err(InformationRefusal::CadenceSpanOverflow { interval_ns, tick_count }) => {
            return Err(HistoryOperatorError::CadenceSpanOverflow {
                interval_ns,
                ticks: u64::from(tick_count),
            });
        }
        Err(
            InformationRefusal::DecisionOffCadence { .. }
            | InformationRefusal::OutcomeOffCadence { .. }
            | InformationRefusal::ProspectiveFuture { .. }
            | InformationRefusal::HeldOutFuture { .. }
            | InformationRefusal::RetrospectiveDenied { .. }
            | InformationRefusal::ForwardOnlyDeployment { .. }
            | InformationRefusal::DelayOverflow { .. },
        ) => return Err(HistoryOperatorError::CadenceUncountable),
    };
    let interval_ns = grid.interval_ns();
    let ticks = u64::from(grid.tick_count());
    let Some(covered) = ticks.checked_mul(interval_ns) else {
        return Err(HistoryOperatorError::CadenceSpanOverflow { interval_ns, ticks });
    };
    if end_ns == 0 || end_ns % interval_ns != 0 || end_ns > covered {
        return Err(crosses(end_ns, Some(interval_ns), Some(ticks)));
    }
    Ok(interval_ns)
}

fn crosses(
    end_ns: u64,
    interval_ns: Option<u64>,
    declared_ticks: Option<u64>,
) -> HistoryOperatorError {
    HistoryOperatorError::HorizonCrossesCadence { end_ns, interval_ns, declared_ticks }
}

fn admit_action(frame: &OperatorFrame, at_ns: u64) -> Result<(), HistoryOperatorError> {
    let interval_ns = frame.interval_ns();
    if at_ns % interval_ns != 0 || at_ns >= frame.horizon().end_ns() {
        return Err(HistoryOperatorError::ActionMissesCadence { at_ns, interval_ns });
    }
    Ok(())
}

fn admit_weight(role: WeightRole, value: f64) -> Result<(), HistoryOperatorError> {
    if !value.is_finite() {
        return Err(HistoryOperatorError::NonFiniteWeight { role, value });
    }
    if value <= 0.0 {
        return Err(HistoryOperatorError::NonPositiveWeight { role, value });
    }
    Ok(())
}

fn admit_name(field: &'static str, name: Arc<str>) -> Result<Arc<str>, HistoryOperatorError> {
    if is_blank(&name) { Err(HistoryOperatorError::EmptyName { field }) } else { Ok(name) }
}

const fn finite_bits(value: f64) -> Result<u64, HistoryOperatorError> {
    if !value.is_finite() {
        return Err(HistoryOperatorError::NonFiniteLevel { value });
    }
    let bits = value.to_bits();
    if bits == (-0.0f64).to_bits() { Ok(0.0f64.to_bits()) } else { Ok(bits) }
}

fn horizon_steps(history: &HistoryConstruction) -> Result<u32, HistoryOperatorError> {
    let steps = history.frame().horizon().end_ns() / history.frame().interval_ns();
    u32::try_from(steps).map_err(|_| HistoryOperatorError::HorizonStepsUncountable)
}

const fn action_bounds(action: DelayedAction) -> (i32, i32) {
    match action {
        DelayedAction::Pulse { at_step, .. } => (at_step, at_step),
        DelayedAction::Sustained { from_step, until_step, .. } => (from_step, until_step),
    }
}

fn admit_action_window(action: DelayedAction, steps: u32) -> Result<(), HistoryOperatorError> {
    let Ok(horizon_steps) = i32::try_from(steps) else {
        return Err(HistoryOperatorError::HorizonStepsUncountable);
    };
    let (from_step, until_step) = action_bounds(action);
    if until_step >= horizon_steps || from_step > until_step {
        return Err(HistoryOperatorError::ActionOutsideHorizon {
            step: until_step,
            horizon_steps: steps,
        });
    }
    Ok(())
}

const fn required_history_ticks(delay: Option<FeedbackDelay>, action: DelayedAction) -> u32 {
    let delay_ticks = match delay {
        Some(delay) => delay.ticks(),
        None => 0,
    };
    let (from_step, _) = action_bounds(action);
    let action_lag = if from_step < 0 { from_step.unsigned_abs() } else { 0 };
    if delay_ticks > action_lag { delay_ticks } else { action_lag }
}

fn admit_truncation(
    history: HistoryWindow,
    delay: Option<FeedbackDelay>,
    action: DelayedAction,
    truncation: Option<TruncationCertificate>,
) -> Result<(), HistoryOperatorError> {
    let Some(retained_ticks) = history.max_lag_steps() else {
        return Ok(());
    };
    let required_ticks = required_history_ticks(delay, action);
    if retained_ticks >= required_ticks {
        return Ok(());
    }
    let certified =
        truncation.is_some_and(|certificate| certificate.retained_ticks() == retained_ticks);
    if certified {
        return Ok(());
    }
    Err(HistoryOperatorError::UncertifiedTruncation { retained_ticks, required_ticks })
}

fn admit_search(
    search_cap: Option<FeedbackSearchCap>,
    ancestry_steps: u32,
) -> Result<FeedbackSearch, HistoryOperatorError> {
    let Some(cap) = search_cap else {
        return Ok(FeedbackSearch::PlannerDefault);
    };
    if ancestry_steps > cap.steps() {
        return Err(HistoryOperatorError::SearchNotIdentified { cap: cap.steps(), ancestry_steps });
    }
    Ok(FeedbackSearch::Identified { cap })
}

fn admit_feedback_information(
    cadence: &TimeIndex,
    steps: u32,
    delay_ticks: u32,
    action: DelayedAction,
) -> Result<Option<AvailableOutcome>, HistoryOperatorError> {
    if delay_ticks == 0 {
        return Ok(None);
    }
    let (_, until_step) = action_bounds(action);
    // The last intervened step arrives last, so it binds the admission. An
    // arrival before the origin is already in the history and is judged at the
    // origin tick.
    let arrival = (i64::from(until_step) + i64::from(delay_ticks)).max(0);
    let arrival =
        u32::try_from(arrival).map_err(|_| HistoryOperatorError::HorizonStepsUncountable)?;
    let decision = steps.checked_sub(1).ok_or(HistoryOperatorError::HorizonStepsUncountable)?;
    match admit_delayed_outcome(
        cadence,
        CadenceTick::new(decision),
        CadenceTick::new(arrival),
        0,
        InformationUse::Prospective,
        RetrospectivePermission::Denied,
        ForwardOnly::Enforced,
    ) {
        Ok(available) => Ok(Some(available)),
        Err(refusal) => Err(HistoryOperatorError::InformationRefused { refusal }),
    }
}

/// Level and second-moment spread of one evaluated target.
struct LevelSpread {
    level: f64,
    spread: f64,
}

fn finite_horizon_level(
    feedback: &DelayedFeedback,
    dynamics: &DelayedDynamics,
) -> Result<f64, HistoryOperatorError> {
    let delay = feedback.delay.map_or(0, FeedbackDelay::ticks);
    let mut state = dynamics.initial_state();
    let autoregression = dynamics.autoregression();
    let input_gain = dynamics.input_gain();
    for step in 0..feedback.steps {
        let source = i64::from(step) - i64::from(delay);
        let Ok(source) = i32::try_from(source) else {
            return Err(HistoryOperatorError::HorizonStepsUncountable);
        };
        let level = feedback.action.level_at(source);
        state = autoregression * state + input_gain * level;
        if !state.is_finite() {
            return Err(HistoryOperatorError::NonFiniteLevel { value: state });
        }
    }
    Ok(state)
}

fn admit_stable(dynamics: &DelayedDynamics) -> Result<(), HistoryOperatorError> {
    let autoregression = dynamics.autoregression();
    if autoregression.abs() < 1.0 {
        return Ok(());
    }
    if autoregression.to_bits() == 1.0f64.to_bits() {
        return Err(HistoryOperatorError::NoFixedPoint);
    }
    Err(HistoryOperatorError::UnstableAutoregression { autoregression })
}

fn fixed_point(dynamics: &DelayedDynamics, permanent: f64) -> Result<f64, HistoryOperatorError> {
    admit_stable(dynamics)?;
    let level = dynamics.input_gain() * permanent / (1.0 - dynamics.autoregression());
    if !level.is_finite() {
        return Err(HistoryOperatorError::NonFiniteLevel { value: level });
    }
    Ok(level)
}

fn stationary_law(
    dynamics: &DelayedDynamics,
    permanent: f64,
) -> Result<LevelSpread, HistoryOperatorError> {
    let level = fixed_point(dynamics, permanent)?;
    let autoregression = dynamics.autoregression();
    let spread = dynamics.innovation_variance() / (1.0 - autoregression * autoregression);
    if !spread.is_finite() {
        return Err(HistoryOperatorError::NonFiniteLevel { value: spread });
    }
    Ok(LevelSpread { level, spread })
}

fn continuous_level(
    feedback: &DelayedFeedback,
    dynamics: &DelayedDynamics,
) -> Result<f64, HistoryOperatorError> {
    let delay_ticks = feedback.delay.map_or(0, FeedbackDelay::ticks);
    let decay = continuous_decay(dynamics, f64::from(feedback.steps));
    let driven = match feedback.action {
        DelayedAction::Pulse { at_step, level_bits } => continuous_pulse(
            dynamics,
            f64::from_bits(level_bits),
            at_step,
            delay_ticks,
            feedback.steps,
        ),
        DelayedAction::Sustained { from_step, until_step, level_bits } => continuous_sustained(
            dynamics,
            f64::from_bits(level_bits),
            (from_step, until_step),
            delay_ticks,
            feedback.steps,
        ),
    };
    let level = dynamics.initial_state() * decay + driven;
    if !level.is_finite() {
        return Err(HistoryOperatorError::NonFiniteLevel { value: level });
    }
    Ok(level)
}

/// Decay of the initial state over `duration` steps, `exp((a - 1) duration)`.
fn continuous_decay(dynamics: &DelayedDynamics, duration: f64) -> f64 {
    ((dynamics.autoregression() - 1.0) * duration).exp()
}

/// Convolution of the held action with the continuous kernel.
///
/// Only the sustained window, shifted by the delay and cut to the horizon,
/// drives the state. A window that arrives at or after the horizon, or ends
/// before the origin, contributes nothing.
fn continuous_sustained(
    dynamics: &DelayedDynamics,
    level: f64,
    (from_step, until_step): (i32, i32),
    delay_ticks: u32,
    steps: u32,
) -> f64 {
    let horizon = f64::from(steps);
    let shift = f64::from(delay_ticks);
    let start = (f64::from(from_step) + shift).max(0.0);
    let end = (f64::from(until_step) + 1.0 + shift).min(horizon);
    if end <= start {
        return 0.0;
    }
    let gain = dynamics.input_gain() * level;
    let drift = dynamics.autoregression() - 1.0;
    if drift.to_bits() == 0.0f64.to_bits() {
        return gain * (end - start);
    }
    gain * ((drift * (horizon - start)).exp() - (drift * (horizon - end)).exp()) / drift
}

fn continuous_pulse(
    dynamics: &DelayedDynamics,
    level: f64,
    at_step: i32,
    delay_ticks: u32,
    steps: u32,
) -> f64 {
    let arrival = i64::from(at_step) + i64::from(delay_ticks);
    let Ok(arrival) = i32::try_from(arrival) else {
        return 0.0;
    };
    let Ok(steps_i) = i32::try_from(steps) else {
        return 0.0;
    };
    if arrival < 0 || arrival >= steps_i {
        return 0.0;
    }
    let elapsed = f64::from(steps) - f64::from(arrival);
    dynamics.input_gain() * level * continuous_decay(dynamics, elapsed)
}

/// The one multi-variable lag-aligned scan of a series.
///
/// A scan plans the windows `[t - max_lag, t]` over any set of lagged columns and gathers them
/// into reusable storage in one of the three window disciplines the temporal routes need:
/// every window must be intact ([`Self::complete`]), incomplete windows are kept in place and
/// flagged ([`Self::with_gaps`]), or incomplete windows are removed ([`Self::windows`] with
/// `gaps`). Every route engine reads its windows through this operator, so history
/// construction has one implementation; [`HistoryConstruction::scan_columns`] binds it to a
/// declared frame.
#[derive(Debug)]
pub struct HistoryScan {
    plan: antecedent_data::LaggedSamplePlan,
    workspace: antecedent_data::LaggedSampleWorkspace,
}

impl HistoryScan {
    /// Plan the windows of `columns` over `data`, `max_lag` lags deep.
    ///
    /// # Errors
    ///
    /// The refusals of [`antecedent_data::LaggedSamplePlan::new`].
    pub fn plan(
        data: &antecedent_data::TimeSeriesData,
        max_lag: u32,
        columns: impl Into<Arc<[antecedent_data::LaggedColumn]>>,
    ) -> Result<Self, antecedent_data::DataError> {
        Ok(Self {
            plan: data.plan_lagged_sample(max_lag, columns)?,
            workspace: antecedent_data::LaggedSampleWorkspace::default(),
        })
    }

    /// The planned windows.
    #[must_use]
    pub const fn lag_plan(&self) -> &antecedent_data::LaggedSamplePlan {
        &self.plan
    }

    /// Gather the windows; every window must be intact.
    ///
    /// # Errors
    ///
    /// Missing or non-float64 columns, a length mismatch, or missing values.
    pub fn complete(
        &mut self,
        data: &antecedent_data::TimeSeriesData,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<antecedent_data::LaggedPreparedSample<'_>, antecedent_data::DataError> {
        self.plan.prepare(data, &mut self.workspace, policy)
    }

    /// Gather the windows keeping the incomplete ones in place; `complete[i]` flags window
    /// `i`.
    ///
    /// # Errors
    ///
    /// Missing or non-float64 columns, a length mismatch, or a row-hiding analysis mask.
    pub fn with_gaps(
        &mut self,
        data: &antecedent_data::TimeSeriesData,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<antecedent_data::GappedSample<'_>, antecedent_data::DataError> {
        self.plan.prepare_with_gaps(data, &mut self.workspace, policy)
    }

    /// Gather the windows, removing the incomplete ones when `gaps` is set. The flags are
    /// `Some` exactly when `gaps` is set, one per planned window.
    ///
    /// # Errors
    ///
    /// As [`Self::complete`] and [`Self::with_gaps`].
    pub fn windows(
        &mut self,
        data: &antecedent_data::TimeSeriesData,
        policy: &antecedent_core::KernelPolicy,
        gaps: bool,
    ) -> Result<
        (antecedent_data::LaggedPreparedSample<'_>, Option<Vec<bool>>),
        antecedent_data::DataError,
    > {
        if gaps {
            let (sample, complete) =
                self.plan.prepare_complete_windows(data, &mut self.workspace, policy)?;
            Ok((sample, Some(complete)))
        } else {
            Ok((self.plan.prepare(data, &mut self.workspace, policy)?, None))
        }
    }
}

impl HistoryConstruction {
    /// Plan the multi-variable scan of `columns` on this frame.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::HistoryShorterThanColumns`] when the declared history cap is
    /// shorter than the deepest column (the history is not extended), or
    /// [`HistoryExecutionError::Scan`] when the plan is refused.
    pub fn scan_columns(
        &self,
        data: &antecedent_data::TimeSeriesData,
        columns: impl Into<Arc<[antecedent_data::LaggedColumn]>>,
    ) -> Result<HistoryScan, HistoryExecutionError> {
        let columns: Arc<[antecedent_data::LaggedColumn]> = columns.into();
        let needed = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
        if let Some(cap) = self.history.resolved_cap() {
            if needed > cap {
                return Err(HistoryExecutionError::HistoryShorterThanColumns { cap, needed });
            }
        }
        HistoryScan::plan(data, needed, columns)
            .map_err(|error| HistoryExecutionError::Scan { message: error.to_string() })
    }
}

// ---------------------------------------------------------------------------
// Execution
//
// The operators above are values over one frame. The code below runs them: one
// scan of the series builds the history windows every consumer shares, the
// sequential regression is fit on those windows under the row weights, the
// transition steps the fitted mechanism one tick, the policy is evaluated on the
// rolled history, and the reduction folds the trajectory. Every consumer of a
// temporal program reads these same operators instead of its own scan.
// ---------------------------------------------------------------------------

/// Refusal while executing compiled operators on a series.
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum HistoryExecutionError {
    /// The outcome and action series have different lengths.
    #[error(
        "outcome has {outcome} ticks and action has {action}; next attempt: align both on the \
         cadence"
    )]
    SeriesLengthMismatch {
        /// Outcome length.
        outcome: usize,
        /// Action length.
        action: usize,
    },
    /// A value is not finite.
    #[error("the value at tick {tick} is not finite; next attempt: fill or drop the tick")]
    NonFiniteValue {
        /// Tick of the first offending value.
        tick: usize,
    },
    /// The history cap is the planner default and has no number.
    #[error(
        "the history cap is an unresolved planner default; next attempt: declare or resolve it"
    )]
    HistoryCapUnresolved,
    /// A history of zero ticks carries no lagged state to roll forward.
    #[error("a history of zero ticks has no state to scan; next attempt: retain at least one lag")]
    HistoryDepthZero,
    /// The series is too short for the regression the history declares.
    #[error(
        "a history of {depth} ticks fits {columns} coefficients and needs {needed} ticks; the \
         series has {got}; next attempt: supply a longer series or a shorter history"
    )]
    SeriesTooShort {
        /// Retained lags.
        depth: u32,
        /// Coefficients of the regression.
        columns: usize,
        /// Ticks needed.
        needed: usize,
        /// Ticks supplied.
        got: usize,
    },
    /// Two operators of one program were compiled over different frames.
    #[error(
        "{operator} was compiled over another cadence, horizon or initial law; next attempt: \
         compile every operator over one frame"
    )]
    FrameMismatch {
        /// Operator whose frame differs from the history's.
        operator: &'static str,
    },
    /// A row weight or probability is outside its range.
    #[error("{role} at tick {tick} is {value}; next attempt: supply a probability in (0, 1]")]
    ProbabilityOutOfRange {
        /// Which weight the probability belongs to.
        role: WeightRole,
        /// Window index.
        tick: usize,
        /// The refused value.
        value: f64,
    },
    /// The weight vector does not have one entry per window.
    #[error("{got} row weights for {windows} windows; next attempt: weight every window")]
    WeightShape {
        /// Windows scanned.
        windows: usize,
        /// Weights supplied.
        got: usize,
    },
    /// The weighted normal equations are singular.
    #[error(
        "the history windows do not identify the transition (singular weighted normal \
         equations); next attempt: vary the action and outcome history or shorten it"
    )]
    SingularHistory,
    /// A sampling weight is negative or not finite.
    #[error(
        "sampling weight {value} at window {window} is not a finite non-negative weight; next \
         attempt: declare weights of at least zero"
    )]
    SamplingWeightInvalid {
        /// Window of the weight.
        window: usize,
        /// The refused value.
        value: f64,
    },
    /// No window carries a positive sampling weight.
    #[error(
        "no lag window has a positive sampling weight; the weighted population is empty; next \
         attempt: give at least one window a positive weight"
    )]
    NoPositiveSamplingWeight,
    /// A named initial law needs the state it names.
    #[error("initial law `{law}` is named, not observed; next attempt: supply its state")]
    InitialStateRequired {
        /// The law's name.
        law: String,
    },
    /// The supplied initial state does not have one value per retained lag.
    #[error("initial state has {outcomes} outcomes and {actions} actions for {depth} lags")]
    InitialStateShape {
        /// Outcomes supplied.
        outcomes: usize,
        /// Actions supplied.
        actions: usize,
        /// Retained lags.
        depth: u32,
    },
    /// A level is not finite.
    #[error("policy level {value} is not finite; next attempt: supply a finite level")]
    NonFinitePolicyLevel {
        /// The refused level.
        value: f64,
    },
    /// The reduction functional is not one this operator evaluates.
    #[error(
        "functional `{name}` is not a reduction this operator evaluates; next attempt: use \
         final, mean, sum or maximum"
    )]
    UnknownFunctional {
        /// The refused name.
        name: String,
    },
    /// A scanned column reaches further back than the declared history.
    #[error(
        "a column lags {needed} ticks but the history keeps {cap}; next attempt: declare a \
         history of at least {needed} ticks (the history is not extended)"
    )]
    HistoryShorterThanColumns {
        /// Declared history, in ticks.
        cap: u32,
        /// Deepest lag scanned.
        needed: u32,
    },
    /// The scan plan was refused.
    #[error("the lag-aligned scan was refused: {message}")]
    Scan {
        /// The data layer's refusal.
        message: String,
    },
    /// The horizon has more ticks than a step counter holds.
    #[error("the horizon has more ticks than can be rolled; next attempt: shorten it")]
    HorizonUncountable,
    /// The rolled trajectory left the finite range.
    #[error(
        "the rolled trajectory is not finite at step {step}; the fitted transition is unstable"
    )]
    UnstableRollout {
        /// Step of the first non-finite value.
        step: u32,
    },
}

/// Outcome and action observed on the ticks of one cadence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickSeries<'a> {
    outcome: &'a [f64],
    action: &'a [f64],
}

impl<'a> TickSeries<'a> {
    /// Series with one outcome and one action per tick.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::SeriesLengthMismatch`] or
    /// [`HistoryExecutionError::NonFiniteValue`].
    #[must_use = "a refused series is not an input"]
    pub fn try_new(outcome: &'a [f64], action: &'a [f64]) -> Result<Self, HistoryExecutionError> {
        if outcome.len() != action.len() {
            return Err(HistoryExecutionError::SeriesLengthMismatch {
                outcome: outcome.len(),
                action: action.len(),
            });
        }
        if let Some(tick) =
            outcome.iter().zip(action).position(|(y, a)| !y.is_finite() || !a.is_finite())
        {
            return Err(HistoryExecutionError::NonFiniteValue { tick });
        }
        Ok(Self { outcome, action })
    }

    /// Ticks.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.outcome.len()
    }

    /// Whether the series has no tick.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.outcome.is_empty()
    }
}

/// The windows one scan of a series produced.
///
/// Row `r` is the window ending at series tick `first_tick + r`:
/// `[1, y_{t-1}, …, y_{t-d}, a_t, a_{t-1}, …, a_{t-d}]` with target `y_t`, `d` the
/// retained lags.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryWindows {
    depth: u32,
    first_tick: usize,
    columns: usize,
    design: Vec<f64>,
    target: Vec<f64>,
}

impl HistoryWindows {
    /// Retained lags.
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    /// Series tick of the first window.
    #[must_use]
    pub const fn first_tick(&self) -> usize {
        self.first_tick
    }

    /// Windows scanned.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.target.len()
    }

    /// Coefficients of the regression the windows feed.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Row-major `rows × columns` design.
    #[must_use]
    pub fn design(&self) -> &[f64] {
        &self.design
    }

    /// Outcome of each window.
    #[must_use]
    pub fn target(&self) -> &[f64] {
        &self.target
    }
}

impl HistoryConstruction {
    /// Scan `series` once into the windows every consumer of the frame shares.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::HistoryCapUnresolved`],
    /// [`HistoryExecutionError::HistoryDepthZero`] or
    /// [`HistoryExecutionError::SeriesTooShort`].
    pub fn scan(&self, series: &TickSeries<'_>) -> Result<HistoryWindows, HistoryExecutionError> {
        let depth =
            self.history.resolved_cap().ok_or(HistoryExecutionError::HistoryCapUnresolved)?;
        if depth == 0 {
            return Err(HistoryExecutionError::HistoryDepthZero);
        }
        let d = depth as usize;
        let columns = 2 * d + 2;
        let needed = d + columns + 1;
        if series.len() < needed {
            return Err(HistoryExecutionError::SeriesTooShort {
                depth,
                columns,
                needed,
                got: series.len(),
            });
        }
        let rows = series.len() - d;
        let mut design = Vec::with_capacity(rows * columns);
        let mut target = Vec::with_capacity(rows);
        for t in d..series.len() {
            design.push(1.0);
            for lag in 1..=d {
                design.push(series.outcome[t - lag]);
            }
            for lag in 0..=d {
                design.push(series.action[t - lag]);
            }
            target.push(series.outcome[t]);
        }
        Ok(HistoryWindows { depth, first_tick: d, columns, design, target })
    }
}

/// The fitted one-tick mechanism of a [`SequentialRegression`].
///
/// Coefficients in the order of the windows' columns: intercept, outcome lags
/// `1..=d`, then actions at lags `0..=d`.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionFit {
    depth: u32,
    coefficients: Vec<f64>,
    residual_variance: f64,
    rows: usize,
}

impl TransitionFit {
    /// Retained lags.
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.depth
    }

    /// Fitted coefficients.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Weighted residual variance.
    #[must_use]
    pub const fn residual_variance(&self) -> f64 {
        self.residual_variance
    }

    /// Windows the fit used.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Weighted contemporaneous action coefficient.
    #[must_use]
    pub fn action_effect(&self, lag: u32) -> Option<f64> {
        (lag <= self.depth).then(|| self.coefficients[1 + self.depth as usize + lag as usize])
    }
}

impl SequentialRegression {
    /// Fit the one-tick mechanism on `windows` by weighted least squares.
    ///
    /// `weights` is one non-negative weight per window (see
    /// [`CensoringAssignmentWeight::row_weights`]); `None` weights every window equally.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::WeightShape`] or
    /// [`HistoryExecutionError::SingularHistory`].
    pub fn fit(
        &self,
        windows: &HistoryWindows,
        weights: Option<&[f64]>,
    ) -> Result<TransitionFit, HistoryExecutionError> {
        let (rows, p) = (windows.rows(), windows.columns());
        let columns: Vec<Vec<f64>> =
            (0..p).map(|j| (0..rows).map(|r| windows.design[r * p + j]).collect()).collect();
        let views: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let fit = self.fit_columns(&views, &windows.target, weights)?;
        Ok(TransitionFit {
            depth: windows.depth,
            coefficients: fit.coefficients,
            residual_variance: fit.residual_variance,
            rows,
        })
    }

    /// Fit the regression of `target` on any set of aligned `columns` (any number of
    /// variables and lags, as scanned by [`HistoryScan`]) by weighted least squares.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::WeightShape`] when a column, the target or the weights
    /// disagree in length, or [`HistoryExecutionError::SingularHistory`].
    pub fn fit_columns(
        &self,
        columns: &[&[f64]],
        target: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<ColumnFit, HistoryExecutionError> {
        let (rows, p) = (target.len(), columns.len());
        if columns.iter().any(|c| c.len() != rows) || weights.is_some_and(|w| w.len() != rows) {
            return Err(HistoryExecutionError::WeightShape {
                windows: rows,
                got: weights.map_or(columns.first().map_or(0, |c| c.len()), <[f64]>::len),
            });
        }
        let mut design = Vec::with_capacity(rows * p);
        for column in columns {
            design.extend_from_slice(column);
        }
        let mut workspace = antecedent_stats::LeastSquaresWorkspace::default();
        let fit = Self::solve_design(&design, rows, p, target, weights, &mut workspace)
            .map_err(|_| HistoryExecutionError::SingularHistory)?;
        let total_weight: f64 = weights.map_or(rows as f64, |w| w.iter().sum());
        let residual_variance =
            fit.rss / (total_weight * (1.0 - p as f64 / rows as f64)).max(f64::MIN_POSITIVE);
        Ok(ColumnFit { coefficients: fit.coefficients, residual_variance, rows })
    }

    /// The one least-squares solve every temporal route engine reads: `min ||W^(1/2)(X b - y)||`
    /// for the column-major design `x_colmajor` (`nrows` by `ncols`).
    ///
    /// `weights` of `None` is the ordinary fit (the backend's column-pivoted QR, bit for bit);
    /// `Some` folds the square root of each weight into the factorization without building a
    /// second design, which is the same fit as scaling every row and the target by the root
    /// weight beforehand. The fit's residuals and `rss` are those of the scaled rows.
    ///
    /// # Errors
    ///
    /// The backend's shape and rank refusals.
    pub fn solve_design(
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
        weights: Option<&[f64]>,
        workspace: &mut antecedent_stats::LeastSquaresWorkspace,
    ) -> Result<antecedent_stats::LeastSquaresFit, antecedent_stats::StatsError> {
        use antecedent_stats::DenseLinearAlgebra;
        let backend = antecedent_stats::FaerBackend;
        match weights {
            None => backend.least_squares(x_colmajor, nrows, ncols, y, workspace),
            Some(weights) => {
                let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
                let scaled: Vec<f64> = y.iter().zip(&root).map(|(y, r)| y * r).collect();
                backend
                    .row_scaled_least_squares(x_colmajor, nrows, ncols, &root, &scaled, workspace)
            }
        }
    }

    /// The coefficients of [`Self::solve_design`] alone, for resampling loops that never read
    /// residuals. The unweighted fit is the backend's coefficient-only path.
    ///
    /// # Errors
    ///
    /// The backend's shape and rank refusals.
    pub fn solve_coefficients(
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
    ) -> Result<Vec<f64>, antecedent_stats::StatsError> {
        antecedent_stats::FaerBackend.least_squares_coefficients(x_colmajor, nrows, ncols, y)
    }
}

/// A weighted least-squares fit of [`SequentialRegression::fit_columns`].
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnFit {
    /// Coefficients, in the order of the columns.
    pub coefficients: Vec<f64>,
    /// Weighted residual variance.
    pub residual_variance: f64,
    /// Rows fitted.
    pub rows: usize,
}

impl CensoringAssignmentWeight {
    /// Inverse-probability row weights `censoring · assignment / (pc_r · pa_r)`.
    ///
    /// The declared censoring and assignment values are the common multipliers of the
    /// weights; weighted least squares is invariant to them, so they fix the scale only.
    /// A probability outside `(0, 1]` is a positivity violation and is refused.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::ProbabilityOutOfRange`] or
    /// [`HistoryExecutionError::WeightShape`].
    pub fn row_weights(
        &self,
        censoring_probability: &[f64],
        assignment_probability: &[f64],
    ) -> Result<Vec<f64>, HistoryExecutionError> {
        if censoring_probability.len() != assignment_probability.len() {
            return Err(HistoryExecutionError::WeightShape {
                windows: censoring_probability.len(),
                got: assignment_probability.len(),
            });
        }
        let check = |role: WeightRole, tick: usize, value: f64| {
            if value.is_finite() && value > 0.0 && value <= 1.0 {
                Ok(value)
            } else {
                Err(HistoryExecutionError::ProbabilityOutOfRange { role, tick, value })
            }
        };
        censoring_probability
            .iter()
            .zip(assignment_probability)
            .enumerate()
            .map(|(tick, (&pc, &pa))| {
                let pc = check(WeightRole::Censoring, tick, pc)?;
                let pa = check(WeightRole::Assignment, tick, pa)?;
                Ok(self.censoring * self.assignment / (pc * pa))
            })
            .collect()
    }
}

/// Sampling weights of a series' lag windows, as the weight operator turns them into row
/// weights.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowWeights {
    /// One weight per window, scaled to mean one over the windows of positive weight. A
    /// window of weight zero keeps its place with weight zero.
    pub weights: Vec<f64>,
    /// Windows with positive weight.
    pub rows_used: usize,
    /// Windows that kept their place with weight zero.
    pub zero_weight_windows: usize,
    /// Sum of the declared weights over the windows used, before normalization.
    pub weight_sum: f64,
    /// Kish effective sample size of the windows used.
    pub effective_sample_size: f64,
}

impl CensoringAssignmentWeight {
    /// Row weights of the windows from the series' declared sampling weights.
    ///
    /// The declared censoring and assignment values multiply every weight, and the result is
    /// scaled to mean one over the windows of positive weight; weighted least squares is
    /// invariant to that common factor, so it cancels in the normalization.
    /// A window of weight zero stays in the design, scaled to nothing, so windows keep their
    /// consecutive times.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::SamplingWeightInvalid`] for a negative or non-finite weight,
    /// or [`HistoryExecutionError::NoPositiveSamplingWeight`].
    pub fn window_weights(&self, sampling: &[f64]) -> Result<WindowWeights, HistoryExecutionError> {
        if let Some((window, &value)) =
            sampling.iter().enumerate().find(|(_, w)| !w.is_finite() || **w < 0.0)
        {
            return Err(HistoryExecutionError::SamplingWeightInvalid { window, value });
        }
        let positive: Vec<f64> = sampling.iter().copied().filter(|w| *w > 0.0).collect();
        if positive.is_empty() {
            return Err(HistoryExecutionError::NoPositiveSamplingWeight);
        }
        let weight_sum: f64 = positive.iter().sum();
        let scale = positive.len() as f64 / weight_sum;
        Ok(WindowWeights {
            weights: sampling.iter().map(|w| w * scale).collect(),
            rows_used: positive.len(),
            zero_weight_windows: sampling.len() - positive.len(),
            weight_sum,
            effective_sample_size: antecedent_kernels::kish_ess(&positive),
        })
    }
}

/// The action rule a [`PolicyEvaluation`] evaluates on the rolled history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolicyRule {
    /// The same level on every tick.
    Hold(f64),
    /// `level` on the action tick only, `otherwise` on every other tick.
    Pulse {
        /// Level at the action tick.
        level: f64,
        /// Level elsewhere.
        otherwise: f64,
    },
    /// `level` from the action tick to the end of the horizon, `otherwise` before it.
    Sustained {
        /// Level from the action tick.
        level: f64,
        /// Level before it.
        otherwise: f64,
    },
    /// From the action tick, `at_or_above` when the previous outcome reached `cutoff` and
    /// `below` when it did not; `otherwise` before the action tick.
    Reactive {
        /// Cutoff on the previous outcome.
        cutoff: f64,
        /// Level when the previous outcome is below the cutoff.
        below: f64,
        /// Level when it is at or above.
        at_or_above: f64,
        /// Level before the action tick.
        otherwise: f64,
    },
}

impl PolicyRule {
    fn levels(self) -> [f64; 4] {
        match self {
            Self::Hold(level) => [level, level, level, level],
            Self::Pulse { level, otherwise } | Self::Sustained { level, otherwise } => {
                [level, otherwise, level, otherwise]
            }
            Self::Reactive { cutoff, below, at_or_above, otherwise } => {
                [cutoff, below, at_or_above, otherwise]
            }
        }
    }

    fn admit(self) -> Result<Self, HistoryExecutionError> {
        match self.levels().iter().find(|v| !v.is_finite()) {
            Some(&value) => Err(HistoryExecutionError::NonFinitePolicyLevel { value }),
            None => Ok(self),
        }
    }

    fn action(self, step: u32, start: u32, previous_outcome: f64) -> f64 {
        match self {
            Self::Hold(level) => level,
            Self::Pulse { level, otherwise } => {
                if step == start {
                    level
                } else {
                    otherwise
                }
            }
            Self::Sustained { level, otherwise } => {
                if step >= start {
                    level
                } else {
                    otherwise
                }
            }
            Self::Reactive { cutoff, below, at_or_above, otherwise } => {
                if step < start {
                    otherwise
                } else if previous_outcome >= cutoff {
                    at_or_above
                } else {
                    below
                }
            }
        }
    }
}

/// State at the analysis origin: the last retained outcomes and actions, oldest first.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialState {
    /// Outcomes at ticks `-d..-1`.
    pub outcomes: Vec<f64>,
    /// Actions at ticks `-d..-1`.
    pub actions: Vec<f64>,
}

/// Functional a [`TrajectoryReduction`] evaluates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionFunctional {
    /// The outcome at the last tick of the horizon.
    Final,
    /// Mean outcome over the horizon.
    Mean,
    /// Sum of the outcomes over the horizon.
    Sum,
    /// Largest outcome over the horizon.
    Maximum,
}

impl ReductionFunctional {
    fn parse(name: &str) -> Result<Self, HistoryExecutionError> {
        match name {
            "final" => Ok(Self::Final),
            "mean" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            "maximum" => Ok(Self::Maximum),
            other => Err(HistoryExecutionError::UnknownFunctional { name: other.to_owned() }),
        }
    }

    fn reduce(self, trajectory: &[f64]) -> f64 {
        match self {
            Self::Final => trajectory.last().copied().unwrap_or(f64::NAN),
            Self::Mean => trajectory.iter().sum::<f64>() / trajectory.len() as f64,
            Self::Sum => trajectory.iter().sum(),
            Self::Maximum => trajectory.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

impl TrajectoryReduction {
    /// The functional this operator evaluates.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::UnknownFunctional`].
    pub fn evaluated(&self) -> Result<ReductionFunctional, HistoryExecutionError> {
        ReductionFunctional::parse(&self.functional)
    }

    /// Reduce `trajectory` to the named functional.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::UnknownFunctional`].
    pub fn reduce(&self, trajectory: &[f64]) -> Result<f64, HistoryExecutionError> {
        Ok(self.evaluated()?.reduce(trajectory))
    }
}

impl Transition {
    /// Mean of the next outcome given the fit, the last `d` outcomes and the last `d` actions
    /// (oldest first) and the action taken now.
    #[must_use]
    pub fn step(fit: &TransitionFit, outcomes: &[f64], actions: &[f64], action_now: f64) -> f64 {
        let d = fit.depth as usize;
        let c = &fit.coefficients;
        let mut value = c[0];
        for lag in 1..=d {
            value += c[lag] * outcomes[outcomes.len() - lag];
        }
        value += c[d + 1] * action_now;
        for lag in 1..=d {
            value += c[d + 1 + lag] * actions[actions.len() - lag];
        }
        value
    }
}

/// Everything one execution of a [`HistoryProgram`] produced.
#[derive(Clone, Debug, PartialEq)]
pub struct TrajectoryExecution {
    /// The fitted mechanism both worlds were rolled through.
    pub fit: TransitionFit,
    /// Windows the one scan produced.
    pub windows: usize,
    /// Mean outcome path under the policy.
    pub policy: Vec<f64>,
    /// Mean outcome path under the reference policy.
    pub reference: Vec<f64>,
    /// The functional of the policy path.
    pub policy_value: f64,
    /// The functional of the reference path.
    pub reference_value: f64,
}

impl TrajectoryExecution {
    /// Policy value minus reference value.
    #[must_use]
    pub fn contrast(&self) -> f64 {
        self.policy_value - self.reference_value
    }
}

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
    history: HistoryConstruction,
    timing: ActionTiming,
    policy: PolicyEvaluation,
    transition: Transition,
    regression: SequentialRegression,
    weights: Option<CensoringAssignmentWeight>,
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
            return Err(HistoryExecutionError::FrameMismatch { operator: *operator });
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
    /// # Errors
    ///
    /// Any [`HistoryExecutionError`].
    pub fn execute(
        &self,
        series: &TickSeries<'_>,
        row_weights: Option<&[f64]>,
        policy: PolicyRule,
        reference: PolicyRule,
        initial: Option<&InitialState>,
    ) -> Result<TrajectoryExecution, HistoryExecutionError> {
        let policy = policy.admit()?;
        let reference = reference.admit()?;
        let windows = self.history.scan(series)?;
        let weights = if self.weights.is_some() { row_weights } else { None };
        let fit = self.regression.fit(&windows, weights)?;
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
        let roll = |rule: PolicyRule| -> Result<Vec<f64>, HistoryExecutionError> {
            let mut outcomes = state.outcomes.clone();
            let mut actions = state.actions.clone();
            let mut path = Vec::with_capacity(steps as usize);
            for step in 0..steps {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn regular(interval_ns: u64, ticks: usize) -> TimeIndex {
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length: ticks }
    }

    fn origin() -> InitialLaw {
        InitialLaw::named("origin").unwrap()
    }

    fn frame_inputs() -> (TimeIndex, Horizon, InitialLaw) {
        (regular(10, 4), Horizon::ending_at(40), origin())
    }

    fn assert_crosses(error: HistoryOperatorError, end_ns: u64) {
        match error {
            HistoryOperatorError::HorizonCrossesCadence { end_ns: got, .. } => {
                assert_eq!(got, end_ns);
            }
            other => panic!("expected a horizon that crosses the cadence, got {other:?}"),
        }
    }

    #[test]
    fn each_operator_records_the_declared_frame() {
        let (cadence, horizon, law) = frame_inputs();
        let history = HistoryConstruction::try_new(
            cadence.clone(),
            horizon,
            law.clone(),
            HistoryWindow::max_lag(2),
        )
        .unwrap();
        let action = ActionTiming::try_new(cadence.clone(), horizon, law.clone(), 20).unwrap();
        let policy =
            PolicyEvaluation::try_new(cadence.clone(), horizon, law.clone(), "treat").unwrap();
        let transition = Transition::try_new(cadence.clone(), horizon, law.clone()).unwrap();
        let regression =
            SequentialRegression::try_new(cadence.clone(), horizon, law.clone(), "y").unwrap();
        let weights =
            CensoringAssignmentWeight::try_new(cadence.clone(), horizon, law.clone(), 1.0, 0.5)
                .unwrap();
        let reduction = TrajectoryReduction::try_new(cadence, horizon, law, "mean").unwrap();

        assert_eq!(history.history(), HistoryWindow::max_lag(2));
        assert_eq!(history.frame().horizon(), horizon);
        assert_eq!(history.frame().initial_law().name(), Some("origin"));
        assert_eq!(history.frame().cadence().length, 4);
        assert_eq!(action.at_ns(), 20);
        assert_eq!(policy.policy(), "treat");
        assert_eq!(transition.step_ns(), 10);
        assert_eq!(regression.outcome(), "y");
        assert_eq!(weights.censoring().to_bits(), 1.0f64.to_bits());
        assert_eq!(weights.assignment().to_bits(), 0.5f64.to_bits());
        assert_eq!(reduction.functional(), "mean");
        assert_eq!(reduction.frame().interval_ns(), 10);
    }

    #[test]
    fn a_contemporaneous_initial_law_is_present() {
        let (cadence, horizon, _) = frame_inputs();
        let transition =
            Transition::try_new(cadence, horizon, InitialLaw::contemporaneous_observed()).unwrap();
        assert!(transition.frame().initial_law().name().is_none());
    }

    #[test]
    fn a_horizon_between_ticks_is_refused() {
        let cadence = regular(10, 4);
        let horizon = Horizon::ending_at(25);
        let law = origin();
        assert_crosses(
            HistoryConstruction::try_new(
                cadence.clone(),
                horizon,
                law.clone(),
                HistoryWindow::planner_default(),
            )
            .unwrap_err(),
            25,
        );
        assert_crosses(
            ActionTiming::try_new(cadence.clone(), horizon, law.clone(), 0).unwrap_err(),
            25,
        );
        assert_crosses(
            PolicyEvaluation::try_new(cadence.clone(), horizon, law.clone(), "treat").unwrap_err(),
            25,
        );
        assert_crosses(Transition::try_new(cadence.clone(), horizon, law.clone()).unwrap_err(), 25);
        assert_crosses(
            SequentialRegression::try_new(cadence.clone(), horizon, law.clone(), "y").unwrap_err(),
            25,
        );
        assert_crosses(
            CensoringAssignmentWeight::try_new(cadence.clone(), horizon, law.clone(), 1.0, 1.0)
                .unwrap_err(),
            25,
        );
        assert_crosses(
            TrajectoryReduction::try_new(cadence, horizon, law, "mean").unwrap_err(),
            25,
        );
    }

    #[test]
    fn a_horizon_past_the_cadence_is_refused() {
        let error =
            Transition::try_new(regular(10, 4), Horizon::ending_at(50), origin()).unwrap_err();
        match error {
            HistoryOperatorError::HorizonCrossesCadence { end_ns, interval_ns, declared_ticks } => {
                assert_eq!(end_ns, 50);
                assert_eq!(interval_ns, Some(10));
                assert_eq!(declared_ticks, Some(4));
            }
            other => panic!("expected a horizon past the cadence, got {other:?}"),
        }
    }

    #[test]
    fn an_irregular_or_zero_cadence_is_refused() {
        let irregular = TimeIndex { regularity: SamplingRegularity::Irregular, length: 4 };
        let irregular_error =
            Transition::try_new(irregular, Horizon::ending_at(10), origin()).unwrap_err();
        match irregular_error {
            HistoryOperatorError::HorizonCrossesCadence {
                interval_ns: None, end_ns: 10, ..
            } => {}
            other => panic!("expected an irregular cadence to be crossed, got {other:?}"),
        }
        let zero =
            Transition::try_new(regular(0, 4), Horizon::ending_at(10), origin()).unwrap_err();
        match zero {
            HistoryOperatorError::HorizonCrossesCadence { interval_ns: Some(0), .. } => {}
            other => panic!("expected a zero interval to be crossed, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_horizon_is_refused() {
        assert_crosses(
            Transition::try_new(regular(10, 4), Horizon::ending_at(0), origin()).unwrap_err(),
            0,
        );
    }

    #[test]
    fn a_non_finite_weight_is_refused() {
        let (cadence, horizon, law) = frame_inputs();
        let cases = [
            (WeightRole::Censoring, f64::NAN, 1.0),
            (WeightRole::Censoring, f64::INFINITY, 1.0),
            (WeightRole::Assignment, 1.0, f64::NEG_INFINITY),
        ];
        for (role, censoring, assignment) in cases {
            let error = CensoringAssignmentWeight::try_new(
                cadence.clone(),
                horizon,
                law.clone(),
                censoring,
                assignment,
            )
            .unwrap_err();
            match error {
                HistoryOperatorError::NonFiniteWeight { role: got, value } => {
                    assert_eq!(got, role);
                    assert!(!value.is_finite());
                }
                other => panic!("expected a non-finite {role} weight, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_zero_or_negative_inverse_probability_weight_is_refused() {
        let (cadence, horizon, law) = frame_inputs();
        let cases = [
            (WeightRole::Censoring, 0.0, 1.0),
            (WeightRole::Censoring, -2.0, 1.0),
            (WeightRole::Assignment, 1.0, 0.0),
            (WeightRole::Assignment, 1.0, -0.5),
        ];
        for (role, censoring, assignment) in cases {
            let error = CensoringAssignmentWeight::try_new(
                cadence.clone(),
                horizon,
                law.clone(),
                censoring,
                assignment,
            )
            .unwrap_err();
            match error {
                HistoryOperatorError::NonPositiveWeight { role: got, value } => {
                    assert_eq!(got, role);
                    assert!(value <= 0.0);
                }
                other => panic!("expected a non-positive {role} weight, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_action_off_the_cadence_or_past_the_horizon_is_refused() {
        let (cadence, horizon, law) = frame_inputs();
        let off_tick =
            ActionTiming::try_new(cadence.clone(), horizon, law.clone(), 15).unwrap_err();
        assert_eq!(
            off_tick,
            HistoryOperatorError::ActionMissesCadence { at_ns: 15, interval_ns: 10 }
        );
        let at_end = ActionTiming::try_new(cadence, horizon, law, 40).unwrap_err();
        assert_eq!(
            at_end,
            HistoryOperatorError::ActionMissesCadence { at_ns: 40, interval_ns: 10 }
        );
    }

    #[test]
    fn a_blank_name_is_refused() {
        let (cadence, horizon, law) = frame_inputs();
        let policy =
            PolicyEvaluation::try_new(cadence.clone(), horizon, law.clone(), " ").unwrap_err();
        assert_eq!(policy, HistoryOperatorError::EmptyName { field: "policy" });
        let outcome =
            SequentialRegression::try_new(cadence.clone(), horizon, law.clone(), "").unwrap_err();
        assert_eq!(outcome, HistoryOperatorError::EmptyName { field: "outcome" });
        let functional = TrajectoryReduction::try_new(cadence, horizon, law, "\t").unwrap_err();
        assert_eq!(functional, HistoryOperatorError::EmptyName { field: "functional" });
    }

    #[test]
    fn a_cadence_uses_the_observed_tick_grid() {
        let past_u32 = u64::from(u32::MAX) + 1;
        let Ok(length) = usize::try_from(past_u32) else {
            return;
        };
        let uncountable =
            Transition::try_new(regular(2, length), Horizon::ending_at(2), origin()).unwrap_err();
        assert_eq!(uncountable, HistoryOperatorError::CadenceUncountable);
        let tick_count = u32::MAX;
        let ticks = u64::from(tick_count);
        let interval_ns = (1_u64 << 32) + 2;
        let overflow = Transition::try_new(
            regular(interval_ns, usize::try_from(tick_count).unwrap()),
            Horizon::ending_at(interval_ns),
            origin(),
        )
        .unwrap_err();
        assert_eq!(overflow, HistoryOperatorError::CadenceSpanOverflow { interval_ns, ticks });
    }

    fn unit_dynamics() -> DelayedDynamics {
        dynamics(0.5, 1.0, 1.0)
    }

    fn dynamics(autoregression: f64, input_gain: f64, innovation_variance: f64) -> DelayedDynamics {
        DelayedDynamics::try_new(autoregression, input_gain, innovation_variance, origin(), 0.0)
            .unwrap()
    }

    fn delayed_request(action: DelayedAction) -> DelayedFeedbackRequest {
        let variable = antecedent_core::VariableId::from_raw(0);
        let (intervention, temporal) = match action {
            DelayedAction::Pulse { at_step, level_bits } => (
                Intervention::set(variable, Value::f64(f64::from_bits(level_bits))),
                TemporalPolicy::pulse(at_step),
            ),
            DelayedAction::Sustained { from_step, until_step, level_bits } => (
                Intervention::set(variable, Value::f64(f64::from_bits(level_bits))),
                TemporalPolicy::sustained(from_step, until_step),
            ),
        };
        DelayedFeedbackRequest {
            cadence: regular(1, 8),
            horizon: Horizon::ending_at(3),
            initial_law: origin(),
            history: HistoryWindow::planner_default(),
            delay: Some(FeedbackDelay::delayed(1).unwrap()),
            intervention,
            temporal,
            initial_state_change: None,
            truncation: None,
            search_cap: Some(FeedbackSearchCap::declared(2)),
            ancestry_steps: 2,
        }
    }

    #[test]
    fn a_shift_or_initial_state_change_is_not_executed_as_the_supplied_set() {
        let variable = antecedent_core::VariableId::from_raw(0);
        let shift = Intervention::shift(variable, Value::f64(1.0));
        let pulse = TemporalPolicy::pulse(0);
        let refused = delayed_action_of(&shift, &pulse).unwrap_err();
        assert_eq!(
            refused,
            HistoryOperatorError::UnsupportedOperation {
                operation: "shift",
                reason: "a shift is an additive change on the factual scale and is not executed \
                         as a set",
            }
        );
        let soft = Intervention::soft(variable, antecedent_core::MechanismOverride::constant(1.0));
        let soft_refused = delayed_action_of(&soft, &pulse).unwrap_err();
        assert!(matches!(
            soft_refused,
            HistoryOperatorError::UnsupportedOperation { operation: "soft", .. }
        ));
        let set = Intervention::set(variable, Value::f64(1.0));
        let from_pulse = delayed_action_of(&set, &pulse).unwrap();
        let from_sustained = delayed_action_of(&set, &TemporalPolicy::sustained(0, 0)).unwrap();
        assert_ne!(from_pulse, from_sustained);
        assert!(matches!(from_pulse, DelayedAction::Pulse { at_step: 0, .. }));
        assert!(matches!(
            from_sustained,
            DelayedAction::Sustained { from_step: 0, until_step: 0, .. }
        ));

        let mut request = delayed_request(DelayedAction::pulse(0, 1.0).unwrap());
        request.intervention = shift;
        request.temporal = pulse;
        let as_request = DelayedFeedback::try_new(request).unwrap_err();
        assert!(matches!(
            as_request,
            HistoryOperatorError::UnsupportedOperation { operation: "shift", .. }
        ));
        let initial = refuse_initial_state_change("x0", 2.0).unwrap_err();
        assert!(matches!(
            initial,
            HistoryOperatorError::UnsupportedOperation { operation: "initial_state_change", .. }
        ));
        let mut change = delayed_request(DelayedAction::pulse(0, 1.0).unwrap());
        change.initial_state_change =
            Some(InitialStateChange { law_name: "x0".to_owned(), level: 2.0 });
        assert!(matches!(
            DelayedFeedback::try_new(change).unwrap_err(),
            HistoryOperatorError::UnsupportedOperation { operation: "initial_state_change", .. }
        ));
    }

    #[test]
    fn finite_horizon_is_not_equilibrium_stationary_or_continuous() {
        let action = DelayedAction::sustained(0, 1, 1.0).unwrap();
        let feedback = DelayedFeedback::try_new(delayed_request(action)).unwrap();
        let dynamics = unit_dynamics();
        assert_eq!(feedback.target(), ResponseTarget::FiniteHorizon);
        assert_eq!(feedback.initial_law().name(), Some("origin"));
        assert_eq!(feedback.transition().step_ns(), 1);
        assert_eq!(feedback.history().history(), HistoryWindow::planner_default());

        let targets = [
            ResponseTarget::FiniteHorizon,
            ResponseTarget::StaticEquilibrium,
            ResponseTarget::Stationary,
            ResponseTarget::ContinuousTime,
        ];
        let responses = targets.map(|target| feedback.response(target, &dynamics).unwrap());
        for (response, target) in responses.iter().zip(targets) {
            assert_eq!(response.target(), target);
        }
        for (index, response) in responses.iter().enumerate() {
            for other in responses.iter().skip(index + 1) {
                assert_ne!(response, other);
            }
        }
        assert_eq!(responses[0].level_bits(), 1.5f64.to_bits());
        assert_eq!(responses[1].level_bits(), 2.0f64.to_bits());
        assert_eq!(responses[1].spread_bits(), 0.0f64.to_bits());
        assert_eq!(responses[2].level_bits(), 2.0f64.to_bits());
        assert_eq!(responses[2].spread_bits(), (1.0_f64 / 0.75).to_bits());
        assert_ne!(responses[3].level_bits(), responses[0].level_bits());
        assert_ne!(responses[3].level_bits(), responses[1].level_bits());
    }

    #[test]
    fn pulse_and_sustained_differ_on_the_same_delayed_system() {
        let dynamics = unit_dynamics();
        let pulse =
            DelayedFeedback::try_new(delayed_request(DelayedAction::pulse(0, 1.0).unwrap()))
                .unwrap();
        let sustained =
            DelayedFeedback::try_new(delayed_request(DelayedAction::sustained(0, 1, 1.0).unwrap()))
                .unwrap();
        let one_step =
            DelayedFeedback::try_new(delayed_request(DelayedAction::sustained(0, 0, 1.0).unwrap()))
                .unwrap();
        let pulse_response = pulse.response(ResponseTarget::FiniteHorizon, &dynamics).unwrap();
        let sustained_response =
            sustained.response(ResponseTarget::FiniteHorizon, &dynamics).unwrap();
        let one_step_response =
            one_step.response(ResponseTarget::FiniteHorizon, &dynamics).unwrap();
        assert_eq!(pulse_response.level_bits(), 0.5f64.to_bits());
        assert_eq!(sustained_response.level_bits(), 1.5f64.to_bits());
        assert_ne!(pulse_response, sustained_response);
        assert_ne!(pulse_response.action(), one_step_response.action());
        assert_ne!(pulse_response, one_step_response);
        assert_eq!(pulse.information().unwrap().decision().get(), 2);
        assert_eq!(pulse.information().unwrap().outcome().get(), 1);
    }

    #[test]
    fn an_off_cadence_delayed_horizon_is_refused() {
        let mut request = delayed_request(DelayedAction::pulse(0, 1.0).unwrap());
        request.cadence = regular(10, 4);
        request.horizon = Horizon::ending_at(25);
        let error = DelayedFeedback::try_new(request).unwrap_err();
        assert_crosses(error, 25);
    }

    #[test]
    fn an_uncertified_truncation_is_refused_and_not_extended() {
        let mut request = delayed_request(DelayedAction::pulse(0, 1.0).unwrap());
        request.delay = Some(FeedbackDelay::delayed(2).unwrap());
        request.history = HistoryWindow::max_lag(0);
        let error = DelayedFeedback::try_new(request.clone()).unwrap_err();
        assert_eq!(
            error,
            HistoryOperatorError::UncertifiedTruncation { retained_ticks: 0, required_ticks: 2 }
        );
        request.truncation = Some(TruncationCertificate::retaining(0));
        let admitted = DelayedFeedback::try_new(request).unwrap();
        assert_eq!(admitted.history().history().max_lag_steps(), Some(0));
        assert_eq!(admitted.delay().unwrap().ticks(), 2);
    }

    #[test]
    fn raising_the_search_cap_is_not_identification() {
        let mut request = delayed_request(DelayedAction::pulse(-1, 1.0).unwrap());
        request.search_cap = Some(FeedbackSearchCap::declared(2));
        request.ancestry_steps = 5;
        let error = DelayedFeedback::try_new(request.clone()).unwrap_err();
        assert_eq!(error, HistoryOperatorError::SearchNotIdentified { cap: 2, ancestry_steps: 5 });
        request.ancestry_steps = 2;
        let admitted = DelayedFeedback::try_new(request).unwrap();
        assert_eq!(
            admitted.search(),
            FeedbackSearch::Identified { cap: FeedbackSearchCap::declared(2) }
        );
    }

    #[test]
    fn feedback_after_the_decision_is_refused() {
        let mut request = delayed_request(DelayedAction::pulse(0, 1.0).unwrap());
        request.delay = Some(FeedbackDelay::delayed(5).unwrap());
        let error = DelayedFeedback::try_new(request).unwrap_err();
        assert_eq!(
            error,
            HistoryOperatorError::InformationRefused {
                refusal: crate::information_time::InformationRefusal::ProspectiveFuture {
                    decision: 2,
                    outcome: 5,
                }
            }
        );
    }

    fn long_request(action: DelayedAction) -> DelayedFeedbackRequest {
        let mut request = delayed_request(action);
        request.horizon = Horizon::ending_at(8);
        request
    }

    #[test]
    fn a_window_shorter_than_the_horizon_drives_only_its_own_steps() {
        let held = dynamics(1.0, 2.0, 1.0);
        let short =
            DelayedFeedback::try_new(long_request(DelayedAction::sustained(1, 3, 1.0).unwrap()))
                .unwrap();
        let long =
            DelayedFeedback::try_new(long_request(DelayedAction::sustained(1, 5, 1.0).unwrap()))
                .unwrap();
        let pulse =
            DelayedFeedback::try_new(long_request(DelayedAction::pulse(1, 1.0).unwrap())).unwrap();
        let finite = |feedback: &DelayedFeedback| {
            feedback.response(ResponseTarget::FiniteHorizon, &held).unwrap().level()
        };
        let continuous = |feedback: &DelayedFeedback| {
            feedback.response(ResponseTarget::ContinuousTime, &held).unwrap().level()
        };
        // Three held steps of gain two, whichever way they are integrated.
        assert_eq!(finite(&short).to_bits(), 6.0f64.to_bits());
        assert_eq!(continuous(&short).to_bits(), 6.0f64.to_bits());
        assert_eq!(finite(&long).to_bits(), 10.0f64.to_bits());
        assert_eq!(continuous(&long).to_bits(), 10.0f64.to_bits());
        assert_eq!(finite(&pulse).to_bits(), 2.0f64.to_bits());
        assert_ne!(finite(&pulse).to_bits(), finite(&short).to_bits());
    }

    #[test]
    fn a_decaying_window_agrees_with_the_rollout_as_the_step_shrinks() {
        // With a near one the continuous kernel is the held sum.
        let near = dynamics(1.0 - 1e-9, 1.0, 1.0);
        let window =
            DelayedFeedback::try_new(long_request(DelayedAction::sustained(2, 4, 1.0).unwrap()))
                .unwrap();
        let finite = window.response(ResponseTarget::FiniteHorizon, &near).unwrap().level();
        let continuous = window.response(ResponseTarget::ContinuousTime, &near).unwrap().level();
        assert!((finite - continuous).abs() < 1e-6);
    }

    #[test]
    fn feedback_arriving_after_the_last_included_tick_is_refused() {
        // The last intervened step arrives at 2 + 1 = 3, past the decision at 2.
        let error =
            DelayedFeedback::try_new(delayed_request(DelayedAction::sustained(0, 2, 1.0).unwrap()))
                .unwrap_err();
        assert_eq!(
            error,
            HistoryOperatorError::InformationRefused {
                refusal: crate::information_time::InformationRefusal::ProspectiveFuture {
                    decision: 2,
                    outcome: 3,
                }
            }
        );
        // A window that starts in the history is still judged on its last step.
        let mut late = delayed_request(DelayedAction::sustained(-2, 1, 1.0).unwrap());
        late.delay = Some(FeedbackDelay::delayed(5).unwrap());
        assert!(matches!(
            DelayedFeedback::try_new(late).unwrap_err(),
            HistoryOperatorError::InformationRefused { .. }
        ));
        let pulse = delayed_request(DelayedAction::pulse(-3, 1.0).unwrap());
        let admitted = DelayedFeedback::try_new(pulse).unwrap();
        // The arrival at -2 is already in the history and is judged at the origin.
        assert_eq!(admitted.information().unwrap().outcome().get(), 0);
        let inside = delayed_request(DelayedAction::sustained(-2, 1, 1.0).unwrap());
        let admitted = DelayedFeedback::try_new(inside).unwrap();
        assert_eq!(admitted.information().unwrap().outcome().get(), 2);
    }

    #[test]
    fn an_unstable_autoregression_has_no_equilibrium_or_stationary_law() {
        let action = DelayedAction::sustained(0, 1, 1.0).unwrap();
        let feedback = DelayedFeedback::try_new(delayed_request(action)).unwrap();
        for autoregression in [1.5, -1.0, -2.0] {
            let unstable = dynamics(autoregression, 1.0, 1.0);
            for target in [ResponseTarget::StaticEquilibrium, ResponseTarget::Stationary] {
                assert_eq!(
                    feedback.response(target, &unstable).unwrap_err(),
                    HistoryOperatorError::UnstableAutoregression { autoregression }
                );
            }
            // The finite rollout never needed the map to be stable.
            assert!(feedback.response(ResponseTarget::FiniteHorizon, &unstable).is_ok());
        }
        assert_eq!(
            feedback
                .response(ResponseTarget::StaticEquilibrium, &dynamics(1.0, 1.0, 1.0))
                .unwrap_err(),
            HistoryOperatorError::NoFixedPoint
        );
    }

    #[test]
    fn the_stationary_spread_comes_from_the_innovation_variance() {
        let action = DelayedAction::sustained(0, 1, 1.0).unwrap();
        let feedback = DelayedFeedback::try_new(delayed_request(action)).unwrap();
        let wide = feedback.response(ResponseTarget::Stationary, &dynamics(0.5, 1.0, 3.0)).unwrap();
        assert_eq!(wide.spread_bits(), (3.0_f64 / 0.75).to_bits());
        let still =
            feedback.response(ResponseTarget::Stationary, &dynamics(0.5, 1.0, 0.0)).unwrap();
        assert_eq!(still.spread_bits(), 0.0f64.to_bits());
        let negative = DelayedDynamics::try_new(0.5, 1.0, -1.0, origin(), 0.0).unwrap_err();
        assert_eq!(negative, HistoryOperatorError::NegativeInnovationVariance { value: -1.0 });
    }

    #[test]
    fn dynamics_for_another_initial_law_are_refused() {
        let action = DelayedAction::sustained(0, 1, 1.0).unwrap();
        let feedback = DelayedFeedback::try_new(delayed_request(action)).unwrap();
        let other = InitialLaw::named("other").unwrap();
        for law in [other, InitialLaw::contemporaneous_observed()] {
            let declared = DelayedDynamics::try_new(0.5, 1.0, 1.0, law.clone(), 0.0).unwrap();
            for target in [
                ResponseTarget::FiniteHorizon,
                ResponseTarget::StaticEquilibrium,
                ResponseTarget::Stationary,
                ResponseTarget::ContinuousTime,
            ] {
                assert_eq!(
                    feedback.response(target, &declared).unwrap_err(),
                    HistoryOperatorError::InitialLawMismatch {
                        frame: origin(),
                        dynamics: law.clone()
                    }
                );
            }
        }
        let matching = DelayedDynamics::try_new(0.5, 1.0, 1.0, origin(), 2.0).unwrap();
        let response = feedback.response(ResponseTarget::FiniteHorizon, &matching).unwrap();
        // The state rolls 2 -> 1 -> 1.5 -> 1.75 with the delayed input from the second step.
        assert_eq!(matching.initial_law(), feedback.initial_law());
        assert_eq!(response.level_bits(), 1.75f64.to_bits());
    }

    #[test]
    fn an_irregular_or_zero_cadence_has_no_horizon_end() {
        let irregular = TimeIndex { regularity: SamplingRegularity::Irregular, length: 4 };
        assert_eq!(
            horizon_ending_after(&irregular, 2).unwrap_err(),
            HistoryOperatorError::CadenceWithoutInterval { interval_ns: None, steps: 2 }
        );
        assert_eq!(
            horizon_ending_after(&regular(0, 4), 2).unwrap_err(),
            HistoryOperatorError::CadenceWithoutInterval { interval_ns: Some(0), steps: 2 }
        );
        assert_eq!(horizon_ending_after(&regular(10, 4), 2).unwrap(), Horizon::ending_at(20));
        assert_eq!(
            horizon_ending_after(&regular(u64::MAX, 4), 2).unwrap_err(),
            HistoryOperatorError::HorizonEndOverflow { interval_ns: u64::MAX, steps: 2 }
        );
    }
}

#[cfg(test)]
mod execution_tests {
    use super::*;

    const C: f64 = 0.5;
    const PHI: f64 = 0.6;
    const B0: f64 = 0.9;
    const B1: f64 = 0.4;

    fn regular(interval_ns: u64, ticks: usize) -> TimeIndex {
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length: ticks }
    }

    fn frame_parts(law: InitialLaw) -> (TimeIndex, Horizon, InitialLaw) {
        (regular(10, 400), Horizon::ending_at(40), law)
    }

    fn program_with(
        law: InitialLaw,
        history: HistoryWindow,
        weights: Option<(f64, f64)>,
        functional: &str,
    ) -> Result<HistoryProgram, HistoryExecutionError> {
        let (cadence, horizon, law) = frame_parts(law);
        let h =
            HistoryConstruction::try_new(cadence.clone(), horizon, law.clone(), history).unwrap();
        let timing = ActionTiming::try_new(cadence.clone(), horizon, law.clone(), 0).unwrap();
        let policy =
            PolicyEvaluation::try_new(cadence.clone(), horizon, law.clone(), "pulse").unwrap();
        let transition = Transition::try_new(cadence.clone(), horizon, law.clone()).unwrap();
        let regression =
            SequentialRegression::try_new(cadence.clone(), horizon, law.clone(), "y").unwrap();
        let weights = weights.map(|(c, a)| {
            CensoringAssignmentWeight::try_new(cadence.clone(), horizon, law.clone(), c, a).unwrap()
        });
        let reduction = TrajectoryReduction::try_new(cadence, horizon, law, functional).unwrap();
        HistoryProgram::try_new(h, timing, policy, transition, regression, weights, reduction)
    }

    fn program(functional: &str) -> HistoryProgram {
        program_with(
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::max_lag(1),
            None,
            functional,
        )
        .unwrap()
    }

    fn actions(n: usize) -> Vec<f64> {
        (0..n).map(|t| ((t * 7919 + 13) % 11) as f64 / 5.0 - 1.0).collect()
    }

    fn noise(state: &mut u64) -> f64 {
        let mut uniform = || {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        (0..12).map(|_| uniform()).sum::<f64>() - 6.0
    }

    /// `y_t = C + PHI y_{t-1} + B0 a_t + B1 a_{t-1} + sd * e_t`.
    fn simulate(n: usize, sd: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let a = actions(n);
        let mut state = seed;
        let mut y = vec![0.0; n];
        for t in 1..n {
            y[t] = C + PHI * y[t - 1] + B0 * a[t] + B1 * a[t - 1] + sd * noise(&mut state);
        }
        (y, a)
    }

    fn close(actual: f64, expected: f64, tolerance: f64, label: &str) {
        assert!((actual - expected).abs() <= tolerance, "{label}: {actual} vs {expected}");
    }

    #[test]
    fn a_pulse_and_a_sustained_action_follow_the_analytic_recursion_exactly() {
        let (y, a) = simulate(300, 0.0, 1);
        let series = TickSeries::try_new(&y, &a).unwrap();
        let final_program = program("final");
        let pulse = final_program
            .execute(
                &series,
                None,
                PolicyRule::Pulse { level: 1.0, otherwise: 0.0 },
                PolicyRule::Hold(0.0),
                None,
            )
            .unwrap();
        // The scan produced one window per tick after the first lag and the fit recovers the
        // data-generating mechanism.
        assert_eq!(pulse.windows, 299);
        for (got, want) in pulse.fit.coefficients().iter().zip([C, PHI, B0, B1]) {
            close(*got, want, 1e-8, "coefficient");
        }
        // Effect of a unit pulse at the first horizon tick: 0.9, then phi * 0.9 + 0.4,
        // then geometric decay at phi.
        let d0 = B0;
        let d1 = PHI * d0 + B1;
        let d2 = PHI * d1;
        let d3 = PHI * d2;
        for (step, want) in [d0, d1, d2, d3].into_iter().enumerate() {
            close(pulse.policy[step] - pulse.reference[step], want, 1e-8, "pulse path");
        }
        close(pulse.contrast(), d3, 1e-8, "final");

        let sustained = program("mean")
            .execute(
                &series,
                None,
                PolicyRule::Sustained { level: 1.0, otherwise: 0.0 },
                PolicyRule::Hold(0.0),
                None,
            )
            .unwrap();
        let s0 = B0;
        let s1 = PHI * s0 + B0 + B1;
        let s2 = PHI * s1 + B0 + B1;
        let s3 = PHI * s2 + B0 + B1;
        close(sustained.contrast(), (s0 + s1 + s2 + s3) / 4.0, 1e-8, "mean");
        let total = program("sum")
            .execute(
                &series,
                None,
                PolicyRule::Sustained { level: 1.0, otherwise: 0.0 },
                PolicyRule::Hold(0.0),
                None,
            )
            .unwrap();
        close(total.contrast(), s0 + s1 + s2 + s3, 1e-7, "sum");
    }

    #[test]
    fn a_reactive_policy_reads_the_rolled_history_not_the_observed_one() {
        let (y, a) = simulate(300, 0.0, 1);
        let series = TickSeries::try_new(&y, &a).unwrap();
        let (low, high, cutoff) = (-1.0, 1.0, 1.5);
        let run = program("sum")
            .execute(
                &series,
                None,
                PolicyRule::Reactive { cutoff, below: high, at_or_above: low, otherwise: 0.0 },
                PolicyRule::Hold(0.0),
                None,
            )
            .unwrap();
        // Independent recursion with the true mechanism from the last observed tick.
        let (mut prev_y, mut prev_a) = (y[299], a[299]);
        let mut expected = Vec::new();
        for _ in 0..4 {
            let action = if prev_y >= cutoff { low } else { high };
            let next = C + PHI * prev_y + B0 * action + B1 * prev_a;
            expected.push(next);
            prev_y = next;
            prev_a = action;
        }
        for (got, want) in run.policy.iter().zip(&expected) {
            close(*got, *want, 1e-7, "reactive path");
        }
    }

    #[test]
    fn a_noisy_series_estimates_the_pulse_effect_within_sampling_error() {
        let (y, a) = simulate(20_000, 0.3, 77);
        let series = TickSeries::try_new(&y, &a).unwrap();
        let run = program("final")
            .execute(
                &series,
                None,
                PolicyRule::Pulse { level: 1.0, otherwise: 0.0 },
                PolicyRule::Hold(0.0),
                None,
            )
            .unwrap();
        close(run.contrast(), PHI * PHI * (PHI * B0 + B1), 0.02, "noisy final effect");
        close(run.fit.residual_variance().sqrt(), 0.3, 0.02, "residual sd");
    }

    #[test]
    fn row_weights_are_the_weighted_least_squares_of_the_duplicated_rows() {
        let (y, a) = simulate(2_000, 0.5, 5);
        let series = TickSeries::try_new(&y, &a).unwrap();
        let program = program_with(
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::max_lag(1),
            Some((1.0, 1.0)),
            "final",
        )
        .unwrap();
        let windows = program.history.scan(&series).unwrap();
        let censoring: Vec<f64> =
            (0..windows.rows()).map(|r| if r % 3 == 0 { 0.5 } else { 1.0 }).collect();
        let assignment = vec![1.0; windows.rows()];
        let weight_operator = program.weights.as_ref().unwrap();
        let weights = weight_operator.row_weights(&censoring, &assignment).unwrap();
        assert!(weights.iter().zip(&censoring).all(|(w, p)| (w - 1.0 / p).abs() < 1e-12));
        let weighted = program.regression.fit(&windows, Some(&weights)).unwrap();
        // Reference: a plain fit on a table where every half-probability row appears twice.
        let p = windows.columns();
        let mut design = Vec::new();
        let mut target = Vec::new();
        for r in 0..windows.rows() {
            let copies = if r % 3 == 0 { 2 } else { 1 };
            for _ in 0..copies {
                design.extend_from_slice(&windows.design()[r * p..(r + 1) * p]);
                target.push(windows.target()[r]);
            }
        }
        let duplicated = HistoryWindows {
            depth: windows.depth(),
            first_tick: windows.first_tick(),
            columns: p,
            design,
            target,
        };
        let plain = program.regression.fit(&duplicated, None).unwrap();
        for (got, want) in weighted.coefficients().iter().zip(plain.coefficients()) {
            close(*got, *want, 1e-9, "weighted vs duplicated");
        }
        // And the weights moved the fit away from the unweighted one.
        let unweighted = program.regression.fit(&windows, None).unwrap();
        assert!(
            weighted
                .coefficients()
                .iter()
                .zip(unweighted.coefficients())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    #[test]
    fn refusals_name_their_next_attempt() {
        let (y, a) = simulate(300, 0.0, 1);
        let series = TickSeries::try_new(&y, &a).unwrap();
        let hold = PolicyRule::Hold(0.0);

        // A named initial law needs the state it names; a supplied state is used.
        let named = program_with(
            InitialLaw::named("clinic_start").unwrap(),
            HistoryWindow::max_lag(1),
            None,
            "final",
        )
        .unwrap();
        let error = named.execute(&series, None, hold, hold, None).unwrap_err();
        assert!(matches!(error, HistoryExecutionError::InitialStateRequired { .. }), "{error}");
        let state = InitialState { outcomes: vec![2.0], actions: vec![0.0] };
        let run = named.execute(&series, None, hold, hold, Some(&state)).unwrap();
        let mut prev = 2.0;
        let mut prev_a = 0.0;
        for value in &run.policy {
            let want = C + PHI * prev + B0 * 0.0 + B1 * prev_a;
            close(*value, want, 1e-7, "named-law start");
            prev = *value;
            prev_a = 0.0;
        }
        let wrong = InitialState { outcomes: vec![2.0, 1.0], actions: vec![0.0] };
        assert!(matches!(
            named.execute(&series, None, hold, hold, Some(&wrong)),
            Err(HistoryExecutionError::InitialStateShape { .. })
        ));

        // An unresolved planner default, a zero history and a short series.
        let unresolved = program_with(
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::planner_default(),
            None,
            "final",
        )
        .unwrap();
        assert_eq!(
            unresolved.execute(&series, None, hold, hold, None).unwrap_err(),
            HistoryExecutionError::HistoryCapUnresolved
        );
        let zero = program_with(
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::max_lag(0),
            None,
            "final",
        )
        .unwrap();
        assert_eq!(
            zero.execute(&series, None, hold, hold, None).unwrap_err(),
            HistoryExecutionError::HistoryDepthZero
        );
        let short = TickSeries::try_new(&y[..4], &a[..4]).unwrap();
        assert!(matches!(
            program("final").execute(&short, None, hold, hold, None),
            Err(HistoryExecutionError::SeriesTooShort { .. })
        ));

        // A functional the reduction cannot evaluate is refused when the program is bound.
        let (cadence, horizon, law) = frame_parts(InitialLaw::contemporaneous_observed());
        let history = HistoryConstruction::try_new(
            cadence.clone(),
            horizon,
            law.clone(),
            HistoryWindow::max_lag(1),
        )
        .unwrap();
        let reduction =
            TrajectoryReduction::try_new(cadence.clone(), horizon, law.clone(), "median").unwrap();
        let rest = (
            ActionTiming::try_new(cadence.clone(), horizon, law.clone(), 0).unwrap(),
            PolicyEvaluation::try_new(cadence.clone(), horizon, law.clone(), "p").unwrap(),
            Transition::try_new(cadence.clone(), horizon, law.clone()).unwrap(),
            SequentialRegression::try_new(cadence.clone(), horizon, law.clone(), "y").unwrap(),
        );
        let error = HistoryProgram::try_new(
            history.clone(),
            rest.0.clone(),
            rest.1.clone(),
            rest.2.clone(),
            rest.3.clone(),
            None,
            reduction,
        )
        .unwrap_err();
        assert!(matches!(error, HistoryExecutionError::UnknownFunctional { .. }), "{error}");

        // An operator over another horizon is not part of the program.
        let other = ActionTiming::try_new(cadence, Horizon::ending_at(30), law, 0).unwrap();
        let reduction = TrajectoryReduction::try_new(
            regular(10, 400),
            Horizon::ending_at(40),
            InitialLaw::contemporaneous_observed(),
            "final",
        )
        .unwrap();
        let error =
            HistoryProgram::try_new(history, other, rest.1, rest.2, rest.3, None, reduction)
                .unwrap_err();
        assert_eq!(error, HistoryExecutionError::FrameMismatch { operator: "action_timing" });

        // A probability outside (0, 1] is a positivity violation, not a weight.
        let weights = CensoringAssignmentWeight::try_new(
            regular(10, 400),
            Horizon::ending_at(40),
            InitialLaw::contemporaneous_observed(),
            1.0,
            1.0,
        )
        .unwrap();
        assert!(matches!(
            weights.row_weights(&[1.0, 0.0], &[1.0, 1.0]),
            Err(HistoryExecutionError::ProbabilityOutOfRange {
                role: WeightRole::Censoring,
                tick: 1,
                ..
            })
        ));
        assert!(matches!(
            TickSeries::try_new(&[1.0], &[1.0, 2.0]),
            Err(HistoryExecutionError::SeriesLengthMismatch { .. })
        ));
        assert!(matches!(
            TickSeries::try_new(&[f64::NAN], &[1.0]),
            Err(HistoryExecutionError::NonFiniteValue { tick: 0 })
        ));
    }

    #[test]
    fn the_shared_scan_aligns_several_columns_and_refuses_a_history_shorter_than_its_columns() {
        use antecedent_core::{ExecutionContext, Lag, VariableId};
        use antecedent_data::{LaggedColumn, TimeSeriesData};
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        let y: Vec<f64> = (0..10).map(|t| 10.0 * f64::from(t)).collect();
        let data = TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
            .unwrap();
        let columns = vec![
            LaggedColumn { variable: VariableId::from_raw(1), lag: Lag::CONTEMPORANEOUS },
            LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::from_raw(2) },
        ];
        let history = |lags| {
            HistoryConstruction::try_new(
                regular(10, 400),
                Horizon::ending_at(40),
                InitialLaw::contemporaneous_observed(),
                HistoryWindow::max_lag(lags),
            )
            .unwrap()
        };
        let ctx = ExecutionContext::for_tests(1);
        let mut scan = history(2).scan_columns(&data, columns.clone()).unwrap();
        let sample = scan.complete(&data, &ctx.kernel_policy).unwrap();
        // Window t reads y_t and x_{t-2}, for t = 2..10.
        assert_eq!(sample.n, 8);
        assert_eq!(sample.column(0), &y[2..]);
        assert_eq!(sample.column(1), &x[..8]);
        // The history is not extended to the column's depth.
        let error = history(1).scan_columns(&data, columns).unwrap_err();
        assert_eq!(error, HistoryExecutionError::HistoryShorterThanColumns { cap: 1, needed: 2 });
    }

    #[test]
    fn the_sequential_regression_fits_any_number_of_scanned_columns() {
        let program = program("final");
        let x: Vec<f64> = (0..50).map(|t| ((t * 7) % 13) as f64).collect();
        let z: Vec<f64> = (0..50).map(|t| ((t * 5) % 11) as f64 - 4.0).collect();
        let ones = vec![1.0; 50];
        let y: Vec<f64> = x.iter().zip(&z).map(|(a, b)| 1.0 + 2.0 * a + 3.0 * b).collect();
        let fit = program
            .regression
            .fit_columns(&[ones.as_slice(), x.as_slice(), z.as_slice()], &y, None)
            .unwrap();
        for (got, want) in fit.coefficients.iter().zip([1.0, 2.0, 3.0]) {
            close(*got, want, 1e-9, "multi-variable coefficient");
        }
        assert!(fit.residual_variance < 1e-12);
        assert!(matches!(
            program.regression.fit_columns(&[x.as_slice()], &y[..10], None),
            Err(HistoryExecutionError::WeightShape { .. })
        ));
    }

    #[test]
    fn every_kind_lowers_to_a_legal_operator_edge() {
        let program = program("final");
        let kinds = program.kinds();
        assert_eq!(kinds.len(), 6);
        for kind in kinds.into_iter().chain([HistoryOperatorKind::Weights]) {
            let operator = kind.operator().unwrap_or_else(|e| panic!("{}: {e}", kind.as_str()));
            assert!(!operator.outputs().is_empty(), "{}", kind.as_str());
        }
        assert_eq!(
            HistoryOperatorKind::Transition.operator().unwrap().effect(),
            antecedent_core::OperatorEffect::Scan
        );
        assert_eq!(program.policy_name(), "pulse");
        assert_eq!(program.step_ns(), 10);
    }

    fn design_fixture(rows: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..rows).map(|t| ((t * 7) % 13) as f64).collect();
        let z: Vec<f64> = (0..rows).map(|t| ((t * 5) % 11) as f64 - 4.0).collect();
        let y: Vec<f64> = (0..rows)
            .map(|t| 1.0 + 2.0 * x[t] - 0.5 * z[t] + (((t * 31) % 17) as f64 - 8.0) * 0.05)
            .collect();
        let mut design = vec![1.0; rows];
        design.extend_from_slice(&x);
        design.extend_from_slice(&z);
        (design, y)
    }

    #[test]
    fn the_shared_solve_is_the_backend_fit_unweighted_and_the_root_scaled_fit_weighted() {
        use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
        let rows = 60;
        let (design, y) = design_fixture(rows);
        let mut ws = LeastSquaresWorkspace::default();
        let shared =
            SequentialRegression::solve_design(&design, rows, 3, &y, None, &mut ws).unwrap();
        let direct = FaerBackend.least_squares(&design, rows, 3, &y, &mut ws).unwrap();
        assert_eq!(shared.coefficients, direct.coefficients);
        assert_eq!(shared.rss, direct.rss);
        let only = SequentialRegression::solve_coefficients(&design, rows, 3, &y).unwrap();
        assert_eq!(only, direct.coefficients);

        // Weighted: identical to rescaling every row and the target by the root weight.
        let weights: Vec<f64> =
            (0..rows).map(|r| if r % 4 == 0 { 0.0 } else { 0.5 + (r % 3) as f64 }).collect();
        let weighted =
            SequentialRegression::solve_design(&design, rows, 3, &y, Some(&weights), &mut ws)
                .unwrap();
        let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
        let mut scaled_design = design.clone();
        for column in scaled_design.chunks_exact_mut(rows) {
            for (v, r) in column.iter_mut().zip(&root) {
                *v *= r;
            }
        }
        let scaled_y: Vec<f64> = y.iter().zip(&root).map(|(v, r)| v * r).collect();
        let reference =
            FaerBackend.least_squares(&scaled_design, rows, 3, &scaled_y, &mut ws).unwrap();
        for (got, want) in weighted.coefficients.iter().zip(&reference.coefficients) {
            close(*got, *want, 1e-10, "weighted vs root-scaled");
        }
        close(weighted.rss, reference.rss, 1e-9, "weighted rss");
        // The weights moved the fit.
        assert!(
            weighted
                .coefficients
                .iter()
                .zip(&direct.coefficients)
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );
    }

    #[test]
    fn fit_columns_reports_the_weighted_residual_variance_of_the_shared_solve() {
        let program = program("final");
        let rows = 40;
        let (design, y) = design_fixture(rows);
        let columns: Vec<&[f64]> = design.chunks_exact(rows).collect();
        let weights: Vec<f64> = (0..rows).map(|r| 1.0 + (r % 5) as f64).collect();
        let fit = program.regression.fit_columns(&columns, &y, Some(&weights)).unwrap();
        let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
        let solved =
            SequentialRegression::solve_design(&design, rows, 3, &y, Some(&weights), &mut ws)
                .unwrap();
        assert_eq!(fit.coefficients, solved.coefficients);
        let total: f64 = weights.iter().sum();
        close(
            fit.residual_variance,
            solved.rss / (total * (1.0 - 3.0 / rows as f64)),
            1e-12,
            "variance",
        );
        // A duplicated column is not identified.
        let twice = [columns[0], columns[1], columns[1]];
        assert_eq!(
            program.regression.fit_columns(&twice, &y, None),
            Err(HistoryExecutionError::SingularHistory)
        );
    }

    #[test]
    fn window_weights_normalize_over_positive_windows_and_keep_zero_windows_in_place() {
        let program = program_with(
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::max_lag(1),
            Some((2.0, 3.0)),
            "final",
        )
        .unwrap();
        let operator = program.weights.as_ref().unwrap();
        let sampling = [2.0, 0.0, 6.0, 4.0, 0.0];
        let windowed = operator.window_weights(&sampling).unwrap();
        assert_eq!(windowed.rows_used, 3);
        assert_eq!(windowed.zero_weight_windows, 2);
        close(windowed.weight_sum, 12.0, 1e-12, "weight sum");
        // Mean one over the positive windows; zero windows keep a zero weight in place.
        close(windowed.weights.iter().sum::<f64>(), 3.0, 1e-12, "mean one");
        assert_eq!(windowed.weights[1], 0.0);
        assert_eq!(windowed.weights[4], 0.0);
        close(windowed.weights[2] / windowed.weights[0], 3.0, 1e-12, "ratio kept");
        // Kish: (sum w)^2 / sum w^2 = 144 / 56.
        close(windowed.effective_sample_size, 144.0 / 56.0, 1e-12, "ess");
        // The declared multipliers are a common factor and cancel.
        let unit = program_with(
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::max_lag(1),
            Some((1.0, 1.0)),
            "final",
        )
        .unwrap();
        assert_eq!(unit.weights.as_ref().unwrap().window_weights(&sampling).unwrap(), windowed);
        assert_eq!(
            operator.window_weights(&[0.0, 0.0]),
            Err(HistoryExecutionError::NoPositiveSamplingWeight)
        );
        assert!(matches!(
            operator.window_weights(&[1.0, -1.0]),
            Err(HistoryExecutionError::SamplingWeightInvalid { window: 1, .. })
        ));
        assert!(matches!(
            operator.window_weights(&[1.0, f64::NAN]),
            Err(HistoryExecutionError::SamplingWeightInvalid { window: 1, .. })
        ));
    }
}
