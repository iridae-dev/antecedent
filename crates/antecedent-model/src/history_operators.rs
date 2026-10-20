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
//! Nothing here fits a mechanism, reads a column, or scans a table.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::sync::Arc;

use antecedent_core::{HistoryWindow, InitialLaw, is_blank};
use antecedent_data::{SamplingRegularity, TimeIndex};
use thiserror::Error;

use crate::information_time::{
    AvailableOutcome, CadenceTick, ForwardOnly, InformationRefusal, InformationUse,
    RetrospectivePermission, admit_delayed_outcome,
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
    /// [`HistoryOperatorError::MissingInitialLaw`] when `initial_law` is
    /// absent. [`HistoryOperatorError::HorizonCrossesCadence`] when the
    /// horizon does not end on a tick of `cadence` or runs past the ticks
    /// `cadence` declares. [`HistoryOperatorError::CadenceSpanOverflow`] when
    /// the declared tick count times the interval does not fit in `u64`.
    /// [`HistoryOperatorError::CadenceUncountable`] when the tick count does
    /// not fit in `u64`.
    #[must_use = "a refused frame is not a cadence, horizon, and initial law"]
    pub fn try_new(
        cadence: TimeIndex,
        horizon: Horizon,
        initial_law: Option<InitialLaw>,
    ) -> Result<Self, HistoryOperatorError> {
        let Some(initial_law) = initial_law else {
            return Err(HistoryOperatorError::MissingInitialLaw);
        };
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
    /// The tick count does not fit in `u64`.
    #[error("cadence tick count exceeds u64; next attempt: shorten the cadence")]
    CadenceUncountable,
    /// No initial-state law was supplied.
    #[error("initial law is missing; next attempt: supply it, a time column does not")]
    MissingInitialLaw,
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
        initial_law: Option<InitialLaw>,
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
        initial_law: Option<InitialLaw>,
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
        initial_law: Option<InitialLaw>,
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
        initial_law: Option<InitialLaw>,
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
        initial_law: Option<InitialLaw>,
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
        initial_law: Option<InitialLaw>,
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
        initial_law: Option<InitialLaw>,
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
    /// Initial-state law. Absence is refused.
    pub initial_law: Option<InitialLaw>,
    /// History window. An explicit cap shorter than the delay needs a certificate.
    pub history: HistoryWindow,
    /// Delay. [`None`] is a finite horizon with contemporaneous transmission.
    pub delay: Option<FeedbackDelay>,
    /// Pulse or sustained action.
    pub action: DelayedAction,
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
            action,
            truncation,
            search_cap,
            ancestry_steps,
        } = request;
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
    let SamplingRegularity::Regular { interval_ns } = cadence.regularity else {
        return Err(crosses(end_ns, None, None));
    };
    let declared_ticks = u64::try_from(cadence.length).ok();
    if interval_ns == 0 || end_ns == 0 || end_ns % interval_ns != 0 {
        return Err(crosses(end_ns, Some(interval_ns), declared_ticks));
    }
    let Some(ticks) = declared_ticks else {
        return Err(HistoryOperatorError::CadenceUncountable);
    };
    let Some(covered) = ticks.checked_mul(interval_ns) else {
        return Err(HistoryOperatorError::CadenceSpanOverflow { interval_ns, ticks });
    };
    if end_ns > covered {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn regular(interval_ns: u64, ticks: usize) -> TimeIndex {
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length: ticks }
    }

    fn origin() -> InitialLaw {
        InitialLaw::named("origin").unwrap()
    }

    fn frame_inputs() -> (TimeIndex, Horizon, Option<InitialLaw>) {
        (regular(10, 4), Horizon::ending_at(40), Some(origin()))
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
            Transition::try_new(cadence, horizon, Some(InitialLaw::contemporaneous_observed()))
                .unwrap();
        assert!(transition.frame().initial_law().name().is_none());
    }

    #[test]
    fn a_horizon_between_ticks_is_refused() {
        let cadence = regular(10, 4);
        let horizon = Horizon::ending_at(25);
        let law = Some(origin());
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
        let error = Transition::try_new(regular(10, 4), Horizon::ending_at(50), Some(origin()))
            .unwrap_err();
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
            Transition::try_new(irregular, Horizon::ending_at(10), Some(origin())).unwrap_err();
        match irregular_error {
            HistoryOperatorError::HorizonCrossesCadence {
                interval_ns: None, end_ns: 10, ..
            } => {}
            other => panic!("expected an irregular cadence to be crossed, got {other:?}"),
        }
        let zero =
            Transition::try_new(regular(0, 4), Horizon::ending_at(10), Some(origin())).unwrap_err();
        match zero {
            HistoryOperatorError::HorizonCrossesCadence { interval_ns: Some(0), .. } => {}
            other => panic!("expected a zero interval to be crossed, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_horizon_is_refused() {
        assert_crosses(
            Transition::try_new(regular(10, 4), Horizon::ending_at(0), Some(origin())).unwrap_err(),
            0,
        );
    }

    #[test]
    fn a_missing_initial_law_is_refused() {
        let error = HistoryConstruction::try_new(
            regular(10, 4),
            Horizon::ending_at(40),
            None,
            HistoryWindow::max_lag(0),
        )
        .unwrap_err();
        assert_eq!(error, HistoryOperatorError::MissingInitialLaw);
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
    fn a_cadence_span_that_overflows_u64_is_refused() {
        let ticks = u64::MAX;
        let Ok(length) = usize::try_from(ticks) else {
            return;
        };
        let error = Transition::try_new(regular(2, length), Horizon::ending_at(2), Some(origin()))
            .unwrap_err();
        assert_eq!(error, HistoryOperatorError::CadenceSpanOverflow { interval_ns: 2, ticks });
    }

    fn unit_dynamics() -> DelayedDynamics {
        dynamics(0.5, 1.0, 1.0)
    }

    fn dynamics(autoregression: f64, input_gain: f64, innovation_variance: f64) -> DelayedDynamics {
        DelayedDynamics::try_new(autoregression, input_gain, innovation_variance, origin(), 0.0)
            .unwrap()
    }

    fn delayed_request(action: DelayedAction) -> DelayedFeedbackRequest {
        DelayedFeedbackRequest {
            cadence: regular(1, 8),
            horizon: Horizon::ending_at(3),
            initial_law: Some(origin()),
            history: HistoryWindow::planner_default(),
            delay: Some(FeedbackDelay::delayed(1).unwrap()),
            action,
            truncation: None,
            search_cap: Some(FeedbackSearchCap::declared(2)),
            ancestry_steps: 2,
        }
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
