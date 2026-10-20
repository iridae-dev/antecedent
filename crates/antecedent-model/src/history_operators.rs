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
//! finite is refused. Nothing here fits a mechanism, reads a column, or scans
//! a table.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::sync::Arc;

use antecedent_core::{HistoryWindow, InitialLaw, is_blank};
use antecedent_data::{SamplingRegularity, TimeIndex};
use thiserror::Error;

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
}
