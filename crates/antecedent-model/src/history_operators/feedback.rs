use super::{
    AvailableOutcome, DelayedAction, DelayedDynamics, DelayedFeedback, DelayedFeedbackRequest,
    FeedbackDelay, FeedbackSearch, HistoryConstruction, HistoryOperatorError, InitialLaw,
    LevelSpread, ResponseTarget, TargetResponse, Transition, admit_action_window,
    admit_feedback_information, admit_search, admit_truncation, continuous_level,
    delayed_action_of, finite_bits, finite_horizon_level, fixed_point, horizon_steps,
    refuse_initial_state_change, stationary_law,
};

impl DelayedFeedback {
    /// Bind history, the initial law, and one transition on a finite horizon.
    ///
    /// # Errors
    ///
    /// The frame errors of [`OperatorFrame::try_new`](super::OperatorFrame::try_new).
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
