use super::{
    CensoringAssignmentWeight, HistoryConstruction, SequentialRegression, TrajectoryReduction,
    Transition, WeightRole,
};
use thiserror::Error;

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
    /// Supplied weights do not match the program's declared weight operator.
    #[error(
        "the program declares weights={declared}, but row weights supplied={supplied}; next \
         attempt: supply row weights exactly when the program declares the weight operator"
    )]
    WeightDeclarationMismatch {
        /// Whether the program declares a weight operator.
        declared: bool,
        /// Whether row weights were supplied to execution.
        supplied: bool,
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
    /// The admission pool cannot hold the rollouts. Nothing was allocated for them.
    #[error(
        "the rollouts need {required_bytes} bytes and the admission pool holds \
         {available_bytes}; next attempt: shorten the horizon or raise the admission pool"
    )]
    RolloutMemoryRefused {
        /// Bytes both rollouts hold at once.
        required_bytes: u64,
        /// Bytes the pool still had.
        available_bytes: u64,
    },
    /// The rollouts would charge more work than the cap allows.
    #[error(
        "the rollouts charge {work} work units (ticks x retained state, both worlds) and the \
         cap is {cap}; next attempt: shorten the horizon or the history"
    )]
    RolloutWorkCapExceeded {
        /// Work the rollouts charge, or had charged when they were stopped.
        work: u64,
        /// The cap.
        cap: u64,
    },
    /// The execution context was cancelled; no result is reported.
    #[error("the execution was cancelled; no trajectory is reported")]
    Cancelled,
}

/// Outcome and action observed on the ticks of one cadence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickSeries<'a> {
    pub(super) outcome: &'a [f64],
    pub(super) action: &'a [f64],
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
    pub(super) depth: u32,
    pub(super) first_tick: usize,
    pub(super) columns: usize,
    pub(super) design: Vec<f64>,
    pub(super) target: Vec<f64>,
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
    /// variables and lags, as scanned by [`HistoryScan`](super::HistoryScan)) by weighted least squares.
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

/// The action rule a [`PolicyEvaluation`](super::PolicyEvaluation) evaluates on the rolled history.
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

    pub(super) fn admit(self) -> Result<Self, HistoryExecutionError> {
        match self.levels().iter().find(|v| !v.is_finite()) {
            Some(&value) => Err(HistoryExecutionError::NonFinitePolicyLevel { value }),
            None => Ok(self),
        }
    }

    pub(super) fn action(self, step: u32, start: u32, previous_outcome: f64) -> f64 {
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

/// Everything one execution of a [`HistoryProgram`](super::HistoryProgram) produced.
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
