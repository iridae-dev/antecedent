//! Doubly robust survival effects of an observational treatment at a declared horizon.
//!
//! The estimands are, for a binary treatment `A` with declared active and control levels and a
//! declared horizon `t*`:
//!
//! * the survival probability difference `S_1(t*) - S_0(t*)`, with `S_a(t) = P(T(a) > t)`;
//! * the restricted mean survival time difference `R_1(t*) - R_0(t*)`, with
//!   `R_a(t*) = E[min(T(a), t*)] = integral_0^{t*} S_a(u) du`.
//!
//! Identification needs (1) conditional exchangeability of the treatment given the declared
//! covariates `Z`, (2) positivity of both treatment levels, (3) consistency and no
//! interference, and (4) coarsening at random: given the arm and `Z`, the hazard of being
//! censored does not depend on the event time. Independent censoring within each arm is the
//! special case in which the censoring hazard does not depend on `Z` either. The assumption
//! is declared with [`CensoringAssumption`] and has no default.
//!
//! The estimator is the cross-fitted augmented inverse-probability and censoring-weighted
//! score of the efficient influence function. For arm `a` and a subject with covariates `Z`,
//! observed time `X`, event indicator `D`, outcome model `S_a(. | Z)` (a Cox model with a
//! Breslow baseline fitted in arm `a`), censoring survival `G_a(. | Z)` and propensity
//! `e_a(Z)`,
//!
//! ```text
//! psi_S = S_a(t*|Z) - 1{A = a} / e_a(Z) * S_a(t*|Z) * Q,
//! psi_R = R_a(Z)    - 1{A = a} / e_a(Z) * int_0^{t*} [int_u^{t*} S_a(s|Z) ds] / (S_a(u|Z) G_a(u-|Z)) dM(u),
//! Q     = int_0^{t*} dM(u) / (S_a(u|Z) G_a(u-|Z)),
//! M(u)  = 1{X <= u, D = 1} - Lambda_a(min(X, u) | Z).
//! ```
//!
//! The mean of `psi` is consistent for the estimand when either the propensity and censoring
//! models, or the outcome model, are correct. All nuisance models are fitted on the other
//! folds. The standard error is the standard deviation of the scores over the square root of
//! the rows; its calibration is unmeasured and no interval is produced here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{CoxBreslowFit, GlmOptions, LeastSquaresWorkspace, cox_breslow};

use crate::crossfit_plan::{
    CrossfitPlan, FoldRows, FoldSource, SHARED_FOLD_ID_ERRORS, rows_colmajor,
};
use crate::error::EstimationError;
use crate::propensity::{CategoryTrainingRows, fit_category_probabilities};
use crate::util::{mean, refuse, sample_std};

/// Default cross-fit fold count.
pub const DEFAULT_SURVIVAL_FOLDS: usize = 5;

/// Default positivity floor on the propensity of either treatment level.
pub const DEFAULT_SURVIVAL_MIN_PROPENSITY: f64 = 0.01;

/// Default floor on the probability of remaining uncensored through the horizon.
pub const DEFAULT_MIN_CENSORING_SURVIVAL: f64 = 0.05;

/// The assumption under which censoring is treated as ignorable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CensoringAssumption {
    /// Censoring is independent of the event time and of the covariates within each arm:
    /// the censoring survival is one curve per arm (Nelson-Aalen).
    IndependentWithinArm,
    /// Coarsening at random: given the arm and the declared covariates, the censoring hazard
    /// does not depend on the event time. It is a Cox model on the declared covariates
    /// within each arm.
    AtRandomGivenCovariates,
}

impl CensoringAssumption {
    /// Stable spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndependentWithinArm => "independent_within_arm",
            Self::AtRandomGivenCovariates => "at_random_given_covariates",
        }
    }
}

/// Columns of an observational survival problem. `NaN` is a missing cell.
#[derive(Clone, Copy, Debug)]
pub struct SurvivalData<'a> {
    /// Observed follow-up time.
    pub duration: &'a [f64],
    /// One when the event was observed at `duration`, zero when the unit was censored there.
    pub event: &'a [f64],
    /// Observed treatment level.
    pub treatment: &'a [f64],
    /// Declared adjustment columns, each with one value per row.
    pub covariates: &'a [&'a [f64]],
}

/// A survival effect request.
#[derive(Clone, Debug)]
pub struct ObservationalSurvival {
    /// Treatment level of the arm in the minuend.
    pub active: f64,
    /// Treatment level of the arm in the subtrahend.
    pub control: f64,
    /// The horizon `t*` both estimands are evaluated at.
    pub horizon: f64,
    /// The declared censoring assumption. There is no default.
    pub censoring: CensoringAssumption,
    /// Cross-fit folds.
    pub folds: usize,
    /// Smallest propensity of either arm tolerated at any row.
    pub min_propensity: f64,
    /// Smallest probability of remaining uncensored through the horizon tolerated at any row
    /// weighted by it.
    pub min_censoring_survival: f64,
    /// GLM options of the propensity model.
    pub glm_options: GlmOptions,
}

impl ObservationalSurvival {
    /// A request with the default folds and positivity floors.
    #[must_use]
    pub fn new(active: f64, control: f64, horizon: f64, censoring: CensoringAssumption) -> Self {
        Self {
            active,
            control,
            horizon,
            censoring,
            folds: DEFAULT_SURVIVAL_FOLDS,
            min_propensity: DEFAULT_SURVIVAL_MIN_PROPENSITY,
            min_censoring_survival: DEFAULT_MIN_CENSORING_SURVIVAL,
            glm_options: GlmOptions::default(),
        }
    }
}

/// One arm's standardized survival summaries at the horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmSurvival {
    /// Treatment level of the arm.
    pub level: f64,
    /// Doubly robust `S_a(t*)`.
    pub survival: f64,
    /// Influence-function standard error of `survival`.
    pub survival_se: f64,
    /// Doubly robust `R_a(t*)`.
    pub rmst: f64,
    /// Influence-function standard error of `rmst`.
    pub rmst_se: f64,
    /// Complete-case rows in the arm.
    pub rows: u64,
    /// Observed events in the arm.
    pub events: u64,
    /// Smallest out-of-fold propensity of the arm over all rows.
    pub propensity_min: f64,
    /// Smallest out-of-fold probability of remaining uncensored through the horizon over the
    /// rows of the arm.
    pub censoring_survival_min: f64,
}

/// An active-minus-control contrast and its standard error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurvivalContrast {
    /// Contrast value.
    pub estimate: f64,
    /// Influence-function standard error (calibration unmeasured).
    pub se: f64,
}

/// Survival effects at the declared horizon.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalSurvivalEstimate {
    /// The horizon the estimands refer to.
    pub horizon: f64,
    /// The censoring assumption the estimate was computed under.
    pub censoring: CensoringAssumption,
    /// The active arm.
    pub active: ArmSurvival,
    /// The control arm.
    pub control: ArmSurvival,
    /// `S_active(t*) - S_control(t*)`.
    pub survival_difference: SurvivalContrast,
    /// `R_active(t*) - R_control(t*)`.
    pub rmst_difference: SurvivalContrast,
    /// Complete-case rows the estimate was computed on.
    pub n_obs: u64,
    /// Rows dropped for a missing or non-finite cell.
    pub rows_dropped_missing: u64,
    /// Original row index of each complete-case row.
    pub row_index: Vec<antecedent_data::GlobalId>,
    /// Per-row score of the survival difference.
    pub survival_difference_scores: Vec<f64>,
    /// Per-row score of the restricted mean survival time difference.
    pub rmst_difference_scores: Vec<f64>,
}

impl ObservationalSurvival {
    /// Estimate the survival and restricted mean survival time differences at the horizon.
    ///
    /// # Errors
    ///
    /// `invalid_argument` for malformed columns, event codes, levels, horizon, folds or
    /// floors; `treatment_level_undeclared` for a row at neither level; `arm_not_populated`
    /// for an arm with too few rows or events; `positivity_violated` for a horizon beyond an
    /// arm's follow-up, a propensity below the floor, or a probability of remaining uncensored
    /// below the floor; `cancelled_no_claim`; and GLM or Cox failures.
    pub fn estimate(
        &self,
        data: &SurvivalData<'_>,
        ctx: &ExecutionContext,
    ) -> Result<ObservationalSurvivalEstimate, EstimationError> {
        self.validate()?;
        let prepared = Prepared::new(self, data)?;
        let n = prepared.n;
        if self.folds > n {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "survival_effect.invalid_folds",
                "cross-fitting folds cannot exceed the complete-case rows",
            ));
        }
        let plan = CrossfitPlan::derive(
            n,
            self.folds,
            FoldSource::Stratified {
                strata: &prepared.arm,
                units: &prepared.row_index,
                seed: ctx.rng.master_seed(),
            },
            &SHARED_FOLD_ID_ERRORS,
        )?;
        let mut scores = Scores::new(n);
        let mut glm_workspace = LeastSquaresWorkspace::default();
        for fold in 0..self.folds {
            if ctx.cancellation.is_cancelled() {
                return Err(refuse(
                    antecedent_core::reason_code!("cancelled_no_claim"),
                    "survival_effect.cancelled",
                    "the cross-fitted survival fit was cancelled before every fold was fit; no estimate is reported",
                ));
            }
            self.score_fold(&prepared, plan.rows(fold), &mut glm_workspace, &mut scores)?;
        }
        self.finish(prepared, &scores)
    }

    fn validate(&self) -> Result<(), EstimationError> {
        let invalid = |detail: &str, message: &str| {
            refuse(antecedent_core::reason_code!("invalid_argument"), detail, message)
        };
        if !self.active.is_finite()
            || !self.control.is_finite()
            || self.active.partial_cmp(&self.control) == Some(std::cmp::Ordering::Equal)
        {
            return Err(invalid(
                "survival_effect.invalid_levels",
                "the active and control treatment levels must be distinct finite values",
            ));
        }
        if !self.horizon.is_finite() || self.horizon <= 0.0 {
            return Err(invalid(
                "survival_effect.invalid_horizon",
                "the horizon must be finite and positive",
            ));
        }
        if self.folds < 2 {
            return Err(invalid(
                "survival_effect.invalid_folds",
                "cross-fitting requires at least two folds",
            ));
        }
        if !self.min_propensity.is_finite()
            || !(0.0..0.5).contains(&self.min_propensity)
            || !self.min_censoring_survival.is_finite()
            || !(0.0..1.0).contains(&self.min_censoring_survival)
        {
            return Err(invalid(
                "survival_effect.invalid_floor",
                "the propensity floor must lie in [0, 0.5) and the censoring survival floor in [0, 1)",
            ));
        }
        Ok(())
    }

    /// Fit the nuisance models on the training rows of a fold and score its rows.
    fn score_fold(
        &self,
        prepared: &Prepared,
        rows: FoldRows<'_>,
        glm_workspace: &mut LeastSquaresWorkspace,
        scores: &mut Scores,
    ) -> Result<(), EstimationError> {
        let (n, ncols) = (prepared.n, prepared.ncols);
        let train_arm: Vec<u32> = rows.train.iter().map(|&i| prepared.arm[i]).collect();
        let nv = rows.valid.len();
        let propensity = fit_category_probabilities(
            &CategoryTrainingRows {
                design: &rows_colmajor(&prepared.design, n, ncols, rows.train),
                category: &train_arm,
                n_categories: 2,
                ncols,
            },
            &rows_colmajor(&prepared.design, n, ncols, rows.valid),
            nv,
            &self.glm_options,
            glm_workspace,
        )?;
        for arm in 0..2usize {
            let models = self.fit_arm(prepared, rows.train, arm)?;
            let grid = ArmGrid::new(&models, self.horizon);
            for (r, &i) in rows.valid.iter().enumerate() {
                let covariates = prepared.covariate_row(i);
                let own = prepared.arm[i] as usize == arm;
                let e = propensity[arm * nv + r];
                scores.propensity_min[arm] = scores.propensity_min[arm].min(e);
                let evaluated = grid.evaluate(
                    &models,
                    &covariates,
                    own.then_some((prepared.duration[i], prepared.event[i])),
                );
                if own {
                    scores.censoring_survival_min[arm] =
                        scores.censoring_survival_min[arm].min(evaluated.censoring_survival);
                }
                let weight = if own { 1.0 / e.max(f64::MIN_POSITIVE) } else { 0.0 };
                scores.survival[arm][i] =
                    evaluated.survival - weight * evaluated.survival_augmentation;
                scores.rmst[arm][i] = evaluated.rmst - weight * evaluated.rmst_augmentation;
            }
        }
        Ok(())
    }

    /// The event and censoring models of one arm, fitted on the training rows in the arm.
    fn fit_arm(
        &self,
        prepared: &Prepared,
        train: &[usize],
        arm: usize,
    ) -> Result<ArmModels, EstimationError> {
        let rows: Vec<usize> =
            train.iter().copied().filter(|&i| prepared.arm[i] as usize == arm).collect();
        let level = if arm == 1 { self.active } else { self.control };
        let time: Vec<f64> = rows.iter().map(|&i| prepared.duration[i]).collect();
        let events: Vec<bool> = rows.iter().map(|&i| prepared.event[i]).collect();
        let censored: Vec<bool> = events.iter().map(|&d| !d).collect();
        let p = prepared.ncols - 1;
        let covariates = |p_used: usize| -> Vec<f64> {
            if p_used == 0 {
                Vec::new()
            } else {
                rows_colmajor(&prepared.design[prepared.n..], prepared.n, p, &rows)
            }
        };
        if !events.contains(&true) || rows.len() <= p + 1 {
            return Err(refuse(
                antecedent_core::reason_code!("arm_not_populated"),
                "survival_effect.training_fold_without_events",
                &format!(
                    "a cross-fit training fold has {} rows and no observed event in the arm at treatment level {level}, so its outcome model cannot be fitted; use fewer folds or a larger sample",
                    rows.len()
                ),
            ));
        }
        let event = cox_breslow(&time, &events, &covariates(p), p)
            .map_err(crate::error::EstimationError::from)?;
        let censoring_p = match self.censoring {
            CensoringAssumption::IndependentWithinArm => 0,
            CensoringAssumption::AtRandomGivenCovariates => p,
        };
        // An arm with no censored row in the training fold has no censoring to model: the
        // probability of remaining uncensored is one.
        let censoring = if censored.contains(&true) {
            Some(
                cox_breslow(&time, &censored, &covariates(censoring_p), censoring_p)
                    .map_err(crate::error::EstimationError::from)?,
            )
        } else {
            None
        };
        Ok(ArmModels { event, censoring, censoring_p })
    }

    fn finish(
        &self,
        prepared: Prepared,
        scores: &Scores,
    ) -> Result<ObservationalSurvivalEstimate, EstimationError> {
        let n = prepared.n;
        for (arm, level) in [(0usize, self.control), (1, self.active)] {
            let lowest = scores.propensity_min[arm];
            if lowest.is_nan() || lowest < self.min_propensity {
                return Err(refuse(
                    antecedent_core::reason_code!("positivity_violated"),
                    "survival_effect.treatment_positivity",
                    &format!(
                        "the propensity of treatment level {level} falls to {lowest:.6} on a row, below the declared floor {}; the doubly robust score weights by its inverse. Restrict the population or the covariates, or lower the floor knowingly",
                        self.min_propensity
                    ),
                ));
            }
        }
        for (arm, level) in [(0usize, self.control), (1, self.active)] {
            let lowest = scores.censoring_survival_min[arm];
            if lowest.is_nan() || lowest < self.min_censoring_survival {
                return Err(refuse(
                    antecedent_core::reason_code!("positivity_violated"),
                    "survival_effect.censoring_support",
                    &format!(
                        "in the arm at treatment level {level} the probability of remaining uncensored through the horizon {} falls to {lowest:.6}, below the declared floor {}; the censoring weights are unbounded. Shorten the horizon or declare a smaller floor knowingly",
                        self.horizon, self.min_censoring_survival
                    ),
                ));
            }
        }
        let arm_summary = |arm: usize, level: f64| {
            let se = |values: &[f64]| sample_std(values) / (n as f64).sqrt();
            ArmSurvival {
                level,
                survival: mean(&scores.survival[arm]),
                survival_se: se(&scores.survival[arm]),
                rmst: mean(&scores.rmst[arm]),
                rmst_se: se(&scores.rmst[arm]),
                rows: prepared.rows_by_arm[arm],
                events: prepared.events_by_arm[arm],
                propensity_min: scores.propensity_min[arm],
                censoring_survival_min: scores.censoring_survival_min[arm],
            }
        };
        let difference = |values: &[Vec<f64>; 2]| -> Vec<f64> {
            values[1].iter().zip(&values[0]).map(|(a, c)| a - c).collect()
        };
        let survival_difference_scores = difference(&scores.survival);
        let rmst_difference_scores = difference(&scores.rmst);
        let contrast = |values: &[f64]| SurvivalContrast {
            estimate: mean(values),
            se: sample_std(values) / (n as f64).sqrt(),
        };
        Ok(ObservationalSurvivalEstimate {
            horizon: self.horizon,
            censoring: self.censoring,
            active: arm_summary(1, self.active),
            control: arm_summary(0, self.control),
            survival_difference: contrast(&survival_difference_scores),
            rmst_difference: contrast(&rmst_difference_scores),
            n_obs: n as u64,
            rows_dropped_missing: prepared.dropped,
            row_index: prepared.row_index,
            survival_difference_scores,
            rmst_difference_scores,
        })
    }
}

/// Per-arm, per-row scores and the support diagnostics gathered while scoring the folds.
struct Scores {
    survival: [Vec<f64>; 2],
    rmst: [Vec<f64>; 2],
    propensity_min: [f64; 2],
    censoring_survival_min: [f64; 2],
}

impl Scores {
    fn new(n: usize) -> Self {
        Self {
            survival: [vec![0.0; n], vec![0.0; n]],
            rmst: [vec![0.0; n], vec![0.0; n]],
            propensity_min: [f64::INFINITY; 2],
            censoring_survival_min: [f64::INFINITY; 2],
        }
    }
}

/// The nuisance hazard models of one arm.
struct ArmModels {
    event: CoxBreslowFit,
    /// `None` when the training rows of the arm hold no censored subject.
    censoring: Option<CoxBreslowFit>,
    /// Covariates the censoring model uses: none when censoring is independent.
    censoring_p: usize,
}

/// The event-model jump grid up to the horizon, with the censoring baseline before each jump.
struct ArmGrid {
    horizon: f64,
    times: Vec<f64>,
    /// Cumulative event baseline hazard just after each jump.
    cumulative: Vec<f64>,
    /// Baseline hazard increment of each jump.
    increment: Vec<f64>,
    /// Cumulative censoring baseline hazard over times strictly before each jump.
    censoring_before: Vec<f64>,
}

/// One row's plug-in values and augmentation terms for one arm.
struct RowEvaluation {
    survival: f64,
    rmst: f64,
    /// `S_a(t*|Z) Q`; zero for a row outside the arm.
    survival_augmentation: f64,
    /// The restricted-mean-survival augmentation integral; zero for a row outside the arm.
    rmst_augmentation: f64,
    /// Probability of remaining uncensored through the horizon given the covariates.
    censoring_survival: f64,
}

impl ArmGrid {
    fn new(models: &ArmModels, horizon: f64) -> Self {
        let event = &models.event;
        let jumps = event.jump_times.partition_point(|&t| t <= horizon);
        let times = event.jump_times[..jumps].to_vec();
        let cumulative = event.cumulative_baseline[..jumps].to_vec();
        let increment = cumulative
            .iter()
            .enumerate()
            .map(|(j, &c)| c - if j == 0 { 0.0 } else { cumulative[j - 1] })
            .collect();
        let censoring_before = times
            .iter()
            .map(|&t| models.censoring.as_ref().map_or(0.0, |c| c.baseline_before(t)))
            .collect();
        Self { horizon, times, cumulative, increment, censoring_before }
    }

    /// Plug-in survival and restricted mean survival at the horizon for one covariate row and,
    /// when `observed` gives the row's own `(time, event)`, the augmentation terms.
    fn evaluate(
        &self,
        models: &ArmModels,
        covariates: &[f64],
        observed: Option<(f64, bool)>,
    ) -> RowEvaluation {
        let jumps = self.times.len();
        let t = self.horizon;
        let hazard_scale = models.event.linear_predictor(covariates).exp();
        let censoring_scale = models
            .censoring
            .as_ref()
            .map(|c| c.linear_predictor(&covariates[..models.censoring_p]).exp());
        let uncensored_before =
            |baseline: f64| censoring_scale.map_or(1.0, |scale| (-scale * baseline).exp());
        let at_horizon = if jumps == 0 { 0.0 } else { self.cumulative[jumps - 1] };
        let survival = (-hazard_scale * at_horizon).exp();
        // mean_residual[j] = (integral of S from times[j] to the horizon) / S(times[j]).
        let mut mean_residual = vec![0.0; jumps];
        for j in (0..jumps).rev() {
            let length =
                if j + 1 < jumps { self.times[j + 1] - self.times[j] } else { t - self.times[j] };
            let carried = if j + 1 < jumps {
                (-hazard_scale * self.increment[j + 1]).exp() * mean_residual[j + 1]
            } else {
                0.0
            };
            mean_residual[j] = length + carried;
        }
        let rmst = if jumps == 0 {
            t
        } else {
            self.times[0] + (-hazard_scale * self.increment[0]).exp() * mean_residual[0]
        };
        let censoring_survival =
            uncensored_before(models.censoring.as_ref().map_or(0.0, |c| c.baseline_at(t)));
        let Some((time, event)) = observed else {
            return RowEvaluation {
                survival,
                rmst,
                survival_augmentation: 0.0,
                rmst_augmentation: 0.0,
                censoring_survival,
            };
        };
        let (mut compensator_survival, mut compensator_rmst) = (0.0, 0.0);
        let reached = self.times.partition_point(|&u| u <= time);
        for (j, residual) in mean_residual.iter().enumerate().take(reached) {
            let weight =
                hazard_scale * self.increment[j] / uncensored_before(self.censoring_before[j]);
            compensator_survival +=
                weight * (-hazard_scale * (at_horizon - self.cumulative[j])).exp();
            compensator_rmst += weight * residual;
        }
        let (mut event_survival, mut event_rmst) = (0.0, 0.0);
        if event && time <= t {
            let at_time = models.event.baseline_at(time);
            let g_before = uncensored_before(
                models.censoring.as_ref().map_or(0.0, |c| c.baseline_before(time)),
            );
            event_survival = (-hazard_scale * (at_horizon - at_time)).exp() / g_before;
            let next = self.times.partition_point(|&u| u <= time);
            let residual = if next == jumps {
                t - time
            } else {
                (self.times[next] - time)
                    + (-hazard_scale * self.increment[next]).exp() * mean_residual[next]
            };
            event_rmst = residual / g_before;
        }
        RowEvaluation {
            survival,
            rmst,
            survival_augmentation: event_survival - compensator_survival,
            rmst_augmentation: event_rmst - compensator_rmst,
            censoring_survival,
        }
    }
}

/// The complete-case design of a survival problem.
struct Prepared {
    n: usize,
    ncols: usize,
    /// Column-major `[1 | Z...]`.
    design: Vec<f64>,
    /// Zero for the control arm, one for the active arm.
    arm: Vec<u32>,
    duration: Vec<f64>,
    event: Vec<bool>,
    row_index: Vec<antecedent_data::GlobalId>,
    rows_by_arm: [u64; 2],
    events_by_arm: [u64; 2],
    dropped: u64,
}

impl Prepared {
    #[allow(
        clippy::float_cmp,
        reason = "declared treatment levels and event codes are exact labels, not computed values"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one linear preparation: validate the request and data, then assemble the prepared rows"
    )]
    fn new(
        request: &ObservationalSurvival,
        data: &SurvivalData<'_>,
    ) -> Result<Self, EstimationError> {
        let invalid = |detail: &str, message: &str| {
            refuse(antecedent_core::reason_code!("invalid_argument"), detail, message)
        };
        let rows = data.duration.len();
        if rows == 0
            || data.event.len() != rows
            || data.treatment.len() != rows
            || data.covariates.iter().any(|c| c.len() != rows)
        {
            return Err(invalid(
                "survival_effect.invalid_data",
                "duration, event, treatment and covariates must be non-empty and of equal length",
            ));
        }
        let infinite = |values: &[f64]| values.iter().any(|v| v.is_infinite());
        if infinite(data.duration)
            || infinite(data.event)
            || infinite(data.treatment)
            || data.covariates.iter().any(|c| infinite(c))
        {
            return Err(invalid("survival_effect.invalid_data", "no value may be infinite"));
        }
        if request.censoring == CensoringAssumption::AtRandomGivenCovariates
            && data.covariates.is_empty()
        {
            return Err(invalid(
                "survival_effect.censoring_covariates_missing",
                "censoring at random given covariates needs declared covariates; with none, declare independent censoring within each arm",
            ));
        }
        let keep: Vec<usize> = (0..rows)
            .filter(|&i| {
                !data.duration[i].is_nan()
                    && !data.event[i].is_nan()
                    && !data.treatment[i].is_nan()
                    && data.covariates.iter().all(|c| !c[i].is_nan())
            })
            .collect();
        let n = keep.len();
        let ncols = 1 + data.covariates.len();
        if n <= 4 * ncols {
            return Err(invalid(
                "survival_effect.too_few_rows",
                "the complete-case rows cannot support the nuisance models",
            ));
        }
        let mut arm = Vec::with_capacity(n);
        let mut duration = Vec::with_capacity(n);
        let mut event = Vec::with_capacity(n);
        let mut rows_by_arm = [0u64; 2];
        let mut events_by_arm = [0u64; 2];
        let mut follow_up = [f64::NEG_INFINITY; 2];
        for &i in &keep {
            let a = if data.treatment[i] == request.active {
                1
            } else if data.treatment[i] == request.control {
                0
            } else {
                return Err(refuse(
                    antecedent_core::reason_code!("treatment_level_undeclared"),
                    "survival_effect.undeclared_level",
                    &format!(
                        "row {i} has treatment {}, which is neither the active level {} nor the control level {}",
                        data.treatment[i], request.active, request.control
                    ),
                ));
            };
            if data.duration[i] < 0.0 || (data.event[i] != 0.0 && data.event[i] != 1.0) {
                return Err(invalid(
                    "survival_effect.invalid_event",
                    &format!("row {i} needs a nonnegative duration and an event code of 0 or 1"),
                ));
            }
            arm.push(a);
            duration.push(data.duration[i]);
            event.push(data.event[i] == 1.0);
            rows_by_arm[a as usize] += 1;
            events_by_arm[a as usize] += u64::from(data.event[i] == 1.0);
            follow_up[a as usize] = follow_up[a as usize].max(data.duration[i]);
        }
        for (a, level) in [(0usize, request.control), (1, request.active)] {
            if rows_by_arm[a] == 0 {
                return Err(refuse(
                    antecedent_core::reason_code!("arm_not_populated"),
                    "survival_effect.arm_not_populated",
                    &format!("no complete-case row has treatment level {level}"),
                ));
            }
            if follow_up[a] < request.horizon {
                return Err(refuse(
                    antecedent_core::reason_code!("positivity_violated"),
                    "survival_effect.horizon_beyond_follow_up",
                    &format!(
                        "no unit in the arm at treatment level {level} is followed to the horizon {}: the longest follow-up is {}. Survival at the horizon cannot be estimated without extrapolating; choose a horizon within follow-up",
                        request.horizon, follow_up[a]
                    ),
                ));
            }
        }
        let mut design = vec![0.0; n * ncols];
        design[..n].fill(1.0);
        for (j, column) in data.covariates.iter().enumerate() {
            for (r, &i) in keep.iter().enumerate() {
                design[(1 + j) * n + r] = column[i];
            }
        }
        let row_index = keep
            .iter()
            .map(|&i| {
                u64::try_from(i).map(antecedent_data::GlobalId::new).map_err(|_| {
                    EstimationError::data_msg(
                        "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            n,
            ncols,
            design,
            arm,
            duration,
            event,
            row_index,
            rows_by_arm,
            events_by_arm,
            dropped: (rows - n) as u64,
        })
    }

    /// The covariates (without the intercept) of complete-case row `i`.
    fn covariate_row(&self, i: usize) -> Vec<f64> {
        (1..self.ncols).map(|j| self.design[j * self.n + i]).collect()
    }
}
