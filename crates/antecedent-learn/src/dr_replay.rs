//! A pure, bounded replay of the cross-fitted doubly-robust (DR-learner) average effect.
//!
//! [`replay_dr_ate`] takes the complete-case columns of an unweighted, independent-row,
//! all-observed-population analysis with a binary 0/1 treatment and recomputes, from those
//! columns and the declared settings alone, what the estimator layer reports: the seeded
//! arm-stratified fold plan, the three cross-fitted nuisances, the overlap clip, the
//! doubly-robust scores, the trimmed marginal mean (the point estimate), and the final-stage
//! regression of the scores with its portable predictor and support box. Every step is the
//! one definition the estimator calls: the planner in [`crate::crossfit_plan`], the nuisance
//! fits in [`crate::aipw_crossfit`], the helpers of this module for trimming and averaging, and
//! the learner factories of [`crate::spec`]. Nothing is read from a cache, a table or the
//! environment other than the execution context, so identical inputs give identical bits.
//!
//! Everything the result depends on is a field of [`DrReplaySpec`] or [`DrReplayInput`]; there
//! are no defaults. Row, column and fold counts are checked with checked arithmetic and the
//! working memory is admitted through the execution context before any buffer is allocated.
//! Cancellation is observed before the cross-fit, per fold, and around the final fit, and a
//! cancelled replay returns [`DrReplayError::Cancelled`] and no partial result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionError;
use antecedent_core::treatment_indicator::is_treated_indicator;
use antecedent_stats::{SandwichKind, coefficient_covariance, iid_se};
use thiserror::Error;

use crate::aipw_crossfit::{
    AipwCrossFitError, AipwLearners, ColumnMajorDesign, aipw_scores, clip_propensities,
    cross_fit_aipw_nuisances, design_for_spec,
};
use crate::crossfit::diagnose;
use crate::crossfit_plan::crossfit_fold_plan;
use crate::design::{DesignView, RowSelection, TargetView};
use crate::error::LearnError;
use crate::fitted_effect::{EffectSupport, FittedEffect, FittedEffectError};
use crate::learner::{LearnerProvenance, PredictionTask};
use crate::portable::PortablePredictor;
use crate::spec::{LearnerSpec, resolve_for};

/// Most rows a replay accepts: fold ids are recorded as `u32` row units.
pub const MAX_REPLAY_ROWS: usize = 4_294_967_295;
/// Most adjustment columns a replay accepts.
pub const MAX_REPLAY_COVARIATES: usize = 4096;
/// Most folds a replay accepts: fold ids are recorded as `u16`.
pub const MAX_REPLAY_FOLDS: usize = 65_536;

/// The message of a trim that leaves no row.
pub const TRIM_REMOVED_EVERY_ROW: &str =
    "overlap trim removed every row; no common-support units remain";

/// Whether both treatment arms are populated, reading arms with the one arm predicate.
///
/// Preparation validates binary values with a numerical tolerance; reading the arms with the
/// predicate the folds execute with keeps accepted near-zero and near-one values in their arms.
#[must_use]
pub fn both_arms_populated(treatment: &[f64]) -> bool {
    treatment.iter().any(|&value| !is_treated_indicator(value))
        && treatment.iter().any(|&value| is_treated_indicator(value))
}

/// Whether every propensity lies strictly inside the unit interval.
///
/// A propensity of exactly 0 or 1, or one that is not finite, carries an infinite
/// inverse-probability weight.
#[must_use]
pub fn propensities_are_interior(scores: &[f64]) -> bool {
    scores.iter().all(|&e| e > 0.0 && e < 1.0)
}

/// Indices of the rows whose **raw** (pre-clip) propensity lies inside `[trim, 1 - trim]`, or
/// `None` when no trim is configured (every row is retained).
///
/// Trimming redefines the estimand to the common-support population.
///
/// # Errors
///
/// [`TRIM_REMOVED_EVERY_ROW`] when every row falls outside the band.
pub fn trim_retained_rows(
    raw_scores: &[f64],
    trim: Option<f64>,
) -> Result<Option<Vec<usize>>, &'static str> {
    let Some(t) = trim else { return Ok(None) };
    let retained: Vec<usize> = raw_scores
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| (p >= t && p <= 1.0 - t).then_some(i))
        .collect();
    if retained.is_empty() {
        return Err(TRIM_REMOVED_EVERY_ROW);
    }
    Ok(Some(retained))
}

/// The unweighted marginal mean of the scores over the retained rows: the point estimate of the
/// cross-fitted doubly-robust average effect.
///
/// `rows` must be non-empty and index `phi`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "the retained row count is far below 2^53, so the conversion is exact"
)]
pub fn retained_mean(phi: &[f64], rows: &[usize]) -> f64 {
    rows.iter().map(|&i| phi[i]).sum::<f64>() / rows.len() as f64
}

/// The influence values of the unweighted marginal mean of `phi` over `rows`, embedded in the
/// original row universe: `(phi[i] - ate) * n / kept` for a retained row and zero elsewhere.
///
/// `rows` must be non-empty and index `phi`; `ate` is [`retained_mean`] of the same rows.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "row counts are far below 2^53, so the conversions are exact"
)]
pub fn retained_influence(phi: &[f64], rows: &[usize], ate: f64) -> Vec<f64> {
    let n = phi.len();
    let mass = rows.len() as f64;
    let mut influence = vec![0.0; n];
    for &i in rows {
        influence[i] = (phi[i] - ate) * n as f64 / mass;
    }
    influence
}

/// The analytic standard error of the unweighted marginal mean: the iid standard error of its
/// influence values.
#[must_use]
pub fn retained_standard_error(phi: &[f64], rows: &[usize], ate: f64) -> f64 {
    iid_se(&retained_influence(phi, rows, ate))
}

/// HC0 sandwich standard errors of a linear CATE regression of orthogonal scores.
///
/// The regression uses `rows` (the retained common-support rows; all rows when `None`) and the
/// covariance is the stats crate's HC0 `(X'X)^-1 sum e^2 x x' (X'X)^-1`, evaluated at every
/// row's features. Penalized or nonlinear finals are not this OLS map, so they return `None`
/// instead of an invented interval. `weights` are the sampling weights of a weighted fit.
#[must_use]
pub fn linear_cate_pointwise_se(
    spec: LearnerSpec,
    x: DesignView<'_>,
    phi: &[f64],
    cate: &[f64],
    rows: Option<&[usize]>,
    weights: Option<&[f64]>,
) -> Option<Vec<f64>> {
    if !matches!(spec, LearnerSpec::Linear(_)) {
        return None;
    }
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 || phi.len() != n || cate.len() != n {
        return None;
    }
    let mut design = vec![0.0; n.saturating_mul(p)];
    for c in 0..p {
        for r in 0..n {
            design[c * n + r] = x.get(r, c).ok()?;
        }
    }
    let all_rows: Vec<usize>;
    let used: &[usize] = if let Some(rows) = rows {
        rows
    } else {
        all_rows = (0..n).collect();
        &all_rows
    };
    let m = used.len();
    let mut sub = vec![0.0; m * p];
    for c in 0..p {
        for (k, &r) in used.iter().enumerate() {
            sub[c * m + k] = design[c * n + r];
        }
    }
    let mut residuals: Vec<f64> = used.iter().map(|&r| phi[r] - cate[r]).collect();
    if let Some(weights) = weights {
        for (k, &r) in used.iter().enumerate() {
            let root = weights[r].sqrt();
            residuals[k] *= root;
            for c in 0..p {
                sub[c * m + k] *= root;
            }
        }
    }
    let cov = coefficient_covariance(&sub, m, p, &residuals, SandwichKind::Hc0).ok()?;
    let mut se = vec![0.0; n];
    for i in 0..n {
        let mut v = 0.0;
        for j in 0..p {
            let mut cj = 0.0;
            for k in 0..p {
                cj += cov[j * p + k] * design[k * n + i];
            }
            v += design[j * n + i] * cj;
        }
        if !(v.is_finite() && v >= 0.0) {
            return None;
        }
        se[i] = v.sqrt();
    }
    Some(se)
}

/// Overlap handling of the replay: the propensity clip and trim bounds of the producer's
/// overlap policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrOverlap {
    /// Clip propensities into `[clip, 1 - clip]` when `Some`; the bound lies in `(0, 0.5)`.
    pub clip: Option<f64>,
    /// Drop rows whose raw propensity lies outside `[trim, 1 - trim]` when `Some`; the bound
    /// lies in `(0, 0.5)`.
    pub trim: Option<f64>,
    /// Whether the caller declared the bounds. An undeclared default carries its default clip
    /// but does not decide a positivity question: a propensity fit that reaches the boundary
    /// refuses instead of being clipped. A declared policy with a clip or a trim admits it.
    pub declared: bool,
}

impl DrOverlap {
    /// Whether the bounds govern propensities at the boundary, so a fit that reaches it is
    /// admitted and the clip or trim applies.
    #[must_use]
    pub const fn admits_boundary_saturation(self) -> bool {
        self.declared && (self.clip.is_some() || self.trim.is_some())
    }

    /// Check that the bounds lie in `(0, 0.5)`.
    ///
    /// # Errors
    ///
    /// [`DrReplayError::Overlap`] for a bound outside the open interval or not finite.
    pub fn validate(self) -> Result<(), DrReplayError> {
        let inside = |bound: f64| bound.is_finite() && bound > 0.0 && bound < 0.5;
        if self.clip.is_some_and(|bound| !inside(bound)) {
            return Err(DrReplayError::Overlap("overlap clip must lie in (0, 0.5)"));
        }
        if self.trim.is_some_and(|bound| !inside(bound)) {
            return Err(DrReplayError::Overlap("overlap trim must lie in (0, 0.5)"));
        }
        Ok(())
    }
}

/// Every declared setting the replayed result depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrReplaySpec {
    /// Cross-fit folds: at least two and at most [`MAX_REPLAY_FOLDS`].
    pub folds: usize,
    /// The master seed of the analysis: the seed of the fold plan, which must also be the
    /// master seed of the execution context the replay runs under.
    pub seed: u64,
    /// The learner of the per-arm outcome regressions.
    pub outcome: LearnerSpec,
    /// The learner of the treatment propensity.
    pub treatment: LearnerSpec,
    /// The final-stage regression of the doubly-robust scores.
    pub final_learner: LearnerSpec,
    /// Clip and trim of the propensities.
    pub overlap: DrOverlap,
    /// Whether the declared active treatment level is numerically above the declared control
    /// level. The fold strata are `1` at the higher level and `0` at the lower, so swapping the
    /// declared levels swaps the strata labels and so the folds dealt.
    pub active_level_above_control: bool,
}

/// The complete-case columns of the analysis.
#[derive(Clone, Copy, Debug)]
pub struct DrReplayInput<'a> {
    /// Treatment arm indicator of every row, exactly `0.0` or `1.0`.
    pub treatment: &'a [f64],
    /// Outcome of every row.
    pub outcome: &'a [f64],
    /// Adjustment columns in the order of the design (the intercept excluded), each of the
    /// same length as the treatment.
    pub covariates: &'a [&'a [f64]],
    /// The source-table row each row came from. Rows sharing one id are one physical unit and
    /// share one fold; the order of the ids sets the order the units are dealt in.
    pub row_ids: &'a [u64],
    /// Whether the design carries a leading constant column. The estimator's design always
    /// does, so only `true` is supported.
    pub intercept: bool,
}

/// Why a replay stopped.
#[derive(Debug, Error)]
pub enum DrReplayError {
    /// The declaration lies outside the supported operation.
    #[error("unsupported doubly-robust replay: {0}")]
    Unsupported(&'static str),
    /// A column length, count, bound or value does not fit the operation.
    #[error("doubly-robust replay shape error: {0}")]
    Shape(&'static str),
    /// One of the treatment arms has no row.
    #[error("both treatment arms must remain populated after complete-case selection")]
    ArmNotPopulated,
    /// The cancellation token of the execution context stopped the work; nothing is reported.
    #[error("the doubly-robust replay was cancelled")]
    Cancelled,
    /// Worker or scratch admission refused the work (other than by cancellation).
    #[error(transparent)]
    Admission(AdmissionError),
    /// A learner could not be resolved, fitted or evaluated, or the planner refused.
    #[error(transparent)]
    Learner(LearnError),
    /// The overlap bounds are invalid, or the propensities leave no usable support.
    #[error("doubly-robust replay overlap refusal: {0}")]
    Overlap(&'static str),
}

/// What a replay recomputed.
#[derive(Clone, Debug)]
pub struct DrReplay {
    /// The point estimate: the mean of [`Self::phi`] over the retained rows.
    pub ate: f64,
    /// The cross-fitted propensity of every row before the clip.
    pub raw_propensity: Vec<f64>,
    /// The propensity of every row after the clip.
    pub propensity: Vec<f64>,
    /// The doubly-robust score of every row.
    pub phi: Vec<f64>,
    /// The fold of every row.
    pub fold_ids: Vec<u16>,
    /// The rows of the common-support band, or `None` when no trim is declared.
    pub retained: Option<Vec<usize>>,
    /// The analytic standard error of [`Self::ate`] ([`retained_standard_error`]).
    pub standard_error: f64,
    /// Out-of-fold R² of the outcome nuisance at each row's observed arm, as the estimator
    /// reports it; `None` when it cannot be scored.
    pub outcome_oof_r2: Option<f64>,
    /// Out-of-fold log loss of the cross-fitted propensity model, as the estimator reports it.
    pub treatment_oof_logloss: Option<f64>,
    /// The final model's prediction for every row.
    pub cate: Vec<f64>,
    /// Pointwise HC0 standard errors of the CATE for a linear final stage, else `None`.
    pub cate_se: Option<Vec<f64>>,
    /// The final model as a portable predictor, or `None` for a learner without a codec.
    pub predictor: Option<PortablePredictor>,
    /// The box of the covariates over the rows the final model was fitted on, or `None` when
    /// it cannot be computed.
    pub support: Option<EffectSupport>,
    /// Whether the final model's design carries a leading constant column.
    pub intercept: bool,
    /// The number of adjustment columns, in the order they were given.
    pub features: usize,
    /// The identities of the cross-fitted propensity models, fold by fold.
    pub nuisance_provenance: Vec<LearnerProvenance>,
    /// The identity of the final model.
    pub final_provenance: LearnerProvenance,
}

impl DrReplay {
    /// The retained effect model as the estimator layer stores it, naming the adjustment
    /// columns by `features`; `None` when the final learner has no portable form.
    ///
    /// # Errors
    ///
    /// A feature list of the wrong length or duplicates, or an invalid predictor.
    pub fn fitted_effect(
        &self,
        features: Vec<u32>,
    ) -> Result<Option<FittedEffect>, FittedEffectError> {
        let Some(predictor) = self.predictor.clone() else { return Ok(None) };
        if features.len() != self.features {
            return Err(FittedEffectError::InvalidArgument("invalid fitted effect schema"));
        }
        let model = FittedEffect {
            version: 1,
            features,
            intercept: self.intercept,
            predictor,
            support: self.support.clone(),
        };
        model.validate()?;
        Ok(Some(model))
    }
}

fn admission_error(error: AdmissionError) -> DrReplayError {
    match error {
        AdmissionError::Cancelled => DrReplayError::Cancelled,
        other => DrReplayError::Admission(other),
    }
}

fn learn_error(error: LearnError) -> DrReplayError {
    match error {
        LearnError::Cancelled
        | LearnError::Stats(
            antecedent_stats::StatsError::Cancelled
            | antecedent_stats::StatsError::Admission(AdmissionError::Cancelled),
        )
        | LearnError::Admission(AdmissionError::Cancelled)
        | LearnError::Probability(
            antecedent_prob::ProbError::Admission(AdmissionError::Cancelled)
            | antecedent_prob::ProbError::Cancelled { .. },
        ) => DrReplayError::Cancelled,
        LearnError::Admission(error)
        | LearnError::Stats(antecedent_stats::StatsError::Admission(error))
        | LearnError::Probability(antecedent_prob::ProbError::Admission(error)) => {
            DrReplayError::Admission(error)
        }
        other => DrReplayError::Learner(other),
    }
}

fn cross_fit_error(error: AipwCrossFitError) -> DrReplayError {
    match error {
        AipwCrossFitError::Learner(error) => learn_error(error),
        AipwCrossFitError::Cancelled => DrReplayError::Cancelled,
        AipwCrossFitError::Admission(error) => admission_error(error),
        AipwCrossFitError::WeightsUnsupported { .. } => {
            DrReplayError::Unsupported("sampling weights are not part of this operation")
        }
    }
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), DrReplayError> {
    if ctx.cancellation.is_cancelled() { Err(DrReplayError::Cancelled) } else { Ok(()) }
}

/// The validated dimensions of a replay.
struct Dimensions {
    nrows: usize,
    design_ncols: usize,
    bytes: u64,
}

/// Check every count and length with checked arithmetic and price the working memory, before
/// any buffer is allocated.
fn check_dimensions(
    input: &DrReplayInput<'_>,
    spec: &DrReplaySpec,
) -> Result<Dimensions, DrReplayError> {
    if !input.intercept {
        return Err(DrReplayError::Unsupported(
            "the estimator's design always carries a leading intercept column",
        ));
    }
    if spec.folds < 2 {
        return Err(DrReplayError::Unsupported("cross-fitting requires at least two folds"));
    }
    if spec.folds > MAX_REPLAY_FOLDS {
        return Err(DrReplayError::Shape("the fold count exceeds the supported maximum"));
    }
    if input.covariates.len() > MAX_REPLAY_COVARIATES {
        return Err(DrReplayError::Shape(
            "the adjustment column count exceeds the supported maximum",
        ));
    }
    let nrows = input.treatment.len();
    if input.outcome.len() != nrows
        || input.row_ids.len() != nrows
        || input.covariates.iter().any(|column| column.len() != nrows)
    {
        return Err(DrReplayError::Shape("treatment, outcome, row ids and covariates must align"));
    }
    if nrows > MAX_REPLAY_ROWS {
        return Err(DrReplayError::Shape("the row count exceeds the supported maximum"));
    }
    if nrows < spec.folds {
        return Err(DrReplayError::Shape("cross-fitting folds cannot exceed the row count"));
    }
    let design_ncols = input
        .covariates
        .len()
        .checked_add(1)
        .ok_or(DrReplayError::Shape("design column count overflow"))?;
    // The design, the fold plan's train and validation lists (about one pass per fold), a dozen
    // per-row vectors (nuisances, scores, retained rows, predictions) and the planner's maps.
    let words = [design_ncols, 12, spec.folds]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(DrReplayError::Shape("working size overflow"))?;
    let bytes = u64::try_from(nrows)
        .ok()
        .and_then(|rows| rows.checked_mul(u64::try_from(words).ok()?))
        .and_then(|cells| cells.checked_mul(8))
        .and_then(|cells| {
            u64::try_from(nrows).ok()?.checked_mul(128).and_then(|maps| cells.checked_add(maps))
        })
        .ok_or(DrReplayError::Shape("working size overflow"))?;
    Ok(Dimensions { nrows, design_ncols, bytes })
}

fn check_values(input: &DrReplayInput<'_>) -> Result<(), DrReplayError> {
    #[allow(
        clippy::float_cmp,
        reason = "the prepared treatment is exactly the recoded 0/1 indicator"
    )]
    let binary = input.treatment.iter().all(|&value| value == 0.0 || value == 1.0);
    if !binary {
        return Err(DrReplayError::Shape("the treatment must be coded exactly 0 or 1"));
    }
    let finite = input.outcome.iter().all(|value| value.is_finite())
        && input.covariates.iter().all(|column| column.iter().all(|value| value.is_finite()));
    if !finite {
        return Err(DrReplayError::Shape("outcome and covariates must be finite"));
    }
    if !both_arms_populated(input.treatment) {
        return Err(DrReplayError::ArmNotPopulated);
    }
    Ok(())
}

/// Replay the cross-fitted doubly-robust average effect of one analysis.
///
/// The execution context's master seed must equal [`DrReplaySpec::seed`], so the seed that
/// plans the folds is also the seed the learners run under.
///
/// # Errors
///
/// [`DrReplayError::Unsupported`] for a declaration outside the operation (no intercept, fewer
/// than two folds, a context seed that differs from the declared one);
/// [`DrReplayError::Shape`] for misaligned or over-limit columns and non-finite or non-binary
/// values; [`DrReplayError::ArmNotPopulated`]; [`DrReplayError::Overlap`] for invalid bounds, a
/// propensity of exactly 0 or 1 with no clip, or a trim that leaves fewer than two rows;
/// [`DrReplayError::Cancelled`] when the token of `ctx` stopped the work;
/// [`DrReplayError::Admission`] when the working memory is refused;
/// [`DrReplayError::Learner`] for a learner or planner failure.
#[allow(
    clippy::too_many_lines,
    reason = "one cohesive pipeline: validate, plan folds, cross-fit, score, average, fit the final stage"
)]
pub fn replay_dr_ate(
    input: DrReplayInput<'_>,
    spec: DrReplaySpec,
    ctx: &ExecutionContext,
) -> Result<DrReplay, DrReplayError> {
    check_cancelled(ctx)?;
    if ctx.rng.master_seed() != spec.seed {
        return Err(DrReplayError::Unsupported(
            "the execution context's master seed must equal the declared seed",
        ));
    }
    spec.overlap.validate()?;
    let Dimensions { nrows, design_ncols, bytes } = check_dimensions(&input, &spec)?;
    check_values(&input)?;
    let _working = ctx.acquire_admission(0, bytes).map_err(admission_error)?;

    let mut design = vec![0.0; nrows * design_ncols];
    design[..nrows].fill(1.0);
    for (column, values) in input.covariates.iter().enumerate() {
        let base = (column + 1) * nrows;
        design[base..base + nrows].copy_from_slice(values);
    }
    let strata: Vec<u32> = input
        .treatment
        .iter()
        .map(|&t| u32::from(is_treated_indicator(t) == spec.active_level_above_control))
        .collect();
    let fold_ids: Vec<u16> = crossfit_fold_plan(&strata, input.row_ids, spec.folds, spec.seed)
        .map_err(DrReplayError::Learner)?
        .into_iter()
        .map(|id| {
            u16::try_from(id).map_err(|_| DrReplayError::Shape("a fold id exceeds the u16 range"))
        })
        .collect::<Result<_, _>>()?;
    drop(strata);

    check_cancelled(ctx)?;
    let (mu0, mu1, raw_propensity, treat) = cross_fit_aipw_nuisances(
        AipwLearners { outcome: spec.outcome, treatment: spec.treatment },
        ColumnMajorDesign { values: &design, nrows, ncols: design_ncols },
        input.outcome,
        input.treatment,
        (spec.folds, fold_ids),
        ctx,
        None,
        spec.overlap.admits_boundary_saturation(),
    )
    .map_err(cross_fit_error)?;

    let mut propensity = raw_propensity.clone();
    clip_propensities(&mut propensity, spec.overlap.clip);
    if !propensities_are_interior(&propensity) {
        return Err(DrReplayError::Overlap(
            "a fitted propensity is 0 or 1 (or not finite) and no clip is set; the \
             inverse-probability weight is infinite",
        ));
    }
    let phi = aipw_scores(input.treatment, input.outcome, &propensity, &mu0, &mu1);
    // The nuisance diagnostics the estimator publishes, from the same predictions.
    let observed_arm_prediction: Vec<f64> = input
        .treatment
        .iter()
        .zip(&mu0)
        .zip(&mu1)
        .map(|((&t, &m0), &m1)| if is_treated_indicator(t) { m1 } else { m0 })
        .collect();
    let outcome_oof_r2 =
        diagnose(PredictionTask::Regression, input.outcome, &observed_arm_prediction).r2;
    let treatment_oof_logloss = treat.validation.logloss;
    if phi.iter().any(|score| !score.is_finite()) {
        return Err(DrReplayError::Overlap("the doubly-robust scores are not all finite"));
    }
    let retained =
        trim_retained_rows(&raw_propensity, spec.overlap.trim).map_err(DrReplayError::Overlap)?;
    let kept: Vec<usize>;
    let rows: &[usize] = if let Some(rows) = retained.as_deref() {
        rows
    } else {
        kept = (0..nrows).collect();
        &kept
    };
    if rows.len() < 2 {
        return Err(DrReplayError::Overlap("the average effect requires two retained rows"));
    }
    let ate = retained_mean(&phi, rows);
    let standard_error = retained_standard_error(&phi, rows, ate);

    check_cancelled(ctx)?;
    let factory =
        resolve_for(spec.final_learner, PredictionTask::Regression).map_err(learn_error)?;
    let view = design_for_spec(
        spec.final_learner,
        ColumnMajorDesign { values: &design, nrows, ncols: design_ncols },
    )
    .map_err(learn_error)?;
    // The reported average covers the common-support rows only, so the final stage is fitted on
    // those rows too.
    let selection: Option<Vec<u64>> = retained
        .as_deref()
        .map(|kept_rows| {
            kept_rows
                .iter()
                .map(|&i| {
                    u64::try_from(i).map_err(|_| {
                        DrReplayError::Shape("a row index exceeds the 64-bit selection range")
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let fit_view = match &selection {
        Some(selected) => view.with_rows(RowSelection::new(selected)).map_err(learn_error)?,
        None => view,
    };
    let fitted = factory.fit(fit_view, TargetView::new(&phi), None, ctx).map_err(learn_error)?;
    let mut cate = vec![0.0; nrows];
    fitted.predict(view, &mut cate, ctx).map_err(learn_error)?;
    let cate_se =
        linear_cate_pointwise_se(spec.final_learner, view, &phi, &cate, retained.as_deref(), None);
    let predictor = match fitted.portable() {
        Ok(predictor) => Some(predictor),
        Err(LearnError::Unsupported { .. }) => None,
        Err(error) => return Err(learn_error(error)),
    };
    // The support contract is the box of the rows the final model was fitted on.
    let support = EffectSupport::from_rows(input.covariates, retained.as_deref());
    check_cancelled(ctx)?;
    Ok(DrReplay {
        ate,
        raw_propensity,
        propensity,
        phi,
        outcome_oof_r2,
        treatment_oof_logloss,
        standard_error,
        cate_se,
        fold_ids: treat.fold_assignment.clone(),
        retained,
        cate,
        predictor,
        support,
        intercept: view.ncols() == design_ncols,
        features: input.covariates.len(),
        nuisance_provenance: treat.model_provenance,
        final_provenance: fitted.provenance(),
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names,
    reason = "the controls build small synthetic tables from small indices and compare exact copies"
)]
mod tests {
    use antecedent_core::execution::AdmissionPool;

    use super::*;
    use crate::spec::{LinearSpec, LogisticSpec};

    const LINEAR: LearnerSpec = LearnerSpec::Linear(LinearSpec {});
    const LOGISTIC: LearnerSpec = LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 });
    const CLIPPED: DrOverlap = DrOverlap { clip: Some(0.05), trim: None, declared: true };
    const BARE: DrOverlap = DrOverlap { clip: None, trim: None, declared: true };

    fn spec(folds: usize, seed: u64, overlap: DrOverlap) -> DrReplaySpec {
        DrReplaySpec {
            folds,
            seed,
            outcome: LINEAR,
            treatment: LOGISTIC,
            final_learner: LINEAR,
            overlap,
            active_level_above_control: true,
        }
    }

    /// `y = a + b t + c x` with the treatment alternating and overlapping `x` in every fold.
    struct Noiseless {
        t: Vec<f64>,
        y: Vec<f64>,
        x: Vec<f64>,
        rows: Vec<u64>,
    }

    const EFFECT: f64 = 2.0;

    fn noiseless(n: usize) -> Noiseless {
        let (a, c) = (1.5, -0.75);
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 13.0).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let y: Vec<f64> = (0..n).map(|i| a + EFFECT * t[i] + c * x[i]).collect();
        Noiseless { t, y, x, rows: (0..n as u64).collect() }
    }

    /// Treated fraction two fifths, an outcome with a deterministic scatter, one unit per row.
    fn intercept_only_table(n: usize) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
        let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 5 < 2))).collect();
        let y: Vec<f64> =
            (0..n).map(|i| ((i * i) % 7) as f64 + 2.0 * t[i] + 0.25 * (i % 3) as f64).collect();
        (t, y, (0..n as u64).collect())
    }

    fn run(
        table: &Noiseless,
        declared: DrReplaySpec,
        ctx: &ExecutionContext,
    ) -> Result<DrReplay, DrReplayError> {
        let covariates: [&[f64]; 1] = [&table.x];
        replay_dr_ate(
            DrReplayInput {
                treatment: &table.t,
                outcome: &table.y,
                covariates: &covariates,
                row_ids: &table.rows,
                intercept: true,
            },
            declared,
            ctx,
        )
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn a_noiseless_linear_outcome_recovers_the_effect_and_the_final_stage_is_the_ols_line() {
        let table = noiseless(52);
        let replay = run(&table, spec(4, 5, CLIPPED), &ExecutionContext::for_tests(5)).unwrap();
        assert!((replay.ate - EFFECT).abs() <= 1e-9, "{}", replay.ate);
        assert!(replay.retained.is_none());
        assert_eq!(replay.fold_ids.len(), 52);
        assert!(replay.fold_ids.iter().all(|&f| f < 4));
        // The final stage regresses the scores on [1, x]; the line computed by the normal
        // equations of that two-column design must be the prediction of every row.
        let n = 52.0;
        let sx: f64 = table.x.iter().sum();
        let sxx: f64 = table.x.iter().map(|v| v * v).sum();
        let sy: f64 = replay.phi.iter().sum();
        let sxy: f64 = table.x.iter().zip(&replay.phi).map(|(a, b)| a * b).sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n;
        for (i, &x) in table.x.iter().enumerate() {
            let line = intercept + slope * x;
            assert!((replay.cate[i] - line).abs() <= 1e-9, "row {i}: {} vs {line}", replay.cate[i]);
            assert!((replay.cate[i] - EFFECT).abs() <= 1e-9, "row {i}");
        }
        let support = replay.support.as_ref().expect("support");
        assert_eq!(support.rows, 52);
        let lo = table.x.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = table.x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!((support.ranges[0].lo, support.ranges[0].hi), (lo, hi));
        assert!(replay.intercept);
        assert_eq!(replay.features, 1);
        let predictor = replay.predictor.as_ref().expect("a linear final stage is portable");
        assert_eq!(predictor.columns, 2);
        let effect = replay.fitted_effect(vec![9]).unwrap().expect("fitted effect");
        assert_eq!(effect.features, vec![9]);
        assert!(replay.fitted_effect(vec![1, 2]).is_err());
    }

    #[test]
    fn an_intercept_only_design_scores_match_the_hand_computed_fold_means() {
        let n = 60;
        let (t, y, rows) = intercept_only_table(n);
        let none: [&[f64]; 0] = [];
        let replay = replay_dr_ate(
            DrReplayInput {
                treatment: &t,
                outcome: &y,
                covariates: &none,
                row_ids: &rows,
                intercept: true,
            },
            spec(3, 4, BARE),
            &ExecutionContext::for_tests(4),
        )
        .unwrap();
        let mean = |keep: &dyn Fn(usize) -> bool, values: &[f64]| {
            let picked: Vec<f64> = (0..n).filter(|&j| keep(j)).map(|j| values[j]).collect();
            picked.iter().sum::<f64>() / picked.len() as f64
        };
        let mut reference = vec![0.0; n];
        for i in 0..n {
            let fold = replay.fold_ids[i];
            let train = |j: usize| replay.fold_ids[j] != fold;
            let m0 = mean(&|j| train(j) && t[j] < 0.5, &y);
            let m1 = mean(&|j| train(j) && t[j] > 0.5, &y);
            let e = mean(&train, &t);
            reference[i] =
                (m1 - m0) + t[i] * (y[i] - m1) / e - (1.0 - t[i]) * (y[i] - m0) / (1.0 - e);
            // The unpenalized logistic MLE agrees with the training fraction to about 1e-8.
            assert!((replay.propensity[i] - e).abs() <= 1e-7, "row {i}");
        }
        for (i, expected) in reference.iter().enumerate() {
            assert!((replay.phi[i] - expected).abs() <= 1e-6, "score row {i}");
        }
        let ate = reference.iter().sum::<f64>() / n as f64;
        assert!((replay.ate - ate).abs() <= 1e-6, "{} vs {ate}", replay.ate);
        // An intercept-only final stage predicts the mean score everywhere.
        for (i, c) in replay.cate.iter().enumerate() {
            assert!((c - replay.ate).abs() <= 1e-12 * (1.0 + replay.ate.abs()), "cate row {i}");
        }
        assert!(replay.support.is_none());
        assert_eq!(replay.features, 0);
    }

    #[test]
    fn a_trim_keeps_only_the_common_support_rows_and_fits_the_final_stage_on_them() {
        let table = noiseless(52);
        let overlap = DrOverlap { clip: Some(0.05), trim: Some(0.3), declared: true };
        let replay = run(&table, spec(4, 5, overlap), &ExecutionContext::for_tests(5)).unwrap();
        let expected = trim_retained_rows(&replay.raw_propensity, Some(0.3)).unwrap();
        assert_eq!(replay.retained, expected);
        let kept = replay.retained.as_ref().unwrap();
        assert!(kept.iter().all(|&i| replay.raw_propensity[i] >= 0.3));
        let direct = kept.iter().map(|&i| replay.phi[i]).sum::<f64>() / kept.len() as f64;
        assert_eq!(replay.ate.to_bits(), direct.to_bits());
        assert_eq!(replay.support.as_ref().unwrap().rows, kept.len() as u64);
    }

    #[test]
    fn the_shared_helpers_have_the_documented_semantics() {
        assert_eq!(trim_retained_rows(&[0.5], None), Ok(None));
        assert_eq!(
            trim_retained_rows(&[0.05, 0.2, 0.5, 0.8, 0.96], Some(0.1)),
            Ok(Some(vec![1, 2, 3]))
        );
        assert_eq!(trim_retained_rows(&[0.01, 0.99], Some(0.1)), Err(TRIM_REMOVED_EVERY_ROW));
        assert!(propensities_are_interior(&[0.2, 1e-12]));
        assert!(!propensities_are_interior(&[0.2, 0.0]));
        assert!(!propensities_are_interior(&[1.0]));
        assert!(!propensities_are_interior(&[f64::NAN]));
        assert!(both_arms_populated(&[1e-13, 1.0 - 1e-13]));
        assert!(!both_arms_populated(&[1e-13, -1e-13]));
        assert!(!both_arms_populated(&[1.0, 1.0]));
        assert_eq!(retained_mean(&[1.0, 10.0, 3.0], &[0, 2]), 2.0);
        assert!(BARE.validate().is_ok());
        assert!(!BARE.admits_boundary_saturation());
        assert!(CLIPPED.admits_boundary_saturation());
        assert!(
            !DrOverlap { clip: Some(0.01), trim: None, declared: false }
                .admits_boundary_saturation()
        );
    }

    #[test]
    fn declarations_outside_the_operation_are_refused_as_unsupported() {
        let table = noiseless(52);
        let ctx = ExecutionContext::for_tests(5);
        let covariates: [&[f64]; 1] = [&table.x];
        let input = DrReplayInput {
            treatment: &table.t,
            outcome: &table.y,
            covariates: &covariates,
            row_ids: &table.rows,
            intercept: false,
        };
        let refused = replay_dr_ate(input, spec(4, 5, CLIPPED), &ctx);
        assert!(matches!(refused, Err(DrReplayError::Unsupported(_))), "{refused:?}");
        for folds in [0, 1] {
            let refused = run(&table, spec(folds, 5, CLIPPED), &ctx);
            assert!(matches!(refused, Err(DrReplayError::Unsupported(_))), "{refused:?}");
        }
        let seeded = run(&table, spec(4, 6, CLIPPED), &ctx);
        assert!(matches!(seeded, Err(DrReplayError::Unsupported(_))), "{seeded:?}");
        // A linear learner is not a propensity learner: the shared resolver refuses it.
        let mut wrong = spec(4, 5, CLIPPED);
        wrong.treatment = LINEAR;
        let refused = run(&table, wrong, &ctx);
        assert!(matches!(refused, Err(DrReplayError::Learner(_))), "{refused:?}");
    }

    #[test]
    fn invalid_overlap_bounds_and_an_empty_trim_are_overlap_refusals() {
        let table = noiseless(52);
        let ctx = ExecutionContext::for_tests(5);
        for overlap in [
            DrOverlap { clip: Some(0.5), trim: None, declared: true },
            DrOverlap { clip: Some(f64::NAN), trim: None, declared: true },
            DrOverlap { clip: None, trim: Some(0.0), declared: true },
        ] {
            let refused = run(&table, spec(4, 5, overlap), &ctx);
            assert!(matches!(refused, Err(DrReplayError::Overlap(_))), "{refused:?}");
        }
        // Without covariates every propensity is the training treated fraction, here about
        // 0.4, so a band around one half retains no row.
        let (t, y, rows) = intercept_only_table(60);
        let none: [&[f64]; 0] = [];
        let all_out = DrOverlap { clip: None, trim: Some(0.45), declared: true };
        let refused = replay_dr_ate(
            DrReplayInput {
                treatment: &t,
                outcome: &y,
                covariates: &none,
                row_ids: &rows,
                intercept: true,
            },
            spec(3, 5, all_out),
            &ctx,
        );
        assert!(
            matches!(refused, Err(DrReplayError::Overlap(TRIM_REMOVED_EVERY_ROW))),
            "{refused:?}"
        );
    }

    #[test]
    fn one_arm_ragged_non_finite_and_non_binary_inputs_are_refused() {
        let ctx = ExecutionContext::for_tests(5);
        let mut table = noiseless(52);
        table.t = vec![1.0; 52];
        let refused = run(&table, spec(4, 5, CLIPPED), &ctx);
        assert!(matches!(refused, Err(DrReplayError::ArmNotPopulated)), "{refused:?}");

        let table = noiseless(52);
        let short = &table.x[..51];
        let covariates: [&[f64]; 1] = [short];
        let ragged = replay_dr_ate(
            DrReplayInput {
                treatment: &table.t,
                outcome: &table.y,
                covariates: &covariates,
                row_ids: &table.rows,
                intercept: true,
            },
            spec(4, 5, CLIPPED),
            &ctx,
        );
        assert!(matches!(ragged, Err(DrReplayError::Shape(_))), "{ragged:?}");

        let mut table = noiseless(52);
        table.rows.pop();
        let refused = run(&table, spec(4, 5, CLIPPED), &ctx);
        assert!(matches!(refused, Err(DrReplayError::Shape(_))), "{refused:?}");

        for poison in [f64::NAN, f64::INFINITY] {
            let mut table = noiseless(52);
            table.y[3] = poison;
            let refused = run(&table, spec(4, 5, CLIPPED), &ctx);
            assert!(matches!(refused, Err(DrReplayError::Shape(_))), "{refused:?}");
            let mut table = noiseless(52);
            table.x[7] = poison;
            let refused = run(&table, spec(4, 5, CLIPPED), &ctx);
            assert!(matches!(refused, Err(DrReplayError::Shape(_))), "{refused:?}");
        }
        let mut table = noiseless(52);
        table.t[2] = 0.5;
        let refused = run(&table, spec(4, 5, CLIPPED), &ctx);
        assert!(matches!(refused, Err(DrReplayError::Shape(_))), "{refused:?}");

        let table = noiseless(3);
        let refused = run(&table, spec(4, 5, CLIPPED), &ctx);
        assert!(matches!(refused, Err(DrReplayError::Shape(_))), "{refused:?}");
    }

    #[test]
    fn a_cancelled_context_returns_the_cancellation_case() {
        let table = noiseless(52);
        let ctx = ExecutionContext::for_tests(5);
        ctx.cancellation.cancel();
        let outcome = run(&table, spec(4, 5, CLIPPED), &ctx);
        assert!(matches!(outcome, Err(DrReplayError::Cancelled)), "{outcome:?}");
    }

    #[test]
    fn over_limit_sizes_are_refused_before_any_allocation() {
        let table = noiseless(52);
        let ctx = ExecutionContext::for_tests(5);
        let refused = run(&table, spec(MAX_REPLAY_FOLDS + 1, 5, CLIPPED), &ctx);
        assert!(matches!(refused, Err(DrReplayError::Shape(_))), "{refused:?}");

        let column: &[f64] = &table.x;
        let wide: Vec<&[f64]> = vec![column; MAX_REPLAY_COVARIATES + 1];
        let refused = replay_dr_ate(
            DrReplayInput {
                treatment: &table.t,
                outcome: &table.y,
                covariates: &wide,
                row_ids: &table.rows,
                intercept: true,
            },
            spec(4, 5, CLIPPED),
            &ctx,
        );
        assert!(matches!(refused, Err(DrReplayError::Shape(_))), "{refused:?}");

        // A pool that cannot fund the working memory refuses the replay and stays untouched.
        let pool = AdmissionPool::fixed(1, 1024);
        let tight = ExecutionContext::for_tests(5).with_admission(pool.clone());
        let refused = run(&table, spec(4, 5, CLIPPED), &tight);
        assert!(matches!(refused, Err(DrReplayError::Admission(_))), "{refused:?}");
        assert_eq!(pool.available().bytes(), 1024);
    }

    #[test]
    fn identical_inputs_give_identical_bits_and_a_new_seed_changes_the_folds() {
        let table = noiseless(60);
        let first = run(&table, spec(3, 9, CLIPPED), &ExecutionContext::for_tests(9)).unwrap();
        let second = run(&table, spec(3, 9, CLIPPED), &ExecutionContext::for_tests(9)).unwrap();
        assert_eq!(first.ate.to_bits(), second.ate.to_bits());
        assert_eq!(bits(&first.phi), bits(&second.phi));
        assert_eq!(bits(&first.cate), bits(&second.cate));
        assert_eq!(bits(&first.propensity), bits(&second.propensity));
        assert_eq!(first.fold_ids, second.fold_ids);
        assert_eq!(first.predictor, second.predictor);
        assert_eq!(first.support, second.support);
        let other = run(&table, spec(3, 10, CLIPPED), &ExecutionContext::for_tests(10)).unwrap();
        assert_ne!(first.fold_ids, other.fold_ids);
    }

    #[test]
    fn swapping_the_declared_levels_deals_different_folds() {
        let table = noiseless(60);
        let mut flipped = spec(3, 9, CLIPPED);
        flipped.active_level_above_control = false;
        let a = run(&table, spec(3, 9, CLIPPED), &ExecutionContext::for_tests(9)).unwrap();
        let b = run(&table, flipped, &ExecutionContext::for_tests(9)).unwrap();
        assert_ne!(a.fold_ids, b.fold_ids);
    }
}
