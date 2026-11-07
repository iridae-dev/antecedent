// SPDX-License-Identifier: MIT OR Apache-2.0
//! Conditional temporal effects, slopes, and nested fit moments.

use super::{
    AnalyticSeKind, Arc, AssumptionSet, EstimationError, EstimationWorkspace, GapSelection,
    HistoryScan, IdentifiedEstimand, KernelPolicy, LaggedColumn, PreparedEstimationProblem,
    SequentialRegression, TargetPopulation, TemporalEffectQuery, TemporalIndexer,
    TemporalLinearAdjustment, TimeSeriesData, VariableId, intervention_f64, offset_to_lag,
};

pub(super) fn slope_ridge_coefficients(
    design: &[f64],
    n: usize,
    ncols: usize,
    y: &[f64],
    ols_coef: &[f64],
    scale: f64,
    invalid_scale: &'static str,
) -> Result<Vec<f64>, EstimationError> {
    crate::util::validate_prior_scale(scale, invalid_scale)?;
    let sigma2 = crate::util::ols_sigma2(design, n, ncols, y, ols_coef);
    if !sigma2.is_finite() {
        return Err(EstimationError::stats_msg(
            "temporal slope prior needs a finite residual variance and positive residual degrees of freedom",
        ));
    }
    let sigma2 = sigma2.max(1e-12);
    let penalty = sigma2 / (scale * scale);
    let aug_n = n + (ncols - 1);
    let mut aug_x = vec![0.0; aug_n * ncols];
    let mut aug_y = vec![0.0; aug_n];
    for col in 0..ncols {
        aug_x[col * aug_n..col * aug_n + n].copy_from_slice(&design[col * n..(col + 1) * n]);
    }
    aug_y[..n].copy_from_slice(y);
    let root = penalty.sqrt();
    for col in 1..ncols {
        let row = n + col - 1;
        aug_x[col * aug_n + row] = root;
    }
    let mut aug_ws = antecedent_stats::LeastSquaresWorkspace::default();
    Ok(SequentialRegression::solve_design(&aug_x, aug_n, ncols, &aug_y, None, &mut aug_ws)
        .map_err(crate::error::EstimationError::from)?
        .coefficients)
}

/// Marginal temporal conditional effect over a modifier set.
///
/// The unfolded regression is `Y ~ 1 + T + W₁…W_q + T×W₁…T×W_q + Z`. The published point is
/// `(β_T + Σ_j β_{T×W_j} · mean(W_j)) · (active − control)`, the same functional as the
/// static conditional effect over its modifier set, on the lag-aligned rows of the
/// temporal backdoor design. `prior_scale` is the isotropic Gaussian prior
/// standard deviation on the slope coefficients. `None` is ordinary least
/// squares. `Some` replaces that point with the posterior mean under that
/// prior, using the least-squares residual variance as a plug-in. No interval
/// is returned.
///
/// # Errors
///
/// The query names no effect modifier (or names one twice), the design is
/// singular or too small, or the temporal sample cannot be aligned.
pub fn temporal_conditional_marginal_effect(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<f64, EstimationError> {
    temporal_conditional_marginal_effect_with_gaps(
        data,
        estimand,
        query,
        indexer,
        policy,
        prior_scale,
        false,
    )
    .map(|(effect, _)| effect)
}

/// [`temporal_conditional_marginal_effect`] on a series that may have missing time points.
///
/// With `allow_gaps` the windows that read a missing value are dropped and the surviving
/// windows keep their own times and lags (nothing is concatenated across a gap); without it a
/// missing value refuses as before.
///
/// # Errors
///
/// As [`temporal_conditional_marginal_effect`]. Also returns, with `allow_gaps`, which windows
/// were intact.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded estimation procedure keeps its related calculations together"
)]
pub fn temporal_conditional_marginal_effect_with_gaps(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
    allow_gaps: bool,
) -> Result<(f64, Option<GapSelection>), EstimationError> {
    let modifiers = query.effect_modifiers.as_ref();
    let q = modifiers.len();
    if q == 0 || modifiers.iter().enumerate().any(|(i, modifier)| modifiers[..i].contains(modifier))
    {
        return Err(EstimationError::unsupported(
            "temporal conditional effect needs at least one effect modifier, each named once",
        ));
    }
    query.validate()?;
    if data.storage().weights().is_some() {
        return Err(crate::sampling_weights::unsupported_route(
            "the conditional effect over declared modifiers centres each modifier at its window \
             mean, which a weighted series would have to centre at the weighted mean; the \
             interaction design does not read the series' sampling weights",
        ));
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::TargetPopulation);
    }
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    let delta = active - control;
    if delta == 0.0 {
        return Err(EstimationError::unsupported(
            "active and control treatment levels must differ",
        ));
    }
    let sample_anchor = query.outcome_offset()?.max(0);
    let t_lag =
        offset_to_lag(query.try_treatment_offset()?.checked_sub(sample_anchor).ok_or_else(
            || EstimationError::unsupported("treatment offset exceeds the temporal lag range"),
        )?)?;
    let y_lag =
        offset_to_lag(query.outcome_offset()?.checked_sub(sample_anchor).ok_or_else(|| {
            EstimationError::unsupported("outcome offset exceeds the temporal lag range")
        })?)?;
    let mut cols = vec![
        LaggedColumn { variable: query.treatment, lag: t_lag },
        LaggedColumn { variable: query.outcome, lag: y_lag },
    ];
    cols.extend(modifiers.iter().map(|&variable| LaggedColumn { variable, lag: y_lag }));
    let mut adjustment = Vec::new();
    for &dense_var in estimand.adjustment_set.iter() {
        let key = indexer.key_of(dense_var.raw())?;
        let lag = offset_to_lag(key.offset - sample_anchor).map_err(|_| {
            EstimationError::unsupported(
                "adjustment node sits after this horizon's sample anchor and cannot be lag-aligned",
            )
        })?;
        // Only the modifier's own column (same variable at the outcome's lag) is read once; the
        // same variable at another lag is a different adjustment node and stays.
        if modifiers.contains(&key.variable) && lag == y_lag {
            continue;
        }
        cols.push(LaggedColumn { variable: key.variable, lag });
        adjustment.push(key.variable);
    }
    let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
    let mut scan = HistoryScan::plan(data, max_lag, Arc::<[LaggedColumn]>::from(cols))
        .map_err(EstimationError::from)?;
    let (prep, complete) = scan.windows(data, policy, allow_gaps).map_err(EstimationError::from)?;
    let selection =
        complete.map(|complete| GapSelection { first_time: max_lag as usize, complete });
    let n = prep.n;
    if n < 8 {
        return Err(EstimationError::data_msg(
            "too few aligned rows for a temporal conditional effect",
        ));
    }
    let t = prep.column(0);
    let y = prep.column(1);
    let n_z = adjustment.len();
    let ncols = 2 + 2 * q + n_z;
    let mut design = vec![0.0; n * ncols];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = t[i];
    }
    for j in 0..q {
        let w = prep.column(2 + j);
        for i in 0..n {
            design[(2 + j) * n + i] = w[i];
            design[(2 + q + j) * n + i] = t[i] * w[i];
        }
    }
    for k in 0..n_z {
        let column = prep.column(2 + q + k);
        let base = (2 + 2 * q + k) * n;
        design[base..base + n].copy_from_slice(column);
    }
    let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
    let fit = SequentialRegression::solve_design(&design, n, ncols, y, None, &mut ws)
        .map_err(crate::error::EstimationError::from)?;
    let mut coef = fit.coefficients;
    if let Some(scale) = prior_scale {
        coef = slope_ridge_coefficients(
            &design,
            n,
            ncols,
            y,
            &coef,
            scale,
            "temporal conditional prior scale must be finite and positive",
        )?;
    }
    let mut slope = coef[1];
    for j in 0..q {
        let w_bar = prep.column(2 + j).iter().sum::<f64>() / n as f64;
        slope += coef[2 + q + j] * w_bar;
    }
    Ok((slope * delta, selection))
}

/// The conditional effect over a declared modifier set with its standard error, and the
/// prepared problem it was fitted from.
#[derive(Clone, Debug)]
pub struct TemporalConditionalFit {
    /// Marginal conditional effect over the modifier set, per the query's contrast.
    pub effect: f64,
    /// Delta-method standard error (rows independent; modifier-mean term included).
    pub standard_error: f64,
    /// Lag windows the regression read.
    pub rows: usize,
    /// The prepared problem: centred interaction design, fit by [`TemporalLinearAdjustment::fit`].
    pub problem: PreparedEstimationProblem,
    /// Intact windows, when gaps were allowed.
    pub selection: Option<GapSelection>,
}

/// [`temporal_conditional_marginal_effect_with_gaps`] with a standard error, through the
/// prepared-problem path of [`TemporalLinearAdjustment`].
///
/// The point is the treatment coefficient of `Y ~ 1 + T + (W - W_bar) + T (W - W_bar) + Z`,
/// scaled by the contrast, which equals the unfolded-regression functional
/// `(beta_T + sum_j beta_{T x W_j} mean(W_j)) (active - control)`. The standard error is the
/// delta-method one of [`conditional_effect_variance`]: it treats windows as independent rows
/// and includes the sampling variability of the modifier means. No ridge prior is applied.
///
/// # Errors
///
/// The query declares no modifier, a multi-step schedule, a target population other than all
/// observed units, a weighted series, a design that is singular or has too few rows, or a
/// covariance kind the conditional variance does not read.
pub fn temporal_conditional_marginal_fit(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    policy: &antecedent_core::KernelPolicy,
    allow_gaps: bool,
    se_kind: AnalyticSeKind,
) -> Result<TemporalConditionalFit, EstimationError> {
    if query.effect_modifiers.is_empty() {
        return Err(EstimationError::unsupported(
            "temporal conditional effect needs at least one effect modifier, each named once",
        ));
    }
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.inference.se_kind = se_kind;
    let (problem, selection) = if allow_gaps {
        let (problem, selection) =
            estimator.prepare_with_gaps(data, estimand, query, indexer, policy)?;
        (problem, Some(selection))
    } else {
        (estimator.prepare(data, estimand, query, indexer, None, policy)?, None)
    };
    let mut workspace = EstimationWorkspace::default();
    let estimate =
        estimator.fit_conditional_point(&problem, &mut workspace, AssumptionSet::default())?;
    Ok(TemporalConditionalFit {
        effect: estimate.ate,
        standard_error: estimate.uncertainty.analytic.se,
        rows: problem.design.nrows,
        problem,
        selection,
    })
}

/// The outcome, treatments and their time offsets of a [`temporal_linear_slopes`] fit.
#[derive(Clone, Copy, Debug)]
pub struct TemporalSlopeSpec<'a> {
    /// Outcome variable (at the outcome offset).
    pub outcome: VariableId,
    /// Treatments regressed on, none equal to the outcome.
    pub treatments: &'a [VariableId],
    /// Offset of the treatments from the policy origin.
    pub treatment_offset: i32,
    /// Offset of the outcome from the policy origin.
    pub outcome_offset: i32,
    /// Treatments read at the outcome's own lag rather than the treatment offset.
    pub contemporaneous: &'a [VariableId],
}

/// Intercept and partial slopes of one outcome on the named treatments.
///
/// The design is the unfolded temporal back-door regression: intercept, each
/// treatment at the policy origin, then the identified adjustment nodes.
/// `prior_scale` replaces the slope coefficients with their posterior means
/// under independent `N(0, scale²)` priors and the OLS residual variance; the
/// intercept stays unpenalized. No interval is computed.
///
/// # Errors
///
/// Refuses an empty treatment list, a treatment equal to the outcome, a
/// non-positive prior scale, fewer than eight aligned rows, or a lag the
/// sample cannot place.
pub fn temporal_linear_slopes(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    spec: &TemporalSlopeSpec<'_>,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<(f64, Vec<f64>), EstimationError> {
    let TemporalSlopeSpec {
        outcome,
        treatments,
        treatment_offset,
        outcome_offset,
        contemporaneous,
    } = *spec;
    if treatments.is_empty() || treatments.contains(&outcome) {
        return Err(EstimationError::unsupported(
            "temporal linear slope needs treatments distinct from the outcome",
        ));
    }
    let sample_anchor = outcome_offset.max(0);
    let t_lag = offset_to_lag(treatment_offset - sample_anchor)?;
    let y_lag = offset_to_lag(outcome_offset - sample_anchor)?;
    let mut cols = Vec::with_capacity(treatments.len() + 1);
    for &treatment in treatments {
        let lag = if contemporaneous.contains(&treatment) { y_lag } else { t_lag };
        cols.push(LaggedColumn { variable: treatment, lag });
    }
    cols.push(LaggedColumn { variable: outcome, lag: y_lag });
    let treatment_count = treatments.len();
    let mut adjustment = Vec::new();
    for &dense_var in estimand.adjustment_set.iter() {
        let key = indexer.key_of(dense_var.raw())?;
        if key.variable == outcome || treatments.contains(&key.variable) {
            continue;
        }
        let lag = offset_to_lag(key.offset - sample_anchor).map_err(|_| {
            EstimationError::unsupported(
                "adjustment node sits after this horizon's sample anchor and cannot be lag-aligned",
            )
        })?;
        cols.push(LaggedColumn { variable: key.variable, lag });
        adjustment.push(key.variable);
    }
    let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
    let mut scan = HistoryScan::plan(data, max_lag, Arc::<[LaggedColumn]>::from(cols))
        .map_err(EstimationError::from)?;
    let prep = scan.complete(data, policy).map_err(EstimationError::from)?;
    let n = prep.n;
    if n < 8 {
        return Err(EstimationError::data_msg("too few aligned rows for a temporal linear slope"));
    }
    let y = prep.column(treatment_count);
    let n_z = adjustment.len();
    let ncols = 1 + treatment_count + n_z;
    let mut design = vec![0.0; n * ncols];
    design[..n].fill(1.0);
    for k in 0..treatment_count {
        let column = prep.column(k);
        design[(1 + k) * n..(2 + k) * n].copy_from_slice(column);
    }
    for k in 0..n_z {
        let column = prep.column(treatment_count + 1 + k);
        let base = (1 + treatment_count + k) * n;
        design[base..base + n].copy_from_slice(column);
    }
    let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
    let fit = SequentialRegression::solve_design(&design, n, ncols, y, None, &mut ws)
        .map_err(crate::error::EstimationError::from)?;
    let mut coef = fit.coefficients;
    if let Some(scale) = prior_scale {
        coef = slope_ridge_coefficients(
            &design,
            n,
            ncols,
            y,
            &coef,
            scale,
            "temporal slope prior scale must be finite and positive",
        )?;
    }
    Ok((coef[0], coef[1..=treatment_count].to_vec()))
}

/// Moments of the nested natural-direct-effect regression on a time series.
///
/// The design is the same empty-adjustment regression
/// [`temporal_linear_slopes`] fits for a nested point: intercept, treatment
/// one step before the outcome, and the mediator at the outcome time. A
/// positive `prior_scale` adds the slope-only ridge that function uses, so
/// the intercept moment stays the aligned row count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestedDirectFitMoments {
    /// Aligned rows that entered the gram.
    pub complete_case_rows: u64,
    /// Row-major gram of `[1, treatment, mediator]`.
    pub gram: [f64; 9],
    /// Cross product of that design with the outcome.
    pub outcome_cross: [f64; 3],
}

/// Outcome-regression moments behind a temporal nested natural direct effect.
///
/// # Errors
///
/// Same row, lag, and prior-scale refusals as [`temporal_linear_slopes`] for
/// this two-treatment design, plus a non-finite aligned column.
pub fn nested_direct_fit_moments(
    data: &TimeSeriesData,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    policy: &KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<NestedDirectFitMoments, EstimationError> {
    if treatment == outcome || mediator == outcome || treatment == mediator {
        return Err(EstimationError::unsupported(
            "temporal nested moments need distinct treatment, mediator, and outcome",
        ));
    }
    let t_lag = offset_to_lag(-1)?;
    let y_lag = offset_to_lag(0)?;
    let cols = [
        LaggedColumn { variable: treatment, lag: t_lag },
        LaggedColumn { variable: mediator, lag: y_lag },
        LaggedColumn { variable: outcome, lag: y_lag },
    ];
    let mut scan = HistoryScan::plan(data, 1, Arc::<[LaggedColumn]>::from(cols))
        .map_err(EstimationError::from)?;
    let prep = scan.complete(data, policy).map_err(EstimationError::from)?;
    let n = prep.n;
    if n < 8 {
        return Err(EstimationError::data_msg("too few aligned rows for a temporal linear slope"));
    }
    let y = prep.column(2);
    let treatment_column = prep.column(0);
    let mediator_column = prep.column(1);
    if y.iter().chain(treatment_column).chain(mediator_column).any(|value| !value.is_finite()) {
        return Err(EstimationError::data_msg(
            "temporal nested moments require finite treatment, mediator, and outcome rows",
        ));
    }
    let ncols = 3;
    let mut design = vec![0.0; n * ncols];
    design[..n].fill(1.0);
    design[n..2 * n].copy_from_slice(treatment_column);
    design[2 * n..3 * n].copy_from_slice(mediator_column);
    let mut gram = [0.0_f64; 9];
    let mut outcome_cross = [0.0_f64; 3];
    for i in 0..n {
        let row = [design[i], design[n + i], design[2 * n + i]];
        for left in 0..3 {
            outcome_cross[left] += row[left] * y[i];
            for right in 0..3 {
                gram[left * 3 + right] += row[left] * row[right];
            }
        }
    }
    if let Some(scale) = prior_scale {
        crate::util::validate_prior_scale(
            scale,
            "temporal slope prior scale must be finite and positive",
        )?;
        let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
        let coef = SequentialRegression::solve_design(&design, n, ncols, y, None, &mut ws)
            .map_err(crate::error::EstimationError::from)?
            .coefficients;
        let sigma2 = crate::util::ols_sigma2(&design, n, ncols, y, &coef);
        if !sigma2.is_finite() {
            return Err(EstimationError::stats_msg(
                "temporal slope prior needs a finite residual variance and positive residual degrees of freedom",
            ));
        }
        let sigma2 = sigma2.max(1e-12);
        let penalty = sigma2 / (scale * scale);
        gram[4] += penalty;
        gram[8] += penalty;
    }
    let Ok(complete_case_rows) = u64::try_from(n) else {
        return Err(EstimationError::data_msg("aligned row count does not fit the moment record"));
    };
    Ok(NestedDirectFitMoments { complete_case_rows, gram, outcome_cross })
}
