//! The value of whole static treatment regimes on a continuous end-of-follow-up outcome under
//! time-varying confounding, from observational subject trajectories.
//!
//! # Target and identification
//!
//! Each subject is observed in the same `K` periods. In period `k` the confounders `L_k` are
//! measured, then a binary treatment `A_k` is taken; the outcome `Y` is measured once, at the end
//! of follow-up, after the last treatment. A static regime is a whole treatment path
//! `a = (a_0, ..., a_{K-1})`, and the target is the mean outcome had every subject followed it,
//! `E[Y^a]`, together with the contrast of an active path against a reference path. Past
//! treatment shapes later confounders, so adjusting for the history in one regression does not
//! identify it; the g-formula does. The target is identified by sequential exchangeability
//! (`Y^a` independent of `A_k` given the measured history `(L_0..L_k, A_0..A_{k-1})` in every
//! period), sequential positivity (`P(A_k = a_k | history) > 0` wherever the history is
//! consistent with the path), consistency, no interference between subjects and complete
//! follow-up. No unmeasured time-varying confounder may influence both a later treatment and the
//! outcome. These are declared, not checked.
//!
//! # Estimators
//!
//! * [`RegimeMethod::GFormula`] is the iterated-conditional-expectation g-formula: working
//!   backwards from `Y`, the pseudo-outcome of period `k + 1` is regressed on the history and
//!   treatment of period `k` (a linear model with the treatment-by-current-confounder
//!   interaction), and its prediction with `A_k` set to `a_k` becomes the pseudo-outcome of
//!   period `k`.
//! * [`RegimeMethod::InverseProbability`] is the saturated marginal structural model for one path:
//!   subjects whose treatments follow the path are weighted by the inverse of the product of the
//!   fitted probabilities of their path actions (one logistic model per period on the history), and the mean is the weight-normalized average outcome.
//! * [`RegimeMethod::DoublyRobust`] is the recursive augmented estimator: the regression
//!   prediction of each period is corrected by the inverse-probability-weighted residual of the
//!   pseudo-outcome of the next, which is consistent when either the outcome regressions or the
//!   treatment models are correct.
//!
//! # Uncertainty
//!
//! The nuisance fits are part of the procedure, so both intervals refit all of them. The
//! frequentist interval is the percentile interval of a unit bootstrap that refits every
//! treatment and outcome model in each replicate (at least 99 replicates; more than one percent
//! failed replicates refuse). The Bayesian posterior is the weighted-likelihood bootstrap: each
//! draw takes Dirichlet weights over units and refits every model under those weights, with a
//! Gaussian coefficient prior of the declared scale on the treatment models. Neither is
//! calibrated: no coverage of these intervals has been measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use super::linalg::{cholesky, cholesky_solve, fit_logistic};
use super::{MIN_CLUSTERS, PosteriorSummary, dense_labels, invalid, posterior_by_cluster_weights};
use crate::error::EstimationError;
use crate::util::refuse;

/// Stream key of the unit bootstrap.
const BOOTSTRAP_STREAM: u64 = 0x7E5D_4E61_0000_0000;
/// Fewest bootstrap replicates of the whole-procedure interval.
pub const MIN_REPLICATES: usize = 99;
/// Largest share of bootstrap replicates allowed to fail before the interval is refused.
const MAX_FAILED_SHARE: f64 = 0.01;
/// Ridge of the outcome regressions; far below any data information, not a prior.
const OUTCOME_RIDGE: f64 = 1e-6;

/// Coordinate names of a regime contrast.
pub const REGIME_COORDINATES: [&str; 3] = ["regime_effect", "active_mean", "reference_mean"];

/// How the value of a regime is estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegimeMethod {
    /// Iterated-conditional-expectation g-formula.
    GFormula,
    /// Inverse-probability-weighted saturated marginal structural model of one path.
    InverseProbability,
    /// Recursive augmented inverse-probability estimator.
    DoublyRobust,
}

impl RegimeMethod {
    /// A short stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::GFormula => "g_formula",
            Self::InverseProbability => "inverse_probability_weighting",
            Self::DoublyRobust => "doubly_robust",
        }
    }

    const fn needs_treatment_models(self) -> bool {
        !matches!(self, Self::GFormula)
    }
}

/// Row-aligned subject trajectories.
#[derive(Clone, Copy, Debug)]
pub struct RegimeRows<'a> {
    /// Subject of each row.
    pub unit: &'a [usize],
    /// Period of each row; every subject has the same periods.
    pub period: &'a [i64],
    /// Outcome of each row; only the row of a subject's last period is read.
    pub outcome: &'a [f64],
    /// Treatment taken in the period.
    pub treated: &'a [bool],
    /// Confounder columns measured at the start of the period, each row-aligned.
    pub confounders: &'a [Vec<f64>],
}

/// The two static treatment paths of a contrast, one action per period.
#[derive(Clone, Copy, Debug)]
pub struct StaticRegimes<'a> {
    /// The active path.
    pub active: &'a [bool],
    /// The reference path.
    pub reference: &'a [bool],
}

/// Validated, subject-major trajectories.
#[derive(Clone, Debug)]
pub struct RegimePanel {
    units: usize,
    periods: usize,
    covariates: usize,
    /// `treated[unit * periods + k]`.
    treated: Vec<bool>,
    /// `covariate[(unit * periods + k) * covariates + c]`.
    covariate: Vec<f64>,
    /// End-of-follow-up outcome of each subject.
    outcome: Vec<f64>,
}

impl RegimePanel {
    /// Group the rows of every subject by period.
    ///
    /// # Errors
    /// Misaligned rows, a non-finite confounder or final outcome, fewer than two subjects, or
    /// subjects observed in different periods (the outcome is read at the end of a complete
    /// follow-up).
    pub fn new(rows: RegimeRows<'_>) -> Result<Self, EstimationError> {
        let n = rows.unit.len();
        if n == 0
            || rows.period.len() != n
            || rows.outcome.len() != n
            || rows.treated.len() != n
            || rows.confounders.iter().any(|column| column.len() != n)
        {
            return Err(invalid("trajectory arrays must be aligned and non-empty"));
        }
        if rows.confounders.iter().flatten().any(|value| !value.is_finite()) {
            return Err(invalid("every confounder value must be finite"));
        }
        let (dense, units) = dense_labels(rows.unit);
        if units < MIN_CLUSTERS {
            return Err(invalid("the regime design needs at least two subjects"));
        }
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); units];
        for (row, &subject) in dense.iter().enumerate() {
            members[subject].push(row);
        }
        for subject in &mut members {
            subject.sort_by_key(|&row| rows.period[row]);
        }
        let schedule: Vec<i64> = members[0].iter().map(|&row| rows.period[row]).collect();
        let periods = schedule.len();
        if schedule.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid("each subject needs one row per period"));
        }
        let covariates = rows.confounders.len();
        let mut treated = Vec::with_capacity(units * periods);
        let mut covariate = Vec::with_capacity(units * periods * covariates);
        let mut outcome = Vec::with_capacity(units);
        for subject in &members {
            if subject.len() != periods
                || subject.iter().zip(&schedule).any(|(&row, &period)| rows.period[row] != period)
            {
                return Err(invalid(
                    "every subject must be observed in the same periods: the outcome is read at \
                     the end of a complete follow-up",
                ));
            }
            for &row in subject {
                treated.push(rows.treated[row]);
                for column in rows.confounders {
                    covariate.push(column[row]);
                }
            }
            let last = rows.outcome[subject[periods - 1]];
            if !last.is_finite() {
                return Err(invalid("the end-of-follow-up outcome must be finite"));
            }
            outcome.push(last);
        }
        Ok(Self { units, periods, covariates, treated, covariate, outcome })
    }

    /// Number of subjects.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// Number of periods per subject.
    #[must_use]
    pub fn periods(&self) -> usize {
        self.periods
    }

    fn action(&self, unit: usize, k: usize) -> bool {
        self.treated[unit * self.periods + k]
    }

    fn confounders_at(&self, unit: usize, k: usize) -> &[f64] {
        let start = (unit * self.periods + k) * self.covariates;
        &self.covariate[start..start + self.covariates]
    }

    /// Width of the treatment model of period `k`: intercept, every confounder up to `k` and
    /// every earlier treatment.
    fn treatment_width(&self, k: usize) -> usize {
        1 + (k + 1) * self.covariates + k
    }

    /// Width of the outcome regression of period `k`: the treatment-model history, the period's
    /// treatment and its interaction with the period's confounders.
    fn outcome_width(&self, k: usize) -> usize {
        self.treatment_width(k) + 1 + self.covariates
    }

    fn push_history(&self, unit: usize, k: usize, out: &mut Vec<f64>) {
        out.push(1.0);
        for j in 0..=k {
            out.extend_from_slice(self.confounders_at(unit, j));
        }
        for j in 0..k {
            out.push(f64::from(u8::from(self.action(unit, j))));
        }
    }

    fn push_outcome_row(&self, unit: usize, k: usize, treated: bool, out: &mut Vec<f64>) {
        self.push_history(unit, k, out);
        let a = f64::from(u8::from(treated));
        out.push(a);
        out.extend(self.confounders_at(unit, k).iter().map(|value| a * value));
    }
}

/// A fitted regime contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeFit {
    /// `[active - reference, active, reference]` at the full-data fit.
    pub estimate: [f64; 3],
    /// Bootstrap standard deviation of each coordinate (frequentist).
    pub standard_error: Option<[f64; 3]>,
    /// Percentile interval of each coordinate (frequentist).
    pub interval: Option<[(f64, f64); 3]>,
    /// Replicates that succeeded, and those that failed.
    pub replicates: (usize, usize),
    /// Posterior (Bayesian).
    pub posterior: Option<PosteriorSummary>,
    /// Smallest fitted probability of a path action among the units the estimator uses; not a
    /// number for the g-formula, which fits no treatment model.
    pub minimum_probability: f64,
    /// Subjects.
    pub units: usize,
    /// Periods per subject.
    pub periods: usize,
}

/// Weighted least squares with a vanishing ridge.
fn weighted_least_squares(
    x: &[f64],
    p: usize,
    y: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let mut gram = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for (i, (&target, &w)) in y.iter().zip(weights).enumerate() {
        if w <= 0.0 {
            continue;
        }
        let row = &x[i * p..(i + 1) * p];
        for a in 0..p {
            xty[a] += w * row[a] * target;
            for b in 0..p {
                gram[a * p + b] += w * row[a] * row[b];
            }
        }
    }
    for a in 0..p {
        gram[a * p + a] += OUTCOME_RIDGE;
    }
    let factor = cholesky(&gram, p).ok_or_else(|| {
        EstimationError::stats_msg(
            "an outcome regression is singular (collinear history or too few subjects)",
        )
    })?;
    Ok(cholesky_solve(&factor, p, &xty))
}

fn dot(row: &[f64], coefficients: &[f64]) -> f64 {
    row.iter().zip(coefficients).map(|(a, b)| a * b).sum()
}

fn positivity_error(arm: f64, floor: f64) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("positivity_violated"),
        "temporal_design.positivity",
        &format!(
            "a fitted probability {arm} of a regime action is below the declared floor {floor}. \
             Next attempt: a regime the history supports, fewer or better-scaled history columns, \
             or a lower declared floor"
        ),
    )
}

/// `P(A_k = 1 | history)` of every subject and period, from one logistic model per period.
fn fit_treatment_models(
    panel: &RegimePanel,
    weights: &[f64],
    ridge: f64,
) -> Result<Vec<Vec<f64>>, EstimationError> {
    let n = panel.units;
    let mut probabilities = Vec::with_capacity(panel.periods);
    for k in 0..panel.periods {
        let width = panel.treatment_width(k);
        let mut x = Vec::with_capacity(n * width);
        let mut y = Vec::with_capacity(n);
        for unit in 0..n {
            panel.push_history(unit, k, &mut x);
            y.push(f64::from(u8::from(panel.action(unit, k))));
        }
        let fit = fit_logistic(&x, width, &y, Some(weights), ridge)?;
        probabilities.push(
            (0..n)
                .map(|unit| {
                    antecedent_kernels::sigmoid(dot(
                        &x[unit * width..(unit + 1) * width],
                        &fit.coefficients,
                    ))
                })
                .collect(),
        );
    }
    Ok(probabilities)
}

struct RegimeValue {
    mean: f64,
    minimum_probability: f64,
}

fn arm_probability(treatment: &[Vec<f64>], k: usize, unit: usize, action: bool) -> f64 {
    let p = treatment[k][unit];
    if action { p } else { 1.0 - p }
}

/// The weight-normalized mean outcome of the subjects who followed the path.
fn inverse_probability_mean(
    panel: &RegimePanel,
    regime: &[bool],
    weights: &[f64],
    treatment: &[Vec<f64>],
    floor: f64,
) -> Result<RegimeValue, EstimationError> {
    let (mut numerator, mut denominator) = (0.0, 0.0);
    let mut minimum = 1.0_f64;
    let weights = weights.get(..panel.units).ok_or_else(|| {
        EstimationError::unsupported("regime weights do not cover every panel unit")
    })?;
    for (unit, &w) in weights.iter().enumerate() {
        if w <= 0.0 {
            continue;
        }
        let mut cumulative = 1.0;
        let mut follower = true;
        for (k, &action) in regime.iter().enumerate() {
            if panel.action(unit, k) != action {
                follower = false;
                break;
            }
            let arm = arm_probability(treatment, k, unit, action);
            if arm.is_nan() || arm < floor {
                return Err(positivity_error(arm, floor));
            }
            minimum = minimum.min(arm);
            cumulative /= arm;
        }
        if follower {
            numerator += w * cumulative * panel.outcome[unit];
            denominator += w * cumulative;
        }
    }
    if denominator.is_nan() || denominator <= 0.0 {
        return Err(refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "temporal_design.arm_not_populated",
            "no subject followed the regime through every period, so its value is not estimable",
        ));
    }
    Ok(RegimeValue { mean: numerator / denominator, minimum_probability: minimum })
}

/// The g-formula and doubly robust recursions over the periods, last to first.
fn regression_mean(
    panel: &RegimePanel,
    regime: &[bool],
    method: RegimeMethod,
    weights: &[f64],
    treatment: Option<&[Vec<f64>]>,
    floor: f64,
) -> Result<RegimeValue, EstimationError> {
    let n = panel.units;
    let mut pseudo = panel.outcome.clone();
    let mut minimum = f64::NAN;
    for k in (0..panel.periods).rev() {
        let width = panel.outcome_width(k);
        let mut x = Vec::with_capacity(n * width);
        for unit in 0..n {
            panel.push_outcome_row(unit, k, panel.action(unit, k), &mut x);
        }
        let coefficients = weighted_least_squares(&x, width, &pseudo, weights)?;
        let mut next = vec![0.0; n];
        let mut row = Vec::with_capacity(width);
        for unit in 0..n {
            row.clear();
            panel.push_outcome_row(unit, k, regime[k], &mut row);
            let prediction = dot(&row, &coefficients);
            next[unit] = prediction;
            if method == RegimeMethod::DoublyRobust
                && weights[unit] > 0.0
                && panel.action(unit, k) == regime[k]
            {
                let arm = arm_probability(treatment.unwrap_or(&[]), k, unit, regime[k]);
                if arm.is_nan() || arm < floor {
                    return Err(positivity_error(arm, floor));
                }
                minimum = if minimum.is_nan() { arm } else { minimum.min(arm) };
                next[unit] = prediction + (pseudo[unit] - prediction) / arm;
            }
        }
        pseudo = next;
    }
    let total: f64 = weights.iter().sum();
    let weighted: f64 = pseudo.iter().zip(weights).map(|(value, w)| value * w).sum();
    Ok(RegimeValue { mean: weighted / total, minimum_probability: minimum })
}

struct Evaluation {
    values: [f64; 3],
    minimum_probability: f64,
}

/// Fit every model under unit weights and evaluate both paths.
fn evaluate(
    panel: &RegimePanel,
    regimes: StaticRegimes<'_>,
    method: RegimeMethod,
    unit_weights: Option<&[f64]>,
    ridge: f64,
    floor: f64,
) -> Result<Evaluation, EstimationError> {
    let ones;
    let weights: &[f64] = if let Some(weights) = unit_weights {
        weights
    } else {
        ones = vec![1.0; panel.units];
        &ones
    };
    let treatment = if method.needs_treatment_models() {
        Some(fit_treatment_models(panel, weights, ridge)?)
    } else {
        None
    };
    let mut values = [0.0; 3];
    let mut minimum = f64::NAN;
    for (slot, regime) in [regimes.active, regimes.reference].into_iter().enumerate() {
        let value = match method {
            RegimeMethod::InverseProbability => inverse_probability_mean(
                panel,
                regime,
                weights,
                treatment.as_deref().unwrap_or(&[]),
                floor,
            )?,
            RegimeMethod::GFormula | RegimeMethod::DoublyRobust => {
                regression_mean(panel, regime, method, weights, treatment.as_deref(), floor)?
            }
        };
        values[1 + slot] = value.mean;
        minimum = minimum.min(value.minimum_probability);
    }
    values[0] = values[1] - values[2];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::stats_msg("the regime contrast is not finite"));
    }
    Ok(Evaluation { values, minimum_probability: minimum })
}

fn check_regimes(panel: &RegimePanel, regimes: StaticRegimes<'_>) -> Result<(), EstimationError> {
    if regimes.active.len() != panel.periods || regimes.reference.len() != panel.periods {
        return Err(invalid(&format!(
            "each regime names one action per period: the panel has {} periods",
            panel.periods
        )));
    }
    if regimes.active == regimes.reference {
        return Err(invalid("the active and reference regimes must differ in some period"));
    }
    Ok(())
}

fn check_floor(floor: f64) -> Result<(), EstimationError> {
    if floor.is_finite() && floor > 0.0 && floor <= 0.5 {
        Ok(())
    } else {
        Err(invalid("the positivity floor must be in (0, 0.5]"))
    }
}

/// The full-data estimate of `[active - reference, active, reference]` and the smallest fitted
/// probability of a path action, with no uncertainty.
///
/// # Errors
/// Regimes of the wrong length or equal, a floor outside `(0, 0.5]`, a fitted probability below
/// the floor, a singular regression or logistic fit, or a path no subject followed.
pub fn regime_point_estimate(
    panel: &RegimePanel,
    regimes: StaticRegimes<'_>,
    method: RegimeMethod,
    floor: f64,
) -> Result<([f64; 3], f64), EstimationError> {
    check_floor(floor)?;
    check_regimes(panel, regimes)?;
    let point = evaluate(panel, regimes, method, None, 0.0, floor)?;
    Ok((point.values, point.minimum_probability))
}

/// Fit the contrast of two static regimes.
///
/// `posterior_draws` selects the weighted-likelihood bootstrap with a Gaussian coefficient prior
/// of scale `prior_scale` on the treatment models; otherwise `replicates` whole-procedure unit
/// bootstrap replicates give the percentile interval and the treatment models are fitted by
/// maximum likelihood.
///
/// # Errors
/// As [`regime_point_estimate`], plus fewer than [`MIN_REPLICATES`] replicates, more than one
/// percent failed replicates, a non-positive prior scale, admission or cancellation.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one fit: the full-data estimate, then the bootstrap or posterior loop"
)]
pub fn fit_static_regimes(
    panel: &RegimePanel,
    regimes: StaticRegimes<'_>,
    method: RegimeMethod,
    floor: f64,
    prior_scale: f64,
    replicates: usize,
    posterior_draws: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<RegimeFit, EstimationError> {
    check_floor(floor)?;
    check_regimes(panel, regimes)?;
    let (units, periods) = (panel.units, panel.periods);
    if let Some(draws) = posterior_draws {
        if !(prior_scale.is_finite() && prior_scale > 0.0) {
            return Err(invalid("the prior scale must be positive"));
        }
        let ridge = 1.0 / (prior_scale * prior_scale);
        let point = evaluate(panel, regimes, method, None, ridge, floor)?;
        let posterior = posterior_by_cluster_weights(units, 3, draws, ctx, |weights| {
            evaluate(panel, regimes, method, Some(weights), ridge, floor)
                .map(|evaluation| evaluation.values.to_vec())
        })
        .map(|mut summary| {
            summary.method = "weighted_likelihood_bootstrap_refitting_regime_models";
            summary
        })?;
        return Ok(RegimeFit {
            estimate: point.values,
            standard_error: None,
            interval: None,
            replicates: (0, 0),
            posterior: Some(posterior),
            minimum_probability: point.minimum_probability,
            units,
            periods,
        });
    }
    if replicates < MIN_REPLICATES {
        return Err(refuse(
            antecedent_core::reason_code!("required_option_missing"),
            "temporal_design.bootstrap_replicates",
            "the regime interval refits every treatment and outcome model in each unit-bootstrap \
             replicate and needs at least 99 replicates. Next attempt: set bootstrap_replicates \
             to 99 or more",
        ));
    }
    let point = evaluate(panel, regimes, method, None, 0.0, floor)?;
    let _lease = crate::util::admit_array::<f64>(ctx, replicates, 3 + units)?;
    let mut columns: [Vec<f64>; 3] = [
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
    ];
    let mut counts = vec![0.0; units];
    let mut failed = 0_usize;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, BOOTSTRAP_STREAM | replicate as u64);
        counts.iter_mut().for_each(|value| *value = 0.0);
        for _ in 0..units {
            counts[antecedent_kernels::unbiased_index(&mut rng, units)] += 1.0;
        }
        match evaluate(panel, regimes, method, Some(&counts), 0.0, floor) {
            Ok(evaluation) => {
                for (column, value) in columns.iter_mut().zip(evaluation.values) {
                    column.push(value);
                }
            }
            Err(_) => failed += 1,
        }
    }
    if failed as f64 > MAX_FAILED_SHARE * replicates as f64 {
        return Err(refuse(
            antecedent_core::reason_code!("positivity_violated"),
            "temporal_design.bootstrap_failures",
            &format!(
                "{failed} of {replicates} unit-bootstrap replicates could not refit the regime \
                 models (positivity, separation or an unpopulated path). Next attempt: more \
                 subjects, fewer or better-scaled history columns, or a lower floor"
            ),
        ));
    }
    let mut standard_error = [0.0; 3];
    let mut interval = [(0.0, 0.0); 3];
    for (j, column) in columns.iter_mut().enumerate() {
        standard_error[j] = crate::util::sample_std(column);
        column.sort_by(f64::total_cmp);
        interval[j] = antecedent_stats::equal_tail_interval_sorted(
            column,
            super::NOMINAL_LEVEL,
            antecedent_stats::QuantileRule::ExchangeableRank,
        );
    }
    Ok(RegimeFit {
        estimate: point.values,
        standard_error: Some(standard_error),
        interval: Some(interval),
        replicates: (replicates - failed, failed),
        posterior: None,
        minimum_probability: point.minimum_probability,
        units,
        periods,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x6A09_E667_F3BC_C908;
        state ^= state >> 31;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    struct Trajectories {
        unit: Vec<usize>,
        period: Vec<i64>,
        outcome: Vec<f64>,
        treated: Vec<bool>,
        confounder: Vec<f64>,
    }

    /// Two periods with treatment-confounder feedback. `L0` is centered uniform and drives
    /// `A0`; `A0` raises `L1`; `L1` drives `A1`; the outcome reads both treatments, both
    /// confounders and noise. Under the structural equations
    /// `E[Y^{a0,a1}] = 0.5 + (1 + 1.5 * 0.8) a0 + 2 a1`, because `E[L0] = 0` and the noise terms
    /// are centered, so the truth is derived from the equations, not from any estimator.
    fn simulate(units: u64) -> Trajectories {
        let mut out = Trajectories {
            unit: Vec::new(),
            period: Vec::new(),
            outcome: Vec::new(),
            treated: Vec::new(),
            confounder: Vec::new(),
        };
        for i in 0..units {
            let key = i * 16;
            let l0 = 2.0 * uniform(key) - 1.0;
            let a0 = uniform(key + 1) < antecedent_kernels::sigmoid(1.5 * l0);
            let a0_value = f64::from(u8::from(a0));
            let l1 = 0.6 * l0 + 0.8 * a0_value + (uniform(key + 2) - 0.5);
            let a1 = uniform(key + 3) < antecedent_kernels::sigmoid(1.2 * (l1 - 0.4));
            let a1_value = f64::from(u8::from(a1));
            let y = 0.5
                + a0_value
                + 2.0 * a1_value
                + 1.5 * l1
                + 2.0 * l0
                + 0.6 * (uniform(key + 4) - 0.5);
            let unit = usize::try_from(i).unwrap();
            out.unit.extend([unit, unit]);
            out.period.extend([0, 1]);
            out.outcome.extend([0.0, y]);
            out.treated.extend([a0, a1]);
            out.confounder.extend([l0, l1]);
        }
        out
    }

    fn truth(a0: bool, a1: bool) -> f64 {
        0.5 + 2.2 * f64::from(u8::from(a0)) + 2.0 * f64::from(u8::from(a1))
    }

    fn panel_of(data: &Trajectories) -> RegimePanel {
        let confounders = vec![data.confounder.clone()];
        RegimePanel::new(RegimeRows {
            unit: &data.unit,
            period: &data.period,
            outcome: &data.outcome,
            treated: &data.treated,
            confounders: &confounders,
        })
        .unwrap()
    }

    fn naive_always_versus_never(data: &Trajectories) -> f64 {
        let (mut treated_sum, mut treated_n, mut control_sum, mut control_n) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..data.unit.len() / 2 {
            let (a0, a1, y) =
                (data.treated[2 * i], data.treated[2 * i + 1], data.outcome[2 * i + 1]);
            if a0 && a1 {
                treated_sum += y;
                treated_n += 1.0;
            } else if !a0 && !a1 {
                control_sum += y;
                control_n += 1.0;
            }
        }
        treated_sum / treated_n - control_sum / control_n
    }

    const ALWAYS: [bool; 2] = [true, true];
    const NEVER: [bool; 2] = [false, false];

    #[test]
    fn every_estimator_recovers_the_regime_contrast_that_a_naive_contrast_misses() {
        let data = simulate(4000);
        let panel = panel_of(&data);
        let target = truth(true, true) - truth(false, false);
        assert!((naive_always_versus_never(&data) - target).abs() > 0.5);
        let regimes = StaticRegimes { active: &ALWAYS, reference: &NEVER };
        for (method, tolerance) in [
            (RegimeMethod::GFormula, 0.25),
            (RegimeMethod::DoublyRobust, 0.35),
            (RegimeMethod::InverseProbability, 0.8),
        ] {
            let (estimate, minimum) = regime_point_estimate(&panel, regimes, method, 0.01).unwrap();
            assert!((estimate[0] - target).abs() < tolerance, "{method:?} {estimate:?}");
            assert!((estimate[1] - truth(true, true)).abs() < tolerance + 0.2, "{estimate:?}");
            assert!((estimate[2] - truth(false, false)).abs() < tolerance + 0.2, "{estimate:?}");
            if method == RegimeMethod::GFormula {
                assert!(minimum.is_nan());
            } else {
                assert!(minimum > 0.01 && minimum < 0.5, "{minimum}");
            }
        }
    }

    #[test]
    fn a_path_that_switches_treatment_is_a_different_regime() {
        let data = simulate(4000);
        let panel = panel_of(&data);
        let switch_on = [false, true];
        let switch_off = [true, false];
        let regimes = StaticRegimes { active: &switch_off, reference: &switch_on };
        let (estimate, _) =
            regime_point_estimate(&panel, regimes, RegimeMethod::GFormula, 0.01).unwrap();
        let target = truth(true, false) - truth(false, true);
        assert!((estimate[0] - target).abs() < 0.3, "{estimate:?} vs {target}");
        let (doubly, _) =
            regime_point_estimate(&panel, regimes, RegimeMethod::DoublyRobust, 0.01).unwrap();
        assert!((doubly[0] - target).abs() < 0.5, "{doubly:?} vs {target}");
    }

    #[test]
    fn the_frequentist_interval_refits_every_model_and_brackets_the_truth() {
        let data = simulate(1200);
        let panel = panel_of(&data);
        let regimes = StaticRegimes { active: &ALWAYS, reference: &NEVER };
        let ctx = ExecutionContext::for_tests(11);
        let target = truth(true, true) - truth(false, false);
        let fit = fit_static_regimes(
            &panel,
            regimes,
            RegimeMethod::DoublyRobust,
            0.01,
            1.0,
            99,
            None,
            &ctx,
        )
        .unwrap();
        let se = fit.standard_error.unwrap()[0];
        let (low, high) = fit.interval.unwrap()[0];
        assert!(se > 0.0 && low < fit.estimate[0] && fit.estimate[0] < high);
        assert!((fit.estimate[0] - target).abs() < 4.0 * se + 0.1, "{:?} se {se}", fit.estimate);
        assert_eq!(fit.replicates.0 + fit.replicates.1, 99);
        assert_eq!((fit.units, fit.periods), (1200, 2));
    }

    #[test]
    fn the_posterior_repeats_under_the_seed_and_centers_near_the_truth() {
        let data = simulate(800);
        let panel = panel_of(&data);
        let regimes = StaticRegimes { active: &ALWAYS, reference: &NEVER };
        let ctx = ExecutionContext::for_tests(5);
        let target = truth(true, true) - truth(false, false);
        let first = fit_static_regimes(
            &panel,
            regimes,
            RegimeMethod::GFormula,
            0.01,
            5.0,
            0,
            Some(80),
            &ctx,
        )
        .unwrap();
        let second = fit_static_regimes(
            &panel,
            regimes,
            RegimeMethod::GFormula,
            0.01,
            5.0,
            0,
            Some(80),
            &ctx,
        )
        .unwrap();
        assert_eq!(first.posterior, second.posterior);
        let posterior = first.posterior.unwrap();
        assert!((posterior.mean[0] - target).abs() < 0.4, "{:?}", posterior.mean);
        assert!(posterior.standard_deviation[0] > 0.0);
        assert!(posterior.lower[0] < posterior.mean[0] && posterior.mean[0] < posterior.upper[0]);
    }

    #[test]
    fn the_treatment_history_changes_the_adjusted_answer() {
        // Removing the feedback (A0 no longer moves L1) must change what the estimator reports,
        // so the result reads the data rather than a constant.
        let with_feedback = simulate(3000);
        let mut without = simulate(3000);
        for i in 0..without.unit.len() / 2 {
            let a0 = f64::from(u8::from(without.treated[2 * i]));
            without.confounder[2 * i + 1] -= 0.8 * a0;
            without.outcome[2 * i + 1] -= 1.5 * 0.8 * a0;
        }
        let regimes = StaticRegimes { active: &ALWAYS, reference: &NEVER };
        let (a, _) =
            regime_point_estimate(&panel_of(&with_feedback), regimes, RegimeMethod::GFormula, 0.01)
                .unwrap();
        let (b, _) =
            regime_point_estimate(&panel_of(&without), regimes, RegimeMethod::GFormula, 0.01)
                .unwrap();
        // The mediated path adds 1.5 * 0.8 = 1.2 to the contrast.
        assert!((a[0] - b[0] - 1.2).abs() < 0.3, "{a:?} {b:?}");
    }

    #[test]
    fn separated_arms_unbalanced_panels_and_bad_regimes_are_refused() {
        let mut data = simulate(300);
        let regimes = StaticRegimes { active: &ALWAYS, reference: &NEVER };
        let ctx = ExecutionContext::for_tests(1);
        // Treatment is a deterministic function of the baseline confounder: positivity fails.
        for i in 0..data.unit.len() / 2 {
            data.treated[2 * i] = data.confounder[2 * i] > 0.0;
        }
        let separated = panel_of(&data);
        assert!(
            regime_point_estimate(&separated, regimes, RegimeMethod::InverseProbability, 0.01)
                .is_err()
        );
        let ordinary = simulate(300);
        let panel = panel_of(&ordinary);
        assert!(
            fit_static_regimes(&panel, regimes, RegimeMethod::GFormula, 0.01, 1.0, 10, None, &ctx)
                .is_err()
        );
        let same = StaticRegimes { active: &ALWAYS, reference: &ALWAYS };
        assert!(regime_point_estimate(&panel, same, RegimeMethod::GFormula, 0.01).is_err());
        let short = StaticRegimes { active: &[true], reference: &NEVER };
        assert!(regime_point_estimate(&panel, short, RegimeMethod::GFormula, 0.01).is_err());
        // A subject missing its last period breaks the complete follow-up.
        let confounders = vec![ordinary.confounder[..ordinary.confounder.len() - 1].to_vec()];
        let truncated = RegimePanel::new(RegimeRows {
            unit: &ordinary.unit[..ordinary.unit.len() - 1],
            period: &ordinary.period[..ordinary.period.len() - 1],
            outcome: &ordinary.outcome[..ordinary.outcome.len() - 1],
            treated: &ordinary.treated[..ordinary.treated.len() - 1],
            confounders: &confounders,
        });
        assert!(truncated.is_err());
    }

    #[test]
    fn weighted_least_squares_recovers_noiseless_coefficients() {
        let x: Vec<f64> = (0..6).flat_map(|i| [1.0, f64::from(i), f64::from(i * i)]).collect();
        let y: Vec<f64> =
            (0..6).map(|i| 2.0 - 3.0 * f64::from(i) + 0.5 * f64::from(i * i)).collect();
        let weights = [1.0, 2.0, 0.0, 1.0, 3.0, 1.0];
        let coefficients = weighted_least_squares(&x, 3, &y, &weights).unwrap();
        for (found, expected) in coefficients.iter().zip([2.0, -3.0, 0.5]) {
            assert!((found - expected).abs() < 1e-4, "{coefficients:?}");
        }
    }
}
