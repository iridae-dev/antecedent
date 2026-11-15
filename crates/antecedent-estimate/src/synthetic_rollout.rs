//! Synthetic control with intervals: conformal inference for one treated unit and placebo
//! variance for a staggered multi-unit rollout.
//!
//! **Conformal intervals** (Chernozhukov, Wüthrich and Zhu 2021). For a hypothesized effect
//! `θ` at one post-intervention period, the treated outcome there is replaced by `Y − θ`, the
//! simplex donor weights are refit on the pre-periods together with that adjusted period, and
//! the adjusted period's absolute residual is ranked among all the residuals. The p-value is
//! its rank share; the interval is every `θ` whose p-value exceeds 5%. The constant-effect
//! interval does the same with every post period adjusted by one `θ`, a statistic summing the
//! post-period absolute residuals and moving-block (cyclic) permutations of the time axis.
//! The fit is symmetric in time under the null, so when the residual series is exchangeable
//! (iid) the test is exact in finite samples; under weak serial dependence it is approximate.
//! The interval is the hull of the accepted set found by a grid over a range scaled by the
//! pre-fit error and refined by bisection; a set that reaches the widest searched range is
//! reported as unbounded rather than truncated.
//!
//! **Multi-unit staggered rollouts.** Each treated unit is fit against the never-treated
//! donors only, at its own adoption period, with simplex synthetic-control weights. The
//! synthetic difference-in-differences option is refused until its unit-weight objective
//! includes the required intercept and ridge penalty. The
//! overall effect averages the per-unit effects weighted by each unit's post-period count.
//! Its uncertainty is the placebo variance of Arkhangelsky et al.: each replicate assigns the
//! observed adoption periods to randomly chosen never-treated units, refits the whole
//! procedure on the remaining donors, and the variance of the placebo estimates gives a normal
//! interval. That variance is valid when the treated units' untreated errors are exchangeable
//! with the donors'; with very few donors it is unstable.
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
#[path = "synthetic_rollout/tests.rs"]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, SyntheticPanelMethod};

use crate::EstimationError;
use crate::rollout_resampling::{
    DesignEffect, DesignEffectKind, ROLLOUT_LEVEL, resample_draws, simultaneous_critical_value,
    uniform_index,
};

/// Semantic operation tag of the placebo-variance resample.
const PLACEBO_OPERATION: u64 = 0x5359_4e52_0001;

/// Iteration cap and tolerance of the accelerated simplex fit.
const SIMPLEX_ITERATIONS: usize = 5_000;
const SIMPLEX_TOLERANCE: f64 = 1e-10;

/// Grid points and bisection steps of the conformal test inversion.
const GRID_POINTS: usize = 41;
const BISECTION_STEPS: usize = 30;
/// Refinement of the grid spacing tried once when the first grid accepts nothing.
const FINE_GRID_FACTOR: usize = 8;
/// Range expansions before a conformal set is reported unbounded.
const RANGE_EXPANSIONS: usize = 4;

/// The row-aligned long panel of a synthetic design.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticRolloutPanel<'a> {
    /// Outcome of each row.
    pub outcome: &'a [f64],
    /// Unit of each row.
    pub units: &'a [Arc<str>],
    /// Calendar period of each row.
    pub periods: &'a [i64],
}

/// A balanced unit-by-period panel in label and period order.
#[derive(Clone, Debug)]
struct Panel {
    labels: Vec<String>,
    periods: Vec<i64>,
    /// `values[unit][period index]`.
    values: Vec<Vec<f64>>,
}

impl Panel {
    fn unit(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|candidate| candidate == label)
    }
}

fn refused(message: impl Into<String>) -> EstimationError {
    EstimationError::refused("option_not_applicable", message)
}

fn invalid(message: impl Into<String>) -> EstimationError {
    EstimationError::refused("invalid_argument", message)
}

/// Read the rows into a balanced panel, leasing its cells from the admission pool first.
fn balanced_panel(
    panel: &SyntheticRolloutPanel<'_>,
    ctx: &ExecutionContext,
) -> Result<(Panel, antecedent_core::execution::AdmissionLease), EstimationError> {
    let n = panel.outcome.len();
    if n == 0 || panel.units.len() != n || panel.periods.len() != n {
        return Err(invalid("synthetic panel columns must be row-aligned and non-empty"));
    }
    if panel.outcome.iter().any(|value| !value.is_finite()) {
        return Err(invalid("synthetic panel outcomes must be finite"));
    }
    let mut rows: BTreeMap<&str, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut grid = BTreeSet::new();
    for ((unit, period), value) in panel.units.iter().zip(panel.periods).zip(panel.outcome) {
        grid.insert(*period);
        if rows.entry(unit.as_ref()).or_default().insert(*period, *value).is_some() {
            return Err(invalid("each unit has exactly one outcome per period"));
        }
    }
    let periods: Vec<i64> = grid.into_iter().collect();
    if rows.values().any(|observed| observed.len() != periods.len()) {
        return Err(refused(
            "a synthetic donor pool needs every unit observed in every period; next attempt: \
             restrict the units and periods to a common window in which every kept unit is \
             complete",
        ));
    }
    let lease = crate::util::admit_array::<f64>(ctx, rows.len(), periods.len())?;
    let labels: Vec<String> = rows.keys().map(|label| (*label).to_string()).collect();
    let values = rows.values().map(|observed| observed.values().copied().collect()).collect();
    Ok((Panel { labels, periods, values }, lease))
}

/// Euclidean projection onto the probability simplex.
fn project_simplex(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut theta = 0.0;
    for (index, value) in sorted.iter().enumerate() {
        cumulative += value;
        let candidate = (cumulative - 1.0) / (index + 1) as f64;
        if index + 1 == sorted.len() || sorted[index + 1] <= candidate {
            theta = candidate;
            break;
        }
    }
    values.iter().map(|value| (value - theta).max(0.0)).collect()
}

/// Simplex weights minimizing the mean squared gap between `target` and the weighted
/// `donors` (each a series aligned with `target`), by accelerated projected gradient from
/// `start` (uniform when absent).
fn fit_simplex(target: &[f64], donors: &[Vec<f64>], start: Option<&[f64]>) -> Result<Vec<f64>, ()> {
    fit_simplex_with_limit(target, donors, start, SIMPLEX_ITERATIONS)
}

/// The operands of one fit after the common-level subtraction and scaling, with donor values
/// stored period-major so one period's row is contiguous.
struct NormalizedProblem {
    k: usize,
    n: usize,
    target: Vec<f64>,
    donors: Vec<f64>,
}

fn normalize_problem(target: &[f64], donors: &[Vec<f64>]) -> Result<NormalizedProblem, ()> {
    let k = donors.len();
    if k == 0 || target.is_empty() || donors.iter().any(|d| d.len() != target.len()) {
        return Err(());
    }
    // Subtract the first donor from all donors and the target. Since weights
    // sum to one this preserves the objective but removes irrelevant common
    // levels that otherwise dominate the Lipschitz bound.
    if target.iter().chain(donors.iter().flatten()).any(|x| !x.is_finite()) {
        return Err(());
    }
    // Use direct subtraction when representable: normalizing the operands
    // first can erase small differences atop a large common level. If any
    // subtraction overflows, halve every operand before subtraction so the
    // same common scale is used for the complete objective.
    let half = (0..target.len()).any(|t| {
        !(target[t] - donors[0][t]).is_finite()
            || donors.iter().any(|donor| !(donor[t] - donors[0][t]).is_finite())
    });
    let difference = |x: f64, t: usize| {
        if half { x * 0.5 - donors[0][t] * 0.5 } else { x - donors[0][t] }
    };
    let mut spread = 0.0_f64;
    for t in 0..target.len() {
        spread = spread.max(difference(target[t], t).abs());
        for donor in donors {
            spread = spread.max(difference(donor[t], t).abs());
        }
    }
    let spread = if spread == 0.0 { 1.0 } else { spread };
    let value = |x: f64, t: usize| difference(x, t) / spread;
    // The normalized operands are fixed for the whole fit: compute each once.
    let normalized_target: Vec<f64> =
        target.iter().enumerate().map(|(t, x)| value(*x, t)).collect();
    let mut normalized_donors = vec![0.0_f64; k * target.len()];
    for (j, donor) in donors.iter().enumerate() {
        for (t, x) in donor.iter().enumerate() {
            normalized_donors[t * k + j] = value(*x, t);
        }
    }
    Ok(NormalizedProblem {
        k,
        n: target.len(),
        target: normalized_target,
        donors: normalized_donors,
    })
}

/// `Zᵀ Z / n` (row-major, symmetric) of period-major `rows` with `k` columns, and `Zᵀ y / n`.
fn gram_of(rows: &[f64], y: &[f64], k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gram = vec![0.0_f64; k * k];
    let mut moment = vec![0.0_f64; k];
    for t in 0..n {
        let row = &rows[t * k..(t + 1) * k];
        for (i, xi) in row.iter().enumerate() {
            for (j, xj) in row.iter().enumerate().skip(i) {
                gram[i * k + j] += xi * xj;
            }
            moment[i] += xi * y[t];
        }
    }
    let n = n as f64;
    for i in 0..k {
        for j in i..k {
            let value = gram[i * k + j] / n;
            gram[i * k + j] = value;
            gram[j * k + i] = value;
        }
        moment[i] /= n;
    }
    (gram, moment)
}

/// Gradient `2 r (G w - b)` of the mean squared residual from its Gram form, `r` being
/// `factor`.
fn gram_gradient(gram: &[f64], moment: &[f64], factor: f64, weights: &[f64]) -> Vec<f64> {
    let k = weights.len();
    (0..k)
        .map(|i| {
            let row = &gram[i * k..(i + 1) * k];
            2.0 * factor * (row.iter().zip(weights).map(|(g, w)| g * w).sum::<f64>() - moment[i])
        })
        .collect()
}

/// The per-row gradient of the mean squared residual (the evaluation the Gram form replaces).
#[cfg(test)]
fn rowwise_gradient(problem: &NormalizedProblem, weights: &[f64]) -> Vec<f64> {
    let (k, n) = (problem.k, problem.n as f64);
    let mut grad = vec![0.0; k];
    for (t, target_value) in problem.target.iter().enumerate() {
        let row = &problem.donors[t * k..(t + 1) * k];
        let residual = row.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>() - target_value;
        for (g, x) in grad.iter_mut().zip(row) {
            *g += 2.0 * x * residual / n;
        }
    }
    grad
}

/// The simplex fit by the per-row gradient (the original evaluation), kept to check the Gram
/// form against.
#[cfg(test)]
fn fit_simplex_rowwise_with_limit(
    target: &[f64],
    donors: &[Vec<f64>],
    start: Option<&[f64]>,
    iterations: usize,
) -> Result<Vec<f64>, ()> {
    let problem = normalize_problem(target, donors)?;
    let frobenius: f64 = problem.donors.iter().map(|x| x.powi(2)).sum();
    let lipschitz = 2.0 * frobenius / problem.n as f64;
    let step = if lipschitz > 0.0 { 1.0 / lipschitz } else { 1.0 };
    if !step.is_finite() {
        return Err(());
    }
    projected_gradient(problem.k, step, start, iterations, |weights| {
        rowwise_gradient(&problem, weights)
    })
}

fn fit_simplex_with_limit(
    target: &[f64],
    donors: &[Vec<f64>],
    start: Option<&[f64]>,
    iterations: usize,
) -> Result<Vec<f64>, ()> {
    let problem = normalize_problem(target, donors)?;
    let (gram, moment) = gram_of(&problem.donors, &problem.target, problem.k, problem.n);
    let trace: f64 = (0..problem.k).map(|i| gram[i * problem.k + i]).sum();
    let lipschitz = 2.0 * trace;
    let step = if lipschitz > 0.0 { 1.0 / lipschitz } else { 1.0 };
    if !step.is_finite() {
        return Err(());
    }
    projected_gradient(problem.k, step, start, iterations, |weights| {
        gram_gradient(&gram, &moment, 1.0, weights)
    })
}

/// A simplex fit whose target has `theta` subtracted from every period from `from` on, refit
/// for many `theta` against the same donors. The donor Gram matrix does not depend on `theta`
/// and the moment vector is affine in it, so each refit costs `O(k)` setup plus `O(k²)` per
/// iteration instead of re-forming the problem from the `k n` raw values.
struct ShiftedTargetFit<'a> {
    target: &'a [f64],
    donors: &'a [Vec<f64>],
    from: usize,
    k: usize,
    n: usize,
    /// Largest absolute donor-minus-first-donor value.
    donor_spread: f64,
    /// Largest absolute target-minus-first-donor value over the unshifted periods.
    target_spread: f64,
    /// Common scale of the stored operands.
    common: f64,
    /// `Zᵀ Z / n`, `Zᵀ y₀ / n` and `Zᵀ 1_shifted / n` of the operands divided by `common`.
    gram: Vec<f64>,
    moment: Vec<f64>,
    shift_moment: Vec<f64>,
}

impl<'a> ShiftedTargetFit<'a> {
    /// `None` when the operands need the general path (non-finite or overflowing values).
    fn new(target: &'a [f64], donors: &'a [Vec<f64>], from: usize) -> Option<Self> {
        let k = donors.len();
        let n = target.len();
        if k == 0 || n == 0 || from > n || donors.iter().any(|d| d.len() != n) {
            return None;
        }
        if target.iter().chain(donors.iter().flatten()).any(|x| !x.is_finite()) {
            return None;
        }
        let base = &donors[0];
        let mut donor_spread = 0.0_f64;
        for t in 0..n {
            for donor in donors {
                let difference = donor[t] - base[t];
                if !difference.is_finite() {
                    return None;
                }
                donor_spread = donor_spread.max(difference.abs());
            }
        }
        let mut target_spread = 0.0_f64;
        for t in 0..from {
            let difference = target[t] - base[t];
            if !difference.is_finite() {
                return None;
            }
            target_spread = target_spread.max(difference.abs());
        }
        if (from..n).any(|t| !(target[t] - base[t]).is_finite()) {
            return None;
        }
        let common = if donor_spread == 0.0 { 1.0 } else { donor_spread };
        let mut rows = vec![0.0_f64; k * n];
        for (j, donor) in donors.iter().enumerate() {
            for t in 0..n {
                rows[t * k + j] = (donor[t] - base[t]) / common;
            }
        }
        let scaled_target: Vec<f64> = (0..n).map(|t| (target[t] - base[t]) / common).collect();
        let (gram, moment) = gram_of(&rows, &scaled_target, k, n);
        let mut shift_moment = vec![0.0_f64; k];
        for t in from..n {
            for (j, value) in shift_moment.iter_mut().enumerate() {
                *value += rows[t * k + j];
            }
        }
        for value in &mut shift_moment {
            *value /= n as f64;
        }
        if gram.iter().chain(&moment).chain(&shift_moment).any(|x| !x.is_finite()) {
            return None;
        }
        Some(Self {
            target,
            donors,
            from,
            k,
            n,
            donor_spread,
            target_spread,
            common,
            gram,
            moment,
            shift_moment,
        })
    }

    /// The target with `theta` subtracted from the shifted periods.
    fn adjusted(&self, theta: f64) -> Vec<f64> {
        shifted_target(self.target, self.from, theta)
    }

    /// The moment vector and gradient factor for the target shifted by `theta`; `None` when
    /// the general path must be used.
    fn shifted_moment(&self, theta: f64) -> Option<(Vec<f64>, f64, f64)> {
        let k = self.k;
        let base = &self.donors[0];
        let mut spread = self.donor_spread.max(self.target_spread);
        for t in self.from..self.n {
            let difference = (self.target[t] - theta) - base[t];
            if !difference.is_finite() {
                return None;
            }
            spread = spread.max(difference.abs());
        }
        let spread = if spread == 0.0 { 1.0 } else { spread };
        let factor = (self.common / spread).powi(2);
        let offset = theta / self.common;
        let moment: Vec<f64> =
            self.moment.iter().zip(&self.shift_moment).map(|(m, s)| m - offset * s).collect();
        let trace: f64 = (0..k).map(|i| self.gram[i * k + i]).sum();
        let lipschitz = 2.0 * factor * trace;
        let step = if lipschitz > 0.0 { 1.0 / lipschitz } else { 1.0 };
        if !step.is_finite() || moment.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some((moment, factor, step))
    }

    /// The simplex weights for the target shifted by `theta`.
    fn fit(&self, theta: f64, start: Option<&[f64]>, iterations: usize) -> Result<Vec<f64>, ()> {
        let Some((moment, factor, step)) = self.shifted_moment(theta) else {
            return fit_simplex_with_limit(&self.adjusted(theta), self.donors, start, iterations);
        };
        projected_gradient(self.k, step, start, iterations, |weights| {
            gram_gradient(&self.gram, &moment, factor, weights)
        })
    }
}

/// Accelerated projected gradient with restart over the simplex for the given `gradient`
/// and fixed `step`.
#[allow(
    clippy::too_many_lines,
    reason = "the accelerated projected-gradient loop keeps its restart and momentum steps together"
)]
fn projected_gradient(
    k: usize,
    step: f64,
    start: Option<&[f64]>,
    iterations: usize,
    gradient: impl Fn(&[f64]) -> Vec<f64>,
) -> Result<Vec<f64>, ()> {
    let mut weights = start.map_or_else(|| vec![1.0 / k as f64; k], <[f64]>::to_vec);
    if weights.len() != k || weights.iter().any(|x| !x.is_finite()) {
        return Err(());
    }
    weights = project_simplex(&weights);
    let mut momentum = weights.clone();
    let mut t = 1.0_f64;
    for _ in 0..iterations {
        let current_gradient = gradient(&weights);
        if current_gradient.iter().any(|v| !v.is_finite()) {
            return Err(());
        }
        let projected = project_simplex(
            &weights.iter().zip(&current_gradient).map(|(w, g)| w - g).collect::<Vec<_>>(),
        );
        let stationarity =
            projected.iter().zip(&weights).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        if stationarity < SIMPLEX_TOLERANCE {
            return Ok(weights);
        }
        // The gradient at the momentum point is the gradient just computed whenever the
        // momentum point is bit-for-bit the current weights (the first iteration, and after
        // every restart), so it is reused rather than recomputed.
        let grad = if momentum.iter().zip(&weights).all(|(m, w)| m.to_bits() == w.to_bits()) {
            current_gradient
        } else {
            let grad = gradient(&momentum);
            if grad.iter().any(|v| !v.is_finite()) {
                return Err(());
            }
            grad
        };
        let next = project_simplex(
            &momentum.iter().zip(&grad).map(|(m, g)| m - step * g).collect::<Vec<_>>(),
        );
        // Restart acceleration when momentum points uphill; this avoids
        // oscillation around active simplex boundaries.
        let restart: f64 = momentum
            .iter()
            .zip(&next)
            .zip(&weights)
            .map(|((m, next), old)| (m - next) * (next - old))
            .sum();
        let t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
        momentum = if restart > 0.0 {
            t = 1.0;
            next.clone()
        } else {
            let out = next
                .iter()
                .zip(&weights)
                .map(|(new, old)| new + (t - 1.0) / t_next * (new - old))
                .collect();
            t = t_next;
            out
        };
        weights = next;
    }
    Err(())
}

fn synthetic(donors: &[Vec<f64>], weights: &[f64], t: usize) -> f64 {
    donors.iter().zip(weights).map(|(donor, w)| donor[t] * w).sum()
}

/// A conformal synthetic-control fit of one treated unit.
#[derive(Clone, Debug, PartialEq)]
pub struct ConformalSyntheticFit {
    /// Average post-period effect, with the constant-effect conformal interval.
    pub overall: DesignEffect,
    /// Per-period effects, each with its own conformal interval.
    pub periods: Vec<DesignEffect>,
    /// Donor weights of the pre-period fit, in donor label order.
    pub donor_weights: Vec<(String, f64)>,
    /// Pre-period root mean squared fit gap.
    pub pre_treatment_rmse: f64,
    /// Pre-periods.
    pub pre_periods: usize,
    /// Post-periods.
    pub post_periods: usize,
    /// Smallest attainable per-period p-value, `1 / (pre-periods + 1)`.
    pub smallest_period_p_value: f64,
    /// Intervals reported unbounded at the widest searched range.
    pub unbounded_intervals: usize,
}

/// Conformal synthetic control for `treated` adopting at `intervention`.
///
/// # Errors
///
/// An unbalanced or misaligned panel, fewer than two donors, too few pre-periods for a 5%
/// conformal test (at least 19), no post-period, cancellation or a refused admission.
#[allow(
    clippy::too_many_lines,
    reason = "one procedure: the pre-period fit, the per-period effects and the conformal inversion share the donor series"
)]
pub fn conformal_synthetic_control(
    panel: &SyntheticRolloutPanel<'_>,
    treated: &str,
    intervention: i64,
    ctx: &ExecutionContext,
) -> Result<ConformalSyntheticFit, EstimationError> {
    conformal_synthetic_control_at_level(panel, treated, intervention, ROLLOUT_LEVEL, ctx)
}

/// Conformal synthetic-control inversion at an explicit level, enabling joint
/// Bonferroni regions over multiple declared rollout targets.
/// # Errors
/// Invalid level, insufficient pre-period resolution, malformed panel or admission refusal.
pub fn conformal_synthetic_control_at_level(
    panel: &SyntheticRolloutPanel<'_>,
    treated: &str,
    intervention: i64,
    level: f64,
    ctx: &ExecutionContext,
) -> Result<ConformalSyntheticFit, EstimationError> {
    conformal_synthetic_control_for(
        panel,
        treated,
        intervention,
        level,
        ConformalIntervals::All,
        ctx,
    )
}

/// Which conformal intervals a fit inverts. Every estimate and the donor weights are always
/// reported; an interval that is not selected is absent (`None`) and is not counted as
/// unbounded. Each selected interval is the same number the full fit reports for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConformalIntervals {
    /// Every per-period interval and the constant-effect interval.
    All,
    /// Only the constant-effect interval of the average effect.
    Overall,
    /// Only the interval of the post period at this calendar period.
    Period(i64),
}

impl ConformalIntervals {
    fn includes_overall(self) -> bool {
        matches!(self, Self::All | Self::Overall)
    }

    fn includes_period(self, period: i64) -> bool {
        match self {
            Self::All => true,
            Self::Overall => false,
            Self::Period(selected) => selected == period,
        }
    }
}

/// Conformal synthetic-control inversion at an explicit level, inverting only the selected
/// intervals.
/// # Errors
/// Invalid level, insufficient pre-period resolution, malformed panel or admission refusal.
#[allow(
    clippy::too_many_lines,
    reason = "one procedure: the pre-period fit, the per-period effects and the conformal inversion share the donor series"
)]
pub fn conformal_synthetic_control_for(
    panel: &SyntheticRolloutPanel<'_>,
    treated: &str,
    intervention: i64,
    level: f64,
    intervals: ConformalIntervals,
    ctx: &ExecutionContext,
) -> Result<ConformalSyntheticFit, EstimationError> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(invalid("conformal level must be strictly between zero and one"));
    }
    let (panel, _lease) = balanced_panel(panel, ctx)?;
    let unit = panel
        .unit(treated)
        .ok_or_else(|| invalid("the treated unit is not in the synthetic panel"))?;
    let pre: Vec<usize> =
        (0..panel.periods.len()).filter(|t| panel.periods[*t] < intervention).collect();
    let post: Vec<usize> =
        (0..panel.periods.len()).filter(|t| panel.periods[*t] >= intervention).collect();
    let alpha = 1.0 - level;
    require_conformal_periods(pre.len(), post.len(), alpha)?;
    let donors: Vec<usize> = (0..panel.labels.len()).filter(|u| *u != unit).collect();
    if donors.len() < 2 {
        return Err(invalid("conformal synthetic control needs at least two donors"));
    }
    let series = |u: usize, times: &[usize]| -> Vec<f64> {
        times.iter().map(|t| panel.values[u][*t]).collect()
    };
    let donor_pre: Vec<Vec<f64>> = donors.iter().map(|u| series(*u, &pre)).collect();
    let target_pre = series(unit, &pre);
    let warm = fit_simplex(&target_pre, &donor_pre, None).map_err(|()| optimizer_refusal())?;
    let pre_rmse = (pre
        .iter()
        .enumerate()
        .map(|(i, _)| (target_pre[i] - synthetic(&donor_pre, &warm, i)).powi(2))
        .sum::<f64>()
        / pre.len() as f64)
        .sqrt();
    let scale = pre_rmse.max(1e-8 * (1.0 + target_pre.iter().map(|v| v.abs()).fold(0.0, f64::max)));
    let mut unbounded = 0usize;
    let mut periods = Vec::with_capacity(post.len());
    for &s in &post {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported(
                "execution cancelled during conformal inversion",
            ));
        }
        let estimate = panel.values[unit][s]
            - donors.iter().zip(&warm).map(|(u, w)| panel.values[*u][s] * w).sum::<f64>();
        let interval = if intervals.includes_period(panel.periods[s]) {
            let times: Vec<usize> = pre.iter().copied().chain(std::iter::once(s)).collect();
            let donor_series: Vec<Vec<f64>> = donors.iter().map(|u| series(*u, &times)).collect();
            let target = series(unit, &times);
            let refit = ShiftedTargetFit::new(&target, &donor_series, target.len() - 1);
            let accept = |theta: f64| {
                let adjusted = shifted_target(&target, target.len() - 1, theta);
                let p =
                    conformal_p_refit(refit.as_ref(), theta, &adjusted, &donor_series, 1, &warm)?;
                Ok(p > alpha)
            };
            let interval = invert(estimate, scale, accept)?;
            unbounded += usize::from(interval.is_none());
            interval
        } else {
            None
        };
        periods.push(DesignEffect {
            period: Some(panel.periods[s]),
            unit: Some(treated.to_string()),
            interval,
            treated: 1,
            comparison: donors.len(),
            ..DesignEffect::point(DesignEffectKind::Period, estimate)
        });
    }
    let average = periods.iter().map(|effect| effect.estimate).sum::<f64>() / post.len() as f64;
    let interval = if intervals.includes_overall() {
        let all: Vec<usize> = pre.iter().chain(&post).copied().collect();
        let donor_all: Vec<Vec<f64>> = donors.iter().map(|u| series(*u, &all)).collect();
        let target_all = series(unit, &all);
        let refit = ShiftedTargetFit::new(&target_all, &donor_all, pre.len());
        let accept = |theta: f64| {
            let adjusted = shifted_target(&target_all, pre.len(), theta);
            let p =
                conformal_p_refit(refit.as_ref(), theta, &adjusted, &donor_all, post.len(), &warm)?;
            Ok(p > alpha)
        };
        let interval = invert(average, scale, accept)?;
        unbounded += usize::from(interval.is_none());
        interval
    } else {
        None
    };
    Ok(ConformalSyntheticFit {
        overall: DesignEffect {
            unit: Some(treated.to_string()),
            interval,
            treated: 1,
            comparison: donors.len(),
            ..DesignEffect::point(DesignEffectKind::Overall, average)
        },
        periods,
        donor_weights: donors
            .iter()
            .zip(&warm)
            .map(|(u, w)| (panel.labels[*u].clone(), *w))
            .collect(),
        pre_treatment_rmse: pre_rmse,
        pre_periods: pre.len(),
        post_periods: post.len(),
        smallest_period_p_value: 1.0 / (pre.len() + 1) as f64,
        unbounded_intervals: unbounded,
    })
}

/// Require enough pre-periods for the requested finite conformal rank. The
/// caller derives positive finite alpha from its checked level in (0, 1).
fn require_conformal_periods(pre: usize, post: usize, alpha: f64) -> Result<(), EstimationError> {
    if post == 0 {
        return Err(invalid("conformal synthetic control needs at least one post-period"));
    }
    if ((pre + 1) as f64) * alpha < 1.0 {
        let needed = (1.0 / alpha).ceil() - 1.0;
        return Err(refused(format!(
            "the requested conformal level needs at least {needed:.0} pre-periods so that one over pre-periods plus one is no greater than alpha; the panel has {pre}; next attempt: a longer pre-period, or the placebo-variance interval"
        )));
    }
    Ok(())
}

/// Moving-block conformal p-value of `target` (adjusted under the null) against `donors`, the
/// last `post` positions holding the tested periods.
fn conformal_p(
    target: &[f64],
    donors: &[Vec<f64>],
    post: usize,
    warm: &[f64],
) -> Result<f64, EstimationError> {
    let weights = fit_simplex(target, donors, Some(warm)).map_err(|()| optimizer_refusal())?;
    Ok(conformal_p_at(target, donors, post, &weights))
}

/// The conformal p-value of `target` given the simplex `weights` already fit to it.
fn conformal_p_at(target: &[f64], donors: &[Vec<f64>], post: usize, weights: &[f64]) -> f64 {
    let n = target.len();
    let residuals: Vec<f64> =
        (0..n).map(|t| (target[t] - synthetic(donors, weights, t)).abs()).collect();
    let statistic = |shift: usize| -> f64 {
        (n - post..n).map(|position| residuals[(position + n - shift) % n]).sum()
    };
    let observed = statistic(0);
    let tolerance = 1e-12 * (1.0 + observed.abs());
    let at_least = (0..n).filter(|shift| statistic(*shift) >= observed - tolerance).count();
    at_least as f64 / n as f64
}

/// `target` with `theta` subtracted from every period from `from` on.
fn shifted_target(target: &[f64], from: usize, theta: f64) -> Vec<f64> {
    let mut adjusted = target.to_vec();
    for value in &mut adjusted[from..] {
        *value -= theta;
    }
    adjusted
}

/// The conformal p-value of the shifted `adjusted` target, refit through the precomputed Gram
/// pieces when available and from scratch otherwise.
fn conformal_p_refit(
    refit: Option<&ShiftedTargetFit<'_>>,
    theta: f64,
    adjusted: &[f64],
    donors: &[Vec<f64>],
    post: usize,
    warm: &[f64],
) -> Result<f64, EstimationError> {
    match refit {
        Some(refit) => {
            let weights = refit
                .fit(theta, Some(warm), SIMPLEX_ITERATIONS)
                .map_err(|()| optimizer_refusal())?;
            Ok(conformal_p_at(adjusted, donors, post, &weights))
        }
        None => conformal_p(adjusted, donors, post, warm),
    }
}

/// The hull of `{θ : accept(θ)}` around `center`, searched on a grid scaled by `scale` and
/// refined by bisection; `None` when the accepted set reaches the widest range searched or
/// is empty. A grid that accepts nothing is searched once more at a finer spacing before the
/// set is called empty, so a set narrower than the first spacing is found.
fn invert(
    center: f64,
    scale: f64,
    accept: impl Fn(f64) -> Result<bool, EstimationError>,
) -> Result<Option<[f64; 2]>, EstimationError> {
    let mut radius = 10.0 * scale;
    let mut points = GRID_POINTS;
    let mut expansions = 0;
    while expansions <= RANGE_EXPANSIONS {
        let grid: Vec<f64> = (0..points)
            .map(|i| center - radius + 2.0 * radius * i as f64 / (points - 1) as f64)
            .collect();
        let accepted: Vec<bool> =
            grid.iter().map(|theta| accept(*theta)).collect::<Result<_, _>>()?;
        let (Some(first), Some(last)) =
            (accepted.iter().position(|ok| *ok), accepted.iter().rposition(|ok| *ok))
        else {
            if points > GRID_POINTS {
                return Ok(None);
            }
            points = FINE_GRID_FACTOR * (GRID_POINTS - 1) + 1;
            continue;
        };
        if first == 0 || last == points - 1 {
            radius *= 4.0;
            expansions += 1;
            continue;
        }
        let refine = |mut inside: f64, mut outside: f64| -> Result<f64, EstimationError> {
            for _ in 0..BISECTION_STEPS {
                let middle = 0.5 * (inside + outside);
                if accept(middle)? {
                    inside = middle;
                } else {
                    outside = middle;
                }
            }
            Ok(inside)
        };
        return Ok(Some([
            refine(grid[first], grid[first - 1])?,
            refine(grid[last], grid[last + 1])?,
        ]));
    }
    Ok(None)
}

/// A multi-unit synthetic rollout fit.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiUnitSyntheticFit {
    /// Post-period-weighted average of the per-unit effects.
    pub overall: DesignEffect,
    /// Per-unit average post-adoption effects.
    pub units: Vec<DesignEffect>,
    /// Never-treated donors.
    pub donors: usize,
    /// Placebo record, when placebo replicates were run.
    pub placebo: Option<PlaceboRecord>,
}

/// Counts and critical value of the placebo-variance replicates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaceboRecord {
    /// Requested placebo replicates.
    pub requested: u32,
    /// Replicates with a finite placebo estimate for every quantity.
    pub replicates_ok: u32,
    /// Replicates without one.
    pub replicates_failed: u32,
    /// Sup-t critical value of the per-unit band.
    pub critical_value: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RolloutFailure {
    Support,
    Optimizer,
}

fn optimizer_refusal() -> EstimationError {
    refused("synthetic rollout weight iteration did not meet its stopping criterion")
}

/// Per-unit effects and their post-weighted average for `treated` (unit index, adoption
/// period) against `donors`.
fn rollout_effects(
    panel: &Panel,
    treated: &[(usize, i64)],
    donors: &[usize],
    method: SyntheticPanelMethod,
) -> Result<(f64, Vec<f64>, Vec<usize>), RolloutFailure> {
    let mut effects = Vec::with_capacity(treated.len());
    let mut post_counts = Vec::with_capacity(treated.len());
    for &(unit, adoption) in treated {
        let pre: Vec<usize> =
            (0..panel.periods.len()).filter(|t| panel.periods[*t] < adoption).collect();
        let post: Vec<usize> =
            (0..panel.periods.len()).filter(|t| panel.periods[*t] >= adoption).collect();
        if pre.len() < 2 || post.is_empty() || donors.len() < 2 {
            return Err(RolloutFailure::Support);
        }
        let donor_pre: Vec<Vec<f64>> =
            donors.iter().map(|u| pre.iter().map(|t| panel.values[*u][*t]).collect()).collect();
        let target_pre: Vec<f64> = pre.iter().map(|t| panel.values[unit][*t]).collect();
        let weights =
            fit_simplex(&target_pre, &donor_pre, None).map_err(|()| RolloutFailure::Optimizer)?;
        let post_mean =
            |u: usize| post.iter().map(|t| panel.values[u][*t]).sum::<f64>() / post.len() as f64;
        let treated_post = post_mean(unit);
        let synthetic_post: f64 = donors.iter().zip(&weights).map(|(u, w)| w * post_mean(*u)).sum();
        let effect = match method {
            SyntheticPanelMethod::Control => treated_post - synthetic_post,
            SyntheticPanelMethod::DifferenceInDifferences => {
                let donor_post: Vec<f64> = donors.iter().map(|u| post_mean(*u)).collect();
                let target = demean(&donor_post);
                let profiles: Vec<Vec<f64>> = pre
                    .iter()
                    .map(|t| {
                        demean(&donors.iter().map(|u| panel.values[*u][*t]).collect::<Vec<_>>())
                    })
                    .collect();
                let time_weights = fit_simplex(&target, &profiles, None)
                    .map_err(|()| RolloutFailure::Optimizer)?;
                let treated_pre: f64 =
                    pre.iter().zip(&time_weights).map(|(t, l)| l * panel.values[unit][*t]).sum();
                let donor_pre_joint: f64 = donors
                    .iter()
                    .zip(&weights)
                    .map(|(u, w)| {
                        w * pre
                            .iter()
                            .zip(&time_weights)
                            .map(|(t, l)| l * panel.values[*u][*t])
                            .sum::<f64>()
                    })
                    .sum();
                (treated_post - treated_pre) - (synthetic_post - donor_pre_joint)
            }
        };
        if !effect.is_finite() {
            return Err(RolloutFailure::Support);
        }
        effects.push(effect);
        post_counts.push(post.len());
    }
    let total: usize = post_counts.iter().sum();
    let overall =
        effects.iter().zip(&post_counts).map(|(e, c)| e * *c as f64).sum::<f64>() / total as f64;
    Ok((overall, effects, post_counts))
}

fn demean(values: &[f64]) -> Vec<f64> {
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    values.iter().map(|value| value - mean).collect()
}

/// Multi-unit synthetic control or synthetic difference in differences of a staggered
/// rollout, with placebo-variance intervals when `replicates > 0`.
///
/// # Errors
///
/// An unbalanced panel, an adoption unit missing from the panel or listed twice, a unit with
/// fewer than two pre-periods, fewer never-treated donors than the treated units plus two
/// (placebo replicates need donors left over), cancellation or a refused admission.
pub fn multi_unit_synthetic(
    panel: &SyntheticRolloutPanel<'_>,
    adoptions: &[(Arc<str>, i64)],
    method: SyntheticPanelMethod,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<MultiUnitSyntheticFit, EstimationError> {
    if method == SyntheticPanelMethod::DifferenceInDifferences {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "synthetic rollout difference-in-differences requires a unit-weight intercept and ridge penalty; the current simplex-only objective does not implement that construction",
        ));
    }
    let (panel, _lease) = balanced_panel(panel, ctx)?;
    let mut treated = Vec::with_capacity(adoptions.len());
    for (label, adoption) in adoptions {
        let unit = panel
            .unit(label)
            .ok_or_else(|| invalid("an adopting unit is not in the synthetic panel"))?;
        if treated.iter().any(|(seen, _)| *seen == unit) {
            return Err(invalid("each adopting unit is listed once"));
        }
        treated.push((unit, *adoption));
    }
    if treated.is_empty() {
        return Err(invalid("a synthetic rollout needs at least one adopting unit"));
    }
    let donors: Vec<usize> =
        (0..panel.labels.len()).filter(|u| !treated.iter().any(|(t, _)| t == u)).collect();
    if donors.len() < treated.len() + 2 {
        return Err(refused(format!(
            "placebo variance reassigns the {} adoptions to never-treated units and needs at \
             least two donors left over; the panel has {} never-treated units; next attempt: \
             more never-treated units, or one treated unit with conformal intervals",
            treated.len(),
            donors.len()
        )));
    }
    let (overall, effects, post_counts) = rollout_effects(&panel, &treated, &donors, method)
        .map_err(|failure| match failure {
            RolloutFailure::Optimizer => optimizer_refusal(),
            RolloutFailure::Support => refused(
                "every adopting unit needs at least two pre-periods and one post-period in the \
                 panel; next attempt: drop units adopting at the first or second period",
            ),
        })?;
    let mut fit = MultiUnitSyntheticFit {
        overall: DesignEffect {
            treated: treated.len(),
            comparison: donors.len(),
            ..DesignEffect::point(DesignEffectKind::Overall, overall)
        },
        units: treated
            .iter()
            .zip(&effects)
            .zip(&post_counts)
            .map(|(((unit, adoption), effect), posts)| DesignEffect {
                unit: Some(panel.labels[*unit].clone()),
                group: Some(*adoption),
                treated: *posts,
                comparison: donors.len(),
                ..DesignEffect::point(DesignEffectKind::Unit, *effect)
            })
            .collect(),
        donors: donors.len(),
        placebo: None,
    };
    if replicates > 0 {
        let width = 1 + treated.len();
        let optimizer_failed = std::sync::atomic::AtomicBool::new(false);
        let draws = resample_draws(ctx, PLACEBO_OPERATION, replicates, width, |rng| {
            let mut pool = donors.clone();
            let mut placebo = Vec::with_capacity(treated.len());
            for (index, (_, adoption)) in treated.iter().enumerate() {
                let pick = index + uniform_index(rng, pool.len() - index);
                pool.swap(index, pick);
                placebo.push((pool[index], *adoption));
            }
            let remaining = &pool[treated.len()..];
            match rollout_effects(&panel, &placebo, remaining, method) {
                Ok((overall, effects, _)) => {
                    Some(std::iter::once(overall).chain(effects).collect())
                }
                Err(RolloutFailure::Support) => None,
                Err(RolloutFailure::Optimizer) => {
                    optimizer_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
        })?;
        if optimizer_failed.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(optimizer_refusal());
        }
        fit.placebo = Some(attach_placebo(&mut fit, &draws, replicates));
    }
    Ok(fit)
}

/// Placebo standard errors, normal intervals and the per-unit sup-t band.
fn attach_placebo(
    fit: &mut MultiUnitSyntheticFit,
    draws: &[Option<Vec<f64>>],
    requested: u32,
) -> PlaceboRecord {
    let complete: Vec<&Vec<f64>> =
        draws.iter().flatten().filter(|draw| draw.iter().all(|value| value.is_finite())).collect();
    let replicates_ok = u32::try_from(complete.len()).unwrap_or(u32::MAX);
    let width = 1 + fit.units.len();
    let z = antecedent_stats::NORMAL_Q975;
    let moments: Vec<Option<(f64, f64)>> = (0..width)
        .map(|k| {
            if complete.len() < 2 {
                return None;
            }
            let mean = complete.iter().map(|draw| draw[k]).sum::<f64>() / complete.len() as f64;
            let variance = complete.iter().map(|draw| (draw[k] - mean).powi(2)).sum::<f64>()
                / complete.len() as f64;
            (variance > 0.0 && variance.is_finite()).then(|| (mean, variance.sqrt()))
        })
        .collect();
    let mut maxima: Vec<f64> = complete
        .iter()
        .filter_map(|draw| {
            (1..width)
                .filter_map(|k| moments[k].map(|(mean, se)| (draw[k] - mean).abs() / se))
                .reduce(f64::max)
        })
        .collect();
    let critical_value = simultaneous_critical_value(&mut maxima, || true);
    let apply = |effect: &mut DesignEffect, moment: Option<(f64, f64)>, band: bool| {
        if let Some((_, se)) = moment {
            effect.standard_error = Some(se);
            effect.interval = Some([effect.estimate - z * se, effect.estimate + z * se]);
            if band {
                effect.band =
                    critical_value.map(|c| [effect.estimate - c * se, effect.estimate + c * se]);
            }
        }
    };
    apply(&mut fit.overall, moments[0], false);
    for (effect, moment) in fit.units.iter_mut().zip(&moments[1..]) {
        apply(effect, *moment, true);
    }
    PlaceboRecord {
        requested,
        replicates_ok,
        replicates_failed: requested.saturating_sub(replicates_ok),
        critical_value,
    }
}
