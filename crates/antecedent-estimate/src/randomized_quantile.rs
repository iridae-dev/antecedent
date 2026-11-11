//! Frequentist quantile treatment effect of a randomized two-arm trial.
//!
//! The target is the `tau`-quantile of each potential outcome law and their difference, the
//! same target the Bayesian randomized quantile answers. Under randomization (an empty
//! adjustment set) the arm's potential-outcome distribution function is the arm's empirical
//! distribution function, `F_a(c) = #{i in arm a: Y_i <= c} / n_a`, evaluated on a strictly
//! increasing threshold grid. The quantile is the piecewise-linear inversion of that grid
//! distribution function ([`crate::quantile::invert_cdf_quantile`], shared with the
//! cross-fitted cell), so every frequentist quantile route reads one inversion.
//!
//! Uncertainty is the delta method on the inversion. The row influence of the grid
//! distribution function is `phi_F(c)_i = (1{T_i = a} / p_a) (1{Y_i <= c} - F_a(c))` with
//! `p_a = n_a / n` (the ratio estimator's influence under independent assignment), blended
//! across the inversion bracket with the bracket weight, and the influence of the quantile is
//! `-phi_F(q) / f_a(q)`. The density `f_a(q)` is not the slope of the grid bracket (a fixed
//! pooled bracket holds only a handful of an arm's rows, and the reciprocal of so noisy a slope
//! makes the studentized effect heavy-tailed). It is the arm's own Siddiqui (1960) sparsity
//! estimate `f_a = 2h / (Q_a(tau + h) - Q_a(tau - h))`, with `Q_a` the type-7 empirical quantile
//! function of the arm's outcomes and `h` the Hall and Sheather (1988) bandwidth
//! `h = n_a^(-1/3) z^(2/3) [1.5 phi(Phi^-1(tau))^2 / (2 Phi^-1(tau)^2 + 1)]^(1/3)` at the
//! interval's normal point `z = 1.959963984540054`. When `tau -/+ h` would leave (0, 1) the
//! bandwidth shrinks symmetrically to [`SPARSITY_EDGE_FRACTION`] of the room to the nearer
//! edge, and an arm whose outcomes are tied across the window (no spread) is refused. The
//! treated-minus-control influence is the row difference, the standard error is the iid
//! standard error of that influence and the interval is the 95% normal interval around the
//! difference. The point estimate conditions on the grid: grid selection and interpolation
//! bias are excluded, as in the cross-fitted cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::special::normal_ppf;
use antecedent_stats::{NORMAL_Q975, QuantileRule, quantile_sorted};

use crate::error::EstimationError;
use crate::quantile::{empirical_threshold_grid, invert_cdf_quantile};

/// Fewest rows each arm must carry; the arm's distribution function is not estimated below it.
pub const MIN_RANDOMIZED_QUANTILE_ARM_ROWS: usize = 10;

/// Points of the data-driven threshold grid built from the pooled outcomes.
pub const RANDOMIZED_QUANTILE_GRID_POINTS: usize = 19;

/// Largest share of the room `min(tau, 1 - tau)` the sparsity bandwidth may take, so the outer
/// levels `tau -/+ h` stay strictly inside (0, 1) and away from the arm's extreme order statistics.
pub const SPARSITY_EDGE_FRACTION: f64 = 0.9;

/// Hall and Sheather (1988) bandwidth for the sparsity at level `tau` with `rows` observations,
/// at the two-sided 95% normal point, shrunk to [`SPARSITY_EDGE_FRACTION`] of the room to the
/// nearer edge of (0, 1).
fn hall_sheather_bandwidth(rows: usize, tau: f64) -> f64 {
    let x = normal_ppf(tau);
    let pdf = antecedent_kernels::norm_pdf(x);
    let shape = 1.5 * pdf * pdf / (2.0 * x * x + 1.0);
    let h = (NORMAL_Q975 * NORMAL_Q975 * shape / rows as f64).cbrt();
    h.min(SPARSITY_EDGE_FRACTION * tau.min(1.0 - tau))
}

/// Siddiqui (1960) density at the `tau`-quantile of one arm, `2h / (Q(tau + h) - Q(tau - h))`
/// with the type-7 empirical quantile of the arm's outcomes and the Hall-Sheather bandwidth.
fn siddiqui_density(arm_outcomes: &mut [f64], tau: f64) -> Result<f64, EstimationError> {
    arm_outcomes.sort_by(f64::total_cmp);
    let h = hall_sheather_bandwidth(arm_outcomes.len(), tau);
    let upper = quantile_sorted(arm_outcomes, tau + h, QuantileRule::Interpolated);
    let lower = quantile_sorted(arm_outcomes, tau - h, QuantileRule::Interpolated);
    let spread = upper - lower;
    let density = 2.0 * h / spread;
    let supported = h > 0.0 && spread > 0.0 && density.is_finite();
    if !supported {
        return Err(EstimationError::unsupported(
            "randomized quantile refused: the arm's outcomes have no spread across the sparsity window",
        ));
    }
    Ok(density)
}

/// Quantile treatment effect of a randomized two-arm trial with its influence-function interval.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedQuantileEffect {
    /// Quantile level.
    pub tau: f64,
    /// Treated minus control quantile.
    pub difference: f64,
    /// Control-arm quantile.
    pub control_quantile: f64,
    /// Treated-arm quantile.
    pub treated_quantile: f64,
    /// Siddiqui sparsity densities at the control and treated quantiles (Hall-Sheather bandwidth).
    pub densities: [f64; 2],
    /// Centered row influence of the difference.
    pub influence: Vec<f64>,
    /// Standard error of the difference: the iid standard error of the influence.
    pub se: f64,
    /// 95% normal interval `difference -/+ NORMAL_Q975 * se`.
    pub interval_95: [f64; 2],
    /// Rows in the control arm.
    pub n_control: usize,
    /// Rows in the treated arm.
    pub n_treated: usize,
    /// The threshold grid the distribution functions were read on.
    pub thresholds: Vec<f64>,
}

/// Quantile treatment effect on a threshold grid the caller declares.
///
/// `thresholds` is strictly increasing with at least two points.
///
/// # Errors
///
/// Misaligned or non-finite rows, an arm with fewer than
/// [`MIN_RANDOMIZED_QUANTILE_ARM_ROWS`] rows, an invalid grid or level, a level outside the
/// grid distribution function of an arm, a grid density too small to invert, or an arm with no
/// spread across its sparsity window.
pub fn randomized_quantile_effect_on_grid(
    outcome: &[f64],
    treated: &[bool],
    thresholds: &[f64],
    tau: f64,
) -> Result<RandomizedQuantileEffect, EstimationError> {
    if outcome.len() != treated.len() {
        return Err(EstimationError::data_msg("outcome and treatment must align"));
    }
    if outcome.iter().any(|y| !y.is_finite()) {
        return Err(EstimationError::data_msg("the quantile outcome must be finite"));
    }
    let n = outcome.len();
    let n_treated = treated.iter().filter(|t| **t).count();
    let n_control = n - n_treated;
    if n_treated < MIN_RANDOMIZED_QUANTILE_ARM_ROWS || n_control < MIN_RANDOMIZED_QUANTILE_ARM_ROWS
    {
        return Err(EstimationError::refused_detail(
            antecedent_core::reason_code!("positivity_violated"),
            antecedent_core::reason_details::randomized_quantile_effect::ARM_TOO_SMALL,
            "the randomized quantile needs at least ten rows in each arm",
        ));
    }
    let arm = |active: bool| -> Result<(f64, Vec<f64>, f64), EstimationError> {
        let size = (if active { n_treated } else { n_control }) as f64;
        let share = size / n as f64;
        let cdf: Vec<f64> = thresholds
            .iter()
            .map(|&c| {
                let count =
                    outcome.iter().zip(treated).filter(|(y, t)| **t == active && **y <= c).count();
                count as f64 / size
            })
            .collect();
        let phi: Vec<Vec<f64>> = thresholds
            .iter()
            .zip(&cdf)
            .map(|(&c, &f)| {
                outcome
                    .iter()
                    .zip(treated)
                    .map(|(y, t)| {
                        if *t == active { (f64::from(u8::from(*y <= c)) - f) / share } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        // The grid inversion gives the quantile and `-phi_F(q) / f_grid`; the influence is
        // rescaled onto the arm's sparsity density, so only the density changes.
        let (quantile, grid_influence, grid_density) =
            invert_cdf_quantile(thresholds, &cdf, &phi, tau)?;
        let mut arm_outcomes: Vec<f64> =
            outcome.iter().zip(treated).filter(|(_, t)| **t == active).map(|(y, _)| *y).collect();
        let density = siddiqui_density(&mut arm_outcomes, tau)?;
        let ratio = grid_density / density;
        let influence = grid_influence.iter().map(|v| v * ratio).collect();
        Ok((quantile, influence, density))
    };
    let (control_quantile, control_influence, control_density) = arm(false)?;
    let (treated_quantile, treated_influence, treated_density) = arm(true)?;
    let influence: Vec<f64> =
        treated_influence.iter().zip(&control_influence).map(|(a, c)| a - c).collect();
    let se = antecedent_stats::iid_se(&influence);
    let difference = treated_quantile - control_quantile;
    if !difference.is_finite() || !se.is_finite() {
        return Err(EstimationError::data_msg("the randomized quantile effect is not finite"));
    }
    let radius = NORMAL_Q975 * se;
    Ok(RandomizedQuantileEffect {
        tau,
        difference,
        control_quantile,
        treated_quantile,
        densities: [control_density, treated_density],
        influence,
        se,
        interval_95: [difference - radius, difference + radius],
        n_control,
        n_treated,
        thresholds: thresholds.to_vec(),
    })
}

/// Quantile treatment effect on the empirical grid of the pooled outcomes, the grid the
/// cross-fitted quantile cell uses ([`RANDOMIZED_QUANTILE_GRID_POINTS`] points).
///
/// # Errors
///
/// As [`randomized_quantile_effect_on_grid`], or fewer than sixteen finite outcomes or a
/// collapsed grid.
pub fn randomized_quantile_effect(
    outcome: &[f64],
    treated: &[bool],
    tau: f64,
) -> Result<RandomizedQuantileEffect, EstimationError> {
    let grid = empirical_threshold_grid(outcome, RANDOMIZED_QUANTILE_GRID_POINTS)?;
    randomized_quantile_effect_on_grid(outcome, treated, &grid, tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_arm_is_refused() {
        let outcome: Vec<f64> = (0..24).map(f64::from).collect();
        let mut treated = vec![false; 24];
        for flag in treated.iter_mut().take(5) {
            *flag = true;
        }
        assert!(randomized_quantile_effect(&outcome, &treated, 0.5).is_err());
    }
}
