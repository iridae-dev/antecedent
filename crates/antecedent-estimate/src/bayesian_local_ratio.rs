//! Bayesian fuzzy-jump and regression-kink local ratio.
//!
//! The estimand is the same ratio as the frequentist local-polynomial row: the outcome's jump (or
//! slope change) at the cutoff divided by the received treatment's jump (or slope change). The
//! design premises are unchanged and no prior supplies them: continuity of the untreated
//! potential outcomes and of the treatment propensity at the cutoff away from the declared change,
//! exclusion of the cutoff from the outcome except through treatment, and (for the jump) monotone
//! compliance.
//!
//! The outcome model is the fixed-bandwidth local-linear Gaussian model of the sharp posterior in
//! [`crate::bayesian_rd`], fitted to the outcome and to the treatment on the same design
//! `1 + A + (R-c) + A (R-c)` with `A = 1{R >= c}`, independent zero-centered Normal coefficient
//! priors whose scale is in residual standard deviations of the respective response (the prior
//! covariance of a response's coefficients is that response's residual variance times
//! `prior_sd^2`) and a plug-in 2x2 residual covariance. The two numerator and denominator coefficients
//! are drawn jointly from their bivariate Normal posterior, so the draws of the ratio carry the
//! outcome-treatment residual correlation.
//!
//! A ratio of two Normal posteriors has no finite mean and, when the denominator posterior has
//! mass near zero, no credible interval. The posterior mean of the ratio is never reported; the
//! posterior mass of the denominator within a declared floor of zero is reported, and the interval
//! is withheld when that mass exceeds the tail mass of the interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_prob::{LaplaceWorkspace, sample_gaussian_mvn};

use crate::error::EstimationError;
use crate::util::DrawSpec;

/// Denominator mass within the floor above which the ratio interval is withheld (the interval's
/// own tail mass).
const WEAK_FIRST_STAGE_MASS: f64 = 0.025;
/// Floor for a plug-in residual variance.
const VARIANCE_FLOOR: f64 = 1e-12;
/// Largest residual correlation allowed in the joint draw, so the covariance stays positive definite.
const CORRELATION_LIMIT: f64 = 1.0 - 1e-6;

/// Which pair of discontinuities forms the ratio.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalRatioKind {
    /// Outcome jump over treatment jump at the cutoff (fuzzy regression discontinuity).
    FuzzyJump,
    /// Outcome slope change over treatment slope change at the cutoff (regression kink).
    Kink,
}

/// Row-aligned observed vectors.
#[derive(Clone, Copy, Debug)]
pub struct LocalRatioSample<'a> {
    /// Running-variable values.
    pub running: &'a [f64],
    /// Received treatment or policy variable.
    pub treatment: &'a [f64],
    /// Outcome values.
    pub outcome: &'a [f64],
}

/// Declared design and model of a local-ratio posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalRatioSpec {
    /// Jump or kink.
    pub kind: LocalRatioKind,
    /// Cutoff of the running variable.
    pub cutoff: f64,
    /// Fixed rectangular bandwidth.
    pub bandwidth: f64,
    /// Prior standard deviation of every local-linear coefficient, in residual standard
    /// deviations of the response it belongs to.
    pub prior_sd: f64,
    /// Absolute denominator magnitude below which the ratio is treated as unidentified.
    pub first_stage_floor: f64,
}

/// Joint posterior of the numerator, denominator and ratio.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianLocalRatioEstimate {
    /// Ratio draws, paired by index with the two coefficient draws below.
    pub ratio_draws: Vec<f64>,
    /// Outcome jump (or slope change) draws.
    pub reduced_form_draws: Vec<f64>,
    /// Treatment jump (or slope change) draws.
    pub first_stage_draws: Vec<f64>,
    /// Posterior median of the ratio. The posterior mean is not reported: it does not exist.
    pub ratio_median: f64,
    /// Equal-tailed 95% interval, or `None` when the denominator has mass near zero.
    pub ratio_interval: Option<[f64; 2]>,
    /// Posterior mean of the denominator coefficient.
    pub first_stage_mean: f64,
    /// Posterior standard deviation of the denominator coefficient.
    pub first_stage_sd: f64,
    /// Posterior probability that the denominator coefficient is positive.
    pub first_stage_positive_probability: f64,
    /// Posterior mass of `|denominator| < first_stage_floor`.
    pub first_stage_near_zero_mass: f64,
    /// True when that mass exceeds the interval tail mass, so no interval is reported.
    pub weak_first_stage: bool,
    /// Fixed bandwidth used.
    pub bandwidth: f64,
    /// Number of rows in the window.
    pub n_window: usize,
    /// Explicit model scope.
    pub model: &'static str,
}

struct TwoResponseFit {
    beta_y: [f64; 4],
    beta_t: [f64; 4],
    inverse: [[f64; 4]; 4],
    residual_covariance: [[f64; 2]; 2],
}

fn fit_two_responses(
    rows: &[[f64; 4]],
    y: &[f64],
    t: &[f64],
    prior_sd: f64,
) -> Result<TwoResponseFit, EstimationError> {
    let mut gram = [[0.0; 4]; 4];
    let mut rhs_y = [0.0; 4];
    let mut rhs_t = [0.0; 4];
    for ((row, yi), ti) in rows.iter().zip(y).zip(t) {
        for i in 0..4 {
            rhs_y[i] += row[i] * yi;
            rhs_t[i] += row[i] * ti;
            for j in 0..4 {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    let precision = 1.0 / (prior_sd * prior_sd);
    for (i, row) in gram.iter_mut().enumerate() {
        row[i] += precision;
    }
    let inverse = crate::util::invert_fixed(&gram)
        .ok_or_else(|| EstimationError::stats_msg("local-ratio posterior design is singular"))?;
    let solve = |rhs: &[f64; 4]| -> [f64; 4] {
        std::array::from_fn(|i| (0..4).map(|j| inverse[i][j] * rhs[j]).sum())
    };
    let (beta_y, beta_t) = (solve(&rhs_y), solve(&rhs_t));
    let mut cross = [[0.0; 2]; 2];
    for ((row, yi), ti) in rows.iter().zip(y).zip(t) {
        let fitted = |beta: &[f64; 4]| (0..4).map(|j| row[j] * beta[j]).sum::<f64>();
        let residual = [yi - fitted(&beta_y), ti - fitted(&beta_t)];
        for a in 0..2 {
            for b in 0..2 {
                cross[a][b] += residual[a] * residual[b];
            }
        }
    }
    let dof = (rows.len() - 4) as f64;
    let mut covariance = cross.map(|r| r.map(|v| v / dof));
    covariance[0][0] = covariance[0][0].max(VARIANCE_FLOOR);
    covariance[1][1] = covariance[1][1].max(VARIANCE_FLOOR);
    let limit = CORRELATION_LIMIT * (covariance[0][0] * covariance[1][1]).sqrt();
    let off_diagonal = covariance[0][1].clamp(-limit, limit);
    covariance[0][1] = off_diagonal;
    covariance[1][0] = off_diagonal;
    Ok(TwoResponseFit { beta_y, beta_t, inverse, residual_covariance: covariance })
}

/// Design rows `1, A, R-c, A (R-c)` with the outcome and treatment of the rows inside the window.
/// Design rows, outcomes and treatments inside the bandwidth window.
type WindowRows = (Vec<[f64; 4]>, Vec<f64>, Vec<f64>);

fn window_rows(
    sample: &LocalRatioSample<'_>,
    cutoff: f64,
    bandwidth: f64,
) -> Result<WindowRows, EstimationError> {
    let LocalRatioSample { running, treatment, outcome } = *sample;
    let (mut rows, mut y, mut t) = (Vec::new(), Vec::new(), Vec::new());
    let (mut left, mut right) = (0_usize, 0_usize);
    for i in 0..running.len() {
        let x = running[i] - cutoff;
        if x.abs() <= bandwidth {
            let on_right = running[i] >= cutoff;
            if on_right {
                right += 1;
            } else {
                left += 1;
            }
            let a = f64::from(on_right);
            rows.push([1.0, a, x, a * x]);
            y.push(outcome[i]);
            t.push(treatment[i]);
        }
    }
    if rows.len() < 12 || left < 4 || right < 4 {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "local_ratio_insufficient_window_support",
            &format!(
                "window has {} rows ({left} left, {right} right); need at least 12 total and 4 \
                 on each side. Next attempt: a wider bandwidth, or more observations near the \
                 cutoff",
                rows.len()
            ),
        ));
    }
    Ok((rows, y, t))
}

/// Fit the joint local-linear posterior and form the ratio draws.
pub fn fit_bayesian_local_ratio(
    sample: &LocalRatioSample<'_>,
    spec: &LocalRatioSpec,
    sampling: DrawSpec,
) -> Result<BayesianLocalRatioEstimate, EstimationError> {
    let LocalRatioSample { running, treatment, outcome } = *sample;
    let LocalRatioSpec { kind, cutoff, bandwidth, prior_sd, first_stage_floor } = *spec;
    let n = running.len();
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if n == 0
        || treatment.len() != n
        || outcome.len() != n
        || running.iter().chain(treatment).chain(outcome).any(|v| !v.is_finite())
        || !cutoff.is_finite()
        || !positive(bandwidth)
        || !positive(prior_sd)
        || !positive(first_stage_floor)
        || sampling.draws < 2
    {
        return Err(EstimationError::stats_msg(
            "Bayesian local ratio needs equal finite vectors, finite cutoff, bandwidth, prior_sd and first_stage_floor > 0, and draws >= 2",
        ));
    }
    let (rows, y, t) = window_rows(sample, cutoff, bandwidth)?;
    let fit = fit_two_responses(&rows, &y, &t, prior_sd)?;
    let coefficient = match kind {
        LocalRatioKind::FuzzyJump => 1,
        LocalRatioKind::Kink => 3,
    };
    let scale = fit.inverse[coefficient][coefficient];
    let s = fit.residual_covariance;
    let covariance = [scale * s[0][0], scale * s[0][1], scale * s[1][0], scale * s[1][1]];
    let mean = [fit.beta_y[coefficient], fit.beta_t[coefficient]];
    let draws = sample_gaussian_mvn(
        &mean,
        &covariance,
        sampling.draws,
        sampling.seed,
        &mut LaplaceWorkspace::default(),
    )
    .map_err(|_| {
        EstimationError::stats_msg(
            "Bayesian local-ratio posterior covariance is not positive definite",
        )
    })?;
    let (reduced_form_draws, first_stage_draws) = draws.split_at(sampling.draws);
    let reduced_form_draws = reduced_form_draws.to_vec();
    let first_stage_draws = first_stage_draws.to_vec();
    let ratio_draws: Vec<f64> =
        reduced_form_draws.iter().zip(&first_stage_draws).map(|(num, den)| num / den).collect();
    let count = sampling.draws as f64;
    let near_zero_mass =
        first_stage_draws.iter().filter(|d| d.abs() < first_stage_floor).count() as f64 / count;
    let positive_probability =
        first_stage_draws.iter().filter(|d| **d > 0.0).count() as f64 / count;
    let first_stage_mean = first_stage_draws.iter().sum::<f64>() / count;
    let first_stage_sd =
        (first_stage_draws.iter().map(|d| (d - first_stage_mean).powi(2)).sum::<f64>()
            / (count - 1.0))
            .sqrt();
    let mut sorted = ratio_draws.clone();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| {
        antecedent_stats::quantile_sorted(
            &sorted,
            p,
            antecedent_stats::QuantileRule::ExchangeableRank,
        )
    };
    let weak_first_stage = near_zero_mass > WEAK_FIRST_STAGE_MASS;
    Ok(BayesianLocalRatioEstimate {
        ratio_median: q(0.5),
        ratio_interval: (!weak_first_stage).then(|| [q(0.025), q(0.975)]),
        ratio_draws,
        reduced_form_draws,
        first_stage_draws,
        first_stage_mean,
        first_stage_sd,
        first_stage_positive_probability: positive_probability,
        first_stage_near_zero_mass: near_zero_mass,
        weak_first_stage,
        bandwidth,
        n_window: rows.len(),
        model: "fixed-bandwidth local-linear Gaussian outcome and treatment; joint bivariate coefficient posterior; plug-in residual covariance; ratio of paired draws",
    })
}

#[cfg(test)]
mod tests;
