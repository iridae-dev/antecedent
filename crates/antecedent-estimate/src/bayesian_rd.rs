//! Bayesian sharp regression discontinuity under a local-linear Gaussian model.
//!
//! The effect is the posterior distribution of the jump at the cutoff for a fixed rectangular
//! bandwidth. The model is `Y ~ 1 + T + (R-c) + T(R-c)` with independent zero-centered Normal
//! coefficient priors whose scale is `prior_sd` residual standard deviations (the prior covariance
//! is `v prior_sd^2 I` for the plug-in residual variance `v`, so rescaling the outcome rescales
//! the posterior) and a plug-in residual variance taken from the residuals of the posterior-mean
//! fit, which a tight prior therefore inflates. These claims apply only to this model and
//! declared bandwidth; bandwidth comparisons are an explicit sensitivity analysis.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;
use crate::util::DrawSpec;

/// Posterior summary for a sharp-RD jump under a fixed local-linear Gaussian model.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianRdEstimate {
    /// Posterior jump draws (coefficient of treatment at the cutoff).
    pub jump_draws: Vec<f64>,
    /// Posterior mean jump.
    pub mean: f64,
    /// Equal-tailed lower posterior quantile.
    pub lower: f64,
    /// Equal-tailed upper posterior quantile.
    pub upper: f64,
    /// Fixed bandwidth used in this fit.
    pub bandwidth: f64,
    /// Prior standard deviation of each local-linear coefficient in units of the plug-in
    /// residual standard deviation.
    pub prior_sd: f64,
    /// Number of observations in the window.
    pub n_window: usize,
    /// Explicit model scope.
    pub model: &'static str,
}

/// The observed vectors of a sharp-RD fit, aligned by row.
#[derive(Clone, Copy, Debug)]
pub struct RdSample<'a> {
    /// Running-variable values.
    pub running: &'a [f64],
    /// Treatment values (must equal `1{running >= cutoff}`).
    pub treatment: &'a [f64],
    /// Outcome values.
    pub outcome: &'a [f64],
}

/// Fit a local-linear Bayesian sharp RD at a caller-specified bandwidth.
///
/// Sharp assignment `T = 1{R >= cutoff}` is checked row by row. The posterior conditions on
/// that design, the fixed bandwidth, and the plug-in variance. At least 12 in-window rows and
/// four rows on each side are required.
pub fn fit_bayesian_sharp_rd(
    sample: &RdSample<'_>,
    cutoff: f64,
    bandwidth: f64,
    prior_sd: f64,
    sampling: DrawSpec,
) -> Result<BayesianRdEstimate, EstimationError> {
    let RdSample { running, treatment, outcome } = *sample;
    let DrawSpec { draws, seed } = sampling;
    let n = running.len();
    if n == 0
        || treatment.len() != n
        || outcome.len() != n
        || running.iter().chain(treatment).chain(outcome).any(|v| !v.is_finite())
        || !cutoff.is_finite()
        || !bandwidth.is_finite()
        || bandwidth <= 0.0
        || !prior_sd.is_finite()
        || prior_sd <= 0.0
        || draws < 2
    {
        return Err(EstimationError::stats_msg(
            "Bayesian sharp RD needs equal finite vectors, finite cutoff, bandwidth and prior_sd > 0, and draws >= 2",
        ));
    }
    let mut rows = Vec::new();
    let mut y = Vec::new();
    let mut left = 0;
    let mut right = 0;
    for i in 0..n {
        let assigned = f64::from(running[i] >= cutoff);
        // Sharp assignment is an exact binary design condition, not a noisy measurement.
        #[expect(clippy::float_cmp, reason = "the contract requires exact sharp 0/1 assignment")]
        let assignment_mismatch = treatment[i] != assigned;
        if assignment_mismatch {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("rd_assignment_not_sharp"),
                format!("row {i} has T={}, but sharp assignment implies {assigned}", treatment[i]),
            ));
        }
        let x = running[i] - cutoff;
        if x.abs() <= bandwidth {
            #[expect(
                clippy::float_cmp,
                reason = "assigned is constructed as the exact binary indicator 0 or 1"
            )]
            let on_right = assigned == 1.0;
            if on_right {
                right += 1
            } else {
                left += 1
            }
            rows.push([1.0, assigned, x, assigned * x]);
            y.push(outcome[i]);
        }
    }
    if rows.len() < 12 || left < 4 || right < 4 {
        return Err(EstimationError::stats_msg(format!(
            "rd_insufficient_window_support: window has {} rows ({left} left, {right} right); need at least 12 total and 4 on each side",
            rows.len()
        )));
    }
    let (beta, cov) = posterior(&rows, &y, prior_sd)?;
    let mut jump_draws = draw_column(beta, &cov, 1, draws, seed)?;
    jump_draws.sort_by(f64::total_cmp);
    let mean = jump_draws.iter().sum::<f64>() / draws as f64;
    let lower = q(&jump_draws, 0.025);
    let upper = q(&jump_draws, 0.975);
    Ok(BayesianRdEstimate {
        jump_draws,
        mean,
        lower,
        upper,
        bandwidth,
        prior_sd,
        n_window: rows.len(),
        model: "sharp-assignment local-linear Gaussian RD; fixed bandwidth; plug-in variance",
    })
}

/// Fit the same model over declared bandwidths and return sensitivity summaries.
pub fn bayesian_rd_bandwidth_sensitivity(
    sample: &RdSample<'_>,
    cutoff: f64,
    bandwidths: &[f64],
    prior_sd: f64,
    sampling: DrawSpec,
) -> Result<Vec<BayesianRdEstimate>, EstimationError> {
    if bandwidths.is_empty() {
        return Err(EstimationError::stats_msg(
            "bandwidth sensitivity requires at least one declared bandwidth",
        ));
    }
    bandwidths
        .iter()
        .enumerate()
        .map(|(i, b)| {
            fit_bayesian_sharp_rd(
                sample,
                cutoff,
                *b,
                prior_sd,
                DrawSpec::new(sampling.draws, sampling.seed.wrapping_add(i as u64)),
            )
        })
        .collect()
}

/// Refit the fixed-bandwidth model over caller-declared coefficient prior scales.
pub fn bayesian_rd_prior_sensitivity(
    sample: &RdSample<'_>,
    cutoff: f64,
    bandwidth: f64,
    prior_sds: &[f64],
    sampling: DrawSpec,
) -> Result<Vec<BayesianRdEstimate>, EstimationError> {
    if prior_sds.is_empty() {
        return Err(EstimationError::stats_msg(
            "prior sensitivity requires at least one declared prior scale",
        ));
    }
    prior_sds
        .iter()
        .enumerate()
        .map(|(i, sd)| {
            fit_bayesian_sharp_rd(
                sample,
                cutoff,
                bandwidth,
                *sd,
                DrawSpec::new(sampling.draws, sampling.seed.wrapping_add(i as u64)),
            )
        })
        .collect()
}

fn q(a: &[f64], p: f64) -> f64 {
    antecedent_stats::quantile_sorted(a, p, antecedent_stats::QuantileRule::ExchangeableRank)
}
fn posterior(
    x: &[[f64; 4]],
    y: &[f64],
    sd: f64,
) -> Result<([f64; 4], [[f64; 4]; 4]), EstimationError> {
    let mut a = [[0.0; 4]; 4];
    let mut b = [0.0; 4];
    for (row, yi) in x.iter().zip(y) {
        for i in 0..4 {
            b[i] += row[i] * yi;
            for j in 0..4 {
                a[i][j] += row[i] * row[j];
            }
        }
    }
    let p = 1.0 / (sd * sd);
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += p;
    }
    let inv = invert(a)?;
    let beta = std::array::from_fn(|i| (0..4).map(|j| inv[i][j] * b[j]).sum());
    let rss = x
        .iter()
        .zip(y)
        .map(|(r, yi)| (yi - (0..4).map(|j| r[j] * beta[j]).sum::<f64>()).powi(2))
        .sum::<f64>();
    let v = (rss / (y.len() - 4) as f64).max(1e-12);
    Ok((beta, inv.map(|r| r.map(|z| z * v))))
}
fn invert(a: [[f64; 4]; 4]) -> Result<[[f64; 4]; 4], EstimationError> {
    crate::util::invert_fixed(&a)
        .ok_or_else(|| EstimationError::stats_msg("Bayesian RD posterior design is singular"))
}
/// `n` seeded draws of one posterior coordinate, whose marginal is `N(mean[col], cov[col][col])`.
fn draw_column(
    mean: [f64; 4],
    cov: &[[f64; 4]; 4],
    col: usize,
    n: usize,
    seed: u64,
) -> Result<Vec<f64>, EstimationError> {
    crate::util::normal_draws(mean[col], cov[col][col], n, seed).ok_or_else(|| {
        EstimationError::stats_msg("Bayesian RD posterior covariance is not positive definite")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_linear_jump_recovers_known_truth_and_reports_bandwidth_sensitivity() {
        let n: usize = 400;
        let running: Vec<f64> = (0..n)
            .map(|i| (f64::from(i32::try_from(i).expect("test index fits i32")) - 200.0) / 100.0)
            .collect();
        let t: Vec<f64> = running.iter().map(|r| f64::from(*r >= 0.0)).collect();
        let y: Vec<f64> = running
            .iter()
            .enumerate()
            .map(|(i, r)| {
                1.0 + 2.5 * t[i] + 0.8 * r + 1.2 * t[i] * r + ((i % 5) as f64 - 2.0) * 0.01
            })
            .collect();
        let fit = fit_bayesian_sharp_rd(
            &RdSample { running: &running, treatment: &t, outcome: &y },
            0.0,
            0.8,
            20.0,
            DrawSpec::new(1000, 4),
        )
        .unwrap();
        assert!((fit.mean - 2.5).abs() < 0.08, "{}", fit.mean);
        assert!(fit.lower < 2.5 && fit.upper > 2.5);
        let sens = bayesian_rd_bandwidth_sensitivity(
            &RdSample { running: &running, treatment: &t, outcome: &y },
            0.0,
            &[0.5, 0.8],
            20.0,
            DrawSpec::new(300, 8),
        )
        .unwrap();
        assert_eq!(sens.len(), 2);
        assert_ne!(sens[0].n_window, sens[1].n_window);
        let prior = bayesian_rd_prior_sensitivity(
            &RdSample { running: &running, treatment: &t, outcome: &y },
            0.0,
            0.8,
            &[0.5, 20.0],
            DrawSpec::new(300, 8),
        )
        .unwrap();
        assert_eq!(prior.len(), 2);
        assert!((prior[0].mean - prior[1].mean).abs() > 1e-3);
    }
    #[test]
    fn adjacent_seeds_do_not_produce_the_same_first_draw() {
        let mean = [0.0; 4];
        let mut cov = [[0.0; 4]; 4];
        cov[1][1] = 1.0;
        let a = draw_column(mean, &cov, 1, 1, 1).unwrap();
        let b = draw_column(mean, &cov, 1, 1, 2).unwrap();
        assert!((a[0] - b[0]).abs() > 1e-3, "{} vs {}", a[0], b[0]);
    }

    #[test]
    fn non_sharp_assignment_is_refused() {
        let r: Vec<f64> = (0..20).map(|i| f64::from(i) - 10.0).collect();
        let mut t: Vec<f64> = r.iter().map(|x| f64::from(*x >= 0.0)).collect();
        t[8] = 1.0;
        let y = r.clone();
        let e = fit_bayesian_sharp_rd(
            &RdSample { running: &r, treatment: &t, outcome: &y },
            0.0,
            5.0,
            2.0,
            DrawSpec::new(20, 3),
        )
        .unwrap_err();
        assert!(e.to_string().contains("rd_assignment_not_sharp"));
    }
    #[test]
    fn the_posterior_inverse_is_scale_free_and_still_refuses_collinear_designs() {
        // Pivots of order 1e-16 were refused by the former absolute 1e-12 cutoff.
        let tiny = [
            [1e-16, 0.0, 0.0, 0.0],
            [0.0, 4e-16, 0.0, 0.0],
            [0.0, 0.0, 9e-16, 0.0],
            [0.0, 0.0, 0.0, 16e-16],
        ];
        assert!(invert(tiny).is_ok());
        let collinear = [
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let e = invert(collinear).unwrap_err();
        assert!(e.to_string().contains("Bayesian RD posterior design is singular"), "{e}");
    }
}
