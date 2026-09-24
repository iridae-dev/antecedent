//! Bayesian sharp regression discontinuity under a local-linear Gaussian model.
//!
//! The effect is the posterior distribution of the jump at the cutoff for a fixed rectangular
//! bandwidth. The model is `Y ~ 1 + T + (R-c) + T(R-c)` with independent zero-centered Normal
//! coefficient priors and a plug-in residual variance. These claims apply only to this model and
//! declared bandwidth; bandwidth comparisons are an explicit sensitivity analysis.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;

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
    /// Prior standard deviation for each local-linear coefficient.
    pub prior_sd: f64,
    /// Number of observations in the window.
    pub n_window: usize,
    /// Explicit model scope.
    pub model: &'static str,
}

/// Fit a local-linear Bayesian sharp RD at a caller-specified bandwidth.
///
/// Sharp assignment `T = 1{R >= cutoff}` is checked row by row. The posterior conditions on
/// that design, the fixed bandwidth, and the plug-in variance. At least 12 in-window rows and
/// four rows on each side are required.
pub fn fit_bayesian_sharp_rd(
    running: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    cutoff: f64,
    bandwidth: f64,
    prior_sd: f64,
    draws: usize,
    seed: u64,
) -> Result<BayesianRdEstimate, EstimationError> {
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
        if treatment[i] != assigned {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("rd_assignment_not_sharp"),
                format!("row {i} has T={}, but sharp assignment implies {assigned}", treatment[i]),
            ));
        }
        let x = running[i] - cutoff;
        if x.abs() <= bandwidth {
            if assigned == 1.0 {
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
    running: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    cutoff: f64,
    bandwidths: &[f64],
    prior_sd: f64,
    draws: usize,
    seed: u64,
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
                running,
                treatment,
                outcome,
                cutoff,
                *b,
                prior_sd,
                draws,
                seed.wrapping_add(i as u64),
            )
        })
        .collect()
}

/// Refit the fixed-bandwidth model over caller-declared coefficient prior scales.
pub fn bayesian_rd_prior_sensitivity(
    running: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    cutoff: f64,
    bandwidth: f64,
    prior_sds: &[f64],
    draws: usize,
    seed: u64,
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
                running,
                treatment,
                outcome,
                cutoff,
                bandwidth,
                *sd,
                draws,
                seed.wrapping_add(i as u64),
            )
        })
        .collect()
}

fn q(a: &[f64], p: f64) -> f64 {
    let k = p * (a.len() - 1) as f64;
    let lo = k.floor() as usize;
    let hi = k.ceil() as usize;
    a[lo] + (k - lo as f64) * (a[hi] - a[lo])
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
    let mut aug = [[0.0; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = a[i][j];
        }
        aug[i][i + 4] = 1.0;
    }
    for c in 0..4 {
        let p = (c..4).max_by(|i, j| aug[*i][c].abs().total_cmp(&aug[*j][c].abs())).unwrap_or(c);
        if aug[p][c].abs() < 1e-12 {
            return Err(EstimationError::stats_msg("Bayesian RD posterior design is singular"));
        }
        aug.swap(c, p);
        let d = aug[c][c];
        for j in 0..8 {
            aug[c][j] /= d;
        }
        for i in 0..4 {
            if i != c {
                let f = aug[i][c];
                for j in 0..8 {
                    aug[i][j] -= f * aug[c][j];
                }
            }
        }
    }
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = aug[i][j + 4];
        }
    }
    Ok(out)
}
fn draw_column(
    mean: [f64; 4],
    cov: &[[f64; 4]; 4],
    col: usize,
    n: usize,
    seed: u64,
) -> Result<Vec<f64>, EstimationError> {
    let mut l = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let s = cov[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return Err(EstimationError::stats_msg(
                        "Bayesian RD posterior covariance is not positive definite",
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let z = [normal(&mut state), normal(&mut state), normal(&mut state), normal(&mut state)];
        out.push(mean[col] + (0..4).map(|j| l[col][j] * z[j]).sum::<f64>());
    }
    Ok(out)
}
fn normal(s: &mut u64) -> f64 {
    fn u(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }
    let r = (-2.0 * u(s).ln()).sqrt();
    r * (std::f64::consts::TAU * u(s)).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_linear_jump_recovers_known_truth_and_reports_bandwidth_sensitivity() {
        let n = 400;
        let running: Vec<f64> = (0..n).map(|i| (i as f64 - 200.0) / 100.0).collect();
        let t: Vec<f64> = running.iter().map(|r| f64::from(*r >= 0.0)).collect();
        let y: Vec<f64> = running
            .iter()
            .enumerate()
            .map(|(i, r)| {
                1.0 + 2.5 * t[i] + 0.8 * r + 1.2 * t[i] * r + ((i % 5) as f64 - 2.0) * 0.01
            })
            .collect();
        let fit = fit_bayesian_sharp_rd(&running, &t, &y, 0.0, 0.8, 20.0, 1000, 4).unwrap();
        assert!((fit.mean - 2.5).abs() < 0.08, "{}", fit.mean);
        assert!(fit.lower < 2.5 && fit.upper > 2.5);
        let sens =
            bayesian_rd_bandwidth_sensitivity(&running, &t, &y, 0.0, &[0.5, 0.8], 20.0, 300, 8)
                .unwrap();
        assert_eq!(sens.len(), 2);
        assert_ne!(sens[0].n_window, sens[1].n_window);
        let prior = bayesian_rd_prior_sensitivity(&running, &t, &y, 0.0, 0.8, &[0.5, 20.0], 300, 8)
            .unwrap();
        assert_eq!(prior.len(), 2);
        assert!((prior[0].mean - prior[1].mean).abs() > 1e-3);
    }
    #[test]
    fn non_sharp_assignment_is_refused() {
        let r: Vec<f64> = (0..20).map(|i| i as f64 - 10.0).collect();
        let mut t: Vec<f64> = r.iter().map(|x| f64::from(*x >= 0.0)).collect();
        t[8] = 1.0;
        let y = r.clone();
        let e = fit_bayesian_sharp_rd(&r, &t, &y, 0.0, 5.0, 2.0, 20, 3).unwrap_err();
        assert!(e.to_string().contains("rd_assignment_not_sharp"));
    }
}
