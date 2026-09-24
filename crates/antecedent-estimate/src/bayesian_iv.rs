//! Dedicated Bayesian linear IV estimator, scoped to a control-function structural model.
//!
//! The first stage is `T ~ 1 + Z`; the structural equation is `Y ~ 1 + T + v`,
//! where `v` is the structural treatment disturbance and its coefficient in the outcome
//! equation captures correlated structural errors (control function). Posterior draws propagate first-stage coefficient uncertainty and weight those draws by the
//! integrated conditional outcome likelihood before drawing outcome coefficients. This is a
//! joint posterior under declared fixed variances, not a generic IV guarantee. A
//! first-stage F statistic below the configured threshold is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;

/// Scoped Bayesian IV output. Intervals are equal-tailed posterior intervals conditional on
/// the selected model, prior scale, and plug-in Gaussian residual variance.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianIvEstimate {
    /// Posterior treatment-effect draws.
    pub effect_draws: Vec<f64>,
    /// Posterior mean of the treatment coefficient.
    pub mean: f64,
    /// Equal-tailed lower posterior quantile.
    pub lower: f64,
    /// Equal-tailed upper posterior quantile.
    pub upper: f64,
    /// Conventional first-stage F statistic for the instrument block.
    pub first_stage_f: f64,
    /// Prior standard deviation used for structural coefficients.
    pub prior_sd: f64,
    /// Explicit model scope.
    pub model: &'static str,
}

/// Fit the linear control-function IV model and sample its conditional Gaussian posterior.
///
/// `instrument` and `treatment` must be finite vectors of at least 8 rows. The instrument is
/// binary or continuous; the first-stage design includes an intercept. The structural model
/// currently uses one instrument and a homoskedastic Gaussian likelihood. This function does
/// not claim generic IV robustness or identification outside that model.
pub fn fit_bayesian_iv(
    instrument: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    prior_sd: f64,
    draws: usize,
    seed: u64,
    weak_f_threshold: f64,
) -> Result<BayesianIvEstimate, EstimationError> {
    fit_bayesian_iv_inner(
        instrument,
        treatment,
        outcome,
        prior_sd,
        draws,
        seed,
        weak_f_threshold,
        None,
    )
}

/// Fit using fixed, declared stage variances instead of estimating them from the observations.
/// The posterior remains model-based and limited to the stated control-function system.
pub fn fit_bayesian_iv_fixed_variances(
    instrument: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    prior_sd: f64,
    treatment_variance: f64,
    outcome_variance: f64,
    draws: usize,
    seed: u64,
    weak_f_threshold: f64,
) -> Result<BayesianIvEstimate, EstimationError> {
    if !treatment_variance.is_finite()
        || treatment_variance <= 0.0
        || !outcome_variance.is_finite()
        || outcome_variance <= 0.0
    {
        return Err(EstimationError::stats_msg(
            "declared IV stage variances must be finite and positive",
        ));
    }
    fit_bayesian_iv_inner(
        instrument,
        treatment,
        outcome,
        prior_sd,
        draws,
        seed,
        weak_f_threshold,
        Some((treatment_variance, outcome_variance)),
    )
}

/// Fit a jointly Gaussian structural IV model with a *fixed unit loading* on
/// the treatment disturbance in the outcome equation:
///
/// `T = alpha + pi Z + v`, `Y = mu + beta T + v + epsilon`.
///
/// The declared positive variances are those of independent `v` and `epsilon`.
/// Independent zero-mean `N(0, prior_sd²)` priors are placed on
/// `(alpha, pi, mu, beta)`. The fixed loading is a substantive restriction:
/// this route must not be used when the outcome disturbance has an unknown
/// loading on `v`.
pub fn fit_bayesian_iv_joint_fixed_loading(
    instrument: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    prior_sd: f64,
    treatment_variance: f64,
    outcome_variance: f64,
    draws: usize,
    seed: u64,
    weak_f_threshold: f64,
) -> Result<BayesianIvEstimate, EstimationError> {
    let n = instrument.len();
    if n < 8
        || treatment.len() != n
        || outcome.len() != n
        || instrument.iter().chain(treatment).chain(outcome).any(|v| !v.is_finite())
        || !prior_sd.is_finite()
        || prior_sd <= 0.0
        || !treatment_variance.is_finite()
        || treatment_variance <= 0.0
        || !outcome_variance.is_finite()
        || outcome_variance <= 0.0
        || draws < 2
        || !weak_f_threshold.is_finite()
        || weak_f_threshold < 0.0
    {
        return Err(EstimationError::stats_msg(
            "joint Bayesian IV requires equal finite vectors (n >= 8), positive prior and stage variances, draws >= 2, and a nonnegative weak-F threshold",
        ));
    }
    let zbar = mean(instrument);
    let tbar = mean(treatment);
    let zss = instrument.iter().map(|z| (z - zbar).powi(2)).sum::<f64>();
    if zss <= 1e-12 {
        return Err(EstimationError::stats_msg("Bayesian IV instrument has no variation"));
    }
    let pi =
        instrument.iter().zip(treatment).map(|(z, t)| (z - zbar) * (t - tbar)).sum::<f64>() / zss;
    let alpha = tbar - pi * zbar;
    let rss =
        instrument.iter().zip(treatment).map(|(z, t)| (t - alpha - pi * z).powi(2)).sum::<f64>();
    let first_stage_f = pi * pi * zss / (rss / (n - 2) as f64).max(1e-15);
    if first_stage_f < weak_f_threshold {
        return Err(EstimationError::stats_msg(format!(
            "weak_instrument: first-stage F={first_stage_f:.3} is below threshold {weak_f_threshold:.3}"
        )));
    }
    let mut precision = [[0.0; 4]; 4];
    let mut rhs = [0.0; 4];
    for ((&z, &t), &y) in instrument.iter().zip(treatment).zip(outcome) {
        let rows = [
            ([1.0, z, 0.0, 0.0], t, 1.0 / treatment_variance),
            ([-1.0, -z, 1.0, t], y - t, 1.0 / outcome_variance),
        ];
        for (x, target, weight) in rows {
            for i in 0..4 {
                rhs[i] += weight * x[i] * target;
                for j in 0..4 {
                    precision[i][j] += weight * x[i] * x[j];
                }
            }
        }
    }
    for (i, row) in precision.iter_mut().enumerate() {
        row[i] += 1.0 / prior_sd.powi(2);
    }
    let covariance = invert4(precision)?;
    let posterior_mean: [f64; 4] =
        std::array::from_fn(|i| (0..4).map(|j| covariance[i][j] * rhs[j]).sum());
    let mut chol = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let s = covariance[i][j] - (0..j).map(|k| chol[i][k] * chol[j][k]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return Err(EstimationError::stats_msg(
                        "joint Bayesian IV covariance is not positive definite",
                    ));
                }
                chol[i][j] = s.sqrt();
            } else {
                chol[i][j] = s / chol[j][j];
            }
        }
    }
    let mut state = seed.max(1);
    let mut effect_draws = Vec::with_capacity(draws);
    for _ in 0..draws {
        let z = [normal(&mut state), normal(&mut state), normal(&mut state), normal(&mut state)];
        effect_draws.push(posterior_mean[3] + (0..=3).map(|j| chol[3][j] * z[j]).sum::<f64>());
    }
    effect_draws.sort_by(f64::total_cmp);
    Ok(BayesianIvEstimate {
        mean: effect_draws.iter().sum::<f64>() / draws as f64,
        lower: quantile(&effect_draws, 0.025),
        upper: quantile(&effect_draws, 0.975),
        effect_draws,
        first_stage_f,
        prior_sd,
        model: "joint Gaussian structural IV; fixed unit disturbance loading and declared stage variances",
    })
}

fn invert4(a: [[f64; 4]; 4]) -> Result<[[f64; 4]; 4], EstimationError> {
    let mut aug = [[0.0; 8]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = a[i][j];
        }
        aug[i][i + 4] = 1.0;
    }
    for c in 0..4 {
        let pivot = (c..4).max_by(|i, j| aug[*i][c].abs().total_cmp(&aug[*j][c].abs())).unwrap();
        if aug[pivot][c].abs() < 1e-12 {
            return Err(EstimationError::stats_msg("joint Bayesian IV precision is singular"));
        }
        aug.swap(c, pivot);
        let divisor = aug[c][c];
        for j in 0..8 {
            aug[c][j] /= divisor;
        }
        for i in 0..4 {
            if i == c {
                continue;
            }
            let factor = aug[i][c];
            for j in 0..8 {
                aug[i][j] -= factor * aug[c][j];
            }
        }
    }
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| aug[i][j + 4])))
}

fn fit_bayesian_iv_inner(
    instrument: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    prior_sd: f64,
    draws: usize,
    seed: u64,
    weak_f_threshold: f64,
    known_variances: Option<(f64, f64)>,
) -> Result<BayesianIvEstimate, EstimationError> {
    let n = instrument.len();
    if n < 8
        || treatment.len() != n
        || outcome.len() != n
        || instrument.iter().chain(treatment).chain(outcome).any(|v| !v.is_finite())
        || !prior_sd.is_finite()
        || prior_sd <= 0.0
        || draws < 2
        || !weak_f_threshold.is_finite()
        || weak_f_threshold < 0.0
    {
        return Err(EstimationError::stats_msg(
            "Bayesian IV needs equal finite vectors (n >= 8), prior_sd > 0, draws >= 2, and a nonnegative weak-F threshold",
        ));
    }
    let zbar = mean(instrument);
    let tbar = mean(treatment);
    let zss = instrument.iter().map(|z| (z - zbar).powi(2)).sum::<f64>();
    if zss <= 1e-12 {
        return Err(EstimationError::stats_msg("Bayesian IV instrument has no variation"));
    }
    let pi =
        instrument.iter().zip(treatment).map(|(z, t)| (z - zbar) * (t - tbar)).sum::<f64>() / zss;
    let alpha = tbar - pi * zbar;
    let residual: Vec<f64> =
        instrument.iter().zip(treatment).map(|(z, t)| t - alpha - pi * z).collect();
    let rss = residual.iter().map(|v| v * v).sum::<f64>();
    let first_stage_f = pi * pi * zss / (rss / (n - 2) as f64).max(1e-15);
    if first_stage_f < weak_f_threshold {
        return Err(EstimationError::stats_msg(format!(
            "weak_instrument: first-stage F={first_stage_f:.3} is below threshold {weak_f_threshold:.3}"
        )));
    }
    let (stage_mean, stage_cov) =
        first_stage_posterior(instrument, treatment, prior_sd, known_variances.map(|x| x.0))?;
    let stage_draws = first_stage_draws(stage_mean, &stage_cov, draws, seed)?;
    let mut effect_draws = Vec::with_capacity(draws);
    let mut log_weights = Vec::with_capacity(draws);
    for (draw_idx, stage) in stage_draws.chunks_exact(2).enumerate() {
        let e: Vec<f64> =
            (0..n).map(|i| treatment[i] - stage[0] - stage[1] * instrument[i]).collect();
        let x: Vec<[f64; 3]> = (0..n).map(|i| [1.0, treatment[i], e[i]]).collect();
        let (beta, cov, log_evidence) =
            normal_posterior(&x, outcome, prior_sd, known_variances.map(|x| x.1))?;
        let sampled =
            gaussian_draws(&beta, &cov, 1, 1, seed.wrapping_add(draw_idx as u64).wrapping_add(1))?;
        effect_draws.push(sampled[1]);
        log_weights.push(log_evidence);
    }
    if known_variances.is_some() {
        effect_draws =
            weighted_resample(&effect_draws, &log_weights, draws, seed.wrapping_add(0xD1B5_4A32));
    }
    effect_draws.sort_by(f64::total_cmp);
    let mean_effect = effect_draws.iter().sum::<f64>() / draws as f64;
    let lower = quantile(&effect_draws, 0.025);
    let upper = quantile(&effect_draws, 0.975);
    Ok(BayesianIvEstimate {
        effect_draws,
        mean: mean_effect,
        lower,
        upper,
        first_stage_f,
        prior_sd,
        model: if known_variances.is_some() {
            "joint linear Gaussian control-function IV; declared fixed variances"
        } else {
            "linear Gaussian control-function IV; plug-in variance"
        },
    })
}

/// Evaluate structural-prior sensitivity on caller-declared scales.
pub fn bayesian_iv_prior_sensitivity(
    instrument: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    prior_sds: &[f64],
    draws: usize,
    seed: u64,
    weak_f_threshold: f64,
) -> Result<Vec<(f64, f64, f64)>, EstimationError> {
    if prior_sds.is_empty() {
        return Err(EstimationError::stats_msg(
            "prior sensitivity requires at least one prior scale",
        ));
    }
    prior_sds
        .iter()
        .enumerate()
        .map(|(i, sd)| {
            let fit = fit_bayesian_iv(
                instrument,
                treatment,
                outcome,
                *sd,
                draws,
                seed.wrapping_add(i as u64),
                weak_f_threshold,
            )?;
            Ok((*sd, fit.lower, fit.upper))
        })
        .collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let at = p * (sorted.len() - 1) as f64;
    let lo = at.floor() as usize;
    let hi = at.ceil() as usize;
    sorted[lo] + (at - lo as f64) * (sorted[hi] - sorted[lo])
}

fn normal_posterior(
    x: &[[f64; 3]],
    y: &[f64],
    sd: f64,
    known_variance: Option<f64>,
) -> Result<([f64; 3], [[f64; 3]; 3], f64), EstimationError> {
    let mut gram = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for (row, yi) in x.iter().zip(y) {
        for i in 0..3 {
            b[i] += row[i] * yi;
            for j in 0..3 {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    let ols_inv = invert(gram)?;
    let ols_beta = mul_vec(ols_inv, b);
    let rss = x.iter().zip(y).map(|(r, yi)| (yi - dot(r, &ols_beta)).powi(2)).sum::<f64>();
    let variance = known_variance.unwrap_or_else(|| (rss / (y.len() - 3) as f64).max(1e-12));
    let mut precision = gram;
    let ridge = variance / (sd * sd);
    for (i, row) in precision.iter_mut().enumerate() {
        row[i] += ridge;
    }
    let inv = invert(precision)?;
    let beta = mul_vec(inv, b);
    let cov = inv.map(|row| row.map(|v| v * variance));
    let det = determinant(precision);
    let log_det_covariance =
        y.len() as f64 * variance.ln() + det.ln() + 3.0 * ((sd * sd) / variance).ln();
    let quadratic = (y.iter().map(|v| v * v).sum::<f64>()
        - (0..3).map(|i| b[i] * beta[i]).sum::<f64>())
        / variance;
    let log_evidence =
        -0.5 * (y.len() as f64 * (std::f64::consts::TAU).ln() + log_det_covariance + quadratic);
    Ok((beta, cov, log_evidence))
}

fn determinant(a: [[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}
fn weighted_resample(values: &[f64], log_weights: &[f64], n: usize, seed: u64) -> Vec<f64> {
    let max = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weights.iter().map(|v| (v - max).exp()).collect();
    let total = weights.iter().sum::<f64>();
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut running = 0.0;
    for w in weights {
        running += w / total;
        cumulative.push(running);
    }
    let mut state = seed.max(1);
    let offset = uniform(&mut state) / n as f64;
    let mut out = Vec::with_capacity(n);
    let mut j = 0;
    for i in 0..n {
        let u = offset + i as f64 / n as f64;
        while j + 1 < cumulative.len() && cumulative[j] < u {
            j += 1;
        }
        out.push(values[j]);
    }
    out
}
fn uniform(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
}
fn first_stage_posterior(
    z: &[f64],
    t: &[f64],
    sd: f64,
    known_variance: Option<f64>,
) -> Result<([f64; 2], [[f64; 2]; 2]), EstimationError> {
    let n = z.len() as f64;
    let mut a =
        [[n, z.iter().sum::<f64>()], [z.iter().sum::<f64>(), z.iter().map(|v| v * v).sum::<f64>()]];
    let rhs = [t.iter().sum::<f64>(), z.iter().zip(t).map(|(zi, ti)| zi * ti).sum::<f64>()];
    let det0 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det0 <= 1e-12 {
        return Err(EstimationError::stats_msg("Bayesian IV first-stage posterior is singular"));
    }
    let ols = [
        (a[1][1] * rhs[0] - a[0][1] * rhs[1]) / det0,
        (-a[1][0] * rhs[0] + a[0][0] * rhs[1]) / det0,
    ];
    let rss = z.iter().zip(t).map(|(zi, ti)| (ti - ols[0] - ols[1] * zi).powi(2)).sum::<f64>();
    let variance = known_variance.unwrap_or_else(|| (rss / (z.len() - 2) as f64).max(1e-12));
    let ridge = variance / (sd * sd);
    a[0][0] += ridge;
    a[1][1] += ridge;
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let inv = [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]];
    let cov = inv.map(|row| row.map(|v| v * variance));
    let beta = [inv[0][0] * rhs[0] + inv[0][1] * rhs[1], inv[1][0] * rhs[0] + inv[1][1] * rhs[1]];
    Ok((beta, cov))
}
fn first_stage_draws(
    mean: [f64; 2],
    cov: &[[f64; 2]; 2],
    n: usize,
    seed: u64,
) -> Result<Vec<f64>, EstimationError> {
    let mut l = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..=i {
            let s = cov[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return Err(EstimationError::stats_msg(
                        "Bayesian IV first-stage covariance is not positive definite",
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(2 * n);
    for _ in 0..n {
        let z = [normal(&mut state), normal(&mut state)];
        for i in 0..2 {
            out.push(mean[i] + (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>());
        }
    }
    Ok(out)
}
fn invert(a: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], EstimationError> {
    let mut aug = [[0.0; 6]; 3];
    for i in 0..3 {
        for j in 0..3 {
            aug[i][j] = a[i][j];
        }
        aug[i][i + 3] = 1.0;
    }
    for c in 0..3 {
        let p = (c..3).max_by(|i, j| aug[*i][c].abs().total_cmp(&aug[*j][c].abs())).unwrap_or(c);
        if aug[p][c].abs() < 1e-12 {
            return Err(EstimationError::stats_msg("Bayesian IV posterior design is singular"));
        }
        aug.swap(c, p);
        let d = aug[c][c];
        for j in 0..6 {
            aug[c][j] /= d;
        }
        for i in 0..3 {
            if i != c {
                let f = aug[i][c];
                for j in 0..6 {
                    aug[i][j] -= f * aug[c][j];
                }
            }
        }
    }
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = aug[i][j + 3];
        }
    }
    Ok(out)
}
fn mul_vec(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * b[j]).sum())
}
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn gaussian_draws(
    mean: &[f64; 3],
    cov: &[[f64; 3]; 3],
    _cols: usize,
    n: usize,
    seed: u64,
) -> Result<Vec<f64>, EstimationError> {
    // Cholesky factor of the symmetric posterior covariance.
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let s = cov[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return Err(EstimationError::stats_msg(
                        "Bayesian IV posterior covariance is not positive definite",
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut rng = seed.max(1);
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        let z = [normal(&mut rng), normal(&mut rng), normal(&mut rng)];
        for i in 0..3 {
            out.push(mean[i] + (0..=i).map(|j| l[i][j] * z[j]).sum::<f64>());
        }
    }
    Ok(out)
}
fn normal(state: &mut u64) -> f64 {
    fn u(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        ((*s >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }
    let r = (-2.0 * u(state).ln()).sqrt();
    r * (std::f64::consts::TAU * u(state)).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_fixed_loading_intervals_calibrate_under_the_declared_law() {
        let mut covered = 0usize;
        let reps = 300usize;
        for rep in 0..reps {
            let mut state = (1_000_003_u64.wrapping_mul(rep as u64 + 1)).max(1);
            let z: Vec<f64> = (0..180).map(|i| f64::from(i % 2)).collect();
            let mut t = Vec::with_capacity(180);
            let mut y = Vec::with_capacity(180);
            for &zi in &z {
                let v = normal(&mut state);
                let ti = zi + v;
                t.push(ti);
                y.push(1.0 + 2.0 * ti + v + normal(&mut state));
            }
            let fit = fit_bayesian_iv_joint_fixed_loading(
                &z,
                &t,
                &y,
                10.0,
                1.0,
                1.0,
                800,
                30_000 + rep as u64,
                10.0,
            )
            .unwrap();
            covered += usize::from(fit.lower <= 2.0 && fit.upper >= 2.0);
        }
        let rate = covered as f64 / reps as f64;
        let mcse = (0.95 * 0.05 / reps as f64).sqrt();
        eprintln!("joint fixed-loading IV 95% interval coverage: {covered}/{reps} ({rate:.3})");
        assert!((rate - 0.95).abs() <= 3.0 * mcse, "joint IV coverage={rate:.3}");
    }
    #[test]
    fn strong_instrument_recovers_structural_effect_and_prior_sensitivity_is_visible() {
        let n = 240;
        let z: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
        let u: Vec<f64> = (0..n).map(|i| if i % 4 < 2 { -0.2 } else { 0.2 }).collect();
        let t: Vec<f64> = z.iter().zip(&u).map(|(z, u)| 0.25 + 0.65 * z + u).collect();
        let y: Vec<f64> = t.iter().zip(&u).map(|(t, u)| 1.0 + 2.0 * t + 1.5 * u).collect();
        let fit = fit_bayesian_iv(&z, &t, &y, 10.0, 2000, 23, 10.0).unwrap();
        assert!((fit.mean - 2.0).abs() < 0.15, "{}", fit.mean);
        assert!(fit.first_stage_f > 10.0);
        let narrow = fit_bayesian_iv(&z, &t, &y, 0.2, 1000, 4, 10.0).unwrap();
        assert!(
            (narrow.mean - fit.mean).abs() > 0.01,
            "prior sensitivity means: narrow={} broad={}",
            narrow.mean,
            fit.mean
        );
        let fixed =
            fit_bayesian_iv_fixed_variances(&z, &t, &y, 10.0, 0.04, 0.01, 1000, 17, 10.0).unwrap();
        assert!(fixed.model.contains("declared fixed variances"));
        assert!((fixed.mean - 2.0).abs() < 0.15);
    }
    #[test]
    fn weak_instrument_is_refused() {
        let n = 100;
        let z: Vec<f64> = (0..n).map(|i| ((i / 2) % 2) as f64).collect();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let y = t.clone();
        let e = fit_bayesian_iv(&z, &t, &y, 2.0, 100, 9, 10.0).unwrap_err();
        assert!(e.to_string().contains("weak_instrument"));
    }
}
