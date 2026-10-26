//! Dedicated Bayesian linear IV estimator, scoped to a control-function structural model.
//!
//! The first stage is `T ~ 1 + Z`; the structural equation is `Y ~ 1 + T + v`,
//! where `v` is the structural treatment disturbance and its coefficient in the outcome
//! equation captures correlated structural errors (control function). Posterior draws propagate
//! first-stage coefficient uncertainty. Under declared fixed variances those draws are weighted
//! by the integrated conditional outcome likelihood, giving a joint posterior; with plug-in
//! variances they are left unweighted. This is not a generic IV guarantee. A first-stage F
//! statistic below the configured threshold is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;
use crate::linalg::SpdFactor;
use crate::util::{DrawSpec, dot, mean};

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

/// The observed vectors of a single-instrument IV fit, aligned by row.
#[derive(Clone, Copy, Debug)]
pub struct IvSample<'a> {
    /// Instrument values.
    pub instrument: &'a [f64],
    /// Treatment values.
    pub treatment: &'a [f64],
    /// Outcome values.
    pub outcome: &'a [f64],
}

/// Declared stage variances of the fixed-variance control-function model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageVariances {
    /// Variance of the first-stage (treatment) disturbance.
    pub treatment: f64,
    /// Variance of the outcome disturbance.
    pub outcome: f64,
}

impl StageVariances {
    fn validate(self) -> Result<Self, EstimationError> {
        if !self.treatment.is_finite()
            || self.treatment <= 0.0
            || !self.outcome.is_finite()
            || self.outcome <= 0.0
        {
            return Err(EstimationError::stats_msg(
                "declared IV stage variances must be finite and positive",
            ));
        }
        Ok(self)
    }
}

/// Fit the linear control-function IV model and sample its conditional Gaussian posterior.
///
/// `sample` vectors must be finite and at least 8 rows. The instrument is
/// binary or continuous; the first-stage design includes an intercept. The structural model
/// currently uses one instrument and a homoskedastic Gaussian likelihood. This function does
/// not claim generic IV robustness or identification outside that model.
pub fn fit_bayesian_iv(
    sample: &IvSample<'_>,
    prior_sd: f64,
    sampling: DrawSpec,
    weak_f_threshold: f64,
) -> Result<BayesianIvEstimate, EstimationError> {
    fit_bayesian_iv_inner(sample, prior_sd, sampling, weak_f_threshold, None)
}

/// Fit using fixed, declared stage variances instead of estimating them from the observations.
/// The posterior remains model-based and limited to the stated control-function system.
pub fn fit_bayesian_iv_fixed_variances(
    sample: &IvSample<'_>,
    prior_sd: f64,
    variances: StageVariances,
    sampling: DrawSpec,
    weak_f_threshold: f64,
) -> Result<BayesianIvEstimate, EstimationError> {
    let variances = variances.validate()?;
    fit_bayesian_iv_inner(sample, prior_sd, sampling, weak_f_threshold, Some(variances))
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
    sample: &IvSample<'_>,
    prior_sd: f64,
    variances: StageVariances,
    sampling: DrawSpec,
    weak_f_threshold: f64,
) -> Result<BayesianIvEstimate, EstimationError> {
    let IvSample { instrument, treatment, outcome } = *sample;
    let StageVariances { treatment: treatment_variance, outcome: outcome_variance } = variances;
    let DrawSpec { draws, seed } = sampling;
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
    if antecedent_stats::column_is_constant(instrument, n, 0) {
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
    let covariance = crate::util::invert_fixed(&precision)
        .ok_or_else(|| EstimationError::stats_msg("joint Bayesian IV precision is singular"))?;
    let posterior_mean: [f64; 4] =
        std::array::from_fn(|i| (0..4).map(|j| covariance[i][j] * rhs[j]).sum());
    // Only the effect coordinate is reported, and its marginal posterior is Gaussian.
    let mut effect_draws =
        crate::util::normal_draws(posterior_mean[3], covariance[3][3], draws, seed).ok_or_else(
            || EstimationError::stats_msg("joint Bayesian IV covariance is not positive definite"),
        )?;
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

fn fit_bayesian_iv_inner(
    sample: &IvSample<'_>,
    prior_sd: f64,
    sampling: DrawSpec,
    weak_f_threshold: f64,
    known_variances: Option<StageVariances>,
) -> Result<BayesianIvEstimate, EstimationError> {
    let IvSample { instrument, treatment, outcome } = *sample;
    let DrawSpec { draws, seed } = sampling;
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
    if antecedent_stats::column_is_constant(instrument, n, 0) {
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
    let (stage_mean, stage_cov) = first_stage_posterior(
        instrument,
        treatment,
        prior_sd,
        known_variances.map(|v| v.treatment),
    )?;
    let stage_draws = first_stage_draws(stage_mean, &stage_cov, draws, seed)?;
    let _admitted = crate::util::admit_bound_array::<f64>(draws, 2)?;
    let mut effect_draws = Vec::with_capacity(draws);
    let mut log_weights = Vec::with_capacity(draws);
    for draw_idx in 0..draws {
        let stage = [stage_draws[0][draw_idx], stage_draws[1][draw_idx]];
        let e: Vec<f64> =
            (0..n).map(|i| treatment[i] - stage[0] - stage[1] * instrument[i]).collect();
        let x: Vec<[f64; 3]> = (0..n).map(|i| [1.0, treatment[i], e[i]]).collect();
        let NormalPosterior { beta, covariance: cov, log_evidence } =
            normal_posterior(&x, outcome, prior_sd, known_variances.map(|v| v.outcome))?;
        // The effect coordinate's marginal posterior, one seeded draw per first-stage draw.
        let sampled = crate::util::normal_draws(
            beta[1],
            cov[1][1],
            1,
            seed.wrapping_add(draw_idx as u64).wrapping_add(1),
        )
        .ok_or_else(|| {
            EstimationError::stats_msg("Bayesian IV posterior covariance is not positive definite")
        })?;
        effect_draws.push(sampled[0]);
        log_weights.push(log_evidence);
    }
    if known_variances.is_some() {
        effect_draws =
            weighted_resample(&effect_draws, &log_weights, draws, seed.wrapping_add(0xD1B5_4A32));
    }
    effect_draws.sort_by(f64::total_cmp);
    let mean_effect = mean(&effect_draws);
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

/// Exchangeable-rank (type-6) quantile of sorted posterior draws, the rule every
/// posterior-draw interval in the crate uses; `NaN` for an empty sample.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    antecedent_stats::quantile_sorted(sorted, p, antecedent_stats::QuantileRule::ExchangeableRank)
}

/// Posterior of the control-function outcome regression.
#[derive(Debug)]
struct NormalPosterior {
    /// Posterior mean of `(intercept, effect, control-function loading)`.
    beta: [f64; 3],
    /// Posterior covariance of the coefficients.
    covariance: [[f64; 3]; 3],
    /// Log marginal likelihood of the outcome under the prior.
    log_evidence: f64,
}

/// Plug-in residual variance `rss / df`, floored at a fixed fraction of the sample variance of
/// the response `y` (so the floor carries the response's units) and never below the smallest
/// positive `f64`.
fn plug_in_variance(rss: f64, df: usize, y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let centre = mean(y);
    let y_variance = y.iter().map(|v| (v - centre).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (rss / df as f64).max((VARIANCE_FLOOR_FRACTION * y_variance).max(f64::MIN_POSITIVE))
}

/// Fraction of `var(y)` below which a plug-in residual variance is treated as an exact fit.
const VARIANCE_FLOOR_FRACTION: f64 = 1e-12;

fn normal_posterior(
    x: &[[f64; 3]],
    y: &[f64],
    sd: f64,
    known_variance: Option<f64>,
) -> Result<NormalPosterior, EstimationError> {
    let singular =
        || EstimationError::stats_msg("Bayesian IV posterior design is singular");
    let mut gram = [0.0; 9];
    let mut b = [0.0; 3];
    for (row, yi) in x.iter().zip(y) {
        for i in 0..3 {
            b[i] += row[i] * yi;
            for j in 0..3 {
                gram[i * 3 + j] += row[i] * row[j];
            }
        }
    }
    let ols_beta = SpdFactor::new(&gram, 3).and_then(|f| f.solve(&b)).ok_or_else(singular)?;
    let rss = x.iter().zip(y).map(|(r, yi)| (yi - dot(r, &ols_beta)).powi(2)).sum::<f64>();
    let variance = known_variance.unwrap_or_else(|| plug_in_variance(rss, y.len() - 3, y));
    let ridge = variance / (sd * sd);
    let mut precision = gram;
    for i in 0..3 {
        precision[i * 4] += ridge;
    }
    let factor = SpdFactor::new(&precision, 3).ok_or_else(singular)?;
    let beta_vec = factor.solve(&b).ok_or_else(singular)?;
    let beta = [beta_vec[0], beta_vec[1], beta_vec[2]];
    let inverse = factor.inverse();
    let cov: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| inverse[i * 3 + j] * variance));
    let log_det_covariance =
        y.len() as f64 * variance.ln() + factor.log_det() + 3.0 * ((sd * sd) / variance).ln();
    // `yᵀy − bᵀβ` at the posterior mean is the penalized residual `‖y − Xβ‖² + ridge·‖β‖²`;
    // the residual form does not cancel a large `yᵀy` against a nearly equal `bᵀβ`.
    let penalized_rss = x.iter().zip(y).map(|(r, yi)| (yi - dot(r, &beta)).powi(2)).sum::<f64>()
        + ridge * beta.iter().map(|v| v * v).sum::<f64>();
    let quadratic = penalized_rss / variance;
    let log_evidence =
        -0.5 * (y.len() as f64 * (std::f64::consts::TAU).ln() + log_det_covariance + quadratic);
    Ok(NormalPosterior { beta, covariance: cov, log_evidence })
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
    let offset = antecedent_core::CausalRng::from_seed(seed).next_f64() / n as f64;
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
fn first_stage_posterior(
    z: &[f64],
    t: &[f64],
    sd: f64,
    known_variance: Option<f64>,
) -> Result<([f64; 2], [[f64; 2]; 2]), EstimationError> {
    let singular = || EstimationError::stats_msg("Bayesian IV first-stage posterior is singular");
    let n = z.len() as f64;
    let sum_z = z.iter().sum::<f64>();
    let mut a = [[n, sum_z], [sum_z, z.iter().map(|v| v * v).sum::<f64>()]];
    let rhs = [t.iter().sum::<f64>(), z.iter().zip(t).map(|(zi, ti)| zi * ti).sum::<f64>()];
    let ols_inv = crate::util::invert_fixed(&a).ok_or_else(singular)?;
    let ols = [
        ols_inv[0][0] * rhs[0] + ols_inv[0][1] * rhs[1],
        ols_inv[1][0] * rhs[0] + ols_inv[1][1] * rhs[1],
    ];
    let rss = z.iter().zip(t).map(|(zi, ti)| (ti - ols[0] - ols[1] * zi).powi(2)).sum::<f64>();
    let variance = known_variance.unwrap_or_else(|| plug_in_variance(rss, z.len() - 2, t));
    let ridge = variance / (sd * sd);
    a[0][0] += ridge;
    a[1][1] += ridge;
    let inv = crate::util::invert_fixed(&a).ok_or_else(singular)?;
    let cov = inv.map(|row| row.map(|v| v * variance));
    let beta = [inv[0][0] * rhs[0] + inv[0][1] * rhs[1], inv[1][0] * rhs[0] + inv[1][1] * rhs[1]];
    Ok((beta, cov))
}
/// `n` draws of each first-stage coefficient: `[intercept draws, slope draws]`, drawn jointly
/// by the shared multivariate sampler.
fn first_stage_draws(
    mean: [f64; 2],
    cov: &[[f64; 2]; 2],
    n: usize,
    seed: u64,
) -> Result<[Vec<f64>; 2], EstimationError> {
    let flat = [cov[0][0], cov[0][1], cov[1][0], cov[1][1]];
    let draws = antecedent_prob::sample_gaussian_mvn(
        &mean,
        &flat,
        n,
        seed,
        &mut antecedent_prob::LaplaceWorkspace::default(),
    )
    .map_err(|_| {
        EstimationError::stats_msg("Bayesian IV first-stage covariance is not positive definite")
    })?;
    Ok([draws[..n].to_vec(), draws[n..2 * n].to_vec()])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn consecutive_seeds_start_independent_standard_normal_streams() {
        let first: Vec<f64> = (0..4000_u64)
            .map(|seed| crate::util::normal_draws(0.0, 1.0, 1, 42 + seed).unwrap()[0])
            .collect();
        let mean = first.iter().sum::<f64>() / first.len() as f64;
        let var = first.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / first.len() as f64;
        let lag1 = first.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum::<f64>()
            / (first.len() - 1) as f64
            / var;
        assert!((var - 1.0).abs() < 0.1, "first-draw variance {var}");
        assert!(lag1.abs() < 0.1, "lag-1 correlation {lag1}");
    }

    #[test]
    fn joint_fixed_loading_intervals_calibrate_under_the_declared_law() {
        let mut covered = 0usize;
        let reps = 300usize;
        for rep in 0..reps {
            let mut rng = antecedent_core::CausalRng::from_seed(1_000_003 * (rep as u64 + 1));
            let z: Vec<f64> = (0..180).map(|i| f64::from(i % 2)).collect();
            let mut t = Vec::with_capacity(180);
            let mut y = Vec::with_capacity(180);
            for &zi in &z {
                let v = antecedent_kernels::standard_normal(&mut rng);
                let ti = zi + v;
                t.push(ti);
                y.push(1.0 + 2.0 * ti + v + antecedent_kernels::standard_normal(&mut rng));
            }
            let fit = fit_bayesian_iv_joint_fixed_loading(
                &IvSample { instrument: &z, treatment: &t, outcome: &y },
                10.0,
                StageVariances { treatment: 1.0, outcome: 1.0 },
                DrawSpec::new(800, 30_000 + rep as u64),
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
        let fit = fit_bayesian_iv(
            &IvSample { instrument: &z, treatment: &t, outcome: &y },
            10.0,
            DrawSpec::new(2000, 23),
            10.0,
        )
        .unwrap();
        assert!((fit.mean - 2.0).abs() < 0.15, "{}", fit.mean);
        assert!(fit.first_stage_f > 10.0);
        let narrow = fit_bayesian_iv(
            &IvSample { instrument: &z, treatment: &t, outcome: &y },
            0.2,
            DrawSpec::new(1000, 4),
            10.0,
        )
        .unwrap();
        assert!(
            (narrow.mean - fit.mean).abs() > 0.01,
            "prior sensitivity means: narrow={} broad={}",
            narrow.mean,
            fit.mean
        );
        let fixed = fit_bayesian_iv_fixed_variances(
            &IvSample { instrument: &z, treatment: &t, outcome: &y },
            10.0,
            StageVariances { treatment: 0.04, outcome: 0.01 },
            DrawSpec::new(1000, 17),
            10.0,
        )
        .unwrap();
        assert!(fixed.model.contains("declared fixed variances"));
        assert!((fixed.mean - 2.0).abs() < 0.15);
    }
    #[test]
    fn weak_instrument_is_refused() {
        let n = 100;
        let z: Vec<f64> = (0..n).map(|i| f64::from((i / 2) % 2)).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
        let y = t.clone();
        let e = fit_bayesian_iv(
            &IvSample { instrument: &z, treatment: &t, outcome: &y },
            2.0,
            DrawSpec::new(100, 9),
            10.0,
        )
        .unwrap_err();
        assert!(e.to_string().contains("weak_instrument"));
    }
    #[test]
    fn the_posterior_is_scale_free_and_still_refuses_collinear_designs() {
        // Columns in units of 1e-8 are well conditioned once the scale is accounted for.
        let rows = |scale: f64, collinear: bool| -> Vec<[f64; 3]> {
            (0..12)
                .map(|i| {
                    let u = f64::from(i % 5) - 2.0;
                    let w = if collinear { 2.0 * u } else { f64::from((i * 7) % 4) - 1.5 };
                    [scale, scale * u, scale * w]
                })
                .collect()
        };
        let y: Vec<f64> = (0..12).map(|i| f64::from(i % 3) + 0.1 * f64::from(i)).collect();
        assert!(normal_posterior(&rows(1e-8, false), &y, 10.0, Some(1.0)).is_ok());
        let e = normal_posterior(&rows(1.0, true), &y, 10.0, Some(1.0)).unwrap_err();
        assert!(e.to_string().contains("Bayesian IV posterior design is singular"), "{e}");
    }

    #[test]
    fn the_plug_in_variance_floor_scales_with_the_response() {
        let y: Vec<f64> = (0..10).map(|i| 1e6 * f64::from(i)).collect();
        let floor = plug_in_variance(0.0, 7, &y);
        let small: Vec<f64> = y.iter().map(|v| v * 1e-12).collect();
        assert!((floor / plug_in_variance(0.0, 7, &small) - 1e24).abs() / 1e24 < 1e-9);
    }
}
