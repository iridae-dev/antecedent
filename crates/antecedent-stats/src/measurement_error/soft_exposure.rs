//! Per-unit soft exposures: a stated probability that each unit's exposure is present.
//!
//! Each unit carries `q_i = P(X_i = 1 | what the analysis conditions on)`, a calibrated
//! probability, rather than a recorded 0/1 value. Two combinations are distinct:
//!
//! * **posterior-mean substitution** replaces `X_i` by `q_i` in a linear outcome model
//!   `E[Y | X, Z] = V' beta`. With `X_i = q_i + eta_i`, `E[eta_i | q_i, Z_i] = 0` and
//!   `Var(eta_i | q_i) = q_i (1 - q_i)`, so least squares on `q` is consistent for the
//!   coefficient of `X` and its residual carries the extra variance `beta^2 q (1 - q)`.
//!   The reported covariance is the HC0 sandwich, which includes that variance; it is split
//!   into the part the exposure uncertainty contributes under that algebra and the rest;
//! * **multiple imputation** draws `X_i` from its law *given the outcome* and combines one
//!   fit per draw by Rubin's rules. A draw from `q_i` alone is Berkson noise and attenuates
//!   the slope, so the draws are conditioned on the outcome through a Gaussian outcome
//!   model fitted by EM, with the model's parameters redrawn by a bootstrap for each
//!   imputation.
//!
//! A reliability diagnostic compares the stated probabilities with repeat codings of the
//! same units, in bins of the stated probability. It is a report; nothing here gates on it.

use antecedent_core::CausalRng;

use crate::gram::{invert_square, solve_spd};

use super::{
    BinaryErrorRates, CombinedImputations, ErrorVariance, LinearErrorData, LinearMoments,
    MeasurementErrorError, combine_imputations_linear, finite_all,
};

/// Columns of a linear outcome analysis with a soft exposure.
#[derive(Clone, Copy, Debug)]
pub struct SoftExposureData<'a> {
    /// Outcome.
    pub outcome: &'a [f64],
    /// Per-unit probability that the exposure is present, each in `[0, 1]`.
    pub probability: &'a [f64],
    /// Error-free covariates, column-major `n x covariate_count`.
    pub covariates: &'a [f64],
    /// Number of covariate columns.
    pub covariate_count: usize,
}

fn check_soft(data: &SoftExposureData<'_>) -> Result<usize, MeasurementErrorError> {
    let n = data.outcome.len();
    if data.probability.len() != n
        || data.covariate_count.checked_mul(n) != Some(data.covariates.len())
    {
        return Err(MeasurementErrorError::Shape("soft exposure columns differ in length"));
    }
    if !(finite_all(data.outcome) && finite_all(data.covariates) && finite_all(data.probability)) {
        return Err(MeasurementErrorError::NonFinite("outcome, probabilities or covariates"));
    }
    if data.probability.iter().any(|q| !(0.0..=1.0).contains(q)) {
        return Err(MeasurementErrorError::InvalidParameter(
            "stated exposure probabilities must lie in [0, 1]",
        ));
    }
    Ok(n)
}

/// A linear fit on posterior-mean exposures, with the exposure-uncertainty term.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftExposureSubstitution {
    /// Rows.
    pub rows: usize,
    /// Design width; coefficient 0 is the intercept and 1 the exposure.
    pub dim: usize,
    /// Least-squares coefficients on the stated probabilities.
    pub coefficients: Vec<f64>,
    /// HC0 covariance, row-major `dim x dim`.
    pub covariance: Vec<f64>,
    /// Mean posterior variance `mean(q (1 - q))` of the exposure.
    pub mean_posterior_variance: f64,
    /// Variance of the exposure coefficient.
    pub variance: f64,
    /// Part of that variance the exposure uncertainty contributes: the extra residual
    /// variance `beta^2 mean(q (1 - q))` passed through `(M^{-1})_{11} / n`, at most the total.
    pub variance_exposure_uncertainty: f64,
    /// The remainder: sampling variance given the exposure.
    pub variance_sampling: f64,
}

/// Regress the outcome on the per-unit posterior mean of the exposure.
///
/// # Errors
/// A shape or range violation, too few rows or a rank-deficient design refuse.
pub fn soft_exposure_substitution(
    data: &SoftExposureData<'_>,
) -> Result<SoftExposureSubstitution, MeasurementErrorError> {
    let n = check_soft(data)?;
    let exact = ErrorVariance::known(0.0, 0.0)?;
    let raw = LinearErrorData {
        y: data.outcome,
        w: data.probability,
        covariates: data.covariates,
        covariate_count: data.covariate_count,
        replicate_counts: None,
    };
    let moments = LinearMoments::accumulate(&raw, &exact, 0.0)?;
    let fit = moments.solve()?;
    let d = fit.dim;
    let beta = fit.naive_coefficients[1];
    let variance = fit.naive_covariance[d + 1];
    let mean_var = data.probability.iter().map(|q| q * (1.0 - q)).sum::<f64>() / n as f64;
    let inverse = invert_square(&moments.gram, d).ok_or(MeasurementErrorError::SingularDesign)?;
    let from_exposure = (beta * beta * mean_var * inverse[d + 1] / n as f64).clamp(0.0, variance);
    Ok(SoftExposureSubstitution {
        rows: n,
        dim: d,
        coefficients: fit.naive_coefficients,
        covariance: fit.naive_covariance,
        mean_posterior_variance: mean_var,
        variance,
        variance_exposure_uncertainty: from_exposure,
        variance_sampling: variance - from_exposure,
    })
}

/// Multiple imputation of the exposure over outcome-conditioned draws, by Rubin's rules.
///
/// `draws` is draw-major, `draw_count` vectors of `n` values in `{0, 1}`, each drawn from the
/// exposure's law given the outcome.
///
/// # Errors
/// Fewer than two draws, a shape mismatch or a singular fit refuse.
pub fn soft_exposure_imputation(
    outcome: &[f64],
    draws: &[f64],
    draw_count: usize,
    covariates: &[f64],
    covariate_count: usize,
) -> Result<CombinedImputations, MeasurementErrorError> {
    combine_imputations_linear(outcome, draws, draw_count, covariates, covariate_count)
}

/// Outcome-conditioned exposure draws.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftExposureDraws {
    /// Draw-major `{0, 1}` exposures, `draw_count * n` values.
    pub draws: Vec<f64>,
    /// Number of draws.
    pub draw_count: usize,
    /// Coefficients of the Gaussian outcome model fitted to the full data by EM (intercept,
    /// exposure, covariates).
    pub coefficients: Vec<f64>,
    /// Residual variance of that fit.
    pub residual_variance: f64,
    /// EM iterations of the full-data fit.
    pub iterations: usize,
}

struct GaussianMixtureFit {
    beta: Vec<f64>,
    sigma2: f64,
    iterations: usize,
}

/// Posterior probability that unit `i`'s exposure is present given its outcome.
fn responsibility(q: f64, y: f64, mu0: f64, mu1: f64, sigma2: f64) -> f64 {
    if q <= 0.0 {
        return 0.0;
    }
    if q >= 1.0 {
        return 1.0;
    }
    let log_odds =
        (q / (1.0 - q)).ln() + ((y - mu0) * (y - mu0) - (y - mu1) * (y - mu1)) / (2.0 * sigma2);
    1.0 / (1.0 + (-log_odds).exp())
}

/// EM for `Y_i ~ N(V_i(x)' beta, sigma^2)` with latent `X_i ~ Bernoulli(q_i)` over the rows
/// listed in `rows` (with multiplicity).
fn fit_mixture(
    data: &SoftExposureData<'_>,
    rows: &[usize],
    start: Option<(&[f64], f64)>,
) -> Result<GaussianMixtureFit, MeasurementErrorError> {
    let n = data.outcome.len();
    let p = data.covariate_count;
    let d = p + 2;
    let m = rows.len() as f64;
    let design = |i: usize, x: f64| -> Vec<f64> {
        let mut v = Vec::with_capacity(d);
        v.push(1.0);
        v.push(x);
        for j in 0..p {
            v.push(data.covariates[j * n + i]);
        }
        v
    };
    let m_step = |resp: &[f64]| -> Result<(Vec<f64>, f64), MeasurementErrorError> {
        let mut a = vec![0.0; d * d];
        let mut b = vec![0.0; d];
        for (slot, &i) in rows.iter().enumerate() {
            let r = resp[slot];
            for (x, w) in [(1.0, r), (0.0, 1.0 - r)] {
                if w <= 0.0 {
                    continue;
                }
                let v = design(i, x);
                for s in 0..d {
                    b[s] += w * v[s] * data.outcome[i];
                    for t in 0..d {
                        a[s * d + t] += w * v[s] * v[t];
                    }
                }
            }
        }
        let beta = solve_spd(&a, d, &b).ok_or(MeasurementErrorError::SingularDesign)?;
        let mut rss = 0.0;
        for (slot, &i) in rows.iter().enumerate() {
            let r = resp[slot];
            for (x, w) in [(1.0, r), (0.0, 1.0 - r)] {
                if w <= 0.0 {
                    continue;
                }
                let mu: f64 = design(i, x).iter().zip(&beta).map(|(v, c)| v * c).sum();
                rss += w * (data.outcome[i] - mu) * (data.outcome[i] - mu);
            }
        }
        Ok((beta, rss / m))
    };
    let (mut beta, mut sigma2) = if let Some((b, s)) = start {
        (b.to_vec(), s)
    } else {
        let resp: Vec<f64> = rows.iter().map(|&i| data.probability[i]).collect();
        m_step(&resp)?
    };
    let floor = 1e-10;
    let mut iterations = 0;
    for step in 1..=500 {
        iterations = step;
        if !(sigma2.is_finite() && sigma2 > floor) {
            return Err(MeasurementErrorError::NonPositiveVariance(
                "residual of the soft-exposure outcome model",
            ));
        }
        let resp: Vec<f64> = rows
            .iter()
            .map(|&i| {
                let mu0: f64 = design(i, 0.0).iter().zip(&beta).map(|(v, c)| v * c).sum();
                let mu1: f64 = design(i, 1.0).iter().zip(&beta).map(|(v, c)| v * c).sum();
                responsibility(data.probability[i], data.outcome[i], mu0, mu1, sigma2)
            })
            .collect();
        let (next, next_sigma2) = m_step(&resp)?;
        let change = next
            .iter()
            .zip(&beta)
            .map(|(a, b)| (a - b).abs())
            .fold((next_sigma2 - sigma2).abs(), f64::max);
        beta = next;
        sigma2 = next_sigma2;
        if change < 1e-9 {
            break;
        }
    }
    Ok(GaussianMixtureFit { beta, sigma2, iterations })
}

/// Draw outcome-conditioned exposures for multiple imputation.
///
/// A Gaussian outcome model with the exposure as a binary regressor is fitted by EM with
/// the stated probabilities as the exposure's prior. For each draw the rows are resampled,
/// the model refitted on the resample, and every unit's exposure drawn from its law given
/// its outcome under that refit. The draws are proper imputations only when the outcome
/// model is right, the probabilities are the exposure's prior given the covariates and the
/// exposure is not informative about the outcome beyond the model.
///
/// # Errors
/// A shape or range violation, fewer than two draws, too few rows or a rank-deficient or
/// degenerate fit refuse.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a uniform draw scaled to a row index below the row count"
)]
pub fn soft_exposure_outcome_conditioned_draws(
    data: &SoftExposureData<'_>,
    draw_count: usize,
    seed: u64,
) -> Result<SoftExposureDraws, MeasurementErrorError> {
    let n = check_soft(data)?;
    if draw_count < 2 {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "Rubin's rules need at least two imputations",
        ));
    }
    let d = data.covariate_count + 2;
    if n < 3 * d {
        return Err(MeasurementErrorError::TooFewRows { rows: n, minimum: 3 * d });
    }
    let all: Vec<usize> = (0..n).collect();
    let full = fit_mixture(data, &all, None)?;
    let mut rng = CausalRng::from_seed(seed);
    let mut draws = Vec::with_capacity(draw_count * n);
    for _ in 0..draw_count {
        let resample: Vec<usize> =
            (0..n).map(|_| ((rng.next_f64() * n as f64) as usize).min(n - 1)).collect();
        let fit = fit_mixture(data, &resample, Some((full.beta.as_slice(), full.sigma2)))?;
        for i in 0..n {
            let mut mu = [0.0, 0.0];
            for (slot, x) in [0.0, 1.0].into_iter().enumerate() {
                mu[slot] = fit.beta[0]
                    + fit.beta[1] * x
                    + (0..data.covariate_count)
                        .map(|j| fit.beta[2 + j] * data.covariates[j * n + i])
                        .sum::<f64>();
            }
            let r = responsibility(data.probability[i], data.outcome[i], mu[0], mu[1], fit.sigma2);
            draws.push(if rng.next_f64() < r { 1.0 } else { 0.0 });
        }
    }
    Ok(SoftExposureDraws {
        draws,
        draw_count,
        coefficients: full.beta,
        residual_variance: full.sigma2,
        iterations: full.iterations,
    })
}

/// One bin of the reliability diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct ReliabilityBin {
    /// Inclusive lower edge of the stated probability.
    pub lower: f64,
    /// Upper edge (inclusive for the last bin).
    pub upper: f64,
    /// Units in the bin.
    pub units: usize,
    /// Mean stated probability.
    pub mean_stated: f64,
    /// Mean rate the repeat codings should show if the probabilities are calibrated.
    pub mean_expected: f64,
    /// Mean of the units' repeat-coding shares.
    pub mean_observed: f64,
    /// Observed minus expected.
    pub gap: f64,
    /// Standard error of the gap over units; `None` for a bin of fewer than two units.
    pub standard_error: Option<f64>,
}

/// Binned comparison of stated probabilities with repeat codings.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftExposureReliability {
    /// Non-empty bins, in increasing stated probability.
    pub bins: Vec<ReliabilityBin>,
    /// Units compared.
    pub units: usize,
    /// Largest absolute gap over bins.
    pub max_abs_gap: f64,
    /// Sum of squared standardized gaps over bins with a standard error.
    pub chi_square: f64,
    /// Bins entering `chi_square`.
    pub chi_square_bins: usize,
    /// Unit-weighted mean absolute gap.
    pub mean_abs_gap: f64,
}

/// Compare stated exposure probabilities with repeat codings, in equal-width bins.
///
/// `codings` holds every unit's 0/1 repeat codings back to back and `sizes` the number of
/// codings of each unit. Under calibration and conditionally independent codings, a coding of
/// a unit with stated probability `q` is `1` with probability `Se q + (1 - Sp)(1 - q)`; with
/// no `reference` the codings are taken as exact (`Se = Sp = 1`). The result is a report:
/// no threshold is applied.
///
/// # Errors
/// Mismatched lengths, values outside `{0, 1}`, a unit with no coding, no units or fewer than
/// one bin refuse.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a probability scaled to a bin index below the bin count"
)]
pub fn soft_exposure_reliability(
    probability: &[f64],
    codings: &[u8],
    sizes: &[u32],
    bins: usize,
    reference: Option<&BinaryErrorRates>,
) -> Result<SoftExposureReliability, MeasurementErrorError> {
    let n = probability.len();
    let total: usize = sizes.iter().map(|&k| k as usize).sum();
    if n == 0 || sizes.len() != n || codings.len() != total || bins == 0 || bins > 1000 {
        return Err(MeasurementErrorError::Shape("reliability inputs differ in length"));
    }
    if sizes.contains(&0) {
        return Err(MeasurementErrorError::Shape("every unit needs at least one repeat coding"));
    }
    if codings.iter().any(|&c| c > 1) {
        return Err(MeasurementErrorError::InvalidParameter("repeat codings must be 0 or 1"));
    }
    if !finite_all(probability) || probability.iter().any(|q| !(0.0..=1.0).contains(q)) {
        return Err(MeasurementErrorError::InvalidParameter(
            "stated exposure probabilities must lie in [0, 1]",
        ));
    }
    let (se, sp) = reference.map_or((1.0, 1.0), |r| (r.sensitivity, r.specificity));
    let mut members: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); bins];
    let mut at = 0_usize;
    for (i, &k) in sizes.iter().enumerate() {
        let k = k as usize;
        let share = codings[at..at + k].iter().map(|&c| f64::from(c)).sum::<f64>() / k as f64;
        at += k;
        let q = probability[i];
        let expected = se * q + (1.0 - sp) * (1.0 - q);
        let slot = ((q * bins as f64) as usize).min(bins - 1);
        members[slot].push((q, expected, share));
    }
    let mut out = Vec::new();
    let mut max_abs_gap = 0.0_f64;
    let mut chi_square = 0.0;
    let mut chi_bins = 0_usize;
    let mut weighted_gap = 0.0;
    for (slot, bin) in members.iter().enumerate() {
        if bin.is_empty() {
            continue;
        }
        let units = bin.len();
        let u = units as f64;
        let mean_stated = bin.iter().map(|b| b.0).sum::<f64>() / u;
        let mean_expected = bin.iter().map(|b| b.1).sum::<f64>() / u;
        let mean_observed = bin.iter().map(|b| b.2).sum::<f64>() / u;
        let gap = mean_observed - mean_expected;
        let standard_error = (units >= 2).then(|| {
            let diffs: Vec<f64> = bin.iter().map(|b| b.2 - b.1).collect();
            let mean = diffs.iter().sum::<f64>() / u;
            let var = diffs.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (u - 1.0);
            (var / u).sqrt()
        });
        if let Some(se_gap) = standard_error
            && se_gap > 0.0
        {
            chi_square += (gap / se_gap) * (gap / se_gap);
            chi_bins += 1;
        }
        max_abs_gap = max_abs_gap.max(gap.abs());
        weighted_gap += gap.abs() * u;
        out.push(ReliabilityBin {
            lower: slot as f64 / bins as f64,
            upper: (slot + 1) as f64 / bins as f64,
            units,
            mean_stated,
            mean_expected,
            mean_observed,
            gap,
            standard_error,
        });
    }
    Ok(SoftExposureReliability {
        bins: out,
        units: n,
        max_abs_gap,
        chi_square,
        chi_square_bins: chi_bins,
        mean_abs_gap: weighted_gap / n as f64,
    })
}
