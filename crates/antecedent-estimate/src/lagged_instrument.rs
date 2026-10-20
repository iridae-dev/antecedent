//! Lagged-instrument contrast for one ordered series.
//!
//! At each time `t` the instrument is the series value at `t - lag`. That lag has to predict
//! treatment. It is excluded from the outcome equation: the outcome contributes only through its
//! reduced form on the lag, and the reported number is the effect of treatment. It is not a
//! coefficient on the lag, and it is not temporal linear adjustment.
//!
//! [`fit_lagged_instrument`] is the just-identified Wald ratio and a homoskedastic standard
//! error. [`fit_lagged_instrument_posterior`] is a different object: draws from a Gaussian
//! posterior of the reduced-form ratio `γ / π`, with an independent normal prior on the
//! reduced-form coefficients and a plug-in residual covariance. A penalty on the Wald fit is
//! not that posterior. The posterior is not the static control-function model.
//!
//! Both treat lag-aligned rows as independent. The standard error and the likelihood do not
//! account for serial dependence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{QuantileRule, column_is_constant, quantile_sorted};

use crate::error::EstimationError;
use crate::util::{DrawSpec, mean};

/// Minimum number of lag-aligned rows. Two reduced-form coefficients need residual degrees of
/// freedom, and the same floor as the static linear IV fit keeps the plug-in variance from
/// resting on two or three points.
const MIN_ALIGNED_ROWS: usize = 8;

/// Numerical floor for a zero residual variance inside the plug-in covariance only. The
/// first-stage F statistic does not use it. A zero residual still reports an infinite F and a
/// zero Wald standard error.
const PLUG_IN_VARIANCE_FLOOR: f64 = 1e-8;

/// Wald model tag. Distinct from the posterior tag.
const WALD_MODEL: &str = concat!(
    "just-identified lagged-instrument Wald contrast; ",
    "the lag is excluded from the outcome equation",
);

/// Posterior model tag. Distinct from the Wald tag and from the static control-function model.
const POSTERIOR_MODEL: &str = concat!(
    "Gaussian reduced-form ratio posterior; ",
    "the lag is excluded from the outcome equation; plug-in residual covariance",
);

/// Equal-length series in time order.
#[derive(Clone, Copy, Debug)]
pub struct LaggedInstrumentSample<'a> {
    /// Instrument series. The value at `t - lag` is the instrument for time `t`.
    pub instrument: &'a [f64],
    /// Treatment series.
    pub treatment: &'a [f64],
    /// Outcome series.
    pub outcome: &'a [f64],
    /// Steps from the instrument to the treatment it predicts. At least 1.
    pub lag: usize,
}

/// Just-identified Wald contrast. The effect is `Cov(Y_t, Z_{t-lag}) / Cov(T_t, Z_{t-lag})`,
/// not an adjustment coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct LaggedInstrumentContrast {
    /// Effect of treatment under exclusion of the lagged instrument.
    pub effect: f64,
    /// Intercept of `Y = α + β T + e` at the Wald `β`.
    pub intercept: f64,
    /// First-stage slope of treatment on the lagged instrument.
    pub first_stage_slope: f64,
    /// First-stage intercept.
    pub first_stage_intercept: f64,
    /// Outcome reduced-form slope on the lagged instrument. Equals `effect * first_stage_slope`
    /// when both slopes are finite.
    pub reduced_form_outcome_slope: f64,
    /// Homoskedastic standard error of [`Self::effect`]. Zero when the structural residual is
    /// identically zero. Not a serial-dependence standard error.
    pub homoskedastic_se: f64,
    /// First-stage F statistic of the lagged instrument. Infinite when that stage fits perfectly.
    pub first_stage_f: f64,
    /// Rows that remain after dropping the first `lag` times.
    pub aligned_rows: usize,
    /// Lag used to build the instrument.
    pub lag: usize,
    /// Model tag.
    pub model: &'static str,
}

/// Posterior draws of the reduced-form ratio. Not a penalized Wald estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct LaggedInstrumentPosterior {
    /// Draws of `γ / π`, sorted ascending.
    pub effect_draws: Vec<f64>,
    /// Mean of [`Self::effect_draws`].
    pub mean: f64,
    /// Equal-tailed 0.025 quantile of the draws.
    pub lower: f64,
    /// Equal-tailed 0.975 quantile of the draws.
    pub upper: f64,
    /// OLS first-stage slope of treatment on the lagged instrument. This is the relevance
    /// diagnostic shared with the Wald contrast, not a posterior mean of `π`.
    pub first_stage_slope: f64,
    /// OLS first-stage F of the lagged instrument.
    pub first_stage_f: f64,
    /// Prior standard deviation of each reduced-form coefficient.
    pub prior_sd: f64,
    /// Rows that remain after dropping the first `lag` times.
    pub aligned_rows: usize,
    /// Lag used to build the instrument.
    pub lag: usize,
    /// Model tag.
    pub model: &'static str,
}

/// Lag-aligned instrument, treatment, and outcome.
struct AlignedSeries {
    instrument: Vec<f64>,
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    lag: usize,
}

/// Fit the lagged-instrument Wald contrast.
///
/// `weak_f_threshold` refuses a finite first-stage F below that value. A perfect first stage
/// has an infinite F and is not refused. A threshold of zero still refuses a lagged instrument
/// that does not vary or that has zero covariance with treatment.
pub fn fit_lagged_instrument(
    sample: &LaggedInstrumentSample<'_>,
    weak_f_threshold: f64,
) -> Result<LaggedInstrumentContrast, EstimationError> {
    require_threshold(weak_f_threshold)?;
    let series = align(sample)?;
    wald(&series, weak_f_threshold)
}

/// Draw the lagged-instrument reduced-form ratio posterior.
///
/// The reduced forms are `T = α + π Z_{t-lag} + v` and `Y = c + γ Z_{t-lag} + u`. Exclusion
/// puts the treatment effect at the ratio `β = γ / π`; the prior is on `(α, π, c, γ)`, not on
/// `β`. Each coefficient has an independent `N(0, prior_sd²)` prior. The residual covariance
/// is plugged in. A non-finite ratio draw refuses the fit rather than being dropped.
pub fn fit_lagged_instrument_posterior(
    sample: &LaggedInstrumentSample<'_>,
    prior_sd: f64,
    sampling: DrawSpec,
    weak_f_threshold: f64,
) -> Result<LaggedInstrumentPosterior, EstimationError> {
    if !prior_sd.is_finite() || prior_sd <= 0.0 || sampling.draws < 2 {
        return Err(EstimationError::stats_msg(
            "lagged-instrument posterior needs prior_sd > 0 and at least 2 draws",
        ));
    }
    let contrast = fit_lagged_instrument(sample, weak_f_threshold)?;
    let series = align(sample)?;
    ratio_posterior(&series, &contrast, prior_sd, sampling)
}

fn require_threshold(weak_f_threshold: f64) -> Result<(), EstimationError> {
    if !weak_f_threshold.is_finite() || weak_f_threshold < 0.0 {
        return Err(EstimationError::stats_msg(
            "lagged-instrument weak-F threshold must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn align(sample: &LaggedInstrumentSample<'_>) -> Result<AlignedSeries, EstimationError> {
    let LaggedInstrumentSample { instrument, treatment, outcome, lag } = *sample;
    let n = instrument.len();
    if treatment.len() != n || outcome.len() != n {
        return Err(EstimationError::stats_msg(
            "lagged instrument, treatment, and outcome must have the same length",
        ));
    }
    if instrument.iter().chain(treatment).chain(outcome).any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "lagged instrument, treatment, and outcome must be finite",
        ));
    }
    if lag == 0 {
        return Err(EstimationError::stats_msg(
            "lagged-instrument lag must be at least 1; a contemporaneous instrument is not this \
             contrast",
        ));
    }
    if lag >= n || n - lag < MIN_ALIGNED_ROWS {
        return Err(EstimationError::stats_msg(
            "lagged instrument needs at least 8 rows after the lag",
        ));
    }
    let aligned = n - lag;
    let mut z = Vec::with_capacity(aligned);
    let mut t = Vec::with_capacity(aligned);
    let mut y = Vec::with_capacity(aligned);
    for time in lag..n {
        z.push(instrument[time - lag]);
        t.push(treatment[time]);
        y.push(outcome[time]);
    }
    Ok(AlignedSeries { instrument: z, treatment: t, outcome: y, lag })
}

fn wald(
    series: &AlignedSeries,
    weak_f_threshold: f64,
) -> Result<LaggedInstrumentContrast, EstimationError> {
    let z = &series.instrument;
    let t = &series.treatment;
    let y = &series.outcome;
    let n = z.len();
    if column_is_constant(z, n, 0) {
        return Err(EstimationError::stats_msg("lagged instrument has no variation"));
    }
    let z_mean = mean(z);
    let t_mean = mean(t);
    let y_mean = mean(y);
    let mut z_ss = 0.0;
    let mut zt = 0.0;
    let mut zy = 0.0;
    for ((&zi, &ti), &yi) in z.iter().zip(t).zip(y) {
        let zc = zi - z_mean;
        z_ss += zc * zc;
        zt += zc * (ti - t_mean);
        zy += zc * (yi - y_mean);
    }
    if !z_ss.is_finite() || z_ss <= 0.0 || !zt.is_finite() || zt.abs() <= 0.0 {
        return Err(EstimationError::stats_msg("lagged instrument does not predict treatment"));
    }
    let first_stage_slope = zt / z_ss;
    let first_stage_intercept = t_mean - first_stage_slope * z_mean;
    let reduced_form_outcome_slope = zy / z_ss;
    let effect = zy / zt;
    let intercept = y_mean - effect * t_mean;
    if !effect.is_finite() || !first_stage_slope.is_finite() {
        return Err(EstimationError::stats_msg("lagged instrument does not predict treatment"));
    }
    let df = (n - 2) as f64;
    let mut treatment_rss = 0.0;
    let mut structural_rss = 0.0;
    for ((&zi, &ti), &yi) in z.iter().zip(t).zip(y) {
        let first_stage_residual = ti - first_stage_intercept - first_stage_slope * zi;
        treatment_rss += first_stage_residual * first_stage_residual;
        let structural = yi - intercept - effect * ti;
        structural_rss += structural * structural;
    }
    let first_stage_f = if treatment_rss <= 0.0 {
        f64::INFINITY
    } else {
        first_stage_slope * first_stage_slope * z_ss / (treatment_rss / df)
    };
    // A NaN F is not a pass: `NaN < threshold` is false.
    if first_stage_f.is_nan() || first_stage_f < weak_f_threshold {
        return Err(EstimationError::stats_msg(format!(
            "weak_instrument: lagged-instrument first-stage F={first_stage_f:.3} is below \
             threshold {weak_f_threshold:.3}"
        )));
    }
    // Homoskedastic SE of the just-identified slope with an intercept:
    // σ² · Σ z̃² / (Σ z̃ t̃)², with σ² = Σ(y − α − β t)² / (n − 2).
    let homoskedastic_se = if structural_rss <= 0.0 {
        0.0
    } else {
        (structural_rss / df * z_ss / (zt * zt)).sqrt()
    };
    Ok(LaggedInstrumentContrast {
        effect,
        intercept,
        first_stage_slope,
        first_stage_intercept,
        reduced_form_outcome_slope,
        homoskedastic_se,
        first_stage_f,
        aligned_rows: n,
        lag: series.lag,
        model: WALD_MODEL,
    })
}

fn ratio_posterior(
    series: &AlignedSeries,
    contrast: &LaggedInstrumentContrast,
    prior_sd: f64,
    sampling: DrawSpec,
) -> Result<LaggedInstrumentPosterior, EstimationError> {
    let (wtt, wyy, wty) = residual_weight(series, contrast)?;
    let mut precision = [[0.0_f64; 4]; 4];
    let mut rhs = [0.0_f64; 4];
    for ((&z, &t), &y) in series.instrument.iter().zip(&series.treatment).zip(&series.outcome) {
        let treatment_row = [1.0, z, 0.0, 0.0];
        let outcome_row = [0.0, 0.0, 1.0, z];
        for (i, (&treatment_coef, &outcome_coef)) in
            treatment_row.iter().zip(outcome_row.iter()).enumerate()
        {
            rhs[i] += wtt * treatment_coef * t
                + wyy * outcome_coef * y
                + wty * (treatment_coef * y + outcome_coef * t);
            for (j, (&treatment_coef_j, &outcome_coef_j)) in
                treatment_row.iter().zip(outcome_row.iter()).enumerate()
            {
                precision[i][j] += wtt * treatment_coef * treatment_coef_j
                    + wyy * outcome_coef * outcome_coef_j
                    + wty * (treatment_coef * outcome_coef_j + outcome_coef * treatment_coef_j);
            }
        }
    }
    let prior_precision = 1.0 / (prior_sd * prior_sd);
    for (i, row) in precision.iter_mut().enumerate() {
        row[i] += prior_precision;
    }
    let covariance = crate::util::invert_fixed(&precision).ok_or_else(|| {
        EstimationError::stats_msg("lagged-instrument reduced-form posterior is singular")
    })?;
    let posterior_mean: [f64; 4] =
        std::array::from_fn(|i| (0..4).map(|j| covariance[i][j] * rhs[j]).sum());
    if posterior_mean.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "lagged-instrument reduced-form posterior is singular",
        ));
    }
    let flat: Vec<f64> = covariance.iter().flatten().copied().collect();
    let samples = antecedent_prob::sample_gaussian_mvn(
        &posterior_mean,
        &flat,
        sampling.draws,
        sampling.seed,
        &mut antecedent_prob::LaplaceWorkspace::default(),
    )
    .map_err(|_| {
        EstimationError::stats_msg(
            "lagged-instrument posterior covariance is not positive definite",
        )
    })?;
    let mut effect_draws = Vec::with_capacity(sampling.draws);
    for draw in 0..sampling.draws {
        let pi = samples[sampling.draws + draw];
        let gamma = samples[3 * sampling.draws + draw];
        let ratio = gamma / pi;
        if !ratio.is_finite() {
            return Err(EstimationError::stats_msg(
                "lagged-instrument ratio posterior is not finite",
            ));
        }
        effect_draws.push(ratio);
    }
    effect_draws.sort_by(f64::total_cmp);
    let mean_effect = mean(&effect_draws);
    Ok(LaggedInstrumentPosterior {
        lower: quantile(&effect_draws, 0.025),
        upper: quantile(&effect_draws, 0.975),
        mean: mean_effect,
        effect_draws,
        first_stage_slope: contrast.first_stage_slope,
        first_stage_f: contrast.first_stage_f,
        prior_sd,
        aligned_rows: contrast.aligned_rows,
        lag: contrast.lag,
        model: POSTERIOR_MODEL,
    })
}

/// Inverse of the plug-in reduced-form residual covariance, as `(W_tt, W_yy, W_ty)`.
fn residual_weight(
    series: &AlignedSeries,
    contrast: &LaggedInstrumentContrast,
) -> Result<(f64, f64, f64), EstimationError> {
    let z_mean = mean(&series.instrument);
    let y_mean = mean(&series.outcome);
    let gamma = contrast.reduced_form_outcome_slope;
    let outcome_intercept = y_mean - gamma * z_mean;
    let mut treatment_rss = 0.0;
    let mut outcome_rss = 0.0;
    let mut cross = 0.0;
    for ((&z, &t), &y) in series.instrument.iter().zip(&series.treatment).zip(&series.outcome) {
        let treatment_residual =
            t - contrast.first_stage_intercept - contrast.first_stage_slope * z;
        let outcome_residual = y - outcome_intercept - gamma * z;
        treatment_rss += treatment_residual * treatment_residual;
        outcome_rss += outcome_residual * outcome_residual;
        cross += treatment_residual * outcome_residual;
    }
    let df = (series.instrument.len() - 2) as f64;
    let treatment_variance = (treatment_rss / df).max(PLUG_IN_VARIANCE_FLOOR);
    let outcome_variance = (outcome_rss / df).max(PLUG_IN_VARIANCE_FLOOR);
    let mut covariance = cross / df;
    let mut det = treatment_variance * outcome_variance - covariance * covariance;
    if !det.is_finite() || det <= 0.0 {
        covariance = 0.0;
        det = treatment_variance * outcome_variance;
    }
    if !det.is_finite() || det <= 0.0 {
        return Err(EstimationError::stats_msg(
            "lagged-instrument reduced-form residual covariance is singular",
        ));
    }
    Ok((outcome_variance / det, treatment_variance / det, -covariance / det))
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    quantile_sorted(sorted, p, QuantileRule::ExchangeableRank)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "fixtures compare exact Wald constants, an infinite first-stage F, and a copied fit"
)]
mod tests {
    use super::*;

    /// Prefix a lag-1 series so index `t` of the returned treatment is predicted by index `t - 1`
    /// of the instrument. The unused treatment and outcome at time 0 are zero. The unused final
    /// instrument value is zero.
    fn lag1(z: &[f64], treatment: &[f64], outcome: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        assert_eq!(z.len(), treatment.len());
        assert_eq!(z.len(), outcome.len());
        let mut instrument = Vec::with_capacity(z.len() + 1);
        instrument.extend_from_slice(z);
        instrument.push(0.0);
        let mut treatment_series = Vec::with_capacity(z.len() + 1);
        treatment_series.push(0.0);
        treatment_series.extend_from_slice(treatment);
        let mut outcome_series = Vec::with_capacity(z.len() + 1);
        outcome_series.push(0.0);
        outcome_series.extend_from_slice(outcome);
        (instrument, treatment_series, outcome_series)
    }

    fn binary_instrument(n: usize) -> Vec<f64> {
        (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect()
    }

    /// `+1, +1, -1, -1`, orthogonal to [`binary_instrument`] when `n` is a multiple of 4.
    fn paired_signs(n: usize) -> Vec<f64> {
        (0..n).map(|i| if i % 4 < 2 { 1.0 } else { -1.0 }).collect()
    }

    fn sample<'a>(
        instrument: &'a [f64],
        treatment: &'a [f64],
        outcome: &'a [f64],
        lag: usize,
    ) -> LaggedInstrumentSample<'a> {
        LaggedInstrumentSample { instrument, treatment, outcome, lag }
    }

    #[test]
    fn wald_contrast_matches_the_hand_derived_confounded_series() {
        // Eight aligned rows. Z alternates 0, 1. U is +1,+1,-1,-1 and is orthogonal to Z.
        // T = Z + U, Y = 2 T + 3 U. Exclusion holds, so the Wald ratio is 2.
        // The regression Y = 5 T − 3 Z fits with residual zero, so the adjustment coefficient
        // on treatment is 5. Σ z̃² = 2, Σ z̃ t̃ = 2, structural rss = 72, df = 6,
        // homoskedastic variance = 12 · 2 / 2² = 6. First-stage rss = 8, so F = 1.5.
        let z = binary_instrument(8);
        let u = paired_signs(8);
        let treatment: Vec<f64> = z.iter().zip(&u).map(|(zi, ui)| zi + ui).collect();
        let outcome: Vec<f64> =
            treatment.iter().zip(&u).map(|(ti, ui)| 2.0 * ti + 3.0 * ui).collect();
        let adjustment_rss: f64 = outcome
            .iter()
            .zip(&treatment)
            .zip(&z)
            .map(|((yi, ti), zi)| {
                let residual = yi - 5.0 * ti + 3.0 * zi;
                residual * residual
            })
            .sum();
        assert!(adjustment_rss < 1e-9, "adjustment rss {adjustment_rss}");
        let (instrument, treatment_series, outcome_series) = lag1(&z, &treatment, &outcome);
        let fit = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            1.0,
        )
        .unwrap();
        assert!((fit.effect - 2.0).abs() < 1e-9, "effect {effect}", effect = fit.effect);
        assert!((fit.effect - 5.0).abs() > 1.0);
        assert!((fit.first_stage_slope - 1.0).abs() < 1e-9);
        assert!(fit.first_stage_intercept.abs() < 1e-9);
        assert!(fit.intercept.abs() < 1e-9);
        assert!((fit.reduced_form_outcome_slope - fit.effect * fit.first_stage_slope).abs() < 1e-9);
        assert!(
            (fit.homoskedastic_se.powi(2) - 6.0).abs() < 1e-9,
            "se {se}",
            se = fit.homoskedastic_se
        );
        assert!((fit.first_stage_f - 1.5).abs() < 1e-9, "F {f}", f = fit.first_stage_f);
        assert_eq!(fit.aligned_rows, 8);
        assert_eq!(fit.lag, 1);
        assert!(fit.model.contains("Wald"));
    }

    #[test]
    fn direct_lag_effect_stays_in_the_wald_ratio() {
        // Y = 2 T + 4 Z on the same orthogonal design. Adjustment recovers 2. The Wald ratio
        // does not partial the lag out, so it equals 6.
        let z = binary_instrument(8);
        let u = paired_signs(8);
        let treatment: Vec<f64> = z.iter().zip(&u).map(|(zi, ui)| zi + ui).collect();
        let outcome: Vec<f64> =
            treatment.iter().zip(&z).map(|(ti, zi)| 2.0 * ti + 4.0 * zi).collect();
        let adjustment_rss: f64 = outcome
            .iter()
            .zip(&treatment)
            .zip(&z)
            .map(|((yi, ti), zi)| {
                let residual = yi - 2.0 * ti - 4.0 * zi;
                residual * residual
            })
            .sum();
        assert!(adjustment_rss < 1e-9, "adjustment rss {adjustment_rss}");
        let (instrument, treatment_series, outcome_series) = lag1(&z, &treatment, &outcome);
        let fit = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            1.0,
        )
        .unwrap();
        assert!((fit.effect - 6.0).abs() < 1e-9, "effect {effect}", effect = fit.effect);
        assert!((fit.first_stage_slope - 1.0).abs() < 1e-9);
    }

    #[test]
    fn first_stage_uses_the_lagged_instrument() {
        // T_t = 1 + 2 Z_{t-1} and Y_t = 4 + 3 T_t. The contemporaneous pairing of this series
        // has first-stage slope −2. The lag pairing has slope 2 and effect 3.
        let z = binary_instrument(8);
        let treatment: Vec<f64> = z.iter().map(|zi| 1.0 + 2.0 * zi).collect();
        let outcome: Vec<f64> = treatment.iter().map(|ti| 4.0 + 3.0 * ti).collect();
        let (mut instrument, treatment_series, outcome_series) = lag1(&z, &treatment, &outcome);
        let fit = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            10.0,
        )
        .unwrap();
        assert!((fit.effect - 3.0).abs() < 1e-9, "effect {effect}", effect = fit.effect);
        assert!(
            (fit.first_stage_slope - 2.0).abs() < 1e-9,
            "slope {slope}",
            slope = fit.first_stage_slope
        );
        assert!(fit.first_stage_f.is_infinite());
        assert_eq!(fit.homoskedastic_se, 0.0);
        let last = instrument.len() - 1;
        instrument[last] = 99.0;
        let untouched_tail = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            10.0,
        )
        .unwrap();
        assert_eq!(untouched_tail, fit);
        instrument[last] = 0.0;
        instrument[0] = 1.0;
        let changed_lag = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            1.0,
        )
        .unwrap();
        assert!((changed_lag.first_stage_slope - 2.0).abs() > 0.1);
    }

    fn noiseless_series() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let z = binary_instrument(8);
        let treatment: Vec<f64> = z.iter().map(|zi| 1.0 + 2.0 * zi).collect();
        let outcome: Vec<f64> = treatment.iter().map(|ti| 4.0 + 3.0 * ti).collect();
        lag1(&z, &treatment, &outcome)
    }

    #[test]
    fn refuses_a_lag_that_is_not_an_instrument() {
        let (instrument, treatment_series, outcome_series) = noiseless_series();
        let contemporaneous = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 0),
            10.0,
        )
        .unwrap_err();
        assert!(contemporaneous.to_string().contains("at least 1"));
        let short = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 2),
            10.0,
        )
        .unwrap_err();
        assert!(short.to_string().contains("at least 8"));
        let mismatch = fit_lagged_instrument(
            &sample(&instrument[..instrument.len() - 1], &treatment_series, &outcome_series, 1),
            10.0,
        )
        .unwrap_err();
        assert!(mismatch.to_string().contains("same length"));
        let mut with_nan = instrument.clone();
        with_nan[1] = f64::NAN;
        let non_finite =
            fit_lagged_instrument(&sample(&with_nan, &treatment_series, &outcome_series, 1), 10.0)
                .unwrap_err();
        assert!(non_finite.to_string().contains("finite"));
        let constant = vec![1.0; instrument.len()];
        let flat =
            fit_lagged_instrument(&sample(&constant, &treatment_series, &outcome_series, 1), 10.0)
                .unwrap_err();
        assert!(flat.to_string().contains("no variation"));
        let bad_threshold = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            f64::NAN,
        )
        .unwrap_err();
        assert!(bad_threshold.to_string().contains("threshold"));
    }

    #[test]
    fn refuses_a_lag_that_does_not_predict_treatment() {
        let z = binary_instrument(8);
        let orthogonal = paired_signs(8);
        let (instrument, treatment_series, outcome_series) = lag1(&z, &orthogonal, &orthogonal);
        let none = fit_lagged_instrument(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            0.0,
        )
        .unwrap_err();
        assert!(none.to_string().contains("does not predict"));
        let weak_treatment: Vec<f64> =
            z.iter().zip(&orthogonal).map(|(zi, ui)| 0.01 * zi + ui).collect();
        let (weak_instrument, weak_treatment_series, weak_outcome_series) =
            lag1(&z, &weak_treatment, &weak_treatment);
        let weak = fit_lagged_instrument(
            &sample(&weak_instrument, &weak_treatment_series, &weak_outcome_series, 1),
            10.0,
        )
        .unwrap_err();
        assert!(weak.to_string().contains("weak_instrument"), "{weak}");
    }

    #[test]
    fn diffuse_posterior_concentrates_on_the_structural_effect() {
        let z = binary_instrument(8);
        let treatment: Vec<f64> = z.iter().map(|zi| 1.0 + 2.0 * zi).collect();
        let outcome: Vec<f64> = treatment.iter().map(|ti| 4.0 + 3.0 * ti).collect();
        let (instrument, treatment_series, outcome_series) = lag1(&z, &treatment, &outcome);
        let sample = sample(&instrument, &treatment_series, &outcome_series, 1);
        let wald = fit_lagged_instrument(&sample, 10.0).unwrap();
        let posterior =
            fit_lagged_instrument_posterior(&sample, 10.0, DrawSpec::new(400, 5), 10.0).unwrap();
        assert!((wald.effect - 3.0).abs() < 1e-9);
        assert!(
            (posterior.mean - 3.0).abs() < 1e-3,
            "posterior mean {mean}",
            mean = posterior.mean
        );
        assert!((posterior.lower - 3.0).abs() < 1e-2, "lower {lower}", lower = posterior.lower);
        assert!((posterior.upper - 3.0).abs() < 1e-2, "upper {upper}", upper = posterior.upper);
        assert!((posterior.first_stage_slope - wald.first_stage_slope).abs() < 1e-12);
        assert_eq!(posterior.effect_draws.len(), 400);
        assert!(posterior.model.contains("posterior"));
        assert_ne!(posterior.model, wald.model);
    }

    #[test]
    fn tight_prior_moves_the_ratio_away_from_the_wald_contrast() {
        // Forty aligned rows. T = Z, so the first stage is perfect. Y = 2 Z + e with
        // e = +1,+1,-1,-1 orthogonal to Z, so the Wald effect is 2.
        // Plug-in outcome variance is 40/38. A N(0, 0.05²) prior on the raw coefficients has
        // posterior precision [[438, 19], [19, 419]] and outcome-slope mean 15922/183161.
        // The perfect first stage keeps π at 1, so the ratio mean sits on that slope. The Wald
        // contrast has no prior and stays at 2.
        let n = 40;
        let z = binary_instrument(n);
        let e = paired_signs(n);
        let treatment = z.clone();
        let outcome: Vec<f64> = z.iter().zip(&e).map(|(zi, ei)| 2.0 * zi + ei).collect();
        let (instrument, treatment_series, outcome_series) = lag1(&z, &treatment, &outcome);
        let sample = sample(&instrument, &treatment_series, &outcome_series, 1);
        let wald = fit_lagged_instrument(&sample, 10.0).unwrap();
        assert!((wald.effect - 2.0).abs() < 1e-9, "wald {effect}", effect = wald.effect);
        assert!(wald.first_stage_f.is_infinite());
        let sampling = DrawSpec::new(2000, 19);
        let diffuse =
            fit_lagged_instrument_posterior(&sample, 100.0, sampling, 10.0).unwrap();
        let tight = fit_lagged_instrument_posterior(&sample, 0.05, sampling, 10.0).unwrap();
        let shrunk_slope = 15_922.0 / 183_161.0;
        assert!((diffuse.mean - 2.0).abs() < 0.05, "diffuse {mean}", mean = diffuse.mean);
        assert!(
            (tight.mean - shrunk_slope).abs() < 0.02,
            "tight {mean} target {shrunk_slope}",
            mean = tight.mean
        );
        assert!((tight.mean - wald.effect).abs() > 1.0);
        assert!((tight.mean - diffuse.mean).abs() > 1.0);
        assert_eq!(tight.first_stage_slope, wald.first_stage_slope);
        assert_eq!(tight.prior_sd, 0.05);
        assert_eq!(diffuse.prior_sd, 100.0);
    }

    #[test]
    fn posterior_refuses_a_weak_lagged_instrument() {
        let z = binary_instrument(8);
        let noise = paired_signs(8);
        let treatment: Vec<f64> = z.iter().zip(&noise).map(|(zi, ui)| 0.01 * zi + ui).collect();
        let (instrument, treatment_series, outcome_series) = lag1(&z, &treatment, &treatment);
        let weak = fit_lagged_instrument_posterior(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            10.0,
            DrawSpec::new(20, 3),
            10.0,
        )
        .unwrap_err();
        assert!(weak.to_string().contains("weak_instrument"), "{weak}");
    }

    #[test]
    fn posterior_refuses_a_nonpositive_prior() {
        let (instrument, treatment_series, outcome_series) = noiseless_series();
        let bad_prior = fit_lagged_instrument_posterior(
            &sample(&instrument, &treatment_series, &outcome_series, 1),
            0.0,
            DrawSpec::new(20, 3),
            10.0,
        )
        .unwrap_err();
        assert!(bad_prior.to_string().contains("prior_sd"));
    }
}
