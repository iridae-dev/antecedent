//! Statistical primitives for explicitly modeled outcome-observation mechanisms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp)]

use antecedent_kernels::{norm_cdf, norm_sf};

use crate::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, StatsError, fit_glm,
};

/// Fitted logistic probabilities for an outcome-observation indicator.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationProbabilityFit {
    /// Logistic coefficients, including the leading intercept.
    pub coefficients: Vec<f64>,
    /// Fitted observation probabilities, clipped to the requested floor.
    pub probabilities: Vec<f64>,
    /// Probability floor used for positivity protection.
    pub probability_floor: f64,
}

impl ObservationProbabilityFit {
    /// Observation probability for one covariate row, clipped to this fit's floor.
    ///
    /// `covariates` excludes the intercept and is ordered as at fit time. This exists so a
    /// caller can evaluate the model on rows it was not fit on, which is what cross-fitting
    /// requires; [`Self::probabilities`] are in-sample by construction and must not be
    /// reused for held-out rows.
    ///
    /// # Errors
    ///
    /// The row length does not match the fitted coefficients.
    pub fn probability_at(&self, covariates: &[f64]) -> Result<f64, StatsError> {
        if covariates.len() + 1 != self.coefficients.len() {
            return Err(StatsError::Shape {
                message: "observation probability row does not match the fitted coefficients",
            });
        }
        let eta = self.coefficients[0]
            + self.coefficients[1..]
                .iter()
                .zip(covariates)
                .map(|(beta, value)| beta * value)
                .sum::<f64>();
        Ok(GlmFamily::BinomialLogit
            .mean_from_eta(eta)
            .clamp(self.probability_floor, 1.0 - self.probability_floor))
    }
}

/// Fit `P(R=1 | X)` by logistic regression.
///
/// `covariates_colmajor` excludes the intercept. Indicators must be exactly zero or one.
///
/// # Errors
///
/// Invalid shapes/values, an invalid probability floor, or a failed logistic fit.
pub fn fit_observation_logistic(
    indicator: &[f64],
    covariates_colmajor: &[f64],
    ncols: usize,
    probability_floor: f64,
) -> Result<ObservationProbabilityFit, StatsError> {
    let n = indicator.len();
    if n < 3 || covariates_colmajor.len() != n * ncols {
        return Err(StatsError::Shape { message: "observation logistic design shape mismatch" });
    }
    if !probability_floor.is_finite() || !(0.0..0.5).contains(&probability_floor) {
        return Err(StatsError::Unsupported {
            message: "observation probability floor must lie in (0, 0.5)",
        });
    }
    if indicator.iter().any(|&r| r != 0.0 && r != 1.0) {
        return Err(StatsError::Unsupported {
            message: "observation indicators must be exactly zero or one",
        });
    }
    if !indicator.contains(&0.0) || !indicator.contains(&1.0) {
        return Err(StatsError::Unsupported {
            message: "observation model requires observed and unobserved rows",
        });
    }
    let mut design = vec![1.0; n * (ncols + 1)];
    design[n..].copy_from_slice(covariates_colmajor);
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &design, nrows: n, ncols: ncols + 1, y: indicator },
        &FaerBackend,
        &mut workspace,
        &GlmOptions::default(),
    )?;
    fit.require_ok()?;
    let probabilities = (0..n)
        .map(|row| {
            let eta = fit
                .coefficients
                .iter()
                .enumerate()
                .map(|(col, beta)| beta * design[col * n + row])
                .sum::<f64>();
            GlmFamily::BinomialLogit
                .mean_from_eta(eta)
                .clamp(probability_floor, 1.0 - probability_floor)
        })
        .collect();
    Ok(ObservationProbabilityFit {
        coefficients: fit.coefficients,
        probabilities,
        probability_floor,
    })
}

/// Construct IPW or augmented-IPW pseudo-outcomes for a selected outcome.
///
/// Without `outcome_regression`, returns `R Y / p`. With it, returns
/// `m(X) + R (Y-m(X))/p`. Values of `observed` on rows with `R=0` are ignored.
///
/// # Errors
///
/// Shape mismatch, non-binary indicators, invalid probabilities, missing observed values,
/// or non-finite outcome-regression predictions.
pub fn selected_outcome_pseudo_values(
    observed: &[f64],
    indicator: &[f64],
    probabilities: &[f64],
    outcome_regression: Option<&[f64]>,
) -> Result<Vec<f64>, StatsError> {
    let n = observed.len();
    if indicator.len() != n
        || probabilities.len() != n
        || outcome_regression.is_some_and(|m| m.len() != n)
    {
        return Err(StatsError::Shape { message: "selected-outcome inputs must align" });
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = indicator[i];
        let p = probabilities[i];
        if (r != 0.0 && r != 1.0) || !p.is_finite() || !(0.0..=1.0).contains(&p) || p == 0.0 {
            return Err(StatsError::Unsupported {
                message: "selected-outcome indicators/probabilities are invalid",
            });
        }
        let m = outcome_regression.map_or(0.0, |values| values[i]);
        if !m.is_finite() || (r == 1.0 && !observed[i].is_finite()) {
            return Err(StatsError::Unsupported {
                message: "selected outcomes or outcome predictions are non-finite",
            });
        }
        out.push(if r == 1.0 { m + (observed[i] - m) / p } else { m });
    }
    Ok(out)
}

/// Kaplan–Meier inverse censoring weights, and whether the censoring survival's observed
/// support covers the whole follow-up.
#[derive(Clone, Debug, PartialEq)]
pub struct KaplanMeierIpcw {
    /// Zero for censored rows; `G(entry_i−) / G(time_i−)` for observed rows.
    pub weights: Vec<f64>,
    /// The censoring survival's first drop to (or below) `survival_floor`, if any.
    ///
    /// `None` means the estimated censoring survival stays at or above the floor through
    /// every recorded time, so nothing in the data indicates the tail is unidentified.
    /// `Some(tau)` means every unit still at risk at `tau` was censored there (an
    /// administrative-censoring boundary, or the numerical equivalent of one): no row's
    /// event can be observed at or beyond `tau`, so `Y ≥ tau` has probability that is not
    /// identified from these data. The unweighted mean of the Horvitz–Thompson pseudo-
    /// outcome `event · Y / G` then estimates the restricted mean `E[Y · 1{Y < tau}]`, not
    /// `E[Y]`; the caller must label the result accordingly rather than pass it off as the
    /// unrestricted mean.
    pub tail_restriction: Option<f64>,
}

/// Kaplan–Meier inverse censoring weights evaluated just before each observed time.
///
/// `event=1` means the scientific event/outcome is uncensored; `event=0` is a censoring
/// event. Optional `entry` implements delayed entry by using risk sets
/// `entry_i <= t <= time_i` and the left-truncated IPCW
/// `G(entry_i−) / G(time_i−)` (reducing to `1 / G(time_i−)` when there is no entry).
/// Censored rows receive zero weight.
///
/// The censoring survival function's own support is only ever as wide as the data: it
/// cannot rule out outcomes beyond the point where every unit still at risk was censored.
/// [`KaplanMeierIpcw::tail_restriction`] reports that boundary instead of pretending
/// the corrected mean covers the unbounded outcome.
///
/// # Errors
///
/// Invalid inputs, empty risk sets, or censoring survival below `survival_floor` at an
/// observed event.
pub fn kaplan_meier_ipcw(
    time: &[f64],
    event: &[f64],
    entry: Option<&[f64]>,
    survival_floor: f64,
) -> Result<KaplanMeierIpcw, StatsError> {
    let n = time.len();
    if n == 0 || event.len() != n || entry.is_some_and(|v| v.len() != n) {
        return Err(StatsError::Shape {
            message: "Kaplan-Meier inputs must align and be nonempty",
        });
    }
    if !survival_floor.is_finite() || !(0.0..1.0).contains(&survival_floor) {
        return Err(StatsError::Unsupported {
            message: "censoring-survival floor must lie in (0, 1)",
        });
    }
    for i in 0..n {
        let start = entry.map_or(f64::NEG_INFINITY, |v| v[i]);
        if !time[i].is_finite()
            || entry.is_some_and(|_| !start.is_finite())
            || start > time[i]
            || (event[i] != 0.0 && event[i] != 1.0)
        {
            return Err(StatsError::Unsupported {
                message: "invalid censoring time, event, or entry",
            });
        }
    }
    // Risk sets from one sort of the times (and entries): a row is at risk at `t` when
    // `entry ≤ t ≤ time`, and every row with `entry > t` also has `time > t`, so
    // `risk(t) = #{time ≥ t} − #{entry > t}`. `+ 0.0` maps `−0.0` to `0.0` so `total_cmp`
    // ordering and `==` agree on ties.
    let sorted = |values: &[f64]| -> Vec<f64> {
        let mut v: Vec<f64> = values.iter().map(|&x| x + 0.0).collect();
        v.sort_by(f64::total_cmp);
        v
    };
    let times_sorted = sorted(time);
    let entries_sorted = entry.map(sorted);
    let censor_times = {
        let censored: Vec<f64> =
            (0..n).filter(|&i| event[i] == 0.0).map(|i| time[i] + 0.0).collect();
        sorted(&censored)
    };
    let mut survival = 1.0;
    let mut steps: Vec<(f64, f64)> = Vec::new();
    let mut idx = 0usize;
    while idx < censor_times.len() {
        let t = censor_times[idx];
        let mut end = idx + 1;
        while end < censor_times.len() && censor_times[end] == t {
            end += 1;
        }
        let censored = end - idx;
        idx = end;
        let time_at_least = n - times_sorted.partition_point(|&x| x < t);
        let not_yet_entered =
            entries_sorted.as_ref().map_or(0, |e| n - e.partition_point(|&x| x <= t));
        let risk = time_at_least.saturating_sub(not_yet_entered);
        if risk == 0 || censored > risk {
            return Err(StatsError::Unsupported {
                message: "invalid delayed-entry censoring risk set",
            });
        }
        survival *= 1.0 - censored as f64 / risk as f64;
        steps.push((t, survival));
    }
    // Left-limit of the censoring survival: product-limit value after every jump strictly
    // before `at`. Equals 1 when no censoring precedes `at`.
    let survival_before = |at: f64| -> f64 {
        let jumps_before = steps.partition_point(|(t, _)| *t < at);
        if jumps_before == 0 { 1.0 } else { steps[jumps_before - 1].1 }
    };
    // The first jump time at which the whole remaining risk set was censored (survival
    // reaches, or is driven below, the floor): an administrative-censoring boundary past
    // which no event can ever be observed. Checked here, over every jump, rather than only
    // at rows with `event=1`: a data set can have no event past this boundary at all, in
    // which case the per-row floor check below never fires even though the tail is exactly
    // as unidentified.
    let tail_restriction =
        steps.iter().find(|(_, survival)| *survival < survival_floor).map(|&(t, _)| t);
    let weights = (0..n)
        .map(|i| {
            if event[i] == 0.0 {
                return Ok(0.0);
            }
            let at_event = survival_before(time[i]);
            // Conditional on delayed entry at L, P(C > T | C > L) = G(T−)/G(L−), so the
            // IPCW is G(L−)/G(T−). Without entry, G(L−)=1 and this collapses to 1/G(T−).
            let at_entry = entry.map_or(1.0, |values| survival_before(values[i]));
            if at_event < survival_floor || at_entry < survival_floor {
                return Err(StatsError::Unsupported {
                    message: "censoring survival is below the configured positivity floor",
                });
            }
            if at_event <= 0.0 {
                return Err(StatsError::Unsupported {
                    message: "censoring survival at the event time is non-positive",
                });
            }
            Ok(at_entry / at_event)
        })
        .collect::<Result<Vec<f64>, StatsError>>()?;
    Ok(KaplanMeierIpcw { weights, tail_restriction })
}

/// One Gaussian observation-likelihood contribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussianObservation {
    /// Exact observation.
    Exact(f64),
    /// Latent value is at most the bound.
    LeftCensored(f64),
    /// Latent value is at least the bound.
    RightCensored(f64),
    /// Latent value lies in `[lower, upper]`.
    IntervalCensored {
        /// Lower endpoint.
        lower: f64,
        /// Upper endpoint.
        upper: f64,
    },
    /// Exact observation conditional on lying inside optional sampling bounds.
    Truncated {
        /// Recorded value.
        value: f64,
        /// Lower sampling bound, or negative infinity when absent.
        lower: f64,
        /// Upper sampling bound, or positive infinity when absent.
        upper: f64,
    },
}

/// Gaussian log likelihood for exact, censored, interval-censored, or truncated values.
///
/// # Errors
///
/// Invalid scale, shape, bounds, values, or numerically zero interval/truncation mass.
pub fn gaussian_observation_log_likelihood(
    observations: &[GaussianObservation],
    means: &[f64],
    sigma: f64,
) -> Result<f64, StatsError> {
    if observations.len() != means.len() || observations.is_empty() {
        return Err(StatsError::Shape {
            message: "Gaussian observation likelihood shape mismatch",
        });
    }
    if !sigma.is_finite() || sigma <= 0.0 || means.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Unsupported {
            message: "Gaussian means/scale must be finite and scale positive",
        });
    }
    let log_sigma = sigma.ln();
    observations.iter().zip(means).try_fold(0.0, |sum, (observation, &mu)| {
        let log_term = match *observation {
            GaussianObservation::Exact(y) => {
                require_finite(y)?;
                log_norm_pdf((y - mu) / sigma) - log_sigma
            }
            GaussianObservation::LeftCensored(bound) => {
                require_finite(bound)?;
                log_norm_sf(-(bound - mu) / sigma)
            }
            GaussianObservation::RightCensored(bound) => {
                require_finite(bound)?;
                log_norm_sf((bound - mu) / sigma)
            }
            GaussianObservation::IntervalCensored { lower, upper } => {
                require_ordered(lower, upper)?;
                log_interval_mass((lower - mu) / sigma, (upper - mu) / sigma)?
            }
            GaussianObservation::Truncated { value, lower, upper } => {
                require_finite(value)?;
                require_ordered(lower, upper)?;
                if value < lower || value > upper {
                    return Err(StatsError::Unsupported {
                        message: "truncated Gaussian observation lies outside sampling bounds",
                    });
                }
                let log_mass = log_interval_mass((lower - mu) / sigma, (upper - mu) / sigma)?;
                log_norm_pdf((value - mu) / sigma) - log_sigma - log_mass
            }
        };
        if log_term.is_finite() {
            Ok(sum + log_term)
        } else {
            Err(StatsError::Unsupported {
                message: "Gaussian observation likelihood has zero probability",
            })
        }
    })
}

fn require_finite(value: f64) -> Result<(), StatsError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(StatsError::Unsupported { message: "Gaussian observation endpoint must be finite" })
    }
}

fn require_ordered(lower: f64, upper: f64) -> Result<(), StatsError> {
    if lower < upper && !lower.is_nan() && !upper.is_nan() {
        Ok(())
    } else {
        Err(StatsError::Unsupported { message: "Gaussian observation bounds must be ordered" })
    }
}

/// Standardized value from which `ln Φ̄(z)` switches from `ln(norm_sf(z))` (which underflows
/// to `−∞` near `z ≈ 38.6`) to its asymptotic series.
const LOG_SF_ASYMPTOTIC_FROM: f64 = 30.0;

/// `ln φ(z)`, computed analytically so it stays finite (`−z²/2 − ½ ln 2π`) where `φ(z)`
/// underflows.
fn log_norm_pdf(z: f64) -> f64 {
    -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// `ln Φ̄(z)` (log survival function) without underflow in the upper tail.
///
/// Beyond [`LOG_SF_ASYMPTOTIC_FROM`] it uses the Mills-ratio series
/// `Φ̄(z) = φ(z)/z · (1 − 1/z² + 3/z⁴ − 15/z⁶ + 105/z⁸ − 945/z¹⁰ + …)`, whose truncation
/// error there is below `2e-14`.
fn log_norm_sf(z: f64) -> f64 {
    if z < LOG_SF_ASYMPTOTIC_FROM {
        return norm_sf(z).ln();
    }
    if z == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    let inv2 = 1.0 / (z * z);
    let series = 1.0
        - inv2 * (1.0 - 3.0 * inv2 * (1.0 - 5.0 * inv2 * (1.0 - 7.0 * inv2 * (1.0 - 9.0 * inv2))));
    log_norm_pdf(z) - z.ln() + series.ln()
}

/// `ln(1 − eᵈ)` for `d ≤ 0`.
fn ln_one_minus_exp(d: f64) -> f64 {
    if d > -std::f64::consts::LN_2 { (-d.exp_m1()).ln() } else { (-d.exp()).ln_1p() }
}

/// `ln(Φ(upper) − Φ(lower))` for standardized bounds, computed in the tail nearer to the
/// interval so neither the mass nor the difference of near-1 CDF values cancels or
/// underflows.
fn log_interval_mass(lower: f64, upper: f64) -> Result<f64, StatsError> {
    let log_mass = if lower > 0.0 {
        // Both bounds in the upper tail: Φ̄(l) − Φ̄(u).
        let log_l = log_norm_sf(lower);
        log_l + ln_one_minus_exp(log_norm_sf(upper) - log_l)
    } else if upper < 0.0 {
        // Mirror image in the lower tail: Φ(u) − Φ(l), with Φ(z) = Φ̄(−z).
        let log_u = log_norm_sf(-upper);
        log_u + ln_one_minus_exp(log_norm_sf(-lower) - log_u)
    } else {
        // The interval contains 0, so its mass is at least the mass of a unit-scale sliver
        // around it minus nothing that cancels catastrophically.
        (norm_cdf(upper) - norm_cdf(lower)).ln()
    };
    if log_mass.is_finite() {
        Ok(log_mass)
    } else {
        Err(StatsError::Unsupported {
            message: "Gaussian observation interval has zero probability",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_kernels::norm_pdf;

    #[test]
    fn aipw_collapses_to_predictions_on_unobserved_rows() {
        let got = selected_outcome_pseudo_values(
            &[2.0, f64::NAN, 4.0],
            &[1.0, 0.0, 1.0],
            &[0.5, 0.5, 1.0],
            Some(&[1.0, 3.0, 4.0]),
        )
        .unwrap();
        assert_eq!(got, vec![3.0, 3.0, 4.0]);
    }

    #[test]
    fn delayed_entry_changes_censoring_risk_set() {
        let without =
            kaplan_meier_ipcw(&[1.0, 2.0, 3.0], &[0.0, 1.0, 1.0], None, 0.01).unwrap().weights;
        let with =
            kaplan_meier_ipcw(&[1.0, 2.0, 3.0], &[0.0, 1.0, 1.0], Some(&[0.0, 1.5, 0.0]), 0.01)
                .unwrap()
                .weights;
        assert!((without[1] - 1.5).abs() < 1e-12);
        // Left-truncated IPCW is G(L−)/G(T−). Unit 1 enters after the only censoring jump, so
        // G(L−)=G(T−)=1/2 and the weight is 1 — not the untruncated 1/G(T−)=2.
        assert!((with[1] - 1.0).abs() < 1e-12);
        assert!((with[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn tail_restriction_is_none_when_censoring_survival_never_reaches_the_floor() {
        // Two events, one censoring jump that leaves half the risk set alive: the
        // censoring survival never drops below the floor, so there is no evidence in the
        // data that the tail is unidentified.
        let fit =
            kaplan_meier_ipcw(&[1.0, 2.0, 3.0, 4.0], &[0.0, 1.0, 0.0, 1.0], None, 0.1).unwrap();
        assert_eq!(fit.tail_restriction, None);
    }

    #[test]
    fn tail_restriction_flags_an_administrative_censoring_boundary() {
        // Exp(1) censored at tau=1: every subject alive at tau=1 is censored there, driving
        // the censoring survival to exactly zero. No event can ever be observed at or past
        // tau=1, so the tail beyond it is unidentified from these data.
        let fit =
            kaplan_meier_ipcw(&[0.2, 0.5, 1.0, 1.0, 1.0], &[1.0, 1.0, 0.0, 0.0, 0.0], None, 0.01)
                .unwrap();
        assert_eq!(fit.tail_restriction, Some(1.0));
        // Events strictly before the boundary still have full weight: the censoring
        // survival at their time is 1 (no censoring has occurred yet).
        assert!((fit.weights[0] - 1.0).abs() < 1e-12);
        assert!((fit.weights[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_censoring_and_truncation_terms_match_closed_form() {
        let observations = [
            GaussianObservation::LeftCensored(0.0),
            GaussianObservation::RightCensored(0.0),
            GaussianObservation::IntervalCensored { lower: -1.0, upper: 1.0 },
            GaussianObservation::Truncated { value: 0.0, lower: -1.0, upper: 1.0 },
        ];
        let got = gaussian_observation_log_likelihood(&observations, &[0.0; 4], 1.0).unwrap();
        let mass = norm_cdf(1.0) - norm_cdf(-1.0);
        let expected = 0.5_f64.ln() * 2.0 + mass.ln() + norm_pdf(0.0).ln() - mass.ln();
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn gaussian_observation_likelihood_survives_the_far_upper_tail() {
        // ln φ(50) = −1250 − ½ ln 2π; φ(50) underflows to 0, so `ln(pdf)` was −∞ and the
        // likelihood errored instead of returning a very negative number.
        let ll =
            gaussian_observation_log_likelihood(&[GaussianObservation::Exact(50.0)], &[0.0], 1.0)
                .unwrap();
        let expected = -1250.0 - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((ll - expected).abs() < 1e-9, "{ll} vs {expected}");
        // With sigma = 2 the log-scale term enters: −z²/2 − ½ ln 2π − ln 2, z = 25.
        let ll =
            gaussian_observation_log_likelihood(&[GaussianObservation::Exact(50.0)], &[0.0], 2.0)
                .unwrap();
        let expected = -312.5 - 0.5 * (2.0 * std::f64::consts::PI).ln() - 2.0_f64.ln();
        assert!((ll - expected).abs() < 1e-9, "{ll} vs {expected}");

        // Interval [8.3, 9] sigma above the mean: Φ(9) and Φ(8.3) both round to 1, so the
        // difference was exactly 0 ("zero probability"). Reference ln(Φ(9) − Φ(8.3)) from a
        // 50-digit evaluation: −37.496387817470645.
        let ll = gaussian_observation_log_likelihood(
            &[GaussianObservation::IntervalCensored { lower: 8.3, upper: 9.0 }],
            &[0.0],
            1.0,
        )
        .unwrap();
        assert!((ll - (-37.496_387_817_470_645)).abs() < 1e-9, "{ll}");
        // Mirror image below the mean.
        let ll = gaussian_observation_log_likelihood(
            &[GaussianObservation::IntervalCensored { lower: -9.0, upper: -8.3 }],
            &[0.0],
            1.0,
        )
        .unwrap();
        assert!((ll - (-37.496_387_817_470_645)).abs() < 1e-9, "{ll}");

        // Right-censoring at 45 sigma: Φ̄(45) underflows; ln Φ̄(45) = −1017.2260942419524
        // from a 50-digit evaluation.
        let ll = gaussian_observation_log_likelihood(
            &[GaussianObservation::RightCensored(45.0)],
            &[0.0],
            1.0,
        )
        .unwrap();
        assert!((ll - (-1017.226_094_241_952_4)).abs() < 1e-8, "{ll}");
        // Left-censoring at −45 sigma is the mirror image.
        let ll = gaussian_observation_log_likelihood(
            &[GaussianObservation::LeftCensored(-45.0)],
            &[0.0],
            1.0,
        )
        .unwrap();
        assert!((ll - (-1017.226_094_241_952_4)).abs() < 1e-8, "{ll}");
    }

    /// The original O(n · distinct censor times) implementation, kept as the reference.
    fn brute_force_km_weights(
        time: &[f64],
        event: &[f64],
        entry: Option<&[f64]>,
        floor: f64,
    ) -> Vec<f64> {
        let n = time.len();
        let mut censor: Vec<f64> = (0..n).filter(|&i| event[i] == 0.0).map(|i| time[i]).collect();
        censor.sort_by(f64::total_cmp);
        censor.dedup();
        let mut survival = 1.0;
        let mut steps: Vec<(f64, f64)> = Vec::new();
        for &t in &censor {
            let risk = (0..n).filter(|&i| entry.is_none_or(|v| v[i] <= t) && time[i] >= t).count();
            let cens = (0..n).filter(|&i| event[i] == 0.0 && time[i] == t).count();
            survival *= 1.0 - cens as f64 / risk as f64;
            steps.push((t, survival));
        }
        let before =
            |at: f64| steps.iter().take_while(|(t, _)| *t < at).last().map_or(1.0, |(_, s)| *s);
        (0..n)
            .map(|i| {
                if event[i] == 0.0 {
                    return 0.0;
                }
                let g_t = before(time[i]);
                assert!(g_t >= floor);
                entry.map_or(1.0, |v| before(v[i])) / g_t
            })
            .collect()
    }

    #[test]
    fn sorted_sweep_kaplan_meier_matches_the_brute_force_definition() {
        // Deterministic pseudo-random data with heavy ties in time and entry.
        let n = 300usize;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 33) as f64 / f64::from(1u32 << 31)
        };
        let mut time = Vec::with_capacity(n);
        let mut event = Vec::with_capacity(n);
        let mut entry = Vec::with_capacity(n);
        for _ in 0..n {
            let e = (next() * 8.0).floor();
            let t = e + (next() * 12.0).floor();
            time.push(t);
            entry.push(e);
            // Roughly 30% censored.
            event.push(if next() < 0.3 { 0.0 } else { 1.0 });
        }
        for entry in [None, Some(entry.as_slice())] {
            let got = kaplan_meier_ipcw(&time, &event, entry, 1e-6).unwrap().weights;
            let want = brute_force_km_weights(&time, &event, entry, 1e-6);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-12 * w.abs().max(1.0), "row {i}: {g} vs {w}");
            }
        }
        // Signed zeros are one tie, not two censoring jumps.
        let signed = kaplan_meier_ipcw(&[-0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], None, 0.01).unwrap();
        let plain = kaplan_meier_ipcw(&[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], None, 0.01);
        assert_eq!(signed.weights.len(), 3);
        assert_eq!(signed.weights, plain.unwrap().weights);
    }

    #[test]
    fn observation_primitives_match_frozen_paper_equation_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/response/observation_primitives/expected.json"
        ))
        .unwrap();
        let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
        let selected = &fixture["selected_outcome"];
        let observed = selected["observed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        let numbers = |field: &serde_json::Value| {
            field
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let pseudo = selected_outcome_pseudo_values(
            &observed,
            &numbers(&selected["indicator"]),
            &numbers(&selected["probabilities"]),
            Some(&numbers(&selected["outcome_regression"])),
        )
        .unwrap();
        let expected_pseudo = numbers(&selected["expected_aipw_pseudo_values"]);
        assert!(
            pseudo
                .iter()
                .zip(expected_pseudo)
                .all(|(got, expected)| (*got - expected).abs() <= atol)
        );

        let km = &fixture["kaplan_meier_ipcw"];
        let time = numbers(&km["time"]);
        let event = numbers(&km["event"]);
        let entry = numbers(&km["entry"]);
        let floor = km["survival_floor"].as_f64().unwrap();
        let without = kaplan_meier_ipcw(&time, &event, None, floor).unwrap().weights;
        let with = kaplan_meier_ipcw(&time, &event, Some(&entry), floor).unwrap().weights;
        assert!(
            without
                .iter()
                .zip(numbers(&km["expected_without_entry"]))
                .all(|(got, expected)| (*got - expected).abs() <= atol)
        );
        assert!(
            with.iter()
                .zip(numbers(&km["expected_with_entry"]))
                .all(|(got, expected)| (*got - expected).abs() <= atol)
        );

        let gaussian = &fixture["gaussian_observation_likelihood"];
        let observations = [
            GaussianObservation::LeftCensored(0.0),
            GaussianObservation::RightCensored(0.0),
            GaussianObservation::IntervalCensored { lower: -1.0, upper: 1.0 },
            GaussianObservation::Truncated { value: 0.0, lower: -1.0, upper: 1.0 },
        ];
        let got = gaussian_observation_log_likelihood(
            &observations,
            &numbers(&gaussian["means"]),
            gaussian["sigma"].as_f64().unwrap(),
        )
        .unwrap();
        assert!((got - gaussian["expected_log_likelihood"].as_f64().unwrap()).abs() <= atol);
    }
}
