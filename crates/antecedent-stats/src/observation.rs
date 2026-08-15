//! Statistical primitives for explicitly modeled outcome-observation mechanisms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

use antecedent_kernels::{norm_cdf, norm_pdf, norm_sf};

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

/// Kaplan–Meier inverse censoring weights evaluated just before each observed time.
///
/// `event=1` means the scientific event/outcome is uncensored; `event=0` is a censoring
/// event. Optional `entry` implements delayed entry by using risk sets
/// `entry_i <= t <= time_i` and the left-truncated IPCW
/// `G(entry_i−) / G(time_i−)` (reducing to `1 / G(time_i−)` when there is no entry).
/// Censored rows receive zero weight.
///
/// # Errors
///
/// Invalid inputs, empty risk sets, or censoring survival below `survival_floor`.
pub fn kaplan_meier_ipcw(
    time: &[f64],
    event: &[f64],
    entry: Option<&[f64]>,
    survival_floor: f64,
) -> Result<Vec<f64>, StatsError> {
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
    let mut censor_times: Vec<f64> = (0..n).filter(|&i| event[i] == 0.0).map(|i| time[i]).collect();
    censor_times.sort_by(f64::total_cmp);
    censor_times.dedup_by(|a, b| a.total_cmp(b).is_eq());
    let mut survival = 1.0;
    let mut steps = Vec::with_capacity(censor_times.len());
    for &t in &censor_times {
        let risk = (0..n).filter(|&i| entry.is_none_or(|v| v[i] <= t) && time[i] >= t).count();
        let censored = (0..n).filter(|&i| event[i] == 0.0 && time[i] == t).count();
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
        steps.iter().take_while(|(t, _)| *t < at).last().map_or(1.0, |(_, after)| *after)
    };
    (0..n)
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
        .collect()
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
                norm_pdf((y - mu) / sigma).ln() - log_sigma
            }
            GaussianObservation::LeftCensored(bound) => {
                require_finite(bound)?;
                norm_cdf((bound - mu) / sigma).ln()
            }
            GaussianObservation::RightCensored(bound) => {
                require_finite(bound)?;
                norm_sf((bound - mu) / sigma).ln()
            }
            GaussianObservation::IntervalCensored { lower, upper } => {
                require_ordered(lower, upper)?;
                probability_log(norm_cdf((upper - mu) / sigma) - norm_cdf((lower - mu) / sigma))?
            }
            GaussianObservation::Truncated { value, lower, upper } => {
                require_finite(value)?;
                require_ordered(lower, upper)?;
                if value < lower || value > upper {
                    return Err(StatsError::Unsupported {
                        message: "truncated Gaussian observation lies outside sampling bounds",
                    });
                }
                let mass = norm_cdf((upper - mu) / sigma) - norm_cdf((lower - mu) / sigma);
                norm_pdf((value - mu) / sigma).ln() - log_sigma - probability_log(mass)?
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

fn probability_log(probability: f64) -> Result<f64, StatsError> {
    if probability.is_finite() && probability > 0.0 {
        Ok(probability.ln())
    } else {
        Err(StatsError::Unsupported {
            message: "Gaussian observation interval has zero probability",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let without = kaplan_meier_ipcw(&[1.0, 2.0, 3.0], &[0.0, 1.0, 1.0], None, 0.01).unwrap();
        let with =
            kaplan_meier_ipcw(&[1.0, 2.0, 3.0], &[0.0, 1.0, 1.0], Some(&[0.0, 1.5, 0.0]), 0.01)
                .unwrap();
        assert!((without[1] - 1.5).abs() < 1e-12);
        // Left-truncated IPCW is G(L−)/G(T−). Unit 1 enters after the only censoring jump, so
        // G(L−)=G(T−)=1/2 and the weight is 1 — not the untruncated 1/G(T−)=2.
        assert!((with[1] - 1.0).abs() < 1e-12);
        assert!((with[2] - 2.0).abs() < 1e-12);
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
        let without = kaplan_meier_ipcw(&time, &event, None, floor).unwrap();
        let with = kaplan_meier_ipcw(&time, &event, Some(&entry), floor).unwrap();
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
