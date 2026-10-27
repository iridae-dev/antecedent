//! Responder-weighted intention-to-treat contrast for randomized outcomes that are missing at
//! random given the arm and declared baseline covariates.
//!
//! Within each arm, the probability of observing the outcome is modelled by a logistic
//! regression on the declared covariates, each responder is weighted by the inverse of its
//! fitted probability, and the arm mean is the Hájek ratio of the weighted outcome total to
//! the weight total. The denominator is the whole arm: an assigned participant whose outcome is
//! missing keeps its place in the arm's score and variance, so the randomized assignment is
//! never narrowed to the responders.
//!
//! The variance is the sum over arms of the sample variance of the arm's estimated influence
//! function divided by the arm size. The influence function includes the score of the fitted
//! observation model, so estimating the observation probabilities is accounted for rather
//! than treated as known. No interval is published from it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::fit_observation_logistic;

use crate::EstimationError;

/// Observation probabilities below this are refused as a positivity failure.
pub const OBSERVATION_PROBABILITY_FLOOR: f64 = 0.02;

/// One arm of a responder-weighted contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponderWeightedArm {
    /// Assigned participants in the arm, responders and non-responders.
    pub assigned: usize,
    /// Participants whose outcome is observed.
    pub responders: usize,
    /// Hájek weighted outcome mean.
    pub mean: f64,
    /// Estimated variance of the mean, `Var(influence) / assigned`.
    pub variance: f64,
    /// Smallest fitted observation probability among the arm's responders.
    pub min_observation_probability: f64,
    /// Kish effective sample size of the responders' inverse-probability weights.
    pub effective_responders: f64,
}

/// A responder-weighted two-arm contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponderWeightedItt {
    /// Treated mean minus control mean.
    pub effect: f64,
    /// Sum of the arms' estimated mean variances.
    pub variance: f64,
    /// Control arm then treated arm.
    pub arms: [ResponderWeightedArm; 2],
}

/// Estimate the intention-to-treat contrast with responder weighting.
///
/// `outcome[i]` is read only where `observed[i]`; `covariates_colmajor` holds `ncov` complete
/// baseline columns of length `n` (no intercept).
///
/// # Errors
///
/// Misaligned inputs, an arm with no responders, an observation model that is separated or
/// has a responder below the positivity floor, a singular observation-model information
/// matrix, or a non-finite result.
pub fn responder_weighted_itt(
    outcome: &[f64],
    observed: &[bool],
    treated: &[bool],
    covariates_colmajor: &[f64],
    ncov: usize,
) -> Result<ResponderWeightedItt, EstimationError> {
    let n = outcome.len();
    if observed.len() != n
        || treated.len() != n
        || covariates_colmajor.len() != n.saturating_mul(ncov)
    {
        return Err(EstimationError::Unsupported {
            message: "responder weighting inputs must align with the assigned rows",
        });
    }
    if covariates_colmajor.iter().any(|x| !x.is_finite()) {
        return Err(EstimationError::Unsupported {
            message: "responder weighting requires complete finite observation-model covariates",
        });
    }
    let arm = |group: bool| -> Result<ResponderWeightedArm, EstimationError> {
        let rows: Vec<usize> = (0..n).filter(|&i| treated[i] == group).collect();
        let covariates: Vec<f64> = (0..ncov)
            .flat_map(|c| rows.iter().map(move |&i| covariates_colmajor[c * n + i]))
            .collect();
        let y: Vec<f64> = rows.iter().map(|&i| outcome[i]).collect();
        let r: Vec<bool> = rows.iter().map(|&i| observed[i]).collect();
        weighted_arm(&y, &r, &covariates, ncov)
    };
    let control = arm(false)?;
    let active = arm(true)?;
    let effect = active.mean - control.mean;
    let variance = active.variance + control.variance;
    if !effect.is_finite() || !variance.is_finite() || variance < 0.0 {
        return Err(EstimationError::Unsupported {
            message: "responder-weighted effect or variance is not finite",
        });
    }
    Ok(ResponderWeightedItt { effect, variance, arms: [control, active] })
}

/// Hájek mean, influence-function variance and weight diagnostics of one arm.
fn weighted_arm(
    y: &[f64],
    observed: &[bool],
    covariates_colmajor: &[f64],
    ncov: usize,
) -> Result<ResponderWeightedArm, EstimationError> {
    let n = y.len();
    let responders = observed.iter().filter(|r| **r).count();
    if n < 3 || responders == 0 {
        return Err(EstimationError::Unsupported {
            message: "responder weighting requires at least three assigned participants and one responder in each arm",
        });
    }
    if observed[..].iter().zip(y).any(|(r, v)| *r && !v.is_finite()) {
        return Err(EstimationError::Unsupported {
            message: "responder weighting requires finite observed outcomes",
        });
    }
    let probabilities = observation_probabilities(observed, covariates_colmajor, ncov)?;
    let mut weight_total = 0.0;
    let mut weighted_outcome = 0.0;
    let mut weight_square = 0.0;
    let mut min_probability = f64::INFINITY;
    for i in (0..n).filter(|&i| observed[i]) {
        let w = 1.0 / probabilities[i];
        weight_total += w;
        weight_square += w * w;
        weighted_outcome += w * y[i];
        min_probability = min_probability.min(probabilities[i]);
    }
    let mean = weighted_outcome / weight_total;
    let weight_mean = weight_total / n as f64;
    // Estimating the observation probabilities contributes the score term h' I^-1 s_i.
    let score = score_terms(y, observed, &probabilities, covariates_colmajor, ncov, mean)?;
    let mut sum_square = 0.0;
    for i in 0..n {
        let pi = probabilities[i];
        let u = if observed[i] { (y[i] - mean) / pi } else { 0.0 };
        let influence = (u - score[i]) / weight_mean;
        sum_square += influence * influence;
    }
    let variance = sum_square / (n as f64 * (n as f64 - 1.0));
    Ok(ResponderWeightedArm {
        assigned: n,
        responders,
        mean,
        variance,
        min_observation_probability: min_probability,
        effective_responders: weight_total * weight_total / weight_square,
    })
}

/// Per-row estimation term `h' I^{-1} s_i` of an inverse-probability-weighted mean whose
/// observation probabilities come from a logistic fit (stacked M-estimation): `s_i =
/// (R_i - pi_i) x_i` is the observation-model score, `I = mean(pi (1 - pi) x x')` its
/// information and `h = mean(R (Y - centre)(1 - pi) / pi x)` the derivative of the weighted
/// estimating equation in the model coefficients. Subtracting it from a row's weighted
/// contribution gives the influence of the mean with the estimated probabilities, whose
/// variance is the honest one; the term is zero when every row is observed.
fn score_terms(
    y: &[f64],
    observed: &[bool],
    probabilities: &[f64],
    covariates_colmajor: &[f64],
    ncov: usize,
    centre: f64,
) -> Result<Vec<f64>, EstimationError> {
    let n = y.len();
    if observed.iter().all(|seen| *seen) {
        return Ok(vec![0.0; n]);
    }
    let p = ncov + 1;
    let row = |i: usize| -> Vec<f64> {
        std::iter::once(1.0).chain((0..ncov).map(|c| covariates_colmajor[c * n + i])).collect()
    };
    let mut correction = vec![0.0; p];
    let mut information = vec![0.0; p * p];
    for i in 0..n {
        let pi = probabilities[i];
        let x = row(i);
        if observed[i] {
            let scale = (y[i] - centre) * (1.0 - pi) / pi;
            for (slot, xv) in correction.iter_mut().zip(&x) {
                *slot += scale * xv / n as f64;
            }
        }
        for a in 0..p {
            for b in 0..p {
                information[a * p + b] += pi * (1.0 - pi) * x[a] * x[b] / n as f64;
            }
        }
    }
    let adjustment = solve_symmetric(&mut information, &mut correction, p).ok_or(
        EstimationError::Unsupported {
            message: "responder weighting observation model information matrix is singular",
        },
    )?;
    Ok((0..n)
        .map(|i| {
            let x = row(i);
            adjustment.iter().zip(&x).map(|(g, xv)| g * xv).sum::<f64>()
                * (f64::from(u8::from(observed[i])) - probabilities[i])
        })
        .collect())
}

/// Fitted probability of being observed for each row of one group: 1 when every row is
/// observed, otherwise a logistic fit on the declared covariates.
///
/// # Errors
///
/// A separated or too-small observation model, or a responder whose probability is at the
/// positivity floor.
fn observation_probabilities(
    observed: &[bool],
    covariates_colmajor: &[f64],
    ncov: usize,
) -> Result<Vec<f64>, EstimationError> {
    let n = observed.len();
    if observed.iter().all(|seen| *seen) {
        return Ok(vec![1.0; n]);
    }
    let indicator: Vec<f64> = observed.iter().map(|r| f64::from(u8::from(*r))).collect();
    let fit = fit_observation_logistic(
        &indicator,
        covariates_colmajor,
        ncov,
        OBSERVATION_PROBABILITY_FLOOR,
    )
    .map_err(crate::observation::observation_fit_error)?;
    let at_floor = (0..n).any(|i| {
        observed[i] && fit.probabilities[i] <= OBSERVATION_PROBABILITY_FLOOR * (1.0 + 1e-9)
    });
    if at_floor {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("positivity_violated"),
            "responder weighting has an observation probability at the positivity floor",
        ));
    }
    Ok(fit.probabilities)
}

/// Inverse-probability-of-observation pseudo-outcomes `R Y / pi` for designs whose own
/// estimator and variance are then applied to every assigned unit.
///
/// `group[i]` is the unit's assignment group (arm, factorial cell, action); the observation
/// model is fitted within each group on the declared complete covariates. A non-responder's
/// pseudo-outcome is zero, so it keeps its place in the design's denominators and totals.
/// Each row also carries the estimation term of the fitted observation model (see
/// `score_terms`), so the design's variance accounts for estimating `pi`; the group means are
/// unchanged. A non-responder's pseudo-outcome is therefore the negative of that term rather
/// than exactly zero, and the term is zero in a group where everyone responded.
///
/// # Errors
///
/// Misaligned inputs, a group without a responder or with fewer than three units, or the
/// observation-model errors of [`responder_weighted_itt`].
pub fn responder_pseudo_outcomes(
    outcome: &[f64],
    observed: &[bool],
    group: &[usize],
    covariates_colmajor: &[f64],
    ncov: usize,
) -> Result<Vec<f64>, EstimationError> {
    let n = outcome.len();
    if observed.len() != n
        || group.len() != n
        || covariates_colmajor.len() != n.saturating_mul(ncov)
    {
        return Err(EstimationError::Unsupported {
            message: "responder weighting inputs must align with the assigned rows",
        });
    }
    if covariates_colmajor.iter().any(|x| !x.is_finite()) {
        return Err(EstimationError::Unsupported {
            message: "responder weighting requires complete finite observation-model covariates",
        });
    }
    let mut pseudo = vec![0.0; n];
    let mut groups: Vec<usize> = group.to_vec();
    groups.sort_unstable();
    groups.dedup();
    for g in groups {
        let rows: Vec<usize> = (0..n).filter(|&i| group[i] == g).collect();
        if rows.len() < 3 || !rows.iter().any(|&i| observed[i]) {
            return Err(EstimationError::Unsupported {
                message: "responder weighting requires at least three assigned units and one responder in every assignment group",
            });
        }
        let seen: Vec<bool> = rows.iter().map(|&i| observed[i]).collect();
        let covariates: Vec<f64> = (0..ncov)
            .flat_map(|c| rows.iter().map(move |&i| covariates_colmajor[c * n + i]))
            .collect();
        let probabilities = observation_probabilities(&seen, &covariates, ncov)?;
        let values: Vec<f64> =
            rows.iter().map(|&i| if observed[i] { outcome[i] } else { 0.0 }).collect();
        if rows.iter().any(|&i| observed[i] && !outcome[i].is_finite()) {
            return Err(EstimationError::Unsupported {
                message: "responder weighting requires finite observed outcomes",
            });
        }
        // The group mean of the adjusted pseudo-outcomes is the weighted mean exactly (the
        // score terms average to zero), and each row carries the influence of that mean
        // including the estimated observation probabilities.
        let score = score_terms(&values, &seen, &probabilities, &covariates, ncov, 0.0)?;
        for (k, &i) in rows.iter().enumerate() {
            pseudo[i] = if observed[i] { outcome[i] / probabilities[k] } else { 0.0 } - score[k];
        }
    }
    Ok(pseudo)
}

/// Solve `A x = b` for a small dense symmetric positive-definite `A` (row-major, `p x p`) by
/// Gaussian elimination with partial pivoting; `None` when `A` is numerically singular.
fn solve_symmetric(a: &mut [f64], b: &mut [f64], p: usize) -> Option<Vec<f64>> {
    for col in 0..p {
        let pivot =
            (col..p).max_by(|&i, &j| a[i * p + col].abs().total_cmp(&a[j * p + col].abs()))?;
        if a[pivot * p + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..p {
                a.swap(col * p + k, pivot * p + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..p {
            let factor = a[row * p + col] / a[col * p + col];
            for k in col..p {
                a[row * p + k] -= factor * a[col * p + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; p];
    for row in (0..p).rev() {
        let tail: f64 = (row + 1..p).map(|k| a[row * p + k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row * p + row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform stream in (0, 1) (`SplitMix64`).
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u, v) = (self.next().max(1e-300), self.next());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// Complete case is biased when responding depends on a covariate that moves the outcome.
    struct Trial {
        outcome: Vec<f64>,
        observed: Vec<bool>,
        treated: Vec<bool>,
        baseline: Vec<f64>,
    }

    const TRUE_EFFECT: f64 = 0.3;

    fn trial(n: usize, seed: u64) -> Trial {
        let mut rng = Stream(seed);
        let (mut outcome, mut observed, mut treated, mut baseline) =
            (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let arm = i % 2 == 1;
            let x = rng.normal().clamp(-2.0, 2.0);
            let y = 1.0 + TRUE_EFFECT * f64::from(u8::from(arm)) + 2.0 * x + rng.normal();
            // Responding falls with the baseline, more steeply in the treated arm.
            // Baseline X ~ N(0,1) clipped to [-2, 2] (so no fitted response probability can reach
            // the positivity floor), outcome slope 2 and response logit 0.8 - b_a X with b = 0.8
            // (control) and 1.6 (treated). By the probit approximation of the logistic,
            // E[X | responded, arm] = -c/sqrt(1+c^2) * phi(m)/Phi(m), c = b/1.7, m = 0.47/sqrt(1+c^2):
            // -0.233 (control) and -0.406 (treated), so the complete-case effect is biased by
            // 2 * (-0.406 + 0.233) = -0.35, far outside any sampling error at the sizes used here.
            let eta = 0.8 - (0.8 + 0.8 * f64::from(u8::from(arm))) * x;
            let p = crate::util::sigmoid(eta);
            outcome.push(y);
            observed.push(rng.next() < p);
            treated.push(arm);
            baseline.push(x);
        }
        Trial { outcome, observed, treated, baseline }
    }

    fn complete_case_effect(t: &Trial) -> f64 {
        let mean = |group: bool| {
            let v: Vec<f64> = (0..t.outcome.len())
                .filter(|&i| t.treated[i] == group && t.observed[i])
                .map(|i| t.outcome[i])
                .collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        mean(true) - mean(false)
    }

    #[test]
    fn weighting_recovers_the_effect_where_complete_case_does_not() {
        let t = trial(40_000, 7);
        let fit =
            responder_weighted_itt(&t.outcome, &t.observed, &t.treated, &t.baseline, 1).unwrap();
        let se = fit.variance.sqrt();
        assert!(
            (fit.effect - TRUE_EFFECT).abs() < 4.0 * se,
            "weighted {} vs truth {TRUE_EFFECT} (se {se})",
            fit.effect
        );
        let naive = complete_case_effect(&t);
        assert!(
            (naive - TRUE_EFFECT).abs() > 0.2,
            "the fixture must make complete case biased, got {naive}"
        );
        assert_eq!(fit.arms[0].assigned + fit.arms[1].assigned, 40_000);
    }

    #[test]
    fn saturated_binary_covariate_equals_the_poststratified_mean() {
        // Independent computation: with a binary covariate and a logistic model on it, the
        // fitted probability is the cell response rate, so the Hájek arm mean is the
        // cell-size-weighted mean of the cell responder means.
        let mut rng = Stream(11);
        let n = 4_000;
        let (mut y, mut r, mut a, mut x) = (vec![], vec![], vec![], vec![]);
        for i in 0..n {
            let arm = i % 2 == 1;
            let cell = f64::from(u8::from(rng.next() < 0.4));
            let outcome = 2.0 * cell + 0.5 * f64::from(u8::from(arm)) + rng.normal();
            let respond = rng.next() < if cell > 0.5 { 0.9 } else { 0.5 };
            y.push(outcome);
            r.push(respond);
            a.push(arm);
            x.push(cell);
        }
        let fit = responder_weighted_itt(&y, &r, &a, &x, 1).unwrap();
        for (slot, group) in [(0usize, false), (1, true)] {
            let idx: Vec<usize> = (0..n).filter(|&i| a[i] == group).collect();
            let mut expected = 0.0;
            for cell in [0.0, 1.0] {
                let in_cell: Vec<usize> =
                    idx.iter().copied().filter(|&i| (x[i] - cell).abs() < 0.5).collect();
                let responders: Vec<f64> =
                    in_cell.iter().copied().filter(|&i| r[i]).map(|i| y[i]).collect();
                expected += in_cell.len() as f64 / idx.len() as f64
                    * (responders.iter().sum::<f64>() / responders.len() as f64);
            }
            assert!(
                (fit.arms[slot].mean - expected).abs() < 1e-6,
                "arm {slot}: {} vs {expected}",
                fit.arms[slot].mean
            );
        }
    }

    #[test]
    fn without_covariates_the_mean_is_the_responder_mean() {
        // Missing completely at random within arm: pi is the arm response rate, constant.
        let t = trial(2_000, 3);
        let fit = responder_weighted_itt(&t.outcome, &t.observed, &t.treated, &[], 0).unwrap();
        assert!((fit.effect - complete_case_effect(&t)).abs() < 1e-9);
    }

    #[test]
    fn no_missing_values_reduces_to_the_difference_in_means_and_neyman_variance() {
        let t = trial(600, 5);
        let observed = vec![true; 600];
        let fit = responder_weighted_itt(&t.outcome, &observed, &t.treated, &[], 0).unwrap();
        let moments = |group: bool| {
            let v: Vec<f64> =
                (0..600).filter(|&i| t.treated[i] == group).map(|i| t.outcome[i]).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let s2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0);
            (m, s2 / v.len() as f64)
        };
        let (m1, v1) = moments(true);
        let (m0, v0) = moments(false);
        assert!((fit.effect - (m1 - m0)).abs() < 1e-12);
        assert!((fit.variance - (v1 + v0)).abs() < 1e-12);
    }

    #[test]
    fn reported_standard_error_matches_the_sampling_spread() {
        // 300 replications of a 600-participant trial with a covariate-dependent response.
        let (mut effects, mut variances) = (vec![], vec![]);
        for rep in 0..300u64 {
            let t = trial(600, 1_000 + rep);
            let fit = responder_weighted_itt(&t.outcome, &t.observed, &t.treated, &t.baseline, 1)
                .unwrap();
            effects.push(fit.effect);
            variances.push(fit.variance);
        }
        let mean = effects.iter().sum::<f64>() / effects.len() as f64;
        let spread = (effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>()
            / (effects.len() as f64 - 1.0))
            .sqrt();
        let reported = (variances.iter().sum::<f64>() / variances.len() as f64).sqrt();
        assert!(
            (reported / spread - 1.0).abs() < 0.12,
            "reported se {reported} vs empirical sd {spread}"
        );
        assert!((mean - TRUE_EFFECT).abs() < 0.05, "mean effect {mean}");
    }

    /// Difference of group means of `values` and its two-sample variance estimate.
    fn contrast(values: &[f64], group: &[usize]) -> (f64, f64) {
        let moments = |g: usize| {
            let v: Vec<f64> =
                (0..values.len()).filter(|&i| group[i] == g).map(|i| values[i]).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let s2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0);
            (m, s2 / v.len() as f64)
        };
        let (m1, v1) = moments(1);
        let (m0, v0) = moments(0);
        (m1 - m0, v1 + v0)
    }

    #[test]
    fn pseudo_outcome_variance_includes_the_estimation_of_the_response_probabilities() {
        // The group means of the adjusted pseudo-outcomes are the inverse-probability
        // weighted means; the variance of their contrast, computed from the per-unit
        // pseudo-outcomes, must match the sampling spread over repeated trials.
        let (mut effects, mut corrected) = (vec![], vec![]);
        for rep in 0..300u64 {
            let t = trial(600, 5_000 + rep);
            let group: Vec<usize> = t.treated.iter().map(|a| usize::from(*a)).collect();
            let pseudo =
                responder_pseudo_outcomes(&t.outcome, &t.observed, &group, &t.baseline, 1).unwrap();
            let (effect, variance) = contrast(&pseudo, &group);
            effects.push(effect);
            corrected.push(variance);
        }
        let mean = effects.iter().sum::<f64>() / effects.len() as f64;
        let spread = (effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>()
            / (effects.len() as f64 - 1.0))
            .sqrt();
        let reported = (corrected.iter().sum::<f64>() / corrected.len() as f64).sqrt();
        assert!(
            (reported / spread - 1.0).abs() < 0.12,
            "pseudo-outcome se {reported} vs sampling spread {spread}"
        );
        assert!((mean - TRUE_EFFECT).abs() < 0.08, "mean pseudo-outcome effect {mean}");
    }

    #[test]
    fn an_arm_without_responders_is_refused() {
        let t = trial(100, 9);
        let observed: Vec<bool> = (0..100).map(|i| !t.treated[i]).collect();
        assert!(responder_weighted_itt(&t.outcome, &observed, &t.treated, &[], 0).is_err());
    }

    #[test]
    fn misaligned_inputs_are_refused() {
        let t = trial(100, 9);
        assert!(responder_weighted_itt(&t.outcome[..99], &t.observed, &t.treated, &[], 0).is_err());
        assert!(responder_weighted_itt(&t.outcome, &t.observed, &t.treated, &[1.0], 1).is_err());
    }
}
