//! Adaptively weighted augmented inverse-probability estimation.
//!
//! For unit `i` enrolled after history `H_{i-1}`, the assignment policy gave
//! arm `a` probability `e_i(a)` (a function of `H_{i-1}` and of randomization
//! only) and the unit received arm `A_i` with outcome `Y_i`. With an outcome
//! model `m_i(a)` that is a function of `H_{i-1}` alone (a *predictable*
//! model: here a prior-shrunk running mean of earlier outcomes of that arm),
//! the score
//!
//! ```text
//! Gamma_i(a) = m_i(a) + 1[A_i = a] (Y_i - m_i(a)) / e_i(a)
//! ```
//!
//! satisfies `E[Gamma_i(a) | H_{i-1}] = mu_a` however the policy adapts, so
//! `Gamma_i(a) - mu_a` is a martingale difference sequence. Any weights
//! `h_i` that are functions of `H_{i-1}` keep `sum h_i (Gamma_i - mu)` a
//! martingale; the self-normalized mean
//!
//! ```text
//! mu_hat = sum h_i Gamma_i / sum h_i,
//! V_hat  = sum h_i^2 (Gamma_i - mu_hat)^2 / (sum h_i)^2
//! ```
//!
//! is the adaptively weighted estimator of Hadad, Hirshberg, Zhan, Wager and
//! Athey (2021, PNAS 118(15)), whose martingale central limit theorem makes
//! `V_hat` the plug-in variance when the weights stabilize the conditional
//! variance. Uniform weights are plain augmented inverse-probability
//! weighting; [`WeightScheme::VarianceStabilizing`] uses
//! `h_i = (sum_{a in estimand} 1 / e_i(a))^(-1/2)`, which for one arm is the
//! paper's square-root weight `sqrt(e_i(a))`. For a two-arm contrast the
//! paper treats arm values; the contrast weight above is this crate's choice
//! of a predictable weight, and its central limit theorem is not claimed.
//!
//! The variance is not a coverage statement: it is reported uncalibrated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::AdaptiveInferenceError as Error;

/// Most units one adaptive analysis accepts.
pub const MAX_ADAPTIVE_UNITS: usize = 5_000_000;
/// Most arms one weighted analysis accepts.
pub const MAX_WEIGHTED_ARMS: usize = 64;

const PROBABILITY_ROW_TOLERANCE: f64 = 1e-8;
const CANCEL_STRIDE: usize = 4096;

/// How per-unit scores are weighted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WeightScheme {
    /// Every unit has weight one (augmented inverse-probability weighting).
    Uniform,
    /// Weight `(sum over the estimand's arms of 1 / e_i(arm))^(-1/2)`, the
    /// inverse conditional standard deviation of the score.
    VarianceStabilizing,
}

impl WeightScheme {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::VarianceStabilizing => "variance_stabilizing",
        }
    }
}

/// Prior of the predictable outcome model: the running mean of an arm is
/// shrunk to `mean` with `strength` pseudo-observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutcomeModelPrior {
    /// Prior mean of an arm's outcome.
    pub mean: f64,
    /// Pseudo-observations behind the prior mean; positive.
    pub strength: f64,
}

/// The quantity estimated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimand {
    /// The mean outcome under one arm.
    ArmMean(usize),
    /// The difference of two arm means, `plus - minus`.
    Contrast {
        /// Arm entering with a plus sign.
        plus: usize,
        /// Arm entering with a minus sign.
        minus: usize,
    },
}

/// An adaptively weighted estimate and its plug-in variance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveEstimate {
    /// The weighted mean of the scores.
    pub estimate: f64,
    /// Plug-in variance of the estimate (uncalibrated).
    pub variance: f64,
    /// Units that contributed.
    pub units: usize,
    /// Sum of the weights.
    pub weight_sum: f64,
    /// `(sum h)^2 / sum h^2`, the effective number of units.
    pub effective_units: f64,
    /// Largest single weight as a share of the weight sum.
    pub max_weight_share: f64,
}

fn check_history(arms: usize, assigned: &[usize], probabilities: &[f64]) -> Result<usize, Error> {
    if !(2..=MAX_WEIGHTED_ARMS).contains(&arms) {
        return Err(Error::Shape("an adaptive analysis needs between 2 and 64 arms"));
    }
    let n = assigned.len();
    if n < 2 {
        return Err(Error::Shape("an adaptive analysis needs at least two units"));
    }
    if n > MAX_ADAPTIVE_UNITS {
        return Err(Error::Limit("more than five million units"));
    }
    if probabilities.len() != n * arms {
        return Err(Error::Shape("assignment probabilities must hold one row of arms per unit"));
    }
    for (unit, row) in probabilities.chunks_exact(arms).enumerate() {
        if row.iter().any(|p| !(p.is_finite() && (0.0..=1.0).contains(p))) {
            return Err(Error::InvalidParameter("an assignment probability is outside [0, 1]"));
        }
        if (row.iter().sum::<f64>() - 1.0).abs() > PROBABILITY_ROW_TOLERANCE {
            return Err(Error::InvalidParameter("assignment probabilities do not sum to one"));
        }
        let arm = assigned[unit];
        if arm >= arms {
            return Err(Error::Shape("an assigned arm is outside the declared arms"));
        }
        if row[arm] <= 0.0 {
            return Err(Error::InvalidParameter(
                "a unit was assigned an arm the policy gave probability zero",
            ));
        }
    }
    Ok(n)
}

/// Per-unit augmented inverse-probability scores, row-major `n x arms`.
///
/// # Errors
/// A malformed history (shapes, probabilities that are not distributions, an
/// assigned arm of probability zero, a non-finite outcome), an invalid prior,
/// or cancellation.
pub fn adaptive_scores(
    arms: usize,
    assigned: &[usize],
    probabilities: &[f64],
    outcomes: &[f64],
    prior: OutcomeModelPrior,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<f64>, Error> {
    let n = check_history(arms, assigned, probabilities)?;
    if outcomes.len() != n {
        return Err(Error::Shape("outcomes must have one value per unit"));
    }
    if outcomes.iter().any(|y| !y.is_finite()) {
        return Err(Error::InvalidParameter("an outcome is not finite"));
    }
    if !(prior.mean.is_finite() && prior.strength.is_finite() && prior.strength > 0.0) {
        return Err(Error::InvalidParameter(
            "the outcome model prior needs a finite mean and a positive strength",
        ));
    }
    let mut sum = vec![0.0; arms];
    let mut count = vec![0.0_f64; arms];
    let mut scores = vec![0.0; n * arms];
    for (unit, (&arm, &y)) in assigned.iter().zip(outcomes).enumerate() {
        if unit % CANCEL_STRIDE == 0 && cancelled() {
            return Err(Error::Cancelled);
        }
        let row = unit * arms;
        for (a, (total, seen)) in sum.iter().zip(&count).enumerate() {
            let model = (prior.strength * prior.mean + total) / (prior.strength + seen);
            let correction = if a == arm { (y - model) / probabilities[row + a] } else { 0.0 };
            scores[row + a] = model + correction;
        }
        sum[arm] += y;
        count[arm] += 1.0;
    }
    Ok(scores)
}

/// The adaptively weighted estimate of `estimand` from scores of
/// [`adaptive_scores`].
///
/// `unit_multiplier` scales each unit's weight (cluster or subject size when
/// the estimand averages over measurements rather than over units); it must be
/// finite and positive. `probability_floor` is the smallest assignment
/// probability an estimand arm may have had at any unit.
///
/// # Errors
/// Shapes that disagree, an arm outside the declared arms, a floor outside
/// `(0, 1]`, an estimand arm whose probability was below the floor at some
/// unit ([`Error::Positivity`]), or a non-positive multiplier.
#[allow(
    clippy::needless_range_loop,
    reason = "the unit index addresses several parallel columns by computed offsets"
)]
pub fn weighted_estimate(
    arms: usize,
    probabilities: &[f64],
    scores: &[f64],
    estimand: Estimand,
    scheme: WeightScheme,
    unit_multiplier: Option<&[f64]>,
    probability_floor: f64,
) -> Result<AdaptiveEstimate, Error> {
    if !(2..=MAX_WEIGHTED_ARMS).contains(&arms) {
        return Err(Error::Shape("an adaptive analysis needs between 2 and 64 arms"));
    }
    if scores.len() % arms != 0 || probabilities.len() != scores.len() {
        return Err(Error::Shape("scores and probabilities must be n x arms"));
    }
    let n = scores.len() / arms;
    if n < 2 {
        return Err(Error::Shape("an adaptive analysis needs at least two units"));
    }
    if !(probability_floor.is_finite() && probability_floor > 0.0 && probability_floor <= 1.0) {
        return Err(Error::InvalidParameter("the probability floor must lie in (0, 1]"));
    }
    if unit_multiplier
        .is_some_and(|m| m.len() != n || m.iter().any(|c| !(c.is_finite() && *c > 0.0)))
    {
        return Err(Error::InvalidParameter(
            "a unit multiplier must be finite and positive, one per unit",
        ));
    }
    let members: Vec<usize> = match estimand {
        Estimand::ArmMean(arm) => vec![arm],
        Estimand::Contrast { plus, minus } => {
            if plus == minus {
                return Err(Error::InvalidParameter("a contrast needs two different arms"));
            }
            vec![plus, minus]
        }
    };
    if members.iter().any(|&arm| arm >= arms) {
        return Err(Error::Shape("an estimand arm is outside the declared arms"));
    }
    let mut weights = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for unit in 0..n {
        let mut inverse_sum = 0.0;
        for &arm in &members {
            let probability = probabilities[unit * arms + arm];
            if probability < probability_floor {
                return Err(Error::Positivity { unit, arm, probability, floor: probability_floor });
            }
            inverse_sum += 1.0 / probability;
        }
        let base = match scheme {
            WeightScheme::Uniform => 1.0,
            WeightScheme::VarianceStabilizing => 1.0 / inverse_sum.sqrt(),
        };
        weights.push(base * unit_multiplier.map_or(1.0, |m| m[unit]));
        values.push(match estimand {
            Estimand::ArmMean(arm) => scores[unit * arms + arm],
            Estimand::Contrast { plus, minus } => {
                scores[unit * arms + plus] - scores[unit * arms + minus]
            }
        });
    }
    let weight_sum: f64 = weights.iter().sum();
    let estimate = weights.iter().zip(&values).map(|(h, v)| h * v).sum::<f64>() / weight_sum;
    let variance =
        weights.iter().zip(&values).map(|(h, v)| (h * (v - estimate)).powi(2)).sum::<f64>()
            / (weight_sum * weight_sum);
    let square_sum: f64 = weights.iter().map(|h| h * h).sum();
    let largest = weights.iter().copied().fold(0.0_f64, f64::max);
    Ok(AdaptiveEstimate {
        estimate,
        variance,
        units: n,
        weight_sum,
        effective_units: weight_sum * weight_sum / square_sum,
        max_weight_share: largest / weight_sum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unit_assigned_an_arm_of_probability_zero_is_refused() {
        let outcome = adaptive_scores(
            2,
            &[0, 1],
            &[1.0, 0.0, 1.0, 0.0],
            &[0.0, 1.0],
            OutcomeModelPrior { mean: 0.5, strength: 1.0 },
            &|| false,
        );
        assert!(matches!(outcome, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn a_floor_failure_names_the_unit_and_arm() {
        let probabilities = [0.5, 0.5, 0.9, 0.1];
        let scores = [0.0, 0.0, 0.0, 0.0];
        let outcome = weighted_estimate(
            2,
            &probabilities,
            &scores,
            Estimand::ArmMean(1),
            WeightScheme::Uniform,
            None,
            0.2,
        );
        assert!(matches!(outcome, Err(Error::Positivity { unit: 1, arm: 1, .. })));
    }
}
