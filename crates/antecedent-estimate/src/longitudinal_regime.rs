//! Prespecified binary longitudinal regime value with supplied sequential probabilities.
//!
//! This is a point estimator. The probabilities and subject-level fold ownership
//! are inputs, not nuisance models fitted or verified by this kernel.
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Point summary for one prescribed longitudinal regime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeValueSummary {
    /// Horvitz--Thompson mean over all enrolled subjects.
    pub value: f64,
    /// Effective sample size among observed matching trajectories.
    pub effective_sample_size: f64,
    /// Fraction of subjects observed through the requested trajectory.
    pub matched_observed_fraction: f64,
    /// Largest cumulative treatment and censoring weight.
    pub maximum_weight: f64,
    /// Smallest conditional probability of the prescribed action.
    pub minimum_action_probability: f64,
    /// Smallest conditional probability of remaining uncensored.
    pub minimum_censoring_probability: f64,
}

/// Evaluate supplied conditional period rewards under a prescribed regime.
/// Predictions are subject-major and must be produced without reusing the
/// subject's outcome when excluded-fold ownership is declared upstream.
pub fn evaluate_g_formula_value(
    predictions: &[f64],
    actions: &[bool],
    treatment_probability: &[f64],
    censoring_probability: &[f64],
    subjects: usize,
    periods: usize,
    minimum_probability: f64,
) -> Result<RegimeValueSummary, &'static str> {
    let cells = subjects.checked_mul(periods).ok_or("longitudinal dimensions overflow")?;
    if subjects == 0
        || periods == 0
        || predictions.len() != cells
        || actions.len() != cells
        || treatment_probability.len() != cells
        || censoring_probability.len() != cells
    {
        return Err("g-formula arrays must align by subject and period");
    }
    if !minimum_probability.is_finite()
        || !(0.0 < minimum_probability && minimum_probability <= 0.5)
    {
        return Err("minimum probability must be finite and in (0, 0.5]");
    }
    let mut total = 0.0;
    let mut minimum_action_probability: f64 = 1.0;
    let mut minimum_censoring_probability: f64 = 1.0;
    for j in 0..cells {
        let q = predictions[j];
        let p = treatment_probability[j];
        let c = censoring_probability[j];
        if !q.is_finite() {
            return Err("period outcome predictions must be finite");
        }
        if !p.is_finite() || p < minimum_probability || p > 1.0 - minimum_probability {
            return Err("sequential treatment positivity fails at the declared floor");
        }
        if !c.is_finite() || c < minimum_probability || c > 1.0 {
            return Err("sequential censoring positivity fails at the declared floor");
        }
        minimum_action_probability =
            minimum_action_probability.min(if actions[j] { p } else { 1.0 - p });
        minimum_censoring_probability = minimum_censoring_probability.min(c);
        total += q;
        if !total.is_finite() {
            return Err("summed outcome predictions overflowed");
        }
    }
    Ok(RegimeValueSummary {
        value: total / subjects as f64,
        effective_sample_size: subjects as f64,
        matched_observed_fraction: 1.0,
        maximum_weight: 1.0,
        minimum_action_probability,
        minimum_censoring_probability,
    })
}

/// Evaluate a caller-prescribed static or history-dependent regime on one row
/// per subject. All two-dimensional inputs are flattened in subject-major order.
///
/// `treatment_probability` is the conditional probability of action 1, and
/// `censoring_probability` is the conditional probability of remaining observed
/// at that decision. Values are unconditional Horvitz--Thompson averages, so
/// unmatched or censored subjects contribute zero to the numerator.
pub fn evaluate_regime_value(
    outcomes: &[f64],
    treatment: &[bool],
    actions: &[bool],
    treatment_probability: &[f64],
    outcome_observed: &[bool],
    censoring_probability: &[f64],
    periods: usize,
    minimum_probability: f64,
) -> Result<RegimeValueSummary, &'static str> {
    let n = outcomes.len();
    let cells = n.checked_mul(periods).ok_or("longitudinal dimensions overflow")?;
    if n == 0
        || periods == 0
        || treatment.len() != cells
        || actions.len() != cells
        || treatment_probability.len() != cells
        || censoring_probability.len() != cells
        || outcome_observed.len() != n
    {
        return Err("regime arrays must align by subject and period");
    }
    if !minimum_probability.is_finite()
        || !(0.0 < minimum_probability && minimum_probability <= 0.5)
    {
        return Err("minimum probability must be finite and in (0, 0.5]");
    }
    let mut total = 0.0;
    let mut weight_sum = 0.0;
    let mut weight_squares = 0.0;
    let mut maximum_weight: f64 = 0.0;
    let mut minimum_action_probability: f64 = 1.0;
    let mut minimum_censoring_probability: f64 = 1.0;
    let mut matched = 0usize;
    for i in 0..n {
        if !outcomes[i].is_finite() {
            return Err("longitudinal outcomes must be finite");
        }
        let mut follows = true;
        let mut weight = 1.0;
        for t in 0..periods {
            let j = i * periods + t;
            let p = treatment_probability[j];
            let c = censoring_probability[j];
            if !p.is_finite() || p < minimum_probability || p > 1.0 - minimum_probability {
                return Err("sequential treatment positivity fails at the declared floor");
            }
            if !c.is_finite() || c < minimum_probability || c > 1.0 {
                return Err("sequential censoring positivity fails at the declared floor");
            }
            let action_probability = if actions[j] { p } else { 1.0 - p };
            minimum_action_probability = minimum_action_probability.min(action_probability);
            minimum_censoring_probability = minimum_censoring_probability.min(c);
            follows &= treatment[j] == actions[j];
            if follows {
                weight /= action_probability * c;
                if !weight.is_finite() {
                    return Err("sequential weight overflowed");
                }
            }
        }
        if follows && outcome_observed[i] {
            matched += 1;
            total += weight * outcomes[i];
            weight_sum += weight;
            weight_squares += weight * weight;
            maximum_weight = maximum_weight.max(weight);
        }
    }
    if matched == 0 {
        return Err("no observed subjects followed the requested regime");
    }
    if !total.is_finite() || !weight_squares.is_finite() {
        return Err("longitudinal weighted score overflowed");
    }
    Ok(RegimeValueSummary {
        value: total / n as f64,
        effective_sample_size: weight_sum * weight_sum / weight_squares,
        matched_observed_fraction: matched as f64 / n as f64,
        maximum_weight,
        minimum_action_probability,
        minimum_censoring_probability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_truth_two_period_bernoulli() {
        let summary = evaluate_regime_value(
            &[4.0, 0.0, 0.0, 0.0],
            &[true, true, true, false, false, true, false, false],
            &[true, true, true, true, true, true, true, true],
            &[0.5; 8],
            &[true; 4],
            &[1.0; 8],
            2,
            0.01,
        )
        .unwrap();
        assert_eq!(summary.value, 4.0);
        assert_eq!(summary.matched_observed_fraction, 0.25);
        assert_eq!(summary.effective_sample_size, 1.0);
    }

    #[test]
    fn refuses_missing_support_and_positivity() {
        let mut p = [0.5; 4];
        assert!(evaluate_regime_value(
            &[1.0, 2.0],
            &[false; 4],
            &[true; 4],
            &p,
            &[true; 2],
            &[1.0; 4],
            2,
            0.01
        )
        .unwrap_err()
        .contains("no observed"));
        p[0] = 0.0;
        assert!(evaluate_regime_value(
            &[1.0, 2.0],
            &[false; 4],
            &[true; 4],
            &p,
            &[true; 2],
            &[1.0; 4],
            2,
            0.01
        )
        .unwrap_err()
        .contains("positivity"));
    }
}
