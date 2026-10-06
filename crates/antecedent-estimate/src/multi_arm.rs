//! Randomized multi-arm Horvitz--Thompson means and conservative contrasts.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// An observed randomized arm mean with its observed-arm variance contribution.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmMean {
    /// Horvitz--Thompson mean for the declared arm.
    pub value: f64,
    /// Observed-arm contribution to the design variance bound.
    pub variance_bound: f64,
    /// Number of observed units assigned to this arm.
    pub observed_support: usize,
}

/// Estimate all declared arms under independent unit-level assignment.
///
/// The variance contribution is the design expectation of the observed-arm
/// estimate for each potential-outcome arm. Twice the sum of two contributions
/// bounds their contrast variance without assuming their unknown covariance.
pub fn estimate_multi_arm(
    outcomes: &[f64],
    assignment: &[usize],
    probabilities: &[Vec<f64>],
) -> Result<Vec<ArmMean>, &'static str> {
    let n = outcomes.len();
    let arms = probabilities.first().map_or(0, Vec::len);
    if n == 0 || arms < 2 || assignment.len() != n || probabilities.len() != n {
        return Err("outcome, assignment, and multi-arm probability rows must align");
    }
    if outcomes.iter().any(|value| !value.is_finite()) {
        return Err("outcomes must be finite");
    }
    let mut estimates =
        vec![ArmMean { value: 0.0, variance_bound: 0.0, observed_support: 0 }; arms];
    for i in 0..n {
        if assignment[i] >= arms {
            return Err("assigned action index is outside the action set");
        }
        let row = &probabilities[i];
        let total = row.iter().sum::<f64>();
        if row.len() != arms
            || row.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
            || (total - 1.0).abs() > 1e-8
        {
            return Err("each action propensity must be positive and rows must sum to one");
        }
        let arm = assignment[i];
        let p = row[arm];
        estimates[arm].value += outcomes[i] / p / n as f64;
        estimates[arm].variance_bound +=
            (1.0 - p) * outcomes[i].powi(2) / p.powi(2) / (n as f64).powi(2);
        estimates[arm].observed_support += 1;
    }
    if estimates.iter().any(|arm| arm.observed_support == 0) {
        return Err("multi-arm positivity failure: every declared arm needs observed support");
    }
    if estimates.iter().any(|arm| !arm.value.is_finite() || !arm.variance_bound.is_finite()) {
        return Err("multi-arm estimate or variance overflowed finite precision");
    }
    Ok(estimates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_three_arm_means_and_missing_arm_refusal() {
        let y = [0.0, 2.0, 5.0, 0.0, 2.0, 5.0];
        let assignment = [0, 1, 2, 0, 1, 2];
        let probabilities = vec![vec![1.0 / 3.0; 3]; 6];
        let fit = estimate_multi_arm(&y, &assignment, &probabilities).unwrap();
        assert_eq!(fit.iter().map(|arm| arm.value).collect::<Vec<_>>(), [0.0, 2.0, 5.0]);
        assert_eq!(fit.iter().map(|arm| arm.observed_support).collect::<Vec<_>>(), [2, 2, 2]);
        assert!(estimate_multi_arm(&y, &[0, 1, 1, 0, 1, 1], &probabilities).is_err());
    }

    #[test]
    fn extreme_positive_assignment_probability_cannot_publish_infinite_results() {
        let probabilities = vec![vec![1e-320, 1.0], vec![0.5, 0.5]];
        assert!(estimate_multi_arm(&[1.0, 1.0], &[0, 1], &probabilities).is_err());
    }
}
