//! Frozen-score doubly robust policy value utility.
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Result from evaluating one fixed policy and its reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyValueScores {
    /// Net policy value per evaluation subject.
    pub policy_value: f64,
    /// Net reference value per evaluation subject.
    pub reference_value: f64,
    /// Paired incremental net value.
    pub incremental_value: f64,
    /// Relative value gap (`reference - policy`).
    pub relative_value_gap: f64,
    /// Fraction assigned treatment by the policy.
    pub treatment_rate: f64,
    /// Sum of assigned policy costs.
    pub total_cost: f64,
    /// Row-score standard error for policy net value.
    pub policy_standard_error: f64,
    /// Row-score standard error for reference net value.
    pub reference_standard_error: f64,
    /// Paired row-score standard error for incremental net value.
    pub incremental_standard_error: f64,
    /// Minimum declared propensity.
    pub propensity_min: f64,
    /// Maximum declared propensity.
    pub propensity_max: f64,
}

/// Evaluate fixed held-out binary policy scores using randomized AIPW.
///
/// Propensities and nuisance predictions are caller-supplied design inputs.
/// Standard errors assume independent evaluation subjects.
pub fn evaluate_policy_value_scores(
    outcome: &[f64], assignment: &[bool], actions: &[bool], propensity: &[f64],
    mu0: &[f64], mu1: &[f64], reference: &[bool], costs: &[f64], reference_costs: &[f64],
) -> Result<PolicyValueScores, &'static str> {
    evaluate_scores(outcome, assignment, actions, propensity, Some((mu0, mu1)), reference, costs, reference_costs)
}

/// Evaluate a fixed binary policy using known randomized probabilities only.
/// The paired row-score standard error assumes independent evaluation subjects.
pub fn evaluate_policy_value_ipw_scores(
    outcome: &[f64], assignment: &[bool], actions: &[bool], propensity: &[f64],
    reference: &[bool], costs: &[f64], reference_costs: &[f64],
) -> Result<PolicyValueScores, &'static str> {
    evaluate_scores(outcome, assignment, actions, propensity, None, reference, costs, reference_costs)
}

fn evaluate_scores(
    outcome: &[f64], assignment: &[bool], actions: &[bool], propensity: &[f64],
    nuisance: Option<(&[f64], &[f64])>, reference: &[bool], costs: &[f64], reference_costs: &[f64],
) -> Result<PolicyValueScores, &'static str> {
    let n = outcome.len();
    if n < 2 || assignment.len() != n || actions.len() != n || reference.len() != n
        || nuisance.is_some_and(|(mu0, mu1)| mu0.len() != n || mu1.len() != n) {
        return Err("outcome, assignment, policies, and predictions must align on at least two rows");
    }
    if ![propensity.len(), costs.len(), reference_costs.len()].iter().all(|length| *length == 1 || *length == n) {
        return Err("propensity and costs must be scalar or row-aligned");
    }
    let at = |values: &[f64], i: usize| values[if values.len() == 1 { 0 } else { i }];
    if outcome.iter().any(|v| !v.is_finite())
        || nuisance.is_some_and(|(mu0, mu1)| mu0.iter().chain(mu1).any(|v| !v.is_finite()))
        || (0..n).any(|i| !at(propensity, i).is_finite() || at(propensity, i) <= 0.0 || at(propensity, i) >= 1.0)
        || costs.iter().chain(reference_costs).any(|c| !c.is_finite() || *c < 0.0)
    { return Err("outcomes, predictions, propensities, or costs are invalid"); }
    let mut ps = Vec::with_capacity(n);
    let mut rs = Vec::with_capacity(n);
    let mut ds = Vec::with_capacity(n);
    let mut treated = 0usize;
    let mut total_cost = 0.0;
    let mut pmin = f64::INFINITY;
    let mut pmax: f64 = 0.0;
    for i in 0..n {
        let p = at(propensity, i);
        pmin = pmin.min(p);
        pmax = pmax.max(p);
        let score = |a: bool| {
            let expected = nuisance.map_or(0.0, |(mu0, mu1)| if a { mu1[i] } else { mu0[i] });
            expected + if assignment[i] == a {
                (outcome[i] - expected) / if a { p } else { 1.0 - p }
            } else { 0.0 }
        };
        let pc = at(costs, i) * f64::from(actions[i]);
        let rc = at(reference_costs, i) * f64::from(reference[i]);
        let pv = score(actions[i]) - pc;
        let rv = score(reference[i]) - rc;
        treated += usize::from(actions[i]);
        total_cost += pc;
        ps.push(pv); rs.push(rv); ds.push(pv - rv);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
    let se = |v: &[f64], m: f64| (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((n - 1) as f64 * n as f64)).sqrt();
    let pv = mean(&ps); let rv = mean(&rs); let dv = mean(&ds);
    Ok(PolicyValueScores { policy_value: pv, reference_value: rv, incremental_value: dv,
        relative_value_gap: rv-pv, treatment_rate: treated as f64/n as f64, total_cost,
        policy_standard_error: se(&ps,pv), reference_standard_error: se(&rs,rv),
        incremental_standard_error: se(&ds,dv), propensity_min: pmin, propensity_max: pmax })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_constant_outcome_values() {
        let result = evaluate_policy_value_scores(
            &[1.0, 3.0, 1.0, 3.0],
            &[false, true, false, true],
            &[false, true, true, false],
            &[0.5],
            &[1.0; 4],
            &[3.0; 4],
            &[false; 4],
            &[0.0],
            &[0.0],
        ).unwrap();
        assert!((result.policy_value - 2.0).abs() < 1e-12);
        assert!((result.reference_value - 1.0).abs() < 1e-12);
        assert!((result.incremental_value - 1.0).abs() < 1e-12);
        assert!(result.incremental_standard_error.is_finite());
    }

    #[test]
    fn refuses_missing_overlap() {
        assert!(evaluate_policy_value_scores(
            &[1.0, 2.0], &[false, true], &[false, true], &[0.0], &[1.0; 2],
            &[2.0; 2], &[false; 2], &[0.0], &[0.0],
        ).is_err());
    }

    #[test]
    fn ipw_uses_known_assignment_probabilities_without_predictions() {
        let result = evaluate_policy_value_ipw_scores(
            &[1.0, 3.0, 1.0, 3.0],
            &[false, true, false, true],
            &[false, true, false, true],
            &[0.5],
            &[false; 4],
            &[0.0],
            &[0.0],
        ).unwrap();
        assert!((result.policy_value - 4.0).abs() < 1e-12);
        assert!((result.reference_value - 1.0).abs() < 1e-12);
        assert!((result.incremental_value - 3.0).abs() < 1e-12);
        assert!(result.incremental_standard_error.is_finite());
    }
}
