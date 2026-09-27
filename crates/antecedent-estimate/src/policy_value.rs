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

/// Held-out inverse-probability uplift for one descending score bin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpliftBinScore {
    /// Zero-based descending score rank.
    pub rank: usize,
    /// Horvitz-Thompson treatment contrast in this bin.
    pub effect: f64,
    /// Independent-subject row-score standard error; no interval claim.
    pub standard_error: f64,
    /// Number of evaluation subjects in the bin.
    pub evaluation_rows: usize,
}

/// Point estimate of one randomized action effect against control in a fixed baseline stratum.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiActionCatePoint {
    /// Pre-treatment group label.
    pub group: String,
    /// Action label compared with the first (control) action.
    pub action: String,
    /// Horvitz-Thompson action minus control contrast.
    pub effect: f64,
    /// Total evaluation rows in this group.
    pub evaluation_rows: usize,
    /// Rows assigned this action.
    pub observed_action_rows: usize,
    /// Rows assigned control.
    pub observed_control_rows: usize,
}

/// Estimate stratum-specific multi-action effects using known randomization probabilities.
/// Groups must have been fixed before observing evaluation outcomes. No interval is licensed.
pub fn evaluate_multi_action_cate(
    outcome: &[f64], policy: &antecedent_core::MultiActionPolicyInputs,
) -> Result<Vec<MultiActionCatePoint>, &'static str> {
    if policy.cate_groups.is_empty() { return Ok(Vec::new()); }
    policy.validate().map_err(|_| "invalid multi-action CATE design")?;
    if outcome.len() != policy.assignment.len() || outcome.iter().any(|y| !y.is_finite()) {
        return Err("multi-action CATE outcomes must be finite and row-aligned");
    }
    let n = outcome.len();
    let k = policy.action_labels.len();
    let mut groups = policy.cate_groups.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    groups.sort_unstable();
    groups.dedup();
    let mut points = Vec::new();
    for group in groups {
        let rows = (0..n).filter(|&i| policy.cate_groups[i].as_ref() == group).collect::<Vec<_>>();
        let control_rows = rows.iter().filter(|&&i| policy.assignment[i] == 0).count();
        if rows.len() < 2 || control_rows == 0 {
            return Err("each CATE stratum requires at least two rows and an observed control");
        }
        let control = rows.iter().map(|&i| {
            if policy.assignment[i] == 0 { outcome[i] / policy.propensities[i * k] } else { 0.0 }
        }).sum::<f64>() / rows.len() as f64;
        for action in 1..k {
            let action_rows = rows.iter().filter(|&&i| policy.assignment[i] == action).count();
            if action_rows == 0 { return Err("each CATE stratum requires an observed row for every action"); }
            let treated = rows.iter().map(|&i| {
                if policy.assignment[i] == action { outcome[i] / policy.propensities[i * k + action] } else { 0.0 }
            }).sum::<f64>() / rows.len() as f64;
            points.push(MultiActionCatePoint {
                group: group.to_owned(), action: policy.action_labels[action].to_string(),
                effect: treated - control, evaluation_rows: rows.len(),
                observed_action_rows: action_rows, observed_control_rows: control_rows,
            });
        }
    }
    Ok(points)
}

/// Evaluate frozen held-out score bins using known binary randomization probabilities.
pub fn evaluate_uplift_bins(
    outcome: &[f64], assignment: &[bool], propensity: &[f64],
    score_bins: &[usize], bin_count: usize,
) -> Result<Vec<UpliftBinScore>, &'static str> {
    let n = outcome.len();
    if n < 2 || assignment.len() != n || score_bins.len() != n || bin_count == 0
        || bin_count > n || !(propensity.len() == 1 || propensity.len() == n)
    { return Err("uplift rows, probabilities, and bin count must align"); }
    let mut groups = vec![Vec::<f64>::new(); bin_count];
    for i in 0..n {
        let p = propensity[if propensity.len() == 1 { 0 } else { i }];
        if !outcome[i].is_finite() || !p.is_finite() || p <= 0.0 || p >= 1.0
            || score_bins[i] >= bin_count
        { return Err("uplift requires finite outcomes, strict overlap, and valid score bins"); }
        groups[score_bins[i]].push(if assignment[i] { outcome[i] / p } else { -outcome[i] / (1.0 - p) });
    }
    groups.into_iter().enumerate().map(|(rank, scores)| {
        let count = scores.len();
        if count < 2 { return Err("every uplift score bin requires at least two evaluation rows for a row-score standard error"); }
        let effect = scores.iter().sum::<f64>() / count as f64;
        let standard_error = (scores.iter().map(|x| (x - effect).powi(2)).sum::<f64>()
            / (count * (count - 1)) as f64).sqrt();
        Ok(UpliftBinScore { rank, effect, standard_error, evaluation_rows: count })
    }).collect()
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

/// Evaluate frozen multi-action recommendations with randomized HT row scores.
pub fn evaluate_multi_action_policy_value_scores(
    outcome: &[f64], policy: &antecedent_core::MultiActionPolicyInputs,
) -> Result<PolicyValueScores, &'static str> {
    let n = outcome.len();
    let k = policy.action_labels.len();
    if n != policy.assignment.len() || outcome.iter().any(|y| !y.is_finite()) {
        return Err("multi-action outcomes must be finite and align with evaluation rows");
    }
    policy.validate().map_err(|_| "invalid multi-action policy design or constraints")?;
    let mut ps = Vec::with_capacity(n);
    let mut rs = Vec::with_capacity(n);
    let mut ds = Vec::with_capacity(n);
    let mut total_cost = 0.0;
    let mut non_control = 0usize;
    let mut pmin = f64::INFINITY;
    let mut pmax: f64 = 0.0;
    for i in 0..n {
        let assigned = policy.assignment[i];
        let action = policy.actions[i];
        let reference = policy.reference[i];
        let probability = policy.propensities[i * k + assigned];
        for &p in &policy.propensities[i * k..(i + 1) * k] {
            pmin = pmin.min(p);
            pmax = pmax.max(p);
        }
        let cost = policy.costs[action];
        let pv = if assigned == action { outcome[i] / probability } else { 0.0 } - cost;
        let rv = if assigned == reference { outcome[i] / probability } else { 0.0 }
            - policy.reference_costs[reference];
        total_cost += cost;
        non_control += usize::from(action != 0);
        ps.push(pv);
        rs.push(rv);
        ds.push(pv - rv);
    }
    let mean = |scores: &[f64]| scores.iter().sum::<f64>() / n as f64;
    let se = |scores: &[f64], average: f64| {
        (scores.iter().map(|x| (x - average).powi(2)).sum::<f64>() / (n * (n - 1)) as f64).sqrt()
    };
    let pv = mean(&ps);
    let rv = mean(&rs);
    let dv = mean(&ds);
    Ok(PolicyValueScores {
        policy_value: pv, reference_value: rv, incremental_value: dv,
        relative_value_gap: rv - pv, treatment_rate: non_control as f64 / n as f64,
        total_cost, policy_standard_error: se(&ps, pv), reference_standard_error: se(&rs, rv),
        incremental_standard_error: se(&ds, dv), propensity_min: pmin, propensity_max: pmax,
    })
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

    #[test]
    fn ranked_uplift_recovers_known_bin_contrasts_and_refuses_singletons() {
        let bins = evaluate_uplift_bins(
            &[9.0, 5.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            &[true, false, true, false, true, false, true, false],
            &[0.5], &[0, 0, 0, 0, 1, 1, 1, 1], 2,
        ).unwrap();
        assert!((bins[0].effect - 4.0).abs() < 1e-12);
        assert!(bins[1].effect.abs() < 1e-12);
        assert!(bins.iter().all(|bin| bin.standard_error.is_finite()));
        assert!(evaluate_uplift_bins(&[1.0, 2.0], &[false, true], &[0.5], &[0, 1], 2).is_err());
    }
}
