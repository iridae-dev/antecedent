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

/// Pointwise 95% Wald intervals for independent held-out evaluation subjects.
/// The incremental interval uses the variance of each *paired difference*.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyValueIntervals95 {
    /// Net policy value lower and upper endpoints.
    pub policy: [f64; 2],
    /// Policy-minus-reference lower and upper endpoints.
    pub incremental: [f64; 2],
}

/// Form intervals only when the policy and reference have empirical assignment
/// support and nondegenerate score variance. This is an asymptotic claim: a
/// caller must separately establish fixed recommendations, known randomized
/// propensities, independent subjects, and held-out nuisance predictions.
pub fn pointwise_intervals_95(
    scores: &PolicyValueScores,
    evaluation_rows: usize,
    policy_matches: usize,
    reference_matches: usize,
) -> Option<PolicyValueIntervals95> {
    if evaluation_rows < 30 || policy_matches < 10 || reference_matches < 10
        || !scores.policy_standard_error.is_finite()
        || !scores.incremental_standard_error.is_finite()
        || scores.policy_standard_error <= 0.0
        || scores.incremental_standard_error <= 0.0
    { return None; }
    let z = antecedent_stats::normal_ppf(0.975);
    let span = z * scores.policy_standard_error;
    let difference_span = z * scores.incremental_standard_error;
    let intervals = PolicyValueIntervals95 {
        policy: [scores.policy_value - span, scores.policy_value + span],
        incremental: [scores.incremental_value - difference_span, scores.incremental_value + difference_span],
    };
    intervals.policy.iter().chain(intervals.incremental.iter()).all(|v| v.is_finite()).then_some(intervals)
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
    fn pointwise_intervals_use_paired_incremental_scores_and_refuse_weak_support() {
        let n = 200;
        let outcomes = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect::<Vec<_>>();
        let assignment = (0..n).map(|i| i % 2 == 1).collect::<Vec<_>>();
        let actions = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>();
        let reference = vec![false; n];
        let score = evaluate_policy_value_ipw_scores(
            &outcomes, &assignment, &actions, &[0.5], &reference, &[0.25], &[0.0],
        ).unwrap();
        let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
        let reference_matches = assignment.iter().filter(|&&a| !a).count();
        let intervals = pointwise_intervals_95(&score, n, policy_matches, reference_matches).unwrap();
        let z = antecedent_stats::normal_ppf(0.975);
        assert!((intervals.policy[0] - (score.policy_value - z * score.policy_standard_error)).abs() < 1e-12);
        assert!((intervals.incremental[0] - (score.incremental_value - z * score.incremental_standard_error)).abs() < 1e-12);
        assert!(pointwise_intervals_95(&score, n, 9, reference_matches).is_none());
        assert!(pointwise_intervals_95(&score, 29, policy_matches, reference_matches).is_none());
    }

    #[test]
    fn pointwise_policy_interval_repeated_sampling_coverage() {
        // Independently randomized evaluation subjects, fixed recommendations,
        // and a shared outcome shock for the two potential outcomes. A paired
        // difference score must retain the within-row covariance.
        let n = 512;
        let simulations = 600;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / ((1_u64 << 53) as f64)
        };
        let actions = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>();
        let reference = vec![false; n];
        let truth_policy = 1.0 + 2.0 * actions.iter().filter(|&&a| a).count() as f64 / n as f64 - 0.15 * actions.iter().filter(|&&a| a).count() as f64 / n as f64;
        let truth_incremental = truth_policy - 1.0;
        let mut policy_hits_95 = 0;
        let mut incremental_hits_95 = 0;
        let mut policy_hits_90 = 0;
        let mut incremental_hits_90 = 0;
        for _ in 0..simulations {
            let assignment = (0..n).map(|_| uniform() < 0.5).collect::<Vec<_>>();
            let outcomes = assignment.iter().map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0).collect::<Vec<_>>();
            let score = evaluate_policy_value_ipw_scores(&outcomes, &assignment, &actions, &[0.5], &reference, &[0.15], &[0.0]).unwrap();
            let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
            let reference_matches = assignment.iter().filter(|&&a| !a).count();
            let interval = pointwise_intervals_95(&score, n, policy_matches, reference_matches).unwrap();
            policy_hits_95 += usize::from(interval.policy[0] <= truth_policy && truth_policy <= interval.policy[1]);
            incremental_hits_95 += usize::from(interval.incremental[0] <= truth_incremental && truth_incremental <= interval.incremental[1]);
            let z90 = antecedent_stats::normal_ppf(0.95);
            policy_hits_90 += usize::from((score.policy_value - truth_policy).abs() <= z90 * score.policy_standard_error);
            incremental_hits_90 += usize::from((score.incremental_value - truth_incremental).abs() <= z90 * score.incremental_standard_error);
        }
        for (hits, target) in [(policy_hits_95, 0.95), (incremental_hits_95, 0.95), (policy_hits_90, 0.90), (incremental_hits_90, 0.90)] {
            let rate = hits as f64 / simulations as f64;
            let mcse = (target * (1.0 - target) / simulations as f64).sqrt();
            assert!((rate - target).abs() <= 3.0 * mcse, "coverage {rate} missed target {target}");
        }
    }

    #[test]
    fn held_out_aipw_and_constrained_multi_action_intervals_cover_known_values() {
        use std::sync::Arc;
        let n = 600;
        let simulations = 500;
        let mut state = 0xD1B5_4A32_4F6C_91E7_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / ((1_u64 << 53) as f64)
        };
        let binary_reference = vec![false; n];
        let multi_reference = vec![0; n];
        let multi_costs = [0.0, 0.1, 0.2];
        let multi_truth = (1.0 + 1.9 + 3.8) / 3.0;
        let binary_truth = 1.0 + 1.85 / 3.0;
        let mut counts = [0_usize; 4];
        for _ in 0..simulations {
            // Baseline features and their fixed policy recommendations are
            // sampled independently for each evaluation subject.
            let binary_actions = (0..n).map(|_| uniform() < 1.0 / 3.0).collect::<Vec<_>>();
            let assignment = (0..n).map(|_| uniform() < 0.5).collect::<Vec<_>>();
            let outcomes = assignment.iter().map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0).collect::<Vec<_>>();
            let score = evaluate_policy_value_scores(
                &outcomes, &assignment, &binary_actions, &[0.5], &vec![1.0; n], &vec![3.0; n],
                &binary_reference, &[0.15], &[0.0],
            ).unwrap();
            let pm = assignment.iter().zip(&binary_actions).filter(|(a, b)| a == b).count();
            let rm = assignment.iter().filter(|&&a| !a).count();
            let ci = pointwise_intervals_95(&score, n, pm, rm).unwrap();
            counts[0] += usize::from(ci.policy[0] <= binary_truth && binary_truth <= ci.policy[1]);
            counts[1] += usize::from(ci.incremental[0] <= binary_truth - 1.0 && binary_truth - 1.0 <= ci.incremental[1]);

            let multi_actions = (0..n).map(|_| (uniform() * 3.0).floor() as usize).collect::<Vec<_>>();
            let multi_assignment = (0..n).map(|_| { let u = uniform(); if u < 0.5 { 0 } else if u < 0.8 { 1 } else { 2 } }).collect::<Vec<_>>();
            let multi_outcomes = multi_assignment.iter().map(|&a| [1.0, 2.0, 4.0][a] + (uniform() - 0.5) * 2.0).collect::<Vec<_>>();
            let multi = antecedent_core::MultiActionPolicyInputs {
                action_labels: Arc::from([Arc::from("control"), Arc::from("small"), Arc::from("large")]),
                assignment: Arc::from(multi_assignment), actions: Arc::from(multi_actions.clone()),
                reference: Arc::from(multi_reference.clone()),
                propensities: Arc::from((0..n).flat_map(|_| [0.5, 0.3, 0.2]).collect::<Vec<_>>()),
                available: Arc::from(vec![true; n * 3]), costs: Arc::from(multi_costs),
                reference_costs: Arc::from([0.0, 0.0, 0.0]),
                capacities: Arc::from([n, n, n]), reference_capacities: Arc::from([n, n, n]),
                budget: Some(n as f64 * 0.2), reference_budget: Some(0.0), cate_groups: Arc::from([]),
            };
            let score = evaluate_multi_action_policy_value_scores(&multi_outcomes, &multi).unwrap();
            let pm = multi.assignment.iter().zip(multi.actions.iter()).filter(|(a, b)| a == b).count();
            let rm = multi.assignment.iter().filter(|&&a| a == 0).count();
            let ci = pointwise_intervals_95(&score, n, pm, rm).unwrap();
            counts[2] += usize::from(ci.policy[0] <= multi_truth && multi_truth <= ci.policy[1]);
            counts[3] += usize::from(ci.incremental[0] <= multi_truth - 1.0 && multi_truth - 1.0 <= ci.incremental[1]);
        }
        let mcse = (0.95_f64 * 0.05 / simulations as f64).sqrt();
        for (index, hits) in counts.into_iter().enumerate() {
            let coverage = hits as f64 / simulations as f64;
            assert!((coverage - 0.95).abs() <= 3.0 * mcse, "cell {index} coverage {coverage}");
        }
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
