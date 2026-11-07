//! Frozen-score doubly robust policy value utility.
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod ipw;
pub use ipw::{
    BinaryIpwDesign, BinaryIpwPolicy, BinaryPolicyLimits, MultiActionIpwDesign,
    MultiActionIpwPolicy, MultiActionLimits, ScoreBinUplift, evaluate_binary_ipw_policy,
    evaluate_multi_action_ipw_policy, uplift_by_score_bin,
};

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
#[must_use]
pub fn pointwise_intervals_95(
    scores: &PolicyValueScores,
    evaluation_rows: usize,
    policy_matches: usize,
    reference_matches: usize,
) -> Option<PolicyValueIntervals95> {
    if evaluation_rows < 120
        || policy_matches < 10
        || reference_matches < 10
        || !scores.policy_standard_error.is_finite()
        || !scores.incremental_standard_error.is_finite()
        || scores.policy_standard_error <= 0.0
        || scores.incremental_standard_error <= 0.0
    {
        return None;
    }
    let z = antecedent_stats::normal_ppf(0.975);
    let span = z * scores.policy_standard_error;
    let difference_span = z * scores.incremental_standard_error;
    let intervals = PolicyValueIntervals95 {
        policy: [scores.policy_value - span, scores.policy_value + span],
        incremental: [
            scores.incremental_value - difference_span,
            scores.incremental_value + difference_span,
        ],
    };
    intervals
        .policy
        .iter()
        .chain(intervals.incremental.iter())
        .all(|v| v.is_finite())
        .then_some(intervals)
}

/// Regret against a prespecified finite class of binary policies, evaluated on
/// subjects independent of policy selection and construction. The selected
/// policy must be one member of `candidates`; the oracle is the best member of
/// this class, not the best unrestricted policy.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedCandidateRegret {
    /// Estimated net value of each candidate in input order.
    pub candidate_values: Vec<f64>,
    /// Plug-in best-candidate minus selected-candidate value.
    pub regret: f64,
    /// Paired standard error of each candidate-minus-selected contrast, in
    /// candidate order. The selected member has a zero contrast and SE.
    pub contrast_standard_errors: Vec<f64>,
    /// Simultaneous 95% interval for finite-class regret. It covers all paired
    /// candidate contrasts through a Bonferroni family bound.
    pub interval_95: [f64; 2],
    /// Index of the prespecified selected candidate.
    pub selected_index: usize,
}

/// Evaluate fixed recommendations with known Bernoulli propensities and paired
/// Horvitz-Thompson scores. `held_out_from_selection` is an explicit caller
/// declaration; a retained route must bind it to fold/data provenance before
/// publishing this interval. Candidate actions may depend on baseline features
/// but cannot depend on these evaluation outcomes or assignments.
pub fn evaluate_fixed_candidate_regret(
    outcome: &[f64],
    assignment: &[bool],
    propensity: &[f64],
    candidates: &[Vec<bool>],
    selected_index: usize,
    treatment_cost: &[f64],
    held_out_from_selection: bool,
) -> Result<FixedCandidateRegret, &'static str> {
    let n = outcome.len();
    let k = candidates.len();
    if !held_out_from_selection {
        return Err("finite-class regret requires held-out candidate selection and construction");
    }
    if n < 400
        || !(2..=16).contains(&k)
        || selected_index >= k
        || assignment.len() != n
        || candidates.iter().any(|actions| actions.len() != n)
        || !(propensity.len() == 1 || propensity.len() == n)
        || !(treatment_cost.len() == 1 || treatment_cost.len() == n)
    {
        return Err("finite-class regret requires 400 subjects and 2–16 aligned fixed candidates");
    }
    let at = |values: &[f64], i: usize| values[if values.len() == 1 { 0 } else { i }];
    if outcome.iter().any(|value| !value.is_finite())
        || (0..n).any(|i| {
            !at(propensity, i).is_finite()
                || !(0.2..=0.8).contains(&at(propensity, i))
                || !at(treatment_cost, i).is_finite()
                || at(treatment_cost, i) < 0.0
        })
    {
        return Err(
            "finite-class regret requires finite outcomes, costs, and randomized overlap at least 0.2",
        );
    }
    let mut scores = Vec::with_capacity(k);
    for actions in candidates {
        let matches = (0..n).filter(|&i| assignment[i] == actions[i]).count();
        if matches < 50 {
            return Err("each candidate requires at least 50 observed matching assignments");
        }
        scores.push(
            (0..n)
                .map(|i| {
                    let p = at(propensity, i);
                    let response = if assignment[i] == actions[i] {
                        outcome[i] / if actions[i] { p } else { 1.0 - p }
                    } else {
                        0.0
                    };
                    response - f64::from(actions[i]) * at(treatment_cost, i)
                })
                .collect::<Vec<_>>(),
        );
    }
    let means = scores.iter().map(|row| row.iter().sum::<f64>() / n as f64).collect::<Vec<_>>();
    let selected = &scores[selected_index];
    let critical = antecedent_stats::normal_ppf(1.0 - 0.05 / (2.0 * (k - 1) as f64));
    let mut lower: f64 = 0.0;
    let mut upper: f64 = 0.0;
    let mut regret: f64 = 0.0;
    let mut contrast_standard_errors = vec![0.0; k];
    for (j, candidate) in scores.iter().enumerate() {
        if j == selected_index {
            continue;
        }
        let difference = means[j] - means[selected_index];
        let variance = candidate
            .iter()
            .zip(selected)
            .map(|(a, b)| {
                let centered = (a - b) - difference;
                centered * centered
            })
            .sum::<f64>()
            / (n * (n - 1)) as f64;
        let span = critical * variance.sqrt();
        if !span.is_finite() {
            return Err("finite-class regret paired scores must have finite variance");
        }
        regret = regret.max(difference);
        contrast_standard_errors[j] = variance.sqrt();
        lower = lower.max(difference - span);
        upper = upper.max(difference + span);
    }
    Ok(FixedCandidateRegret {
        candidate_values: means,
        regret,
        contrast_standard_errors,
        interval_95: [lower, upper],
        selected_index,
    })
}

/// Held-out inverse-probability uplift for one descending score bin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpliftBinScore {
    /// Zero-based descending score rank.
    pub rank: usize,
    /// Horvitz-Thompson treatment contrast in this bin.
    pub effect: f64,
    /// Independent-subject row-score standard error.
    pub standard_error: f64,
    /// Number of evaluation subjects in the bin.
    pub evaluation_rows: usize,
    /// Pointwise 95% interval when held-out rank and randomized support pass.
    pub interval_95: Option<[f64; 2]>,
}

/// One randomized action effect against control in a fixed baseline stratum.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiActionCatePoint {
    /// Pre-treatment group label.
    pub group: String,
    /// Action label compared with the first (control) action.
    pub action: String,
    /// Horvitz-Thompson action minus control contrast.
    pub effect: f64,
    /// Standard error of the paired action-minus-control row scores.
    pub standard_error: f64,
    /// Pointwise interval conditional on a fixed stratum and randomized support.
    pub interval_95: Option<[f64; 2]>,
    /// Total evaluation rows in this group.
    pub evaluation_rows: usize,
    /// Rows assigned this action.
    pub observed_action_rows: usize,
    /// Rows assigned control.
    pub observed_control_rows: usize,
}

/// Estimate stratum-specific multi-action effects using known randomization probabilities.
/// Groups must have been fixed before observing evaluation outcomes. Intervals
/// additionally require independent held-out subjects, known assignment
/// probabilities, and enough observed rows in both compared arms.
pub fn evaluate_multi_action_cate(
    outcome: &[f64],
    policy: &antecedent_core::MultiActionPolicyInputs,
) -> Result<Vec<MultiActionCatePoint>, &'static str> {
    if policy.cate_groups.is_empty() {
        return Ok(Vec::new());
    }
    policy.validate().map_err(|_| "invalid multi-action CATE design")?;
    if outcome.len() != policy.assignment.len() || outcome.iter().any(|y| !y.is_finite()) {
        return Err("multi-action CATE outcomes must be finite and row-aligned");
    }
    let n = outcome.len();
    let k = policy.action_labels.len();
    let mut groups = policy.cate_groups.iter().map(std::convert::AsRef::as_ref).collect::<Vec<_>>();
    groups.sort_unstable();
    groups.dedup();
    let mut points = Vec::new();
    for group in groups {
        let rows = (0..n).filter(|&i| policy.cate_groups[i].as_ref() == group).collect::<Vec<_>>();
        let control_rows = rows.iter().filter(|&&i| policy.assignment[i] == 0).count();
        if rows.len() < 2 || control_rows == 0 {
            return Err("each CATE stratum requires at least two rows and an observed control");
        }
        for action in 1..k {
            let action_rows = rows.iter().filter(|&&i| policy.assignment[i] == action).count();
            if action_rows == 0 {
                return Err("each CATE stratum requires an observed row for every action");
            }
            let scores = rows
                .iter()
                .map(|&i| match policy.assignment[i] {
                    assigned if assigned == action => {
                        outcome[i] / policy.propensities[i * k + action]
                    }
                    0 => -outcome[i] / policy.propensities[i * k],
                    _ => 0.0,
                })
                .collect::<Vec<_>>();
            let effect = scores.iter().sum::<f64>() / rows.len() as f64;
            let standard_error = (scores.iter().map(|score| (score - effect).powi(2)).sum::<f64>()
                / (rows.len() * (rows.len() - 1)) as f64)
                .sqrt();
            let strong_overlap = rows.iter().all(|&i| {
                policy.propensities[i * k] >= 0.2 && policy.propensities[i * k + action] >= 0.2
            });
            let interval_95 = if rows.len() >= 300
                && action_rows >= 50
                && control_rows >= 50
                && strong_overlap
                && standard_error.is_finite()
                && standard_error > 0.0
            {
                let span = antecedent_stats::normal_ppf(0.975) * standard_error;
                let bounds = [effect - span, effect + span];
                bounds.iter().all(|value| value.is_finite()).then_some(bounds)
            } else {
                None
            };
            points.push(MultiActionCatePoint {
                group: group.to_owned(),
                action: policy.action_labels[action].to_string(),
                effect,
                standard_error,
                interval_95,
                evaluation_rows: rows.len(),
                observed_action_rows: action_rows,
                observed_control_rows: control_rows,
            });
        }
    }
    Ok(points)
}

/// Evaluate frozen held-out score bins using known binary randomization probabilities.
pub fn evaluate_uplift_bins(
    outcome: &[f64],
    assignment: &[bool],
    propensity: &[f64],
    score_bins: &[usize],
    bin_count: usize,
) -> Result<Vec<UpliftBinScore>, &'static str> {
    let n = outcome.len();
    if n < 2
        || assignment.len() != n
        || score_bins.len() != n
        || bin_count == 0
        || bin_count > n
        || !(propensity.len() == 1 || propensity.len() == n)
    {
        return Err("uplift rows, probabilities, and bin count must align");
    }
    let mut groups = vec![Vec::<f64>::new(); bin_count];
    let mut treated_rows = vec![0_usize; bin_count];
    let mut control_rows = vec![0_usize; bin_count];
    for i in 0..n {
        let p = propensity[if propensity.len() == 1 { 0 } else { i }];
        if !outcome[i].is_finite()
            || !p.is_finite()
            || p <= 0.0
            || p >= 1.0
            || score_bins[i] >= bin_count
        {
            return Err("uplift requires finite outcomes, strict overlap, and valid score bins");
        }
        groups[score_bins[i]].push(if assignment[i] {
            outcome[i] / p
        } else {
            -outcome[i] / (1.0 - p)
        });
        if assignment[i] {
            treated_rows[score_bins[i]] += 1;
        } else {
            control_rows[score_bins[i]] += 1;
        }
    }
    groups.into_iter().enumerate().map(|(rank, scores)| {
        let count = scores.len();
        if count < 2 { return Err("every uplift score bin requires at least two evaluation rows for a row-score standard error"); }
        let effect = scores.iter().sum::<f64>() / count as f64;
        let standard_error = (scores.iter().map(|x| (x - effect).powi(2)).sum::<f64>()
            / (count * (count - 1)) as f64).sqrt();
        let interval_95 = if count >= 300 && treated_rows[rank] >= 50 && control_rows[rank] >= 50
            && standard_error.is_finite() && standard_error > 0.0 {
            let span = antecedent_stats::normal_ppf(0.975) * standard_error;
            let bounds = [effect - span, effect + span];
            bounds.iter().all(|value| value.is_finite()).then_some(bounds)
        } else { None };
        Ok(UpliftBinScore { rank, effect, standard_error, evaluation_rows: count, interval_95 })
    }).collect()
}

/// A frozen binary policy, its reference and the known randomization, aligned on the
/// evaluation rows. `propensity`, `costs` and `reference_costs` are scalar or row-aligned.
#[derive(Clone, Copy, Debug)]
pub struct BinaryPolicyDesign<'a> {
    /// Realized randomized assignment.
    pub assignment: &'a [bool],
    /// Fixed policy action per row.
    pub actions: &'a [bool],
    /// Known assignment propensity.
    pub propensity: &'a [f64],
    /// Fixed reference action per row.
    pub reference: &'a [bool],
    /// Policy cost of treating a row.
    pub costs: &'a [f64],
    /// Reference-policy cost of treating a row.
    pub reference_costs: &'a [f64],
}

/// Evaluate fixed held-out binary policy scores using randomized AIPW.
///
/// Propensities and nuisance predictions are caller-supplied design inputs.
/// Standard errors assume independent evaluation subjects. `mu0` and `mu1` are the
/// frozen control and treated conditional outcome predictions.
pub fn evaluate_policy_value_scores(
    outcome: &[f64],
    design: &BinaryPolicyDesign<'_>,
    mu0: &[f64],
    mu1: &[f64],
) -> Result<PolicyValueScores, &'static str> {
    evaluate_scores(outcome, design, Some((mu0, mu1)))
}

/// Evaluate a fixed binary policy using known randomized probabilities only.
/// The paired row-score standard error assumes independent evaluation subjects.
pub fn evaluate_policy_value_ipw_scores(
    outcome: &[f64],
    design: &BinaryPolicyDesign<'_>,
) -> Result<PolicyValueScores, &'static str> {
    evaluate_scores(outcome, design, None)
}

/// Evaluate frozen multi-action recommendations with randomized HT row scores.
pub fn evaluate_multi_action_policy_value_scores(
    outcome: &[f64],
    policy: &antecedent_core::MultiActionPolicyInputs,
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
    #[allow(clippy::needless_range_loop, reason = "index used for multiple aligned per-row slices")]
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
        (scores.iter().map(|x| (x - average).powi(2)).sum::<f64>() / (n as f64 * (n - 1) as f64))
            .sqrt()
    };
    let pv = mean(&ps);
    let rv = mean(&rs);
    let dv = mean(&ds);
    Ok(PolicyValueScores {
        policy_value: pv,
        reference_value: rv,
        incremental_value: dv,
        relative_value_gap: rv - pv,
        treatment_rate: non_control as f64 / n as f64,
        total_cost,
        policy_standard_error: se(&ps, pv),
        reference_standard_error: se(&rs, rv),
        incremental_standard_error: se(&ds, dv),
        propensity_min: pmin,
        propensity_max: pmax,
    })
}

/// Per-row scores of a fixed binary policy, its reference and their paired difference.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryPolicyRowScores {
    /// Net policy score per evaluation row.
    pub policy: Vec<f64>,
    /// Net reference score per evaluation row.
    pub reference: Vec<f64>,
    /// Policy-minus-reference score per evaluation row.
    pub incremental: Vec<f64>,
}

/// The per-row scores of [`evaluate_policy_value_scores`] (or the point-only IPW form when
/// `nuisance` is `None`), before they are averaged. Same validation and the same scores the
/// summary averages, so a posterior over them is a posterior of the same functional.
pub fn binary_policy_row_scores(
    outcome: &[f64],
    design: &BinaryPolicyDesign<'_>,
    nuisance: Option<(&[f64], &[f64])>,
) -> Result<BinaryPolicyRowScores, &'static str> {
    let rows = row_scores(outcome, design, nuisance)?;
    Ok(BinaryPolicyRowScores {
        policy: rows.policy,
        reference: rows.reference,
        incremental: rows.incremental,
    })
}

struct RowScores {
    policy: Vec<f64>,
    reference: Vec<f64>,
    incremental: Vec<f64>,
    treated: usize,
    total_cost: f64,
    propensity_min: f64,
    propensity_max: f64,
}

fn row_scores(
    outcome: &[f64],
    design: &BinaryPolicyDesign<'_>,
    nuisance: Option<(&[f64], &[f64])>,
) -> Result<RowScores, &'static str> {
    let BinaryPolicyDesign { assignment, actions, propensity, reference, costs, reference_costs } =
        *design;
    let n = outcome.len();
    if n < 2
        || assignment.len() != n
        || actions.len() != n
        || reference.len() != n
        || nuisance.is_some_and(|(mu0, mu1)| mu0.len() != n || mu1.len() != n)
    {
        return Err(
            "outcome, assignment, policies, and predictions must align on at least two rows",
        );
    }
    if ![propensity.len(), costs.len(), reference_costs.len()]
        .iter()
        .all(|length| *length == 1 || *length == n)
    {
        return Err("propensity and costs must be scalar or row-aligned");
    }
    let at = |values: &[f64], i: usize| values[if values.len() == 1 { 0 } else { i }];
    if outcome.iter().any(|v| !v.is_finite())
        || nuisance.is_some_and(|(mu0, mu1)| mu0.iter().chain(mu1).any(|v| !v.is_finite()))
        || (0..n).any(|i| {
            !at(propensity, i).is_finite() || at(propensity, i) <= 0.0 || at(propensity, i) >= 1.0
        })
        || costs.iter().chain(reference_costs).any(|c| !c.is_finite() || *c < 0.0)
    {
        return Err("outcomes, predictions, propensities, or costs are invalid");
    }
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
            expected
                + if assignment[i] == a {
                    (outcome[i] - expected) / if a { p } else { 1.0 - p }
                } else {
                    0.0
                }
        };
        let pc = at(costs, i) * f64::from(actions[i]);
        let rc = at(reference_costs, i) * f64::from(reference[i]);
        let pv = score(actions[i]) - pc;
        let rv = score(reference[i]) - rc;
        treated += usize::from(actions[i]);
        total_cost += pc;
        ps.push(pv);
        rs.push(rv);
        ds.push(pv - rv);
    }
    Ok(RowScores {
        policy: ps,
        reference: rs,
        incremental: ds,
        treated,
        total_cost,
        propensity_min: pmin,
        propensity_max: pmax,
    })
}

fn evaluate_scores(
    outcome: &[f64],
    design: &BinaryPolicyDesign<'_>,
    nuisance: Option<(&[f64], &[f64])>,
) -> Result<PolicyValueScores, &'static str> {
    let RowScores {
        policy: ps,
        reference: rs,
        incremental: ds,
        treated,
        total_cost,
        propensity_min: pmin,
        propensity_max: pmax,
    } = row_scores(outcome, design, nuisance)?;
    let n = outcome.len();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
    let se = |v: &[f64], m: f64| {
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((n - 1) as f64 * n as f64)).sqrt()
    };
    let pv = mean(&ps);
    let rv = mean(&rs);
    let dv = mean(&ds);
    Ok(PolicyValueScores {
        policy_value: pv,
        reference_value: rv,
        incremental_value: dv,
        relative_value_gap: rv - pv,
        treatment_rate: treated as f64 / n as f64,
        total_cost,
        policy_standard_error: se(&ps, pv),
        reference_standard_error: se(&rs, rv),
        incremental_standard_error: se(&ds, dv),
        propensity_min: pmin,
        propensity_max: pmax,
    })
}

#[cfg(test)]
mod tests;
