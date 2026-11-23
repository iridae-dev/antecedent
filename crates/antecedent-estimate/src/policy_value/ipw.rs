//! Point-only inverse-probability policy values and score bins.

use std::borrow::Cow;

/// Point-only binary inverse-probability policy value.
///
/// Distinct from [`super::evaluate_policy_value_ipw_scores`], which requires two rows
/// and returns score standard errors. This keeps the capacity, budget, and
/// availability refusals of the Python entry point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryIpwPolicy {
    /// Policy value minus mean assigned cost.
    pub policy_net: f64,
    /// Reference value minus mean assigned reference cost.
    pub reference_net: f64,
    /// Policy net minus reference net.
    pub incremental_value: f64,
    /// Reference net minus policy net.
    pub regret: f64,
    /// Share of rows the policy treats.
    pub treatment_rate: f64,
    /// Sum of assigned policy costs.
    pub total_cost: f64,
}

/// One held-out score bin's Horvitz–Thompson contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreBinUplift {
    /// Bin index.
    pub index: usize,
    /// Mean Horvitz–Thompson score in the bin.
    pub effect: f64,
    /// Row-score standard error. A singleton bin has a zero error.
    pub standard_error: f64,
    /// Rows in the bin.
    pub rows: usize,
}

/// Point-only multi-action inverse-probability policy value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiActionIpwPolicy {
    /// Policy value minus mean assigned cost.
    pub policy_value: f64,
    /// Reference value minus mean assigned reference cost.
    pub reference_value: f64,
    /// Policy value minus reference value.
    pub incremental_value: f64,
    /// Share of rows assigned an action other than the first.
    pub non_control_rate: f64,
    /// Sum of assigned policy costs.
    pub policy_cost: f64,
}

fn binary_ipw_value(
    outcome: &[f64],
    assignment: &[bool],
    actions: &[bool],
    propensity: &[f64],
) -> Result<f64, &'static str> {
    let n = outcome.len();
    let mut value = 0.0;
    for i in 0..n {
        let e = propensity[if propensity.len() == 1 { 0 } else { i }];
        if !e.is_finite() || e <= 0.0 || e >= 1.0 {
            return Err("assignment propensities must be finite and strictly between zero and one");
        }
        if assignment[i] == actions[i] {
            let action_probability = if actions[i] { e } else { 1.0 - e };
            value += outcome[i] / action_probability;
        }
    }
    Ok(value / n as f64)
}

/// A binary policy under known randomization, with the optional reference, reference
/// costs and per-row availability the point-only evaluator defaults.
#[derive(Clone, Copy, Debug)]
pub struct BinaryIpwDesign<'a> {
    /// Realized randomized assignment.
    pub assignment: &'a [bool],
    /// Fixed policy action per row.
    pub actions: &'a [bool],
    /// Known assignment propensity, scalar or row-aligned.
    pub propensity: &'a [f64],
    /// Reference action per row; all control when `None`.
    pub reference: Option<&'a [bool]>,
    /// Policy cost of treating a row, scalar or row-aligned.
    pub costs: &'a [f64],
    /// Reference-policy costs; the policy costs when `None`.
    pub reference_costs: Option<&'a [f64]>,
    /// Whether the action is available per row; all available when `None`.
    pub available: Option<&'a [bool]>,
}

/// Declared treatment capacity and budget of a binary policy and its reference.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BinaryPolicyLimits {
    /// Most rows the policy may treat.
    pub capacity: Option<usize>,
    /// Most total cost the policy may incur.
    pub budget: Option<f64>,
    /// Most rows the reference policy may treat.
    pub reference_capacity: Option<usize>,
    /// Most total cost the reference policy may incur.
    pub reference_budget: Option<f64>,
}

/// Known-propensity binary policy value with the historical constraint checks.
#[allow(
    clippy::too_many_lines,
    reason = "binary IPW keeps the historical refusal order in one function"
)]
pub fn evaluate_binary_ipw_policy(
    outcome: &[f64],
    design: &BinaryIpwDesign<'_>,
    limits: &BinaryPolicyLimits,
) -> Result<BinaryIpwPolicy, &'static str> {
    let BinaryIpwDesign {
        assignment,
        actions,
        propensity,
        reference,
        costs,
        reference_costs,
        available,
    } = *design;
    let BinaryPolicyLimits { capacity, budget, reference_capacity, reference_budget } = *limits;
    let n = outcome.len();
    if n == 0 || assignment.len() != n || actions.len() != n {
        return Err("outcome, assignment, and policy actions must have equal non-zero length");
    }
    if propensity.len() != 1 && propensity.len() != n {
        return Err("propensity must have one value or one value per evaluation row");
    }
    let reference_rows = reference.map_or_else(|| Cow::Owned(vec![false; n]), Cow::Borrowed);
    if reference_rows.len() != n {
        return Err("reference policy length must match evaluation rows");
    }
    let available_rows = available.map_or_else(|| Cow::Owned(vec![true; n]), Cow::Borrowed);
    if available_rows.len() != n {
        return Err("availability length must match evaluation rows");
    }
    if costs.len() != 1 && costs.len() != n {
        return Err("costs must be scalar or one value per row");
    }
    let reference_cost_rows =
        reference_costs.map_or_else(|| Cow::Owned(costs.to_vec()), Cow::Borrowed);
    if reference_cost_rows.len() != 1 && reference_cost_rows.len() != n {
        return Err("reference_costs must be scalar or one value per row");
    }
    if reference_cost_rows.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err("reference costs must be finite and non-negative");
    }
    if costs.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err("treatment costs must be finite and non-negative");
    }
    if [budget, reference_budget]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return Err("budgets must be finite and non-negative");
    }
    if outcome.iter().any(|value| !value.is_finite()) {
        return Err("outcomes must be finite");
    }
    let cost = |i: usize, values: &[f64]| values[if values.len() == 1 { 0 } else { i }];
    let assigned_cost = |policy: &[bool], values: &[f64]| -> f64 {
        antecedent_kernels::numeric::CompensatedSum::of(
            policy.iter().enumerate().map(|(i, treat)| f64::from(*treat) * cost(i, values)),
        )
    };
    let policy_actions = actions.iter().filter(|&&treat| treat).count();
    if actions.iter().enumerate().any(|(i, treat)| *treat && !available_rows[i]) {
        return Err("policy treats a unit whose action is unavailable");
    }
    if reference_rows.iter().enumerate().any(|(i, treat)| *treat && !available_rows[i]) {
        return Err("reference policy treats a unit whose action is unavailable");
    }
    if capacity.is_some_and(|limit| policy_actions > limit) {
        return Err("policy exceeds the declared treatment capacity");
    }
    let reference_actions = reference_rows.iter().filter(|&&treat| treat).count();
    if reference_capacity.is_some_and(|limit| reference_actions > limit) {
        return Err("reference policy exceeds the declared treatment capacity");
    }
    let total_cost = assigned_cost(actions, costs);
    if !total_cost.is_finite()
        || budget.is_some_and(|limit| {
            !antecedent_core::policy_assignment_cost_within_budget(total_cost, limit)
        })
    {
        return Err("policy exceeds the declared treatment budget");
    }
    let reference_total_cost = assigned_cost(&reference_rows, &reference_cost_rows);
    if !reference_total_cost.is_finite()
        || reference_budget.is_some_and(|limit| {
            !antecedent_core::policy_assignment_cost_within_budget(reference_total_cost, limit)
        })
    {
        return Err("reference policy exceeds the declared treatment budget");
    }
    let policy_value = binary_ipw_value(outcome, assignment, actions, propensity)?;
    let reference_value = binary_ipw_value(outcome, assignment, &reference_rows, propensity)?;
    let policy_net = policy_value - total_cost / n as f64;
    let reference_net = reference_value - reference_total_cost / n as f64;
    Ok(BinaryIpwPolicy {
        policy_net,
        reference_net,
        incremental_value: policy_net - reference_net,
        regret: reference_net - policy_net,
        treatment_rate: policy_actions as f64 / n as f64,
        total_cost,
    })
}

/// A multi-action policy under known randomization, with the optional reference, costs
/// and availability the point-only evaluator defaults.
///
/// `probabilities` is row-major with `probability_rows` by `action_count`.
#[derive(Clone, Copy, Debug)]
pub struct MultiActionIpwDesign<'a> {
    /// Realized action index per row.
    pub assigned: &'a [usize],
    /// Fixed policy action index per row.
    pub actions: &'a [usize],
    /// Row-major randomized action probabilities.
    pub probabilities: &'a [f64],
    /// Rows of `probabilities`; must equal the evaluation row count.
    pub probability_rows: usize,
    /// Columns of `probabilities`: the number of actions.
    pub action_count: usize,
    /// Reference action index per row; all first (control) action when `None`.
    pub reference: Option<&'a [usize]>,
    /// Per-action policy cost; zero when `None`.
    pub costs: Option<&'a [f64]>,
    /// Per-action reference cost; the policy costs when `None`.
    pub reference_costs: Option<&'a [f64]>,
    /// Per-row, per-action availability; all available when `None`.
    pub available: Option<&'a [Vec<bool>]>,
}

/// Declared per-action capacities and total budgets of a multi-action policy and its
/// reference. `None` is unconstrained.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MultiActionLimits<'a> {
    /// Per-action policy capacity.
    pub capacities: Option<&'a [usize]>,
    /// Per-action reference capacity.
    pub reference_capacities: Option<&'a [usize]>,
    /// Most total policy cost.
    pub budget: Option<f64>,
    /// Most total reference cost.
    pub reference_budget: Option<f64>,
}

/// Known-randomization multi-action policy value with the historical constraint checks.
#[allow(
    clippy::too_many_lines,
    reason = "multi-action IPW keeps the historical refusal order in one function"
)]
pub fn evaluate_multi_action_ipw_policy(
    outcome: &[f64],
    design: &MultiActionIpwDesign<'_>,
    limits: &MultiActionLimits<'_>,
) -> Result<MultiActionIpwPolicy, &'static str> {
    let MultiActionIpwDesign {
        assigned,
        actions,
        probabilities,
        probability_rows,
        action_count,
        reference,
        costs,
        reference_costs,
        available,
    } = *design;
    let MultiActionLimits { capacities, reference_capacities, budget, reference_budget } = *limits;
    let n = outcome.len();
    let width = probability_rows.saturating_mul(action_count);
    if n == 0
        || probability_rows != n
        || probabilities.len() != width
        || assigned.len() != n
        || actions.len() != n
    {
        return Err(
            "outcome, assignment, policy, and propensity rows must have equal non-zero length",
        );
    }
    if action_count < 2 || assigned.iter().chain(actions).any(|&a| a >= action_count) {
        return Err("action index is outside the declared action set");
    }
    if outcome.iter().any(|value| !value.is_finite()) {
        return Err("outcomes must be finite");
    }
    for row in 0..n {
        let values = &probabilities[row * action_count..(row + 1) * action_count];
        let total = values.iter().sum::<f64>();
        if values.iter().any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0)
            || (total - 1.0).abs() > 1e-8
        {
            return Err("each action propensity must be positive and rows must sum to one");
        }
    }
    let reference_rows = reference.map_or_else(|| Cow::Owned(vec![0; n]), Cow::Borrowed);
    if reference_rows.len() != n || reference_rows.iter().any(|&a| a >= action_count) {
        return Err("reference action indexes are invalid");
    }
    let cost_rows = costs.map_or_else(|| Cow::Owned(vec![0.0; action_count]), Cow::Borrowed);
    if cost_rows.len() != action_count || cost_rows.iter().any(|c| !c.is_finite() || *c < 0.0) {
        return Err("costs must contain one finite non-negative value per action");
    }
    let reference_cost_rows =
        reference_costs.map_or_else(|| Cow::Owned(cost_rows.to_vec()), Cow::Borrowed);
    if reference_cost_rows.len() != action_count
        || reference_cost_rows.iter().any(|c| !c.is_finite() || *c < 0.0)
    {
        return Err("reference_costs must contain one finite non-negative value per action");
    }
    if [budget, reference_budget]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite() || value < 0.0)
    {
        return Err("budgets must be finite and non-negative");
    }
    let available_rows =
        available.map_or_else(|| Cow::Owned(vec![vec![true; action_count]; n]), Cow::Borrowed);
    if available_rows.len() != n || available_rows.iter().any(|row| row.len() != action_count) {
        return Err("availability must have one bool per row and action");
    }
    let capacity_rows = capacities.map_or_else(|| Cow::Owned(vec![n; action_count]), Cow::Borrowed);
    if capacity_rows.len() != action_count {
        return Err("capacities must contain one limit per action");
    }
    let reference_capacity_rows =
        reference_capacities.map_or_else(|| Cow::Owned(vec![n; action_count]), Cow::Borrowed);
    if reference_capacity_rows.len() != action_count {
        return Err("reference_capacities must contain one limit per action");
    }
    let validate_policy = |policy: &[usize],
                           limits: &[usize],
                           policy_costs: &[f64],
                           policy_budget: Option<f64>|
     -> Result<(f64, usize), &'static str> {
        let mut total_cost = antecedent_kernels::numeric::CompensatedSum::default();
        let mut counts = vec![0_usize; action_count];
        for (i, &action) in policy.iter().enumerate() {
            if !available_rows[i][action] {
                return Err("policy selects an unavailable action");
            }
            counts[action] += 1;
            total_cost.add(policy_costs[action]);
        }
        if counts.iter().zip(limits).any(|(count, limit)| count > limit) {
            return Err("policy exceeds an action capacity");
        }
        let total_cost = total_cost.total();
        if !total_cost.is_finite()
            || policy_budget.is_some_and(|limit| {
                !antecedent_core::policy_assignment_cost_within_budget(total_cost, limit)
            })
        {
            return Err("policy exceeds the declared budget");
        }
        Ok((total_cost, counts.iter().skip(1).sum()))
    };
    let (policy_cost, non_control) = validate_policy(actions, &capacity_rows, &cost_rows, budget)?;
    let (reference_cost, _) = validate_policy(
        &reference_rows,
        &reference_capacity_rows,
        &reference_cost_rows,
        reference_budget,
    )?;
    let value = |policy: &[usize]| -> Result<f64, &'static str> {
        let mut sum = 0.0;
        for i in 0..n {
            if assigned[i] == policy[i] {
                sum += outcome[i] / probabilities[i * action_count + assigned[i]];
            }
        }
        Ok(sum / n as f64)
    };
    let policy_value = value(actions)? - policy_cost / n as f64;
    let reference_value = value(&reference_rows)? - reference_cost / n as f64;
    Ok(MultiActionIpwPolicy {
        policy_value,
        reference_value,
        incremental_value: policy_value - reference_value,
        non_control_rate: non_control as f64 / n as f64,
        policy_cost,
    })
}

/// Horvitz–Thompson uplift within caller-supplied score bins.
///
/// Distinct from [`evaluate_uplift_bins`]: a bin may contain one row, and this
/// function does not form an interval.
pub fn uplift_by_score_bin(
    outcome: &[f64],
    assignment: &[bool],
    score_bin: &[usize],
    propensity: &[f64],
    bin_count: usize,
) -> Result<Vec<ScoreBinUplift>, &'static str> {
    let n = outcome.len();
    if n == 0
        || bin_count == 0
        || assignment.len() != n
        || score_bin.len() != n
        || (propensity.len() != 1 && propensity.len() != n)
    {
        return Err("uplift arrays and non-zero bin count must align");
    }
    if bin_count > n {
        // Every bin must hold a row, so this cannot succeed; refuse before sizing by it.
        return Err("every score bin must contain evaluation rows");
    }
    let mut values = vec![Vec::<f64>::new(); bin_count];
    for i in 0..n {
        let pi = propensity[if propensity.len() == 1 { 0 } else { i }];
        if !outcome[i].is_finite() || !pi.is_finite() || pi <= 0.0 || pi >= 1.0 {
            return Err("outcomes must be finite and propensities in (0, 1)");
        }
        if score_bin[i] >= bin_count {
            return Err("score bin index is outside bin_count");
        }
        let score = if assignment[i] { outcome[i] / pi } else { -outcome[i] / (1.0 - pi) };
        values[score_bin[i]].push(score);
    }
    values
        .into_iter()
        .enumerate()
        .map(|(index, group)| {
            if group.is_empty() {
                return Err("every score bin must contain evaluation rows");
            }
            let n_group = group.len();
            let effect = group.iter().sum::<f64>() / n_group as f64;
            let variance = if n_group > 1 {
                group.iter().map(|v| (v - effect).powi(2)).sum::<f64>()
                    / ((n_group - 1) as f64 * n_group as f64)
            } else {
                f64::NAN
            };
            Ok(ScoreBinUplift { index, effect, standard_error: variance.sqrt(), rows: n_group })
        })
        .collect()
}
