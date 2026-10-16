//! Frozen-score doubly robust policy value utility.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;

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

fn evaluate_scores(
    outcome: &[f64],
    design: &BinaryPolicyDesign<'_>,
    nuisance: Option<(&[f64], &[f64])>,
) -> Result<PolicyValueScores, &'static str> {
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

/// Point-only binary inverse-probability policy value.
///
/// Distinct from [`evaluate_policy_value_ipw_scores`], which requires two rows
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
        policy.iter().enumerate().map(|(i, treat)| f64::from(*treat) * cost(i, values)).sum()
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
    if budget.is_some_and(|limit| total_cost > limit) {
        return Err("policy exceeds the declared treatment budget");
    }
    let reference_total_cost = assigned_cost(&reference_rows, &reference_cost_rows);
    if reference_budget.is_some_and(|limit| reference_total_cost > limit) {
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
        let mut total_cost = 0.0;
        let mut counts = vec![0_usize; action_count];
        for (i, &action) in policy.iter().enumerate() {
            if !available_rows[i][action] {
                return Err("policy selects an unavailable action");
            }
            counts[action] += 1;
            total_cost += policy_costs[action];
        }
        if counts.iter().zip(limits).any(|(count, limit)| count > limit) {
            return Err("policy exceeds an action capacity");
        }
        if policy_budget.is_some_and(|limit| total_cost > limit) {
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
            let variance = group.iter().map(|v| (v - effect).powi(2)).sum::<f64>()
                / ((n_group.saturating_sub(1).max(1)) as f64 * n_group as f64);
            Ok(ScoreBinUplift { index, effect, standard_error: variance.sqrt(), rows: n_group })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![cfg_attr(
        test,
        allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "fixtures derive small nonnegative counts/indices from deterministic values"
        )
    )]
    use super::*;

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one end-to-end estimator fixture; splitting would fragment the flow"
    )]
    fn fixed_multi_action_cate_intervals_cover_group_truth_and_refuse_weak_support() {
        use std::sync::Arc;
        const PER_GROUP: usize = 300;
        const DRAWS: usize = 2_000;
        let n = 2 * PER_GROUP;
        let groups = (0..n)
            .map(|i| Arc::<str>::from(if i < PER_GROUP { "g0" } else { "g1" }))
            .collect::<Vec<_>>();
        let truths = [2.0, 3.0, 1.0, 4.0];
        let mut covered = [0_usize; 4];
        let mut supported = [0_usize; 4];
        let mut state = 0x8a71_b65d_53cc_f093_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        for _ in 0..DRAWS {
            let assignment = (0..n).map(|_| (3.0 * uniform()).floor() as usize).collect::<Vec<_>>();
            let outcome = assignment
                .iter()
                .enumerate()
                .map(|(i, &action)| {
                    let group = usize::from(i >= PER_GROUP);
                    let action_effects = if group == 0 { [0.0, 2.0, 3.0] } else { [0.0, 1.0, 4.0] };
                    1.0 + action_effects[action] + 2.0 * (uniform() - 0.5)
                })
                .collect::<Vec<_>>();
            let policy = antecedent_core::MultiActionPolicyInputs {
                action_labels: Arc::from([Arc::from("control"), Arc::from("A"), Arc::from("B")]),
                assignment: Arc::from(assignment),
                actions: Arc::from(vec![0; n]),
                reference: Arc::from(vec![0; n]),
                propensities: Arc::from((0..n).flat_map(|_| [1.0 / 3.0; 3]).collect::<Vec<_>>()),
                available: Arc::from(vec![true; n * 3]),
                costs: Arc::from([0.0; 3]),
                reference_costs: Arc::from([0.0; 3]),
                capacities: Arc::from([n; 3]),
                reference_capacities: Arc::from([n; 3]),
                budget: None,
                reference_budget: None,
                cate_groups: Arc::from(groups.clone()),
            };
            let points = evaluate_multi_action_cate(&outcome, &policy).unwrap();
            assert_eq!(points.len(), 4);
            for (index, point) in points.iter().enumerate() {
                if let Some(bounds) = point.interval_95 {
                    supported[index] += 1;
                    covered[index] +=
                        usize::from(bounds[0] <= truths[index] && truths[index] <= bounds[1]);
                }
            }
        }
        for (index, hits) in covered.into_iter().enumerate() {
            let rate = hits as f64 / supported[index] as f64;
            eprintln!("multi-action CATE {index}: {hits}/{}, coverage={rate:.4}", supported[index]);
            assert!(supported[index] >= 1_990);
            assert!((0.93..=0.97).contains(&rate), "CATE contrast {index} coverage {rate}");
        }

        // The interval requires at least 300 rows in each fixed baseline stratum.
        let mut assignment = (0..299).map(|i| i % 3).collect::<Vec<_>>();
        let outcome = assignment.iter().map(|&action| 1.0 + action as f64).collect::<Vec<_>>();
        let policy = antecedent_core::MultiActionPolicyInputs {
            action_labels: Arc::from([Arc::from("control"), Arc::from("A"), Arc::from("B")]),
            assignment: Arc::from(std::mem::take(&mut assignment)),
            actions: Arc::from(vec![0; 299]),
            reference: Arc::from(vec![0; 299]),
            propensities: Arc::from((0..299).flat_map(|_| [1.0 / 3.0; 3]).collect::<Vec<_>>()),
            available: Arc::from(vec![true; 299 * 3]),
            costs: Arc::from([0.0; 3]),
            reference_costs: Arc::from([0.0; 3]),
            capacities: Arc::from([299; 3]),
            reference_capacities: Arc::from([299; 3]),
            budget: None,
            reference_budget: None,
            cate_groups: Arc::from(vec![Arc::<str>::from("g0"); 299]),
        };
        assert!(
            evaluate_multi_action_cate(&outcome, &policy)
                .unwrap()
                .iter()
                .all(|p| p.interval_95.is_none())
        );
        let mut sparse = policy.clone();
        sparse.assignment = Arc::from(
            (0..300)
                .map(|i| {
                    if i < 49 {
                        1
                    } else if i < 149 {
                        0
                    } else {
                        2
                    }
                })
                .collect::<Vec<_>>(),
        );
        sparse.actions = Arc::from(vec![0; 300]);
        sparse.reference = Arc::from(vec![0; 300]);
        sparse.propensities = Arc::from((0..300).flat_map(|_| [1.0 / 3.0; 3]).collect::<Vec<_>>());
        sparse.available = Arc::from(vec![true; 300 * 3]);
        sparse.capacities = Arc::from([300; 3]);
        sparse.reference_capacities = Arc::from([300; 3]);
        sparse.cate_groups = Arc::from(vec![Arc::<str>::from("g0"); 300]);
        let outcome =
            sparse.assignment.iter().map(|&action| 1.0 + action as f64).collect::<Vec<_>>();
        let points = evaluate_multi_action_cate(&outcome, &sparse).unwrap();
        assert_eq!(points[0].observed_action_rows, 49);
        assert!(points[0].interval_95.is_none());
        let mut weak_overlap = sparse.clone();
        weak_overlap.assignment = Arc::from((0..300).map(|i| i % 3).collect::<Vec<_>>());
        weak_overlap.propensities =
            Arc::from((0..300).flat_map(|_| [0.45, 0.45, 0.10]).collect::<Vec<_>>());
        let outcome =
            weak_overlap.assignment.iter().map(|&action| 1.0 + action as f64).collect::<Vec<_>>();
        let points = evaluate_multi_action_cate(&outcome, &weak_overlap).unwrap();
        assert!(points[1].interval_95.is_none());
    }

    #[test]
    fn boundary_probability_multi_action_cate_intervals_cover_known_truth() {
        use std::sync::Arc;
        const N: usize = 300;
        const DRAWS: usize = 2_000;
        let mut state = 0xb41d_8ac3_67f0_2219_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        let mut covered = [0_usize; 2];
        let mut supported = [0_usize; 2];
        for _ in 0..DRAWS {
            let assignment = (0..N)
                .map(|_| {
                    let u = uniform();
                    if u < 0.5 {
                        0
                    } else if u < 0.8 {
                        1
                    } else {
                        2
                    }
                })
                .collect::<Vec<_>>();
            let outcome = assignment
                .iter()
                .map(|&action| [1.0, 3.0, 4.0][action] + 2.0 * (uniform() - 0.5))
                .collect::<Vec<_>>();
            let policy = antecedent_core::MultiActionPolicyInputs {
                action_labels: Arc::from([Arc::from("control"), Arc::from("A"), Arc::from("B")]),
                assignment: Arc::from(assignment),
                actions: Arc::from(vec![0; N]),
                reference: Arc::from(vec![0; N]),
                propensities: Arc::from((0..N).flat_map(|_| [0.5, 0.3, 0.2]).collect::<Vec<_>>()),
                available: Arc::from(vec![true; N * 3]),
                costs: Arc::from([0.0; 3]),
                reference_costs: Arc::from([0.0; 3]),
                capacities: Arc::from([N; 3]),
                reference_capacities: Arc::from([N; 3]),
                budget: None,
                reference_budget: None,
                cate_groups: Arc::from(vec![Arc::<str>::from("g0"); N]),
            };
            let points = evaluate_multi_action_cate(&outcome, &policy).unwrap();
            for (index, point) in points.iter().enumerate() {
                if let Some(bounds) = point.interval_95 {
                    supported[index] += 1;
                    let truth = [2.0, 3.0][index];
                    covered[index] += usize::from(bounds[0] <= truth && truth <= bounds[1]);
                }
            }
        }
        for (index, hits) in covered.into_iter().enumerate() {
            let coverage = hits as f64 / supported[index] as f64;
            eprintln!("boundary CATE {index}: {hits}/{}, coverage={coverage:.4}", supported[index]);
            assert!(supported[index] >= 1_700);
            assert!((0.93..=0.985).contains(&coverage));
        }
    }

    #[test]
    fn held_out_uplift_bin_intervals_cover_fixed_rank_truth() {
        const BIN_ROWS: usize = 300;
        const DRAWS: usize = 2_000;
        let bins = (0..BIN_ROWS * 2).map(|i| usize::from(i >= BIN_ROWS)).collect::<Vec<_>>();
        let mut state = 0x3721_f0de_8b64_5c19_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        let truths = [2.0, 0.5];
        let mut covered = [0_usize; 2];
        for _ in 0..DRAWS {
            let assignment = (0..BIN_ROWS * 2).map(|_| uniform() < 0.5).collect::<Vec<_>>();
            let outcome = (0..BIN_ROWS * 2)
                .map(|i| 1.0 + truths[bins[i]] * f64::from(assignment[i]) + 2.0 * (uniform() - 0.5))
                .collect::<Vec<_>>();
            let points = evaluate_uplift_bins(&outcome, &assignment, &[0.5], &bins, 2).unwrap();
            for (index, point) in points.iter().enumerate() {
                let interval = point
                    .interval_95
                    .expect("300 independently randomized subjects per fixed rank bin");
                covered[index] +=
                    usize::from(interval[0] <= truths[index] && truths[index] <= interval[1]);
            }
        }
        for (index, hits) in covered.into_iter().enumerate() {
            let rate = hits as f64 / DRAWS as f64;
            eprintln!("uplift bin {index}: {hits}/{DRAWS}, coverage={rate:.4}");
            assert!((0.93..=0.97).contains(&rate), "pointwise 95% uplift-bin coverage {rate}");
        }
        let assignment = (0..299).map(|i| i % 2 == 0).collect::<Vec<_>>();
        let point =
            evaluate_uplift_bins(&vec![1.0; 299], &assignment, &[0.5], &vec![0; 299], 1).unwrap();
        assert!(point[0].interval_95.is_none());
        let assignment = (0..300).map(|i| i < 49).collect::<Vec<_>>();
        let point =
            evaluate_uplift_bins(&vec![1.0; 300], &assignment, &[0.5], &vec![0; 300], 1).unwrap();
        assert!(point[0].interval_95.is_none());
    }

    #[test]
    fn recovers_known_constant_outcome_values() {
        let result = evaluate_policy_value_scores(
            &[1.0, 3.0, 1.0, 3.0],
            &BinaryPolicyDesign {
                assignment: &[false, true, false, true],
                actions: &[false, true, true, false],
                propensity: &[0.5],
                reference: &[false; 4],
                costs: &[0.0],
                reference_costs: &[0.0],
            },
            &[1.0; 4],
            &[3.0; 4],
        )
        .unwrap();
        assert!((result.policy_value - 2.0).abs() < 1e-12);
        assert!((result.reference_value - 1.0).abs() < 1e-12);
        assert!((result.incremental_value - 1.0).abs() < 1e-12);
        assert!(result.incremental_standard_error.is_finite());
    }

    #[test]
    fn refuses_missing_overlap() {
        assert!(
            evaluate_policy_value_scores(
                &[1.0, 2.0],
                &BinaryPolicyDesign {
                    assignment: &[false, true],
                    actions: &[false, true],
                    propensity: &[0.0],
                    reference: &[false; 2],
                    costs: &[0.0],
                    reference_costs: &[0.0]
                },
                &[1.0; 2],
                &[2.0; 2]
            )
            .is_err()
        );
    }

    #[test]
    fn ipw_uses_known_assignment_probabilities_without_predictions() {
        let result = evaluate_policy_value_ipw_scores(
            &[1.0, 3.0, 1.0, 3.0],
            &BinaryPolicyDesign {
                assignment: &[false, true, false, true],
                actions: &[false, true, false, true],
                propensity: &[0.5],
                reference: &[false; 4],
                costs: &[0.0],
                reference_costs: &[0.0],
            },
        )
        .unwrap();
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
            &outcomes,
            &BinaryPolicyDesign {
                assignment: &assignment,
                actions: &actions,
                propensity: &[0.5],
                reference: &reference,
                costs: &[0.25],
                reference_costs: &[0.0],
            },
        )
        .unwrap();
        let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
        let reference_matches = assignment.iter().filter(|&&a| !a).count();
        let intervals =
            pointwise_intervals_95(&score, n, policy_matches, reference_matches).unwrap();
        let z = antecedent_stats::normal_ppf(0.975);
        assert!(
            (intervals.policy[0] - (score.policy_value - z * score.policy_standard_error)).abs()
                < 1e-12
        );
        assert!(
            (intervals.incremental[0]
                - (score.incremental_value - z * score.incremental_standard_error))
                .abs()
                < 1e-12
        );
        assert!(pointwise_intervals_95(&score, n, 9, reference_matches).is_none());
        assert!(pointwise_intervals_95(&score, 119, policy_matches, reference_matches).is_none());
    }

    #[test]
    fn pointwise_policy_interval_repeated_sampling_coverage() {
        // Independently randomized evaluation subjects, fixed recommendations,
        // and a shared outcome shock for the two potential outcomes. A paired
        // difference score must retain the within-row covariance.
        let n = 120;
        let simulations = 2_000;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        let reference = vec![false; n];
        let truth_policy = 1.0 + 1.85 / 3.0;
        let truth_incremental = truth_policy - 1.0;
        let mut policy_hits_95 = 0;
        let mut incremental_hits_95 = 0;
        let mut policy_hits_90 = 0;
        let mut incremental_hits_90 = 0;
        for _ in 0..simulations {
            let actions = (0..n).map(|_| uniform() < 1.0 / 3.0).collect::<Vec<_>>();
            let assignment = (0..n).map(|_| uniform() < 0.5).collect::<Vec<_>>();
            let outcomes = assignment
                .iter()
                .map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0)
                .collect::<Vec<_>>();
            let score = evaluate_policy_value_ipw_scores(
                &outcomes,
                &BinaryPolicyDesign {
                    assignment: &assignment,
                    actions: &actions,
                    propensity: &[0.5],
                    reference: &reference,
                    costs: &[0.15],
                    reference_costs: &[0.0],
                },
            )
            .unwrap();
            let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
            let reference_matches = assignment.iter().filter(|&&a| !a).count();
            let interval =
                pointwise_intervals_95(&score, n, policy_matches, reference_matches).unwrap();
            policy_hits_95 += usize::from(
                interval.policy[0] <= truth_policy && truth_policy <= interval.policy[1],
            );
            incremental_hits_95 += usize::from(
                interval.incremental[0] <= truth_incremental
                    && truth_incremental <= interval.incremental[1],
            );
            let z90 = antecedent_stats::normal_ppf(0.95);
            policy_hits_90 += usize::from(
                (score.policy_value - truth_policy).abs() <= z90 * score.policy_standard_error,
            );
            incremental_hits_90 += usize::from(
                (score.incremental_value - truth_incremental).abs()
                    <= z90 * score.incremental_standard_error,
            );
        }
        for (index, (hits, target)) in [
            (policy_hits_95, 0.95),
            (incremental_hits_95, 0.95),
            (policy_hits_90, 0.90),
            (incremental_hits_90, 0.90),
        ]
        .into_iter()
        .enumerate()
        {
            let rate = hits as f64 / f64::from(simulations);
            let mcse = (target * (1.0 - target) / f64::from(simulations)).sqrt();
            assert!(
                (rate - target).abs() <= 3.0 * mcse,
                "cell {index} coverage {rate} missed target {target}"
            );
        }
    }

    #[test]
    fn held_out_aipw_and_constrained_multi_action_intervals_cover_known_values() {
        use std::sync::Arc;
        let n = 300;
        let simulations = 2_000;
        let mut state = 0xD1B5_4A32_4F6C_91E7_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        let binary_reference = vec![false; n];
        let multi_reference = vec![0; n];
        let multi_costs = [0.0, 0.1, 0.2];
        let multi_truth = (1.0 + 1.9 + 3.8) / 3.0;
        let binary_truth = 1.0 + 1.85 / 3.0;
        let mut counts = [0_usize; 5];
        for _ in 0..simulations {
            // Baseline features and their fixed policy recommendations are
            // sampled independently for each evaluation subject.
            let binary_actions = (0..n).map(|_| uniform() < 1.0 / 3.0).collect::<Vec<_>>();
            let assignment = (0..n).map(|_| uniform() < 0.5).collect::<Vec<_>>();
            let outcomes = assignment
                .iter()
                .map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0)
                .collect::<Vec<_>>();
            let score = evaluate_policy_value_scores(
                &outcomes,
                &BinaryPolicyDesign {
                    assignment: &assignment,
                    actions: &binary_actions,
                    propensity: &[0.5],
                    reference: &binary_reference,
                    costs: &[0.15],
                    reference_costs: &[0.0],
                },
                &vec![1.0; n],
                &vec![3.0; n],
            )
            .unwrap();
            let pm = assignment.iter().zip(&binary_actions).filter(|(a, b)| a == b).count();
            let rm = assignment.iter().filter(|&&a| !a).count();
            let ci = pointwise_intervals_95(&score, n, pm, rm).unwrap();
            counts[0] += usize::from(ci.policy[0] <= binary_truth && binary_truth <= ci.policy[1]);
            counts[1] += usize::from(
                ci.incremental[0] <= binary_truth - 1.0 && binary_truth - 1.0 <= ci.incremental[1],
            );
            let uplift =
                evaluate_uplift_bins(&outcomes, &assignment, &[0.5], &vec![0; n], 1).unwrap();
            let uplift_ci =
                uplift[0].interval_95.expect("one held-out randomized bin of 300 subjects");
            counts[4] += usize::from(uplift_ci[0] <= 2.0 && 2.0 <= uplift_ci[1]);

            let multi_actions =
                (0..n).map(|_| (uniform() * 3.0).floor() as usize).collect::<Vec<_>>();
            let multi_assignment = (0..n)
                .map(|_| {
                    let u = uniform();
                    if u < 0.5 {
                        0
                    } else if u < 0.8 {
                        1
                    } else {
                        2
                    }
                })
                .collect::<Vec<_>>();
            let multi_outcomes = multi_assignment
                .iter()
                .map(|&a| [1.0, 2.0, 4.0][a] + (uniform() - 0.5) * 2.0)
                .collect::<Vec<_>>();
            let multi = antecedent_core::MultiActionPolicyInputs {
                action_labels: Arc::from([
                    Arc::from("control"),
                    Arc::from("small"),
                    Arc::from("large"),
                ]),
                assignment: Arc::from(multi_assignment),
                actions: Arc::from(multi_actions.clone()),
                reference: Arc::from(multi_reference.clone()),
                propensities: Arc::from((0..n).flat_map(|_| [0.5, 0.3, 0.2]).collect::<Vec<_>>()),
                available: Arc::from(vec![true; n * 3]),
                costs: Arc::from(multi_costs),
                reference_costs: Arc::from([0.0, 0.0, 0.0]),
                capacities: Arc::from([n, n, n]),
                reference_capacities: Arc::from([n, n, n]),
                budget: Some(n as f64 * 0.2),
                reference_budget: Some(0.0),
                cate_groups: Arc::from([]),
            };
            let score = evaluate_multi_action_policy_value_scores(&multi_outcomes, &multi).unwrap();
            let pm =
                multi.assignment.iter().zip(multi.actions.iter()).filter(|(a, b)| a == b).count();
            let rm = multi.assignment.iter().filter(|&&a| a == 0).count();
            let ci = pointwise_intervals_95(&score, n, pm, rm).unwrap();
            counts[2] += usize::from(ci.policy[0] <= multi_truth && multi_truth <= ci.policy[1]);
            counts[3] += usize::from(
                ci.incremental[0] <= multi_truth - 1.0 && multi_truth - 1.0 <= ci.incremental[1],
            );
        }
        let mcse = (0.95_f64 * 0.05 / f64::from(simulations)).sqrt();
        for (index, hits) in counts.into_iter().enumerate() {
            let coverage = hits as f64 / f64::from(simulations);
            assert!((coverage - 0.95).abs() <= 3.0 * mcse, "cell {index} coverage {coverage}");
        }
    }

    #[test]
    fn crossfit_aipw_policy_intervals_cover_known_values() {
        // Independently randomized evaluation subjects with five-fold
        // cross-fitted outcome nuisances: each fold's mu0/mu1 predictions are
        // the control and treated sample means computed on the other four
        // folds, so no row uses its own outcome. Cross-fitting removes
        // own-observation bias and the independent-row influence-function
        // variance stays valid, so the pointwise 95% coverage holds.
        const N: usize = 400;
        const FOLDS: usize = 5;
        let simulations = 2_000;
        let mut state = 0x51ED_270B_6A11_C3D7_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        let cost = 0.15;
        let truth_policy = 1.0 + 2.0 * (1.0 / 3.0) - cost * (1.0 / 3.0);
        let truth_incremental = truth_policy - 1.0;
        let reference = vec![false; N];
        let mut policy_covered = 0usize;
        let mut incremental_covered = 0usize;
        for _ in 0..simulations {
            let actions = (0..N).map(|_| uniform() < 1.0 / 3.0).collect::<Vec<_>>();
            let assignment = (0..N).map(|_| uniform() < 0.5).collect::<Vec<_>>();
            let outcomes = assignment
                .iter()
                .map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0)
                .collect::<Vec<_>>();
            let fold = (0..N).map(|i| i % FOLDS).collect::<Vec<_>>();
            let mut mu0 = vec![0.0; N];
            let mut mu1 = vec![0.0; N];
            for held in 0..FOLDS {
                let (mut treated_sum, mut treated_n, mut control_sum, mut control_n) =
                    (0.0, 0usize, 0.0, 0usize);
                for i in 0..N {
                    if fold[i] == held {
                        continue;
                    }
                    if assignment[i] {
                        treated_sum += outcomes[i];
                        treated_n += 1;
                    } else {
                        control_sum += outcomes[i];
                        control_n += 1;
                    }
                }
                let treated_mean = treated_sum / treated_n as f64;
                let control_mean = control_sum / control_n as f64;
                for i in 0..N {
                    if fold[i] == held {
                        mu1[i] = treated_mean;
                        mu0[i] = control_mean;
                    }
                }
            }
            let score = evaluate_policy_value_scores(
                &outcomes,
                &BinaryPolicyDesign {
                    assignment: &assignment,
                    actions: &actions,
                    propensity: &[0.5],
                    reference: &reference,
                    costs: &[cost],
                    reference_costs: &[0.0],
                },
                &mu0,
                &mu1,
            )
            .unwrap();
            let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
            let reference_matches = assignment.iter().filter(|&&a| !a).count();
            let interval =
                pointwise_intervals_95(&score, N, policy_matches, reference_matches).unwrap();
            policy_covered += usize::from(
                interval.policy[0] <= truth_policy && truth_policy <= interval.policy[1],
            );
            incremental_covered += usize::from(
                interval.incremental[0] <= truth_incremental
                    && truth_incremental <= interval.incremental[1],
            );
        }
        for (label, hits) in [("policy", policy_covered), ("incremental", incremental_covered)] {
            let coverage = hits as f64 / f64::from(simulations);
            eprintln!(
                "crossfit AIPW {label} known-truth coverage: {hits}/{simulations} = {coverage:.4}"
            );
            assert!((0.93..=0.985).contains(&coverage), "crossfit {label} coverage {coverage}");
        }
    }

    #[test]
    fn ranked_uplift_recovers_known_bin_contrasts_and_refuses_singletons() {
        let bins = evaluate_uplift_bins(
            &[9.0, 5.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            &[true, false, true, false, true, false, true, false],
            &[0.5],
            &[0, 0, 0, 0, 1, 1, 1, 1],
            2,
        )
        .unwrap();
        assert!((bins[0].effect - 4.0).abs() < 1e-12);
        assert!(bins[1].effect.abs() < 1e-12);
        assert!(bins.iter().all(|bin| bin.standard_error.is_finite()));
        assert!(evaluate_uplift_bins(&[1.0, 2.0], &[false, true], &[0.5], &[0, 1], 2).is_err());
    }

    #[test]
    fn finite_fixed_class_regret_simultaneous_interval_covers_oracle_gap() {
        const DRAWS: usize = 2_000;
        const N: usize = 400;
        // The even-subgroup policy is the class oracle. Its net gain is 0.9,
        // versus 0.3 for selected treat-all, so the finite-class regret truth
        // is 0.6. This is the target the simultaneous interval must cover.
        const ORACLE_REGRET_TRUTH: f64 = 0.6;
        let candidates = vec![
            vec![false; N],
            vec![true; N],
            (0..N).map(|i| i % 2 == 0).collect(),
            (0..N).map(|i| i % 2 == 1).collect(),
        ];
        let mut state = 0x6d3a_9f12_98e7_b40d_u64;
        let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
        for p in [0.2, 0.5] {
            let mut covered = 0;
            for _ in 0..DRAWS {
                let assignment = (0..N).map(|_| uniform() < p).collect::<Vec<_>>();
                let outcome = assignment
                    .iter()
                    .enumerate()
                    .map(|(i, &treated)| {
                        1.0 + f64::from(i % 2 == 1) * 0.5
                            + if treated { if i % 2 == 0 { 2.0 } else { -1.0 } } else { 0.0 }
                            + 2.0 * (uniform() - 0.5)
                    })
                    .collect::<Vec<_>>();
                let estimate = evaluate_fixed_candidate_regret(
                    &outcome,
                    &assignment,
                    &[p],
                    &candidates,
                    1,
                    &[0.2],
                    true,
                )
                .unwrap();
                covered += usize::from(
                    estimate.interval_95[0] <= ORACLE_REGRET_TRUTH
                        && ORACLE_REGRET_TRUTH <= estimate.interval_95[1],
                );
                assert!(estimate.interval_95[0] >= 0.0);
                assert!(estimate.interval_95[0] <= estimate.interval_95[1]);
            }
            let coverage = covered as f64 / DRAWS as f64;
            println!(
                "finite-class regret p={p}: {covered}/{DRAWS} simultaneous intervals cover {ORACLE_REGRET_TRUTH}"
            );
            assert!(
                (0.94..=1.0).contains(&coverage),
                "p={p} simultaneous regret coverage {covered}/{DRAWS}"
            );
        }
    }

    #[test]
    fn finite_fixed_class_regret_refuses_unowned_selection_and_sparse_support() {
        let n = 400;
        let outcome = vec![1.0; n];
        let assignments = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
        let candidates = vec![vec![false; n], vec![true; n]];
        let evaluate = |assignment: &[bool], p: f64, held_out| {
            evaluate_fixed_candidate_regret(
                &outcome,
                assignment,
                &[p],
                &candidates,
                0,
                &[0.0],
                held_out,
            )
        };
        assert!(evaluate(&assignments, 0.5, false).is_err());
        assert!(evaluate(&assignments, 0.1, true).is_err());
        assert!(evaluate(&vec![true; n], 0.5, true).is_err());
        assert!(
            evaluate_fixed_candidate_regret(
                &outcome,
                &assignments,
                &[0.5],
                &candidates,
                2,
                &[0.0],
                true,
            )
            .is_err()
        );
    }
}
