//! Finite-history adaptive regimes evaluated against treatment-affected state transitions.
// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::{BTreeMap, BTreeSet};

/// One deterministic action at one complete observed history, before the action is assigned.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveAction {
    /// States measured through the current period.
    pub states: Vec<usize>,
    /// Actions before the current period.
    pub past_actions: Vec<usize>,
    /// Action prescribed at this history.
    pub action: usize,
}

/// A prespecified policy as a finite table of history-to-action mappings.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptivePolicy {
    /// Stable policy identity.
    pub id: String,
    /// Rules for every reachable history under this policy.
    pub rules: Vec<AdaptiveAction>,
}

/// One complete independent longitudinal unit and its jointly measured outcomes.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveEpisode {
    /// Original independent unit identity.
    pub id: String,
    /// Pretreatment state at every period, including states affected by earlier treatments.
    pub states: Vec<usize>,
    /// Realized treatment at every period.
    pub actions: Vec<usize>,
    /// Terminal outcomes in the recipe's declared coordinate order.
    pub outcomes: Vec<f64>,
}

/// Complete source and scientific assumptions for two adaptive policies.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointAdaptiveRegimeRecipe {
    /// Source population identity.
    pub population: String,
    /// Identity of the observed discrete state variable.
    pub state_variable_id: String,
    /// Identity of the discrete treatment variable.
    pub action_variable_id: String,
    /// Declared outcome identities and units, with values stored in episodes.
    pub outcomes: Vec<(String, String, String)>,
    /// Active and reference deterministic history policies.
    pub policies: [AdaptivePolicy; 2],
    /// Original complete independent subject histories.
    pub episodes: Vec<AdaptiveEpisode>,
    /// Policies were prespecified independently of the evaluation outcomes.
    pub policy_prespecification: String,
    /// Population positivity for each reachable policy action given recorded history.
    pub positivity: String,
    /// Sequential exchangeability given the entire recorded state and action history.
    pub exchangeability: String,
    /// Treatment consistency assumption.
    pub consistency: String,
    /// Complete followup and state measurement assumption.
    pub followup: String,
    /// Independent sampling and no between-unit interference assumption.
    pub isolation: String,
}

/// Joint means and whole-subject jackknife covariance; all failed deletions are retained.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointAdaptiveRegimeLaw {
    /// Active outcomes followed by reference outcomes.
    pub means: Vec<f64>,
    /// Joint covariance, absent when any deletion loses support.
    pub covariance: Option<Vec<f64>>,
    /// Failure reason for each deleted subject, aligned to the original episode order.
    pub failures: Vec<Option<String>>,
    /// Independent complete subjects.
    pub independent_units: usize,
}

fn value(
    sample: &[&AdaptiveEpisode],
    policy: &AdaptivePolicy,
    states: &mut Vec<usize>,
    past: &mut Vec<usize>,
    weight: f64,
    out: &mut [f64],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), String> {
    super::budget::check_cancelled(ctx)?;
    let action = policy
        .rules
        .iter()
        .find(|r| r.states == *states && r.past_actions == *past)
        .ok_or_else(|| {
            format!("policy {} has no rule for history {:?}/{:?}", policy.id, states, past)
        })?
        .action;
    past.push(action);
    let cell: Vec<_> = sample
        .iter()
        .copied()
        .filter(|e| e.states.starts_with(states) && e.actions.starts_with(past))
        .collect();
    if cell.is_empty() {
        return Err(format!(
            "policy {} lacks observed support for history {:?}/{:?}",
            policy.id, states, past
        ));
    }
    if states.len() == cell[0].states.len() {
        for (j, target) in out.iter_mut().enumerate() {
            *target += weight * cell.iter().map(|e| e.outcomes[j]).sum::<f64>() / cell.len() as f64;
        }
    } else {
        let mut next = BTreeMap::new();
        for e in &cell {
            *next.entry(e.states[states.len()]).or_insert(0usize) += 1;
        }
        for (state, count) in next {
            states.push(state);
            value(
                sample,
                policy,
                states,
                past,
                weight * count as f64 / cell.len() as f64,
                out,
                ctx,
            )?;
            states.pop();
        }
    }
    past.pop();
    Ok(())
}

fn means(
    recipe: &JointAdaptiveRegimeRecipe,
    sample: &[&AdaptiveEpisode],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<f64>, String> {
    let mut initial = BTreeMap::new();
    for e in sample {
        *initial.entry(e.states[0]).or_insert(0usize) += 1;
    }
    let mut result = Vec::with_capacity(2 * recipe.outcomes.len());
    for policy in &recipe.policies {
        let mut out = vec![0.0; recipe.outcomes.len()];
        for (&state, &count) in &initial {
            value(
                sample,
                policy,
                &mut vec![state],
                &mut Vec::new(),
                count as f64 / sample.len() as f64,
                &mut out,
                ctx,
            )?;
        }
        result.extend(out);
    }
    Ok(result)
}

/// Evaluate deterministic adaptive policies and retain aligned whole-subject deletions.
///
/// # Errors
/// Invalid source, missing scientific premises, missing policy rules, no observed support,
/// cancellation, or resource admission failure.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded estimation procedure keeps its related calculations together"
)]
pub fn joint_adaptive_regime(
    recipe: &JointAdaptiveRegimeRecipe,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointAdaptiveRegimeLaw, String> {
    let n = recipe.episodes.len();
    let h = recipe.episodes.first().map_or(0, |e| e.states.len());
    let k = recipe.outcomes.len();
    if n < 3
        || !(1..=6).contains(&h)
        || k == 0
        || k > 16
        || [
            &recipe.population,
            &recipe.state_variable_id,
            &recipe.action_variable_id,
            &recipe.policy_prespecification,
            &recipe.positivity,
            &recipe.exchangeability,
            &recipe.consistency,
            &recipe.followup,
            &recipe.isolation,
        ]
        .iter()
        .any(|v| v.trim().is_empty())
        || recipe
            .outcomes
            .iter()
            .any(|(name, var, units)| [name, var, units].iter().any(|v| v.trim().is_empty()))
        || recipe.outcomes.iter().map(|(name, _, _)| name).collect::<BTreeSet<_>>().len() != k
        || recipe.policies[0].id.trim().is_empty()
        || recipe.policies[1].id.trim().is_empty()
        || recipe.policies[0].id == recipe.policies[1].id
        || recipe.episodes.iter().map(|e| &e.id).collect::<BTreeSet<_>>().len() != n
        || recipe.episodes.iter().any(|e| {
            e.id.trim().is_empty()
                || e.states.len() != h
                || e.actions.len() != h
                || e.outcomes.len() != k
                || e.outcomes.iter().any(|v| !v.is_finite())
        })
    {
        return Err("invalid adaptive regime source or scientific premises".into());
    }
    let max_rules = recipe.policies.iter().map(|p| p.rules.len()).max().unwrap_or(0);
    // One evaluation visits at most one history per rule of a policy (a visited history
    // needs its rule, and rules are distinct); each visit scans the episodes and the rules
    // once, comparing prefixes of at most `h` entries. The jackknife repeats the evaluation
    // once per subject.
    let work_bound = n
        .checked_mul(max_rules.max(1))
        .and_then(|v| v.checked_mul(h))
        .and_then(|v| v.checked_mul(n.saturating_add(max_rules)));
    if work_bound.is_none_or(|v| v > 100_000_000) {
        return Err("adaptive regime exceeds its bounded history work budget".into());
    }
    for policy in &recipe.policies {
        if policy.rules.len() > n.saturating_mul(h) {
            return Err("adaptive policy exceeds the bounded history-rule budget".into());
        }
        let mut seen = BTreeSet::new();
        for r in &policy.rules {
            if r.states.is_empty()
                || r.states.len() > h
                || r.past_actions.len() + 1 != r.states.len()
                || !seen.insert((&r.states, &r.past_actions))
            {
                return Err("invalid or duplicated adaptive policy rule".into());
            }
        }
    }
    let _lease = super::budget::admit_cells(
        ctx,
        "adaptive regime jackknife",
        n.saturating_mul(2 * k).saturating_add(4 * k * k),
        1,
    )?;
    let all: Vec<_> = recipe.episodes.iter().collect();
    let point = means(recipe, &all, ctx)?;
    let mut rows = Vec::with_capacity(n);
    let mut failures = Vec::with_capacity(n);
    for excluded in 0..n {
        super::budget::check_cancelled(ctx)?;
        let sample: Vec<_> =
            all.iter().enumerate().filter_map(|(i, e)| (i != excluded).then_some(*e)).collect();
        match means(recipe, &sample, ctx) {
            Ok(row) => {
                rows.push(row);
                failures.push(None);
            }
            Err(e) => {
                super::budget::check_cancelled(ctx)?;
                failures.push(Some(e));
            }
        }
    }
    let width = 2 * k;
    let covariance = if rows.len() == n {
        let center: Vec<_> =
            (0..width).map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
        let mut covariance = vec![0.0; width * width];
        for row in rows {
            for i in 0..width {
                for j in 0..width {
                    covariance[i * width + j] +=
                        (n - 1) as f64 / n as f64 * (row[i] - center[i]) * (row[j] - center[j]);
                }
            }
        }
        Some(covariance)
    } else {
        None
    };
    Ok(JointAdaptiveRegimeLaw { means: point, covariance, failures, independent_units: n })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> JointAdaptiveRegimeRecipe {
        let mut episodes = Vec::new();
        for a in 0..2 {
            for state in 0..2 {
                for b in 0..2 {
                    for repeat in 0..if a == state { 6 } else { 2 } {
                        let y = (2 * a + 3 * b + 4 * state + a * b) as f64;
                        episodes.push(AdaptiveEpisode {
                            id: format!("{a}/{state}/{b}/{repeat}"),
                            states: vec![0, state],
                            actions: vec![a, b],
                            outcomes: vec![y, 2.0 * y],
                        });
                    }
                }
            }
        }
        let policy = |id: &str, first: usize, adaptive: bool| AdaptivePolicy {
            id: id.into(),
            rules: vec![
                AdaptiveAction { states: vec![0], past_actions: vec![], action: first },
                AdaptiveAction { states: vec![0, 0], past_actions: vec![first], action: 0 },
                AdaptiveAction {
                    states: vec![0, 1],
                    past_actions: vec![first],
                    action: usize::from(adaptive),
                },
            ],
        };
        JointAdaptiveRegimeRecipe {
            population: "pop".into(),
            state_variable_id: "state".into(),
            action_variable_id: "treatment".into(),
            outcomes: vec![
                ("y".into(), "y".into(), "u".into()),
                ("z".into(), "z".into(), "u".into()),
            ],
            policies: [policy("adaptive", 1, true), policy("reference", 0, false)],
            episodes,
            policy_prespecification: "fixed before evaluation".into(),
            positivity: "population support".into(),
            exchangeability: "conditional on full measured history".into(),
            consistency: "consistency".into(),
            followup: "complete".into(),
            isolation: "independent units".into(),
        }
    }

    #[test]
    fn adaptive_full_history_gformula_and_joint_deletions() {
        let r = fixture();
        let law =
            joint_adaptive_regime(&r, &antecedent_core::ExecutionContext::for_tests(1)).unwrap();
        // Independent arithmetic: P(L1=1 | A0=1)=3/4 and P(L1=1 | A0=0)=1/4.
        // The adaptive action is A1=L1, so E[Y(policy)] = 1/4*2 + 3/4*10 = 8;
        // the reference action is always zero and its value is 1.
        assert_eq!(law.means, vec![8.0, 16.0, 1.0, 2.0]);
        let covariance = law.covariance.unwrap();
        assert!((covariance[1] - 2.0 * covariance[0]).abs() < 1e-12);
        assert!(law.failures.iter().all(Option::is_none));
        let mut invalid = r.clone();
        invalid.policies[0].rules.pop();
        assert!(
            joint_adaptive_regime(&invalid, &antecedent_core::ExecutionContext::for_tests(1))
                .is_err()
        );
        // The work of the jackknife grows with the rules a policy can visit, not with the
        // square of the subjects: a few rules over many subjects are admitted.
        let mut large = r.clone();
        large.episodes = (0..40)
            .flat_map(|copy| {
                r.episodes
                    .iter()
                    .map(move |e| AdaptiveEpisode { id: format!("{copy}/{}", e.id), ..e.clone() })
            })
            .collect();
        assert_eq!(large.episodes.len(), 1280);
        let law = joint_adaptive_regime(&large, &antecedent_core::ExecutionContext::for_tests(1))
            .unwrap();
        assert_eq!(law.means, vec![8.0, 16.0, 1.0, 2.0]);
        let mut sparse = r;
        // The policy remains supported in the full sample; one deletion loses a necessary cell.
        sparse
            .episodes
            .retain(|e| !(e.actions == [1, 1] && e.states == [0, 1]) || e.id.ends_with("/0"));
        let law = joint_adaptive_regime(&sparse, &antecedent_core::ExecutionContext::for_tests(1))
            .unwrap();
        assert!(law.covariance.is_none());
        assert!(law.failures.iter().any(Option::is_some));
    }
}
