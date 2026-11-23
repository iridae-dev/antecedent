//! Joint multi-outcome sequentially randomized excursion scores.
// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::temporal_design::{
    ClusterScores, PanelRows, contrasts::two_arm_scores, dense_labels, lagged_pairs,
    sampling_summary,
};
use std::collections::{BTreeMap, BTreeSet};

/// Outcome coordinate and measurement units retained by the native scientific target.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcursionOutcome {
    /// Stable component name used by downstream expressions.
    pub component_name: String,
    /// Stable measured-variable identity.
    pub variable_id: String,
    /// Measurement units.
    pub units: String,
    /// Original aligned measurements, including lagged follow-up.
    pub values: Vec<f64>,
}

/// Complete recorded unit-period experiment and its scientific premises.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointExcursionRecipe {
    /// Source population.
    pub population: String,
    /// Exact active intervention identity.
    pub treated_regime_id: String,
    /// Exact reference intervention identity.
    pub control_regime_id: String,
    /// Independent unit labels per observation row.
    pub unit_ids: Vec<String>,
    /// Integer period per row.
    pub periods: Vec<i64>,
    /// Recorded conditional treatment probability per decision.
    pub treatment_probabilities: Vec<f64>,
    /// Realized treatment assignment per row.
    pub assignments: Vec<bool>,
    /// Aligned measured outcomes and their scientific identities.
    pub outcomes: Vec<ExcursionOutcome>,
    /// Decision horizon, retained separately from the measured lag.
    pub horizon: u32,
    /// Outcome lag in periods.
    pub lag: u32,
    /// Probability floor for both arms.
    pub min_probability: f64,
    /// Identity of the sequential randomization law.
    pub assignment_law: String,
    /// Independent-unit premise.
    pub independence: String,
    /// Administrative end-of-follow-up premise.
    pub followup: String,
    /// Consistency and treatment-version premise.
    pub consistency: String,
    /// No between-unit interference premise.
    pub isolation: String,
}

/// Arm-coordinate means and full cluster covariance of the excursion.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointExcursionLaw {
    /// Treated coordinates followed by control coordinates, in outcome order.
    pub means: Vec<f64>,
    /// Row-major covariance of the same coordinates.
    pub covariance: Vec<f64>,
    /// Units with at least one paired decision.
    pub independent_units: usize,
    /// Total observed decision/outcome pairs.
    pub paired_decisions: usize,
}

/// Evaluate the declared excursion over independent units.
///
/// # Errors
/// Malformed rows, unsupported probabilities, missing internal follow-up or invalid scores.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded estimation procedure keeps its related calculations together"
)]
pub fn joint_excursion(
    recipe: &JointExcursionRecipe,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointExcursionLaw, String> {
    if ctx.cancellation.is_cancelled() {
        return Err("joint excursion cancelled".into());
    }
    let n = recipe.unit_ids.len();
    let k = recipe.outcomes.len();
    if n == 0
        || recipe.periods.len() != n
        || recipe.assignments.len() != n
        || recipe.treatment_probabilities.len() != n
        || k == 0
        || !recipe.min_probability.is_finite()
        || !(0.0..=0.5).contains(&recipe.min_probability)
        || recipe.min_probability == 0.0
        || recipe.treated_regime_id.trim().is_empty()
        || recipe.control_regime_id.trim().is_empty()
        || recipe.treated_regime_id == recipe.control_regime_id
        || [
            &recipe.population,
            &recipe.assignment_law,
            &recipe.independence,
            &recipe.followup,
            &recipe.consistency,
            &recipe.isolation,
        ]
        .iter()
        .any(|v| v.trim().is_empty())
        || recipe.unit_ids.iter().any(|v| v.trim().is_empty())
        || recipe.outcomes.iter().any(|v| {
            v.component_name.trim().is_empty()
                || v.variable_id.trim().is_empty()
                || v.units.trim().is_empty()
                || v.values.len() != n
                || v.values.iter().any(|x| !x.is_finite())
        })
        || recipe.outcomes.iter().map(|v| &v.component_name).collect::<BTreeSet<_>>().len() != k
        || recipe.treatment_probabilities.iter().any(|p| {
            !p.is_finite() || *p < recipe.min_probability || 1.0 - *p < recipe.min_probability
        })
    {
        return Err(
            "invalid excursion source, outcomes, probability law or scientific premises".into()
        );
    }
    let (dense, _) = dense_labels(&recipe.unit_ids);
    let pairs = lagged_pairs(PanelRows { unit: &dense, period: &recipe.periods }, recipe.lag)
        .map_err(|e| e.to_string())?;
    let treated = pairs.decision.iter().filter(|&&i| recipe.assignments[i]).count();
    if treated == 0 || treated == pairs.decision.len() {
        return Err("both randomized arms require observed paired decisions".into());
    }
    let mut max = BTreeMap::new();
    let observed = recipe
        .unit_ids
        .iter()
        .zip(&recipe.periods)
        .map(|(u, p)| (u.as_str(), *p))
        .collect::<BTreeSet<_>>();
    for (unit, period) in recipe.unit_ids.iter().zip(&recipe.periods) {
        max.entry(unit.as_str())
            .and_modify(|v: &mut i64| *v = (*v).max(*period))
            .or_insert(*period);
    }
    for (unit, period) in recipe.unit_ids.iter().zip(&recipe.periods) {
        if ctx.cancellation.is_cancelled() {
            return Err("joint excursion cancelled".into());
        }
        let target = period.checked_add(i64::from(recipe.lag)).ok_or("outcome lag overflows")?;
        if !observed.contains(&(unit.as_str(), target)) && max[unit.as_str()] >= target {
            return Err("missing internal lagged outcomes are not administrative follow-up".into());
        }
    }
    let sets = recipe
        .outcomes
        .iter()
        .map(|outcome| {
            if ctx.cancellation.is_cancelled() {
                return Err(crate::error::EstimationError::stats_msg("joint excursion cancelled"));
            }
            two_arm_scores(
                &pairs,
                &outcome.values,
                &recipe.assignments,
                &recipe.treatment_probabilities,
                &dense,
                recipe.min_probability,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let width = k.checked_mul(2).ok_or("excursion coordinate overflow")?;
    let capacity = pairs.decision.len().checked_mul(width).ok_or("excursion score overflow")?;
    let mut scores = Vec::with_capacity(capacity);
    for row in 0..pairs.decision.len() {
        if ctx.cancellation.is_cancelled() {
            return Err("joint excursion cancelled".into());
        }
        for s in &sets {
            scores.push(s.scores[row * 3 + 1]);
        }
        for s in &sets {
            scores.push(s.scores[row * 3 + 2]);
        }
    }
    let cluster = ClusterScores::accumulate(&scores, width, &sets[0].cluster_of_pair)
        .map_err(|e| e.to_string())?;
    let summary = sampling_summary(&cluster).map_err(|e| e.to_string())?;
    if ctx.cancellation.is_cancelled() {
        return Err("joint excursion cancelled".into());
    }
    Ok(JointExcursionLaw {
        means: summary.estimate,
        covariance: summary.covariance,
        independent_units: cluster.clusters(),
        paired_decisions: pairs.decision.len(),
    })
}
