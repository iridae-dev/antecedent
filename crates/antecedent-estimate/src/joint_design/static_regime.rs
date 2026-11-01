//! Retained multi-outcome whole static-regime point functionals.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::excursion::ExcursionOutcome;
use crate::temporal_design::{
    dense_labels,
    regime::{RegimeMethod, RegimePanel, RegimeRows, StaticRegimes, regime_point_estimate},
};
use std::collections::{BTreeMap, BTreeSet};

/// Original complete subject histories, requested paths and identifying premises.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointStaticRegimeRecipe {
    /// Source population.
    pub population: String,
    /// Exact active intervention identity.
    pub treated_regime_id: String,
    /// Exact reference intervention identity.
    pub control_regime_id: String,
    /// Subject identity per observation.
    pub unit_ids: Vec<String>,
    /// Period per observation.
    pub periods: Vec<i64>,
    /// Recorded treatment at each period.
    pub assignments: Vec<bool>,
    /// Confounder identities, aligned with columns.
    pub confounder_ids: Vec<String>,
    /// Measured confounder history columns.
    pub confounders: Vec<Vec<f64>>,
    /// Outcome coordinates and measurements.
    pub outcomes: Vec<ExcursionOutcome>,
    /// Complete active path.
    pub active_path: Vec<bool>,
    /// Complete reference path.
    pub reference_path: Vec<bool>,
    /// Supported existing estimator.
    pub method: String,
    /// Positivity floor.
    pub min_probability: f64,
    /// Sequential positivity premise.
    pub positivity: String,
    /// Sequential exchangeability premise.
    pub exchangeability: String,
    /// Consistency premise.
    pub consistency: String,
    /// Complete follow-up premise.
    pub followup: String,
    /// Correctness premise for the selected outcome or treatment models.
    pub model_assumption: String,
    /// Between-unit isolation premise.
    pub isolation: String,
}

/// Whole-path point coordinates only; no joint sampling or posterior law is asserted.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointStaticRegimeLaw {
    /// Active coordinates, then reference coordinates.
    pub means: Vec<f64>,
    /// Observed minimum probability for each outcome, unavailable for g-formula.
    pub minimum_probabilities: Vec<Option<f64>>,
    /// Number of independent complete subjects.
    pub independent_units: usize,
}

/// Refit each existing static-regime point functional on the same declared histories.
///
/// # Errors
/// Invalid premises, trajectories, probabilities, fit, or cancellation.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded estimation procedure keeps its related calculations together"
)]
pub fn joint_static_regime(
    recipe: &JointStaticRegimeRecipe,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointStaticRegimeLaw, String> {
    let n = recipe.unit_ids.len();
    let k = recipe.outcomes.len();
    if n == 0
        || recipe.periods.len() != n
        || recipe.assignments.len() != n
        || k == 0
        || recipe.active_path.is_empty()
        || recipe.active_path.len() != recipe.reference_path.len()
        || recipe.active_path == recipe.reference_path
        || recipe.confounder_ids.is_empty()
        || recipe.confounder_ids.len() != recipe.confounders.len()
        || recipe.confounder_ids.iter().any(|id| id.trim().is_empty())
        || recipe.confounder_ids.iter().collect::<BTreeSet<_>>().len()
            != recipe.confounder_ids.len()
        || recipe.confounders.iter().any(|x| x.len() != n || x.iter().any(|v| !v.is_finite()))
        || recipe.outcomes.iter().any(|o| {
            o.component_name.trim().is_empty()
                || o.variable_id.trim().is_empty()
                || o.units.trim().is_empty()
                || o.values.len() != n
                || o.values.iter().any(|v| !v.is_finite())
        })
        || recipe.outcomes.iter().map(|o| &o.component_name).collect::<BTreeSet<_>>().len() != k
        || !recipe.min_probability.is_finite()
        || recipe.min_probability <= 0.0
        || recipe.min_probability > 0.5
        || recipe.treated_regime_id.trim().is_empty()
        || recipe.control_regime_id.trim().is_empty()
        || recipe.treated_regime_id == recipe.control_regime_id
        || [
            &recipe.population,
            &recipe.positivity,
            &recipe.exchangeability,
            &recipe.consistency,
            &recipe.followup,
            &recipe.model_assumption,
            &recipe.isolation,
        ]
        .iter()
        .any(|x| x.trim().is_empty())
    {
        return Err(
            "invalid whole-regime source, history, outcomes or identification premises".into()
        );
    }
    let method = match recipe.method.as_str() {
        "g_formula" => RegimeMethod::GFormula,
        "inverse_probability_weighting" => RegimeMethod::InverseProbability,
        "doubly_robust" => RegimeMethod::DoublyRobust,
        _ => return Err("unsupported static-regime method".into()),
    };
    let mut schedules = BTreeMap::<&str, Vec<i64>>::new();
    for (unit, period) in recipe.unit_ids.iter().zip(&recipe.periods) {
        if unit.trim().is_empty() {
            return Err("empty subject identity".into());
        }
        schedules.entry(unit).or_default().push(*period);
    }
    for schedule in schedules.values_mut() {
        schedule.sort_unstable();
    }
    if schedules.len() < 2
        || schedules.values().next().is_none_or(|first| {
            first.len() != recipe.active_path.len()
                || first.windows(2).any(|w| w[0] >= w[1])
                || schedules.values().any(|v| v != first)
        })
    {
        return Err("each subject requires the same complete unique schedule".into());
    }
    let (units, _) = dense_labels(&recipe.unit_ids);
    if !observed_path_support(recipe, &units, &vec![1.0; schedules.len()]) {
        return Err("one or both regime paths have no observed prefix support".into());
    }
    let mut active = Vec::with_capacity(k);
    let mut reference = Vec::with_capacity(k);
    let mut minima = Vec::with_capacity(k);
    for outcome in &recipe.outcomes {
        if ctx.cancellation.is_cancelled() {
            return Err("joint static regime cancelled".into());
        }
        let panel = RegimePanel::new(RegimeRows {
            unit: &units,
            period: &recipe.periods,
            outcome: &outcome.values,
            treated: &recipe.assignments,
            confounders: &recipe.confounders,
        })
        .map_err(|e| e.to_string())?;
        let (point, minimum) = regime_point_estimate(
            &panel,
            StaticRegimes { active: &recipe.active_path, reference: &recipe.reference_path },
            method,
            recipe.min_probability,
        )
        .map_err(|e| e.to_string())?;
        active.push(point[1]);
        reference.push(point[2]);
        minima.push(minimum.is_finite().then_some(minimum));
    }
    if ctx.cancellation.is_cancelled() {
        return Err("joint static regime cancelled".into());
    }
    active.extend(reference);
    Ok(JointStaticRegimeLaw {
        means: active,
        minimum_probabilities: minima,
        independent_units: schedules.len(),
    })
}

/// Aligned whole-subject bootstrap refits, including every failed slot.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointStaticRegimeBootstrap {
    /// Full-data point and positivity diagnostics.
    pub point: JointStaticRegimeLaw,
    /// Complete replicate vectors in original slot order; failures stay absent in place.
    pub realizations: Vec<Option<Vec<f64>>>,
    /// Original error for each failed slot.
    pub failures: Vec<Option<String>>,
    /// Joint covariance only when every declared refit succeeded.
    pub covariance: Option<Vec<f64>>,
}

/// Refit all outcomes on the same whole-subject bootstrap weights for every replicate.
///
/// No failed slot is discarded: any failure withholds the joint covariance. This output is
/// sampling uncertainty and never becomes posterior draws or calibrated coverage by itself.
///
/// # Errors
/// Invalid source, fewer than 99 refits, overflow, resource or cancellation refusal.
#[allow(
    clippy::too_many_lines,
    reason = "the bounded estimation procedure keeps its related calculations together"
)]
pub fn joint_static_regime_bootstrap(
    recipe: &JointStaticRegimeRecipe,
    replicates: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointStaticRegimeBootstrap, String> {
    use crate::temporal_design::regime::{MIN_REPLICATES, regime_point_estimate_weighted};
    use antecedent_core::StreamDomain;
    if replicates < MIN_REPLICATES {
        return Err("whole-subject bootstrap requires at least 99 refits".into());
    }
    let point = joint_static_regime(recipe, ctx)?;
    let width = recipe.outcomes.len().checked_mul(2).ok_or("bootstrap coordinate overflow")?;
    let covariance_size = width.checked_mul(width).ok_or("bootstrap covariance overflow")?;
    let _total = replicates.checked_mul(width).ok_or("bootstrap allocation overflow")?;
    let (units, _) = dense_labels(&recipe.unit_ids);
    let panels = recipe
        .outcomes
        .iter()
        .map(|outcome| {
            RegimePanel::new(RegimeRows {
                unit: &units,
                period: &recipe.periods,
                outcome: &outcome.values,
                treated: &recipe.assignments,
                confounders: &recipe.confounders,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    let method = match recipe.method.as_str() {
        "g_formula" => RegimeMethod::GFormula,
        "inverse_probability_weighting" => RegimeMethod::InverseProbability,
        "doubly_robust" => RegimeMethod::DoublyRobust,
        _ => return Err("unsupported static-regime method".into()),
    };
    let mut realizations = Vec::with_capacity(replicates);
    let mut failures = Vec::with_capacity(replicates);
    let mut weights = vec![0.0; point.independent_units];
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err("joint regime bootstrap cancelled".into());
        }
        let stream = u64::try_from(replicate).map_err(|_| "bootstrap stream overflow")?;
        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, 0x7e5d_4e63_0000_0000 ^ stream);
        weights.fill(0.0);
        for _ in 0..point.independent_units {
            weights[antecedent_kernels::unbiased_index(&mut rng, point.independent_units)] += 1.0;
        }
        let mut active = Vec::with_capacity(recipe.outcomes.len());
        let mut reference = Vec::with_capacity(recipe.outcomes.len());
        let mut failure = if observed_path_support(recipe, &units, &weights) {
            None
        } else {
            Some("one or both regime paths have no bootstrap prefix support".into())
        };
        for panel in &panels {
            if failure.is_some() {
                break;
            }
            if ctx.cancellation.is_cancelled() {
                return Err("joint regime bootstrap cancelled".into());
            }
            match regime_point_estimate_weighted(
                panel,
                StaticRegimes { active: &recipe.active_path, reference: &recipe.reference_path },
                method,
                recipe.min_probability,
                &weights,
            ) {
                Ok((values, _)) => {
                    active.push(values[1]);
                    reference.push(values[2]);
                }
                Err(error) => {
                    failure = Some(error.to_string());
                    break;
                }
            }
        }
        if let Some(error) = failure {
            realizations.push(None);
            failures.push(Some(error));
        } else {
            active.extend(reference);
            realizations.push(Some(active));
            failures.push(None);
        }
    }
    let covariance = if failures.iter().any(Option::is_some) {
        None
    } else {
        let mut mean = vec![0.0; width];
        for row in realizations.iter().flatten() {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += v / replicates as f64;
            }
        }
        let mut cov = vec![0.0; covariance_size];
        for row in realizations.iter().flatten() {
            for i in 0..width {
                for j in 0..width {
                    cov[i * width + j] +=
                        (row[i] - mean[i]) * (row[j] - mean[j]) / (replicates - 1) as f64;
                }
            }
        }
        if cov.iter().any(|v| !v.is_finite()) {
            return Err("bootstrap covariance overflow".into());
        }
        Some(cov)
    };
    Ok(JointStaticRegimeBootstrap { point, realizations, failures, covariance })
}

fn observed_path_support(
    recipe: &JointStaticRegimeRecipe,
    dense: &[usize],
    weights: &[f64],
) -> bool {
    let mut histories = vec![Vec::new(); weights.len()];
    for ((&unit, &period), &assignment) in
        dense.iter().zip(&recipe.periods).zip(&recipe.assignments)
    {
        if unit >= histories.len() {
            return false;
        }
        histories[unit].push((period, assignment));
    }
    for history in &mut histories {
        history.sort_unstable_by_key(|(period, _)| *period);
    }
    [&recipe.active_path, &recipe.reference_path].into_iter().all(|path| {
        (0..path.len()).all(|k| {
            histories.iter().zip(weights).any(|(history, weight)| {
                *weight > 0.0
                    && history.len() == path.len()
                    && (0..=k).all(|j| history[j].1 == path[j])
            })
        })
    })
}
