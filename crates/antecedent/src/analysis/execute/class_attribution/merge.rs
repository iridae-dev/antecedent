//! Merging class-attribution payloads and disclosing posterior envelopes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ResultPayload, StudyResult,
};
use crate::CausalPosterior;
use antecedent_attribution::{
    AnomalyScores, CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget,
};
use antecedent_core::ComponentId;
use antecedent_prob::PosteriorQuantityKind;
use std::collections::BTreeSet;

/// What merging the members' payloads produced.
pub(super) struct Merged {
    pub(super) payload: ResultPayload,
    /// Components of the union of the members, ascending.
    pub(super) components: Vec<ComponentId>,
    /// The value of each component in each member (zero where a member has no such player).
    pub(super) per_member: Vec<Vec<f64>>,
    pub(super) mean: Vec<f64>,
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
}

pub(super) fn normalized(weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

/// Weighted mean, minimum, maximum and weighted standard deviation across members, per column.
fn spread(per_member: &[Vec<f64>], weights: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let width = per_member.first().map_or(0, Vec::len);
    let mut mean = vec![0.0; width];
    let mut lower = vec![f64::INFINITY; width];
    let mut upper = vec![f64::NEG_INFINITY; width];
    for (values, weight) in per_member.iter().zip(weights) {
        for (column, value) in values.iter().enumerate() {
            mean[column] += weight * value;
            lower[column] = lower[column].min(*value);
            upper[column] = upper[column].max(*value);
        }
    }
    let mut deviation = vec![0.0; width];
    for (values, weight) in per_member.iter().zip(weights) {
        for (column, value) in values.iter().enumerate() {
            let centred = value - mean[column];
            deviation[column] += weight * centred * centred;
        }
    }
    for value in &mut deviation {
        *value = value.sqrt();
    }
    (mean, lower, upper, deviation)
}

/// Merge the members' anomaly or change payloads.
pub(super) fn merge_payload(
    identified: &[(usize, StudyResult)],
    weights: &[f64],
    hedge_components: &[ComponentId],
) -> Result<Merged, CausalError> {
    match &identified[0].1.payload {
        ResultPayload::ChangeAttribution(_) => {
            let parts: Vec<&ChangeAttributionResult> = identified
                .iter()
                .filter_map(|(_, result)| match &result.payload {
                    ResultPayload::ChangeAttribution(change) => Some(change),
                    _ => None,
                })
                .collect();
            if parts.len() != identified.len() {
                return Err(class_payload_mismatch());
            }
            Ok(merge_change(&parts, weights, hedge_components))
        }
        ResultPayload::Anomaly(_) => {
            let parts: Vec<&[AnomalyScores]> = identified
                .iter()
                .filter_map(|(_, result)| match &result.payload {
                    ResultPayload::Anomaly(scores) => Some(scores.as_slice()),
                    _ => None,
                })
                .collect();
            if parts.len() != identified.len() {
                return Err(class_payload_mismatch());
            }
            merge_anomaly(&parts, weights)
        }
        _ => Err(class_payload_mismatch()),
    }
}

fn class_payload_mismatch() -> CausalError {
    CausalError::Compile {
        message: "the members of a class attribution published different result kinds".into(),
    }
}

pub(super) fn merge_change(
    parts: &[&ChangeAttributionResult],
    weights: &[f64],
    hedge_components: &[ComponentId],
) -> Merged {
    let components: Vec<ComponentId> = parts
        .iter()
        .flat_map(|part| part.contributions.iter().map(|c| c.component))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let per_member: Vec<Vec<f64>> = parts
        .iter()
        .map(|part| {
            components
                .iter()
                .map(|component| {
                    part.contributions
                        .iter()
                        .find(|c| c.component == *component)
                        .map_or(0.0, |c| c.contribution)
                })
                .collect()
        })
        .collect();
    let (mean, lower, upper, deviation) = spread(&per_member, weights);
    let total_change: f64 =
        parts.iter().zip(weights).map(|(part, weight)| weight * part.total_change).sum();
    let observed_change = parts
        .iter()
        .zip(weights)
        .map(|(part, weight)| part.observed_change.map(|observed| weight * observed))
        .sum::<Option<f64>>();
    let unidentified: Vec<ComponentId> = parts
        .iter()
        .flat_map(|part| part.unidentified.iter().copied())
        .chain(hedge_components.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    // A component's permutation-sampling error is carried only where every member holds it and
    // the members agree on its value: a mean over disagreeing members is a different quantity
    // from the one each member's error describes. The widest member error and interval bound.
    let contributions: Vec<ComponentContribution> = components
        .iter()
        .enumerate()
        .zip(&mean)
        .map(|((position, component), contribution)| {
            let held: Option<Vec<&ComponentContribution>> = parts
                .iter()
                .map(|part| part.contributions.iter().find(|c| c.component == *component))
                .collect();
            let agree =
                (upper[position] - lower[position]).abs() <= 1e-12 * (1.0 + mean[position].abs());
            let carried = held.filter(|_| agree);
            let widest = |read: fn(&ComponentContribution) -> Option<f64>,
                          pick: fn(f64, f64) -> f64| {
                carried.as_ref().and_then(|held| {
                    held.iter()
                        .map(|c| read(c))
                        .collect::<Option<Vec<f64>>>()
                        .and_then(|values| values.into_iter().reduce(pick))
                })
            };
            ComponentContribution {
                component: *component,
                contribution: *contribution,
                stderr: widest(|c| c.stderr, f64::max),
                ci_low: widest(|c| c.ci_low, f64::min),
                ci_high: widest(|c| c.ci_high, f64::max),
            }
        })
        .collect();
    // The permutation-sampling errors of independent members combine in quadrature under the
    // weights of the mean.
    let monte_carlo_stderr = parts
        .iter()
        .zip(weights)
        .map(|(part, weight)| part.monte_carlo_stderr.map(|se| (weight * se).powi(2)))
        .sum::<Option<f64>>()
        .map(f64::sqrt);
    let budget = ComputeBudget {
        evaluations: parts.iter().map(|part| part.budget.evaluations).sum(),
        samples: parts.iter().map(|part| part.budget.samples).sum(),
        exact_coalitions: parts.iter().map(|part| part.budget.exact_coalitions).sum(),
    };
    let result = ChangeAttributionResult {
        outcome: parts[0].outcome,
        total_change,
        observed_change,
        contributions: Arc::from(contributions),
        interactions: Arc::from([]),
        path_breakdown: Arc::from([]),
        unidentified: Arc::from(unidentified),
        graph_sensitivity: Some(Arc::from(deviation)),
        budget,
        monte_carlo_stderr,
        component_mc_stderr: None,
        cache_stats: CacheStats::default(),
        fit_uncertainty: None,
    };
    Merged {
        payload: ResultPayload::ChangeAttribution(result),
        components,
        per_member,
        mean,
        lower,
        upper,
    }
}

/// Merge the members' anomaly scores: the factual scores do not depend on the graph, so the
/// rows and scores are the first member's; the contributions are the weighted mean over members
/// on the union of their players. The identified set is read on the first target.
fn merge_anomaly(parts: &[&[AnomalyScores]], weights: &[f64]) -> Result<Merged, CausalError> {
    let targets = parts[0].len();
    if parts.iter().any(|part| part.len() != targets) {
        return Err(class_payload_mismatch());
    }
    let mut merged_scores = Vec::with_capacity(targets);
    let mut first: Option<(Vec<ComponentId>, Vec<Vec<f64>>)> = None;
    for target in 0..targets {
        let base = &parts[0][target];
        let rows = base.rows.len();
        if parts.iter().any(|part| {
            part[target].target != base.target
                || part[target].rows.len() != rows
                || part[target].baseline_scores.len() != rows
        }) {
            return Err(class_payload_mismatch());
        }
        let components: Vec<ComponentId> = parts
            .iter()
            .flat_map(|part| part[target].noise_components.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let width = components.len();
        let mut contributions = vec![0.0; rows * width];
        let mut baselines = vec![0.0; rows];
        let mut per_member = Vec::with_capacity(parts.len());
        for (part, weight) in parts.iter().zip(weights) {
            let scores = &part[target];
            let held = scores.noise_components.len();
            let positions: Vec<usize> = scores
                .noise_components
                .iter()
                .filter_map(|component| components.iter().position(|c| c == component))
                .collect();
            if positions.len() != held || scores.noise_contributions.len() != rows * held {
                return Err(class_payload_mismatch());
            }
            let mut row_means = vec![0.0; width];
            for row in 0..rows {
                baselines[row] += weight * scores.baseline_scores[row];
                for (column, &position) in positions.iter().enumerate() {
                    let value = scores.noise_contributions[row * held + column];
                    contributions[row * width + position] += weight * value;
                    row_means[position] += value / rows as f64;
                }
            }
            per_member.push(row_means);
        }
        let absolute: Vec<f64> = (0..rows)
            .map(|row| contributions[row * width..(row + 1) * width].iter().map(|v| v.abs()).sum())
            .collect();
        merged_scores.push(AnomalyScores {
            target: base.target,
            rows: Arc::clone(&base.rows),
            scores: Arc::clone(&base.scores),
            baseline_scores: Arc::from(baselines),
            abs_contribution_sum: Arc::from(absolute),
            noise_components: Arc::from(components.clone()),
            noise_contributions: Arc::from(contributions),
        });
        if first.is_none() {
            first = Some((components, per_member));
        }
    }
    let (components, per_member) = first.ok_or_else(class_payload_mismatch)?;
    let (mean, lower, upper, _) = spread(&per_member, weights);
    Ok(Merged {
        payload: ResultPayload::Anomaly(merged_scores),
        components,
        per_member,
        mean,
        lower,
        upper,
    })
}

/// The posterior of the class, when it has one.
///
/// A member's posterior is conditional on its own structure and the enumeration weights are not
/// posterior mass, so members' posteriors are never mixed. A posterior is published only when
/// every member published the very same draws, as a quantity that does not depend on the
/// structure does (the mean anomaly score): it is then each member's own posterior.
pub(super) fn shared_posterior(
    identified: &[(usize, StudyResult)],
    (unidentified_mass, unevaluable_mass): (f64, f64),
) -> Option<CausalPosterior> {
    let first = identified[0].1.posterior.as_ref()?;
    for (_, result) in &identified[1..] {
        if result.posterior.as_ref()?.draws != first.draws {
            return None;
        }
    }
    let mut shared = first.clone();
    shared.unidentified_mass = unidentified_mass;
    shared.unevaluable_mass = unevaluable_mass;
    Some(shared)
}

/// The envelope of the members' 95% credible intervals, per quantity: the interval for the
/// identified set of a quantity the members' posteriors disagree on. A member without the
/// quantity (a component that is not one of its players) contributes the point zero.
pub(super) fn posterior_envelope_diagnostic(
    identified: &[(usize, StudyResult)],
) -> Option<Diagnostic> {
    let posteriors: Vec<&CausalPosterior> =
        identified.iter().map(|(_, result)| result.posterior.as_ref()).collect::<Option<_>>()?;
    if posteriors.len() < 2 {
        return None;
    }
    let name_of = |kind: &PosteriorQuantityKind| match kind {
        PosteriorQuantityKind::Effect { name } | PosteriorQuantityKind::Scalar { name } => {
            Some(Arc::clone(name))
        }
        _ => None,
    };
    let mut names: Vec<Arc<str>> = Vec::new();
    for posterior in &posteriors {
        for name in posterior.draws.schema.quantities.iter().filter_map(name_of) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    let mut ranges = Vec::with_capacity(names.len());
    for name in &names {
        let (mut lower, mut upper) = (f64::INFINITY, f64::NEG_INFINITY);
        for posterior in &posteriors {
            let position = posterior
                .draws
                .schema
                .quantities
                .iter()
                .position(|kind| name_of(kind).as_ref() == Some(name));
            let (low, high) = position.map_or((0.0, 0.0), |index| {
                (posterior.summaries.q025[index], posterior.summaries.q975[index])
            });
            lower = lower.min(low);
            upper = upper.max(high);
        }
        ranges.push(format!("{name}: [{lower}, {upper}]"));
    }
    Some(Diagnostic::new(
        "attribution.class.posterior_envelope",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the members' posteriors differ, so the class publishes no mixture of them \
             (enumeration weights are not posterior mass; each posterior stays on its atom): the \
             envelope of their 95% credible intervals is the interval for the identified set: {}",
            ranges.join("; ")
        ),
    ))
}
