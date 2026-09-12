//! Graph-sensitive root-cause ranking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ComponentId, ExecutionContext};
use antecedent_prob::{PosteriorDraws, WeightedGraphSamples};

use crate::error::AttributionError;
use crate::result::{ChangeAttributionResult, RootCauseRank};

/// Stable weighted central moments; missing structural components are explicit
/// zero contributions in the model-collection aggregation.
#[derive(Default)]
struct ContributionMoments {
    mass: f64,
    mean: f64,
    m2: f64,
}

impl ContributionMoments {
    fn add(&mut self, weight: f64, value: f64) -> Result<(), AttributionError> {
        if weight == 0.0 {
            return Ok(());
        }
        if !value.is_finite() {
            return Err(AttributionError::invalid_input("contributions must be finite"));
        }
        let total = self.mass + weight;
        let delta = value - self.mean;
        self.mean += (weight / total) * delta;
        self.m2 += weight * delta * (value - self.mean);
        self.mass = total;
        Ok(())
    }

    fn rank(self, component: ComponentId) -> RootCauseRank {
        RootCauseRank {
            component,
            score: self.mean.abs(),
            graph_std: Some((self.m2 / self.mass).max(0.0).sqrt()),
        }
    }
}

fn weight_scale(weights: &[f64]) -> Result<f64, AttributionError> {
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(AttributionError::invalid_input(
            "graph weights must be finite and nonnegative",
        ));
    }
    let scale = weights.iter().copied().fold(0.0, f64::max);
    if scale == 0.0 {
        return Err(AttributionError::invalid_input("graph weights require positive mass"));
    }
    Ok(scale)
}

/// Rank components by absolute contribution, optionally aggregating across a
/// graph ensemble's contribution draws.
///
/// When `graph_samples` and `contribution_draws` are provided, each draw column
/// is a component contribution; weights come from the graph ensemble. Results
/// include per-rank `graph_std`.
///
/// # Errors
///
/// Shape mismatches or empty contributions.
pub fn root_cause_rank(
    attribution: &ChangeAttributionResult,
    graph_samples: Option<&WeightedGraphSamples>,
    contribution_draws: Option<&PosteriorDraws>,
    _ctx: &ExecutionContext,
) -> Result<Vec<RootCauseRank>, AttributionError> {
    if attribution.contributions.is_empty() {
        return Err(AttributionError::invalid_input(
            "root_cause_rank requires non-empty contributions",
        ));
    }

    if let (Some(gs), Some(draws)) = (graph_samples, contribution_draws) {
        return rank_with_graph_uncertainty(attribution, gs, draws);
    }

    let mut ranks: Vec<RootCauseRank> = attribution
        .contributions
        .iter()
        .map(|c| RootCauseRank {
            component: c.component,
            score: c.contribution.abs(),
            graph_std: None,
        })
        .collect();
    ranks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ranks)
}

/// Aggregate attribution results over a [`ModelCollection`](antecedent_model::ModelCollection) by weighted mean.
///
/// # Errors
///
/// Empty collection, invalid weights, duplicate components, or nonfinite contributions.
/// A component absent from a model has zero contribution in that model.
pub fn aggregate_model_collection_ranks(
    per_model: &[(f64, ChangeAttributionResult)],
) -> Result<Vec<RootCauseRank>, AttributionError> {
    if per_model.is_empty() {
        return Err(AttributionError::invalid_input(
            "aggregate_model_collection_ranks requires ≥1 model result",
        ));
    }
    let weights = per_model.iter().map(|(weight, _)| *weight).collect::<Vec<_>>();
    let scale = weight_scale(&weights)?;
    let total: f64 = weights.iter().map(|weight| weight / scale).sum();
    let mut acc = std::collections::BTreeMap::<ComponentId, ContributionMoments>::new();
    for (weight, result) in per_model {
        let mut seen = std::collections::HashSet::new();
        for contribution in result.contributions.iter() {
            if !seen.insert(contribution.component) {
                return Err(AttributionError::invalid_input(
                    "duplicate component in model attribution",
                ));
            }
            acc.entry(contribution.component)
                .or_default()
                .add(*weight / scale, contribution.contribution)?;
        }
    }
    let mut ranks = Vec::with_capacity(acc.len());
    for (component, mut moments) in acc {
        // Retain the existing union-of-components convention: absent mechanisms
        // have zero contribution in that model, rather than silently changing
        // the conditioning population for each component.
        moments.add((total - moments.mass).max(0.0), 0.0)?;
        ranks.push(moments.rank(component));
    }
    ranks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ranks)
}

/// Process posterior contribution draws in bounded blocks into a summary ranking.
///
/// # Errors
///
/// Posterior shape errors.
pub fn posterior_contribution_ranks(
    draws: &PosteriorDraws,
    components: &[ComponentId],
    block_size: usize,
) -> Result<Vec<RootCauseRank>, AttributionError> {
    if components.is_empty() {
        return Err(AttributionError::invalid_input(
            "posterior_contribution_ranks requires components",
        ));
    }
    if block_size == 0 {
        return Err(AttributionError::Budget { message: "block_size must be ≥ 1".into() });
    }
    let n_draws = draws.n_draws;
    if n_draws == 0 || draws.n_quantities() < components.len() {
        return Err(AttributionError::invalid_input(
            "nonempty contribution draws must cover every component",
        ));
    }
    let mut moments =
        (0..components.len()).map(|_| ContributionMoments::default()).collect::<Vec<_>>();
    let mut start = 0usize;
    while start < n_draws {
        let len = block_size.min(n_draws - start);
        let batch = draws.batch(start, len)?;
        for (q, moment) in moments.iter_mut().enumerate() {
            for &value in batch.column(q)? {
                moment.add(1.0, value)?;
            }
        }
        start += len;
    }
    let mut ranks = moments
        .into_iter()
        .zip(components)
        .map(|(moment, &component)| moment.rank(component))
        .collect::<Vec<_>>();
    ranks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ranks)
}

fn rank_with_graph_uncertainty(
    attribution: &ChangeAttributionResult,
    gs: &WeightedGraphSamples,
    draws: &PosteriorDraws,
) -> Result<Vec<RootCauseRank>, AttributionError> {
    let n_comp = attribution.contributions.len();
    if draws.n_quantities() < n_comp {
        return Err(AttributionError::invalid_input(
            "contribution_draws columns < contribution count",
        ));
    }
    if gs.n_samples != draws.n_draws || gs.weights.len() != gs.n_samples {
        return Err(AttributionError::invalid_input(
            "one contribution draw is required per graph sample",
        ));
    }
    let scale = weight_scale(&gs.weights)?;
    let mut ranks = Vec::with_capacity(n_comp);
    for (i, component) in attribution.contributions.iter().enumerate() {
        let col = draws.column(i)?;
        let mut moments = ContributionMoments::default();
        for (&weight, &value) in gs.weights.iter().zip(col) {
            moments.add(weight / scale, value)?;
        }
        ranks.push(moments.rank(component.component));
    }
    ranks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(ranks)
}

/// Build a [`PosteriorDraws`] matrix (column-major) of contribution samples.
///
/// # Errors
///
/// Prob shape errors.
pub fn contribution_posterior_from_rows(
    n_components: usize,
    n_draws: usize,
    values_colmajor: &[f64],
) -> Result<PosteriorDraws, AttributionError> {
    use antecedent_prob::{PosteriorQuantityKind, PosteriorSchema};
    let quantities: Vec<_> = (0..n_components)
        .map(|i| PosteriorQuantityKind::Scalar { name: Arc::from(format!("contrib_{i}")) })
        .collect();
    let schema = PosteriorSchema { quantities: Arc::from(quantities) };
    Ok(PosteriorDraws::from_column_major(schema, n_draws, values_colmajor)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        posterior_draws_colmajor: Vec<f64>,
        n_draws: usize,
        expected_ranking: Vec<ExpectedRank>,
    }

    #[derive(Deserialize)]
    struct ExpectedRank {
        component_raw: u32,
        score: f64,
        graph_std: f64,
    }

    fn attribution(value: f64) -> ChangeAttributionResult {
        ChangeAttributionResult {
            outcome: antecedent_core::VariableId::from_raw(0),
            total_change: value,
            contributions: Arc::from([crate::result::ComponentContribution {
                component: ComponentId::from_raw(0),
                contribution: value,
                stderr: None,
                ci_low: None,
                ci_high: None,
            }]),
            interactions: Arc::from([]),
            path_breakdown: Arc::from([]),
            unidentified: Arc::from([]),
            graph_sensitivity: None,
            budget: crate::result::ComputeBudget::default(),
            monte_carlo_stderr: None,
            component_mc_stderr: None,
            cache_stats: crate::result::CacheStats::default(),
        }
    }

    #[test]
    fn graph_ranks_are_weight_scale_invariant_and_keep_small_variance() {
        for weight in [1e-30, 1.0, 1e300] {
            let ranks = aggregate_model_collection_ranks(&[
                (weight, attribution(1e12 - 1.0)),
                (weight, attribution(1e12 + 1.0)),
            ])
            .unwrap();
            assert!((ranks[0].score - 1e12).abs() < 1e-6);
            assert!((ranks[0].graph_std.unwrap() - 1.0).abs() < 1e-12);
        }
        let draws = contribution_posterior_from_rows(1, 2, &[1e12 - 1.0, 1e12 + 1.0]).unwrap();
        let ranks = posterior_contribution_ranks(&draws, &[ComponentId::from_raw(0)], 1).unwrap();
        assert!((ranks[0].graph_std.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn graph_ranks_require_aligned_samples_and_valid_mass() {
        use antecedent_prob::GraphIdentFlag;
        let graphs = WeightedGraphSamples::new(
            vec![0.5, 0.5],
            vec![GraphIdentFlag::Identified; 2],
            vec![1, 2],
        )
        .unwrap();
        let draws = contribution_posterior_from_rows(1, 1, &[2.0]).unwrap();
        assert!(
            root_cause_rank(
                &attribution(2.0),
                Some(&graphs),
                Some(&draws),
                &ExecutionContext::for_tests(1)
            )
            .is_err()
        );
        for weight in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
            assert!(aggregate_model_collection_ranks(&[(weight, attribution(2.0))]).is_err());
        }
    }

    #[test]
    fn posterior_ranking_matches_closed_form_moments() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../conformance/attribution/anomaly_root_cause/expected.json"
        ))
        .unwrap();
        let components = [ComponentId::from_raw(0), ComponentId::from_raw(1)];
        let draws = contribution_posterior_from_rows(
            components.len(),
            fixture.n_draws,
            &fixture.posterior_draws_colmajor,
        )
        .unwrap();
        for block_size in [1, 2, 3, 8] {
            let actual = posterior_contribution_ranks(&draws, &components, block_size).unwrap();
            for (got, expected) in actual.iter().zip(fixture.expected_ranking.iter()) {
                assert_eq!(got.component, ComponentId::from_raw(expected.component_raw));
                assert!((got.score - expected.score).abs() < 1e-12);
                assert!((got.graph_std.unwrap() - expected.graph_std).abs() < 1e-12);
            }
        }
    }
}
