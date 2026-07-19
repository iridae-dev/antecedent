//! Graph-sensitive root-cause ranking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ComponentId, ExecutionContext};
use causal_prob::{PosteriorDraws, WeightedGraphSamples};

use crate::error::AttributionError;
use crate::result::{ChangeAttributionResult, RootCauseRank};

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

/// Aggregate attribution results over a [`ModelCollection`] by weighted mean.
///
/// # Errors
///
/// Empty collection.
pub fn aggregate_model_collection_ranks(
    per_model: &[(f64, ChangeAttributionResult)],
) -> Result<Vec<RootCauseRank>, AttributionError> {
    if per_model.is_empty() {
        return Err(AttributionError::invalid_input(
            "aggregate_model_collection_ranks requires ≥1 model result",
        ));
    }
    let wsum: f64 = per_model.iter().map(|(w, _)| *w).sum::<f64>().max(1e-12);
    let mut acc: Vec<(ComponentId, f64, f64)> = Vec::new();
    for (w, res) in per_model {
        for c in res.contributions.iter() {
            if let Some(e) = acc.iter_mut().find(|(id, _, _)| *id == c.component) {
                e.1 += *w * c.contribution;
                e.2 += *w * c.contribution * c.contribution;
            } else {
                acc.push((c.component, *w * c.contribution, *w * c.contribution * c.contribution));
            }
        }
    }
    let mut ranks: Vec<RootCauseRank> = acc
        .into_iter()
        .map(|(component, s, s2)| {
            let mean = s / wsum;
            let var = (s2 / wsum - mean * mean).max(0.0);
            RootCauseRank { component, score: mean.abs(), graph_std: Some(var.sqrt()) }
        })
        .collect();
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
    let mut means = vec![0.0; components.len()];
    let mut m2 = vec![0.0; components.len()];
    let mut start = 0usize;
    while start < n_draws {
        let len = block_size.min(n_draws - start);
        let batch = draws.batch(start, len)?;
        for q in 0..components.len() {
            let col = batch.column(q)?;
            for &v in col {
                means[q] += v;
                m2[q] += v * v;
            }
        }
        start += len;
    }
    let n = n_draws.max(1) as f64;
    let mut ranks = Vec::with_capacity(components.len());
    for (i, &component) in components.iter().enumerate() {
        let mean = means[i] / n;
        let var = (m2[i] / n - mean * mean).max(0.0);
        ranks.push(RootCauseRank { component, score: mean.abs(), graph_std: Some(var.sqrt()) });
    }
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
    let wsum = gs.total_weight().max(1e-12);
    let mut ranks = Vec::with_capacity(n_comp);
    for (i, c) in attribution.contributions.iter().enumerate() {
        let col = draws.column(i)?;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let n = col.len().min(gs.n_samples);
        for s in 0..n {
            let w = gs.weights[s] / wsum;
            let v = col[s];
            mean += w * v;
            m2 += w * v * v;
        }
        let var = (m2 - mean * mean).max(0.0);
        ranks.push(RootCauseRank {
            component: c.component,
            score: mean.abs(),
            graph_std: Some(var.sqrt()),
        });
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
    use causal_prob::{PosteriorQuantityKind, PosteriorSchema};
    let quantities: Vec<_> = (0..n_components)
        .map(|i| PosteriorQuantityKind::Scalar { name: Arc::from(format!("contrib_{i}")) })
        .collect();
    let schema = PosteriorSchema { quantities: Arc::from(quantities) };
    Ok(PosteriorDraws::from_column_major(schema, n_draws, values_colmajor)?)
}
