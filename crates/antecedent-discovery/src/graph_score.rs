//! Shared DAG scoring for Bayesian discovery (exact + MCMC).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_data::TabularData;
use antecedent_state::{GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LocalScoreCache};

use crate::error::DiscoveryError;
use crate::graph_posterior::{GraphPrior, edge_required, log_prior_mask, parents_of, set_edge};
use crate::pc::collect_float_columns;

/// Build column-major [`GraphScoreData`] from tabular float columns.
pub(crate) fn tabular_score_data(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<GraphScoreData, DiscoveryError> {
    let col_owned = collect_float_columns(data, variables)?;
    let n_rows = col_owned[0].len();
    if n_rows < 2 {
        return Err(DiscoveryError::stats_msg("insufficient rows for graph score"));
    }
    for c in &col_owned {
        if c.len() != n_rows {
            return Err(DiscoveryError::data_msg("column length mismatch"));
        }
    }
    let n_vars = variables.len();
    let mut flat = Vec::with_capacity(n_vars.saturating_mul(n_rows));
    for c in &col_owned {
        flat.extend_from_slice(c.as_ref());
    }
    Ok(GraphScoreData::new(n_rows, n_vars, Arc::from(flat))?)
}

/// Score one DAG: BIC + log prior (`None` if invalid).
///
/// Uses [`LocalScoreCache::local_score`] only (does not mutate cached parent sets),
/// so MCMC proposals can reject without restoring graph state.
pub(crate) fn score_dag_mask(
    mask: u64,
    n: usize,
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
    prior: &GraphPrior,
    variables: &[VariableId],
) -> Option<f64> {
    let lp = log_prior_mask(mask, n, prior, variables)?;
    let mut total = lp;
    for node in 0..n {
        let pa = Arc::from(parents_of(mask, n, node));
        let s = cache.local_score(data, u32::try_from(node).ok()?, &pa).ok()?;
        if !s.is_finite() {
            return None;
        }
        total += s;
    }
    Some(total)
}

/// Start mask samplers inside the constrained support rather than hoping that
/// one-edge proposals can cross a region of zero prior probability.
pub(crate) fn initial_scored_mask(
    n: usize,
    data: &GraphScoreData,
    family: GraphScoreFamily,
    prior: &GraphPrior,
    variables: &[VariableId],
) -> Result<(u64, f64), DiscoveryError> {
    let mut mask = 0u64;
    for i in 0..n {
        for j in 0..n {
            if i != j && edge_required(&prior.constraints, variables, i, j) {
                mask = set_edge(mask, n, i, j, true);
            }
        }
    }
    let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
        data_version: 1,
        family,
        var_fingerprint: u64::try_from(n).unwrap_or(u64::MAX),
        penalty_fingerprint: u64::try_from(data.n_rows).unwrap_or(u64::MAX),
    });
    let score = score_dag_mask(mask, n, data, &mut cache, prior, variables)
        .filter(|s| s.is_finite())
        .ok_or_else(|| {
            DiscoveryError::unsupported(
                "required-edge graph is cyclic, violates constraints, or has no finite score",
            )
        })?;
    Ok((mask, score))
}
