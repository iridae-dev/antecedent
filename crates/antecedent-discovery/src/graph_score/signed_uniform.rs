//! Local factors for the signed-uniform Gaussian structural equation model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Score every admissible family. A numerical integration failure invalidates the
/// posterior, since dropping that family would change its normalization.
pub(crate) fn signed_uniform_local_table(
    data: &GraphScoreData,
    prior: &GraphPrior,
    variables: &[VariableId],
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(GraphPriorTable, Vec<Vec<f64>>), DiscoveryError> {
    let n = variables.len();
    if prior.constraints.max_parents.unwrap_or(n - 1).min(n - 1) > 3 {
        return Err(DiscoveryError::unsupported(
            "GaussianSignedUniform supports at most three parents per node; specify a parent cap",
        ));
    }
    let table = GraphPriorTable::new(n, prior, variables);
    let key = graph_score_cache_key(GraphScoreFamily::GaussianSignedUniform, n, data.n_rows);
    let mut cache = LocalScoreCache::new(key);
    let mut scores = vec![vec![f64::NEG_INFINITY; 1 << n]; n];
    for (node, row) in scores.iter_mut().enumerate() {
        for (subset, slot) in row.iter_mut().enumerate() {
            if subset & (1 << node) != 0 {
                continue;
            }
            let mask = AdjacencyMask::from_word(subset as u64);
            if table.node_log_prior(node, &mask).is_none() {
                continue;
            }
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let parents = mask.set_bits().collect::<Arc<[u32]>>();
            let score = cache.local_score(data, crate::indexing::dense_u32(node), &parents)?;
            if !score.is_finite() {
                return Err(DiscoveryError::stats_msg("non-finite GaussianSignedUniform score"));
            }
            *slot = score;
        }
    }
    Ok((table, scores))
}

impl DagScorer {
    pub(crate) fn signed_uniform(
        data: &GraphScoreData,
        prior: &GraphPrior,
        variables: &[VariableId],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, DiscoveryError> {
        let (prior, scores) = signed_uniform_local_table(data, prior, variables, ctx)?;
        let local = scores.into_iter().flatten().map(|s| s.is_finite().then_some(s)).collect();
        Ok(Self { n: variables.len(), prior, local })
    }
}
