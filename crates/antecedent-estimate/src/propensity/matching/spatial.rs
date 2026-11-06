//! Exact spatial matching query passes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    BoxTreeIndex, EstimationError, ExecutionContext, MATCH_CANCELLATION_CHUNK, MatchingIndex,
    StatsError,
};

type NearestQuery<'a> =
    dyn Fn(&[f64], usize, &mut Vec<(usize, f64)>) -> Result<(), StatsError> + 'a;

/// [`match_nearest_k`] and the single-neighbour pass answered by a [`BoxTreeIndex`], with the
/// same output contract.
/// Single-neighbour pass over a multivariate index that keeps every donor tied at the nearest
/// distance, as the scalar pass does: a query with several equidistant donors (duplicated
/// covariate rows, a discrete covariate) matches the whole tied set, whose outcomes are averaged
/// with equal weight, rather than the donor listed first. The estimate then does not depend on
/// the order of the rows.
///
/// Returns the tied runs (`(start, len)` into the pool, `len == 0` for an unmatched query) and the
/// pool. A tied set shared by several queries is stored once. `donor_rows[q]` holds the first
/// tied donor and `distances[q]` the shared distance, `usize::MAX` and infinity when the nearest
/// distance is non-finite or past the caliper.
#[expect(
    clippy::too_many_arguments,
    reason = "the search closure, the queries, the caliper and the output buffers are independent"
)]
#[allow(
    clippy::type_complexity,
    reason = "the type preserves the estimator output and resource structure"
)]
pub(super) fn match_nearest_tied(
    nearest_k: &NearestQuery<'_>,
    query_features: &[f64],
    dim: usize,
    n_donors: usize,
    caliper: Option<f64>,
    donor_rows: &mut [usize],
    distances: &mut [f64],
    ctx: &ExecutionContext,
) -> Result<(Vec<(usize, usize)>, Vec<usize>), EstimationError> {
    let mut found = Vec::new();
    let mut runs = vec![(0_usize, 0_usize); donor_rows.len()];
    let mut pool: Vec<usize> = Vec::new();
    let mut shared = std::collections::HashMap::<Vec<usize>, usize>::new();
    for (q, query) in query_features.chunks_exact(dim).enumerate() {
        if q % MATCH_CANCELLATION_CHUNK == 0 && ctx.cancellation.is_cancelled() {
            return Err(EstimationError::from(StatsError::Cancelled));
        }
        nearest_k(query, 2.min(n_donors), &mut found)?;
        let nearest = found.first().map_or(f64::INFINITY, |&(_, d)| d);
        let within = nearest.is_finite() && caliper.is_none_or(|limit| nearest <= limit);
        if !within {
            donor_rows[q] = usize::MAX;
            distances[q] = f64::INFINITY;
            continue;
        }
        // Widen the neighbourhood until it holds a donor farther than the nearest, or all of them.
        let mut k = found.len();
        while k < n_donors
            && found.len() == k
            && found.last().is_some_and(|&(_, d)| d.total_cmp(&nearest).is_eq())
        {
            k = (k * 2).min(n_donors);
            nearest_k(query, k, &mut found)?;
        }
        let tied: Vec<usize> = found
            .iter()
            .take_while(|&&(_, d)| d.total_cmp(&nearest).is_eq())
            .map(|&(row, _)| row)
            .collect();
        donor_rows[q] = tied[0];
        distances[q] = nearest;
        let start = if tied.len() == 1 {
            pool.push(tied[0]);
            pool.len() - 1
        } else {
            *shared.entry(tied.clone()).or_insert_with(|| {
                let start = pool.len();
                pool.extend_from_slice(&tied);
                start
            })
        };
        runs[q] = (start, tied.len());
    }
    Ok((runs, pool))
}

#[expect(clippy::too_many_arguments, reason = "mirrors the split-plane passes it replaces")]
pub(super) fn match_with_box_tree(
    index: &BoxTreeIndex,
    query_features: &[f64],
    n_queries: usize,
    matches: usize,
    caliper: Option<f64>,
    donor_rows: &mut [usize],
    distances: &mut [f64],
    ctx: &ExecutionContext,
) -> Result<(), EstimationError> {
    if matches == 1 {
        index
            .match_all_within(query_features, n_queries, caliper, donor_rows, distances, ctx)
            .map_err(crate::error::EstimationError::from)?;
        return Ok(());
    }
    match_nearest_k_with(
        index.dim(),
        &|query, k, out| index.nearest_k(query, k, out),
        query_features,
        matches,
        caliper,
        donor_rows,
        distances,
        ctx,
    )
}

/// Fill `donor_rows` (`matches` slots per query) with each query's `matches` nearest donors.
/// A query whose farthest match is beyond the caliper, or at a non-finite distance, keeps
/// `usize::MAX` in its first slot and `distances[q] = ∞`; otherwise `distances[q]` is the
/// farthest of its matches.
pub(super) fn match_nearest_k(
    index: &MatchingIndex,
    query_features: &[f64],
    matches: usize,
    caliper: Option<f64>,
    donor_rows: &mut [usize],
    distances: &mut [f64],
    ctx: &ExecutionContext,
) -> Result<(), EstimationError> {
    match_nearest_k_with(
        index.dim,
        &|query, k, out| index.nearest_k(query, k, out),
        query_features,
        matches,
        caliper,
        donor_rows,
        distances,
        ctx,
    )
}

#[expect(clippy::too_many_arguments, reason = "the index query and output buffers are independent")]
fn match_nearest_k_with(
    dim: usize,
    nearest_k: &NearestQuery<'_>,
    query_features: &[f64],
    matches: usize,
    caliper: Option<f64>,
    donor_rows: &mut [usize],
    distances: &mut [f64],
    ctx: &ExecutionContext,
) -> Result<(), EstimationError> {
    let mut found = Vec::with_capacity(matches);
    for (q, query) in query_features.chunks_exact(dim).enumerate() {
        if q % MATCH_CANCELLATION_CHUNK == 0 && ctx.cancellation.is_cancelled() {
            return Err(EstimationError::from(StatsError::Cancelled));
        }
        nearest_k(query, matches, &mut found).map_err(crate::error::EstimationError::from)?;
        let farthest = found.last().map_or(f64::INFINITY, |&(_, d)| d);
        let within = found.len() == matches
            && farthest.is_finite()
            && caliper.is_none_or(|limit| farthest <= limit);
        if within {
            for (slot, &(row, _)) in
                donor_rows[q * matches..(q + 1) * matches].iter_mut().zip(&found)
            {
                *slot = row;
            }
            distances[q] = farthest;
        } else {
            donor_rows[q * matches] = usize::MAX;
            distances[q] = f64::INFINITY;
        }
    }
    Ok(())
}
