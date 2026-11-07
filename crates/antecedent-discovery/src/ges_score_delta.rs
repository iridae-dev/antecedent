// SPDX-License-Identifier: MIT OR Apache-2.0
//! Parent-set construction and score deltas for GES operators.

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_graph::{Cpdag, DenseNodeId};
use antecedent_state::{GraphScoreData, LocalScoreCache};

use super::{GesDeltaCache, local_score_or_degenerate, na_yx};
use crate::error::DiscoveryError;

/// The parent sets Insert(X, Y, T) compares: `Pa(Y) ∪ NA(Y,X) ∪ T` and the same with `{X}`
/// added, each sorted and without repeats.
///
/// The larger set is what the operator scores, so it is also what counts against
/// `max_parents`: counting only `Pa` and `T` would let neighbours in `NA(Y,X)` push a node past
/// the limit while the operator is still accepted.
pub(super) fn insert_parent_sets(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    na: &[DenseNodeId],
    t: &[DenseNodeId],
) -> (Vec<u32>, Vec<u32>) {
    let mut base = parents_u32(cpdag, y);
    for n in na.iter().chain(t.iter()) {
        base.push(n.raw());
    }
    base.sort_unstable();
    base.dedup();
    let mut neu = base.clone();
    neu.push(x.raw());
    neu.sort_unstable();
    neu.dedup();
    (base, neu)
}

fn parents_u32(cpdag: &Cpdag, y: DenseNodeId) -> Vec<u32> {
    let mut out: Vec<_> =
        cpdag.parents(y).into_iter().map(antecedent_graph::DenseNodeId::raw).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
pub(super) fn insert_delta(
    cpdag: &Cpdag,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    data: &GraphScoreData,
    x: DenseNodeId,
    y: DenseNodeId,
    t: &[DenseNodeId],
) -> Result<f64, DiscoveryError> {
    // Corollary 16: s(Y, NA∪T∪Pa∪{X}) − s(Y, NA∪T∪Pa)
    let (base, neu) = insert_parent_sets(cpdag, x, y, &na_yx(cpdag, y, x), t);
    cached_operator_delta(cache, deltas, data, y.raw(), &base, &neu)
}

pub(super) fn delete_delta(
    cpdag: &Cpdag,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    data: &GraphScoreData,
    x: DenseNodeId,
    y: DenseNodeId,
    h: &[DenseNodeId],
) -> Result<f64, DiscoveryError> {
    // Corollary 18: s(Y, (NA\H) ∪ (Pa\{X})) − s(Y, (NA\H) ∪ Pa)
    let na = na_yx(cpdag, y, x);
    let h_set: HashSet<DenseNodeId> = h.iter().copied().collect();
    let na_rest: Vec<u32> = na
        .into_iter()
        .filter(|n| !h_set.contains(n))
        .map(antecedent_graph::DenseNodeId::raw)
        .collect();
    let mut pa = parents_u32(cpdag, y);
    let mut with_x = pa.clone();
    with_x.push(x.raw());
    with_x.extend_from_slice(&na_rest);
    with_x.sort_unstable();
    with_x.dedup();
    pa.retain(|&p| p != x.raw());
    let mut without_x = pa;
    without_x.extend_from_slice(&na_rest);
    without_x.sort_unstable();
    without_x.dedup();
    cached_operator_delta(cache, deltas, data, y.raw(), &with_x, &without_x)
}

pub(super) fn cached_operator_delta(
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    data: &GraphScoreData,
    node: u32,
    old_parents: &[u32],
    new_parents: &[u32],
) -> Result<f64, DiscoveryError> {
    deltas.scratch.clear();
    deltas.scratch.push(node);
    deltas.scratch.push(crate::indexing::dense_u32(old_parents.len()));
    deltas.scratch.extend_from_slice(old_parents);
    deltas.scratch.extend_from_slice(new_parents);
    if let Some(delta) = deltas.entries.get(deltas.scratch.as_slice()).copied() {
        return Ok(delta);
    }
    let old = local_score_or_degenerate(cache, data, node, &Arc::from(old_parents))?;
    let new = local_score_or_degenerate(cache, data, node, &Arc::from(new_parents))?;
    let delta = operator_delta(old, new);
    deltas.entries.insert(deltas.scratch.as_slice().into(), delta);
    Ok(delta)
}

/// `new − old`, with a non-finite candidate scoring `−∞` (never chosen) and a non-finite
/// current score yielding `+∞` when the candidate is finite (leaving the degenerate set).
pub(super) fn operator_delta(old: f64, new: f64) -> f64 {
    if !new.is_finite() {
        f64::NEG_INFINITY
    } else if !old.is_finite() {
        f64::INFINITY
    } else {
        new - old
    }
}
