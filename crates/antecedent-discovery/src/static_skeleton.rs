//! Shared PC skeleton search for the static algorithms ([`crate::Pc`], [`crate::Fci`],
//! [`crate::Rfci`]).
//!
//! Adjacency search removes an edge as soon as some conditioning set drawn from the current
//! neighbours of either endpoint separates it. All three algorithms run this identical
//! phase; they differ only in what they do with the skeleton afterwards.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments, clippy::zero_sized_map_values)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_stats::{CiBatchRequest, CiQuery, ConditionalIndependence, ConfidenceMethod};

use crate::combinations::for_each_combination_vars;
use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pc::{adjacent_vars, edge_key, sorted_edge_pairs};
use crate::result::{DiscoveryIteration, EdgeEvidence, LaggedLink, PcSepsets, ScoredLink};

/// Outcome of the adjacency search.
pub(crate) struct StaticSkeleton {
    /// Surviving undirected edges (`edge_key` order).
    pub adj: HashMap<(u32, u32), ()>,
    /// Weakest CI evidence per retained, tested edge. Required edges have no entry: no test
    /// was ever run on them.
    pub edge_scores: HashMap<(u32, u32), ScoredLink>,
    /// Separating sets of removed edges (both directions).
    pub sepsets: PcSepsets,
    /// CI tests run.
    pub ci_tests: u64,
    /// One entry per conditioning-set size.
    pub iterations: Vec<DiscoveryIteration>,
    /// Every CI p-value, aligned with `family_edge`.
    pub family_p: Vec<f64>,
    /// Edge each entry of `family_p` tested.
    pub family_edge: Vec<(u32, u32)>,
}

/// Inputs of [`run_static_skeleton`].
pub(crate) struct StaticSkeletonInput<'a> {
    pub ci: &'a dyn ConditionalIndependence,
    pub constraints: &'a DiscoveryConstraints,
    pub cols: &'a [&'a [f64]],
    pub var_index: &'a HashMap<VariableId, usize>,
    pub variables: &'a [VariableId],
    /// Prefix of the per-depth iteration labels (`{label}.{depth}`).
    pub label: &'static str,
}

/// One CI test `x ⊥ y | z` on the prepared workspace test.
///
/// # Errors
///
/// Unknown variables, CI failures, or a non-finite statistic / p-value (a NaN input must not
/// pass as evidence either way).
pub(crate) fn static_ci_test(
    ci: &dyn ConditionalIndependence,
    constraints: &DiscoveryConstraints,
    cols: &[&[f64]],
    var_index: &HashMap<VariableId, usize>,
    x: VariableId,
    y: VariableId,
    z: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<(f64, f64), DiscoveryError> {
    let xi = *var_index.get(&x).ok_or_else(|| DiscoveryError::data_msg("missing x"))?;
    let yi = *var_index.get(&y).ok_or_else(|| DiscoveryError::data_msg("missing y"))?;
    workspace.z_flat.clear();
    for &v in z {
        let zi = *var_index.get(&v).ok_or_else(|| DiscoveryError::data_msg("missing z"))?;
        workspace.z_flat.push(zi);
    }
    let prepared = workspace
        .prepared_ci
        .as_ref()
        .ok_or(DiscoveryError::Unsupported { message: "CI test used before prepare()" })?;
    let queries = [CiQuery { x: xi, y: yi, z_start: 0, z_len: workspace.z_flat.len() }];
    let req = CiBatchRequest {
        columns: cols,
        queries: &queries,
        z_flat: &workspace.z_flat,
        significance: constraints.significance,
        confidence: ConfidenceMethod::default(),
    };
    let out =
        ci.test_batch(prepared, &req, &mut workspace.ci, ctx).map_err(DiscoveryError::from)?;
    let result = out
        .results
        .into_iter()
        .next()
        .ok_or_else(|| DiscoveryError::stats_msg("CI batch returned no results"))?;
    if !result.statistic.is_finite() || !result.p_value.is_finite() {
        return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
    }
    Ok((result.statistic, result.p_value))
}

/// Record `sep` as the separating set of `{x, y}` in both directions.
pub(crate) fn record_sepset(
    sepsets: &mut PcSepsets,
    x: VariableId,
    y: VariableId,
    sep: &[VariableId],
) {
    let sep_lagged: Arc<[(VariableId, Lag)]> =
        Arc::from(sep.iter().map(|&v| (v, Lag::CONTEMPORANEOUS)).collect::<Vec<_>>());
    sepsets.insert((x, Lag::CONTEMPORANEOUS, y, Lag::CONTEMPORANEOUS), Arc::clone(&sep_lagged));
    sepsets.insert((y, Lag::CONTEMPORANEOUS, x, Lag::CONTEMPORANEOUS), sep_lagged);
}

/// PC adjacency search (order-dependent, deterministic traversal — see
/// [`sorted_edge_pairs`]).
///
/// The complete graph minus forbidden edges is thinned by conditioning-set size. A retained
/// edge is summarised by its *weakest* dependence evidence over every test it faced at every
/// depth (the largest p-value): an edge is only as well established as its least significant
/// separation attempt.
///
/// # Errors
///
/// CI failures.
#[allow(clippy::too_many_lines)] // one linear derivation; splitting it would scatter the argument
pub(crate) fn run_static_skeleton(
    input: &StaticSkeletonInput<'_>,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<StaticSkeleton, DiscoveryError> {
    let StaticSkeletonInput { ci, constraints, cols, var_index, variables, label } = *input;
    let alpha = constraints.alpha;
    let max_cond = constraints.max_cond_size;

    let mut skel = StaticSkeleton {
        adj: HashMap::new(),
        edge_scores: HashMap::new(),
        sepsets: PcSepsets::default(),
        ci_tests: 0,
        iterations: Vec::new(),
        family_p: Vec::new(),
        family_edge: Vec::new(),
    };

    for i in 0..variables.len() {
        for j in (i + 1)..variables.len() {
            let (a, b) = (variables[i], variables[j]);
            if constraints.static_forbidden(a, b) {
                continue;
            }
            skel.adj.insert(edge_key(a, b), ());
        }
    }

    let mut combo_scratch = Vec::new();
    let mut depth = 0usize;
    loop {
        let mut depth_tests = 0u64;
        // See `sorted_edge_pairs` doc: this loop mutates `adj` (edges removed below), so
        // traversal order must be deterministic for the skeleton to be reproducible.
        let edges = sorted_edge_pairs(&skel.adj);

        for &(x, y) in &edges {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let key = edge_key(x, y);
            if !skel.adj.contains_key(&key) {
                continue;
            }
            if constraints.static_required(x, y) {
                continue;
            }
            let nx: Vec<VariableId> =
                adjacent_vars(x, &skel.adj, variables).into_iter().filter(|&v| v != y).collect();
            let ny: Vec<VariableId> =
                adjacent_vars(y, &skel.adj, variables).into_iter().filter(|&v| v != x).collect();
            let mut cand_sets: Vec<Vec<VariableId>> = Vec::new();
            if nx.len() >= depth {
                for_each_combination_vars(&nx, depth, &mut combo_scratch, |c| {
                    cand_sets.push(c.to_vec());
                    true
                });
            }
            if ny.len() >= depth {
                for_each_combination_vars(&ny, depth, &mut combo_scratch, |c| {
                    cand_sets.push(c.to_vec());
                    true
                });
            }
            cand_sets.sort_unstable();
            cand_sets.dedup();

            let mut separating: Option<&Vec<VariableId>> = None;
            // Statistic / p-value of the least significant test this edge faced at this
            // depth (max p).
            let mut weakest: Option<(f64, f64)> = None;
            for z in &cand_sets {
                let (stat, p) =
                    static_ci_test(ci, constraints, cols, var_index, x, y, z, workspace, ctx)?;
                skel.ci_tests += 1;
                depth_tests += 1;
                skel.family_p.push(p);
                skel.family_edge.push(key);
                if p > alpha {
                    separating = Some(z);
                    break;
                }
                if weakest.is_none_or(|(_, wp)| p > wp) {
                    weakest = Some((stat, p));
                }
            }

            if let Some(z) = separating {
                skel.adj.remove(&key);
                record_sepset(&mut skel.sepsets, x, y, z);
            } else if let Some((statistic, p_value)) = weakest {
                let link = ScoredLink {
                    link: LaggedLink {
                        source: x,
                        source_lag: Lag::CONTEMPORANEOUS,
                        target: y,
                        target_lag: Lag::CONTEMPORANEOUS,
                    },
                    statistic,
                    p_value,
                    adjusted_p_value: None,
                };
                // Across depths keep the weakest evidence too, not the strongest.
                skel.edge_scores
                    .entry(key)
                    .and_modify(|s| {
                        if p_value > s.p_value {
                            *s = link;
                        }
                    })
                    .or_insert(link);
            }
        }

        skel.iterations.push(DiscoveryIteration {
            label: Arc::from(format!("{label}.{depth}")),
            ci_tests: depth_tests,
        });

        depth += 1;
        if depth > max_cond {
            break;
        }
        // Stop when no remaining edge has enough neighbours for larger conditioning sets.
        let mut degree: HashMap<u32, usize> = HashMap::new();
        for &(lo, hi) in skel.adj.keys() {
            *degree.entry(lo).or_default() += 1;
            *degree.entry(hi).or_default() += 1;
        }
        let max_deg = skel
            .adj
            .keys()
            .map(|&(lo, hi)| degree[&lo].max(degree[&hi]).saturating_sub(1))
            .max()
            .unwrap_or(0);
        if max_deg < depth {
            break;
        }
    }
    Ok(skel)
}

/// One [`ScoredLink`] per retained edge in deterministic order.
///
/// Edges the skeleton never tested (required by the caller's constraints) carry `NaN`
/// statistic and p-value: "not tested", never a fabricated significance.
pub(crate) fn skeleton_scored_links(skel: &StaticSkeleton) -> Vec<ScoredLink> {
    let mut keys: Vec<(u32, u32)> = skel.adj.keys().copied().collect();
    keys.sort_unstable();
    keys.into_iter()
        .map(|(lo, hi)| {
            skel.edge_scores.get(&(lo, hi)).copied().unwrap_or(ScoredLink {
                link: LaggedLink {
                    source: VariableId::from_raw(lo),
                    source_lag: Lag::CONTEMPORANEOUS,
                    target: VariableId::from_raw(hi),
                    target_lag: Lag::CONTEMPORANEOUS,
                },
                statistic: f64::NAN,
                p_value: f64::NAN,
                adjusted_p_value: None,
            })
        })
        .collect()
}

/// Per-edge evidence records for a static skeleton result.
///
/// A `NaN` statistic / p-value (an untested, constraint-required edge) becomes `None` with
/// provenance `"required"` rather than a fabricated number.
pub(crate) fn static_edge_evidence(
    scored: &[ScoredLink],
    sepsets: &PcSepsets,
    algorithm: &str,
) -> Vec<EdgeEvidence> {
    scored
        .iter()
        .map(|s| {
            let separating_set = sepsets
                .get(&(s.link.source, Lag::CONTEMPORANEOUS, s.link.target, Lag::CONTEMPORANEOUS))
                .cloned();
            let tested = s.p_value.is_finite() && s.statistic.is_finite();
            let mut provenance: Vec<Arc<str>> = vec![Arc::from(algorithm)];
            if !tested {
                provenance.push(Arc::from("required"));
            }
            EdgeEvidence {
                link: s.link,
                statistic: tested.then_some(s.statistic),
                p_value: tested.then_some(s.p_value),
                adjusted_p_value: s.adjusted_p_value,
                interval: None,
                separating_set,
                provenance: provenance.into(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(lo: u32, hi: u32, statistic: f64, p_value: f64) -> ScoredLink {
        ScoredLink {
            link: LaggedLink {
                source: VariableId::from_raw(lo),
                source_lag: Lag::CONTEMPORANEOUS,
                target: VariableId::from_raw(hi),
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic,
            p_value,
            adjusted_p_value: None,
        }
    }

    #[test]
    fn untested_edge_reports_no_evidence() {
        let scored = [link(0, 1, 0.4, 0.01), link(0, 2, f64::NAN, f64::NAN)];
        let ev = static_edge_evidence(&scored, &PcSepsets::default(), "pc");
        assert_eq!(ev[0].p_value, Some(0.01));
        assert_eq!(ev[0].statistic, Some(0.4));
        assert_eq!(ev[0].provenance.len(), 1);
        assert_eq!(ev[1].p_value, None);
        assert_eq!(ev[1].statistic, None);
        assert_eq!(&*ev[1].provenance[1], "required");
    }
}
