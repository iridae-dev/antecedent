//! Shared PC skeleton search for the static algorithms ([`crate::Pc`], [`crate::Fci`],
//! [`crate::Rfci`]).
//!
//! Adjacency search removes an edge as soon as some conditioning set drawn from the current
//! neighbours of either endpoint separates it. All three algorithms run this identical
//! phase; they differ only in what they do with the skeleton afterwards.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_stats::{CiQuery, ConditionalIndependence};

use crate::ci::{ChunkSchedule, run_ci_batch};
use crate::combinations::Combinations;
use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pc::{adjacent_vars, edge_key, sorted_edge_pairs};
use crate::result::{DiscoveryIteration, EdgeEvidence, LaggedLink, PcSepsets, ScoredLink};

/// Undirected skeleton as one bit-row per variable.
///
/// Row `i` bit `j` is the edge between `variables[i]` and `variables[j]`. A
/// `HashSet<(u32, u32)>` of [`crate::pc::edge_key`] pairs has the same membership.
/// Degrees are updated on insert and remove, so the depth check does not rebuild
/// them by scanning every edge into a map. [`Self::edge_pairs`] is sorted by raw id,
/// which is the order the skeleton search already imposed before it mutated the graph.
pub(crate) struct UndirectedAdj {
    ids: Vec<u32>,
    pos: HashMap<u32, usize>,
    words: usize,
    rows: Vec<u64>,
    degree: Vec<usize>,
    n_edges: usize,
}

impl UndirectedAdj {
    /// Empty skeleton on `variables` (last duplicate id wins, as a `HashMap` index does).
    pub(crate) fn from_variables(variables: &[VariableId]) -> Self {
        let n = variables.len();
        let mut pos = HashMap::with_capacity(n);
        for (i, v) in variables.iter().enumerate() {
            pos.insert(v.raw(), i);
        }
        let words = n.div_ceil(64);
        Self {
            ids: variables.iter().map(|v| v.raw()).collect(),
            pos,
            words,
            rows: vec![0; n.saturating_mul(words)],
            degree: vec![0; n],
            n_edges: 0,
        }
    }

    fn positions(&self, lo: u32, hi: u32) -> Option<(usize, usize)> {
        let i = *self.pos.get(&lo)?;
        let j = *self.pos.get(&hi)?;
        (i != j).then_some((i, j))
    }

    fn mask(&self, i: usize, j: usize) -> (usize, u64) {
        (i * self.words + j / 64, 1u64 << (j % 64))
    }

    fn test(&self, i: usize, j: usize) -> bool {
        let (slot, bit) = self.mask(i, j);
        self.rows[slot] & bit != 0
    }

    fn set(&mut self, i: usize, j: usize) {
        let (slot, bit) = self.mask(i, j);
        self.rows[slot] |= bit;
    }

    fn clear_bit(&mut self, i: usize, j: usize) {
        let (slot, bit) = self.mask(i, j);
        self.rows[slot] &= !bit;
    }

    /// Insert the undirected edge. Returns whether it was absent.
    ///
    /// An id that is not one of the skeleton variables is ignored.
    pub(crate) fn insert(&mut self, edge: (u32, u32)) -> bool {
        let Some((i, j)) = self.positions(edge.0, edge.1) else {
            return false;
        };
        if self.test(i, j) {
            return false;
        }
        self.set(i, j);
        self.set(j, i);
        self.degree[i] += 1;
        self.degree[j] += 1;
        self.n_edges += 1;
        true
    }

    /// Remove the undirected edge. Returns whether it was present.
    pub(crate) fn remove(&mut self, edge: (u32, u32)) -> bool {
        let Some((i, j)) = self.positions(edge.0, edge.1) else {
            return false;
        };
        if !self.test(i, j) {
            return false;
        }
        self.clear_bit(i, j);
        self.clear_bit(j, i);
        self.degree[i] -= 1;
        self.degree[j] -= 1;
        self.n_edges -= 1;
        true
    }

    pub(crate) fn contains(&self, edge: (u32, u32)) -> bool {
        self.positions(edge.0, edge.1).is_some_and(|(i, j)| self.test(i, j))
    }

    pub(crate) fn len(&self) -> usize {
        self.n_edges
    }

    /// Keep edges for which `pred` is true.
    pub(crate) fn retain(&mut self, mut pred: impl FnMut(&(u32, u32)) -> bool) {
        for edge in self.edge_pairs() {
            if !pred(&edge) {
                self.remove(edge);
            }
        }
    }

    /// Undirected edges as `(lo, hi)` raw ids, sorted.
    pub(crate) fn edge_pairs(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.n_edges);
        let n = self.ids.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.test(i, j) {
                    let (a, b) = (self.ids[i], self.ids[j]);
                    out.push(if a <= b { (a, b) } else { (b, a) });
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Outcome of the adjacency search.
pub(crate) struct StaticSkeleton {
    /// Surviving undirected edges.
    pub adj: UndirectedAdj,
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
    pub ci: &'a (dyn ConditionalIndependence + Sync),
    pub constraints: &'a DiscoveryConstraints,
    pub cols: &'a [&'a [f64]],
    pub var_index: &'a HashMap<VariableId, usize>,
    pub variables: &'a [VariableId],
    /// Prefix of the per-depth iteration labels (`{label}.{depth}`).
    pub label: &'static str,
}

/// A static-graph CI test source with the workspace, context and level it is evaluated at,
/// counting the tests it runs.
pub(crate) struct StaticCiRun<'a> {
    pub input: &'a StaticSkeletonInput<'a>,
    pub workspace: &'a mut DiscoveryWorkspace,
    pub ctx: &'a ExecutionContext,
    /// Significance level: `p > alpha` accepts independence.
    pub alpha: f64,
    /// CI tests evaluated through [`Self::test`].
    pub tests: u64,
}

impl<'a> StaticCiRun<'a> {
    pub(crate) fn new(
        input: &'a StaticSkeletonInput<'a>,
        workspace: &'a mut DiscoveryWorkspace,
        ctx: &'a ExecutionContext,
        alpha: f64,
    ) -> Self {
        Self { input, workspace, ctx, alpha, tests: 0 }
    }

    /// One counted CI test `x ⊥ y | z`; returns the p-value.
    pub(crate) fn p_value(
        &mut self,
        x: VariableId,
        y: VariableId,
        z: &[VariableId],
    ) -> Result<f64, DiscoveryError> {
        let (_stat, p) = static_ci_test(self.input, x, y, z, self.workspace, self.ctx)?;
        self.tests += 1;
        Ok(p)
    }

    /// Smallest subset of `z` (dropping one element at a time, in order) that still
    /// separates `x` and `y`.
    pub(crate) fn minimize_sepset(
        &mut self,
        x: VariableId,
        y: VariableId,
        z: &[VariableId],
    ) -> Result<Vec<VariableId>, DiscoveryError> {
        let mut s = z.to_vec();
        let mut i = 0;
        while i < s.len() {
            let mut trial = s.clone();
            trial.remove(i);
            if self.p_value(x, y, &trial)? > self.alpha {
                s = trial;
            } else {
                i += 1;
            }
        }
        Ok(s)
    }
}

/// One CI test `x ⊥ y | z` on the prepared workspace test.
///
/// # Errors
///
/// Unknown variables, CI failures, or a non-finite statistic / p-value (a NaN input must not
/// pass as evidence either way).
pub(crate) fn static_ci_test(
    input: &StaticSkeletonInput<'_>,
    x: VariableId,
    y: VariableId,
    z: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<(f64, f64), DiscoveryError> {
    let StaticSkeletonInput { ci, constraints, cols, var_index, .. } = *input;
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
    let req =
        crate::ci::ci_batch_request(cols, &queries, &workspace.z_flat, constraints.significance);
    let result = run_ci_batch(ci, prepared, &req, &mut workspace.ci, ctx)
        .map_err(DiscoveryError::from)?
        .into_iter()
        .next()
        .ok_or_else(|| DiscoveryError::stats_msg("CI batch returned no results"))?;
    if !result.statistic.is_finite() || !result.p_value.is_finite() {
        return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
    }
    Ok((result.statistic, result.p_value))
}

/// `x ⊥ y | z` for every `z` of `zs` as one (possibly threaded) CI batch, or `None` when it
/// cannot be batched (a single test, an unknown variable, no prepared test, a failing batch).
/// `None` sends the caller down [`static_ci_test`], which reports a failure at the test that
/// causes it.
fn static_ci_batch(
    input: &StaticSkeletonInput<'_>,
    x: VariableId,
    y: VariableId,
    zs: &[Vec<VariableId>],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Option<Vec<antecedent_stats::CiResult>> {
    if zs.len() < 2 {
        return None;
    }
    let StaticSkeletonInput { ci, constraints, cols, var_index, .. } = *input;
    let xi = *var_index.get(&x)?;
    let yi = *var_index.get(&y)?;
    let mut queries = Vec::with_capacity(zs.len());
    let mut z_flat = Vec::new();
    for z in zs {
        let z_start = z_flat.len();
        for v in z {
            z_flat.push(*var_index.get(v)?);
        }
        queries.push(CiQuery { x: xi, y: yi, z_start, z_len: z.len() });
    }
    let req = crate::ci::ci_batch_request(cols, &queries, &z_flat, constraints.significance);
    let prepared = workspace.prepared_ci.as_ref()?;
    let results = run_ci_batch(ci, prepared, &req, &mut workspace.ci, ctx).ok()?;
    (results.len() == zs.len()).then_some(results)
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

/// Neighbours of `v` other than `other`, in ascending id order.
fn sorted_neighbours(
    v: VariableId,
    other: VariableId,
    adj: &UndirectedAdj,
    variables: &[VariableId],
) -> Vec<VariableId> {
    let mut out: Vec<VariableId> =
        adjacent_vars(v, adj, variables).into_iter().filter(|&u| u != other).collect();
    out.sort_unstable();
    out
}

/// Merge of two ascending streams without repeats into one ascending stream without repeats.
fn sorted_union<T: Ord>(
    a: impl Iterator<Item = T>,
    b: impl Iterator<Item = T>,
) -> impl Iterator<Item = T> {
    let (mut a, mut b) = (a.peekable(), b.peekable());
    std::iter::from_fn(move || match (a.peek(), b.peek()) {
        (None, None) => None,
        (Some(_), None) => a.next(),
        (None, Some(_)) => b.next(),
        (Some(l), Some(r)) => match l.cmp(r) {
            std::cmp::Ordering::Less => a.next(),
            std::cmp::Ordering::Greater => b.next(),
            std::cmp::Ordering::Equal => {
                b.next();
                a.next()
            }
        },
    })
}

/// Largest number of neighbours an endpoint of a remaining edge has besides the other endpoint.
///
/// That quantity is `max(degree) - 1`: a node of maximum degree is an endpoint of some
/// remaining edge, and the other endpoint of that edge is one of its neighbours.
fn max_excess_degree(adj: &UndirectedAdj) -> usize {
    adj.degree.iter().copied().max().unwrap_or(0).saturating_sub(1)
}

/// The edge `x — y` (`key`) and the conditioning-set size a scan of it runs at.
#[derive(Clone, Copy)]
struct EdgeScan {
    x: VariableId,
    y: VariableId,
    key: (u32, u32),
    depth: usize,
}

/// Separating set found for an edge (if any) and the statistic and p-value of the least
/// significant test it faced (max p).
type EdgeScanOutcome = (Option<Vec<VariableId>>, Option<(f64, f64)>);

/// Test `x ⊥ y` under the size-`depth` conditioning candidates drawn from the neighbours of `x`
/// and of `y`, in ascending order and without repeats, until a set separates them.
///
/// Upcoming candidates are evaluated in speculative batches (see [`ChunkSchedule`]); results
/// past the separating set are discarded, so what is counted and recorded is exactly what a
/// one-test-at-a-time scan would. Returns the separating set, if any, and the statistic and
/// p-value of the least significant test faced (max p).
fn scan_edge(
    input: &StaticSkeletonInput<'_>,
    edge: EdgeScan,
    skel: &mut StaticSkeleton,
    depth_tests: &mut u64,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<EdgeScanOutcome, DiscoveryError> {
    let EdgeScan { x, y, key, depth } = edge;
    let alpha = input.constraints.alpha;
    let nx = sorted_neighbours(x, y, &skel.adj, input.variables);
    let ny = sorted_neighbours(y, x, &skel.adj, input.variables);
    let mut cand_sets = sorted_union(Combinations::new(&nx, depth), Combinations::new(&ny, depth));
    let mut weakest: Option<(f64, f64)> = None;
    let mut schedule = ChunkSchedule::new(ctx);
    loop {
        let chunk: Vec<Vec<VariableId>> = cand_sets.by_ref().take(schedule.next_size()).collect();
        if chunk.is_empty() {
            return Ok((None, weakest));
        }
        let batch = static_ci_batch(input, x, y, &chunk, workspace, ctx);
        for (k, z) in chunk.into_iter().enumerate() {
            let (stat, p) = match &batch {
                Some(results) => {
                    let r = &results[k];
                    if !r.statistic.is_finite() || !r.p_value.is_finite() {
                        return Err(DiscoveryError::stats_msg(
                            "non-finite CI statistic or p-value",
                        ));
                    }
                    (r.statistic, r.p_value)
                }
                None => static_ci_test(input, x, y, &z, workspace, ctx)?,
            };
            skel.ci_tests += 1;
            *depth_tests += 1;
            skel.family_p.push(p);
            skel.family_edge.push(key);
            if p > alpha {
                return Ok((Some(z), weakest));
            }
            if weakest.is_none_or(|(_, wp)| p > wp) {
                weakest = Some((stat, p));
            }
        }
    }
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
pub(crate) fn run_static_skeleton(
    input: &StaticSkeletonInput<'_>,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<StaticSkeleton, DiscoveryError> {
    let StaticSkeletonInput { constraints, variables, label, .. } = *input;
    let max_cond = constraints.max_cond_size;

    let mut skel = StaticSkeleton {
        adj: UndirectedAdj::from_variables(variables),
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
            skel.adj.insert(edge_key(a, b));
        }
    }

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
            if !skel.adj.contains(key) {
                continue;
            }
            if constraints.static_required(x, y) {
                continue;
            }
            let (separating, weakest) = scan_edge(
                input,
                EdgeScan { x, y, key, depth },
                &mut skel,
                &mut depth_tests,
                workspace,
                ctx,
            )?;

            if let Some(z) = separating {
                skel.adj.remove(key);
                record_sepset(&mut skel.sepsets, x, y, &z);
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
        let max_deg = max_excess_degree(&skel.adj);
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
    let mut keys: Vec<(u32, u32)> = skel.adj.edge_pairs();
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

    /// Bit rows agree with a `HashSet` of `edge_key` pairs: membership, the sorted
    /// edge list, `retain`, and the excess degree the depth check used to recompute
    /// by scanning edges into a map. Ids are not `0..n`.
    #[test]
    fn bit_rows_match_hashset_edges_and_excess_degree() {
        use std::collections::{HashMap, HashSet};

        let ids = [4u32, 1, 9, 2, 7];
        let vars: Vec<_> = ids.into_iter().map(VariableId::from_raw).collect();
        let mut bits = UndirectedAdj::from_variables(&vars);
        let mut set = HashSet::new();
        for (a, b) in [(4u32, 1), (9, 2), (1, 7), (4, 9), (2, 7), (4, 2)] {
            let key = if a <= b { (a, b) } else { (b, a) };
            assert_eq!(bits.insert(key), set.insert(key));
        }
        assert!(!bits.insert((3, 4)), "an id outside the variable list is not an edge");
        assert_eq!(bits.remove((1, 4)), set.remove(&(1, 4)));
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = (ids[i], ids[j]);
                let key = if a <= b { (a, b) } else { (b, a) };
                assert_eq!(bits.contains(key), set.contains(&key), "{key:?}");
            }
        }
        let mut keys: Vec<_> = set.iter().copied().collect();
        keys.sort_unstable();
        assert_eq!(bits.edge_pairs(), keys);
        assert_eq!(bits.len(), set.len());

        let mut degree: HashMap<u32, usize> = HashMap::new();
        for &(lo, hi) in &set {
            *degree.entry(lo).or_default() += 1;
            *degree.entry(hi).or_default() += 1;
        }
        let rescanned = set
            .iter()
            .map(|&(lo, hi)| degree[&lo].max(degree[&hi]).saturating_sub(1))
            .max()
            .unwrap_or(0);
        assert_eq!(max_excess_degree(&bits), rescanned);

        bits.retain(|edge| edge.0 != 9 && edge.1 != 9);
        set.retain(|edge| edge.0 != 9 && edge.1 != 9);
        let mut keys: Vec<_> = set.into_iter().collect();
        keys.sort_unstable();
        assert_eq!(bits.edge_pairs(), keys);
        assert_eq!(max_excess_degree(&bits), {
            let mut degree: HashMap<u32, usize> = HashMap::new();
            for &(lo, hi) in &keys {
                *degree.entry(lo).or_default() += 1;
                *degree.entry(hi).or_default() += 1;
            }
            keys.iter()
                .map(|&(lo, hi)| degree[&lo].max(degree[&hi]).saturating_sub(1))
                .max()
                .unwrap_or(0)
        });
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
