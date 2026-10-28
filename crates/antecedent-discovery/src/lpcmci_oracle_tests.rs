//! LPCMCI under a population d-separation oracle on random stationary structural vector
//! autoregressions with one hidden process.
//!
//! The oracle answers every conditional-independence query exactly from the generating graph,
//! unrolled over ten time steps, so any disagreement is the algorithm's rather than sampling
//! error.  The d-separation check is the active-trail reachability (Bayes-ball) procedure; it
//! shares no code with the moralisation check used for the static FCI references.  The
//! reference quantities are computed here from definitions, not from LPCMCI's routines:
//!
//! * adjacency: `X_{t-tau}` and `Y_t` (or the contemporaneous pair `X_t`, `Y_t` with `X` before
//!   `Y`) are adjacent in the window MAG exactly when no subset of the other observed window
//!   nodes `{(V, l): l <= max_lag}` d-separates them, found by exhausting subsets;
//! * recorded separating sets: every one d-separates its pair in the unrolled graph;
//! * mark soundness: an arrowhead at `B` on an edge with `A` means `B` is not an ancestor of
//!   `A` in the generating graph (hidden nodes included), a tail means it is.
//!
//! Orientation completeness is not asserted; that is a separate claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{DenseNodeId, Endpoint, NodeRef};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence, PreparedCiTest,
    StatsError,
};

use crate::constraints::{DiscoveryConstraints, TemporalConstraints};
use crate::engine::DiscoveryWorkspace;
use crate::lpcmci::Lpcmci;

const MAX_LAG: u32 = 2;
/// Lags materialised in the lagged frame (`2 * MAX_LAG` plus the contemporaneous column).
const FRAME_LAGS: usize = 2 * MAX_LAG as usize + 1;
const STEPS: usize = 10;
const NOW: usize = STEPS - 1;

/// The generating graph over `m` processes (observed and hidden) unrolled over `STEPS` steps.
/// Node `(v, t)` is `t * m + v`; edges only run from lower to higher index.
struct Unrolled {
    m: usize,
    parents: Vec<Vec<usize>>,
    children: Vec<Vec<usize>>,
}

impl Unrolled {
    fn node(&self, v: usize, t: usize) -> usize {
        t * self.m + v
    }

    fn ancestors_or_self(&self, seeds: &[usize]) -> HashSet<usize> {
        let mut seen: HashSet<usize> = seeds.iter().copied().collect();
        let mut stack: Vec<usize> = seeds.to_vec();
        while let Some(v) = stack.pop() {
            for &p in &self.parents[v] {
                if seen.insert(p) {
                    stack.push(p);
                }
            }
        }
        seen
    }

    /// `a` is an ancestor of `b` (or `a == b`).
    fn is_ancestor(&self, a: usize, b: usize) -> bool {
        self.ancestors_or_self(&[b]).contains(&a)
    }

    /// d-separation by active-trail reachability from `x` given `z` (Bayes-ball).
    fn d_separated(&self, x: usize, y: usize, z: &[usize]) -> bool {
        let in_z: HashSet<usize> = z.iter().copied().collect();
        let anc_z = self.ancestors_or_self(z);
        // (node, arrived_from_child)
        let mut visited: HashSet<(usize, bool)> = HashSet::new();
        let mut stack = vec![(x, true)];
        while let Some((v, up)) = stack.pop() {
            if !visited.insert((v, up)) {
                continue;
            }
            if v == y && !in_z.contains(&v) {
                return false;
            }
            if up {
                if !in_z.contains(&v) {
                    stack.extend(self.parents[v].iter().map(|&p| (p, true)));
                    stack.extend(self.children[v].iter().map(|&c| (c, false)));
                }
            } else {
                if !in_z.contains(&v) {
                    stack.extend(self.children[v].iter().map(|&c| (c, false)));
                }
                if anc_z.contains(&v) {
                    stack.extend(self.parents[v].iter().map(|&p| (p, true)));
                }
            }
        }
        true
    }
}

struct TemporalDsepOracle {
    graph: Arc<Unrolled>,
    /// Process index of each observed variable slot.
    observed: Vec<usize>,
}

impl TemporalDsepOracle {
    fn node_of_column(&self, column: usize) -> usize {
        let slot = column / FRAME_LAGS;
        let lag = column % FRAME_LAGS;
        self.graph.node(self.observed[slot], NOW - lag)
    }
}

impl ConditionalIndependence for TemporalDsepOracle {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let results = request
            .queries
            .iter()
            .map(|q| {
                let cond = &request.z_flat[q.z_start..q.z_start + q.z_len];
                // A pair conditioned on one of its own endpoints is not a conditional
                // independence question; a real test would find a degenerate design.
                if cond.contains(&q.x) || cond.contains(&q.y) {
                    return Err(StatsError::Unsupported {
                        message: "conditioning set contains a tested endpoint",
                    });
                }
                let z: Vec<usize> = cond.iter().map(|&c| self.node_of_column(c)).collect();
                let separated = self.graph.d_separated(
                    self.node_of_column(q.x),
                    self.node_of_column(q.y),
                    &z,
                );
                Ok(CiResult {
                    statistic: if separated { 0.0 } else { 1.0 },
                    p_value: if separated { 1.0 } else { 0.0 },
                    df: 0.0,
                    ci: None,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CiBatchResult { results })
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn chance(&mut self, per_mille: u64) -> bool {
        self.next() % 1000 < per_mille
    }
}

/// A random stationary process graph over `m` processes with one hidden process, unrolled.
/// Every process has a lag-one self edge; other lag-one edges, lag-two edges and
/// contemporaneous edges (from lower to higher process index) are drawn independently.
/// Returns the unrolled graph and the observed process indices.
fn random_case(rng: &mut Lcg, m: usize) -> (Unrolled, Vec<usize>) {
    let mut lagged: Vec<(usize, usize, usize)> = Vec::new(); // (source, target, lag)
    let mut contemp: Vec<(usize, usize)> = Vec::new();
    for target in 0..m {
        lagged.push((target, target, 1));
        for source in 0..m {
            if source != target && rng.chance(250) {
                lagged.push((source, target, 1));
            }
            if rng.chance(100) {
                lagged.push((source, target, 2));
            }
            if source < target && rng.chance(300) {
                contemp.push((source, target));
            }
        }
    }
    let mut parents = vec![Vec::new(); m * STEPS];
    let mut children = vec![Vec::new(); m * STEPS];
    let mut add = |p: usize, c: usize| {
        if !parents[c].contains(&p) {
            parents[c].push(p);
            children[p].push(c);
        }
    };
    for t in 0..STEPS {
        for &(s, g) in &contemp {
            add(t * m + s, t * m + g);
        }
        for &(s, g, lag) in &lagged {
            if t >= lag {
                add((t - lag) * m + s, t * m + g);
            }
        }
    }
    let hidden = (rng.next() as usize) % m;
    let observed = (0..m).filter(|&v| v != hidden).collect();
    (Unrolled { m, parents, children }, observed)
}

fn subsets_of(items: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    (0..(1u32 << items.len())).map(move |mask| {
        items.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, &v)| v).collect()
    })
}

fn series(k: usize, n: usize) -> (TimeSeriesData, Vec<VariableId>) {
    let mut schema = CausalSchemaBuilder::new();
    for i in 0..k {
        schema
            .add_variable(
                format!("v{i}").as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = schema.build().unwrap();
    // Values are never read by the oracle; they only have to form a valid frame.
    let columns = (0..k)
        .map(|i| {
            let values: Vec<f64> = (0..n).map(|t| ((t * (i + 3)) % 7) as f64).collect();
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    (data, (0..k).map(|i| VariableId::from_raw(i as u32)).collect())
}

fn lagged_key(graph: &antecedent_graph::TemporalPag, id: DenseNodeId) -> (usize, u32) {
    match graph.nodes()[id.as_usize()] {
        NodeRef::Lagged { variable, lag } => (variable.raw() as usize, lag.raw()),
        _ => panic!("LPCMCI graphs hold lagged nodes only"),
    }
}

#[test]
fn lpcmci_under_a_dsep_oracle_recovers_window_adjacencies_with_sound_marks_and_sepsets() {
    let mut rng = Lcg(0x1CC0_0002);
    let (mut edges, mut absent, mut marks, mut sepsets_checked) = (0, 0, 0, 0);
    let mut failures: Vec<String> = Vec::new();
    for case in 0..60 {
        let m = 4;
        let (graph, observed) = random_case(&mut rng, m);
        let graph = Arc::new(graph);
        let k = observed.len();
        let (data, variables) = series(k, 40);
        let algorithm = Lpcmci::new()
            .with_fdr(false)
            .with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    min_lag: Lag::CONTEMPORANEOUS,
                    max_lag: Lag::from_raw(MAX_LAG),
                },
                alpha: 0.5,
                // Large enough that the cap never binds in a window of k * (MAX_LAG + 1) nodes.
                max_cond_size: 16,
                ..DiscoveryConstraints::default()
            })
            .with_ci(Arc::new(TemporalDsepOracle {
                graph: Arc::clone(&graph),
                observed: observed.clone(),
            }));
        let result = algorithm
            .run(
                &data,
                &variables,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(case as u64 + 1),
            )
            .unwrap();
        let pag = &result.review.graph;
        let node_at = |slot: usize, lag: u32| -> usize { graph.node(observed[slot], NOW - lag as usize) };
        let id_of = |slot: usize, lag: u32| -> DenseNodeId {
            let index = pag
                .nodes()
                .iter()
                .position(|n| {
                    matches!(n, NodeRef::Lagged { variable, lag: l }
                        if variable.raw() as usize == slot && l.raw() == lag)
                })
                .expect("window node");
            DenseNodeId::from_raw(index as u32)
        };

        // Window nodes as (slot, lag).
        let window: Vec<(usize, u32)> =
            (0..k).flat_map(|s| (0..=MAX_LAG).map(move |l| (s, l))).collect();
        for target in 0..k {
            for &(source, lag) in &window {
                if lag == 0 && source >= target {
                    continue;
                }
                let others: Vec<usize> = window
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| **w != (source, lag) && **w != (target, 0))
                    .map(|(i, _)| i)
                    .collect();
                let x = node_at(source, lag);
                let y = node_at(target, 0);
                let separable = subsets_of(&others).any(|subset| {
                    let z: Vec<usize> =
                        subset.iter().map(|&i| node_at(window[i].0, window[i].1)).collect();
                    graph.d_separated(x, y, &z)
                });
                let a = id_of(source, lag);
                let b = id_of(target, 0);
                let edge = pag.edge_between(a, b);
                if separable == edge.is_some() {
                    let key = (
                        VariableId::from_raw(source as u32),
                        Lag::from_raw(lag),
                        VariableId::from_raw(target as u32),
                        Lag::CONTEMPORANEOUS,
                    );
                    let sep = result.sepsets.get(&key).map(|s| {
                        s.iter().map(|(v, l)| (v.raw(), l.raw())).collect::<Vec<_>>()
                    });
                    failures.push(format!(
                        "case {case}: pair ({source},{lag})-({target},0) separable={separable} \
                         edge={} recorded sepset={sep:?}",
                        edge.is_some()
                    ));
                }
                if separable {
                    absent += 1;
                } else {
                    edges += 1;
                }
            }
        }

        // Soundness of every mark that is set.
        for a_raw in 0..pag.node_count() {
            for b_raw in a_raw + 1..pag.node_count() {
                let (a, b) = (DenseNodeId::from_raw(a_raw as u32), DenseNodeId::from_raw(b_raw as u32));
                let Some(edge) = pag.edge_between(a, b) else { continue };
                let (sa, la) = lagged_key(pag, edge.a);
                let (sb, lb) = lagged_key(pag, edge.b);
                let na = node_at(sa, la);
                let nb = node_at(sb, lb);
                for (mark, at, other) in [(edge.at_a, na, nb), (edge.at_b, nb, na)] {
                    match mark {
                        Endpoint::Arrow => {
                            marks += 1;
                            if graph.is_ancestor(at, other) {
                                failures.push(format!(
                                    "case {case}: arrowhead at an ancestor, edge ({sa},{la})-({sb},{lb})"
                                ));
                            }
                        }
                        Endpoint::Tail => {
                            marks += 1;
                            if !graph.is_ancestor(at, other) {
                                failures.push(format!(
                                    "case {case}: tail at a non-ancestor, edge ({sa},{la})-({sb},{lb})"
                                ));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Every recorded separating set separates its pair in the generating graph.
        for ((src, src_lag, tgt, tgt_lag), set) in &result.sepsets {
            let z: Vec<usize> = set
                .iter()
                .map(|(v, l)| node_at(v.raw() as usize, l.raw()))
                .collect();
            let x = node_at(src.raw() as usize, src_lag.raw());
            let y = node_at(tgt.raw() as usize, tgt_lag.raw());
            sepsets_checked += 1;
            if !graph.d_separated(x, y, &z) {
                failures.push(format!("case {case}: recorded sepset does not separate"));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures, first: {:#?}", failures.len(), &failures[..failures.len().min(12)]);
    // The comparison is not vacuous: both adjacent and separable pairs, directed marks and
    // recorded separating sets occur.
    assert!(edges > 200 && absent > 200, "edges {edges}, absent {absent}");
    assert!(marks > 200, "marks {marks}");
    assert!(sepsets_checked > 100, "sepsets {sepsets_checked}");
}

/// The reference itself: a chain `a -> c -> b` is blocked by `c`, a collider `a -> c <- b` is
/// opened by `c` or by a descendant of `c`, so a wrong oracle cannot make the comparison pass.
#[test]
fn bayes_ball_reference_matches_textbook_cases() {
    let build = |edges: &[(usize, usize)], n: usize| {
        let mut parents = vec![Vec::new(); n];
        let mut children = vec![Vec::new(); n];
        for &(p, c) in edges {
            parents[c].push(p);
            children[p].push(c);
        }
        Unrolled { m: n, parents, children }
    };
    let chain = build(&[(0, 1), (1, 2)], 3);
    assert!(!chain.d_separated(0, 2, &[]));
    assert!(chain.d_separated(0, 2, &[1]));
    let fork = build(&[(1, 0), (1, 2)], 3);
    assert!(!fork.d_separated(0, 2, &[]));
    assert!(fork.d_separated(0, 2, &[1]));
    let collider = build(&[(0, 1), (2, 1), (1, 3)], 4);
    assert!(collider.d_separated(0, 2, &[]));
    assert!(!collider.d_separated(0, 2, &[1]));
    assert!(!collider.d_separated(0, 2, &[3]));
    // M-graph: conditioning on the collider opens a path between the outer nodes.
    let m = build(&[(0, 2), (1, 2), (1, 3), (4, 0), (4, 3)], 5);
    assert!(m.d_separated(0, 3, &[4]));
    assert!(!m.d_separated(0, 3, &[]));
    assert!(!m.d_separated(0, 1, &[2]));
    assert!(m.d_separated(0, 1, &[]));
}
