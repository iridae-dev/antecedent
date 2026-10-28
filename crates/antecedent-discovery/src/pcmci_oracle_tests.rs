//! PCMCI and PCMCI+ under a population d-separation oracle on random stationary structural
//! vector autoregressions with every process observed.
//!
//! The oracle answers each conditional-independence query exactly from the generating graph
//! unrolled over ten steps (active-trail reachability), so any disagreement is the algorithm's.
//! The reference quantities come from definitions, not from the routines under test:
//!
//! * PCMCI (lagged parents only, no contemporaneous edges): the retained links are exactly the
//!   generating lagged edges `X_{t-tau} -> Y_t` with `tau <= max_lag`. PC1 never discards a true
//!   parent (a parent is d-connected to its child given anything), and MCI conditions on a
//!   superset of the child's parents drawn from its non-descendants, which d-separates every
//!   non-parent by weak union;
//! * PCMCI+ (contemporaneous edges from lower to higher process index): the lagged and
//!   contemporaneous adjacencies are exactly the generating edges, lagged links are oriented
//!   forward in time, every oriented contemporaneous edge has the generating direction (a
//!   compelled edge of the equivalence class is the same in every member), and every recorded
//!   separating set d-separates its pair in the generating graph. Orientation completeness is
//!   not asserted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{DenseNodeId, NodeRef};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence, PreparedCiTest,
    StatsError,
};

use crate::constraints::{DiscoveryConstraints, TemporalConstraints};
use crate::engine::DiscoveryWorkspace;
use crate::pcmci::Pcmci;
use crate::pcmci_plus::PcmciPlus;

const MAX_LAG: u32 = 2;
/// Lags materialised in the lagged frame (`2 * MAX_LAG` plus the contemporaneous column).
const FRAME_LAGS: usize = 2 * MAX_LAG as usize + 1;
const STEPS: usize = 10;
const NOW: usize = STEPS - 1;

/// The generating graph over `m` processes unrolled over `STEPS` steps.
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


/// A random stationary process graph over `m` observed processes, unrolled. Every process has a
/// lag-one self edge; other lag-one and lag-two edges are drawn independently, and with
/// `contemporaneous` also edges from lower to higher process index. Returns the unrolled graph,
/// the lagged edges `(source, target, lag)` and the contemporaneous edges.
type Case = (Unrolled, Vec<(usize, usize, usize)>, Vec<(usize, usize)>);

fn random_case(rng: &mut Lcg, m: usize, contemporaneous: bool) -> Case {
    let mut lagged: Vec<(usize, usize, usize)> = Vec::new();
    let mut contemp: Vec<(usize, usize)> = Vec::new();
    for target in 0..m {
        lagged.push((target, target, 1));
        for source in 0..m {
            if source != target && rng.chance(300) {
                lagged.push((source, target, 1));
            }
            if rng.chance(150) {
                lagged.push((source, target, 2));
            }
            if contemporaneous && source < target && rng.chance(350) {
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
    lagged.sort_unstable();
    lagged.dedup();
    (Unrolled { m, parents, children }, lagged, contemp)
}

fn key_of(nodes: &[NodeRef], id: DenseNodeId) -> (usize, u32) {
    match nodes[id.as_usize()] {
        NodeRef::Lagged { variable, lag } => (variable.raw() as usize, lag.raw()),
        _ => panic!("lagged nodes only"),
    }
}

fn constraints(min_lag: Lag) -> DiscoveryConstraints {
    DiscoveryConstraints {
        temporal: TemporalConstraints { min_lag, max_lag: Lag::from_raw(MAX_LAG) },
        alpha: 0.5,
        max_cond_size: 16,
        ..DiscoveryConstraints::default()
    }
}

#[test]
fn pcmci_under_a_dsep_oracle_returns_exactly_the_lagged_parents() {
    let mut rng = Lcg(0x70C0_0001);
    let (mut present, mut absent) = (0, 0);
    for case in 0..80u64 {
        let m = 3 + (case as usize % 3);
        let (graph, lagged, _) = random_case(&mut rng, m, false);
        let graph = Arc::new(graph);
        let (data, variables) = series(m, 40);
        let result = Pcmci::new()
            .with_fdr(false)
            .with_constraints(constraints(Lag::from_raw(1)))
            .with_ci(Arc::new(TemporalDsepOracle {
                graph: Arc::clone(&graph),
                observed: (0..m).collect(),
            }))
            .run(
                &data,
                &variables,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(case + 1),
            )
            .unwrap();
        let nodes = result.evidence.graph.nodes();
        let found: BTreeSet<(usize, usize, u32)> = result
            .evidence
            .graph
            .edges()
            .map(|e| {
                let (p, c) = e.parent_child().expect("PCMCI links are directed");
                let (src, lag) = key_of(nodes, p);
                let (tgt, tgt_lag) = key_of(nodes, c);
                assert_eq!(tgt_lag, 0, "case {case}: links end at the present");
                (src, tgt, lag)
            })
            .collect();
        let want: BTreeSet<(usize, usize, u32)> =
            lagged.iter().map(|&(s, t, l)| (s, t, l as u32)).collect();
        assert_eq!(found, want, "case {case}: m={m}");
        present += want.len();
        absent += m * m * MAX_LAG as usize - want.len();
    }
    assert!(present > 300 && absent > 300, "present {present}, absent {absent}");
}

#[test]
fn pcmci_plus_under_a_dsep_oracle_recovers_adjacencies_and_sound_orientations() {
    let mut rng = Lcg(0x70C0_0002);
    let (mut adjacent, mut oriented, mut sepsets_checked, mut undirected, mut extra_links) =
        (0, 0, 0, 0, 0usize);
    for case in 0..80u64 {
        let m = 3 + (case as usize % 3);
        let (graph, lagged, contemp) = random_case(&mut rng, m, true);
        let graph = Arc::new(graph);
        let (data, variables) = series(m, 40);
        let result = PcmciPlus::new()
            .with_fdr(false)
            .with_constraints(constraints(Lag::CONTEMPORANEOUS))
            .with_ci(Arc::new(TemporalDsepOracle {
                graph: Arc::clone(&graph),
                observed: (0..m).collect(),
            }))
            .run(
                &data,
                &variables,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(case + 1),
            )
            .unwrap();
        let cpdag = &result.evidence.graph;
        let nodes = cpdag.nodes();
        let mut found_lagged: BTreeSet<(usize, usize, u32)> = BTreeSet::new();
        let mut found_contemp: BTreeSet<(usize, usize)> = BTreeSet::new();
        for e in cpdag.edges() {
            let (sa, la) = key_of(nodes, e.a);
            let (sb, lb) = key_of(nodes, e.b);
            if la == 0 && lb == 0 {
                found_contemp.insert((sa.min(sb), sa.max(sb)));
                match e.parent_child() {
                    Some((p, c)) => {
                        oriented += 1;
                        // The generating contemporaneous edges run from lower to higher index.
                        assert!(
                            key_of(nodes, p).0 < key_of(nodes, c).0,
                            "case {case}: oriented {sa}-{sb} against its generating direction"
                        );
                    }
                    None => undirected += 1,
                }
            } else {
                let (p, c) = e.parent_child().unwrap_or_else(|| {
                    panic!("case {case}: lagged link ({sa},{la})-({sb},{lb}) is not directed")
                });
                let (src, lag) = key_of(nodes, p);
                let (tgt, tgt_lag) = key_of(nodes, c);
                assert_eq!(tgt_lag, 0, "case {case}: lagged links point into the present");
                found_lagged.insert((src, tgt, lag));
            }
        }
        let want_lagged: BTreeSet<(usize, usize, u32)> =
            lagged.iter().map(|&(s, t, l)| (s, t, l as u32)).collect();
        // Every generating link is found: an oracle never separates d-connected variables. The
        // converse can fail, because the contemporaneous phase tests one conditioning set per
        // size (the strongest neighbours), not every subset; see the counterexample below.
        assert!(want_lagged.is_subset(&found_lagged), "case {case}: lagged link missing, m={m}");
        let want_contemp: BTreeSet<(usize, usize)> = contemp.iter().copied().collect();
        assert!(want_contemp.is_subset(&found_contemp), "case {case}: contemporaneous link missing");
        extra_links += found_lagged.len() + found_contemp.len() - want_lagged.len() - want_contemp.len();
        adjacent += want_lagged.len() + want_contemp.len();

        let node_at = |slot: usize, lag: u32| graph.node(slot, NOW - lag as usize);
        for ((src, src_lag, tgt, tgt_lag), set) in &result.sepsets {
            let z: Vec<usize> =
                set.iter().map(|(v, l)| node_at(v.raw() as usize, l.raw())).collect();
            sepsets_checked += 1;
            assert!(
                graph.d_separated(
                    node_at(src.raw() as usize, src_lag.raw()),
                    node_at(tgt.raw() as usize, tgt_lag.raw()),
                    &z
                ),
                "case {case}: recorded separating set {set:?} does not separate ({src:?},{src_lag:?})-({tgt:?},{tgt_lag:?})"
            );
        }
    }
    assert!(adjacent > 400 && sepsets_checked > 100, "adjacent {adjacent}, sepsets {sepsets_checked}");
    assert!(oriented > 50, "oriented contemporaneous edges {oriented} (undirected {undirected})");
    // The single-subset search leaves a few d-separable links in; they must stay rare.
    assert!(extra_links * 20 < adjacent, "extra links {extra_links} of {adjacent}");
}
