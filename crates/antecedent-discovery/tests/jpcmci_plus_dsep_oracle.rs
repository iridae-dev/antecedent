//! J-PCMCI+ under a population d-separation oracle on random stationary structural vector
//! autoregressions observed in two environments.
//!
//! The generating model has `m` observed processes with lagged and contemporaneous edges, and an
//! unobserved environment indicator that shifts a random non-empty subset of the processes at
//! every step; the pooled search sees it as the space dummy, a time-invariant root with an edge
//! into the shifted processes. The oracle answers each conditional-independence query exactly
//! from the generating graph unrolled over ten steps with that root (active-trail reachability),
//! so any disagreement is the algorithm's. The reference quantities come from definitions:
//!
//! * every generating lagged edge, contemporaneous adjacency and dummy-to-process edge is
//!   retained (an oracle never separates d-connected variables), and the extra links the
//!   single-subset contemporaneous phase can leave in stay rare;
//! * every dummy link is directed out of the dummy, every lagged link points forward in time,
//!   and every oriented contemporaneous edge has the generating direction (a compelled edge of
//!   the equivalence class is the same in every member);
//! * every recorded separating set d-separates its pair in the generating graph, the dummy
//!   included.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage, SamplingRegularity,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, JpcmciPlus, TemporalConstraints,
};
use antecedent_graph::NodeRef;
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence, PreparedCiTest,
    StatsError,
};

const MAX_LAG: u32 = 2;
/// Lag slots per variable in the pooled frame (`2 * MAX_LAG` plus the contemporaneous column).
const FRAME_LAGS: usize = 2 * MAX_LAG as usize + 1;
const STEPS: usize = 10;
const NOW: usize = STEPS - 1;

/// The generating graph unrolled; node `(v, t)` is `t * m + v` and the dummy is the last node.
struct Unrolled {
    m: usize,
    /// Whether an observed space-context root precedes the dummy among the exogenous nodes.
    has_context: bool,
    parents: Vec<Vec<usize>>,
    children: Vec<Vec<usize>>,
}

impl Unrolled {
    fn node(&self, v: usize, t: usize) -> usize {
        t * self.m + v
    }

    fn context(&self) -> usize {
        self.m * STEPS
    }

    fn dummy(&self) -> usize {
        self.m * STEPS + usize::from(self.has_context)
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

struct PooledOracle {
    graph: Arc<Unrolled>,
}

impl PooledOracle {
    /// Columns are variable-major with `FRAME_LAGS` lag slots; the observed context (when there
    /// is one) follows the processes and the dummy follows the context.
    fn node_of_column(&self, column: usize) -> usize {
        let (slot, lag) = (column / FRAME_LAGS, column % FRAME_LAGS);
        if slot < self.graph.m {
            self.graph.node(slot, NOW - lag)
        } else if self.graph.has_context && slot == self.graph.m {
            self.graph.context()
        } else {
            self.graph.dummy()
        }
    }
}

impl ConditionalIndependence for PooledOracle {
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
                if cond.contains(&q.x) || cond.contains(&q.y) {
                    return Err(StatsError::Unsupported {
                        message: "conditioning set contains a tested endpoint",
                    });
                }
                let z: Vec<usize> = cond.iter().map(|&c| self.node_of_column(c)).collect();
                let separated =
                    self.graph.d_separated(self.node_of_column(q.x), self.node_of_column(q.y), &z);
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
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn chance(&mut self, per_mille: u64) -> bool {
        self.next() % 1000 < per_mille
    }
}

fn series(k: usize, n: usize, offset: usize) -> TimeSeriesData {
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
            let values: Vec<f64> = (0..n).map(|t| (((t + offset) * (i + 3)) % 7) as f64).collect();
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
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

type Lagged = Vec<(usize, usize, usize)>;

/// A random process graph over `m` processes with lag-one self edges, other lagged edges,
/// contemporaneous edges from lower to higher index and a non-empty set of dummy children.
type Case = (Unrolled, Lagged, Vec<(usize, usize)>, Vec<usize>, Vec<usize>);

fn random_case(rng: &mut Lcg, m: usize, has_context: bool) -> Case {
    let mut lagged: Lagged = Vec::new();
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
            if source < target && rng.chance(350) {
                contemp.push((source, target));
            }
        }
    }
    let mut shifted: Vec<usize> = (0..m).filter(|_| rng.chance(400)).collect();
    if shifted.is_empty() {
        shifted.push((rng.next() as usize) % m);
    }
    let mut context_children: Vec<usize> = Vec::new();
    if has_context {
        context_children = (0..m).filter(|_| rng.chance(400)).collect();
        if context_children.is_empty() {
            context_children.push((rng.next() as usize) % m);
        }
    }
    let total = m * STEPS + 1 + usize::from(has_context);
    let dummy_node = m * STEPS + usize::from(has_context);
    let mut parents = vec![Vec::new(); total];
    let mut children = vec![Vec::new(); total];
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
        for &v in &shifted {
            add(dummy_node, t * m + v);
        }
        for &v in &context_children {
            add(m * STEPS, t * m + v);
        }
    }
    lagged.sort_unstable();
    lagged.dedup();
    (Unrolled { m, has_context, parents, children }, lagged, contemp, shifted, context_children)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Key {
    Process(usize, u32),
    Context,
    Dummy,
}

fn key_of(
    nodes: &[NodeRef],
    id: antecedent_graph::DenseNodeId,
    m: usize,
    has_context: bool,
) -> Key {
    match nodes[id.as_usize()] {
        NodeRef::Lagged { variable, lag } => Key::Process(variable.raw() as usize, lag.raw()),
        NodeRef::Context { variable, .. } => {
            if has_context && variable.raw() as usize == m {
                Key::Context
            } else {
                assert!(variable.raw() as usize > m - 1, "a context node is never a process");
                Key::Dummy
            }
        }
        other => panic!("unexpected node {other:?}"),
    }
}

fn run_cases(with_context: bool, seed: u64, cases: u64, extra_per_mille: usize) {
    let mut rng = Lcg(seed);
    let (mut adjacent, mut exogenous_found, mut oriented, mut sepsets_checked, mut extra) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for case in 0..cases {
        let m = 3 + (case as usize % 2);
        let (graph, lagged, contemp, shifted, context_children) =
            random_case(&mut rng, m, with_context);
        let graph = Arc::new(graph);
        let columns = m + usize::from(with_context);
        let multi = MultiEnvironmentData::try_new(Arc::from([
            series(columns, 40, 0),
            series(columns, 40, 3),
        ]))
        .unwrap();
        let variables: Vec<VariableId> = (0..m as u32).map(VariableId::from_raw).collect();
        let mut constraints = DiscoveryConstraints {
            temporal: TemporalConstraints {
                min_lag: Lag::CONTEMPORANEOUS,
                max_lag: Lag::from_raw(MAX_LAG),
            },
            alpha: 0.5,
            max_cond_size: 16,
            ..DiscoveryConstraints::default()
        };
        if with_context {
            constraints.multi_dataset.context_variables =
                Arc::from([VariableId::from_raw(m as u32)]);
        }
        let result = JpcmciPlus::new()
            .with_fdr(false)
            .with_constraints(constraints)
            .with_ci(Arc::new(PooledOracle { graph: Arc::clone(&graph) }))
            .run(
                &multi,
                &variables,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(case + 1),
            )
            .unwrap_or_else(|e| panic!("case {case}: {e:?}"));
        let cpdag = &result.evidence.graph;
        let nodes = cpdag.nodes();
        let key = |id| key_of(nodes, id, m, with_context);

        let mut found_lagged: BTreeSet<(usize, usize, u32)> = BTreeSet::new();
        let mut found_contemp: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut found_dummy: BTreeSet<usize> = BTreeSet::new();
        let mut found_context: BTreeSet<usize> = BTreeSet::new();
        for e in cpdag.edges() {
            let (ka, kb) = (key(e.a), key(e.b));
            match (ka, kb) {
                (Key::Dummy | Key::Context, Key::Process(v, lag))
                | (Key::Process(v, lag), Key::Dummy | Key::Context) => {
                    assert_eq!(
                        lag, 0,
                        "case {case}: an exogenous node acts at the present step only"
                    );
                    let (p, _) = e.parent_child().unwrap_or_else(|| {
                        panic!("case {case}: exogenous link to process {v} is not directed")
                    });
                    match key(p) {
                        Key::Dummy => found_dummy.insert(v),
                        Key::Context => found_context.insert(v),
                        Key::Process(..) => panic!("case {case}: directed into an exogenous node"),
                    };
                }
                (Key::Process(sa, la), Key::Process(sb, lb)) if la == 0 && lb == 0 => {
                    found_contemp.insert((sa.min(sb), sa.max(sb)));
                    if let Some((p, c)) = e.parent_child() {
                        oriented += 1;
                        let (Key::Process(ps, _), Key::Process(cs, _)) = (key(p), key(c)) else {
                            unreachable!()
                        };
                        assert!(
                            ps < cs,
                            "case {case}: oriented {sa}-{sb} against its generating direction"
                        );
                    }
                }
                (Key::Process(..), Key::Process(..)) => {
                    let (p, c) = e.parent_child().unwrap_or_else(|| {
                        panic!("case {case}: lagged link {ka:?}-{kb:?} is not directed")
                    });
                    let (Key::Process(src, lag), Key::Process(tgt, tgt_lag)) = (key(p), key(c))
                    else {
                        unreachable!()
                    };
                    assert_eq!(tgt_lag, 0, "case {case}: lagged links point into the present");
                    found_lagged.insert((src, tgt, lag));
                }
                _ => panic!("case {case}: a link between two exogenous nodes"),
            }
        }
        let want_lagged: BTreeSet<(usize, usize, u32)> =
            lagged.iter().map(|&(s, t, l)| (s, t, l as u32)).collect();
        let want_contemp: BTreeSet<(usize, usize)> = contemp.iter().copied().collect();
        let want_dummy: BTreeSet<usize> = shifted.iter().copied().collect();
        let want_context: BTreeSet<usize> = context_children.iter().copied().collect();
        assert!(want_lagged.is_subset(&found_lagged), "case {case}: lagged link missing, m={m}");
        assert!(
            want_contemp.is_subset(&found_contemp),
            "case {case}: contemporaneous link missing"
        );
        assert!(
            want_dummy.is_subset(&found_dummy),
            "case {case}: dummy link missing {want_dummy:?} vs {found_dummy:?}"
        );
        assert!(
            want_context.is_subset(&found_context),
            "case {case}: context link missing {want_context:?} vs {found_context:?}"
        );
        extra += found_lagged.len() + found_contemp.len() + found_dummy.len() + found_context.len()
            - want_lagged.len()
            - want_contemp.len()
            - want_dummy.len()
            - want_context.len();
        adjacent += want_lagged.len() + want_contemp.len();
        exogenous_found += want_dummy.len() + want_context.len();

        let node_at = |slot: usize, lag: u32| graph.node(slot, NOW - lag as usize);
        let node_of = |v: VariableId, lag: Lag| {
            let raw = v.raw() as usize;
            if raw < m {
                node_at(raw, lag.raw())
            } else if with_context && raw == m {
                graph.context()
            } else {
                graph.dummy()
            }
        };
        for ((src, src_lag, tgt, tgt_lag), set) in &result.sepsets {
            let z: Vec<usize> = set.iter().map(|(v, l)| node_of(*v, *l)).collect();
            sepsets_checked += 1;
            assert!(
                graph.d_separated(node_of(*src, *src_lag), node_of(*tgt, *tgt_lag), &z),
                "case {case}: recorded separating set {set:?} does not separate ({src:?},{src_lag:?})-({tgt:?},{tgt_lag:?})"
            );
        }
    }
    assert!(
        adjacent > 200 && exogenous_found > 40 && sepsets_checked > 80,
        "{adjacent} {exogenous_found} {sepsets_checked}"
    );
    assert!(oriented > 20, "oriented contemporaneous edges {oriented}");
    assert!(
        extra * 1000 < extra_per_mille * (adjacent + exogenous_found),
        "extra links {extra} of {}",
        adjacent + exogenous_found
    );
}

#[test]
fn jpcmci_plus_under_a_dsep_oracle_recovers_dummy_links_and_sound_orientations() {
    run_cases(false, 0x70C0_0003, 60, 50);
}

#[test]
fn jpcmci_plus_under_a_dsep_oracle_recovers_context_and_dummy_links_together() {
    // Two exogenous roots leave more d-separable links in the single-subset context phase.
    run_cases(true, 0x70C0_0004, 60, 100);
}
