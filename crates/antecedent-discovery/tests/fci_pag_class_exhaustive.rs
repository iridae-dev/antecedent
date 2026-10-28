//! Static FCI under an m-separation oracle against the definition of a Markov equivalence class
//! of maximal ancestral graphs, exhaustively on three and four nodes.
//!
//! Reference, sharing no code with the product: enumerate every mixed graph whose pair states are
//! absent, `->`, `<-` or `<->`; keep those with no directed cycle, no almost-directed cycle and
//! every non-adjacent pair m-separated by some set (a simple-path oracle); group them by their
//! m-separation signature. The invariant PAG of a class keeps an endpoint mark exactly where all
//! members agree and puts a circle elsewhere. FCI, fed the signature as a perfect
//! conditional-independence oracle, must return that PAG (the orientation rules without selection
//! variables are complete), so every missed orientation, every unsound mark and every wrong
//! adjacency is a disagreement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{DiscoveryWorkspace, Fci, Rfci};
use antecedent_graph::{DenseNodeId, Endpoint};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, StatsError,
};

fn pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect()
}

/// Pair states: 0 absent, 1 `i -> j`, 2 `j -> i`, 3 `i <-> j`.
type Code = Vec<u8>;

struct Mixed {
    n: usize,
    arrow: Vec<Vec<bool>>,
    bi: Vec<Vec<bool>>,
    anc: Vec<Vec<bool>>,
}

impl Mixed {
    fn new(n: usize, code: &[u8]) -> Self {
        let mut arrow = vec![vec![false; n]; n];
        let mut bi = vec![vec![false; n]; n];
        for (&(i, j), &s) in pairs(n).iter().zip(code) {
            match s {
                1 => arrow[i][j] = true,
                2 => arrow[j][i] = true,
                3 => {
                    bi[i][j] = true;
                    bi[j][i] = true;
                }
                _ => {}
            }
        }
        let mut anc = arrow.clone();
        for (v, row) in anc.iter_mut().enumerate() {
            row[v] = true;
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if anc[i][k] && anc[k][j] {
                        anc[i][j] = true;
                    }
                }
            }
        }
        Self { n, arrow, bi, anc }
    }

    fn adjacent(&self, a: usize, b: usize) -> bool {
        self.arrow[a][b] || self.arrow[b][a] || self.bi[a][b]
    }

    fn ancestral(&self) -> bool {
        (0..self.n).all(|a| {
            (0..self.n).all(|b| {
                a == b
                    || !(self.arrow[a][b] && self.anc[b][a]) && !(self.bi[a][b] && self.anc[a][b])
            })
        })
    }

    fn head_at(&self, v: usize, w: usize) -> bool {
        self.arrow[w][v] || self.bi[v][w]
    }

    fn msep(&self, x: usize, y: usize, z: usize) -> bool {
        let inz: Vec<bool> = (0..self.n).map(|v| z >> v & 1 == 1).collect();
        let mut path = vec![x];
        !self.connects(&mut path, y, &inz)
    }

    fn connects(&self, path: &mut Vec<usize>, y: usize, z: &[bool]) -> bool {
        let cur = *path.last().unwrap();
        for next in 0..self.n {
            if next == cur || path.contains(&next) || !self.adjacent(cur, next) {
                continue;
            }
            if path.len() >= 2 {
                let prev = path[path.len() - 2];
                let open = if self.head_at(cur, prev) && self.head_at(cur, next) {
                    (0..self.n).any(|w| z[w] && self.anc[cur][w])
                } else {
                    !z[cur]
                };
                if !open {
                    continue;
                }
            }
            if next == y {
                return true;
            }
            path.push(next);
            let found = self.connects(path, y, z);
            path.pop();
            if found {
                return true;
            }
        }
        false
    }

    /// Verdict of every pair against every conditioning mask over the other nodes.
    fn table(&self) -> BTreeMap<(usize, usize, usize), bool> {
        let mut out = BTreeMap::new();
        for mask in 0..(1usize << self.n) {
            for (x, y) in pairs(self.n) {
                if mask >> x & 1 == 0 && mask >> y & 1 == 0 {
                    out.insert((x, y, mask), self.msep(x, y, mask));
                }
            }
        }
        out
    }

    fn maximal(&self) -> bool {
        pairs(self.n).into_iter().all(|(x, y)| {
            self.adjacent(x, y)
                || (0..(1usize << self.n))
                    .any(|m| m >> x & 1 == 0 && m >> y & 1 == 0 && self.msep(x, y, m))
        })
    }

    /// Endpoint of the edge `a`-`b` at `at` (`at` is `a` or `b`), for an adjacent pair.
    fn mark(&self, a: usize, b: usize, at: usize) -> Endpoint {
        let other = if at == a { b } else { a };
        if self.head_at(at, other) { Endpoint::Arrow } else { Endpoint::Tail }
    }
}

struct TableOracle {
    table: BTreeMap<(usize, usize, usize), bool>,
}

impl ConditionalIndependenceTest for TableOracle {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let results = request
            .queries
            .iter()
            .map(|q| {
                let mask = request.z_flat[q.z_start..q.z_start + q.z_len]
                    .iter()
                    .fold(0usize, |m, &c| m | (1 << c));
                let (x, y) = if q.x < q.y { (q.x, q.y) } else { (q.y, q.x) };
                let separated = self.table[&(x, y, mask)];
                CiResult {
                    statistic: if separated { 0.0 } else { 1.0 },
                    p_value: if separated { 1.0 } else { 0.0 },
                    df: 0.0,
                    ci: None,
                }
            })
            .collect();
        Ok(CiBatchResult { results })
    }
}

fn run_fci(
    n: usize,
    table: BTreeMap<(usize, usize, usize), bool>,
    restricted: bool,
) -> antecedent_graph::Pag {
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let column: Vec<f64> = (0..40).map(|r| f64::from(r) * 0.1).collect();
    let data = TabularData::from_f64_columns(names.iter().map(|s| (s.as_str(), column.as_slice())))
        .unwrap();
    let variables: Vec<VariableId> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
    let oracle = Arc::new(TableOracle { table });
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(3);
    if restricted {
        let mut rfci = Rfci::new().with_ci(oracle);
        rfci.constraints.max_cond_size = n;
        rfci.constraints.alpha = 0.5;
        rfci.run(&data, &variables, &mut ws, &ctx).unwrap().evidence.graph
    } else {
        let mut fci = Fci::new().with_fdr(false).with_ci(oracle);
        fci.constraints.max_cond_size = n;
        fci.constraints.alpha = 0.5;
        fci.run(&data, &variables, &mut ws, &ctx).unwrap().evidence.graph
    }
}

fn mark_in(pag: &antecedent_graph::Pag, a: usize, b: usize, at: usize) -> Endpoint {
    let edge = pag
        .edge_between(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32))
        .expect("edge");
    if edge.a.as_usize() == at { edge.at_a } else { edge.at_b }
}

fn classes(n: usize) -> Vec<Vec<Code>> {
    let k = pairs(n).len();
    let mut by_signature: BTreeMap<Vec<bool>, Vec<Code>> = BTreeMap::new();
    for mut c in 0..4usize.pow(u32::try_from(k).unwrap()) {
        let code: Code = (0..k)
            .map(|_| {
                let s = (c % 4) as u8;
                c /= 4;
                s
            })
            .collect();
        let m = Mixed::new(n, &code);
        if m.ancestral() && m.maximal() {
            by_signature.entry(m.table().into_values().collect()).or_default().push(code);
        }
    }
    by_signature.into_values().collect()
}

fn sweep(n: usize, expected_classes: Option<usize>, restricted: bool) -> (usize, usize) {
    let all = classes(n);
    if let Some(e) = expected_classes {
        assert_eq!(all.len(), e, "class count on {n} nodes");
    }
    let (mut circles, mut tails) = (0, 0);
    for members in &all {
        let graphs: Vec<Mixed> = members.iter().map(|c| Mixed::new(n, c)).collect();
        let pag = run_fci(n, graphs[0].table(), restricted);
        for (a, b) in pairs(n) {
            let adj: BTreeSet<bool> = graphs.iter().map(|g| g.adjacent(a, b)).collect();
            assert_eq!(adj.len(), 1, "skeleton is a class invariant");
            let adjacent = *adj.iter().next().unwrap();
            assert_eq!(
                pag.has_edge(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32)),
                adjacent,
                "adjacency {a}-{b} in class {members:?}"
            );
            if !adjacent {
                continue;
            }
            for at in [a, b] {
                let seen: BTreeSet<bool> =
                    graphs.iter().map(|g| g.mark(a, b, at) == Endpoint::Arrow).collect();
                let expected = if seen.len() > 1 {
                    circles += 1;
                    Endpoint::Circle
                } else if seen.contains(&true) {
                    Endpoint::Arrow
                } else {
                    tails += 1;
                    Endpoint::Tail
                };
                assert_eq!(
                    mark_in(&pag, a, b, at),
                    expected,
                    "mark at {at} on {a}-{b} in class {members:?}"
                );
            }
        }
    }
    (circles, tails)
}

#[test]
fn fci_returns_the_invariant_pag_of_every_three_node_class() {
    let (circles, _) = sweep(3, Some(11), false);
    assert!(circles > 0);
}

#[test]
fn fci_returns_the_invariant_pag_of_every_four_node_class() {
    let (circles, tails) = sweep(4, None, false);
    assert!(circles > 0 && tails > 0, "circles {circles}, tails {tails}");
}

#[test]
#[ignore = "about 3 minutes unoptimized (19 s with --release); run with --ignored"]
fn fci_returns_the_invariant_pag_of_every_five_node_class() {
    let (circles, tails) = sweep(5, None, false);
    assert!(circles > 0 && tails > 0, "circles {circles}, tails {tails}");
}

#[test]
fn rfci_returns_the_invariant_pag_of_every_three_and_four_node_class() {
    sweep(3, Some(11), true);
    sweep(4, None, true);
}
