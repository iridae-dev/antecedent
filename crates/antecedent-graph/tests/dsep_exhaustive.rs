//! d-separation against the textbook path definition, exhaustively on small DAGs.
//!
//! The oracle below is written from the definition (Pearl 1988; Verma and Pearl 1990): a path
//! between `x` and `y` is active given `Z` when every non-collider on it lies outside `Z` and
//! every collider is in `Z` or has a descendant in `Z`; `x` and `y` are d-separated when no
//! simple path is active.  It enumerates simple paths over an adjacency matrix and takes
//! descendants from a Warshall closure, so it shares no traversal, bitset or ancestor code with
//! the product's Bayes-ball and moral-graph routines.
//!
//! Every edge subset of the fixed order `0 < 1 < ... < n-1` is a DAG, and every DAG is a
//! relabelling of one of these, so sweeping all graphs, all unordered pairs and all
//! conditioning sets (in both argument orders) is a complete check up to `n = 5`.  Six- and
//! seven-node graphs are checked on deterministic subsets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{DSeparationWorkspace, Dag, DenseNodeId, SeparationResult};

struct Oracle {
    n: usize,
    /// `arrow[a][b]`: edge a -> b.
    arrow: Vec<Vec<bool>>,
    /// `desc[a][b]`: a directed path a -> ... -> b with at least one edge.
    desc: Vec<Vec<bool>>,
}

impl Oracle {
    fn new(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut arrow = vec![vec![false; n]; n];
        for &(a, b) in edges {
            arrow[a][b] = true;
        }
        let mut desc = arrow.clone();
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if desc[i][k] && desc[k][j] {
                        desc[i][j] = true;
                    }
                }
            }
        }
        Self { n, arrow, desc }
    }

    fn separated(&self, x: usize, y: usize, z: &[bool]) -> bool {
        let mut visited = vec![false; self.n];
        visited[x] = true;
        !self.active_from(x, None, y, z, &mut visited)
    }

    /// Extend a simple path currently at `cur` (reached from `prev`) towards `target`.
    fn active_from(
        &self,
        cur: usize,
        prev: Option<usize>,
        target: usize,
        z: &[bool],
        visited: &mut [bool],
    ) -> bool {
        for next in 0..self.n {
            if visited[next] || !(self.arrow[cur][next] || self.arrow[next][cur]) {
                continue;
            }
            if let Some(p) = prev {
                let into_cur = self.arrow[p][cur];
                let into_cur_from_next = self.arrow[next][cur];
                let collider = into_cur && into_cur_from_next;
                let open = if collider {
                    z[cur] || (0..self.n).any(|d| z[d] && self.desc[cur][d])
                } else {
                    !z[cur]
                };
                if !open {
                    continue;
                }
            }
            if next == target {
                return true;
            }
            visited[next] = true;
            let found = self.active_from(next, Some(cur), target, z, visited);
            visited[next] = false;
            if found {
                return true;
            }
        }
        false
    }
}

fn build(n: usize, edges: &[(usize, usize)]) -> Dag {
    let mut g = Dag::with_variables(u32::try_from(n).unwrap());
    for &(a, b) in edges {
        g.insert_directed(id(a), id(b)).unwrap();
    }
    g
}

fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

fn edges_of(n: usize, mask: u64) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let mut bit = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if mask >> bit & 1 == 1 {
                edges.push((i, j));
            }
            bit += 1;
        }
    }
    edges
}

/// Check one graph over all unordered pairs and every Z disjoint from the pair.
/// Returns the number of queries checked.
fn check_graph(n: usize, edges: &[(usize, usize)], ws: &mut DSeparationWorkspace) -> usize {
    let g = build(n, edges);
    let oracle = Oracle::new(n, edges);
    let mut checked = 0;
    for x in 0..n {
        for y in (x + 1)..n {
            let others: Vec<usize> = (0..n).filter(|&v| v != x && v != y).collect();
            for sub in 0..(1u32 << others.len()) {
                let mut zmask = vec![false; n];
                let mut z = Vec::new();
                for (k, &v) in others.iter().enumerate() {
                    if sub >> k & 1 == 1 {
                        zmask[v] = true;
                        z.push(id(v));
                    }
                }
                let want = oracle.separated(x, y, &zmask);
                let got = g.is_d_separated(id(x), id(y), &z, ws).unwrap();
                let got_rev = g.is_d_separated(id(y), id(x), &z, ws).unwrap();
                assert_eq!(got, want, "x={x} y={y} z={z:?} edges={edges:?}");
                assert_eq!(got_rev, want, "reverse x={y} y={x} z={z:?} edges={edges:?}");
                // The witness route (moral ancestral graph) must reach the same verdict.
                let witness = g.d_separation(id(x), id(y), &z, ws).unwrap();
                assert_eq!(
                    matches!(witness, SeparationResult::Separated { .. }),
                    want,
                    "witness x={x} y={y} z={z:?} edges={edges:?}"
                );
                checked += 1;
            }
        }
    }
    checked
}

#[test]
fn every_dag_on_up_to_five_nodes_matches_the_path_definition() {
    let mut ws = DSeparationWorkspace::default();
    let mut total = 0;
    for n in 2..=5usize {
        let pairs = n * (n - 1) / 2;
        for mask in 0..(1u64 << pairs) {
            total += check_graph(n, &edges_of(n, mask), &mut ws);
        }
    }
    // 2: 2 graphs x 1 pair; 5: 1024 graphs x 10 pairs x 8 sets dominate.
    assert!(total > 80_000, "{total}");
}

#[test]
fn sampled_six_and_seven_node_dags_match_the_path_definition() {
    let mut ws = DSeparationWorkspace::default();
    // Deterministic scatter through the edge-subset space (odd multiplier mod 2^pairs).
    for (n, count) in [(6usize, 700u64), (7, 120)] {
        let pairs = n * (n - 1) / 2;
        let space = 1u64 << pairs;
        for k in 0..count {
            let mask = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 7) % space;
            check_graph(n, &edges_of(n, mask), &mut ws);
        }
    }
}

/// The documented policy for ill-posed queries: an endpoint in Z, or x equal to y, is "not
/// separated" in the boolean API, and the witness API refuses an endpoint in Z.
#[test]
fn ill_posed_queries_are_not_reported_as_separated() {
    let mut ws = DSeparationWorkspace::default();
    let g = build(3, &[(0, 1), (1, 2)]);
    assert!(!g.is_d_separated(id(0), id(0), &[], &mut ws).unwrap());
    assert!(!g.is_d_separated(id(0), id(2), &[id(0)], &mut ws).unwrap());
    assert!(!g.is_d_separated(id(0), id(2), &[id(2)], &mut ws).unwrap());
    assert!(g.d_separation(id(0), id(2), &[id(0)], &mut ws).is_err());
    // An isolated pair is separated; the chain is blocked only by its middle node.
    let isolated = build(2, &[]);
    assert!(isolated.is_d_separated(id(0), id(1), &[], &mut ws).unwrap());
    assert!(!g.is_d_separated(id(0), id(2), &[], &mut ws).unwrap());
    assert!(g.is_d_separated(id(0), id(2), &[id(1)], &mut ws).unwrap());
}
