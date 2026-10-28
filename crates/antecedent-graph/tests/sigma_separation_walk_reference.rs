//! σ-separation against the walk definition, by a search over walks with no length bound.
//!
//! Reference: Definition A.16 of Bongers, Forré, Peters and Mooij restated directly. A walk from
//! `x` to `y` is σ-open given `C` when `x` and `y` lie outside `C`, every collider (two arrowheads
//! at the node) lies in `An(C)`, and no non-collider in `C` has an adjacent edge `v -> w` with `w`
//! outside the strongly connected component of `v`. The search runs over (node, incoming edge)
//! states, so it allows repeated nodes and shares neither the acyclification nor the simple-path
//! enumeration of the product. The product has a four-node ceiling on its own walk oracle; the
//! walk search here has none, so it also checks five-node mixed graphs on a seeded sample.
//!
//! Every directed-and-bidirected graph on four nodes is swept on a stride, with every pair of
//! distinct endpoints and every conditioning set drawn from the remaining nodes (and the endpoint
//! pair itself, which must block). The result must be non-vacuous: both verdicts must occur.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{CyclicMixedGraph, DenseNodeId, SigmaSeparationWorkspace, sigma_separates};

fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

/// One traversal of an edge: `from -> to`, with an arrowhead at each end or not.
#[derive(Clone, Copy)]
struct Step {
    from: usize,
    to: usize,
    head_from: bool,
    head_to: bool,
}

struct Reference {
    n: usize,
    steps: Vec<Step>,
    directed: Vec<Vec<bool>>,
    same_component: Vec<Vec<bool>>,
}

impl Reference {
    fn new(n: usize, directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> Self {
        let mut reach = vec![vec![false; n]; n];
        let mut arrow = vec![vec![false; n]; n];
        for (v, row) in reach.iter_mut().enumerate() {
            row[v] = true;
        }
        let mut steps = Vec::new();
        for &(a, b) in directed {
            arrow[a][b] = true;
            reach[a][b] = true;
            steps.push(Step { from: a, to: b, head_from: false, head_to: true });
            steps.push(Step { from: b, to: a, head_from: true, head_to: false });
        }
        for &(a, b) in bidirected {
            steps.push(Step { from: a, to: b, head_from: true, head_to: true });
            steps.push(Step { from: b, to: a, head_from: true, head_to: true });
        }
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if reach[i][k] && reach[k][j] {
                        reach[i][j] = true;
                    }
                }
            }
        }
        let same_component = (0..n)
            .map(|i| (0..n).map(|j| reach[i][j] && reach[j][i]).collect())
            .collect();
        Self { n, steps, directed: reach, same_component }
    }

    fn ancestor_of_set(&self, v: usize, c: &[bool]) -> bool {
        (0..self.n).any(|w| c[w] && self.directed[v][w])
    }

    /// The edge `step` leaves its source along `source -> other` with `other` outside
    /// the source's component.
    fn leaves_component(&self, step: &Step, at: usize) -> bool {
        let (tail_here, other) = if step.from == at {
            (!step.head_from, step.to)
        } else {
            (!step.head_to, step.from)
        };
        tail_here && !self.same_component[at][other]
    }

    fn separated(&self, x: usize, y: usize, c: &[bool]) -> bool {
        if c[x] || c[y] {
            return true;
        }
        if x == y {
            return false;
        }
        let mut seen = vec![false; self.steps.len()];
        let mut stack: Vec<usize> = Vec::new();
        for (i, s) in self.steps.iter().enumerate() {
            if s.from == x {
                if s.to == y {
                    return false;
                }
                seen[i] = true;
                stack.push(i);
            }
        }
        while let Some(i) = stack.pop() {
            let incoming = self.steps[i];
            let v = incoming.to;
            for (j, outgoing) in self.steps.iter().enumerate() {
                if outgoing.from != v {
                    continue;
                }
                let collider = incoming.head_to && outgoing.head_from;
                let open = if collider {
                    self.ancestor_of_set(v, c)
                } else {
                    !(c[v]
                        && (self.leaves_component(&incoming, v)
                            || self.leaves_component(outgoing, v)))
                };
                if !open {
                    continue;
                }
                if outgoing.to == y {
                    return false;
                }
                if !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        true
    }
}

fn product_graph(
    n: usize,
    directed: &[(usize, usize)],
    bidirected: &[(usize, usize)],
) -> CyclicMixedGraph {
    let mut g = CyclicMixedGraph::with_variables(u32::try_from(n).unwrap());
    for &(a, b) in directed {
        g.insert_directed(id(a), id(b)).unwrap();
    }
    for &(a, b) in bidirected {
        g.insert_bidirected(id(a), id(b)).unwrap();
    }
    g
}

/// Compare every endpoint pair and conditioning set; return (separated, not separated) counts.
fn compare(n: usize, directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> (u64, u64) {
    let reference = Reference::new(n, directed, bidirected);
    let graph = product_graph(n, directed, bidirected);
    let mut ws = SigmaSeparationWorkspace::new();
    let (mut sep, mut open) = (0u64, 0u64);
    for mask in 0..(1usize << n) {
        let c: Vec<bool> = (0..n).map(|v| mask >> v & 1 == 1).collect();
        let z: Vec<DenseNodeId> = (0..n).filter(|&v| c[v]).map(id).collect();
        for x in 0..n {
            for y in 0..n {
                let got = sigma_separates(&graph, id(x), id(y), &z, &mut ws).unwrap().separated();
                let want = reference.separated(x, y, &c);
                assert_eq!(
                    got, want,
                    "n={n} directed={directed:?} bidirected={bidirected:?} x={x} y={y} z={z:?}"
                );
                if want {
                    sep += 1;
                } else {
                    open += 1;
                }
            }
        }
    }
    (sep, open)
}

fn ordered_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).collect()
}

fn unordered_pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect()
}

fn pick(pairs: &[(usize, usize)], code: u64) -> Vec<(usize, usize)> {
    pairs.iter().enumerate().filter(|(k, _)| code >> k & 1 == 1).map(|(_, p)| *p).collect()
}

#[test]
fn four_node_mixed_graphs_match_the_walk_definition_on_a_stride() {
    let dpairs = ordered_pairs(4);
    let bpairs = unordered_pairs(4);
    let total = (1u64 << dpairs.len()) * (1u64 << bpairs.len());
    let (mut sep, mut open, mut graphs, mut cyclic) = (0, 0, 0u64, 0u64);
    let mut code = 0u64;
    while code < total {
        let directed = pick(&dpairs, code >> bpairs.len());
        let bidirected = pick(&bpairs, code & ((1 << bpairs.len()) - 1));
        let (s, o) = compare(4, &directed, &bidirected);
        sep += s;
        open += o;
        graphs += 1;
        if directed.iter().any(|&(a, b)| directed.contains(&(b, a))) {
            cyclic += 1;
        }
        code += 37;
    }
    assert!(graphs > 7000, "{graphs}");
    assert!(cyclic > 1000, "swept cyclic graphs {cyclic}");
    assert!(sep > 100_000 && open > 100_000, "separated {sep}, not separated {open}");
}

#[test]
fn five_node_mixed_graphs_match_the_walk_definition_beyond_the_product_oracle_bound() {
    let dpairs = ordered_pairs(5);
    let bpairs = unordered_pairs(5);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut sep, mut open) = (0, 0);
    for _ in 0..400 {
        let density = 3 + next() % 4;
        let directed: Vec<_> = dpairs.iter().copied().filter(|_| next() % 10 < density).collect();
        let bidirected: Vec<_> = bpairs.iter().copied().filter(|_| next() % 10 < 2).collect();
        let (s, o) = compare(5, &directed, &bidirected);
        sep += s;
        open += o;
    }
    assert!(sep > 5_000 && open > 5_000, "separated {sep}, not separated {open}");
}
