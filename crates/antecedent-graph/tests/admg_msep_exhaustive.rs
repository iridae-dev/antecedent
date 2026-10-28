//! m-separation in ADMGs against the textbook path definition, exhaustively on small graphs.
//!
//! The oracle follows the definition (Richardson 2003; Richardson and Spirtes 2002): a path
//! between `x` and `y` over directed and bidirected edges is m-connecting given `Z` when every
//! non-collider on it lies outside `Z` and every collider is an ancestor of some member of `Z`
//! (counting a member as its own ancestor). `x` and `y` are m-separated when no simple path is
//! m-connecting. It walks simple paths over adjacency matrices with a Warshall ancestor closure,
//! so it shares no Bayes-ball, moralisation or bitset code with the product.
//!
//! Every ADMG is a relabelling of one whose directed edges all run from a lower to a higher
//! index, so each unordered pair of the fixed order carries one of four states (no edge, `->`,
//! `<->`, or both `->` and `<->`). All 4^6 graphs on four nodes are swept completely; five-node
//! graphs are swept on a stride. Every pair, every conditioning set and both argument orders are
//! compared, and the set form of the query is compared against the oracle applied pairwise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{Admg, DSeparationWorkspace, DenseNodeId};

fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

struct Oracle {
    n: usize,
    arrow: Vec<Vec<bool>>,
    bi: Vec<Vec<bool>>,
    /// `anc[a][b]`: `a` is `b` or a proper ancestor of `b`.
    anc: Vec<Vec<bool>>,
}

impl Oracle {
    fn new(n: usize, directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> Self {
        let mut arrow = vec![vec![false; n]; n];
        let mut bi = vec![vec![false; n]; n];
        for &(a, b) in directed {
            arrow[a][b] = true;
        }
        for &(a, b) in bidirected {
            bi[a][b] = true;
            bi[b][a] = true;
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

    fn separated(&self, x: usize, y: usize, z: &[bool]) -> bool {
        if x == y || z[x] || z[y] {
            return false;
        }
        let mut visited = vec![false; self.n];
        visited[x] = true;
        // `None` marks the start: there is no arrowhead at the first node.
        !self.connects(x, None, y, z, &mut visited)
    }

    /// Extend a simple path at `cur`, which was entered over an edge whose mark at `cur` is an
    /// arrowhead exactly when `head_in` (`None` at the start). Parallel `->` and `<->` edges
    /// between one pair are distinct steps.
    fn connects(
        &self,
        cur: usize,
        head_in: Option<bool>,
        target: usize,
        z: &[bool],
        visited: &mut [bool],
    ) -> bool {
        for next in 0..self.n {
            if visited[next] {
                continue;
            }
            // (edge exists, arrowhead at `cur`, arrowhead at `next`)
            let steps = [
                (self.arrow[cur][next], false, true),
                (self.arrow[next][cur], true, false),
                (self.bi[cur][next], true, true),
            ];
            for (exists, head_cur, head_next) in steps {
                if !exists {
                    continue;
                }
                if let Some(head_in) = head_in {
                    let collider = head_in && head_cur;
                    let open = if collider {
                        (0..self.n).any(|w| z[w] && self.anc[cur][w])
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
                let found = self.connects(next, Some(head_next), target, z, visited);
                visited[next] = false;
                if found {
                    return true;
                }
            }
        }
        false
    }
}

fn edges_of(n: usize, code: u64) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let (mut directed, mut bidirected) = (Vec::new(), Vec::new());
    let mut rest = code;
    for i in 0..n {
        for j in (i + 1)..n {
            let state = rest % 4;
            rest /= 4;
            if state == 1 || state == 3 {
                directed.push((i, j));
            }
            if state >= 2 {
                bidirected.push((i, j));
            }
        }
    }
    (directed, bidirected)
}

fn build(n: usize, directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> Admg {
    let mut g = Admg::with_variables(u32::try_from(n).unwrap());
    for &(a, b) in directed {
        g.insert_directed(id(a), id(b)).unwrap();
    }
    for &(a, b) in bidirected {
        g.insert_bidirected(id(a), id(b)).unwrap();
    }
    g
}

fn check_graph(n: usize, code: u64, ws: &mut DSeparationWorkspace) -> (usize, usize) {
    let (directed, bidirected) = edges_of(n, code);
    let g = build(n, &directed, &bidirected);
    let oracle = Oracle::new(n, &directed, &bidirected);
    let (mut checked, mut connected) = (0, 0);
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
                let context = format!("x={x} y={y} z={z:?} directed={directed:?} bi={bidirected:?}");
                assert_eq!(g.is_m_separated(id(x), id(y), &z, ws).unwrap(), want, "{context}");
                assert_eq!(g.is_m_separated(id(y), id(x), &z, ws).unwrap(), want, "rev {context}");
                let witnessed = g.m_separation(id(x), id(y), &z, ws).unwrap();
                assert_eq!(
                    matches!(witnessed, antecedent_graph::SeparationResult::Separated { .. }),
                    want,
                    "witness {context}"
                );
                checked += 1;
                connected += usize::from(!want);
            }
        }
    }
    (checked, connected)
}

#[test]
fn every_four_node_admg_matches_the_path_definition() {
    let mut ws = DSeparationWorkspace::default();
    let (mut checked, mut connected) = (0, 0);
    for code in 0..4u64.pow(6) {
        let (c, k) = check_graph(4, code, &mut ws);
        checked += c;
        connected += k;
    }
    assert_eq!(checked, 4u64.pow(6) as usize * 6 * 4);
    assert!(connected > 20_000 && checked - connected > 5_000, "{connected}/{checked}");
}

#[test]
fn strided_five_node_admgs_match_the_path_definition() {
    let mut ws = DSeparationWorkspace::default();
    let (mut checked, mut connected) = (0, 0);
    for code in (0..4u64.pow(10)).step_by(331) {
        let (c, k) = check_graph(5, code, &mut ws);
        checked += c;
        connected += k;
    }
    assert!(checked > 20_000 && connected > 2_000 && checked - connected > 500);
}

#[test]
fn set_form_agrees_with_the_pairwise_oracle_on_every_four_node_admg() {
    let mut ws = DSeparationWorkspace::default();
    for code in (0..4u64.pow(6)).step_by(7) {
        let (directed, bidirected) = edges_of(4, code);
        let g = build(4, &directed, &bidirected);
        let oracle = Oracle::new(4, &directed, &bidirected);
        // Assign each node to sources (1), targets (2), conditioning (3) or unused (0).
        for assignment in 0..4u32.pow(4) {
            let role = |v: usize| (assignment / 4u32.pow(u32::try_from(v).unwrap())) % 4;
            let pick = |r: u32| -> Vec<usize> { (0..4).filter(|&v| role(v) == r).collect() };
            let (sources, targets, cond) = (pick(1), pick(2), pick(3));
            let mut zmask = vec![false; 4];
            for &v in &cond {
                zmask[v] = true;
            }
            let want = sources.iter().all(|&s| targets.iter().all(|&t| oracle.separated(s, t, &zmask)));
            let got = g
                .is_m_separated_sets(
                    &sources.iter().map(|&v| id(v)).collect::<Vec<_>>(),
                    &targets.iter().map(|&v| id(v)).collect::<Vec<_>>(),
                    &cond.iter().map(|&v| id(v)).collect::<Vec<_>>(),
                    &mut ws,
                )
                .unwrap();
            let expected = sources.is_empty() || targets.is_empty() || want;
            assert_eq!(got, expected, "{sources:?} {targets:?} {cond:?} {directed:?} {bidirected:?}");
        }
    }
}

#[test]
fn the_oracle_distinguishes_the_known_cases() {
    // x <-> y with a collider at m: x -> m <- y is opened by conditioning on m.
    let o = Oracle::new(3, &[(0, 2), (1, 2)], &[]);
    assert!(o.separated(0, 1, &[false, false, false]));
    assert!(!o.separated(0, 1, &[false, false, true]));
    // A bidirected edge connects regardless of the remaining nodes.
    let o = Oracle::new(3, &[], &[(0, 1)]);
    assert!(!o.separated(0, 1, &[false, false, false]));
    // Directed ancestor of a conditioning node opens a bidirected collider: x -> m <-> y, m -> w.
    let o = Oracle::new(4, &[(0, 2), (2, 3)], &[(1, 2)]);
    assert!(o.separated(0, 1, &[false, false, false, false]));
    assert!(!o.separated(0, 1, &[false, false, false, true]));
}

// --- latent projection: m-separation of the projected ADMG equals d-separation in the DAG ---

use antecedent_graph::{Dag, latent_project};

fn dag_arcs(n: usize, code: u64) -> Vec<(usize, usize)> {
    // Base-2 code over the pairs of the fixed order: every DAG is a relabelling of one of these.
    let mut arcs = Vec::new();
    let mut bit = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if code >> bit & 1 == 1 {
                arcs.push((i, j));
            }
            bit += 1;
        }
    }
    arcs
}

fn projection_agrees(n: usize, code: u64, ws: &mut DSeparationWorkspace) -> usize {
    let arcs = dag_arcs(n, code);
    let mut dag = Dag::with_variables(u32::try_from(n).unwrap());
    for &(a, b) in &arcs {
        dag.insert_directed(id(a), id(b)).unwrap();
    }
    // The oracle treats the DAG as an ADMG without bidirected edges: the same definition.
    let full = Oracle::new(n, &arcs, &[]);
    let mut checked = 0;
    for mask in 0usize..(1 << n) {
        // Observed nodes in descending index order, so dense ids of the projection differ from the
        // original ones.
        let observed: Vec<usize> = (0..n).rev().filter(|&v| mask >> v & 1 == 1).collect();
        if observed.len() < 2 {
            continue;
        }
        let projected = latent_project(&dag, &observed.iter().map(|&v| id(v)).collect::<Vec<_>>())
            .unwrap();
        let k = observed.len();
        for i in 0..k {
            for j in (i + 1)..k {
                let rest: Vec<usize> = (0..k).filter(|&m| m != i && m != j).collect();
                for sub in 0usize..(1 << rest.len()) {
                    let chosen: Vec<usize> =
                        rest.iter().enumerate().filter(|(b, _)| sub >> b & 1 == 1).map(|(_, &m)| m).collect();
                    let mut zmask = vec![false; n];
                    for &m in &chosen {
                        zmask[observed[m]] = true;
                    }
                    let want = full.separated(observed[i], observed[j], &zmask);
                    let got = projected
                        .is_m_separated(
                            id(i),
                            id(j),
                            &chosen.iter().map(|&m| id(m)).collect::<Vec<_>>(),
                            ws,
                        )
                        .unwrap();
                    assert_eq!(
                        got, want,
                        "arcs={arcs:?} observed={observed:?} pair=({},{}) z={chosen:?}",
                        observed[i], observed[j]
                    );
                    checked += 1;
                }
            }
        }
    }
    checked
}

#[test]
fn projection_preserves_separation_for_every_four_node_dag_and_observed_subset() {
    let mut ws = DSeparationWorkspace::default();
    let checked: usize = (0..1u64 << 6).map(|code| projection_agrees(4, code, &mut ws)).sum();
    assert!(checked > 3_000, "{checked}");
}

#[test]
fn projection_preserves_separation_on_a_stride_of_the_five_node_dags_and_every_observed_subset() {
    let mut ws = DSeparationWorkspace::default();
    let checked: usize =
        (0..1u64 << 10).step_by(13).map(|code| projection_agrees(5, code, &mut ws)).sum();
    assert!(checked > 20_000, "{checked}");
}
