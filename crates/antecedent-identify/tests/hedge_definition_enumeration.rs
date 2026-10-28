//! The hedge criterion restated from Shpitser and Pearl's definition and compared with ID.
//!
//! The reference enumerates node sets `F' ⊆ F`, root sets `R`, and edge subsets directly from
//! the definition of an `R`-rooted C-forest (bidirected-connected, every node has at most one
//! child, the childless nodes are exactly `R`) and of a hedge (`F ∩ X ≠ ∅`, `F' ∩ X = ∅`,
//! `F' ⊆ F` as subgraphs, `R ⊆ An(Y)` after cutting the arrows into `X`). It shares no code with
//! `HedgeCertificate::verify` or with ID.
//!
//! Two properties are checked on every ADMG of three nodes and on the four-node ADMGs listed in
//! each test:
//!
//! * completeness: ID reports `NotIdentified` exactly when a hedge exists for some sub-query
//!   `(X', Y')` with non-empty `X' ⊆ X`, `Y' ⊆ Y` (Shpitser and Pearl 2006, Theorem 4);
//! * the shipped verifier accepts a node-set pair exactly when the definition has an edge
//!   subset realising it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    Intervention, InterventionalDistributionQuery, Value, VariableId, CausalQuery,
};
use antecedent_graph::{Admg, BitSet, DenseNodeId};
use antecedent_identify::{
    HedgeCertificate, IdIdentifier, IdentificationStatus, IdentificationWorkspace,
};

fn d(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}
fn v(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

struct Graph {
    n: usize,
    /// `child[a][b]`: a -> b.
    child: Vec<Vec<bool>>,
    /// `bi[a][b]`: a <-> b (symmetric).
    bi: Vec<Vec<bool>>,
}

impl Graph {
    fn from_masks(n: usize, dir_mask: usize, bi_mask: usize) -> Self {
        let pairs: Vec<(usize, usize)> =
            (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
        let mut child = vec![vec![false; n]; n];
        let mut bi = vec![vec![false; n]; n];
        for (k, &(a, b)) in pairs.iter().enumerate() {
            if dir_mask >> k & 1 == 1 {
                child[a][b] = true;
            }
            if bi_mask >> k & 1 == 1 {
                bi[a][b] = true;
                bi[b][a] = true;
            }
        }
        Self { n, child, bi }
    }

    fn admg(&self) -> Admg {
        let mut g = Admg::with_variables(u32::try_from(self.n).unwrap());
        for a in 0..self.n {
            for b in 0..self.n {
                if self.child[a][b] {
                    g.insert_directed(d(a), d(b)).unwrap();
                }
                if a < b && self.bi[a][b] {
                    g.insert_bidirected(d(a), d(b)).unwrap();
                }
            }
        }
        g
    }

    /// Ancestors (each node its own) of `y` after cutting the arrows into `x`.
    fn ancestors_after_cut(&self, x: usize, y: usize) -> usize {
        let mut anc = y;
        loop {
            let mut grew = anc;
            for a in 0..self.n {
                for b in 0..self.n {
                    if self.child[a][b] && anc >> b & 1 == 1 && x >> b & 1 == 0 {
                        grew |= 1 << a;
                    }
                }
            }
            if grew == anc {
                return anc;
            }
            anc = grew;
        }
    }

    fn bidirected_connected(&self, set: usize) -> bool {
        let Some(start) = (0..self.n).find(|&i| set >> i & 1 == 1) else { return false };
        let mut seen = 1usize << start;
        let mut stack = vec![start];
        while let Some(a) = stack.pop() {
            for b in 0..self.n {
                if set >> b & 1 == 1 && self.bi[a][b] && seen >> b & 1 == 0 {
                    seen |= 1 << b;
                    stack.push(b);
                }
            }
        }
        seen == set
    }

    /// Whether some edge subset makes `big` an `r`-rooted C-forest whose restriction to `small`
    /// is an `r`-rooted C-forest of `small` (a sub-forest: every kept parent edge of a node of
    /// `small` stays inside `small`).
    fn forest_pair_exists(&self, big: usize, small: usize, r: usize) -> bool {
        if !self.bidirected_connected(big) || !self.bidirected_connected(small) {
            return false;
        }
        // Each non-root node of `big` chooses exactly one child inside `big`; roots keep none.
        let nodes: Vec<usize> = (0..self.n).filter(|&i| big >> i & 1 == 1 && r >> i & 1 == 0).collect();
        self.choose(&nodes, 0, big, small)
    }

    fn choose(&self, nodes: &[usize], at: usize, big: usize, small: usize) -> bool {
        let Some(&node) = nodes.get(at) else { return true };
        for c in 0..self.n {
            if !self.child[node][c] || big >> c & 1 == 0 {
                continue;
            }
            // A non-root node of the smaller forest needs its one child inside it.
            if small >> node & 1 == 1 && small >> c & 1 == 0 {
                continue;
            }
            if self.choose(nodes, at + 1, big, small) {
                return true;
            }
        }
        false
    }

    /// Whether `(f, f_prime)` is a hedge for `P_x(y)`.
    fn is_hedge(&self, f: usize, f_prime: usize, x: usize, y: usize) -> bool {
        if f_prime == 0 || f_prime & !f != 0 || f & x == 0 || f_prime & x != 0 {
            return false;
        }
        let candidates = f_prime & self.ancestors_after_cut(x, y);
        (1..(1usize << self.n))
            .filter(|&r| r & !candidates == 0)
            .any(|r| self.forest_pair_exists(f, f_prime, r))
    }

    fn some_hedge(&self, x: usize, y: usize) -> bool {
        let all = (1usize << self.n) - 1;
        (1..=all).any(|f| (1..=f).any(|fp| fp & !f == 0 && self.is_hedge(f, fp, x, y)))
    }

    fn some_hedge_for_a_sub_outcome(&self, x: usize, y: usize) -> bool {
        (1..=y).filter(|ys| ys & !y == 0).any(|ys| self.some_hedge(x, ys))
    }
}

fn members(mask: usize, n: usize) -> Vec<usize> {
    (0..n).filter(|&i| mask >> i & 1 == 1).collect()
}

fn query(outcomes: &[usize], treatments: &[usize]) -> CausalQuery {
    let interventions: Vec<_> =
        treatments.iter().map(|&t| Intervention::set(v(t), Value::f64(1.0))).collect();
    CausalQuery::Distribution(
        InterventionalDistributionQuery::new(v(outcomes[0]), interventions)
            .with_outcomes(outcomes.iter().map(|&o| v(o)).collect::<Vec<_>>()),
    )
}

fn sweep(n: usize, stride: usize, single_only: bool) -> (usize, usize) {
    let id = IdIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let pairs = n * (n - 1) / 2;
    let (mut hedged, mut free) = (0, 0);
    let mut index = 0usize;
    for dir_mask in 0..1usize << pairs {
        for bi_mask in 0..1usize << pairs {
            index += 1;
            if index % stride != 0 {
                continue;
            }
            let g = Graph::from_masks(n, dir_mask, bi_mask);
            let prepared = id.prepare(&g.admg()).unwrap();
            for x in 1..1usize << n {
                for y in 1..1usize << n {
                    if x & y != 0 || (single_only && (x.count_ones() > 1 || y.count_ones() > 1)) {
                        continue;
                    }
                    let res = id
                        .identify(&prepared, &query(&members(y, n), &members(x, n)), &mut ws)
                        .unwrap();
                    let expected_hedge = g.some_hedge_for_a_sub_outcome(x, y);
                    let context = format!(
                        "n={n} dir={dir_mask:b} bi={bi_mask:b} X={:?} Y={:?}",
                        members(x, n),
                        members(y, n)
                    );
                    match res.status {
                        IdentificationStatus::NotIdentified => {
                            hedged += 1;
                            assert!(expected_hedge, "ID refuses but no hedge exists: {context}");
                        }
                        IdentificationStatus::NonparametricallyIdentified => {
                            free += 1;
                            assert!(!expected_hedge, "ID identifies but a hedge exists: {context}");
                        }
                        other => panic!("{context}: unexpected {other:?}"),
                    }
                }
            }
        }
    }
    (hedged, free)
}

#[test]
fn id_refuses_exactly_when_a_hedge_exists_on_every_three_node_admg() {
    let (hedged, free) = sweep(3, 1, false);
    assert!(hedged > 100 && free > 500, "hedged={hedged} free={free}");
}

#[test]
fn id_refuses_exactly_when_a_hedge_exists_on_every_four_node_single_query() {
    let (hedged, free) = sweep(4, 1, true);
    assert!(hedged > 5_000 && free > 5_000, "hedged={hedged} free={free}");
}

#[test]
fn id_refuses_exactly_when_a_hedge_exists_on_four_node_joint_queries() {
    let (hedged, free) = sweep(4, 3, false);
    assert!(hedged > 2_000 && free > 2_000, "hedged={hedged} free={free}");
}

fn bitset(n: usize, mask: usize) -> BitSet {
    let mut set = BitSet::with_len(n);
    for i in members(mask, n) {
        set.insert(d(i));
    }
    set
}

fn verifier_agrees(n: usize, stride: usize) -> (usize, usize) {
    let id = IdIdentifier::new();
    let pairs = n * (n - 1) / 2;
    let (mut accepted, mut rejected) = (0, 0);
    let mut index = 0usize;
    for dir_mask in 0..1usize << pairs {
        for bi_mask in 0..1usize << pairs {
            index += 1;
            if index % stride != 0 {
                continue;
            }
            let g = Graph::from_masks(n, dir_mask, bi_mask);
            let prepared = id.prepare(&g.admg()).unwrap();
            for x in 1..1usize << n {
                for y in 1..1usize << n {
                    if x & y != 0 {
                        continue;
                    }
                    let (xs, ys): (Vec<_>, Vec<_>) =
                        (members(x, n).into_iter().map(v).collect(), members(y, n).into_iter().map(v).collect());
                    for f in 1..1usize << n {
                        for fp in 1..=f {
                            if fp & !f != 0 {
                                continue;
                            }
                            let certificate =
                                HedgeCertificate::from_sets(&bitset(n, f), &bitset(n, fp), |node| {
                                    v(node.as_usize())
                                });
                            let shipped = certificate.verify(&prepared, &xs, &ys).is_ok();
                            let defined = g.is_hedge(f, fp, x, y);
                            assert_eq!(
                                shipped, defined,
                                "n={n} dir={dir_mask:b} bi={bi_mask:b} X={x:b} Y={y:b} F={f:b} F'={fp:b}"
                            );
                            if defined {
                                accepted += 1;
                            } else {
                                rejected += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    (accepted, rejected)
}

#[test]
fn the_shipped_verifier_matches_the_definition_on_every_three_node_admg() {
    let (accepted, rejected) = verifier_agrees(3, 1);
    assert!(accepted > 100 && rejected > 10_000, "{accepted} {rejected}");
}

#[test]
fn the_shipped_verifier_matches_the_definition_on_sampled_four_node_admgs() {
    let (accepted, rejected) = verifier_agrees(4, 97);
    assert!(accepted > 100 && rejected > 10_000, "{accepted} {rejected}");
}

#[test]
fn the_reference_recovers_the_textbook_hedges() {
    // Bow arc x -> y with x <-> y: F = {x, y}, F' = {y}.
    let bow = Graph::from_masks(2, 0b1, 0b1);
    assert!(bow.is_hedge(0b11, 0b10, 0b01, 0b10));
    // Without the bidirected arc the pair is not a C-forest pair.
    assert!(!Graph::from_masks(2, 0b1, 0).some_hedge(0b01, 0b10));
    // Front door with a confounded mediator: x -> m -> y, x <-> y, m <-> y has a hedge.
    let confounded = Graph::from_masks(3, 0b101, 0b110);
    assert!(confounded.some_hedge_for_a_sub_outcome(0b001, 0b100));
    // The classic front door (x <-> y only) has none.
    let front = Graph::from_masks(3, 0b101, 0b010);
    assert!(!front.some_hedge_for_a_sub_outcome(0b001, 0b100));
}
