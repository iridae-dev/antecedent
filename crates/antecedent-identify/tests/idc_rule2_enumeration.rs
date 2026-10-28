//! IDC checked on every four-node ADMG against a rule-2 closure computed from the definition.
//!
//! Shpitser and Pearl's IDC moves a conditioning variable `z` into the intervention set when
//! `Y ⊥ z | X, Z \ {z}` in `G_{X̄ z̲}` and then asks ID for `P_{X ∪ Z'}(Y ∪ Z'')`. The reference
//! here decides the separations by enumerating every simple path of the mutilated mixed graph
//! and applying the collider rule with ancestors (it shares no code with the product's
//! d-separation workspace), explores every order in which the moves can be taken, and requires
//! all orders to end in the same identification status as the product. The final unconditional
//! question is put to ID, whose refusals are separately checked against a hedge enumeration in
//! `hedge_definition_enumeration.rs`; a reduced query with no intervention is identified.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, Intervention, InterventionalDistributionQuery, Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    IdIdentifier, IdcIdentifier, IdentificationStatus, IdentificationWorkspace,
};

fn d(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}
fn v(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

#[derive(Clone)]
struct Graph {
    n: usize,
    child: Vec<Vec<bool>>,
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

    /// `G_{X̄ z̲}`: no arrowhead at a node of `x`, no arrow leaving `z` (if any).
    fn mutilated(&self, x: usize, z: Option<usize>) -> Self {
        let mut g = self.clone();
        for a in 0..self.n {
            for b in 0..self.n {
                if x >> b & 1 == 1 {
                    g.child[a][b] = false;
                    g.bi[a][b] = false;
                    g.bi[b][a] = false;
                }
                if z == Some(a) {
                    g.child[a][b] = false;
                }
            }
        }
        g
    }

    fn ancestors_of(&self, set: usize) -> usize {
        let mut anc = set;
        loop {
            let mut next = anc;
            for a in 0..self.n {
                for b in 0..self.n {
                    if self.child[a][b] && anc >> b & 1 == 1 {
                        next |= 1 << a;
                    }
                }
            }
            if next == anc {
                return anc;
            }
            anc = next;
        }
    }

    /// Arrowhead of the edge `a - b` at `b`, if the edge exists in that orientation class.
    fn edges(&self, a: usize, b: usize) -> Vec<(bool, bool)> {
        // (head at a, head at b)
        let mut out = Vec::new();
        if self.child[a][b] {
            out.push((false, true));
        }
        if self.child[b][a] {
            out.push((true, false));
        }
        if self.bi[a][b] {
            out.push((true, true));
        }
        out
    }

    /// Whether some simple path from `from` to `to` is m-connecting given `cond`.
    fn connected(&self, from: usize, to: usize, cond: usize) -> bool {
        let anc_cond = self.ancestors_of(cond);
        let mut path = vec![from];
        let mut heads: Vec<bool> = Vec::new(); // head at the later end of each edge step
        self.extend(&mut path, &mut heads, to, cond, anc_cond)
    }

    fn extend(
        &self,
        path: &mut Vec<usize>,
        heads: &mut Vec<bool>,
        to: usize,
        cond: usize,
        anc_cond: usize,
    ) -> bool {
        let last = *path.last().unwrap();
        if last == to {
            return true;
        }
        for next in 0..self.n {
            if path.contains(&next) {
                continue;
            }
            for (head_last, head_next) in self.edges(last, next) {
                // Judge `last` as an interior node once the new edge is known.
                if path.len() >= 2 {
                    let head_in = *heads.last().unwrap();
                    let collider = head_in && head_last;
                    let blocked =
                        if collider { anc_cond >> last & 1 == 0 } else { cond >> last & 1 == 1 };
                    if blocked {
                        continue;
                    }
                }
                path.push(next);
                heads.push(head_next);
                let found = self.extend(path, heads, to, cond, anc_cond);
                path.pop();
                heads.pop();
                if found {
                    return true;
                }
            }
        }
        false
    }

    fn separated(&self, ys: usize, z: usize, cond: usize) -> bool {
        (0..self.n).filter(|&y| ys >> y & 1 == 1).all(|y| !self.connected(y, z, cond))
    }
}

fn members(mask: usize, n: usize) -> Vec<usize> {
    (0..n).filter(|&i| mask >> i & 1 == 1).collect()
}

fn set_query(outcomes: &[usize], treatments: &[usize], conditioning: &[usize]) -> CausalQuery {
    let interventions: Vec<_> =
        treatments.iter().map(|&t| Intervention::set(v(t), Value::f64(1.0))).collect();
    CausalQuery::Distribution(
        InterventionalDistributionQuery::new(v(outcomes[0]), interventions)
            .with_outcomes(outcomes.iter().map(|&o| v(o)).collect::<Vec<_>>())
            .with_conditioning(conditioning.iter().map(|&c| v(c)).collect::<Vec<_>>()),
    )
}

/// Every final `(X', Z'')` reachable by taking rule-2 moves in any order.
fn closures(g: &Graph, y: usize, x: usize, z: usize, out: &mut Vec<(usize, usize)>) {
    let mut moved = false;
    for zi in members(z, g.n) {
        let rest = z & !(1 << zi);
        let cut = g.mutilated(x, Some(zi));
        if cut.separated(y, zi, x | rest) {
            moved = true;
            closures(g, y, x | 1 << zi, rest, out);
        }
    }
    if !moved && !out.contains(&(x, z)) {
        out.push((x, z));
    }
}

fn run(n: usize, stride: usize) -> (usize, usize, usize) {
    let id = IdIdentifier::new();
    let idc = IdcIdentifier::new();
    let mut ws = IdentificationWorkspace::default();
    let pairs = n * (n - 1) / 2;
    let (mut refused, mut identified, mut moved_cases) = (0, 0, 0);
    let mut index = 0usize;
    for dir_mask in 0..1usize << pairs {
        for bi_mask in 0..1usize << pairs {
            index += 1;
            if index % stride != 0 {
                continue;
            }
            let g = Graph::from_masks(n, dir_mask, bi_mask);
            let prepared = id.prepare(&g.admg()).unwrap();
            for role in 0..4usize.pow(u32::try_from(n).unwrap()) {
                let (mut y, mut x, mut z) = (0usize, 0usize, 0usize);
                for i in 0..n {
                    match role / 4usize.pow(u32::try_from(i).unwrap()) % 4 {
                        1 => y |= 1 << i,
                        2 => x |= 1 << i,
                        3 => z |= 1 << i,
                        _ => {}
                    }
                }
                if y == 0 || z == 0 {
                    continue;
                }
                let context = format!(
                    "n={n} dir={dir_mask:b} bi={bi_mask:b} Y={:?} X={:?} Z={:?}",
                    members(y, n),
                    members(x, n),
                    members(z, n)
                );
                let mut finals = Vec::new();
                closures(&g, y, x, z, &mut finals);
                // Every move order must end in one status.
                let mut statuses = Vec::new();
                for &(x2, z2) in &finals {
                    let ids = y | z2;
                    let status = if x2 == 0 {
                        IdentificationStatus::NonparametricallyIdentified
                    } else {
                        id.identify(
                            &prepared,
                            &set_query(&members(ids, n), &members(x2, n), &[]),
                            &mut ws,
                        )
                        .unwrap()
                        .status
                    };
                    statuses.push(status);
                }
                assert!(
                    statuses.windows(2).all(|w| w[0] == w[1]),
                    "move orders disagree ({statuses:?}): {context}"
                );
                if finals.iter().any(|&(x2, _)| x2 != x) {
                    moved_cases += 1;
                }
                let got = idc
                    .identify(
                        &prepared,
                        &set_query(&members(y, n), &members(x, n), &members(z, n)),
                        &mut ws,
                    )
                    .unwrap_or_else(|e| panic!("{context}: {e}"));
                assert_eq!(got.status, statuses[0], "{context}");
                match got.status {
                    IdentificationStatus::NotIdentified => refused += 1,
                    IdentificationStatus::NonparametricallyIdentified => identified += 1,
                    other => panic!("{context}: unexpected {other:?}"),
                }
            }
        }
    }
    println!("n={n} refused={refused} identified={identified} with_moves={moved_cases}");
    (refused, identified, moved_cases)
}

#[test]
fn idc_matches_the_rule_two_closure_on_every_three_node_admg() {
    let (refused, identified, moved) = run(3, 1);
    assert!(refused > 50 && identified > 500 && moved > 100, "{refused} {identified} {moved}");
}

#[test]
fn idc_matches_the_rule_two_closure_on_sampled_four_node_admgs() {
    let (refused, identified, moved) = run(4, 1);
    assert!(
        refused > 1_000 && identified > 1_000 && moved > 1_000,
        "{refused} {identified} {moved}"
    );
}

#[test]
fn the_reference_separation_recovers_textbook_cases() {
    // Chain 0 -> 1 -> 2: 0 and 2 are connected unconditionally, separated given 1.
    let chain = Graph::from_masks(3, 0b101, 0);
    assert!(chain.connected(0, 2, 0));
    assert!(!chain.connected(0, 2, 0b010));
    // Collider 0 -> 2 <- 1: separated unconditionally, connected given 2.
    let collider = Graph::from_masks(3, 0b110, 0);
    assert!(!collider.connected(0, 1, 0));
    assert!(collider.connected(0, 1, 0b100));
    // Bidirected 0 <-> 2 is open unless an arrowhead at 2 is cut.
    let confounded = Graph::from_masks(3, 0, 0b010);
    assert!(confounded.connected(0, 2, 0));
    assert!(!confounded.mutilated(0b100, None).connected(0, 2, 0));
}
