//! PAG completion against the definition of a Markov equivalence class of maximal ancestral
//! graphs, exhaustively on three and four nodes.
//!
//! Reference: enumerate every mixed graph whose pair states are absent, `->`, `<-` or `<->`.
//! A graph is a MAG when it has no directed cycle, no almost-directed cycle (`a <-> b` with `a` an
//! ancestor of `b`) and every non-adjacent pair is m-separated by some set. Its m-separation
//! signature (the verdict of every pair against every conditioning set, from a simple-path
//! oracle that shares no code with the product) defines its Markov class.
//!
//! * For each class, the PAG keeps the marks all members agree on and puts a circle elsewhere. The
//!   sampler must return exactly the members of that class, and report no ambiguity.
//! * For every partially marked skeleton on three nodes (each pair absent or any of the eight
//!   non-undirected mark pairs), the compatible MAGs are grouped by signature. When one class
//!   remains the sampler must return exactly the compatible members of it; when several
//!   remain the sampler must return nothing and flag the ambiguity, since it refuses to
//!   invent membership for a partial graph that is no PAG. Only compatible MAGs whose
//!   unshielded colliders are exactly the triples with two arrowheads in the partial graph
//!   count, the invariant a PAG carries. On three nodes that rule leaves no ambiguous graph, so
//!   the ambiguity branch is not exercised here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_graph::{CompletionSampler, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};

fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

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
        for a in 0..self.n {
            for b in 0..self.n {
                if a != b && self.arrow[a][b] && self.anc[b][a] {
                    return false;
                }
                if a != b && self.bi[a][b] && self.anc[a][b] {
                    return false;
                }
            }
        }
        true
    }

    /// Simple-path m-separation: collider needs an ancestor in `z`, non-collider outside `z`.
    fn msep(&self, x: usize, y: usize, z: &[bool]) -> bool {
        let mut path = vec![x];
        !self.connects(&mut path, y, z)
    }

    fn connects(&self, path: &mut Vec<usize>, y: usize, z: &[bool]) -> bool {
        let cur = *path.last().unwrap();
        for next in 0..self.n {
            if next == cur || path.contains(&next) || !self.adjacent(cur, next) {
                continue;
            }
            // Open-ness of `cur` as an interior node is decided once the next step is known.
            if path.len() >= 2 && !self.interior_open(path[path.len() - 2], cur, next, z) {
                continue;
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

    /// Head at `v` on the edge between `v` and `w`.
    fn head_at(&self, v: usize, w: usize) -> bool {
        self.arrow[w][v] || self.bi[v][w]
    }

    fn interior_open(&self, prev: usize, v: usize, next: usize, z: &[bool]) -> bool {
        // Parallel edges between a pair cannot occur: each pair has exactly one state here.
        let collider = self.head_at(v, prev) && self.head_at(v, next);
        if collider {
            (0..self.n).any(|w| z[w] && self.anc[v][w])
        } else {
            !z[v]
        }
    }

    fn signature(&self) -> Vec<bool> {
        let n = self.n;
        let mut out = Vec::new();
        for mask in 0..(1usize << n) {
            let z: Vec<bool> = (0..n).map(|v| mask >> v & 1 == 1).collect();
            for (x, y) in pairs(n) {
                if z[x] || z[y] {
                    continue;
                }
                out.push(self.msep(x, y, &z));
            }
        }
        out
    }

    fn maximal(&self) -> bool {
        let n = self.n;
        for (x, y) in pairs(n) {
            if self.adjacent(x, y) {
                continue;
            }
            let separable = (0..(1usize << n)).any(|mask| {
                let z: Vec<bool> = (0..n).map(|v| mask >> v & 1 == 1).collect();
                !z[x] && !z[y] && self.msep(x, y, &z)
            });
            if !separable {
                return false;
            }
        }
        true
    }
}

fn all_codes(n: usize) -> Vec<Code> {
    let k = pairs(n).len();
    (0..4usize.pow(u32::try_from(k).unwrap()))
        .map(|mut c| {
            (0..k)
                .map(|_| {
                    let s = (c % 4) as u8;
                    c /= 4;
                    s
                })
                .collect()
        })
        .collect()
}

fn endpoints(s: u8) -> Option<(Endpoint, Endpoint)> {
    match s {
        1 => Some((Endpoint::Tail, Endpoint::Arrow)),
        2 => Some((Endpoint::Arrow, Endpoint::Tail)),
        3 => Some((Endpoint::Arrow, Endpoint::Arrow)),
        _ => None,
    }
}

fn pag_from(n: usize, marks: &[Option<(Endpoint, Endpoint)>]) -> Option<Pag> {
    let mut g = Pag::with_variables(u32::try_from(n).unwrap());
    for (&(i, j), m) in pairs(n).iter().zip(marks) {
        if let Some((at_a, at_b)) = *m {
            let edge =
                MarkedEdge { a: id(i), b: id(j), at_a, at_b, middle: MiddleMark::Empty };
            g.insert_marked(edge).ok()?;
        }
    }
    Some(g)
}

fn code_of(n: usize, g: &Pag) -> Code {
    pairs(n)
        .iter()
        .map(|&(i, j)| match g.edge_between(id(i), id(j)) {
            None => 0,
            Some(e) => {
                let (at_i, at_j) = if e.a == id(i) { (e.at_a, e.at_b) } else { (e.at_b, e.at_a) };
                match (at_i, at_j) {
                    (Endpoint::Tail, Endpoint::Arrow) => 1,
                    (Endpoint::Arrow, Endpoint::Tail) => 2,
                    (Endpoint::Arrow, Endpoint::Arrow) => 3,
                    other => panic!("non-MAG completion {other:?}"),
                }
            }
        })
        .collect()
}

fn mags(n: usize) -> Vec<(Code, Vec<bool>)> {
    all_codes(n)
        .into_iter()
        .filter_map(|c| {
            let m = Mixed::new(n, &c);
            (m.ancestral() && m.maximal()).then(|| {
                let s = m.signature();
                (c, s)
            })
        })
        .collect()
}

#[test]
fn every_markov_class_of_mags_is_recovered_from_its_pag() {
    for (n, expected_classes) in [(3usize, 11usize), (4, 0)] {
        let all = mags(n);
        let mut classes: BTreeMap<Vec<bool>, Vec<Code>> = BTreeMap::new();
        for (c, s) in &all {
            classes.entry(s.clone()).or_default().push(c.clone());
        }
        if expected_classes > 0 {
            assert_eq!(classes.len(), expected_classes, "MEC count of MAGs on {n} nodes");
        }
        assert!(classes.len() >= 11);
        let (mut with_circles, mut multi) = (0, 0);
        for members in classes.values() {
            let marks: Vec<Option<(Endpoint, Endpoint)>> = (0..pairs(n).len())
                .map(|k| {
                    let states: BTreeSet<u8> = members.iter().map(|m| m[k]).collect();
                    let first = *states.iter().next().unwrap();
                    if first == 0 {
                        return None;
                    }
                    let side = |pick: fn((Endpoint, Endpoint)) -> Endpoint| {
                        let ends: BTreeSet<_> = states
                            .iter()
                            .map(|&s| format!("{:?}", pick(endpoints(s).unwrap())))
                            .collect();
                        if ends.len() == 1 {
                            pick(endpoints(first).unwrap())
                        } else {
                            Endpoint::Circle
                        }
                    };
                    Some((side(|p| p.0), side(|p| p.1)))
                })
                .collect();
            let pag = pag_from(n, &marks).expect("class PAG is legal");
            let sampler = CompletionSampler::new(pag, 10_000).unwrap();
            let report = sampler.validation_report();
            assert!(!report.ambiguous_global_class, "{n} nodes {members:?}");
            assert!(!sampler.hit_cap());
            let got: BTreeSet<Code> = sampler.map(|c| code_of(n, &c.graph)).collect();
            let want: BTreeSet<Code> = members.iter().cloned().collect();
            assert_eq!(got, want, "{n} nodes, marks {marks:?}");
            if marks.iter().flatten().any(|m| m.0 == Endpoint::Circle || m.1 == Endpoint::Circle) {
                with_circles += 1;
            }
            if members.len() > 1 {
                multi += 1;
            }
        }
        assert!(with_circles > 3 && multi > 3, "{with_circles} {multi}");
    }
}

#[test]
fn every_partially_marked_skeleton_on_three_nodes_gets_the_class_or_a_refusal() {
    let n = 3;
    let all = mags(n);
    let sig: BTreeMap<Code, Vec<bool>> = all.iter().cloned().collect();
    let k = pairs(n).len();
    let kinds = [Endpoint::Tail, Endpoint::Arrow, Endpoint::Circle];
    let mut options: Vec<Option<(Endpoint, Endpoint)>> = vec![None];
    for a in kinds {
        for b in kinds {
            if !(a == Endpoint::Tail && b == Endpoint::Tail) {
                options.push(Some((a, b)));
            }
        }
    }
    let compatible = |m: Endpoint, s: Endpoint| m == Endpoint::Circle || m == s;
    // A PAG encodes an unshielded collider exactly where both marks at the middle node are
    // arrowheads; a circle there leaves the triple a non-collider for every completion.
    let triples_kept = |marks: &[Option<(Endpoint, Endpoint)>], m: &Code| -> bool {
        let ps = pairs(n);
        let mark_at = |v: usize, w: usize, source: bool| -> bool {
            let k = ps.iter().position(|&p| p == (v.min(w), v.max(w))).unwrap();
            if source {
                let (a, b) = marks[k].unwrap();
                if v < w { a == Endpoint::Arrow } else { b == Endpoint::Arrow }
            } else {
                let (a, b) = endpoints(m[k]).unwrap();
                if v < w { a == Endpoint::Arrow } else { b == Endpoint::Arrow }
            }
        };
        let present = |v: usize, w: usize| {
            let k = ps.iter().position(|&p| p == (v.min(w), v.max(w))).unwrap();
            marks[k].is_some()
        };
        (0..n).all(|mid| {
            (0..n).all(|l| {
                (l + 1..n).all(|r| {
                    if l == mid || r == mid || !present(l, mid) || !present(r, mid) || present(l, r)
                    {
                        return true;
                    }
                    // `mark_at(v, w)`: arrowhead at the first node of the edge to the second.
                    let at_mid = |s: bool, o: usize| mark_at(mid, o, s);
                    (at_mid(true, l) && at_mid(true, r)) == (at_mid(false, l) && at_mid(false, r))
                })
            })
        })
    };
    let (mut single, mut ambiguous, mut empty) = (0, 0, 0);
    for code in 0..options.len().pow(u32::try_from(k).unwrap()) {
        let mut c = code;
        let marks: Vec<_> = (0..k)
            .map(|_| {
                let o = options[c % options.len()];
                c /= options.len();
                o
            })
            .collect();
        let Some(pag) = pag_from(n, &marks) else { continue };
        let compat: Vec<&Code> = sig
            .keys()
            .filter(|m| {
                m.iter().zip(&marks).all(|(&s, mark)| match (*mark, endpoints(s)) {
                    (None, None) => true,
                    (Some((a, b)), Some((sa, sb))) => compatible(a, sa) && compatible(b, sb),
                    _ => false,
                }) && triples_kept(&marks, m)
            })
            .collect();
        let classes: BTreeSet<&Vec<bool>> = compat.iter().map(|m| &sig[*m]).collect();
        let sampler = CompletionSampler::new(pag, 10_000).unwrap();
        let ambiguous_flag = sampler.validation_report().ambiguous_global_class;
        let got: BTreeSet<Code> = sampler.map(|c| code_of(n, &c.graph)).collect();
        match classes.len() {
            0 => {
                assert!(got.is_empty(), "{marks:?}");
                empty += 1;
            }
            1 => {
                let want: BTreeSet<Code> = compat.iter().map(|m| (*m).clone()).collect();
                assert_eq!(got, want, "{marks:?}");
                assert!(!ambiguous_flag);
                single += 1;
            }
            _ => {
                assert!(got.is_empty() && ambiguous_flag, "{marks:?} yielded {got:?}");
                ambiguous += 1;
            }
        }
    }
    assert!(single > 100 && empty > 20, "{single} {ambiguous} {empty}");
}
