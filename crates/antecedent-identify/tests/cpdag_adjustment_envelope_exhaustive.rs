//! The CPDAG adjustment envelope against the back-door definition, over every Markov class of
//! three- and four-node DAGs.
//!
//! Reference: group the labelled DAGs by skeleton and unshielded colliders (Verma and Pearl),
//! build each class's essential graph as the edges every member orients identically, and for
//! each ordered treatment/outcome pair count the members that have some valid back-door set,
//! decided by Pearl's criterion with a moral-graph d-separation (no member of the set descends
//! from the treatment; the set separates treatment and outcome once the treatment's outgoing
//! arrows are cut). The envelope must hold exactly the class's members, one case per member, with
//! identified weight equal to that count, and its status must follow the mix: fully identified
//! when every member has a set and the reported sets agree, partially identified when every member
//! identifies with different reported sets (no single estimand), each reported set a valid back-door
//! set of its own member, graph dependent when only some do, and not identified
//! when none does.
//! The per-member sets differ across a class (not one shared set), which the envelope is
//! expected to report member by member rather than as a class-wide adjustment set; the test also
//! counts classes where a single set is valid in every member against classes that identify
//! member by member, to show that distinction is exercised.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{AverageEffectQuery, VariableId};
use antecedent_graph::{Cpdag, DenseNodeId, Endpoint};
use antecedent_identify::{GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier, IdentificationStatus};

type Arcs = Vec<(usize, usize)>;

fn acyclic(n: usize, arcs: &[(usize, usize)]) -> bool {
    let mut indegree = vec![0usize; n];
    for &(_, to) in arcs {
        indegree[to] += 1;
    }
    let mut ready: Vec<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    let mut seen = 0;
    while let Some(v) = ready.pop() {
        seen += 1;
        for &(from, to) in arcs {
            if from == v {
                indegree[to] -= 1;
                if indegree[to] == 0 {
                    ready.push(to);
                }
            }
        }
    }
    seen == n
}

fn all_dags(n: usize) -> Vec<Arcs> {
    let pairs: Vec<(usize, usize)> =
        (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
    let mut out = Vec::new();
    for code in 0..3usize.pow(u32::try_from(pairs.len()).unwrap()) {
        let (mut arcs, mut rest) = (Vec::new(), code);
        for &(i, j) in &pairs {
            match rest % 3 {
                1 => arcs.push((i, j)),
                2 => arcs.push((j, i)),
                _ => {}
            }
            rest /= 3;
        }
        if acyclic(n, &arcs) {
            arcs.sort_unstable();
            out.push(arcs);
        }
    }
    out
}

fn reach(n: usize, arcs: &[(usize, usize)], of: &[usize], forward: bool) -> Vec<bool> {
    let mut r = vec![false; n];
    for &v in of {
        r[v] = true;
    }
    loop {
        let mut grew = false;
        for &(a, b) in arcs {
            let (from, to) = if forward { (a, b) } else { (b, a) };
            if r[from] && !r[to] {
                r[to] = true;
                grew = true;
            }
        }
        if !grew {
            return r;
        }
    }
}

fn d_separated(n: usize, arcs: &[(usize, usize)], x: usize, y: usize, z: &[usize]) -> bool {
    let mut seeds = vec![x, y];
    seeds.extend_from_slice(z);
    let keep = reach(n, arcs, &seeds, false);
    let mut adj = vec![vec![false; n]; n];
    for &(a, b) in arcs {
        if keep[a] && keep[b] {
            adj[a][b] = true;
            adj[b][a] = true;
        }
    }
    for child in 0..n {
        let parents: Vec<usize> =
            arcs.iter().filter(|&&(f, t)| t == child && keep[f] && keep[t]).map(|a| a.0).collect();
        for &a in &parents {
            for &b in &parents {
                if a != b {
                    adj[a][b] = true;
                }
            }
        }
    }
    let mut seen = vec![false; n];
    let mut stack = vec![x];
    seen[x] = true;
    while let Some(v) = stack.pop() {
        if v == y {
            return false;
        }
        for w in 0..n {
            if keep[w] && adj[v][w] && !seen[w] && !z.contains(&w) {
                seen[w] = true;
                stack.push(w);
            }
        }
    }
    true
}

fn backdoor_valid(n: usize, arcs: &[(usize, usize)], t: usize, y: usize, z: &[usize]) -> bool {
    let de = reach(n, arcs, &[t], true);
    if z.iter().any(|&v| de[v]) {
        return false;
    }
    let cut: Arcs = arcs.iter().copied().filter(|&(from, _)| from != t).collect();
    d_separated(n, &cut, t, y, z)
}

fn valid_sets(n: usize, arcs: &[(usize, usize)], t: usize, y: usize) -> Vec<Vec<usize>> {
    let pool: Vec<usize> = (0..n).filter(|&v| v != t && v != y).collect();
    (0..(1usize << pool.len()))
        .map(|mask| {
            pool.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, &v)| v).collect()
        })
        .filter(|z: &Vec<usize>| backdoor_valid(n, arcs, t, y, z))
        .collect()
}

type Invariant = (BTreeSet<(usize, usize)>, BTreeSet<(usize, usize, usize)>);

fn invariant(arcs: &[(usize, usize)]) -> Invariant {
    let adjacent = |a: usize, b: usize| arcs.contains(&(a, b)) || arcs.contains(&(b, a));
    let skeleton = arcs.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
    let mut colliders = BTreeSet::new();
    for &(a, m) in arcs {
        for &(b, m2) in arcs {
            if m == m2 && a < b && !adjacent(a, b) {
                colliders.insert((a, m, b));
            }
        }
    }
    (skeleton, colliders)
}

fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

fn run(n: usize) -> (usize, usize, usize, usize, usize) {
    let mut classes: BTreeMap<Invariant, Vec<Arcs>> = BTreeMap::new();
    for dag in all_dags(n) {
        classes.entry(invariant(&dag)).or_default().push(dag);
    }
    let identifier = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig { max_completions: 4096, ..Default::default() },
    };
    let (mut full, mut none, mut partial, mut member_only, mut differing) = (0, 0, 0, 0, 0);
    for members in classes.values() {
        let mut cpdag = Cpdag::with_variables(u32::try_from(n).unwrap());
        let mut undirected = BTreeSet::new();
        let mut directed = BTreeSet::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let forward = members.iter().filter(|m| m.contains(&(a, b))).count();
                let backward = members.iter().filter(|m| m.contains(&(b, a))).count();
                if forward == members.len() {
                    directed.insert((a, b));
                } else if backward == members.len() {
                    directed.insert((b, a));
                } else if forward + backward == members.len() {
                    undirected.insert((a, b));
                }
            }
        }
        for &(a, b) in &undirected {
            cpdag.insert_undirected(id(a), id(b)).unwrap();
        }
        for &(a, b) in &directed {
            cpdag.insert_directed(id(a), id(b)).unwrap();
        }
        for t in 0..n {
            for y in 0..n {
                if t == y {
                    continue;
                }
                let per_member: Vec<Vec<Vec<usize>>> =
                    members.iter().map(|m| valid_sets(n, m, t, y)).collect();
                let identified = per_member.iter().filter(|s| !s.is_empty()).count();
                let common = per_member
                    .iter()
                    .map(|sets| sets.iter().cloned().collect::<BTreeSet<_>>())
                    .reduce(|a, b| a.intersection(&b).cloned().collect())
                    .unwrap();
                let query = AverageEffectQuery::binary_ate(
                    VariableId::from_raw(u32::try_from(t).unwrap()),
                    VariableId::from_raw(u32::try_from(y).unwrap()),
                );
                let envelope = identifier.identify_cpdag_envelope(&cpdag, &query).unwrap();
                let tag = format!("n={n} members={members:?} t={t} y={y}");
                assert_eq!(envelope.cases.len(), members.len(), "{tag}");
                let got_members: BTreeSet<Arcs> = envelope
                    .cases
                    .iter()
                    .map(|c| {
                        let mut arcs: Arcs = c
                            .graph
                            .edges()
                            .map(|e| match (e.at_a, e.at_b) {
                                (Endpoint::Tail, Endpoint::Arrow) => (e.a.raw() as usize, e.b.raw() as usize),
                                _ => (e.b.raw() as usize, e.a.raw() as usize),
                            })
                            .collect();
                        arcs.sort_unstable();
                        arcs
                    })
                    .collect();
                assert_eq!(got_members, members.iter().cloned().collect(), "{tag}");
                for case in &envelope.cases {
                    let mut arcs: Arcs = case
                        .graph
                        .edges()
                        .map(|e| match (e.at_a, e.at_b) {
                            (Endpoint::Tail, Endpoint::Arrow) => (e.a.raw() as usize, e.b.raw() as usize),
                            _ => (e.b.raw() as usize, e.a.raw() as usize),
                        })
                        .collect();
                    arcs.sort_unstable();
                    let has_set = !valid_sets(n, &arcs, t, y).is_empty();
                    assert_eq!(case.result.estimands.is_empty(), !has_set, "{tag} member {arcs:?}");
                    if let Some(est) = case.result.estimands.first() {
                        let z: Vec<usize> =
                            est.adjustment_set.iter().map(|v| v.raw() as usize).collect();
                        assert!(backdoor_valid(n, &arcs, t, y, &z), "{tag} member {arcs:?} set {z:?}");
                    }
                }
                let want_id = identified as f64;
                let want_un = (members.len() - identified) as f64;
                assert!((envelope.identified_weight.0 - want_id).abs() < 1e-12, "{tag}");
                assert!((envelope.unidentified_weight.0 - want_un).abs() < 1e-12, "{tag}");
                if identified == members.len() {
                    // Every member identifies. The envelope keeps one estimand only when the
                    // members' adjustment sets agree; otherwise it holds none and is partial.
                    let sets: BTreeSet<Vec<usize>> = envelope
                        .cases
                        .iter()
                        .map(|c| {
                            let mut z: Vec<usize> = c.result.estimands[0]
                                .adjustment_set
                                .iter()
                                .map(|v| v.raw() as usize)
                                .collect();
                            z.sort_unstable();
                            z
                        })
                        .collect();
                    if sets.len() == 1 {
                        assert_eq!(envelope.status, IdentificationStatus::NonparametricallyIdentified, "{tag}");
                        assert!(envelope.invariant.is_some(), "{tag}");
                    } else {
                        assert_eq!(envelope.status, IdentificationStatus::PartiallyIdentified, "{tag}");
                        assert!(envelope.invariant.is_none(), "{tag}");
                        differing += 1;
                    }
                    full += 1;
                    if common.is_empty() {
                        member_only += 1;
                    }
                } else if identified == 0 {
                    assert_eq!(envelope.status, IdentificationStatus::NotIdentified, "{tag}");
                    none += 1;
                } else {
                    assert_eq!(envelope.status, IdentificationStatus::GraphDependent, "{tag}");
                    partial += 1;
                }
            }
        }
    }
    (full, none, partial, member_only, differing)
}

#[test]
fn cpdag_envelope_counts_members_with_a_valid_back_door_set_on_three_nodes() {
    let (full, none, partial, _, _) = run(3);
    assert!(full > 10 && none > 5 && partial > 0, "{full} {none} {partial}");
}

#[test]
fn cpdag_envelope_counts_members_with_a_valid_back_door_set_on_four_nodes() {
    let (full, none, partial, member_only, differing) = run(4);
    assert!(full > 100 && none > 50 && partial > 5 && differing > 5, "{full} {none} {partial}");
    println!("fully identified {full}; no shared set {member_only}; members' sets differ {differing}");
}
