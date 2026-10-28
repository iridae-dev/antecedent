//! CPDAG completion against brute-force Markov equivalence classes of all small DAGs.
//!
//! The reference groups every labelled DAG on `n` nodes by the Verma-Pearl invariant (same
//! skeleton and same unshielded colliders), builds each class's CPDAG straight from the
//! definition (an edge is directed exactly when every member orients it the same way, otherwise
//! undirected), and then requires the completion sampler to stream exactly the members of that
//! class, `is_mec_member` to accept exactly those DAGs among all DAGs of the same size, and no
//! completion to be repeated. It uses no orientation rule and no shared search code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_graph::{Cpdag, CpdagCompletionSampler, Dag, DenseNodeId, is_mec_member};

type Arcs = Vec<(usize, usize)>;

fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

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
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
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

/// (skeleton, unshielded colliders) of a DAG.
fn invariant(n: usize, arcs: &[(usize, usize)]) -> (BTreeSet<(usize, usize)>, BTreeSet<(usize, usize, usize)>) {
    let adjacent = |a: usize, b: usize| arcs.contains(&(a, b)) || arcs.contains(&(b, a));
    let skeleton = arcs.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
    let mut colliders = BTreeSet::new();
    for b in 0..n {
        for a in 0..n {
            for c in (a + 1)..n {
                if a != b && c != b && arcs.contains(&(a, b)) && arcs.contains(&(c, b)) && !adjacent(a, c) {
                    colliders.insert((a, b, c));
                }
            }
        }
    }
    (skeleton, colliders)
}

fn to_dag(n: usize, arcs: &[(usize, usize)]) -> Dag {
    let mut g = Dag::with_variables(u32::try_from(n).unwrap());
    for &(a, b) in arcs {
        g.insert_directed(id(a), id(b)).unwrap();
    }
    g
}

fn completions(cpdag: &Cpdag, cap: usize) -> Vec<Arcs> {
    let sampler = CpdagCompletionSampler::new(cpdag.clone(), cap).unwrap();
    let mut out: Vec<Arcs> = sampler
        .map(|c| {
            let mut arcs: Arcs = c
                .graph
                .directed_edge_pairs_sorted()
                .into_iter()
                .map(|(a, b)| (a as usize, b as usize))
                .collect();
            arcs.sort_unstable();
            arcs
        })
        .collect();
    out.sort();
    out
}

fn check(n: usize, class_stride: usize, membership_stride: usize) -> (usize, usize) {
    let dags = all_dags(n);
    let mut classes: BTreeMap<_, Vec<Arcs>> = BTreeMap::new();
    for arcs in &dags {
        classes.entry(invariant(n, arcs)).or_default().push(arcs.clone());
    }
    let (mut checked, mut undirected_total) = (0, 0);
    for (index, (key, members)) in classes.iter().enumerate() {
        if index % class_stride != 0 {
            continue;
        }
        // CPDAG from the definition: directed iff all members agree.
        let mut cpdag = Cpdag::with_variables(u32::try_from(n).unwrap());
        for &(a, b) in &key.0 {
            let forward = members.iter().all(|m| m.contains(&(a, b)));
            let backward = members.iter().all(|m| m.contains(&(b, a)));
            match (forward, backward) {
                (true, false) => cpdag.insert_directed(id(a), id(b)).unwrap(),
                (false, true) => cpdag.insert_directed(id(b), id(a)).unwrap(),
                (false, false) => {
                    cpdag.insert_undirected(id(a), id(b)).unwrap();
                    undirected_total += 1;
                }
                (true, true) => unreachable!("an edge has one orientation per DAG"),
            }
        }
        let mut expected: Vec<Arcs> = members.clone();
        expected.sort();
        let got = completions(&cpdag, 100_000);
        assert_eq!(got, expected, "class {key:?}");
        // Membership over every DAG of this size (strided for the larger sizes).
        for (position, arcs) in dags.iter().enumerate() {
            if position % membership_stride != 0 && !members.contains(arcs) {
                continue;
            }
            let want = members.contains(arcs);
            assert_eq!(is_mec_member(&cpdag, &to_dag(n, arcs)), want, "class {key:?} dag {arcs:?}");
        }
        checked += 1;
    }
    (checked, undirected_total)
}

#[test]
fn completions_equal_the_class_for_every_three_node_class() {
    let (checked, undirected) = check(3, 1, 1);
    assert_eq!(checked, 11);
    assert!(undirected > 5);
}

#[test]
fn completions_equal_the_class_for_every_four_node_class() {
    let (checked, undirected) = check(4, 1, 1);
    assert_eq!(checked, 185);
    assert!(undirected > 200);
}

#[test]
fn completions_equal_the_class_for_a_stride_of_five_node_classes() {
    let (checked, undirected) = check(5, 17, 211);
    assert!(checked > 50 && undirected > 200, "{checked} {undirected}");
}

#[test]
fn the_completion_cap_yields_distinct_members() {
    // Undirected 4-cycle-free tree: a star with three leaves has 4 members.
    let mut cpdag = Cpdag::with_variables(4);
    for leaf in 1..4 {
        cpdag.insert_undirected(id(0), id(leaf)).unwrap();
    }
    let all = completions(&cpdag, 100);
    assert_eq!(all.len(), 4);
    let capped = completions(&cpdag, 2);
    assert_eq!(capped.len(), 2);
    let distinct: BTreeSet<_> = capped.iter().collect();
    assert_eq!(distinct.len(), 2);
    assert!(capped.iter().all(|c| all.contains(c)));
}
