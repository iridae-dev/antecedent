//! The completion enumeration and the CPDAG admission test against every small mixed graph.
//!
//! Reference: group every labelled DAG on `n` nodes by the Verma-Pearl invariant (same
//! skeleton, same unshielded colliders), and build each class's essential graph from the
//! definition (an edge is directed exactly when every member orients it the same way). A
//! partially directed graph is then a CPDAG exactly when it equals one of those essential
//! graphs, and its completions are exactly that class's members. Every one of the `4^(n choose 2)`
//! partially directed graphs on three and four nodes (each pair absent, undirected, or directed
//! either way) is submitted: the enumerator must return the class for a CPDAG and refuse
//! everything else with the not-a-CPDAG detail. The enumeration shares no search code with
//! this reference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{ExecutionContext, SearchLimits};
use antecedent_graph::{Cpdag, DenseNodeId};
use antecedent_identify::sid::cpdag_completion::{
    CPDAG_NOT_A_CPDAG_DETAIL, CpdagCompletionInput, enumerate_cpdag_completions,
};

type Arcs = Vec<(u32, u32)>;

const BIG: SearchLimits = SearchLimits { operations: 10_000_000, depth: 64 };

fn id(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn acyclic(n: u32, arcs: &[(u32, u32)]) -> bool {
    let mut indegree = vec![0usize; n as usize];
    for &(_, to) in arcs {
        indegree[to as usize] += 1;
    }
    let mut ready: Vec<u32> = (0..n).filter(|&v| indegree[v as usize] == 0).collect();
    let mut seen = 0;
    while let Some(v) = ready.pop() {
        seen += 1;
        for &(from, to) in arcs {
            if from == v {
                indegree[to as usize] -= 1;
                if indegree[to as usize] == 0 {
                    ready.push(to);
                }
            }
        }
    }
    seen == n
}

fn pairs(n: u32) -> Vec<(u32, u32)> {
    (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect()
}

fn all_dags(n: u32) -> Vec<Arcs> {
    let pairs = pairs(n);
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

type Invariant = (BTreeSet<(u32, u32)>, BTreeSet<(u32, u32, u32)>);

fn invariant(n: u32, arcs: &[(u32, u32)]) -> Invariant {
    let adjacent = |a: u32, b: u32| arcs.contains(&(a, b)) || arcs.contains(&(b, a));
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

/// Essential-graph description: per skeleton pair, `0` undirected, `1` low -> high, `2` high -> low.
fn essential(members: &[Arcs], skeleton: &BTreeSet<(u32, u32)>) -> BTreeMap<(u32, u32), u8> {
    skeleton
        .iter()
        .map(|&(a, b)| {
            let forward = members.iter().all(|m| m.contains(&(a, b)));
            let backward = members.iter().all(|m| m.contains(&(b, a)));
            ((a, b), if forward { 1 } else if backward { 2 } else { 0 })
        })
        .collect()
}

fn run(n: u32) -> (usize, usize, usize) {
    let mut classes: BTreeMap<Invariant, Vec<Arcs>> = BTreeMap::new();
    for arcs in all_dags(n) {
        classes.entry(invariant(n, &arcs)).or_default().push(arcs);
    }
    // Essential graph (as a state per pair) -> the class members.
    let mut by_essential: BTreeMap<Vec<u8>, Vec<Arcs>> = BTreeMap::new();
    for (key, members) in &classes {
        let state = essential(members, &key.0);
        let code = pairs(n).iter().map(|p| state.get(p).map_or(3, |&s| s)).collect::<Vec<u8>>();
        // Distinct classes have distinct essential graphs.
        assert!(by_essential.insert(code, members.clone()).is_none());
    }
    let pairs = pairs(n);
    let (mut accepted, mut refused, mut unbuildable) = (0, 0, 0);
    // State per pair: 0 absent, 1 undirected, 2 low -> high, 3 high -> low.
    for code in 0..4usize.pow(u32::try_from(pairs.len()).unwrap()) {
        let mut rest = code;
        let mut graph = Cpdag::with_variables(n);
        let mut key = Vec::new();
        let mut ok = true;
        for &(a, b) in &pairs {
            let state = rest % 4;
            rest /= 4;
            let result = match state {
                0 => Ok(()),
                1 => graph.insert_undirected(id(a), id(b)).map(|_| ()),
                2 => graph.insert_directed(id(a), id(b)).map(|_| ()),
                _ => graph.insert_directed(id(b), id(a)).map(|_| ()),
            };
            ok &= result.is_ok();
            // The essential-graph code: 3 absent, 0 undirected, 1 low -> high, 2 high -> low.
            key.push(match state {
                0 => 3,
                1 => 0,
                2 => 1,
                _ => 2,
            });
        }
        if !ok {
            unbuildable += 1;
            continue;
        }
        let outcome = enumerate_cpdag_completions(
            &CpdagCompletionInput::new(graph),
            BIG,
            &ExecutionContext::for_tests(1),
        );
        match (by_essential.get(&key), outcome) {
            (Some(members), Ok(enumeration)) => {
                accepted += 1;
                let mut got: Vec<Arcs> = enumeration
                    .completions
                    .iter()
                    .map(|c| {
                        let mut arcs: Arcs = c.edges.iter().map(|(a, b)| (a.raw(), b.raw())).collect();
                        arcs.sort_unstable();
                        arcs
                    })
                    .collect();
                got.sort();
                let mut want = members.clone();
                want.sort();
                assert_eq!(got, want, "n={n} graph code {key:?}");
                assert!(enumeration.is_exportable() && enumeration.not_enumerated == 0);
            }
            (None, Err(error)) => {
                refused += 1;
                assert_eq!(error.detail, CPDAG_NOT_A_CPDAG_DETAIL, "n={n} graph code {key:?}");
            }
            (Some(_), Err(error)) => panic!("n={n}: a CPDAG {key:?} was refused: {}", error.detail),
            (None, Ok(_)) => panic!("n={n}: {key:?} is not an essential graph yet was enumerated"),
        }
    }
    assert_eq!(accepted, by_essential.len(), "every class is submitted exactly once");
    (accepted, refused, unbuildable)
}

#[test]
fn every_three_node_partially_directed_graph_is_a_cpdag_or_refused() {
    let (accepted, refused, unbuildable) = run(3);
    assert_eq!(accepted, 11);
    assert_eq!(accepted + refused + unbuildable, 64);
    assert!(refused > 20);
}

#[test]
fn every_four_node_partially_directed_graph_is_a_cpdag_or_refused() {
    let (accepted, refused, unbuildable) = run(4);
    assert_eq!(accepted, 185);
    assert_eq!(accepted + refused + unbuildable, 4096);
    assert!(refused > 1000);
}
