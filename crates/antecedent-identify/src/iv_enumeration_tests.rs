//! Exhaustive check of the instrumental-variable graphical criterion.
//!
//! The reference is a closed-form restatement that shares no code with the identifier: with an
//! empty conditioning set, two nodes of a DAG are d-connected exactly when they have a common
//! ancestor (counting each node as its own ancestor), because a path with no collider is open
//! and every collider-free path is a fork or a chain through such an ancestor. A node `z` is
//! then an instrument for `t -> y` when it is not a descendant of `t`, shares an ancestor with
//! `t`, and shares no ancestor with `y` once `t`'s outgoing edges are removed. Every labelled
//! DAG on four nodes and every ordered (treatment, outcome) pair is compared.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, CausalQuery, VariableId};
use antecedent_graph::{Dag, DenseNodeId};

use super::*;
use crate::result::IdentificationStatus;

const N: usize = 4;

fn acyclic(arcs: &[(usize, usize)]) -> bool {
    let mut indegree = [0usize; N];
    for &(_, to) in arcs {
        indegree[to] += 1;
    }
    let mut ready: Vec<usize> = (0..N).filter(|&v| indegree[v] == 0).collect();
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
    seen == N
}

/// `reach[a][b]` is true when `a` is `b` or an ancestor of `b`.
fn ancestor_or_self(arcs: &[(usize, usize)]) -> [[bool; N]; N] {
    let mut reach = [[false; N]; N];
    for (v, row) in reach.iter_mut().enumerate() {
        row[v] = true;
    }
    for &(from, to) in arcs {
        reach[from][to] = true;
    }
    for k in 0..N {
        for i in 0..N {
            for j in 0..N {
                if reach[i][k] && reach[k][j] {
                    reach[i][j] = true;
                }
            }
        }
    }
    reach
}

fn share_ancestor(reach: &[[bool; N]; N], a: usize, b: usize) -> bool {
    (0..N).any(|w| reach[w][a] && reach[w][b])
}

fn expected_instruments(arcs: &[(usize, usize)], t: usize, y: usize) -> Vec<u32> {
    let full = ancestor_or_self(arcs);
    let cut: Vec<(usize, usize)> = arcs.iter().copied().filter(|&(from, _)| from != t).collect();
    let mutilated = ancestor_or_self(&cut);
    (0..N)
        .filter(|&z| z != t && z != y)
        .filter(|&z| !full[t][z])
        .filter(|&z| share_ancestor(&full, z, t))
        .filter(|&z| !share_ancestor(&mutilated, z, y))
        .map(|z| u32::try_from(z).unwrap())
        .collect()
}

fn reported_instruments(
    id: &InstrumentalVariableIdentifier,
    g: &Dag,
    t: u32,
    y: u32,
) -> (IdentificationStatus, Vec<u32>) {
    let prepared = id.prepare(g).unwrap();
    let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(t),
        VariableId::from_raw(y),
    ));
    let result = id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
    let mut found: Vec<u32> = result.estimands.iter().map(|e| e.instruments[0].raw()).collect();
    found.sort_unstable();
    (result.status, found)
}

fn dag_of(arcs: &[(usize, usize)]) -> Dag {
    let mut g = Dag::with_variables(u32::try_from(N).unwrap());
    for &(from, to) in arcs {
        g.insert_directed(
            DenseNodeId::from_raw(u32::try_from(from).unwrap()),
            DenseNodeId::from_raw(u32::try_from(to).unwrap()),
        )
        .unwrap();
    }
    g
}

#[test]
fn search_agrees_with_the_common_ancestor_criterion_on_every_four_node_dag() {
    let pairs: Vec<(usize, usize)> =
        (0..N).flat_map(|i| ((i + 1)..N).map(move |j| (i, j))).collect();
    let id = InstrumentalVariableIdentifier::new();
    let (mut dags, mut with_instrument, mut without) = (0, 0, 0);
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
        if !acyclic(&arcs) {
            continue;
        }
        dags += 1;
        let g = dag_of(&arcs);
        for t in 0..N {
            for y in 0..N {
                if t == y {
                    continue;
                }
                let expected = expected_instruments(&arcs, t, y);
                let (status, found) = reported_instruments(
                    &id,
                    &g,
                    u32::try_from(t).unwrap(),
                    u32::try_from(y).unwrap(),
                );
                assert_eq!(found, expected, "arcs={arcs:?} t={t} y={y}");
                if expected.is_empty() {
                    without += 1;
                    assert_eq!(status, IdentificationStatus::NotIdentified, "arcs={arcs:?}");
                } else {
                    with_instrument += 1;
                    assert_eq!(
                        status,
                        IdentificationStatus::IdentifiedUnderParametricRestrictions,
                        "arcs={arcs:?}"
                    );
                }
            }
        }
    }
    assert_eq!(dags, 543, "labelled DAGs on four nodes");
    // The comparison must exercise both verdicts, or agreement would be vacuous.
    assert!(with_instrument > 100 && without > 100, "{with_instrument} / {without}");
}

#[test]
fn a_candidate_sharing_a_cause_with_the_outcome_is_not_an_instrument() {
    // Z <- U -> Y with Z -> T -> Y: relevant, but the open path Z <- U -> Y violates
    // exogeneity once T's out-edges are cut, so neither Z nor U qualifies.
    let g = dag_of(&[(0, 1), (1, 2), (3, 0), (3, 2)]);
    let (status, found) = reported_instruments(&InstrumentalVariableIdentifier::new(), &g, 1, 2);
    assert!(found.is_empty());
    assert_eq!(status, IdentificationStatus::NotIdentified);
}

#[test]
fn the_result_limit_truncates_the_listing_in_candidate_order() {
    // Z1 -> T <- Z2, U -> T, U -> Y, T -> Y has two valid instruments; a limit of one reports
    // the first candidate only. The limit is a caller-set output bound, not a validity test.
    let g = dag_of(&[(0, 2), (1, 2), (2, 3)]);
    let full = InstrumentalVariableIdentifier::new();
    assert_eq!(reported_instruments(&full, &g, 2, 3).1, vec![0, 1]);
    let capped =
        InstrumentalVariableIdentifier { config: InstrumentSearchConfig { max_results: 1 } };
    assert_eq!(reported_instruments(&capped, &g, 2, 3).1, vec![0]);
}
