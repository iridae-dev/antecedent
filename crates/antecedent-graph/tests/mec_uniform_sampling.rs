//! Exact class counts by clique-picking and uniform draws of class members.
//!
//! * The clique-picking count of random chordal undirected graphs on up to twelve nodes equals
//!   the number of members the depth-first completion enumerator streams.
//! * Closed forms: a path on `n` nodes has `n` members, a complete graph `n!`, disjoint
//!   components multiply, compelled edges add none.
//! * Draws are exactly uniform: a chi-square goodness-of-fit statistic over `10^5` draws of a
//!   24-member class stays far inside its null distribution; every draw is a member; a seed
//!   reproduces the draws exactly and another seed does not.
//! * PAG draws: every draw is a member the completion enumerator lists, the draws over a
//!   25-member class pass the same goodness-of-fit check, and a starved budget is refused.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{ExecutionContext, SearchLimits, SearchStop, splitmix64};
use antecedent_graph::{
    CompletionSampler, Cpdag, CpdagCompletionSampler, DenseNodeId, MecUniform, Pag,
    PagUniformError, dag_fingerprint, is_mec_member, mag_fingerprint, mec_size, sample_pag_members,
};

fn id(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn undirected(n: u32, edges: &[(u32, u32)]) -> Cpdag {
    let mut g = Cpdag::with_variables(n);
    for &(a, b) in edges {
        g.insert_undirected(id(a), id(b)).unwrap();
    }
    g
}

fn limits() -> SearchLimits {
    SearchLimits { operations: 50_000_000, depth: usize::MAX }
}

/// A random chordal graph on `n` nodes: each new node joins a random clique around an
/// earlier node (a perfect elimination order read backwards).
fn random_chordal(n: u32, state: &mut u64) -> Vec<(u32, u32)> {
    let size = usize::try_from(n).unwrap();
    let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); size];
    let mut edges = Vec::new();
    for v in 1..size {
        let anchor = usize::try_from(splitmix64(state) % u64::try_from(v).unwrap()).unwrap();
        let density = [30_u64, 60, 90][usize::try_from(splitmix64(state) % 3).unwrap()];
        let mut clique = vec![anchor];
        let candidates: Vec<usize> = adjacency[anchor].iter().copied().collect();
        for w in candidates {
            if clique.iter().all(|c| adjacency[*c].contains(&w))
                && splitmix64(state) % 100 < density
            {
                clique.push(w);
            }
        }
        for c in clique {
            adjacency[c].insert(v);
            adjacency[v].insert(c);
            edges.push((u32::try_from(c).unwrap(), u32::try_from(v).unwrap()));
        }
    }
    edges
}

fn enumerated(g: &Cpdag, cap: usize) -> Option<Vec<u64>> {
    let mut sampler = CpdagCompletionSampler::new(g.clone(), cap).unwrap();
    let members: Vec<u64> = sampler.by_ref().map(|c| dag_fingerprint(&c.graph)).collect();
    (!sampler.hit_cap()).then_some(members)
}

#[test]
fn clique_picking_counts_equal_enumeration_on_random_chordal_graphs() {
    let mut state = 0x5eed_u64;
    let mut compared = 0;
    for trial in 0..300_u32 {
        let n = 2 + trial % 11;
        let edges = random_chordal(n, &mut state);
        let g = undirected(n, &edges);
        let counted = mec_size(&g).unwrap();
        if counted > 60_000 {
            continue;
        }
        let Some(members) = enumerated(&g, 100_000) else {
            continue;
        };
        assert_eq!(counted, u128::try_from(members.len()).unwrap(), "n = {n}, edges = {edges:?}");
        compared += 1;
    }
    assert!(compared >= 200, "only {compared} graphs were compared");
}

#[test]
fn closed_form_class_sizes() {
    let path: Vec<(u32, u32)> = (0..11).map(|i| (i, i + 1)).collect();
    assert_eq!(mec_size(&undirected(12, &path)), Ok(12));
    let complete: Vec<(u32, u32)> =
        (0..7).flat_map(|a| ((a + 1)..7).map(move |b| (a, b))).collect();
    assert_eq!(mec_size(&undirected(7, &complete)), Ok(5_040));
    // A triangle beside a path of four: 3! * 4.
    let mixed = undirected(7, &[(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (5, 6)]);
    assert_eq!(mec_size(&mixed), Ok(24));
    // A compelled collider x -> z <- y beside an undirected path u - v - w.
    let mut compelled = undirected(6, &[(3, 4), (4, 5)]);
    compelled.insert_directed(id(0), id(2)).unwrap();
    compelled.insert_directed(id(1), id(2)).unwrap();
    let class = MecUniform::new(&compelled, limits(), &ExecutionContext::for_tests(0)).unwrap();
    assert_eq!(class.class_size(), 3);
    for dag in class.sample_members(30, 5, 0).unwrap() {
        assert!(is_mec_member(&compelled, &dag));
    }
    // 40 disjoint undirected edges: 2^40 members, counted without enumeration.
    let disjoint: Vec<(u32, u32)> = (0..40).map(|i| (2 * i, 2 * i + 1)).collect();
    assert_eq!(mec_size(&undirected(80, &disjoint)), Ok(1_u128 << 40));
}

/// Chi-square statistic of `counts` against the uniform law over `cells` cells.
fn chi_square(counts: &BTreeMap<u64, usize>, cells: usize, draws: usize) -> f64 {
    let expected = draws as f64 / cells as f64;
    let seen: f64 = counts.values().map(|&o| (o as f64 - expected).powi(2) / expected).sum();
    let unseen = (cells - counts.len()) as f64 * expected;
    seen + unseen
}

/// A bound the statistic exceeds with negligible probability under uniformity: the null mean
/// plus eight null standard deviations.
fn chi_square_bound(cells: usize) -> f64 {
    let df = (cells - 1) as f64;
    df + 8.0 * (2.0 * df).sqrt()
}

#[test]
fn cpdag_draws_are_uniform_members_reproduced_by_their_seed() {
    // Triangles {0,1,2} and {3,4,5} joined by the edge 2 - 3: 24 members.
    let g = undirected(6, &[(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)]);
    let members: BTreeSet<u64> = enumerated(&g, 1_000).unwrap().into_iter().collect();
    assert_eq!(members.len(), 24);
    let class = MecUniform::new(&g, limits(), &ExecutionContext::for_tests(0)).unwrap();
    assert_eq!(class.class_size(), 24);
    let draws = 100_000;
    let drawn = class.sample_members(draws, 11, 3).unwrap();
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for dag in &drawn {
        let print = dag_fingerprint(dag);
        assert!(members.contains(&print), "every draw is a member");
        *counts.entry(print).or_default() += 1;
    }
    let statistic = chi_square(&counts, members.len(), draws);
    assert!(statistic < chi_square_bound(members.len()), "chi-square {statistic}");

    let again: Vec<u64> =
        class.sample_members(200, 11, 3).unwrap().iter().map(dag_fingerprint).collect();
    let first: Vec<u64> = drawn[..200].iter().map(dag_fingerprint).collect();
    assert_eq!(again, first, "the seed reproduces the draws");
    let other: Vec<u64> =
        class.sample_members(200, 12, 3).unwrap().iter().map(dag_fingerprint).collect();
    assert_ne!(other, first, "another seed draws otherwise");
}

#[test]
fn a_complete_class_is_drawn_uniformly() {
    let complete: Vec<(u32, u32)> =
        (0..4).flat_map(|a| ((a + 1)..4).map(move |b| (a, b))).collect();
    let g = undirected(4, &complete);
    let class = MecUniform::new(&g, limits(), &ExecutionContext::for_tests(0)).unwrap();
    assert_eq!(class.class_size(), 24);
    let draws = 48_000;
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for dag in class.sample_members(draws, 2, 9).unwrap() {
        assert!(is_mec_member(&g, &dag));
        *counts.entry(dag_fingerprint(&dag)).or_default() += 1;
    }
    assert_eq!(counts.len(), 24);
    assert!(chi_square(&counts, 24, draws) < chi_square_bound(24));
}

/// Two disjoint chains `a o-o b o-o c`, each with five members (no collider at the middle):
/// 25 members.
fn two_circle_chains() -> Pag {
    let mut pag = Pag::with_variables(6);
    for (a, b) in [(0, 1), (1, 2), (3, 4), (4, 5)] {
        pag.insert_circle_circle(id(a), id(b)).unwrap();
    }
    pag
}

#[test]
fn pag_draws_are_uniform_over_the_enumerated_members() {
    let pag = two_circle_chains();
    let listed: Vec<Pag> =
        CompletionSampler::new(pag.clone(), 1_000).unwrap().map(|c| c.graph).collect();
    let members: BTreeSet<u64> = listed.iter().map(mag_fingerprint).collect();
    assert_eq!(members.len(), 25);
    let draws = 20_000;
    let ctx = ExecutionContext::for_tests(0);
    let sample = sample_pag_members(&pag, &listed[0], draws, (17, 4), limits(), &ctx).unwrap();
    assert_eq!(sample.members.len(), draws);
    assert!(sample.attempts >= u64::try_from(draws).unwrap());
    let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
    for member in &sample.members {
        let print = mag_fingerprint(member);
        assert!(members.contains(&print), "every draw is an enumerated member");
        *counts.entry(print).or_default() += 1;
    }
    let statistic = chi_square(&counts, members.len(), draws);
    assert!(statistic < chi_square_bound(members.len()), "chi-square {statistic}");

    let again = sample_pag_members(&pag, &listed[0], 100, (17, 4), limits(), &ctx).unwrap();
    let first: Vec<u64> = sample.members[..100].iter().map(mag_fingerprint).collect();
    let reprinted: Vec<u64> = again.members.iter().map(mag_fingerprint).collect();
    assert_eq!(first, reprinted, "the seed reproduces the draws");
}

#[test]
fn a_pag_class_beyond_the_budget_is_refused() {
    let pag = two_circle_chains();
    let reference = CompletionSampler::new(pag.clone(), 1).unwrap().next().unwrap().graph;
    let ctx = ExecutionContext::for_tests(0);
    let starved = SearchLimits { operations: 2_000, depth: usize::MAX };
    let refused = sample_pag_members(&pag, &reference, 10_000, (1, 1), starved, &ctx);
    assert!(matches!(
        refused,
        Err(PagUniformError::BudgetExhausted { stop: SearchStop::Operations, .. })
    ));
    // A reference that is not a completion is refused before anything is drawn.
    assert!(matches!(
        sample_pag_members(&pag, &pag, 1, (1, 1), limits(), &ctx),
        Err(PagUniformError::ReferenceNotMember)
    ));
}
