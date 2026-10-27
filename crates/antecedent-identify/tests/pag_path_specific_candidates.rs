//! Path-specific natural effects over a PAG: the ADMGs a MAG cannot tell apart.
//!
//! A MAG fixes a latent projection only up to its m-separation model, and a path-specific
//! effect is not a function of that model. These tests pin the three consequences the route
//! relies on, each against something that shares no code with the candidate search:
//!
//! * the latent projection of a randomly drawn structural model with hidden variables is
//!   always among the candidates of its own MAG, and every candidate has that MAG (the MAG
//!   is rebuilt here from the definition, by searching for inducing paths);
//! * two structural models with the same MAG can have different path-specific effects, so the
//!   functional of one MAG-shaped graph is wrong in the other model, and the route that reads
//!   the MAG through its candidates refuses instead of reporting that functional;
//! * when every candidate is identified, the functionals the envelope carries contain the
//!   effect of each model the MAG represents, computed by enumerating its exogenous states.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

#[path = "support/path_specific_scm.rs"]
mod path_specific_scm;

use std::collections::BTreeSet;

use antecedent_core::{CausalQuery, IdentificationStatus, PathSpecificEffectQuery};
use antecedent_graph::{Admg, Dag, Pag, latent_project};
use antecedent_identify::{GeneralizedAdjustmentIdentifier, carries_identified_mass};
use path_specific_scm::latent_scm::{LatentScm, Rng};
use path_specific_scm::{admg, d, evaluate, has_rule, identify, path_specific_truth, v};

fn linked(graph: &Admg, a: usize, b: usize) -> bool {
    graph.children(d(a)).contains(&d(b))
        || graph.children(d(b)).contains(&d(a))
        || graph.bidirected_neighbors(d(a)).contains(&d(b))
}

fn head_into(graph: &Admg, from: usize, into: usize) -> bool {
    graph.children(d(from)).contains(&d(into))
        || graph.bidirected_neighbors(d(from)).contains(&d(into))
}

/// Whether an inducing path joins `a` and `b`: every internal node is a collider on the path
/// and an ancestor of `a` or `b` (Richardson & Spirtes 2002, Def. 3.8).
fn inducing_path(graph: &Admg, a: usize, b: usize) -> bool {
    let n = graph.node_count();
    let ancestor: Vec<bool> = (0..n)
        .map(|x| x != a && x != b && (graph.reaches(d(x), d(a)) || graph.reaches(d(x), d(b))))
        .collect();
    fn extend(
        graph: &Admg,
        seen: &mut [bool],
        ancestor: &[bool],
        previous: usize,
        current: usize,
        end: usize,
    ) -> bool {
        for next in 0..graph.node_count() {
            if seen[next] || !linked(graph, current, next) {
                continue;
            }
            if !(head_into(graph, previous, current)
                && head_into(graph, next, current)
                && ancestor[current])
            {
                continue;
            }
            if next == end {
                return true;
            }
            seen[next] = true;
            let found = extend(graph, seen, ancestor, current, next, end);
            seen[next] = false;
            if found {
                return true;
            }
        }
        false
    }
    let mut seen = vec![false; n];
    seen[a] = true;
    for first in 0..n {
        if first == a || first == b || !linked(graph, a, first) {
            continue;
        }
        seen[first] = true;
        let found = extend(graph, &mut seen, &ancestor, a, first, b);
        seen[first] = false;
        if found {
            return true;
        }
    }
    false
}

/// Pair relations of the MAG of `graph`, by the definition: adjacent iff linked or joined by
/// an inducing path, a tail at the ancestor, bidirected when neither is an ancestor.
/// `0` absent, `1` lower index is the tail, `2` higher index is the tail, `3` bidirected.
fn mag_relations(graph: &Admg) -> Vec<u8> {
    let n = graph.node_count();
    let mut out = vec![0u8; n * n];
    for a in 0..n {
        for b in (a + 1)..n {
            if !(linked(graph, a, b) || inducing_path(graph, a, b)) {
                continue;
            }
            out[a * n + b] = if graph.reaches(d(a), d(b)) {
                1
            } else if graph.reaches(d(b), d(a)) {
                2
            } else {
                3
            };
        }
    }
    out
}

fn mag_of(graph: &Admg) -> Pag {
    let n = graph.node_count();
    let relations = mag_relations(graph);
    let mut mag = Pag::with_variables(n as u32);
    for a in 0..n {
        for b in (a + 1)..n {
            match relations[a * n + b] {
                1 => mag.insert_directed(d(a), d(b)).unwrap(),
                2 => mag.insert_directed(d(b), d(a)).unwrap(),
                3 => mag.insert_bidirected(d(a), d(b)).unwrap(),
                _ => {}
            }
        }
    }
    mag
}

fn edges(graph: &Admg) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    (graph.directed_edge_pairs_sorted(), graph.bidirected_edge_pairs_sorted())
}

fn shuffled(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    order
}

/// A random DAG over four observed and two hidden nodes, projected onto the observed ones.
fn random_projection(seed: u64) -> Admg {
    let mut rng = Rng::new(seed);
    let order = shuffled(&mut rng, 6);
    let mut dag = Dag::with_variables(6);
    for i in 0..6 {
        for j in (i + 1)..6 {
            if rng.unit() < 0.4 {
                dag.insert_directed(d(order[i]), d(order[j])).unwrap();
            }
        }
    }
    latent_project(&dag, &[d(0), d(1), d(2), d(3)]).unwrap()
}

#[test]
fn the_projection_of_a_hidden_variable_model_is_a_candidate_of_its_own_mag() {
    let identifier = GeneralizedAdjustmentIdentifier::new();
    let mut multi_candidate = 0;
    for seed in 0..24u64 {
        let projection = random_projection(seed);
        let mag = mag_of(&projection);
        let candidates = identifier.admg_candidates_of_mag(&mag).unwrap();
        assert!(!candidates.bounded, "seed {seed}: the candidate search must finish");
        let listed: BTreeSet<_> = candidates.members.iter().map(edges).collect();
        assert!(
            listed.contains(&edges(&projection)),
            "seed {seed}: the projection {:?} is missing from the candidates {listed:?}",
            edges(&projection)
        );
        let relations = mag_relations(&projection);
        for member in &candidates.members {
            assert_eq!(
                mag_relations(member),
                relations,
                "seed {seed}: candidate {:?} has another MAG",
                edges(member)
            );
        }
        assert_eq!(listed.len(), candidates.members.len(), "seed {seed}: candidates repeat");
        multi_candidate += usize::from(candidates.members.len() > 1);
    }
    assert!(multi_candidate > 0, "the sweep must reach MAGs with several candidates");
}

/// `T -> M -> Y`, `T -> Y` with every edge present: all three edges are invisible, so each
/// may hide a latent common cause.
fn tournament() -> Pag {
    let mut mag = Pag::with_variables(3);
    for (from, to) in [(0, 1), (1, 2), (0, 2)] {
        mag.insert_directed(d(from), d(to)).unwrap();
    }
    mag
}

fn effect_through_m() -> CausalQuery {
    CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)]))
}

/// Two models with one MAG and different effects: the functional that reads the MAG as the
/// ADMG is the effect in one and is wrong in the other, so the route refuses.
#[test]
fn two_models_with_one_mag_disagree_and_the_route_refuses() {
    let directed = [(0, 1), (1, 2), (0, 2)];
    let unconfounded = admg(3, &directed, &[]);
    let confounded = admg(3, &directed, &[(1, 2)]);
    let mag = tournament();
    // Both models have the complete tournament as MAG: every pair is adjacent and the lower
    // index is the tail, since the bidirected edge of the second joins a node to a descendant.
    let tournament_relations = vec![0, 1, 1, 0, 0, 1, 0, 0, 0];
    assert_eq!(mag_relations(&unconfounded), tournament_relations);
    assert_eq!(mag_relations(&confounded), tournament_relations);

    let identifier = GeneralizedAdjustmentIdentifier::new();
    let candidates = identifier.admg_candidates_of_mag(&mag).unwrap();
    let listed: BTreeSet<_> = candidates.members.iter().map(edges).collect();
    assert!(listed.contains(&edges(&unconfounded)), "{listed:?}");
    assert!(listed.contains(&edges(&confounded)), "{listed:?}");

    // The MAG read as the ADMG identifies the effect; it is right in the unconfounded model
    // and wrong in the confounded one.
    let naive = identify(&unconfounded, &effect_through_m());
    assert_eq!(naive.status, IdentificationStatus::NonparametricallyIdentified);
    let mut widest_gap: f64 = 0.0;
    for seed in 0..12u64 {
        let right = LatentScm::random(&mut Rng::new(seed), &[2, 2, 2], &directed, &[], 3);
        assert!(
            (evaluate(&naive, &right) - path_specific_truth(&right, 0, 2, &[1])).abs() < 1e-10,
            "seed {seed}: the functional must be the effect without confounding"
        );
        let wrong = LatentScm::random(&mut Rng::new(seed), &[2, 2, 2], &directed, &[(1, 2)], 3);
        widest_gap = widest_gap
            .max((evaluate(&naive, &wrong) - path_specific_truth(&wrong, 0, 2, &[1])).abs());
    }
    assert!(widest_gap > 1e-3, "the MAG reading must miss the confounded effect: {widest_gap}");

    // The confounded candidate is a recanting district: no functional is right for both.
    let refused = identify(&confounded, &effect_through_m());
    assert_eq!(refused.status, IdentificationStatus::NotIdentified);
    assert!(
        has_rule(&refused, "path_specific.recanting_district"),
        "{:?}",
        refused.derivation.steps
    );

    let envelope =
        identifier.identify_pag_path_specific_envelope(&mag, &effect_through_m()).unwrap();
    assert!(envelope.unidentified_weight.0 > 0.0, "{:?}", envelope.status);
    assert_eq!(envelope.identified_weight.0, 0.0);
    assert!(
        envelope
            .cases
            .iter()
            .all(|case| !carries_identified_mass(case.result.status)
                && case.result.estimands.is_empty()),
        "a refused candidate must leave no point behind"
    );
    assert!(
        envelope.cases.iter().any(|case| case
            .result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identify.path_specific_pag.unidentified_candidate")),
        "the refusal must name the candidate"
    );
}

/// One edge with two open ends: its three MAG completions each leave the effect of one end on
/// the other unidentified (an unobserved common cause cannot be excluded, or there is no
/// directed path), so the envelope carries no identified mass.
#[test]
fn an_edge_with_two_circle_marks_leaves_the_effect_unidentified() {
    let mut pag = Pag::with_variables(2);
    pag.insert_circle_circle(d(0), d(1)).unwrap();
    let query = CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(v(0), v(1)));
    let envelope = GeneralizedAdjustmentIdentifier::new()
        .identify_pag_path_specific_envelope(&pag, &query)
        .unwrap();
    assert!(envelope.cases.len() >= 2, "A->B, A<-B and A<->B: {}", envelope.cases.len());
    assert_eq!(envelope.identified_weight.0, 0.0);
    assert!(envelope.unidentified_weight.0 > 0.0);
    assert_eq!(envelope.status, IdentificationStatus::NotIdentified);
}

/// `C -> T -> M -> Y` with every mark decided. `C -> T` is the only invisible edge, and a
/// hidden common cause of `C` and `T` does not reach the effect of `T` on `Y`, so both
/// candidates are identified; the functionals the envelope carries contain the effect of each
/// of them.
#[test]
fn identified_candidates_cover_the_effect_of_every_model_the_mag_represents() {
    let directed = [(0, 1), (1, 2), (2, 3)];
    let mut mag = Pag::with_variables(4);
    for (from, to) in directed {
        mag.insert_directed(d(from), d(to)).unwrap();
    }
    let identifier = GeneralizedAdjustmentIdentifier::new();
    let candidates = identifier.admg_candidates_of_mag(&mag).unwrap();
    assert!(!candidates.bounded);
    let listed: BTreeSet<_> = candidates.members.iter().map(edges).collect();
    assert_eq!(
        listed,
        BTreeSet::from([edges(&admg(4, &directed, &[])), edges(&admg(4, &directed, &[(0, 1)])),])
    );

    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(1), v(3)).with_path_nodes([v(2)]),
    );
    let envelope = identifier.identify_pag_path_specific_envelope(&mag, &query).unwrap();
    assert_eq!(envelope.unidentified_weight.0, 0.0, "{:?}", envelope.critical_graph_features);
    assert!(envelope.identified_weight.0 > 0.0);
    for bidirected in [&[][..], &[(0, 1)][..]] {
        for seed in 0..6u64 {
            let scm = LatentScm::random(
                &mut Rng::new(40 + seed),
                &[2, 2, 2, 2],
                &directed,
                bidirected,
                3,
            );
            let truth = path_specific_truth(&scm, 1, 3, &[2]);
            assert!(
                envelope
                    .cases
                    .iter()
                    .any(|case| (evaluate(&case.result, &scm) - truth).abs() < 1e-9),
                "seed {seed}, latent edges {bidirected:?}: no carried functional equals the \
                 effect {truth}"
            );
        }
    }
}
