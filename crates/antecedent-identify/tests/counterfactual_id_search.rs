//! 2.2B X8: counterfactual identification of the effect of treatment on the
//! treated on bounded ADMGs, checked against exact enumerated latent-variable
//! structural models, explicit two-model non-identifiability witnesses, the
//! shared search budget, the declared bounds and invariances.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::result_large_err,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities; exact bit comparisons are the assertion"
)]

#[path = "support/latent_scm.rs"]
mod latent_scm;

use std::sync::Arc;

use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, EdgeRoute, ExecutionContext, ExogenousCoupling,
    MemoryBudget, RegimeId, SearchLimits, SearchStop, Value, VariableId, WorldId, WorldSpec,
};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_DEFAULT_LIMITS, COUNTERFACTUAL_ID_MAX_DEPTH,
    COUNTERFACTUAL_ID_MAX_OPERATIONS, COUNTERFACTUAL_ID_MEMORY_BYTES, CfValue,
    CounterfactualFunctional, CounterfactualIdDerivation, CounterfactualIdProblem,
    CounterfactualIdRefusal, CounterfactualIdShape, CounterfactualObstruction,
    decide_counterfactual_id, evaluate_counterfactual_functional,
};
use latent_scm::{LatentScm, ResponseType, Rng};

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn binary(n: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0, 1.0]; n]
}

fn law(cards: &[usize], probabilities: Vec<f64>) -> ExactDiscreteLaw {
    let axes: Vec<DiscreteAxis> = cards
        .iter()
        .enumerate()
        .map(|(i, &c)| DiscreteAxis {
            variable: v(u32::try_from(i).unwrap()),
            values: (0..c).map(|l| Value::f64(l as f64)).collect::<Vec<_>>().into(),
        })
        .collect();
    ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        axes,
        probabilities,
        "latent-scm",
        LawTolerance::default(),
    )
    .unwrap()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn decide(
    problem: &CounterfactualIdProblem,
    query: &CounterfactualEventQuery,
) -> Result<CounterfactualIdDerivation, CounterfactualIdRefusal> {
    decide_counterfactual_id(
        problem,
        query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        &ctx(),
    )
}

fn ett(x: u32, active: f64, observed: f64, y: u32, level: f64) -> CounterfactualEventQuery {
    CounterfactualEventQuery::effect_on_treated(v(x), active, observed, v(y), level).unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs()))
}

/// The ETT from the functional, for every outcome level, checked against the
/// enumerated counterfactual truth of `scm`.
fn assert_matches_truth(
    scm: &LatentScm,
    directed: &[(u32, u32)],
    bidirected: &[(u32, u32)],
    x: u32,
    y: u32,
) -> usize {
    let levels: Vec<Vec<f64>> =
        scm.cards.iter().map(|&c| (0..c).map(|l| l as f64).collect()).collect();
    let problem = CounterfactualIdProblem::new(levels, directed, bidirected).unwrap();
    let joint = law(&scm.cards, scm.observational());
    let mut checked = 0;
    for active in 0..scm.cards[x as usize] {
        for observed in 0..scm.cards[x as usize] {
            if active == observed {
                continue;
            }
            let query = ett(x, active as f64, observed as f64, y, 0.0);
            let derivation = decide(&problem, &query).unwrap();
            let point =
                evaluate_counterfactual_functional(&problem, &derivation, &joint, &ctx()).unwrap();
            let truths = scm.ett_numerators(x as usize, active, observed, y as usize);
            for (level, numerator) in &point.numerators {
                let truth = truths[*level as usize];
                assert!(
                    close(*numerator, truth),
                    "numerator {numerator} != enumerated truth {truth} at level {level}"
                );
                checked += 1;
            }
            let truth = truths[0] / truths.iter().sum::<f64>();
            assert!(close(point.probability, truth), "{} != {truth}", point.probability);
        }
    }
    checked
}

// ---------------------------------------------------------------- positives

/// Front door `X -> M -> Y`, `X <-> Y`: `P(Y_x = y | X = x') = sum_m P(m | x) P(y | m, x')`.
fn frontdoor() -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    (vec![(0, 1), (1, 2)], vec![(0, 2)])
}

#[test]
fn frontdoor_ett_is_identified_as_the_frontdoor_functional() {
    let (directed, bidirected) = frontdoor();
    let problem = CounterfactualIdProblem::new(binary(3), &directed, &bidirected).unwrap();
    let derivation = decide(&problem, &ett(0, 1.0, 0.0, 2, 1.0)).unwrap();
    assert!(matches!(derivation.shape, CounterfactualIdShape::EffectOnTreated { .. }));
    assert_eq!(derivation.numerators.len(), 2, "one functional per level of Y");
    // Two district terms per level: P_x(m) and P_m(x', y).
    let terms = derivation.terms();
    assert_eq!(terms.len(), 4);
    let signatures: Vec<(Vec<u32>, Vec<u32>)> = terms
        .iter()
        .map(|t| {
            (t.intervention.iter().map(|p| p.0).collect(), t.event.iter().map(|p| p.0).collect())
        })
        .collect();
    assert!(signatures.contains(&(vec![0], vec![1])), "P_x(m): {signatures:?}");
    assert!(signatures.contains(&(vec![1], vec![0, 2])), "P_m(x', y): {signatures:?}");
    let graph = derivation.counterfactual_graph.as_ref().unwrap();
    // X (observed x'), M_x (free), Y_x (event): X and Y_x share the latent.
    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(graph.districts.len(), 2);
    // Numerically: the front-door ETT formula computed straight from the law.
    let mut rng = Rng::new(11);
    let scm = LatentScm::random(&mut rng, &[2, 2, 2], &[(0, 1), (1, 2)], &[(0, 2)], 2);
    let joint = scm.observational();
    let p = |x: usize, m: usize, y: usize| joint[x * 4 + m * 2 + y];
    let p_x = |x: usize| {
        (0..2).flat_map(|m| (0..2).map(move |y| (m, y))).map(|(m, y)| p(x, m, y)).sum::<f64>()
    };
    let p_xm = |x: usize, m: usize| p(x, m, 0) + p(x, m, 1);
    let formula: f64 = (0..2).map(|m| (p_xm(1, m) / p_x(1)) * (p(0, m, 1) / p_xm(0, m))).sum();
    let point =
        evaluate_counterfactual_functional(&problem, &derivation, &law(&[2, 2, 2], joint), &ctx())
            .unwrap();
    assert!(close(point.probability, formula), "{} vs {formula}", point.probability);
}

#[test]
fn frontdoor_ett_matches_the_enumerated_latent_scm() {
    let (directed, bidirected) = frontdoor();
    let mut rng = Rng::new(2026);
    let mut checked = 0;
    for cards in [[2, 2, 2], [3, 2, 2], [2, 3, 3]] {
        let d: Vec<(usize, usize)> =
            directed.iter().map(|&(a, b)| (a as usize, b as usize)).collect();
        let b: Vec<(usize, usize)> =
            bidirected.iter().map(|&(a, b)| (a as usize, b as usize)).collect();
        let scm = LatentScm::random(&mut rng, &cards, &d, &b, 3);
        checked += assert_matches_truth(&scm, &directed, &bidirected, 0, 2);
    }
    assert!(checked >= 20, "{checked}");
}

#[test]
fn mediator_confounded_with_outcome_matches_the_enumerated_truth() {
    // X -> M -> Y, X -> Y, M <-> Y: the hidden edge does not touch X.
    let directed = [(0, 1), (1, 2), (0, 2)];
    let bidirected = [(1, 2)];
    let problem = CounterfactualIdProblem::new(binary(3), &directed, &bidirected).unwrap();
    let derivation = decide(&problem, &ett(0, 1.0, 0.0, 2, 1.0)).unwrap();
    // P(x') and P_x(m, y): the ETT equals the interventional law (X is unconfounded).
    let signatures: Vec<(Vec<u32>, Vec<u32>)> = derivation
        .terms()
        .iter()
        .map(|t| {
            (t.intervention.iter().map(|p| p.0).collect(), t.event.iter().map(|p| p.0).collect())
        })
        .collect();
    assert!(signatures.contains(&(vec![0], vec![1, 2])), "{signatures:?}");
    assert!(signatures.contains(&(vec![], vec![0])), "{signatures:?}");
    let mut rng = Rng::new(77);
    let mut checked = 0;
    for cards in [[2, 2, 2], [2, 3, 2], [3, 2, 3]] {
        let scm = LatentScm::random(&mut rng, &cards, &[(0, 1), (1, 2), (0, 2)], &[(1, 2)], 3);
        checked += assert_matches_truth(&scm, &directed, &bidirected, 0, 2);
    }
    assert!(checked >= 20, "{checked}");
}

/// A random ADMG over `n` nodes: a random DAG under the identity order plus
/// random bidirected edges.
fn random_graph(rng: &mut Rng, n: usize) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if rng.unit() < 0.45 {
                directed.push((a as u32, b as u32));
            }
            if rng.unit() < 0.35 {
                bidirected.push((a as u32, b as u32));
            }
        }
    }
    (directed, bidirected)
}

#[test]
fn every_identified_ett_on_random_admgs_matches_the_enumerated_truth() {
    let mut rng = Rng::new(4242);
    let (mut identified, mut conflicts, mut hedges, mut checked, mut nontrivial) = (0, 0, 0, 0, 0);
    for round in 0..160 {
        let n = 3 + round % 3;
        let (mut directed, bidirected) = random_graph(&mut rng, n);
        // X precedes Y and reaches it, so the counterfactual is not the factual law.
        let x = rng.below(n - 1) as u32;
        let y = x + 1 + rng.below(n - 1 - x as usize) as u32;
        let reaches = |edges: &[(u32, u32)]| {
            let mut seen = vec![x];
            let mut i = 0;
            while i < seen.len() {
                let a = seen[i];
                seen.extend(
                    edges
                        .iter()
                        .filter(|e| e.0 == a && !seen.contains(&e.1))
                        .map(|e| e.1)
                        .collect::<Vec<_>>(),
                );
                i += 1;
            }
            seen.contains(&y)
        };
        if !reaches(&directed) {
            directed.push((x, y));
        }
        let cards: Vec<usize> =
            (0..n).map(|i| if i == x as usize { 2 } else { 2 + rng.below(2) }).collect();
        let levels: Vec<Vec<f64>> =
            cards.iter().map(|&c| (0..c).map(|l| l as f64).collect()).collect();
        let problem = CounterfactualIdProblem::new(levels, &directed, &bidirected).unwrap();
        match decide(&problem, &ett(x, 1.0, 0.0, y, 0.0)) {
            Ok(_) => {
                identified += 1;
                let d: Vec<(usize, usize)> =
                    directed.iter().map(|&(a, b)| (a as usize, b as usize)).collect();
                let b: Vec<(usize, usize)> =
                    bidirected.iter().map(|&(a, b)| (a as usize, b as usize)).collect();
                let scm = LatentScm::random(&mut rng, &cards, &d, &b, 2);
                checked += assert_matches_truth(&scm, &directed, &bidirected, x, y);
                // Non-vacuous: the counterfactual differs from the observed conditional.
                let observed = scm.ett_numerators(x as usize, 0, 0, y as usize);
                let counterfactual = scm.ett_numerators(x as usize, 1, 0, y as usize);
                if (observed[0] - counterfactual[0]).abs() > 1e-6 {
                    nontrivial += 1;
                }
            }
            Err(refusal) if refusal.detail == "counterfactual_id.conflicting_subscripts" => {
                conflicts += 1;
                refusal.obstruction.as_ref().unwrap().verify().unwrap();
            }
            Err(refusal) if refusal.detail == "counterfactual_id.counterfactual_hedge" => {
                hedges += 1;
                refusal.obstruction.as_ref().unwrap().verify().unwrap();
            }
            Err(other) => panic!("unexpected refusal {other}"),
        }
    }
    eprintln!(
        "identified {identified}, conflicts {conflicts}, hedges {hedges}, checked {checked}, \
         nontrivial {nontrivial}"
    );
    assert!(nontrivial >= 50, "nontrivial {nontrivial}");
    assert!(identified >= 60, "identified {identified}");
    assert!(conflicts >= 20, "conflicts {conflicts}");
    assert!(checked >= 200, "checked {checked}");
    // Recorded: no ID hedge was ever reached on the effect-on-the-treated shape.
    assert_eq!(hedges, 0, "an ID hedge on the ETT shape: record it");
}

// ---------------------------------------------------------------- negatives

/// A model on `X`, `Y` (both binary) and optionally `Z`, with one binary latent
/// `U` (`P(U = 1) = 1/2`) on `X <-> Y`.
fn two_level_scm(
    cards: Vec<usize>,
    parents: Vec<Vec<usize>>,
    latents: Vec<(usize, usize, f64)>,
    tables: Vec<Vec<(f64, Vec<usize>)>>,
) -> LatentScm {
    LatentScm {
        cards,
        parents,
        latents,
        types: tables
            .into_iter()
            .map(|list| {
                list.into_iter()
                    .map(|(probability, table)| ResponseType { probability, table })
                    .collect()
            })
            .collect(),
    }
}

fn assert_two_models_agree_on_p_v_and_differ_on_the_ett(
    first: &LatentScm,
    second: &LatentScm,
    x: usize,
    y: usize,
) {
    let (a, b) = (first.observational(), second.observational());
    assert_eq!(a.len(), b.len());
    for (p, q) in a.iter().zip(&b) {
        assert!((p - q).abs() <= 1e-15, "the models disagree on P(V): {a:?} vs {b:?}");
    }
    assert!(a.iter().all(|p| *p > 0.0) || a.iter().any(|p| *p > 0.0));
    let (t1, t2) = (first.ett(x, 1, 0, y, 1), second.ett(x, 1, 0, y, 1));
    assert!((t1 - t2).abs() > 0.1, "the models agree on the ETT: {t1} vs {t2}");
}

#[test]
fn the_bow_arc_refuses_with_conflicting_subscripts() {
    let problem = CounterfactualIdProblem::new(binary(2), &[(0, 1)], &[(0, 1)]).unwrap();
    let refusal = decide(&problem, &ett(0, 1.0, 0.0, 1, 1.0)).unwrap_err();
    assert_eq!(refusal.code, "cross_world_not_identified");
    assert_eq!(refusal.detail, "counterfactual_id.conflicting_subscripts");
    let Some(obstruction) = refusal.obstruction.as_deref() else { panic!("no obstruction") };
    obstruction.verify().unwrap();
    let CounterfactualObstruction::ConflictingSubscripts { variable, subscript, event, .. } =
        obstruction
    else {
        panic!("not a conflict")
    };
    assert_eq!(*variable, 0);
    assert_eq!(*subscript, CfValue::Level(1.0f64.to_bits()));
    assert_eq!(*event, Some(CfValue::Level(0.0f64.to_bits())));
}

#[test]
fn two_models_on_the_bow_arc_agree_on_p_v_and_differ_on_the_ett() {
    // X = U; model one: Y = X; model two: Y = U. Same observational law (X = Y
    // = U), but among the untreated (U = 0) Y_{x=1} is 1 in one and 0 in the other.
    let parents = vec![vec![], vec![0]];
    let latents = vec![(0, 1, 0.5)];
    // X's table: inputs (U); Y's table: inputs (X, U).
    let x_table = vec![(1.0, vec![0, 1])];
    let first = two_level_scm(
        vec![2, 2],
        parents.clone(),
        latents.clone(),
        vec![x_table.clone(), vec![(1.0, vec![0, 0, 1, 1])]],
    );
    let second =
        two_level_scm(vec![2, 2], parents, latents, vec![x_table, vec![(1.0, vec![0, 1, 0, 1])]]);
    assert_two_models_agree_on_p_v_and_differ_on_the_ett(&first, &second, 0, 1);
    assert!((first.ett(0, 1, 0, 1, 1) - 1.0).abs() < 1e-15);
    assert!(second.ett(0, 1, 0, 1, 1).abs() < 1e-15);
}

#[test]
fn the_instrument_graph_refuses_with_cross_world_not_identified() {
    // Z -> X -> Y, X <-> Y.
    let problem = CounterfactualIdProblem::new(binary(3), &[(0, 1), (1, 2)], &[(1, 2)]).unwrap();
    let refusal = decide(&problem, &ett(1, 1.0, 0.0, 2, 1.0)).unwrap_err();
    assert_eq!(refusal.code, "cross_world_not_identified");
    assert_eq!(refusal.detail, "counterfactual_id.conflicting_subscripts");
    refusal.obstruction.as_ref().unwrap().verify().unwrap();
}

#[test]
fn two_models_on_the_instrument_graph_agree_on_p_v_and_differ_on_the_ett() {
    // Z ~ Bern(1/2), U ~ Bern(1/2), X = Z and U; Y = X in one model and
    // Y = X and U in the other. X = 1 forces U = 1, so both give Y = X on
    // every observed state; among X = 0 with U = 0, Y_{x=1} is 1 or 0.
    let parents = vec![vec![], vec![0], vec![1]];
    let latents = vec![(1, 2, 0.5)];
    let z = vec![(0.5, vec![0]), (0.5, vec![1])];
    // X inputs (Z, U): and.
    let x = vec![(1.0, vec![0, 0, 0, 1])];
    let first = two_level_scm(
        vec![2, 2, 2],
        parents.clone(),
        latents.clone(),
        vec![z.clone(), x.clone(), vec![(1.0, vec![0, 0, 1, 1])]],
    );
    let second =
        two_level_scm(vec![2, 2, 2], parents, latents, vec![z, x, vec![(1.0, vec![0, 0, 0, 1])]]);
    assert_two_models_agree_on_p_v_and_differ_on_the_ett(&first, &second, 1, 2);
}

#[test]
fn a_conflict_obstruction_names_the_district_and_pair_and_replays() {
    let problem = CounterfactualIdProblem::new(binary(3), &[(0, 1), (1, 2)], &[(1, 2)]).unwrap();
    let query = ett(1, 1.0, 0.0, 2, 1.0);
    let first = decide(&problem, &query).unwrap_err();
    let second = decide(&problem, &query).unwrap_err();
    assert_eq!(first, second, "the obstruction is deterministic and replays");
    let Some(CounterfactualObstruction::ConflictingSubscripts { graph, variable, .. }) =
        first.obstruction.as_deref()
    else {
        panic!("no conflict")
    };
    assert_eq!(graph.districts.len(), 1);
    assert!(graph.nodes.iter().any(|n| n.variable == *variable && n.value.is_some()));
    assert!(graph.nodes.iter().any(|n| n.subscript.iter().any(|(s, _)| s == variable)));
    // A tampered record fails its own check: the pair no longer conflicts.
    let mut tampered = first.obstruction.clone().unwrap();
    if let CounterfactualObstruction::ConflictingSubscripts { subscript, event, .. } =
        tampered.as_mut()
    {
        *subscript = event.unwrap();
    }
    assert!(tampered.verify().is_err());
    let mut split = first.obstruction.clone().unwrap();
    if let CounterfactualObstruction::ConflictingSubscripts { graph, .. } = split.as_mut() {
        graph.bidirected.clear();
    }
    assert!(split.verify().is_err(), "a district that is not connected is refused");
}

#[test]
fn an_event_contradicting_itself_within_one_world_is_an_exact_zero() {
    let (directed, bidirected) = frontdoor();
    let problem = CounterfactualIdProblem::new(binary(3), &directed, &bidirected).unwrap();
    let worlds =
        || vec![WorldSpec::new([], []).unwrap(), WorldSpec::new([(v(0), 1.0)], []).unwrap()];
    let at = |w: u8, var: u32, level: f64| {
        CounterfactualEvent::new(WorldId::new(w), v(var), level).unwrap()
    };
    let given = [at(0, 0, 0.0)];
    let joint = law(&[2, 2, 2], {
        let mut rng = Rng::new(5);
        LatentScm::random(&mut rng, &[2, 2, 2], &[(0, 1), (1, 2)], &[(0, 2)], 2).observational()
    });
    // Y_x read at two levels in one world; X_x read away from the level its world sets.
    for event in [vec![at(1, 2, 0.0), at(1, 2, 1.0)], vec![at(1, 0, 0.0)]] {
        let query = CounterfactualEventQuery::new(
            worlds(),
            ExogenousCoupling::SharedLatentExogenous,
            event,
            given,
        )
        .unwrap();
        let derivation = decide(&problem, &query).unwrap();
        assert_eq!(derivation.shape, CounterfactualIdShape::ExactZero);
        assert_eq!(derivation.numerators, vec![(0, CounterfactualFunctional::Zero)]);
        let point =
            evaluate_counterfactual_functional(&problem, &derivation, &joint, &ctx()).unwrap();
        assert_eq!(point.probability.to_bits(), 0.0f64.to_bits(), "an exact zero");
        assert!(point.conditioning_probability > 0.0);
        assert_eq!(point.effect, None);
    }
    // A contradictory conditioning event is a null event: refused, not zero.
    let query = CounterfactualEventQuery::new(
        worlds(),
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 1.0)],
        [at(0, 0, 0.0), at(0, 0, 1.0)],
    )
    .unwrap();
    let refusal = decide(&problem, &query).unwrap_err();
    assert_eq!(
        (refusal.code, refusal.detail),
        ("invalid_argument", "counterfactual_id.invalid_query")
    );
}

#[test]
fn shapes_outside_the_contract_refuse_with_their_details() {
    let (directed, bidirected) = frontdoor();
    let problem = CounterfactualIdProblem::new(binary(3), &directed, &bidirected).unwrap();
    let at = |w: u8, var: u32, level: f64| {
        CounterfactualEvent::new(WorldId::new(w), v(var), level).unwrap()
    };
    let two = || vec![WorldSpec::new([], []).unwrap(), WorldSpec::new([(v(0), 1.0)], []).unwrap()];
    let expect = |query: &CounterfactualEventQuery, code: &str, detail: &str| {
        let refusal = decide(&problem, query).unwrap_err();
        assert_eq!((refusal.code, refusal.detail), (code, detail), "{}", refusal.message);
    };
    // A joint of two potential outcomes, no conditioning.
    let joint = CounterfactualEventQuery::new(
        two(),
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 1.0), at(0, 2, 0.0)],
        [],
    )
    .unwrap();
    expect(&joint, "route_not_supported", "counterfactual_id.query_outside_contract");
    // The abduction coupling of the fixed-DAG cell.
    let abduced = CounterfactualEventQuery::new(
        two(),
        ExogenousCoupling::SharedAbducedExogenous,
        [at(1, 2, 1.0)],
        [at(0, 0, 0.0)],
    )
    .unwrap();
    expect(&abduced, "route_not_supported", "counterfactual_id.query_outside_contract");
    // A world that routes an edge: path-specific, deferred to 2.3.
    let routed = CounterfactualEventQuery::new(
        vec![
            WorldSpec::new([], []).unwrap(),
            WorldSpec::new(
                [(v(0), 1.0)],
                [EdgeRoute { parent: v(1), child: v(2), source: WorldId::new(0) }],
            )
            .unwrap(),
        ],
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 1.0)],
        [at(0, 0, 0.0)],
    )
    .unwrap();
    expect(&routed, "route_not_supported", "counterfactual_id.path_specific_deferred");
    // x = x' (the observed conditional), an unknown level, an unknown variable.
    let same = CounterfactualEventQuery::new(
        two(),
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 1.0)],
        [at(0, 0, 1.0)],
    )
    .unwrap();
    expect(&same, "invalid_argument", "counterfactual_id.invalid_query");
    expect(&ett(0, 1.0, 0.0, 2, 5.0), "invalid_argument", "counterfactual_id.invalid_query");
    expect(&ett(0, 1.0, 0.0, 9, 1.0), "invalid_argument", "counterfactual_id.invalid_query");
    // A cyclic graph is outside the graph contract.
    let cyclic = CounterfactualIdProblem::new(binary(2), &[(0, 1), (1, 0)], &[]).unwrap_err();
    assert_eq!(
        (cyclic.code, cyclic.detail),
        ("cell_not_licensed", "counterfactual_id.graph_outside_contract")
    );
}

// ---------------------------------------------------------------- budget

#[test]
fn the_one_budget_stops_with_receipts_never_a_verdict() {
    let (directed, bidirected) = frontdoor();
    let problem = CounterfactualIdProblem::new(binary(3), &directed, &bidirected).unwrap();
    let query = ett(0, 1.0, 0.0, 2, 1.0);
    let full = decide(&problem, &query).unwrap();
    let total = full.search.operations_consumed;
    assert!(total > 10, "{total}");
    assert_eq!(full.search.operations_limit, COUNTERFACTUAL_ID_DEFAULT_LIMITS.operations);
    // Operations: one short of the whole decision (every outcome level shares it).
    let short =
        SearchLimits { operations: total - 1, depth: COUNTERFACTUAL_ID_DEFAULT_LIMITS.depth };
    let refusal =
        decide_counterfactual_id(&problem, &query, short, COUNTERFACTUAL_ID_MEMORY_BYTES, &ctx())
            .unwrap_err();
    assert_eq!(
        (refusal.code, refusal.detail),
        ("transport_budget_cancel", "counterfactual_id.budget")
    );
    let receipt = refusal.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_limit, total - 1);
    assert_eq!(receipt.operations_consumed, Some(total - 1));
    // The first level finished within the shared budget; the second did not.
    assert_eq!(receipt.explored.len(), 1, "{receipt:?}");
    assert_eq!(receipt.unevaluated.len(), 1, "{receipt:?}");
    assert!(refusal.obstruction.is_none(), "a stop is never an obstruction");
    // The exact budget suffices.
    let exact = SearchLimits { operations: total, depth: COUNTERFACTUAL_ID_DEFAULT_LIMITS.depth };
    assert!(
        decide_counterfactual_id(&problem, &query, exact, COUNTERFACTUAL_ID_MEMORY_BYTES, &ctx())
            .is_ok()
    );
    // Depth: the decision reached depth `full.search.depth_reached`.
    let deep = full.search.depth_reached;
    let shallow = SearchLimits { operations: 10_000, depth: deep - 1 };
    let refusal =
        decide_counterfactual_id(&problem, &query, shallow, COUNTERFACTUAL_ID_MEMORY_BYTES, &ctx())
            .unwrap_err();
    assert_eq!(refusal.receipt.unwrap().stop, SearchStop::Depth);
    let enough = SearchLimits { operations: 10_000, depth: deep };
    assert!(
        decide_counterfactual_id(&problem, &query, enough, COUNTERFACTUAL_ID_MEMORY_BYTES, &ctx())
            .is_ok()
    );
    // Memory: the context's hard limit caps the search.
    let mut small = ctx();
    small.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(64) };
    let refusal = decide_counterfactual_id(
        &problem,
        &query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        &small,
    )
    .unwrap_err();
    let receipt = refusal.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(64));
    // Cancellation before entry.
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let refusal = decide_counterfactual_id(
        &problem,
        &query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        &cancelled,
    )
    .unwrap_err();
    assert_eq!(refusal.receipt.unwrap().stop, SearchStop::Cancelled);
}

#[test]
fn the_memory_charged_is_cumulative_across_the_decision() {
    let (directed, bidirected) = frontdoor();
    let query = ett(0, 1.0, 0.0, 2, 1.0);
    // The smallest cap a decision fits under (exact, by bisection).
    let smallest = |problem: &CounterfactualIdProblem| {
        let fits = |cap: u64| {
            decide_counterfactual_id(problem, &query, COUNTERFACTUAL_ID_DEFAULT_LIMITS, cap, &ctx())
                .is_ok()
        };
        let (mut low, mut high) = (1u64, 1u64 << 20);
        assert!(fits(high));
        while low < high {
            let mid = (low + high) / 2;
            if fits(mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        low
    };
    let both = CounterfactualIdProblem::new(binary(3), &directed, &bidirected).unwrap();
    let whole = smallest(&both);
    let refusal = decide_counterfactual_id(
        &both,
        &query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        whole - 1,
        &ctx(),
    )
    .unwrap_err();
    let receipt = refusal.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(whole - 1));
    // One outcome level alone (Y has the single level 1) needs strictly less: the
    // first level's terms stay charged while the second level is derived.
    let one_level = CounterfactualIdProblem::new(
        vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0]],
        &directed,
        &bidirected,
    )
    .unwrap();
    let single = smallest(&one_level);
    assert!(whole > single, "cumulative cap {whole}, one level {single}");
}

// ---------------------------------------------------------------- bounds

#[test]
fn bounds_admit_the_cap_and_refuse_one_more() {
    let bounds = |r: CounterfactualIdRefusal| {
        assert_eq!(
            (r.code, r.detail),
            ("route_not_supported", "counterfactual_id.bounds_exceeded")
        );
    };
    // Six variables pass; seven refuse.
    let chain: Vec<(u32, u32)> = (0..5).map(|i| (i, i + 1)).collect();
    let six = CounterfactualIdProblem::new(binary(6), &chain, &[(0, 5)]).unwrap();
    assert!(decide(&six, &ett(1, 1.0, 0.0, 5, 1.0)).is_ok());
    bounds(CounterfactualIdProblem::new(binary(7), &chain, &[]).unwrap_err());
    // Four levels pass; five refuse.
    let four = CounterfactualIdProblem::new(
        vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0, 3.0]],
        &[(0, 1)],
        &[],
    )
    .unwrap();
    assert!(decide(&four, &ett(0, 1.0, 0.0, 1, 3.0)).is_ok());
    bounds(
        CounterfactualIdProblem::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0, 3.0, 4.0]],
            &[(0, 1)],
            &[],
        )
        .unwrap_err(),
    );
    // Search limits at the caps pass; one more refuses before any search.
    let at_cap = SearchLimits {
        operations: COUNTERFACTUAL_ID_MAX_OPERATIONS,
        depth: COUNTERFACTUAL_ID_MAX_DEPTH,
    };
    let query = ett(0, 1.0, 0.0, 1, 1.0);
    assert!(
        decide_counterfactual_id(&four, &query, at_cap, COUNTERFACTUAL_ID_MEMORY_BYTES, &ctx())
            .is_ok()
    );
    for over in [
        SearchLimits { operations: COUNTERFACTUAL_ID_MAX_OPERATIONS + 1, ..at_cap },
        SearchLimits { depth: COUNTERFACTUAL_ID_MAX_DEPTH + 1, ..at_cap },
    ] {
        bounds(
            decide_counterfactual_id(&four, &query, over, COUNTERFACTUAL_ID_MEMORY_BYTES, &ctx())
                .unwrap_err(),
        );
    }
}

// ---------------------------------------------------------------- invariances

fn point_of(
    problem: &CounterfactualIdProblem,
    query: &CounterfactualEventQuery,
    joint: &ExactDiscreteLaw,
) -> f64 {
    let derivation = decide(problem, query).unwrap();
    evaluate_counterfactual_functional(problem, &derivation, joint, &ctx()).unwrap().probability
}

#[test]
fn the_answer_is_invariant_to_presentation() {
    let (directed, bidirected) = frontdoor();
    let mut rng = Rng::new(99);
    let scm = LatentScm::random(&mut rng, &[2, 3, 2], &[(0, 1), (1, 2)], &[(0, 2)], 3);
    let joint_probs = scm.observational();
    let levels = vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0], vec![0.0, 1.0]];
    let problem = CounterfactualIdProblem::new(levels.clone(), &directed, &bidirected).unwrap();
    let query = ett(0, 1.0, 0.0, 2, 1.0);
    let base = point_of(&problem, &query, &law(&[2, 3, 2], joint_probs.clone()));

    // Bidirected orientation and duplicate edges: the same canonical problem, bit for bit.
    let flipped =
        CounterfactualIdProblem::new(levels.clone(), &[(1, 2), (0, 1), (0, 1)], &[(2, 0), (0, 2)])
            .unwrap();
    assert_eq!(flipped, problem);
    assert_eq!(
        point_of(&flipped, &query, &law(&[2, 3, 2], joint_probs.clone())).to_bits(),
        base.to_bits()
    );

    // Node order: relabel (X, M, Y) -> (Y', X', M') = variables (1, 2, 0).
    let relabel = [1u32, 2, 0];
    let cards = [2usize, 2, 3]; // new variable 0 is old Y, 1 is old X, 2 is old M
    let mut permuted = vec![0.0; joint_probs.len()];
    for x in 0..2 {
        for m in 0..3 {
            for y in 0..2 {
                // new layout: (Y, X, M)
                permuted[y * 6 + x * 3 + m] = joint_probs[x * 6 + m * 2 + y];
            }
        }
    }
    let moved = CounterfactualIdProblem::new(
        vec![vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0, 2.0]],
        &[(relabel[0], relabel[1]), (relabel[1], relabel[2])],
        &[(relabel[0], relabel[2])],
    )
    .unwrap();
    let moved_point =
        point_of(&moved, &ett(relabel[0], 1.0, 0.0, relabel[2], 1.0), &law(&cards, permuted));
    assert!(close(moved_point, base), "{moved_point} vs {base}");

    // Value relabelling: M's levels renamed 0,1,2 -> 7,-3,0.5 everywhere.
    let renamed_levels = vec![vec![0.0, 1.0], vec![7.0, -3.0, 0.5], vec![0.0, 1.0]];
    let renamed = CounterfactualIdProblem::new(renamed_levels, &directed, &bidirected).unwrap();
    let axes = vec![
        DiscreteAxis { variable: v(0), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) },
        DiscreteAxis {
            variable: v(1),
            values: Arc::from([Value::f64(7.0), Value::f64(-3.0), Value::f64(0.5)]),
        },
        DiscreteAxis { variable: v(2), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) },
    ];
    let renamed_law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        axes,
        joint_probs.clone(),
        "latent-scm",
        LawTolerance::default(),
    )
    .unwrap();
    assert_eq!(point_of(&renamed, &query, &renamed_law).to_bits(), base.to_bits());

    // An isolated extra variable with an independent law.
    let with_extra = CounterfactualIdProblem::new(
        vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0], vec![0.0, 1.0], vec![0.0, 1.0]],
        &directed,
        &bidirected,
    )
    .unwrap();
    let extra: Vec<f64> = joint_probs.iter().flat_map(|p| [p * 0.3, p * 0.7]).collect();
    let extra_point = point_of(&with_extra, &query, &law(&[2, 3, 2, 2], extra));
    assert!(close(extra_point, base), "{extra_point} vs {base}");

    // Event order: conjunctions are canonical, so the query text is order free.
    let at = |w: u8, var: u32, level: f64| {
        CounterfactualEvent::new(WorldId::new(w), v(var), level).unwrap()
    };
    let worlds =
        || vec![WorldSpec::new([], []).unwrap(), WorldSpec::new([(v(0), 1.0)], []).unwrap()];
    let one = CounterfactualEventQuery::new(
        worlds(),
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 0.0), at(1, 2, 1.0)],
        [at(0, 0, 0.0)],
    )
    .unwrap();
    let two = CounterfactualEventQuery::new(
        worlds(),
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 1.0), at(1, 2, 0.0)],
        [at(0, 0, 0.0)],
    )
    .unwrap();
    assert_eq!(decide(&problem, &one).unwrap(), decide(&problem, &two).unwrap());
}
