//! 2.2B B1 (X2): conditional transport against enumerated latent-SCM truth.
//!
//! The oracle never touches the symbolic layer: each bidirected edge is a binary
//! latent, the target population shifts the mechanism of every selected node,
//! and the truth is `P*(y | do(x), w) = P*(y, w | do(x)) / P*(w | do(x))`
//! enumerated in the target model. The route's answer is the checked formula,
//! bound to a catalog of every source experiment plus the target observational
//! joint, evaluated by the exact provider.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "binary levels and small graph coordinates"
)]

#[path = "common/admg_conditional_scm.rs"]
mod admg_conditional_scm;

use admg_conditional_scm::{Scm, query, request, v};
use antecedent_core::{ExecutionContext, Value};
use antecedent_estimate::{EstimationError, prepare_exact_admg_conditional_transport};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, BoundConditionalTransportFunctional,
    ConditionalTransportDecision, ConditionalTransportQuery, decide_admg_conditional_transport,
};

enum Outcome {
    Identified,
    NotCertified,
}

/// Decide, and when identified compare the point at every level of `x ∪ w`.
fn check(scm: &Scm, y: &[usize], x: &[usize], w: &[usize]) -> Outcome {
    let ctx = ExecutionContext::for_tests(3);
    let (catalog, data) = scm.catalog_and_laws();
    let decision = decide_admg_conditional_transport(
        &scm.diagram(),
        &query(y, x, w),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap();
    let functional = match decision {
        ConditionalTransportDecision::Identified(bound) => bound,
        ConditionalTransportDecision::NotCertified(_) => return Outcome::NotCertified,
        other => panic!("a full catalog is never missing evidence or exhausted: {other:?}"),
    };
    for level in 0..(1usize << scm.n) {
        // One representative per level of x ∪ w.
        if (0..scm.n).any(|i| (level >> i) & 1 == 1 && !x.contains(&i) && !w.contains(&i)) {
            continue;
        }
        let expected = scm.truth(y, x, w, level).expect("positive parameterization");
        let point = evaluate(&functional, &data, x, w, level).unwrap();
        assert_eq!(point.probabilities.len(), expected.len());
        for (atom, (actual, truth)) in
            point.atoms.iter().zip(point.probabilities.iter().zip(&expected))
        {
            assert_eq!(atom.len(), y.len());
            assert!(
                (actual - truth).abs() < 1e-10,
                "y={y:?} x={x:?} w={w:?} level={level} directed={:?} bidirected={:?} selected={:?}: {actual} != {truth}",
                scm.directed,
                scm.bidirected,
                scm.selected
            );
        }
    }
    Outcome::Identified
}

fn evaluate(
    functional: &BoundConditionalTransportFunctional,
    data: &ExactTransportData,
    x: &[usize],
    w: &[usize],
    level: usize,
) -> Result<antecedent_expr::ExactDistribution, EstimationError> {
    let ctx = ExecutionContext::for_tests(3);
    prepare_exact_admg_conditional_transport(
        functional,
        data.clone(),
        &request(x, w, level),
        ExactEvaluationLimits::default(),
        &ctx,
    )?
    .evaluate(&ctx)
}

#[test]
fn every_three_node_conditional_query_matches_the_enumerated_truth() {
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    let (mut identified, mut not_certified) = (0usize, 0usize);
    for directed in 0..8usize {
        for bidirected in 0..8usize {
            for selected in 0..8usize {
                for params in 0..2 {
                    let scm = Scm {
                        n: 3,
                        directed: pairs
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| directed & (1 << i) != 0)
                            .map(|(_, e)| *e)
                            .collect(),
                        bidirected: pairs
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| bidirected & (1 << i) != 0)
                            .map(|(_, e)| *e)
                            .collect(),
                        selected: (0..3).filter(|i| selected & (1 << i) != 0).collect(),
                        params,
                        target_zero: Vec::new(),
                    };
                    for y in 0..3usize {
                        for w in (0..3usize).filter(|w| *w != y) {
                            let x = 3 - y - w;
                            for treatments in [vec![x], vec![]] {
                                match check(&scm, &[y], &treatments, &[w]) {
                                    Outcome::Identified => identified += 1,
                                    Outcome::NotCertified => not_certified += 1,
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("three-node sweep: identified {identified}, not certified {not_certified}");
    // 64 graphs x 8 selections x 12 query shapes x 2 parameterizations.
    assert_eq!(identified + not_certified, 64 * 8 * 12 * 2);
    assert!(identified > 11_000, "identified {identified}");
    assert!(not_certified > 900, "not certified {not_certified}");
}

/// Deterministic 64-bit generator (splitmix64).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn a_seeded_four_node_sample_matches_the_enumerated_truth() {
    let pairs: Vec<(usize, usize)> = (0..4).flat_map(|a| (a + 1..4).map(move |b| (a, b))).collect();
    let mut rng = Rng(0x2A2B_B1C0);
    let (mut identified, mut not_certified, mut multi) = (0usize, 0usize, 0usize);
    for _ in 0..400 {
        let directed = rng.below(64);
        let bidirected = rng.below(64);
        let scm = Scm {
            n: 4,
            directed: pairs
                .iter()
                .enumerate()
                .filter(|(i, _)| directed & (1 << i) != 0)
                .map(|(_, e)| *e)
                .collect(),
            bidirected: pairs
                .iter()
                .enumerate()
                .filter(|(i, _)| bidirected & (1 << i) != 0)
                .map(|(_, e)| *e)
                .collect(),
            selected: (0..4).filter(|_| rng.below(3) == 0).collect(),
            params: rng.below(3),
            target_zero: Vec::new(),
        };
        // A random role for every node: outcome, treatment, conditioned, or absent.
        let (mut y, mut x, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for node in 0..4 {
            match rng.below(4) {
                0 => y.push(node),
                1 => x.push(node),
                2 => w.push(node),
                _ => {}
            }
        }
        if y.is_empty() || w.is_empty() {
            continue;
        }
        multi += usize::from(y.len() > 1 || w.len() > 1);
        match check(&scm, &y, &x, &w) {
            Outcome::Identified => identified += 1,
            Outcome::NotCertified => not_certified += 1,
        }
    }
    eprintln!(
        "four-node sample: identified {identified}, not certified {not_certified}, multi {multi}"
    );
    assert!(identified > 150, "identified {identified}");
    assert!(not_certified > 15, "not certified {not_certified}");
    assert!(multi > 100, "multi-variable outcomes or conditioned sets {multi}");
}

#[test]
fn a_zero_mass_conditioning_event_refuses_as_a_support_failure() {
    // X(0) -> Y(1) -> W(2), X <-> Y, selection on W, and the target never sets W = 1.
    let scm = Scm {
        n: 3,
        directed: vec![(0, 1), (1, 2)],
        bidirected: vec![(0, 1)],
        selected: vec![2],
        params: 0,
        target_zero: vec![2],
    };
    let ctx = ExecutionContext::for_tests(3);
    let (catalog, data) = scm.catalog_and_laws();
    let ConditionalTransportDecision::Identified(functional) = decide_admg_conditional_transport(
        &scm.diagram(),
        &query(&[1], &[0], &[2]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        panic!("identified");
    };
    // W = 0 has positive mass and matches the truth.
    let point = evaluate(&functional, &data, &[0], &[2], 0b000).unwrap();
    let truth = scm.truth(&[1], &[0], &[2], 0b000).unwrap();
    assert!((point.probabilities[1] - truth[1]).abs() < 1e-12);
    // W = 1 has zero mass under the target: a support failure, never a number.
    for level in [0b100, 0b101] {
        assert!(scm.truth(&[1], &[0], &[2], level).is_none());
        let error = evaluate(&functional, &data, &[0], &[2], level).unwrap_err();
        let EstimationError::Refused { code, message } = error else {
            panic!("typed refusal");
        };
        assert_eq!(code, "transport_support_failure");
        assert!(message.starts_with("admg_transport.support_failure"), "{message}");
    }
}

#[test]
fn a_request_must_bind_exactly_the_treatments_and_conditioned_variables() {
    let scm = Scm {
        n: 3,
        directed: vec![(0, 1), (1, 2)],
        bidirected: vec![(0, 1)],
        selected: vec![2],
        params: 1,
        target_zero: Vec::new(),
    };
    let ctx = ExecutionContext::for_tests(3);
    let (catalog, data) = scm.catalog_and_laws();
    let ConditionalTransportDecision::Identified(functional) = decide_admg_conditional_transport(
        &scm.diagram(),
        &query(&[1], &[0], &[2]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        panic!("identified");
    };
    let invalid = |request: Assignment| {
        let error = prepare_exact_admg_conditional_transport(
            &functional,
            data.clone(),
            &request,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap_err();
        let EstimationError::Refused { code, message } = error else { panic!("typed") };
        assert_eq!(code, "invalid_argument");
        assert!(message.starts_with("admg_transport.invalid_request"), "{message}");
    };
    invalid(Assignment::from_pairs([(v(0), Value::Int64(1))]));
    invalid(Assignment::from_pairs([
        (v(0), Value::Int64(1)),
        (v(2), Value::Int64(1)),
        (v(1), Value::Int64(0)),
    ]));
    // A conditioning level outside the laws' support.
    let plan = prepare_exact_admg_conditional_transport(
        &functional,
        data.clone(),
        &Assignment::from_pairs([(v(0), Value::Int64(1)), (v(2), Value::Int64(7))]),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    let EstimationError::Refused { code, message } = plan.evaluate(&ctx).unwrap_err() else {
        panic!("typed")
    };
    assert_eq!(code, "invalid_argument");
    assert!(message.starts_with("admg_transport.invalid_request"), "{message}");
}

#[test]
fn query_coordinate_order_does_not_change_the_point() {
    // Two outcomes and two conditioned variables, declared in both orders.
    let scm = Scm {
        n: 5,
        directed: vec![(0, 1), (1, 2), (3, 1), (2, 4)],
        bidirected: vec![(0, 2)],
        selected: vec![4],
        params: 2,
        target_zero: Vec::new(),
    };
    let ctx = ExecutionContext::for_tests(3);
    let (catalog, data) = scm.catalog_and_laws();
    let decide = |q: &ConditionalTransportQuery| match decide_admg_conditional_transport(
        &scm.diagram(),
        q,
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap()
    {
        ConditionalTransportDecision::Identified(bound) => bound,
        other => panic!("identified: {other:?}"),
    };
    let forward = decide(&query(&[1, 2], &[0], &[3, 4]));
    let backward = decide(&query(&[2, 1], &[0], &[4, 3]));
    let mut compared = 0;
    for level in 0..32usize {
        if level & 0b00110 != 0 {
            continue;
        }
        let a = evaluate(&forward, &data, &[0], &[3, 4], level).unwrap();
        let b = evaluate(&backward, &data, &[0], &[4, 3], level).unwrap();
        let truth = scm.truth(&[1, 2], &[0], &[3, 4], level).unwrap();
        // Atom (y1, y2) of the forward order is atom (y2, y1) of the backward one.
        for (atom, p) in a.atoms.iter().zip(a.probabilities.iter()) {
            let swapped: Vec<Value> = vec![atom[1].clone(), atom[0].clone()];
            let q = b.atoms.iter().position(|other| other.as_ref() == swapped.as_slice()).unwrap();
            assert!((p - b.probabilities[q]).abs() < 1e-14, "{p} vs {}", b.probabilities[q]);
            let index = atom.iter().fold(0, |i, x| i * 2 + x.as_f64().unwrap() as usize);
            assert!((p - truth[index]).abs() < 1e-10);
            compared += 1;
        }
    }
    assert_eq!(compared, 8 * 4);
}
