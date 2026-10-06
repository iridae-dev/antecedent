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
    ConditionalTransportDecision, ConditionalTransportQuery, WitnessSearch,
    decide_admg_conditional_transport, search_conditional_witness,
};

enum Outcome {
    Identified,
    Proven,
    NotCertified,
}

/// Exact fraction for the independent witness check (separate code from the
/// library's verifier: different representation and enumeration order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Frac(i128, i128);

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

impl Frac {
    fn norm(n: i128, d: i128) -> Self {
        let g = gcd(n, d).max(1) * d.signum();
        Self(n / g, d / g)
    }
    fn parse(text: &str) -> Self {
        let (n, d) = text.split_once('/').expect("p/q");
        Self::norm(n.parse().unwrap(), d.parse().unwrap())
    }
    fn add(self, o: Self) -> Self {
        Self::norm(self.0 * o.1 + o.0 * self.1, self.1 * o.1)
    }
    fn mul(self, o: Self) -> Self {
        let (g1, g2) = (gcd(self.0, o.1).max(1), gcd(o.0, self.1).max(1));
        Self::norm((self.0 / g1) * (o.0 / g2), (self.1 / g2) * (o.1 / g1))
    }
    fn complement(self) -> Self {
        Self::norm(self.1 - self.0, self.1)
    }
}

/// One witness model as lookup tables keyed by node / edge, over `n` nodes.
struct WitnessScm<'a> {
    n: usize,
    model: &'a antecedent_identify::WitnessModelRecord,
}

impl WitnessScm<'_> {
    /// Joint mass of `world` (the intervened nodes' bits fixed) in one population,
    /// summed over every latent configuration (first latent varies slowest).
    fn mass(&self, target: bool, intervened: usize, world: usize) -> Frac {
        let cards: Vec<usize> = self.model.latents.iter().map(|l| l.probabilities.len()).collect();
        let configs: usize = cards.iter().product();
        let mut total = Frac(0, 1);
        for code in 0..configs {
            // Decode the mixed-radix latent configuration.
            let mut u = vec![0usize; cards.len()];
            let mut rest = code;
            for e in (0..cards.len()).rev() {
                u[e] = rest % cards[e];
                rest /= cards[e];
            }
            let mut m = Frac(1, 1);
            for (e, latent) in self.model.latents.iter().enumerate() {
                m = m.mul(Frac::parse(&latent.probabilities[u[e]]));
            }
            for node in 0..self.n {
                if intervened & (1 << node) != 0 {
                    continue;
                }
                let kernel = self
                    .model
                    .target
                    .iter()
                    .find(|k| target && k.node as usize == node)
                    .unwrap_or_else(|| {
                        self.model.source.iter().find(|k| k.node as usize == node).unwrap()
                    });
                let mut row = 0usize;
                for p in &kernel.parents {
                    row = row * 2 + ((world >> *p) & 1);
                }
                for edge in &kernel.latents {
                    let e = self.model.latents.iter().position(|l| l.edge == *edge).unwrap();
                    row = row * cards[e] + u[e];
                }
                let p = Frac::parse(&kernel.ones[row]);
                m = m.mul(if (world >> node) & 1 == 1 { p } else { p.complement() });
            }
            total = total.add(m);
        }
        total
    }
}

/// Independent re-check of a verified witness on the three-node test class
/// (node `i` is variable `i`): both models agree on every source experiment and
/// the target observational law, and the conditional differs at the level.
fn independently_verified(
    n: usize,
    w: &antecedent_identify::ConditionalWitnessRecord,
    y: &[usize],
    x: &[usize],
    c: &[usize],
) -> bool {
    let (a, b) = (WitnessScm { n, model: &w.first }, WitnessScm { n, model: &w.second });
    for mask in 0..(1usize << n) {
        for world in 0..(1usize << n) {
            if a.mass(false, mask, world) != b.mass(false, mask, world) {
                return false;
            }
        }
    }
    for world in 0..(1usize << n) {
        if a.mass(true, 0, world) != b.mass(true, 0, world) {
            return false;
        }
    }
    let bits = |vs: &[usize], levels: &[u8]| {
        vs.iter()
            .zip(levels)
            .fold((0usize, 0usize), |(m, s), (v, l)| (m | (1 << v), s | (usize::from(*l) << v)))
    };
    let (xm, xs) = bits(x, &w.treatment_level);
    let (cm, cs) = bits(c, &w.conditioned_level);
    let (ym, ys) = bits(y, &w.outcome_level);
    let value = |m: &WitnessScm<'_>| {
        let (mut num, mut den) = (Frac(0, 1), Frac(0, 1));
        for world in (0..(1usize << n)).filter(|wd| wd & xm == xs && wd & cm == cs) {
            let p = m.mass(true, xm, world);
            den = den.add(p);
            if world & ym == ys {
                num = num.add(p);
            }
        }
        assert!(den.0 > 0, "positive conditioning mass");
        Frac::norm(num.0 * den.1, num.1 * den.0)
    };
    let (va, vb) = (value(&a), value(&b));
    va != vb && Frac::parse(&w.first_value) == va && Frac::parse(&w.second_value) == vb
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
        ConditionalTransportDecision::ProvenNonTransportable(proof) => {
            // Every verified witness is re-checked by the independent verifier
            // above and by the library verifier from its record.
            assert!(
                independently_verified(scm.n, proof.witness(), y, x, w),
                "directed={:?} bidirected={:?} selected={:?}",
                scm.directed,
                scm.bidirected,
                scm.selected
            );
            proof.recheck(&scm.diagram(), &query(y, x, w), &ctx).unwrap();
            return Outcome::Proven;
        }
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
    let (mut identified, mut proven, mut not_certified, mut cross_checked) =
        (0usize, 0usize, 0usize, 0usize);
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
                                    Outcome::Identified => {
                                        identified += 1;
                                        // Cross-check on a sample: no witness search
                                        // ever verifies on an identified query.
                                        if params == 0
                                            && (directed + bidirected + selected + y) % 7 == 0
                                        {
                                            let ctx = ExecutionContext::for_tests(3);
                                            let search = search_conditional_witness(
                                                &scm.diagram(),
                                                &query(&[y], &treatments, &[w]),
                                                &mut |_| Ok(()),
                                                &ctx,
                                            )
                                            .unwrap();
                                            assert_eq!(search, WitnessSearch::NotFound);
                                            cross_checked += 1;
                                        }
                                    }
                                    Outcome::Proven => proven += 1,
                                    Outcome::NotCertified => not_certified += 1,
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "three-node sweep: identified {identified}, proven {proven}, not certified {not_certified}, identified cross-checked {cross_checked}"
    );
    // 64 graphs x 8 selections x 12 query shapes x 2 parameterizations.
    assert_eq!(identified + proven + not_certified, 64 * 8 * 12 * 2);
    assert_eq!(identified, 11_304, "identified {identified}");
    // Every reduced-joint s-hedge on three nodes (984 before the witness stage)
    // now carries an exactly verified two-model witness.
    assert_eq!(proven, 984, "proven {proven}");
    assert_eq!(not_certified, 0, "not certified {not_certified}");
    assert!(cross_checked > 100, "cross-checked {cross_checked}");
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
    let (mut identified, mut proven, mut not_certified, mut multi) =
        (0usize, 0usize, 0usize, 0usize);
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
            Outcome::Proven => proven += 1,
            Outcome::NotCertified => not_certified += 1,
        }
    }
    eprintln!(
        "four-node sample: identified {identified}, proven {proven}, not certified {not_certified}, multi {multi}"
    );
    assert!(identified > 150, "identified {identified}");
    assert!(proven + not_certified > 15, "proven {proven}, not certified {not_certified}");
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

/// Counted laws would make the point an empirical plug-in, which this route does
/// not license: the estimate-level preparation refuses them, as the facade and
/// Python preparations do, before compiling anything.
#[test]
fn counted_laws_are_refused_by_the_estimate_level_preparation() {
    let scm = Scm {
        n: 3,
        directed: vec![(0, 1), (1, 2)],
        bidirected: vec![(0, 1)],
        selected: vec![2],
        params: 0,
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
    let mut laws = data.laws().to_vec();
    let first = &laws[0];
    let cells = first.probabilities().len();
    laws[0] = antecedent_expr::ExactDiscreteLaw::try_new(
        first.population(),
        first.regime(),
        first.interventions().to_vec(),
        first.axes().to_vec(),
        vec![1.0 / cells as f64; cells],
        first.snapshot_identity(),
        antecedent_expr::LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(vec![1; cells])
    .unwrap();
    let counted = ExactTransportData::try_new(laws, 1_000_000).unwrap();
    let error = prepare_exact_admg_conditional_transport(
        &functional,
        counted,
        &request(&[0], &[2], 0),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap_err();
    let EstimationError::Refused { code, message } = error else { panic!("typed refusal") };
    assert_eq!(code, "cell_not_licensed");
    assert!(message.starts_with("admg_transport.interval_withheld"), "{message}");
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
