//! Whole-procedure test inversion of an adaptive procedure against an independent oracle.
//!
//! The procedure is deterministic UCB1 assignment (exploration 2) over six one-unit looks with
//! one-sided Wald boundaries on the contrast of arm 1 against arm 0 and, at the horizon, the
//! arm with the highest observed mean. UCB1 pulls each arm once first, so every terminal state
//! has both arms observed and the contrast is always defined. The terminal-state law is
//! enumerated here by a recursion over histories; the stagewise ordering, the tail
//! probabilities, the supremum over the nuisance baselines and the acceptance rule are written
//! again from their stated definitions, so the library's `inversion_interval` is compared with
//! a computation that shares none of its code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, AssignmentRule, BetaPrior, DERIVED_UNMEASURED, ExposureModel,
    FinalSelection, InversionGrid, MAX_INVERSION_LAWS, ObservedTerminal, OutcomeKernel,
    OutcomeObservation, ProcedureDeclaration, ProcedureSpec, Sidedness, StoppingRule,
    enumerate_law, inversion_interval, ordering_p_values,
};

const UNITS: usize = 6;
/// Not a value any small-count statistic hits exactly, so no boundary tie depends on rounding.
const CRITICAL: f64 = 1.37;
const EXPLORATION: f64 = 2.0;
const TIE: f64 = 1e-12;

fn never() -> bool {
    false
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn declaration(arms: usize) -> ProcedureDeclaration {
    ProcedureDeclaration {
        arms,
        context_probabilities: None,
        assignment: Some(AssignmentRule::UpperConfidenceBound { exploration: EXPLORATION }),
        enrolment: Some(vec![1; UNITS]),
        observation: Some(OutcomeObservation::COMPLETE),
        exposure: Some(ExposureModel::none(arms)),
        stopping: Some(StoppingRule::WaldBoundaries {
            critical: vec![CRITICAL; UNITS],
            sidedness: Sidedness::OneSidedUpper,
        }),
        selection: Some(FinalSelection::HighestObservedMean),
    }
}

fn adaptive_spec() -> ProcedureSpec {
    ProcedureSpec::new(declaration(2)).expect("a valid declaration")
}

// -- independent enumeration of the procedure ------------------------------------------------

type Counts = [(u32, u32); 2];

/// UCB1: the untried arm with the lowest index, else the arm with the larger
/// `mean + sqrt(2 ln t / n)`, ties to arm 0.
fn ucb_arm(c: &Counts) -> usize {
    let n = [c[0].0 + c[0].1, c[1].0 + c[1].1];
    if n[0] == 0 {
        return 0;
    }
    if n[1] == 0 {
        return 1;
    }
    let t = f64::from(n[0] + n[1]);
    let score = |a: usize| {
        f64::from(c[a].0) / f64::from(n[a]) + (EXPLORATION * t.ln() / f64::from(n[a])).sqrt()
    };
    usize::from(score(1) > score(0))
}

/// The one-sided upper Wald rule: the standardized contrast reaches the critical value.
fn wald_stop(c: &Counts) -> bool {
    let (n0, n1) = (f64::from(c[0].0 + c[0].1), f64::from(c[1].0 + c[1].1));
    if n0 == 0.0 || n1 == 0.0 {
        return false;
    }
    let (p0, p1) = (f64::from(c[0].0) / n0, f64::from(c[1].0) / n1);
    let variance = p0 * (1.0 - p0) / n0 + p1 * (1.0 - p1) / n1;
    if variance <= 0.0 {
        return false;
    }
    (p1 - p0) / variance.sqrt() >= CRITICAL
}

/// Arm decided at a stop (always arm 1 under the upper boundary) or selected at the horizon
/// (higher observed mean, arm 0 on a tie).
fn decision(c: &Counts, stopped: bool) -> usize {
    if stopped {
        return 1;
    }
    let (n0, n1) = (c[0].0 + c[0].1, c[1].0 + c[1].1);
    usize::from(c[1].0 * n0 > c[0].0 * n1)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct State {
    look: usize,
    stopped: bool,
    counts: Counts,
}

type Key = (usize, bool, [u32; 4]);

fn key(state: &State) -> Key {
    let c = state.counts;
    (state.look, state.stopped, [c[0].0, c[0].1, c[1].0, c[1].1])
}

fn walk(
    unit: usize,
    counts: Counts,
    probability: f64,
    theta: [f64; 2],
    out: &mut BTreeMap<Key, (State, f64)>,
) {
    let arm = ucb_arm(&counts);
    for (success, outcome) in [(true, theta[arm]), (false, 1.0 - theta[arm])] {
        let weight = probability * outcome;
        if weight <= 0.0 {
            continue;
        }
        let mut next = counts;
        if success {
            next[arm].0 += 1;
        } else {
            next[arm].1 += 1;
        }
        let stopped = wald_stop(&next);
        if stopped || unit + 1 == UNITS {
            let state = State { look: unit, stopped, counts: next };
            out.entry(key(&state)).or_insert((state, 0.0)).1 += weight;
        } else {
            walk(unit + 1, next, weight, theta, out);
        }
    }
}

fn terminal_law(theta: [f64; 2]) -> Vec<(State, f64)> {
    let mut out = BTreeMap::new();
    walk(0, [(0, 0); 2], 1.0, theta, &mut out);
    out.into_values().collect()
}

// -- independent stagewise ordering ----------------------------------------------------------

type Rank = (u8, i64, f64);

/// Rank for the contrast of arm 1 against arm 0: class 2 for a stop in favour of arm 1
/// (earlier higher), class 0 for a stop in favour of arm 0 (later higher), class 1 otherwise;
/// ties inside a class are broken by the estimate.
fn rank(state: &State, estimate: f64) -> Rank {
    let look = i64::try_from(state.look).unwrap();
    match (state.stopped, decision(&state.counts, state.stopped)) {
        (true, 1) => (2, -look, estimate),
        (true, 0) => (0, look, estimate),
        _ => (1, 0, estimate),
    }
}

fn estimate(c: &Counts) -> f64 {
    let (n0, n1) = (c[0].0 + c[0].1, c[1].0 + c[1].1);
    f64::from(c[1].0) / f64::from(n1) - f64::from(c[0].0) / f64::from(n0)
}

/// -1, 0, +1 of `a` against `b` under the ordering.
fn cmp(a: Rank, b: Rank) -> i32 {
    if a.0 != b.0 {
        return if a.0 > b.0 { 1 } else { -1 };
    }
    if a.1 != b.1 {
        return if a.1 > b.1 { 1 } else { -1 };
    }
    if (a.2 - b.2).abs() <= TIE {
        0
    } else if a.2 > b.2 {
        1
    } else {
        -1
    }
}

/// `(upper, lower)` tail masses of `target` under truth `theta`.
fn tails(theta: [f64; 2], target: Rank) -> (f64, f64) {
    let (mut upper, mut lower) = (0.0, 0.0);
    for (state, p) in terminal_law(theta) {
        match cmp(rank(&state, estimate(&state.counts)), target) {
            1 => upper += p,
            -1 => lower += p,
            _ => {
                upper += p;
                lower += p;
            }
        }
    }
    (upper, lower)
}

/// An early stop in favour of arm 1 (look 3, estimate 0.5).
fn efficacy_stop() -> (ObservedTerminal, Rank) {
    (
        ObservedTerminal { look: 3, rule_stopped: true, decision: Some(1), estimate: 0.5 },
        (2, -3, 0.5),
    )
}

/// The horizon reached with arm 1 behind (estimate -0.4).
fn horizon_behind() -> (ObservedTerminal, Rank) {
    (
        ObservedTerminal { look: 5, rule_stopped: false, decision: Some(0), estimate: -0.4 },
        (1, 0, -0.4),
    )
}

fn grid() -> InversionGrid {
    InversionGrid { contrasts: vec![-0.6, -0.3, 0.0, 0.3, 0.6], baselines: vec![0.2, 0.4, 0.6] }
}

/// The accepted contrasts and laws evaluated, from the definitions.
fn oracle_interval(grid: &InversionGrid, alpha: f64, target: Rank) -> (Vec<f64>, usize) {
    let mut accepted = Vec::new();
    let mut evaluated = 0;
    for &delta in &grid.contrasts {
        let (mut upper_sup, mut lower_sup) = (0.0_f64, 0.0_f64);
        for &baseline in &grid.baselines {
            let other = baseline + delta;
            if !(0.0..=1.0).contains(&other) {
                continue;
            }
            evaluated += 1;
            let (upper, lower) = tails([baseline, other], target);
            upper_sup = upper_sup.max(upper);
            lower_sup = lower_sup.max(lower);
        }
        if upper_sup >= alpha / 2.0 && lower_sup >= alpha / 2.0 {
            accepted.push(delta);
        }
    }
    (accepted, evaluated)
}

#[test]
fn the_enumerated_terminal_law_matches_a_hand_computed_atom() {
    // UCB1 pulls arm 0 then arm 1. At counts arm0 (1,0), arm1 (1,0) the scores tie and the
    // lowest index is pulled: arm 0 fails. Counts arm0 (1,1), arm1 (1,0): standardized
    // contrast (1 - 0.5) / sqrt(0.5 * 0.5 / 2) = 1.414 >= 1.37, so the rule stops at look 2
    // for arm 1. No other history reaches those counts at that look.
    let theta = [0.4, 0.6];
    let by_hand = theta[0] * theta[1] * (1.0 - theta[0]);
    close(by_hand, 0.144, 1e-12, "hand value");
    let law = enumerate_law(
        &adaptive_spec(),
        &OutcomeKernel::Fixed { success_probabilities: theta.to_vec() },
        &never,
    )
    .unwrap();
    let atoms: Vec<_> = law
        .atoms
        .iter()
        .filter(|a| a.look == 2 && a.rule_stopped && a.observed == vec![1, 1, 1, 0])
        .collect();
    assert_eq!(atoms.len(), 1);
    assert_eq!(atoms[0].decision, Some(1));
    close(atoms[0].probability, by_hand, 1e-12, "library atom");
    let oracle: f64 = terminal_law(theta)
        .iter()
        .filter(|(s, _)| s.look == 2 && s.stopped && s.counts == [(1, 1), (1, 0)])
        .map(|(_, p)| *p)
        .sum();
    close(oracle, by_hand, 1e-12, "oracle atom");
    let total: f64 = terminal_law(theta).iter().map(|(_, p)| p).sum();
    close(total, 1.0, 1e-12, "oracle mass");
    close(law.atoms.iter().map(|a| a.probability).sum::<f64>(), 1.0, 1e-9, "library mass");
}

#[test]
fn ordering_tails_of_the_adaptive_procedure_match_the_enumeration() {
    let spec = adaptive_spec();
    for (observed, target) in [efficacy_stop(), horizon_behind()] {
        for theta in [[0.2, 0.5], [0.5, 0.5], [0.7, 0.3]] {
            let law = enumerate_law(
                &spec,
                &OutcomeKernel::Fixed { success_probabilities: theta.to_vec() },
                &never,
            )
            .unwrap();
            let library = ordering_p_values(&law, 1, 0, &observed).unwrap();
            let (upper, lower) = tails(theta, target);
            close(library.upper, upper, 1e-9, "upper tail");
            close(library.lower, lower, 1e-9, "lower tail");
            // Both arms are observed in every terminal state of this procedure.
            assert_eq!(library.undefined_mass, 0.0);
        }
    }
}

#[test]
fn the_inverted_adaptive_interval_matches_the_enumerated_supremum_tests() {
    let spec = adaptive_spec();
    let grid = grid();
    let alpha = 0.3;
    for (observed, target) in [efficacy_stop(), horizon_behind()] {
        let interval = inversion_interval(&spec, &observed, &grid, alpha, &never).unwrap();
        assert_eq!(interval.calibration, DERIVED_UNMEASURED);
        let (accepted, evaluated) = oracle_interval(&grid, alpha, target);
        assert_eq!(interval.accepted, accepted);
        assert_eq!(interval.lower, accepted.first().copied());
        assert_eq!(interval.upper, accepted.last().copied());
        // Baselines that put the plus arm outside [0, 1] are skipped.
        assert_eq!(interval.laws_evaluated, evaluated);
        assert!(interval.accepted.windows(2).all(|w| w[0] < w[1]));
    }
    // Under contrast -0.6 only the baseline 0.6 is admissible, which makes arm 1 never
    // succeed: no standardized contrast is positive, so the upper tail of an observed stop
    // for arm 1 is exactly zero and that contrast is rejected.
    let (observed, _) = efficacy_stop();
    let interval = inversion_interval(&spec, &observed, &grid, alpha, &never).unwrap();
    assert!(!interval.accepted.contains(&-0.6));
    assert_eq!(interval.laws_evaluated, 1 + 2 + 3 + 3 + 2);
}

#[test]
fn inversion_refuses_what_its_premises_do_not_cover() {
    let (observed, _) = efficacy_stop();
    let small = InversionGrid { contrasts: vec![0.0, 0.3], baselines: vec![0.4] };

    // Units of two contexts are not identically distributed within an arm.
    let mut contextual = declaration(2);
    contextual.context_probabilities = Some(vec![0.5, 0.5]);
    contextual.assignment = Some(AssignmentRule::ThompsonBeta { prior: BetaPrior::UNIFORM });
    contextual.stopping = Some(StoppingRule::NoStopping);
    contextual.selection = Some(FinalSelection::NoSelection);
    contextual.enrolment = Some(vec![1, 1]);
    let spec = ProcedureSpec::new(contextual).unwrap();
    assert_eq!(
        inversion_interval(&spec, &observed, &small, 0.3, &never),
        Err(Error::PremiseFailed {
            detail: "adaptive_inference.inversion_premise_contexts",
            message: "inversion is defined for a contrast of arms within one context",
        })
    );

    // More than two arms: a contrast of arm 1 against arm 0 is not the contrast asked for.
    let mut three = declaration(3);
    three.assignment = Some(AssignmentRule::Fixed { probabilities: vec![1.0 / 3.0; 3] });
    three.stopping = Some(StoppingRule::NoStopping);
    three.enrolment = Some(vec![3]);
    let three = ProcedureSpec::new(three).unwrap();
    assert!(matches!(
        inversion_interval(&three, &observed, &small, 0.3, &never),
        Err(Error::Shape(_))
    ));

    let spec = adaptive_spec();
    // The level, the grid values and the grid size.
    for alpha in [0.0, 1.0, f64::NAN, -0.1] {
        assert!(matches!(
            inversion_interval(&spec, &observed, &small, alpha, &never),
            Err(Error::InvalidParameter(_))
        ));
    }
    let outside = InversionGrid { contrasts: vec![1.5], baselines: vec![0.4] };
    assert!(matches!(
        inversion_interval(&spec, &observed, &outside, 0.3, &never),
        Err(Error::InvalidParameter(_))
    ));
    let not_finite = InversionGrid { contrasts: vec![0.0], baselines: vec![f64::NAN] };
    assert!(matches!(
        inversion_interval(&spec, &observed, &not_finite, 0.3, &never),
        Err(Error::InvalidParameter(_))
    ));
    let nothing = InversionGrid { contrasts: vec![0.0], baselines: vec![] };
    assert!(matches!(
        inversion_interval(&spec, &observed, &nothing, 0.3, &never),
        Err(Error::Shape(_))
    ));
    let mut estimate_not_finite = observed;
    estimate_not_finite.estimate = f64::NAN;
    assert!(matches!(
        inversion_interval(&spec, &estimate_not_finite, &small, 0.3, &never),
        Err(Error::InvalidParameter(_)) | Err(Error::Shape(_))
    ));

    // The law bound: 64 x 64 is admitted, 65 x 64 is one past it.
    assert_eq!(MAX_INVERSION_LAWS, 4096);
    let over = InversionGrid { contrasts: vec![0.0; 65], baselines: vec![0.4; 64] };
    assert!(over.contrasts.len() * over.baselines.len() > MAX_INVERSION_LAWS);
    assert!(matches!(
        inversion_interval(&spec, &observed, &over, 0.3, &never),
        Err(Error::Limit(_))
    ));
}

#[test]
fn cancellation_stops_the_inversion_with_no_partial_interval() {
    let spec = adaptive_spec();
    let (observed, _) = efficacy_stop();
    let small = InversionGrid { contrasts: vec![0.0, 0.3], baselines: vec![0.4] };
    let result = inversion_interval(&spec, &observed, &small, 0.3, &|| true);
    assert_eq!(result, Err(Error::Cancelled));
}
