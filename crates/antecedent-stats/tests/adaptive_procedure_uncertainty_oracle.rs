//! Independent oracles for the uncertainty derived for a complete adaptive
//! procedure.
//!
//! The exact law of a procedure is checked against enumerations written here,
//! not against the code under test: recursions over assignment, outcome and
//! stopping that use integer counts and the integer-shape sum for the
//! probability that one Beta variable exceeds another, closed forms
//! (binomial mixtures, the mixture-Hoeffding radius, the normal boundary
//! spending) and a separately integrated bivariate normal. Each procedure
//! family, outcome-delay structure, exposure model and monitoring strategy
//! the matrix names is its own test, because evidence for one does not
//! transfer to another.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "enumerations index parallel per-arm arrays by arm and count small integers"
)]

use std::collections::BTreeMap;

use antecedent_kernels::{norm_pdf, norm_sf};
use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, AssignmentRule, Attribution, BetaPrior, ContractGap,
    DERIVED_UNMEASURED, Estimand, EstimandStatement, ExposureModel, FinalSelection, Functional,
    InversionGrid, ObservedTerminal, OutcomeKernel, OutcomeModelPrior, OutcomeObservation,
    ProcedureDeclaration, ProcedureSpec, Sidedness, SimulationRequest, SpendingFunction,
    StagewiseLook, StoppingRule, UncertaintyParadigm, WeightScheme, admit,
    anytime_exclusion_probability, anytime_valid_report, bayesian_report, boundaries,
    decision_report, enumerate_law, inversion_interval, ordering_p_values, posterior_mean_gaps,
    simulate_procedure, stagewise_inference, stagewise_upper_p_value,
};
use antecedent_stats::ln_gamma;

fn never() -> bool {
    false
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn declaration(assignment: AssignmentRule, enrolment: Vec<u32>) -> ProcedureDeclaration {
    ProcedureDeclaration {
        arms: 2,
        context_probabilities: None,
        assignment: Some(assignment),
        enrolment: Some(enrolment),
        observation: Some(OutcomeObservation::COMPLETE),
        exposure: Some(ExposureModel::none(2)),
        stopping: Some(StoppingRule::NoStopping),
        selection: Some(FinalSelection::NoSelection),
    }
}

fn build(declaration: ProcedureDeclaration) -> ProcedureSpec {
    ProcedureSpec::new(declaration).expect("a valid declaration")
}

fn fixed_half() -> AssignmentRule {
    AssignmentRule::Fixed { probabilities: vec![0.5, 0.5] }
}

fn thompson() -> AssignmentRule {
    AssignmentRule::ThompsonBeta { prior: BetaPrior::UNIFORM }
}

fn law_at(
    spec: &ProcedureSpec,
    truth: &[f64],
) -> antecedent_stats::adaptive_inference::ProcedureLaw {
    enumerate_law(spec, &OutcomeKernel::Fixed { success_probabilities: truth.to_vec() }, &never)
        .expect("a law")
}

type TerminalMap = BTreeMap<(usize, bool, Vec<u16>), f64>;

fn terminal_map(law: &antecedent_stats::adaptive_inference::ProcedureLaw) -> TerminalMap {
    let mut out = TerminalMap::new();
    for atom in &law.atoms {
        *out.entry((atom.look, atom.rule_stopped, atom.observed.clone())).or_insert(0.0) +=
            atom.probability;
    }
    out
}

fn assert_same_law(actual: &TerminalMap, expected: &TerminalMap, tolerance: f64, what: &str) {
    for (key, &p) in expected {
        let got = actual.get(key).copied().unwrap_or(0.0);
        assert!((got - p).abs() <= tolerance, "{what}: state {key:?} expected {p}, got {got}");
    }
    for (key, &p) in actual {
        assert!(
            expected.contains_key(key) || p <= tolerance,
            "{what}: unexpected state {key:?} with probability {p}"
        );
    }
}

// -- Independent enumeration of the immediate, one-unit-per-look procedure ------------------------

type Counts = [(u32, u32); 2];
type GapCase = (fn(&mut ProcedureDeclaration), ContractGap, &'static str);

/// `P(theta_b > theta_a)` for integer shapes (the finite sum of Evan Miller).
fn miller_b_exceeds_a(a: (u32, u32), b: (u32, u32)) -> f64 {
    let ln_b = |x: f64, y: f64| ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y);
    let (aa, ba) = (f64::from(a.0), f64::from(a.1));
    let bb = f64::from(b.1);
    (0..b.0)
        .map(|i| {
            let i = f64::from(i);
            (ln_b(aa + i, ba + bb) - ln_b(1.0 + i, bb) - ln_b(aa, ba)).exp() / (bb + i)
        })
        .sum()
}

fn thompson_zero(counts: &Counts) -> f64 {
    1.0 - miller_b_exceeds_a((counts[0].0 + 1, counts[0].1 + 1), (counts[1].0 + 1, counts[1].1 + 1))
}

fn observed_vector(counts: &Counts) -> Vec<u16> {
    let narrow = |x: u32| u16::try_from(x).expect("a small count");
    vec![narrow(counts[0].0), narrow(counts[0].1), narrow(counts[1].0), narrow(counts[1].1)]
}

#[allow(clippy::too_many_arguments, reason = "one recursion carries its whole state explicitly")]
fn tree(
    unit: usize,
    units: usize,
    counts: Counts,
    probability: f64,
    theta: [f64; 2],
    arm_zero: &dyn Fn(&Counts) -> f64,
    stop: &dyn Fn(&Counts) -> bool,
    out: &mut TerminalMap,
) {
    let q = arm_zero(&counts);
    for (arm, chosen) in [(0_usize, q), (1, 1.0 - q)] {
        if chosen <= 0.0 {
            continue;
        }
        for (success, outcome) in [(true, theta[arm]), (false, 1.0 - theta[arm])] {
            let mut next = counts;
            if success {
                next[arm].0 += 1;
            } else {
                next[arm].1 += 1;
            }
            let weight = probability * chosen * outcome;
            let stopped = stop(&next);
            if stopped || unit + 1 == units {
                *out.entry((unit, stopped, observed_vector(&next))).or_insert(0.0) += weight;
            } else {
                tree(unit + 1, units, next, weight, theta, arm_zero, stop, out);
            }
        }
    }
}

fn enumerated(
    units: usize,
    theta: [f64; 2],
    arm_zero: &dyn Fn(&Counts) -> f64,
    stop: &dyn Fn(&Counts) -> bool,
) -> TerminalMap {
    let mut out = TerminalMap::new();
    tree(0, units, [(0, 0); 2], 1.0, theta, arm_zero, stop, &mut out);
    out
}

// -- Contract refusals -------------------------------------------------------------------------------

#[test]
fn every_missing_element_of_the_procedure_is_its_own_refusal() {
    let complete = declaration(fixed_half(), vec![4]);
    let cases: [GapCase; 6] = [
        (
            |d| d.assignment = None,
            ContractGap::AssignmentPolicyMissing,
            "adaptive_inference.policy_history_missing",
        ),
        (
            |d| d.enrolment = None,
            ContractGap::EnrolmentScheduleMissing,
            "adaptive_inference.interim_analyses_missing",
        ),
        (
            |d| d.observation = None,
            ContractGap::OutcomeObservationMissing,
            "adaptive_inference.outcome_availability_missing",
        ),
        (
            |d| d.exposure = None,
            ContractGap::ExposureModelMissing,
            "adaptive_inference.exposure_history_missing",
        ),
        (
            |d| d.stopping = None,
            ContractGap::StoppingRuleMissing,
            "adaptive_inference.stopping_rule_missing",
        ),
        (
            |d| d.selection = None,
            ContractGap::FinalSelectionMissing,
            "adaptive_inference.policy_selection_missing",
        ),
    ];
    for (remove, gap, detail) in cases {
        let mut d = complete.clone();
        remove(&mut d);
        let error = ProcedureSpec::new(d).expect_err("a missing element is refused");
        assert_eq!(error.gaps, vec![gap], "exactly the missing element is named");
        assert_eq!(gap.detail(), detail);
        assert!(!gap.remedy().is_empty());
    }
    // All six missing at once: six gaps, nothing defaulted.
    let mut empty = complete;
    empty.assignment = None;
    empty.enrolment = None;
    empty.observation = None;
    empty.exposure = None;
    empty.stopping = None;
    empty.selection = None;
    assert_eq!(ProcedureSpec::new(empty).expect_err("refused").gaps.len(), 6);
}

#[test]
fn inconsistent_elements_are_refused_with_their_own_details() {
    let invalid = |mut d: ProcedureDeclaration, change: fn(&mut ProcedureDeclaration)| {
        change(&mut d);
        ProcedureSpec::new(d).expect_err("refused").gaps
    };
    let base = declaration(thompson(), vec![2, 2]);
    let gaps = invalid(base.clone(), |d| d.enrolment = Some(vec![0, 3]));
    assert_eq!(gaps, vec![ContractGap::Invalid("adaptive_inference.looks_not_ordered")]);
    let gaps = invalid(base.clone(), |d| {
        d.observation =
            Some(OutcomeObservation { lag_looks: 1, retain_success: 0.0, retain_failure: 1.0 });
    });
    assert_eq!(gaps, vec![ContractGap::Invalid("adaptive_inference.outcome_availability_invalid")]);
    let gaps = invalid(base.clone(), |d| {
        d.exposure = Some(ExposureModel {
            failure_probability: vec![0.1, 0.1],
            fallback_arm: 5,
            attribution: Attribution::Delivered,
        });
    });
    assert_eq!(gaps, vec![ContractGap::Invalid("adaptive_inference.exposure_invalid")]);
    // Boundaries need one critical value per look and two arms.
    let gaps = invalid(base.clone(), |d| {
        d.stopping = Some(StoppingRule::WaldBoundaries {
            critical: vec![2.0],
            sidedness: Sidedness::OneSidedUpper,
        });
    });
    assert_eq!(gaps, vec![ContractGap::Invalid("adaptive_inference.rule_mismatch")]);
    let gaps = invalid(base, |d| {
        d.assignment = Some(AssignmentRule::Fixed { probabilities: vec![0.7, 0.7] });
    });
    assert_eq!(gaps, vec![ContractGap::Invalid("adaptive_inference.rule_mismatch")]);
}

#[test]
fn procedure_names_follow_the_matrix_axes() {
    let single = build(declaration(thompson(), vec![1, 1, 1]));
    assert_eq!(single.family_name(), "thompson_beta_bernoulli");
    assert_eq!(single.monitoring_name(), "continuous_monitoring");
    assert_eq!(single.stopping_name(), "no_stopping");
    let batched = build(declaration(thompson(), vec![5, 5]));
    assert_eq!(batched.family_name(), "batched_adaptive");
    assert_eq!(batched.monitoring_name(), "fixed_interim_looks");
    let mut d = declaration(fixed_half(), vec![10]);
    d.stopping = Some(StoppingRule::NoStopping);
    let fixed = build(d);
    assert_eq!(fixed.family_name(), "fixed_randomization");
    assert_eq!(fixed.monitoring_name(), "final_analysis_only");
}

// -- Fixed randomization, final analysis only ------------------------------------------------------

#[test]
fn fixed_design_law_matches_the_binomial_mixture() {
    let spec = build(declaration(fixed_half(), vec![4]));
    let theta = [0.3, 0.6];
    let law = law_at(&spec, &theta);
    // Independent enumeration: k units to arm 0 ~ Bin(4, 1/2), then
    // binomial successes in each arm.
    let choose = |n: u32, k: u32| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
    };
    let mut expected = TerminalMap::new();
    for k in 0..=4_u32 {
        let pk = choose(4, k) / 16.0;
        for s0 in 0..=k {
            let p0 =
                choose(k, s0) * theta[0].powi(s0 as i32) * (1.0 - theta[0]).powi((k - s0) as i32);
            for s1 in 0..=(4 - k) {
                let p1 = choose(4 - k, s1)
                    * theta[1].powi(s1 as i32)
                    * (1.0 - theta[1]).powi((4 - k - s1) as i32);
                let key =
                    (0, false, vec![s0 as u16, (k - s0) as u16, s1 as u16, (4 - k - s1) as u16]);
                *expected.entry(key).or_insert(0.0) += pk * p0 * p1;
            }
        }
    }
    assert_same_law(&terminal_map(&law), &expected, 1e-12, "fixed design");
    // An undefined contrast needs an arm with no unit: probability 2 / 16.
    let contrast = law.estimator_law(Functional::Contrast { plus: 1, minus: 0 }).unwrap();
    close(contrast.undefined_mass, 2.0 / 16.0, 1e-12, "undefined mass");
    // Unbiased given both arms observed: E[ybar_1 - ybar_0] = 0.3.
    close(contrast.mean().unwrap(), 0.3, 1e-12, "mean contrast");
    assert!(law.work_units > 0 && law.peak_states >= 1, "work is observed");
}

#[test]
fn changed_truth_at_unchanged_shape_changes_the_law() {
    let spec = build(declaration(fixed_half(), vec![3]));
    let a = law_at(&spec, &[0.3, 0.6]);
    let b = law_at(&spec, &[0.3, 0.61]);
    assert_eq!(a.atoms.len(), b.atoms.len(), "same shape");
    assert_ne!(terminal_map(&a), terminal_map(&b), "a changed truth is a changed law");
}

// -- Thompson sampling: assignment, delay, stopping, selection --------------------------------------

#[test]
fn thompson_law_matches_the_enumerated_history_tree() {
    let spec = build(declaration(thompson(), vec![1, 1, 1]));
    let theta = [0.3, 0.7];
    let law = law_at(&spec, &theta);
    let expected = enumerated(3, theta, &thompson_zero, &|_| false);
    assert_same_law(&terminal_map(&law), &expected, 1e-8, "thompson tree");
}

#[test]
fn thompson_stopping_time_matches_the_enumerated_tree() {
    // Stop when the posterior probability that an arm is best reaches 0.72, avoiding the exact tie at 0.7 (Beta(2,1) against Beta(2,2)):
    // after one unit the best is 2/3, after two matching units 3/4.
    let mut d = declaration(thompson(), vec![1, 1, 1, 1]);
    d.stopping =
        Some(StoppingRule::PosteriorProbabilityBest { threshold: 0.72, prior: BetaPrior::UNIFORM });
    d.selection = Some(FinalSelection::HighestObservedMean);
    let spec = build(d);
    let theta = [0.35, 0.65];
    let law = law_at(&spec, &theta);
    let stop = |c: &Counts| {
        let zero = thompson_zero(c);
        zero.max(1.0 - zero) >= 0.72
    };
    let expected = enumerated(4, theta, &thompson_zero, &stop);
    assert_same_law(&terminal_map(&law), &expected, 1e-8, "stopped thompson tree");
    // No stop can occur after a single unit (best probability 2/3 < 0.72).
    assert!(law.rule_stop_probabilities()[0].abs() < 1e-12);
    // The stopping probabilities and the horizon probability exhaust the mass.
    let total: f64 = law.rule_stop_probabilities().iter().sum::<f64>() + law.horizon_probability();
    close(total, 1.0, 1e-9, "stop and horizon mass");
    // Expected sample size agrees with the enumeration.
    let expected_units: f64 = expected.iter().map(|(k, p)| p * (k.0 as f64 + 1.0)).sum();
    close(law.expected_enrolled(), expected_units, 1e-8, "expected units");
}

#[test]
fn outcome_delay_hides_pending_outcomes_from_the_policy() {
    // Lag of one look, Thompson, enrolment [1, 1, 1]: the third unit is
    // assigned from the outcome of the first alone, and the terminal
    // observed counts hold the first two units only.
    let mut d = declaration(thompson(), vec![1, 1, 1]);
    d.observation = Some(OutcomeObservation::delayed(1));
    let spec = build(d);
    let theta = [0.3, 0.7];
    let law = law_at(&spec, &theta);
    let mut expected = TerminalMap::new();
    let p_zero_after = |first_arm: usize, success: bool| -> f64 {
        let mut c: Counts = [(0, 0); 2];
        if success {
            c[first_arm].0 += 1;
        } else {
            c[first_arm].1 += 1;
        }
        thompson_zero(&c)
    };
    for a0 in 0..2_usize {
        for y0 in [true, false] {
            let p0 = 0.5 * if y0 { theta[a0] } else { 1.0 - theta[a0] };
            for a1 in 0..2_usize {
                for y1 in [true, false] {
                    let p1 = 0.5 * if y1 { theta[a1] } else { 1.0 - theta[a1] };
                    // Unit 2's arm is irrelevant to the observed counts, and
                    // its policy uses unit 0 only.
                    let q = p_zero_after(a0, y0);
                    assert!((0.0..=1.0).contains(&q));
                    let mut c: Counts = [(0, 0); 2];
                    for (arm, ok) in [(a0, y0), (a1, y1)] {
                        if ok {
                            c[arm].0 += 1;
                        } else {
                            c[arm].1 += 1;
                        }
                    }
                    *expected.entry((2, false, observed_vector(&c))).or_insert(0.0) += p0 * p1;
                }
            }
        }
    }
    assert_same_law(&terminal_map(&law), &expected, 1e-9, "delayed thompson");
    // The same procedure without delay observes all three units.
    let immediate = law_at(&build(declaration(thompson(), vec![1, 1, 1])), &theta);
    assert!(immediate.atoms.iter().all(|a| a.cell_units(0) + a.cell_units(1) == 3));
    assert!(law.atoms.iter().all(|a| a.cell_units(0) + a.cell_units(1) == 2));
}

#[test]
fn a_follow_up_look_reveals_pending_outcomes() {
    // Fixed assignment to arm 0 only; two units then a follow-up look.
    let mut d = declaration(AssignmentRule::Fixed { probabilities: vec![1.0, 0.0] }, vec![2, 0]);
    d.observation = Some(OutcomeObservation::delayed(1));
    let with_follow_up = law_at(&build(d.clone()), &[0.4, 0.5]);
    for atom in &with_follow_up.atoms {
        assert_eq!(atom.cell_units(0), 2, "the follow-up look reveals both units");
    }
    // Success count ~ Bin(2, 0.4): P(2) = 0.16.
    let two: f64 = with_follow_up
        .atoms
        .iter()
        .filter(|a| a.cell_counts(0).0 == 2)
        .map(|a| a.probability)
        .sum();
    close(two, 0.16, 1e-12, "P(two successes)");
    // Without the follow-up look nothing was observed at the end.
    d.enrolment = Some(vec![2]);
    let without = law_at(&build(d), &[0.4, 0.5]);
    assert!(without.atoms.iter().all(|a| a.cell_units(0) == 0));
}

#[test]
fn a_resource_stop_before_outcomes_arrive_observes_nothing() {
    let mut d = declaration(AssignmentRule::Fixed { probabilities: vec![0.5, 0.5] }, vec![2, 2]);
    d.observation = Some(OutcomeObservation::delayed(1));
    d.stopping = Some(StoppingRule::ResourceCap { max_units: 2 });
    let law = law_at(&build(d), &[0.4, 0.5]);
    assert_eq!(law.atoms.len(), 1);
    let atom = &law.atoms[0];
    assert!(atom.rule_stopped && atom.look == 0 && atom.enrolled == 2);
    assert!(atom.observed.iter().all(|&c| c == 0));
    close(atom.probability, 1.0, 1e-12, "mass");
}

// -- Other assignment families -----------------------------------------------------------------------

#[test]
fn epsilon_greedy_law_matches_integer_arithmetic_enumeration() {
    let epsilon = 0.3;
    let spec = build(declaration(
        AssignmentRule::EpsilonGreedy { epsilon, prior: BetaPrior::UNIFORM },
        vec![1, 1, 1, 1],
    ));
    let theta = [0.4, 0.6];
    let law = law_at(&spec, &theta);
    // Running mean (1 + s) / (2 + n), compared by integer cross-multiplication;
    // tied arms share the greedy mass.
    let arm_zero = |c: &Counts| -> f64 {
        let (n0, n1) = (2 + c[0].0 + c[0].1, 2 + c[1].0 + c[1].1);
        let (m0, m1) = ((1 + c[0].0) * n1, (1 + c[1].0) * n0);
        let greedy_zero = match m0.cmp(&m1) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Less => 0.0,
            std::cmp::Ordering::Equal => 0.5,
        };
        epsilon / 2.0 + (1.0 - epsilon) * greedy_zero
    };
    let expected = enumerated(4, theta, &arm_zero, &|_| false);
    assert_same_law(&terminal_map(&law), &expected, 1e-12, "epsilon greedy");
}

#[test]
fn upper_confidence_bound_law_is_deterministic_in_its_first_pulls() {
    let spec = build(declaration(
        AssignmentRule::UpperConfidenceBound { exploration: 2.0 },
        vec![1, 1, 1, 1],
    ));
    let theta = [0.4, 0.6];
    let law = law_at(&spec, &theta);
    // UCB1 by hand: untried arms first, then mean + sqrt(2 ln t / n).
    let arm_zero = |c: &Counts| -> f64 {
        let n = [c[0].0 + c[0].1, c[1].0 + c[1].1];
        if n[0] == 0 {
            return 1.0;
        }
        if n[1] == 0 {
            return 0.0;
        }
        let t = f64::from(n[0] + n[1]);
        let score = |a: usize| {
            f64::from(c[a].0) / f64::from(n[a]) + (2.0 * t.ln() / f64::from(n[a])).sqrt()
        };
        if score(1) > score(0) { 0.0 } else { 1.0 }
    };
    let expected = enumerated(4, theta, &arm_zero, &|_| false);
    assert_same_law(&terminal_map(&law), &expected, 1e-12, "ucb");
    // After the first two pulls each arm has exactly one unit.
    let two = law_at(
        &build(declaration(AssignmentRule::UpperConfidenceBound { exploration: 2.0 }, vec![1, 1])),
        &theta,
    );
    for atom in &two.atoms {
        assert_eq!((atom.cell_units(0), atom.cell_units(1)), (1, 1));
    }
    let both: f64 = two
        .atoms
        .iter()
        .filter(|a| a.cell_counts(0).0 == 1 && a.cell_counts(1).0 == 1)
        .map(|a| a.probability)
        .sum();
    close(both, 0.4 * 0.6, 1e-12, "P(both succeed)");
}

#[test]
fn contextual_assignment_runs_independent_strata() {
    // Two equally likely contexts, stratified Thompson, two units.
    let mut d = declaration(thompson(), vec![1, 1]);
    d.context_probabilities = Some(vec![0.5, 0.5]);
    let spec = build(d);
    assert_eq!(spec.family_name(), "contextual_bandit");
    let theta = [0.2, 0.7, 0.5, 0.9];
    let law = law_at(&spec, &theta);
    // The first unit is in each cell with probability 1/4. The second lands in
    // the same cell with probability 1/2 (context) times 2/3 after a success
    // and 1/3 after a failure (Beta(2,1) against Beta(1,1)).
    let both_in_cell_zero: f64 =
        law.atoms.iter().filter(|a| a.cell_units(0) == 2).map(|a| a.probability).sum();
    let expected = 0.25 * (theta[0] * (0.5 * 2.0 / 3.0) + (1.0 - theta[0]) * (0.5 / 3.0));
    close(both_in_cell_zero, expected, 1e-9, "same cell twice");
    let total: f64 = law.atoms.iter().map(|a| a.probability).sum();
    close(total, 1.0, 1e-9, "mass");
}

#[test]
fn batched_assignment_updates_only_between_batches() {
    // One batch of two under Thompson: both units see the empty history, so
    // each is assigned uniformly and independently.
    let spec = build(declaration(thompson(), vec![2]));
    assert_eq!(spec.family_name(), "batched_adaptive");
    let law = law_at(&spec, &[0.5, 0.5]);
    let both_arm_zero: f64 =
        law.atoms.iter().filter(|a| a.cell_units(0) == 2).map(|a| a.probability).sum();
    close(both_arm_zero, 0.25, 1e-9, "both units in arm 0");
}

// -- Exposure failures and outcome-dependent loss -------------------------------------------------

#[test]
fn exploration_failures_change_what_the_arm_mean_is() {
    // Always assign arm 0; a quarter of units receive arm 1 instead.
    let assignment = AssignmentRule::Fixed { probabilities: vec![1.0, 0.0] };
    let mut delivered = declaration(assignment.clone(), vec![2]);
    delivered.exposure = Some(ExposureModel {
        failure_probability: vec![0.25, 0.0],
        fallback_arm: 1,
        attribution: Attribution::Delivered,
    });
    let theta = [0.2, 0.8];
    let delivered_spec = build(delivered.clone());
    let law = law_at(&delivered_spec, &theta);
    // Arm 0 has no delivered unit with probability 0.25^2.
    let arm_zero = law.estimator_law(Functional::CellMean(0)).unwrap();
    close(arm_zero.undefined_mass, 0.0625, 1e-12, "no delivered unit");
    close(arm_zero.mean().unwrap(), 0.2, 1e-12, "delivered mean of arm 0");
    assert_eq!(
        admit(&delivered_spec, UncertaintyParadigm::BayesianPosterior).unwrap(),
        EstimandStatement::DeliveredArmMean
    );
    // Attributed to the assignment, every outcome is arm 0's and the mean is
    // the mixture 0.75 * 0.2 + 0.25 * 0.8 = 0.35.
    let mut assigned = delivered;
    if let Some(exposure) = assigned.exposure.as_mut() {
        exposure.attribution = Attribution::Assigned;
    }
    let assigned_spec = build(assigned);
    let itt = law_at(&assigned_spec, &theta).estimator_law(Functional::CellMean(0)).unwrap();
    close(itt.undefined_mass, 0.0, 1e-12, "all units attributed");
    close(itt.mean().unwrap(), 0.35, 1e-12, "intention-to-treat mean");
    assert_eq!(
        admit(&assigned_spec, UncertaintyParadigm::BayesianPosterior).unwrap(),
        EstimandStatement::AssignedArmIntentToTreat
    );
}

#[test]
fn outcome_dependent_loss_biases_the_observed_mean_by_the_closed_form() {
    // Successes always observed, failures with probability 1/2. Given an
    // observation the success probability is 0.5 / (0.5 + 0.25) = 2/3.
    let mut d = declaration(AssignmentRule::Fixed { probabilities: vec![1.0, 0.0] }, vec![3]);
    d.observation =
        Some(OutcomeObservation { lag_looks: 0, retain_success: 1.0, retain_failure: 0.5 });
    let spec = build(d);
    let law = law_at(&spec, &[0.5, 0.5]);
    let observed_mean = law.estimator_law(Functional::CellMean(0)).unwrap();
    close(observed_mean.mean().unwrap(), 2.0 / 3.0, 1e-12, "observed mean");
    // No observation at all: (1/4)^3.
    close(observed_mean.undefined_mass, 1.0 / 64.0, 1e-12, "nothing observed");
    // The Bayesian and anytime-valid premises fail, with named details.
    for paradigm in
        [UncertaintyParadigm::BayesianPosterior, UncertaintyParadigm::AnytimeValidFrequentist]
    {
        match admit(&spec, paradigm) {
            Err(Error::PremiseFailed { detail, .. }) => {
                assert!(detail.contains("outcome_dependent"), "{detail}");
            }
            other => panic!("expected a premise failure, got {other:?}"),
        }
    }
}

// -- Bayesian posterior validity ----------------------------------------------------------------------

#[test]
fn posterior_mean_is_a_martingale_of_an_ignorable_adaptive_procedure() {
    let mut d = declaration(thompson(), vec![1, 1, 1, 1]);
    d.stopping =
        Some(StoppingRule::PosteriorProbabilityBest { threshold: 0.7, prior: BetaPrior::UNIFORM });
    d.observation =
        Some(OutcomeObservation { lag_looks: 1, retain_success: 0.8, retain_failure: 0.8 });
    let spec = build(d);
    let priors = [BetaPrior { alpha: 2.0, beta: 1.0 }, BetaPrior::UNIFORM];
    let gaps = posterior_mean_gaps(&spec, &priors, &never).unwrap();
    for gap in gaps {
        close(gap.gap, 0.0, 1e-9, "posterior mean gap under ignorable loss and delay");
    }
}

#[test]
fn outcome_dependent_loss_breaks_the_posterior_by_the_hand_computed_gap() {
    // One unit, uniform assignment, failures observed with probability 1/2:
    // E[posterior mean of arm 0] = 25/48, against the prior mean 1/2.
    let mut d = declaration(fixed_half(), vec![1]);
    d.observation =
        Some(OutcomeObservation { lag_looks: 0, retain_success: 1.0, retain_failure: 0.5 });
    let spec = build(d);
    let gaps = posterior_mean_gaps(&spec, &[BetaPrior::UNIFORM; 2], &never).unwrap();
    close(gaps[0].expected_posterior_mean, 25.0 / 48.0, 1e-12, "expected posterior mean");
    close(gaps[0].gap, 1.0 / 48.0, 1e-12, "gap");
    assert!(matches!(
        bayesian_report(&spec, BetaPrior::UNIFORM, &[1, 0, 0, 0], &never),
        Err(Error::PremiseFailed { .. })
    ));
}

#[test]
fn bayesian_report_matches_closed_forms() {
    let spec = build(declaration(thompson(), vec![1, 1]));
    // Arm 0: 3 successes, 1 failure -> Beta(4, 2); arm 1: Beta(2, 2).
    let report = bayesian_report(&spec, BetaPrior::UNIFORM, &[3, 1, 1, 1], &never).unwrap();
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
    close(report.posteriors[0].mean(), 4.0 / 6.0, 1e-15, "posterior mean");
    let best_zero = 1.0 - miller_b_exceeds_a((4, 2), (2, 2));
    close(report.probability_best[0], best_zero, 1e-9, "probability best");
    close(report.probability_best[1], 1.0 - best_zero, 1e-9, "complement");
    assert!(matches!(
        bayesian_report(&spec, BetaPrior::UNIFORM, &[1, 0], &never),
        Err(Error::Shape(_))
    ));
}

// -- Decision-theoretic stopping -----------------------------------------------------------------------

#[test]
fn expected_loss_rule_decides_at_the_first_look_with_closed_form_regret() {
    // After one unit the smaller expected loss is 1/12 < 0.1 for either
    // arm and outcome (hand integration), so the rule always stops at look 0.
    let mut d = declaration(fixed_half(), vec![1, 1]);
    d.stopping = Some(StoppingRule::PosteriorExpectedLoss {
        loss_threshold: 0.1,
        prior: BetaPrior::UNIFORM,
    });
    let spec = build(d);
    let truth = [0.4, 0.7];
    let report = decision_report(&spec, &truth, &never).unwrap();
    close(report.stop_probabilities[0], 1.0, 1e-9, "stop at look 0");
    close(report.expected_enrolled, 1.0, 1e-9, "one unit");
    // Decide arm 0 after (arm 0, success) or (arm 1, failure).
    let decide_zero = 0.5 * truth[0] + 0.5 * (1.0 - truth[1]);
    close(report.decision_probabilities[0], decide_zero, 1e-9, "decide arm 0");
    close(report.wrong_decision_probability, decide_zero, 1e-9, "wrong decision");
    close(report.expected_regret, decide_zero * 0.3, 1e-9, "regret");
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
    // A fixed-sample design is not a decision-theoretic procedure.
    let plain = build(declaration(fixed_half(), vec![2]));
    assert!(matches!(decision_report(&plain, &truth, &never), Err(Error::PremiseFailed { .. })));
}

// -- Anytime-valid ---------------------------------------------------------------------------------------

fn radius(count: f64, alpha: f64, tuning: f64) -> f64 {
    0.5 * ((count + tuning) * ((count + tuning) / (tuning * alpha * alpha)).ln()).sqrt() / count
}

#[test]
fn anytime_exclusion_probability_matches_the_binomial_enumeration() {
    // Fixed design, one look, four units; the bound of each arm is the
    // mixture-Hoeffding radius at alpha / 2 around the observed mean.
    let spec = build(declaration(fixed_half(), vec![4]));
    let theta = [0.5, 0.5];
    let (alpha, tuning) = (0.5, 2.0);
    let value = anytime_exclusion_probability(&spec, &theta, alpha, tuning, &never).unwrap();
    let choose = |n: u32, k: u32| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
    };
    let per_cell = alpha / 2.0;
    let mut expected = 0.0;
    for k in 0..=4_u32 {
        for s0 in 0..=k {
            for s1 in 0..=(4 - k) {
                let probability = choose(4, k) / 16.0
                    * choose(k, s0)
                    * 0.5_f64.powi(k as i32)
                    * choose(4 - k, s1)
                    * 0.5_f64.powi((4 - k) as i32);
                let misses = |s: u32, n: u32| -> bool {
                    if n == 0 {
                        return false;
                    }
                    let mean = f64::from(s) / f64::from(n);
                    let r = radius(f64::from(n), per_cell, tuning);
                    (mean - r).max(0.0) > 0.5 || (mean + r).min(1.0) < 0.5
                };
                if misses(s0, k) || misses(s1, 4 - k) {
                    expected += probability;
                }
            }
        }
    }
    close(value, expected, 1e-12, "exclusion probability");
}

#[test]
fn anytime_bound_holds_at_the_stopping_time_of_an_adaptive_procedure() {
    // Thompson sampling with a posterior-probability stopping rule, ten
    // looks: the time-uniform theorem bounds the exclusion probability at
    // the data-dependent stopping time by alpha, for every truth tried.
    let mut d = declaration(thompson(), vec![1; 10]);
    d.stopping =
        Some(StoppingRule::PosteriorProbabilityBest { threshold: 0.85, prior: BetaPrior::UNIFORM });
    let spec = build(d);
    let alpha = 0.2;
    for truth in [[0.5, 0.5], [0.2, 0.8], [0.0, 1.0], [0.9, 0.1], [0.45, 0.55]] {
        let value = anytime_exclusion_probability(&spec, &truth, alpha, 5.0, &never).unwrap();
        assert!(value <= alpha + 1e-12, "exclusion {value} at {truth:?} exceeds {alpha}");
    }
    let report = anytime_valid_report(&spec, &[3, 1, 0, 0], 0.2, 5.0).unwrap();
    assert!((report.alpha_per_cell - 0.1).abs() < 1e-12);
    assert_eq!((report.bounds[1].lower, report.bounds[1].upper), (0.0, 1.0));
    close(
        report.bounds[0].upper,
        (0.75 + radius(4.0, 0.1, 5.0)).min(1.0),
        1e-15,
        "arm 0 upper bound",
    );
}

#[test]
fn anytime_premise_fails_for_contexts() {
    let mut d = declaration(thompson(), vec![1, 1]);
    d.context_probabilities = Some(vec![0.5, 0.5]);
    let spec = build(d);
    assert!(matches!(
        admit(&spec, UncertaintyParadigm::AnytimeValidFrequentist),
        Err(Error::PremiseFailed { detail: "adaptive_inference.anytime_premise_contexts", .. })
    ));
}

// -- Fixed sample and group sequential gates ---------------------------------------------------------

#[test]
fn fixed_sample_and_group_sequential_are_refused_for_adaptive_designs() {
    let adaptive = build(declaration(thompson(), vec![5, 5]));
    assert!(matches!(
        admit(&adaptive, UncertaintyParadigm::FrequentistFixedSample),
        Err(Error::PremiseFailed { detail: "adaptive_inference.fixed_sample_design_adaptive", .. })
    ));
    let mut d = declaration(thompson(), vec![5, 5]);
    d.stopping = Some(StoppingRule::WaldBoundaries {
        critical: vec![2.5, 2.0],
        sidedness: Sidedness::OneSidedUpper,
    });
    assert!(matches!(
        admit(&build(d), UncertaintyParadigm::FrequentistGroupSequential),
        Err(Error::PremiseFailed {
            detail: "adaptive_inference.group_sequential_assignment_adaptive",
            ..
        })
    ));
    let mut data_dependent = declaration(fixed_half(), vec![5, 5]);
    data_dependent.stopping = Some(StoppingRule::ResourceCap { max_units: 5 });
    assert!(matches!(
        admit(&build(data_dependent), UncertaintyParadigm::FrequentistFixedSample),
        Err(Error::PremiseFailed {
            detail: "adaptive_inference.fixed_sample_data_dependent_stopping",
            ..
        })
    ));
    // The fixed design with boundaries is admitted.
    let mut gs = declaration(fixed_half(), vec![5, 5]);
    gs.stopping = Some(StoppingRule::WaldBoundaries {
        critical: vec![2.5, 2.0],
        sidedness: Sidedness::TwoSidedSymmetric,
    });
    assert!(admit(&build(gs), UncertaintyParadigm::FrequentistGroupSequential).is_ok());
}

#[test]
fn wald_boundaries_stop_on_the_declared_statistic() {
    // One look after four fixed-probability units, boundary 1.0: the
    // statistic is the unpooled Wald z of arm 1 against arm 0.
    let mut d = declaration(fixed_half(), vec![4, 0]);
    d.stopping = Some(StoppingRule::WaldBoundaries {
        critical: vec![1.0, 1.0],
        sidedness: Sidedness::OneSidedUpper,
    });
    let spec = build(d);
    let law = law_at(&spec, &[0.2, 0.8]);
    // Independent enumeration of the z rule at look 0.
    let choose = |n: u32, k: u32| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
    };
    let (theta0, theta1) = (0.2_f64, 0.8_f64);
    let mut stop = 0.0;
    for k in 1..=3_u32 {
        for s0 in 0..=k {
            for s1 in 0..=(4 - k) {
                let (n0, n1) = (f64::from(k), f64::from(4 - k));
                let (p0, p1) = (f64::from(s0) / n0, f64::from(s1) / n1);
                let variance = p0 * (1.0 - p0) / n0 + p1 * (1.0 - p1) / n1;
                if variance > 0.0 && (p1 - p0) / variance.sqrt() >= 1.0 {
                    stop += choose(4, k) / 16.0
                        * choose(k, s0)
                        * theta0.powi(s0 as i32)
                        * (1.0 - theta0).powi((k - s0) as i32)
                        * choose(4 - k, s1)
                        * theta1.powi(s1 as i32)
                        * (1.0 - theta1).powi((4 - k - s1) as i32);
                }
            }
        }
    }
    close(law.rule_stop_probabilities()[0], stop, 1e-12, "Wald stop probability");
    assert!(stop > 0.0 && stop < 1.0);
}

// -- Stagewise ordering, inversion --------------------------------------------------------------------

#[test]
fn ordering_p_values_match_the_binomial_enumeration() {
    let spec = build(declaration(fixed_half(), vec![4]));
    let theta = [0.4, 0.6];
    let law = law_at(&spec, &theta);
    let observed = ObservedTerminal { look: 0, rule_stopped: false, decision: None, estimate: 0.5 };
    let p = ordering_p_values(&law, 1, 0, &observed).unwrap();
    let choose = |n: u32, k: u32| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
    };
    let (mut upper, mut lower, mut undefined) = (0.0, 0.0, 0.0);
    for k in 0..=4_u32 {
        for s0 in 0..=k {
            for s1 in 0..=(4 - k) {
                let probability = choose(4, k) / 16.0
                    * choose(k, s0)
                    * theta[0].powi(s0 as i32)
                    * (1.0 - theta[0]).powi((k - s0) as i32)
                    * choose(4 - k, s1)
                    * theta[1].powi(s1 as i32)
                    * (1.0 - theta[1]).powi((4 - k - s1) as i32);
                if k == 0 || k == 4 {
                    undefined += probability;
                    continue;
                }
                let estimate = f64::from(s1) / f64::from(4 - k) - f64::from(s0) / f64::from(k);
                if estimate >= 0.5 - 1e-12 {
                    upper += probability;
                }
                if estimate <= 0.5 + 1e-12 {
                    lower += probability;
                }
            }
        }
    }
    close(p.upper, upper + undefined, 1e-12, "upper tail");
    close(p.lower, lower + undefined, 1e-12, "lower tail");
    close(p.undefined_mass, undefined, 1e-12, "undefined mass");
}

#[test]
fn stagewise_ordering_ranks_an_early_efficacy_stop_above_a_late_one() {
    let mut d = declaration(fixed_half(), vec![4, 4]);
    d.stopping = Some(StoppingRule::WaldBoundaries {
        critical: vec![1.5, 1.5],
        sidedness: Sidedness::OneSidedUpper,
    });
    let spec = build(d);
    let law = law_at(&spec, &[0.3, 0.7]);
    let early = ObservedTerminal { look: 0, rule_stopped: true, decision: Some(1), estimate: 0.6 };
    let late = ObservedTerminal { look: 1, rule_stopped: true, decision: Some(1), estimate: 0.6 };
    let p_early = ordering_p_values(&law, 1, 0, &early).unwrap();
    let p_late = ordering_p_values(&law, 1, 0, &late).unwrap();
    assert!(p_early.upper < p_late.upper, "{} vs {}", p_early.upper, p_late.upper);
    // A stop for efficacy outranks any terminal state that did not stop.
    let horizon = ObservedTerminal { look: 1, rule_stopped: false, decision: None, estimate: 0.9 };
    let p_horizon = ordering_p_values(&law, 1, 0, &horizon).unwrap();
    assert!(p_late.upper < p_horizon.upper);
}

#[test]
fn inversion_matches_the_enumerated_supremum_tests() {
    let spec = build(declaration(fixed_half(), vec![4]));
    let observed = ObservedTerminal { look: 0, rule_stopped: false, decision: None, estimate: 0.5 };
    let grid =
        InversionGrid { contrasts: vec![-0.4, 0.0, 0.2, 0.4, 0.8], baselines: vec![0.1, 0.3, 0.5] };
    let alpha = 0.3;
    let interval = inversion_interval(&spec, &observed, &grid, alpha, &never).unwrap();
    assert_eq!(interval.calibration, DERIVED_UNMEASURED);
    // Independent enumeration of the supremum tail probabilities.
    let choose = |n: u32, k: u32| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * f64::from(n - i) / f64::from(i + 1))
    };
    let mut accepted = Vec::new();
    for &delta in &grid.contrasts {
        let (mut upper_sup, mut lower_sup) = (0.0_f64, 0.0_f64);
        for &b in &grid.baselines {
            let (t0, t1) = (b, b + delta);
            if !(0.0..=1.0).contains(&t1) {
                continue;
            }
            let (mut upper, mut lower) = (0.0, 0.0);
            for k in 0..=4_u32 {
                for s0 in 0..=k {
                    for s1 in 0..=(4 - k) {
                        let probability = choose(4, k) / 16.0
                            * choose(k, s0)
                            * t0.powi(s0 as i32)
                            * (1.0 - t0).powi((k - s0) as i32)
                            * choose(4 - k, s1)
                            * t1.powi(s1 as i32)
                            * (1.0 - t1).powi((4 - k - s1) as i32);
                        if k == 0 || k == 4 {
                            upper += probability;
                            lower += probability;
                            continue;
                        }
                        let estimate =
                            f64::from(s1) / f64::from(4 - k) - f64::from(s0) / f64::from(k);
                        if estimate >= 0.5 - 1e-12 {
                            upper += probability;
                        }
                        if estimate <= 0.5 + 1e-12 {
                            lower += probability;
                        }
                    }
                }
            }
            upper_sup = upper_sup.max(upper);
            lower_sup = lower_sup.max(lower);
        }
        if upper_sup >= alpha / 2.0 && lower_sup >= alpha / 2.0 {
            accepted.push(delta);
        }
    }
    assert_eq!(interval.accepted, accepted);
    // Baselines that would put the plus arm outside [0, 1] are skipped:
    // 1 + 3 + 3 + 3 + 1 laws for the five contrasts.
    assert_eq!(interval.laws_evaluated, 11);
    // A grid with no admissible baseline and the single-context rule.
    let empty = InversionGrid { contrasts: vec![], baselines: vec![0.5] };
    assert!(matches!(
        inversion_interval(&spec, &observed, &empty, alpha, &never),
        Err(Error::Shape(_))
    ));
}

// -- Group-sequential stagewise inference (normal model) -----------------------------------------------

#[test]
fn stagewise_p_value_at_the_boundary_is_the_alpha_spent() {
    let fractions = [0.5, 1.0];
    let alpha = 0.05;
    let b = boundaries(
        &fractions,
        alpha,
        SpendingFunction::OBrienFlemingType,
        Sidedness::TwoSidedSymmetric,
        &never,
    )
    .unwrap();
    // At the first look, z = c_1: upper tail of the spend at t = 1/2.
    let spent_half = 2.0 * norm_sf(1.959_963_984_540_054 / 0.5_f64.sqrt());
    let at_first = stagewise_upper_p_value(
        &fractions,
        &b.critical,
        Sidedness::TwoSidedSymmetric,
        StagewiseLook { look: 0, statistic: b.critical[0] },
        0.0,
        &never,
    )
    .unwrap();
    close(at_first, spent_half / 2.0, 1e-8, "p at the first boundary");
    // At the final boundary the whole upper alpha is spent.
    let at_last = stagewise_upper_p_value(
        &fractions,
        &b.critical,
        Sidedness::TwoSidedSymmetric,
        StagewiseLook { look: 1, statistic: b.critical[1] },
        0.0,
        &never,
    )
    .unwrap();
    close(at_last, alpha / 2.0, 1e-8, "p at the final boundary");
}

#[test]
fn single_look_stagewise_inference_is_the_normal_interval() {
    let z = 2.3;
    let inference = stagewise_inference(
        &[1.0],
        &[1.96],
        Sidedness::OneSidedUpper,
        StagewiseLook { look: 0, statistic: z },
        0.95,
        &never,
    )
    .unwrap();
    close(inference.median_unbiased, z, 1e-8, "median unbiased");
    close(inference.lower, z - 1.959_963_984_540_054, 1e-7, "lower");
    close(inference.upper, z + 1.959_963_984_540_054, 1e-7, "upper");
    close(inference.p_value_at_null, norm_sf(z), 1e-9, "null p-value");
}

#[test]
fn two_look_stagewise_p_value_matches_a_separate_bivariate_integration() {
    // One-sided boundaries 2.5 then 1.9; observed final statistic 1.5 at the
    // second of two equally spaced looks, drift 0.7.
    let (c1, z2, drift) = (2.5_f64, 1.5_f64, 0.7_f64);
    let value = stagewise_upper_p_value(
        &[0.5, 1.0],
        &[c1, 1.9],
        Sidedness::OneSidedUpper,
        StagewiseLook { look: 1, statistic: z2 },
        drift,
        &never,
    )
    .unwrap();
    // P(Z1 >= c1) + integral over z1 < c1 of density(z1) P(Z2 >= z2 | z1),
    // with Z1 ~ N(drift sqrt(1/2), 1) and, given z1,
    // Z2 ~ N(z1 sqrt(1/2) + drift / 2, 1/2).
    let mean1 = drift * 0.5_f64.sqrt();
    let (lo, steps) = (-9.0_f64, 200_000_usize);
    let width = (c1 - lo) / steps as f64;
    let mut integral = 0.0;
    for i in 0..steps {
        let z1 = lo + (i as f64 + 0.5) * width;
        let conditional = norm_sf((z2 - (z1 * 0.5_f64.sqrt() + drift * 0.5)) / 0.5_f64.sqrt());
        integral += norm_pdf(z1 - mean1) * conditional * width;
    }
    let expected = norm_sf(c1 - mean1) + integral;
    close(value, expected, 1e-8, "stagewise p-value");
}

// -- Simulation of order-dependent estimators -----------------------------------------------------------

fn weighting_request(replications: u32) -> SimulationRequest {
    SimulationRequest {
        replications,
        seed: 0x5EED_0001,
        estimand: Estimand::Contrast { plus: 1, minus: 0 },
        scheme: WeightScheme::Uniform,
        outcome_prior: OutcomeModelPrior { mean: 0.5, strength: 2.0 },
        probability_floor: 0.01,
    }
}

#[test]
fn simulated_weighted_estimator_recovers_the_contrast_and_replays() {
    // Epsilon-greedy over 30 sequential units, known contrast 0.2.
    let spec = build(declaration(
        AssignmentRule::EpsilonGreedy { epsilon: 0.4, prior: BetaPrior::UNIFORM },
        vec![2; 15],
    ));
    let truth = [0.4, 0.6];
    let law = simulate_procedure(&spec, &truth, &weighting_request(600), &never).unwrap();
    let summary = law.summarize(0.2, 1.96).unwrap();
    assert!(summary.defined > 590, "defined {}", summary.defined);
    assert!(
        summary.bias.abs() < 4.0 * summary.monte_carlo_standard_error,
        "bias {} with Monte Carlo standard error {}",
        summary.bias,
        summary.monte_carlo_standard_error
    );
    // Streams are keyed by replicate: fewer replications are a prefix.
    let prefix = simulate_procedure(&spec, &truth, &weighting_request(50), &never).unwrap();
    assert_eq!(prefix.replicates[..], law.replicates[..50]);
    close(law.mean_enrolled(), 30.0, 1e-12, "no stopping enrols the horizon");
}

#[test]
fn simulation_under_a_stopping_rule_reports_the_stopped_estimator() {
    let mut d = declaration(thompson(), vec![5; 8]);
    d.stopping =
        Some(StoppingRule::PosteriorProbabilityBest { threshold: 0.9, prior: BetaPrior::UNIFORM });
    d.observation = Some(OutcomeObservation::delayed(1));
    let spec = build(d);
    let law = simulate_procedure(&spec, &[0.2, 0.8], &weighting_request(200), &never).unwrap();
    assert!(law.rule_stop_fraction() > 0.5, "the rule stops early at a large contrast");
    assert!(law.mean_enrolled() < 40.0);
    // A delayed outcome is not analysed before it is observed: at a stop the
    // most recent batch has not entered the estimator.
    assert!(
        law.replicates.iter().all(|r| r.analysed_units < r.enrolled as usize || !r.rule_stopped)
    );
}

#[test]
fn deterministic_assignment_has_no_weighted_estimate() {
    let spec =
        build(declaration(AssignmentRule::UpperConfidenceBound { exploration: 2.0 }, vec![1; 12]));
    let outcome = simulate_procedure(&spec, &[0.4, 0.6], &weighting_request(5), &never);
    assert!(matches!(outcome, Err(Error::Positivity { .. })), "{outcome:?}");
}

#[test]
fn simulation_and_law_refuse_a_truth_of_the_wrong_shape() {
    let spec = build(declaration(fixed_half(), vec![3]));
    assert!(matches!(
        simulate_procedure(&spec, &[0.5], &weighting_request(3), &never),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        enumerate_law(&spec, &OutcomeKernel::Fixed { success_probabilities: vec![0.5] }, &never),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        enumerate_law(
            &spec,
            &OutcomeKernel::Fixed { success_probabilities: vec![0.5, 1.5] },
            &never
        ),
        Err(Error::InvalidParameter(_))
    ));
}
