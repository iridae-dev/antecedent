//! Whole-procedure calibration designs for the participant-level adaptive family.
//!
//! Each design simulates the entire loop in every replicate: recruitment in batches, adaptive
//! (Thompson with an exploration floor) assignment from the exact posterior of the completed
//! participants, correlated beta-binomial responses, the posterior update, the expected-loss
//! stopping rule, the selection of an arm, and the reading of the anytime-valid claim at every
//! look. The retained history of the replicate is then read back under the declared procedure, so
//! the numbers scored are the ones a consumer of the sealed contract would recompute.
//!
//! Scored per replicate (claims about the *procedure*, not about a component):
//!
//! * simultaneous coverage of the anytime-valid family: every arm's sequence contains its true
//!   mean at every look up to the stop, and at the stopping look;
//! * coverage of the selected-arm comparison with the control (the data-selected arm, the
//!   contrast the procedure reports), at every look and at the stop;
//! * the half-width of the arms' sequences at the stop and the width of the selected comparison;
//! * selection error (the selected arm is not among the best arms), regret (the best true mean
//!   less the selected arm's), best-arm identification, and false superiority (some treatment
//!   comparison's lower bound exceeds zero at some look, which under the null the simultaneous
//!   family keeps below its alpha);
//! * the stopping-look and stopping-reason distributions and the sample used.
//!
//! The scenarios are null effects, near ties, separated effects, strong within-participant
//! correlation and misleading historical priors, each at 600, 1,200 and 2,400 participants with
//! four arms and six responses per participant. Every case is its own test, named by the
//! acceptance-manifest case it binds, and its thresholds are the manifest's frozen targets (a
//! default-suite test checks that the two agree). The measured summaries are printed as one
//! `adaptive_whole_procedure_summary` JSON line per case. Nothing is measured at implementation
//! time: the cases are `#[ignore]`d and run only at the final measurement step.
//!
//! The default-suite tests check the harness against scenarios with an exactly known answer: a
//! recruitment cap that fixes the stopping look and the sample used, a truth with an
//! unmistakable best arm, and the summary arithmetic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    reason = "bounded simulation sizes, small indices and rates of counts"
)]

use std::collections::BTreeMap;

use antecedent_stats::adaptive_inference::{
    AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan, DecisionRules, HistoricalPrior,
    LookDecision, LookReading, ParticipantDeclaration, ParticipantSpec, StopReason,
    simulate_participant_history,
};
use serde_json::{Value, json};

fn never() -> bool {
    false
}

const ARMS: usize = 4;
const RESPONSES: u32 = 6;
const LOOKS: usize = 8;
const REPLICATES: u32 = 2000;
const BASE: f64 = 0.4;
const ALPHA: f64 = 0.05;
const LOSS_THRESHOLD: f64 = 0.005;
const FLOOR: f64 = 0.1;

// -- Scenarios -------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    NullEffects,
    NearTies,
    SeparatedEffects,
    StrongCorrelation,
    MisleadingPriors,
}

impl Scenario {
    const fn name(self) -> &'static str {
        match self {
            Self::NullEffects => "null_effects",
            Self::NearTies => "near_ties",
            Self::SeparatedEffects => "separated_effects",
            Self::StrongCorrelation => "strong_correlation",
            Self::MisleadingPriors => "misleading_priors",
        }
    }

    /// Effects over the control, as in the manifest.
    const fn effects(self) -> [f64; ARMS] {
        match self {
            Self::NullEffects => [0.0, 0.0, 0.0, 0.0],
            Self::NearTies => [0.0, 0.0, 0.02, 0.03],
            Self::SeparatedEffects => [0.0, 0.05, 0.1, 0.2],
            Self::StrongCorrelation | Self::MisleadingPriors => [0.0, 0.05, 0.1, 0.15],
        }
    }

    const fn correlation(self) -> f64 {
        match self {
            Self::StrongCorrelation => 0.7,
            _ => 0.3,
        }
    }

    fn truth(self) -> Vec<f64> {
        self.effects().iter().map(|e| BASE + e).collect()
    }
}

/// What a case demands beyond coverage and width.
#[derive(Clone, Copy, Debug)]
enum Extra {
    Nothing,
    /// At most this rate of replicates in which a treatment comparison excluded zero.
    FalseSuperiorityAtMost(f64),
    /// At most this mean regret.
    MeanRegretAtMost(f64),
    /// At least this rate of replicates selecting a best arm.
    BestIdentificationAtLeast(f64),
}

#[derive(Clone, Copy, Debug)]
struct Case {
    id: &'static str,
    scenario: Scenario,
    participants: u32,
    seed: u64,
    max_median_halfwidth: f64,
    extra: Extra,
}

const FLOOR_COVERAGE: f64 = 0.93;

const CASES: [Case; 15] = [
    Case {
        id: "adaptive_full_procedure_null_effects_600",
        scenario: Scenario::NullEffects,
        participants: 600,
        seed: 4_688_630,
        max_median_halfwidth: 0.2,
        extra: Extra::FalseSuperiorityAtMost(0.05),
    },
    Case {
        id: "adaptive_full_procedure_null_effects_1200",
        scenario: Scenario::NullEffects,
        participants: 1200,
        seed: 8_547_647,
        max_median_halfwidth: 0.14,
        extra: Extra::FalseSuperiorityAtMost(0.05),
    },
    Case {
        id: "adaptive_full_procedure_null_effects_2400",
        scenario: Scenario::NullEffects,
        participants: 2400,
        seed: 1_240_419,
        max_median_halfwidth: 0.1,
        extra: Extra::FalseSuperiorityAtMost(0.05),
    },
    Case {
        id: "adaptive_full_procedure_near_ties_600",
        scenario: Scenario::NearTies,
        participants: 600,
        seed: 3_364_870,
        max_median_halfwidth: 0.2,
        extra: Extra::MeanRegretAtMost(0.02),
    },
    Case {
        id: "adaptive_full_procedure_near_ties_1200",
        scenario: Scenario::NearTies,
        participants: 1200,
        seed: 1_599_552,
        max_median_halfwidth: 0.14,
        extra: Extra::MeanRegretAtMost(0.02),
    },
    Case {
        id: "adaptive_full_procedure_near_ties_2400",
        scenario: Scenario::NearTies,
        participants: 2400,
        seed: 3_922_140,
        max_median_halfwidth: 0.1,
        extra: Extra::MeanRegretAtMost(0.02),
    },
    Case {
        id: "adaptive_full_procedure_separated_effects_600",
        scenario: Scenario::SeparatedEffects,
        participants: 600,
        seed: 4_345_524,
        max_median_halfwidth: 0.2,
        extra: Extra::BestIdentificationAtLeast(0.6),
    },
    Case {
        id: "adaptive_full_procedure_separated_effects_1200",
        scenario: Scenario::SeparatedEffects,
        participants: 1200,
        seed: 1_828_172,
        max_median_halfwidth: 0.14,
        extra: Extra::BestIdentificationAtLeast(0.9),
    },
    Case {
        id: "adaptive_full_procedure_separated_effects_2400",
        scenario: Scenario::SeparatedEffects,
        participants: 2400,
        seed: 3_269_392,
        max_median_halfwidth: 0.1,
        extra: Extra::BestIdentificationAtLeast(0.99),
    },
    Case {
        id: "adaptive_full_procedure_strong_correlation_600",
        scenario: Scenario::StrongCorrelation,
        participants: 600,
        seed: 2_966_363,
        max_median_halfwidth: 0.2,
        extra: Extra::Nothing,
    },
    Case {
        id: "adaptive_full_procedure_strong_correlation_1200",
        scenario: Scenario::StrongCorrelation,
        participants: 1200,
        seed: 4_097_367,
        max_median_halfwidth: 0.14,
        extra: Extra::Nothing,
    },
    Case {
        id: "adaptive_full_procedure_strong_correlation_2400",
        scenario: Scenario::StrongCorrelation,
        participants: 2400,
        seed: 1_627_083,
        max_median_halfwidth: 0.1,
        extra: Extra::Nothing,
    },
    Case {
        id: "adaptive_full_procedure_misleading_priors_600",
        scenario: Scenario::MisleadingPriors,
        participants: 600,
        seed: 1_421_029,
        max_median_halfwidth: 0.2,
        extra: Extra::Nothing,
    },
    Case {
        id: "adaptive_full_procedure_misleading_priors_1200",
        scenario: Scenario::MisleadingPriors,
        participants: 1200,
        seed: 5_792_820,
        max_median_halfwidth: 0.14,
        extra: Extra::Nothing,
    },
    Case {
        id: "adaptive_full_procedure_misleading_priors_2400",
        scenario: Scenario::MisleadingPriors,
        participants: 2400,
        seed: 1_978_792,
        max_median_halfwidth: 0.1,
        extra: Extra::Nothing,
    },
];

// -- The procedure under test ----------------------------------------------------------------------

struct Design {
    schedule: Vec<u32>,
    lag_looks: usize,
    correlation: f64,
    borrowing: BorrowingPlan,
    decisions: DecisionRules,
    tuning_samples: f64,
}

fn declare(design: &Design) -> ParticipantSpec {
    ParticipantSpec::new(ParticipantDeclaration {
        arms: ARMS,
        responses: Some(RESPONSES),
        correlation: Some(design.correlation),
        correlation_prior: None,
        borrowing: Some(design.borrowing.clone()),
        enrolment: Some(design.schedule.clone()),
        lag_looks: design.lag_looks,
        min_exploration: Some(FLOOR),
        decisions: Some(design.decisions),
        anytime: Some(AnytimeSettings { alpha: ALPHA, tuning_samples: design.tuning_samples }),
    })
    .expect("a valid procedure")
}

/// Historical sources whose rate is `shift` above each arm's true mean, worth half their trials.
fn history_shifted(truth: &[f64], shift: f64) -> BorrowingPlan {
    BorrowingPlan {
        arms: truth
            .iter()
            .enumerate()
            .map(|(arm, &mean)| ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![HistoricalPrior {
                    source: format!("registry {arm}"),
                    successes: (200.0 * (mean + shift)).round() as u64,
                    trials: 200,
                    discount: 0.5,
                }],
            })
            .collect(),
    }
}

fn case_design(case: &Case) -> Design {
    let truth = case.scenario.truth();
    let borrowing = if case.scenario == Scenario::MisleadingPriors {
        history_shifted(&truth, 0.15)
    } else {
        BorrowingPlan::uniform(ARMS, BetaPrior::UNIFORM)
    };
    Design {
        schedule: vec![case.participants / LOOKS as u32; LOOKS],
        lag_looks: 1,
        correlation: case.scenario.correlation(),
        borrowing,
        decisions: DecisionRules {
            loss_threshold: LOSS_THRESHOLD,
            futility: None,
            participant_cap: None,
        },
        // The sequences are tightest around a quarter of the arm's final sample.
        tuning_samples: f64::from(case.participants) / (ARMS as f64 * 4.0),
    }
}

// -- Scoring one replicate -------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Score {
    every_look: bool,
    at_stop: bool,
    selected_every_look: bool,
    selected_at_stop: bool,
    /// Mean over arms of the sequence half-width at the stop.
    half_width: f64,
    /// Width of the selected arm's comparison with the control at the stop (zero for the control).
    selected_width: f64,
    false_superiority: bool,
    selected: usize,
    regret: f64,
    identified: bool,
    enrolled: u32,
    completed: u32,
    stop_look: usize,
    reason: StopReason,
}

fn contains(lower: f64, upper: f64, value: f64) -> bool {
    lower <= value && value <= upper
}

/// Score the readings of one replicate against the true arm means.
fn score(readings: &[LookReading], truth: &[f64]) -> Option<Score> {
    let last = readings.last()?;
    let LookDecision::Stop { reason, arm: selected } = last.decision else {
        return None;
    };
    let best = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let arm_covered = |reading: &LookReading| {
        reading.frequentist.arms.iter().zip(truth).all(|(a, &t)| contains(a.lower, a.upper, t))
    };
    let selected_covered = |reading: &LookReading| {
        selected == 0
            || reading.frequentist.comparisons.iter().any(|c| {
                c.arm == selected && contains(c.lower, c.upper, truth[selected] - truth[0])
            })
    };
    let arms_at_stop = &last.frequentist.arms;
    let widths: Vec<f64> =
        arms_at_stop.iter().filter(|a| a.count > 0).map(|a| 0.5 * (a.upper - a.lower)).collect();
    let half_width =
        if widths.is_empty() { 0.5 } else { widths.iter().sum::<f64>() / widths.len() as f64 };
    let selected_width = last
        .frequentist
        .comparisons
        .iter()
        .find(|c| c.arm == selected)
        .map_or(0.0, |c| c.upper - c.lower);
    Some(Score {
        every_look: readings.iter().all(&arm_covered),
        at_stop: arm_covered(last),
        selected_every_look: readings.iter().all(&selected_covered),
        selected_at_stop: selected_covered(last),
        half_width,
        selected_width,
        false_superiority: readings
            .iter()
            .any(|r| r.frequentist.comparisons.iter().any(|c| c.lower > 0.0)),
        selected,
        regret: (best - truth[selected]).max(0.0),
        identified: truth[selected] >= best - 1e-12,
        enrolled: last.enrolled,
        completed: last.completed.iter().sum(),
        stop_look: last.look,
        reason,
    })
}

/// One full replicate: run the procedure, retain its history, read it back, score it.
fn replicate(spec: &ParticipantSpec, truth: &[f64], correlation: f64, seed: u64) -> Option<Score> {
    let history =
        simulate_participant_history(spec, truth, Some(correlation), seed, &never).ok()?;
    let readings = spec.read_history(&history, &never).ok()?;
    score(&readings, truth)
}

fn run(
    spec: &ParticipantSpec,
    truth: &[f64],
    correlation: f64,
    seed: u64,
    replications: u32,
) -> Vec<Option<Score>> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from).min(16);
    let mut slots: Vec<Option<Score>> = vec![None; replications as usize];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    (t..replications as usize)
                        .step_by(threads)
                        .map(|r| {
                            let stream =
                                seed.wrapping_add((r as u64 + 1).wrapping_mul(0x9E37_79B9));
                            (r, replicate(spec, truth, correlation, stream))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for handle in handles {
            for (r, outcome) in handle.join().expect("a replicate thread") {
                slots[r] = outcome;
            }
        }
    });
    slots
}

// -- The published summary -------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Summary {
    replications: usize,
    reporting_rate: f64,
    simultaneous_coverage: f64,
    coverage_at_stop: f64,
    selected_coverage_every_look: f64,
    selected_coverage_at_stop: f64,
    median_half_width: f64,
    p90_half_width: f64,
    median_selected_width: f64,
    false_superiority_rate: f64,
    selection_error_rate: f64,
    best_arm_identification_rate: f64,
    mean_regret: f64,
    stop_look_counts: Vec<usize>,
    reason_fractions: BTreeMap<&'static str, f64>,
    mean_enrolled: f64,
    mean_completed: f64,
    sample_use: f64,
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

fn summarize(scores: &[Option<Score>], planned: u32, looks: usize) -> Summary {
    let reported: Vec<&Score> = scores.iter().flatten().collect();
    let n = reported.len() as f64;
    let rate = |f: &dyn Fn(&Score) -> bool| reported.iter().filter(|s| f(s)).count() as f64 / n;
    let mut stop_look_counts = vec![0; looks];
    let mut reasons: BTreeMap<&'static str, f64> = BTreeMap::new();
    for s in &reported {
        stop_look_counts[s.stop_look] += 1;
        *reasons.entry(s.reason.as_str()).or_insert(0.0) += 1.0 / n;
    }
    let mut widths: Vec<f64> = reported.iter().map(|s| s.half_width).collect();
    let mut selected: Vec<f64> = reported.iter().map(|s| s.selected_width).collect();
    let mean_enrolled = reported.iter().map(|s| f64::from(s.enrolled)).sum::<f64>() / n;
    Summary {
        replications: scores.len(),
        reporting_rate: n / scores.len() as f64,
        simultaneous_coverage: rate(&|s| s.every_look),
        coverage_at_stop: rate(&|s| s.at_stop),
        selected_coverage_every_look: rate(&|s| s.selected_every_look),
        selected_coverage_at_stop: rate(&|s| s.selected_at_stop),
        median_half_width: quantile(&mut widths, 0.5),
        p90_half_width: quantile(&mut widths, 0.9),
        median_selected_width: quantile(&mut selected, 0.5),
        false_superiority_rate: rate(&|s| s.false_superiority),
        selection_error_rate: rate(&|s| !s.identified),
        best_arm_identification_rate: rate(&|s| s.identified),
        mean_regret: reported.iter().map(|s| s.regret).sum::<f64>() / n,
        stop_look_counts,
        reason_fractions: reasons,
        mean_enrolled,
        mean_completed: reported.iter().map(|s| f64::from(s.completed)).sum::<f64>() / n,
        sample_use: mean_enrolled / f64::from(planned),
    }
}

impl Summary {
    fn to_json(&self, case: &Case) -> Value {
        json!({
            "case": case.id,
            "scenario": case.scenario.name(),
            "participants": case.participants,
            "arms": ARMS,
            "responses_per_participant": RESPONSES,
            "looks": LOOKS,
            "replications": self.replications,
            "reporting_rate": self.reporting_rate,
            "simultaneous_coverage_all_looks": self.simultaneous_coverage,
            "coverage_at_stop": self.coverage_at_stop,
            "selected_comparison_coverage_all_looks": self.selected_coverage_every_look,
            "selected_comparison_coverage_at_stop": self.selected_coverage_at_stop,
            "median_arm_half_width_at_stop": self.median_half_width,
            "p90_arm_half_width_at_stop": self.p90_half_width,
            "median_selected_comparison_width_at_stop": self.median_selected_width,
            "false_superiority_rate": self.false_superiority_rate,
            "selection_error_rate": self.selection_error_rate,
            "best_arm_identification_rate": self.best_arm_identification_rate,
            "mean_regret": self.mean_regret,
            "stopping_look_counts": self.stop_look_counts,
            "stopping_reason_fractions": self.reason_fractions,
            "mean_participants_enrolled": self.mean_enrolled,
            "mean_participants_completed": self.mean_completed,
            "sample_use_fraction_of_plan": self.sample_use,
            "calibration": "measured_properties_of_this_simulation_not_a_coverage_claim",
        })
    }
}

fn run_case(case: &Case) {
    let truth = case.scenario.truth();
    let design = case_design(case);
    let spec = declare(&design);
    let scores = run(&spec, &truth, case.scenario.correlation(), case.seed, REPLICATES);
    let summary = summarize(&scores, case.participants, LOOKS);
    println!("adaptive_whole_procedure_summary {}", summary.to_json(case));

    assert!(
        (summary.reporting_rate - 1.0).abs() < f64::EPSILON,
        "{}: every replicate reports ({})",
        case.id,
        summary.reporting_rate
    );
    assert!(
        summary.simultaneous_coverage >= FLOOR_COVERAGE,
        "{}: simultaneous coverage across looks and stopping {}",
        case.id,
        summary.simultaneous_coverage
    );
    assert!(
        summary.coverage_at_stop >= FLOOR_COVERAGE,
        "{}: coverage at the stopping look {}",
        case.id,
        summary.coverage_at_stop
    );
    assert!(
        summary.selected_coverage_every_look >= FLOOR_COVERAGE,
        "{}: selected-arm comparison coverage across looks {}",
        case.id,
        summary.selected_coverage_every_look
    );
    assert!(
        summary.selected_coverage_at_stop >= FLOOR_COVERAGE,
        "{}: selected-arm comparison coverage at the stop {}",
        case.id,
        summary.selected_coverage_at_stop
    );
    assert!(
        summary.median_half_width <= case.max_median_halfwidth,
        "{}: median half-width {} exceeds {}",
        case.id,
        summary.median_half_width,
        case.max_median_halfwidth
    );
    match case.extra {
        Extra::Nothing => {}
        Extra::FalseSuperiorityAtMost(limit) => assert!(
            summary.false_superiority_rate <= limit,
            "{}: false superiority {}",
            case.id,
            summary.false_superiority_rate
        ),
        Extra::MeanRegretAtMost(limit) => assert!(
            summary.mean_regret <= limit,
            "{}: mean regret {}",
            case.id,
            summary.mean_regret
        ),
        Extra::BestIdentificationAtLeast(floor) => assert!(
            summary.best_arm_identification_rate >= floor,
            "{}: best-arm identification {}",
            case.id,
            summary.best_arm_identification_rate
        ),
    }
}

fn run_indexed(index: usize, id: &str) {
    let case = &CASES[index];
    assert_eq!(case.id, id);
    run_case(case);
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_null_effects_600() {
    run_indexed(0, "adaptive_full_procedure_null_effects_600");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_null_effects_1200() {
    run_indexed(1, "adaptive_full_procedure_null_effects_1200");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_null_effects_2400() {
    run_indexed(2, "adaptive_full_procedure_null_effects_2400");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_near_ties_600() {
    run_indexed(3, "adaptive_full_procedure_near_ties_600");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_near_ties_1200() {
    run_indexed(4, "adaptive_full_procedure_near_ties_1200");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_near_ties_2400() {
    run_indexed(5, "adaptive_full_procedure_near_ties_2400");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_separated_effects_600() {
    run_indexed(6, "adaptive_full_procedure_separated_effects_600");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_separated_effects_1200() {
    run_indexed(7, "adaptive_full_procedure_separated_effects_1200");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_separated_effects_2400() {
    run_indexed(8, "adaptive_full_procedure_separated_effects_2400");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_strong_correlation_600() {
    run_indexed(9, "adaptive_full_procedure_strong_correlation_600");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_strong_correlation_1200() {
    run_indexed(10, "adaptive_full_procedure_strong_correlation_1200");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_strong_correlation_2400() {
    run_indexed(11, "adaptive_full_procedure_strong_correlation_2400");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_misleading_priors_600() {
    run_indexed(12, "adaptive_full_procedure_misleading_priors_600");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_misleading_priors_1200() {
    run_indexed(13, "adaptive_full_procedure_misleading_priors_1200");
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn adaptive_full_procedure_misleading_priors_2400() {
    run_indexed(14, "adaptive_full_procedure_misleading_priors_2400");
}

// -- Default suite: the harness against scenarios with a known answer ------------------------------

fn small_design(cap: Option<u32>, loss_threshold: f64) -> Design {
    Design {
        schedule: vec![20; 4],
        lag_looks: 0,
        correlation: 0.3,
        borrowing: BorrowingPlan::uniform(ARMS, BetaPrior::UNIFORM),
        decisions: DecisionRules { loss_threshold, futility: None, participant_cap: cap },
        tuning_samples: 20.0,
    }
}

#[test]
fn a_recruitment_cap_fixes_the_stopping_look_and_the_sample_used() {
    // The cap is reached at the second batch whatever the data: the stopping look, the reason,
    // the participants enrolled and completed and the fraction of the plan used are exact.
    let spec = declare(&small_design(Some(40), 1e-300));
    let truth = [0.4; ARMS];
    let scores = run(&spec, &truth, 0.3, 11, 40);
    let summary = summarize(&scores, 80, 4);
    assert!((summary.reporting_rate - 1.0).abs() < f64::EPSILON);
    assert_eq!(summary.stop_look_counts, vec![0, 40, 0, 0]);
    assert_eq!(summary.reason_fractions.len(), 1);
    assert!((summary.reason_fractions["recruitment_cap"] - 1.0).abs() < 1e-12);
    assert!((summary.mean_enrolled - 40.0).abs() < 1e-12);
    assert!((summary.mean_completed - 40.0).abs() < 1e-12);
    assert!((summary.sample_use - 0.5).abs() < 1e-12);
    // Equal arms have no regret and no selection error, whichever arm is selected.
    assert!(summary.mean_regret.abs() < 1e-15);
    assert!(summary.selection_error_rate.abs() < 1e-15);
    // The sequences of a long-run-valid family cover at these sizes, with a width that is positive.
    assert!(summary.simultaneous_coverage >= 0.9, "{}", summary.simultaneous_coverage);
    assert!(summary.median_half_width > 0.0 && summary.median_half_width <= 0.5);
}

#[test]
fn an_unmistakable_best_arm_is_selected_with_no_regret() {
    // Arm 3 always succeeds and the others never do: it is selected by the expected-loss rule in
    // every replicate, regret is zero, and the exact truths 0 and 1 are covered.
    let spec = declare(&small_design(None, 0.005));
    let truth = [0.0, 0.0, 0.0, 1.0];
    let scores = run(&spec, &truth, 0.3, 5, 30);
    let summary = summarize(&scores, 80, 4);
    assert!((summary.reporting_rate - 1.0).abs() < f64::EPSILON);
    assert!((summary.best_arm_identification_rate - 1.0).abs() < 1e-15);
    assert!(summary.mean_regret.abs() < 1e-15);
    assert!((summary.simultaneous_coverage - 1.0).abs() < 1e-15);
    assert!((summary.selected_coverage_at_stop - 1.0).abs() < 1e-15);
    assert!((summary.reason_fractions["expected_loss"] - 1.0).abs() < 1e-12);
    assert!(summary.sample_use < 1.0);
    for s in scores.iter().flatten() {
        assert_eq!(s.selected, 3);
    }
}

#[test]
fn the_summary_arithmetic_matches_a_hand_computation() {
    let score = |every_look: bool, half_width: f64, regret: f64, identified: bool, look: usize| {
        Some(Score {
            every_look,
            at_stop: true,
            selected_every_look: true,
            selected_at_stop: true,
            half_width,
            selected_width: 2.0 * half_width,
            false_superiority: !every_look,
            selected: usize::from(identified),
            regret,
            identified,
            enrolled: 100 * (look as u32 + 1),
            completed: 90 * (look as u32 + 1),
            stop_look: look,
            reason: if look == 2 { StopReason::Horizon } else { StopReason::ExpectedLoss },
        })
    };
    let scores = vec![
        score(true, 0.1, 0.0, true, 0),
        score(true, 0.2, 0.0, true, 1),
        score(false, 0.3, 0.1, false, 2),
        score(true, 0.4, 0.0, true, 1),
        None,
    ];
    let summary = summarize(&scores, 400, 3);
    assert_eq!(summary.replications, 5);
    assert!((summary.reporting_rate - 0.8).abs() < 1e-15);
    assert!((summary.simultaneous_coverage - 0.75).abs() < 1e-15);
    assert!((summary.false_superiority_rate - 0.25).abs() < 1e-15);
    assert!((summary.selection_error_rate - 0.25).abs() < 1e-15);
    assert!((summary.mean_regret - 0.025).abs() < 1e-15);
    assert_eq!(summary.stop_look_counts, vec![1, 2, 1]);
    assert!((summary.reason_fractions["expected_loss"] - 0.75).abs() < 1e-15);
    assert!((summary.reason_fractions["horizon"] - 0.25).abs() < 1e-15);
    // Enrolled: 100, 200, 300, 200; mean 200; used half the plan of 400.
    assert!((summary.mean_enrolled - 200.0).abs() < 1e-12);
    assert!((summary.sample_use - 0.5).abs() < 1e-12);
    // Half-widths 0.1, 0.2, 0.3, 0.4: the median index rounds (3 * 0.5) up to the third value.
    assert!((summary.median_half_width - 0.3).abs() < 1e-15);
    assert!((summary.p90_half_width - 0.4).abs() < 1e-15);
}

#[test]
fn scoring_reads_the_selected_comparison_and_false_superiority_from_the_readings() {
    // A replicate whose procedure never stops cannot be scored, and a history read back under
    // the declared procedure always ends in a stop.
    let spec = declare(&small_design(None, 1e-300));
    let history =
        simulate_participant_history(&spec, &[0.4; ARMS], Some(0.3), 3, &never).expect("a history");
    let readings = spec.read_history(&history, &never).expect("replays");
    assert!(score(&readings[..readings.len() - 1], &[0.4; ARMS]).is_none());
    let full = score(&readings, &[0.4; ARMS]).expect("scores");
    assert_eq!(full.stop_look, readings.len() - 1);
    assert_eq!(full.reason, StopReason::Horizon);
    assert_eq!(full.enrolled, 80);
    // The selected arm's comparison is the one the reading reports for it.
    let last = readings.last().unwrap();
    let width = last
        .frequentist
        .comparisons
        .iter()
        .find(|c| c.arm == full.selected)
        .map_or(0.0, |c| c.upper - c.lower);
    assert!((full.selected_width - width).abs() < 1e-15);
}

#[test]
fn the_cases_agree_with_the_frozen_acceptance_manifest() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/acceptance_manifest.toml");
    let text = std::fs::read_to_string(path).expect("the acceptance manifest");
    for case in &CASES {
        let marker = format!("id = \"{}\"", case.id);
        let start = text.find(&marker).unwrap_or_else(|| panic!("{} is in the manifest", case.id));
        let rest = &text[start..];
        let block = &rest[..rest.find("[[case]]").unwrap_or(rest.len())];
        let mut expected = vec![
            format!("participants = {}", case.participants),
            format!("scenario = \"{}\"", case.scenario.name()),
            format!("replicates = {REPLICATES}"),
            format!("seed = {}", case.seed),
            format!("simultaneous_coverage_floor = {FLOOR_COVERAGE:?}"),
            format!("coverage_floor = {FLOOR_COVERAGE:?}"),
            "min_reporting_rate = 1.0".to_owned(),
            format!("max_median_cs_halfwidth = {:?}", case.max_median_halfwidth),
            format!("arms = {ARMS}"),
            format!("responses_per_participant = {RESPONSES}"),
        ];
        match case.extra {
            Extra::Nothing => {}
            Extra::FalseSuperiorityAtMost(x) => {
                expected.push(format!("max_false_superiority_rate = {x:?}"));
            }
            Extra::MeanRegretAtMost(x) => expected.push(format!("max_mean_regret = {x:?}")),
            Extra::BestIdentificationAtLeast(x) => {
                expected.push(format!("min_best_arm_identification_rate = {x:?}"));
            }
        }
        for needle in expected {
            assert!(block.contains(&needle), "{}: the manifest lacks `{needle}`", case.id);
        }
    }
}
