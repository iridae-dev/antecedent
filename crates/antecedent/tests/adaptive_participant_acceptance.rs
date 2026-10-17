//! Acceptance cases of the participant-level adaptive family through the facade.
//!
//! Each case is a practitioner's use of `ParticipantProcedure` at the size the case states:
//! 600, 1,200 and 2,400 participants with three or four arms and three or six binary responses
//! each, participant-level randomization, historical informative priors, Thompson allocation with
//! a minimum exploration probability, and decisions at completed-participant batch boundaries.
//! The simulated truth is a base rate of 0.4 plus the effects 0, 0.05, 0.1, 0.15 over the
//! control, with an intra-participant correlation of 0.3.
//!
//! Oracles are independent of the code under test: the confidence-sequence readings are compared
//! with a hand-rolled running intersection of the closed-form mixture radius, the posterior
//! spread with the conjugate Beta that ignores the participant structure, the stopping rules with
//! the expected losses and exceedances the readings themselves report, and coverage with the
//! simulated truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "bounded trial sizes and small indices; the oracles compare exact floats on purpose"
)]

use antecedent::analysis::adaptive_inference::{
    AdaptiveInferenceRefusal, AdaptiveInferenceSession, InferenceParadigm,
    consume_adaptive_monitoring_contract_artifact,
};
use antecedent::analysis::adaptive_participant_procedure::ParticipantProcedure;
use antecedent_core::{CancellationToken, ExecutionContext};
use antecedent_stats::adaptive_inference::{
    AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan, DERIVED_UNMEASURED, DecisionRules,
    FutilityRule, HistoricalPrior, LookDecision, LookReading, ParticipantDeclaration,
    ParticipantHistory, ParticipantSimulationRequest, StopReason,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

const BASE: f64 = 0.4;
const EFFECTS: [f64; 4] = [0.0, 0.05, 0.1, 0.15];
const CORRELATION: f64 = 0.3;
const FLOOR: f64 = 0.1;
const LOSS_THRESHOLD: f64 = 0.005;
const ALPHA: f64 = 0.05;
/// An expected-loss threshold no run with this much uncertainty meets: every look is taken.
const NEVER: f64 = 1e-300;

fn truth(arms: usize) -> Vec<f64> {
    EFFECTS[..arms].iter().map(|effect| BASE + effect).collect()
}

fn refused<T: std::fmt::Debug>(
    outcome: Result<T, AdaptiveInferenceRefusal>,
) -> AdaptiveInferenceRefusal {
    outcome.expect_err("the request is refused")
}

fn rules(loss_threshold: f64) -> DecisionRules {
    DecisionRules { loss_threshold, futility: None, participant_cap: None }
}

/// Two historical sources per arm, each worth half its trials, centred on the arm's true rate.
fn historical(arms: usize, discount: f64) -> BorrowingPlan {
    let source = |label: String, rate: f64, trials: u64| HistoricalPrior {
        source: label,
        successes: (rate * trials as f64).round() as u64,
        trials,
        discount,
    };
    BorrowingPlan {
        arms: (0..arms)
            .map(|arm| ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![
                    source(format!("registry {arm}"), BASE + EFFECTS[arm], 150),
                    source(format!("pilot {arm}"), BASE + EFFECTS[arm], 60),
                ],
            })
            .collect(),
    }
}

fn uniform(arms: usize) -> BorrowingPlan {
    BorrowingPlan::uniform(arms, BetaPrior::UNIFORM)
}

struct Shape {
    participants: u32,
    batch: u32,
    arms: usize,
    responses: u32,
    lag: usize,
}

impl Shape {
    fn looks(&self) -> usize {
        (self.participants / self.batch) as usize
    }
}

fn declaration(
    shape: &Shape,
    decisions: DecisionRules,
    borrowing: BorrowingPlan,
) -> ParticipantDeclaration {
    ParticipantDeclaration {
        arms: shape.arms,
        responses: Some(shape.responses),
        correlation: Some(CORRELATION),
        correlation_prior: None,
        borrowing: Some(borrowing),
        enrolment: Some(vec![shape.batch; shape.looks()]),
        lag_looks: shape.lag,
        min_exploration: Some(FLOOR),
        decisions: Some(decisions),
        anytime: Some(AnytimeSettings {
            alpha: ALPHA,
            tuning_samples: f64::from(shape.participants) / (shape.arms as f64 * 4.0),
        }),
    }
}

fn procedure(
    shape: &Shape,
    decisions: DecisionRules,
    borrowing: BorrowingPlan,
) -> ParticipantProcedure {
    ParticipantProcedure::declare(declaration(shape, decisions, borrowing))
        .expect("a complete declaration")
}

fn labels(arms: usize) -> Vec<String> {
    (0..arms).map(|arm| format!("arm {arm}")).collect()
}

// -- Planned procedures execute with checked counters -----------------------------------------------

/// The procedure of the case runs to a declared stop, once with its retained history and receipt
/// and then as a few replications whose receipt adds up.
fn procedure_runs_to_a_declared_stop(participants: u32, batch: u32, seed: u64) {
    let shape = Shape { participants, batch, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(LOSS_THRESHOLD), uniform(4));
    let report = p.run(&truth(4), Some(CORRELATION), seed, &ctx()).expect("the procedure runs");
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
    let run = &report.run;
    assert!(run.complete, "a run nobody cancelled reaches its stop");
    assert_eq!(run.looks_planned, shape.looks());
    let performed = run.readings.len();
    assert!((1..=shape.looks()).contains(&performed));
    let LookDecision::Stop { reason, arm } = run.readings[performed - 1].decision else {
        panic!("the procedure ended without a declared stop");
    };
    assert!(matches!(reason, StopReason::ExpectedLoss | StopReason::Horizon), "{reason:?}");
    assert!(arm < 4);
    assert!(run.readings[..performed - 1].iter().all(|r| r.decision == LookDecision::Continue));
    if reason == StopReason::Horizon {
        assert_eq!(performed, shape.looks());
    }
    // Whole batches only.
    assert_eq!(run.history.batches, vec![batch; performed]);
    let enrolled = batch * performed as u32;

    // The receipt's counters: looks, states, work units and bytes, against what was admitted.
    let receipt = run.receipt;
    assert_eq!(receipt.replications, 1);
    assert_eq!(receipt.looks, performed as u64);
    assert_eq!(
        receipt.states,
        4 * (3 + 1),
        "four arms times the four success counts of a participant"
    );
    assert_eq!(receipt.work_planned, u64::from(participants) * 3);
    assert_eq!(receipt.work_done, u64::from(enrolled) * 3);
    assert_eq!(receipt.peak_units, enrolled);
    assert!(receipt.estimated_peak_bytes > 0);

    // A few replications add up in the same counters.
    let replications = 4;
    let simulation = p
        .simulate(
            &truth(4),
            &ParticipantSimulationRequest {
                replications,
                seed,
                true_correlation: Some(CORRELATION),
            },
            &ctx(),
        )
        .expect("the replications run");
    let receipt = simulation.simulation.receipt;
    assert_eq!(receipt.replications, replications);
    assert_eq!(receipt.work_planned, u64::from(replications) * u64::from(participants) * 3);
    let replicates = &simulation.simulation.replicates;
    assert_eq!(replicates.len(), replications as usize);
    assert!(
        replicates
            .iter()
            .all(|r| matches!(r.reason, StopReason::ExpectedLoss | StopReason::Horizon))
    );
    assert_eq!(receipt.looks, replicates.iter().map(|r| r.look as u64 + 1).sum::<u64>());
    assert_eq!(
        receipt.work_done,
        replicates.iter().map(|r| u64::from(r.enrolled) * 3).sum::<u64>()
    );
    assert_eq!(receipt.peak_units, replicates.iter().map(|r| r.enrolled).max().unwrap());
    assert_eq!(receipt.states, 16);
    assert!(receipt.work_done <= receipt.work_planned);
    assert!(receipt.estimated_peak_bytes > 0);
}

#[test]
fn adaptive_procedure_executes_600_participants() {
    procedure_runs_to_a_declared_stop(600, 50, 6_562_783);
}

#[test]
fn adaptive_procedure_executes_1200_participants() {
    procedure_runs_to_a_declared_stop(1_200, 100, 6_860_415);
}

#[test]
fn adaptive_procedure_executes_2400_participants() {
    procedure_runs_to_a_declared_stop(2_400, 200, 7_453_855);
}

// -- Cancellation between batches ---------------------------------------------------------------------

#[test]
fn adaptive_run_cancellation_returns_partial_receipt() {
    let shape = Shape { participants: 2_400, batch: 200, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(NEVER), uniform(4));
    let seed = 6_718_126;
    let whole = p.run(&truth(4), Some(CORRELATION), seed, &ctx()).unwrap().run;
    assert!(whole.complete);
    assert_eq!(whole.readings.len(), 12);

    // The token is polled before the first batch and after each batch that does not end the
    // run: it trips at the check after the third look.
    let mut tripping = ctx();
    tripping.cancellation = CancellationToken::cancel_after_checks(3);
    let partial = p
        .run(&truth(4), Some(CORRELATION), seed, &tripping)
        .expect("a cancelled run is a partial result, not a refusal")
        .run;
    assert!(!partial.complete, "the receipt marks the run incomplete");
    assert_eq!(partial.looks_planned, 12);
    // Honoured within one batch: three whole batches were taken, none of a fourth.
    assert_eq!(partial.readings.len(), 3);
    assert_eq!(partial.history.batches, vec![200; 3]);
    assert_eq!(partial.history.arms.len(), 600);
    assert_eq!(partial.history.successes.len(), 600);
    assert_eq!(partial.history.assignment.len(), 600);
    // One look of delay: the first two batches completed, the third had not.
    assert!(partial.history.successes[..400].iter().all(Option::is_some));
    assert!(partial.history.successes[400..].iter().all(Option::is_none));
    // The receipt counts what was done against what was admitted.
    let receipt = partial.receipt;
    assert_eq!(receipt.replications, 1);
    assert_eq!(receipt.looks, 3);
    assert_eq!(receipt.peak_units, 600);
    assert_eq!(receipt.work_done, 600 * 3);
    assert_eq!(receipt.work_planned, 2_400 * 3);
    assert!(receipt.work_done < receipt.work_planned);
    // What completed is exactly the beginning of the uncancelled run, and nothing is decided on it.
    assert_eq!(&partial.readings[..], &whole.readings[..3]);
    assert_eq!(&partial.history.arms[..], &whole.history.arms[..600]);
    assert!(partial.readings.iter().all(|r| r.decision == LookDecision::Continue));

    // A token that was cancelled before the first batch enrols no one.
    let before = ctx();
    before.cancellation.cancel();
    let none = p.run(&truth(4), Some(CORRELATION), seed, &before).unwrap().run;
    assert!(!none.complete);
    assert!(none.readings.is_empty() && none.history.arms.is_empty());
    assert_eq!((none.receipt.looks, none.receipt.work_done, none.receipt.peak_units), (0, 0, 0));
    assert_eq!(none.receipt.work_planned, 2_400 * 3);

    // A run that reached its stopping look before the token tripped is complete.
    let decisive = procedure(&shape, rules(0.05), uniform(4));
    let mut late = ctx();
    late.cancellation = CancellationToken::cancel_after_checks(11);
    let finished = decisive.run(&[0.1, 0.9, 0.1, 0.1], Some(CORRELATION), seed, &late).unwrap().run;
    assert!(finished.complete);
    assert!(finished.readings.len() < 12);
    assert!(matches!(
        finished.readings.last().unwrap().decision,
        LookDecision::Stop { reason: StopReason::ExpectedLoss, arm: 1 }
    ));
}

// -- The artifact limit is its own refusal ------------------------------------------------------------

#[test]
fn adaptive_artifact_size_limit_distinct_refusal() {
    let shape = Shape { participants: 2_400, batch: 200, arms: 4, responses: 6, lag: 1 };
    let p = procedure(&shape, rules(NEVER), historical(4, 0.5));
    let run = p.run(&truth(4), Some(CORRELATION), 5_827_042, &ctx()).unwrap().run;
    assert!(run.complete, "the run itself completed");
    assert_eq!(run.history.arms.len(), 2_400);
    let declaration = p.retained_declaration("trial", labels(4), &run.history, &ctx()).unwrap();
    let session = AdaptiveInferenceSession::prepare(declaration).expect("a complete contract");
    let paradigm = InferenceParadigm::BayesianPosterior;
    let full = session.export_contract_with_procedure(paradigm, &p, "trial").expect("seals");

    // An artifact that exactly fits its budget is the same artifact.
    let within = session
        .export_contract_with_procedure_within(paradigm, &p, "trial", full.len())
        .expect("fits exactly");
    assert_eq!(within, full);

    // One byte less is refused as the artifact limit, naming the bytes needed and the budget.
    let budget = full.len() - 1;
    let refusal =
        refused(session.export_contract_with_procedure_within(paradigm, &p, "trial", budget));
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "adaptive_inference.artifact_size_exceeded");
    assert!(refusal.message.contains(&full.len().to_string()), "{}", refusal.message);
    assert!(refusal.message.contains(&budget.to_string()), "{}", refusal.message);
    assert!(refusal.message.contains("export budget"), "{}", refusal.message);
    assert!(refusal.message.contains("run is complete"), "{}", refusal.message);
    for other in [
        "adaptive_inference.retained_record_exceeded",
        "adaptive_inference.simulation_memory_exceeded",
        "adaptive_inference.simulation_work_exceeded",
        "adaptive_inference.declared_units_exceeded",
        "adaptive_inference.exact_law_states_exceeded",
    ] {
        assert_ne!(refusal.detail, other, "the artifact limit is not a limit of the run");
    }
    let tiny = refused(session.export_contract_with_procedure_within(paradigm, &p, "trial", 1_024));
    assert_eq!(tiny.detail, "adaptive_inference.artifact_size_exceeded");

    // The refusal leaves the run and its procedure untouched: the history still reads, and the
    // artifact still exports and is consumed by replay.
    assert_eq!(p.read_history(&run.history, &ctx()).unwrap().readings.len(), run.readings.len());
    let consumed = consume_adaptive_monitoring_contract_artifact(&full, None).unwrap();
    assert_eq!(consumed.procedure_record().unwrap().readings().len(), run.readings.len());
}

// -- All four capabilities in one procedure ------------------------------------------------------------

/// Closed-form standard deviation of a Beta posterior that treats every response as an
/// independent trial, the model the participant structure rules out.
fn independent_response_sd(prior: &BetaPrior, successes: f64, failures: f64) -> f64 {
    let (a, b) = (prior.alpha + successes, prior.beta + failures);
    (a * b / ((a + b).powi(2) * (a + b + 1.0))).sqrt()
}

fn all_four_capabilities(shape: &Shape, seed: u64) {
    let discount = 0.5;
    let p = procedure(shape, rules(LOSS_THRESHOLD), historical(shape.arms, discount));
    let arms = shape.arms;
    let run = p.run(&truth(arms), Some(CORRELATION), seed, &ctx()).expect("runs").run;
    assert!(run.complete);
    let report = p.read_history(&run.history, &ctx()).expect("the history replays");
    let readings = &report.readings;
    assert_eq!(readings.len(), run.readings.len());
    let history = &run.history;
    let enrolled: usize = history.batches.iter().map(|&n| n as usize).sum();

    // Participant-level randomization: every participant has one arm, drawn under the allocation
    // of the batch they entered, and gives at most `responses` binary responses.
    assert_eq!(history.arms.len(), enrolled);
    let mut participant = 0;
    for (look, &size) in history.batches.iter().enumerate() {
        for _ in 0..size {
            assert!(history.arms[participant] < arms);
            assert_eq!(history.assignment[participant], readings[look].assignment_probabilities);
            if let Some(successes) = history.successes[participant] {
                assert!(successes <= shape.responses);
            }
            participant += 1;
        }
    }

    // One allocation policy with a minimum exploration probability: no participant was
    // randomized under a probability below it, every row is a distribution, and the policy adapts
    // to the data rather than splitting evenly.
    let even = 1.0 / arms as f64;
    for row in &history.assignment {
        assert_eq!(row.len(), arms);
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(row.iter().all(|&q| q >= FLOOR - 1e-12), "floor violated: {row:?}");
    }
    assert!(
        history.assignment.iter().any(|row| row.iter().any(|&q| (q - even).abs() > 1e-6)),
        "the allocation adapts"
    );

    // The participant-aware likelihood: each arm's posterior is wider than the one that treats
    // every response as an independent trial.
    let last = readings.last().unwrap();
    for arm in 0..arms {
        let (mut successes, mut failures) = (0.0, 0.0);
        for (&a, s) in history.arms.iter().zip(&history.successes) {
            if let (true, Some(s)) = (a == arm, s) {
                successes += f64::from(*s);
                failures += f64::from(shape.responses - *s);
            }
        }
        assert!(successes + failures > 0.0, "arm {arm} has completed participants");
        let naive = independent_response_sd(&p.spec().priors()[arm].prior, successes, failures);
        assert!(
            last.posterior.sd[arm] > naive,
            "arm {arm}: sd {} not above the independent-response sd {naive}",
            last.posterior.sd[arm]
        );
    }

    // Decision-based stopping: no earlier look met the expected-loss threshold; the last look
    // either decided for the arm of least expected loss or ran the schedule.
    let least = |reading: &LookReading| {
        reading.posterior.expected_loss.iter().copied().fold(f64::INFINITY, f64::min)
    };
    for reading in &readings[..readings.len() - 1] {
        assert_eq!(reading.decision, LookDecision::Continue);
        assert!(least(reading) >= LOSS_THRESHOLD);
    }
    match last.decision {
        LookDecision::Stop { reason: StopReason::ExpectedLoss, arm } => {
            assert!(least(last) < LOSS_THRESHOLD);
            assert_eq!(last.posterior.expected_loss[arm], least(last));
        }
        LookDecision::Stop { reason: StopReason::Horizon, .. } => {
            assert_eq!(readings.len(), shape.looks());
            assert!(least(last) >= LOSS_THRESHOLD);
        }
        other => panic!("not a declared stop: {other:?}"),
    }

    // Historical sources, discounts and effective sample sizes are recorded, and the posterior
    // before any participant completes is the resolved prior.
    assert_eq!(report.priors.len(), arms);
    for (arm, prior) in report.priors.iter().enumerate() {
        assert_eq!(prior.sources.len(), 2);
        assert_eq!(prior.sources[0].source, format!("registry {arm}"));
        assert_eq!(prior.sources[1].source, format!("pilot {arm}"));
        assert!(prior.sources.iter().all(|s| s.discount == discount));
        let rate = BASE + EFFECTS[arm];
        let (s1, s2) = ((150.0 * rate).round(), (60.0 * rate).round());
        assert_eq!(prior.prior.alpha, 1.0 + discount * (s1 + s2));
        assert_eq!(prior.prior.beta, 1.0 + discount * ((150.0 - s1) + (60.0 - s2)));
        assert_eq!(prior.borrowed_sample_size, discount * 210.0);
        assert_eq!(prior.effective_sample_size, 2.0 + discount * 210.0);
        assert_eq!(report.borrowing[arm].effective_historical_sample_size, discount * 210.0);
        assert_eq!(report.borrowing[arm].sources, prior.sources);
        assert_eq!(readings[0].completed, vec![0; arms], "nobody has completed at the first look");
        let (a, b) = (prior.prior.alpha, prior.prior.beta);
        assert!((readings[0].posterior.mean[arm] - a / (a + b)).abs() < 1e-9);
    }
}

#[test]
fn adaptive_beta_binomial_three_arm_600_three_responses() {
    let shape = Shape { participants: 600, batch: 50, arms: 3, responses: 3, lag: 1 };
    all_four_capabilities(&shape, 7_103_449);
}

#[test]
fn adaptive_beta_binomial_four_arm_1200_six_responses() {
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 6, lag: 1 };
    all_four_capabilities(&shape, 5_402_708);
}

#[test]
fn adaptive_beta_binomial_four_arm_2400() {
    let shape = Shape { participants: 2_400, batch: 200, arms: 4, responses: 6, lag: 1 };
    all_four_capabilities(&shape, 6_681_309);
}

// -- Borrowing is frozen before recruitment ------------------------------------------------------------

#[test]
fn adaptive_borrowing_frozen_before_recruitment() {
    let shape = Shape { participants: 600, batch: 50, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(NEVER), historical(4, 0.5));
    let other = historical(4, 0.8);

    // Before recruitment the borrowing is open: another plan is another declaration.
    let amended = p.amend_borrowing(&other, 0).expect("open before the first participant");
    assert_ne!(amended.identity(), p.identity());
    assert_eq!(amended.spec().borrowing(), &other);

    // After the first participant it is frozen: the data would choose the prior.
    let run = p.run(&truth(4), Some(CORRELATION), 2_756_012, &ctx()).unwrap().run;
    let performed = run.readings.len();
    assert!(performed >= 1);
    for looks in [1, performed] {
        let frozen = refused(p.amend_borrowing(&other, looks));
        assert_eq!(frozen.code, "adaptive_assumption_violated");
        assert_eq!(frozen.detail, "adaptive_inference.borrowing_frozen");
        assert!(frozen.message.contains("frozen"), "{}", frozen.message);
    }
    // Re-declaring the same plan is no change.
    assert_eq!(
        p.amend_borrowing(p.spec().borrowing(), performed).unwrap().identity(),
        p.identity()
    );

    // A procedure with another plan cannot read the history that ran under this one: the
    // allocation it produced does not replay.
    let replay = refused(amended.read_history(&run.history, &ctx()));
    assert_eq!(replay.detail, "adaptive_inference.assignment_not_replayed");
}

// -- The minimum exploration probability ---------------------------------------------------------------

#[test]
fn adaptive_minimum_exploration_enforced() {
    // Arm 0 is hopelessly dominated: unfloored Thompson sampling would stop assigning it.
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 3, lag: 0 };
    let p = procedure(&shape, rules(NEVER), uniform(4));
    let dominated = [0.1, 0.7, 0.7, 0.7];
    let run = p.run(&dominated, Some(CORRELATION), 3_445_060, &ctx()).unwrap().run;
    assert!(run.complete);
    assert_eq!(run.readings.len(), 12);
    let history = &run.history;
    let rows: Vec<&Vec<f64>> = history.assignment.iter().collect();
    assert_eq!(rows.len(), 1_200);
    let smallest = rows.iter().flat_map(|row| row.iter().copied()).fold(f64::INFINITY, f64::min);
    assert!(smallest >= FLOOR - 1e-12, "an assignment probability {smallest} is below the floor");
    // The floor binds on the dominated arm at some look, so the guarantee is not idle.
    let dominated_arm: Vec<f64> = rows.iter().map(|row| row[0]).collect();
    assert!(
        dominated_arm.iter().any(|&q| (q - FLOOR).abs() < 1e-9),
        "the dominated arm sits on the floor at some look"
    );
    // The arm keeps being assigned: roughly a tenth of the later participants.
    let later = &history.arms[100..];
    let share = later.iter().filter(|&&arm| arm == 0).count() as f64 / later.len() as f64;
    assert!((0.07..0.2).contains(&share), "share of the dominated arm {share}");
    // Each posterior probability of being best is far below the floor for the dominated arm, so
    // it is the floor, not the data, that keeps the arm in the trial.
    let last = run.readings.last().unwrap();
    assert!(last.posterior.probability_best[0] < FLOOR / 10.0);

    // A floor no allocation could honour is refused at declaration, not clipped.
    let mut too_high = declaration(&shape, rules(NEVER), uniform(4));
    too_high.min_exploration = Some(0.3);
    let refusals = ParticipantProcedure::declare(too_high).unwrap_err();
    assert_eq!(refusals[0].detail, "adaptive_inference.rule_mismatch");
}

// -- Decisions at completed-participant batch boundaries -----------------------------------------------

/// The batch boundary a stop was taken at: a whole number of batches enrolled, the participants
/// of the last `lag` batches not yet complete, and a recorded look index.
fn assert_whole_batches(
    readings: &[LookReading],
    history: &ParticipantHistory,
    batch: u32,
    lag: usize,
) {
    let last = readings.last().unwrap();
    assert_eq!(last.look, readings.len() - 1, "the decision records its look index");
    assert_eq!(last.enrolled, batch * readings.len() as u32);
    let completed: u32 = last.completed.iter().sum();
    assert_eq!(completed, batch * (readings.len() - lag.min(readings.len())) as u32);
    for (look, reading) in readings.iter().enumerate() {
        assert_eq!(reading.enrolled % batch, 0, "no decision on a partial batch");
        assert_eq!(reading.enrolled, batch * (look as u32 + 1));
    }
    assert_eq!(history.batches, vec![batch; readings.len()]);
    assert_eq!(history.arms.len() as u32, last.enrolled);
}

#[test]
fn adaptive_expected_loss_stop_at_completed_batch_boundary() {
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(LOSS_THRESHOLD), uniform(4));
    let run = p.run(&[0.2, 0.7, 0.2, 0.2], Some(CORRELATION), 9_274_147, &ctx()).unwrap().run;
    let report = p.read_history(&run.history, &ctx()).unwrap();
    let readings = &report.readings;
    assert!(readings.len() < shape.looks(), "a decisive truth stops before the horizon");
    let last = readings.last().unwrap();
    assert!(matches!(
        last.decision,
        LookDecision::Stop { reason: StopReason::ExpectedLoss, arm: 1 }
    ));
    assert!(last.posterior.expected_loss[1] < LOSS_THRESHOLD);
    for reading in &readings[..readings.len() - 1] {
        assert_eq!(reading.decision, LookDecision::Continue);
        assert!(
            reading.posterior.expected_loss.iter().all(|&loss| loss >= LOSS_THRESHOLD),
            "no earlier look met the threshold"
        );
    }
    assert_whole_batches(readings, &run.history, 100, 1);
}

#[test]
fn adaptive_futility_stop_at_completed_batch_boundary() {
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 3, lag: 1 };
    let futility = FutilityRule { margin: 0.1, threshold: 0.15 };
    let decisions =
        DecisionRules { loss_threshold: NEVER, futility: Some(futility), participant_cap: None };
    let p = procedure(&shape, decisions, uniform(4));
    let run = p.run(&[0.3; 4], Some(CORRELATION), 7_073_496, &ctx()).unwrap().run;
    let report = p.read_history(&run.history, &ctx()).unwrap();
    let readings = &report.readings;
    let last = readings.last().unwrap();
    assert!(matches!(last.decision, LookDecision::Stop { reason: StopReason::Futility, arm: 0 }));
    assert!(readings.len() <= shape.looks());
    // No treatment is likely to beat the control by the margin at the stop; some was before it.
    assert!(last.posterior.exceeds_control.iter().all(|&q| q < futility.threshold));
    for reading in &readings[..readings.len() - 1] {
        assert_eq!(reading.decision, LookDecision::Continue);
        assert!(reading.posterior.exceeds_control.iter().any(|&q| q >= futility.threshold));
    }
    assert_whole_batches(readings, &run.history, 100, 1);
}

#[test]
fn adaptive_recruitment_cap_stop_at_completed_batch_boundary() {
    let shape = Shape { participants: 600, batch: 50, arms: 4, responses: 3, lag: 1 };
    let capped = |cap: u32| {
        let decisions =
            DecisionRules { loss_threshold: NEVER, futility: None, participant_cap: Some(cap) };
        procedure(&shape, decisions, uniform(4))
    };
    // A cap on a boundary ends the run there.
    let p = capped(300);
    let run = p.run(&truth(4), Some(CORRELATION), 2_822_811, &ctx()).unwrap().run;
    let readings = &p.read_history(&run.history, &ctx()).unwrap().readings;
    let last = readings.last().unwrap();
    assert!(matches!(last.decision, LookDecision::Stop { reason: StopReason::RecruitmentCap, .. }));
    assert_eq!((last.look, last.enrolled), (5, 300));
    assert_whole_batches(readings, &run.history, 50, 1);

    // A cap between two boundaries waits for the next: a batch is never split.
    let p = capped(310);
    let run = p.run(&truth(4), Some(CORRELATION), 2_822_811, &ctx()).unwrap().run;
    let readings = &p.read_history(&run.history, &ctx()).unwrap().readings;
    let last = readings.last().unwrap();
    assert!(matches!(last.decision, LookDecision::Stop { reason: StopReason::RecruitmentCap, .. }));
    assert_eq!((last.look, last.enrolled), (6, 350));
    assert_whole_batches(readings, &run.history, 50, 1);
}

#[test]
fn adaptive_partial_participant_batch_refused() {
    let shape = Shape { participants: 600, batch: 50, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(NEVER), uniform(4));
    let run = p.run(&truth(4), Some(CORRELATION), 3_914_842, &ctx()).unwrap().run;
    assert_eq!(run.readings.len(), 12);
    assert!(p.read_history(&run.history, &ctx()).is_ok(), "the whole history reads");

    // Seven participants of a batch whose follow-up has elapsed have no complete responses.
    let mut partial = run.history.clone();
    for participant in 0..7 {
        partial.successes[participant] = None;
    }
    let refusal = refused(p.read_history(&partial, &ctx()));
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.batch_incomplete");
    assert!(refusal.message.contains("participants without complete responses"));
    assert!(refusal.message.contains("never on a partial batch"));
    // The same history cannot be retained as a contract either.
    let retained = refused(p.retained_declaration("partial", labels(4), &partial, &ctx()));
    assert_eq!(retained.detail, "adaptive_inference.batch_incomplete");
    // One incomplete participant is enough.
    let mut one = run.history.clone();
    one.successes[49] = None;
    assert_eq!(refused(p.read_history(&one, &ctx())).detail, "adaptive_inference.batch_incomplete");
    // A participant reported complete before the delay elapsed is a different disagreement.
    let mut early = run.history.clone();
    let last = early.successes.len() - 1;
    early.successes[last] = Some(1);
    assert_ne!(
        refused(p.read_history(&early, &ctx())).detail,
        "adaptive_inference.batch_incomplete"
    );
}

// -- Confidence sequences at every look ------------------------------------------------------------------

/// The closed-form radius of the normal-mixture Hoeffding sequence after `n` observations.
fn radius(n: f64, alpha: f64, tuning: f64) -> f64 {
    0.5 * ((n + tuning) * ((n + tuning) / (tuning * alpha * alpha)).ln()).sqrt() / n
}

/// The running-intersection sequence of `values` (in pull order).
fn running_intersection(values: &[f64], alpha: f64, tuning: f64) -> (f64, f64) {
    let (mut lower, mut upper, mut sum) = (0.0_f64, 1.0_f64, 0.0);
    for (index, value) in values.iter().enumerate() {
        sum += value;
        let n = index as f64 + 1.0;
        let mean = sum / n;
        let r = radius(n, alpha, tuning);
        lower = lower.max(mean - r);
        upper = upper.min(mean + r);
    }
    (lower, upper)
}

#[test]
fn adaptive_confidence_sequence_reported_at_every_look() {
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(NEVER), uniform(4));
    let run = p.run(&truth(4), Some(CORRELATION), 7_634_056, &ctx()).unwrap().run;
    let report = p.read_history(&run.history, &ctx()).unwrap();
    let readings = &report.readings;
    assert_eq!(readings.len(), 12, "twelve looks were performed");
    let history = &run.history;
    let tuning = 1_200.0 / 16.0;
    // Alpha over the three comparisons with the control, each over its two arms.
    let per_arm = ALPHA / (3.0 * 2.0);
    for (look, reading) in readings.iter().enumerate() {
        assert_eq!(reading.look, look);
        let f = &reading.frequentist;
        assert_eq!(f.paradigm, "anytime_valid_frequentist");
        assert_eq!(f.calibration, DERIVED_UNMEASURED);
        assert!((f.alpha - ALPHA).abs() < 1e-15);
        assert!((f.alpha_per_arm - per_arm).abs() < 1e-15);
        assert_eq!(f.arms.len(), 4);
        assert_eq!(f.comparisons.len(), 3);
        // Participants whose follow-up completed by this look: the batches before it.
        let end = 100 * look;
        for arm in 0..4 {
            let proportions: Vec<f64> = (0..end)
                .filter(|&i| history.arms[i] == arm)
                .map(|i| f64::from(history.successes[i].unwrap()) / 3.0)
                .collect();
            assert_eq!(f.arms[arm].count as usize, proportions.len());
            let (lower, upper) = running_intersection(&proportions, per_arm, tuning);
            assert!((f.arms[arm].lower - lower).abs() < 1e-12, "look {look} arm {arm}");
            assert!((f.arms[arm].upper - upper).abs() < 1e-12, "look {look} arm {arm}");
            assert!(f.arms[arm].lower <= f.arms[arm].upper);
        }
        for comparison in &f.comparisons {
            assert!(
                (comparison.lower - (f.arms[comparison.arm].lower - f.arms[0].upper)).abs() < 1e-15
            );
            assert!(
                (comparison.upper - (f.arms[comparison.arm].upper - f.arms[0].lower)).abs() < 1e-15
            );
        }
    }
    // The sequences only narrow, look after look.
    for pair in readings.windows(2) {
        for arm in 0..4 {
            assert!(pair[1].frequentist.arms[arm].lower >= pair[0].frequentist.arms[arm].lower);
            assert!(pair[1].frequentist.arms[arm].upper <= pair[0].frequentist.arms[arm].upper);
        }
    }
}

// -- Simultaneous inference and the data-selected alternative ---------------------------------------------

const COVERAGE_REPLICATES: u64 = 120;
const COVERAGE_FLOOR: f64 = 0.93;

/// Run `COVERAGE_REPLICATES` retained histories of the 1,200-participant, four-arm trial under
/// `truth`, read each look by look, and hand every replicate's readings to `score`.
fn replicate_readings(truth: &[f64], seed: u64, mut score: impl FnMut(&[LookReading])) {
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(LOSS_THRESHOLD), uniform(4));
    for replicate in 0..COVERAGE_REPLICATES {
        let context = ctx();
        let history = p
            .simulate_history(truth, Some(CORRELATION), seed + replicate, &context)
            .expect("a history");
        let report = p.read_history(&history, &context).expect("the history replays");
        score(&report.readings);
    }
}

#[test]
fn adaptive_simultaneous_comparisons_against_control() {
    let truth = truth(4);
    let mut covered = 0_u64;
    let mut checked_looks = 0_usize;
    replicate_readings(&truth, 7_574_789, |readings| {
        // Every contrast with the control, at every look to the stop, at once.
        let all = readings.iter().all(|reading| {
            let f = &reading.frequentist;
            assert_eq!(f.comparisons.len(), 3, "one contrast per arm versus the control");
            assert!((f.alpha_per_comparison - ALPHA / 3.0).abs() < 1e-15);
            f.comparisons.iter().enumerate().all(|(index, c)| {
                assert_eq!(c.arm, index + 1);
                c.lower <= truth[c.arm] - truth[0] && truth[c.arm] - truth[0] <= c.upper
            })
        });
        covered += u64::from(all);
        checked_looks += readings.len();
    });
    let coverage = covered as f64 / COVERAGE_REPLICATES as f64;
    assert!(checked_looks >= COVERAGE_REPLICATES as usize);
    assert!(
        coverage >= COVERAGE_FLOOR,
        "simultaneous coverage of the three contrasts at every look: {coverage}"
    );
}

#[test]
fn adaptive_selected_alternative_data_chosen() {
    // Equal arms: the "best" alternative is picked from noise, which is where a naive interval
    // for the picked arm under-covers. The family covers whichever arm the data pick.
    let truth = [BASE; 4];
    let (mut at_stop, mut every_look) = (0_u64, 0_u64);
    let mut picked = [0_u32; 4];
    replicate_readings(&truth, 9_173_130, |readings| {
        let covers = |reading: &LookReading| {
            let f = &reading.frequentist;
            let chosen = f.best_treatment;
            // The picked alternative is the treatment of highest posterior mean, and its
            // interval is the contrast the family reports for it.
            let means = &reading.posterior.mean;
            assert!((1..4).all(|arm| means[chosen] >= means[arm]));
            assert_eq!(f.best_treatment_comparison.arm, chosen);
            f.best_treatment_comparison.lower <= 0.0 && 0.0 <= f.best_treatment_comparison.upper
        };
        let last = readings.last().unwrap();
        picked[last.frequentist.best_treatment] += 1;
        at_stop += u64::from(covers(last));
        every_look += u64::from(readings.iter().all(covers));
    });
    let n = COVERAGE_REPLICATES as f64;
    assert!(at_stop as f64 / n >= COVERAGE_FLOOR, "at the stop: {}", at_stop as f64 / n);
    assert!(every_look as f64 / n >= COVERAGE_FLOOR, "at every look: {}", every_look as f64 / n);
    assert_eq!(picked[0], 0, "the control is never the picked alternative");
    assert!(picked.iter().filter(|&&count| count > 0).count() >= 2, "{picked:?}: picked by data");
}

// -- Replay and separate claims -----------------------------------------------------------------------------

#[test]
fn adaptive_replay_contract_retains_assignment_outcomes_priors_decisions() {
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(NEVER), historical(4, 0.5));
    let run = p.run(&truth(4), Some(CORRELATION), 5_803_458, &ctx()).unwrap().run;
    let history = &run.history;
    let declaration = p.retained_declaration("trial", labels(4), history, &ctx()).unwrap();

    // The contract keeps what was randomized and observed, bit for bit.
    assert_eq!(declaration.assignment_probabilities.as_ref().unwrap(), &history.assignment);
    let availability = declaration.availability.as_ref().unwrap();
    for (participant, success) in history.successes.iter().enumerate() {
        let look = participant / 100;
        match success {
            Some(_) => {
                assert_eq!(availability.available_at_look[participant], Some(look as u32 + 1))
            }
            None => assert_eq!(availability.available_at_look[participant], None),
        }
    }
    assert_eq!(declaration.outcomes.len(), 1_200);
    for (participant, success) in history.successes.iter().enumerate() {
        let ones = declaration.outcomes[participant].iter().filter(|&&y| y == 1.0).count();
        assert_eq!(declaration.outcomes[participant].len(), usize::from(success.is_some()) * 3);
        assert_eq!(ones as u32, success.unwrap_or(0));
    }
    // One policy state per look (the prior update that governs the next batch).
    let epochs = declaration.policy_history.as_ref().unwrap();
    assert_eq!(epochs.len(), 12);
    assert_eq!(
        epochs.iter().map(|e| e.first_unit).collect::<Vec<_>>(),
        (0..12).map(|i| i * 100).collect::<Vec<_>>()
    );
    let distinct: std::collections::BTreeSet<_> =
        epochs.iter().map(|e| format!("{:?}", e.policy_identity)).collect();
    assert_eq!(distinct.len(), 12, "each look updates the policy state");
    // The decisions: the declared rule and that no rule ended the run.
    assert!(!declaration.stopping_rule.as_ref().unwrap().ended_by_rule);

    // Sealed with its procedure, the artifact replays to the same readings, decisions included.
    let session = AdaptiveInferenceSession::prepare(declaration).unwrap();
    let bytes = session
        .export_contract_with_procedure(InferenceParadigm::BayesianPosterior, &p, "replay")
        .unwrap();
    let consumed = consume_adaptive_monitoring_contract_artifact(&bytes, None).unwrap();
    let record = consumed.procedure_record().expect("the procedure section is replayed");
    let report = p.read_history(history, &ctx()).unwrap();
    assert_eq!(record.readings(), &report.readings[..], "replay reproduces the result");
    assert_eq!(record.spec().identity(), p.identity());
    assert_eq!(record.spec().priors(), p.spec().priors());
    for (stored, fresh) in record.readings().iter().zip(&report.readings) {
        assert_eq!(stored.decision, fresh.decision);
        assert_eq!(stored.assignment_probabilities, fresh.assignment_probabilities);
    }
    assert_eq!(
        consumed.contract().declaration().assignment_probabilities.as_ref().unwrap(),
        &history.assignment
    );
}

#[test]
fn adaptive_posterior_and_frequentist_claims_separate() {
    let shape = Shape { participants: 1_200, batch: 100, arms: 4, responses: 3, lag: 1 };
    let p = procedure(&shape, rules(NEVER), historical(4, 0.5));
    let run = p.run(&truth(4), Some(CORRELATION), 3_827_572, &ctx()).unwrap().run;
    let report = p.read_history(&run.history, &ctx()).unwrap();
    for reading in &report.readings {
        // Two claims, two fields, two paradigms.
        assert_eq!(reading.posterior.paradigm, "bayesian_posterior");
        assert_eq!(reading.frequentist.paradigm, "anytime_valid_frequentist");
        assert_eq!(reading.frequentist.calibration, DERIVED_UNMEASURED);
        assert_eq!(reading.posterior.mean.len(), 4);
        assert_eq!(reading.frequentist.arms.len(), 4);
    }

    // Separate assumptions: the posterior rests on the declared priors and likelihood, the
    // anytime-valid claim on the level and tuning alone. A procedure that differs only in the
    // level replays the same history with the same posterior claim and other sequences.
    let mut stricter = declaration(&shape, rules(NEVER), historical(4, 0.5));
    stricter.anytime = Some(AnytimeSettings { alpha: 0.01, tuning_samples: 75.0 });
    let stricter = ParticipantProcedure::declare(stricter).unwrap();
    assert_ne!(stricter.identity(), p.identity());
    let other = stricter.read_history(&run.history, &ctx()).unwrap();
    for (a, b) in report.readings.iter().zip(&other.readings) {
        assert_eq!(a.posterior, b.posterior, "the level is not an assumption of the posterior");
        assert_eq!(a.assignment_probabilities, b.assignment_probabilities);
    }
    let (loose, strict) = (&report.readings[11].frequentist, &other.readings[11].frequentist);
    assert!(
        strict.arms[1].upper - strict.arms[1].lower > loose.arms[1].upper - loose.arms[1].lower
    );

    // Each paradigm is sealed and consumed as its own claim.
    let declaration = p.retained_declaration("trial", labels(4), &run.history, &ctx()).unwrap();
    let session = AdaptiveInferenceSession::prepare(declaration).unwrap();
    let mut sealed = Vec::new();
    for paradigm in [
        InferenceParadigm::BayesianPosterior,
        InferenceParadigm::AnytimeValidFrequentist,
        InferenceParadigm::DecisionTheoreticStopping,
    ] {
        let bytes = session.export_contract_with_procedure(paradigm, &p, "claims").unwrap();
        let consumed = consume_adaptive_monitoring_contract_artifact(&bytes, None).unwrap();
        assert_eq!(consumed.paradigm(), paradigm);
        sealed.push(bytes);
    }
    sealed.sort();
    sealed.dedup();
    assert_eq!(sealed.len(), 3, "the three paradigms are three different claims");
}
