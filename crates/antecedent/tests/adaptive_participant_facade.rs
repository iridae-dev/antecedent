//! The participant-level adaptive family through the facade: declare it, simulate the whole of
//! it under a budget, retain a run, seal the retained contract together with the procedure
//! section, and consume it by replay.
//!
//! The replay is the point: a fresh consumer rebuilds the declared procedure through its own
//! checks, recomputes every batch's allocation from the completed participants and the declared
//! (frozen) priors, compares it with the retained probabilities, replays the decisions, and
//! recomputes every stored reading. Every edit below is *resealed* (the stored identity is
//! recomputed after the edit), so only the replay can refuse it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::adaptive_inference::{
    AdaptiveInferenceRefusal, AdaptiveInferenceSession, InferenceParadigm,
    consume_adaptive_monitoring_contract_artifact,
};
use antecedent::analysis::adaptive_participant_procedure::ParticipantProcedure;
use antecedent_core::ExecutionContext;
use antecedent_io::adaptive_monitoring_contract_artifact::{
    AdaptiveMonitoringContractArtifact, AdaptiveMonitoringContractError as ArtifactError,
};
use antecedent_stats::adaptive_inference::{
    AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan, CorrelationPrior, DecisionRules,
    FutilityRule, HistoricalPrior, LookDecision, ParticipantDeclaration,
    ParticipantSimulationRequest, StopReason,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn borrowing(discount: f64) -> BorrowingPlan {
    BorrowingPlan {
        arms: vec![
            ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![
                    HistoricalPrior {
                        source: "control registry".to_owned(),
                        successes: 40,
                        trials: 100,
                        discount,
                    },
                    HistoricalPrior {
                        source: "earlier pilot".to_owned(),
                        successes: 9,
                        trials: 30,
                        discount: 1.0,
                    },
                ],
            },
            ArmPrior::without_history(BetaPrior::UNIFORM),
            ArmPrior::without_history(BetaPrior::UNIFORM),
        ],
    }
}

fn declaration() -> ParticipantDeclaration {
    ParticipantDeclaration {
        arms: 3,
        responses: Some(6),
        correlation: Some(0.25),
        correlation_prior: None,
        borrowing: Some(borrowing(0.5)),
        enrolment: Some(vec![40; 5]),
        lag_looks: 1,
        min_exploration: Some(0.15),
        decisions: Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: Some(FutilityRule { margin: 0.1, threshold: 0.02 }),
            participant_cap: None,
        }),
        anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 30.0 }),
    }
}

fn procedure() -> ParticipantProcedure {
    ParticipantProcedure::declare(declaration()).expect("a complete declaration")
}

fn labels() -> Vec<String> {
    ["control", "low dose", "high dose"].map(str::to_owned).to_vec()
}

fn refused<T: std::fmt::Debug>(
    outcome: Result<T, AdaptiveInferenceRefusal>,
) -> AdaptiveInferenceRefusal {
    outcome.expect_err("the request is refused")
}

// -- declaration ----------------------------------------------------------------------------------

#[test]
fn declaring_refuses_each_missing_element_with_its_own_detail_and_a_remedy() {
    let mut empty = declaration();
    empty.responses = None;
    empty.correlation = None;
    empty.borrowing = None;
    empty.anytime = None;
    let refusals = ParticipantProcedure::declare(empty).unwrap_err();
    let details: Vec<&str> = refusals.iter().map(|r| r.detail.as_str()).collect();
    assert_eq!(
        details,
        [
            "adaptive_inference.outcome_structure_missing",
            "adaptive_inference.within_unit_correlation_missing",
            "adaptive_inference.historical_borrowing_missing",
            "adaptive_inference.anytime_settings_missing",
        ]
    );
    assert!(
        refusals.iter().all(|r| r.code == "adaptive_contract_incomplete" && !r.message.is_empty())
    );

    let mut invalid = declaration();
    invalid.correlation = Some(1.0);
    let refusals = ParticipantProcedure::declare(invalid).unwrap_err();
    assert_eq!(refusals[0].code, "invalid_argument");
    assert_eq!(refusals[0].detail, "adaptive_inference.correlation_invalid");
}

#[test]
fn the_declaration_identity_includes_borrowing_and_names_the_matrix_cell() {
    let a = procedure();
    let mut other = declaration();
    other.borrowing = Some(borrowing(0.6));
    let b = ParticipantProcedure::declare(other).unwrap();
    assert_ne!(a.identity(), b.identity());
    assert_eq!(a.identity(), procedure().identity());
    assert_eq!(
        a.cell().id(),
        "thompson_beta_bernoulli/repeated_measures/fixed_interim_looks/decision_theoretic"
    );
    // The priors, sources, discounts and the prior effective sample size are recorded.
    let control = &a.spec().priors()[0];
    assert!((control.prior.alpha - (1.0 + 20.0 + 9.0)).abs() < 1e-12);
    assert!((control.prior.beta - (1.0 + 30.0 + 21.0)).abs() < 1e-12);
    assert!((control.effective_sample_size - 82.0).abs() < 1e-12);
    assert!((control.borrowed_sample_size - 80.0).abs() < 1e-12);
    assert_eq!(control.sources.len(), 2);
}

#[test]
fn borrowing_is_frozen_after_the_first_look_with_an_assumption_refusal() {
    let a = procedure();
    let amended = a.amend_borrowing(&borrowing(0.6), 0).expect("open before recruitment");
    assert_ne!(amended.identity(), a.identity());
    let refusal = refused(a.amend_borrowing(&borrowing(0.6), 1));
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.borrowing_frozen");
    assert!(a.amend_borrowing(&borrowing(0.5), 4).is_ok(), "no change is not an amendment");
}

// -- capacity and cancellation through the facade -------------------------------------------------

#[test]
fn capacity_refusals_name_their_class_of_limit() {
    let mut huge = declaration();
    huge.enrolment = Some(vec![40_000, 40_000]);
    let refusals = ParticipantProcedure::declare(huge).unwrap_err();
    assert_eq!(refusals[0].detail, "adaptive_inference.declared_units_exceeded");
    assert_eq!(refusals[0].code, "invalid_argument");
    assert!(refusals[0].message.contains("stages"));

    let p = procedure();
    let truth = [0.4, 0.4, 0.4];
    // 20,000 replications of 64,000 participants with six responses each: 7.7e9 response draws,
    // over the work budget, refused before a replicate runs.
    let mut large = declaration();
    large.enrolment = Some(vec![32_000, 32_000]);
    let large = ParticipantProcedure::declare(large).expect("65,535 units may be declared");
    let work = refused(large.simulate(
        &truth,
        &ParticipantSimulationRequest { replications: 20_000, seed: 1, true_correlation: None },
        &ctx(),
    ));
    assert_eq!(work.detail, "adaptive_inference.simulation_work_exceeded");
    assert!(work.message.contains("remedy"));
    // A memory budget on the execution context is honoured.
    let mut small = ctx();
    small.memory.hard_limit_bytes = Some(1_000);
    let memory = refused(p.simulate(
        &truth,
        &ParticipantSimulationRequest { replications: 10, seed: 1, true_correlation: None },
        &small,
    ));
    assert_eq!(memory.detail, "adaptive_inference.simulation_memory_exceeded");
    // Cancellation is a refusal that is not a verdict on the data.
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let stopped = refused(p.simulate(
        &truth,
        &ParticipantSimulationRequest { replications: 10, seed: 1, true_correlation: None },
        &cancelled,
    ));
    assert_eq!(stopped.code, "cancelled_no_claim");
}

#[test]
fn a_simulation_returns_a_receipt_and_labelled_measured_properties() {
    let report = procedure()
        .simulate(
            &[0.4, 0.4, 0.4],
            &ParticipantSimulationRequest { replications: 30, seed: 8, true_correlation: None },
            &ctx(),
        )
        .unwrap();
    assert_eq!(report.calibration, "derived_unmeasured");
    let receipt = report.simulation.receipt;
    assert_eq!(receipt.replications, 30);
    assert_eq!(receipt.work_planned, 30 * 200 * 6);
    assert!(receipt.work_done <= receipt.work_planned);
    assert_eq!(report.summary.replications, 30);
    // The simultaneous family covers the truth at every look, whatever stopped the run.
    assert!(report.summary.covered_every_look >= 0.9);
}

// -- retention, sealing and replay ----------------------------------------------------------------

struct Run {
    procedure: ParticipantProcedure,
    session: AdaptiveInferenceSession,
    bytes: Vec<u8>,
}

fn sealed_run() -> Run {
    let procedure = procedure();
    let history =
        procedure.simulate_history(&[0.35, 0.45, 0.4], None, 12, &ctx()).expect("a history");
    assert_eq!(history.batches.len(), 5, "no decision ends this run early");
    let declaration =
        procedure.retained_declaration("dose finding", labels(), &history, &ctx()).unwrap();
    let session = AdaptiveInferenceSession::prepare(declaration).expect("a complete contract");
    let bytes = session
        .export_contract_with_procedure(InferenceParadigm::BayesianPosterior, &procedure, "run-1")
        .expect("seals");
    Run { procedure, session, bytes }
}

#[test]
fn a_retained_run_seals_with_its_procedure_and_a_fresh_consumer_replays_it() {
    let run = sealed_run();
    let history = run.procedure.simulate_history(&[0.35, 0.45, 0.4], None, 12, &ctx()).unwrap();
    let report = run.procedure.read_history(&history, &ctx()).unwrap();
    assert_eq!(report.calibration, "derived_unmeasured");
    assert!(matches!(report.decision, LookDecision::Stop { .. }));

    let consumed = consume_adaptive_monitoring_contract_artifact(&run.bytes, None).unwrap();
    let record = consumed.procedure_record().expect("the procedure section is replayed");
    // The consumer's recomputed readings are the producer's, and each look carries the posterior
    // and the frequentist claim in separate, labelled fields.
    assert_eq!(record.readings(), &report.readings[..]);
    assert_eq!(record.spec().identity(), run.procedure.identity());
    for reading in record.readings() {
        assert_eq!(reading.posterior.paradigm, "bayesian_posterior");
        assert_eq!(reading.frequentist.paradigm, "anytime_valid_frequentist");
        assert_eq!(reading.frequentist.calibration, "derived_unmeasured");
        assert_eq!(reading.frequentist.comparisons.len(), 2);
    }
    // The retained identity is accepted; the artifact is deterministic.
    assert!(
        consume_adaptive_monitoring_contract_artifact(&run.bytes, Some(consumed.identity()))
            .is_ok()
    );
    let again = run
        .session
        .export_contract_with_procedure(
            InferenceParadigm::BayesianPosterior,
            &run.procedure,
            "run-1",
        )
        .unwrap();
    assert_eq!(again, run.bytes);
    // Every paradigm the cell is licensed for may carry the section; one it is not licensed for may not.
    for paradigm in
        [InferenceParadigm::DecisionTheoreticStopping, InferenceParadigm::AnytimeValidFrequentist]
    {
        run.session.export_contract_with_procedure(paradigm, &run.procedure, "run-1").unwrap();
    }
    let fixed = refused(run.session.export_contract_with_procedure(
        InferenceParadigm::FrequentistFixedSample,
        &run.procedure,
        "run-1",
    ));
    assert_eq!(fixed.code, "adaptive_design_ignored");
    // A contract sealed without the section replays without one.
    let plain = run.session.export_contract(InferenceParadigm::BayesianPosterior, "run-1").unwrap();
    assert_ne!(plain, run.bytes);
    let plain = consume_adaptive_monitoring_contract_artifact(&plain, None).unwrap();
    assert!(plain.procedure_record().is_none());
}

#[test]
fn a_procedure_that_estimates_the_correlation_seals_and_a_fresh_consumer_replays_it() {
    let mut estimated = declaration();
    estimated.correlation = None;
    estimated.correlation_prior = Some(CorrelationPrior { alpha: 2.0, beta: 6.0 });
    estimated.enrolment = Some(vec![40; 4]);
    estimated.decisions =
        Some(DecisionRules { loss_threshold: 1e-12, futility: None, participant_cap: None });
    let procedure = ParticipantProcedure::declare(estimated).expect("a complete declaration");
    assert_ne!(procedure.identity(), self::procedure().identity());
    // A procedure that estimates the correlation needs the true one to simulate.
    refused(procedure.simulate_history(&[0.35, 0.45, 0.4], None, 12, &ctx()));
    let history =
        procedure.simulate_history(&[0.35, 0.45, 0.4], Some(0.4), 12, &ctx()).expect("a history");
    assert_eq!(history.batches.len(), 4, "no decision ends this run early");
    let report = procedure.read_history(&history, &ctx()).unwrap();
    let last = &report.readings.last().unwrap().posterior;
    assert!(last.correlation_sd > 0.0 && last.correlation_mean > 0.0);
    let declaration =
        procedure.retained_declaration("dose finding", labels(), &history, &ctx()).unwrap();
    let session = AdaptiveInferenceSession::prepare(declaration).expect("a complete contract");
    let bytes = session
        .export_contract_with_procedure(InferenceParadigm::BayesianPosterior, &procedure, "run-2")
        .expect("seals");
    let consumed = consume_adaptive_monitoring_contract_artifact(&bytes, None).unwrap();
    let record = consumed.procedure_record().expect("the procedure section is replayed");
    assert_eq!(record.readings(), &report.readings[..]);
    assert_eq!(record.spec().correlation(), None);
    assert_eq!(record.spec().correlation_prior(), Some(CorrelationPrior { alpha: 2.0, beta: 6.0 }));
}

type Artifact = AdaptiveMonitoringContractArtifact;

fn resealed(
    run: &Run,
    edit: impl FnOnce(&mut antecedent_io::adaptive_monitoring_contract_artifact::MonitoringContractMeta),
) -> Result<Artifact, ArtifactError> {
    let mut meta = Artifact::decode_unchecked(&run.bytes).expect("decodes");
    edit(&mut meta);
    Artifact::reseal_identity(&mut meta);
    Artifact::from_meta(&meta, None)
}

fn replay_detail(error: &ArtifactError) -> (&'static str, String) {
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    (code, detail)
}

#[test]
fn a_resealed_edit_of_the_procedure_section_does_not_replay() {
    let run = sealed_run();
    assert!(resealed(&run, |_| {}).is_ok(), "an unedited reseal is accepted");

    // A stored reading that was changed.
    let error = resealed(&run, |meta| {
        let reading = &mut meta.participant_procedure.as_mut().unwrap().looks[1];
        reading.numbers_bits[0] = (f64::from_bits(reading.numbers_bits[0]) + 0.01).to_bits();
    })
    .unwrap_err();
    assert_eq!(error, ArtifactError::ReplayMismatch("procedure readings"));
    assert_eq!(replay_detail(&error).1, "adaptive_inference.report_replay_mismatch");

    // A stored decision that was changed.
    let error = resealed(&run, |meta| {
        meta.participant_procedure.as_mut().unwrap().looks[0].decision = "futility".to_owned();
    })
    .unwrap_err();
    assert_eq!(error, ArtifactError::ReplayMismatch("procedure readings"));

    // A borrowed discount that was changed: the declaration no longer has the stored identity.
    let error = resealed(&run, |meta| {
        meta.participant_procedure.as_mut().unwrap().arms[0].historical[0].discount_bits =
            1.0_f64.to_bits();
    })
    .unwrap_err();
    assert_eq!(error, ArtifactError::ReplayMismatch("procedure identity"));

    // A prior effective sample size that was changed.
    let error = resealed(&run, |meta| {
        meta.participant_procedure.as_mut().unwrap().arms[0].effective_sample_size_bits =
            500.0_f64.to_bits();
    })
    .unwrap_err();
    assert_eq!(error, ArtifactError::ReplayMismatch("resolved priors"));

    // More readings than a record holds.
    let error = resealed(&run, |meta| {
        let section = meta.participant_procedure.as_mut().unwrap();
        let first = section.looks[0].clone();
        section.looks = vec![first; 65];
    })
    .unwrap_err();
    assert_eq!(replay_detail(&error).1, "adaptive_inference.retained_record_exceeded");

    // An element that was removed: the declared procedure is rebuilt through its own checks.
    let error = resealed(&run, |meta| {
        meta.participant_procedure.as_mut().unwrap().enrolment = vec![200];
    })
    .unwrap_err();
    assert_eq!(replay_detail(&error).0, "invalid_argument");
    assert_eq!(replay_detail(&error).1, "adaptive_inference.looks_not_ordered");

    // A procedure section on a contract of another cell.
    let other_cell = resealed(&run, |meta| {
        meta.declaration.stopping = "efficacy".to_owned();
    });
    assert!(other_cell.is_err());
}

#[test]
fn a_retained_probability_that_the_declared_priors_do_not_produce_is_refused() {
    let run = sealed_run();
    // The producer's allocation probabilities are retained in the contract. Another allocation
    // for the first batch, sealed consistently as a contract of its own and carrying the original
    // procedure section, is not what the declared procedure computes.
    let original = Artifact::decode_unchecked(&run.bytes).expect("decodes");
    let mut wire = original.declaration.clone();
    wire.assignment_probability_bits.as_mut().unwrap()[0] =
        vec![0.5_f64.to_bits(), 0.3_f64.to_bits(), 0.2_f64.to_bits()];
    let declaration =
        antecedent_io::adaptive_monitoring_contract_artifact::declaration_from_wire(&wire).unwrap();
    let contract = antecedent_contract::adaptive_design::AdaptiveDesignContract::new(declaration)
        .expect("a consistent contract");
    let (_, mut meta) = Artifact::seal(&contract, InferenceParadigm::BayesianPosterior).unwrap();
    meta.participant_procedure = original.participant_procedure.clone();
    Artifact::reseal_identity(&mut meta);
    let error = Artifact::from_meta(&meta, None).unwrap_err();
    let (code, detail) = replay_detail(&error);
    assert_eq!(code, "adaptive_assumption_violated");
    assert_eq!(detail, "adaptive_inference.assignment_not_replayed");
}

#[test]
fn sealing_under_other_borrowing_is_refused_because_the_allocation_does_not_replay() {
    let run = sealed_run();
    // The retained allocation was computed under a discount of 0.5; a procedure declared with
    // another discount (amendable only before the first look) cannot be sealed with this history.
    let other = run.procedure.amend_borrowing(&borrowing(1.0), 0).unwrap();
    let refusal = refused(run.session.export_contract_with_procedure(
        InferenceParadigm::BayesianPosterior,
        &other,
        "run-1",
    ));
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.assignment_not_replayed");
    // And once the procedure has made a look, the borrowing cannot be changed at all.
    let frozen = refused(run.procedure.amend_borrowing(&borrowing(1.0), 5));
    assert_eq!(frozen.detail, "adaptive_inference.borrowing_frozen");
}

#[test]
fn a_run_that_stopped_early_replays_and_a_later_look_than_the_stop_is_refused() {
    // Equal arms, no delay and a recruitment cap of 80: the cap ends the run at its second look.
    let mut capped = declaration();
    capped.lag_looks = 0;
    capped.decisions =
        Some(DecisionRules { loss_threshold: 1e-12, futility: None, participant_cap: Some(80) });
    let capped = ParticipantProcedure::declare(capped).unwrap();
    let truth = [0.4, 0.4, 0.4];
    let history = capped.simulate_history(&truth, None, 5, &ctx()).unwrap();
    assert_eq!(history.batches, vec![40, 40]);
    let report = capped.read_history(&history, &ctx()).unwrap();
    assert!(matches!(
        report.decision,
        LookDecision::Stop { reason: StopReason::RecruitmentCap, .. }
    ));
    // The retained contract records that a decision, not the schedule, ended the trial, and it
    // seals and replays.
    let declaration = capped.retained_declaration("capped", labels(), &history, &ctx()).unwrap();
    assert!(declaration.stopping_rule.unwrap().ended_by_rule);
    let session = AdaptiveInferenceSession::prepare(declaration).unwrap();
    let bytes = session
        .export_contract_with_procedure(InferenceParadigm::BayesianPosterior, &capped, "capped")
        .unwrap();
    let consumed = consume_adaptive_monitoring_contract_artifact(&bytes, None).unwrap();
    assert_eq!(consumed.procedure_record().unwrap().readings(), &report.readings[..]);

    // The same schedule without the cap runs to its horizon; read under the cap, that history
    // went on past the look at which the declared decisions stop the procedure.
    let mut uncapped = declaration_without_cap();
    uncapped.lag_looks = 0;
    let uncapped = ParticipantProcedure::declare(uncapped).unwrap();
    let longer = uncapped.simulate_history(&truth, None, 5, &ctx()).unwrap();
    assert_eq!(longer.batches.len(), 5);
    let refusal = refused(capped.read_history(&longer, &ctx()));
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.stopping_not_replayed");
}

fn declaration_without_cap() -> ParticipantDeclaration {
    let mut d = declaration();
    d.decisions =
        Some(DecisionRules { loss_threshold: 1e-12, futility: None, participant_cap: None });
    d
}
