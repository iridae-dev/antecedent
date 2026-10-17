//! Capacity of the adaptive-procedure routes: each class of limit is its own refusal.
//!
//! A declaration may enrol far more than the 400 participants these routes once accepted;
//! what bounds a request is the cost of deriving the procedure's uncertainty, and that
//! cost has a different bound on each route. The exact law is bounded by the distinct states and
//! the unit expansions of its dynamic program, so a large procedure is refused there with the
//! exact-law details; the whole-procedure simulation that large procedures use instead is bounded
//! by replications times units times measurements and by an estimated peak memory, both checked
//! before any replicate runs, and returns a receipt of the work it did. Planned procedures of 600,
//! 1,200 and 2,400 participants run on every simulation route, single outcome and repeated
//! measures, and cancellation stops each.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CancellationToken;
use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, AnalysisEstimator, AnytimeSettings, AssignmentRule, BetaPrior,
    BorrowingPlan, CapacityError, ContractGap, DecisionRules, Estimand, ExposureModel,
    FinalAnalysis, FinalSelection, MAX_PARTICIPANT_LOOKS, MAX_PROCEDURE_UNITS, MAX_UNIT_BATCHES,
    OutcomeKernel, OutcomeModelPrior, OutcomeObservation, OutcomeStructure, ParticipantDeclaration,
    ParticipantSimulationRequest, ParticipantSpec, ProcedureDeclaration, ProcedureSpec,
    RepeatedMeasuresDeclaration, RepeatedMeasuresRequest, RepeatedMeasuresSpec, ResourceBudget,
    SimulationRequest, StoppingRule, UnitAssignment, UnitSummary, WeightScheme,
    WithinUnitCorrelation, enumerate_law, simulate_participant_history,
    simulate_participant_procedure, simulate_participant_procedure_within, simulate_procedure,
    simulate_procedure_within, simulate_repeated_measures, simulate_repeated_measures_within,
};

fn never() -> bool {
    false
}

const PLANNED: [u32; 3] = [600, 1_200, 2_400];

// -- single-outcome procedures --------------------------------------------------------------------

/// Three arms, Thompson allocation with a 10 percent floor, `looks` equal batches.
fn thompson(units: u32, looks: u32) -> ProcedureSpec {
    ProcedureSpec::new(ProcedureDeclaration {
        arms: 3,
        context_probabilities: None,
        assignment: Some(AssignmentRule::ThompsonBetaFloored {
            prior: BetaPrior::UNIFORM,
            min_probability: 0.1,
        }),
        enrolment: Some(vec![units / looks; looks as usize]),
        observation: Some(OutcomeObservation::COMPLETE),
        exposure: Some(ExposureModel::none(3)),
        stopping: Some(StoppingRule::NoStopping),
        selection: Some(FinalSelection::NoSelection),
    })
    .expect("a valid procedure")
}

fn weighting(replications: u32) -> SimulationRequest {
    SimulationRequest {
        replications,
        seed: 17,
        estimand: Estimand::Contrast { plus: 1, minus: 0 },
        scheme: WeightScheme::Uniform,
        outcome_prior: OutcomeModelPrior { mean: 0.5, strength: 2.0 },
        probability_floor: 0.05,
    }
}

#[test]
fn planned_single_outcome_procedures_run_with_a_receipt() {
    for units in PLANNED {
        let spec = thompson(units, 12);
        let law = simulate_procedure(&spec, &[0.3, 0.45, 0.6], &weighting(3), &never)
            .unwrap_or_else(|e| panic!("{units} participants: {e}"));
        assert_eq!(law.replicates.len(), 3);
        for replicate in &law.replicates {
            assert_eq!(replicate.enrolled, units);
            assert_eq!(replicate.analysed_units, units as usize);
            assert!(replicate.estimate.is_some());
        }
        let receipt = law.receipt;
        assert_eq!(receipt.replications, 3);
        assert_eq!(receipt.work_planned, 3 * u64::from(units));
        assert_eq!(receipt.work_done, 3 * u64::from(units));
        assert_eq!(receipt.peak_units, units);
        assert!(receipt.estimated_peak_bytes > 0);
    }
}

// -- repeated measures ----------------------------------------------------------------------------

fn repeated(units: u32, batches: u32, measurements: u32) -> RepeatedMeasuresSpec {
    RepeatedMeasuresSpec::new(RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(UnitAssignment::EpsilonGreedy {
            epsilon: 0.4,
            prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 },
        }),
        enrolment: Some(vec![units / batches; batches as usize]),
        lag_looks: 1,
        structure: Some(OutcomeStructure::RepeatedMeasures {
            measurements,
            variance: 1.0,
            correlation: Some(WithinUnitCorrelation::CompoundSymmetric { rho: 0.3 }),
        }),
        analysis: Some(FinalAnalysis {
            estimator: AnalysisEstimator::AdaptivelyWeighted {
                summary: UnitSummary::Mean,
                scheme: WeightScheme::Uniform,
                prior: OutcomeModelPrior { mean: 0.0, strength: 1.0 },
                probability_floor: 0.05,
            },
            plus: 1,
            minus: 0,
        }),
        stopping: Some(StoppingRule::NoStopping),
    })
    .expect("a valid procedure")
}

#[test]
fn planned_repeated_measures_procedures_run_with_a_receipt() {
    for units in PLANNED {
        let spec = repeated(units, 12, 3);
        let request = RepeatedMeasuresRequest { replications: 3, seed: 5 };
        let law = simulate_repeated_measures(&spec, &[0.0, 0.3], &request, &never)
            .unwrap_or_else(|e| panic!("{units} participants: {e}"));
        assert_eq!(law.replicates.len(), 3);
        assert!(law.replicates.iter().all(|r| r.enrolled == units && r.estimate.is_some()));
        // One batch of follow-up has not completed when the analysis is made.
        assert!(law.replicates.iter().all(|r| r.analysed_units == (units / 12 * 11) as usize));
        let receipt = law.receipt;
        assert_eq!(receipt.replications, 3);
        assert_eq!(receipt.work_planned, 3 * u64::from(units) * 3);
        assert_eq!(receipt.work_done, receipt.work_planned);
        assert_eq!(receipt.peak_units, units);
    }
}

#[test]
fn a_fully_sequential_repeated_measures_design_past_400_batches_is_declared() {
    // One participant per batch, 2,400 batches, past the 400 batches these routes once accepted.
    let spec = repeated(2_400, 2_400, 3);
    assert_eq!(spec.enrolment().len(), 2_400);
    const { assert!(MAX_UNIT_BATCHES >= 2_400) };
}

// -- the participant-level family -----------------------------------------------------------------

fn participant(units: u32, looks: u32) -> ParticipantDeclaration {
    ParticipantDeclaration {
        arms: 4,
        responses: Some(6),
        correlation: Some(0.2),
        correlation_prior: None,
        borrowing: Some(BorrowingPlan::uniform(4, BetaPrior::UNIFORM)),
        enrolment: Some(vec![units / looks; looks as usize]),
        lag_looks: 1,
        min_exploration: Some(0.1),
        // Four arms with equal means keep a posterior expected loss of about a hundredth, so this
        // threshold never ends a run.
        decisions: Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: None,
            participant_cap: None,
        }),
        anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 50.0 }),
    }
}

const EQUAL_ARMS: [f64; 4] = [0.4; 4];

fn participant_spec(units: u32, looks: u32) -> ParticipantSpec {
    ParticipantSpec::new(participant(units, looks)).expect("a valid procedure")
}

#[test]
fn planned_participant_procedures_run_with_a_receipt() {
    for units in PLANNED {
        let spec = participant_spec(units, 12);
        let request =
            ParticipantSimulationRequest { replications: 2, seed: 9, true_correlation: None };
        let simulation = simulate_participant_procedure(&spec, &EQUAL_ARMS, &request, &never)
            .unwrap_or_else(|e| panic!("{units} participants: {e}"));
        assert_eq!(simulation.replicates.len(), 2);
        assert!(simulation.replicates.iter().all(|r| r.enrolled == units));
        let receipt = simulation.receipt;
        assert_eq!(receipt.replications, 2);
        assert_eq!(receipt.work_planned, 2 * u64::from(units) * 6);
        assert_eq!(receipt.work_done, receipt.work_planned);
        assert_eq!(receipt.peak_units, units);
    }
}

#[test]
fn a_run_ended_early_reports_less_work_than_it_was_admitted_for() {
    let mut declaration = participant(1_200, 12);
    declaration.lag_looks = 0;
    declaration.decisions =
        Some(DecisionRules { loss_threshold: 0.01, futility: None, participant_cap: None });
    let spec = ParticipantSpec::new(declaration).expect("valid");
    let request = ParticipantSimulationRequest { replications: 4, seed: 3, true_correlation: None };
    let simulation =
        simulate_participant_procedure(&spec, &[0.1, 0.9, 0.1, 0.1], &request, &never).unwrap();
    let receipt = simulation.receipt;
    assert_eq!(receipt.work_planned, 4 * 1_200 * 6);
    assert!(receipt.work_done < receipt.work_planned, "a decisive truth stops early");
    assert!(simulation.replicates.iter().all(|r| r.reason.ended_by_rule()));
}

// -- declaration capacity -------------------------------------------------------------------------

#[test]
fn declaration_capacity_is_separate_from_ordering() {
    // Participant procedure.
    let units_over = |declaration: ParticipantDeclaration| {
        ParticipantSpec::new(declaration).expect_err("refused").gaps
    };
    let mut too_many_units = participant(2_400, 12);
    too_many_units.enrolment = Some(vec![MAX_PROCEDURE_UNITS / 2 + 1, MAX_PROCEDURE_UNITS / 2 + 1]);
    assert_eq!(
        units_over(too_many_units),
        vec![ContractGap::Capacity(CapacityError::DeclaredUnits {
            limit: u64::from(MAX_PROCEDURE_UNITS),
            requested: u64::from(MAX_PROCEDURE_UNITS) + 1,
        })]
    );
    let mut too_many_looks = participant(2_400, 12);
    too_many_looks.enrolment = Some(vec![1; MAX_PARTICIPANT_LOOKS + 1]);
    assert_eq!(
        units_over(too_many_looks),
        vec![ContractGap::Capacity(CapacityError::DeclaredLooks {
            limit: MAX_PARTICIPANT_LOOKS as u64,
            requested: MAX_PARTICIPANT_LOOKS as u64 + 1,
        })]
    );
    let mut unordered = participant(2_400, 12);
    unordered.enrolment = Some(vec![5, 0, 5]);
    assert_eq!(
        units_over(unordered),
        vec![ContractGap::Invalid("adaptive_inference.looks_not_ordered")]
    );

    // Repeated measures: zero-unit follow-up batches past the batch bound.
    let declaration = |schedule: Vec<u32>| RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(UnitAssignment::Fixed { probabilities: vec![0.5, 0.5] }),
        enrolment: Some(schedule),
        lag_looks: 0,
        structure: Some(OutcomeStructure::RepeatedMeasures {
            measurements: 3,
            variance: 1.0,
            correlation: Some(WithinUnitCorrelation::CompoundSymmetric { rho: 0.3 }),
        }),
        analysis: Some(FinalAnalysis {
            estimator: AnalysisEstimator::ClusterRobust { summary: UnitSummary::Mean },
            plus: 1,
            minus: 0,
        }),
        stopping: Some(StoppingRule::NoStopping),
    };
    let mut long = vec![0; MAX_UNIT_BATCHES + 1];
    long[0] = 1;
    assert_eq!(
        RepeatedMeasuresSpec::new(declaration(long)).expect_err("refused").gaps,
        vec![ContractGap::Capacity(CapacityError::DeclaredLooks {
            limit: MAX_UNIT_BATCHES as u64,
            requested: MAX_UNIT_BATCHES as u64 + 1,
        })]
    );
    assert_eq!(
        RepeatedMeasuresSpec::new(declaration(vec![MAX_PROCEDURE_UNITS + 1])).expect_err("r").gaps,
        vec![ContractGap::Capacity(CapacityError::DeclaredUnits {
            limit: u64::from(MAX_PROCEDURE_UNITS),
            requested: u64::from(MAX_PROCEDURE_UNITS) + 1,
        })]
    );
    assert!(RepeatedMeasuresSpec::new(declaration(vec![2_400])).is_ok());
}

#[test]
fn each_capacity_class_has_its_own_detail_and_remedy() {
    let classes = [
        CapacityError::DeclaredUnits { limit: 1, requested: 2 },
        CapacityError::DeclaredLooks { limit: 1, requested: 2 },
        CapacityError::LawStates { limit: 1, requested: 2 },
        CapacityError::LawWork { limit: 1, requested: 2 },
        CapacityError::SimulationReplications { limit: 1, requested: 2 },
        CapacityError::SimulationWork { limit: 1, requested: 2 },
        CapacityError::SimulationMemory { limit: 1, requested: 2 },
        CapacityError::RetainedRecord { limit: 1, requested: 2 },
    ];
    let mut details: Vec<&str> = classes.iter().map(|c| c.detail()).collect();
    details.sort_unstable();
    details.dedup();
    assert_eq!(details.len(), classes.len(), "no two classes share a detail");
    let mut remedies: Vec<&str> = classes.iter().map(|c| c.remedy()).collect();
    remedies.sort_unstable();
    remedies.dedup();
    assert_eq!(remedies.len(), classes.len(), "no two classes share a remedy");
    for class in classes {
        assert!(class.detail().starts_with("adaptive_inference."));
        assert_ne!(class.detail(), "adaptive_inference.looks_not_ordered");
        assert_eq!((class.limit(), class.requested()), (1, 2));
        assert!(class.to_string().contains(&class.class().replace('_', " ")));
    }
}

// -- exact-law bounds -----------------------------------------------------------------------------

fn fixed_law_spec(arms: usize, contexts: usize, units: u32) -> ProcedureSpec {
    ProcedureSpec::new(ProcedureDeclaration {
        arms,
        context_probabilities: (contexts > 1).then(|| vec![1.0 / contexts as f64; contexts]),
        assignment: Some(AssignmentRule::Fixed { probabilities: vec![1.0 / arms as f64; arms] }),
        enrolment: Some(vec![units]),
        observation: Some(OutcomeObservation::COMPLETE),
        exposure: Some(ExposureModel::none(arms)),
        stopping: Some(StoppingRule::NoStopping),
        selection: Some(FinalSelection::NoSelection),
    })
    .expect("a valid procedure")
}

fn law_error(spec: &ProcedureSpec) -> CapacityError {
    let cells = spec.cells();
    let kernel = OutcomeKernel::Fixed { success_probabilities: vec![0.5; cells] };
    match enumerate_law(spec, &kernel, &never) {
        Err(Error::Capacity(capacity)) => capacity,
        other => panic!("expected a capacity refusal, got {other:?}"),
    }
}

#[test]
fn the_exact_law_refuses_too_many_live_states_with_its_own_detail() {
    // Eight cells and one batch of thirty: the compositions of the batch alone pass a million.
    let refused = law_error(&fixed_law_spec(4, 2, 30));
    assert!(matches!(refused, CapacityError::LawStates { limit: 1_000_000, .. }), "{refused:?}");
    assert_eq!(refused.detail(), "adaptive_inference.exact_law_states_exceeded");
    assert!(refused.remedy().contains("simulate"));
}

#[test]
#[ignore = "enumerates to the exact-law work bound, about ten seconds"]
fn the_exact_law_refuses_too_much_work_with_its_own_detail() {
    // 2,400 participants in one batch: far inside the declaration capacity, far outside what the
    // dynamic program enumerates. The simulation route runs the same procedure.
    let spec = fixed_law_spec(2, 1, 2_400);
    let refused = law_error(&spec);
    assert!(
        matches!(
            refused,
            CapacityError::LawWork { limit: 60_000_000, .. } | CapacityError::LawStates { .. }
        ),
        "{refused:?}"
    );
    assert!(refused.detail().starts_with("adaptive_inference.exact_law_"));
    let simulated = simulate_procedure(&spec, &[0.4, 0.6], &weighting(2), &never);
    assert!(simulated.is_ok(), "the simulation route runs what the enumeration cannot");
}

// -- simulation budgets ---------------------------------------------------------------------------

fn work_budget(max_work: u64) -> ResourceBudget {
    ResourceBudget { max_work, max_memory_bytes: u64::MAX }
}

fn memory_budget(max_memory_bytes: u64) -> ResourceBudget {
    ResourceBudget { max_work: u64::MAX, max_memory_bytes }
}

#[test]
fn a_work_budget_refusal_names_the_work_class_before_any_replicate_runs() {
    // replications x units x measurements, per route.
    let single = thompson(600, 12);
    let refused = simulate_procedure_within(
        &single,
        &[0.3, 0.45, 0.6],
        &weighting(10),
        &work_budget(1_000),
        &never,
    )
    .unwrap_err();
    assert_eq!(
        refused,
        Error::Capacity(CapacityError::SimulationWork { limit: 1_000, requested: 6_000 })
    );

    let repeated_spec = repeated(600, 12, 3);
    let refused = simulate_repeated_measures_within(
        &repeated_spec,
        &[0.0, 0.3],
        &RepeatedMeasuresRequest { replications: 10, seed: 1 },
        &work_budget(1_000),
        &never,
    )
    .unwrap_err();
    assert_eq!(
        refused,
        Error::Capacity(CapacityError::SimulationWork { limit: 1_000, requested: 18_000 })
    );

    let family = participant_spec(600, 12);
    let refused = simulate_participant_procedure_within(
        &family,
        &EQUAL_ARMS,
        &ParticipantSimulationRequest { replications: 10, seed: 1, true_correlation: None },
        &work_budget(1_000),
        &never,
    )
    .unwrap_err();
    assert_eq!(
        refused,
        Error::Capacity(CapacityError::SimulationWork { limit: 1_000, requested: 36_000 })
    );
    let Error::Capacity(capacity) = refused else { unreachable!() };
    assert_eq!(capacity.detail(), "adaptive_inference.simulation_work_exceeded");
}

#[test]
fn the_default_work_budget_refuses_a_request_beyond_it_without_running_it() {
    // 500,000 replications of 2,400 participants with six responses each: 7.2e9 response draws.
    let family = participant_spec(2_400, 12);
    let refused = simulate_participant_procedure(
        &family,
        &EQUAL_ARMS,
        &ParticipantSimulationRequest { replications: 500_000, seed: 1, true_correlation: None },
        &never,
    )
    .unwrap_err();
    assert_eq!(
        refused,
        Error::Capacity(CapacityError::SimulationWork {
            limit: 4_000_000_000,
            requested: 7_200_000_000,
        })
    );
}

#[test]
fn replications_past_the_bound_are_refused_by_every_simulation_route() {
    let expected = Error::Capacity(CapacityError::SimulationReplications {
        limit: 1_000_000,
        requested: 1_000_001,
    });
    assert_eq!(
        simulate_procedure(&thompson(600, 12), &[0.3, 0.45, 0.6], &weighting(1_000_001), &never)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        simulate_repeated_measures(
            &repeated(600, 12, 3),
            &[0.0, 0.3],
            &RepeatedMeasuresRequest { replications: 1_000_001, seed: 1 },
            &never,
        )
        .unwrap_err(),
        expected
    );
    assert_eq!(
        simulate_participant_procedure(
            &participant_spec(600, 12),
            &EQUAL_ARMS,
            &ParticipantSimulationRequest {
                replications: 1_000_001,
                seed: 1,
                true_correlation: None
            },
            &never,
        )
        .unwrap_err(),
        expected
    );
    // Zero replications are an invalid request, not a capacity.
    assert!(matches!(
        simulate_participant_procedure(
            &participant_spec(600, 12),
            &EQUAL_ARMS,
            &ParticipantSimulationRequest { replications: 0, seed: 1, true_correlation: None },
            &never,
        ),
        Err(Error::InvalidParameter(_))
    ));
}

#[test]
fn a_memory_budget_refusal_names_the_memory_class() {
    let single = thompson(600, 12);
    let refused = simulate_procedure_within(
        &single,
        &[0.3, 0.45, 0.6],
        &weighting(10),
        &memory_budget(1_000),
        &never,
    )
    .unwrap_err();
    assert!(
        matches!(refused, Error::Capacity(CapacityError::SimulationMemory { limit: 1_000, .. })),
        "{refused:?}"
    );
    let family = participant_spec(600, 12);
    let refused = simulate_participant_procedure_within(
        &family,
        &EQUAL_ARMS,
        &ParticipantSimulationRequest { replications: 10, seed: 1, true_correlation: None },
        &memory_budget(1_000),
        &never,
    )
    .unwrap_err();
    let Error::Capacity(capacity) = refused else { panic!("{refused:?}") };
    assert_eq!(capacity.detail(), "adaptive_inference.simulation_memory_exceeded");
    assert!(capacity.requested() > 1_000);
}

// -- cancellation ---------------------------------------------------------------------------------

#[test]
fn cancellation_stops_every_large_simulation_route() {
    let single = thompson(2_400, 12);
    let token = CancellationToken::cancel_after_checks(4);
    let cancelled = || token.is_cancelled();
    assert_eq!(
        simulate_procedure(&single, &[0.3, 0.45, 0.6], &weighting(400), &cancelled).unwrap_err(),
        Error::Cancelled
    );

    let repeated_spec = repeated(2_400, 12, 3);
    let token = CancellationToken::cancel_after_checks(4);
    let cancelled = || token.is_cancelled();
    assert_eq!(
        simulate_repeated_measures(
            &repeated_spec,
            &[0.0, 0.3],
            &RepeatedMeasuresRequest { replications: 400, seed: 1 },
            &cancelled,
        )
        .unwrap_err(),
        Error::Cancelled
    );

    let family = participant_spec(2_400, 12);
    let token = CancellationToken::cancel_after_checks(4);
    let cancelled = || token.is_cancelled();
    assert_eq!(
        simulate_participant_procedure(
            &family,
            &EQUAL_ARMS,
            &ParticipantSimulationRequest { replications: 400, seed: 1, true_correlation: None },
            &cancelled,
        )
        .unwrap_err(),
        Error::Cancelled
    );

    // The exact law checks at every look and every thousand unit expansions.
    let law = fixed_law_spec(2, 1, 40);
    let kernel = OutcomeKernel::Fixed { success_probabilities: vec![0.4, 0.6] };
    assert_eq!(enumerate_law(&law, &kernel, &|| true).unwrap_err(), Error::Cancelled);
}

// -- acceptance cases of the capacity classes ---------------------------------------------------------

#[test]
fn adaptive_repeated_measures_batch_ceiling_removed() {
    // 2,400 participants in batches of 200 with six measurements each: a batch the repeated-measures
    // routes once refused.
    let spec = repeated(2_400, 12, 6);
    assert_eq!(spec.enrolment().len(), 12);
    assert!(spec.enrolment().iter().all(|&batch| batch == 200));
    let request = RepeatedMeasuresRequest { replications: 3, seed: 5_641_339 };
    let law = simulate_repeated_measures(&spec, &[0.0, 0.3], &request, &never)
        .expect("a batch of 200 participants x 6 measurements is accepted");
    assert_eq!(law.replicates.len(), 3);
    assert!(law.replicates.iter().all(|r| r.enrolled == 2_400 && r.estimate.is_some()));
    let receipt = law.receipt;
    assert_eq!(receipt.replications, 3);
    assert_eq!(receipt.work_planned, 3 * 2_400 * 6);
    assert_eq!(receipt.work_done, receipt.work_planned);
    assert_eq!(receipt.looks, 3 * 12);
    assert_eq!(receipt.states, 4, "a sum and a count for each of two arms");
    assert_eq!(receipt.peak_units, 2_400);
    assert!(receipt.estimated_peak_bytes > 0);

    // The participant-level family takes the same batch: 200 participants x 6 responses each.
    let family = participant_spec(2_400, 12);
    let history = simulate_participant_history(&family, &EQUAL_ARMS, None, 5_641_339, &never)
        .expect("a retained history");
    assert_eq!(history.batches, vec![200; 12]);
    assert_eq!(history.arms.len(), 2_400);
    assert!(history.successes.iter().flatten().all(|&successes| successes <= 6));
    assert_eq!(history.successes.iter().flatten().count(), 2_200, "the last batch is in follow-up");
    let request =
        ParticipantSimulationRequest { replications: 2, seed: 5_641_339, true_correlation: None };
    let receipt = simulate_participant_procedure(&family, &EQUAL_ARMS, &request, &never)
        .expect("a simulation")
        .receipt;
    assert_eq!((receipt.replications, receipt.looks, receipt.states), (2, 24, 4 * 7));
    assert_eq!(receipt.work_planned, 2 * 2_400 * 6);
    assert_eq!(receipt.work_done, receipt.work_planned);
}

#[test]
fn adaptive_declared_capacity_separate_from_exact_law_limits() {
    // 2,400 participants are declared without complaint...
    let spec = fixed_law_spec(4, 2, 2_400);
    assert_eq!(spec.enrolment().iter().map(|&n| u64::from(n)).sum::<u64>(), 2_400);
    // ...and it is the exact law that cannot hold their histories: the refusal names the exact-law
    // state limit, not the declared capacity.
    let refused = law_error(&spec);
    assert!(matches!(refused, CapacityError::LawStates { limit: 1_000_000, .. }), "{refused:?}");
    assert_eq!(refused.class(), "exact_law_states");
    assert_eq!(refused.detail(), "adaptive_inference.exact_law_states_exceeded");
    assert!(refused.requested() > refused.limit());
    assert!(refused.remedy().contains("simulate"));
    for declared in
        ["adaptive_inference.declared_units_exceeded", "adaptive_inference.declared_looks_exceeded"]
    {
        assert_ne!(refused.detail(), declared);
    }
    // The simulation route runs the same declaration.
    let simulated =
        simulate_procedure(&spec, &[0.3, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5, 0.6], &weighting(2), &never);
    assert!(simulated.is_ok(), "{simulated:?}");

    // The converse: a declaration past the declared capacity is refused as that class, before any
    // law is asked for.
    let past = ProcedureSpec::new(ProcedureDeclaration {
        arms: 4,
        context_probabilities: Some(vec![0.5, 0.5]),
        assignment: Some(AssignmentRule::Fixed { probabilities: vec![0.25; 4] }),
        enrolment: Some(vec![MAX_PROCEDURE_UNITS + 1]),
        observation: Some(OutcomeObservation::COMPLETE),
        exposure: Some(ExposureModel::none(4)),
        stopping: Some(StoppingRule::NoStopping),
        selection: Some(FinalSelection::NoSelection),
    })
    .expect_err("past the declaration capacity");
    assert_eq!(
        past.gaps,
        vec![ContractGap::Capacity(CapacityError::DeclaredUnits {
            limit: u64::from(MAX_PROCEDURE_UNITS),
            requested: u64::from(MAX_PROCEDURE_UNITS) + 1,
        })]
    );
}

#[test]
fn adaptive_simulation_work_and_memory_limits_refuse() {
    let family = participant_spec(2_400, 12);
    let request = |replications| ParticipantSimulationRequest {
        replications,
        seed: 9_827_611,
        true_correlation: None,
    };
    // Work: 300,000 replications of 2,400 participants x 6 responses is 4.32e9 response draws,
    // over the 4e9 bound; refused before a replicate runs, naming the bound and the request.
    let refused = simulate_participant_procedure(&family, &EQUAL_ARMS, &request(300_000), &never)
        .unwrap_err();
    assert_eq!(
        refused,
        Error::Capacity(CapacityError::SimulationWork {
            limit: 4_000_000_000,
            requested: 4_320_000_000,
        })
    );
    let Error::Capacity(work) = refused else { unreachable!() };
    assert_eq!(work.detail(), "adaptive_inference.simulation_work_exceeded");
    assert!(work.to_string().contains("4000000000") && work.to_string().contains("4320000000"));
    assert!(work.remedy().contains("work budget"));
    // Just inside the bound is admitted (checked on the budget, not run).
    let inside = simulate_participant_procedure_within(
        &family,
        &EQUAL_ARMS,
        &request(1),
        &work_budget(2_400 * 6),
        &never,
    );
    assert_eq!(inside.unwrap().receipt.work_planned, 2_400 * 6);
    let one_short = simulate_participant_procedure_within(
        &family,
        &EQUAL_ARMS,
        &request(1),
        &work_budget(2_400 * 6 - 1),
        &never,
    )
    .unwrap_err();
    assert_eq!(
        one_short,
        Error::Capacity(CapacityError::SimulationWork {
            limit: 2_400 * 6 - 1,
            requested: 2_400 * 6
        })
    );

    // Memory: the replicate records and the retained history of one replicate against the budget.
    let refused = simulate_participant_procedure_within(
        &family,
        &EQUAL_ARMS,
        &request(1_000),
        &memory_budget(100_000),
        &never,
    )
    .unwrap_err();
    let Error::Capacity(memory) = refused else { panic!("{refused:?}") };
    assert!(matches!(memory, CapacityError::SimulationMemory { limit: 100_000, .. }));
    assert_eq!(memory.detail(), "adaptive_inference.simulation_memory_exceeded");
    assert!(memory.requested() > 100_000);
    assert!(memory.to_string().contains("100000"));
    assert_ne!(memory.detail(), work.detail());
}
