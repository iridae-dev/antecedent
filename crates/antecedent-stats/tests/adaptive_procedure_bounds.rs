//! The declared size bounds of an adaptive procedure: each bound is accepted at its value and
//! refused, with a named detail, one past it. Nothing here claims a statistic; the refusals
//! are the supported-bounds evidence of the procedure contract and of the simulation route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, AssignmentRule, BetaPrior, CapacityError, ContractGap,
    Estimand, ExposureModel, FinalSelection, MAX_OUTCOME_LAG, MAX_PROCEDURE_ARMS,
    MAX_PROCEDURE_CELLS, MAX_PROCEDURE_CONTEXTS, MAX_PROCEDURE_LOOKS, MAX_PROCEDURE_UNITS,
    MAX_SIMULATION_REPLICATIONS, OutcomeModelPrior, OutcomeObservation, ProcedureDeclaration,
    ProcedureSpec, SimulationRequest, StoppingRule, WeightScheme, simulate_procedure,
};

fn never() -> bool {
    false
}

/// A valid fixed-design declaration over `arms` arms, `looks` looks of one unit each.
fn base(arms: usize, looks: usize) -> ProcedureDeclaration {
    ProcedureDeclaration {
        arms,
        context_probabilities: None,
        assignment: Some(AssignmentRule::Fixed { probabilities: vec![1.0 / arms as f64; arms] }),
        enrolment: Some(vec![1; looks]),
        observation: Some(OutcomeObservation::COMPLETE),
        exposure: Some(ExposureModel::none(arms)),
        stopping: Some(StoppingRule::NoStopping),
        selection: Some(FinalSelection::NoSelection),
    }
}

fn gaps(declaration: ProcedureDeclaration) -> Vec<ContractGap> {
    ProcedureSpec::new(declaration).expect_err("a declaration past a bound is refused").gaps
}

#[test]
fn arms_are_bounded() {
    assert_eq!(MAX_PROCEDURE_ARMS, 4);
    assert!(ProcedureSpec::new(base(MAX_PROCEDURE_ARMS, 2)).is_ok());
    assert_eq!(
        gaps(base(MAX_PROCEDURE_ARMS + 1, 2)),
        vec![ContractGap::Invalid("adaptive_inference.arms_invalid")]
    );
    assert_eq!(gaps(base(1, 2)), vec![ContractGap::Invalid("adaptive_inference.arms_invalid")]);
}

#[test]
fn contexts_and_cells_are_bounded() {
    assert_eq!(MAX_PROCEDURE_CONTEXTS, 4);
    assert_eq!(MAX_PROCEDURE_CELLS, 8);
    let contextual = |arms: usize, contexts: usize| {
        let mut d = base(arms, 2);
        d.assignment = Some(AssignmentRule::ThompsonBeta { prior: BetaPrior::UNIFORM });
        d.context_probabilities = Some(vec![1.0 / contexts as f64; contexts]);
        d
    };
    // Four contexts of two arms: the context bound and the cell bound are both met exactly.
    assert!(ProcedureSpec::new(contextual(2, MAX_PROCEDURE_CONTEXTS)).is_ok());
    // One context more.
    assert_eq!(
        gaps(contextual(2, MAX_PROCEDURE_CONTEXTS + 1)),
        vec![ContractGap::Invalid("adaptive_inference.contexts_invalid")]
    );
    // Within the context bound and over the cell bound: three contexts of four arms.
    assert_eq!(
        gaps(contextual(4, 3)),
        vec![ContractGap::Invalid("adaptive_inference.contexts_invalid")]
    );
}

#[test]
fn looks_and_units_are_bounded() {
    assert_eq!(MAX_PROCEDURE_LOOKS, 24);
    assert_eq!(MAX_PROCEDURE_UNITS, 65_535);
    assert!(ProcedureSpec::new(base(2, MAX_PROCEDURE_LOOKS)).is_ok());
    assert_eq!(
        gaps(base(2, MAX_PROCEDURE_LOOKS + 1)),
        vec![ContractGap::Capacity(CapacityError::DeclaredLooks { limit: 24, requested: 25 })]
    );

    let mut at_units = base(2, 1);
    at_units.enrolment = Some(vec![MAX_PROCEDURE_UNITS]);
    assert!(ProcedureSpec::new(at_units).is_ok());
    // The old 400-unit bound is gone: a planned 2,400-participant procedure is declared.
    let mut planned = base(2, 1);
    planned.enrolment = Some(vec![2_400]);
    assert!(ProcedureSpec::new(planned).is_ok());
    let over = |enrolment: Vec<u32>, requested: u64| {
        let mut declaration = base(2, 1);
        declaration.enrolment = Some(enrolment);
        assert_eq!(
            gaps(declaration),
            vec![ContractGap::Capacity(CapacityError::DeclaredUnits {
                limit: u64::from(MAX_PROCEDURE_UNITS),
                requested,
            })]
        );
    };
    over(vec![MAX_PROCEDURE_UNITS + 1], 65_536);
    // The total is bounded, not each look.
    over(vec![MAX_PROCEDURE_UNITS / 2, MAX_PROCEDURE_UNITS / 2 + 2], 65_536);

    // An empty schedule, an empty first look and a schedule of no units are ordering errors,
    // not capacity errors.
    let ordering = vec![ContractGap::Invalid("adaptive_inference.looks_not_ordered")];
    for schedule in [vec![], vec![0, 5], vec![0]] {
        let mut declaration = base(2, 1);
        declaration.enrolment = Some(schedule);
        assert_eq!(gaps(declaration), ordering);
    }
}

#[test]
fn outcome_delay_is_bounded() {
    assert_eq!(MAX_OUTCOME_LAG, 4);
    let lagged = |lag_looks: usize| {
        let mut d = base(2, 8);
        d.observation =
            Some(OutcomeObservation { lag_looks, retain_success: 1.0, retain_failure: 1.0 });
        d
    };
    assert!(ProcedureSpec::new(lagged(MAX_OUTCOME_LAG)).is_ok());
    assert_eq!(
        gaps(lagged(MAX_OUTCOME_LAG + 1)),
        vec![ContractGap::Invalid("adaptive_inference.outcome_availability_invalid")]
    );
}

#[test]
fn simulation_replications_are_bounded() {
    assert_eq!(MAX_SIMULATION_REPLICATIONS, 1_000_000);
    let spec = ProcedureSpec::new(base(2, 3)).unwrap();
    let request = |replications: u32| SimulationRequest {
        replications,
        seed: 1,
        estimand: Estimand::Contrast { plus: 1, minus: 0 },
        scheme: WeightScheme::Uniform,
        outcome_prior: OutcomeModelPrior { mean: 0.5, strength: 2.0 },
        probability_floor: 0.01,
    };
    // Past the bound, refused before any replicate runs, with the replication class of limit.
    assert_eq!(
        simulate_procedure(&spec, &[0.4, 0.6], &request(MAX_SIMULATION_REPLICATIONS + 1), &never)
            .unwrap_err(),
        Error::Capacity(CapacityError::SimulationReplications {
            limit: 1_000_000,
            requested: 1_000_001,
        })
    );
    // Zero replications are an invalid request, not a capacity.
    assert!(matches!(
        simulate_procedure(&spec, &[0.4, 0.6], &request(0), &never),
        Err(Error::InvalidParameter(_))
    ));
    // A request inside the bound runs.
    assert!(simulate_procedure(&spec, &[0.4, 0.6], &request(5), &never).is_ok());
}
