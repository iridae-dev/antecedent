//! Interim-look accounting of an adaptive design against hand-built look schedules.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, unused_imports)]

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, BoundarySidedness,
    CellStatus, DelayMechanism, InferenceParadigm, InterimAnalysis, MonitoringStrategy,
    OutcomeAvailability, OutcomeStructure, PolicyEpoch, ProcedureCell, SpendingKind,
    StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec, procedure_cells, procedure_status,
    supported_routes,
};
use antecedent_core::SemanticDigest;

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn cell(
    assignment: AssignmentFamily,
    outcome: OutcomeStructure,
    monitoring: MonitoringStrategy,
    stopping: StoppingRuleKind,
) -> ProcedureCell {
    ProcedureCell { assignment, outcome, monitoring, stopping }
}

fn thompson_cell() -> ProcedureCell {
    cell(
        AssignmentFamily::ThompsonBetaBernoulli,
        OutcomeStructure::SingleOutcome,
        MonitoringStrategy::FixedInterimLooks,
        StoppingRuleKind::Efficacy,
    )
}

fn rule_for(cell: ProcedureCell) -> StoppingRuleDeclaration {
    let spec = match cell.stopping {
        StoppingRuleKind::Efficacy if cell.monitoring == MonitoringStrategy::GroupSequential => {
            StoppingSpec::AlphaSpending {
                spending: SpendingKind::OBrienFlemingType,
                total_alpha: 0.05,
                sidedness: BoundarySidedness::TwoSidedSymmetric,
            }
        }
        StoppingRuleKind::Efficacy => StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
        StoppingRuleKind::DecisionTheoretic => {
            StoppingSpec::PosteriorExpectedLoss { loss_threshold: 0.05 }
        }
        StoppingRuleKind::Resource => StoppingSpec::ResourceCap { max_units: 8 },
        StoppingRuleKind::Futility => {
            StoppingSpec::PosteriorFutility { margin: 0.05, threshold: 0.1 }
        }
        _ => StoppingSpec::NoStopping,
    };
    StoppingRuleDeclaration { kind: cell.stopping, spec, ended_by_rule: false }
}

/// Eight units, two arms, looks after four and eight units.
fn declaration(cell: ProcedureCell) -> AdaptiveDesignDeclaration {
    let assigned_arm = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let varying = cell.assignment != AssignmentFamily::FixedRandomization;
    let probabilities = assigned_arm
        .iter()
        .enumerate()
        .map(|(unit, _)| if varying && unit >= 4 { vec![0.3, 0.7] } else { vec![0.5, 0.5] })
        .collect();
    let final_only = cell.monitoring == MonitoringStrategy::FinalAnalysisOnly;
    let looks = if final_only {
        vec![InterimAnalysis { units_enrolled: 8, information_fraction: 1.0 }]
    } else {
        vec![
            InterimAnalysis { units_enrolled: 4, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 8, information_fraction: 1.0 },
        ]
    };
    let available_at_look = (0..8).map(|unit| Some(u32::from(!(final_only || unit < 4)))).collect();
    let repeated = cell.outcome == OutcomeStructure::RepeatedMeasures;
    let outcomes = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        .iter()
        .map(|&y| if repeated { vec![y, 1.0 - y] } else { vec![y] })
        .collect();
    AdaptiveDesignDeclaration {
        label: "trial".to_owned(),
        cell,
        arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![
            PolicyEpoch { first_unit: 0, policy_identity: digest(1) },
            PolicyEpoch { first_unit: 4, policy_identity: digest(2) },
        ]),
        assignment_probabilities: Some(probabilities),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look,
            delay: DelayMechanism::Immediate,
        }),
        outcomes,
        interim_analyses: Some(looks),
        stopping_rule: Some(rule_for(cell)),
        probability_floor: 0.05,
    }
}

fn thompson() -> AdaptiveDesignDeclaration {
    declaration(thompson_cell())
}

/// Looks after 3, 6 and 8 enrolled units. The outcomes of units 0-2 are available at look 0, of
/// units 3 and 5 at look 1, and of units 4, 6 and 7 at look 2 (unit 4 reports one look late).
/// By hand the analysed sets are {0,1,2}, {0,1,2,3,5} and all eight units, so the three looks
/// hold 3 + 5 + 8 = 16 slots over 8 distinct units with 8 repeated.
#[test]
fn three_delayed_looks_analyse_the_hand_derived_nested_sets() {
    let mut declaration = thompson();
    declaration.interim_analyses = Some(vec![
        InterimAnalysis { units_enrolled: 3, information_fraction: 0.375 },
        InterimAnalysis { units_enrolled: 6, information_fraction: 0.75 },
        InterimAnalysis { units_enrolled: 8, information_fraction: 1.0 },
    ]);
    let availability = declaration.availability.as_mut().unwrap();
    availability.available_at_look = vec![
        Some(0),
        Some(0),
        Some(0),
        Some(1),
        Some(2),
        Some(1),
        Some(2),
        Some(2),
    ];
    availability.delay = DelayMechanism::IgnorableAdministrative;
    let contract = AdaptiveDesignContract::new(declaration).unwrap();
    assert_eq!(contract.analysis_units(0).unwrap(), vec![0, 1, 2]);
    assert_eq!(contract.analysis_units(1).unwrap(), vec![0, 1, 2, 3, 5]);
    assert_eq!(contract.analysis_units(2).unwrap(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let evidence = contract.interim_evidence().unwrap();
    assert_eq!(evidence.mass.declared_slots, 16);
    assert_eq!(evidence.mass.distinct_observations, 8);
    assert_eq!(evidence.mass.repeated_slots, 8);
    assert_eq!(evidence.look_sources.len(), 3);
}
