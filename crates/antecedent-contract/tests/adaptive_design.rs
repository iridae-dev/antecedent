//! The adaptive analysis contract and the procedure matrix: completeness,
//! consistency, canonical identity, paradigm authorization and the
//! observation accounting of interim analyses.
//!
//! Expected values are written down independently of the code under test: the
//! matrix counts come from the axis sizes and the delivered cells, the
//! mass accounting from the sizes of nested prefixes, and the refusal
//! vocabulary from the registry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, BoundarySidedness,
    CellStatus, DelayMechanism, InferenceParadigm, InterimAnalysis, MonitoringStrategy,
    OutcomeAvailability, OutcomeStructure, PolicyEpoch, ProcedureCell, SpendingKind,
    StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec, licensed_routes, procedure_cells,
    procedure_status,
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
        StoppingRuleKind::Futility => StoppingSpec::PosteriorProbabilityBest { threshold: 0.9 },
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

fn details(error: &antecedent_contract::adaptive_design::AdaptiveContractError) -> Vec<String> {
    error.refusals.iter().map(|r| r.detail.clone()).collect()
}

// -- the matrix ----------------------------------------------------------------------------

#[test]
fn the_matrix_lists_every_combination_once_and_classifies_each() {
    let cells = procedure_cells();
    assert_eq!(cells.len(), 6 * 3 * 4 * 5);
    let identifiers: BTreeSet<String> = cells.iter().map(ProcedureCell::identifier).collect();
    assert_eq!(identifiers.len(), cells.len(), "no cell is listed twice");
    let mut licensed = 0;
    let mut incoherent = 0;
    let mut pairing = 0;
    let mut carried = 0;
    for cell in &cells {
        match procedure_status(cell) {
            CellStatus::Licensed { paradigms } => {
                assert!(!paradigms.is_empty());
                licensed += 1;
            }
            CellStatus::ClosedIncoherent { reason } => {
                assert!(!reason.is_empty());
                incoherent += 1;
            }
            CellStatus::ClosedPairing { reason } => {
                assert!(reason.contains("pairing"));
                pairing += 1;
            }
            CellStatus::CarriedForward { owner, reason } => {
                assert!(owner.starts_with("adaptive_inference/"));
                assert!(!reason.is_empty());
                carried += 1;
            }
            _ => panic!("an unclassified cell"),
        }
    }
    // Incoherent: stopping without looks (4 x 18), looks without a rule
    // (3 x 18 x ... see the axis sizes) and alpha spending with a
    // decision-theoretic rule. Counted from the axes, not from the code.
    let per_cell_group = 6 * 3; // assignment x outcome
    let stopping_without_looks = per_cell_group * 4; // final-only with one of 4 rules
    let looks_without_rule = per_cell_group * 3; // 3 monitoring strategies with no rule
    let spending_with_decision_theory = per_cell_group; // group sequential x decision theoretic
    assert_eq!(
        incoherent,
        stopping_without_looks + looks_without_rule + spending_with_decision_theory
    );
    assert_eq!((licensed, pairing, carried), (6, 18, 192));
    assert_eq!(licensed + incoherent + pairing + carried, cells.len());
}

#[test]
fn exactly_the_delivered_cells_are_licensed_with_their_own_paradigms() {
    use AssignmentFamily as A;
    use InferenceParadigm as P;
    use MonitoringStrategy as M;
    use OutcomeStructure as O;
    use StoppingRuleKind as S;
    let expected = [
        (
            cell(A::ThompsonBetaBernoulli, O::SingleOutcome, M::FixedInterimLooks, S::Efficacy),
            vec![P::BayesianPosterior, P::AnytimeValidFrequentist],
        ),
        (
            cell(
                A::ThompsonBetaBernoulli,
                O::SingleOutcome,
                M::FixedInterimLooks,
                S::DecisionTheoretic,
            ),
            vec![P::BayesianPosterior, P::DecisionTheoreticStopping],
        ),
        (
            cell(A::EpsilonGreedy, O::SingleOutcome, M::FixedInterimLooks, S::Efficacy),
            vec![P::FrequentistAdaptiveWeighting, P::AnytimeValidFrequentist],
        ),
        (
            cell(A::EpsilonGreedy, O::RepeatedMeasures, M::FinalAnalysisOnly, S::NoStopping),
            vec![P::FrequentistAdaptiveWeighting],
        ),
        (
            cell(A::FixedRandomization, O::SingleOutcome, M::GroupSequential, S::Efficacy),
            vec![P::FrequentistGroupSequential],
        ),
        (
            cell(A::FixedRandomization, O::SingleOutcome, M::FinalAnalysisOnly, S::NoStopping),
            vec![P::FrequentistFixedSample],
        ),
    ];
    let routes = licensed_routes();
    assert_eq!(routes.len(), expected.len());
    for (cell, paradigms) in expected {
        let found = routes.iter().find(|(c, _)| *c == cell).expect("delivered cell is licensed");
        assert_eq!(found.1, paradigms.as_slice(), "{}", cell.identifier());
    }
    // A guarantee for two components does not license their pairing: each of
    // these shares an assignment family or a monitoring strategy with a
    // licensed cell and is not licensed itself.
    for pairing in [
        cell(A::ThompsonBetaBernoulli, O::SingleOutcome, M::GroupSequential, S::Efficacy),
        cell(A::EpsilonGreedy, O::SingleOutcome, M::GroupSequential, S::Efficacy),
        cell(A::EpsilonGreedy, O::RepeatedMeasures, M::GroupSequential, S::Efficacy),
        cell(A::FixedRandomization, O::RepeatedMeasures, M::FixedInterimLooks, S::Efficacy),
    ] {
        assert!(
            matches!(procedure_status(&pairing), CellStatus::ClosedPairing { .. }),
            "{}",
            pairing.identifier()
        );
    }
}

/// Parse the `[[cell]]` rows of `parity/adaptive_procedures.toml`:
/// `(id, status, owner, paradigms)`.
fn registry_rows() -> Vec<(String, String, Option<String>, Vec<String>)> {
    let text = include_str!("../../../parity/adaptive_procedures.toml");
    let mut rows: Vec<(String, String, Option<String>, Vec<String>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let value = |key: &str| {
            line.strip_prefix(key)
                .and_then(|rest| rest.trim_start().strip_prefix('='))
                .map(|rest| rest.trim().trim_matches('"').to_owned())
        };
        if let Some(id) = value("id") {
            rows.push((id, String::new(), None, Vec::new()));
        } else if let Some(status) = value("status") {
            rows.last_mut().expect("status follows id").1 = status;
        } else if let Some(owner) = value("owner") {
            rows.last_mut().expect("owner follows id").2 = Some(owner);
        } else if let Some(list) = line.strip_prefix("paradigms") {
            let inner = list.trim_start().trim_start_matches('=').trim().trim_matches(['[', ']']);
            rows.last_mut().expect("paradigms follow id").3 =
                inner.split(',').map(|p| p.trim().trim_matches('"').to_owned()).collect();
        }
    }
    rows
}

#[test]
fn the_registry_table_and_the_executable_matrix_agree_cell_by_cell() {
    let rows = registry_rows();
    let cells = procedure_cells();
    assert_eq!(rows.len(), cells.len(), "every cell is in the registry, none unlisted");
    for (row, cell) in rows.iter().zip(&cells) {
        assert_eq!(row.0, cell.identifier(), "registry order is the matrix order");
        match procedure_status(cell) {
            CellStatus::Licensed { paradigms } => {
                assert_eq!(row.1, "licensed", "{}", row.0);
                let names: Vec<&str> = paradigms.iter().map(|p| p.as_str()).collect();
                assert_eq!(row.3, names, "{}", row.0);
            }
            CellStatus::ClosedIncoherent { .. } => {
                assert_eq!(row.1, "closed_incoherent", "{}", row.0)
            }
            CellStatus::ClosedPairing { .. } => assert_eq!(row.1, "closed_pairing", "{}", row.0),
            CellStatus::CarriedForward { owner, .. } => {
                assert_eq!(row.1, "carried_forward", "{}", row.0);
                assert_eq!(row.2.as_deref(), Some(owner), "{}", row.0);
            }
            _ => panic!("an unclassified cell"),
        }
    }
}

// -- completeness ---------------------------------------------------------------------------

#[test]
fn a_complete_declaration_is_accepted() {
    let contract = AdaptiveDesignContract::new(thompson()).expect("complete");
    assert_eq!(contract.units(), 8);
    assert_eq!(contract.arms(), 2);
    assert_eq!(contract.looks(), 2);
    assert!(contract.is_adaptive());
}

#[test]
fn every_missing_element_is_its_own_typed_refusal() {
    type Strip = fn(&mut AdaptiveDesignDeclaration);
    let cases: [(&str, Strip); 6] = [
        ("adaptive_inference.policy_history_missing", |d| d.policy_history = None),
        ("adaptive_inference.assignment_probabilities_missing", |d| {
            d.assignment_probabilities = None;
        }),
        ("adaptive_inference.exposure_history_missing", |d| d.exposure = None),
        ("adaptive_inference.outcome_availability_missing", |d| d.availability = None),
        ("adaptive_inference.interim_analyses_missing", |d| d.interim_analyses = None),
        ("adaptive_inference.stopping_rule_missing", |d| d.stopping_rule = None),
    ];
    for (detail, strip) in &cases {
        let mut declaration = thompson();
        strip(&mut declaration);
        let error = AdaptiveDesignContract::new(declaration).expect_err("one element missing");
        assert_eq!(details(&error), vec![(*detail).to_owned()], "exactly one detail per element");
        assert_eq!(error.refusals[0].code, "adaptive_contract_incomplete");
        assert!(error.refusals[0].remedy.is_some_and(|r| r.starts_with("retain")));
    }
    let mut stripped = thompson();
    for (_, strip) in &cases {
        strip(&mut stripped);
    }
    let error = AdaptiveDesignContract::new(stripped).expect_err("everything missing");
    let all: BTreeSet<String> = details(&error).into_iter().collect();
    let wanted: BTreeSet<String> = cases.iter().map(|(d, _)| (*d).to_owned()).collect();
    assert_eq!(all, wanted, "one detail per missing element and nothing else");
}

#[test]
fn inconsistent_records_are_refused_with_distinct_details() {
    let mutate = |edit: fn(&mut AdaptiveDesignDeclaration)| {
        let mut declaration = thompson();
        edit(&mut declaration);
        details(&AdaptiveDesignContract::new(declaration).expect_err("inconsistent"))
    };
    assert_eq!(
        mutate(|d| d.assignment_probabilities.as_mut().unwrap()[0] = vec![0.5, 0.6]),
        ["adaptive_inference.probabilities_invalid"]
    );
    assert_eq!(
        mutate(|d| d.assignment_probabilities.as_mut().unwrap()[0] = vec![0.0, 1.0]),
        ["adaptive_inference.assigned_probability_zero"]
    );
    assert_eq!(mutate(|d| d.assigned_arm[2] = 7), ["adaptive_inference.arm_out_of_range"]);
    assert_eq!(
        mutate(|d| d.interim_analyses.as_mut().unwrap()[1].units_enrolled = 7),
        ["adaptive_inference.looks_not_ordered"]
    );
    assert_eq!(
        mutate(|d| d.interim_analyses.as_mut().unwrap()[0].information_fraction = 1.0),
        ["adaptive_inference.looks_not_ordered"]
    );
    assert_eq!(
        mutate(|d| d.availability.as_mut().unwrap().available_at_look[5] = Some(0)),
        ["adaptive_inference.shape_mismatch"],
        "an outcome cannot be available before its unit is enrolled"
    );
    assert_eq!(
        mutate(|d| d.availability.as_mut().unwrap().available_at_look[0] = None),
        ["adaptive_inference.shape_mismatch"],
        "availability and observed outcomes must agree"
    );
    assert_eq!(
        mutate(|d| d.stopping_rule.as_mut().unwrap().kind = StoppingRuleKind::Resource),
        ["adaptive_inference.rule_mismatch"]
    );
    assert_eq!(
        mutate(|d| d.outcomes[1] = vec![0.5]),
        ["adaptive_inference.outcome_invalid"],
        "Beta-Bernoulli is defined for {{0, 1}}"
    );
    assert_eq!(
        mutate(|d| d.policy_history.as_mut().unwrap()[0].first_unit = 1),
        ["adaptive_inference.policy_history_invalid"]
    );
    assert_eq!(
        mutate(|d| d.probability_floor = 0.0),
        ["adaptive_inference.probability_floor_invalid"]
    );
    // A fixed-randomization declaration whose probabilities vary is not one.
    let mut fixed = declaration(cell(
        AssignmentFamily::FixedRandomization,
        OutcomeStructure::SingleOutcome,
        MonitoringStrategy::FinalAnalysisOnly,
        StoppingRuleKind::NoStopping,
    ));
    fixed.assignment_probabilities.as_mut().unwrap()[3] = vec![0.4, 0.6];
    let error = AdaptiveDesignContract::new(fixed).expect_err("probabilities vary");
    assert_eq!(details(&error), ["adaptive_inference.rule_mismatch"]);
    // The refusals use only registered codes.
    for refusal in &error.refusals {
        assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code));
    }
}

// -- identity -----------------------------------------------------------------------------------

#[test]
fn identity_is_deterministic_and_sensitive_to_every_element() {
    let base = AdaptiveDesignContract::new(thompson()).unwrap().identity();
    assert_eq!(AdaptiveDesignContract::new(thompson()).unwrap().identity(), base);
    let mutations: [fn(&mut AdaptiveDesignDeclaration); 9] = [
        |d| d.label = "other".to_owned(),
        |d| d.assigned_arm.swap(0, 1),
        |d| d.assignment_probabilities.as_mut().unwrap()[4] = vec![0.3 + 1e-12, 0.7 - 1e-12],
        |d| d.outcomes[3] = vec![0.0],
        |d| d.interim_analyses.as_mut().unwrap()[0].information_fraction = 0.25,
        |d| {
            d.stopping_rule.as_mut().unwrap().spec =
                StoppingSpec::PosteriorProbabilityBest { threshold: 0.975 };
        },
        |d| d.policy_history.as_mut().unwrap()[1].policy_identity = digest(9),
        |d| d.probability_floor = 0.01,
        |d| d.arm_labels.swap(0, 1),
    ];
    let mut seen = BTreeSet::from([base.to_hex()]);
    for (index, edit) in mutations.iter().enumerate() {
        let mut declaration = thompson();
        edit(&mut declaration);
        // Swapping assigned arms 0 and 1 changes exposure consistency only at
        // authorization, so every mutation still constructs.
        let identity = AdaptiveDesignContract::new(declaration)
            .unwrap_or_else(|e| panic!("mutation {index}: {e}"))
            .identity();
        assert!(seen.insert(identity.to_hex()), "mutation {index} did not change the identity");
    }
}

// -- authorization ----------------------------------------------------------------------

#[test]
fn fixed_sample_inference_of_adaptive_data_is_refused() {
    let contract = AdaptiveDesignContract::new(thompson()).unwrap();
    let refusal = contract.admit_paradigm(InferenceParadigm::FrequentistFixedSample).unwrap_err();
    assert_eq!(refusal.code, "adaptive_design_ignored");
    assert_eq!(refusal.detail, "adaptive_inference.fixed_sample_on_adaptive_data");
    // Adaptive in any one axis suffices: a fixed assignment with interim looks.
    let looked_at = declaration(cell(
        AssignmentFamily::FixedRandomization,
        OutcomeStructure::SingleOutcome,
        MonitoringStrategy::GroupSequential,
        StoppingRuleKind::Efficacy,
    ));
    let contract = AdaptiveDesignContract::new(looked_at).unwrap();
    let refusal = contract.admit_paradigm(InferenceParadigm::FrequentistFixedSample).unwrap_err();
    assert_eq!(refusal.detail, "adaptive_inference.fixed_sample_on_adaptive_data");
    // A design fixed in advance is the one case the paradigm is authorized.
    let fixed = declaration(cell(
        AssignmentFamily::FixedRandomization,
        OutcomeStructure::SingleOutcome,
        MonitoringStrategy::FinalAnalysisOnly,
        StoppingRuleKind::NoStopping,
    ));
    let contract = AdaptiveDesignContract::new(fixed).unwrap();
    assert!(!contract.is_adaptive());
    let granted = contract.admit_paradigm(InferenceParadigm::FrequentistFixedSample).unwrap();
    assert_eq!(granted.contract_identity, contract.identity());
}

#[test]
fn a_paradigm_never_transfers_to_a_cell_it_is_not_licensed_for() {
    let contract = AdaptiveDesignContract::new(thompson()).unwrap();
    assert!(contract.admit_paradigm(InferenceParadigm::BayesianPosterior).is_ok());
    assert!(contract.admit_paradigm(InferenceParadigm::AnytimeValidFrequentist).is_ok());
    let refusal =
        contract.admit_paradigm(InferenceParadigm::FrequentistAdaptiveWeighting).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "adaptive_inference.paradigm_not_licensed_for_cell");
    let refusal =
        contract.admit_paradigm(InferenceParadigm::FrequentistGroupSequential).unwrap_err();
    assert_eq!(refusal.detail, "adaptive_inference.paradigm_not_licensed_for_cell");
}

#[test]
fn closed_carried_and_paired_cells_are_refused_with_their_own_details() {
    use AssignmentFamily as A;
    use MonitoringStrategy as M;
    use OutcomeStructure as O;
    use StoppingRuleKind as S;
    let refused = |cell: ProcedureCell| {
        let contract = AdaptiveDesignContract::new(declaration(cell)).expect("constructs");
        contract.admit_paradigm(InferenceParadigm::BayesianPosterior).unwrap_err()
    };
    let carried =
        refused(cell(A::UpperConfidenceBound, O::SingleOutcome, M::FixedInterimLooks, S::Efficacy));
    assert_eq!(carried.detail, "adaptive_inference.cell_carried_forward");
    assert_eq!(carried.offending.as_deref(), Some("adaptive_inference/upper_confidence_bound"));
    let paired =
        refused(cell(A::ThompsonBetaBernoulli, O::SingleOutcome, M::GroupSequential, S::Efficacy));
    assert_eq!(paired.detail, "adaptive_inference.pairing_not_licensed");
    let closed = refused(cell(
        A::ThompsonBetaBernoulli,
        O::SingleOutcome,
        M::FinalAnalysisOnly,
        S::Efficacy,
    ));
    assert_eq!(closed.detail, "adaptive_inference.cell_closed");
    for refusal in [&carried, &paired, &closed] {
        assert_eq!(refusal.code, "cell_not_licensed");
    }
}

#[test]
fn noncompliance_and_outcome_dependent_delay_are_refused_before_any_number() {
    let mut declaration = thompson();
    declaration.exposure.as_mut().unwrap()[2] = 1;
    let contract = AdaptiveDesignContract::new(declaration).unwrap();
    let refusal = contract.admit_paradigm(InferenceParadigm::BayesianPosterior).unwrap_err();
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.exposure_differs_from_assignment");

    let mut declaration = thompson();
    declaration.availability.as_mut().unwrap().delay = DelayMechanism::OutcomeDependent;
    let contract = AdaptiveDesignContract::new(declaration).unwrap();
    let refusal = contract.admit_paradigm(InferenceParadigm::BayesianPosterior).unwrap_err();
    assert_eq!(refusal.detail, "adaptive_inference.delay_not_ignorable");

    let mut declaration = thompson();
    declaration.availability.as_mut().unwrap().delay = DelayMechanism::IgnorableAdministrative;
    let contract = AdaptiveDesignContract::new(declaration).unwrap();
    assert!(contract.admit_paradigm(InferenceParadigm::BayesianPosterior).is_ok());
}

// -- delayed outcomes and the evidence ledger ---------------------------------------------------------

#[test]
fn delayed_outcomes_enter_the_analysis_only_when_observed() {
    let mut declaration = thompson();
    // Units 1 and 3 report one look late; unit 7 never reports.
    let availability = declaration.availability.as_mut().unwrap();
    availability.available_at_look[1] = Some(1);
    availability.available_at_look[3] = Some(1);
    availability.available_at_look[7] = None;
    declaration.outcomes[7] = Vec::new();
    availability.delay = DelayMechanism::IgnorableAdministrative;
    let contract = AdaptiveDesignContract::new(declaration).unwrap();
    assert_eq!(contract.analysis_units(0), vec![0, 2]);
    assert_eq!(contract.analysis_units(1), vec![0, 1, 2, 3, 4, 5, 6]);
    // Nested: each look's data contain the previous look's.
    assert!(contract.analysis_units(0).iter().all(|u| contract.analysis_units(1).contains(u)));
}

#[test]
fn interim_looks_count_each_observation_once() {
    let contract = AdaptiveDesignContract::new(thompson()).unwrap();
    let evidence = contract.interim_evidence().unwrap();
    // Looks analyse 4 and 8 units: 12 slots, 8 distinct, 4 counted twice.
    assert_eq!(evidence.mass.declared_slots, 12);
    assert_eq!(evidence.mass.distinct_observations, 8);
    assert_eq!(evidence.mass.repeated_slots, 4);
    assert!(evidence.mass.unknown_sources.is_empty());
    assert_eq!(evidence.look_sources.len(), 2);
    assert!(evidence.ledger.result(evidence.reuse).is_some());
    // The ledger never licenses an interval for the repeated use.
    let error = evidence.ledger.request_interval(evidence.reuse).unwrap_err();
    assert_eq!(error.code, "cell_not_licensed");
}

#[test]
fn the_looks_are_never_pooled_as_independent_evidence() {
    let contract = AdaptiveDesignContract::new(thompson()).unwrap();
    let evidence = contract.interim_evidence().unwrap();
    let refusal = contract.refuse_independent_pooling_of_looks(&evidence).unwrap_err();
    assert_eq!(refusal.code, "evidence_double_counted");
    assert_eq!(refusal.detail, "adaptive_inference.looks_share_observations");
    // With only a final analysis there is no earlier look to double count.
    let single = declaration(cell(
        AssignmentFamily::FixedRandomization,
        OutcomeStructure::SingleOutcome,
        MonitoringStrategy::FinalAnalysisOnly,
        StoppingRuleKind::NoStopping,
    ));
    let single = AdaptiveDesignContract::new(single).unwrap();
    let evidence = single.interim_evidence().unwrap();
    assert_eq!(evidence.mass.repeated_slots, 0);
    assert!(single.refuse_independent_pooling_of_looks(&evidence).is_ok());
}

#[test]
fn delays_change_the_accounting_by_the_sizes_of_the_nested_looks() {
    let mut declaration = thompson();
    let availability = declaration.availability.as_mut().unwrap();
    availability.available_at_look[1] = Some(1);
    availability.available_at_look[3] = Some(1);
    availability.delay = DelayMechanism::IgnorableAdministrative;
    let contract = AdaptiveDesignContract::new(declaration).unwrap();
    let evidence = contract.interim_evidence().unwrap();
    // Look 0 analyses units {0, 2}; look 1 analyses all eight.
    assert_eq!(evidence.mass.declared_slots, 10);
    assert_eq!(evidence.mass.distinct_observations, 8);
    assert_eq!(evidence.mass.repeated_slots, 2);
}

#[test]
fn observation_identities_are_stable_per_contract_and_unit() {
    let a = AdaptiveDesignContract::new(thompson()).unwrap();
    let mut other = thompson();
    other.label = "another".to_owned();
    let b = AdaptiveDesignContract::new(other).unwrap();
    assert_eq!(a.observation_id(3), a.observation_id(3));
    assert_ne!(a.observation_id(3), a.observation_id(4));
    assert_ne!(a.observation_id(3), b.observation_id(3));
}
