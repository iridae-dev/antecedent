//! The closed transported row of the cross-world capability matrix.
//!
//! No joint theorem composes a transport licence with factual abduction in a target
//! population, so every transported request whose quantity and model have no theorem-scoped
//! route of their own is refused with the closed row's code, detail, remedy and the named
//! unmet prerequisites. This file sweeps the whole quantity-by-model-by-everything-else surface
//! of `decide` for the transported population, checks that the only transported cell that opens
//! is the static additive linear one (and that it opens to its own route rather than to an
//! abduction row), and checks the executor's use of the row: a target population other than
//! the declared one refuses before any work, while the declared population itself executes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::matrix::CapabilityStatus;
use antecedent_counterfactual::selective::{
    ConditioningKind, EvidenceKind, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteQuantity, FiniteScm, FiniteScmExecutor, FiniteVariable,
    FiniteWorld, HistoryKind, InterventionKind, MatrixRequest, ModelClass, NoiseCoupling,
    PopulationKind, QuantityClass, WorldAtom, decide, rows,
};

const QUANTITIES: [QuantityClass; 5] = [
    QuantityClass::CounterfactualMarginal,
    QuantityClass::CounterfactualJoint,
    QuantityClass::NaturalPathSpecific,
    QuantityClass::Temporal,
    QuantityClass::TransportedCrossWorld,
];
const MODELS: [ModelClass; 6] = [
    ModelClass::FiniteMarkovianScm,
    ModelClass::FiniteSharedExogenousScm,
    ModelClass::AdditiveLinearTwoSlice,
    ModelClass::AdditiveLinearStaticTransported,
    ModelClass::SemiMarkovian,
    ModelClass::NonparametricAdmg,
];

#[test]
fn every_transported_cell_is_closed_except_the_static_additive_theorem_route() {
    let theorem_cells: Vec<_> = rows()
        .iter()
        .filter(|r| r.status == CapabilityStatus::ExistingTheoremRoute)
        .map(|r| (r.quantity, r.model, r.id))
        .collect();
    assert_eq!(theorem_cells.len(), 1, "exactly one transported theorem route exists");
    let mut opened = 0;
    let mut closed = 0;
    for quantity in QUANTITIES {
        for model in MODELS {
            for conditioning in [ConditioningKind::NoEvidence, ConditioningKind::FactualEvidence] {
                for interventions in [InterventionKind::Hard, InterventionKind::Soft] {
                    for history in [
                        HistoryKind::NoHistory,
                        HistoryKind::FixedPopulationTwoSlice,
                        HistoryKind::Incompatible,
                    ] {
                        for evidence in [
                            EvidenceKind::DeclaredMechanisms,
                            EvidenceKind::SuppliedFittedIdentity,
                            EvidenceKind::DataToFit,
                        ] {
                            let request = MatrixRequest {
                                quantity,
                                model,
                                conditioning,
                                interventions,
                                population: PopulationKind::Transported,
                                history,
                                evidence,
                            };
                            let expected_open =
                                theorem_cells.iter().any(|(q, m, _)| *q == quantity && *m == model);
                            match decide(&request) {
                                Ok(row) => {
                                    assert!(expected_open, "{request:?} opened {}", row.id);
                                    assert_eq!(row.status, CapabilityStatus::ExistingTheoremRoute);
                                    assert_eq!(row.id, theorem_cells[0].2);
                                    opened += 1;
                                }
                                Err(refusal) => {
                                    assert!(!expected_open, "{request:?}");
                                    assert_eq!(refusal.code, "cell_not_licensed");
                                    assert_eq!(
                                        refusal.detail,
                                        "cross_world_execution.transported_row_closed"
                                    );
                                    assert_eq!(
                                        refusal.subject,
                                        "transported.cross_world_execution"
                                    );
                                    assert!(!refusal.remedy.is_empty());
                                    for gate in [
                                        "transport_license_missing",
                                        "target_population_abduction_receipt",
                                        "shared_exogenous_invariance_across_populations",
                                    ] {
                                        assert!(
                                            refusal.missing.iter().any(|m| m == gate),
                                            "{gate} missing from {:?}",
                                            refusal.missing
                                        );
                                    }
                                    assert!(
                                        !refusal
                                            .missing
                                            .iter()
                                            .any(|m| m.contains("fixed_population")),
                                        "the fixed-population gate is treated as met: {:?}",
                                        refusal.missing
                                    );
                                    closed += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(opened > 0 && closed > opened, "opened {opened}, closed {closed}");
}

fn scm() -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.5, 0.5] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.25, 0.5, 0.125, 0.125] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
        ],
    }
}

fn request(target: Option<&str>) -> FiniteExecutionRequest {
    let world = |n: &str, l| FiniteWorld {
        name: n.to_owned(),
        interventions: vec![(0, l)],
        routes: vec![],
    };
    FiniteExecutionRequest {
        evidence_history: "regime-1".to_owned(),
        scm: scm(),
        units: vec![FactualUnit { unit: "u".to_owned(), observed: vec![(0, 1), (1, 1)] }],
        worlds: vec![world("x0", 0), world("x1", 1)],
        quantity: FiniteQuantity::Probability(WorldAtom { world: 0, variable: 1, level: 0 }),
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: target.map(str::to_owned),
        reading: RequestedReading::Point,
    }
}

#[test]
fn a_different_target_population_refuses_before_any_work_and_the_declared_one_executes() {
    let ctx = ExecutionContext::for_tests(5);
    let mut exec = FiniteScmExecutor::new();
    for target in ["pop-b", "POP-A", "pop-a ", "", "pop-a2"] {
        let err = exec.execute(&request(Some(target)), &ctx).expect_err(target);
        let refusal = err.refusal().unwrap();
        assert_eq!(refusal.code, "cell_not_licensed", "{target:?}");
        assert_eq!(refusal.detail, "cross_world_execution.transported_row_closed", "{target:?}");
        assert!(err.missing().iter().any(|m| m == "target_population_abduction_receipt"));
        let c = exec.counters();
        assert_eq!((c.abductions, c.world_predictions, c.shared_draws), (0, 0, 0), "{target:?}");
        assert_eq!(exec.retained_len(), 0);
    }
    // The population the mechanisms were declared for is not a transport request.
    // P(Y_{x0} = 0 | X = 1, Y = 1): complier 0.5, always 0.125 consistent; complier gives 0.
    let own = exec.execute(&request(Some("pop-a")), &ctx).unwrap();
    assert!((own.value - 0.5 / 0.625).abs() < 1e-14, "{}", own.value);
    let none = exec.execute(&request(None), &ctx).unwrap();
    assert_eq!(own.value.to_bits(), none.value.to_bits());
}
