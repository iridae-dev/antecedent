//! Facade integration of scientific interface contracts: a compatible pair, a
//! pair that matches syntactically and fails scientifically, a feedback cycle,
//! export, independent inspection and consumption against a retained identity.
//!
//! The expectations are enumerated from the declarations: which dimensions of
//! two quantities differ, which operations a producer lacks, which path closes
//! the cycle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::interface_contract::{
    CalibrationRequirement, CalibrationStatus, ComponentConnection, ConsumedPort,
    ContractOperation, DependenceRequirement, InterfaceContract, ProducedPort, SupportDomain,
};
use antecedent::analysis::{
    check_interface_composition, check_interface_connection, consume_interface_contract_artifact,
    export_interface_contract, inspect_interface_contract_artifact,
};
use antecedent_contract::{DistributionMeaning, QuantityRole, ScientificQuantity};

fn quantity(population: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:ldl".into(),
        variable_name: "ldl".into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: "do(statin=1)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn estimator(population: &str, operations: &[ContractOperation]) -> InterfaceContract {
    let mut port = ProducedPort::new(
        "effect",
        quantity(population, "mg/dL"),
        "estimand:ate",
        SupportDomain::Real,
        operations.iter().copied(),
    );
    if operations.contains(&ContractOperation::Interval) {
        port = port
            .with_uncertainty(DistributionMeaning::EstimatorSampling)
            .with_calibration(CalibrationStatus::Unmeasured);
    }
    InterfaceContract { component: "estimator".into(), produces: vec![port], consumes: vec![] }
}

fn decision(population: &str, operations: &[ContractOperation]) -> InterfaceContract {
    let mut port = ConsumedPort::new(
        "effect",
        quantity(population, "mg/dL"),
        "estimand:ate",
        SupportDomain::Real,
        operations.iter().copied(),
    );
    if operations.contains(&ContractOperation::Law) {
        port = port.with_uncertainty(DistributionMeaning::InterventionalPredictive);
    }
    InterfaceContract { component: "decision".into(), produces: vec![], consumes: vec![port] }
}

#[test]
fn a_compatible_pair_is_connected_and_keeps_the_producers_label() {
    let producer = estimator("adults", &[ContractOperation::Mean, ContractOperation::Interval]);
    let consumer = decision("adults", &[ContractOperation::Mean]);
    let connection =
        check_interface_connection(&producer, "effect", &consumer, "effect").expect("connects");
    assert_eq!(connection.calibration, CalibrationStatus::Unmeasured);
    assert_eq!(connection.operations.len(), 1);
}

#[test]
fn a_pair_that_matches_syntactically_fails_scientifically() {
    // Same variable, same units, same shape: another population, and a mean offered for a law.
    let producer = estimator("trial_cohort", &[ContractOperation::Mean]);
    let consumer = decision("adults", &[ContractOperation::Law]);
    let refusals =
        check_interface_connection(&producer, "effect", &consumer, "effect").expect_err("refused");
    let details: Vec<&str> = refusals.iter().map(|r| r.detail.as_str()).collect();
    assert_eq!(
        details,
        [
            "interface_contract.population_mismatch",
            "interface_contract.operation_unavailable",
            "interface_contract.uncertainty_meaning_missing",
        ]
    );
    assert!(refusals.iter().all(|r| r.remedy.is_some()));
    assert_eq!(refusals[0].expected.as_deref(), Some("adults"));
    assert_eq!(refusals[0].supplied.as_deref(), Some("trial_cohort"));
    // The refusal carries a registered code and renders with the reason prefix.
    assert!(refusals[0].to_string().starts_with("reason=interface_contract_quantity_mismatch"));
}

#[test]
fn a_feedback_loop_is_refused_before_execution_with_the_path() {
    let stage = |name: &str| InterfaceContract {
        component: name.into(),
        consumes: vec![
            ConsumedPort::new(
                "in",
                quantity("adults", "mg/dL"),
                "estimand:ate",
                SupportDomain::Real,
                [ContractOperation::Mean],
            )
            .with_dependence(DependenceRequirement::Unconstrained)
            .with_calibration(CalibrationRequirement::NotRequired),
        ],
        produces: vec![ProducedPort::new(
            "out",
            quantity("adults", "mg/dL"),
            "estimand:ate",
            SupportDomain::Real,
            [ContractOperation::Mean],
        )],
    };
    let components = [stage("mixer"), stage("policy"), stage("plant")];
    let connections = [
        ComponentConnection::new("plant", "out", "mixer", "in"),
        ComponentConnection::new("mixer", "out", "policy", "in"),
        ComponentConnection::new("policy", "out", "plant", "in"),
    ];
    let refusals = check_interface_composition(&components, &connections).expect_err("cyclic");
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].code, "interface_contract_cyclic_composition");
    assert_eq!(refusals[0].offending.as_deref(), Some("mixer -> policy -> plant -> mixer"));
    // Without the closing edge the same components compose feed-forward.
    let composed = check_interface_composition(&components, &connections[..2]).expect("acyclic");
    assert_eq!(composed.order, ["plant", "mixer", "policy"]);
}

#[test]
fn an_exported_contract_is_inspected_and_consumed_against_a_retained_identity() {
    let contract = estimator("adults", &[ContractOperation::Mean, ContractOperation::Interval]);
    let exported = export_interface_contract(&contract, "estimator-contract").expect("exports");
    let meta = inspect_interface_contract_artifact(&exported.bytes).expect("inspects");
    assert_eq!(meta.feature, "interface_contract_v1");
    assert_eq!(meta.contract.component, "estimator");
    assert_eq!(meta.contract.produces[0].calibration.status, "unmeasured");
    assert_eq!(meta.identity, exported.identity);

    let consumed = consume_interface_contract_artifact(&exported.bytes, Some(&exported.identity))
        .expect("consumes");
    assert_eq!(consumed.contract.produces[0].name, "effect");
    assert_eq!(consumed.contract_identity, exported.contract_identity);

    // A different contract, consistently exported, is refused by the retained identity.
    let other = export_interface_contract(
        &estimator("children", &[ContractOperation::Mean, ContractOperation::Interval]),
        "estimator-contract",
    )
    .expect("exports");
    let refusal = consume_interface_contract_artifact(&other.bytes, Some(&exported.identity))
        .expect_err("another contract");
    assert_eq!(refusal.detail, "interface_contract.identity_mismatch");
    assert!(refusal.remedy.is_some());
}
