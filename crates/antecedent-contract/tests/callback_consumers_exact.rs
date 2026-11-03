//! Whether a callback's declared output feeds a consumer, decided by perturbing one declared
//! field of the consumer at a time and reading the refusals off the connection rule.
//!
//! Rule (the record's theorem): a produced port connects to a consumed port exactly when the
//! variable, units, population, regime, horizon, functional, conditioning, transform and
//! semantic identity agree, the consumer's support contains the produced support, every
//! operation the consumer needs is offered, and the consumer's uncertainty meaning (when it
//! states one) is the one produced. Every defect is reported, in the order quantity fields,
//! semantic identity, support, operations, uncertainty.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::callback_protocol::{
    CALLBACK_OUTPUT_PORT, CallbackDeclaration, CallbackDependency, CallbackKind, CallbackOperation,
    CallbackTrust, DependencyDeclaration, DependencyKind, ReplayContract,
};
use antecedent_contract::interface_contract::{
    ConsumedPort, ContractOperation, EvidenceBasis, InterfaceContract, SupportDomain,
    check_connection,
};
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityCondition, QuantityRole,
    ScientificQuantity,
};

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:systolic".into(),
        variable_name: "systolic".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "adults".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn callback(support: SupportDomain, operations: &[CallbackOperation]) -> CallbackDeclaration {
    CallbackDeclaration {
        component: "law".into(),
        kind: CallbackKind::Law,
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "law".into(),
            version_id: "1".into(),
            snapshot_id: "snap".into(),
            request_id: "req".into(),
        },
        quantity: quantity(),
        semantic_identity: "estimand:outcome_law".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support,
        operations: operations.iter().copied().collect(),
        update: None,
        inputs: Vec::new(),
        dependencies: DependencyDeclaration::Complete(vec![CallbackDependency::new(
            DependencyKind::Model,
            "parameters",
            "p",
        )]),
        replay: ReplayContract::DeterministicPure,
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

/// What the consumer states it needs.
#[derive(Clone)]
struct Need {
    quantity: ScientificQuantity,
    semantic: String,
    support: SupportDomain,
    operations: Vec<ContractOperation>,
    uncertainty: Option<DistributionMeaning>,
}

fn need() -> Need {
    Need {
        quantity: quantity(),
        semantic: "estimand:outcome_law".into(),
        support: SupportDomain::Real,
        operations: vec![ContractOperation::Draws],
        uncertainty: Some(DistributionMeaning::InterventionalPredictive),
    }
}

fn details(producer: &CallbackDeclaration, need: &Need) -> Result<(), Vec<String>> {
    let mut port = ConsumedPort::new(
        "input",
        need.quantity.clone(),
        &need.semantic,
        need.support.clone(),
        need.operations.iter().copied(),
    );
    if let Some(meaning) = need.uncertainty {
        port = port.with_uncertainty(meaning);
    }
    let consumer = InterfaceContract {
        component: "consumer".into(),
        produces: vec![],
        consumes: vec![port],
    };
    check_connection(&producer.interface_contract(), CALLBACK_OUTPUT_PORT, &consumer, "input")
        .map(|_| ())
        .map_err(|refusals| refusals.into_iter().map(|r| r.detail).collect())
}

fn full() -> CallbackDeclaration {
    callback(
        SupportDomain::Real,
        &[
            CallbackOperation::Mean,
            CallbackOperation::Sample,
            CallbackOperation::Cdf,
            CallbackOperation::Quantile,
        ],
    )
}

/// Changing exactly one declared coordinate field of the consumer refuses with exactly the one
/// matching detail, and the unchanged consumer connects.
#[test]
fn each_semantic_field_refuses_alone_and_the_matching_consumer_connects() {
    let producer = full();
    assert_eq!(details(&producer, &need()), Ok(()));
    let cases: Vec<(&str, fn(&mut Need), &str)> = vec![
        ("variable", |n| n.quantity.variable_id = "schema:diastolic".into(), "variable_mismatch"),
        ("units", |n| n.quantity.units = "kPa".into(), "units_mismatch"),
        ("population", |n| n.quantity.population_id = "children".into(), "population_mismatch"),
        ("regime", |n| n.quantity.regime_id = "do(a=0)".into(), "regime_mismatch"),
        ("horizon", |n| n.quantity.horizon = 1, "horizon_mismatch"),
        ("functional", |n| n.quantity.functional_id = "mean".into(), "functional_mismatch"),
        (
            "conditioning",
            |n| {
                n.quantity.conditioning = vec![QuantityCondition {
                    variable_id: "schema:age".into(),
                    value_id: "old".into(),
                }];
            },
            "conditioning_mismatch",
        ),
        ("transform", |n| n.quantity.transform_id = "log".into(), "transform_mismatch"),
        ("semantic identity", |n| n.semantic = "estimand:other".into(), "semantic_identity_mismatch"),
    ];
    for (label, change, expected) in cases {
        let mut asked = need();
        change(&mut asked);
        assert_eq!(
            details(&producer, &asked),
            Err(vec![format!("interface_contract.{expected}")]),
            "{label}"
        );
    }
}

/// Several defects are all reported, in the fixed order: units, population, then the operation
/// the callback does not offer (the callback offers mean, draws and law, but no joint draws).
#[test]
fn every_defect_is_reported_in_the_fixed_order() {
    let mut asked = need();
    asked.quantity.population_id = "children".into();
    asked.quantity.units = "kPa".into();
    asked.operations = vec![ContractOperation::Draws, ContractOperation::JointDraws];
    assert_eq!(
        details(&full(), &asked),
        Err(vec![
            "interface_contract.units_mismatch".to_owned(),
            "interface_contract.population_mismatch".to_owned(),
            "interface_contract.operation_unavailable".to_owned(),
        ])
    );
}

/// Offered operations follow the declared callback operations: mean -> mean, sample -> draws,
/// cdf and quantile -> law. Two declaration rules shape what a consumer can state: a port that
/// needs a law or draws must state an uncertainty meaning, and a port that needs only a mean
/// must not (`uncertainty_meaning_required` / `_unused` refuse the consumer's own declaration).
/// So a consumer wanting `Draws` or `Law` states the meaning, and a consumer wanting `Mean` none.
/// By hand the connection then refuses with `operation_unavailable` when the operation is not
/// offered, followed by `uncertainty_meaning_missing` when the consumer states a meaning and the
/// callback offers no law or draws (a mean-only callback carries no uncertainty), in that order.
#[test]
fn operations_connect_exactly_when_the_callback_offers_them() {
    use ContractOperation as Op;
    let offers: Vec<(Vec<CallbackOperation>, Vec<Op>)> = vec![
        (vec![CallbackOperation::Mean], vec![Op::Mean]),
        (vec![CallbackOperation::Sample], vec![Op::Draws]),
        (vec![CallbackOperation::Cdf], vec![Op::Law]),
        (vec![CallbackOperation::Quantile], vec![Op::Law]),
        (vec![CallbackOperation::Mean, CallbackOperation::Sample], vec![Op::Mean, Op::Draws]),
        (vec![CallbackOperation::Cdf, CallbackOperation::Quantile], vec![Op::Law]),
    ];
    for (declared, offered) in offers {
        let producer = callback(SupportDomain::Real, &declared);
        let carries_uncertainty = offered.contains(&Op::Draws) || offered.contains(&Op::Law);
        for wanted in [Op::Mean, Op::Draws, Op::Law, Op::JointDraws] {
            let mut asked = need();
            asked.operations = vec![wanted];
            let states_meaning = wanted != Op::Mean;
            if !states_meaning {
                asked.uncertainty = None;
            }
            let mut expected = Vec::new();
            if !offered.contains(&wanted) {
                expected.push("interface_contract.operation_unavailable".to_owned());
            }
            if states_meaning && !carries_uncertainty {
                expected.push("interface_contract.uncertainty_meaning_missing".to_owned());
            }
            let expected = if expected.is_empty() { Ok(()) } else { Err(expected) };
            assert_eq!(details(&producer, &asked), expected, "{declared:?} / {wanted:?}");
        }
    }
}

/// A consumer that needs draws and names an uncertainty meaning, facing a callback that carries
/// none (a mean-only callback), refuses twice: the draws are not offered and the meaning is
/// missing. A mean-only consumer that states no meaning connects. A consumer that needs only a
/// mean but states a meaning has an invalid declaration of its own (`unused`). Another meaning
/// than the one produced refuses once.
#[test]
fn the_uncertainty_meaning_the_consumer_names_must_be_the_one_produced() {
    let mean_only = callback(SupportDomain::Real, &[CallbackOperation::Mean]);
    assert_eq!(
        details(&mean_only, &need()),
        Err(vec![
            "interface_contract.operation_unavailable".to_owned(),
            "interface_contract.uncertainty_meaning_missing".to_owned(),
        ])
    );
    let mut mean_consumer = need();
    mean_consumer.operations = vec![ContractOperation::Mean];
    assert_eq!(
        details(&mean_only, &mean_consumer),
        Err(vec!["interface_contract.uncertainty_meaning_unused".to_owned()])
    );
    mean_consumer.uncertainty = None;
    assert_eq!(details(&mean_only, &mean_consumer), Ok(()));

    for other in [
        DistributionMeaning::ParameterPosterior,
        DistributionMeaning::PosteriorPredictive,
        DistributionMeaning::EstimatorSampling,
    ] {
        let mut asked = need();
        asked.uncertainty = Some(other);
        let refusals = details(&full(), &asked).unwrap_err();
        assert_eq!(refusals.len(), 1, "{other:?}");
        assert!(refusals[0].starts_with("interface_contract.uncertainty_"), "{refusals:?}");
    }
}

/// Support containment: the consumer's domain must contain the produced one. A real consumer
/// accepts every produced domain; a non-negative consumer only non-negative or binary ones; a
/// binary consumer only binary; an interval consumer an inner interval; an interval is never
/// widened.
#[test]
fn the_consumers_support_must_contain_the_callbacks_support() {
    use SupportDomain as S;
    let produced = [
        S::Real,
        S::NonNegative,
        S::Binary,
        S::interval(0.0, 1.0),
        S::interval(-1.0, 3.0),
    ];
    // Hand-tabulated acceptance: rows are the consumer's domain, columns the produced domains
    // in the order above.
    let consumers: Vec<(S, [bool; 5])> = vec![
        (S::Real, [true, true, true, true, true]),
        (S::NonNegative, [false, true, true, true, false]),
        (S::Binary, [false, false, true, false, false]),
        (S::interval(0.0, 1.0), [false, false, true, true, false]),
        (S::interval(-2.0, 4.0), [false, false, true, true, true]),
    ];
    for (accepting, row) in consumers {
        for (offered, accepted) in produced.iter().zip(row) {
            let producer = callback(offered.clone(), &[CallbackOperation::Sample]);
            let mut asked = need();
            asked.support = accepting.clone();
            let expected = if accepted {
                Ok(())
            } else {
                Err(vec!["interface_contract.support_mismatch".to_owned()])
            };
            assert_eq!(details(&producer, &asked), expected, "{accepting:?} <- {offered:?}");
        }
    }
}
