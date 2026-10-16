//! Interface contracts: what is decided from declarations alone, before any
//! number is computed.
//!
//! Every expectation is a hand-enumerated outcome of the rule it names: which
//! dimensions of two quantities differ, which operations a producer lacks,
//! which path closes a cycle.

use std::collections::BTreeSet;

use antecedent_contract::evidence_ledger::SourceId;
use antecedent_contract::interface_contract::{
    CalibrationRequirement, CalibrationStatus, ComponentConnection, ConsumedPort,
    ContractOperation, DependenceRequirement, InterfaceContract, ProducedPort, SupportDomain,
    UncertaintyParadigm, check_composition, check_connection, operations_of_capabilities,
};
use antecedent_contract::{
    DistributionMeaning, ExternalCapability, ExternalRefusal, QuantityRole, ScientificQuantity,
};
use antecedent_core::SemanticDigest;

const MODEL_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const MODEL_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn source(byte: u8) -> SourceId {
    SourceId::from_digest(SemanticDigest::from_bytes([byte; 32]))
}

fn quantity(units: &str, population: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:systolic".into(),
        variable_name: "systolic".into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn produced(operations: &[ContractOperation]) -> ProducedPort {
    ProducedPort::new(
        "effect",
        quantity("mmHg", "adults", "do(a=1)"),
        "estimand:ate",
        SupportDomain::Real,
        operations.iter().copied(),
    )
}

fn consumed(operations: &[ContractOperation]) -> ConsumedPort {
    ConsumedPort::new(
        "effect",
        quantity("mmHg", "adults", "do(a=1)"),
        "estimand:ate",
        SupportDomain::Real,
        operations.iter().copied(),
    )
}

fn producer_of(port: ProducedPort) -> InterfaceContract {
    InterfaceContract { component: "estimator".into(), produces: vec![port], consumes: vec![] }
}

fn consumer_of(port: ConsumedPort) -> InterfaceContract {
    InterfaceContract { component: "decision".into(), produces: vec![], consumes: vec![port] }
}

fn details(refusals: &[ExternalRefusal]) -> Vec<&str> {
    refusals.iter().map(|r| r.detail.as_str()).collect()
}

fn check(
    producer: &InterfaceContract,
    consumer: &InterfaceContract,
) -> Result<(), Vec<ExternalRefusal>> {
    let produced = producer.produces[0].name.as_str();
    let consumed = consumer.consumes[0].name.as_str();
    check_connection(producer, produced, consumer, consumed).map(|_| ())
}

#[test]
fn compatible_pair_connects_and_carries_what_the_producer_holds() {
    let producer = producer_of(
        produced(&[
            ContractOperation::Point,
            ContractOperation::Mean,
            ContractOperation::Interval,
            ContractOperation::Posterior,
        ])
        .with_uncertainty(DistributionMeaning::CausalFunctionalPosterior)
        .with_calibration(CalibrationStatus::Measured { record_id: "cal-0001".into() })
        .with_measurement_models([MODEL_A])
        .with_evidence([source(1), source(2)])
        .with_provenance(["causal_contract", "claim"]),
    );
    let consumer = consumer_of(
        consumed(&[ContractOperation::Mean, ContractOperation::Posterior])
            .with_uncertainty(DistributionMeaning::CausalFunctionalPosterior)
            .with_calibration(CalibrationRequirement::Measured)
            .with_measurement_models([MODEL_A])
            .with_dependence(DependenceRequirement::DeclaredEvidence)
            .with_provenance(["causal_contract"]),
    );
    let connection = check_connection(&producer, "effect", &consumer, "effect").expect("connects");
    assert_eq!(
        connection.operations,
        BTreeSet::from([ContractOperation::Mean, ContractOperation::Posterior])
    );
    assert_eq!(connection.calibration.label(), "measured");
    assert_eq!(
        connection.calibration,
        CalibrationStatus::Measured { record_id: "cal-0001".into() }
    );
    assert_eq!(
        connection
            .uncertainty
            .map(antecedent_contract::interface_contract::UncertaintyMeaning::paradigm),
        Some(UncertaintyParadigm::Posterior)
    );
    // The same declarations decide the same connection, with the same identity.
    let again = check_connection(&producer, "effect", &consumer, "effect").expect("connects");
    assert_eq!(connection.identity, again.identity);
}

#[test]
fn matching_names_units_and_shape_but_another_population_and_a_mean_for_a_law_refuse() {
    // Same variable name, same units, same dtype and shape: only the population
    // differs, and the producer offers a mean where the consumer needs a law.
    let producer = producer_of(ProducedPort::new(
        "effect",
        quantity("mmHg", "trial_cohort", "do(a=1)"),
        "estimand:ate",
        SupportDomain::Real,
        [ContractOperation::Point, ContractOperation::Mean],
    ));
    let consumer = consumer_of(
        consumed(&[ContractOperation::Law])
            .with_uncertainty(DistributionMeaning::InterventionalPredictive),
    );
    let refusals = check(&producer, &consumer).expect_err("scientifically incompatible");
    let codes: Vec<&str> = refusals.iter().map(|r| r.code).collect();
    assert_eq!(
        details(&refusals),
        [
            "interface_contract.population_mismatch",
            "interface_contract.operation_unavailable",
            "interface_contract.uncertainty_meaning_missing",
        ]
    );
    assert_eq!(
        codes,
        [
            "interface_contract_quantity_mismatch",
            "interface_contract_operation_unavailable",
            "interface_contract_uncertainty_mismatch",
        ]
    );
    let population = &refusals[0];
    assert_eq!(population.expected.as_deref(), Some("adults"));
    assert_eq!(population.supplied.as_deref(), Some("trial_cohort"));
    let operation = &refusals[1];
    assert_eq!(operation.offending.as_deref(), Some("law"));
    assert_eq!(operation.supplied.as_deref(), Some("point,mean"));
    for refusal in &refusals {
        assert_eq!(refusal.validate(), Ok(()), "{refusal:?}");
        assert!(refusal.remedy.is_some_and(|text| !text.is_empty()));
    }
}

#[test]
fn every_quantity_dimension_is_reported_separately() {
    let mut other = quantity("lb", "children", "observational");
    other.horizon = 2;
    other.functional_id = "risk".into();
    other.transform_id = "log".into();
    other.variable_id = "schema:diastolic".into();
    let producer = producer_of(ProducedPort::new(
        "effect",
        other,
        "estimand:att",
        SupportDomain::NonNegative,
        [ContractOperation::Mean],
    ));
    let consumer = consumer_of(ConsumedPort::new(
        "effect",
        quantity("mmHg", "adults", "do(a=1)"),
        "estimand:ate",
        SupportDomain::Binary,
        [ContractOperation::Mean],
    ));
    let refusals = check(&producer, &consumer).expect_err("refused");
    assert_eq!(
        details(&refusals),
        [
            "interface_contract.variable_mismatch",
            "interface_contract.units_mismatch",
            "interface_contract.population_mismatch",
            "interface_contract.regime_mismatch",
            "interface_contract.horizon_mismatch",
            "interface_contract.functional_mismatch",
            "interface_contract.transform_mismatch",
            "interface_contract.semantic_identity_mismatch",
            "interface_contract.support_mismatch",
        ]
    );
}

#[test]
fn support_containment_is_decided_by_declared_domains() {
    let unit = SupportDomain::interval(0.0, 1.0);
    // Producer domain inside the consumer's domain connects.
    assert!(SupportDomain::Real.accepts(&unit));
    assert!(SupportDomain::NonNegative.accepts(&unit));
    assert!(SupportDomain::NonNegative.accepts(&SupportDomain::Binary));
    assert!(unit.accepts(&SupportDomain::Binary));
    assert!(SupportDomain::interval(-1.0, 2.0).accepts(&unit));
    // A wider producer domain does not.
    assert!(!unit.accepts(&SupportDomain::Real));
    assert!(!unit.accepts(&SupportDomain::NonNegative));
    assert!(!SupportDomain::Binary.accepts(&unit));
    assert!(!SupportDomain::interval(-1.0, 0.5).accepts(&unit));
    assert!(!SupportDomain::NonNegative.accepts(&SupportDomain::interval(-0.5, 1.0)));
    let a_b = SupportDomain::Categorical { levels: vec!["a".into(), "b".into()] };
    let a = SupportDomain::Categorical { levels: vec!["a".into()] };
    assert!(a_b.accepts(&a));
    assert!(!a.accepts(&a_b));
    assert!(!SupportDomain::Real.accepts(&a));
    assert!(!SupportDomain::interval(1.0, 0.0).is_well_formed());
    assert!(!SupportDomain::interval(f64::NAN, 1.0).is_well_formed());
}

#[test]
fn measurement_assumptions_must_agree_as_sets_of_model_identities() {
    let base = produced(&[ContractOperation::Mean]);
    let want_a = consumed(&[ContractOperation::Mean]).with_measurement_models([MODEL_A]);
    let producer_b = producer_of(base.clone().with_measurement_models([MODEL_B]));
    let refusals = check(&producer_b, &consumer_of(want_a.clone())).expect_err("model differs");
    assert_eq!(details(&refusals), ["interface_contract.measurement_assumption_mismatch"]);
    assert_eq!(refusals[0].expected.as_deref(), Some(MODEL_A));
    assert_eq!(refusals[0].supplied.as_deref(), Some(MODEL_B));
    // An uncorrected producer does not satisfy a consumer that needs the model.
    assert!(check(&producer_of(base.clone()), &consumer_of(want_a.clone())).is_err());
    // And an equal set connects.
    assert!(
        check(&producer_of(base.with_measurement_models([MODEL_A])), &consumer_of(want_a)).is_ok()
    );
}

#[test]
fn undeclared_dependence_is_refused_where_the_consumer_needs_it_declared() {
    let producer = producer_of(produced(&[ContractOperation::Mean]));
    let consumer = consumer_of(
        consumed(&[ContractOperation::Mean])
            .with_dependence(DependenceRequirement::DeclaredEvidence),
    );
    let refusals = check(&producer, &consumer).expect_err("dependence unknown");
    assert_eq!(details(&refusals), ["interface_contract.dependence_unknown"]);
    assert_eq!(refusals[0].code, "interface_contract_assumption_mismatch");
    let declared = producer_of(produced(&[ContractOperation::Mean]).with_evidence([source(7)]));
    assert!(check(&declared, &consumer).is_ok());
}

#[test]
fn posterior_and_sampling_uncertainty_are_not_interchangeable() {
    // The consumer needs a posterior; the producer offers a sampling interval.
    let producer = producer_of(
        produced(&[ContractOperation::Interval])
            .with_uncertainty(DistributionMeaning::EstimatorSampling),
    );
    let consumer = consumer_of(
        consumed(&[ContractOperation::Posterior])
            .with_uncertainty(DistributionMeaning::CausalFunctionalPosterior),
    );
    let refusals = check(&producer, &consumer).expect_err("refused");
    assert_eq!(
        details(&refusals),
        [
            "interface_contract.operation_unavailable",
            "interface_contract.uncertainty_paradigm_mismatch",
        ]
    );
    assert_eq!(refusals[1].expected.as_deref(), Some("posterior"));
    assert_eq!(refusals[1].supplied.as_deref(), Some("sampling"));

    // Within one paradigm, a posterior over parameters is not one over the functional.
    let parameters = producer_of(
        produced(&[ContractOperation::Posterior])
            .with_uncertainty(DistributionMeaning::ParameterPosterior),
    );
    let refusals = check(&parameters, &consumer).expect_err("meaning differs");
    assert_eq!(details(&refusals), ["interface_contract.uncertainty_meaning_mismatch"]);
    assert_eq!(refusals[0].expected.as_deref(), Some("causal_functional_posterior"));
    assert_eq!(refusals[0].supplied.as_deref(), Some("parameter_posterior"));
}

#[test]
fn measured_calibration_is_required_not_assumed() {
    let consumer = consumer_of(
        consumed(&[ContractOperation::Interval])
            .with_uncertainty(DistributionMeaning::EstimatorSampling)
            .with_calibration(CalibrationRequirement::Measured),
    );
    for status in [CalibrationStatus::None, CalibrationStatus::Unmeasured] {
        let label = status.label();
        let producer = producer_of(
            produced(&[ContractOperation::Interval])
                .with_uncertainty(DistributionMeaning::EstimatorSampling)
                .with_calibration(status),
        );
        let refusals = check(&producer, &consumer).expect_err("not held");
        assert_eq!(details(&refusals), ["interface_contract.calibration_not_held"]);
        assert_eq!(refusals[0].code, "interface_contract_calibration_not_held");
        assert_eq!(refusals[0].supplied.as_deref(), Some(label));
    }
    // An inherited label cannot be resolved from a pair.
    let inheriting = InterfaceContract {
        component: "estimator".into(),
        consumes: vec![consumed(&[ContractOperation::Mean])],
        produces: vec![
            produced(&[ContractOperation::Interval])
                .with_uncertainty(DistributionMeaning::EstimatorSampling)
                .with_calibration(CalibrationStatus::Inherited { input: "effect".into() }),
        ],
    };
    let refusals = check(&inheriting, &consumer).expect_err("unresolved");
    assert_eq!(details(&refusals), ["interface_contract.calibration_inherited_unresolved"]);
    // A producer that holds a measured record connects, and the connection keeps its label.
    let measured = producer_of(
        produced(&[ContractOperation::Interval])
            .with_uncertainty(DistributionMeaning::EstimatorSampling)
            .with_calibration(CalibrationStatus::Measured { record_id: "rec".into() }),
    );
    assert!(check(&measured, &consumer).is_ok());
}

#[test]
fn point_only_stays_point_only_across_a_connection() {
    let producer = producer_of(
        produced(&[ContractOperation::Point, ContractOperation::Mean])
            .with_calibration(CalibrationStatus::PointOnly),
    );
    let consumer = consumer_of(consumed(&[ContractOperation::Mean]));
    let connection = check_connection(&producer, "effect", &consumer, "effect").expect("connects");
    assert_eq!(connection.calibration, CalibrationStatus::PointOnly);
    assert!(connection.uncertainty.is_none());
}

#[test]
fn malformed_declarations_are_refused_with_their_own_detail() {
    let valid = produced(&[ContractOperation::Mean]);
    let mut contract = producer_of(valid.clone());
    assert_eq!(contract.validate(), Ok(()));

    let mut cases: Vec<(InterfaceContract, &str)> = Vec::new();
    // An interval with no stated meaning.
    contract.produces[0].operations.insert(ContractOperation::Interval);
    cases.push((contract.clone(), "interface_contract.uncertainty_meaning_required"));
    // A meaning stated where nothing carries uncertainty.
    cases.push((
        producer_of(valid.clone().with_uncertainty(DistributionMeaning::Bootstrap)),
        "interface_contract.uncertainty_meaning_unused",
    ));
    // A posterior operation with a sampling meaning.
    cases.push((
        producer_of(
            produced(&[ContractOperation::Posterior])
                .with_uncertainty(DistributionMeaning::EstimatorSampling),
        ),
        "interface_contract.posterior_without_posterior_meaning",
    ));
    // Point-only beside an interval.
    cases.push((
        producer_of(
            produced(&[ContractOperation::Interval])
                .with_uncertainty(DistributionMeaning::Bootstrap)
                .with_calibration(CalibrationStatus::PointOnly),
        ),
        "interface_contract.point_only_with_uncertainty",
    ));
    // Measured calibration with no record id, and with no interval operation.
    cases.push((
        producer_of(
            produced(&[ContractOperation::Interval])
                .with_uncertainty(DistributionMeaning::Bootstrap)
                .with_calibration(CalibrationStatus::Measured { record_id: " ".into() }),
        ),
        "interface_contract.calibration_record_blank",
    ));
    cases.push((
        producer_of(
            valid.clone().with_calibration(CalibrationStatus::Measured { record_id: "r".into() }),
        ),
        "interface_contract.calibration_without_interval",
    ));
    // Inheritance from a port the component does not consume.
    cases.push((
        producer_of(
            valid
                .clone()
                .with_calibration(CalibrationStatus::Inherited { input: "missing".into() }),
        ),
        "interface_contract.inherited_input_unknown",
    ));
    // An identity that is not a lowercase hex digest.
    cases.push((
        producer_of(valid.clone().with_measurement_models(["not-a-digest"])),
        "interface_contract.invalid_digest",
    ));
    cases.push((
        producer_of(valid.clone().with_provenance(["not_a_stage"])),
        "interface_contract.unknown_provenance_stage",
    ));
    // A quantity with no units, an unordered interval, no operations.
    let mut no_units = valid.clone();
    no_units.quantity.units.clear();
    cases.push((producer_of(no_units), "interface_contract.invalid_quantity"));
    let mut backwards = valid.clone();
    backwards.support = SupportDomain::interval(2.0, 1.0);
    cases.push((producer_of(backwards), "interface_contract.invalid_support"));
    cases.push((
        producer_of(ProducedPort::new(
            "effect",
            quantity("mmHg", "adults", "do(a=1)"),
            "estimand:ate",
            SupportDomain::Real,
            [],
        )),
        "interface_contract.missing_operations",
    ));
    cases.push((
        InterfaceContract {
            component: "c".into(),
            produces: vec![valid.clone(), valid],
            consumes: vec![],
        },
        "interface_contract.duplicate_port",
    ));
    for (contract, detail) in cases {
        let refusal = contract.validate().expect_err(detail);
        assert_eq!(refusal.detail, detail);
        assert_eq!(refusal.code, "interface_contract_invalid");
        assert_eq!(refusal.validate(), Ok(()));
        // A malformed declaration is refused alone, before any comparison.
        let consumer = consumer_of(consumed(&[ContractOperation::Mean]));
        let refusals = check_connection(&contract, "effect", &consumer, "effect")
            .expect_err("malformed producer");
        assert_eq!(refusals.len(), 1);
        assert_eq!(refusals[0].code, "interface_contract_invalid");
    }
}

#[test]
fn identity_ignores_port_order_and_binds_every_declared_field() {
    let a = produced(&[ContractOperation::Mean]);
    let mut b = a.clone();
    b.name = "other".into();
    let forward = InterfaceContract {
        component: "c".into(),
        produces: vec![a.clone(), b.clone()],
        consumes: vec![],
    };
    let reversed =
        InterfaceContract { component: "c".into(), produces: vec![b, a.clone()], consumes: vec![] };
    assert_eq!(forward.canonical_identity(), reversed.canonical_identity());

    let base = producer_of(a.clone()).canonical_identity();
    let mut changes: Vec<ProducedPort> = Vec::new();
    let mut units = a.clone();
    units.quantity.units = "lb".into();
    changes.push(units);
    let mut population = a.clone();
    population.quantity.population_id = "children".into();
    changes.push(population);
    let mut label = a.clone();
    label.quantity.variable_name = "renamed".into();
    changes.push(label);
    changes.push(a.clone().with_evidence([source(3)]));
    changes.push(a.clone().with_measurement_models([MODEL_A]));
    changes.push(a.clone().with_calibration(CalibrationStatus::Unmeasured));
    let mut support = a.clone();
    support.support = SupportDomain::NonNegative;
    changes.push(support);
    let mut ops = a;
    ops.operations.insert(ContractOperation::Point);
    changes.push(ops);
    let mut seen = BTreeSet::from([base]);
    for change in changes {
        assert!(
            seen.insert(producer_of(change).canonical_identity()),
            "every change moves the identity to a new value"
        );
    }
    assert_eq!(producer_of(produced(&[ContractOperation::Mean])).canonical_identity(), base);
}

fn passthrough(name: &str, calibration: Option<CalibrationStatus>) -> InterfaceContract {
    let mut output = ProducedPort::new(
        "out",
        quantity("mmHg", "adults", "do(a=1)"),
        "estimand:ate",
        SupportDomain::Real,
        [ContractOperation::Mean],
    );
    if let Some(status) = calibration {
        output = output.with_calibration(status);
    }
    InterfaceContract {
        component: name.into(),
        produces: vec![output],
        consumes: vec![ConsumedPort::new(
            "in",
            quantity("mmHg", "adults", "do(a=1)"),
            "estimand:ate",
            SupportDomain::Real,
            [ContractOperation::Mean],
        )],
    }
}

fn link(from: &str, to: &str) -> ComponentConnection {
    ComponentConnection::new(from, "out", to, "in")
}

#[test]
fn a_feed_forward_composition_is_ordered_producers_first() {
    let components = [passthrough("c", None), passthrough("a", None), passthrough("b", None)];
    let composed = check_composition(&components, &[link("a", "b"), link("b", "c")]).expect("ok");
    assert_eq!(composed.order, ["a", "b", "c"]);
    assert_eq!(composed.connections.len(), 2);
    // Declaration order of components and connections does not move the identity.
    let shuffled = [passthrough("b", None), passthrough("c", None), passthrough("a", None)];
    let again = check_composition(&shuffled, &[link("b", "c"), link("a", "b")]).expect("ok");
    assert_eq!(composed.identity, again.identity);
    assert_eq!(composed.order, again.order);
}

#[test]
fn a_cycle_is_refused_before_execution_with_its_path_named() {
    let components = [passthrough("a", None), passthrough("b", None), passthrough("c", None)];
    let refusals =
        check_composition(&components, &[link("a", "b"), link("b", "c"), link("c", "a")])
            .expect_err("cyclic");
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].code, "interface_contract_cyclic_composition");
    assert_eq!(refusals[0].detail, "interface_contract.cyclic_composition");
    assert_eq!(refusals[0].offending.as_deref(), Some("a -> b -> c -> a"));
    assert!(refusals[0].remedy.is_some());
    assert_eq!(refusals[0].validate(), Ok(()));

    // A self-connection is the smallest cycle.
    let refusals =
        check_composition(&[passthrough("a", None)], &[link("a", "a")]).expect_err("self feedback");
    assert_eq!(refusals[0].offending.as_deref(), Some("a -> a"));

    // A component downstream of a cycle is not named in the path.
    let components = [passthrough("a", None), passthrough("b", None), passthrough("tail", None)];
    let refusals =
        check_composition(&components, &[link("a", "b"), link("b", "a"), link("b", "tail")])
            .expect_err("cyclic");
    assert_eq!(refusals[0].offending.as_deref(), Some("a -> b -> a"));
}

#[test]
fn a_composition_with_a_pairwise_defect_is_refused_with_the_defect() {
    let mut downstream = passthrough("b", None);
    downstream.consumes[0].quantity.units = "lb".into();
    let refusals = check_composition(&[passthrough("a", None), downstream], &[link("a", "b")])
        .expect_err("units");
    assert_eq!(details(&refusals), ["interface_contract.units_mismatch"]);
}

#[test]
fn unknown_names_and_a_doubly_fed_port_are_refused() {
    let components = [passthrough("a", None), passthrough("b", None), passthrough("c", None)];
    let refusals = check_composition(&components, &[link("a", "zzz")]).expect_err("unknown");
    assert_eq!(details(&refusals), ["interface_contract.unknown_component"]);
    let refusals =
        check_composition(&components, &[ComponentConnection::new("a", "nope", "b", "in")])
            .expect_err("unknown port");
    assert_eq!(details(&refusals), ["interface_contract.unknown_port"]);
    let refusals =
        check_composition(&components, &[link("a", "c"), link("b", "c")]).expect_err("fed twice");
    assert_eq!(details(&refusals), ["interface_contract.port_multiply_fed"]);
    assert_eq!(refusals[0].offending.as_deref(), Some("c.in"));
    let refusals = check_composition(&[passthrough("a", None), passthrough("a", None)], &[])
        .expect_err("duplicate");
    assert_eq!(details(&refusals), ["interface_contract.duplicate_component"]);
}

fn fan_in(left: u8, right: u8) -> Vec<InterfaceContract> {
    let source_of = |name: &str, byte: u8| {
        let mut contract = passthrough(name, None);
        contract.produces[0] = contract.produces[0].clone().with_evidence([source(byte)]);
        contract.consumes.clear();
        contract
    };
    let joint = InterfaceContract {
        component: "pool".into(),
        produces: vec![],
        consumes: ["left", "right"]
            .iter()
            .map(|name| {
                ConsumedPort::new(
                    *name,
                    quantity("mmHg", "adults", "do(a=1)"),
                    "estimand:ate",
                    SupportDomain::Real,
                    [ContractOperation::Mean],
                )
                .with_dependence(DependenceRequirement::DisjointEvidence)
            })
            .collect(),
    };
    vec![source_of("s1", left), source_of("s2", right), joint]
}

#[test]
fn inputs_that_must_be_disjoint_cannot_share_an_evidence_source() {
    let wires = [
        ComponentConnection::new("s1", "out", "pool", "left"),
        ComponentConnection::new("s2", "out", "pool", "right"),
    ];
    // Distinct sources: accepted.
    assert!(check_composition(&fan_in(1, 2), &wires).is_ok());
    // One source reaching the pool twice: refused, with the count of shared sources.
    let refusals = check_composition(&fan_in(5, 5), &wires).expect_err("double counted");
    assert_eq!(details(&refusals), ["interface_contract.shared_evidence"]);
    assert_eq!(refusals[0].code, "interface_contract_assumption_mismatch");
    assert_eq!(refusals[0].supplied.as_deref(), Some("1 shared source(s)"));
    assert_eq!(refusals[0].offending.as_deref(), Some("s1.out, s2.out"));
}

#[test]
fn inherited_calibration_resolves_against_the_upstream_that_holds_it() {
    let measured = |record: &str| {
        let mut contract = passthrough("a", None);
        contract.consumes.clear();
        contract.produces = vec![
            ProducedPort::new(
                "out",
                quantity("mmHg", "adults", "do(a=1)"),
                "estimand:ate",
                SupportDomain::Real,
                [ContractOperation::Interval],
            )
            .with_uncertainty(DistributionMeaning::EstimatorSampling)
            .with_calibration(CalibrationStatus::Measured { record_id: record.into() }),
        ];
        contract
    };
    let interval_in = |name: &str, require: bool| {
        let mut port = ConsumedPort::new(
            "in",
            quantity("mmHg", "adults", "do(a=1)"),
            "estimand:ate",
            SupportDomain::Real,
            [ContractOperation::Interval],
        )
        .with_uncertainty(DistributionMeaning::EstimatorSampling);
        if require {
            port = port.with_calibration(CalibrationRequirement::Measured);
        }
        let mut contract = passthrough(name, None);
        contract.consumes = vec![port];
        contract.produces = vec![
            ProducedPort::new(
                "out",
                quantity("mmHg", "adults", "do(a=1)"),
                "estimand:ate",
                SupportDomain::Real,
                [ContractOperation::Interval],
            )
            .with_uncertainty(DistributionMeaning::EstimatorSampling)
            .with_calibration(CalibrationStatus::Inherited { input: "in".into() }),
        ];
        contract
    };
    let sink = {
        let mut contract = interval_in("sink", true);
        contract.produces.clear();
        contract
    };
    let wires = [link("a", "mid"), link("mid", "sink")];
    // Measured upstream: the inherited output resolves to the same record.
    let composed =
        check_composition(&[measured("rec-7"), interval_in("mid", true), sink.clone()], &wires)
            .expect("holds");
    let mid_to_sink = composed.connections.iter().find(|c| c.consumer == "sink").expect("edge");
    assert_eq!(mid_to_sink.calibration, CalibrationStatus::Measured { record_id: "rec-7".into() });

    // An upstream that no longer holds a measured record: the inheritance is refused,
    // and the downstream that relies on it is refused too.
    let mut unmeasured = measured("rec-7");
    unmeasured.produces[0].calibration = CalibrationStatus::Unmeasured;
    let refusals = check_composition(&[unmeasured, interval_in("mid", false), sink], &wires)
        .expect_err("not held");
    assert_eq!(
        details(&refusals),
        [
            "interface_contract.calibration_inherited_not_held",
            "interface_contract.calibration_not_held",
        ]
    );
    assert_eq!(refusals[0].offending.as_deref(), Some("out <- in"));
    assert!(refusals.iter().all(|r| r.code == "interface_contract_calibration_not_held"));
}

#[test]
fn provider_capabilities_disclose_operations_one_to_one() {
    let operations = operations_of_capabilities(&[
        ExternalCapability::Mean,
        ExternalCapability::Cdf,
        ExternalCapability::Quantile,
        ExternalCapability::Sample,
        ExternalCapability::Conditional,
    ]);
    assert_eq!(
        operations,
        BTreeSet::from([ContractOperation::Mean, ContractOperation::Law, ContractOperation::Draws])
    );
    // A mean does not disclose a law, and a law does not disclose draws.
    assert!(
        !operations_of_capabilities(&[ExternalCapability::Mean]).contains(&ContractOperation::Law)
    );
    assert!(
        !operations_of_capabilities(&[ExternalCapability::Cdf]).contains(&ContractOperation::Draws)
    );
}
