//! Mechanisms, bridges and measurement models as instances of the one
//! interface-contract schema, decided by the one connection check; and the
//! evidence-ledger overlap that replaces identity-string equality wherever a
//! dependence is judged.
//!
//! Every expectation is a hand-enumerated outcome of the rule it names. A
//! connection is decided from declarations alone: no number is computed.

use std::collections::BTreeSet;

use antecedent_contract::bridge::{
    BRIDGE_SOURCE_PORT, BRIDGE_TARGET_PORT, BridgeAssumption, BridgeContract, BridgeEndpoint,
    BridgeInterfaceBasis, BridgeKind, BridgeMapping, BridgeScope, QuantityClass, ValueKind,
};
use antecedent_contract::callback_protocol::{
    CALLBACK_OUTPUT_PORT, CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind,
    CallbackOperation, CallbackTrust, DependencyDeclaration, DependencyKind, InputSupplier,
    ReplayContract,
};
use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_contract::interface_contract::{
    CalibrationStatus, ComponentConnection, ConsumedPort, ContractOperation, DependenceRequirement,
    EvidenceBasis, InterfaceContract, PortFacts, SupportDomain, check_composition,
    check_composition_with_ledger, disjoint_requirements_without_ledger,
};
use antecedent_contract::measurement::{
    MEASUREMENT_CONSTRUCT_PORT, MEASUREMENT_INDICATOR_PORT, MeasurementInterfaceBasis,
    measurement_interface_contract,
};
use antecedent_contract::{
    DependenceDeclaration, DistributionMeaning, ExternalRefusal, InputAlignment, InputRole,
    MECHANISM_OUTPUT_PORT, MechanismCompositionDeclaration, MechanismDeclaration,
    MechanismInterfaceBasis, MechanismOriginKind, PropagationDependence, PropagationInput,
    PropagationOperation, PropagationRequest, ProviderObjectIdentity, QuantityRole,
    RequestedReading, ScientificQuantity, check_declared_independence_against_ledger,
    check_propagation, check_propagation_with_ledger, check_rows_pairable_across,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};

const MODEL_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn q(variable: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: population.into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn observation(row: u8) -> ObservationId {
    ObservationId::from_key("trial", &[row])
}

fn observed(ledger: &mut EvidenceLedger, label: &str, rows: &[u8]) -> SourceId {
    ledger
        .add_source(
            SourceDeclaration::new(label, SourceKind::ObservationSet)
                .with_observations(rows.iter().copied().map(observation)),
        )
        .expect("source registers")
}

fn undeclared(ledger: &mut EvidenceLedger, label: &str) -> SourceId {
    ledger
        .add_source(SourceDeclaration::new(label, SourceKind::ObservationSet))
        .expect("source registers")
}

fn details(refusals: &[ExternalRefusal]) -> Vec<&str> {
    refusals.iter().map(|r| r.detail.as_str()).collect()
}

// -- the chain: mechanism -> bridge -> callback ---------------------------------------------------

fn mechanism(evidence: SourceId, population: &str) -> MechanismDeclaration {
    MechanismDeclaration {
        output: q("schema:y", population),
        parents: vec![q("schema:w", population)],
        conditioning_variable_ids: vec!["schema:w".into()],
        origin: MechanismOriginKind::Fitted,
        artifact_id: "fit-y".into(),
        evidence_id: evidence.to_hex(),
        latent_causes: Vec::new(),
    }
}

fn mechanism_contract(evidence: SourceId, population: &str) -> InterfaceContract {
    let basis = MechanismInterfaceBasis::new(
        PortFacts::new(SupportDomain::Real),
        [ContractOperation::Mean],
    );
    mechanism(evidence, population).interface_contract(&basis)
}

fn bridge(source_population: &str) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::LatentMeasuredOutcome,
        source: BridgeEndpoint {
            quantity: q("schema:y", source_population),
            class: QuantityClass::MeasuredOutcome,
        },
        target: BridgeEndpoint {
            quantity: q("schema:z", "adults"),
            class: QuantityClass::LatentConstruct,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope {
            population_id: "adults".into(),
            regime_id: "observational".into(),
        },
        mapping: BridgeMapping::Identity,
        assumptions: Vec::<BridgeAssumption>::new(),
        transport: None,
    }
}

fn bridge_contract(source_population: &str, evidence: SourceId) -> InterfaceContract {
    let basis = BridgeInterfaceBasis {
        source: PortFacts::new(SupportDomain::Real),
        target: PortFacts::new(SupportDomain::Real),
        calibration: None,
        evidence: EvidenceBasis::Declared(BTreeSet::from([evidence])),
    };
    bridge(source_population).interface_contract("bridge:y_to_z", &basis)
}

fn callback() -> CallbackDeclaration {
    CallbackDeclaration {
        component: "outcome_model".into(),
        kind: CallbackKind::Law,
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "outcome".into(),
            version_id: "1.0".into(),
            snapshot_id: "snap-1".into(),
            request_id: "req-1".into(),
        },
        quantity: q("schema:w", "adults"),
        semantic_identity: "schema:w".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: [CallbackOperation::Mean].into_iter().collect(),
        update: None,
        inputs: vec![CallbackInput {
            name: "schema:z".into(),
            quantity: q("schema:z", "adults"),
            semantic_identity: "schema:z".into(),
            support: SupportDomain::Real,
            operation: CallbackOperation::Mean,
            uncertainty: None,
            supplier: InputSupplier::External,
        }],
        dependencies: DependencyDeclaration::Complete(vec![CallbackDependency::new(
            DependencyKind::Model,
            "parameters",
            "mu=2",
        )]),
        replay: ReplayContract::DeterministicPure,
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

fn feed_forward() -> Vec<ComponentConnection> {
    vec![
        ComponentConnection::new(
            "mechanism:schema:y",
            MECHANISM_OUTPUT_PORT,
            "bridge:y_to_z",
            BRIDGE_SOURCE_PORT,
        ),
        ComponentConnection::new("bridge:y_to_z", BRIDGE_TARGET_PORT, "outcome_model", "schema:z"),
    ]
}

#[test]
fn a_mechanism_a_bridge_and_a_callback_compose_through_the_one_check() {
    let mut ledger = EvidenceLedger::new();
    let evidence = observed(&mut ledger, "fit-y", &[1, 2, 3]);
    let components = [
        mechanism_contract(evidence, "adults"),
        bridge_contract("adults", evidence),
        callback().interface_contract(),
    ];
    let composed = check_composition(&components, &feed_forward()).expect("compatible chain");
    assert_eq!(composed.order, vec!["mechanism:schema:y", "bridge:y_to_z", "outcome_model"]);
    assert_eq!(composed.connections.len(), 2);
    // The calibration is the producer's, never raised by a consumer: a mean of
    // a fit offered with no uncertainty is point-only all the way.
    assert!(composed.connections.iter().all(|c| c.calibration == CalibrationStatus::PointOnly));
    // The mechanism's evidence was read from its declaration as a ledger id.
    assert_eq!(
        composed.connections[0].evidence,
        EvidenceBasis::Declared(BTreeSet::from([evidence]))
    );
    // The same declarations decide the same composition identity.
    let again = check_composition(&components, &feed_forward()).expect("compatible chain");
    assert_eq!(composed.identity, again.identity);
}

#[test]
fn matching_names_units_and_support_but_another_population_is_refused() {
    // Same variable, units, regime, support and operations: only the
    // population the mechanism was fitted in differs from the one the bridge
    // consumes. The mechanism's own admission of another population is not a
    // license for a connection.
    let mut ledger = EvidenceLedger::new();
    let evidence = observed(&mut ledger, "fit-y", &[1, 2, 3]);
    let components = [
        mechanism_contract(evidence, "children"),
        bridge_contract("adults", evidence),
        callback().interface_contract(),
    ];
    let refusals = check_composition(&components, &feed_forward()).expect_err("population differs");
    assert_eq!(details(&refusals), vec!["interface_contract.population_mismatch"]);
    assert_eq!(refusals[0].expected.as_deref(), Some("adults"));
    assert_eq!(refusals[0].supplied.as_deref(), Some("children"));
}

#[test]
fn a_bridge_consuming_a_mean_is_not_fed_by_a_law() {
    let mut ledger = EvidenceLedger::new();
    let evidence = observed(&mut ledger, "fit-y", &[1]);
    let law_basis = MechanismInterfaceBasis::new(
        PortFacts::new(SupportDomain::Real)
            .with_uncertainty(DistributionMeaning::ParameterPosterior),
        [ContractOperation::Law],
    );
    let producer = mechanism(evidence, "adults").interface_contract(&law_basis);
    let components =
        [producer, bridge_contract("adults", evidence), callback().interface_contract()];
    let refusals = check_composition(&components, &feed_forward()).expect_err("law is not a mean");
    assert_eq!(details(&refusals), vec!["interface_contract.operation_unavailable"]);
    assert_eq!(refusals[0].offending.as_deref(), Some("mean"));
}

#[test]
fn a_measurement_assumption_mismatch_is_refused_between_a_mechanism_and_a_bridge() {
    let mut ledger = EvidenceLedger::new();
    let evidence = observed(&mut ledger, "fit-y", &[1]);
    let basis = MechanismInterfaceBasis::new(
        PortFacts::new(SupportDomain::Real).with_measurement_models([MODEL_B]),
        [ContractOperation::Mean],
    );
    let producer = mechanism(evidence, "adults").interface_contract(&basis);
    let components =
        [producer, bridge_contract("adults", evidence), callback().interface_contract()];
    let refusals =
        check_composition(&components, &feed_forward()).expect_err("measured under another model");
    assert_eq!(details(&refusals), vec!["interface_contract.measurement_assumption_mismatch"]);
    assert_eq!(refusals[0].supplied.as_deref(), Some(MODEL_B));
}

#[test]
fn a_cycle_through_a_mechanism_a_bridge_and_a_callback_is_refused_with_its_path() {
    let mut ledger = EvidenceLedger::new();
    let evidence = observed(&mut ledger, "fit-y", &[1]);
    let components = [
        mechanism_contract(evidence, "adults"),
        bridge_contract("adults", evidence),
        callback().interface_contract(),
    ];
    let mut connections = feed_forward();
    // The callback's output feeds the mechanism's parent: feedback.
    connections.push(ComponentConnection::new(
        "outcome_model",
        CALLBACK_OUTPUT_PORT,
        "mechanism:schema:y",
        "schema:w",
    ));
    let refusals = check_composition(&components, &connections).expect_err("feedback is refused");
    assert_eq!(details(&refusals), vec!["interface_contract.cyclic_composition"]);
    assert_eq!(
        refusals[0].offending.as_deref(),
        Some("bridge:y_to_z -> outcome_model -> mechanism:schema:y -> bridge:y_to_z")
    );
}

// -- measurement model --------------------------------------------------------------------------

fn clinic_model() -> MeasurementModel {
    MeasurementModel {
        model_id: "screening-flag".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "schema:flag".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "binary".into(),
        },
        construct: MeasurementVariable {
            id: "construct:disease".into(),
            kind: MeasurementKind::LatentScientific,
            units: "binary".into(),
        },
        binding: MeasurementBinding {
            population_id: "clinic-a".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.9,
                specificity: 0.8,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 },
        evidence: ParameterEvidence::Declared { declarer: "lab".into() },
    }
}

fn disease(variable: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: "binary".into(),
        functional_id: "prevalence".into(),
        regime_id: "observational".into(),
        ..q(variable, population)
    }
}

fn measurement_basis() -> MeasurementInterfaceBasis {
    MeasurementInterfaceBasis {
        indicator: PortFacts::new(SupportDomain::Binary),
        construct: PortFacts::new(SupportDomain::Real),
        operations: BTreeSet::from([ContractOperation::Mean]),
        calibration: None,
        evidence: EvidenceBasis::Undeclared,
    }
}

#[test]
fn a_measurement_model_is_produced_under_its_identity_and_consumers_state_the_model_they_rely_on() {
    let model = clinic_model();
    let identity = model.identities().model.to_hex();
    let contract = measurement_interface_contract(
        "screening_correction",
        &model,
        &disease("schema:flag", "clinic-a"),
        &disease("construct:disease", "clinic-a"),
        &measurement_basis(),
    )
    .expect("the quantities are the model's");
    assert_eq!(contract.consumes[0].name, MEASUREMENT_INDICATOR_PORT);
    let produced = contract.produced(MEASUREMENT_CONSTRUCT_PORT).expect("construct port");
    assert_eq!(produced.measurement_models, BTreeSet::from([identity.clone()]));
    assert_eq!(produced.calibration, CalibrationStatus::PointOnly);

    let consumer = |models: &str| InterfaceContract {
        component: "decision".into(),
        produces: Vec::new(),
        consumes: vec![
            ConsumedPort::new(
                "disease",
                disease("construct:disease", "clinic-a"),
                "construct:disease",
                SupportDomain::Real,
                [ContractOperation::Mean],
            )
            .with_measurement_models([models.to_owned()]),
        ],
    };
    let connect = |decision: InterfaceContract| {
        check_composition(
            &[contract.clone(), decision],
            &[ComponentConnection::new(
                "screening_correction",
                MEASUREMENT_CONSTRUCT_PORT,
                "decision",
                "disease",
            )],
        )
    };
    connect(consumer(&identity)).expect("relies on exactly this model");
    let refusals = connect(consumer(MODEL_B)).expect_err("relies on another model");
    assert_eq!(details(&refusals), vec!["interface_contract.measurement_assumption_mismatch"]);
}

#[test]
fn a_measurement_model_refuses_quantities_it_does_not_bind() {
    let model = clinic_model();
    let basis = measurement_basis();
    let other_population = measurement_interface_contract(
        "screening_correction",
        &model,
        &disease("schema:flag", "clinic-b"),
        &disease("construct:disease", "clinic-b"),
        &basis,
    )
    .expect_err("population is not the binding");
    assert_eq!(other_population.detail, "measurement_model.population_mismatch");
    let proxy = measurement_interface_contract(
        "screening_correction",
        &model,
        &disease("schema:flag", "clinic-a"),
        &disease("schema:flag", "clinic-a"),
        &basis,
    )
    .expect_err("the indicator is not the construct");
    assert_eq!(proxy.detail, "measurement_model.construct_mismatch");
}

// -- evidence ledger overlap ----------------------------------------------------------------------

fn two_consumers_requiring_disjoint(
    first: SourceId,
    second: SourceId,
) -> (Vec<InterfaceContract>, Vec<ComponentConnection>) {
    let producer = |name: &str, source: SourceId| InterfaceContract {
        component: name.into(),
        produces: vec![
            antecedent_contract::interface_contract::ProducedPort::new(
                "effect",
                q(&format!("schema:{name}"), "adults"),
                format!("schema:{name}"),
                SupportDomain::Real,
                [ContractOperation::Mean],
            )
            .with_evidence([source]),
        ],
        consumes: Vec::new(),
    };
    let port = |name: &str| {
        ConsumedPort::new(
            name,
            q(&format!("schema:{name}"), "adults"),
            format!("schema:{name}"),
            SupportDomain::Real,
            [ContractOperation::Mean],
        )
        .with_dependence(DependenceRequirement::DisjointEvidence)
    };
    let pooled = InterfaceContract {
        component: "pooled".into(),
        produces: Vec::new(),
        consumes: vec![port("a"), port("b")],
    };
    (
        vec![producer("a", first), producer("b", second), pooled],
        vec![
            ComponentConnection::new("a", "effect", "pooled", "a"),
            ComponentConnection::new("b", "effect", "pooled", "b"),
        ],
    )
}

#[test]
fn distinct_sources_that_share_rows_are_not_disjoint_and_unknown_overlap_is_not_disjoint() {
    let mut ledger = EvidenceLedger::new();
    let first = observed(&mut ledger, "trial-a", &[1, 2, 3]);
    let overlapping = observed(&mut ledger, "trial-b", &[3, 4]);
    let disjoint = observed(&mut ledger, "trial-c", &[7, 8]);
    let opaque = undeclared(&mut ledger, "trial-d");

    // The declared source identities differ, so the identity comparison alone
    // accepts the overlapping pair; the ledger does not.
    let (components, connections) = two_consumers_requiring_disjoint(first, overlapping);
    check_composition(&components, &connections).expect("distinct identities pass by identity");
    let shared = check_composition_with_ledger(&components, &connections, &ledger)
        .expect_err("they share observation 3");
    assert_eq!(details(&shared), vec!["interface_contract.shared_evidence"]);
    assert_eq!(shared[0].supplied.as_deref(), Some("1 shared source(s)"));

    let (components, connections) = two_consumers_requiring_disjoint(first, disjoint);
    check_composition_with_ledger(&components, &connections, &ledger).expect("rows are disjoint");

    let (components, connections) = two_consumers_requiring_disjoint(first, opaque);
    let unknown = check_composition_with_ledger(&components, &connections, &ledger)
        .expect_err("undeclared observations are unknown, never disjoint");
    assert_eq!(details(&unknown), vec!["interface_contract.dependence_unknown"]);

    // Without a ledger the requirement is reported as unchecked, not skipped.
    let unchecked = disjoint_requirements_without_ledger(&components);
    assert_eq!(details(&unchecked), vec!["interface_contract.dependence_unknown"; 2]);
    assert_eq!(unchecked[0].offending.as_deref(), Some("pooled.a"));
}

#[test]
fn mechanisms_declared_independent_are_checked_against_the_ledger_rows() {
    let mut ledger = EvidenceLedger::new();
    let first = observed(&mut ledger, "fit-a", &[1, 2, 3]);
    let overlapping = observed(&mut ledger, "fit-b", &[3, 4]);
    let disjoint = observed(&mut ledger, "fit-c", &[7, 8]);
    let opaque = undeclared(&mut ledger, "fit-d");
    let declaration = |second: SourceId| {
        let mut b = mechanism(second, "adults");
        b.output = q("schema:x", "adults");
        b.parents = Vec::new();
        b.conditioning_variable_ids = Vec::new();
        MechanismCompositionDeclaration {
            target_population_id: "adults".into(),
            target_regime_id: "observational".into(),
            mechanisms: vec![
                MechanismDeclaration {
                    parents: Vec::new(),
                    conditioning_variable_ids: Vec::new(),
                    ..mechanism(first, "adults")
                },
                b,
            ],
            edges: Vec::new(),
            admissions: Vec::new(),
            dependence: DependenceDeclaration::IndependentByDesign {
                justification: "separate trials".into(),
            },
            admit_declared_mechanisms: false,
        }
    };
    check_declared_independence_against_ledger(&declaration(disjoint), &ledger)
        .expect("disjoint rows");
    let shared = check_declared_independence_against_ledger(&declaration(overlapping), &ledger)
        .expect_err("row 3 is in both");
    assert_eq!(
        details(&shared),
        vec!["mechanism_composition.posterior_dependence_shared_evidence_declared_independent"]
    );
    let unknown = check_declared_independence_against_ledger(&declaration(opaque), &ledger)
        .expect_err("undeclared rows");
    assert_eq!(
        details(&unknown),
        vec!["mechanism_composition.posterior_dependence_evidence_overlap_unknown"]
    );
    // An evidence id that is not a ledger source is unknown, not independent.
    let mut not_a_source = declaration(disjoint);
    not_a_source.mechanisms[1].evidence_id = "snapshot-7".into();
    let unknown = check_declared_independence_against_ledger(&not_a_source, &ledger)
        .expect_err("not a ledger identity");
    assert_eq!(
        details(&unknown),
        vec!["mechanism_composition.posterior_dependence_evidence_overlap_unknown"]
    );
}

fn estimate(id: &str, evidence: SourceId) -> PropagationInput {
    PropagationInput {
        id: id.into(),
        evidence_id: evidence.to_hex(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    }
}

#[test]
fn a_declared_independence_is_decided_from_ledger_rows_not_evidence_strings() {
    let mut ledger = EvidenceLedger::new();
    let first = observed(&mut ledger, "snapshot-a", &[1, 2, 3]);
    let subset = observed(&mut ledger, "snapshot-b", &[2, 3]);
    let disjoint = observed(&mut ledger, "snapshot-c", &[9]);
    let opaque = undeclared(&mut ledger, "snapshot-d");
    let request = |second: SourceId| PropagationRequest {
        operation: PropagationOperation::Sum,
        inputs: vec![estimate("x", first), estimate("y", second)],
        dependence: PropagationDependence::DeclaredIndependent,
        reading: RequestedReading::Point,
    };
    // By identity string the two snapshots are distinct and independence is
    // accepted: this is the defect.
    check_propagation(&request(subset)).expect("string-distinct passes the string rule");
    let shared = check_propagation_with_ledger(&request(subset), &ledger)
        .expect_err("a subset shares its rows");
    assert_eq!(shared.detail, "joint_law_propagation.shared_evidence_declared_independent");
    assert_eq!(shared.code, "evidence_double_counted");

    let plan = check_propagation_with_ledger(&request(disjoint), &ledger).expect("disjoint rows");
    assert!(plan.shared_evidence.is_empty());

    let unknown = check_propagation_with_ledger(&request(opaque), &ledger)
        .expect_err("undeclared rows are unknown");
    assert_eq!(unknown.detail, "joint_law_propagation.evidence_overlap_unknown");
    assert_eq!(unknown.code, "sampling_dependence_unknown");

    // The forwarded duplicate (the same source twice) shares by identity.
    let twice = PropagationRequest {
        inputs: vec![estimate("x", first), estimate("y", first)],
        ..request(first)
    };
    let shared = check_propagation_with_ledger(&twice, &ledger).expect_err("same source twice");
    assert_eq!(shared.detail, "joint_law_propagation.shared_evidence_declared_independent");
}

#[test]
fn row_pairing_across_inputs_compares_their_real_alignment_identities() {
    let aligned = |id: &str| InputAlignment::Aligned { alignment_id: id.into() };
    let (a, b, c) = (aligned("joint-a"), aligned("joint-a"), aligned("joint-b"));
    check_rows_pairable_across(&[("x", &a), ("y", &b)]).expect("one alignment identity");
    let mismatch = check_rows_pairable_across(&[("x", &a), ("y", &c)])
        .expect_err("two joint artifacts with different identities are not paired");
    assert_eq!(mismatch.detail, "joint_law_propagation.alignment_mismatch");
    let marginals = InputAlignment::IndependentMarginals;
    let unpaired = check_rows_pairable_across(&[("x", &a), ("y", &marginals)])
        .expect_err("independent marginals are never paired by index");
    assert_eq!(unpaired.detail, "joint_law_propagation.index_pairing_refused");
}
