//! Golden fixtures for the artifact families that carry their own version constant, and
//! the refusal of a newer version or an unknown required feature at the byte boundary.
//!
//! Each golden lives at `tests/fixtures/<family>_v<N>.bin`. A loader test reads it, checks
//! its container kind, consumes it as a fresh consumer (re-deriving rather than trusting
//! the stored identity) and requires the identity a producer derives today from the same
//! declaration. A missing golden fails with the command that writes it.
//!
//! Generate or refresh every golden with
//! `cargo test -p antecedent-io-core --test artifact_compat_golden -- --ignored write_`
//! and review the diff: a changed golden is a compatibility decision and needs a
//! migration note.
//!
//! The manifest records the producing library version, so bytes are not compared for
//! equality; the artifact identity, which excludes it, is.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, BoundarySidedness,
    DelayMechanism, InferenceParadigm, InterimAnalysis, MonitoringStrategy, OutcomeAvailability,
    OutcomeStructure, PolicyEpoch, ProcedureCell, SpendingKind, StoppingRuleDeclaration,
    StoppingRuleKind, StoppingSpec,
};
use antecedent_contract::callback_protocol::{
    CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind, CallbackOperation,
    CallbackTrust, DependencyDeclaration, DependencyKind, InputSupplier, ReplayContract,
};
use antecedent_contract::composition_verification::{
    CalibrationDeclaration, CompositionGraph, CompositionNode,
    DependencyKind as GraphDependencyKind, NodeKind, Revalidation,
};
use antecedent_contract::evidence_ledger::{SourceId, TransportIdentity};
use antecedent_contract::interface_contract::{ContractOperation, EvidenceBasis, SupportDomain};
use antecedent_contract::provider_transport::{TransportedClaimOffer, TransportedInputPolicy};
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_discovery::selective::{
    DiscoveredStructure, Mark, ORIENTATION_COMPONENTS, PortableDiscoveryState, SEARCH_COMPONENTS,
    STATISTIC_COMPONENTS, StatisticReceipt, StructureClass, StructureEdge, StructureNode,
    WorkUnits,
};
use antecedent_io_core::adaptive_monitoring_contract_artifact::{
    ADAPTIVE_MONITORING_CONTRACT_FEATURE, ADAPTIVE_MONITORING_CONTRACT_VERSION,
    AdaptiveMonitoringContractArtifact,
};
use antecedent_io_core::cohort_decision::{
    CATE_FUNCTIONAL_ID, COHORT_DECISION_ARTIFACT_KIND, COHORT_DECISION_SECTION, CohortBinding,
    CohortColumn, CohortDecisionSource, CohortExpectation, CohortOptions, RetainedEffectModel,
};
use antecedent_io_core::compact_export::{
    COMPACT_EXPORT_ARTIFACT_KIND, COMPACT_EXPORT_SECTION, CompactExport, ExportInput, ExportLimits,
    ExportSpec, InputSupport, TermSpec,
};
use antecedent_io_core::composition_verification_artifact::{
    COMPOSITION_VERIFICATION_ARTIFACT_FEATURE, COMPOSITION_VERIFICATION_ARTIFACT_VERSION,
    CompositionVerificationArtifact, CompositionVerificationArtifactError,
};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use antecedent_io_core::selective_discovery_artifact::{
    SELECTIVE_DISCOVERY_ARTIFACT_FEATURE, SELECTIVE_DISCOVERY_ARTIFACT_VERSION,
    SelectiveDiscoveryArtifact, SelectiveDiscoveryArtifactError,
};
use antecedent_io_core::transport_handoff_artifact::{
    TRANSPORT_HANDOFF_ARTIFACT_FEATURE, TRANSPORT_HANDOFF_ARTIFACT_VERSION,
    TransportHandoffArtifact, TransportHandoffArtifactError,
};
use antecedent_io_core::{
    ArtifactContainer, CONTENT_CBOR, OutSection, container_kind, decode_container, encode_container,
};
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str =
    "cargo test -p antecedent-io-core --test artifact_compat_golden -- --ignored write_";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(name: &str) -> Vec<u8> {
    read_fixture_bytes(&fixture_dir(), name, &format!("generate it with `{GENERATOR}`"))
}

fn write(name: &str, bytes: &[u8]) {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(name), bytes).expect("write the golden");
}

/// Rewrite the `version` word of a single-section CBOR body and re-wrap it in a valid
/// container, so only the version gate can refuse it.
fn with_body_version(bytes: &[u8], kind: &str, section: &str, max: usize, version: u64) -> Vec<u8> {
    let parts = decode_container(kind, max, &[section], bytes).expect("a readable container");
    let mut value: ciborium::Value =
        ciborium::from_reader(parts[0].as_slice()).expect("a CBOR body");
    let ciborium::Value::Map(entries) = &mut value else { panic!("the body is a map") };
    let mut found = false;
    for (key, entry) in entries.iter_mut() {
        if key.as_text() == Some("version") {
            *entry = ciborium::Value::Integer(version.into());
            found = true;
        }
    }
    assert!(found, "the body names its version");
    let mut body = Vec::new();
    ciborium::into_writer(&value, &mut body).expect("encode the body");
    encode_container(
        &ArtifactContainer {
            kind,
            note: "compat",
            max_bytes: max,
            artifact_id: "compat-1",
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[OutSection { id: section, content_type: CONTENT_CBOR, bytes: &body }],
    )
    .expect("a valid container")
}

// -- transport handoff -------------------------------------------------------------------------

fn handoff_quantity(population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:recovery".into(),
        variable_name: "recovery".into(),
        role: QuantityRole::Outcome,
        units: "days".into(),
        population_id: population.into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn handoff_offer() -> TransportedClaimOffer {
    TransportedClaimOffer {
        component: "transport_result".into(),
        port: "effect".into(),
        quantity: handoff_quantity("target"),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        operations: [ContractOperation::Mean].into_iter().collect(),
        uncertainty: None,
        transport: TransportIdentity {
            source_populations: vec!["trial_a".into(), "trial_b".into()],
            target_population: "target".into(),
            selection_diagram: SemanticDigest::from_bytes([7; 32]),
            assumptions: vec!["recovery mechanism invariant across populations".into()],
        },
        evidence: [SourceId::from_digest(SemanticDigest::from_bytes([5; 32]))]
            .into_iter()
            .collect(),
    }
}

fn handoff_callback() -> CallbackDeclaration {
    CallbackDeclaration {
        component: "downstream_model".into(),
        kind: CallbackKind::Law,
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "downstream".into(),
            version_id: "1.0".into(),
            snapshot_id: "snap-1".into(),
            request_id: "req-1".into(),
        },
        quantity: handoff_quantity("target"),
        semantic_identity: "estimand:downstream_outcome".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: [CallbackOperation::Mean].into_iter().collect(),
        update: None,
        inputs: vec![CallbackInput {
            name: "effect".into(),
            quantity: handoff_quantity("target"),
            semantic_identity: "estimand:recovery_under_treatment".into(),
            support: SupportDomain::Real,
            operation: CallbackOperation::Mean,
            uncertainty: None,
            supplier: InputSupplier::External,
        }],
        dependencies: DependencyDeclaration::Complete(vec![CallbackDependency::new(
            DependencyKind::Model,
            "parameters",
            "beta=0.5",
        )]),
        replay: ReplayContract::DeterministicPure,
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

fn handoff_sealed()
-> (TransportHandoffArtifact, antecedent_io_core::transport_handoff_artifact::TransportHandoffMeta)
{
    TransportHandoffArtifact::seal_for_callback(
        &handoff_offer(),
        &handoff_callback(),
        "effect",
        TransportedInputPolicy::AcceptsTransported,
    )
    .expect("seals")
}

const HANDOFF_GOLDEN: &str = "transport_handoff_v1.bin";

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_transport_handoff_golden() {
    let (_, meta) = handoff_sealed();
    write(
        HANDOFF_GOLDEN,
        &TransportHandoffArtifact::meta_to_bytes(&meta, "golden-transport-handoff").unwrap(),
    );
}

#[test]
fn the_transport_handoff_golden_is_consumed_and_gated() {
    let bytes = load(HANDOFF_GOLDEN);
    assert_eq!(container_kind(&bytes).as_deref(), Some("transport_handoff_v1"));
    let (fresh, _) = handoff_sealed();
    let consumed = TransportHandoffArtifact::from_bytes(&bytes, Some(fresh.identity()))
        .expect("a fresh consumer re-derives the stored handoff");
    assert_eq!(consumed.identity(), fresh.identity());
    assert_eq!(consumed.handoff_identity(), fresh.handoff_identity());
    consumed.verify_against_callback(&handoff_callback()).expect("the declared callback");

    let meta = TransportHandoffArtifact::inspect(&bytes).expect("inspectable");
    assert_eq!(meta.version, TRANSPORT_HANDOFF_ARTIFACT_VERSION);
    assert_eq!(meta.feature, TRANSPORT_HANDOFF_ARTIFACT_FEATURE);

    // A newer major version is refused before the payload is read.
    let mut newer = meta.clone();
    newer.version = TRANSPORT_HANDOFF_ARTIFACT_VERSION + 1;
    TransportHandoffArtifact::reseal_identity(&mut newer);
    let newer_bytes = TransportHandoffArtifact::meta_to_bytes(&newer, "newer").unwrap();
    assert!(matches!(
        TransportHandoffArtifact::from_bytes(&newer_bytes, None).expect_err("newer"),
        TransportHandoffArtifactError::UnsupportedVersion { version: 2 }
    ));
    // An unknown required feature is refused even under a consistent identity.
    let mut feature = meta;
    feature.required_features.push("future_feature".into());
    TransportHandoffArtifact::reseal_identity(&mut feature);
    let feature_bytes = TransportHandoffArtifact::meta_to_bytes(&feature, "feature").unwrap();
    assert_eq!(
        TransportHandoffArtifact::from_bytes(&feature_bytes, None).expect_err("feature"),
        TransportHandoffArtifactError::UnsupportedSemantics("required feature")
    );
}

// -- selective discovery ------------------------------------------------------------------------

fn discovery_state() -> PortableDiscoveryState {
    let node = |variable: u32| StructureNode { kind: 0, variable, aux: 0 };
    let nodes = vec![node(0), node(1), node(2)];
    let edge = |a: u32, b: u32, at_a: Mark, at_b: Mark| StructureEdge {
        a: node(a),
        b: node(b),
        at_a,
        at_b,
        middle: 4,
    };
    let edges = vec![edge(0, 1, Mark::Tail, Mark::Arrow), edge(1, 2, Mark::Tail, Mark::Tail)];
    let structure = DiscoveredStructure::from_parts(StructureClass::Cpdag, nodes, edges).unwrap();
    let mut names: Vec<&str> = STATISTIC_COMPONENTS
        .iter()
        .chain(SEARCH_COMPONENTS.iter())
        .chain(ORIENTATION_COMPONENTS.iter())
        .copied()
        .collect();
    names.sort_unstable();
    let components = names
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n.to_owned(), [u8::try_from(i).unwrap() + 1; 32]))
        .collect();
    let receipt = |tag: u8, p: f64| StatisticReceipt {
        coordinate: [tag; 32],
        statistic: 0.25,
        p_value: p,
        df: 97.0,
        interval: Some((-0.1, 0.3)),
    };
    let mut state = PortableDiscoveryState {
        route: "static_pc".to_owned(),
        components,
        contract_identity: [0; 32],
        data_identity: [9; 32],
        rows: 100,
        variables: vec![0, 1, 2],
        receipts: vec![receipt(1, 0.5), receipt(2, 0.01), receipt(3, 0.9)],
        receipts_identity: [0; 32],
        structure,
        result_identity: [7; 32],
        search_incomplete: false,
        orientation_conflict: false,
        bounded_search_receipts: Vec::new(),
        work: WorkUnits { ci_tests_attempted: 3, ci_tests_completed: 3, ..WorkUnits::default() },
    };
    state.reseal();
    state
}

const DISCOVERY_GOLDEN: &str = "selective_discovery_v1.bin";

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_selective_discovery_golden() {
    write(
        DISCOVERY_GOLDEN,
        &SelectiveDiscoveryArtifact::to_bytes(&discovery_state(), "golden-selective-discovery")
            .unwrap(),
    );
}

#[test]
fn the_selective_discovery_golden_is_consumed_and_gated() {
    let bytes = load(DISCOVERY_GOLDEN);
    assert_eq!(container_kind(&bytes).as_deref(), Some("selective_discovery_v1"));
    let state = discovery_state();
    let fresh = SelectiveDiscoveryArtifact::seal(&state).expect("seals").0;
    let consumed = SelectiveDiscoveryArtifact::from_bytes(&bytes, Some(fresh.identity()))
        .expect("a fresh consumer rebuilds receipts and structure");
    assert_eq!(consumed.identity(), fresh.identity());
    assert_eq!(consumed.state().receipts, state.receipts);
    assert_eq!(consumed.state().structure, state.structure);

    let meta = SelectiveDiscoveryArtifact::inspect(&bytes).expect("inspectable");
    assert_eq!(meta.version, SELECTIVE_DISCOVERY_ARTIFACT_VERSION);
    assert_eq!(meta.feature, SELECTIVE_DISCOVERY_ARTIFACT_FEATURE);

    let mut newer = meta.clone();
    newer.version = SELECTIVE_DISCOVERY_ARTIFACT_VERSION + 1;
    SelectiveDiscoveryArtifact::reseal_identity(&mut newer);
    let newer_bytes = SelectiveDiscoveryArtifact::meta_to_bytes(&newer, "newer").unwrap();
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes(&newer_bytes, None),
        Err(SelectiveDiscoveryArtifactError::UnsupportedVersion { version: 2 })
    ));
    let mut feature = meta;
    feature.required_features = vec!["future_feature".to_owned()];
    SelectiveDiscoveryArtifact::reseal_identity(&mut feature);
    let feature_bytes = SelectiveDiscoveryArtifact::meta_to_bytes(&feature, "feature").unwrap();
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes(&feature_bytes, None),
        Err(SelectiveDiscoveryArtifactError::UnsupportedSemantics("required feature"))
    ));
}

// -- monitoring contract ---------------------------------------------------------------------

fn monitoring_contract() -> AdaptiveDesignContract {
    let cell = ProcedureCell {
        assignment: AssignmentFamily::FixedRandomization,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::GroupSequential,
        stopping: StoppingRuleKind::Efficacy,
    };
    let assigned_arm: Vec<u32> = (0..12).map(|unit| unit % 2).collect();
    AdaptiveDesignContract::new(AdaptiveDesignDeclaration {
        label: "group_sequential_trial".to_owned(),
        cell,
        arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![PolicyEpoch {
            first_unit: 0,
            policy_identity: SemanticDigest::from_bytes([7; 32]),
        }]),
        assignment_probabilities: Some(vec![vec![0.5, 0.5]; 12]),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look: (0..12).map(|unit| Some(u32::from(unit >= 6))).collect(),
            delay: DelayMechanism::Immediate,
        }),
        outcomes: (0..12).map(|unit| vec![f64::from(unit % 5) * 0.5]).collect(),
        interim_analyses: Some(vec![
            InterimAnalysis { units_enrolled: 6, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 12, information_fraction: 1.0 },
        ]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::AlphaSpending {
                spending: SpendingKind::OBrienFlemingType,
                total_alpha: 0.05,
                sidedness: BoundarySidedness::TwoSidedSymmetric,
            },
            ended_by_rule: false,
        }),
        probability_floor: 0.05,
    })
    .expect("complete contract")
}

const MONITORING_GOLDEN: &str = "adaptive_monitoring_contract_v1.bin";

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_adaptive_monitoring_contract_golden() {
    write(
        MONITORING_GOLDEN,
        &AdaptiveMonitoringContractArtifact::to_bytes(
            &monitoring_contract(),
            InferenceParadigm::FrequentistGroupSequential,
            "golden-adaptive-monitoring",
        )
        .unwrap(),
    );
}

#[test]
fn the_adaptive_monitoring_contract_golden_is_consumed_and_gated() {
    let bytes = load(MONITORING_GOLDEN);
    assert_eq!(container_kind(&bytes).as_deref(), Some("adaptive_monitoring_contract_v1"));
    let contract = monitoring_contract();
    let consumed = AdaptiveMonitoringContractArtifact::from_bytes(&bytes, None)
        .expect("a fresh consumer replays the contract");
    assert_eq!(consumed.paradigm(), InferenceParadigm::FrequentistGroupSequential);
    assert_eq!(consumed.contract().identity(), contract.identity());
    // The identity a producer derives today equals the stored one.
    assert!(
        AdaptiveMonitoringContractArtifact::from_bytes(&bytes, Some(consumed.identity())).is_ok()
    );

    let mut meta = AdaptiveMonitoringContractArtifact::decode_unchecked(&bytes).expect("meta");
    assert_eq!(meta.version, ADAPTIVE_MONITORING_CONTRACT_VERSION);
    assert_eq!(meta.feature, ADAPTIVE_MONITORING_CONTRACT_FEATURE);
    meta.version += 1;
    AdaptiveMonitoringContractArtifact::reseal_identity(&mut meta);
    assert!(AdaptiveMonitoringContractArtifact::from_meta(&meta, None).is_err());
}

const PARTICIPANT_GOLDEN: &str = "adaptive_monitoring_contract_participant_v1.bin";

/// A retained run of the participant-level family under the confidence sequence the released
/// golden was written with, and the procedure it ran under.
fn participant_run()
-> (AdaptiveDesignContract, antecedent_stats::adaptive_inference::ParticipantSpec) {
    participant_run_under(antecedent_stats::adaptive_inference::SequenceKind::MixtureHoeffding)
}

fn participant_spec(
    sequence: antecedent_stats::adaptive_inference::SequenceKind,
) -> antecedent_stats::adaptive_inference::ParticipantSpec {
    use antecedent_stats::adaptive_inference::{
        AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan, CorrelationPrior, DecisionRules,
        HistoricalPrior, ParticipantDeclaration, ParticipantSpec,
    };
    let control = ArmPrior {
        initial: BetaPrior::UNIFORM,
        historical: vec![HistoricalPrior {
            source: "registry".to_owned(),
            successes: 40,
            trials: 100,
            discount: 0.5,
        }],
    };
    ParticipantSpec::new(ParticipantDeclaration {
        arms: 3,
        responses: Some(3),
        correlation: None,
        // The correlation is estimated jointly with the arm means under a Beta(2, 6) prior.
        correlation_prior: Some(CorrelationPrior { alpha: 2.0, beta: 6.0 }),
        borrowing: Some(BorrowingPlan {
            arms: vec![
                control,
                ArmPrior::without_history(BetaPrior::UNIFORM),
                ArmPrior::without_history(BetaPrior::UNIFORM),
            ],
        }),
        enrolment: Some(vec![10, 10, 10]),
        lag_looks: 0,
        min_exploration: Some(0.1),
        decisions: Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: None,
            participant_cap: None,
        }),
        anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 10.0 }),
    })
    .expect("a valid procedure")
    .with_sequence_kind(sequence)
}

/// A retained run of the participant-level family and the procedure it ran under, reading the
/// given confidence-sequence construction.
fn participant_run_under(
    sequence: antecedent_stats::adaptive_inference::SequenceKind,
) -> (AdaptiveDesignContract, antecedent_stats::adaptive_inference::ParticipantSpec) {
    use antecedent_stats::adaptive_inference::{LookDecision, simulate_participant_history};
    let spec = participant_spec(sequence);
    let history = simulate_participant_history(&spec, &[0.4, 0.4, 0.4], Some(0.2), 5, &|| false)
        .expect("a history");
    let readings = spec.read_history(&history, &|| false).expect("replays");
    assert!(matches!(readings.last().map(|r| r.decision), Some(LookDecision::Stop { .. })));
    assert!(history.batches.len() >= 2);
    let total: u32 = history.batches.iter().sum();
    let mut cumulative = 0;
    let mut policy_history = Vec::new();
    let mut interim_analyses = Vec::new();
    let mut available_at_look = Vec::new();
    for (look, &size) in history.batches.iter().enumerate() {
        policy_history.push(PolicyEpoch {
            first_unit: cumulative,
            policy_identity: SemanticDigest::from_bytes([u8::try_from(look).unwrap() + 1; 32]),
        });
        cumulative += size;
        interim_analyses.push(InterimAnalysis {
            units_enrolled: cumulative,
            information_fraction: f64::from(cumulative) / f64::from(total),
        });
        available_at_look.extend((0..size).map(|_| Some(u32::try_from(look).unwrap())));
    }
    let arms: Vec<u32> = history.arms.iter().map(|&a| u32::try_from(a).unwrap()).collect();
    let contract = AdaptiveDesignContract::new(AdaptiveDesignDeclaration {
        label: "participant_trial".to_owned(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::ThompsonBetaBernoulli,
            outcome: OutcomeStructure::RepeatedMeasures,
            monitoring: MonitoringStrategy::FixedInterimLooks,
            stopping: StoppingRuleKind::DecisionTheoretic,
        },
        arm_labels: vec!["control".to_owned(), "low".to_owned(), "high".to_owned()],
        assigned_arm: arms.clone(),
        policy_history: Some(policy_history),
        assignment_probabilities: Some(history.assignment.clone()),
        exposure: Some(arms),
        availability: Some(OutcomeAvailability {
            available_at_look,
            delay: DelayMechanism::Immediate,
        }),
        outcomes: history
            .successes
            .iter()
            .map(|count| {
                let ones = count.unwrap_or(0) as usize;
                let mut values = vec![1.0; ones];
                values.resize(3, 0.0);
                values
            })
            .collect(),
        interim_analyses: Some(interim_analyses),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::DecisionTheoretic,
            spec: StoppingSpec::PosteriorExpectedLoss { loss_threshold: 1e-12 },
            ended_by_rule: false,
        }),
        probability_floor: 0.1,
    })
    .expect("complete contract");
    (contract, spec)
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_adaptive_participant_procedure_golden() {
    let (contract, spec) = participant_run();
    write(
        PARTICIPANT_GOLDEN,
        &AdaptiveMonitoringContractArtifact::to_bytes_with_procedure(
            &contract,
            InferenceParadigm::BayesianPosterior,
            &spec,
            "golden-adaptive-participant",
        )
        .unwrap(),
    );
}

#[test]
fn the_adaptive_participant_procedure_golden_is_replayed_and_gated() {
    let bytes = load(PARTICIPANT_GOLDEN);
    assert_eq!(container_kind(&bytes).as_deref(), Some("adaptive_monitoring_contract_v1"));
    let consumed = AdaptiveMonitoringContractArtifact::from_bytes(&bytes, None)
        .expect("a fresh consumer replays the contract and the procedure section");
    // Replay the retained history, rather than simulating it again: adaptive
    // allocation probabilities depend on platform libm rounding. The golden's
    // exact declaration and identity remain the compatibility contract.
    let contract = AdaptiveDesignContract::new(consumed.contract().declaration().clone())
        .expect("the retained declaration validates independently");
    assert_eq!(
        contract.identity().to_hex(),
        "fe8105eb2f91aca3ec7d44ec90c0d7cf733c079766970282ef8f38b3565c3785"
    );
    let record = consumed.procedure_record().expect("the procedure section is replayed");
    let spec =
        participant_spec(antecedent_stats::adaptive_inference::SequenceKind::MixtureHoeffding);
    assert_eq!(record.spec().identity(), spec.identity());
    let (fresh, _) = AdaptiveMonitoringContractArtifact::seal_with_procedure(
        &contract,
        InferenceParadigm::BayesianPosterior,
        &spec,
    )
    .expect("a fresh producer replays the retained history");
    assert_eq!(consumed.contract().identity(), fresh.contract().identity());
    // The released contract names no sequence construction and replays under its own.
    assert_eq!(
        record.spec().sequence_kind(),
        antecedent_stats::adaptive_inference::SequenceKind::MixtureHoeffding
    );
    assert_eq!(record.readings().len(), contract.looks());
    // A reading that was edited and resealed does not replay.
    let mut meta = AdaptiveMonitoringContractArtifact::decode_unchecked(&bytes).expect("meta");
    let look = &mut meta.participant_procedure.as_mut().expect("section").looks[0];
    look.numbers_bits[0] = (f64::from_bits(look.numbers_bits[0]) + 0.01).to_bits();
    AdaptiveMonitoringContractArtifact::reseal_identity(&mut meta);
    assert!(AdaptiveMonitoringContractArtifact::from_meta(&meta, None).is_err());
}

#[test]
fn a_plug_in_sequence_procedure_records_its_construction_and_replays() {
    use antecedent_stats::adaptive_inference::SequenceKind;
    let plug_in = SequenceKind::PredictablePlugInBernstein;
    let (contract, spec) = participant_run_under(plug_in);
    let (_, released) = participant_run();
    // The construction is part of the procedure's identity.
    assert_ne!(spec.identity(), released.identity());
    let bytes = AdaptiveMonitoringContractArtifact::to_bytes_with_procedure(
        &contract,
        InferenceParadigm::BayesianPosterior,
        &spec,
        "plug-in-sequence",
    )
    .unwrap();
    let consumed = AdaptiveMonitoringContractArtifact::from_bytes(&bytes, None)
        .expect("a fresh consumer replays the plug-in sequence");
    let record = consumed.procedure_record().expect("the procedure section is replayed");
    assert_eq!(record.spec().sequence_kind(), plug_in);
    assert_eq!(record.spec().identity(), spec.identity());
    // A section edited to name the released sequence, and resealed, does not replay.
    let mut meta = AdaptiveMonitoringContractArtifact::decode_unchecked(&bytes).expect("meta");
    let section = meta.participant_procedure.as_mut().expect("section");
    assert_eq!(section.anytime_sequence.as_deref(), Some(plug_in.label()));
    section.anytime_sequence = None;
    AdaptiveMonitoringContractArtifact::reseal_identity(&mut meta);
    assert!(AdaptiveMonitoringContractArtifact::from_meta(&meta, None).is_err());
    // A construction the consumer does not know is refused, not defaulted.
    let section = meta.participant_procedure.as_mut().expect("section");
    section.anytime_sequence = Some("unknown_construction".to_owned());
    AdaptiveMonitoringContractArtifact::reseal_identity(&mut meta);
    assert!(AdaptiveMonitoringContractArtifact::from_meta(&meta, None).is_err());
}

// -- composition verification -----------------------------------------------------------------

fn digest_of(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

/// `cohort` feeds `model`; `fit` reuses `model`'s derivation and reads `cohort`; `claim`
/// reads `fit` and inherits its calibration record; `sibling` stands alone.
fn composition_graph(cohort: u8) -> CompositionGraph {
    let held = CalibrationDeclaration::Held { record: digest_of(90), coordinate: digest_of(100) };
    let mut g = CompositionGraph::new();
    let node = |key: &str, kind, id| CompositionNode::new(key, kind, digest_of(id));
    g.add_node(node("cohort", NodeKind::CohortSource, cohort)).unwrap();
    g.add_node(node("model", NodeKind::MeasurementModel, 2).with_structure(digest_of(3))).unwrap();
    g.add_node(
        node("fit", NodeKind::LedgerResult, 4)
            .with_structure(digest_of(5))
            .with_coordinate(digest_of(100))
            .with_calibration(held),
    )
    .unwrap();
    g.add_node(node("claim", NodeKind::Claim, 6).with_coordinate(digest_of(100)).with_calibration(
        CalibrationDeclaration::Inherited {
            from: "fit".to_owned(),
            record: digest_of(90),
            coordinate: digest_of(100),
        },
    ))
    .unwrap();
    g.add_node(node("sibling", NodeKind::Claim, 7)).unwrap();
    g.add_edge("model", "cohort", GraphDependencyKind::Data).unwrap();
    g.add_edge("fit", "model", GraphDependencyKind::StructuralDerivation).unwrap();
    g.add_edge("fit", "cohort", GraphDependencyKind::Data).unwrap();
    g.add_edge("claim", "fit", GraphDependencyKind::Data).unwrap();
    g.add_edge("claim", "fit", GraphDependencyKind::Calibration).unwrap();
    g
}

const COMPOSITION_GOLDEN: &str = "composition_verification_v1.bin";

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_composition_verification_golden() {
    let bytes = CompositionVerificationArtifact::to_bytes(
        Some(&composition_graph(1)),
        &composition_graph(9),
        &BTreeMap::new(),
        &[],
        None,
        "golden-composition-verification",
    )
    .unwrap();
    write(COMPOSITION_GOLDEN, &bytes);
}

#[test]
fn the_composition_verification_golden_is_replayed_and_gated() {
    let bytes = load(COMPOSITION_GOLDEN);
    assert_eq!(container_kind(&bytes).as_deref(), Some("composition_verification_v1"));
    let (fresh, meta) = CompositionVerificationArtifact::seal(
        Some(&composition_graph(1)),
        &composition_graph(9),
        &BTreeMap::new(),
        &[],
        None,
    )
    .expect("seals");
    let consumed = CompositionVerificationArtifact::from_bytes(&bytes, Some(fresh.identity()))
        .expect("a fresh consumer replays the classification");
    assert_eq!(consumed.identity(), fresh.identity());
    // Replacing the cohort's data recomputes everything downstream of it and leaves the
    // sibling valid.
    let revalidation = consumed.revalidation();
    assert_eq!(
        revalidation.keys_of(Revalidation::Recompute),
        vec!["claim", "cohort", "fit", "model"]
    );
    assert_eq!(revalidation.keys_of(Revalidation::StillValid), vec!["sibling"]);

    assert_eq!(meta.version, COMPOSITION_VERIFICATION_ARTIFACT_VERSION);
    assert_eq!(meta.feature, COMPOSITION_VERIFICATION_ARTIFACT_FEATURE);
    let mut newer = meta.clone();
    newer.version += 1;
    CompositionVerificationArtifact::reseal_identity(&mut newer);
    assert!(matches!(
        CompositionVerificationArtifact::from_meta(&newer, None),
        Err(CompositionVerificationArtifactError::UnsupportedVersion { version: 2 })
    ));
    let mut feature = meta;
    feature.required_features.push("future_feature".into());
    CompositionVerificationArtifact::reseal_identity(&mut feature);
    assert!(CompositionVerificationArtifact::from_meta(&feature, None).is_err());
}

// -- compact runtime export and cohort decision source --------------------------------------

fn wire(
    id: &str,
    role: &str,
    units: &str,
    functional: &str,
    regime: &str,
) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: units.into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// `tau(z) = 2 + 0.5 z` on `z` in `[-1, 1]`.
fn export() -> CompactExport {
    CompactExport::build(ExportSpec {
        response: wire("y", "outcome", "mmHg", CATE_FUNCTIONAL_ID, "contrast:do(t=1)-do(t=0)"),
        inputs: vec![ExportInput {
            quantity: wire("z", "covariate", "1", "mean", "observed"),
            support: InputSupport::Range { lo: -1.0, hi: 1.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "z".into() }],
        coefficients: vec![2.0, 0.5],
        covariance: vec![0.04, -0.01, -0.01, 0.09],
        mask: vec![],
    })
    .expect("a valid export")
}

/// Cohort `z = (-1, 0.5, 1)` with weights `(0.2, 0.3, 0.5)`: weighted mean contrast
/// `2 + 0.5 * 0.45 = 2.225`.
fn cohort_source() -> CohortDecisionSource {
    let model = RetainedEffectModel::from_compact_export(&export()).unwrap();
    let cohort = CohortBinding::new(
        "clinic-2026-q1",
        "target",
        vec!["r0".into(), "r1".into(), "r2".into()],
        vec!["z".into()],
        vec![CohortColumn::Numeric(vec![-1.0, 0.5, 1.0])],
        vec![0.2, 0.3, 0.5],
    )
    .unwrap();
    CohortDecisionSource::build(
        &model,
        &cohort,
        "mmHg",
        &CohortOptions::default(),
        &ExecutionContext::for_tests(3),
    )
    .unwrap()
}

const EXPORT_GOLDEN: &str = "compact_runtime_export_v1.bin";
const COHORT_GOLDEN: &str = "cohort_decision_source_v1.bin";

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_compact_runtime_export_golden() {
    write(EXPORT_GOLDEN, &export().to_bytes("golden-compact-export").unwrap());
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_cohort_decision_source_golden() {
    write(COHORT_GOLDEN, &cohort_source().to_bytes("golden-cohort-decision").unwrap());
}

#[test]
fn the_compact_runtime_export_golden_is_consumed_and_gated() {
    let bytes = load(EXPORT_GOLDEN);
    assert_eq!(container_kind(&bytes).as_deref(), Some(COMPACT_EXPORT_ARTIFACT_KIND));
    let identity = export().identity().to_owned();
    let consumed = CompactExport::consume(&bytes, &ExportLimits::default(), &identity)
        .expect("the golden verifies under the identity a producer derives today");
    assert_eq!(consumed.identity(), identity);
    // Another retained identity is refused.
    let error = CompactExport::consume(&bytes, &ExportLimits::default(), &"0".repeat(64))
        .expect_err("another identity");
    assert_eq!(error.detail, "compact_export.identity_unexpected");
    // A newer body version is refused before the body is interpreted.
    let newer = with_body_version(
        &bytes,
        COMPACT_EXPORT_ARTIFACT_KIND,
        COMPACT_EXPORT_SECTION,
        ExportLimits::default().max_bytes,
        2,
    );
    let error = CompactExport::consume(&newer, &ExportLimits::default(), &identity)
        .expect_err("a newer version");
    assert_eq!(error.detail, "compact_export.unsupported_version");
}

#[test]
fn the_cohort_decision_source_golden_is_consumed_and_gated() {
    let bytes = load(COHORT_GOLDEN);
    assert_eq!(container_kind(&bytes).as_deref(), Some(COHORT_DECISION_ARTIFACT_KIND));
    let fresh = cohort_source();
    let expectation = CohortExpectation::for_identity(fresh.identity());
    let consumed = CohortDecisionSource::consume(&bytes, &expectation)
        .expect("a fresh consumer rebuilds the model and the cohort and recomputes the mean");
    assert_eq!(consumed.identity(), fresh.identity());
    assert!((consumed.mean_contrast() - 2.225).abs() < 1e-12);
    // Another retained identity is refused.
    let other = CohortExpectation::for_identity("0".repeat(64));
    assert!(CohortDecisionSource::consume(&bytes, &other).is_err());
    // A newer body version is refused as an unsupported version, not a field error.
    let newer = with_body_version(
        &bytes,
        COHORT_DECISION_ARTIFACT_KIND,
        COHORT_DECISION_SECTION,
        64 * 1024 * 1024,
        2,
    );
    let error = CohortDecisionSource::consume(&newer, &expectation).expect_err("newer");
    assert_eq!(error.detail, "cohort_decision.unsupported_version");
}
