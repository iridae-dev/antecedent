//! Selective-discovery artifact: a fresh consumer rebuilds receipts and structure, recomputes
//! every identity and refuses edits, newer versions, unknown features and labels, and a
//! consistently resealed artifact of another run. The stored metadata is readable as plain
//! CBOR without the library's types.
//!
//! Replay against raw data is tested where data exist (the facade crate's
//! `selective_discovery_facade` tests); this crate has no data layer.

use antecedent_discovery::selective::{
    DiscoveredStructure, Mark, ORIENTATION_COMPONENTS, PortableDiscoveryState, SEARCH_COMPONENTS,
    STATISTIC_COMPONENTS, StatisticReceipt, StatisticStore, StructureClass, StructureEdge,
    StructureNode, WorkUnits, digest_hex,
};
use antecedent_io_core::container_kind;
use antecedent_io_core::reader::ArtifactReader;
use antecedent_io_core::selective_discovery_artifact::{
    SELECTIVE_DISCOVERY_ARTIFACT_FEATURE, SELECTIVE_DISCOVERY_INFERENCE,
    SelectiveDiscoveryArtifact, SelectiveDiscoveryArtifactError, SelectiveDiscoveryIntakeLimits,
};

fn state() -> PortableDiscoveryState {
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

fn bytes_of(state: &PortableDiscoveryState) -> Vec<u8> {
    SelectiveDiscoveryArtifact::to_bytes(state, "selective-1").unwrap()
}

#[test]
fn a_sealed_state_round_trips_and_binds_its_identities() {
    let state = state();
    let bytes = bytes_of(&state);
    let artifact = SelectiveDiscoveryArtifact::from_bytes(&bytes, None).unwrap();
    assert_eq!(artifact.state().receipts, state.receipts);
    assert_eq!(artifact.state().structure, state.structure);
    assert_eq!(artifact.state().graph_identity(), state.graph_identity());
    assert_eq!(artifact.state().result_identity, state.result_identity);
    assert_eq!(artifact.state().work, state.work);
    // Deterministic bytes and identity.
    assert_eq!(bytes_of(&state), bytes);
    let retained = artifact.identity().to_owned();
    assert!(SelectiveDiscoveryArtifact::from_bytes(&bytes, Some(&retained)).is_ok());
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes(&bytes, Some(&"0".repeat(64))),
        Err(SelectiveDiscoveryArtifactError::IdentityMismatch { field: "retained_identity" })
    ));
    // Stored metadata is inspectable without raw data and states no recovery claim.
    let meta = SelectiveDiscoveryArtifact::inspect(&bytes).unwrap();
    assert_eq!(meta.inference, SELECTIVE_DISCOVERY_INFERENCE);
    assert_eq!(meta.receipts.len(), 3);
    assert_eq!(meta.graph_identity, digest_hex(&state.graph_identity()));
}

#[test]
fn an_unsealed_edit_is_refused_and_a_resealed_edit_is_caught_by_a_retained_identity() {
    let state = state();
    let bytes = bytes_of(&state);
    let original = SelectiveDiscoveryArtifact::from_bytes(&bytes, None).unwrap();
    let meta = SelectiveDiscoveryArtifact::inspect(&bytes).unwrap();

    // A receipt's p-value changes and nothing else: the receipt identity does not match.
    let mut edited = meta.clone();
    edited.receipts[1].p_value_bits = 0.6_f64.to_bits();
    let err = SelectiveDiscoveryArtifact::from_meta(&edited, None).unwrap_err();
    assert_eq!(err.refusal().unwrap().1, "selective_discovery.receipts_identity");

    // The receipt identity and the artifact identity are recomputed by an editor: the
    // artifact is internally consistent, so only a retained identity (or a replay against
    // raw data) can tell it from the original.
    let mut resealed = state.clone();
    resealed.receipts[1].p_value = 0.6;
    resealed.reseal();
    let resealed_bytes = bytes_of(&resealed);
    let loaded = SelectiveDiscoveryArtifact::from_bytes(&resealed_bytes, None).unwrap();
    assert_ne!(loaded.identity(), original.identity());
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes(&resealed_bytes, Some(original.identity())),
        Err(SelectiveDiscoveryArtifactError::IdentityMismatch { field: "retained_identity" })
    ));
}

#[test]
fn graph_receipt_and_label_mutations_are_refused() {
    let state = state();
    let meta = SelectiveDiscoveryArtifact::inspect(&bytes_of(&state)).unwrap();

    // The stored structure no longer hashes to the stored graph identity.
    let mut graph = meta.clone();
    graph.structure.edges[1].at_a = "arrow".to_owned();
    SelectiveDiscoveryArtifact::reseal_identity(&mut graph);
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_meta(&graph, None),
        Err(SelectiveDiscoveryArtifactError::IdentityMismatch { field: "graph_identity" })
    ));

    // An out-of-range p-value that is perfectly consistent with its identities.
    let mut invalid = state.clone();
    invalid.receipts[0].p_value = 2.0;
    invalid.reseal();
    let mut invalid_meta = meta.clone();
    invalid_meta.receipts[0].p_value_bits = 2.0_f64.to_bits();
    invalid_meta.receipts_identity =
        digest_hex(&StatisticStore::receipts_identity(&invalid.receipts));
    SelectiveDiscoveryArtifact::reseal_identity(&mut invalid_meta);
    let err = SelectiveDiscoveryArtifact::from_meta(&invalid_meta, None).unwrap_err();
    assert_eq!(err.refusal().unwrap().1, "selective_discovery.receipt_invalid");
    assert_eq!(err.refusal().unwrap().0, "discovery_statistics_invalid");

    // A stronger inference label than this format states.
    let mut label = meta.clone();
    label.inference = "recovers_true_graph".to_owned();
    SelectiveDiscoveryArtifact::reseal_identity(&mut label);
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_meta(&label, None),
        Err(SelectiveDiscoveryArtifactError::UnsupportedSemantics("inference label"))
    ));
}

#[test]
fn newer_versions_unknown_features_and_required_features_are_refused() {
    let meta = SelectiveDiscoveryArtifact::inspect(&bytes_of(&state())).unwrap();
    let mut newer = meta.clone();
    newer.version = 2;
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_meta(&newer, None),
        Err(SelectiveDiscoveryArtifactError::UnsupportedVersion { version: 2 })
    ));
    let mut feature = meta.clone();
    feature.feature = "selective_discovery_v2".to_owned();
    SelectiveDiscoveryArtifact::reseal_identity(&mut feature);
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_meta(&feature, None),
        Err(SelectiveDiscoveryArtifactError::UnsupportedSemantics("feature marker"))
    ));
    let mut required = meta;
    required.required_features = vec!["incremental_statistics".to_owned()];
    SelectiveDiscoveryArtifact::reseal_identity(&mut required);
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_meta(&required, None),
        Err(SelectiveDiscoveryArtifactError::UnsupportedSemantics("required feature"))
    ));
}

#[test]
fn incomplete_declarations_and_corruption_are_refused() {
    let state = state();
    let meta = SelectiveDiscoveryArtifact::inspect(&bytes_of(&state)).unwrap();
    let mut missing = meta.clone();
    missing.components.pop();
    SelectiveDiscoveryArtifact::reseal_identity(&mut missing);
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_meta(&missing, None),
        Err(SelectiveDiscoveryArtifactError::Refused { detail, .. })
            if detail == "selective_discovery.contract_identity"
    ));

    let mut bytes = bytes_of(&state);
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes(&bytes, None),
        Err(SelectiveDiscoveryArtifactError::Container(antecedent_io_core::IoError::ChecksumMismatch { section })) if section == "selective_discovery_meta"
    ));
    assert!(matches!(
        SelectiveDiscoveryArtifact::to_bytes(&state, "  "),
        Err(SelectiveDiscoveryArtifactError::Encode(_))
    ));
}

#[test]
fn stored_metadata_is_plain_cbor_readable_without_library_types() {
    let bytes = bytes_of(&state());
    assert_eq!(container_kind(&bytes).as_deref(), Some("selective_discovery_v1"));
    let mut reader =
        ArtifactReader::open_seek(std::io::Cursor::new(bytes.as_slice())).expect("container");
    let section = reader.load_section("selective_discovery_meta").expect("meta section");
    let value: ciborium::Value = ciborium::from_reader(section.as_bytes()).expect("one CBOR item");
    let ciborium::Value::Map(entries) = value else { panic!("meta is a map") };
    let text = |key: &str| {
        entries
            .iter()
            .find(|(k, _)| k.as_text() == Some(key))
            .and_then(|(_, v)| v.as_text().map(str::to_owned))
    };
    assert_eq!(text("feature").as_deref(), Some(SELECTIVE_DISCOVERY_ARTIFACT_FEATURE));
    assert_eq!(text("inference").as_deref(), Some("algorithmic_match_only"));
    assert_eq!(text("route").as_deref(), Some("static_pc"));
    assert_eq!(text("identity").map(|t| t.len()), Some(64));
}

#[test]
fn caller_intake_limits_refuse_what_the_defaults_admit() {
    let state = state();
    let defaults = SelectiveDiscoveryIntakeLimits::default();
    let bytes =
        SelectiveDiscoveryArtifact::to_bytes_with_limits(&state, "selective-1", &defaults).unwrap();
    SelectiveDiscoveryArtifact::from_bytes_with_limits(&bytes, None, &defaults).unwrap();

    let few_receipts = SelectiveDiscoveryIntakeLimits { max_receipts: 2, ..defaults };
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes_with_limits(&bytes, None, &few_receipts),
        Err(SelectiveDiscoveryArtifactError::LimitsExceeded("receipts"))
    ));
    assert!(matches!(
        SelectiveDiscoveryArtifact::to_bytes_with_limits(&state, "selective-1", &few_receipts),
        Err(SelectiveDiscoveryArtifactError::Refused { detail, .. })
            if detail == "selective_discovery.receipt_limit"
    ));
    let few_items = SelectiveDiscoveryIntakeLimits { max_structure_items: 1, ..defaults };
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes_with_limits(&bytes, None, &few_items),
        Err(SelectiveDiscoveryArtifactError::LimitsExceeded("structure"))
    ));
    let few_bytes = SelectiveDiscoveryIntakeLimits { max_bytes: bytes.len() - 1, ..defaults };
    assert!(matches!(
        SelectiveDiscoveryArtifact::from_bytes_with_limits(&bytes, None, &few_bytes),
        Err(SelectiveDiscoveryArtifactError::LimitsExceeded("artifact bytes"))
    ));
    assert!(matches!(
        SelectiveDiscoveryArtifact::to_bytes_with_limits(&state, "selective-1", &few_bytes),
        Err(SelectiveDiscoveryArtifactError::LimitsExceeded("artifact bytes"))
    ));
}
