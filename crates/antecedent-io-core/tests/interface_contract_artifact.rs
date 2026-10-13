//! Interface contract artifact: a fresh consumer rebuilds the declaration,
//! recomputes both identities and refuses edits, newer versions, unknown
//! features and declarations the library refuses; and the stored metadata is
//! readable as plain CBOR without the library's types.

use antecedent_contract::evidence_ledger::SourceId;
use antecedent_contract::interface_contract::{
    CalibrationRequirement, CalibrationStatus, ConsumedPort, ContractOperation,
    DependenceRequirement, InterfaceContract, ProducedPort, SupportDomain,
};
use antecedent_contract::{DistributionMeaning, QuantityRole, ScientificQuantity};
use antecedent_core::SemanticDigest;
use antecedent_io_core::interface_contract_artifact::{
    INTERFACE_CONTRACT_ARTIFACT_FEATURE, InterfaceContractArtifact, InterfaceContractArtifactError,
    InterfaceContractMeta,
};
use antecedent_io_core::reader::ArtifactReader;
use antecedent_io_core::{ArtifactContainer, CONTENT_CBOR, OutSection, container_kind};

fn quantity(units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:systolic".into(),
        variable_name: "systolic".into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "adults".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn contract() -> InterfaceContract {
    let model = "ab".repeat(32);
    InterfaceContract {
        component: "estimator".into(),
        consumes: vec![
            ConsumedPort::new(
                "data",
                quantity("mmHg"),
                "estimand:ate",
                SupportDomain::interval(0.0, 300.0),
                [ContractOperation::Mean],
            )
            .with_dependence(DependenceRequirement::DeclaredEvidence),
        ],
        produces: vec![
            ProducedPort::new(
                "effect",
                quantity("mmHg"),
                "estimand:ate",
                SupportDomain::Real,
                [ContractOperation::Mean, ContractOperation::Interval],
            )
            .with_uncertainty(DistributionMeaning::EstimatorSampling)
            .with_calibration(CalibrationStatus::Measured { record_id: "cal-0001".into() })
            .with_measurement_models([model])
            .with_evidence([SourceId::from_digest(SemanticDigest::from_bytes([3; 32]))])
            .with_provenance(["causal_contract", "claim"]),
            ProducedPort::new(
                "arm",
                quantity("mmHg"),
                "estimand:arm_mean",
                SupportDomain::Categorical { levels: vec!["a".into(), "b".into()] },
                [ContractOperation::Point],
            )
            .with_calibration(CalibrationStatus::PointOnly),
        ],
    }
}

fn meta_of(contract: &InterfaceContract) -> InterfaceContractMeta {
    InterfaceContractArtifact::seal(contract).expect("seals").1
}

fn bytes_of(meta: &InterfaceContractMeta) -> Vec<u8> {
    InterfaceContractArtifact::meta_to_bytes(meta, "contract-0001").expect("encodes")
}

#[test]
fn a_sealed_contract_round_trips_with_stable_identities() {
    let original = contract();
    let bytes = InterfaceContractArtifact::to_bytes(&original, "contract-0001").expect("seals");
    assert_eq!(container_kind(&bytes).as_deref(), Some("interface_contract_v1"));
    let consumed = InterfaceContractArtifact::from_bytes(&bytes, None).expect("consumes");
    // Ports come back in canonical (name) order; the declaration is otherwise equal.
    assert_eq!(consumed.contract().component, "estimator");
    assert_eq!(consumed.contract().produces.len(), 2);
    assert_eq!(consumed.contract().produces[0].name, "arm");
    assert_eq!(consumed.contract_identity(), original.canonical_identity().to_hex());
    // Retaining the identity independently is accepted, and a different one is refused.
    let identity = consumed.identity().to_owned();
    assert!(InterfaceContractArtifact::from_bytes(&bytes, Some(&identity)).is_ok());
    assert_eq!(
        InterfaceContractArtifact::from_bytes(&bytes, Some(&"0".repeat(64))).unwrap_err(),
        InterfaceContractArtifactError::IdentityMismatch { field: "retained_identity" }
    );
    // The artifact is deterministic.
    let again = InterfaceContractArtifact::to_bytes(&original, "contract-0001").expect("seals");
    assert_eq!(bytes, again);
}

#[test]
fn the_metadata_is_inspectable_as_plain_cbor_without_the_library_types() {
    let bytes = InterfaceContractArtifact::to_bytes(&contract(), "contract-0001").expect("seals");
    let mut reader =
        ArtifactReader::open_seek(std::io::Cursor::new(bytes.as_slice())).expect("container");
    let section = reader.load_section("interface_contract_meta").expect("meta section");
    let value: ciborium::Value = ciborium::from_reader(section.as_bytes()).expect("one CBOR item");
    let ciborium::Value::Map(entries) = value else { panic!("a map") };
    let get = |map: &[(ciborium::Value, ciborium::Value)], key: &str| {
        map.iter()
            .find(|(k, _)| k.as_text() == Some(key))
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| panic!("key {key}"))
    };
    assert_eq!(get(&entries, "version").as_integer().map(i128::from), Some(1));
    assert_eq!(get(&entries, "feature").as_text(), Some(INTERFACE_CONTRACT_ARTIFACT_FEATURE));
    let ciborium::Value::Map(contract_entries) = get(&entries, "contract") else {
        panic!("contract map")
    };
    assert_eq!(get(&contract_entries, "component").as_text(), Some("estimator"));
    let ciborium::Value::Array(produces) = get(&contract_entries, "produces") else {
        panic!("produces array")
    };
    // Ports are sorted by name: `arm` precedes `effect`.
    let ciborium::Value::Map(first) = &produces[0] else { panic!("port map") };
    assert_eq!(get(first, "name").as_text(), Some("arm"));
    let ciborium::Value::Map(quantity_entries) = get(first, "quantity") else {
        panic!("quantity map")
    };
    assert_eq!(get(&quantity_entries, "units").as_text(), Some("mmHg"));
    assert_eq!(get(&entries, "identity").as_text().map(str::len), Some(64));
}

#[test]
fn an_unsealed_edit_is_an_identity_mismatch() {
    let mut meta = meta_of(&contract());
    meta.contract.produces[1].quantity.units = "lb".into();
    assert_eq!(
        InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err(),
        InterfaceContractArtifactError::IdentityMismatch { field: "contract_identity" }
    );
}

#[test]
fn a_resealed_edit_is_caught_by_a_retained_identity() {
    let retained = InterfaceContractArtifact::seal(&contract()).expect("seals").0;
    let retained = retained.identity().to_owned();
    // Another contract, consistently sealed: well formed, so it is accepted on its own ...
    let mut other = contract();
    other.produces[0].quantity.population_id = "children".into();
    let bytes = InterfaceContractArtifact::to_bytes(&other, "contract-0001").expect("seals");
    assert!(InterfaceContractArtifact::from_bytes(&bytes, None).is_ok());
    // ... and refused by a consumer that retained the identity of the original.
    assert_eq!(
        InterfaceContractArtifact::from_bytes(&bytes, Some(&retained)).unwrap_err(),
        InterfaceContractArtifactError::IdentityMismatch { field: "retained_identity" }
    );
}

#[test]
fn a_stale_artifact_identity_is_refused_even_with_the_contract_identity_resealed() {
    let mut meta = meta_of(&contract());
    meta.required_features.clear();
    meta.identity = "0".repeat(64);
    assert_eq!(
        InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err(),
        InterfaceContractArtifactError::IdentityMismatch { field: "identity" }
    );
}

#[test]
fn newer_versions_unknown_features_and_required_features_are_refused() {
    let mut meta = meta_of(&contract());
    meta.version = 2;
    assert_eq!(
        InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err(),
        InterfaceContractArtifactError::UnsupportedVersion { version: 2 }
    );
    let mut meta = meta_of(&contract());
    meta.feature = "interface_contract_v2".into();
    InterfaceContractArtifact::reseal_identity(&mut meta);
    assert_eq!(
        InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err(),
        InterfaceContractArtifactError::UnsupportedSemantics("feature marker")
    );
    let mut meta = meta_of(&contract());
    meta.required_features = vec!["calibrated_composition".into()];
    InterfaceContractArtifact::reseal_identity(&mut meta);
    assert_eq!(
        InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err(),
        InterfaceContractArtifactError::UnsupportedSemantics("required feature")
    );
}

#[test]
fn a_resealed_declaration_the_library_refuses_is_refused_with_its_detail() {
    // An interval operation with no stated meaning, consistently resealed.
    let mut meta = meta_of(&contract());
    meta.contract.produces[1].uncertainty = None;
    InterfaceContractArtifact::reseal_identity(&mut meta);
    let error = InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err();
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    assert_eq!(code, "interface_contract_invalid");
    assert_eq!(detail, "interface_contract.uncertainty_meaning_required");

    // A stated paradigm that is not the meaning's paradigm.
    let mut meta = meta_of(&contract());
    meta.contract.produces[1].uncertainty.as_mut().expect("meaning").paradigm = "posterior".into();
    InterfaceContractArtifact::reseal_identity(&mut meta);
    assert!(matches!(
        InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err(),
        InterfaceContractArtifactError::Malformed(_)
    ));

    // Ports out of canonical order.
    let mut meta = meta_of(&contract());
    meta.contract.produces.reverse();
    InterfaceContractArtifact::reseal_identity(&mut meta);
    assert!(matches!(
        InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err(),
        InterfaceContractArtifactError::Malformed(_)
    ));

    // A consumed port that requires measured calibration without an interval.
    let mut meta = meta_of(&contract());
    meta.contract.consumes[0].calibration = CalibrationRequirement::Measured.as_str().to_owned();
    InterfaceContractArtifact::reseal_identity(&mut meta);
    let error = InterfaceContractArtifact::from_bytes(&bytes_of(&meta), None).unwrap_err();
    assert_eq!(
        error.refusal().map(|(_, detail, _)| detail),
        Some("interface_contract.calibration_without_interval".to_owned())
    );
}

#[test]
fn an_unknown_field_and_a_corrupted_byte_do_not_decode() {
    let meta = meta_of(&contract());
    let mut value = ciborium::Value::serialized(&meta).expect("value");
    if let ciborium::Value::Map(entries) = &mut value {
        entries.push((ciborium::Value::Text("surprise".into()), ciborium::Value::Bool(true)));
    }
    let mut body = Vec::new();
    ciborium::into_writer(&value, &mut body).expect("cbor");
    let bytes = antecedent_io_core::encode_container(
        &ArtifactContainer {
            kind: "interface_contract_v1",
            note: "interface_contract",
            max_bytes: 1 << 20,
            artifact_id: "contract-0001",
            library_version: "0.0.0",
        },
        &[OutSection { id: "interface_contract_meta", content_type: CONTENT_CBOR, bytes: &body }],
    )
    .expect("container");
    assert!(matches!(
        InterfaceContractArtifact::from_bytes(&bytes, None).unwrap_err(),
        InterfaceContractArtifactError::Decode(_)
    ));

    let mut corrupted =
        InterfaceContractArtifact::to_bytes(&contract(), "contract-0001").expect("seals");
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x01;
    assert!(matches!(
        InterfaceContractArtifact::from_bytes(&corrupted, None).unwrap_err(),
        InterfaceContractArtifactError::Decode(_)
    ));
}
