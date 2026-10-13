//! Specification conformance gate.
//!
//! `spec/` holds the normative, versioned specification of the interface
//! contract, evidence ledger and composition bundle formats, a clean-room
//! consumer (`spec/consumer/validate.py`, Python standard library only) and
//! conformance fixtures. This test is the Rust side of that contract:
//!
//! * the fixtures are generated here from real contracts, ledgers and bundles
//!   and must regenerate bit for bit (a change that moves a byte of any fixture
//!   fails here until the specification and fixtures are regenerated on
//!   purpose);
//! * the vocabularies in the schemas and the tables in the specification text
//!   must equal the implementation's;
//! * the clean-room consumer is run as a subprocess over the committed
//!   fixtures, and over mutated copies, which it must reject with the expected
//!   class.
//!
//! Regenerate the fixtures deliberately with
//! `ANTECEDENT_REGENERATE_CONFORMANCE=1 cargo test -p antecedent --test
//! specification_conformance -- --test-threads=1`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, ObservationId, PoolingDependence, SourceDeclaration,
    SourceId, SourceKind, SynthesisKind, SynthesisOperation,
};
use antecedent_contract::interface_contract::{
    CalibrationRequirement, CalibrationStatus, ConsumedPort, ContractOperation,
    DependenceRequirement, InterfaceContract, ProducedPort, SupportDomain, UncertaintyMeaning,
    UncertaintyParadigm,
};
use antecedent_contract::{
    CompositionStage, DistributionMeaning, QuantityRole, ScientificQuantity,
};
use antecedent_core::SemanticDigest;
use antecedent_design::composition_bundle::{
    BundleLimits, BundleNode, CompositionBundle, EvidenceRelationship, FACT_LAW, FACT_REQUIRES_LAW,
    FACT_TRUST, LAW_JOINT, LAW_MEAN_ONLY, NodeKind, ProviderOrData,
};
use antecedent_design::composition_verifiers::evidence_relationship_bytes;
use antecedent_io::container::{
    ArtifactManifest, CompressPolicy, EncodedArtifact, SectionBytes, section_descriptor_with_policy,
};
use antecedent_io::evidence_ledger_artifact::{
    EvidenceLedgerArtifact, SynthesisMethod, SynthesisRecord, covariance_digest,
    covariance_has_cross_terms,
};
use antecedent_io::interface_contract_artifact::{
    INTERFACE_CONTRACT_ARTIFACT_FEATURE, INTERFACE_CONTRACT_ARTIFACT_TAG,
    INTERFACE_CONTRACT_ARTIFACT_VERSION, InterfaceContractArtifact,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::wire::{ArtifactKind, SemanticVersion};
use ciborium::Value;
use serde_json::json;

const IC_KIND: &str = "interface_contract_v1";
const EL_KIND: &str = "evidence_ledger_v1";
const CB_KIND: &str = "composition_bundle_v1";
const IC_SECTION: &str = "interface_contract_meta";
const EL_SECTION: &str = "evidence_ledger_meta";
const CB_SECTION: &str = "bundle.body";

fn spec_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec")
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

// -- container helpers ----------------------------------------------------------------------

type Part = (String, String, Vec<u8>);

fn parts(bytes: &[u8]) -> (ArtifactManifest, Vec<Part>) {
    let encoded = EncodedArtifact::read_from(bytes).expect("a container");
    let sections = encoded
        .manifest
        .sections
        .iter()
        .zip(encoded.sections.iter())
        .map(|(descriptor, section)| {
            (descriptor.id.clone(), descriptor.content_type.clone(), section.data.to_vec())
        })
        .collect();
    (encoded.manifest, sections)
}

/// Write a container with every section uncompressed and a fixed library
/// version, so fixtures do not move with the crate version or the compressor.
fn assemble(mut manifest: ArtifactManifest, sections: Vec<Part>) -> Vec<u8> {
    manifest.library_version = SemanticVersion { major: 0, minor: 0, patch: 0 };
    manifest.sections = sections
        .iter()
        .map(|(id, content_type, data)| {
            section_descriptor_with_policy(
                id.clone(),
                content_type.clone(),
                data,
                CompressPolicy::Never,
            )
        })
        .collect();
    let encoded = EncodedArtifact {
        manifest,
        sections: sections.into_iter().map(|(id, _, data)| SectionBytes::new(id, data)).collect(),
    };
    let mut out = Vec::new();
    encoded.write_to(&mut out).expect("writes");
    out
}

fn normalize(bytes: &[u8]) -> Vec<u8> {
    let (manifest, sections) = parts(bytes);
    assemble(manifest, sections)
}

enum Step<'a> {
    Field(&'a str),
    Index(usize),
}

fn try_at<'v>(value: &'v mut Value, path: &[Step<'_>]) -> Option<&'v mut Value> {
    let mut current = value;
    for step in path {
        current = match (step, current) {
            (Step::Field(name), Value::Map(entries)) => {
                &mut entries.iter_mut().find(|(key, _)| key.as_text() == Some(*name))?.1
            }
            (Step::Index(index), Value::Array(items)) => items.get_mut(*index)?,
            _ => return None,
        };
    }
    Some(current)
}

fn at<'v>(value: &'v mut Value, path: &[Step<'_>]) -> &'v mut Value {
    try_at(value, path).expect("the path fits the value")
}

fn edit_section(bytes: &[u8], section: &str, edit: impl FnOnce(&mut Value)) -> Vec<u8> {
    let (manifest, mut sections) = parts(bytes);
    let entry = sections.iter_mut().find(|(id, _, _)| id == section).expect("the section exists");
    let mut value: Value = ciborium::from_reader(entry.2.as_slice()).expect("one CBOR item");
    edit(&mut value);
    let mut encoded = Vec::new();
    ciborium::into_writer(&value, &mut encoded).expect("encodes");
    entry.2 = encoded;
    assemble(manifest, sections)
}

fn section_value(bytes: &[u8], section: &str) -> Value {
    let (_, sections) = parts(bytes);
    let entry = sections.iter().find(|(id, _, _)| id == section).expect("the section exists");
    ciborium::from_reader(entry.2.as_slice()).expect("one CBOR item")
}

fn flip_hex(value: &mut Value) {
    let Value::Text(text) = value else { panic!("a text field") };
    let last = text.pop().expect("non-empty");
    text.push(if last == '0' { '1' } else { '0' });
}

fn set_text(value: &mut Value, text: &str) {
    *value = Value::Text(text.to_owned());
}

fn set_int(value: &mut Value, number: u64) {
    *value = Value::Integer(number.into());
}

fn add_field(value: &mut Value) {
    let Value::Map(entries) = value else { panic!("a map") };
    entries.push((Value::Text("surprise".into()), Value::Bool(true)));
}

fn json_of(value: &Value) -> serde_json::Value {
    match value {
        Value::Integer(integer) => {
            let number: i128 = (*integer).into();
            if number >= 0 {
                json!(u64::try_from(number).expect("fits u64"))
            } else {
                json!(i64::try_from(number).expect("fits i64"))
            }
        }
        Value::Float(float) => {
            serde_json::Number::from_f64(*float).map(serde_json::Value::Number).expect("finite")
        }
        Value::Text(text) => json!(text),
        Value::Bool(flag) => json!(flag),
        Value::Null => serde_json::Value::Null,
        Value::Array(items) => serde_json::Value::Array(items.iter().map(json_of).collect()),
        Value::Map(entries) => serde_json::Value::Object(
            entries
                .iter()
                .map(|(key, value)| (key.as_text().expect("text key").to_owned(), json_of(value)))
                .collect(),
        ),
        other => panic!("no JSON rendering for {other:?}"),
    }
}

// -- the real objects ----------------------------------------------------------------------

fn quantity(variable: &str, role: QuantityRole, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: format!("schema:{variable}"),
        variable_name: variable.to_owned(),
        role,
        units: units.to_owned(),
        population_id: "adults_18_plus".into(),
        regime_id: "do(statin=1)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn model_identity() -> String {
    "ab".repeat(32)
}

fn estimator_contract() -> InterfaceContract {
    use ContractOperation as Op;
    let mut effect_quantity = quantity("ldl_change", QuantityRole::Outcome, "mg/dL");
    effect_quantity.conditioning.push(antecedent_contract::QuantityCondition {
        variable_id: "schema:age_band".into(),
        value_id: "40_59".into(),
    });
    let source = |byte: u8| SourceId::from_digest(digest(byte));
    InterfaceContract {
        component: "ldl_effect_estimator".into(),
        produces: vec![
            ProducedPort::new(
                "dose",
                quantity("statin_dose", QuantityRole::Treatment, "mg"),
                "estimand:dose",
                SupportDomain::NonNegative,
                [Op::Point, Op::Mean],
            )
            .with_calibration(CalibrationStatus::PointOnly)
            .with_evidence([source(1)])
            .with_provenance(["causal_contract"]),
            ProducedPort::new(
                "effect",
                effect_quantity,
                "estimand:ate",
                SupportDomain::Real,
                [Op::Mean, Op::Draws, Op::Interval, Op::Posterior],
            )
            .with_uncertainty(DistributionMeaning::CausalFunctionalPosterior)
            .with_calibration(CalibrationStatus::Measured {
                record_id: "calibration-record-0001".into(),
            })
            .with_measurement_models([model_identity()])
            .with_evidence([source(2), source(3)])
            .with_provenance(["causal_contract", "claim", "evidence"]),
            ProducedPort::new(
                "marginal",
                quantity("baseline_risk", QuantityRole::Covariate, "probability"),
                "estimand:baseline_risk",
                SupportDomain::interval(-0.0, 1.0),
                [Op::Law, Op::JointDraws],
            )
            .with_uncertainty(DistributionMeaning::InterventionalPredictive)
            .with_calibration(CalibrationStatus::Unmeasured),
            ProducedPort::new(
                "mediated",
                quantity("adherence", QuantityRole::Mediator, "indicator"),
                "estimand:adherence",
                SupportDomain::Binary,
                [Op::Covariance],
            )
            .with_uncertainty(DistributionMeaning::EstimatorSampling)
            .with_calibration(CalibrationStatus::Inherited { input: "observed".into() }),
            ProducedPort::new(
                "selection",
                quantity("enrolled", QuantityRole::Selection, "indicator"),
                "estimand:enrolment",
                SupportDomain::Categorical { levels: vec!["no".into(), "yes".into()] },
                [Op::Point],
            ),
            ProducedPort::new(
                "utility",
                quantity("net_benefit", QuantityRole::Utility, "qaly"),
                "estimand:net_benefit",
                SupportDomain::Real,
                [Op::Mean, Op::Interval],
            )
            .with_uncertainty(DistributionMeaning::Bootstrap)
            .with_calibration(CalibrationStatus::Unmeasured),
        ],
        consumes: vec![
            ConsumedPort::new(
                "observed",
                quantity("ldl", QuantityRole::Outcome, "mg/dL"),
                "estimand:observed_ldl",
                SupportDomain::interval(0.0, 400.0),
                [Op::Mean],
            )
            .with_measurement_models([model_identity()])
            .with_dependence(DependenceRequirement::DeclaredEvidence)
            .with_provenance(["data"]),
            ConsumedPort::new(
                "policy",
                quantity("statin_dose", QuantityRole::Treatment, "mg"),
                "estimand:dose",
                SupportDomain::NonNegative,
                [Op::Law],
            )
            .with_uncertainty(DistributionMeaning::InterventionalPredictive)
            .with_dependence(DependenceRequirement::DisjointEvidence),
            ConsumedPort::new(
                "response",
                quantity("ldl_change", QuantityRole::Outcome, "mg/dL"),
                "estimand:ate",
                SupportDomain::Real,
                [Op::Interval, Op::Posterior],
            )
            .with_uncertainty(DistributionMeaning::CausalFunctionalPosterior)
            .with_calibration(CalibrationRequirement::Measured)
            .with_provenance(["claim"]),
        ],
    }
}

fn consumer_contract() -> InterfaceContract {
    InterfaceContract {
        component: "dose_decision".into(),
        produces: Vec::new(),
        consumes: vec![ConsumedPort::new(
            "effect",
            quantity("ldl_change", QuantityRole::Outcome, "mg/dL"),
            "estimand:ate",
            SupportDomain::Real,
            [ContractOperation::Mean],
        )],
    }
}

fn ledger_fixture() -> Vec<u8> {
    let rows = |range: std::ops::Range<u32>| -> Vec<ObservationId> {
        range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
    };
    let data = |label: &str, range: std::ops::Range<u32>| {
        SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(rows(range))
    };
    let pooling =
        |label: &str, ids: &[SourceId], dependence: PoolingDependence| CombinationRequest {
            label: label.to_owned(),
            operation: SynthesisOperation::StatisticalPooling { sources: ids.to_vec(), dependence },
            replaces: None,
        };
    let mut ledger = EvidenceLedger::new();
    let s1 = ledger.add_source(data("study_1", 0..40)).expect("s1");
    let s2 = ledger.add_source(data("study_2", 40..90)).expect("s2");
    let s3 = ledger.add_source(data("study_3", 90..120)).expect("s3");
    // A fourth study re-using 10 of study 3's observations: the overlap is detected at entry.
    let s4 = ledger.add_source(data("study_4", 110..150)).expect("s4");
    let vendor = ledger
        .add_source(SourceDeclaration::new("vendor_claim", SourceKind::ExternalClaim))
        .expect("vendor");
    let pooled = ledger
        .combine(pooling("independent_pool", &[s1, s2, s3], PoolingDependence::Independent))
        .expect("pool");
    // y = (1, 3), standard errors (1, 2), correlation 1/2: S = [[1, 1], [1, 4]].
    let covariance_matrix = [1.0, 1.0, 1.0, 4.0];
    let covariance = ledger
        .combine(pooling(
            "covariance_pool",
            &[s3, s4],
            PoolingDependence::DeclaredCovariance {
                covariance: covariance_digest(2, &covariance_matrix),
                has_cross_covariance: covariance_has_cross_terms(2, &covariance_matrix),
            },
        ))
        .expect("covariance pool");
    let envelope = ledger
        .combine(pooling(
            "range_pool",
            &[s3, vendor],
            PoolingDependence::AssumptionRange { rho_lo: 0.0, rho_hi: 0.9 },
        ))
        .expect("range pool");
    let records = vec![
        SynthesisRecord {
            result: pooled,
            sources: vec![s1, s2, s3],
            // Weights 1, 1/2, 1/4: estimate (0 + 1 + 1.5) / 1.75 = 10/7, variance 4/7.
            method: SynthesisMethod::FixedEffectInverseVariance {
                estimates: vec![0.0, 2.0, 6.0],
                variances: vec![1.0, 2.0, 4.0],
            },
        },
        SynthesisRecord {
            result: covariance,
            sources: vec![s3, s4],
            method: SynthesisMethod::FixedEffectGls {
                estimates: vec![1.0, 3.0],
                covariance: covariance_matrix.to_vec(),
            },
        },
        SynthesisRecord {
            result: envelope,
            sources: vec![s3, vendor],
            method: SynthesisMethod::DependenceEnvelope {
                estimates: vec![1.0, 3.0],
                standard_errors: vec![1.0, 2.0],
                rho_lo: 0.0,
                rho_hi: 0.9,
            },
        },
    ];
    EvidenceLedgerArtifact::to_bytes(&ledger, &records, "evidence-ledger-fixture")
        .expect("the ledger seals")
}

fn bundle_parts() -> (CompositionBundle, Vec<u8>) {
    let claim_a = BundleNode::reference(
        "claim_a",
        NodeKind::ExternalClaim,
        "claim-a-identity",
        ProviderOrData::Provider {
            provider_id: "reference_lab".into(),
            snapshot_id: "snapshot-9".into(),
            request_fingerprint: "fingerprint-1".into(),
        },
    )
    .with_fact(FACT_LAW, LAW_MEAN_ONLY)
    .with_fact(FACT_TRUST, "externally_attested")
    .with_inspected("mean", 4.25);
    let claim_b = BundleNode::reference(
        "claim_b",
        NodeKind::ExternalClaim,
        "claim-b-identity",
        ProviderOrData::Data { snapshot_id: "cohort-b".into(), digest: "digest-b".into() },
    )
    .with_fact(FACT_LAW, LAW_JOINT)
    .with_inspected("mean", -0.0);
    let decision = BundleNode::reference(
        "decision",
        NodeKind::DecisionResult,
        "decision-identity",
        ProviderOrData::Data {
            snapshot_id: "decision-snapshot".into(),
            digest: "decision-digest".into(),
        },
    )
    .with_fact(FACT_REQUIRES_LAW, LAW_JOINT)
    .with_inspected("expected_utility", 3.0);
    let relationship = BundleNode::embedded(
        "relationship",
        NodeKind::EvidenceRelationship,
        "relationship-identity",
        evidence_relationship_bytes(EvidenceRelationship::SharedData, "claim_a", "claim_b"),
    );
    let pair = |from: &str, to: &str| (from.to_owned(), to.to_owned());
    let bundle = CompositionBundle::new(
        vec![claim_a, claim_b, decision, relationship],
        &[
            pair("claim_a", "relationship"),
            pair("claim_b", "relationship"),
            pair("claim_a", "decision"),
            pair("claim_b", "decision"),
            pair("relationship", "decision"),
        ],
        &BundleLimits::default(),
    )
    .expect("a bundle");
    let bytes = bundle.to_bytes("composition-fixture", &BundleLimits::default()).expect("bytes");
    (bundle, bytes)
}

// -- fixtures --------------------------------------------------------------------------------

struct Fixture {
    file: String,
    kind: &'static str,
    expect: &'static str,
    reject_class: Option<&'static str>,
    identity: Option<String>,
    meta: Option<serde_json::Value>,
    description: &'static str,
    bytes: Vec<u8>,
}

fn accepted(
    name: &str,
    kind: &'static str,
    section: &str,
    bytes: Vec<u8>,
    identity: String,
    description: &'static str,
) -> Fixture {
    let bytes = normalize(&bytes);
    let meta = json_of(&section_value(&bytes, section));
    Fixture {
        file: format!("{name}.artifact"),
        kind,
        expect: "accept",
        reject_class: None,
        identity: Some(identity),
        meta: Some(meta),
        description,
        bytes,
    }
}

fn rejected(
    name: &str,
    kind: &'static str,
    class: &'static str,
    bytes: Vec<u8>,
    description: &'static str,
) -> Fixture {
    Fixture {
        file: format!("{name}.artifact"),
        kind,
        expect: "reject",
        reject_class: Some(class),
        identity: None,
        meta: None,
        description,
        bytes,
    }
}

fn interface_contract_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let artifact = |contract: &InterfaceContract| {
        let bytes = InterfaceContractArtifact::to_bytes(contract, "interface-contract-fixture")
            .expect("seals");
        let identity = InterfaceContractArtifact::from_bytes(&bytes, None)
            .expect("consumes")
            .identity()
            .to_owned();
        (bytes, identity)
    };
    let (rich, rich_identity) = artifact(&estimator_contract());
    out.push(accepted(
        "interface_contract__estimator",
        IC_KIND,
        IC_SECTION,
        rich.clone(),
        rich_identity,
        "an estimator contract with every role, support kind, operation and calibration label",
    ));
    let (minimal, minimal_identity) = artifact(&consumer_contract());
    out.push(accepted(
        "interface_contract__consumer_only",
        IC_KIND,
        IC_SECTION,
        minimal,
        minimal_identity,
        "a consumer with one port and nothing produced",
    ));
    let rich = normalize(&rich);
    let kind = IC_KIND;
    let edit = |f: &dyn Fn(&mut Value)| edit_section(&rich, IC_SECTION, |v: &mut Value| f(v));
    out.push(rejected(
        "interface_contract__identity_flipped",
        kind,
        "identity",
        edit(&|v: &mut Value| flip_hex(at(v, &[Step::Field("identity")]))),
        "the artifact identity differs in its last hex digit",
    ));
    out.push(rejected(
        "interface_contract__contract_edited_unsealed",
        kind,
        "identity",
        edit(&|v: &mut Value| {
            set_text(
                at(
                    v,
                    &[
                        Step::Field("contract"),
                        Step::Field("produces"),
                        Step::Index(1),
                        Step::Field("quantity"),
                        Step::Field("units"),
                    ],
                ),
                "kg",
            );
        }),
        "the units of a produced quantity were edited and no identity was recomputed",
    ));
    out.push(rejected(
        "interface_contract__newer_version",
        kind,
        "version",
        edit(&|v: &mut Value| set_int(at(v, &[Step::Field("version")]), 2)),
        "a metadata version this specification does not define",
    ));
    out.push(rejected(
        "interface_contract__unknown_feature_marker",
        kind,
        "feature",
        edit(&|v: &mut Value| set_text(at(v, &[Step::Field("feature")]), "interface_contract_v2")),
        "a feature marker this specification does not define",
    ));
    out.push(rejected(
        "interface_contract__required_feature",
        kind,
        "required_feature",
        edit(&|v: &mut Value| {
            *at(v, &[Step::Field("required_features")]) =
                Value::Array(vec![Value::Text("calibrated_composition".into())]);
        }),
        "a required feature that version 1 does not define",
    ));
    out.push(rejected(
        "interface_contract__unknown_field",
        kind,
        "schema",
        edit(&|v: &mut Value| add_field(v)),
        "a field the schema does not list",
    ));
    out.push(rejected(
        "interface_contract__missing_field",
        kind,
        "schema",
        edit(&|v: &mut Value| {
            let Value::Map(entries) = v else { panic!("a map") };
            entries.retain(|(key, _)| key.as_text() != Some("contract_identity"));
        }),
        "a required field is absent",
    ));
    // A resealed declaration the library refuses: an interval offered with no stated meaning.
    let mut meta = InterfaceContractArtifact::inspect(&rich).expect("inspects");
    meta.contract.produces[1].uncertainty = None;
    InterfaceContractArtifact::reseal_identity(&mut meta);
    out.push(rejected(
        "interface_contract__resealed_interval_without_meaning",
        kind,
        "semantic",
        normalize(&InterfaceContractArtifact::meta_to_bytes(&meta, "x").expect("encodes")),
        "an interval or posterior is declared without saying what it means; the artifact identity was resealed",
    ));
    let mut meta = InterfaceContractArtifact::inspect(&rich).expect("inspects");
    meta.contract.produces.reverse();
    InterfaceContractArtifact::reseal_identity(&mut meta);
    out.push(rejected(
        "interface_contract__resealed_ports_unsorted",
        kind,
        "structure",
        normalize(&InterfaceContractArtifact::meta_to_bytes(&meta, "x").expect("encodes")),
        "ports are not in name order; the artifact identity was resealed",
    ));
    let mut corrupted = rich.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x01;
    out.push(rejected(
        "interface_contract__checksum_corrupted",
        kind,
        "checksum",
        corrupted,
        "one payload byte differs from the section checksum",
    ));
    let (mut manifest, sections) = parts(&rich);
    manifest.artifact_kind = ArtifactKind::Other(EL_KIND.into());
    out.push(rejected(
        "interface_contract__wrong_kind",
        kind,
        "kind",
        assemble(manifest, sections),
        "the manifest declares another artifact kind",
    ));
    out
}

fn evidence_ledger_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let bytes = ledger_fixture();
    let identity =
        EvidenceLedgerArtifact::from_bytes(&bytes, None).expect("consumes").identity().to_owned();
    out.push(accepted(
        "evidence_ledger__pooled",
        EL_KIND,
        EL_SECTION,
        bytes.clone(),
        identity,
        "three pooled studies, a covariance pool, an assumption-range envelope and a vendor claim with undeclared observations",
    ));
    let base = normalize(&bytes);
    let kind = EL_KIND;
    let edit = |f: &dyn Fn(&mut Value)| edit_section(&base, EL_SECTION, |v: &mut Value| f(v));
    out.push(rejected(
        "evidence_ledger__identity_flipped",
        kind,
        "identity",
        edit(&|v: &mut Value| flip_hex(at(v, &[Step::Field("identity")]))),
        "the artifact identity differs in its last hex digit",
    ));
    out.push(rejected(
        "evidence_ledger__synthesis_value_edited",
        kind,
        "semantic",
        edit(&|v: &mut Value| {
            set_int(
                at(
                    v,
                    &[
                        Step::Field("syntheses"),
                        Step::Index(0),
                        Step::Field("values"),
                        Step::Index(0),
                    ],
                ),
                1.5_f64.to_bits(),
            );
        }),
        "the pooled estimate was edited to 1.5; the stored value is not the recomputed 10/7",
    ));
    out.push(rejected(
        "evidence_ledger__head_edited",
        kind,
        "structure",
        edit(&|v: &mut Value| set_text(at(v, &[Step::Field("head")]), &"0".repeat(64))),
        "the head is not the identity of the last operation",
    ));
    out.push(rejected(
        "evidence_ledger__operation_identity_edited",
        kind,
        "identity",
        edit(&|v: &mut Value| {
            flip_hex(at(v, &[Step::Field("operations"), Step::Index(0), Step::Field("id")]));
        }),
        "an operation identity was edited; the artifact identity binds it",
    ));
    out.push(rejected(
        "evidence_ledger__newer_version",
        kind,
        "version",
        edit(&|v: &mut Value| set_int(at(v, &[Step::Field("version")]), 2)),
        "a metadata version this specification does not define",
    ));
    out.push(rejected(
        "evidence_ledger__unknown_field",
        kind,
        "schema",
        edit(&|v: &mut Value| add_field(v)),
        "a field the schema does not list",
    ));
    out
}

fn composition_bundle_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let (bundle, bytes) = bundle_parts();
    out.push(accepted(
        "composition_bundle__decision",
        CB_KIND,
        CB_SECTION,
        bytes.clone(),
        bundle.identity().to_owned(),
        "two external claims, an embedded evidence relationship and a decision depending on all three",
    ));
    let base = normalize(&bytes);
    let kind = CB_KIND;
    let edit = |f: &dyn Fn(&mut Value)| edit_section(&base, CB_SECTION, |v: &mut Value| f(v));
    out.push(rejected(
        "composition_bundle__identity_flipped",
        kind,
        "identity",
        edit(&|v: &mut Value| flip_hex(at(v, &[Step::Field("identity")]))),
        "the bundle identity differs in its last hex digit",
    ));
    out.push(rejected(
        "composition_bundle__edge_digest_edited",
        kind,
        "identity",
        edit(&|v: &mut Value| {
            flip_hex(at(
                v,
                &[Step::Field("edges"), Step::Index(0), Step::Field("upstream_digest")],
            ));
        }),
        "an edge carries a digest that is not its upstream node's Merkle digest",
    ));
    out.push(rejected(
        "composition_bundle__fact_edited",
        kind,
        "identity",
        edit(&|v: &mut Value| {
            set_text(
                at(v, &[Step::Field("nodes"), Step::Index(0), Step::Field("facts"), Step::Field("law")]),
                "joint_draw",
            );
        }),
        "a reference node declares an aligned joint law where the sealed bundle declared a mean only",
    ));
    let (manifest, mut sections) = parts(&base);
    for (id, _, data) in &mut sections {
        if id == "embedded.relationship" {
            *data = evidence_relationship_bytes(
                EvidenceRelationship::Independent,
                "claim_a",
                "claim_b",
            );
        }
    }
    out.push(rejected(
        "composition_bundle__embedded_bytes_swapped",
        kind,
        "identity",
        assemble(manifest, sections),
        "the embedded evidence relationship was replaced by an independence declaration under a valid checksum",
    ));
    out.push(rejected(
        "composition_bundle__node_kind_unknown",
        kind,
        "schema",
        edit(&|v: &mut Value| {
            set_text(
                at(v, &[Step::Field("nodes"), Step::Index(0), Step::Field("kind")]),
                "nonsense",
            );
        }),
        "a node kind this specification does not define",
    ));
    out.push(rejected(
        "composition_bundle__newer_version",
        kind,
        "version",
        edit(&|v: &mut Value| set_int(at(v, &[Step::Field("version")]), 2)),
        "a body version this specification does not define",
    ));
    out.push(rejected(
        "composition_bundle__unknown_field",
        kind,
        "schema",
        edit(&|v: &mut Value| add_field(v)),
        "a field the schema does not list",
    ));
    out
}

fn all_fixtures() -> Vec<Fixture> {
    let mut all = interface_contract_fixtures();
    all.extend(evidence_ledger_fixtures());
    all.extend(composition_bundle_fixtures());
    all.sort_by(|a, b| a.file.cmp(&b.file));
    all
}

fn blake3_vectors() -> serde_json::Value {
    let lengths = [
        0_usize, 1, 2, 3, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2048, 2049, 3072, 3073,
        4096, 5120, 7168, 8192, 31744,
    ];
    let vectors: Vec<serde_json::Value> = lengths
        .iter()
        .map(|&len| {
            let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            json!({ "len": len, "hash": blake3::hash(&input).to_hex().to_string() })
        })
        .collect();
    json!({
        "description": "BLAKE3 (unkeyed, 32-byte output) of inputs whose byte i is i mod 251, from the reference implementation",
        "vectors": vectors,
    })
}

fn pretty(value: &serde_json::Value) -> Vec<u8> {
    let mut text = serde_json::to_string_pretty(value).expect("json");
    text.push('\n');
    text.into_bytes()
}

/// Every file the conformance directory must contain, by name.
fn expected_files() -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut entries = Vec::new();
    for fixture in all_fixtures() {
        let mut entry = json!({
            "file": fixture.file,
            "kind": fixture.kind,
            "expect": fixture.expect,
            "description": fixture.description,
        });
        if let Some(class) = fixture.reject_class {
            entry["reject_class"] = json!(class);
        }
        if let (Some(identity), Some(meta)) = (&fixture.identity, &fixture.meta) {
            let meta_file = fixture.file.replace(".artifact", ".meta.json");
            entry["identity"] = json!(identity);
            entry["meta_json"] = json!(meta_file);
            files.insert(meta_file, pretty(meta));
        }
        files.insert(fixture.file.clone(), fixture.bytes);
        entries.push(entry);
    }
    files.insert("blake3_vectors.json".to_owned(), pretty(&blake3_vectors()));
    files.insert(
        "manifest.json".to_owned(),
        pretty(&json!({
            "format": "antecedent-conformance",
            "specification_version": 1,
            "fixtures": entries,
            "blake3_vectors": "blake3_vectors.json",
        })),
    );
    files
}

fn conformance_dir() -> PathBuf {
    spec_dir().join("conformance")
}

#[test]
fn committed_fixtures_regenerate_bit_for_bit() {
    let expected = expected_files();
    let directory = conformance_dir();
    if std::env::var_os("ANTECEDENT_REGENERATE_CONFORMANCE").is_some() {
        std::fs::create_dir_all(&directory).expect("directory");
        for entry in std::fs::read_dir(&directory).expect("listing") {
            let path = entry.expect("entry").path();
            if path.is_file() {
                std::fs::remove_file(path).expect("removes");
            }
        }
        for (name, bytes) in &expected {
            std::fs::write(directory.join(name), bytes).expect("writes");
        }
        return;
    }
    let mut present = BTreeSet::new();
    for entry in std::fs::read_dir(&directory).unwrap_or_else(|error| {
        panic!(
            "spec/conformance is missing ({error}); generate it with \
             ANTECEDENT_REGENERATE_CONFORMANCE=1"
        )
    }) {
        let path = entry.expect("entry").path();
        if path.is_file() {
            present.insert(path.file_name().expect("name").to_string_lossy().into_owned());
        }
    }
    let wanted: BTreeSet<String> = expected.keys().cloned().collect();
    assert_eq!(
        present, wanted,
        "the committed fixture set differs from the generated set; regenerate deliberately"
    );
    for (name, bytes) in &expected {
        let committed = std::fs::read(directory.join(name)).expect("reads");
        assert!(
            &committed == bytes,
            "fixture {name} differs from what the implementation produces; the specification \
             and the implementation have drifted (regenerate deliberately with \
             ANTECEDENT_REGENERATE_CONFORMANCE=1 and review the diff)"
        );
    }
}

// -- the clean-room consumer, as a subprocess -------------------------------------------------

fn python() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_owned())
}

fn run_consumer(extra: &[&str]) -> (bool, String) {
    let script = spec_dir().join("consumer/validate.py");
    let output = Command::new(python())
        .arg("-I")
        .arg(&script)
        .args(extra)
        .output()
        .unwrap_or_else(|error| panic!("cannot run {}: {error}", python()));
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

#[test]
fn the_clean_room_consumer_accepts_and_rejects_the_committed_fixtures() {
    let (ok, text) = run_consumer(&[]);
    assert!(ok, "the clean-room consumer failed:\n{text}");
    assert!(text.contains("self-test ok"), "{text}");
    for fixture in all_fixtures() {
        assert!(
            text.contains(&format!("ok   {}", fixture.file)),
            "{} not reported:\n{text}",
            fixture.file
        );
    }
}

/// A mutation returns whether it applied; an inapplicable path is skipped, never forced.
type Edit = (&'static str, &'static str, Box<dyn Fn(&mut Value) -> bool>);

fn text_edit(path: Vec<Step<'static>>, text: &'static str) -> Box<dyn Fn(&mut Value) -> bool> {
    Box::new(move |v: &mut Value| match try_at(v, &path) {
        Some(target) => {
            set_text(target, text);
            true
        }
        None => false,
    })
}

fn flip_edit(path: Vec<Step<'static>>) -> Box<dyn Fn(&mut Value) -> bool> {
    Box::new(move |v: &mut Value| match try_at(v, &path) {
        Some(target) => {
            flip_hex(target);
            true
        }
        None => false,
    })
}

fn int_edit(path: Vec<Step<'static>>, number: u64) -> Box<dyn Fn(&mut Value) -> bool> {
    Box::new(move |v: &mut Value| match try_at(v, &path) {
        Some(target) => {
            set_int(target, number);
            true
        }
        None => false,
    })
}

fn mutations(kind: &str) -> Vec<Edit> {
    use Step::{Field as F, Index as I};
    let extra: Box<dyn Fn(&mut Value) -> bool> = Box::new(|v: &mut Value| {
        add_field(v);
        true
    });
    match kind {
        IC_KIND => vec![
            (
                "population",
                "identity",
                text_edit(
                    vec![F("contract"), F("consumes"), I(0), F("quantity"), F("population_id")],
                    "children",
                ),
            ),
            (
                "regime",
                "identity",
                text_edit(
                    vec![F("contract"), F("produces"), I(0), F("quantity"), F("regime_id")],
                    "observational",
                ),
            ),
            ("component", "identity", text_edit(vec![F("contract"), F("component")], "other")),
            (
                "calibration_record",
                "identity",
                text_edit(
                    vec![F("contract"), F("produces"), I(1), F("calibration"), F("record_id")],
                    "calibration-record-9999",
                ),
            ),
            ("contract_identity", "identity", flip_edit(vec![F("contract_identity")])),
            ("version_zero", "version", int_edit(vec![F("version")], 0)),
            ("version_next", "version", int_edit(vec![F("version")], 2)),
            ("feature", "feature", text_edit(vec![F("feature")], "x")),
            ("extra_field", "schema", extra),
            (
                "support_kind",
                "schema",
                text_edit(
                    vec![F("contract"), F("produces"), I(0), F("support"), F("kind")],
                    "complex",
                ),
            ),
        ],
        EL_KIND => vec![
            ("identity", "identity", flip_edit(vec![F("identity")])),
            ("state_digest", "identity", flip_edit(vec![F("state_digest")])),
            ("version", "version", int_edit(vec![F("version")], 2)),
            ("extra_field", "schema", extra),
            ("method", "schema", text_edit(vec![F("syntheses"), I(0), F("method")], "median")),
        ],
        _ => vec![
            ("identity", "identity", flip_edit(vec![F("identity")])),
            ("node_identity", "identity", flip_edit(vec![F("nodes"), I(1), F("identity")])),
            (
                "edge_duplicated",
                "structure",
                text_edit(vec![F("edges"), I(0), F("to")], "relationship"),
            ),
            ("version", "version", int_edit(vec![F("version")], 2)),
            ("extra_field", "schema", extra),
        ],
    }
}

#[test]
fn the_clean_room_consumer_rejects_mutated_copies_of_every_accepted_fixture() {
    let temp = std::env::temp_dir().join(format!("antecedent-conformance-{}", std::process::id()));
    std::fs::create_dir_all(&temp).expect("temp dir");
    let mut count = 0;
    for fixture in all_fixtures().into_iter().filter(|f| f.expect == "accept") {
        let section = match fixture.kind {
            IC_KIND => IC_SECTION,
            EL_KIND => EL_SECTION,
            _ => CB_SECTION,
        };
        // The unmutated copy is accepted through the single-file path with its retained identity.
        let path = temp.join(&fixture.file);
        std::fs::write(&path, &fixture.bytes).expect("writes");
        let identity = fixture.identity.clone().expect("identity");
        let (ok, text) = run_consumer(&[
            "--kind",
            fixture.kind,
            "--retained-identity",
            &identity,
            path.to_str().expect("utf-8 path"),
        ]);
        assert!(ok, "{} should be accepted: {text}", fixture.file);
        assert!(text.contains(&identity), "{text}");
        let (ok, text) = run_consumer(&[
            "--kind",
            fixture.kind,
            "--retained-identity",
            &"0".repeat(64),
            path.to_str().expect("utf-8 path"),
        ]);
        assert!(!ok && text.contains("REJECT identity"), "{text}");
        let mut classes = BTreeSet::new();
        for (name, class, edit) in mutations(fixture.kind) {
            let mut applied = false;
            let mutated = edit_section(&fixture.bytes, section, |v: &mut Value| {
                applied = edit(v);
            });
            if !applied {
                continue;
            }
            classes.insert(class);
            let path = temp.join(format!("{}.{name}.artifact", fixture.file));
            std::fs::write(&path, mutated).expect("writes");
            let (ok, text) = run_consumer(&["--kind", fixture.kind, path.to_str().expect("path")]);
            assert!(!ok, "the mutation {name} of {} was accepted: {text}", fixture.file);
            assert!(
                text.contains(&format!("REJECT {class}")),
                "the mutation {name} of {} should be rejected as {class}: {text}",
                fixture.file
            );
            count += 1;
        }
        // Every accepted fixture still receives each class that makes sense for it.
        for class in ["identity", "version", "schema"] {
            assert!(classes.contains(class), "no applicable {class} mutation for {}", fixture.file);
        }
    }
    assert!(count >= 20, "mutation coverage is {count}");
    let _ = std::fs::remove_dir_all(&temp);
}

// -- drift between the specification and the implementation --------------------------------

fn load_schema(name: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(spec_dir().join(name)).expect("schema");
    serde_json::from_str(&text).expect("valid JSON")
}

fn enum_of(schema: &serde_json::Value, pointer: &str) -> BTreeSet<String> {
    schema
        .pointer(pointer)
        .unwrap_or_else(|| panic!("schema has no {pointer}"))
        .as_array()
        .expect("an enum array")
        .iter()
        .map(|v| v.as_str().expect("a string").to_owned())
        .collect()
}

fn set<S: ToString>(items: impl IntoIterator<Item = S>) -> BTreeSet<String> {
    items.into_iter().map(|s| s.to_string()).collect()
}

#[test]
fn schema_vocabularies_equal_the_implementations() {
    let ic = load_schema("interface-contract.schema.json");
    assert_eq!(
        enum_of(&ic, "/$defs/operations/items/enum"),
        set(ContractOperation::ALL.iter().map(|o| o.as_str()))
    );
    assert_eq!(
        enum_of(&ic, "/$defs/uncertainty/oneOf/1/properties/meaning/enum"),
        set(UncertaintyMeaning::ALL_MEANINGS.iter().map(|m| UncertaintyMeaning::new(*m).name()))
    );
    assert_eq!(
        enum_of(&ic, "/$defs/uncertainty/oneOf/1/properties/paradigm/enum"),
        set(UncertaintyParadigm::ALL.iter().map(|p| p.as_str()))
    );
    let roles = [
        QuantityRole::Treatment,
        QuantityRole::Outcome,
        QuantityRole::Covariate,
        QuantityRole::Mediator,
        QuantityRole::Selection,
        QuantityRole::Utility,
    ];
    assert_eq!(
        enum_of(&ic, "/$defs/quantity/properties/role/enum"),
        set(roles.iter().map(|r| ScientificQuantityWire::from(&quantity("v", *r, "u")).role))
    );
    let stages = [
        CompositionStage::CausalContract,
        CompositionStage::Evidence,
        CompositionStage::Data,
        CompositionStage::ExternalProvider,
        CompositionStage::DistributionArtifact,
        CompositionStage::Transformation,
        CompositionStage::DecisionContract,
        CompositionStage::SensitivityInput,
        CompositionStage::StudyRankingProvider,
        CompositionStage::Claim,
    ];
    assert_eq!(
        enum_of(&ic, "/$defs/producedPort/properties/provenance/items/enum"),
        set(stages.iter().map(|s| s.as_str()))
    );
    assert_eq!(
        enum_of(&ic, "/$defs/consumedPort/properties/provenance/items/enum"),
        set(stages.iter().map(|s| s.as_str()))
    );
    assert_eq!(
        enum_of(&ic, "/$defs/consumedPort/properties/calibration/enum"),
        set(["not_required", "measured"]
            .map(|label| { CalibrationRequirement::from_name(label).expect("known").as_str() }))
    );
    assert_eq!(
        enum_of(&ic, "/$defs/consumedPort/properties/dependence/enum"),
        set(["unconstrained", "declared_evidence", "disjoint_evidence"]
            .map(|label| { DependenceRequirement::from_name(label).expect("known").as_str() }))
    );
    let labels = [
        CalibrationStatus::None,
        CalibrationStatus::PointOnly,
        CalibrationStatus::Unmeasured,
        CalibrationStatus::Measured { record_id: "r".into() },
        CalibrationStatus::Inherited { input: "i".into() },
    ];
    let schema_labels: BTreeSet<String> = ["none", "point_only", "unmeasured"]
        .iter()
        .chain(["measured"].iter())
        .chain(["inherited"].iter())
        .map(|s| (*s).to_owned())
        .collect();
    assert_eq!(set(labels.iter().map(CalibrationStatus::label)), schema_labels);
    assert_eq!(
        ic.pointer("/properties/version/const").and_then(|v| v.as_u64()),
        Some(u64::from(INTERFACE_CONTRACT_ARTIFACT_VERSION))
    );
    assert_eq!(
        ic.pointer("/properties/feature/const").and_then(|v| v.as_str()),
        Some(INTERFACE_CONTRACT_ARTIFACT_FEATURE)
    );

    let el = load_schema("evidence-ledger.schema.json");
    assert_eq!(
        enum_of(&el, "/$defs/source/properties/kind/enum"),
        set(SourceKind::ALL.iter().map(|k| k.as_str()))
    );
    assert_eq!(
        enum_of(&el, "/$defs/combination/properties/kind/enum"),
        set(SynthesisKind::ALL.iter().map(|k| k.as_str()))
    );
    assert_eq!(
        el.pointer("/properties/feature/const").and_then(|v| v.as_str()),
        Some("evidence_ledger_v1")
    );

    let cb = load_schema("composition-bundle.schema.json");
    assert_eq!(
        enum_of(&cb, "/$defs/node/properties/kind/enum"),
        set(NodeKind::ALL.iter().map(|k| k.as_str()))
    );
}

#[test]
fn the_specification_text_states_the_implementations_tables_and_tags() {
    let text = std::fs::read_to_string(spec_dir().join("canonicalization-and-identity.md"))
        .expect("specification text");
    // Every node kind and the stage the link digest writes for it.
    for kind in NodeKind::ALL {
        let row = format!("| `{}` | `{}` |", kind.as_str(), kind.stage().as_str());
        assert!(text.contains(&row), "the specification has no row {row}");
    }
    // Identity domains named in the text.
    for tag in [
        "antecedent.interface_contract",
        INTERFACE_CONTRACT_ARTIFACT_TAG,
        "antecedent.interface_contract.connection",
        "antecedent.interface_contract.composition",
        "antecedent.evidence_ledger.artifact",
        "antecedent.composition_bundle.node.v1",
        "antecedent.composition_provenance.link.v1",
        "antecedent.composition_bundle.v1",
    ] {
        assert!(text.contains(&format!("`{tag}`")), "the specification does not name {tag}");
    }
    // Operation tags and the refusal classes the consumer reports.
    for operation in ContractOperation::ALL {
        let phrase = format!("{} {}", operation.as_str(), operation.tag());
        assert!(text.contains(&phrase), "the specification does not state `{phrase}`");
    }
    for class in [
        "container",
        "checksum",
        "kind",
        "unsupported",
        "version",
        "feature",
        "required_feature",
        "schema",
        "structure",
        "semantic",
        "identity",
    ] {
        assert!(text.contains(&format!("`{class}`")), "the refusal class {class} is not named");
    }
    // The meaning-to-paradigm table.
    for meaning in UncertaintyMeaning::ALL_MEANINGS {
        let meaning = UncertaintyMeaning::new(meaning);
        assert!(
            text.contains(meaning.name()),
            "the specification does not name the meaning {}",
            meaning.name()
        );
    }
    // Every fixture class the manifest uses is a class the text defines.
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(conformance_dir().join("manifest.json")).expect("manifest"),
    )
    .expect("json");
    assert_eq!(manifest["specification_version"], 1);
}

#[test]
fn the_clean_room_consumer_agrees_with_the_implementation_on_every_distinct_identity() {
    // The two implementations compute the same identity for the same declaration, for a
    // contract that exercises every role, support, operation and label ...
    let (accepted, text) = run_consumer(&[
        "--kind",
        IC_KIND,
        conformance_dir().join("interface_contract__estimator.artifact").to_str().expect("path"),
    ]);
    assert!(accepted, "{text}");
    let contract = estimator_contract();
    let identity = contract.canonical_identity().to_hex();
    let sealed = InterfaceContractArtifact::seal(&contract).expect("seals").0;
    assert_eq!(sealed.contract_identity(), identity);
    assert!(text.contains(sealed.identity()), "{text}");
    // ... and the port declaration order never moves an identity.
    let mut reordered = contract.clone();
    reordered.produces.reverse();
    reordered.consumes.reverse();
    assert_eq!(reordered.canonical_identity().to_hex(), identity);
}

#[test]
fn the_clean_room_consumers_vocabulary_tables_equal_the_implementations() {
    let (ok, text) = run_consumer(&["--dump-tables"]);
    assert!(ok, "{text}");
    let tables: serde_json::Value = serde_json::from_str(text.trim()).expect("JSON tables");
    for operation in ContractOperation::ALL {
        assert_eq!(
            tables["operation_tags"][operation.as_str()].as_u64(),
            Some(operation.tag()),
            "operation tag of {}",
            operation.as_str()
        );
    }
    for meaning in UncertaintyMeaning::ALL_MEANINGS {
        let meaning = UncertaintyMeaning::new(meaning);
        assert_eq!(
            tables["meaning_paradigms"][meaning.name()].as_str(),
            Some(meaning.paradigm().as_str()),
            "paradigm of {}",
            meaning.name()
        );
    }
    for kind in NodeKind::ALL {
        assert_eq!(
            tables["kind_stages"][kind.as_str()].as_str(),
            Some(kind.stage().as_str()),
            "stage of {}",
            kind.as_str()
        );
    }
    let object = |name: &str| tables[name].as_object().expect("an object").len();
    assert_eq!(object("operation_tags"), ContractOperation::ALL.len());
    assert_eq!(object("meaning_paradigms"), UncertaintyMeaning::ALL_MEANINGS.len());
    assert_eq!(object("kind_stages"), NodeKind::ALL.len());
    assert_eq!(object("role_tags"), 6);
    assert_eq!(object("calibration_tags"), 5);
}
