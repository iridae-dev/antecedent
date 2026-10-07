//! The portable composed result (`composition_bundle_v1`): a graph of embedded and
//! referenced nodes with Merkle edge digests, consumed through per-kind verifiers.
//! The expected decision values are hand-enumerated from four equally likely rows:
//! `risky = E[p * q] = (4 + 0 + 4 + 0) / 4 = 2` and `safe = max(3, 0) = 3`.

use std::collections::BTreeMap;

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::composition_bundle::{
    BundleConsumer, BundleError, BundleLimits, BundleNode, BundleStage, ClaimLabel,
    CompositionBundle, ConsumedBundle, EvidenceRelationship, FACT_LAW, FACT_QUANTITY_DIGEST,
    FACT_REQUIRES_LAW, FACT_SNAPSHOT, FACT_TRUST, LAW_JOINT, LAW_MEAN_ONLY, NodeFailure, NodeKind,
    NodeStatus, NodeVerifier, ProviderOrData, SuppliedSources, UpstreamNode, VerifiedNode,
};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, source_digest,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::evaluate_contract;
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use antecedent_io::reader::ArtifactReader;
use serde_json::{Value, json};

const BODY: &str = "bundle.body";

// ---------------------------------------------------------------------------
// Real embedded artifacts: a joint law and a decision contract/result over it.
// ---------------------------------------------------------------------------

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn columns() -> Vec<ScientificQuantity> {
    vec![quantity("p", "do(a=1)"), quantity("q", "do(a=1)"), quantity("safe", "do(a=0)")]
}

fn source_identity() -> DistributionIdentity {
    DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

fn source() -> DistributionArtifact {
    let p = [1.0, 3.0, 2.0, 0.0];
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity: source_identity(),
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn contract() -> DecisionContract {
    let cols = columns();
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![cols[0].clone(), cols[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "safe".into(),
                kind: ActionKind::Policy,
                inputs: vec![cols[2].clone()],
                utility: UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![HardConstraint {
            id: "q-cap".into(),
            expr: UtilityExpr::Input(1),
            bound: 5.0,
            min_probability: 0.75,
            units: "units".into(),
            applies_to: vec!["risky".into()],
        }],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn quantity_text(art: &DistributionArtifact) -> String {
    art.quantities().iter().map(|q| format!("{q:?}")).collect::<Vec<_>>().join(";")
}

fn text_digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn failure(stage: BundleStage, reason: &str) -> NodeFailure {
    NodeFailure { stage, reason: reason.to_owned() }
}

// ---------------------------------------------------------------------------
// Verifiers registered per node kind: each decodes through the artifact's own
// consumer, so the bundle module depends on none of them.
// ---------------------------------------------------------------------------

struct ContractVerifier {
    retained: String,
}

impl NodeVerifier for ContractVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::DecisionContract
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let artifact = DecisionContractArtifact::from_bytes(bytes, &self.retained)
            .map_err(|e| failure(BundleStage::SwappedEvidence, &format!("{e:?}")))?;
        Ok(VerifiedNode {
            identity: artifact.identity().to_owned(),
            facts: BTreeMap::new(),
            values: vec![],
        })
    }
}

struct DistributionVerifier {
    retained: DistributionIdentity,
}

impl NodeVerifier for DistributionVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::Distribution
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let art = DistributionArtifact::from_bytes(bytes, &self.retained)
            .map_err(|e| failure(BundleStage::SwappedEvidence, &format!("{e:?}")))?;
        let identity = &art.metadata().identity;
        let law =
            if identity.alignment == DrawAlignment::Joint { LAW_JOINT } else { LAW_MEAN_ONLY };
        let mut facts = BTreeMap::new();
        facts.insert(FACT_LAW.to_owned(), law.to_owned());
        facts.insert(FACT_QUANTITY_DIGEST.to_owned(), text_digest(quantity_text(&art).as_bytes()));
        facts.insert(
            FACT_SNAPSHOT.to_owned(),
            format!("{}|{}", identity.snapshot_id, identity.causal_contract_id),
        );
        Ok(VerifiedNode { identity: source_digest(&art), facts, values: vec![] })
    }
}

struct ResultVerifier;

impl NodeVerifier for ResultVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::DecisionResult
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let find = |kind: NodeKind| upstream.iter().find(|u| u.kind == kind).map(|u| u.identity);
        let (Some(contract_id), Some(source_id)) =
            (find(NodeKind::DecisionContract), find(NodeKind::Distribution))
        else {
            return Err(failure(
                BundleStage::GraphOrSnapshotMismatch,
                "the result's contract or source is not verified upstream",
            ));
        };
        let artifact = DecisionResultArtifact::from_bytes(bytes, contract_id, source_id)
            .map_err(|e| failure(BundleStage::GraphOrSnapshotMismatch, &format!("{e:?}")))?;
        let mut facts = BTreeMap::new();
        facts.insert(FACT_REQUIRES_LAW.to_owned(), LAW_JOINT.to_owned());
        let values = artifact
            .result()
            .actions
            .iter()
            .map(|a| (format!("{}.expected_utility", a.id), a.expected_utility))
            .collect();
        Ok(VerifiedNode { identity: text_digest(bytes), facts, values })
    }
}

struct QuantityVerifier;

impl NodeVerifier for QuantityVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::QuantityCoordinates
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let digest = text_digest(bytes);
        let mut facts = BTreeMap::new();
        facts.insert(FACT_QUANTITY_DIGEST.to_owned(), digest.clone());
        Ok(VerifiedNode { identity: digest, facts, values: vec![] })
    }
}

struct EvidenceVerifier;

impl NodeVerifier for EvidenceVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::EvidenceRelationship
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let name = std::str::from_utf8(bytes)
            .map_err(|_| failure(BundleStage::SwappedEvidence, "not text"))?;
        EvidenceRelationship::from_name(name)
            .ok_or_else(|| failure(BundleStage::SwappedEvidence, "unknown relationship"))?;
        Ok(VerifiedNode { identity: text_digest(bytes), facts: BTreeMap::new(), values: vec![] })
    }
}

// ---------------------------------------------------------------------------
// The bundle fixture
// ---------------------------------------------------------------------------

struct Fixture {
    contract_bytes: Vec<u8>,
    contract_identity: String,
    dist_bytes: Vec<u8>,
    dist_identity: String,
    result_bytes: Vec<u8>,
    quantity: Vec<u8>,
    evidence: Vec<u8>,
    evidence_identity: String,
    snapshot_fact: String,
}

impl Fixture {
    fn new() -> Self {
        let c = contract();
        let s = source();
        let result = evaluate_contract(&c, &s).unwrap();
        let contract_artifact = DecisionContractArtifact::new(c).unwrap();
        let evidence = b"shared_data".to_vec();
        Self {
            contract_bytes: contract_artifact.to_bytes("contract").unwrap(),
            contract_identity: contract_artifact.identity().to_owned(),
            dist_bytes: s.to_bytes("distribution").unwrap(),
            dist_identity: source_digest(&s),
            result_bytes: DecisionResultArtifact::new(result, &s).to_bytes("result").unwrap(),
            quantity: quantity_text(&s).into_bytes(),
            evidence_identity: text_digest(&evidence),
            evidence,
            snapshot_fact: "enumeration-1|checked-contract".into(),
        }
    }

    fn nodes(&self) -> Vec<BundleNode> {
        vec![
            BundleNode::reference(
                "causal",
                NodeKind::CausalContract,
                "program-1",
                ProviderOrData::Data {
                    snapshot_id: "enumeration-1".into(),
                    digest: "snapshot-digest".into(),
                },
            )
            .with_fact(FACT_SNAPSHOT, &self.snapshot_fact),
            BundleNode::embedded(
                "contract",
                NodeKind::DecisionContract,
                &self.contract_identity,
                self.contract_bytes.clone(),
            ),
            BundleNode::embedded(
                "distribution",
                NodeKind::Distribution,
                &self.dist_identity,
                self.dist_bytes.clone(),
            ),
            BundleNode::embedded(
                "evidence",
                NodeKind::EvidenceRelationship,
                &self.evidence_identity,
                self.evidence.clone(),
            ),
            BundleNode::embedded(
                "quantity",
                NodeKind::QuantityCoordinates,
                &text_digest(&self.quantity),
                self.quantity.clone(),
            ),
            BundleNode::embedded(
                "result",
                NodeKind::DecisionResult,
                &text_digest(&self.result_bytes),
                self.result_bytes.clone(),
            ),
        ]
    }

    fn bundle(&self) -> CompositionBundle {
        CompositionBundle::new(self.nodes(), &deps(), &BundleLimits::default()).unwrap()
    }

    fn consumer(&self) -> BundleConsumer {
        BundleConsumer::new()
            .register(Box::new(ContractVerifier { retained: self.contract_identity.clone() }))
            .register(Box::new(DistributionVerifier { retained: source_identity() }))
            .register(Box::new(ResultVerifier))
            .register(Box::new(QuantityVerifier))
            .register(Box::new(EvidenceVerifier))
    }
}

fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect()
}

fn deps() -> Vec<(String, String)> {
    pairs(&[
        ("causal", "distribution"),
        ("quantity", "distribution"),
        ("distribution", "result"),
        ("contract", "result"),
        ("evidence", "result"),
    ])
}

fn snapshot_supplied() -> SuppliedSources {
    SuppliedSources::default().with_data("enumeration-1", "snapshot-digest")
}

fn consume(
    f: &Fixture,
    bytes: &[u8],
    expected: &str,
    supplied: &SuppliedSources,
) -> Result<ConsumedBundle, BundleError> {
    f.consumer().consume(bytes, &BundleLimits::default(), expected, supplied)
}

fn stage_of(result: Result<ConsumedBundle, BundleError>) -> BundleStage {
    match result {
        Err(error) => error.stage().expect("a structured refusal"),
        Ok(_) => panic!("the bundle was consumed"),
    }
}

fn status_of(consumed: &ConsumedBundle, id: &str) -> NodeStatus {
    consumed.node(id).unwrap().status.clone()
}

fn failed_stage(consumed: &ConsumedBundle, id: &str) -> BundleStage {
    match status_of(consumed, id) {
        NodeStatus::Failed { stage, .. } => stage,
        other => panic!("node `{id}` is {other:?}"),
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

/// Edit the body and the section payloads, then re-wrap in a fresh, valid
/// container: a mutation the outer checksum cannot detect.
fn reseal(bytes: &[u8], edit: impl FnOnce(&mut Value, &mut BTreeMap<String, Vec<u8>>)) -> Vec<u8> {
    let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes)).unwrap();
    let ids: Vec<String> = reader.manifest().sections.iter().map(|s| s.id.clone()).collect();
    let mut blobs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for id in &ids {
        blobs.insert(id.clone(), reader.load_section(id).unwrap().as_bytes().to_vec());
    }
    let manifest = reader.manifest().clone();
    let mut body: Value = from_cbor(&blobs[BODY]).unwrap();
    edit(&mut body, &mut blobs);
    blobs.insert(BODY.to_owned(), to_cbor(&body).unwrap());
    let descriptors = ids
        .iter()
        .map(|id| {
            let content = if id == BODY { "application/cbor" } else { "application/octet-stream" };
            section_descriptor(id.clone(), content, &blobs[id])
        })
        .collect();
    let sections = ids.iter().map(|id| SectionBytes::new(id.clone(), blobs[id].clone())).collect();
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest { sections: descriptors, ..manifest },
        sections,
    };
    let mut out = Vec::new();
    encoded.write_to(&mut out).unwrap();
    out
}

fn node_mut<'a>(body: &'a mut Value, id: &str) -> &'a mut Value {
    body["nodes"].as_array_mut().unwrap().iter_mut().find(|n| n["id"] == id).unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn c3_bundle_round_trip_verifies_every_node_and_reports_the_hand_computed_decision() {
    let f = Fixture::new();
    let bundle = f.bundle();
    let bytes = bundle.to_bytes("bundle-1", &BundleLimits::default()).unwrap();
    let consumed = consume(&f, &bytes, bundle.identity(), &snapshot_supplied()).unwrap();
    consumed.require_verified().unwrap();
    assert!(consumed.all_verified());
    assert_eq!(consumed.identity(), bundle.identity());
    assert_eq!(consumed.edges(), bundle.edges());
    close(consumed.value("result", "safe.expected_utility").unwrap(), 3.0);
    close(consumed.value("result", "risky.expected_utility").unwrap(), 2.0);
    assert_eq!(consumed.node("result").unwrap().claim_label, Some(ClaimLabel::JointDraw));
    assert_eq!(consumed.node("result").unwrap().claim_label.unwrap().as_str(), "joint_draw");
    for node in consumed.nodes() {
        assert_eq!(Some(node.chain_digest.as_str()), bundle.chain_digest(&node.id));
    }
}

#[test]
fn c3_bundle_identity_is_invariant_to_node_and_edge_declaration_order() {
    let f = Fixture::new();
    let a = f.bundle();
    let mut nodes = f.nodes();
    nodes.reverse();
    let mut dependencies = deps();
    dependencies.reverse();
    let b = CompositionBundle::new(nodes, &dependencies, &BundleLimits::default()).unwrap();
    assert_eq!(a.identity(), b.identity());
    assert_eq!(
        a.to_bytes("bundle", &BundleLimits::default()).unwrap(),
        b.to_bytes("bundle", &BundleLimits::default()).unwrap()
    );
}

#[test]
fn c3_bundle_changed_upstream_identity_cascades_into_every_dependent_digest() {
    let f = Fixture::new();
    let base = f.bundle();
    let mut changed = Fixture::new();
    changed.dist_identity = "another-law".into();
    let other = changed.bundle();
    assert_ne!(base.identity(), other.identity());
    for same in ["causal", "contract", "evidence", "quantity"] {
        assert_eq!(base.chain_digest(same), other.chain_digest(same), "{same}");
    }
    for different in ["distribution", "result"] {
        assert_ne!(base.chain_digest(different), other.chain_digest(different), "{different}");
    }
    // A change at the root reaches the sink through the distribution.
    let mut root = Fixture::new();
    root.snapshot_fact = "elsewhere|checked-contract".into();
    let rooted = root.bundle();
    assert_ne!(base.chain_digest("causal"), rooted.chain_digest("causal"));
    assert_ne!(base.chain_digest("distribution"), rooted.chain_digest("distribution"));
    assert_ne!(base.chain_digest("result"), rooted.chain_digest("result"));
    assert_eq!(base.chain_digest("contract"), rooted.chain_digest("contract"));
}

#[test]
fn c3_bundle_resealed_container_edits_are_refused_at_their_stage() {
    let f = Fixture::new();
    let bundle = f.bundle();
    let honest = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let id = bundle.identity().to_owned();
    let supplied = snapshot_supplied();

    let version = reseal(&honest, |body, _| body["version"] = json!(2));
    assert_eq!(stage_of(consume(&f, &version, &id, &supplied)), BundleStage::IncompatibleVersion);

    let kind = reseal(&honest, |body, _| node_mut(body, "quantity")["kind"] = json!("bogus"));
    assert_eq!(stage_of(consume(&f, &kind, &id, &supplied)), BundleStage::UnknownNodeKind);

    // An upstream node's identity edited: its dependents' stated digests no longer hold.
    let upstream = reseal(&honest, |body, _| {
        node_mut(body, "contract")["identity"] = json!("0".repeat(64));
    });
    assert_eq!(stage_of(consume(&f, &upstream, &id, &supplied)), BundleStage::EdgeDigestMismatch);

    // Embedded bytes edited under a fresh checksum.
    let embedded = reseal(&honest, |_, blobs| {
        if let Some(bytes) = blobs.get_mut("embedded.contract") {
            bytes[0] ^= 1;
        }
    });
    assert_eq!(stage_of(consume(&f, &embedded, &id, &supplied)), BundleStage::EdgeDigestMismatch);

    // A stated edge digest edited.
    let edge = reseal(&honest, |body, _| {
        body["edges"][0]["upstream_digest"] = json!("0".repeat(64));
    });
    assert_eq!(stage_of(consume(&f, &edge, &id, &supplied)), BundleStage::EdgeDigestMismatch);

    // A sink node edited: no edge depends on it, so only the identity holds it.
    let sink = reseal(&honest, |_, blobs| {
        if let Some(bytes) = blobs.get_mut("embedded.result") {
            bytes[0] ^= 1;
        }
    });
    assert_eq!(stage_of(consume(&f, &sink, &id, &supplied)), BundleStage::ExpectedIdentityMismatch);

    // The stored identity edited.
    let stored = reseal(&honest, |body, _| body["identity"] = json!("f".repeat(64)));
    assert_eq!(
        stage_of(consume(&f, &stored, &id, &supplied)),
        BundleStage::ExpectedIdentityMismatch
    );

    // The consumer's retained identity differs from the bundle's.
    let wrong = "0".repeat(64);
    assert_eq!(
        stage_of(consume(&f, &honest, &wrong, &supplied)),
        BundleStage::ExpectedIdentityMismatch
    );
}

#[test]
fn c3_bundle_tampered_quantity_fails_the_dependent_node() {
    let mut f = Fixture::new();
    f.quantity = b"some other coordinates".to_vec();
    let bundle = f.bundle();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let consumed = consume(&f, &bytes, bundle.identity(), &snapshot_supplied()).unwrap();
    assert_eq!(failed_stage(&consumed, "distribution"), BundleStage::TamperedQuantity);
    assert_eq!(consumed.value("distribution", "x"), None);
    let error = consumed.require_verified().unwrap_err();
    assert_eq!(error.stage(), Some(BundleStage::TamperedQuantity));
}

#[test]
fn c3_bundle_swapped_evidence_fails_the_evidence_node() {
    let mut f = Fixture::new();
    f.evidence_identity = text_digest(b"shared_prior");
    let bundle = f.bundle();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let consumed = consume(&f, &bytes, bundle.identity(), &snapshot_supplied()).unwrap();
    assert_eq!(failed_stage(&consumed, "evidence"), BundleStage::SwappedEvidence);
    assert_eq!(
        EvidenceRelationship::from_name("shared_prior"),
        Some(EvidenceRelationship::SharedPrior)
    );
    assert_eq!(EvidenceRelationship::SharedFittedModel.as_str(), "shared_fitted_model");
}

#[test]
fn c3_bundle_graph_or_snapshot_mismatch_fails_the_bound_node() {
    let mut f = Fixture::new();
    f.snapshot_fact = "elsewhere|other-contract".into();
    let bundle = f.bundle();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let consumed = consume(&f, &bytes, bundle.identity(), &snapshot_supplied()).unwrap();
    assert_eq!(failed_stage(&consumed, "distribution"), BundleStage::GraphOrSnapshotMismatch);

    // Supplied data that is not the referenced snapshot fails the reference itself.
    let honest = Fixture::new();
    let bundle = honest.bundle();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let wrong = SuppliedSources::default().with_data("enumeration-1", "other-digest");
    let consumed = consume(&honest, &bytes, bundle.identity(), &wrong).unwrap();
    assert_eq!(failed_stage(&consumed, "causal"), BundleStage::GraphOrSnapshotMismatch);
}

#[test]
fn c3_bundle_reference_node_is_unresolved_without_a_source_and_verified_with_one() {
    let f = Fixture::new();
    let bundle = f.bundle();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();

    let without = consume(&f, &bytes, bundle.identity(), &SuppliedSources::default()).unwrap();
    assert!(matches!(
        status_of(&without, "causal"),
        NodeStatus::ReferenceUnresolved { requires: ProviderOrData::Data { .. } }
    ));
    assert!(!without.all_verified());
    // The inspected decision is preserved although the reference cannot be executed.
    assert_eq!(status_of(&without, "result"), NodeStatus::Verified);
    close(without.value("result", "safe.expected_utility").unwrap(), 3.0);
    let error = without.require_verified().unwrap_err();
    assert_eq!(error.stage(), Some(BundleStage::CallbackUnavailable));

    let with = consume(&f, &bytes, bundle.identity(), &snapshot_supplied()).unwrap();
    assert_eq!(status_of(&with, "causal"), NodeStatus::Verified);
}

fn external_mean_bundle(require_joint: bool) -> CompositionBundle {
    let mut decision = BundleNode::reference(
        "decision",
        NodeKind::DecisionResult,
        "decision-id",
        ProviderOrData::Data { snapshot_id: "decision-snap".into(), digest: "d1".into() },
    );
    if require_joint {
        decision = decision.with_fact(FACT_REQUIRES_LAW, LAW_JOINT);
    }
    let nodes = vec![
        BundleNode::reference(
            "attestation",
            NodeKind::Attestation,
            "attestation-id",
            ProviderOrData::Data { snapshot_id: "att-snap".into(), digest: "a1".into() },
        ),
        BundleNode::reference(
            "claim",
            NodeKind::ExternalClaim,
            "claim-id",
            ProviderOrData::Provider {
                provider_id: "lab".into(),
                snapshot_id: "snap-9".into(),
                request_fingerprint: "fp-1".into(),
            },
        )
        .with_fact(FACT_LAW, LAW_MEAN_ONLY)
        .with_fact(FACT_TRUST, "externally_attested")
        .with_inspected("mean", 4.25),
        decision,
    ];
    CompositionBundle::new(
        nodes,
        &pairs(&[("attestation", "claim"), ("claim", "decision")]),
        &BundleLimits::default(),
    )
    .unwrap()
}

fn mean_consume(
    bundle: &CompositionBundle,
    supplied: &SuppliedSources,
) -> Result<ConsumedBundle, BundleError> {
    let bytes = bundle.to_bytes("mean", &BundleLimits::default()).unwrap();
    BundleConsumer::new().consume(&bytes, &BundleLimits::default(), bundle.identity(), supplied)
}

#[test]
fn c3_bundle_external_mean_decision_is_point_only_attested_and_preserved() {
    let bundle = external_mean_bundle(false);
    let none = mean_consume(&bundle, &SuppliedSources::default()).unwrap();
    assert!(matches!(status_of(&none, "claim"), NodeStatus::ReferenceUnresolved { .. }));
    assert!(matches!(status_of(&none, "decision"), NodeStatus::ReferenceUnresolved { .. }));
    assert_eq!(none.node("decision").unwrap().claim_label, Some(ClaimLabel::PointOnlyAttested));
    assert_eq!(ClaimLabel::PointOnlyAttested.as_str(), "point_only_attested");
    // The declared mean is kept for inspection but is not a verified value.
    assert_eq!(none.node("claim").unwrap().inspected, vec![("mean".to_owned(), 4.25)]);
    assert_eq!(none.value("claim", "mean"), None);

    let supplied = SuppliedSources::default()
        .with_provider("lab", "snap-9", "fp-1")
        .with_data("decision-snap", "d1")
        .with_data("att-snap", "a1");
    let all = mean_consume(&bundle, &supplied).unwrap();
    all.require_verified().unwrap();
    assert_eq!(all.node("decision").unwrap().claim_label, Some(ClaimLabel::PointOnlyAttested));
}

#[test]
fn c3_bundle_changed_provider_request_is_refused() {
    let bundle = external_mean_bundle(false);
    let supplied = SuppliedSources::default()
        .with_provider("lab", "snap-9", "fp-CHANGED")
        .with_data("decision-snap", "d1")
        .with_data("att-snap", "a1");
    let consumed = mean_consume(&bundle, &supplied).unwrap();
    assert_eq!(failed_stage(&consumed, "claim"), BundleStage::ProviderRequestChanged);
    let error = consumed.require_verified().unwrap_err();
    assert_eq!(error.stage(), Some(BundleStage::ProviderRequestChanged));
}

#[test]
fn c3_bundle_joint_decision_over_a_mean_only_claim_retains_the_aligned_law_requirement() {
    let bundle = external_mean_bundle(true);
    let consumed = mean_consume(&bundle, &SuppliedSources::default()).unwrap();
    assert_eq!(failed_stage(&consumed, "decision"), BundleStage::UnsupportedLaw);
    assert_eq!(consumed.node("decision").unwrap().claim_label, None);
}

#[test]
fn c3_bundle_embedded_node_without_a_registered_verifier_never_verifies() {
    let nodes = vec![BundleNode::embedded("sens", NodeKind::Sensitivity, "s-1", b"bytes".to_vec())];
    let bundle = CompositionBundle::new(nodes, &[], &BundleLimits::default()).unwrap();
    let bytes = bundle.to_bytes("sens", &BundleLimits::default()).unwrap();
    let consumed = BundleConsumer::new()
        .consume(&bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default())
        .unwrap();
    assert_eq!(failed_stage(&consumed, "sens"), BundleStage::UnknownNodeKind);
}

#[test]
fn c3_bundle_oversize_and_graph_bounds_are_refused() {
    let f = Fixture::new();
    let bundle = f.bundle();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let supplied = snapshot_supplied();

    let tiny = BundleLimits { max_bundle_bytes: 64, ..BundleLimits::default() };
    let result = f.consumer().consume(&bytes, &tiny, bundle.identity(), &supplied);
    assert_eq!(stage_of(result), BundleStage::Oversized);

    let few = BundleLimits { max_nodes: 2, ..BundleLimits::default() };
    let result = f.consumer().consume(&bytes, &few, bundle.identity(), &supplied);
    assert_eq!(stage_of(result), BundleStage::Oversized);
    match CompositionBundle::new(f.nodes(), &deps(), &few) {
        Err(error) => assert_eq!(error.stage(), Some(BundleStage::Oversized)),
        Ok(_) => panic!("too many nodes were accepted"),
    }

    let small = BundleLimits { max_embedded_bytes: 4, ..BundleLimits::default() };
    match CompositionBundle::new(f.nodes(), &deps(), &small) {
        Err(error) => assert_eq!(error.stage(), Some(BundleStage::Oversized)),
        Ok(_) => panic!("an oversized embedded node was accepted"),
    }

    // A cycle and a dangling edge are malformed graphs.
    let cycle = pairs(&[("contract", "result"), ("result", "contract")]);
    match CompositionBundle::new(f.nodes(), &cycle, &BundleLimits::default()) {
        Err(error) => assert_eq!(error.stage(), Some(BundleStage::EdgeDigestMismatch)),
        Ok(_) => panic!("a cycle was accepted"),
    }
    let dangling = pairs(&[("contract", "missing")]);
    match CompositionBundle::new(f.nodes(), &dangling, &BundleLimits::default()) {
        Err(error) => assert_eq!(error.stage(), Some(BundleStage::EdgeDigestMismatch)),
        Ok(_) => panic!("a dangling edge was accepted"),
    }
}

#[test]
fn c3_bundle_stages_map_to_registered_codes_and_namespaced_details() {
    for stage in BundleStage::ALL {
        assert_eq!(stage.detail(), format!("composition_bundle.{}", stage.as_str()));
        assert!(antecedent_core::reason_code::is_registered(stage.code()), "{stage:?}");
    }
    let error = BundleError::Refused {
        stage: BundleStage::UnsupportedLaw,
        reason: "mean only".into(),
        offending: Some("decision".into()),
    };
    let refusal = error.to_refusal().unwrap();
    assert_eq!(refusal.detail, "composition_bundle.unsupported_law");
    assert_eq!(refusal.code, "joint_law_required");
    assert_eq!(refusal.offending.as_deref(), Some("decision"));
    for kind in NodeKind::ALL {
        assert_eq!(NodeKind::from_name(kind.as_str()), Some(kind));
    }
    assert_eq!(NodeKind::from_name("bogus"), None);
}
