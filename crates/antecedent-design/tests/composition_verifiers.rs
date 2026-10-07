//! C3 node verifiers: every artifact kind a composition bundle embeds is decoded
//! through its own consumer, publishes binding facts read from the decoded
//! artifact, and is cross-checked against the nodes upstream of it.
//!
//! Hand-enumerated oracles:
//!
//! * joint decision: four equally likely rows, `risky = E[p * q] = (4 + 0 + 4 + 0) / 4
//!   = 2` and `safe = max(3, 0) = 3`;
//! * point-only decision: `E[Y | do(a)] = 1 + 2a`, so `wait` reads `a = 0` and has
//!   utility `2 * 1 - 1 = 1`, `treat` reads `a = 2` and has utility `2 * 5 - 1 = 9`;
//! * robust decision: structures `s1 = (a: 5, b: 3)` and `s2 = (a: 6, b: 2)`;
//! * ranking: the frozen F14 record, net values `3/20` and `1/40`.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;

use antecedent_core::{
    CancellationToken, CheckedCausalContract, DistributionMeaning, ExternalCapability,
    ExternalResponse, ExternalResult, ExternalResultHeader, ExternalScientificObject,
    ExternalTrustState, ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, SignalProviderContract,
    SupportStatus, bind_external_result,
};
use antecedent_design::composition_bundle::{
    BundleLimits, BundleNode, BundleStage, ClaimLabel, CompositionBundle, ConsumedBundle,
    EvidenceRelationship, FACT_LAW, FACT_REQUIRES_LAW, FACT_SNAPSHOT, FACT_TRUST, LAW_JOINT,
    LAW_MEAN_ONLY, NodeKind, NodeStatus, ProviderOrData, SuppliedSources,
};
use antecedent_design::composition_verifiers::{
    FACT_COORDINATES, FACT_REQUIRES_DISTRIBUTION, TRUST_ATTESTED, TRUST_VERIFIED, VERIFIABLE_KINDS,
    describe_artifact, detect_node_kind, evidence_relationship_bytes, mean_decision_bytes,
    mean_source_of, standard_consumer, verifier_for,
};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, mean_result_to_bytes, source_digest,
};
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, HardConstraint, StructuralPolicy, UncertaintyKind, UtilityExpr,
};
use antecedent_design::decision_eval::{evaluate_contract, evaluate_contract_on_means};
use antecedent_design::decision_robust_artifact::{
    AdmissibleContractArtifact, ExternalCallbackReceipt, ExternalTrustLimit, RobustResultArtifact,
};
use antecedent_design::decision_robustness::{AtomSupport, ClaimProfile, evaluate_robust};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_design::design_ranking_artifact::{DesignRankingArtifactWire, SealInputs, seal};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiRequest, StudyCostSpec, evaluate_evsi,
};
use antecedent_design::sensitivity_decision::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, PointSupport, SamplingStatus,
    SensitivityArtifact, SensitivityParts, SensitivityProvenance, SurfaceQuantity,
    UncertaintyRelationship, UtilityTerm,
};
use antecedent_design::signal::{
    ExternalLaw, ExternalSignal, ExternalSignalBody, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    SamplingPlan,
};
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimTrust, VerificationProbeWire,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_io::reader::ArtifactReader;
use serde_json::{Value, json};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn embed(id: &str, bytes: &[u8]) -> BundleNode {
    let kind = detect_node_kind(bytes).expect("a bundle artifact kind");
    let identity = describe_artifact(kind, bytes).expect("the artifact decodes").identity;
    BundleNode::embedded(id, kind, &identity, bytes.to_vec())
}

fn edges(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect()
}

fn run(
    nodes: Vec<BundleNode>,
    dependencies: &[(&str, &str)],
    supplied: &SuppliedSources,
) -> ConsumedBundle {
    let bundle =
        CompositionBundle::new(nodes, &edges(dependencies), &BundleLimits::default()).unwrap();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    let consumed = standard_consumer()
        .consume(&bytes, &BundleLimits::default(), bundle.identity(), supplied)
        .unwrap();
    assert_eq!(consumed.identity(), bundle.identity());
    consumed
}

fn failed_stage(consumed: &ConsumedBundle, id: &str) -> BundleStage {
    match &consumed.node(id).unwrap().status {
        NodeStatus::Failed { stage, .. } => *stage,
        other => panic!("node `{id}` is {other:?}"),
    }
}

fn fact<'a>(consumed: &'a ConsumedBundle, id: &str, key: &str) -> Option<&'a str> {
    consumed.node(id).unwrap().facts.get(key).map(String::as_str)
}

// ---------------------------------------------------------------------------
// A joint law, a decision contract and its result
// ---------------------------------------------------------------------------

fn outcome(variable: &str, regime: &str) -> ScientificQuantity {
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

fn joint_columns() -> Vec<ScientificQuantity> {
    vec![outcome("p", "do(a=1)"), outcome("q", "do(a=1)"), outcome("safe", "do(a=0)")]
}

fn joint_source() -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &joint_columns(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let p = [1.0, 3.0, 2.0, 0.0];
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
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

fn joint_contract() -> DecisionContract {
    let cols = joint_columns();
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

struct Joint {
    contract: Vec<u8>,
    distribution: Vec<u8>,
    result: Vec<u8>,
}

fn joint() -> Joint {
    let contract = joint_contract();
    let source = joint_source();
    let result = evaluate_contract(&contract, &source).unwrap();
    Joint {
        contract: DecisionContractArtifact::new(contract).unwrap().to_bytes("contract").unwrap(),
        distribution: source.to_bytes("distribution").unwrap(),
        result: DecisionResultArtifact::new(result, &source).to_bytes("result").unwrap(),
    }
}

fn joint_nodes(j: &Joint) -> Vec<BundleNode> {
    vec![
        embed("contract", &j.contract),
        embed("distribution", &j.distribution),
        embed("result", &j.result),
    ]
}

const JOINT_EDGES: [(&str, &str); 2] = [("distribution", "result"), ("contract", "result")];

/// Edit the metadata of a distribution container and re-wrap it with a fresh,
/// valid checksum: a self-consistent artifact the container cannot refuse.
fn reseal_distribution(bytes: &[u8], edit: impl FnOnce(&mut Value)) -> Vec<u8> {
    const META: &str = "distribution.meta";
    const DRAWS: &str = "distribution.draws";
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes)).unwrap();
    let manifest = reader.manifest().clone();
    let meta = reader.load_section(META).unwrap().as_bytes().to_vec();
    let draws = reader.load_section(DRAWS).unwrap().as_bytes().to_vec();
    let mut value: Value = from_cbor(&meta).unwrap();
    edit(&mut value);
    let meta = to_cbor(&value).unwrap();
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            sections: vec![
                section_descriptor(META, "application/cbor", &meta),
                section_descriptor(DRAWS, "application/octet-stream", &draws),
            ],
            ..manifest
        },
        sections: vec![SectionBytes::new(META, meta), SectionBytes::new(DRAWS, draws)],
    };
    let mut out = Vec::new();
    encoded.write_to(&mut out).unwrap();
    out
}

// ---------------------------------------------------------------------------
// An external response claim and a point-only decision over it
// ---------------------------------------------------------------------------

fn claim_quantity(dose: u32) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:y".into(),
        variable_name: "Y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: format!("do(a={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

struct ClaimSpec<'a> {
    provider: &'a str,
    snapshot: &'a str,
    request: &'a str,
    evidence: &'a [&'a str],
    values: [f64; 3],
}

impl Default for ClaimSpec<'_> {
    fn default() -> Self {
        Self {
            provider: "lab",
            snapshot: "snap-9",
            request: "req-1",
            evidence: &["factor:z"],
            values: [1.0, 3.0, 5.0],
        }
    }
}

fn claim(spec: &ClaimSpec<'_>) -> ExternalClaimArtifact {
    let quantities: Vec<ScientificQuantity> = (0..3).map(claim_quantity).collect();
    let evidence: Vec<String> = spec.evidence.iter().map(|e| (*e).to_owned()).collect();
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: spec.provider.into(),
            object_id: "curve".into(),
            version_id: "v3".into(),
            snapshot_id: spec.snapshot.into(),
            request_id: spec.request.into(),
        },
        quantities: quantities.clone(),
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean],
    });
    let contract = CheckedCausalContract {
        graph_id: "graph-1".into(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: quantities.clone(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: evidence.clone(),
        required_assumption_ids: vec!["ignorability".into()],
        equivalences: vec![],
    };
    let response = ExternalResponse {
        header: ExternalResultHeader {
            object,
            graph_id: "graph-1".into(),
            quantities,
            evidence_ids: evidence,
            assumption_ids: vec!["ignorability".into()],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        },
        values: spec.values.to_vec(),
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: Some(vec![
            SupportStatus::Supported,
            SupportStatus::Supported,
            SupportStatus::Supported,
        ]),
    };
    let bound = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    ExternalClaimArtifact::from_bound_claim(&bound, "checked-contract").unwrap()
}

/// The same claim carrying an exact-request verification receipt.
fn verified(artifact: &ExternalClaimArtifact) -> ExternalClaimArtifact {
    let mut meta = artifact.metadata().clone();
    meta.identity.trust = ExternalClaimTrust::ExactRequestVerified;
    meta.identity.verification = Some(vec![VerificationProbeWire {
        kind: "normalization".into(),
        observed: 1.0,
        expected: 1.0,
        tolerance: 1e-9,
    }]);
    ExternalClaimArtifact::new(meta, artifact.values().to_vec()).unwrap()
}

fn mean_contract() -> DecisionContract {
    let affine = || {
        UtilityExpr::difference(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
            UtilityExpr::Const(1.0),
        )
    };
    let action = |id: &str, dose: u32| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![claim_quantity(dose)],
        utility: affine(),
    };
    DecisionContract {
        actions: vec![action("wait", 0), action("treat", 2)],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

/// The point-only decision of `claim_artifact`'s means, as container bytes.
fn point_only_bytes(claim_artifact: &ExternalClaimArtifact) -> Vec<u8> {
    let source = mean_source_of(claim_artifact).unwrap();
    let result = evaluate_contract_on_means(&mean_contract(), &source).unwrap();
    mean_result_to_bytes(&result, &source, "mean-result").unwrap()
}

fn mean_contract_bytes() -> Vec<u8> {
    DecisionContractArtifact::new(mean_contract()).unwrap().to_bytes("mean-contract").unwrap()
}

const MEAN_EDGES: [(&str, &str); 2] = [("claim", "result"), ("contract", "result")];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn c3_verifiers_joint_decision_verifies_every_node_and_reads_the_hand_computed_values() {
    let j = joint();
    let consumed = run(joint_nodes(&j), &JOINT_EDGES, &SuppliedSources::default());
    consumed.require_verified().unwrap();
    close(consumed.value("result", "risky.expected_utility").unwrap(), 2.0);
    close(consumed.value("result", "safe.expected_utility").unwrap(), 3.0);
    close(consumed.value("distribution", "n_draws").unwrap(), 4.0);
    // The mean of p over the four rows: (1 + 3 + 2 + 0) / 4.
    close(consumed.value("distribution", "mean.0").unwrap_or(f64::NAN), 1.5);
    assert_eq!(consumed.node("result").unwrap().claim_label, Some(ClaimLabel::JointDraw));

    // Facts come from the decoded artifacts.
    assert_eq!(fact(&consumed, "distribution", FACT_LAW), Some(LAW_JOINT));
    assert_eq!(fact(&consumed, "distribution", FACT_TRUST), Some("unverified"));
    assert_eq!(
        fact(&consumed, "distribution", FACT_SNAPSHOT),
        Some("enumeration-1|checked-contract")
    );
    assert_eq!(fact(&consumed, "result", FACT_SNAPSHOT), Some("enumeration-1|checked-contract"));
    // The product utility needs an aligned joint law, and the result inherits it.
    assert_eq!(fact(&consumed, "contract", FACT_REQUIRES_LAW), Some(LAW_JOINT));
    assert_eq!(fact(&consumed, "result", FACT_REQUIRES_LAW), Some(LAW_JOINT));
    assert_eq!(fact(&consumed, "result", FACT_REQUIRES_DISTRIBUTION), Some("true"));
    assert!(fact(&consumed, "contract", FACT_COORDINATES).is_some());

    // The identity of an embedded artifact is recomputed, and order does not matter.
    let mut reversed = joint_nodes(&j);
    reversed.reverse();
    let again = run(reversed, &JOINT_EDGES, &SuppliedSources::default());
    assert_eq!(again.identity(), consumed.identity());
}

#[test]
fn c3_verifiers_external_mean_decision_is_point_only_attested_with_verification_only_when_retained()
{
    let attested = claim(&ClaimSpec::default());
    let nodes = |claim_artifact: &ExternalClaimArtifact| {
        vec![
            embed("claim", &claim_artifact.to_bytes("claim").unwrap()),
            embed("contract", &mean_contract_bytes()),
            embed("result", &point_only_bytes(&attested)),
        ]
    };
    let consumed = run(nodes(&attested), &MEAN_EDGES, &SuppliedSources::default());
    consumed.require_verified().unwrap();
    close(consumed.value("result", "wait.expected_utility").unwrap(), 1.0);
    close(consumed.value("result", "treat.expected_utility").unwrap(), 9.0);
    assert_eq!(consumed.node("result").unwrap().claim_label, Some(ClaimLabel::PointOnlyAttested));
    assert_eq!(fact(&consumed, "claim", FACT_LAW), Some(LAW_MEAN_ONLY));
    assert_eq!(fact(&consumed, "claim", FACT_TRUST), Some(TRUST_ATTESTED));
    // An affine expected utility is answered by a mean: nothing demands a joint law.
    assert_eq!(fact(&consumed, "contract", FACT_REQUIRES_LAW), None);
    assert_eq!(fact(&consumed, "result", FACT_REQUIRES_DISTRIBUTION), None);

    // The same claim retaining its verification receipt is upgraded, still never native
    // and still a point-only result.
    let upgraded = run(nodes(&verified(&attested)), &MEAN_EDGES, &SuppliedSources::default());
    assert_eq!(fact(&upgraded, "claim", FACT_TRUST), Some(TRUST_VERIFIED));
    assert_eq!(upgraded.node("result").unwrap().claim_label, Some(ClaimLabel::PointOnlyAttested));
    // Verification is part of the claim's identity: the attested node is not the verified one.
    assert_ne!(consumed.node("claim").unwrap().identity, upgraded.node("claim").unwrap().identity);
}

#[test]
fn c3_verifiers_mean_decision_helper_matches_direct_evaluation_and_refuses_what_a_mean_cannot_answer()
 {
    let artifact = claim(&ClaimSpec::default());
    let claim_bytes = artifact.to_bytes("claim").unwrap();
    let helper = mean_decision_bytes(&mean_contract(), &claim_bytes, "mean-result").unwrap();
    assert_eq!(helper, point_only_bytes(&artifact));

    // A nonlinear utility needs more than a mean: the law stage refuses it.
    let mut squared = mean_contract();
    squared.actions[1].utility = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
    let refused = mean_decision_bytes(&squared, &claim_bytes, "mean-result").unwrap_err();
    assert_eq!(refused.stage, BundleStage::UnsupportedLaw);

    // Bytes that are not a claim are refused by the claim's own consumer.
    let not_a_claim = joint().distribution;
    assert!(mean_decision_bytes(&mean_contract(), &not_a_claim, "mean-result").is_err());
}

#[test]
fn c3_verifiers_changed_mean_value_fails_the_point_only_result() {
    let honest = claim(&ClaimSpec::default());
    let changed = claim(&ClaimSpec { values: [1.0, 3.0, 6.0], ..ClaimSpec::default() });
    let consumed = run(
        vec![
            embed("claim", &changed.to_bytes("claim").unwrap()),
            embed("contract", &mean_contract_bytes()),
            embed("result", &point_only_bytes(&honest)),
        ],
        &MEAN_EDGES,
        &SuppliedSources::default(),
    );
    assert_eq!(consumed.node("claim").unwrap().status, NodeStatus::Verified);
    assert_eq!(failed_stage(&consumed, "result"), BundleStage::TamperedQuantity);
    assert_eq!(consumed.value("result", "treat.expected_utility"), None);
    let refusal = consumed.require_verified().unwrap_err().to_refusal().unwrap();
    assert_eq!(refusal.detail, "composition_bundle.tampered_quantity");
}

#[test]
fn c3_verifiers_joint_requirement_over_a_mean_only_claim_fails_unsupported_law() {
    let j = joint();
    // The claim states the snapshot and causal contract the decision rests on, so only
    // its law can disagree.
    let mean_only = claim(&ClaimSpec { snapshot: "enumeration-1", ..ClaimSpec::default() });
    let mut nodes = joint_nodes(&j);
    nodes.push(embed("claim", &mean_only.to_bytes("claim").unwrap()));
    let consumed = run(
        nodes,
        &[("distribution", "result"), ("contract", "result"), ("claim", "result")],
        &SuppliedSources::default(),
    );
    assert_eq!(consumed.node("claim").unwrap().status, NodeStatus::Verified);
    assert_eq!(failed_stage(&consumed, "result"), BundleStage::UnsupportedLaw);
    assert_eq!(consumed.node("result").unwrap().claim_label, None);
    let refusal = consumed.require_verified().unwrap_err().to_refusal().unwrap();
    assert_eq!(refusal.detail, "composition_bundle.unsupported_law");
    assert_eq!(refusal.code, "joint_law_required");
}

#[test]
fn c3_verifiers_swapped_quantity_fails_the_dependent_result_after_resealing_the_inner_artifact() {
    let j = joint();
    let resealed = reseal_distribution(&j.distribution, |meta| {
        meta["identity"]["quantities"][0]["variable_id"] = json!("p-renamed");
    });

    // A rebuilt bundle: the inner artifact is self-consistent and verifies on its own,
    // yet the result was computed from other draws and coordinates.
    let rebuilt = run(
        vec![
            embed("contract", &j.contract),
            embed("distribution", &resealed),
            embed("result", &j.result),
        ],
        &JOINT_EDGES,
        &SuppliedSources::default(),
    );
    assert_eq!(rebuilt.node("distribution").unwrap().status, NodeStatus::Verified);
    assert_eq!(failed_stage(&rebuilt, "result"), BundleStage::TamperedQuantity);

    // The bundle still declares the original identity: the node itself is refused,
    // and the result with it.
    let original = describe_artifact(NodeKind::Distribution, &j.distribution).unwrap().identity;
    let stale = run(
        vec![
            embed("contract", &j.contract),
            BundleNode::embedded("distribution", NodeKind::Distribution, &original, resealed),
            embed("result", &j.result),
        ],
        &JOINT_EDGES,
        &SuppliedSources::default(),
    );
    assert_eq!(failed_stage(&stale, "distribution"), BundleStage::SwappedEvidence);
    assert_eq!(failed_stage(&stale, "result"), BundleStage::TamperedQuantity);
}

#[test]
fn c3_verifiers_a_decision_over_other_contract_or_unrelated_source_is_refused() {
    let j = joint();
    // A result whose contract is not upstream of it.
    let other_contract = {
        let mut c = joint_contract();
        c.constraints[0].bound = 4.0;
        DecisionContractArtifact::new(c).unwrap().to_bytes("other").unwrap()
    };
    let consumed = run(
        vec![
            embed("contract", &other_contract),
            embed("distribution", &j.distribution),
            embed("result", &j.result),
        ],
        &JOINT_EDGES,
        &SuppliedSources::default(),
    );
    assert_eq!(failed_stage(&consumed, "result"), BundleStage::GraphOrSnapshotMismatch);

    // A result with no source upstream of it.
    let alone = run(
        vec![embed("contract", &j.contract), embed("result", &j.result)],
        &[("contract", "result")],
        &SuppliedSources::default(),
    );
    assert_eq!(failed_stage(&alone, "result"), BundleStage::TamperedQuantity);
}

#[test]
fn c3_verifiers_swapped_snapshot_fails_the_bound_sensitivity_node() {
    let artifact = SensitivityArtifact::new(sensitivity_parts()).unwrap();
    let bytes = artifact.to_bytes("sensitivity").unwrap();
    let reference = |snapshot: &str| {
        BundleNode::reference(
            "causal",
            NodeKind::CausalContract,
            "program-1",
            ProviderOrData::Data { snapshot_id: "snapshot-1".into(), digest: "d".into() },
        )
        .with_fact(FACT_SNAPSHOT, snapshot)
    };
    let supplied = SuppliedSources::default().with_data("snapshot-1", "d");
    let consumed = run(
        vec![reference("snapshot-1|checked-contract"), embed("sensitivity", &bytes)],
        &[("causal", "sensitivity")],
        &supplied,
    );
    consumed.require_verified().unwrap();
    close(consumed.value("sensitivity", "grid_points").unwrap(), 3.0);
    assert_eq!(consumed.node("sensitivity").unwrap().identity, artifact.identity().digest);

    let swapped = run(
        vec![reference("elsewhere|checked-contract"), embed("sensitivity", &bytes)],
        &[("causal", "sensitivity")],
        &supplied,
    );
    assert_eq!(failed_stage(&swapped, "sensitivity"), BundleStage::GraphOrSnapshotMismatch);
}

#[test]
fn c3_verifiers_swapped_evidence_fails_the_relationship_node() {
    let a = claim(&ClaimSpec { request: "req-a", evidence: &["factor:z"], ..ClaimSpec::default() });
    let b = claim(&ClaimSpec { request: "req-b", evidence: &["factor:w"], ..ClaimSpec::default() });
    let relationship = evidence_relationship_bytes(EvidenceRelationship::Independent, "a", "b");
    let nodes = |second: &ExternalClaimArtifact| {
        vec![
            embed("a", &a.to_bytes("a").unwrap()),
            embed("b", &second.to_bytes("b").unwrap()),
            embed("relation", &relationship),
        ]
    };
    let relations = [("a", "relation"), ("b", "relation")];
    let consumed = run(nodes(&b), &relations, &SuppliedSources::default());
    consumed.require_verified().unwrap();

    // The second study is swapped for one resting on the first one's evidence while the
    // declaration still says the two are independent.
    let swapped =
        claim(&ClaimSpec { request: "req-b", evidence: &["factor:z"], ..ClaimSpec::default() });
    let refused = run(nodes(&swapped), &relations, &SuppliedSources::default());
    assert_eq!(refused.node("b").unwrap().status, NodeStatus::Verified);
    assert_eq!(failed_stage(&refused, "relation"), BundleStage::SwappedEvidence);

    // A relationship that names a node that is not upstream of it is refused.
    let dangling = run(nodes(&b), &[("a", "relation")], &SuppliedSources::default());
    assert_eq!(failed_stage(&dangling, "relation"), BundleStage::GraphOrSnapshotMismatch);
}

// -- the robust decision ----------------------------------------------------------------

fn law(a: f64, b: f64) -> Box<DistributionArtifact> {
    let columns = [outcome("a", "do(a=1)"), outcome("b", "do(a=0)")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "structure".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration".into(),
            causal_contract_id: "checked".into(),
        },
    )
    .unwrap();
    Box::new(
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [2, 2],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Exact,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            vec![a, b, a, b],
        )
        .unwrap(),
    )
}

fn evaluated(id: &str, a: f64, b: f64) -> StructuralAtom {
    StructuralAtom {
        id: id.into(),
        probability: None,
        evidence: AtomEvidence::Evaluated(law(a, b)),
    }
}

fn robust_contract() -> AdmissibleDecisionContract {
    let action = |id: &str, variable: &str, regime: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![outcome(variable, regime)],
        utility: UtilityExpr::Input(0),
    };
    AdmissibleDecisionContract {
        contract: DecisionContract {
            actions: vec![action("A", "a", "do(a=1)"), action("B", "b", "do(a=0)")],
            utility_units: "units".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        },
        rules: AdmissibilityRules {
            default_weakest_support: Some(SupportStatus::Supported),
            ..AdmissibilityRules::default()
        },
    }
}

const ROBUST_REQUEST: &str = "d5a1c0ffeed5a1c0ffeed5a1c0ffeed5a1c0ffeed5a1c0ffeed5a1c0ffeed5a1";

struct Robust {
    contract: Vec<u8>,
    s1: Vec<u8>,
    s2: Vec<u8>,
    result: Vec<u8>,
}

fn robust(trust: ExternalTrustLimit) -> Robust {
    let c = robust_contract();
    let atoms = [evaluated("s1", 5.0, 3.0), evaluated("s2", 6.0, 2.0)];
    let claims = ClaimProfile {
        uncertainty: UncertaintyKind::StructuralEnvelope,
        support: vec![
            ("s1".to_owned(), AtomSupport::supported()),
            ("s2".to_owned(), AtomSupport::supported()),
        ],
    };
    let result = evaluate_robust(&c, &atoms, &claims).unwrap().1;
    let receipt = ExternalCallbackReceipt {
        atom_id: "s2".into(),
        provider_id: "lab-model".into(),
        snapshot_id: "snap-9".into(),
        request_fingerprint: ROBUST_REQUEST.into(),
        attested_value: 4.25,
        trust,
    };
    let artifact = RobustResultArtifact::new(&c, result, &claims, &atoms, vec![receipt]).unwrap();
    let AtomEvidence::Evaluated(s1) = &atoms[0].evidence else { unreachable!() };
    let AtomEvidence::Evaluated(s2) = &atoms[1].evidence else { unreachable!() };
    Robust {
        contract: AdmissibleContractArtifact::new(c).unwrap().to_bytes("robust-contract").unwrap(),
        s1: s1.to_bytes("s1").unwrap(),
        s2: s2.to_bytes("s2").unwrap(),
        result: artifact.to_bytes("robust").unwrap(),
    }
}

const ROBUST_EDGES: [(&str, &str); 4] =
    [("contract", "result"), ("s1", "result"), ("s2", "result"), ("claim", "result")];

fn robust_nodes(r: &Robust, claim_artifact: &ExternalClaimArtifact) -> Vec<BundleNode> {
    vec![
        embed("contract", &r.contract),
        embed("s1", &r.s1),
        embed("s2", &r.s2),
        embed("result", &r.result),
        embed("claim", &claim_artifact.to_bytes("claim").unwrap()),
    ]
}

#[test]
fn c3_verifiers_changed_provider_request_fails_the_robust_result() {
    let attested = ExternalTrustLimit::ExternallyAttested { attestor: "outside-lab".into() };
    let r = robust(attested);
    let answered = claim(&ClaimSpec {
        provider: "lab-model",
        snapshot: "snap-9",
        request: ROBUST_REQUEST,
        ..ClaimSpec::default()
    });
    let consumed = run(robust_nodes(&r, &answered), &ROBUST_EDGES, &SuppliedSources::default());
    consumed.require_verified().unwrap();
    assert_eq!(fact(&consumed, "result", FACT_TRUST), Some(TRUST_ATTESTED));
    assert_eq!(consumed.node("result").unwrap().claim_label, Some(ClaimLabel::PointOnlyAttested));
    // A decision resting on an external callback is never labelled natively verified.
    assert_eq!(consumed.value("result", "native_verified"), Some(0.0));

    // The provider answered another request: the result's receipt no longer matches the
    // claim upstream of it, even though the claim itself is a valid artifact.
    let other = claim(&ClaimSpec {
        provider: "lab-model",
        snapshot: "snap-9",
        request: "another-request",
        ..ClaimSpec::default()
    });
    let changed = run(robust_nodes(&r, &other), &ROBUST_EDGES, &SuppliedSources::default());
    assert_eq!(changed.node("claim").unwrap().status, NodeStatus::Verified);
    assert_eq!(failed_stage(&changed, "result"), BundleStage::ProviderRequestChanged);

    // A receipt that says verified over a claim that retains no receipt is refused.
    let inflated = robust(ExternalTrustLimit::VerifiedExtension);
    let refused =
        run(robust_nodes(&inflated, &answered), &ROBUST_EDGES, &SuppliedSources::default());
    assert_eq!(failed_stage(&refused, "result"), BundleStage::SwappedEvidence);
    let genuine = run(
        robust_nodes(&inflated, &verified(&answered)),
        &ROBUST_EDGES,
        &SuppliedSources::default(),
    );
    assert_eq!(genuine.node("result").unwrap().status, NodeStatus::Verified);
    assert_eq!(fact(&genuine, "result", FACT_TRUST), Some(TRUST_VERIFIED));

    // A structure replaced by other draws is not what the result was assessed from.
    let swapped = {
        let mut nodes = robust_nodes(&r, &answered);
        nodes.retain(|n| n.id != "s2");
        nodes.push(embed("s2", &law(9.0, 9.0).to_bytes("s2").unwrap()));
        run(nodes, &ROBUST_EDGES, &SuppliedSources::default())
    };
    assert_eq!(failed_stage(&swapped, "result"), BundleStage::TamperedQuantity);
}

// -- sensitivity ------------------------------------------------------------------------

fn sensitivity_parts() -> SensitivityParts {
    let scientific = |name: &str| ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "utils".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "sensitivity_surface".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    };
    let point = |name: &str, values: &[f64]| SurfaceQuantity {
        quantity: ScientificQuantityWire::from(&scientific(name)),
        lower: values.to_vec(),
        upper: values.to_vec(),
    };
    let act = |id: &str, quantity: &str| ActionUtility {
        id: id.into(),
        utility: UtilityTerm::quantity(quantity),
    };
    SensitivityParts {
        coordinate: AssumptionCoordinate {
            id: "gamma".into(),
            scale: "sensitivity_parameter".into(),
            units: "dimensionless".into(),
            minimum: 0.0,
            maximum: 2.0,
        },
        grid: vec![0.0, 1.0, 2.0],
        support: vec![PointSupport::Supported; 3],
        quantities: vec![point("ua", &[2.0, 1.0, 0.0]), point("ub", &[1.0, 1.0, 1.0])],
        actions: vec![act("A", "ua"), act("B", "ub")],
        uncertainty: UncertaintyRelationship {
            assumption_range: AssumptionRangeStatement {
                kind: "assumption_range".into(),
                interpretation: "assumption range; not a probability".into(),
            },
            identified_bound: None,
            sampling: SamplingStatus::Withheld {
                reason_code: "cell_not_licensed".into(),
                detail: "joint_sensitivity.interval_withheld".into(),
            },
        },
        provenance: SensitivityProvenance {
            source_kind: "supplied_surface".into(),
            query_binding: "c3-test".into(),
            provider_snapshot: "snapshot-1".into(),
            source_regime: "regime:1".into(),
            method: "hand-derived surface".into(),
            causal_contract_id: "checked-contract".into(),
            decision_threshold: None,
            source_tipping: vec![],
        },
    }
}

// -- the ranking ------------------------------------------------------------------------

fn rank_quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: id.into(),
        variable_name: id.into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn signal_request(candidate: &str) -> SignalRequest {
    SignalRequest {
        candidate_id: candidate.into(),
        prior_id: "prior-1".into(),
        state_quantity: rank_quantity("schema:state"),
        observation_quantity: rank_quantity("schema:signal"),
        sample_size: 1,
        rng_seed: 3,
        evidence_lineage: vec!["snapshot:a".into()],
        conditional_independence: "iid_given_state".into(),
        limits: SignalLimits::default(),
    }
}

fn guess_candidate(id: &str, accuracy: f64) -> EvsiCandidate {
    let req = signal_request(id);
    let object = ExternalScientificObject::Signal(SignalProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: format!("signal-{id}"),
            version_id: "v1".into(),
            snapshot_id: "snap".into(),
            request_id: req.fingerprint(),
        },
        candidate_id: req.candidate_id.clone(),
        prior_id: req.prior_id.clone(),
        observation: req.observation_quantity.clone(),
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Update],
    });
    let trust = ExternalTrustState::attest(&object, "lab-qa").unwrap();
    let body = ExternalSignalBody {
        sample_size: req.sample_size,
        state_quantity: req.state_quantity.clone(),
        observation_quantity: req.observation_quantity.clone(),
        law: ExternalLaw::Posterior {
            states: vec![0.0, 1.0],
            statistics: vec![0.0, 1.0],
            predictive: vec![0.5, 0.5],
            posterior: vec![vec![accuracy, 1.0 - accuracy], vec![1.0 - accuracy, accuracy]],
        },
    };
    let provider: Arc<dyn SignalProvider> =
        Arc::new(ExternalSignal::new(object, trust, body).unwrap());
    EvsiCandidate {
        semantic_id: id.into(),
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        signal_request: req,
        provider,
        cost: StudyCostSpec { amount: 0.1, unit: "utility".into() },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn ranking(source_digests: &[String]) -> DesignRankingArtifactWire {
    let utility = AffineUtility::new(vec![1.0, 0.0], vec![-1.0, 1.0]).unwrap();
    let problem: DecisionProblem<usize, f64> =
        DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![]);
    let prior = DecisionPrior::Draws(vec![0.0, 1.0]);
    let map = CostToUtilityMap {
        cost_unit: "utility".into(),
        utility_unit: "utility".into(),
        utility_per_cost: 1.0,
    };
    let request = EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["guess0".into(), "guess1".into()],
        candidates: vec![guess_candidate("cand-1", 0.75), guess_candidate("cand-2", 0.625)],
        cost_map: Some(map),
        require_net_value: false,
        prior_observation_ids: vec!["obs-prior".into()],
        rank_config: DesignRankConfig {
            min_batches: 4,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 0.0,
        },
        rng_seed: 5,
        mc_error_tolerance: 1e-3,
        tie_tolerance: 1e-12,
        max_candidates: 16,
    };
    let report = evaluate_evsi(&problem, &prior, &request, &CancellationToken::new()).unwrap();
    seal(&SealInputs {
        problem: &problem,
        prior: &prior,
        request: &request,
        report: &report,
        source_digests,
    })
    .unwrap()
}

#[test]
fn c3_verifiers_ranking_is_replayed_and_bound_to_its_source_digests() {
    let j = joint();
    let digest = source_digest(&joint_source());
    let bound = ranking(&[digest]).to_bytes("ranking").unwrap();
    let consumed = run(
        vec![embed("distribution", &j.distribution), embed("ranking", &bound)],
        &[("distribution", "ranking")],
        &SuppliedSources::default(),
    );
    consumed.require_verified().unwrap();
    close(consumed.value("ranking", "cand-1.net_value").unwrap(), 0.15);
    close(consumed.value("ranking", "cand-2.net_value").unwrap(), 0.025);
    close(consumed.value("ranking", "cand-1.evsi").unwrap(), 0.25);
    assert_eq!(fact(&consumed, "ranking", FACT_TRUST), Some(TRUST_ATTESTED));

    // A ranking computed over other source digests is not bound to this distribution.
    let unrelated = ranking(&["digest-a".to_owned()]).to_bytes("ranking").unwrap();
    let refused = run(
        vec![embed("distribution", &j.distribution), embed("ranking", &unrelated)],
        &[("distribution", "ranking")],
        &SuppliedSources::default(),
    );
    assert_eq!(refused.node("distribution").unwrap().status, NodeStatus::Verified);
    assert_eq!(failed_stage(&refused, "ranking"), BundleStage::TamperedQuantity);

    // Standing alone it verifies and is never claimed natively replayed.
    let alone = run(vec![embed("ranking", &unrelated)], &[], &SuppliedSources::default());
    alone.require_verified().unwrap();
}

#[test]
fn c3_verifiers_registry_covers_every_artifact_kind_and_refuses_the_rest() {
    let j = joint();
    for kind in VERIFIABLE_KINDS {
        assert!(verifier_for(kind).is_some_and(|v| v.kind() == kind), "{kind:?}");
    }
    assert!(verifier_for(NodeKind::CausalContract).is_none());
    assert_eq!(detect_node_kind(&j.contract), Some(NodeKind::DecisionContract));
    assert_eq!(detect_node_kind(&j.distribution), Some(NodeKind::Distribution));
    assert_eq!(detect_node_kind(&j.result), Some(NodeKind::DecisionResult));
    assert_eq!(
        detect_node_kind(&claim(&ClaimSpec::default()).to_bytes("c").unwrap()),
        Some(NodeKind::ExternalClaim)
    );
    assert_eq!(
        detect_node_kind(
            &SensitivityArtifact::new(sensitivity_parts()).unwrap().to_bytes("s").unwrap()
        ),
        Some(NodeKind::Sensitivity)
    );
    assert_eq!(
        detect_node_kind(&ranking(&["digest".to_owned()]).to_bytes("r").unwrap()),
        Some(NodeKind::StudyRanking)
    );
    assert_eq!(detect_node_kind(b"not a container"), None);

    // An embedded kind with no verifier never verifies.
    let consumed = run(
        vec![BundleNode::embedded(
            "causal",
            NodeKind::CausalContract,
            "program",
            b"bytes".to_vec(),
        )],
        &[],
        &SuppliedSources::default(),
    );
    assert_eq!(failed_stage(&consumed, "causal"), BundleStage::UnknownNodeKind);

    // An artifact of the wrong kind in a node is refused by its own consumer.
    let mismatched = run(
        vec![BundleNode::embedded(
            "contract",
            NodeKind::DecisionContract,
            "id",
            j.distribution.clone(),
        )],
        &[],
        &SuppliedSources::default(),
    );
    assert!(matches!(
        mismatched.node("contract").unwrap().status,
        NodeStatus::Failed { stage: BundleStage::SwappedEvidence, .. }
    ));

    // A describe of a corrupted artifact is a failure, not a panic.
    let mut corrupt = j.distribution.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(describe_artifact(NodeKind::Distribution, &corrupt).is_err());
    let facts: BTreeMap<String, String> =
        describe_artifact(NodeKind::Distribution, &j.distribution).unwrap().facts;
    assert_eq!(facts.get(FACT_LAW).map(String::as_str), Some(LAW_JOINT));
}
