//! C3 cross-surface composition bundle: a Python-built bundle consumed here, and a
//! Rust-built bundle written for the Python consumer.
//!
//! Fixtures live in `conformance/cross_surface/`. A missing `py_*` fixture fails the
//! test: run `python python/tests/generate_cross_surface_bundle_fixtures.py` from the
//! repo root. To refresh the Rust-built bundle:
//! `ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test
//! cross_surface_bundle -- --ignored regenerate_rust_fixtures`.
//!
//! The bundle holds two hand-derived decisions:
//!
//! * joint law (`contract`, `law`, `result`): rows `p = [1, 3, 2, 0]`,
//!   `q = [4, 0, 2, 6]`, `safe = 3`, so `E[p * q] = (4 + 0 + 4 + 0) / 4 = 2` and
//!   `safe = max(3, 0) = 3`;
//! * point only (`claim`, `mean_contract`, `mean_result`): `E[Y | do(a)] = 1 + 2a` gives
//!   `[1, 3, 5]`, so `wait` (`a = 0`) is worth `2 * 1 - 1 = 1` and `treat` (`a = 2`) is
//!   worth `2 * 5 - 1 = 9`.
//!
//! Every consumer-side identity is built here from constants: the expected bundle is
//! assembled from Rust builders, and the identity file Python writes is only compared.

use antecedent_core::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalResponse,
    ExternalResult, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
    ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract, ProviderObjectIdentity,
    QuantityRole, ScientificQuantity, SupportStatus, bind_external_result,
};
use antecedent_design::composition_bundle::{
    BundleLimits, BundleNode, ClaimLabel, CompositionBundle, ConsumedBundle, FACT_LAW,
    FACT_REQUIRES_LAW, FACT_TRUST, LAW_JOINT, LAW_MEAN_ONLY, NodeStatus, SuppliedSources,
};
use antecedent_design::composition_verifiers::{
    TRUST_ATTESTED, describe_artifact, detect_node_kind, mean_decision_bytes, standard_consumer,
};
use antecedent_design::decision_artifact::{DecisionContractArtifact, DecisionResultArtifact};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::evaluate_contract;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::external_claim_artifact::ExternalClaimArtifact;
use antecedent_io::quantity_wire::DistributionMeaningWire;

const PY_COMMAND: &str = "python python/tests/generate_cross_surface_bundle_fixtures.py";
const EDGES: [(&str, &str); 4] = [
    ("claim", "mean_result"),
    ("contract", "result"),
    ("law", "result"),
    ("mean_contract", "mean_result"),
];

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/cross_surface")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing cross-surface fixture {} ({error}); run `{PY_COMMAND}` from the repo root \
             (Rust-built fixtures: ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design \
             --test cross_surface_bundle -- --ignored regenerate_rust_fixtures)",
            path.display()
        )
    })
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

// ---------------------------------------------------------------------------
// The joint-law decision, declared from constants
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

// ---------------------------------------------------------------------------
// The external mean claim and the point-only decision, declared from constants
// ---------------------------------------------------------------------------

/// Outcome `y` in mmHg; the Python surface derives `variable_id == variable_name == "y"`.
fn claim_quantity(dose: u32) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
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

fn claim() -> ExternalClaimArtifact {
    let quantities: Vec<ScientificQuantity> = (0..3).map(claim_quantity).collect();
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "curve".into(),
            version_id: "v3".into(),
            snapshot_id: "snap-9".into(),
            request_id: "req-1".into(),
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
        required_evidence_ids: vec!["factor:z".into()],
        required_assumption_ids: vec!["ignorability".into()],
        equivalences: vec![],
    };
    let response = ExternalResponse {
        header: ExternalResultHeader {
            object,
            graph_id: "graph-1".into(),
            quantities,
            evidence_ids: vec!["factor:z".into()],
            assumption_ids: vec!["ignorability".into()],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        },
        // Closed form: E[Y | do(a)] = 1 + 2a.
        values: vec![1.0, 3.0, 5.0],
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: Some(vec![
            SupportStatus::Supported,
            SupportStatus::Supported,
            SupportStatus::OutsideEmpiricalSupport,
        ]),
    };
    let bound = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    ExternalClaimArtifact::from_bound_claim(&bound, "checked-contract").unwrap()
}

/// `wait` reads `a = 0` and `treat` reads `a = 2`; utility `2 * mean - 1`.
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

// ---------------------------------------------------------------------------
// The bundle
// ---------------------------------------------------------------------------

fn embed(id: &str, bytes: &[u8]) -> BundleNode {
    let kind = detect_node_kind(bytes).expect("a bundle artifact kind");
    let identity = describe_artifact(kind, bytes).expect("the artifact decodes").identity;
    BundleNode::embedded(id, kind, &identity, bytes.to_vec())
}

/// The bundle both surfaces declare, assembled from Rust builders.
fn rust_bundle() -> CompositionBundle {
    let source = joint_source();
    let contract = joint_contract();
    let result = evaluate_contract(&contract, &source).unwrap();
    let claim_bytes = claim().to_bytes("claim").unwrap();
    let means = mean_contract();
    let mean_result = mean_decision_bytes(&means, &claim_bytes, "mean-result").unwrap();
    let nodes = vec![
        embed(
            "contract",
            &DecisionContractArtifact::new(contract).unwrap().to_bytes("contract").unwrap(),
        ),
        embed("law", &source.to_bytes("law").unwrap()),
        embed("result", &DecisionResultArtifact::new(result, &source).to_bytes("result").unwrap()),
        embed("claim", &claim_bytes),
        embed(
            "mean_contract",
            &DecisionContractArtifact::new(means).unwrap().to_bytes("mean-contract").unwrap(),
        ),
        embed("mean_result", &mean_result),
    ];
    let edges: Vec<(String, String)> =
        EDGES.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect();
    CompositionBundle::new(nodes, &edges, &BundleLimits::default()).unwrap()
}

fn consume(bytes: &[u8], identity: &str) -> ConsumedBundle {
    standard_consumer()
        .consume(bytes, &BundleLimits::default(), identity, &SuppliedSources::default())
        .unwrap()
}

fn fact<'a>(consumed: &'a ConsumedBundle, id: &str, key: &str) -> Option<&'a str> {
    consumed.node(id).unwrap().facts.get(key).map(String::as_str)
}

fn assert_hand_derived_truth(consumed: &ConsumedBundle) {
    consumed.require_verified().unwrap();
    assert!(consumed.all_verified());
    close(consumed.value("result", "risky.expected_utility").unwrap(), 2.0);
    close(consumed.value("result", "safe.expected_utility").unwrap(), 3.0);
    close(consumed.value("mean_result", "wait.expected_utility").unwrap(), 1.0);
    close(consumed.value("mean_result", "treat.expected_utility").unwrap(), 9.0);
    assert_eq!(consumed.node("result").unwrap().claim_label, Some(ClaimLabel::JointDraw));
    assert_eq!(
        consumed.node("mean_result").unwrap().claim_label,
        Some(ClaimLabel::PointOnlyAttested)
    );
    assert_eq!(fact(consumed, "law", FACT_LAW), Some(LAW_JOINT));
    assert_eq!(fact(consumed, "result", FACT_REQUIRES_LAW), Some(LAW_JOINT));
    assert_eq!(fact(consumed, "claim", FACT_LAW), Some(LAW_MEAN_ONLY));
    assert_eq!(fact(consumed, "claim", FACT_TRUST), Some(TRUST_ATTESTED));
    assert_eq!(fact(consumed, "mean_contract", FACT_REQUIRES_LAW), None);
    assert_eq!(consumed.nodes().len(), 6);
    assert_eq!(consumed.edges().len(), 4);
}

/// The Python-built bundle verifies under an identity assembled here from constants,
/// and reads the hand-derived values.
#[test]
fn c3_xsurface_python_built_bundle_verifies_under_the_constants_identity() {
    let expected = rust_bundle();
    let consumed = consume(&read_fixture("py_composition_bundle.bin"), expected.identity());
    assert_eq!(consumed.identity(), expected.identity());
    assert_hand_derived_truth(&consumed);
    for node in consumed.nodes() {
        assert_eq!(node.status, NodeStatus::Verified, "node `{}`", node.id);
        // Merkle digests agree with the Rust-assembled twin, node by node.
        assert_eq!(Some(node.chain_digest.as_str()), expected.chain_digest(&node.id));
    }
}

/// The identity file Python writes agrees with the constants but is never trusted.
#[test]
fn c3_xsurface_python_identity_file_agrees_with_the_constants_but_is_never_trusted() {
    let json: serde_json::Value =
        serde_json::from_slice(&read_fixture("py_composition_bundle.identity.json")).unwrap();
    assert_eq!(json["bundle_identity"], rust_bundle().identity());
    assert_eq!(json["claim_label"], "point_only_attested");
    assert_eq!(json["values"]["result.risky.expected_utility"], 2.0);
    assert_eq!(json["values"]["result.safe.expected_utility"], 3.0);
    assert_eq!(json["values"]["mean_result.wait.expected_utility"], 1.0);
    assert_eq!(json["values"]["mean_result.treat.expected_utility"], 9.0);
}

/// Another retained identity or a changed byte refuses the Python-built bundle.
#[test]
fn c3_xsurface_python_built_bundle_refuses_another_identity_and_changed_bytes() {
    let bytes = read_fixture("py_composition_bundle.bin");
    let identity = rust_bundle().identity().to_owned();

    let wrong = standard_consumer().consume(
        &bytes,
        &BundleLimits::default(),
        &"0".repeat(64),
        &SuppliedSources::default(),
    );
    let refusal = wrong.unwrap_err().to_refusal().unwrap();
    assert_eq!(refusal.detail, "composition_bundle.expected_identity_mismatch");

    let mut flipped = bytes;
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xFF;
    let changed = standard_consumer().consume(
        &flipped,
        &BundleLimits::default(),
        &identity,
        &SuppliedSources::default(),
    );
    assert!(changed.is_err());
}

/// The Rust-assembled bundle round-trips through its own bytes, so the Python
/// consumer is handed bytes this side already verifies.
#[test]
fn c3_xsurface_rust_built_bundle_round_trips_in_process() {
    let bundle = rust_bundle();
    let bytes = bundle.to_bytes("composition-bundle", &BundleLimits::default()).unwrap();
    assert_hand_derived_truth(&consume(&bytes, bundle.identity()));
}

/// Writes the Rust-built bundle the Python tests consume.
#[test]
#[ignore = "writes conformance/cross_surface; set ANTECEDENT_WRITE_FIXTURES=1"]
fn regenerate_rust_fixtures() {
    assert_eq!(
        std::env::var("ANTECEDENT_WRITE_FIXTURES").as_deref(),
        Ok("1"),
        "set ANTECEDENT_WRITE_FIXTURES=1 to write fixtures"
    );
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let bundle = rust_bundle();
    std::fs::write(
        dir.join("rust_composition_bundle.bin"),
        bundle.to_bytes("composition-bundle", &BundleLimits::default()).unwrap(),
    )
    .unwrap();
}
