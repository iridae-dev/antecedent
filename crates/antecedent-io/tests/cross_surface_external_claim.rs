//! Cross-surface external response claim: Python-built bytes consumed here, and
//! Rust-built bytes written for the Python consumer.
//!
//! Fixtures live in `conformance/cross_surface/`. A missing `py_*` fixture fails
//! the test: run `python python/tests/generate_cross_surface_fixtures.py`.
//! To refresh the `rust_*` fixtures:
//! `ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-io --test
//! cross_surface_external_claim -- --ignored regenerate_rust_fixtures`.
//!
//! Truth is the enumerated closed form `E[Y | do(a)] = 1 + 2a` over `a` in
//! `0, 1, 2`, i.e. `[1, 3, 5]`, with every identity built from constants.

use antecedent_core::{
    CheckedCausalContract, CompositionStage, DistributionMeaning, ExternalCapability,
    ExternalResponse, ExternalResult, ExternalResultHeader, ExternalScientificObject,
    ExternalTrustState, ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, SupportStatus, bind_external_result,
};
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, ExternalClaimTrust, lineage_wire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/cross_surface")
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing cross-surface fixture {} ({error}); run \
             `python python/tests/generate_cross_surface_fixtures.py` from the repo root \
             (Rust-built fixtures: ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-io \
             --test cross_surface_external_claim -- --ignored regenerate_rust_fixtures)",
            path.display()
        )
    })
}

/// Outcome `y` in mmHg; the Python surface derives `variable_id == variable_name == "y"`.
fn quantity(dose: u32) -> ScientificQuantity {
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

/// Consumer-retained identity, written from constants rather than any artifact.
fn expected_identity() -> ExternalClaimIdentity {
    let mut identity = ExternalClaimIdentity {
        causal_contract_id: "checked-contract".into(),
        graph_id: "graph-1".into(),
        identification: "nonparametrically_identified".into(),
        quantities: (0..3).map(|d| ScientificQuantityWire::from(&quantity(d))).collect(),
        provider_id: "lab".into(),
        object_id: "curve".into(),
        version_id: "v3".into(),
        snapshot_id: "snap-9".into(),
        request_id: "req-1".into(),
        values_blake3: {
            let mut hasher = blake3::Hasher::new();
            for value in [1.0_f64, 3.0, 5.0] {
                hasher.update(&value.to_le_bytes());
            }
            hasher.finalize().to_hex().to_string()
        },
        trust: ExternalClaimTrust::ExternallyAttested,
        point_status: vec![
            "supported".into(),
            "supported".into(),
            "outside_empirical_support".into(),
        ],
        uncertainty_method: None,
        evidence_ids: vec!["factor:z".into()],
        assumption_ids: vec!["ignorability".into()],
        equivalence_ids: vec![],
        // Digests come from the shared core algorithm over the constants below.
        lineage: lineage_wire(&[
            ("contract:checked-contract", "causal_contract", &[]),
            ("evidence:factor:z", "evidence", &["contract:checked-contract"]),
            ("provider:external:lab/curve@v3#snap-9", "external_provider", &["evidence:factor:z"]),
            (
                "claim",
                "claim",
                &["contract:checked-contract", "provider:external:lab/curve@v3#snap-9"],
            ),
        ])
        .unwrap(),
        provider_meaning: "interventional_predictive".into(),
        capabilities: vec!["mean".into()],
        provider_fingerprint: String::new(),
        verification: None,
    };
    identity.provider_fingerprint = identity.compute_fingerprint();
    identity
}

fn bound_artifact() -> ExternalClaimArtifact {
    let quantities: Vec<_> = (0..3).map(quantity).collect();
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
    let claim = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    ExternalClaimArtifact::from_bound_claim(&claim, "checked-contract").unwrap()
}

fn flip_last_byte(bytes: &[u8]) -> Vec<u8> {
    let mut copy = bytes.to_vec();
    *copy.last_mut().expect("non-empty artifact") ^= 0xFF;
    copy
}

#[test]
fn python_built_claim_loads_under_the_constants_built_identity() {
    let bytes = read_fixture("py_external_claim.bin");
    let loaded = ExternalClaimArtifact::from_bytes(&bytes, &expected_identity()).unwrap();

    let values = loaded.values();
    for (got, want) in values.iter().zip([1.0_f64, 3.0, 5.0]) {
        assert!((got - want).abs() < 1e-12, "{got} != {want}");
    }
    assert_eq!(values.len(), 3);
    assert_eq!(loaded.metadata().identity.trust, ExternalClaimTrust::ExternallyAttested);
    assert_eq!(loaded.provenance_label(), "external:lab/curve@v3#snap-9");
    assert!(!loaded.metadata().native_estimation);
    assert_eq!(
        loaded.metadata().identity.point_status,
        ["supported", "supported", "outside_empirical_support"]
    );

    let chain = loaded.metadata().identity.provenance_chain().unwrap();
    let behind = chain.stages_behind("claim").unwrap();
    for stage in [
        CompositionStage::CausalContract,
        CompositionStage::Evidence,
        CompositionStage::ExternalProvider,
    ] {
        assert!(behind.contains(&stage), "missing lineage stage {stage:?}");
    }
}

#[test]
fn python_identity_json_agrees_with_the_constants_but_is_never_trusted() {
    // A cross-check only: loading above uses `expected_identity()` built here.
    let json = read_fixture("py_external_claim.identity.json");
    let from_python: ExternalClaimIdentity = serde_json::from_slice(&json).unwrap();
    assert_eq!(from_python, expected_identity());
}

#[test]
fn python_built_claim_refuses_changed_identity_and_changed_bytes() {
    let bytes = read_fixture("py_external_claim.bin");
    let mut other_snapshot = expected_identity();
    other_snapshot.snapshot_id = "other-snapshot".into();
    other_snapshot.provider_fingerprint = other_snapshot.compute_fingerprint();
    assert!(ExternalClaimArtifact::from_bytes(&bytes, &other_snapshot).is_err());

    let mut verified = expected_identity();
    verified.trust = ExternalClaimTrust::ExactRequestVerified;
    assert!(ExternalClaimArtifact::from_bytes(&bytes, &verified).is_err());

    assert!(
        ExternalClaimArtifact::from_bytes(&flip_last_byte(&bytes), &expected_identity()).is_err()
    );
    assert!(
        ExternalClaimArtifact::from_bytes(&bytes[..bytes.len() - 5], &expected_identity()).is_err()
    );
}

#[test]
fn rust_construction_matches_the_constants_built_identity() {
    // The Rust producer and the retained constants agree before any bytes are shared.
    assert_eq!(bound_artifact().metadata().identity, expected_identity());
}

/// Writes the Rust-built fixtures the Python tests consume.
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
    let artifact = bound_artifact();
    std::fs::write(dir.join("rust_external_claim.bin"), artifact.to_bytes("claim").unwrap())
        .unwrap();
    std::fs::write(
        dir.join("rust_external_claim.identity.json"),
        serde_json::to_vec_pretty(&expected_identity()).unwrap(),
    )
    .unwrap();
}
