//! Independent-process consumption of a bound external response-grid claim.

use antecedent_core::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalResponse,
    ExternalResult, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
    ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract, ProviderObjectIdentity,
    QuantityRole, ScientificQuantity, SupportStatus, bind_external_result,
};
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, ExternalClaimTrust, lineage_wire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

/// Constants-built lineage; the Merkle digests are computed by the shared core
/// algorithm from the ids, stages and parents written here.
fn expected_lineage() -> Vec<antecedent_io::external_claim_artifact::LineageLinkWire> {
    lineage_wire(&[
        ("contract:checked-contract", "causal_contract", &[]),
        ("evidence:factor:z", "evidence", &["contract:checked-contract"]),
        ("provider:external:lab/curve@v3#snap-9", "external_provider", &["evidence:factor:z"]),
        ("claim", "claim", &["contract:checked-contract", "provider:external:lab/curve@v3#snap-9"]),
    ])
    .unwrap()
}

fn quantity(dose: u32) -> ScientificQuantity {
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

/// Consumer-retained identity, written from constants rather than the artifact.
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
        lineage: expected_lineage(),
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

#[test]
fn lineage_records_a_merkle_digest_per_link_and_refuses_tampered_digests() {
    let artifact = bound_artifact();
    let identity = &artifact.metadata().identity;
    let lineage = &identity.lineage;
    assert_eq!(lineage.len(), 4);
    assert!(
        lineage.iter().all(|l| l.digest.len() == 64 && l.parent_digests.len() == l.parents.len())
    );
    assert!(lineage[0].parent_digests.is_empty());
    assert_eq!(lineage[1].parent_digests, [lineage[0].digest.clone()]);
    assert_eq!(lineage[3].parent_digests, [lineage[0].digest.clone(), lineage[2].digest.clone()]);
    let chain = identity.provenance_chain().unwrap();
    assert_eq!(chain.digest_of("claim").unwrap(), lineage[3].digest);

    let refused = |edit: &dyn Fn(&mut ExternalClaimIdentity)| {
        let mut meta = artifact.metadata().clone();
        edit(&mut meta.identity);
        let error = meta.identity.provenance_chain().unwrap_err();
        assert_eq!(error.reason_code(), Some("external_binding_mismatch"), "{error}");
        assert!(ExternalClaimArtifact::new(meta, artifact.values().to_vec()).is_err());
    };
    // A changed own digest, a changed declared predecessor digest, and a dropped one.
    refused(&|i| i.lineage[1].digest = "0".repeat(64));
    refused(&|i| i.lineage[2].parent_digests[0] = "0".repeat(64));
    refused(&|i| {
        i.lineage[3].parent_digests.pop();
    });
    // Relabelling a display id while keeping every carried digest refuses.
    refused(&|i| {
        i.lineage[1].id = "evidence:renamed".into();
        i.lineage[2].parents[0] = "evidence:renamed".into();
    });
}

#[test]
fn fresh_process_consumer_recomputes_closed_form_and_refuses_mutations() {
    const PATH_ENV: &str = "ANTECEDENT_23_EXTERNAL_CLAIM_TEST_PATH";
    if let Ok(path) = std::env::var(PATH_ENV) {
        let bytes = std::fs::read(path).unwrap();
        let loaded = ExternalClaimArtifact::from_bytes(&bytes, &expected_identity()).unwrap();
        let v = loaded.values();
        assert!(((v[2] - v[0]) / 2.0 - 2.0).abs() < 1e-12);
        assert!((v[0] - 1.0).abs() < 1e-12);
        assert_eq!(loaded.provenance_label(), "external:lab/curve@v3#snap-9");
        assert!(!loaded.metadata().native_estimation);
        // The consumer can answer where the numbers came from without the producer.
        let chain = loaded.metadata().identity.provenance_chain().unwrap();
        let behind = chain.stages_behind("claim").unwrap();
        for stage in [
            antecedent_core::CompositionStage::CausalContract,
            antecedent_core::CompositionStage::Evidence,
            antecedent_core::CompositionStage::ExternalProvider,
        ] {
            assert!(behind.contains(&stage));
        }

        let reseal = |edit: &dyn Fn(&mut ExternalClaimIdentity)| {
            let mut meta = loaded.metadata().clone();
            edit(&mut meta.identity);
            // A structurally invalid identity is refused even before resealing.
            ExternalClaimArtifact::new(meta, v.to_vec()).ok().map(|a| a.to_bytes("claim").unwrap())
        };
        let edits: [&dyn Fn(&mut ExternalClaimIdentity); 10] = [
            &|i| i.lineage[1].id = "evidence:other".into(),
            &|i| i.lineage.truncate(3),
            &|i| i.point_status[2] = "supported".into(),
            &|i| i.snapshot_id = "other".into(),
            &|i| i.trust = ExternalClaimTrust::ExactRequestVerified,
            &|i| i.graph_id = "graph-2".into(),
            &|i| i.uncertainty_method = Some("interval".into()),
            &|i| i.equivalence_ids = vec!["backdoor:z".into()],
            &|i| i.quantities.swap(0, 1),
            &|i| i.request_id = "req-2".into(),
        ];
        for edit in edits {
            if let Some(bytes) = reseal(edit) {
                assert!(ExternalClaimArtifact::from_bytes(&bytes, &expected_identity()).is_err());
            }
        }
        // A container can be resealed with valid section checksums, but the
        // independently retained identity still fixes the actual values.
        let mut changed_values = v.to_vec();
        changed_values[1] = 30.0;
        assert!(
            ExternalClaimArtifact::new(loaded.metadata().clone(), changed_values.clone()).is_err()
        );
        let mut resealed_meta = loaded.metadata().clone();
        resealed_meta.identity.values_blake3 = {
            let mut hasher = blake3::Hasher::new();
            for value in &changed_values {
                hasher.update(&value.to_le_bytes());
            }
            hasher.finalize().to_hex().to_string()
        };
        let resealed = ExternalClaimArtifact::new(resealed_meta, changed_values)
            .unwrap()
            .to_bytes("claim")
            .unwrap();
        assert!(ExternalClaimArtifact::from_bytes(&resealed, &expected_identity()).is_err());
        let mut native = loaded.metadata().clone();
        native.native_estimation = true;
        assert!(ExternalClaimArtifact::new(native, v.to_vec()).is_err());
        assert!(
            ExternalClaimArtifact::from_bytes(&bytes[..bytes.len() - 5], &expected_identity())
                .is_err()
        );
        return;
    }
    let artifact = bound_artifact();
    assert_eq!(artifact.metadata().identity, expected_identity());
    let unique =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir()
        .join(format!("antecedent-external-claim-{}-{unique}.bin", std::process::id()));
    std::fs::write(&path, artifact.to_bytes("claim").unwrap()).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("fresh_process_consumer_recomputes_closed_form_and_refuses_mutations")
        .env(PATH_ENV, &path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
