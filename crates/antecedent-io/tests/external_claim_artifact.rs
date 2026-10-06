//! Independent-process consumption of a bound external response-grid claim.

use antecedent_core::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalResponse,
    ExternalResult, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
    ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract, ProviderObjectIdentity,
    QuantityRole, ScientificQuantity, bind_external_result,
};
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, ExternalClaimTrust,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

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
    ExternalClaimIdentity {
        causal_contract_id: "checked-contract".into(),
        graph_id: "graph-1".into(),
        identification: "nonparametrically_identified".into(),
        quantities: (0..3).map(|d| ScientificQuantityWire::from(&quantity(d))).collect(),
        provider_id: "lab".into(),
        object_id: "curve".into(),
        version_id: "v3".into(),
        snapshot_id: "snap-9".into(),
        request_id: "req-1".into(),
        trust: ExternalClaimTrust::ExternallyAttested,
        uncertainty_method: None,
        evidence_ids: vec!["factor:z".into()],
        assumption_ids: vec!["ignorability".into()],
        equivalence_ids: vec![],
    }
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
    };
    let claim = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    ExternalClaimArtifact::from_bound_claim(&claim, "checked-contract").unwrap()
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

        let reseal = |edit: &dyn Fn(&mut ExternalClaimIdentity)| {
            let mut meta = loaded.metadata().clone();
            edit(&mut meta.identity);
            ExternalClaimArtifact::new(meta, v.to_vec()).unwrap().to_bytes("claim").unwrap()
        };
        let edits: [&dyn Fn(&mut ExternalClaimIdentity); 7] = [
            &|i| i.snapshot_id = "other".into(),
            &|i| i.trust = ExternalClaimTrust::ExactRequestVerified,
            &|i| i.graph_id = "graph-2".into(),
            &|i| i.uncertainty_method = Some("interval".into()),
            &|i| i.equivalence_ids = vec!["backdoor:z".into()],
            &|i| i.quantities.swap(0, 1),
            &|i| i.request_id = "req-2".into(),
        ];
        for edit in edits {
            assert!(
                ExternalClaimArtifact::from_bytes(&reseal(edit), &expected_identity()).is_err()
            );
        }
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
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}
