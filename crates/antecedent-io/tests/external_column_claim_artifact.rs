//! A row-level column bound to a checked contract: digest, support and lineage, no rows.

use antecedent_contract::{
    CheckedCausalContract, ColumnSupport, DistributionMeaning, ExternalCapability, ExternalColumn,
    ExternalResult, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
    ExternalUncertaintyMeaning, LawProviderContract, ProviderObjectIdentity, QuantityRole,
    ScientificQuantity, bind_external_result,
};
use antecedent_core::{IdentificationStatus, SupportStatus};
use antecedent_io::external_claim_artifact::ExternalClaimArtifact;
use antecedent_io::external_column_claim_artifact::{
    ExternalColumnClaimArtifact, ExternalColumnIdentity,
};

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:y".into(),
        variable_name: "Y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: "do(a=1) - do(a=0)".into(),
        horizon: 0,
        functional_id: "cate".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn bound(values: &[f64], support: Option<ColumnSupport>) -> ExternalColumnClaimArtifact {
    let quantities = vec![quantity()];
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "learner".into(),
            object_id: "cate:y~a".into(),
            version_id: "v4".into(),
            snapshot_id: "rows-1".into(),
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
        required_evidence_ids: vec![],
        required_assumption_ids: vec!["ignorability".into()],
        equivalences: vec![],
    };
    let column = ExternalColumn {
        header: ExternalResultHeader {
            object,
            graph_id: "graph-1".into(),
            quantities,
            evidence_ids: vec![],
            assumption_ids: vec!["ignorability".into()],
            trust: ExternalTrustState::ExternallyAttested { attestor: "learner".into() },
        },
        values: values.to_vec(),
        uncertainty: ExternalUncertaintyMeaning::None,
        support,
    };
    let claim = bind_external_result(&contract, &ExternalResult::Column(column)).unwrap();
    ExternalColumnClaimArtifact::from_bound_claim(&claim, "checked-contract").unwrap()
}

#[test]
fn a_large_column_exports_a_small_claim_and_reloads_under_the_retained_identity() {
    let rows: Vec<f64> = (0..250_000).map(|i| f64::from(i % 97) * 0.01).collect();
    let support =
        ColumnSupport::PerRow(
            (0..rows.len())
                .map(|i| {
                    if i % 10 == 0 {
                        SupportStatus::Extrapolative
                    } else {
                        SupportStatus::Supported
                    }
                })
                .collect(),
        );
    let artifact = bound(&rows, Some(support));
    let bytes = artifact.to_bytes("cate-claim").unwrap();
    assert!(bytes.len() < 16 * 1024, "{} bytes carry no rows", bytes.len());

    let retained: ExternalColumnIdentity = artifact.metadata().identity.clone();
    assert_eq!(retained.column.rows, 250_000);
    assert_eq!(retained.claim.point_status, ["extrapolative"]);
    let loaded = ExternalColumnClaimArtifact::from_bytes(&bytes, &retained).unwrap();
    assert!(loaded.covers(&rows));
    let mut changed = rows;
    changed[41] += 1e-12;
    assert!(!loaded.covers(&changed));
    assert!(!loaded.covers(&changed[1..]));
    assert!(!loaded.metadata().native_estimation);
    assert_eq!(loaded.provenance_label(), "external:learner/cate:y~a@v4#rows-1");
}

#[test]
fn a_changed_column_digest_provider_or_support_is_another_claim() {
    let artifact = bound(&[0.1, 0.2, 0.3], None);
    let bytes = artifact.to_bytes("claim").unwrap();
    let retained = artifact.metadata().identity.clone();
    assert_eq!(retained.claim.point_status, ["missing_evidence"]);
    for edit in [
        (|i: &mut ExternalColumnIdentity| i.claim.values_blake3 = "0".repeat(64)) as fn(&mut _),
        |i| i.claim.snapshot_id = "other".into(),
        |i| i.column.rows = 4,
        |i| i.column.support[0].status = "supported".into(),
    ] {
        let mut expected = retained.clone();
        edit(&mut expected);
        assert!(ExternalColumnClaimArtifact::from_bytes(&bytes, &expected).is_err());
    }
}

#[test]
fn a_column_claim_is_never_read_as_a_response_grid() {
    let artifact = bound(&[0.1, 0.2], Some(ColumnSupport::Uniform(SupportStatus::Supported)));
    let bytes = artifact.to_bytes("claim").unwrap();
    let identity = artifact.metadata().identity.claim.clone();
    assert!(ExternalClaimArtifact::from_bytes(&bytes, &identity).is_err());
}
