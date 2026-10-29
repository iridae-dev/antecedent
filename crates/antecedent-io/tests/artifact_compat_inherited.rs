//! Compatibility class of artifacts written by the previous release, loaded by the
//! current consumers.
//!
//! The bytes in `tests/fixtures/inherited/` are the previous release's cross-surface
//! fixtures, copied unchanged from version control. They are never resealed or
//! regenerated: this file asserts what the current consumer does with them.
//!
//! | family | class | why |
//! | --- | --- | --- |
//! | `joint_distribution_v1` | byte-compatible | metadata and draws are unchanged; only the
//! |   | | manifest's producing library version differs |
//! | `external_response_claim_v1` | requires regeneration | the lineage digests and the
//! |   | | provider fingerprint are derived under new identity domains, so the stored
//! |   | | identity no longer equals the one a consumer derives from its checked contract |
//!
//! Regeneration of the second: bind the provider response again against the checked
//! contract and write it with `ExternalClaimArtifact::from_bound_claim(..).to_bytes(..)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionProvenance,
    DistributionTrust, DrawAlignment,
};
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, ExternalClaimTrust, lineage_wire, values_digest,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_test_support::read_fixture_bytes;

const HINT: &str = "the inherited fixtures are copies of the previous release's \
    conformance/cross_surface/rust_{joint_law,external_claim}.bin; restore them from version \
    control and do not regenerate them";

fn inherited(name: &str) -> Vec<u8> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inherited");
    read_fixture_bytes(&dir, name, HINT)
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

fn law_quantities() -> Vec<ScientificQuantity> {
    ["x", "y"]
        .iter()
        .map(|name| ScientificQuantity {
            variable_id: (*name).into(),
            variable_name: (*name).into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        })
        .collect()
}

fn law_identity() -> DistributionIdentity {
    DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &law_quantities(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated-law".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "law-snapshot".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap()
}

/// Outcome `y` in mmHg at dose `d`.
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

/// The identity a current consumer writes from constants for the same checked contract.
fn current_claim_identity() -> ExternalClaimIdentity {
    let mut identity = ExternalClaimIdentity {
        causal_contract_id: "checked-contract".into(),
        graph_id: "graph-1".into(),
        identification: "nonparametrically_identified".into(),
        quantities: (0..3).map(|d| ScientificQuantityWire::from(&claim_quantity(d))).collect(),
        provider_id: "lab".into(),
        object_id: "curve".into(),
        version_id: "v3".into(),
        snapshot_id: "snap-9".into(),
        request_id: "req-1".into(),
        values_blake3: values_digest(&[1.0, 3.0, 5.0]),
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

/// Byte-compatible: the previous release's joint law loads under the current identity and
/// reports the enumerated truth (two equally weighted aligned draws (0, 0) and (1, 2):
/// means 1/2 and 1, covariance 1/2, E[XY] = 1).
#[test]
fn an_inherited_joint_law_is_byte_compatible() {
    let loaded = DistributionArtifact::from_bytes(&inherited("joint_law.bin"), &law_identity())
        .expect("the inherited joint law loads under the current identity");
    assert!(near(loaded.mean(0).unwrap(), 0.5));
    assert!(near(loaded.mean(1).unwrap(), 1.0));
    assert!(near(loaded.covariance(0, 1).unwrap(), 0.5));
    assert!(near(loaded.joint_expectation(0, 1, |x, y| x * y).unwrap(), 1.0));
    assert_eq!(loaded.metadata().identity, law_identity());
    assert_eq!(loaded.metadata().trust, DistributionTrust::Unverified);
    assert_eq!(loaded.metadata().calibration, DistributionCalibration::Exact);
    // Re-encoding the loaded artifact preserves the draws bit for bit.
    let reloaded = DistributionArtifact::from_bytes(
        &loaded.to_bytes("inherited-joint-law").unwrap(),
        &law_identity(),
    )
    .unwrap();
    assert_eq!(reloaded.draws(), loaded.draws());
}

/// Requires regeneration: the stored claim identity is not the identity a current
/// consumer derives, so the claim is refused rather than trusted.
#[test]
fn an_inherited_external_claim_requires_regeneration() {
    let error = ExternalClaimArtifact::from_bytes(
        &inherited("external_claim.bin"),
        &current_claim_identity(),
    )
    .expect_err("the stored identity differs from the current derivation");
    let text = error.to_string();
    assert!(text.contains("external_claim.identity_expected"), "{text}");
    // Corruption is still a different failure: the container itself is intact.
    let mut flipped = inherited("external_claim.bin");
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xFF;
    let corrupt = ExternalClaimArtifact::from_bytes(&flipped, &current_claim_identity())
        .expect_err("a flipped byte is refused")
        .to_string();
    assert_eq!(corrupt, "checksum mismatch for section `external_claim.meta`");
}
