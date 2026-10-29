//! Compatibility class of decision and design artifacts written by the previous
//! release, loaded by the current consumers.
//!
//! The bytes in `tests/fixtures/inherited/` are the previous release's cross-surface
//! fixtures, copied unchanged from version control. They are never resealed or
//! regenerated: this file asserts what the current consumer does with them.
//!
//! | family | class | why |
//! | --- | --- | --- |
//! | `joint_distribution_v1` (decision source) | byte-compatible | metadata and draws are
//! |   | | unchanged; only the manifest's producing library version differs |
//! | `decision_contract_v1` | requires regeneration | the canonical contract identity is
//! |   | | derived under a new domain, so the stored identity is never the recomputed one |
//! | `decision_result_v1` | requires regeneration | the contract identity, the source digest
//! |   | | and every lineage digest are derived under new domains |
//! | `design_ranking_v1` | inspectable only | the wire layout and version are unchanged and
//! |   | | the body decodes; the signal and request fingerprints are derived differently, so
//! |   | | the consumer's recomputation refuses it |
//! | `composition_bundle_v1` | requires regeneration | node identities and edge digests are
//! |   | | derived from embedded artifacts whose identities moved; no node verifies |
//!
//! Regeneration of each is a re-declaration and re-export from the retained inputs: the
//! decision contract JSON and the source distribution (both byte-compatible) reproduce the
//! contract, the result and the bundle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_design::composition_bundle::{BundleLimits, SuppliedSources};
use antecedent_design::composition_verifiers::standard_consumer;
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, source_digest,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::design_ranking_artifact::{
    ConsumeExpectation, DESIGN_RANKING_ARTIFACT_KIND, DESIGN_RANKING_ARTIFACT_VERSION,
    DesignRankingArtifactWire, consume,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionProvenance,
    DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;
use antecedent_test_support::read_fixture_bytes;

mod common;

const HINT: &str = "the inherited fixtures are copies of the previous release's \
    conformance/cross_surface/rust_*.bin; restore them from version control and do not \
    regenerate them";

/// Identity and digests stored in the previous release's decision artifacts.
const OLD_CONTRACT_IDENTITY: &str =
    "ea64a7afcfec83a5b786fc865a9cc61be89feaa48a40043c11dea58887307b09";
const OLD_SOURCE_DIGEST: &str = "c50f07f8a5f477e6b2358ce4f1d4b7d0d9943f847e1749f93d7fba23e1aa9ad6";
const OLD_BUNDLE_IDENTITY: &str =
    "5ef964ff45f6987334ba3e05dd7e48884452a4ca6805f62df3f3d2bd6d3aca95";

fn inherited(name: &str) -> Vec<u8> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inherited");
    read_fixture_bytes(&dir, name, HINT)
}

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity { regime_id: regime.into(), ..common::quantity(variable) }
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

/// The consumer re-declares the contract from constants: the same declaration the
/// previous release sealed.
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

fn loaded_source() -> DistributionArtifact {
    DistributionArtifact::from_bytes(&inherited("decision_source.bin"), &source_identity())
        .expect("the inherited decision source loads under the current identity")
}

/// Byte-compatible: the draws of the inherited source are the enumerated `p`, `q` and
/// `safe` columns, read back bit for bit.
#[test]
fn an_inherited_decision_source_is_byte_compatible() {
    let source = loaded_source();
    assert_eq!(source.metadata().identity, source_identity());
    assert_eq!(source.metadata().trust, DistributionTrust::Unverified);
    assert_eq!(source.metadata().calibration, DistributionCalibration::Exact);
    let p = [1.0, 3.0, 2.0, 0.0];
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut expected = Vec::new();
    for i in 0..4 {
        expected.extend([p[i], q[i], 3.0]);
    }
    assert_eq!(source.draws(), expected.as_slice());
}

/// Requires regeneration: the digest a current consumer computes over the same draws is
/// not the one the previous release stored.
#[test]
fn the_source_digest_and_contract_identity_moved() {
    let source = loaded_source();
    let digest = source_digest(&source);
    assert_eq!(digest.len(), 64);
    assert_ne!(digest, OLD_SOURCE_DIGEST);
    let identity = contract().identity().unwrap();
    assert_eq!(identity.len(), 64);
    assert_ne!(identity, OLD_CONTRACT_IDENTITY);
}

#[test]
fn an_inherited_decision_contract_requires_regeneration() {
    let bytes = inherited("decision_contract.bin");
    // Under the identity a current consumer retains, and under the identity the artifact
    // itself stored, the recomputed identity differs: it is refused, not trusted.
    for retained in [contract().identity().unwrap(), OLD_CONTRACT_IDENTITY.to_owned()] {
        let error = DecisionContractArtifact::from_bytes(&bytes, &retained)
            .expect_err("the stored identity is not the recomputed one");
        let text = error.to_string();
        assert!(text.contains("decision_artifact.identity_expected"), "{text}");
    }
    // Regeneration from the declaration seals under the current identity and loads.
    let regenerated = DecisionContractArtifact::new(contract())
        .unwrap()
        .to_bytes("decision-contract-regenerated")
        .unwrap();
    let identity = contract().identity().unwrap();
    assert!(DecisionContractArtifact::from_bytes(&regenerated, &identity).is_ok());
}

#[test]
fn an_inherited_decision_result_requires_regeneration() {
    let bytes = inherited("decision_result.bin");
    let source = loaded_source();
    // A current consumer retains the current contract identity and source digest.
    let error = DecisionResultArtifact::from_bytes(
        &bytes,
        &contract().identity().unwrap(),
        &source_digest(&source),
    )
    .expect_err("the result was computed under another contract identity");
    assert!(error.to_string().contains("decision_artifact.contract_expected"), "{error}");
    // A consumer that retained the previous release's identity and digest passes the
    // binding checks and is refused at the lineage, whose digests moved.
    let error =
        DecisionResultArtifact::from_bytes(&bytes, OLD_CONTRACT_IDENTITY, OLD_SOURCE_DIGEST)
            .expect_err("the stored lineage digests are not the current derivation");
    assert!(error.to_string().contains("lineage"), "{error}");
}

#[test]
fn an_inherited_design_ranking_decodes_but_is_not_consumed() {
    let bytes = inherited("design_ranking.bin");
    let wire = DesignRankingArtifactWire::from_bytes(&bytes)
        .expect("the layout and major version are unchanged");
    assert_eq!(wire.version, DESIGN_RANKING_ARTIFACT_VERSION);
    assert_eq!(wire.kind, DESIGN_RANKING_ARTIFACT_KIND);
    assert_eq!(wire.candidates.len(), 2);
    // Recomputation refuses it: the signal identities it stored are not the ones the
    // current derivation produces.
    let error = consume(&bytes, &ConsumeExpectation::default()).unwrap_err();
    assert!(
        matches!(error, antecedent_design::design_ranking_artifact::DesignRankingArtifactError::DigestMismatch),
        "{error:?}"
    );
}

#[test]
fn an_inherited_composition_bundle_never_verifies() {
    let bytes = inherited("composition_bundle.bin");
    let outcome = standard_consumer().consume(
        &bytes,
        &BundleLimits::default(),
        OLD_BUNDLE_IDENTITY,
        &SuppliedSources::default(),
    );
    // The edge digests of the inherited bundle were derived from embedded artifacts whose
    // identities moved, so the consumer refuses at the edge-digest stage, before any node
    // could verify.
    let error = outcome.expect_err("an inherited bundle never consumes");
    assert_eq!(error.stage(), Some(antecedent_design::composition_bundle::BundleStage::EdgeDigestMismatch));
}
