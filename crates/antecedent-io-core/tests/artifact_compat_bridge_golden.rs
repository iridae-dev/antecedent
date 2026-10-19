//! Golden fixture for the scientific bridge artifact (`scientific_bridge_v1`).
//!
//! The golden lives at `tests/fixtures/scientific_bridge_v1.bin`. The loader reads it,
//! checks its container kind, consumes it as a fresh consumer (replaying the derivation
//! from the stored contract and source claim rather than trusting the stored value) and
//! requires the identity a producer derives today from the same declaration. A missing
//! golden fails with the command that writes it. A changed golden is a compatibility
//! decision and needs a migration note.
//!
//! Generate or refresh it with
//! `cargo test -p antecedent-io-core --test artifact_compat_bridge_golden -- --ignored write_`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;
use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, BridgedClaim, QuantityClass, TrialEffectPair, TrialLevelValidation,
    ValueKind, derive_bridged_claim,
};
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, QuantityRole, ReasoningView,
    ScientificQuantity, SlotAvailability,
};
use antecedent_core::{ContractIdentities, SemanticDigest};
use antecedent_io_core::bridge_artifact::{
    SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE, SCIENTIFIC_BRIDGE_ARTIFACT_VERSION,
    ScientificBridgeArtifact, ScientificBridgeArtifactError,
};
use antecedent_io_core::container_kind;
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str =
    "cargo test -p antecedent-io-core --test artifact_compat_bridge_golden -- --ignored write_";
const GOLDEN: &str = "scientific_bridge_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn quantity(variable: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "adults".into(),
        regime_id: "do(t=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn source_claim() -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(1),
        ContractIdentities::new(
            digest(10),
            digest(11),
            Some(digest(12)),
            Some(digest(13)),
            digest(14),
            digest(15),
            digest(16),
        ),
        ClaimKind::Point,
        Some(0.8),
        Some(Arc::from("mmol_per_l")),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        ClaimDomains::new(
            DomainStatus::Identified,
            DomainStatus::Supported,
            DomainStatus::Evaluated,
        ),
        Some(digest(77)),
        [Arc::<str>::from("prior:p0")],
    )
}

fn surrogate_contract() -> BridgeContract {
    let trials = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
        .into_iter()
        .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
            surrogate_effect,
            terminal_effect,
        })
        .collect();
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity("trial:biomarker", "mmol_per_l"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity("trial:survival", "log_hazard"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials,
            min_trials: 4,
            min_r_squared: 0.9,
        }),
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,

        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

fn derived() -> (BridgedClaim, ClaimEnvelope, ScientificQuantity) {
    let contract = surrogate_contract();
    let source = source_claim();
    let quantity = contract.source.quantity.clone();
    let bridged =
        derive_bridged_claim(&contract, &BridgeSource { claim: &source, quantity: &quantity })
            .expect("licensed bridge");
    (bridged, source, quantity)
}

fn fresh_identity() -> String {
    let (bridged, source, quantity) = derived();
    let (artifact, _) = ScientificBridgeArtifact::seal(
        &bridged,
        &BridgeSource { claim: &source, quantity: &quantity },
    )
    .expect("seal");
    artifact.identity().to_owned()
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_scientific_bridge_golden() {
    let (bridged, source, quantity) = derived();
    let bytes = ScientificBridgeArtifact::to_bytes(
        &bridged,
        &BridgeSource { claim: &source, quantity: &quantity },
        "golden-scientific-bridge",
    )
    .expect("bytes");
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(GOLDEN), bytes).expect("write the golden");
}

#[test]
fn the_scientific_bridge_golden_is_replayed_and_gated() {
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, &format!("generate it with `{GENERATOR}`"));
    assert_eq!(container_kind(&bytes).as_deref(), Some("scientific_bridge_v1"));

    let identity = fresh_identity();
    let consumed = ScientificBridgeArtifact::from_bytes(&bytes, Some(&identity))
        .expect("a fresh consumer replays the stored bridge");
    assert_eq!(consumed.identity(), identity);
    // Hand value of the surrogate bridge: slope 0.92, intercept 0.06, 0.92 * 0.8 + 0.06.
    assert!((consumed.bridged().value - 0.796).abs() <= 1e-12);
    assert!(!consumed.bridged().transported);

    // The version and feature gates refuse a resealed newer artifact.
    let (bridged, source, quantity) = derived();
    let (_, meta) = ScientificBridgeArtifact::seal(
        &bridged,
        &BridgeSource { claim: &source, quantity: &quantity },
    )
    .expect("seal");
    assert_eq!(meta.version, SCIENTIFIC_BRIDGE_ARTIFACT_VERSION);
    assert_eq!(meta.feature, SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE);
    let mut newer = meta;
    newer.version = SCIENTIFIC_BRIDGE_ARTIFACT_VERSION + 1;
    ScientificBridgeArtifact::reseal_identity(&mut newer);
    assert!(matches!(
        ScientificBridgeArtifact::from_meta(&newer, None),
        Err(ScientificBridgeArtifactError::UnsupportedVersion { version: 2 })
    ));
}
