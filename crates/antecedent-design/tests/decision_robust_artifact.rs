//! Durable admissible contracts and robust decision results: canonical identity,
//! round trip, resealed-mutation refusal, replay by recomputation, bounded decode
//! and external-callback receipts that never read as native verification.
//! Expected values are the hand-set payoffs of each structure.

use antecedent_core::{CompositionStage, QuantityRole, ScientificQuantity, SupportStatus};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, MAX_DECISION_ARTIFACT_BYTES, contract_to_json,
};
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, DeclaredExclusion, StructuralPolicy, SupportRule, UncertaintyKind,
    UncertaintyRequirement, UtilityExpr,
};
use antecedent_design::decision_robust_artifact::{
    AdmissibleContractArtifact, ExternalCallbackReceipt, ExternalTrustLimit, RobustResultArtifact,
    admissible_contract_from_json_refusal, admissible_contract_to_json, atom_digests_of,
};
use antecedent_design::decision_robustness::{
    AtomSupport, ClaimProfile, RobustDecisionResult, RobustVerdict, evaluate_robust,
};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::DistributionMeaningWire;
use antecedent_io::reader::ArtifactReader;
use antecedent_io::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

type ContractEdit = Box<dyn Fn(&mut AdmissibleDecisionContract)>;
type ReceiptEdit = Box<dyn Fn(&mut ExternalCallbackReceipt)>;

// ---------------------------------------------------------------------------
// Fixtures: two actions, each reading one quantity; a structure is the exact
// law of the payoffs of A and B on two equally likely rows.
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

fn law(a: f64, b: f64) -> Box<DistributionArtifact> {
    let columns = [quantity("a", "do(a=1)"), quantity("b", "do(a=0)")];
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

fn evaluated(id: &str, p: Option<f64>, a: f64, b: f64) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability: p, evidence: AtomEvidence::Evaluated(law(a, b)) }
}

fn base(policy: StructuralPolicy) -> DecisionContract {
    let action = |id: &str, variable: &str, regime: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(variable, regime)],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("A", "a", "do(a=1)"), action("B", "b", "do(a=0)")],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: policy,
    }
}

fn admissible(policy: StructuralPolicy, rules: AdmissibilityRules) -> AdmissibleDecisionContract {
    AdmissibleDecisionContract { contract: base(policy), rules }
}

fn rich_rules() -> AdmissibilityRules {
    AdmissibilityRules {
        default_weakest_support: Some(SupportStatus::WeakOverlap),
        support_rules: vec![
            SupportRule {
                action_id: "A".into(),
                input: 0,
                weakest_allowed: SupportStatus::Supported,
            },
            SupportRule {
                action_id: "B".into(),
                input: 0,
                weakest_allowed: SupportStatus::Extrapolative,
            },
        ],
        declared_exclusions: vec![
            DeclaredExclusion { action_id: "A".into(), reason: "legal".into() },
            DeclaredExclusion { action_id: "B".into(), reason: "not licensed".into() },
        ],
        uncertainty: UncertaintyRequirement::StructuralEnvelope,
    }
}

fn require_supported() -> AdmissibilityRules {
    AdmissibilityRules {
        default_weakest_support: Some(SupportStatus::Supported),
        ..AdmissibilityRules::default()
    }
}

fn profile(entries: Vec<(&str, AtomSupport)>) -> ClaimProfile {
    ClaimProfile {
        uncertainty: UncertaintyKind::StructuralEnvelope,
        support: entries.into_iter().map(|(id, s)| (id.to_owned(), s)).collect(),
    }
}

fn supported_profile() -> ClaimProfile {
    profile(vec![("s1", AtomSupport::supported()), ("s2", AtomSupport::supported())])
}

fn two_atoms() -> [StructuralAtom; 2] {
    [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)]
}

fn assess(
    c: &AdmissibleDecisionContract,
    atoms: &[StructuralAtom],
    claims: &ClaimProfile,
) -> RobustDecisionResult {
    evaluate_robust(c, atoms, claims).unwrap().1
}

fn message(error: &IoError) -> String {
    match error {
        IoError::Refused { message, .. } => message.clone(),
        other => format!("{other:?}"),
    }
}

fn external(atom: &str, trust: ExternalTrustLimit) -> ExternalCallbackReceipt {
    ExternalCallbackReceipt {
        atom_id: atom.into(),
        provider_id: "lab-model".into(),
        snapshot_id: "snap-9".into(),
        request_fingerprint: "d5a1c0ffee".repeat(6) + "d5a1",
        attested_value: 4.25,
        trust,
    }
}

fn attested() -> ExternalTrustLimit {
    ExternalTrustLimit::ExternallyAttested { attestor: "outside-lab".into() }
}

/// An artifact over the two-structure invariant decision, with `receipts`.
fn sealed(receipts: Vec<ExternalCallbackReceipt>) -> RobustResultArtifact {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = two_atoms();
    let claims = supported_profile();
    let result = assess(&c, &atoms, &claims);
    RobustResultArtifact::new(&c, result, &claims, &atoms, receipts).unwrap()
}

const ROBUST_KIND: &str = "robust_decision_result_v1";
const BODY: &str = "decision.body";

/// Edit the stored body and re-wrap it in a fresh, valid container: a mutation a
/// checksum alone cannot detect.
fn reseal(bytes: &[u8], kind: &str, edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes)).unwrap();
    let section = reader.load_section(BODY).unwrap();
    let mut value: serde_json::Value = from_cbor(section.as_bytes()).unwrap();
    edit(&mut value);
    let body = to_cbor(&value).unwrap();
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: antecedent_io::migrate::STABLE_FORMAT,
            minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(kind.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                .unwrap(),
            artifact_id: "resealed".into(),
            sections: vec![section_descriptor(BODY, "application/cbor", &body)],
            provenance: ProvenanceWire { note: "resealed".into() },
        },
        sections: vec![SectionBytes::new(BODY, body)],
    };
    let mut out = Vec::new();
    encoded.write_to(&mut out).unwrap();
    out
}

// ---------------------------------------------------------------------------
// F4: the admissible contract artifact
// ---------------------------------------------------------------------------

#[test]
fn f4_admissible_contract_round_trips_with_canonical_identity() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, rich_rules());
    let artifact = AdmissibleContractArtifact::new(c.clone()).unwrap();
    assert_eq!(artifact.identity(), c.identity().unwrap());
    let bytes = artifact.to_bytes("contract").unwrap();
    // Deterministic: the same contract writes the same bytes.
    assert_eq!(bytes, artifact.to_bytes("contract").unwrap());
    let loaded = AdmissibleContractArtifact::from_bytes(&bytes, artifact.identity()).unwrap();
    assert_eq!(loaded.contract(), &c);
    assert_eq!(loaded.contract().rules.uncertainty, UncertaintyRequirement::StructuralEnvelope);
    assert_eq!(loaded.contract().rules.default_weakest_support, Some(SupportStatus::WeakOverlap));
    assert_eq!(loaded.identity(), artifact.identity());
    // The JSON declaration a host language builds recovers the same contract.
    let json = admissible_contract_to_json(&c).unwrap();
    assert_eq!(admissible_contract_from_json_refusal(&json).unwrap(), c);
    // Rules are part of the identity: the base contract alone has another one.
    assert_ne!(artifact.identity(), c.contract.identity().unwrap());
}

#[test]
fn f4_admissible_identity_ignores_declaration_order_but_not_semantics() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, rich_rules());
    let id = c.identity().unwrap();

    let mut reordered = c.clone();
    reordered.contract.actions.reverse();
    reordered.rules.support_rules.reverse();
    reordered.rules.declared_exclusions.reverse();
    assert_eq!(reordered.identity().unwrap(), id);
    let a = AdmissibleContractArtifact::new(c.clone()).unwrap();
    let b = AdmissibleContractArtifact::new(reordered).unwrap();
    assert_eq!(a.identity(), b.identity());

    let edits: Vec<ContractEdit> = vec![
        Box::new(|c| c.rules.support_rules[0].weakest_allowed = SupportStatus::WeakOverlap),
        Box::new(|c| c.rules.default_weakest_support = None),
        Box::new(|c| c.rules.declared_exclusions[0].reason = "another reason".into()),
        Box::new(|c| {
            c.rules.declared_exclusions.pop();
        }),
        Box::new(|c| c.rules.uncertainty = UncertaintyRequirement::Credible),
        Box::new(|c| c.contract.structural_policy = StructuralPolicy::Maximin),
    ];
    for edit in edits {
        let mut changed = c.clone();
        edit(&mut changed);
        assert_ne!(changed.identity().unwrap(), id);
    }
}

#[test]
fn f4_resealed_rule_edit_is_refused_under_the_retained_identity() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, rich_rules());
    let original = AdmissibleContractArtifact::new(c.clone()).unwrap();
    let retained = original.identity().to_owned();

    // A self-consistent artifact of an edited rule set (its own identity matches
    // its own contents) is still not the contract the consumer retained.
    let mut edited = c;
    edited.rules.support_rules[0].weakest_allowed = SupportStatus::Extrapolative;
    let forged = AdmissibleContractArtifact::new(edited).unwrap().to_bytes("contract").unwrap();
    let error = AdmissibleContractArtifact::from_bytes(&forged, &retained).unwrap_err();
    assert!(message(&error).contains("identity_expected"), "{error:?}");

    // Editing the stored rules while keeping the stored identity refuses even
    // under the identity the stored bytes claim.
    let bytes = original.to_bytes("contract").unwrap();
    let resealed = reseal(&bytes, "admissible_decision_contract_v1", |body| {
        body["rules"]["uncertainty"] = serde_json::json!("credible");
    });
    let error = AdmissibleContractArtifact::from_bytes(&resealed, &retained).unwrap_err();
    assert!(message(&error).contains("identity_expected"), "{error:?}");

    // A flipped byte is a container error.
    let mut corrupt = bytes;
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0x55;
    assert!(AdmissibleContractArtifact::from_bytes(&corrupt, &retained).is_err());
}

#[test]
fn f4_contract_without_rules_keeps_a0_bytes_and_kinds_do_not_cross() {
    let plain = base(StructuralPolicy::ReportOnly);
    let a0 = DecisionContractArtifact::new(plain.clone()).unwrap();
    let a0_bytes = a0.to_bytes("contract").unwrap();
    // The A0 path is untouched: same kind, deterministic bytes, no rules field.
    assert_eq!(
        a0_bytes,
        DecisionContractArtifact::new(plain.clone()).unwrap().to_bytes("contract").unwrap()
    );
    let reader = ArtifactReader::open_seek(std::io::Cursor::new(&a0_bytes[..])).unwrap();
    assert_eq!(reader.manifest().artifact_kind, ArtifactKind::Other("decision_contract_v1".into()));
    let json = contract_to_json(&plain).unwrap();
    assert!(!json.contains("rules") && !json.contains("admissib"));
    let loaded = DecisionContractArtifact::from_bytes(&a0_bytes, a0.identity()).unwrap();
    assert_eq!(loaded.contract(), &plain);

    // Rules write a different artifact kind, so neither loader reads the other's
    // bytes and rules are never silently dropped.
    let c = AdmissibleDecisionContract { contract: plain, rules: rich_rules() };
    let rules_bytes =
        AdmissibleContractArtifact::new(c.clone()).unwrap().to_bytes("contract").unwrap();
    assert!(
        DecisionContractArtifact::from_bytes(&rules_bytes, &c.contract.identity().unwrap())
            .is_err()
    );
    assert!(AdmissibleContractArtifact::from_bytes(&a0_bytes, &c.identity().unwrap()).is_err());
}

#[test]
fn f4_invalid_rules_refuse_with_structured_detail() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, rich_rules());
    let json = admissible_contract_to_json(&c).unwrap();
    let ghost = "\"action_id\":\"A\",\"input\":0";
    assert!(json.contains(ghost), "{json}");
    let unknown = json.replacen(ghost, "\"action_id\":\"ghost\",\"input\":0", 1);
    let refusal = admissible_contract_from_json_refusal(&unknown).unwrap_err();
    assert_eq!(refusal.detail, "decision_admissibility.unknown_action");
    assert_eq!(refusal.offending.as_deref(), Some("ghost"));

    let wrong_input = json.replacen(ghost, "\"action_id\":\"A\",\"input\":7", 1);
    let refusal = admissible_contract_from_json_refusal(&wrong_input).unwrap_err();
    assert_eq!(refusal.detail, "decision_admissibility.unknown_input");
    assert_eq!(refusal.offending.as_deref(), Some("A[7]"));

    let malformed = admissible_contract_from_json_refusal("{").unwrap_err();
    assert_eq!(malformed.detail, "decision_contract.invalid_declaration");
    let unknown_field = json.replacen("\"rules\":{", "\"rules\":{\"bonus\":1,", 1);
    assert_eq!(
        admissible_contract_from_json_refusal(&unknown_field).unwrap_err().detail,
        "decision_contract.invalid_declaration"
    );
    let bad_label = json.replace("\"weak_overlap\"", "\"mostly_fine\"");
    assert_eq!(
        admissible_contract_from_json_refusal(&bad_label).unwrap_err().detail,
        "decision_contract.invalid_declaration"
    );

    let mut duplicate = c.clone();
    duplicate.rules.support_rules.push(duplicate.rules.support_rules[0].clone());
    assert!(AdmissibleContractArtifact::new(duplicate).is_err());
    let mut blank = c;
    blank.rules.declared_exclusions[0].reason = " ".into();
    assert!(AdmissibleContractArtifact::new(blank).is_err());
}

#[test]
fn f4_decode_is_bounded_before_any_allocation() {
    let huge = vec![0_u8; MAX_DECISION_ARTIFACT_BYTES + 1];
    assert!(matches!(
        AdmissibleContractArtifact::from_bytes(&huge, "x").unwrap_err(),
        IoError::TooLarge
    ));
    assert!(matches!(
        RobustResultArtifact::from_bytes(&huge, "x", &[]).unwrap_err(),
        IoError::TooLarge
    ));
    let declaration = " ".repeat(MAX_DECISION_ARTIFACT_BYTES + 1);
    assert_eq!(
        admissible_contract_from_json_refusal(&declaration).unwrap_err().detail,
        "decision_contract.invalid_declaration"
    );
    assert!(AdmissibleContractArtifact::from_bytes(b"not an artifact", "x").is_err());
}

// ---------------------------------------------------------------------------
// F6: the robust result artifact
// ---------------------------------------------------------------------------

#[test]
fn f6_robust_result_round_trips_and_replays_by_recomputation() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = two_atoms();
    let claims = supported_profile();
    let result = assess(&c, &atoms, &claims);
    assert_eq!(result.verdict, RobustVerdict::StructurallyRobust("A".into()));
    let artifact = RobustResultArtifact::new(&c, result.clone(), &claims, &atoms, vec![]).unwrap();
    assert!(artifact.native_verified());
    let bytes = artifact.to_bytes("robust").unwrap();

    let identity = c.identity().unwrap();
    let digests = atom_digests_of(&atoms);
    assert!(digests.iter().all(|(_, digest)| digest.is_some()));
    let loaded = RobustResultArtifact::from_bytes(&bytes, &identity, &digests).unwrap();
    assert_eq!(loaded.result(), &result);
    assert_eq!(loaded.profile(), &claims);
    assert!(loaded.native_verified() && loaded.receipts().is_empty());
    assert_eq!(loaded.atom_digests(), digests);

    // Replay recomputes the whole assessment from the consumer's own inputs.
    let receipt = loaded.replay(&c, &atoms, &claims).unwrap();
    assert!(receipt.recomputed && receipt.native_verified && receipt.external_atoms.is_empty());

    // The derivation names the contract and each structure's draws.
    let chain = loaded.provenance_chain().unwrap();
    chain
        .require_stages(
            RobustResultArtifact::CLAIM_LINK_ID,
            &[
                CompositionStage::DecisionContract,
                CompositionStage::DistributionArtifact,
                CompositionStage::Claim,
            ],
        )
        .unwrap();
    assert!(
        chain
            .require_stages(
                RobustResultArtifact::CLAIM_LINK_ID,
                &[CompositionStage::ExternalProvider]
            )
            .is_err()
    );
}

#[test]
fn f6_every_robust_verdict_state_round_trips() {
    type Case = (StructuralPolicy, AdmissibilityRules, Vec<StructuralAtom>, ClaimProfile);
    let invariant = StructuralPolicy::RequireInvariantBestAction;
    let supported_both = supported_profile();
    let weak = profile(vec![
        ("s1", AtomSupport::supported()),
        ("s2", AtomSupport::uniform(SupportStatus::Extrapolative)),
    ]);
    let outside = profile(vec![
        ("s1", AtomSupport::uniform(SupportStatus::Extrapolative)),
        ("s2", AtomSupport::uniform(SupportStatus::OutsideEmpiricalSupport)),
    ]);
    let disagree = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let weighted = [evaluated("s1", Some(0.5), 5.0, 3.0), evaluated("s2", Some(0.5), 1.0, 4.0)];
    let unresolved = [
        evaluated("s1", None, 5.0, 3.0),
        StructuralAtom {
            id: "s2".into(),
            probability: None,
            evidence: AtomEvidence::Unevaluated("budget".into()),
        },
    ];
    let both_removed = AdmissibilityRules {
        declared_exclusions: vec![
            DeclaredExclusion { action_id: "A".into(), reason: "legal".into() },
            DeclaredExclusion { action_id: "B".into(), reason: "ethics".into() },
        ],
        ..AdmissibilityRules::default()
    };
    let cases: Vec<(&str, Case)> = vec![
        (
            "structurally_robust",
            (invariant, require_supported(), two_atoms().to_vec(), supported_both.clone()),
        ),
        ("support_robust", (invariant, require_supported(), two_atoms().to_vec(), weak)),
        (
            "graph_dependent",
            (invariant, AdmissibilityRules::default(), disagree.to_vec(), profile(vec![])),
        ),
        ("unsupported", (invariant, require_supported(), two_atoms().to_vec(), outside)),
        (
            "insufficient",
            (invariant, AdmissibilityRules::default(), unresolved.to_vec(), profile(vec![])),
        ),
        ("none_admissible", (invariant, both_removed, two_atoms().to_vec(), profile(vec![]))),
        (
            "worst_case",
            (
                StructuralPolicy::Maximin,
                AdmissibilityRules::default(),
                disagree.to_vec(),
                profile(vec![]),
            ),
        ),
        (
            "bayes",
            (
                StructuralPolicy::BayesOverStructures,
                AdmissibilityRules::default(),
                weighted.to_vec(),
                profile(vec![]),
            ),
        ),
        (
            "report_only",
            (
                StructuralPolicy::ReportOnly,
                AdmissibilityRules::default(),
                two_atoms().to_vec(),
                profile(vec![]),
            ),
        ),
    ];
    for (name, (policy, rules, atoms, claims)) in cases {
        let c = admissible(policy, rules);
        let result = assess(&c, &atoms, &claims);
        let artifact =
            RobustResultArtifact::new(&c, result.clone(), &claims, &atoms, vec![]).unwrap();
        let bytes = artifact.to_bytes("robust").unwrap();
        let loaded = RobustResultArtifact::from_bytes(
            &bytes,
            &c.identity().unwrap(),
            &atom_digests_of(&atoms),
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(loaded.result().verdict, result.verdict, "{name}");
        assert_eq!(loaded.result().unsupported_atoms, result.unsupported_atoms, "{name}");
        assert_eq!(loaded.result().shortfalls.len(), result.shortfalls.len(), "{name}");
        loaded.replay(&c, &atoms, &claims).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
}

#[test]
fn f6_robust_bytes_are_invariant_to_action_order() {
    let rules = require_supported();
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, rules);
    let mut reordered = c.clone();
    reordered.contract.actions.reverse();
    assert_eq!(c.identity().unwrap(), reordered.identity().unwrap());
    let atoms = two_atoms();
    let claims = supported_profile();
    let write = |contract: &AdmissibleDecisionContract| {
        let result = assess(contract, &atoms, &claims);
        RobustResultArtifact::new(contract, result, &claims, &atoms, vec![])
            .unwrap()
            .to_bytes("robust")
            .unwrap()
    };
    assert_eq!(write(&c), write(&reordered));
    // The artifact written under one order replays under the other.
    let loaded = RobustResultArtifact::from_bytes(
        &write(&c),
        &reordered.identity().unwrap(),
        &atom_digests_of(&atoms),
    )
    .unwrap();
    loaded.replay(&reordered, &atoms, &claims).unwrap();
}

#[test]
fn f6_resealed_robust_mutations_are_refused() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = two_atoms();
    let claims = supported_profile();
    let identity = c.identity().unwrap();
    let digests = atom_digests_of(&atoms);
    let with_receipt = sealed(vec![external("s1", attested())]);
    let plain = sealed(vec![]);
    let receipt_bytes = with_receipt.to_bytes("robust").unwrap();
    let plain_bytes = plain.to_bytes("robust").unwrap();
    let load = |bytes: &[u8]| RobustResultArtifact::from_bytes(bytes, &identity, &digests);
    assert!(load(&receipt_bytes).is_ok() && load(&plain_bytes).is_ok());

    // Hiding the external origin by flipping the label.
    let relabelled = reseal(&receipt_bytes, ROBUST_KIND, |body| {
        body["native_verified"] = serde_json::json!(true);
    });
    assert!(message(&load(&relabelled).unwrap_err()).contains("native_label"));

    // Hiding it by dropping the receipt altogether: the lineage no longer matches.
    let stripped = reseal(&receipt_bytes, ROBUST_KIND, |body| {
        body["native_verified"] = serde_json::json!(true);
        body["external_receipts"] = serde_json::json!([]);
    });
    assert!(message(&load(&stripped).unwrap_err()).contains("lineage"));

    // Upgrading the trust label to native.
    for label in ["native_licensed", "native", "verified"] {
        let upgraded = reseal(&receipt_bytes, ROBUST_KIND, |body| {
            body["external_receipts"][0]["trust"]["label"] = serde_json::json!(label);
        });
        assert!(message(&load(&upgraded).unwrap_err()).contains("receipt_trust"), "{label}");
    }

    // Changing the attested value or the request fingerprint breaks the derivation.
    let value = reseal(&receipt_bytes, ROBUST_KIND, |body| {
        body["external_receipts"][0]["attested_value"] = serde_json::json!(9.5);
    });
    assert!(message(&load(&value).unwrap_err()).contains("lineage"));
    let request = reseal(&receipt_bytes, ROBUST_KIND, |body| {
        body["external_receipts"][0]["request_fingerprint"] = serde_json::json!("another-request");
    });
    assert!(message(&load(&request).unwrap_err()).contains("lineage"));

    // A deleted derivation is not accepted.
    let no_lineage = reseal(&plain_bytes, ROBUST_KIND, |body| {
        body["lineage"] = serde_json::json!([]);
    });
    assert!(message(&load(&no_lineage).unwrap_err()).contains("lineage_missing"));

    // A resealed verdict is internally consistent, so it loads, but replay by
    // recomputation does not reproduce it.
    let verdict = reseal(&plain_bytes, ROBUST_KIND, |body| {
        body["result"]["verdict"] = serde_json::json!({"structurally_robust": "B"});
    });
    let forged = load(&verdict).unwrap();
    assert_eq!(forged.result().verdict, RobustVerdict::StructurallyRobust("B".into()));
    let error = forged.replay(&c, &atoms, &claims).unwrap_err();
    assert!(message(&error).contains("replay_mismatch"), "{error:?}");

    // A resealed number in a per-action range fails the same way.
    let range = reseal(&plain_bytes, ROBUST_KIND, |body| {
        body["result"]["actions"][0]["range"] = serde_json::json!([50.0, 60.0]);
    });
    let error = load(&range).unwrap().replay(&c, &atoms, &claims).unwrap_err();
    assert!(message(&error).contains("replay_mismatch"), "{error:?}");
}

#[test]
fn f6_load_and_replay_refuse_other_contracts_structures_and_profiles() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = two_atoms();
    let claims = supported_profile();
    let artifact = sealed(vec![]);
    let bytes = artifact.to_bytes("robust").unwrap();
    let identity = c.identity().unwrap();
    let digests = atom_digests_of(&atoms);

    let other = admissible(StructuralPolicy::Maximin, require_supported());
    let error =
        RobustResultArtifact::from_bytes(&bytes, &other.identity().unwrap(), &digests).unwrap_err();
    assert!(message(&error).contains("contract_expected"));
    let shifted = [evaluated("s1", None, 5.5, 3.0), evaluated("s2", None, 6.0, 2.0)];
    let error = RobustResultArtifact::from_bytes(&bytes, &identity, &atom_digests_of(&shifted))
        .unwrap_err();
    assert!(message(&error).contains("atoms_expected"));
    let renamed = [evaluated("t1", None, 5.0, 3.0), evaluated("s2", None, 6.0, 2.0)];
    assert!(
        RobustResultArtifact::from_bytes(&bytes, &identity, &atom_digests_of(&renamed)).is_err()
    );

    let loaded = RobustResultArtifact::from_bytes(&bytes, &identity, &digests).unwrap();
    let error = loaded.replay(&other, &atoms, &claims).unwrap_err();
    assert!(message(&error).contains("replay_contract"));
    let error = loaded.replay(&c, &shifted, &claims).unwrap_err();
    assert!(message(&error).contains("replay_atoms"));
    let weaker = profile(vec![
        ("s1", AtomSupport::supported()),
        ("s2", AtomSupport::uniform(SupportStatus::WeakOverlap)),
    ]);
    let error = loaded.replay(&c, &atoms, &weaker).unwrap_err();
    assert!(message(&error).contains("replay_profile"));

    // Binding a result to another contract's identity is refused at construction.
    let result = assess(&other, &atoms, &claims);
    assert!(RobustResultArtifact::new(&c, result, &claims, &atoms, vec![]).is_err());
}

// ---------------------------------------------------------------------------
// F6: external-callback receipts
// ---------------------------------------------------------------------------

#[test]
fn f6_external_callback_receipt_is_retained_and_never_native() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let atoms = two_atoms();
    let claims = supported_profile();
    for trust in [attested(), ExternalTrustLimit::VerifiedExtension] {
        assert!(!trust.is_native());
        let receipt = external("s1", trust.clone());
        let artifact = sealed(vec![receipt.clone()]);
        assert!(!artifact.native_verified());
        let bytes = artifact.to_bytes("robust").unwrap();
        let loaded = RobustResultArtifact::from_bytes(
            &bytes,
            &c.identity().unwrap(),
            &atom_digests_of(&atoms),
        )
        .unwrap();
        // The attested value, exact request fingerprint and trust limit survive.
        assert_eq!(loaded.receipts(), &[receipt.clone()]);
        assert_eq!(loaded.receipts()[0].attested_value.to_bits(), 4.25_f64.to_bits());
        assert_eq!(loaded.receipts()[0].request_fingerprint, receipt.request_fingerprint);
        assert_eq!(loaded.receipts()[0].trust, trust);
        assert!(!loaded.native_verified());
        let json = loaded.to_json().unwrap();
        assert!(json.contains("\"native_verified\":false"), "{json}");
        assert!(json.contains(trust.label()), "{json}");

        // Replay recomputes over the draws but does not certify the callback.
        let replayed = loaded.replay(&c, &atoms, &claims).unwrap();
        assert!(replayed.recomputed);
        assert!(!replayed.native_verified);
        assert_eq!(replayed.external_atoms, vec!["s1".to_owned()]);

        // The external provider stands behind the reported decision.
        loaded
            .provenance_chain()
            .unwrap()
            .require_stages(
                RobustResultArtifact::CLAIM_LINK_ID,
                &[
                    CompositionStage::ExternalProvider,
                    CompositionStage::DistributionArtifact,
                    CompositionStage::DecisionContract,
                    CompositionStage::Claim,
                ],
            )
            .unwrap();
    }
    assert_eq!(attested().label(), "externally_attested");
    assert_eq!(ExternalTrustLimit::VerifiedExtension.label(), "verified_extension");
}

#[test]
fn f6_changing_any_receipt_field_changes_the_derivation() {
    let digest = |receipt: ExternalCallbackReceipt| {
        sealed(vec![receipt])
            .provenance_chain()
            .unwrap()
            .digest_of(RobustResultArtifact::CLAIM_LINK_ID)
            .unwrap()
            .to_owned()
    };
    let baseline = digest(external("s1", attested()));
    assert_eq!(baseline, digest(external("s1", attested())));
    let none = sealed(vec![])
        .provenance_chain()
        .unwrap()
        .digest_of(RobustResultArtifact::CLAIM_LINK_ID)
        .unwrap()
        .to_owned();
    assert_ne!(baseline, none);
    let variants: Vec<ReceiptEdit> = vec![
        Box::new(|r| r.request_fingerprint = "a different request".into()),
        Box::new(|r| r.attested_value = 4.5),
        Box::new(|r| r.provider_id = "other-model".into()),
        Box::new(|r| r.snapshot_id = "snap-10".into()),
        Box::new(|r| r.trust = ExternalTrustLimit::VerifiedExtension),
        Box::new(|r| {
            r.trust = ExternalTrustLimit::ExternallyAttested { attestor: "someone-else".into() };
        }),
        Box::new(|r| r.atom_id = "s2".into()),
    ];
    for edit in variants {
        let mut changed = external("s1", attested());
        edit(&mut changed);
        assert_ne!(digest(changed), baseline);
    }
}

#[test]
fn f6_invalid_external_receipts_refuse() {
    let c = admissible(StructuralPolicy::RequireInvariantBestAction, require_supported());
    let claims = supported_profile();
    let atoms = [
        evaluated("s1", None, 5.0, 3.0),
        StructuralAtom { id: "s2".into(), probability: None, evidence: AtomEvidence::Unidentified },
    ];
    let result = assess(&c, &atoms, &claims);
    let new = |receipts: Vec<ExternalCallbackReceipt>| {
        RobustResultArtifact::new(&c, result.clone(), &claims, &atoms, receipts)
    };
    assert!(new(vec![external("s1", attested())]).is_ok());
    // A receipt must name a structure that has draws.
    assert!(message(&new(vec![external("s2", attested())]).unwrap_err()).contains("receipt_atom"));
    assert!(
        message(&new(vec![external("ghost", attested())]).unwrap_err()).contains("receipt_atom")
    );
    // One receipt per structure.
    let twice = vec![external("s1", attested()), external("s1", attested())];
    assert!(message(&new(twice).unwrap_err()).contains("receipt_duplicate"));
    // Every receipt field is required and the value is finite.
    let mut blank = external("s1", attested());
    blank.request_fingerprint = " ".into();
    assert!(message(&new(vec![blank]).unwrap_err()).contains("receipt_invalid"));
    let mut nan = external("s1", attested());
    nan.attested_value = f64::NAN;
    assert!(message(&new(vec![nan]).unwrap_err()).contains("receipt_invalid"));
    let nobody = external("s1", ExternalTrustLimit::ExternallyAttested { attestor: " ".into() });
    assert!(message(&new(vec![nobody]).unwrap_err()).contains("receipt_invalid"));
}
