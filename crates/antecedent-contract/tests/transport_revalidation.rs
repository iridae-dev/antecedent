//! Revalidation of transported claims after a change of target, source evidence,
//! selection diagram or support, on hand-computable scenes.
//!
//! The base scene:
//!
//! ```text
//! source:trial ──evidence──► claim:effect ──data──► claim:decision
//! source:registry ─evidence─►     │  │ ▲
//!                diagram ─structural┘ │ └─ calibration inherited by claim:decision
//!                target ──support─────┘
//! ```
//!
//! `effect` holds a calibration record established at the base target, and `decision`
//! inherits it. `note` is a point-only claim on the registry alone.

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::{
    CalibrationState, NumericalCause, ObservedEffect, PriorAuthority, Revalidation,
    SupportStanding, TrustStanding,
};
use antecedent_contract::transport_revalidation::{
    SceneCalibration, SceneClaim, SceneClaimKind, SceneSource, SceneTarget, TransportReason,
    TransportScene, revalidate_transport, transport_coordinate,
};
use antecedent_core::SemanticDigest;

fn d(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

const TARGET: u8 = 10;
const DIAGRAM: u8 = 20;
const RECORD: u8 = 90;
const QUANTITY: u8 = 50;

fn base() -> TransportScene {
    let mut effect = SceneClaim::new("effect", SceneClaimKind::Transported, d(31), d(QUANTITY));
    effect.structure = Some(d(40));
    effect.sources = vec!["trial".into(), "registry".into()];
    effect.calibration =
        SceneCalibration::Held { record: d(RECORD), established_at_target: d(TARGET) };
    let mut decision = SceneClaim::new("decision", SceneClaimKind::Decision, d(32), d(QUANTITY));
    decision.rests_on = vec!["effect".into()];
    decision.calibration = SceneCalibration::Inherited {
        from: "effect".into(),
        record: d(RECORD),
        established_at_target: d(TARGET),
    };
    let mut note = SceneClaim::new("note", SceneClaimKind::Transported, d(33), d(51));
    note.structure = Some(d(41));
    note.sources = vec!["registry".into()];
    TransportScene {
        target: SceneTarget { identity: d(TARGET), support: SupportStanding::Supported },
        selection_diagram: d(DIAGRAM),
        sources: vec![
            SceneSource { key: "trial".into(), identity: d(1) },
            SceneSource { key: "registry".into(), identity: d(2) },
        ],
        claims: vec![effect, decision, note],
    }
}

fn none() -> BTreeMap<String, ObservedEffect> {
    BTreeMap::new()
}

#[test]
fn unchanged_scene_keeps_every_claim_and_its_earlier_authority() {
    let result = revalidate_transport(&base(), &base(), &none()).unwrap();
    for claim in result.claims() {
        assert_eq!(claim.class, Revalidation::StillValid, "{}", claim.key);
        assert!(claim.reasons.is_empty());
        assert_eq!(claim.authority, PriorAuthority::Retained);
        assert!(claim.refusal().is_none());
    }
    assert_eq!(result.claim("effect").unwrap().calibration, CalibrationState::Held);
    assert_eq!(result.claim("decision").unwrap().calibration, CalibrationState::Inherited);
    assert_eq!(result.claim("note").unwrap().calibration, CalibrationState::NotApplicable);
    assert!(result.removed().is_empty());
}

#[test]
fn changed_source_evidence_recomputes_exactly_the_claims_that_count_it() {
    let mut current = base();
    current.sources[0].identity = d(101); // the trial
    let result = revalidate_transport(&base(), &current, &none()).unwrap();
    let effect = result.claim("effect").unwrap();
    assert_eq!(effect.class, Revalidation::Recompute);
    assert_eq!(
        effect.reasons,
        vec![TransportReason::SourceEvidenceChanged { source: "trial".into() }]
    );
    assert_eq!(effect.authority, PriorAuthority::NotRetained);
    // the calibration record is still at its coordinate: not withdrawn, only stale
    assert!(!effect.calibration_withdrawn());
    let decision = result.claim("decision").unwrap();
    assert_eq!(decision.class, Revalidation::Recompute);
    assert_eq!(
        decision.reasons,
        vec![TransportReason::SourceEvidenceChanged { source: "trial".into() }]
    );
    assert_eq!(result.claim("note").unwrap().class, Revalidation::StillValid);
    assert_eq!(result.keys_of(Revalidation::Recompute), vec!["decision", "effect"]);
}

#[test]
fn changed_selection_diagram_recomputes_transported_claims_and_what_rests_on_them() {
    let mut current = base();
    current.selection_diagram = d(121);
    let result = revalidate_transport(&base(), &current, &none()).unwrap();
    for key in ["effect", "decision", "note"] {
        let claim = result.claim(key).unwrap();
        assert_eq!(claim.class, Revalidation::Recompute, "{key}");
        assert_eq!(claim.reasons, vec![TransportReason::SelectionDiagramChanged], "{key}");
        assert_eq!(claim.authority, PriorAuthority::NotRetained);
    }
    let refusal = result.claim("effect").unwrap().refusal().unwrap();
    assert_eq!(refusal.detail, "composition_verification.recompute_required");
}

#[test]
fn changed_target_recomputes_a_point_claim_and_withdraws_a_claim_calibrated_elsewhere() {
    let mut current = base();
    current.target.identity = d(111);
    let result = revalidate_transport(&base(), &current, &none()).unwrap();

    // point-only claim: the numbers are stale, no license is lost
    let note = result.claim("note").unwrap();
    assert_eq!(note.class, Revalidation::Recompute);
    assert!(note.reasons.contains(&TransportReason::TargetPopulationChanged));
    assert!(note.reasons.iter().all(|r| !r.is_withdrawal()));

    // the record was established at the old target: never inherited by the new one
    let effect = result.claim("effect").unwrap();
    assert_eq!(effect.class, Revalidation::Withdrawn);
    assert!(effect.reasons.contains(&TransportReason::TargetPopulationChanged));
    assert!(effect.reasons.contains(&TransportReason::CalibrationFromAnotherTarget));
    assert!(effect.calibration_withdrawn());
    assert_eq!(effect.calibration, CalibrationState::Lost);
    assert_eq!(effect.authority, PriorAuthority::NotRetained);

    // the decision's inherited record names the old target as well
    let decision = result.claim("decision").unwrap();
    assert_eq!(decision.class, Revalidation::Withdrawn);
    assert!(decision.reasons.contains(&TransportReason::CalibrationFromAnotherTarget));
    assert_eq!(decision.calibration, CalibrationState::Lost);
    assert_eq!(result.keys_of(Revalidation::Withdrawn), vec!["decision", "effect"]);
}

#[test]
fn recalibrating_at_the_new_target_restores_the_claim_after_recomputation() {
    let mut current = base();
    current.target.identity = d(111);
    for claim in &mut current.claims {
        match &mut claim.calibration {
            SceneCalibration::Held { established_at_target, record } => {
                *established_at_target = d(111);
                *record = d(91);
            }
            SceneCalibration::Inherited { established_at_target, record, .. } => {
                *established_at_target = d(111);
                *record = d(91);
            }
            _ => {}
        }
    }
    let result = revalidate_transport(&base(), &current, &none()).unwrap();
    let effect = result.claim("effect").unwrap();
    assert_eq!(effect.class, Revalidation::Recompute);
    assert_eq!(effect.calibration, CalibrationState::Held);
    assert!(!effect.calibration_withdrawn());
    let decision = result.claim("decision").unwrap();
    assert_eq!(decision.class, Revalidation::Recompute);
    assert_eq!(decision.calibration, CalibrationState::Inherited);
}

#[test]
fn inheriting_a_record_from_another_target_is_withdrawn_even_when_upstream_holds_it() {
    // the decision claims the effect's record, but states it was established at another
    // target than the effect's
    let mut current = base();
    if let SceneCalibration::Inherited { established_at_target, .. } =
        &mut current.claims[1].calibration
    {
        *established_at_target = d(77);
    }
    let result = revalidate_transport(&base(), &current, &none()).unwrap();
    assert_eq!(result.claim("effect").unwrap().class, Revalidation::StillValid);
    let decision = result.claim("decision").unwrap();
    assert_eq!(decision.class, Revalidation::Withdrawn);
    assert!(decision.reasons.contains(&TransportReason::CalibrationFromAnotherTarget));
    assert!(decision.calibration_withdrawn());
}

#[test]
fn unsupported_target_withdraws_every_claim_for_support_with_the_coordinates() {
    let mut current = base();
    current.target.support =
        SupportStanding::Unsupported { coordinates: vec!["age>=80".into()] };
    let result = revalidate_transport(&base(), &current, &none()).unwrap();
    for key in ["effect", "decision", "note"] {
        let claim = result.claim(key).unwrap();
        assert_eq!(claim.class, Revalidation::Withdrawn, "{key}");
        assert!(
            claim.reasons.contains(&TransportReason::TargetUnsupported {
                coordinates: "age>=80".into()
            }),
            "{key}"
        );
        assert!(claim.reasons.iter().any(TransportReason::is_withdrawal));
    }
    let refusal = result.claim("note").unwrap().refusal().unwrap();
    assert_eq!(refusal.detail, "composition_verification.support_withdrawn");
}

#[test]
fn claim_level_support_and_provider_trust_withdraw_only_what_relies_on_them() {
    let mut current = base();
    current.claims[2].support = SupportStanding::Unsupported { coordinates: vec!["z".into()] };
    current.claims[0].trust = TrustStanding::Withdrawn;
    let mut previous = base();
    previous.claims[0].trust = TrustStanding::Held;
    let result = revalidate_transport(&previous, &current, &none()).unwrap();
    assert_eq!(
        result.claim("note").unwrap().reasons,
        vec![TransportReason::ClaimUnsupported { claim: "note".into(), coordinates: "z".into() }]
    );
    let effect = result.claim("effect").unwrap();
    assert_eq!(effect.class, Revalidation::Withdrawn);
    assert!(effect.reasons.contains(&TransportReason::ProviderTrustLost { claim: "effect".into() }));
    // a claim withdrawn for provider trust is not still carrying its calibration
    assert_eq!(result.claim("decision").unwrap().class, Revalidation::Withdrawn);
}

#[test]
fn removed_source_retires_the_claims_that_counted_it() {
    let mut current = base();
    current.sources.retain(|s| s.key != "registry");
    current.claims[0].sources = vec!["trial".into()];
    current.claims.retain(|c| c.key != "note");
    let result = revalidate_transport(&base(), &current, &none()).unwrap();
    let effect = result.claim("effect").unwrap();
    assert_eq!(effect.class, Revalidation::Withdrawn);
    assert!(effect.reasons.contains(&TransportReason::InputRetired { input: "registry".into() }));
    assert_eq!(result.removed(), ["note".to_owned()]);
}

#[test]
fn observed_staleness_and_own_changes_are_named_per_claim() {
    let mut current = base();
    current.claims[2].identity = d(133);
    let mut effects = BTreeMap::new();
    effects.insert(
        "claim:effect".to_owned(),
        ObservedEffect::Recomputed { cause: "stage_status:fit".into() },
    );
    let result = revalidate_transport(&base(), &current, &effects).unwrap();
    assert_eq!(
        result.claim("note").unwrap().reasons,
        vec![TransportReason::ClaimChanged {
            claim: "note".into(),
            cause: NumericalCause::StructureReusedFitNotRetained,
        }]
    );
    let effect = result.claim("effect").unwrap();
    assert_eq!(effect.class, Revalidation::Recompute);
    assert_eq!(effect.reasons, vec![TransportReason::ObservedStale { node: "effect".into() }]);
}

#[test]
fn undeclared_references_are_refused_and_identity_ignores_declaration_order() {
    let mut bad = base();
    bad.claims[0].sources.push("missing".into());
    assert!(revalidate_transport(&base(), &bad, &none()).is_err());

    let mut inherit = base();
    inherit.claims[1].calibration = SceneCalibration::Inherited {
        from: "ghost".into(),
        record: d(RECORD),
        established_at_target: d(TARGET),
    };
    assert!(revalidate_transport(&base(), &inherit, &none()).is_err());

    let mut reordered = base();
    reordered.claims.reverse();
    reordered.sources.reverse();
    let a = revalidate_transport(&base(), &base(), &none()).unwrap();
    let b = revalidate_transport(&reordered, &reordered, &none()).unwrap();
    assert_eq!(a.identity(), b.identity());

    let mut changed = base();
    changed.sources[0].identity = d(102);
    let c = revalidate_transport(&base(), &changed, &none()).unwrap();
    assert_ne!(a.identity(), c.identity());
}

#[test]
fn coordinate_depends_on_target_and_quantity() {
    let a = transport_coordinate(d(1), d(2));
    assert_eq!(a, transport_coordinate(d(1), d(2)));
    assert_ne!(a, transport_coordinate(d(3), d(2)));
    assert_ne!(a, transport_coordinate(d(1), d(3)));
}
