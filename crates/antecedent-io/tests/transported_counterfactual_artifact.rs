//! 2.3.0 A5 transported static path-specific counterfactual artifact: hand-derived oracle values
//! through the container, a recomputing consumer, resealed-mutation refusals against a retained
//! identity (premise, law, coefficient, selection, assignment), false-labelled premises,
//! refusals that are never stored, unknown versions and bounds.
//!
//! Structural model (covariate `z`, treatment `a`, mediator `m`, outcome `y`), `a1 = 2`,
//! `a0 = 0.5`:
//!   `m = 0.5 z + (2 - 0.5 z) a + u_m`
//!   `y = (3 + 0.75 z) a + 4 m + 1.5 z + u_y`
//! so `NDE(z) = 4.5 + 1.125 z`. Target law `{0: 0.25, 1: 0.75}` gives `5.34375`, source law
//! `{0: 0.4, 1: 0.4, 2: 0.2}` gives `5.4`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_io::transported_counterfactual_artifact::{
    AffineWire, AssignmentWire, EvidenceWire, LawWire, MAX_COVARIATES, MechanismWire, ModelWire,
    PremisesWire, RefusalWire, SelectionWire, SupportPointWire, TransportedCounterfactualArtifact,
    TransportedCounterfactualArtifactError, TransportedCounterfactualMeta,
    TransportedCounterfactualRequestWire, decode_parts, encode_parts,
};

const TOL: f64 = 1e-12;

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < TOL, "{a} != {b}");
}

fn aff(constant: f64, slopes: &[(&str, f64)]) -> AffineWire {
    AffineWire {
        constant,
        covariate_slopes: slopes.iter().map(|(n, s)| ((*n).to_owned(), *s)).collect(),
    }
}

fn model() -> ModelWire {
    ModelWire {
        treatment: "a".into(),
        covariates: vec!["z".into()],
        mechanisms: vec![
            MechanismWire {
                node: "m".into(),
                intercept: aff(0.0, &[]),
                parents: vec![("z".into(), aff(0.5, &[])), ("a".into(), aff(2.0, &[("z", -0.5)]))],
            },
            MechanismWire {
                node: "y".into(),
                intercept: aff(0.0, &[]),
                parents: vec![
                    ("a".into(), aff(3.0, &[("z", 0.75)])),
                    ("m".into(), aff(4.0, &[])),
                    ("z".into(), aff(1.5, &[])),
                ],
            },
        ],
    }
}

fn law(points: &[(f64, f64)]) -> LawWire {
    LawWire {
        points: points
            .iter()
            .map(|(z, w)| SupportPointWire { values: vec![("z".into(), *z)], weight: *w })
            .collect(),
    }
}

fn evidence(label: &str) -> Vec<EvidenceWire> {
    ["source", "target"]
        .iter()
        .map(|role| EvidenceWire {
            role: (*role).into(),
            regime: "observational".into(),
            label: label.into(),
        })
        .collect()
}

fn declared() -> PremisesWire {
    PremisesWire { additive_noise: true, noise_laws_shared: true, cross_world_independence: true }
}

fn nde() -> AssignmentWire {
    AssignmentWire {
        outcome: "y".into(),
        treated_value: 2.0,
        control_value: 0.5,
        plus: vec!["y".into()],
        minus: vec![],
    }
}

fn nie() -> AssignmentWire {
    AssignmentWire { plus: vec!["m".into(), "y".into()], minus: vec!["y".into()], ..nde() }
}

fn selection(target: &str) -> Vec<SelectionWire> {
    vec![SelectionWire { label: "S".into(), target: target.into() }]
}

fn request() -> TransportedCounterfactualRequestWire {
    TransportedCounterfactualRequestWire {
        model: model(),
        source_law: law(&[(0.0, 0.4), (1.0, 0.4), (2.0, 0.2)]),
        target_law: law(&[(0.0, 0.25), (1.0, 0.75)]),
        selections: selection("z"),
        assignment: nde(),
        premises: declared(),
        evidence: evidence("evidence"),
    }
}

fn seal(request: &TransportedCounterfactualRequestWire) -> TransportedCounterfactualArtifact {
    TransportedCounterfactualArtifact::seal(request).expect("seals")
}

fn refusal_of(request: &TransportedCounterfactualRequestWire) -> RefusalWire {
    TransportedCounterfactualArtifact::seal(request)
        .expect_err("must refuse")
        .refusal()
        .expect("a structured refusal")
}

#[test]
fn x8t_artifact_hand_values_survive_the_container_round_trip() {
    let sealed = seal(&request());
    close(sealed.result().target_contrast, 5.34375);
    close(sealed.result().source_contrast, 5.4);
    let by_z: Vec<f64> = sealed.result().unit_contrasts.iter().map(|u| u.contrast).collect();
    close(by_z[0], 4.5);
    close(by_z[1], 5.625);
    assert_eq!(sealed.result().derivation.claim, "point_only");
    assert!(sealed.result().derivation.declared.iter().any(|d| d == "additive_noise"));
    assert!(
        sealed
            .result()
            .derivation
            .checked
            .iter()
            .any(|c| c == "no_selection_on_mediator_or_outcome")
    );
    let bytes = sealed.to_bytes("x8t-artifact").unwrap();
    let consumed = TransportedCounterfactualArtifact::from_bytes(&bytes, Some(sealed.identity()))
        .expect("replays");
    assert_eq!(consumed.result(), sealed.result());
    assert_eq!(consumed.identity(), sealed.identity());
    assert_eq!(consumed.request(), sealed.request());
    assert_eq!(consumed.report(), sealed.report());
    // A second hand value: the NIE through the same container.
    let nie_request = TransportedCounterfactualRequestWire { assignment: nie(), ..request() };
    let nie_sealed = seal(&nie_request);
    close(nie_sealed.result().target_contrast, 9.75);
}

#[test]
fn x8t_artifact_identity_and_answer_do_not_depend_on_supply_order() {
    let base = seal(&request());
    let mut shuffled = request();
    shuffled.model.mechanisms.reverse();
    for mechanism in &mut shuffled.model.mechanisms {
        mechanism.parents.reverse();
    }
    shuffled.source_law.points.reverse();
    shuffled.target_law.points.reverse();
    shuffled.evidence.reverse();
    shuffled.assignment.plus.reverse();
    let other = seal(&shuffled);
    assert_eq!(other.identity(), base.identity());
    assert_eq!(other.result(), base.result());
    assert_eq!(other.request(), base.request());
}

#[test]
fn x8t_artifact_resealed_changes_are_refused_against_the_retained_identity() {
    let retained = seal(&request()).identity().clone();
    let mut by_field: Vec<(&str, TransportedCounterfactualRequestWire)> = Vec::new();
    // A coefficient: the A coefficient of M.
    let mut changed = request();
    changed.model.mechanisms[0].parents[1].1.constant = 3.0;
    by_field.push(("model", changed));
    // The source law, the target law.
    let mut changed = request();
    changed.source_law = law(&[(0.0, 0.5), (1.0, 0.25), (2.0, 0.25)]);
    by_field.push(("source_law", changed));
    let mut changed = request();
    changed.target_law = law(&[(0.0, 0.5), (1.0, 0.5)]);
    by_field.push(("target_law", changed));
    // The selection diagram: still covariate-only.
    let mut changed = request();
    changed.selections = selection("a");
    by_field.push(("selection", changed));
    // The edge assignment, and the treated value.
    let mut changed = request();
    changed.assignment = nie();
    by_field.push(("assignment", changed));
    let mut changed = request();
    changed.assignment.treated_value = 3.0;
    by_field.push(("assignment", changed));
    // The evidence labels.
    let mut changed = request();
    changed.evidence = evidence("another-fit");
    by_field.push(("evidence", changed));
    for (field, forged_request) in by_field {
        // The forger recomputes everything, so the artifact is self-consistent ...
        let bytes = seal(&forged_request).to_bytes("forged").unwrap();
        assert!(TransportedCounterfactualArtifact::from_bytes(&bytes, None).is_ok(), "{field}");
        // ... and is refused by the identity the consumer retained.
        let error =
            TransportedCounterfactualArtifact::from_bytes(&bytes, Some(&retained)).unwrap_err();
        assert_eq!(error, TransportedCounterfactualArtifactError::IdentityMismatch { field });
        let refusal = error.refusal().unwrap();
        assert_eq!(refusal.detail, "transported_counterfactual.artifact_changed");
        assert_eq!(refusal.code, "route_not_supported");
        assert_eq!(refusal.offending.as_deref(), Some(field));
    }
}

#[test]
fn x8t_artifact_stale_identity_after_an_unresealed_edit_is_refused() {
    let sealed = seal(&request());
    let mut meta = decode_parts(&sealed.to_bytes("x").unwrap()).unwrap();
    // Edit a coefficient and a law in place without updating the stored identity or result.
    meta.request.model.mechanisms[1].parents[0].1.constant = 4.0;
    let error = TransportedCounterfactualArtifact::from_bytes(
        &encode_parts(&meta, "edited").unwrap(),
        None,
    )
    .unwrap_err();
    assert_eq!(error, TransportedCounterfactualArtifactError::IdentityMismatch { field: "model" });
    let mut meta = decode_parts(&sealed.to_bytes("x").unwrap()).unwrap();
    meta.request.target_law = law(&[(0.0, 0.5), (1.0, 0.5)]);
    let error = TransportedCounterfactualArtifact::from_bytes(
        &encode_parts(&meta, "edited").unwrap(),
        None,
    )
    .unwrap_err();
    assert_eq!(
        error,
        TransportedCounterfactualArtifactError::IdentityMismatch { field: "target_law" }
    );
}

#[test]
fn x8t_artifact_false_labelled_premises_are_refused_even_when_resealed() {
    let sealed = seal(&request());
    for (flip, detail) in [
        (0, "transported_counterfactual.nonadditive_mechanism"),
        (1, "transported_counterfactual.noise_law_not_shared"),
        (2, "transported_counterfactual.cross_world_independence_missing"),
    ] {
        let mut meta = decode_parts(&sealed.to_bytes("x").unwrap()).unwrap();
        // Relabel one premise false (earlier premises stay true so the named one is first).
        meta.request.premises = PremisesWire {
            additive_noise: flip != 0,
            noise_laws_shared: flip != 1,
            cross_world_independence: flip != 2,
        };
        let bytes = encode_parts(&meta, "false-labelled").unwrap();
        for expected in [None, Some(sealed.identity())] {
            let refusal = TransportedCounterfactualArtifact::from_bytes(&bytes, expected)
                .unwrap_err()
                .refusal()
                .expect("the core refusal");
            assert_eq!(refusal.detail, detail);
            assert_eq!(refusal.code, "cell_not_licensed");
        }
    }
}

#[test]
fn x8t_artifact_a_selection_resealed_onto_a_mechanism_is_refused_with_its_witness() {
    let sealed = seal(&request());
    let mut meta = decode_parts(&sealed.to_bytes("x").unwrap()).unwrap();
    meta.request.selections = selection("y");
    let refusal = TransportedCounterfactualArtifact::from_bytes(
        &encode_parts(&meta, "on-outcome").unwrap(),
        None,
    )
    .unwrap_err()
    .refusal()
    .unwrap();
    assert_eq!(refusal.code, "transport_proven_non_transportable");
    assert!(refusal.witness.is_some());
}

#[test]
fn x8t_artifact_a_tampered_stored_answer_does_not_replay() {
    let sealed = seal(&request());
    let tampered = |edit: &dyn Fn(&mut TransportedCounterfactualMeta)| {
        let mut meta = decode_parts(&sealed.to_bytes("x").unwrap()).unwrap();
        edit(&mut meta);
        TransportedCounterfactualArtifact::from_bytes(&encode_parts(&meta, "t").unwrap(), None)
            .unwrap_err()
    };
    assert_eq!(
        tampered(&|m| m.result.target_contrast += 0.5),
        TransportedCounterfactualArtifactError::ResultMismatch("target contrast")
    );
    assert_eq!(
        tampered(&|m| m.result.source_contrast += 0.5),
        TransportedCounterfactualArtifactError::ResultMismatch("source contrast")
    );
    assert_eq!(
        tampered(&|m| m.result.unit_contrasts[1].contrast += 1.0),
        TransportedCounterfactualArtifactError::ResultMismatch("per-unit contrasts")
    );
    assert_eq!(
        tampered(&|m| m.result.derivation.declared.retain(|d| d != "additive_noise")),
        TransportedCounterfactualArtifactError::ResultMismatch("derivation")
    );
}

#[test]
fn x8t_artifact_a_changed_stored_identity_field_is_named() {
    let sealed = seal(&request());
    let mut meta = decode_parts(&sealed.to_bytes("x").unwrap()).unwrap();
    meta.identity.premises_digest = "0".repeat(64);
    let error =
        TransportedCounterfactualArtifact::from_bytes(&encode_parts(&meta, "t").unwrap(), None)
            .unwrap_err();
    assert_eq!(
        error,
        TransportedCounterfactualArtifactError::IdentityMismatch { field: "premises" }
    );
}

#[test]
fn x8t_artifact_refusals_are_never_stored_and_keep_the_two_model_witness() {
    let mut on_outcome = request();
    on_outcome.selections = selection("y");
    let refusal = refusal_of(&on_outcome);
    assert_eq!(refusal.code, "transport_proven_non_transportable");
    assert_eq!(refusal.detail, "transported_counterfactual.selection_on_mechanism");
    let witness = refusal.witness.expect("the NDE is sensitive to the outcome mechanism");
    assert_eq!(witness.selected_node, "y");
    assert_eq!(witness.perturbed_parent, "a");
    // The witness carries the canonical (sorted) source model.
    assert_eq!(witness.source_model, seal(&request()).request().model);
    assert_eq!(witness.target_model_a, witness.source_model);
    assert_ne!(witness.target_model_b, witness.target_model_a);
    close(witness.source_contrast, 5.4);
    close(witness.target_contrast_a, 5.34375);
    close(witness.target_contrast_b, 5.34375 + 1.5);
    // The mediator is not witnessed for the NDE: refused, no impossibility claimed.
    let mut on_mediator = request();
    on_mediator.selections = selection("m");
    let refusal = refusal_of(&on_mediator);
    assert_eq!(refusal.code, "cell_not_licensed");
    assert!(refusal.witness.is_none());
    // Undeclared premises refuse with the core's own detail.
    let mut undeclared = request();
    undeclared.premises = PremisesWire::default();
    assert_eq!(refusal_of(&undeclared).detail, "transported_counterfactual.nonadditive_mechanism");
    // Missing evidence names the missing factor; overlap names the point.
    let mut withheld = request();
    withheld.evidence.retain(|e| e.role == "source");
    let refusal = refusal_of(&withheld);
    assert_eq!(refusal.detail, "transported_counterfactual.factor_missing");
    assert_eq!(refusal.missing_factors, ["target:observational"]);
    let mut outside = request();
    outside.target_law = law(&[(0.0, 0.5), (3.0, 0.5)]);
    let refusal = refusal_of(&outside);
    assert_eq!(refusal.detail, "transported_counterfactual.overlap_failure");
    assert_eq!(refusal.offending.as_deref(), Some("z=3"));
    // A repeated evidence factor or an unknown role is an invalid factor.
    let mut repeated = request();
    repeated.evidence.push(repeated.evidence[0].clone());
    assert_eq!(refusal_of(&repeated).detail, "transported_counterfactual.invalid_factor");
}

#[test]
fn x8t_artifact_unknown_versions_features_and_corruption_are_refused() {
    let sealed = seal(&request());
    let bytes = sealed.to_bytes("x").unwrap();
    let mut meta = decode_parts(&bytes).unwrap();
    meta.version = 9;
    assert_eq!(
        TransportedCounterfactualArtifact::from_bytes(&encode_parts(&meta, "v").unwrap(), None)
            .unwrap_err(),
        TransportedCounterfactualArtifactError::UnsupportedVersion { version: 9 }
    );
    // decode_parts refuses another version before interpreting the metadata.
    assert!(matches!(
        decode_parts(&encode_parts(&meta, "v").unwrap()),
        Err(TransportedCounterfactualArtifactError::UnsupportedVersion { version: 9 })
    ));
    let mut meta = decode_parts(&bytes).unwrap();
    meta.feature = "other_v1".into();
    assert_eq!(
        TransportedCounterfactualArtifact::from_bytes(&encode_parts(&meta, "f").unwrap(), None)
            .unwrap_err(),
        TransportedCounterfactualArtifactError::UnsupportedSemantics("feature marker")
    );
    let mut meta = decode_parts(&bytes).unwrap();
    meta.inference_claim = "interval".into();
    assert_eq!(
        TransportedCounterfactualArtifact::from_bytes(&encode_parts(&meta, "c").unwrap(), None)
            .unwrap_err(),
        TransportedCounterfactualArtifactError::UnsupportedSemantics("inference claim")
    );
    assert!(
        TransportedCounterfactualArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err()
    );
    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xFF;
    assert!(TransportedCounterfactualArtifact::from_bytes(&flipped, None).is_err());
    assert!(matches!(
        encode_parts(&decode_parts(&bytes).unwrap(), "  "),
        Err(TransportedCounterfactualArtifactError::Encode(_))
    ));
}

#[test]
fn x8t_artifact_bounds_are_enforced_on_the_request() {
    let mut wide = request();
    wide.model.covariates = (0..=MAX_COVARIATES).map(|i| format!("c{i}")).collect();
    assert_eq!(
        TransportedCounterfactualArtifact::seal(&wide).unwrap_err(),
        TransportedCounterfactualArtifactError::LimitsExceeded("covariates")
    );
}
