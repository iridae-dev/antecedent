//! 2.3.0 A5: the transported static path-specific counterfactual through the `antecedent`
//! facade: evaluate, export and consume, the hand-derived oracle values, the typed refusals
//! (with the two-model witness) mapped onto `CausalError`, and the general closed route that
//! the narrow cell does not open.
//!
//! Model: `m = 0.5 z + (2 - 0.5 z) a + u_m`, `y = (3 + 0.75 z) a + 4 m + 1.5 z + u_y`, with
//! `a1 = 2`, `a0 = 0.5`. Target law `{0: 0.25, 1: 0.75}`: NDE `5.34375`, NIE `9.75`, total
//! `15.09375`; source law `{0: 0.4, 1: 0.4, 2: 0.2}`: NDE `5.4`. Shifting the target law to
//! `{0: 0.5, 1: 0.5}` moves the NDE by exactly `-0.28125`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::float_cmp, reason = "exact dyadic hand values")]

use antecedent::CausalError;
use antecedent::analysis::temporal_counterfactual::{
    TransportedCounterfactualPrerequisites, transported_path_specific,
};
use antecedent::analysis::transported_counterfactual::{
    AffineWire, AssignmentWire, EvidenceWire, LawWire, MechanismWire, ModelWire, PremisesWire,
    SelectionWire, SupportPointWire, TransportedCounterfactual,
    TransportedCounterfactualArtifactError, TransportedCounterfactualRequestWire,
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

fn law(points: &[(f64, f64)]) -> LawWire {
    LawWire {
        points: points
            .iter()
            .map(|(z, w)| SupportPointWire { values: vec![("z".into(), *z)], weight: *w })
            .collect(),
    }
}

fn request(plus: &[&str], minus: &[&str]) -> TransportedCounterfactualRequestWire {
    TransportedCounterfactualRequestWire {
        model: ModelWire {
            treatment: "a".into(),
            covariates: vec!["z".into()],
            mechanisms: vec![
                MechanismWire {
                    node: "m".into(),
                    intercept: aff(0.0, &[]),
                    parents: vec![
                        ("z".into(), aff(0.5, &[])),
                        ("a".into(), aff(2.0, &[("z", -0.5)])),
                    ],
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
        },
        source_law: law(&[(0.0, 0.4), (1.0, 0.4), (2.0, 0.2)]),
        target_law: law(&[(0.0, 0.25), (1.0, 0.75)]),
        selections: vec![SelectionWire { label: "S".into(), target: "z".into() }],
        assignment: AssignmentWire {
            outcome: "y".into(),
            treated_value: 2.0,
            control_value: 0.5,
            plus: plus.iter().map(|s| (*s).into()).collect(),
            minus: minus.iter().map(|s| (*s).into()).collect(),
        },
        premises: PremisesWire {
            additive_noise: true,
            noise_laws_shared: true,
            cross_world_independence: true,
        },
        evidence: ["source", "target"]
            .iter()
            .map(|role| EvidenceWire {
                role: (*role).into(),
                regime: "observational".into(),
                label: "evidence".into(),
            })
            .collect(),
    }
}

fn nde() -> TransportedCounterfactualRequestWire {
    request(&["y"], &[])
}

#[test]
fn x8t_facade_hand_values_nde_nie_total() {
    let direct = TransportedCounterfactual::evaluate(&nde()).unwrap();
    close(direct.target_contrast(), 5.34375);
    close(direct.source_contrast(), 5.4);
    let indirect = TransportedCounterfactual::evaluate(&request(&["m", "y"], &["y"])).unwrap();
    close(indirect.target_contrast(), 9.75);
    let overall = TransportedCounterfactual::evaluate(&request(&["m", "y"], &[])).unwrap();
    close(overall.target_contrast(), 15.09375);
    close(overall.target_contrast(), direct.target_contrast() + indirect.target_contrast());
    assert_eq!(direct.result().derivation.claim, "point_only");
    assert_eq!(direct.report().inference_claim, "point_only");
    assert_eq!(direct.identity(), direct.artifact().identity());
}

#[test]
fn x8t_facade_target_law_shift_moves_the_answer_by_the_hand_amount() {
    let base = TransportedCounterfactual::evaluate(&nde()).unwrap().target_contrast();
    let mut shifted = nde();
    shifted.target_law = law(&[(0.0, 0.5), (1.0, 0.5)]);
    let moved = TransportedCounterfactual::evaluate(&shifted).unwrap().target_contrast();
    close(moved - base, -0.28125);
}

#[test]
fn x8t_facade_export_and_consume_replay_the_answer_and_bind_the_identity() {
    let sealed = TransportedCounterfactual::evaluate(&nde()).unwrap();
    let bytes = sealed.export("x8t-facade").unwrap();
    let consumed = TransportedCounterfactual::consume(&bytes, Some(sealed.identity())).unwrap();
    assert_eq!(consumed.result(), sealed.result());
    assert_eq!(consumed.target_contrast(), sealed.target_contrast());
    // A resealed change of the target law is refused by the retained identity.
    let mut forged = nde();
    forged.target_law = law(&[(0.0, 0.5), (1.0, 0.5)]);
    let forged_bytes =
        TransportedCounterfactual::evaluate(&forged).unwrap().export("forged").unwrap();
    assert!(TransportedCounterfactual::consume(&forged_bytes, None).is_ok());
    let error =
        TransportedCounterfactual::consume(&forged_bytes, Some(sealed.identity())).unwrap_err();
    assert_eq!(
        error,
        TransportedCounterfactualArtifactError::IdentityMismatch { field: "target_law" }
    );
}

#[test]
fn x8t_facade_selection_on_the_outcome_refuses_with_a_witness_and_maps_to_a_causal_error() {
    let mut request = nde();
    request.selections = vec![SelectionWire { label: "S".into(), target: "y".into() }];
    let error = TransportedCounterfactual::evaluate(&request).unwrap_err();
    let refusal = error.refusal().expect("a structured refusal");
    assert_eq!(refusal.code, "transport_proven_non_transportable");
    assert_eq!(refusal.detail, "transported_counterfactual.selection_on_mechanism");
    let witness = refusal.witness.expect("two-model witness");
    close(witness.target_contrast_b - witness.target_contrast_a, 1.5);
    let causal = CausalError::from(error);
    let text = causal.to_string();
    assert!(text.contains("transport_proven_non_transportable"), "{text}");
    assert!(text.contains("transported_counterfactual.selection_on_mechanism"), "{text}");
}

#[test]
fn x8t_facade_undeclared_premises_refuse_and_default_to_undeclared() {
    let mut request = nde();
    request.premises = PremisesWire::default();
    let refusal =
        TransportedCounterfactual::evaluate(&request).unwrap_err().refusal().expect("refusal");
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "transported_counterfactual.nonadditive_mechanism");
    assert_eq!(refusal.offending.as_deref(), Some("additive_noise"));
}

#[test]
fn x8t_facade_the_general_transported_route_stays_closed() {
    // The narrow cell is a different theorem; the general closed gate still refuses, whatever
    // is supplied.
    let refusal = transported_path_specific(
        &TransportedCounterfactualPrerequisites::default(),
        &[],
        &std::collections::BTreeMap::new(),
    )
    .unwrap_err();
    assert_eq!(refusal.refusal.detail, "transported_counterfactual.route_frozen");
    assert_eq!(refusal.refusal.code, "cell_not_licensed");
}
