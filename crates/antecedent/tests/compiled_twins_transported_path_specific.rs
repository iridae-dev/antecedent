//! The direct kernel `evaluate_transported_path_specific` and the compiled
//! `TransportedCounterfactual::compile` give the same transported contrast.
//!
//! The compiled entry seals the request (the artifact's canonical order: mechanisms by node,
//! parents by name, slopes by covariate, support points by value) and calls the kernel on it.
//! The direct call below is given the same model, laws, diagram, assignment, premises and
//! evidence in that canonical order, so every contrast is compared on `to_bits()`. The model is
//! the facade fixture: `m = 0.5 z + (2 - 0.5 z) a + u_m`, `y = (3 + 0.75 z) a + 4 m + 1.5 z + u_y`,
//! `a1 = 2`, `a0 = 0.5`, source law `{0: 0.4, 1: 0.4, 2: 0.2}`, target law `{0: 0.25, 1: 0.75}`;
//! the hand value of the target natural direct effect is `5.34375`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    reason = "the test calls the direct kernel the compiled program is compared with"
)]
#![allow(clippy::float_cmp, reason = "bit-identical equality is the property under test")]

use std::collections::{BTreeMap, BTreeSet};

use antecedent::analysis::transported_counterfactual::{
    AffineWire, AssignmentWire, EvidenceWire, LawWire, ModelWire, PremisesWire, SelectionWire,
    SupportPointWire, TransportedCounterfactual, TransportedCounterfactualRequestWire,
};
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::transported_gate::{PopulationRole, RegimeFactorKey};
use antecedent_counterfactual::transported_path_specific::{
    AdditiveLinearScm, Affine, CovariateLaw, DeclaredAssumptions, EdgeAssignment, LinearMechanism,
    PathSpecificQuery, PathSpecificSelectionDiagram, SelectionNode, TransportedPathSpecificInput,
    evaluate_transported_path_specific,
};
use antecedent_io::transported_counterfactual_artifact::TransportedCounterfactualMechanismWire;

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

/// The natural direct effect request of the facade fixture.
fn request() -> TransportedCounterfactualRequestWire {
    TransportedCounterfactualRequestWire {
        model: ModelWire {
            treatment: "a".into(),
            covariates: vec!["z".into()],
            mechanisms: vec![
                TransportedCounterfactualMechanismWire {
                    node: "m".into(),
                    intercept: aff(0.0, &[]),
                    parents: vec![
                        ("z".into(), aff(0.5, &[])),
                        ("a".into(), aff(2.0, &[("z", -0.5)])),
                    ],
                },
                TransportedCounterfactualMechanismWire {
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
            plus: vec!["y".into()],
            minus: Vec::new(),
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

fn affine(constant: f64, slopes: &[(&str, f64)]) -> Affine {
    Affine {
        constant,
        covariate_slopes: slopes.iter().map(|(n, s)| ((*n).to_owned(), *s)).collect(),
    }
}

fn covariate_law(points: &[(f64, f64)]) -> CovariateLaw {
    CovariateLaw {
        points: points.iter().map(|(z, w)| (BTreeMap::from([("z".to_owned(), *z)]), *w)).collect(),
    }
}

#[test]
fn evaluate_transported_path_specific_equals_the_compiled_program() {
    // The same declaration, in the artifact's canonical order.
    let model = AdditiveLinearScm {
        treatment: "a".into(),
        covariates: vec!["z".into()],
        mechanisms: vec![
            (
                "m".into(),
                LinearMechanism {
                    intercept: affine(0.0, &[]),
                    parents: vec![
                        ("a".into(), affine(2.0, &[("z", -0.5)])),
                        ("z".into(), affine(0.5, &[])),
                    ],
                },
            ),
            (
                "y".into(),
                LinearMechanism {
                    intercept: affine(0.0, &[]),
                    parents: vec![
                        ("a".into(), affine(3.0, &[("z", 0.75)])),
                        ("m".into(), affine(4.0, &[])),
                        ("z".into(), affine(1.5, &[])),
                    ],
                },
            ),
        ],
    };
    let diagram = PathSpecificSelectionDiagram {
        selections: vec![SelectionNode { label: "S".into(), target: "z".into() }],
    };
    let source = covariate_law(&[(0.0, 0.4), (1.0, 0.4), (2.0, 0.2)]);
    let target = covariate_law(&[(0.0, 0.25), (1.0, 0.75)]);
    let query = PathSpecificQuery {
        outcome: "y".into(),
        treated_value: 2.0,
        control_value: 0.5,
        plus: EdgeAssignment { treated: BTreeSet::from(["y".to_owned()]) },
        minus: EdgeAssignment { treated: BTreeSet::new() },
    };
    let factors: BTreeMap<RegimeFactorKey, String> =
        [PopulationRole::Source, PopulationRole::Target]
            .into_iter()
            .map(|role| {
                (RegimeFactorKey { role, regime: "observational".into() }, "evidence".into())
            })
            .collect();
    let direct = evaluate_transported_path_specific(&TransportedPathSpecificInput {
        model: &model,
        diagram: &diagram,
        source_law: &source,
        target_law: &target,
        query: &query,
        assumptions: DeclaredAssumptions {
            additive_noise: true,
            noise_laws_shared: true,
            cross_world_independence: true,
        },
        supplied_factors: &factors,
    })
    .expect("the fixture is in the supported class");

    let compiled =
        TransportedCounterfactual::compile(&request(), &ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(compiled.lowered().route(), "TransportedPathSpecific");
    let result = compiled.counterfactual().result();
    assert_eq!(result.target_contrast.to_bits(), direct.target_contrast.to_bits());
    assert_eq!(result.source_contrast.to_bits(), direct.source_contrast.to_bits());
    assert_eq!(result.unit_contrasts.len(), direct.unit_contrasts.len());
    for (route, kernel) in result.unit_contrasts.iter().zip(&direct.unit_contrasts) {
        let point: BTreeMap<String, f64> = route.point.iter().cloned().collect();
        assert_eq!(point, kernel.point);
        assert_eq!(route.target_weight.to_bits(), kernel.target_weight.to_bits());
        assert_eq!(route.contrast.to_bits(), kernel.contrast.to_bits());
    }
    assert!((direct.target_contrast - 5.343_75).abs() < 1e-12, "the hand value");

    // The direct facade entry seals the same answer.
    let facade = TransportedCounterfactual::evaluate(&request()).unwrap();
    assert_eq!(facade.target_contrast().to_bits(), compiled.target_contrast().to_bits());
    assert_eq!(facade.identity(), compiled.counterfactual().identity());
}
