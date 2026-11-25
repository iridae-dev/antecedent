//! Fitted linear-Gaussian mechanism transport against `transport_linear_gaussian`.
//!
//! Oracle: treatment `x = 1 + u + e` fitted in the source and `x = 3 + u + e` refitted in the
//! target, outcome `y = 0.5 + 2 x + u + e` carried. Selection points into `x`. The contrast
//! `E[y | do(x = 1)] - E[y | do(x = 0)]` is 2, and `E[x] = 3`, `E[y] = 6.5`
//! (`crates/antecedent-model/tests/mechanism_transport.rs`). The facade twin is the chain with
//! `M` refitted to `1.5 + 0.5 X + 2 e`, whose same contrast is also 2
//! (`crates/antecedent/tests/transported_mechanism_facade.rs`).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(deprecated, reason = "the direct facade path is the twin of the compiled program")]
#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::planner::PlanView;
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismRefit, MechanismSource,
    Study, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{
    ContextCoordinate, ContrastArm, Domain, InvarianceDeclaration, LatentDependence, MeanContrast,
    MechanismContext, MechanismDescriptor, MechanismForm, MechanismOrigin, MechanismProvenance,
    ParentBinding, SelectionDiagramDeclaration, SelectionNodeDeclaration,
};
use antecedent_model::mechanism_transport::{MechanismTransportRequest, transport_linear_gaussian};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn bits(left: f64, right: f64) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left} vs {right}");
}

fn assert_unit(view: &PlanView) {
    let support = view.unit_support().expect("the route is bound to its capability record");
    assert_eq!(support.unit(), "transport.fitted_linear_gaussian_mechanism");
    assert!(support.supported());
    assert!(support.missing_evidence().is_empty());
}

fn binding(variable: &str) -> ParentBinding {
    ParentBinding {
        variable: variable.into(),
        units: "u".into(),
        transform: "identity".into(),
        domain: Domain::Real,
    }
}

fn gaussian(
    variable: &str,
    population: &str,
    intercept: f64,
    coefficients: &[(&str, f64)],
    shared_with: &str,
) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: variable.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma: 1.0,
            coefficients: coefficients
                .iter()
                .map(|(name, coefficient)| ((*name).to_owned(), *coefficient))
                .collect(),
        },
        parents: coefficients.iter().map(|(name, _)| binding(name)).collect(),
        conditioning_set: coefficients.iter().map(|(name, _)| (*name).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: MechanismContext { population: population.into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{variable}:{population}"),
            evidence_id: format!("data:{variable}:{population}"),
        },
        latents: vec![LatentDependence {
            latent: "u".into(),
            loading: Some(1.0),
            shared_with: vec![shared_with.into()],
        }],
        uncertainty: None,
    }
}

fn carried_request() -> MechanismTransportRequest {
    MechanismTransportRequest {
        target: MechanismContext { population: "target".into(), regime: "observational".into() },
        diagram: SelectionDiagramDeclaration {
            source_population: "source".into(),
            target_population: "target".into(),
            selection_nodes: vec![SelectionNodeDeclaration {
                name: "S".into(),
                points_into: vec!["x".into()],
            }],
        },
        source_mechanisms: vec![
            gaussian("x", "source", 1.0, &[], "y"),
            gaussian("y", "source", 0.5, &[("x", 2.0)], "x"),
        ],
        target_mechanisms: vec![gaussian("x", "target", 3.0, &[], "y")],
        edges: vec![("x".into(), "y".into())],
        invariances: vec![InvarianceDeclaration {
            variable: "y".into(),
            coordinate: ContextCoordinate::Population,
            source: "source".into(),
            target: "target".into(),
            assumption_id: "assumption:outcome-mechanism-invariant".into(),
        }],
        dependence: antecedent_model::compose::PosteriorDependence::Unknown,
    }
}

fn effect_of_x() -> MeanContrast {
    MeanContrast {
        outcome: "y".into(),
        arms: vec![
            ContrastArm { weight: 1.0, interventions: vec![("x".into(), 1.0)] },
            ContrastArm { weight: -1.0, interventions: vec![("x".into(), 0.0)] },
        ],
    }
}

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: match variable {
            "X" => QuantityRole::Treatment,
            "M" => QuantityRole::Mediator,
            "Y" => QuantityRole::Outcome,
            _ => QuantityRole::Covariate,
        },
        units: "u".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn linear(variable: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismInput {
    let names: Vec<&str> = parents.iter().map(|(parent, _)| *parent).collect();
    let slot = MechanismSlot::LinearGaussian {
        intercept,
        coeffs: Arc::from(parents.iter().map(|(_, coefficient)| *coefficient).collect::<Vec<_>>()),
        sigma,
    };
    MechanismInput {
        declaration: MechanismDeclaration {
            output: quantity(variable),
            parents: names.iter().copied().map(quantity).collect(),
            conditioning_variable_ids: names.iter().copied().map(str::to_owned).collect(),
            origin: MechanismOriginKind::Fitted,
            artifact_id: format!("fit:{variable}"),
            evidence_id: format!("data:{variable}"),
            latent_causes: Vec::new(),
        },
        source: MechanismSource::from_fitted_slot(&slot, &names).unwrap(),
        output_support: Domain::Real,
        parent_domains: names.iter().copied().map(|name| (name.to_owned(), Domain::Real)).collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

#[test]
fn the_compiled_transport_matches_transport_linear_gaussian() {
    let request = carried_request();
    let direct = transport_linear_gaussian(&request).unwrap();
    let compiled = Study::compile_fitted_mechanism_transport(&request, &ctx()).unwrap();
    assert_unit(&compiled.plan_view());
    let carried = compiled.execute(&ctx()).unwrap();
    bits(carried.model().law(&[]).unwrap().mean_of("x").unwrap(), 3.0);
    bits(carried.model().law(&[]).unwrap().mean_of("y").unwrap(), 6.5);
    bits(
        carried.mean_contrast(&effect_of_x()).unwrap(),
        direct.mean_contrast(&effect_of_x()).unwrap(),
    );
    bits(carried.mean_contrast(&effect_of_x()).unwrap(), 2.0);
    assert_eq!(carried.identity_hex(), direct.identity_hex());
}

#[test]
fn the_facade_compiled_path_matches_transport_to() {
    let mechanisms = vec![
        linear("X", 1.0, 2.0, &[]),
        linear("M", 0.5, 1.0, &[("X", 2.0)]),
        linear("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ];
    let mut edges = Vec::new();
    for mechanism in &mechanisms {
        for parent in &mechanism.declaration.parents {
            edges.push((
                parent.variable_id.clone(),
                mechanism.declaration.output.variable_id.clone(),
            ));
        }
    }
    let source = StudyBuilder::mechanism_composition(
        MechanismCompositionRequest {
            target_population_id: "pop".into(),
            target_regime_id: "observational".into(),
            mechanisms,
            edges,
            admissions: Vec::new(),
            dependence: DependenceInput::Unknown,
            admit_declared_mechanisms: false,
        },
        &ctx(),
    )
    .unwrap();
    let declaration = antecedent::analysis::transported_mechanism::MechanismTransportDeclaration {
        target_population_id: "clinic_b".into(),
        target_regime_id: "observational".into(),
        selection_nodes: vec![SelectionNodeDeclaration {
            name: "S".into(),
            points_into: vec!["M".into()],
        }],
        target_refits: vec![MechanismRefit {
            variable_id: "M".into(),
            source: MechanismSource::FittedLinear {
                intercept: 1.5,
                sigma: 2.0,
                coefficients: vec![("X".into(), 0.5)],
            },
            artifact_id: "fit:M:target".into(),
            evidence_id: "data:M:target".into(),
            uncertainty: None,
        }],
        invariances: Vec::new(),
        dependence: DependenceInput::Unknown,
    };
    let contrast = MeanContrast {
        outcome: "Y".into(),
        arms: vec![
            ContrastArm { weight: 1.0, interventions: vec![("X".into(), 1.0)] },
            ContrastArm { weight: -1.0, interventions: vec![("X".into(), 0.0)] },
        ],
    };
    let direct = source.transport_to(&declaration, &ctx()).unwrap();
    let compiled = source.transport_compiled(&declaration, &ctx()).unwrap();
    assert_eq!(compiled.identity_hex(), direct.identity_hex());
    bits(compiled.mean_contrast(&contrast).unwrap(), direct.mean_contrast(&contrast).unwrap());
    bits(compiled.mean_contrast(&contrast).unwrap(), 2.0);
}
