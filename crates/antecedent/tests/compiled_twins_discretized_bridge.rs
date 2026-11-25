//! The discretized linear-Gaussian bridge against `DiscretizedLinearGaussian::new`,
//! `from_spec` and `mean_contrast`.
//!
//! Oracle: the chain `X = 1 + 2 e`, `M = 0.5 + 2 X + e`, `Y = -1 + 0.5 X + 3 M + 0.5 e`,
//! so `E[Y | do(X = v)] = 0.5 + 6.5 v`. Four equal-probability levels. The top-minus-bottom
//! contrast's closed form is `6.5 (v_top - v_bottom)` at the cells' own values
//! (`crates/antecedent/tests/discretized_composition_facade.rs`).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(deprecated, reason = "the direct facade path is the twin of the compiled program")]
#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::analysis::discretized_composition::{
    DiscreteArm, DiscreteContrast, DiscretizationMethod, DiscretizationPlan,
    PreparedDiscretizedComposition,
};
use antecedent::planner::PlanView;
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource, Study,
    StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{
    ComposedLinearGaussian, ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor,
    MechanismForm, MechanismOrigin, MechanismProvenance, PosteriorDependence,
};
use antecedent_model::discretized_scm::DiscretizedLinearGaussian;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn bits(left: f64, right: f64) {
    assert_eq!(left.to_bits(), right.to_bits(), "{left} vs {right}");
}

/// The closed form `0.5 + 6.5 v` is the chain's algebra. The kernel evaluates it at the
/// cell values, and the last bit is the product's, so the facade's tolerance is the check.
fn near(left: f64, right: f64) {
    assert!((left - right).abs() <= 1e-9, "{left} vs {right}");
}

fn assert_unit(view: &PlanView) {
    let support = view.unit_support().expect("the route is bound to its capability record");
    assert_eq!(support.unit(), "model.discretized_linear_gaussian_scm");
    assert!(support.supported());
    assert!(support.missing_evidence().is_empty());
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

fn chain() -> MechanismCompositionRequest {
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
    MechanismCompositionRequest {
        target_population_id: "pop".into(),
        target_regime_id: "observational".into(),
        mechanisms,
        edges,
        admissions: Vec::new(),
        dependence: DependenceInput::Unknown,
        admit_declared_mechanisms: false,
    }
}

fn plan(variables: &[String]) -> DiscretizationPlan {
    DiscretizationPlan::uniform(DiscretizationMethod::EqualProbabilityQuantile, variables, 4)
}

fn contrast() -> DiscreteContrast {
    DiscreteContrast {
        outcome: "Y".into(),
        arms: vec![
            DiscreteArm { weight: 1.0, interventions: vec![("X".into(), 3)] },
            DiscreteArm { weight: -1.0, interventions: vec![("X".into(), 0)] },
        ],
    }
}

fn descriptor(
    variable: &str,
    intercept: f64,
    sigma: f64,
    parents: &[(&str, f64)],
) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: variable.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma,
            coefficients: parents
                .iter()
                .map(|(name, coefficient)| ((*name).to_owned(), *coefficient))
                .collect(),
        },
        parents: parents
            .iter()
            .map(|(name, _)| antecedent_model::compose::ParentBinding {
                variable: (*name).to_owned(),
                units: "u".into(),
                transform: "identity".into(),
                domain: Domain::Real,
            })
            .collect(),
        conditioning_set: parents.iter().map(|(name, _)| (*name).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{variable}"),
            evidence_id: format!("data:{variable}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn specification() -> ComposedModelSpec {
    let mechanisms = vec![
        descriptor("X", 1.0, 2.0, &[]),
        descriptor("M", 0.5, 1.0, &[("X", 2.0)]),
        descriptor("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ];
    let mut edges = Vec::new();
    for mechanism in &mechanisms {
        for parent in &mechanism.parents {
            edges.push((parent.variable.clone(), mechanism.variable.clone()));
        }
    }
    ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: false,
    }
}

fn same_bridge(compiled: &DiscretizedLinearGaussian, direct: &DiscretizedLinearGaussian) {
    assert_eq!(compiled.identity_hex(), direct.identity_hex());
    for variable in direct.variables() {
        let left = compiled.level_values(variable).unwrap();
        let right = direct.level_values(variable).unwrap();
        assert_eq!(left.len(), right.len());
        for (compiled_value, direct_value) in left.iter().zip(right) {
            bits(*compiled_value, *direct_value);
        }
    }
}

#[test]
fn the_compiled_bridge_matches_new_and_the_contrast_matches_mean_contrast() {
    let prepared = StudyBuilder::mechanism_composition(chain(), &ctx()).unwrap();
    let levels = plan(prepared.variables());
    let direct = prepared.discretize(&levels, &ctx()).unwrap();
    let compiled = Study::compile_discretized_composition(&prepared, &levels, &ctx()).unwrap();
    assert_unit(&compiled.plan_view());
    let bridge = compiled.execute(&ctx()).unwrap();
    let assembled = PreparedDiscretizedComposition::from_compiled_bridge(
        levels.clone(),
        bridge,
        prepared.plan_identity_hex(),
        prepared.plan_identity_version(),
    );
    same_bridge(assembled.bridge(), direct.bridge());
    assert_eq!(assembled.source_plan_identity(), direct.source_plan_identity());

    let direct_contrast = direct.mean_contrast(&contrast(), &ctx()).unwrap();
    let compiled_contrast =
        Study::compile_discretized_mean_contrast(direct.bridge(), &contrast(), &ctx()).unwrap();
    assert_unit(&compiled_contrast.plan_view());
    let value = compiled_contrast.execute(&ctx()).unwrap();
    bits(value.value, direct_contrast.value);
    bits(value.closed_form, direct_contrast.closed_form);
    bits(value.absolute_error, direct_contrast.absolute_error);
    bits(value.error_bound, direct_contrast.error_bound);
    let x = direct.bridge().level_values("X").unwrap();
    near(value.closed_form, 6.5 * (x[3] - x[0]));
    near(value.arms[0].closed_form_mean, 0.5 + 6.5 * x[3]);
    near(value.arms[1].closed_form_mean, 0.5 + 6.5 * x[0]);
}

#[test]
fn the_compiled_spec_bridge_matches_from_spec() {
    let spec = specification();
    let model = ComposedLinearGaussian::compile(&spec).unwrap();
    let levels = plan(model.variables());
    let direct = DiscretizedLinearGaussian::from_spec(&spec, &levels, &ctx()).unwrap();
    let compiled =
        Study::compile_discretized_linear_gaussian_from_spec(&spec, &levels, &ctx()).unwrap();
    assert_unit(&compiled.plan_view());
    same_bridge(&compiled.execute(&ctx()).unwrap(), &direct);
    let from_model = Study::compile_discretized_linear_gaussian(&model, &levels, &ctx()).unwrap();
    same_bridge(&from_model.execute(&ctx()).unwrap(), &direct);
}
