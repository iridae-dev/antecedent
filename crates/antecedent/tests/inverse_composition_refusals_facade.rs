//! An inverse query's composed forward source keeps the composition's own typed refusal.
//!
//! A composition that cannot answer an action's query refuses with its own registered code,
//! namespaced detail, offending subject and remedy; the forward source carries that refusal
//! unchanged and keeps `inverse_forward_source.origin_invalid` for a replay that does not
//! reproduce its premises.
//!
//! Oracles (by construction):
//!
//! - Parameter laws declared per mechanism with no stated dependence between them: a contrast
//!   of interventional means has no joint law for its variance, which the composition refuses
//!   as `joint_law_required` / `mechanism_composition.posterior_dependence_unknown`, naming the
//!   first action whose query it refused.
//! - Mechanisms that report no parameter law at all: the composition's findings include
//!   `mechanism_composition.uncertainty_marginal_missing`.
//! - A finite-distribution query of a linear-Gaussian composition is outside the executor's
//!   family: `route_not_supported` / `mechanism_composition.request_unsupported`.
//! - The same mechanisms answered by an intervention law are a replayed point:
//!   `E[Y | do(X = 1)] = -1 + 0.5 + 3 (0.5 + 2) = 7` and `E[Y | do(X = 0)] = 0.5`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one scenario per test")]

use std::sync::Arc;

use antecedent::analysis::inverse_query::{
    Comparison, GridScope, InverseConstraint, InverseQuery, SelectionRule,
};
use antecedent::analysis::inverse_sources::{
    ComposedActionQuery, ComposedSourceRefusal, ForwardSourceRefusal, InverseDemand,
    TransportedInputPolicy, evaluate,
};
use antecedent::analysis::mechanism_composition::CompositionQueryWire;
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{Domain, MechanismUncertainty, ParameterTerm, UncertaintyMeaning};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn role(variable: &str) -> QuantityRole {
    match variable {
        "X" => QuantityRole::Treatment,
        "M" => QuantityRole::Mediator,
        "Y" => QuantityRole::Outcome,
        _ => QuantityRole::Covariate,
    }
}

fn coordinate(variable: &str, regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: role(variable),
        units: "u".into(),
        population_id: "pop".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn declaration(variable: &str, parents: &[&str]) -> MechanismDeclaration {
    MechanismDeclaration {
        output: coordinate(variable, "observational", "realization"),
        parents: parents.iter().map(|p| coordinate(p, "observational", "realization")).collect(),
        conditioning_variable_ids: parents.iter().map(|p| (*p).to_owned()).collect(),
        origin: MechanismOriginKind::Fitted,
        artifact_id: format!("fit:{variable}"),
        evidence_id: format!("data:{variable}"),
        latent_causes: Vec::new(),
    }
}

fn linear(
    variable: &str,
    intercept: f64,
    sigma: f64,
    parents: &[(&str, f64)],
    uncertainty: Option<MechanismUncertainty>,
) -> MechanismInput {
    let names: Vec<&str> = parents.iter().map(|(p, _)| *p).collect();
    let slot = MechanismSlot::LinearGaussian {
        intercept,
        coeffs: Arc::from(parents.iter().map(|(_, c)| *c).collect::<Vec<f64>>()),
        sigma,
    };
    MechanismInput {
        declaration: declaration(variable, &names),
        source: MechanismSource::from_fitted_slot(&slot, &names).unwrap(),
        output_support: Domain::Real,
        parent_domains: names.iter().map(|p| ((*p).to_owned(), Domain::Real)).collect(),
        uncertainty,
        latent_loadings: Vec::new(),
    }
}

fn marginal(terms: Vec<ParameterTerm>, covariance: Vec<f64>) -> MechanismUncertainty {
    MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: terms,
        covariance,
    }
}

fn request_of(mechanisms: Vec<MechanismInput>) -> MechanismCompositionRequest {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.declaration.parents {
            edges.push((p.variable_id.clone(), m.declaration.output.variable_id.clone()));
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

/// `X = 1 + 2 e`, `M = 0.5 + 2 X + e`, `Y = -1 + 0.5 X + 3 M + 0.5 e`, no parameter law.
fn bare_chain() -> Vec<MechanismInput> {
    vec![
        linear("X", 1.0, 2.0, &[], None),
        linear("M", 0.5, 1.0, &[("X", 2.0)], None),
        linear("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)], None),
    ]
}

/// The same chain with a marginal parameter law per mechanism and no stated dependence.
fn uncertain_chain() -> Vec<MechanismInput> {
    let mut mechanisms = bare_chain();
    mechanisms[0].uncertainty = Some(marginal(vec![ParameterTerm::Intercept], vec![0.25]));
    mechanisms[1].uncertainty = Some(marginal(
        vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())],
        vec![0.04, -0.01, -0.01, 0.01],
    ));
    mechanisms[2].uncertainty = Some(marginal(
        vec![
            ParameterTerm::Intercept,
            ParameterTerm::Coefficient("X".into()),
            ParameterTerm::Coefficient("M".into()),
        ],
        vec![0.09, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0025],
    ));
    mechanisms
}

fn contrast_actions(arms: &[f64]) -> Vec<ComposedActionQuery> {
    arms.iter()
        .map(|arm| ComposedActionQuery {
            action_id: format!("x{arm}"),
            coordinate: coordinate("Y", &format!("do(X={arm})"), "mean"),
            query: CompositionQueryWire::GaussianMeanContrast {
                outcome: "Y".into(),
                arms: vec![(1.0, vec![("X".into(), *arm)])],
            },
            level_values: vec![],
        })
        .collect()
}

fn law_actions(arms: &[f64]) -> Vec<ComposedActionQuery> {
    arms.iter()
        .map(|arm| ComposedActionQuery {
            action_id: format!("x{arm}"),
            coordinate: coordinate("Y", &format!("do(X={arm})"), "mean"),
            query: CompositionQueryWire::GaussianInterventionLaw {
                interventions: vec![("X".into(), *arm)],
                outcome: "Y".into(),
            },
            level_values: vec![],
        })
        .collect()
}

fn finite_actions_of_a_gaussian_composition() -> Vec<ComposedActionQuery> {
    vec![ComposedActionQuery {
        action_id: "x0".into(),
        coordinate: coordinate("Y", "do(X=0)", "outcome"),
        query: CompositionQueryWire::FiniteInterventionDistribution {
            interventions: vec![("X".into(), "0".into())],
            outcome: "Y".into(),
        },
        level_values: vec![("0".into(), 0.0), ("1".into(), 1.0)],
    }]
}

fn native() -> TransportedInputPolicy {
    TransportedInputPolicy::NativeTargetOnly
}

#[test]
fn a_mean_contrast_without_a_declared_joint_law_surfaces_the_joint_law_refusal() {
    let prepared =
        StudyBuilder::mechanism_composition(request_of(uncertain_chain()), &ctx()).unwrap();
    let error = prepared
        .inverse_forward_source_typed(&contrast_actions(&[0.0, 1.0]), None, native(), &ctx())
        .unwrap_err();

    // The composition's own refusal, unchanged, naming the first action it refused.
    let ComposedSourceRefusal::Composition { action_id, refusal } = &error else {
        panic!("expected the composition's own refusal, got {error:?}");
    };
    assert_eq!(action_id, "x0");
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(refusal.details(), vec!["mechanism_composition.posterior_dependence_unknown"]);
    assert!(refusal.refusals[0].remedy.is_some());
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.detail(), "mechanism_composition.posterior_dependence_unknown");
    assert_ne!(error.detail(), "inverse_forward_source.origin_invalid");
    assert_eq!(error.action_id(), Some("x0"));

    // The structured form keeps the offending subject, the expected semantics and the remedy.
    let structured = error.to_refusal();
    assert_eq!(structured.code, "joint_law_required");
    assert_eq!(structured.detail, "mechanism_composition.posterior_dependence_unknown");
    assert!(structured.remedy.is_some());
    assert_eq!(error.refusals(), refusal.refusals);

    // The earlier return type still carries the code and the first detail, flattened.
    let flat = prepared
        .inverse_forward_source(&contrast_actions(&[0.0, 1.0]), None, native(), &ctx())
        .unwrap_err();
    assert!(matches!(flat, ForwardSourceRefusal::Artifact { .. }), "{flat:?}");
    assert_eq!(flat.code(), "joint_law_required");
    assert_eq!(flat.detail(), "mechanism_composition.posterior_dependence_unknown");
}

#[test]
fn mechanisms_with_no_parameter_law_refuse_a_contrast_with_the_missing_marginals() {
    let prepared = StudyBuilder::mechanism_composition(request_of(bare_chain()), &ctx()).unwrap();
    let error = prepared
        .inverse_forward_source_typed(&contrast_actions(&[1.0]), None, native(), &ctx())
        .unwrap_err();
    let ComposedSourceRefusal::Composition { refusal, .. } = &error else {
        panic!("expected the composition's own refusal, got {error:?}");
    };
    let details = refusal.details();
    assert!(details.contains(&"mechanism_composition.uncertainty_marginal_missing"), "{details:?}");
    assert!(details.contains(&"mechanism_composition.posterior_dependence_unknown"), "{details:?}");
    assert!(refusal.refusals.iter().all(|r| r.detail != "inverse_forward_source.origin_invalid"));
    assert!(refusal.refusals.iter().any(|r| r.code == "joint_law_required"));
}

#[test]
fn a_finite_query_of_a_gaussian_composition_is_an_unsupported_request_not_an_invalid_origin() {
    let prepared = StudyBuilder::mechanism_composition(request_of(bare_chain()), &ctx()).unwrap();
    let error = prepared
        .inverse_forward_source_typed(
            &finite_actions_of_a_gaussian_composition(),
            None,
            native(),
            &ctx(),
        )
        .unwrap_err();
    let ComposedSourceRefusal::Composition { action_id, refusal } = &error else {
        panic!("expected the composition's own refusal, got {error:?}");
    };
    assert_eq!(action_id, "x0");
    assert_eq!(refusal.code(), "route_not_supported");
    assert_eq!(refusal.details(), vec!["mechanism_composition.request_unsupported"]);
    assert!(refusal.refusals[0].remedy.is_some());
    assert!(refusal.refusals[0].offending.as_deref().is_some_and(|s| s.contains("x0")));
}

#[test]
fn a_composed_refusal_of_the_forward_source_stays_a_source_refusal() {
    // Level coding is the adapter's refusal, not the composition's: it keeps its own type.
    let prepared = StudyBuilder::mechanism_composition(request_of(bare_chain()), &ctx()).unwrap();
    let mut actions = law_actions(&[0.0]);
    actions[0].coordinate = coordinate("Y", "do(X=0)", "outcome");
    let error = prepared
        .inverse_forward_source_typed(&actions, None, native(), &ctx())
        .unwrap_err();
    assert!(
        matches!(
            error,
            ComposedSourceRefusal::Source(ForwardSourceRefusal::ComposedCoordinateIsOutcomeLaw(_))
        ),
        "{error:?}"
    );
    assert_eq!(error.code(), "invalid_argument");
    assert_eq!(error.detail(), "inverse_forward_source.composed_coordinate_is_outcome_law");
    assert!(error.action_id().is_none());
}

#[test]
fn the_same_mechanisms_answered_by_an_intervention_law_are_a_replayed_point() {
    let prepared = StudyBuilder::mechanism_composition(request_of(bare_chain()), &ctx()).unwrap();
    let source = prepared
        .inverse_forward_source_typed(&law_actions(&[0.0, 1.0]), None, native(), &ctx())
        .unwrap();

    let contract = DecisionContract {
        actions: ["0", "1"]
            .iter()
            .map(|arm| DecisionAction {
                id: format!("x{arm}"),
                kind: ActionKind::Intervention,
                inputs: vec![coordinate("Y", &format!("do(X={arm})"), "mean")],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "u".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "pop".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let query = InverseQuery {
        contract,
        grid_order: vec!["x0".into(), "x1".into()],
        grid_scope: GridScope::FiniteEnumeration,
        constraints: vec![InverseConstraint::TargetMean {
            target: 5.0,
            comparison: Comparison::AtLeast,
        }],
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    };
    let sourced = evaluate(&query, &source, InverseDemand::default()).unwrap();
    let values: Vec<f64> =
        sourced.result.actions.iter().map(|a| a.point_values[0].value.unwrap()).collect();
    assert!((values[0] - 0.5).abs() <= 1e-9, "{values:?}");
    assert!((values[1] - 7.0).abs() <= 1e-9, "{values:?}");
    assert_eq!(sourced.result.selected.as_deref(), Some("x1"));
}
