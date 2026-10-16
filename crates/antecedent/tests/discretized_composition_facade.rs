//! A prepared linear-Gaussian composition is discretized into an executable finite
//! model through the facade: a computed error bound beside the closed-form value, a
//! refresh that keeps the declared plan and recomputes grid, tables and bound, and an
//! export a consumer re-derives from the carried premises.
//!
//! Oracle. For the chain `X = 1 + 2 e`, `M = 0.5 + 2 X + e`, `Y = -1 + 0.5 X + 3 M + 0.5 e`
//! the closed-form interventional mean is linear in the intervention value `v`:
//! `E[Y | do(X = v)] = 0.5 + 6.5 v`. A contrast of two top and bottom cells has the
//! closed-form value `6.5 (v_top - v_bottom)` at the cells' own values. After `M` is refit
//! as `1 + 3 X`, `E[Y | do(X = v)] = 2 + 9.5 v`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::discretized_composition::{
    BRIDGE_CALIBRATION, BRIDGE_INFERENCE, DiscreteArm, DiscreteContrast, DiscretizationMethod,
    DiscretizationPlan, DiscretizationRefusal, InferenceParadigm, consume_discretized_composition,
};
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{Domain, TabularRow};
use antecedent_model::discretized_scm::{DETAIL_LEVELS_OUT_OF_RANGE, DETAIL_LEVELS_UNDECLARED};
use antecedent_model::discretized_scm_artifact::DETAIL_IDENTITY_MISMATCH;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-9, "{a} vs {b}");
}

fn quantity(variable: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: match variable {
            "X" => QuantityRole::Treatment,
            "M" => QuantityRole::Mediator,
            "Y" => QuantityRole::Outcome,
            _ => QuantityRole::Covariate,
        },
        units: units.into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn linear(variable: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismInput {
    let names: Vec<&str> = parents.iter().map(|(p, _)| *p).collect();
    let slot = MechanismSlot::LinearGaussian {
        intercept,
        coeffs: Arc::from(parents.iter().map(|(_, c)| *c).collect::<Vec<f64>>()),
        sigma,
    };
    MechanismInput {
        declaration: MechanismDeclaration {
            output: quantity(variable, "u"),
            parents: names.iter().map(|p| quantity(p, "u")).collect(),
            conditioning_variable_ids: names.iter().map(|p| (*p).to_owned()).collect(),
            origin: MechanismOriginKind::Fitted,
            artifact_id: format!("fit:{variable}"),
            evidence_id: format!("data:{variable}"),
            latent_causes: Vec::new(),
        },
        source: MechanismSource::from_fitted_slot(&slot, &names).unwrap(),
        output_support: Domain::Real,
        parent_domains: names.iter().map(|p| ((*p).to_owned(), Domain::Real)).collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
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

fn chain(m_intercept: f64, m_slope: f64) -> MechanismCompositionRequest {
    request_of(vec![
        linear("X", 1.0, 2.0, &[]),
        linear("M", m_intercept, 1.0, &[("X", m_slope)]),
        linear("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ])
}

fn plan(variables: &[String], levels: usize) -> DiscretizationPlan {
    DiscretizationPlan::uniform(DiscretizationMethod::EqualProbabilityQuantile, variables, levels)
}

fn top_minus_bottom() -> DiscreteContrast {
    DiscreteContrast {
        outcome: "Y".into(),
        arms: vec![
            DiscreteArm { weight: 1.0, interventions: vec![("X".into(), 3)] },
            DiscreteArm { weight: -1.0, interventions: vec![("X".into(), 0)] },
        ],
    }
}

#[test]
fn a_discretized_contrast_carries_its_closed_form_and_a_computed_bound() {
    let prepared = StudyBuilder::mechanism_composition(chain(0.5, 2.0), &ctx()).unwrap();
    let bridged = prepared.discretize(&plan(prepared.variables(), 4), &ctx()).unwrap();
    let values = bridged.bridge().level_values("X").unwrap().to_vec();
    assert_eq!(values.len(), 4);
    // The equal-probability cells of N(1, 2^2) are symmetric about the mean, and a cell
    // carries its conditional mean, so the discretized marginal reproduces the mean.
    near(values.iter().sum::<f64>() / 4.0, 1.0);

    let result = bridged.mean_contrast(&top_minus_bottom(), &ctx()).unwrap();
    near(result.closed_form, 6.5 * (values[3] - values[0]));
    assert!(result.error_bound.is_finite() && result.error_bound >= 0.0);
    assert!(result.absolute_error <= result.error_bound + 1e-9);
    assert_eq!(result.inference, BRIDGE_INFERENCE);
    assert_eq!(result.calibration, BRIDGE_CALIBRATION);
    assert!(!result.parameter_uncertainty_not_propagated);

    // An interval, posterior or law reading of a point is refused for both paradigms.
    for paradigm in [InferenceParadigm::Frequentist, InferenceParadigm::Bayesian] {
        let refusal = bridged.uncertainty(paradigm).unwrap_err();
        assert_eq!(refusal.code(), "cell_not_licensed");
        assert_eq!(refusal.detail(), "discretized_scm.point_only");
        assert!(refusal.remedy().is_some());
    }

    let inspection = bridged.inspect();
    assert_eq!(inspection.inference, "point_only");
    assert_eq!(inspection.variables.len(), 3);
    assert!(inspection.variables.iter().all(|(_, levels, _)| *levels == 4));
    assert_eq!(inspection.source_plan_identity, prepared.plan_identity_hex());
}

#[test]
fn a_refresh_keeps_the_plan_and_recomputes_the_grid_the_bound_and_the_identity() {
    let base = StudyBuilder::mechanism_composition(chain(0.5, 2.0), &ctx()).unwrap();
    let bridged = base.discretize(&plan(base.variables(), 4), &ctx()).unwrap();
    let refit = StudyBuilder::mechanism_composition(chain(1.0, 3.0), &ctx()).unwrap();

    let refreshed = bridged.refresh(&refit, &ctx()).unwrap();
    // Kept: the declared method and level counts.
    assert_eq!(refreshed.plan(), bridged.plan());
    // Recomputed: identity, the grid of the refit variable and
    // the closed-form value (E[Y | do(X = v)] = 2 + 9.5 v).
    assert_ne!(refreshed.identity_hex(), bridged.identity_hex());
    // The plan identity covers declarations (coordinates, provenance ids), not numbers.
    assert_eq!(refreshed.source_plan_identity(), bridged.source_plan_identity());
    assert_ne!(
        refreshed.bridge().level_values("M").unwrap(),
        bridged.bridge().level_values("M").unwrap()
    );
    let values = refreshed.bridge().level_values("X").unwrap().to_vec();
    let result = refreshed.mean_contrast(&top_minus_bottom(), &ctx()).unwrap();
    near(result.closed_form, 9.5 * (values[3] - values[0]));
    assert!(result.absolute_error <= result.error_bound + 1e-9);

    // The earlier bridge is untouched.
    let before = bridged.mean_contrast(&top_minus_bottom(), &ctx()).unwrap();
    near(
        before.closed_form,
        6.5 * (bridged.bridge().level_values("X").unwrap()[3]
            - bridged.bridge().level_values("X").unwrap()[0]),
    );

    // A refit whose variables the plan does not cover is refused, never patched.
    let mut with_extra = chain(0.5, 2.0);
    with_extra.mechanisms.push(linear("W", 0.0, 1.0, &[("X", 1.0)]));
    with_extra.edges.push(("X".into(), "W".into()));
    let extra = StudyBuilder::mechanism_composition(with_extra, &ctx()).unwrap();
    let refusal = bridged.refresh(&extra, &ctx()).unwrap_err();
    assert_eq!(refusal.detail(), DETAIL_LEVELS_UNDECLARED);
}

#[test]
fn the_plan_and_the_composition_are_refused_with_their_own_reasons() {
    let prepared = StudyBuilder::mechanism_composition(chain(0.5, 2.0), &ctx()).unwrap();
    // A level count is never defaulted.
    let mut partial = plan(prepared.variables(), 4);
    partial.levels.remove("M");
    let refusal = prepared.discretize(&partial, &ctx()).unwrap_err();
    assert_eq!(refusal.detail(), DETAIL_LEVELS_UNDECLARED);
    // One level is no grid.
    let refusal = prepared.discretize(&plan(prepared.variables(), 1), &ctx()).unwrap_err();
    assert_eq!(refusal.detail(), DETAIL_LEVELS_OUT_OF_RANGE);
    assert!(refusal.remedy().is_some());

    // A finite-table composition has no Gaussian executable to discretize.
    let levels = vec!["0".to_owned(), "1".to_owned()];
    let table = MechanismInput {
        declaration: MechanismDeclaration {
            output: quantity("X", "category"),
            parents: vec![],
            conditioning_variable_ids: vec![],
            origin: MechanismOriginKind::Declared,
            artifact_id: "decl:X".into(),
            evidence_id: "decl:X".into(),
            latent_causes: Vec::new(),
        },
        source: MechanismSource::DeclaredTable {
            levels: levels.clone(),
            rows: vec![TabularRow {
                parent_levels: vec![],
                probabilities: vec![("0".into(), 0.5), ("1".into(), 0.5)],
            }],
        },
        output_support: Domain::Levels(levels),
        parent_domains: vec![],
        uncertainty: None,
        latent_loadings: Vec::new(),
    };
    let mut request = request_of(vec![table]);
    request.admit_declared_mechanisms = true;
    let finite = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    let refusal = finite.discretize(&plan(&["X".to_owned()], 4), &ctx()).unwrap_err();
    assert!(matches!(refusal, DiscretizationRefusal::Composition(_)));
    assert_eq!(refusal.code(), "route_not_supported");
}

#[test]
fn an_export_is_re_derived_by_a_consumer_from_the_carried_premises() {
    let prepared = StudyBuilder::mechanism_composition(chain(0.5, 2.0), &ctx()).unwrap();
    let bridged = prepared.discretize(&plan(prepared.variables(), 4), &ctx()).unwrap();
    let bytes = bridged.export().unwrap();
    let identity = bridged.bridge().identity();

    let consumed = consume_discretized_composition(&bytes, Some(identity), &ctx()).unwrap();
    assert_eq!(consumed.identity(), identity);
    assert_eq!(consumed.level_values("M").unwrap(), bridged.bridge().level_values("M").unwrap());

    // A self-consistent artifact of another composition is refused by a consumer that
    // retained the identity it expects.
    let other = StudyBuilder::mechanism_composition(chain(1.0, 3.0), &ctx()).unwrap();
    let other_bridge = other.discretize(&plan(other.variables(), 4), &ctx()).unwrap();
    let error =
        consume_discretized_composition(&other_bridge.export().unwrap(), Some(identity), &ctx())
            .unwrap_err();
    assert_eq!(error.detail(), DETAIL_IDENTITY_MISMATCH);

    // A damaged artifact is refused.
    let mut damaged = bytes;
    damaged.truncate(damaged.len() - 3);
    assert!(consume_discretized_composition(&damaged, None, &ctx()).is_err());
}
