//! Fitted linear-Gaussian mechanisms carried into a target population's composed model
//! under a declared selection diagram: hand-computed laws of the transported system, the typed
//! receipt and identity, precedence of a native refit, selection-arrow-over-invariance, the
//! scoped parameter variance, the sealed artifact a consumer replays, and the named refusals.
//!
//! Source chain (fitted): `X = 1 + 2 e_X`, `M = 0.5 + 2 X + e_M`, `Y = -1 + 0.5 X + 3 M + 0.5 e_Y`.
//! Target: a selection node `S` points into `M` only. `M` is natively refitted in the target as
//! `M = 1.5 + 0.5 X + 2 e_M`; `X` and `Y` are carried. The transported system is therefore
//! `Y = 3.5 + 2 X + 6 e_M + 0.5 e_Y`, so
//!
//! * `E[X] = 1`, `E[M] = 2`, `E[Y] = 5.5`; `Var X = 4`, `Var M = 5`, `Var Y = 52.25`,
//!   `Cov(X, Y) = 8`;
//! * `do(X = 2)`: `E[M] = 2.5`, `E[Y] = 7.5`, `Var Y = 9 * 4 + 0.25 = 36.25`;
//! * `E[Y | do(X = 1)] - E[Y | do(X = 0)] = 0.5 + 3 * 0.5 = 2`; the source's is `6.5`.
//!
//! With the marginal parameter laws of the source chain and a declared independence, the
//! first-order variance of `E[Y | do(X = 2)]` has gradient `(3, 6, 0, 1, 2, 2.5)` over
//! `(a_M, b_MX, a_X, a_Y, b_YX, b_YM)` and equals `0.36 + 0.09 + 0.04 + 0.015625 = 0.505625`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::mechanism_composition::{
    CompositionQueryWire, CompositionResultWire, consume_mechanism_composition_artifact,
};
use antecedent::analysis::transported_mechanism::{
    DETAIL_REFIT_FAMILY_CHANGED, DETAIL_REFIT_UNKNOWN_VARIABLE, DETAIL_TRANSPORT_IDENTITY_MISMATCH,
    MECHANISM_TRANSPORT_INFERENCE, MECHANISM_TRANSPORT_ROUTE, MECHANISM_TRANSPORT_SUPPORT,
    MechanismRole, MechanismTransportDeclaration, UNCERTAINTY_SCOPE, verify_mechanism_transport,
};
use antecedent::{
    DependenceInput, MechanismCompositionRefusal, MechanismCompositionRequest, MechanismInput,
    MechanismRefit, MechanismSource, PreparedMechanismComposition, StudyBuilder,
};
use antecedent_contract::{
    MechanismDeclaration, MechanismOriginKind, QuantityRole, ScientificQuantity,
};
use antecedent_core::ExecutionContext;
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{
    ContextCoordinate, ContrastArm, Domain, InvarianceDeclaration, MeanContrast,
    MechanismUncertainty, ParameterTerm, SelectionNodeDeclaration, UncertaintyMeaning,
};

const TOL: f64 = 1e-9;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

fn role(variable: &str) -> QuantityRole {
    match variable {
        "X" => QuantityRole::Treatment,
        "M" => QuantityRole::Mediator,
        "Y" => QuantityRole::Outcome,
        _ => QuantityRole::Covariate,
    }
}

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: format!("label of {variable}"),
        role: role(variable),
        units: "u".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn declaration_of(variable: &str, parents: &[&str]) -> MechanismDeclaration {
    MechanismDeclaration {
        output: quantity(variable),
        parents: parents.iter().map(|p| quantity(p)).collect(),
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
    declared: bool,
) -> MechanismInput {
    let names: Vec<&str> = parents.iter().map(|(p, _)| *p).collect();
    let coefficients: Vec<(String, f64)> =
        parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect();
    let source = if declared {
        MechanismSource::DeclaredLinear { intercept, sigma, coefficients }
    } else {
        let slot = MechanismSlot::LinearGaussian {
            intercept,
            coeffs: Arc::from(parents.iter().map(|(_, c)| *c).collect::<Vec<f64>>()),
            sigma,
        };
        MechanismSource::from_fitted_slot(&slot, &names).unwrap()
    };
    MechanismInput {
        declaration: declaration_of(variable, &names),
        source,
        output_support: Domain::Real,
        parent_domains: names.iter().map(|p| ((*p).to_owned(), Domain::Real)).collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

fn chain(declared: bool) -> Vec<MechanismInput> {
    vec![
        linear("X", 1.0, 2.0, &[], declared),
        linear("M", 0.5, 1.0, &[("X", 2.0)], declared),
        linear("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)], declared),
    ]
}

fn marginal(terms: Vec<ParameterTerm>, covariance: Vec<f64>) -> MechanismUncertainty {
    MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: terms,
        covariance,
    }
}

fn m_law() -> MechanismUncertainty {
    marginal(
        vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())],
        vec![0.04, -0.01, -0.01, 0.01],
    )
}

fn uncertain_chain() -> Vec<MechanismInput> {
    let mut mechanisms = chain(false);
    mechanisms[0].uncertainty = Some(marginal(vec![ParameterTerm::Intercept], vec![0.25]));
    mechanisms[1].uncertainty = Some(m_law());
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

fn prepare(
    mechanisms: Vec<MechanismInput>,
    dependence: DependenceInput,
    admit_declared: bool,
) -> PreparedMechanismComposition {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.declaration.parents {
            edges.push((p.variable_id.clone(), m.declaration.output.variable_id.clone()));
        }
    }
    StudyBuilder::mechanism_composition(
        MechanismCompositionRequest {
            target_population_id: "pop".into(),
            target_regime_id: "observational".into(),
            mechanisms,
            edges,
            admissions: Vec::new(),
            dependence,
            admit_declared_mechanisms: admit_declared,
        },
        &ctx(),
    )
    .unwrap()
}

/// `M` refitted in the target: `M = 1.5 + 0.5 X + 2 e_M`.
fn refit_m(intercept: f64, uncertainty: Option<MechanismUncertainty>) -> MechanismRefit {
    MechanismRefit {
        variable_id: "M".into(),
        source: MechanismSource::FittedLinear {
            intercept,
            sigma: 2.0,
            coefficients: vec![("X".into(), 0.5)],
        },
        artifact_id: "fit:M:target".into(),
        evidence_id: "data:M:target".into(),
        uncertainty,
    }
}

fn transport(into: &[&str], refits: Vec<MechanismRefit>) -> MechanismTransportDeclaration {
    MechanismTransportDeclaration {
        target_population_id: "clinic_b".into(),
        target_regime_id: "observational".into(),
        selection_nodes: vec![SelectionNodeDeclaration {
            name: "S".into(),
            points_into: into.iter().map(|v| (*v).to_owned()).collect(),
        }],
        target_refits: refits,
        invariances: Vec::new(),
        dependence: DependenceInput::Unknown,
    }
}

fn at_x(x: f64) -> MeanContrast {
    MeanContrast {
        outcome: "Y".into(),
        arms: vec![ContrastArm { weight: 1.0, interventions: vec![("X".into(), x)] }],
    }
}

fn ate() -> MeanContrast {
    MeanContrast {
        outcome: "Y".into(),
        arms: vec![
            ContrastArm { weight: 1.0, interventions: vec![("X".into(), 1.0)] },
            ContrastArm { weight: -1.0, interventions: vec![("X".into(), 0.0)] },
        ],
    }
}

fn details(refusal: &MechanismCompositionRefusal) -> Vec<&str> {
    refusal.details()
}

#[test]
fn carried_and_refitted_components_give_the_hand_computed_law_and_a_typed_receipt() {
    let source = prepare(chain(false), DependenceInput::Unknown, false);
    let transported =
        source.transport_to(&transport(&["M"], vec![refit_m(1.5, None)]), &ctx()).unwrap();

    near(transported.mean_contrast(&at_x(2.0)).unwrap(), 7.5);
    near(transported.mean_contrast(&ate()).unwrap(), 2.0);
    let law = transported.model().model().law(&[("X", 2.0)]).unwrap();
    near(law.mean_of("M").unwrap(), 2.5);
    near(law.mean_of("Y").unwrap(), 7.5);
    near(law.variance_of("Y").unwrap(), 36.25);
    let observational = transported.model().model().law(&[]).unwrap();
    near(observational.mean_of("Y").unwrap(), 5.5);
    near(observational.variance_of("M").unwrap(), 5.0);
    near(observational.variance_of("Y").unwrap(), 52.25);
    near(observational.covariance_of("X", "Y").unwrap(), 8.0);

    let receipt = transported.receipt();
    assert_eq!(receipt.route, MECHANISM_TRANSPORT_ROUTE);
    assert_eq!(receipt.support, MECHANISM_TRANSPORT_SUPPORT);
    assert_eq!(receipt.inference, MECHANISM_TRANSPORT_INFERENCE);
    assert_eq!(receipt.uncertainty_scope, UNCERTAINTY_SCOPE);
    assert_eq!(receipt.identity, transported.identity_hex());
    assert_eq!(receipt.identity.len(), 64);
    assert_eq!(receipt.source_plan_identity, source.plan_identity_hex());
    assert_eq!(receipt.source_population, "pop");
    assert_eq!(receipt.target_population, "clinic_b");
    assert_eq!(receipt.selection_targets, vec!["M".to_owned()]);
    assert_eq!(transported.transported_variables(), vec!["X", "Y"]);
    let m = receipt.dispositions.iter().find(|d| d.variable == "M").unwrap();
    assert_eq!(m.role, MechanismRole::NativeInTarget);
    assert_eq!(m.fitted_population, "clinic_b");
    assert_eq!(m.evidence_id, "data:M:target");
    let x = receipt.dispositions.iter().find(|d| d.variable == "X").unwrap();
    assert_eq!(x.role, MechanismRole::Transported);
    assert_eq!(x.fitted_population, "pop");

    // The source composition is untouched: its contrast is the source's 6.5.
    near(source.mean_contrast(&ate()).unwrap(), 6.5);
}

#[test]
fn a_refit_replaces_the_carried_mechanism_and_is_never_averaged() {
    let source = prepare(chain(false), DependenceInput::Unknown, false);
    // No selection arrow at all: everything is carried unless a refit is supplied.
    let carried = source.transport_to(&transport(&[], vec![]), &ctx()).unwrap();
    near(carried.mean_contrast(&ate()).unwrap(), 6.5);
    assert_eq!(carried.transported_variables(), vec!["M", "X", "Y"]);
    let replaced = source.transport_to(&transport(&[], vec![refit_m(1.5, None)]), &ctx()).unwrap();
    near(replaced.mean_contrast(&ate()).unwrap(), 2.0);
    assert_ne!(replaced.identity_hex(), carried.identity_hex());
}

#[test]
fn a_changed_number_at_an_unchanged_shape_moves_the_identity_and_a_consumer_rederives_it() {
    let source = prepare(chain(false), DependenceInput::Unknown, false);
    let declaration = transport(&["M"], vec![refit_m(1.5, None)]);
    let base = source.transport_to(&declaration, &ctx()).unwrap();
    let moved = source.transport_to(&transport(&["M"], vec![refit_m(1.75, None)]), &ctx()).unwrap();
    assert_ne!(moved.identity_hex(), base.identity_hex());
    near(moved.mean_contrast(&at_x(2.0)).unwrap(), -1.0 + 1.0 + 3.0 * (1.75 + 1.0));

    let again =
        verify_mechanism_transport(&source, &declaration, &base.identity_hex(), &ctx()).unwrap();
    assert_eq!(again.identity_hex(), base.identity_hex());
    let error =
        verify_mechanism_transport(&source, &declaration, &"0".repeat(64), &ctx()).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_TRANSPORT_IDENTITY_MISMATCH]);
}

#[test]
fn the_variance_is_parameter_only_scoped_and_the_interval_is_closed() {
    let source = prepare(uncertain_chain(), DependenceInput::Unknown, false);
    let mut declaration = transport(&["M"], vec![refit_m(1.5, Some(m_law()))]);
    declaration.dependence = DependenceInput::IndependentByDesign {
        justification: "disjoint trials, independently fitted".into(),
    };
    let transported = source.transport_to(&declaration, &ctx()).unwrap();
    let value = transported.mean_contrast_uncertainty(&at_x(2.0)).unwrap();
    near(value.functional.point, 7.5);
    near(value.functional.variance, 0.505_625);
    near(value.functional.standard_error, 0.505_625_f64.sqrt());
    assert_eq!(value.scope, UNCERTAINTY_SCOPE);
    assert_eq!(value.functional.method, "first_order_delta_method");

    let closed = transported.interval().unwrap_err();
    assert_eq!(closed.code(), "cell_not_licensed");
    assert_eq!(details(&closed), vec!["mechanism_transport.interval_route_closed"]);

    // An unknown dependence refuses the variance, never the point.
    let unknown =
        source.transport_to(&transport(&["M"], vec![refit_m(1.5, Some(m_law()))]), &ctx()).unwrap();
    near(unknown.mean_contrast(&at_x(2.0)).unwrap(), 7.5);
    let refused = unknown.mean_contrast_uncertainty(&at_x(2.0)).unwrap_err();
    assert_eq!(refused.code(), "joint_law_required");
}

#[test]
fn the_sealed_artifact_is_the_existing_one_and_replays_to_the_same_bits() {
    let source = prepare(chain(false), DependenceInput::Unknown, false);
    let transported =
        source.transport_to(&transport(&["M"], vec![refit_m(1.5, None)]), &ctx()).unwrap();
    let query = CompositionQueryWire::GaussianInterventionLaw {
        interventions: vec![("X".into(), 2.0)],
        outcome: "Y".into(),
    };
    let exported = transported.export(query, &ctx()).unwrap();
    let CompositionResultWire::GaussianLaw { mean_bits, variance_bits, .. } = &exported.result
    else {
        panic!("a Gaussian law was asked for");
    };
    near(f64::from_bits(*mean_bits), 7.5);
    near(f64::from_bits(*variance_bits), 36.25);

    let consumed = consume_mechanism_composition_artifact(
        &exported.bytes,
        Some(&exported.premises_digest),
        &ctx(),
    )
    .unwrap();
    let (mean, variance) = consumed.gaussian_law().unwrap();
    assert_eq!(mean.to_bits(), *mean_bits);
    assert_eq!(variance.to_bits(), *variance_bits);
    assert_eq!(consumed.premises_digest, exported.premises_digest);
}

#[test]
fn a_selection_arrow_dominates_an_invariance_declaration() {
    let source = prepare(chain(false), DependenceInput::Unknown, false);
    let mut declaration = transport(&["M"], vec![]);
    declaration.invariances = vec![InvarianceDeclaration {
        variable: "M".into(),
        coordinate: ContextCoordinate::Population,
        source: "pop".into(),
        target: "clinic_b".into(),
        assumption_id: "assume:m-invariant".into(),
    }];
    let error = source.transport_to(&declaration, &ctx()).unwrap_err();
    assert!(
        details(&error)
            .contains(&"mechanism_composition.population_selection_arrow_into_mechanism"),
        "{:?}",
        details(&error)
    );
    assert_eq!(error.code(), "quantity_semantics_mismatch");
    assert!(error.refusals.iter().all(|r| r.remedy.is_some()));
}

#[test]
fn declared_and_unknown_refits_and_declared_sources_are_refused_by_name() {
    let source = prepare(chain(false), DependenceInput::Unknown, false);
    // A target mechanism the analyst wrote down is not a natively refitted one.
    let mut written = refit_m(1.5, None);
    written.source = MechanismSource::DeclaredLinear {
        intercept: 1.5,
        sigma: 2.0,
        coefficients: vec![("X".into(), 0.5)],
    };
    let error = source.transport_to(&transport(&["M"], vec![written]), &ctx()).unwrap_err();
    assert!(
        details(&error)
            .contains(&"mechanism_composition.provenance_declared_mechanism_unacknowledged"),
        "{:?}",
        details(&error)
    );
    // A refit of a variable the composition has no mechanism for.
    let mut stray = refit_m(1.5, None);
    stray.variable_id = "Q".into();
    let error = source.transport_to(&transport(&["M"], vec![stray]), &ctx()).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_REFIT_UNKNOWN_VARIABLE]);
    assert_eq!(error.code(), "quantity_semantics_mismatch");
    // A refit that changes the family.
    let mut table = refit_m(1.5, None);
    table.source = MechanismSource::DeclaredTable { levels: vec!["a".into()], rows: Vec::new() };
    let error = source.transport_to(&transport(&["M"], vec![table]), &ctx()).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_REFIT_FAMILY_CHANGED]);

    // A declared (not fitted) source mechanism is not carried.
    let declared = prepare(chain(true), DependenceInput::Unknown, true);
    let error = declared.transport_to(&transport(&["M"], vec![]), &ctx()).unwrap_err();
    assert!(details(&error).contains(&"mechanism_transport.origin_not_fitted"));
    assert!(
        error.refusals.iter().any(|r| r.code == "mechanism_composition_incoherent"),
        "{:?}",
        error.refusals.iter().map(|r| r.code).collect::<Vec<_>>()
    );
}
