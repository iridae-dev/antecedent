//! Transport of fitted mechanisms into a target-population composition against
//! the closed-form moments of the shared fixture, with the refusals each
//! inadmissible carry owes.
//!
//! The hand-computed oracle is written beside the fixture
//! (`transport_fixture/mod.rs`): with `x` refitted in the target and `y`
//! carried, `E[x]=3`, `E[y]=6.5`, `Var x=2`, `Var y=14`, `Cov(x,y)=5`, and
//! `E[y | do(x=1)] - E[y | do(x=0)] = 2`.

mod transport_fixture;

use antecedent_model::compose::{
    CompositionError, ContrastArm, MeanContrast, MechanismForm, MechanismOrigin, MechanismUncertainty,
    ParameterTerm, PosteriorDependence, SelectionNodeDeclaration, UncertaintyMeaning,
};
use antecedent_model::mechanism_transport::{
    DETAIL_DIAGRAM_SOURCE_MISMATCH, DETAIL_DUPLICATE_MECHANISM, DETAIL_FAMILY_UNSUPPORTED,
    DETAIL_ORIGIN_NOT_FITTED, DETAIL_SELECTION_INTO_LATENT, DETAIL_SELECTION_INTO_MECHANISM,
    DETAIL_TARGET_NOT_NATIVE, MechanismRole, MechanismTransportRequest, UNCERTAINTY_SCOPE,
    selection_targets, transport_linear_gaussian,
};
use transport_fixture::{
    Selection, diagram, edges, outcome_invariance, source_x, source_y, target_context, target_x,
    target_y,
};

const TOL: f64 = 1e-12;

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

fn details(error: &CompositionError) -> Vec<&'static str> {
    error.findings().iter().map(|f| f.detail).collect()
}

fn request(selection: Selection) -> MechanismTransportRequest {
    MechanismTransportRequest {
        target: target_context(),
        diagram: diagram(selection),
        source_mechanisms: vec![source_x(), source_y()],
        target_mechanisms: vec![target_x()],
        edges: edges(),
        invariances: vec![outcome_invariance()],
        dependence: PosteriorDependence::Unknown,
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

#[test]
fn an_invariant_fitted_mechanism_is_carried_and_executes_in_the_target() {
    let carried = transport_linear_gaussian(&request(Selection::TreatmentSelected)).unwrap();
    let law = carried.model().law(&[]).unwrap();
    near(law.mean_of("x").unwrap(), 3.0);
    near(law.mean_of("y").unwrap(), 6.5);
    near(law.variance_of("x").unwrap(), 2.0);
    near(law.variance_of("y").unwrap(), 14.0);
    near(law.covariance_of("x", "y").unwrap(), 5.0);
    near(carried.mean_contrast(&effect_of_x()).unwrap(), 2.0);

    // Treatment was refitted in the target; the outcome mechanism was carried.
    assert_eq!(carried.transported_variables(), vec!["y"]);
    let roles: Vec<(&str, MechanismRole)> =
        carried.dispositions().iter().map(|d| (d.variable.as_str(), d.role)).collect();
    assert_eq!(
        roles,
        vec![("x", MechanismRole::NativeInTarget), ("y", MechanismRole::Transported)]
    );
    assert_eq!(carried.dispositions()[1].fitted_population, "source");
    assert_eq!(selection_targets(&diagram(Selection::TreatmentSelected)), vec!["x".to_owned()]);
}

#[test]
fn a_selection_arrow_into_the_mechanism_refuses_the_carry_even_with_an_invariance_claim() {
    // The analyst's invariance claim cannot override the declared diagram.
    let err = transport_linear_gaussian(&request(Selection::OutcomeSelected)).unwrap_err();
    assert_eq!(details(&err), vec![DETAIL_SELECTION_INTO_MECHANISM]);
    let finding = &err.findings()[0];
    assert_eq!(finding.subject, "y");
    assert_eq!(finding.expected.as_deref(), Some("target"));
    assert_eq!(finding.supplied.as_deref(), Some("source"));
    assert!(finding.remedy.contains("refit it in the target population"));

    // Without a natively refitted outcome mechanism the composition does not
    // exist; the diagram's arrow is the stated reason.
    let mut no_claim = request(Selection::OutcomeSelected);
    no_claim.invariances.clear();
    let err = transport_linear_gaussian(&no_claim).unwrap_err();
    assert_eq!(details(&err), vec![DETAIL_SELECTION_INTO_MECHANISM]);
}

#[test]
fn refitting_the_selected_mechanism_in_the_target_makes_the_composition_exist() {
    let mut req = request(Selection::OutcomeSelected);
    req.target_mechanisms.push(target_y());
    req.invariances.clear();
    let model = transport_linear_gaussian(&req).unwrap();
    assert!(model.transported_variables().is_empty());
    // The native slope 1.5 is used; the source slope 2 is not averaged in.
    near(model.mean_contrast(&effect_of_x()).unwrap(), 1.5);
}

#[test]
fn a_native_target_mechanism_takes_precedence_over_an_invariant_carried_one() {
    let mut req = request(Selection::TreatmentSelected);
    req.target_mechanisms.push(target_y());
    let model = transport_linear_gaussian(&req).unwrap();
    assert!(model.transported_variables().is_empty());
    near(model.mean_contrast(&effect_of_x()).unwrap(), 1.5);
}

#[test]
fn a_latent_loading_mechanism_needs_the_explicit_invariance_the_diagram_does_not_cover() {
    let mut req = request(Selection::TreatmentSelected);
    req.invariances.clear();
    let err = transport_linear_gaussian(&req).unwrap_err();
    assert_eq!(
        details(&err),
        vec!["mechanism_composition.population_latent_blocks_selection_admissibility"]
    );

    // A selection node into the latent cause itself is refused with its own
    // literal, even with the invariance declared.
    let mut req = request(Selection::TreatmentSelected);
    req.diagram.selection_nodes.push(SelectionNodeDeclaration {
        name: "S_u".into(),
        points_into: vec!["u".into()],
    });
    let err = transport_linear_gaussian(&req).unwrap_err();
    assert_eq!(details(&err), vec![DETAIL_SELECTION_INTO_LATENT]);
    assert_eq!(err.findings()[0].subject, "u:y");
}

#[test]
fn only_fitted_linear_gaussian_mechanisms_are_carried() {
    let mut declared = request(Selection::TreatmentSelected);
    declared.source_mechanisms[1].provenance.origin = MechanismOrigin::Declared;
    let err = transport_linear_gaussian(&declared).unwrap_err();
    assert!(details(&err).contains(&DETAIL_ORIGIN_NOT_FITTED), "{err:?}");

    let mut external = request(Selection::TreatmentSelected);
    external.source_mechanisms[1].provenance.origin = MechanismOrigin::ExternalBinding;
    let err = transport_linear_gaussian(&external).unwrap_err();
    assert!(details(&err).contains(&DETAIL_ORIGIN_NOT_FITTED), "{err:?}");

    let mut table = request(Selection::TreatmentSelected);
    table.source_mechanisms[1].form = MechanismForm::Tabular { levels: vec!["a".into()], rows: vec![] };
    let err = transport_linear_gaussian(&table).unwrap_err();
    assert!(details(&err).contains(&DETAIL_FAMILY_UNSUPPORTED), "{err:?}");
}

#[test]
fn the_diagram_must_describe_the_populations_it_is_applied_to() {
    let mut wrong_source = request(Selection::TreatmentSelected);
    wrong_source.diagram.source_population = "elsewhere".into();
    let err = transport_linear_gaussian(&wrong_source).unwrap_err();
    assert!(details(&err).contains(&DETAIL_DIAGRAM_SOURCE_MISMATCH), "{err:?}");

    let mut not_native = request(Selection::TreatmentSelected);
    not_native.target_mechanisms[0].context.population = "source".into();
    let err = transport_linear_gaussian(&not_native).unwrap_err();
    assert!(details(&err).contains(&DETAIL_TARGET_NOT_NATIVE), "{err:?}");

    let mut duplicate = request(Selection::TreatmentSelected);
    duplicate.source_mechanisms.push(source_y());
    let err = transport_linear_gaussian(&duplicate).unwrap_err();
    assert!(details(&err).contains(&DETAIL_DUPLICATE_MECHANISM), "{err:?}");
}

#[test]
fn identity_is_order_free_and_moves_with_a_changed_number_at_unchanged_shape() {
    let base = transport_linear_gaussian(&request(Selection::TreatmentSelected)).unwrap();

    let mut reordered = request(Selection::TreatmentSelected);
    reordered.source_mechanisms.reverse();
    let same = transport_linear_gaussian(&reordered).unwrap();
    assert_eq!(base.identity(), same.identity());
    assert_eq!(base.identity_hex().len(), 64);

    // Same structure, one carried coefficient changed.
    let mut changed = request(Selection::TreatmentSelected);
    if let MechanismForm::LinearGaussian { coefficients, .. } =
        &mut changed.source_mechanisms[1].form
    {
        coefficients[0].1 = 2.5;
    }
    let other = transport_linear_gaussian(&changed).unwrap();
    assert_ne!(base.identity(), other.identity());
    near(other.mean_contrast(&effect_of_x()).unwrap(), 2.5);

    // Same numbers, but the outcome mechanism is refitted rather than carried:
    // a different role is a different composition.
    let mut refit = request(Selection::TreatmentSelected);
    let mut native_y = source_y();
    native_y.context.population = "target".into();
    refit.target_mechanisms.push(native_y);
    let refit = transport_linear_gaussian(&refit).unwrap();
    assert_ne!(base.identity(), refit.identity());
}

#[test]
fn a_failed_carry_leaves_the_request_and_a_later_valid_carry_unaffected() {
    let ok = request(Selection::TreatmentSelected);
    let first = transport_linear_gaussian(&ok).unwrap();
    assert!(transport_linear_gaussian(&request(Selection::OutcomeSelected)).is_err());
    let second = transport_linear_gaussian(&ok).unwrap();
    assert_eq!(first.identity(), second.identity());
}

#[test]
fn contrast_uncertainty_is_the_source_fit_variance_and_says_what_it_omits() {
    // y's fitted covariance: intercept 0.04, slope 0.01 (sd 0.1); x's intercept
    // 0.09. The contrast E[y|do(x=1)] - E[y|do(x=0)] is the y slope, so its
    // delta-method variance is exactly the slope variance 0.01, and the
    // treatment mechanism does not enter under intervention.
    let mut req = request(Selection::TreatmentSelected);
    req.source_mechanisms[1].uncertainty = Some(MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("x".into())],
        covariance: vec![0.04, 0.0, 0.0, 0.01],
    });
    req.target_mechanisms[0].uncertainty = Some(MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: vec![ParameterTerm::Intercept],
        covariance: vec![0.09],
    });
    // Unknown dependence between the two fits is a refusal, never independence.
    let carried = transport_linear_gaussian(&req).unwrap();
    let err = carried.mean_contrast_uncertainty(&effect_of_x()).unwrap_err();
    assert!(
        details(&err).contains(&"mechanism_composition.posterior_dependence_unknown"),
        "{err:?}"
    );

    req.dependence = PosteriorDependence::IndependentByDesign {
        justification: "source trial and target registry are disjoint evidence".into(),
    };
    let carried = transport_linear_gaussian(&req).unwrap();
    let u = carried.mean_contrast_uncertainty(&effect_of_x()).unwrap();
    near(u.functional.point, 2.0);
    near(u.functional.variance, 0.01);
    near(u.functional.standard_error, 0.1);
    assert_eq!(u.scope, UNCERTAINTY_SCOPE);
}
