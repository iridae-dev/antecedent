//! Typed (nonlinear / non-Gaussian) mechanism transport against an independent
//! numpy/scipy target-sample fit, with the refusals each inadmissible carry owes.
//!
//! The fixture `conformance/transport/typed_mechanism_shift` holds a source and a
//! target sample in which the income and price mechanisms are invariant and six
//! outcome mechanisms (Gaussian, Poisson, Bernoulli, gamma, log-normal and
//! heteroscedastic Gaussian, each with a quadratic price effect) shift. Its
//! reference script fits the target sample independently and evaluates the target
//! interventional mean by quadrature over the invariant income law.

use std::collections::BTreeMap;

use antecedent_core::CausalRng;
use antecedent_core::ExecutionContext;
use antecedent_kernels::standard_normal;
use antecedent_model::compose::{
    CompositionError, ContrastArm, MeanContrast, MechanismContext, MechanismOrigin,
    MechanismProvenance, SelectionDiagramDeclaration, SelectionNodeDeclaration,
};
use antecedent_model::mechanism_expr::MechanismExpression;
use antecedent_model::typed_mechanism_transport::{
    fit_mechanism, refit_in_target, transport_typed, typed_refusal_code, FittedMechanism, MeanSpec,
    MechanismRole, MechanismSpec, ParameterDependence, ResponseFamily, TypedFitError,
    TypedTransportRequest, DETAIL_DEPENDENCE_UNKNOWN, DETAIL_DESIGN_NOT_SHARED,
    DETAIL_INVARIANCE_CONTRADICTED, DETAIL_SELECTION_TARGET_UNKNOWN,
    DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE,
};

const OUTCOMES: [&str; 6] = ["demand", "orders", "buys", "spend", "basket", "noise_h"];
const DRAWS: usize = 200_000;

type Columns = BTreeMap<String, Vec<f64>>;

struct Fixture {
    source: Columns,
    target: Columns,
    expected: serde_json::Value,
}

fn columns(value: &serde_json::Value) -> Columns {
    value
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (k.clone(), v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())
        })
        .collect()
}

fn fixture() -> Fixture {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/transport/typed_mechanism_shift/data.json"
    ))
    .unwrap();
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/transport/typed_mechanism_shift/expected.json"
    ))
    .unwrap();
    Fixture { source: columns(&data["source"]), target: columns(&data["target"]), expected }
}

fn context(population: &str) -> MechanismContext {
    MechanismContext { population: population.into(), regime: "observational".into() }
}

fn provenance(population: &str, variable: &str) -> MechanismProvenance {
    MechanismProvenance {
        origin: MechanismOrigin::Fitted,
        artifact_id: format!("fit:{population}:{variable}"),
        evidence_id: format!("evidence:{population}"),
    }
}

fn family_of(name: &str) -> ResponseFamily {
    match name {
        "demand" => ResponseFamily::Gaussian,
        "orders" => ResponseFamily::Poisson,
        "buys" => ResponseFamily::Bernoulli,
        "spend" => ResponseFamily::Gamma,
        "basket" => ResponseFamily::LogNormal,
        "noise_h" => ResponseFamily::HeteroscedasticGaussian,
        other => panic!("{other}"),
    }
}

fn spec_of(variable: &str) -> MechanismSpec {
    match variable {
        "income" => MechanismSpec {
            variable: "income".into(),
            parents: vec![],
            mean: MeanSpec::Polynomial { degree: 1, interactions: false },
            family: ResponseFamily::Gaussian,
        },
        "price" => MechanismSpec {
            variable: "price".into(),
            parents: vec!["income".into()],
            mean: MeanSpec::Polynomial { degree: 1, interactions: false },
            family: ResponseFamily::Gaussian,
        },
        outcome => MechanismSpec {
            variable: outcome.into(),
            parents: vec!["price".into(), "income".into()],
            mean: MeanSpec::Polynomial { degree: 2, interactions: false },
            family: family_of(outcome),
        },
    }
}

fn fit_source(variable: &str, f: &Fixture) -> FittedMechanism {
    fit_mechanism(
        &spec_of(variable),
        &f.source,
        context("source"),
        provenance("source", variable),
        None,
    )
    .unwrap()
}

fn refit_target(source: &FittedMechanism, columns: &Columns) -> FittedMechanism {
    refit_in_target(source, columns, context("target"), provenance("target", source.variable()))
        .unwrap()
}

fn diagram(into: &[&str]) -> SelectionDiagramDeclaration {
    SelectionDiagramDeclaration {
        source_population: "source".into(),
        target_population: "target".into(),
        selection_nodes: vec![SelectionNodeDeclaration {
            name: "S".into(),
            points_into: into.iter().map(|s| (*s).to_owned()).collect(),
        }],
    }
}

fn independent() -> ParameterDependence {
    ParameterDependence::IndependentByDesign {
        justification: "source and target samples are disjoint and independently generated".into(),
    }
}

struct World {
    fixture: Fixture,
    sources: BTreeMap<String, FittedMechanism>,
}

fn world() -> World {
    let fixture = fixture();
    let mut sources = BTreeMap::new();
    for v in ["income", "price"].into_iter().chain(OUTCOMES) {
        sources.insert(v.to_owned(), fit_source(v, &fixture));
    }
    World { fixture, sources }
}

impl World {
    fn request(&self, shifted: &[&str], target_variables: &[&str]) -> TypedTransportRequest {
        TypedTransportRequest {
            target: context("target"),
            diagram: diagram(shifted),
            source_mechanisms: self.sources.values().cloned().collect(),
            target_mechanisms: target_variables
                .iter()
                .map(|v| refit_target(&self.sources[*v], &self.fixture.target))
                .collect(),
            invariances: vec![],
            dependence: independent(),
            invariance_level: 0.01,
        }
    }

    fn full(&self) -> TypedTransportRequest {
        let mut targets: Vec<&str> = vec!["income", "price"];
        targets.extend(OUTCOMES);
        self.request(&OUTCOMES, &targets)
    }
}

fn level(outcome: &str, price: f64) -> MeanContrast {
    MeanContrast {
        outcome: outcome.into(),
        arms: vec![ContrastArm { weight: 1.0, interventions: vec![("price".into(), price)] }],
    }
}

fn details(error: &CompositionError) -> Vec<&'static str> {
    error.findings().iter().map(|f| f.detail).collect()
}

fn array(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn transported_response_matches_the_independent_target_fit_for_every_family() {
    let w = world();
    let model = transport_typed(&w.full()).unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let grid = array(&w.fixture.expected["grid"]);
    for outcome in OUTCOMES {
        let o = &w.fixture.expected["outcomes"][outcome];
        let reference = array(&o["reference"]);
        let truth = array(&o["truth"]);
        let reference_se = array(&o["reference_se"]);
        for (i, p) in grid.iter().enumerate() {
            let est = model.mean_contrast(&level(outcome, *p), DRAWS, &ctx).unwrap();
            assert!(
                (est.value - reference[i]).abs() <= 6.0 * est.monte_carlo_se + 1e-6,
                "{outcome} p={p}: {} vs reference {} (mc se {})",
                est.value,
                reference[i],
                est.monte_carlo_se
            );
            let band = 4.5 * (reference_se[i].powi(2) + est.monte_carlo_se.powi(2)).sqrt();
            assert!(
                (est.value - truth[i]).abs() <= band,
                "{outcome} p={p}: {} vs truth {} (band {band})",
                est.value,
                truth[i]
            );
        }
    }
}

#[test]
fn the_nonlinear_demand_effect_is_the_target_effect_not_the_source_effect() {
    let w = world();
    let model = transport_typed(&w.full()).unwrap();
    let ctx = ExecutionContext::for_tests(11);
    let effect = MeanContrast {
        outcome: "demand".into(),
        arms: vec![
            ContrastArm { weight: 1.0, interventions: vec![("price".into(), 2.0)] },
            ContrastArm { weight: -1.0, interventions: vec![("price".into(), 1.0)] },
        ],
    };
    // Target truth: -0.8 - 0.5 * (4 - 1) = -2.3; the source effect is -1.0.
    let est = model.mean_contrast(&effect, DRAWS, &ctx).unwrap();
    assert!((est.value + 2.3).abs() < 0.12, "{}", est.value);
    assert!((est.value + 1.0).abs() > 1.0, "{}", est.value);
}

#[test]
fn parameter_uncertainty_agrees_with_the_independent_delta_method() {
    let w = world();
    let model = transport_typed(&w.full()).unwrap();
    let ctx = ExecutionContext::for_tests(13);
    for outcome in OUTCOMES {
        let o = &w.fixture.expected["outcomes"][outcome];
        let reference_se = array(&o["reference_se"]);
        let u = model.mean_contrast_uncertainty(&level(outcome, 2.0), 300_000, &ctx).unwrap();
        let relative = (u.standard_error - reference_se[2]).abs() / reference_se[2];
        assert!(relative < 0.08, "{outcome}: {} vs {}", u.standard_error, reference_se[2]);
        assert_eq!(u.scope, antecedent_model::mechanism_transport::UNCERTAINTY_SCOPE);
        assert_eq!(u.dependence, "independent_by_design");
        // Monte Carlo error is reported apart from the parameter variance.
        assert!(u.estimate.monte_carlo_se.is_finite());
    }
}

#[test]
fn dispositions_say_which_mechanisms_were_carried_and_which_refitted() {
    let w = world();
    let model = transport_typed(&w.full()).unwrap();
    assert_eq!(model.transported_variables(), vec!["income", "price"]);
    for d in model.dispositions() {
        if OUTCOMES.contains(&d.variable.as_str()) {
            assert_eq!(d.role, MechanismRole::NativeInTarget);
            assert_eq!(d.covariance_source, "target_fit");
            assert_eq!(d.fitted_population, "target");
            assert!(d.consistency.is_none());
        } else {
            assert_eq!(d.role, MechanismRole::Transported);
            assert_eq!(d.covariance_source, "source_fit");
            assert_eq!(d.fitted_population, "source");
            let check = d.consistency.expect("target evidence was supplied");
            assert!(!check.rejected);
            let expected = &w.fixture.expected["invariance"][d.variable.as_str()];
            let statistic = expected["statistic"].as_f64().unwrap();
            let p = expected["p_value"].as_f64().unwrap();
            assert!((check.statistic - statistic).abs() <= 1e-6 * statistic.max(1.0));
            assert!((check.p_value - p).abs() <= 1e-6);
            assert_eq!(check.degrees_of_freedom, if d.variable == "income" { 2 } else { 3 });
        }
    }
}

#[test]
fn a_shifted_mechanism_without_target_evidence_is_refused_never_carried() {
    let w = world();
    // The selection node points into demand, but only price and income have target fits.
    let request = w.request(&["demand"], &["income", "price"]);
    let error = transport_typed(&request).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE]);
    assert_eq!(error.findings()[0].subject, "demand");
    assert_eq!(
        typed_refusal_code(DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE),
        Some("transport_missing_evidence")
    );
    // The source fit of the variable is present and is not substituted.
    assert!(request.source_mechanisms.iter().any(|m| m.variable() == "demand"));
}

#[test]
fn a_selection_arrow_into_an_unknown_variable_is_refused() {
    let w = world();
    let request = w.request(&["revenue"], &["income", "price"]);
    let error = transport_typed(&request).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_SELECTION_TARGET_UNKNOWN]);
}

#[test]
fn a_wrongly_declared_invariance_is_flagged_by_the_target_fit() {
    let w = world();
    // The price mechanism changed in the target (slope on income 0.5 -> 0.9), but the
    // diagram declares no selection node into it. Target data refute that.
    let mut target = w.fixture.target.clone();
    let shifted = target.remove("price_shifted").unwrap();
    target.insert("price".into(), shifted);
    let refit = refit_target(&w.sources["price"], &target);
    let mut request = w.request(&OUTCOMES, &["income"]);
    request.target_mechanisms.push(refit);
    for o in OUTCOMES {
        request.target_mechanisms.push(refit_target(&w.sources[o], &w.fixture.target));
    }
    let error = transport_typed(&request).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_INVARIANCE_CONTRADICTED]);
    assert_eq!(error.findings()[0].subject, "price");
    assert_eq!(
        typed_refusal_code(DETAIL_INVARIANCE_CONTRADICTED),
        Some("quantity_semantics_mismatch")
    );

    // Declaring the arrow and refitting is the licensed route.
    let mut licensed = request.clone();
    licensed.diagram =
        diagram(&["demand", "orders", "buys", "spend", "basket", "noise_h", "price"]);
    let model = transport_typed(&licensed).unwrap();
    let price = model.dispositions().iter().find(|d| d.variable == "price").unwrap();
    assert_eq!(price.role, MechanismRole::NativeInTarget);
}

#[test]
fn an_untested_invariance_is_recorded_as_such() {
    let w = world();
    // No target fit for price: the declaration cannot be tested, and the disposition says so.
    let mut targets: Vec<&str> = vec!["income"];
    targets.extend(OUTCOMES);
    let model = transport_typed(&w.request(&OUTCOMES, &targets)).unwrap();
    let price = model.dispositions().iter().find(|d| d.variable == "price").unwrap();
    assert_eq!(price.role, MechanismRole::Transported);
    assert!(price.consistency.is_none());
}

#[test]
fn a_target_fit_on_another_design_is_not_comparable() {
    let w = world();
    let independent_fit = fit_mechanism(
        &spec_of("price"),
        &w.fixture.target,
        context("target"),
        provenance("target", "price"),
        None,
    )
    .unwrap();
    let mut request = w.request(&OUTCOMES, &["income"]);
    request.target_mechanisms.push(independent_fit);
    for o in OUTCOMES {
        request.target_mechanisms.push(refit_target(&w.sources[o], &w.fixture.target));
    }
    let error = transport_typed(&request).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_DESIGN_NOT_SHARED]);
}

#[test]
fn unknown_parameter_dependence_refuses_a_multi_mechanism_uncertainty_claim() {
    let w = world();
    let mut request = w.full();
    request.dependence = ParameterDependence::Unknown;
    let model = transport_typed(&request).unwrap();
    let ctx = ExecutionContext::for_tests(3);
    // The point claim is unaffected.
    assert!(model.mean_contrast(&level("demand", 1.0), 10_000, &ctx).is_ok());
    let error = model.mean_contrast_uncertainty(&level("demand", 1.0), 10_000, &ctx).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_DEPENDENCE_UNKNOWN]);
    assert_eq!(
        typed_refusal_code(DETAIL_DEPENDENCE_UNKNOWN),
        Some("uncertainty_representation_insufficient")
    );
    // A claim that involves one mechanism only (income intervened away) needs no dependence.
    let single = MeanContrast {
        outcome: "demand".into(),
        arms: vec![ContrastArm {
            weight: 1.0,
            interventions: vec![("price".into(), 1.0), ("income".into(), 0.0)],
        }],
    };
    assert!(model.mean_contrast_uncertainty(&single, 10_000, &ctx).is_ok());
}

#[test]
fn joint_dependence_changes_the_variance_it_is_given() {
    let w = world();
    let ctx = ExecutionContext::for_tests(5);
    let base = transport_typed(&w.full()).unwrap();
    let contrast = level("demand", 2.0);
    let independent_variance =
        base.mean_contrast_uncertainty(&contrast, 100_000, &ctx).unwrap().variance;

    // A correlation between the income-mean and demand-intercept errors changes the
    // variance; the demand intercept is itself correlated with the squared-term
    // coefficients, so the correlation must stay inside the positive semidefinite range.
    let income = base.mechanisms().iter().find(|m| m.variable() == "income").unwrap();
    let demand = base.mechanisms().iter().find(|m| m.variable() == "demand").unwrap();
    let (ki, kd) = (income.parameters().len(), demand.parameters().len());
    let d = ki + kd;
    let mut joint = vec![0.0; d * d];
    for a in 0..ki {
        for b in 0..ki {
            joint[a * d + b] = income.covariance()[a * ki + b];
        }
    }
    for a in 0..kd {
        for b in 0..kd {
            joint[(ki + a) * d + ki + b] = demand.covariance()[a * kd + b];
        }
    }
    let mut request = w.full();
    request.dependence = ParameterDependence::Joint {
        variables: vec!["income".into(), "demand".into()],
        covariance: joint.clone(),
    };
    let block_diagonal = transport_typed(&request)
        .unwrap()
        .mean_contrast_uncertainty(&contrast, 100_000, &ctx)
        .unwrap()
        .variance;
    assert!((block_diagonal - independent_variance).abs() <= 1e-9 * independent_variance.max(1.0));

    // Couple the income mean (index 0) and the demand intercept (index ki).
    let rho = 0.3;
    let coupling = rho * (joint[0] * joint[ki * d + ki]).sqrt();
    joint[ki] = coupling;
    joint[ki * d] = coupling;
    request.dependence = ParameterDependence::Joint {
        variables: vec!["income".into(), "demand".into()],
        covariance: joint,
    };
    let coupled = transport_typed(&request)
        .unwrap()
        .mean_contrast_uncertainty(&contrast, 100_000, &ctx)
        .unwrap()
        .variance;
    assert!((coupled - independent_variance).abs() > 1e-12);
}

#[test]
fn the_transport_identity_is_order_free_and_moves_with_a_changed_number() {
    let w = world();
    let first = transport_typed(&w.full()).unwrap().identity_hex();
    let mut reordered = w.full();
    reordered.target_mechanisms.reverse();
    reordered.source_mechanisms.reverse();
    assert_eq!(transport_typed(&reordered).unwrap().identity_hex(), first);

    let mut other = w.full();
    other.dependence = ParameterDependence::IndependentByDesign { justification: "another".into() };
    assert_ne!(transport_typed(&other).unwrap().identity_hex(), first);

    let mut level_changed = w.full();
    level_changed.invariance_level = 0.02;
    assert_ne!(transport_typed(&level_changed).unwrap().identity_hex(), first);
}

#[test]
fn populations_must_match_the_diagram() {
    let w = world();
    let mut request = w.full();
    request.diagram.target_population = "elsewhere".into();
    let error = transport_typed(&request).unwrap_err();
    assert!(details(&error)
        .contains(&antecedent_model::mechanism_transport::DETAIL_DIAGRAM_TARGET_MISMATCH));

    let mut request = w.full();
    request.diagram.source_population = "elsewhere".into();
    let error = transport_typed(&request).unwrap_err();
    assert!(details(&error)
        .contains(&antecedent_model::mechanism_transport::DETAIL_DIAGRAM_SOURCE_MISMATCH));

    let mut request = w.full();
    request.invariance_level = 0.7;
    assert!(transport_typed(&request).is_err());
}

#[test]
fn responses_outside_the_family_support_and_separation_are_refused_at_fit() {
    let mut cols: Columns = BTreeMap::new();
    cols.insert("x".into(), (0..40).map(|i| f64::from(i) / 10.0).collect());
    cols.insert("count".into(), (0..40).map(|i| if i == 3 { -1.0 } else { 1.0 }).collect());
    let spec = |family| MechanismSpec {
        variable: "count".into(),
        parents: vec!["x".into()],
        mean: MeanSpec::Polynomial { degree: 1, interactions: false },
        family,
    };
    let fit = |family, cols: &Columns| {
        fit_mechanism(&spec(family), cols, context("p"), provenance("p", "count"), None)
    };
    assert!(matches!(fit(ResponseFamily::Poisson, &cols), Err(TypedFitError::Support { .. })));
    assert!(matches!(fit(ResponseFamily::Gamma, &cols), Err(TypedFitError::Support { .. })));
    assert!(matches!(fit(ResponseFamily::Bernoulli, &cols), Err(TypedFitError::Support { .. })));

    // Complete separation of a Bernoulli response is a non-convergence refusal.
    cols.insert("count".into(), (0..40).map(|i| if i >= 20 { 1.0 } else { 0.0 }).collect());
    let error = fit(ResponseFamily::Bernoulli, &cols).unwrap_err();
    assert!(matches!(error, TypedFitError::NotConverged(_)));
    assert_eq!(error.reason_code(), "mechanism_fit_not_converged");
}

fn tanh_mean() -> MeanSpec {
    // y = p0 + p1 * tanh(p2 * x)
    MeanSpec::Expression {
        expression: MechanismExpression::Add {
            terms: vec![
                MechanismExpression::Parameter { parameter: 0 },
                MechanismExpression::Multiply {
                    factors: vec![
                        MechanismExpression::Parameter { parameter: 1 },
                        MechanismExpression::Tanh {
                            argument: Box::new(MechanismExpression::Multiply {
                                factors: vec![
                                    MechanismExpression::Parameter { parameter: 2 },
                                    MechanismExpression::Variable { variable: 0 },
                                ],
                            }),
                        },
                    ],
                },
            ],
        },
        start: vec![0.0, 1.0, 1.0],
    }
}

#[test]
fn an_expression_mean_is_refit_in_the_target_and_transported() {
    // Source: y = 0.5 + 2 tanh(1.5 x); the target saturates earlier and higher:
    // y = 0.5 + 3 tanh(1.5 x). Both with Gaussian noise sd 0.3. x ~ U(-2, 2).
    let mut rng = CausalRng::from_seed(20_261_027);
    let mut sample = |amplitude: f64, n: usize| {
        let x: Vec<f64> = (0..n).map(|_| -2.0 + 4.0 * rng.next_f64()).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|v| 0.5 + amplitude * (1.5 * v).tanh() + 0.3 * standard_normal(&mut rng))
            .collect();
        let mut cols: Columns = BTreeMap::new();
        cols.insert("x".into(), x);
        cols.insert("y".into(), y);
        cols
    };
    let source = sample(2.0, 4000);
    let target = sample(3.0, 4000);
    let y_spec = MechanismSpec {
        variable: "y".into(),
        parents: vec!["x".into()],
        mean: tanh_mean(),
        family: ResponseFamily::Gaussian,
    };
    let x_spec = spec_of("income");
    let x_spec = MechanismSpec { variable: "x".into(), ..x_spec };
    let source_y =
        fit_mechanism(&y_spec, &source, context("source"), provenance("source", "y"), None)
            .unwrap();
    let source_x =
        fit_mechanism(&x_spec, &source, context("source"), provenance("source", "x"), None)
            .unwrap();
    let target_y =
        refit_in_target(&source_y, &target, context("target"), provenance("target", "y")).unwrap();

    // Parameter recovery within the reported fit covariance.
    let truth = [0.5, 3.0, 1.5];
    let k = target_y.parameters().len();
    for (j, t) in truth.iter().enumerate() {
        let sd = target_y.covariance()[j * k + j].sqrt();
        assert!(
            (target_y.parameters()[j] - t).abs() < 4.5 * sd,
            "parameter {j}: {}",
            target_y.parameters()[j]
        );
    }

    let request = TypedTransportRequest {
        target: context("target"),
        diagram: SelectionDiagramDeclaration {
            source_population: "source".into(),
            target_population: "target".into(),
            selection_nodes: vec![SelectionNodeDeclaration {
                name: "S".into(),
                points_into: vec!["y".into()],
            }],
        },
        source_mechanisms: vec![source_x, source_y.clone()],
        target_mechanisms: vec![target_y],
        invariances: vec![],
        dependence: ParameterDependence::Unknown,
        invariance_level: 0.01,
    };
    let model = transport_typed(&request).unwrap();
    let ctx = ExecutionContext::for_tests(17);
    let effect = MeanContrast {
        outcome: "y".into(),
        arms: vec![
            ContrastArm { weight: 1.0, interventions: vec![("x".into(), 1.0)] },
            ContrastArm { weight: -1.0, interventions: vec![("x".into(), 0.0)] },
        ],
    };
    let u = model.mean_contrast_uncertainty(&effect, 20_000, &ctx).unwrap();
    let truth_effect = 3.0 * (1.5_f64).tanh();
    let source_effect = 2.0 * (1.5_f64).tanh();
    assert!(
        (u.estimate.value - truth_effect).abs() <= 4.5 * u.standard_error.max(1e-3),
        "{} vs {truth_effect} (se {})",
        u.estimate.value,
        u.standard_error
    );
    assert!((u.estimate.value - source_effect).abs() > 0.5);
    assert_eq!(u.dependence, "unknown_single_mechanism");
}

#[test]
fn a_shifted_expression_mechanism_without_target_data_is_refused() {
    let mut rng = CausalRng::from_seed(5);
    let x: Vec<f64> = (0..400).map(|_| -2.0 + 4.0 * rng.next_f64()).collect();
    let y: Vec<f64> =
        x.iter().map(|v| 0.5 + 2.0 * (1.5 * v).tanh() + 0.3 * standard_normal(&mut rng)).collect();
    let mut cols: Columns = BTreeMap::new();
    cols.insert("x".into(), x);
    cols.insert("y".into(), y);
    let y_spec = MechanismSpec {
        variable: "y".into(),
        parents: vec!["x".into()],
        mean: tanh_mean(),
        family: ResponseFamily::Gaussian,
    };
    let source_y =
        fit_mechanism(&y_spec, &cols, context("source"), provenance("source", "y"), None).unwrap();
    let x_spec = MechanismSpec { variable: "x".into(), ..spec_of("income") };
    let source_x =
        fit_mechanism(&x_spec, &cols, context("source"), provenance("source", "x"), None).unwrap();
    let request = TypedTransportRequest {
        target: context("target"),
        diagram: SelectionDiagramDeclaration {
            source_population: "source".into(),
            target_population: "target".into(),
            selection_nodes: vec![SelectionNodeDeclaration {
                name: "S".into(),
                points_into: vec!["y".into()],
            }],
        },
        source_mechanisms: vec![source_x, source_y],
        target_mechanisms: vec![],
        invariances: vec![],
        dependence: ParameterDependence::Unknown,
        invariance_level: 0.01,
    };
    let error = transport_typed(&request).unwrap_err();
    assert_eq!(details(&error), vec![DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE]);
}
