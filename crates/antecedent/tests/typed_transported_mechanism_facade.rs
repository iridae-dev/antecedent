//! Typed (nonlinear / non-Gaussian) mechanisms carried into a target population's composed
//! model through the facade: the typed receipt and identity, the refusal of a shifted
//! mechanism with no target evidence, the flagged wrong invariance, the scoped parameter
//! variance, and the closed interval route.
//!
//! Source: `x ~ N(0, 1)`, count `n | x ~ Poisson(exp(0.2 + 0.3 x))`. Target: `x` unchanged
//! (invariant), the count mechanism shifts to `Poisson(exp(0.5 + 0.2 x + 0.25 x^2))`, so with
//! `x` intervened `E[n | do(x)] = exp(0.5 + 0.2 x + 0.25 x^2)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent::analysis::mechanism_composition::MechanismCompositionRefusal;
use antecedent::analysis::typed_transported_mechanism::{
    fit_typed_mechanism, refit_typed_mechanism_in_target, transport_typed_mechanisms,
    verify_typed_transport, FittedMechanism, MeanSpec, MechanismRole, MechanismSpec,
    ParameterDependence, ResponseFamily, TypedTransportRequest, DETAIL_INVARIANCE_CONTRADICTED,
    DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE, DETAIL_TYPED_IDENTITY_MISMATCH,
    TYPED_TRANSPORT_INFERENCE, TYPED_TRANSPORT_ROUTE, TYPED_TRANSPORT_SUPPORT, UNCERTAINTY_SCOPE,
};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_model::compose::{
    ContrastArm, MeanContrast, MechanismContext, MechanismOrigin, MechanismProvenance,
    SelectionDiagramDeclaration, SelectionNodeDeclaration,
};

type Columns = BTreeMap<String, Vec<f64>>;

fn normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn poisson(lambda: f64, rng: &mut CausalRng) -> f64 {
    let limit = (-lambda).exp();
    let mut p = 1.0;
    let mut k = 0.0;
    loop {
        p *= rng.next_f64();
        if p <= limit {
            return k;
        }
        k += 1.0;
    }
}

fn population(shifted: bool, n: usize, seed: u64) -> Columns {
    let mut rng = CausalRng::from_seed(seed);
    let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let counts: Vec<f64> = x
        .iter()
        .map(|v| {
            let eta = if shifted { 0.5 + 0.2 * v + 0.25 * v * v } else { 0.2 + 0.3 * v };
            poisson(eta.exp(), &mut rng)
        })
        .collect();
    let mut cols = Columns::new();
    cols.insert("x".into(), x);
    cols.insert("n".into(), counts);
    cols
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

fn spec(variable: &str) -> MechanismSpec {
    if variable == "x" {
        MechanismSpec {
            variable: "x".into(),
            parents: vec![],
            mean: MeanSpec::Polynomial { degree: 1, interactions: false },
            family: ResponseFamily::Gaussian,
        }
    } else {
        MechanismSpec {
            variable: "n".into(),
            parents: vec!["x".into()],
            mean: MeanSpec::Polynomial { degree: 2, interactions: false },
            family: ResponseFamily::Poisson,
        }
    }
}

struct Fits {
    source_x: FittedMechanism,
    source_n: FittedMechanism,
    target_x: FittedMechanism,
    target_n: FittedMechanism,
}

fn fits() -> Fits {
    let source = population(false, 5000, 21);
    let target = population(true, 5000, 22);
    let source_x = fit_typed_mechanism(
        &spec("x"),
        &source,
        context("source"),
        provenance("source", "x"),
        None,
    )
    .unwrap();
    let source_n = fit_typed_mechanism(
        &spec("n"),
        &source,
        context("source"),
        provenance("source", "n"),
        None,
    )
    .unwrap();
    let target_x = refit_typed_mechanism_in_target(
        &source_x,
        &target,
        context("target"),
        provenance("target", "x"),
    )
    .unwrap();
    let target_n = refit_typed_mechanism_in_target(
        &source_n,
        &target,
        context("target"),
        provenance("target", "n"),
    )
    .unwrap();
    Fits { source_x, source_n, target_x, target_n }
}

fn request(f: &Fits, shifted: &[&str], targets: Vec<FittedMechanism>) -> TypedTransportRequest {
    TypedTransportRequest {
        target: context("target"),
        diagram: SelectionDiagramDeclaration {
            source_population: "source".into(),
            target_population: "target".into(),
            selection_nodes: vec![SelectionNodeDeclaration {
                name: "S".into(),
                points_into: shifted.iter().map(|s| (*s).to_owned()).collect(),
            }],
        },
        source_mechanisms: vec![f.source_x.clone(), f.source_n.clone()],
        target_mechanisms: targets,
        invariances: vec![],
        dependence: ParameterDependence::IndependentByDesign {
            justification: "disjoint independently generated samples".into(),
        },
        invariance_level: 0.01,
    }
}

fn level(x: f64) -> MeanContrast {
    MeanContrast {
        outcome: "n".into(),
        arms: vec![ContrastArm { weight: 1.0, interventions: vec![("x".into(), x)] }],
    }
}

#[test]
fn a_shifted_count_mechanism_is_refit_and_the_invariant_covariate_is_carried() {
    let f = fits();
    let ctx = ExecutionContext::for_tests(1);
    let composition = transport_typed_mechanisms(
        &request(&f, &["n"], vec![f.target_x.clone(), f.target_n.clone()]),
        &ctx,
    )
    .unwrap();
    let receipt = composition.receipt();
    assert_eq!(receipt.route, TYPED_TRANSPORT_ROUTE);
    assert_eq!(receipt.support, TYPED_TRANSPORT_SUPPORT);
    assert_eq!(receipt.inference, TYPED_TRANSPORT_INFERENCE);
    assert_eq!(receipt.uncertainty_scope, UNCERTAINTY_SCOPE);
    assert_eq!(receipt.selection_targets, vec!["n".to_owned()]);
    assert_eq!(composition.transported_variables(), vec!["x"]);
    let roles: Vec<(&str, MechanismRole)> =
        composition.dispositions().iter().map(|d| (d.variable.as_str(), d.role)).collect();
    assert_eq!(
        roles,
        vec![("n", MechanismRole::NativeInTarget), ("x", MechanismRole::Transported)]
    );
    assert!(composition.dispositions()[1].consistency.is_some_and(|c| !c.rejected));

    for x in [-1.0_f64, 0.0, 1.5] {
        let truth = (0.5 + 0.2 * x + 0.25 * x * x).exp();
        let u = composition.mean_contrast_uncertainty(&level(x), 50_000, &ctx).unwrap();
        let band = 4.5 * (u.standard_error.powi(2) + u.estimate.monte_carlo_se.powi(2)).sqrt();
        assert!((u.estimate.value - truth).abs() <= band, "x={x}: {} vs {truth}", u.estimate.value);
    }
    assert_eq!(composition.interval().unwrap_err().code(), "cell_not_licensed");
}

#[test]
fn a_shifted_mechanism_without_target_evidence_is_refused() {
    let f = fits();
    let ctx = ExecutionContext::for_tests(1);
    let refusal: MechanismCompositionRefusal =
        transport_typed_mechanisms(&request(&f, &["n"], vec![f.target_x.clone()]), &ctx)
            .unwrap_err();
    assert_eq!(refusal.code(), "transport_missing_evidence");
    assert_eq!(refusal.details(), vec![DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE]);
}

#[test]
fn a_wrongly_declared_invariance_is_flagged_by_target_data() {
    let f = fits();
    let ctx = ExecutionContext::for_tests(1);
    // The count mechanism shifted, but the diagram declares no selection node into it.
    let refusal =
        transport_typed_mechanisms(&request(&f, &[], vec![f.target_n.clone()]), &ctx).unwrap_err();
    assert_eq!(refusal.code(), "quantity_semantics_mismatch");
    assert_eq!(refusal.details(), vec![DETAIL_INVARIANCE_CONTRADICTED]);
}

#[test]
fn the_identity_is_verified_only_against_what_it_re_derives() {
    let f = fits();
    let ctx = ExecutionContext::for_tests(1);
    let req = request(&f, &["n"], vec![f.target_x.clone(), f.target_n.clone()]);
    let identity = transport_typed_mechanisms(&req, &ctx).unwrap().identity_hex();
    assert!(verify_typed_transport(&req, &identity, &ctx).is_ok());
    let refusal = verify_typed_transport(&req, &"0".repeat(64), &ctx).unwrap_err();
    assert_eq!(refusal.details(), vec![DETAIL_TYPED_IDENTITY_MISMATCH]);
}
