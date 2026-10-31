//! Facade integration of composed independently fitted mechanisms: declared
//! semantic coordinates and numbers in, one executable joint model out, with
//! the refusals individually valid mechanisms owe, an artifact a consumer
//! replays, and a refresh that keeps the plan and recomputes the numbers.
//!
//! Hand-computed oracle (linear-Gaussian chain), with
//!   X = 1 + 2 `e_X`
//!   M = 0.5 + 2 X + 1 `e_M`
//!   Y = -1 + 0.5 X + 3 M + 0.5 `e_Y`
//! so that Y = 0.5 + 6.5 X + 3 `e_M` + `e_Y`:
//!   E[X]=1, E[M]=2.5, E[Y]=7; Var X=4, Var M=17, Var Y=178.25
//!   Cov(X,M)=8, Cov(X,Y)=26, Cov(M,Y)=55
//!   do(X=2): E[M]=4.5, E[Y]=13.5, Var M=1, Var Y=9.25, Cov(M,Y)=3
//!   do(M=1): E[Y]=2.5, Var Y=1.25
//!   E[Y|do(X=1)] - E[Y|do(X=0)] = 6.5
//! Finite chain (X -> M -> Y, X -> Y, binary): P(Y=1)=0.396,
//!   P(Y=1|do(X=1))=0.72, P(Y=1|do(X=0))=0.18, P(Y=1|do(M=1))=0.66.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::mechanism_composition::CompositionQueryWire;
use antecedent::{
    DependenceInput, MechanismCompositionRefusal, MechanismCompositionRequest, MechanismInput,
    MechanismRefit, MechanismSource, StudyBuilder, consume_mechanism_composition_artifact,
};
use antecedent_contract::{
    AdmissionBasis, AdmissionCoordinate, LatentCauseDeclaration, MechanismDeclaration,
    MechanismOriginKind, QuantityRole, ScientificQuantity, TransportAdmission,
};
use antecedent_core::ExecutionContext;
use antecedent_io::mechanism_composition_artifact::{
    MechanismCompositionArtifactWire, mechanism_composition_identity,
};
use antecedent_model::compile::MechanismSlot;
use antecedent_model::compose::{
    ContrastArm, CrossCovarianceBlock, Domain, MeanContrast, MechanismFamily, MechanismUncertainty,
    ParameterId, ParameterTerm, TabularRow, UncertaintyMeaning,
};

const TOL: f64 = 1e-12;

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

fn quantity(variable: &str, units: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: format!("label of {variable}"),
        role: role(variable),
        units: units.into(),
        population_id: population.into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn declaration(variable: &str, parents: &[&str], units: &str) -> MechanismDeclaration {
    MechanismDeclaration {
        output: quantity(variable, units, "pop"),
        parents: parents.iter().map(|p| quantity(p, units, "pop")).collect(),
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
        // The mapping a fitted registry slot takes: coefficients by position.
        let slot = MechanismSlot::LinearGaussian {
            intercept,
            coeffs: Arc::from(parents.iter().map(|(_, c)| *c).collect::<Vec<f64>>()),
            sigma,
        };
        MechanismSource::from_fitted_slot(&slot, &names).unwrap()
    };
    MechanismInput {
        declaration: declaration(variable, &names, "u"),
        source,
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

fn chain() -> Vec<MechanismInput> {
    vec![
        linear("X", 1.0, 2.0, &[], false),
        linear("M", 0.5, 1.0, &[("X", 2.0)], false),
        linear("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)], false),
    ]
}

fn details(refusal: &MechanismCompositionRefusal) -> Vec<&str> {
    refusal.details()
}

fn ate_at(x: f64) -> MeanContrast {
    MeanContrast {
        outcome: "Y".into(),
        arms: vec![ContrastArm { weight: 1.0, interventions: vec![("X".into(), x)] }],
    }
}

fn marginal(terms: Vec<ParameterTerm>, covariance: Vec<f64>) -> MechanismUncertainty {
    MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: terms,
        covariance,
    }
}

/// Marginal parameter laws of the chain: M (`a_M`, `b_MX`); X (`a_X`); Y listed out
/// of canonical order (`a_Y`, `b_YX`, `b_YM`).
fn uncertain_chain() -> Vec<MechanismInput> {
    let mut mechanisms = chain();
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

fn cross_blocks() -> Vec<CrossCovarianceBlock> {
    let m_terms = vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())];
    let x_terms = vec![ParameterTerm::Intercept];
    let y_terms = vec![
        ParameterTerm::Intercept,
        ParameterTerm::Coefficient("M".into()),
        ParameterTerm::Coefficient("X".into()),
    ];
    let block =
        |first: &str, second: &str, fp: &[ParameterTerm], sp: &[ParameterTerm], v: Vec<f64>| {
            CrossCovarianceBlock {
                first: first.into(),
                second: second.into(),
                first_parameters: fp.to_vec(),
                second_parameters: sp.to_vec(),
                values: v,
            }
        };
    vec![
        // cov(b_MX, b_YM) = 0.002 adds 2 * 6 * 4.5 * 0.002 = 0.108; every other
        // pair is an explicit zero block (a declared independence).
        block("M", "Y", &m_terms, &y_terms, vec![0.0, 0.0, 0.0, 0.0, 0.002, 0.0]),
        block("M", "X", &m_terms, &x_terms, vec![0.0, 0.0]),
        block("X", "Y", &x_terms, &y_terms, vec![0.0, 0.0, 0.0]),
    ]
}

// ---------------------------------------------------------------------------
// Closed-form execution
// ---------------------------------------------------------------------------

#[test]
fn fitted_slots_compose_into_the_hand_computed_scm() {
    let prepared = StudyBuilder::mechanism_composition(request_of(chain()), &ctx()).unwrap();
    assert_eq!(prepared.family(), MechanismFamily::LinearGaussian);
    assert_eq!(prepared.variables(), ["M", "X", "Y"]);
    assert!(prepared.declared_mechanisms().is_empty());

    let obs = prepared.intervention_law(&[]).unwrap();
    near(obs.mean_of("X").unwrap(), 1.0);
    near(obs.mean_of("M").unwrap(), 2.5);
    near(obs.mean_of("Y").unwrap(), 7.0);
    near(obs.variance_of("X").unwrap(), 4.0);
    near(obs.variance_of("M").unwrap(), 17.0);
    near(obs.variance_of("Y").unwrap(), 178.25);
    near(obs.covariance_of("X", "M").unwrap(), 8.0);
    near(obs.covariance_of("X", "Y").unwrap(), 26.0);
    near(obs.covariance_of("M", "Y").unwrap(), 55.0);

    let do_x = prepared.intervention_law(&[("X", 2.0)]).unwrap();
    near(do_x.mean_of("M").unwrap(), 4.5);
    near(do_x.mean_of("Y").unwrap(), 13.5);
    near(do_x.variance_of("M").unwrap(), 1.0);
    near(do_x.variance_of("Y").unwrap(), 9.25);
    near(do_x.covariance_of("M", "Y").unwrap(), 3.0);
    let do_m = prepared.intervention_law(&[("M", 1.0)]).unwrap();
    near(do_m.mean_of("Y").unwrap(), 2.5);
    near(do_m.variance_of("Y").unwrap(), 1.25);

    let contrast = MeanContrast {
        outcome: "Y".into(),
        arms: vec![
            ContrastArm { weight: 1.0, interventions: vec![("X".into(), 1.0)] },
            ContrastArm { weight: -1.0, interventions: vec![("X".into(), 0.0)] },
        ],
    };
    near(prepared.mean_contrast(&contrast).unwrap(), 6.5);
}

#[test]
fn listing_order_does_not_change_the_plan_or_the_law_and_sampling_is_deterministic() {
    let forward = StudyBuilder::mechanism_composition(request_of(chain()), &ctx()).unwrap();
    let mut reversed = chain();
    reversed.reverse();
    let mut request = request_of(reversed);
    request.edges.reverse();
    let backward = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    assert_eq!(forward.plan_identity_hex(), backward.plan_identity_hex());
    assert_eq!(forward.plan_identity_version(), 2);
    assert_eq!(
        forward.intervention_law(&[("X", 2.0)]).unwrap(),
        backward.intervention_law(&[("X", 2.0)]).unwrap()
    );

    let n = 200_000;
    let table = forward.sample(&[("X", 2.0)], n, &ExecutionContext::for_tests(20_260_101)).unwrap();
    let y = &table.columns[table.variables.iter().position(|v| v == "Y").unwrap()];
    let mean = y.iter().sum::<f64>() / n as f64;
    let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    // SE of the mean sqrt(9.25/200000) = 0.0068: 6 SE; relative SE of the
    // variance sqrt(2/n) = 0.0032: 6x.
    assert!((mean - 13.5).abs() < 0.05, "mean {mean}");
    assert!((var / 9.25 - 1.0).abs() < 0.02, "var {var}");
    let again = forward.sample(&[("X", 2.0)], n, &ExecutionContext::for_tests(20_260_101)).unwrap();
    assert_eq!(table, again);
}

#[test]
fn declared_mechanisms_are_labelled_and_need_acknowledgement() {
    let mut mechanisms = chain();
    mechanisms[1] = linear("M", 0.5, 1.0, &[("X", 2.0)], true);
    let refusal =
        StudyBuilder::mechanism_composition(request_of(mechanisms.clone()), &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "mechanism_composition_incoherent");
    assert_eq!(
        details(&refusal),
        vec!["mechanism_composition.provenance_declared_mechanism_unacknowledged"]
    );
    // The label comes from the source of the numbers, not from the declaration.
    mechanisms[1].declaration.origin = MechanismOriginKind::Fitted;
    let mut request = request_of(mechanisms);
    request.admit_declared_mechanisms = true;
    let prepared = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    assert_eq!(prepared.declared_mechanisms(), ["M"]);
    near(prepared.intervention_law(&[("X", 2.0)]).unwrap().mean_of("Y").unwrap(), 13.5);
}

#[test]
fn a_registry_slot_that_is_not_plain_linear_gaussian_does_not_compose() {
    let slot = MechanismSlot::Constant { value: 1.0 };
    let refusal = MechanismSource::from_fitted_slot(&slot, &[]).unwrap_err();
    assert_eq!(refusal.details(), vec!["mechanism_composition.slot_not_linear_gaussian"]);
    let slot = MechanismSlot::LinearGaussian {
        intercept: 0.0,
        coeffs: Arc::from(vec![1.0, 2.0]),
        sigma: 1.0,
    };
    let refusal = MechanismSource::from_fitted_slot(&slot, &["X"]).unwrap_err();
    assert_eq!(refusal.details(), vec!["mechanism_composition.shape_coefficient_parent_mismatch"]);
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn individually_valid_mechanisms_that_cannot_compose_are_refused_with_structure() {
    // Units: M is produced in mmHg but Y was fitted on M in kPa.
    let mut mechanisms = chain();
    mechanisms[1].declaration.output.units = "mmHg".into();
    mechanisms[2].declaration.parents[1].units = "kPa".into();
    let refusal = StudyBuilder::mechanism_composition(request_of(mechanisms), &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "quantity_semantics_mismatch");
    assert!(details(&refusal).contains(&"mechanism_composition.units_mismatch"), "{refusal:?}");
    let first = &refusal.refusals[0];
    assert_eq!(first.offending.as_deref(), Some("M->Y"));
    assert!(first.remedy.is_some());

    // Parents: Y was fitted on {X} only; the declared graph has {X, M}.
    let mut mechanisms = chain();
    mechanisms[2] = linear("Y", -1.0, 0.5, &[("X", 0.5)], false);
    let mut request = request_of(mechanisms);
    request.edges.push(("M".into(), "Y".into()));
    let refusal = StudyBuilder::mechanism_composition(request, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "graph_invalid");
    assert_eq!(details(&refusal), vec!["mechanism_composition.parents_declared_graph_mismatch"]);
    assert_eq!(refusal.refusals[0].expected.as_deref(), Some("{M,X}"));
    assert_eq!(refusal.refusals[0].supplied.as_deref(), Some("{X}"));

    // A cycle is refused before anything executes.
    let cyc = vec![
        linear("A", 0.0, 1.0, &[("B", 0.5)], false),
        linear("B", 0.0, 1.0, &[("A", 0.5)], false),
    ];
    let refusal = StudyBuilder::mechanism_composition(request_of(cyc), &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "graph_invalid");
    assert!(details(&refusal).contains(&"mechanism_composition.cycle_directed_cycle"));

    // Support: Y was fitted on X in [0, 10]; X is declared on the whole line.
    let mut mechanisms = chain();
    mechanisms[2].parent_domains[0].1 = Domain::Interval { lo: 0.0, hi: 10.0 };
    let refusal = StudyBuilder::mechanism_composition(request_of(mechanisms), &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "mechanism_composition_incoherent");
    assert_eq!(details(&refusal), vec!["mechanism_composition.support_not_contained"]);

    // A parent with no mechanism of its own.
    let refusal = StudyBuilder::mechanism_composition(
        request_of(vec![linear("Y", 0.0, 1.0, &[("W", 1.0)], false)]),
        &ctx(),
    )
    .unwrap_err();
    assert!(details(&refusal).contains(&"mechanism_composition.identity_parent_without_mechanism"));

    // Provenance: an unbound mechanism.
    let mut mechanisms = chain();
    mechanisms[1].declaration.artifact_id = " ".into();
    let refusal = StudyBuilder::mechanism_composition(request_of(mechanisms), &ctx()).unwrap_err();
    assert_eq!(details(&refusal), vec!["mechanism_composition.provenance_missing_artifact"]);

    // Families are never mixed.
    let mut mechanisms = chain();
    let mut table = linear("Z", 0.0, 1.0, &[], true);
    table.source = MechanismSource::DeclaredTable {
        levels: vec!["0".into(), "1".into()],
        rows: vec![TabularRow {
            parent_levels: vec![],
            probabilities: vec![("0".into(), 0.5), ("1".into(), 0.5)],
        }],
    };
    table.output_support = Domain::Levels(vec!["0".into(), "1".into()]);
    mechanisms.push(table);
    let mut request = request_of(mechanisms);
    request.admit_declared_mechanisms = true;
    let refusal = StudyBuilder::mechanism_composition(request, &ctx()).unwrap_err();
    assert!(details(&refusal).contains(&"mechanism_composition.family_mixed"), "{refusal:?}");
}

#[test]
fn a_mechanism_from_another_population_needs_a_declared_admission() {
    // M was fitted in `source`: its output and the parents it saw are there;
    // Y, fitted in `pop`, consumes M as a realization of the same quantity.
    let from_source = || {
        let mut mechanisms = chain();
        mechanisms[1].declaration.output.population_id = "source".into();
        mechanisms[1].declaration.parents[0].population_id = "source".into();
        mechanisms
    };
    let refusal =
        StudyBuilder::mechanism_composition(request_of(from_source()), &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "quantity_semantics_mismatch");
    assert_eq!(details(&refusal), vec!["mechanism_composition.population_undeclared_transport"]);
    assert_eq!(refusal.refusals[0].expected.as_deref(), Some("pop"));
    assert_eq!(refusal.refusals[0].supplied.as_deref(), Some("source"));

    let admission = |basis: AdmissionBasis| TransportAdmission {
        variable_id: "M".into(),
        coordinate: AdmissionCoordinate::Population,
        source_id: "source".into(),
        target_id: "pop".into(),
        basis,
    };
    // A selection diagram with no arrow into M admits it.
    let mut request = request_of(from_source());
    request.admissions =
        vec![admission(AdmissionBasis::SelectionDiagram { targets: vec!["X".into()] })];
    assert!(StudyBuilder::mechanism_composition(request, &ctx()).is_ok());
    // One that points into M does not.
    let mut request = request_of(from_source());
    request.admissions =
        vec![admission(AdmissionBasis::SelectionDiagram { targets: vec!["M".into()] })];
    let refusal = StudyBuilder::mechanism_composition(request, &ctx()).unwrap_err();
    assert_eq!(
        details(&refusal),
        vec!["mechanism_composition.population_selection_arrow_into_mechanism"]
    );
    // An explicit invariance assumption admits it regardless.
    let mut request = request_of(from_source());
    request.admissions = vec![admission(AdmissionBasis::DeclaredInvariance {
        assumption_id: "assumption:m-invariant".into(),
    })];
    assert!(StudyBuilder::mechanism_composition(request, &ctx()).is_ok());
}

#[test]
fn shared_latents_are_exact_and_a_dropped_latent_is_refused() {
    let latent = |peer: &str| LatentCauseDeclaration {
        latent_id: "U".into(),
        shared_with_variable_ids: vec![peer.to_owned()],
    };
    // A = 0 + e + 2 U ; B = 1 + e' + 3 U  =>  Var A=5, Var B=10, Cov=6.
    let build = |with_b: bool| {
        let mut a = linear("A", 0.0, 1.0, &[], false);
        a.declaration.latent_causes = vec![latent("B")];
        a.latent_loadings = vec![("U".into(), 2.0)];
        let mut mechanisms = vec![a];
        if with_b {
            let mut b = linear("B", 1.0, 1.0, &[], false);
            b.declaration.latent_causes = vec![latent("A")];
            b.latent_loadings = vec![("U".into(), 3.0)];
            mechanisms.push(b);
        }
        request_of(mechanisms)
    };
    let prepared = StudyBuilder::mechanism_composition(build(true), &ctx()).unwrap();
    let law = prepared.intervention_law(&[]).unwrap();
    near(law.variance_of("A").unwrap(), 5.0);
    near(law.variance_of("B").unwrap(), 10.0);
    near(law.covariance_of("A", "B").unwrap(), 6.0);

    // Dropping B: A's latent is shared with a mechanism that is not here.
    let refusal = StudyBuilder::mechanism_composition(build(false), &ctx()).unwrap_err();
    assert!(
        details(&refusal).contains(&"mechanism_composition.latent_shared_with_absent_mechanism")
    );
    // The dependence exists but its strength is not declared.
    let mut request = build(true);
    request.mechanisms[0].latent_loadings.clear();
    let refusal = StudyBuilder::mechanism_composition(request, &ctx()).unwrap_err();
    assert!(details(&refusal).contains(&"mechanism_composition.latent_loading_undeclared"));
}

// ---------------------------------------------------------------------------
// Uncertainty: explicit cross-covariance or a refusal
// ---------------------------------------------------------------------------

#[test]
fn unknown_dependence_stays_unknown_and_refuses_the_uncertainty_claim() {
    let prepared =
        StudyBuilder::mechanism_composition(request_of(uncertain_chain()), &ctx()).unwrap();
    // Point execution is unaffected ...
    near(prepared.mean_contrast(&ate_at(2.0)).unwrap(), 13.5);
    // ... but the uncertainty claim is a joint-law refusal, not independence.
    let refusal = prepared.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(details(&refusal), vec!["mechanism_composition.posterior_dependence_unknown"]);
    assert!(refusal.refusals[0].remedy.is_some());
    // The interval route is closed regardless.
    let closed = prepared.interval().unwrap_err();
    assert_eq!(closed.code(), "cell_not_licensed");
    assert_eq!(closed.details(), vec!["mechanism_composition.interval_route_closed"]);

    // Independence declared over shared evidence is contradicted, not accepted.
    let mut mechanisms = uncertain_chain();
    mechanisms[1].declaration.evidence_id = "data:X".into();
    let mut request = request_of(mechanisms);
    request.dependence =
        DependenceInput::IndependentByDesign { justification: "disjoint trials".into() };
    let refusal = StudyBuilder::mechanism_composition(request, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(
        details(&refusal),
        vec!["mechanism_composition.posterior_dependence_shared_evidence_declared_independent"]
    );
}

#[test]
fn independent_assumption_and_declared_covariance_give_different_checked_variances() {
    // E[Y | do(X=2)] = a_Y + b_YX*2 + b_YM*(a_M + b_MX*2). Gradient: a_M 3,
    // b_MX 6, a_X 0, a_Y 1, b_YX 2, b_YM 4.5. Independent: M block
    // 0.04*9 + 0.01*36 + 2(-0.01)(3)(6) = 0.36; Y block
    // 0.09 + 0.01*4 + 0.0025*20.25 = 0.180625; total 0.540625.
    let mut request = request_of(uncertain_chain());
    request.dependence = DependenceInput::IndependentByDesign {
        justification: "disjoint trials, independently generated".into(),
    };
    let independent = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    // A product of path coefficients is nonlinear in the parameters: a declared
    // independence is not a joint law for it, whatever its justification.
    let refusal = independent.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(details(&refusal), vec!["joint_law_propagation.nonlinear_needs_aligned_joint_law"]);

    // cov(b_MX, b_YM) = 0.002 adds 2 * 6 * 4.5 * 0.002 = 0.108.
    let mut request = request_of(uncertain_chain());
    request.dependence = DependenceInput::CrossCovariance { blocks: cross_blocks() };
    let declared = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    let v = declared.mean_contrast_uncertainty(&ate_at(2.0)).unwrap();
    assert!((v.variance - 0.648_625).abs() < 1e-12, "{}", v.variance);
    assert_eq!(v.dependence, "declared_cross_covariance");
    assert!((v.variance - 0.540_625 - 0.108).abs() < 1e-12);
    let grad = |mechanism: &str, term: ParameterTerm| {
        v.gradient
            .iter()
            .find(|(p, _)| p.mechanism == mechanism && p.term == term)
            .map(|(_, g)| *g)
            .unwrap()
    };
    near(v.point, 13.5);
    near(grad("M", ParameterTerm::Intercept), 3.0);
    near(grad("M", ParameterTerm::Coefficient("X".into())), 6.0);
    near(grad("Y", ParameterTerm::Intercept), 1.0);
    near(grad("Y", ParameterTerm::Coefficient("X".into())), 2.0);
    near(grad("Y", ParameterTerm::Coefficient("M".into())), 4.5);
    assert!((v.standard_error - 0.648_625_f64.sqrt()).abs() < 1e-12);

    // The same dependence as one aligned joint law, in an arbitrary order.
    let ids: Vec<ParameterId> = vec![
        ("Y", ParameterTerm::Coefficient("X".into())),
        ("M", ParameterTerm::Intercept),
        ("X", ParameterTerm::Intercept),
        ("Y", ParameterTerm::Coefficient("M".into())),
        ("M", ParameterTerm::Coefficient("X".into())),
        ("Y", ParameterTerm::Intercept),
    ]
    .into_iter()
    .map(|(m, t)| ParameterId { mechanism: m.into(), term: t })
    .collect();
    let value = |a: &ParameterId, b: &ParameterId| -> f64 {
        let key = |p: &ParameterId| (p.mechanism.clone(), p.term.clone());
        let (ka, kb) = (key(a), key(b));
        let m_x = ("M".to_owned(), ParameterTerm::Coefficient("X".into()));
        let y_m = ("Y".to_owned(), ParameterTerm::Coefficient("M".into()));
        if ka == kb {
            match (ka.0.as_str(), &ka.1) {
                ("M", ParameterTerm::Intercept) => 0.04,
                ("M", ParameterTerm::Coefficient(_)) => 0.01,
                ("X", _) => 0.25,
                ("Y", ParameterTerm::Intercept) => 0.09,
                ("Y", ParameterTerm::Coefficient(p)) if p == "X" => 0.01,
                _ => 0.0025,
            }
        } else if ka.0 == "M" && kb.0 == "M" {
            -0.01
        } else if (ka == m_x && kb == y_m) || (ka == y_m && kb == m_x) {
            0.002
        } else {
            0.0
        }
    };
    let mut covariance = Vec::new();
    for a in &ids {
        for b in &ids {
            covariance.push(value(a, b));
        }
    }
    let mut request = request_of(uncertain_chain());
    request.dependence = DependenceInput::AlignedJointLaw {
        law_id: "joint-fit-1".into(),
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: ids,
        covariance,
    };
    let joint = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    let w = joint.mean_contrast_uncertainty(&ate_at(2.0)).unwrap();
    assert!((w.variance - 0.648_625).abs() < 1e-12, "{}", w.variance);
    assert_eq!(w.dependence, "joint_gaussian_covariance");
}

#[test]
fn a_missing_cross_covariance_pair_is_unknown_for_that_pair() {
    let mut blocks = cross_blocks();
    blocks.truncate(1);
    let mut request = request_of(uncertain_chain());
    request.dependence = DependenceInput::CrossCovariance { blocks };
    let prepared = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    let refusal = prepared.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    let found = details(&refusal);
    assert_eq!(found.len(), 2);
    assert!(
        found
            .iter()
            .all(|d| *d == "mechanism_composition.posterior_dependence_cross_covariance_missing")
    );
}

// ---------------------------------------------------------------------------
// Finite tables
// ---------------------------------------------------------------------------

fn table_input(variable: &str, parents: &[&str], rows: &[(&[&str], [f64; 2])]) -> MechanismInput {
    let levels = vec!["0".to_owned(), "1".to_owned()];
    let mut input = MechanismInput {
        declaration: declaration(variable, parents, "category"),
        source: MechanismSource::DeclaredTable {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, p)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|(parent, level)| ((*parent).to_owned(), (*level).to_owned()))
                        .collect(),
                    probabilities: vec![("0".to_owned(), p[0]), ("1".to_owned(), p[1])],
                })
                .collect(),
        },
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parents
            .iter()
            .map(|p| ((*p).to_owned(), Domain::Levels(levels.clone())))
            .collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    };
    input.declaration.output.units = "category".into();
    input
}

fn binary_chain() -> MechanismCompositionRequest {
    let mut request = request_of(vec![
        table_input("X", &[], &[(&[], [0.6, 0.4])]),
        table_input("M", &["X"], &[(&["0"], [0.8, 0.2]), (&["1"], [0.3, 0.7])]),
        // parents in declared order (M, X); rows keyed (M, X)
        table_input(
            "Y",
            &["M", "X"],
            &[
                (&["0", "0"], [0.9, 0.1]),
                (&["1", "0"], [0.5, 0.5]),
                (&["0", "1"], [0.7, 0.3]),
                (&["1", "1"], [0.1, 0.9]),
            ],
        ),
    ]);
    request.admit_declared_mechanisms = true;
    request
}

#[test]
fn finite_tables_match_the_enumerated_scm() {
    let prepared = StudyBuilder::mechanism_composition(binary_chain(), &ctx()).unwrap();
    assert_eq!(prepared.family(), MechanismFamily::Tabular);
    assert_eq!(prepared.declared_mechanisms(), ["M", "X", "Y"]);
    // A Gaussian query of a finite composition is refused, not coerced.
    assert!(prepared.intervention_law(&[]).is_err());
    let obs = prepared.finite_law(&[], &ctx()).unwrap();
    near(obs.probability(&[("Y", "1")]).unwrap(), 0.396);
    near(obs.probability(&[("X", "1"), ("M", "1"), ("Y", "1")]).unwrap(), 0.252);
    near(
        prepared.finite_law(&[("X", "1")], &ctx()).unwrap().probability(&[("Y", "1")]).unwrap(),
        0.72,
    );
    near(
        prepared.finite_law(&[("X", "0")], &ctx()).unwrap().probability(&[("Y", "1")]).unwrap(),
        0.18,
    );
    let do_m = prepared.finite_law(&[("M", "1")], &ctx()).unwrap();
    near(do_m.probability(&[("Y", "1")]).unwrap(), 0.66);
    near(do_m.probability(&[("X", "1")]).unwrap(), 0.4);
    near(obs.probabilities().iter().sum::<f64>(), 1.0);

    // A table with a missing parent configuration is refused.
    let mut request = binary_chain();
    if let MechanismSource::DeclaredTable { rows, .. } = &mut request.mechanisms[1].source {
        rows.pop();
    }
    let refusal = StudyBuilder::mechanism_composition(request, &ctx()).unwrap_err();
    assert!(
        details(&refusal).contains(&"mechanism_composition.support_missing_parent_configuration")
    );
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

fn contrast_query() -> CompositionQueryWire {
    CompositionQueryWire::GaussianMeanContrast {
        outcome: "Y".into(),
        arms: vec![(1.0, vec![("X".into(), 2.0)])],
    }
}

fn independent_chain() -> antecedent::PreparedMechanismComposition {
    let mut request = request_of(uncertain_chain());
    request.dependence = DependenceInput::IndependentByDesign {
        justification: "disjoint trials, independently generated".into(),
    };
    StudyBuilder::mechanism_composition(request, &ctx()).unwrap()
}

#[test]
fn artifact_replays_and_refuses_semantic_mutation() {
    let prepared = independent_chain();
    let export = prepared.export(contrast_query(), &ctx()).unwrap();
    let consumed = consume_mechanism_composition_artifact(
        &export.bytes,
        Some(&export.premises_digest),
        &ctx(),
    )
    .unwrap();
    assert_eq!(consumed.family, MechanismFamily::LinearGaussian);
    let (point, variance, dependence, method) = consumed.gaussian_functional().unwrap();
    near(point, 13.5);
    assert!((variance - 0.540_625).abs() < 1e-12);
    assert_eq!(dependence, "declared_independent_by_design");
    assert_eq!(method, "first_order_delta_method");
    assert_eq!(consumed.result, export.result);

    // A retained identity that is not the artifact's is refused.
    let refusal =
        consume_mechanism_composition_artifact(&export.bytes, Some(&"0".repeat(64)), &ctx())
            .unwrap_err();
    assert_eq!(refusal.details(), vec!["mechanism_composition.premises_mismatch"]);
    // Damaged bytes are refused.
    let mut damaged = export.bytes.clone();
    let at = damaged.len() / 2;
    damaged[at] ^= 0xFF;
    assert!(consume_mechanism_composition_artifact(&damaged, None, &ctx()).is_err());

    // An edit that is not resealed is a premises mismatch.
    let mut wire = MechanismCompositionArtifactWire::decode(&export.bytes).unwrap();
    wire.premises.target_regime = "do(Z=1)".into();
    let bytes = wire.export().unwrap();
    let refusal = consume_mechanism_composition_artifact(&bytes, None, &ctx()).unwrap_err();
    assert_eq!(refusal.details(), vec!["mechanism_composition.premises_mismatch"]);

    // A resealed edit that changes the answer is a result mismatch ...
    let mut wire = MechanismCompositionArtifactWire::decode(&export.bytes).unwrap();
    if let CompositionQueryWire::GaussianMeanContrast { arms, .. } = &mut wire.query {
        arms[0].1[0].1 = 3.0;
    }
    wire.premises_digest = mechanism_composition_identity(&wire.premises, &wire.query).unwrap();
    let bytes = wire.export().unwrap();
    let refusal = consume_mechanism_composition_artifact(&bytes, None, &ctx()).unwrap_err();
    assert_eq!(refusal.details(), vec!["mechanism_composition.result_mismatch"]);

    // ... and one that breaks coherence is refused with the structured defect.
    let mut wire = MechanismCompositionArtifactWire::decode(&export.bytes).unwrap();
    let m =
        wire.premises.mechanisms.iter_mut().find(|m| m.variable == "M").expect("M is a premise");
    m.output_units = "mmHg".into();
    wire.premises_digest = mechanism_composition_identity(&wire.premises, &wire.query).unwrap();
    let bytes = wire.export().unwrap();
    let refusal = consume_mechanism_composition_artifact(&bytes, None, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "quantity_semantics_mismatch");
    assert!(details(&refusal).contains(&"mechanism_composition.units_mismatch"));
}

#[test]
fn a_finite_artifact_matches_the_enumerated_truth_and_an_uncertainty_claim_needs_its_law() {
    let prepared = StudyBuilder::mechanism_composition(binary_chain(), &ctx()).unwrap();
    let export = prepared
        .export(
            CompositionQueryWire::FiniteInterventionDistribution {
                interventions: vec![("X".into(), "1".into())],
                outcome: "Y".into(),
            },
            &ctx(),
        )
        .unwrap();
    let consumed = consume_mechanism_composition_artifact(&export.bytes, None, &ctx()).unwrap();
    let distribution = consumed.finite_distribution().unwrap();
    assert_eq!(distribution.len(), 2);
    near(distribution[1].1, 0.72);
    near(distribution[0].1, 0.28);

    // An uncertainty claim over unknown dependence is not exported.
    let unknown =
        StudyBuilder::mechanism_composition(request_of(uncertain_chain()), &ctx()).unwrap();
    let refusal = unknown.export(contrast_query(), &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "joint_law_required");
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[test]
fn refresh_keeps_the_plan_recomputes_the_numbers_and_refuses_other_mechanisms() {
    let prepared = StudyBuilder::mechanism_composition(request_of(chain()), &ctx()).unwrap();
    let before = prepared.intervention_law(&[("X", 2.0)]).unwrap();

    // M refitted on new data: intercept 1, coefficient 3. E[M|do(X=2)] = 7 and
    // E[Y|do(X=2)] = -1 + 1 + 3*7 = 21.
    let refit = MechanismRefit {
        variable_id: "M".into(),
        source: MechanismSource::FittedLinear {
            intercept: 1.0,
            sigma: 1.0,
            coefficients: vec![("X".into(), 3.0)],
        },
        artifact_id: "fit:M:2".into(),
        evidence_id: "data:M:2".into(),
        uncertainty: None,
    };
    let refreshed = prepared.refresh_mechanisms(vec![refit.clone()], None, &ctx()).unwrap();
    let after = refreshed.intervention_law(&[("X", 2.0)]).unwrap();
    near(after.mean_of("M").unwrap(), 7.0);
    near(after.mean_of("Y").unwrap(), 21.0);
    // Kept: the semantic coordinates of the plan.
    assert_eq!(refreshed.plan().edges, prepared.plan().edges);
    assert_eq!(refreshed.plan().target_population_id, prepared.plan().target_population_id);
    for (old, new) in prepared.plan().mechanisms.iter().zip(&refreshed.plan().mechanisms) {
        assert_eq!(old.variable_id, new.variable_id);
        assert_eq!(old.units, new.units);
        assert_eq!(old.parents, new.parents);
        assert_eq!(old.population_id, new.population_id);
    }
    // Recomputed: the provenance, hence the plan identity, and the numbers.
    assert_ne!(refreshed.plan_identity_hex(), prepared.plan_identity_hex());
    // The prepared composition it was called on is unchanged.
    assert_eq!(prepared.intervention_law(&[("X", 2.0)]).unwrap(), before);
    near(before.mean_of("Y").unwrap(), 13.5);

    // Refused: another mechanism family, other parents, an unknown variable.
    let mut changed_family = refit.clone();
    changed_family.source =
        MechanismSource::DeclaredTable { levels: vec!["0".into(), "1".into()], rows: vec![] };
    let refusal = prepared.refresh_mechanisms(vec![changed_family], None, &ctx()).unwrap_err();
    assert_eq!(details(&refusal), vec!["mechanism_composition.family_mixed"]);
    let mut other_parents = refit.clone();
    other_parents.source = MechanismSource::FittedLinear {
        intercept: 1.0,
        sigma: 1.0,
        coefficients: vec![("X".into(), 3.0), ("Z".into(), 1.0)],
    };
    let refusal = prepared.refresh_mechanisms(vec![other_parents], None, &ctx()).unwrap_err();
    assert!(details(&refusal).contains(&"mechanism_composition.shape_coefficient_parent_mismatch"));
    let mut unknown = refit.clone();
    unknown.variable_id = "Q".into();
    let refusal = prepared.refresh_mechanisms(vec![unknown], None, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "quantity_semantics_mismatch");
    assert_eq!(
        details(&refusal),
        vec!["mechanism_composition.identity_declaration_unknown_variable"]
    );
    // Still unchanged after three refusals.
    assert_eq!(prepared.intervention_law(&[("X", 2.0)]).unwrap(), before);
}

#[test]
fn a_dependence_declared_for_the_earlier_fit_is_unknown_again_after_a_refit() {
    let mut request = request_of(uncertain_chain());
    request.dependence = DependenceInput::CrossCovariance { blocks: cross_blocks() };
    let prepared = StudyBuilder::mechanism_composition(request, &ctx()).unwrap();
    let refit = MechanismRefit {
        variable_id: "M".into(),
        source: MechanismSource::FittedLinear {
            intercept: 0.5,
            sigma: 1.0,
            coefficients: vec![("X".into(), 2.0)],
        },
        artifact_id: "fit:M:2".into(),
        evidence_id: "data:M:2".into(),
        uncertainty: Some(marginal(
            vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())],
            vec![0.04, -0.01, -0.01, 0.01],
        )),
    };
    let refusal = prepared.refresh_mechanisms(vec![refit.clone()], None, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(details(&refusal), vec!["mechanism_composition.posterior_dependence_unknown"]);
    // Redeclared, the refit composes and the variance is recomputed.
    let refreshed = prepared
        .refresh_mechanisms(
            vec![refit],
            Some(DependenceInput::IndependentByDesign {
                justification: "refit on a new, disjoint trial".into(),
            }),
            &ctx(),
        )
        .unwrap();
    let refusal = refreshed.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(details(&refusal), vec!["joint_law_propagation.nonlinear_needs_aligned_joint_law"]);
}

#[test]
fn cancellation_stops_before_a_composed_result() {
    let context = ExecutionContext::for_tests(1);
    context.cancellation.cancel();
    let refusal = StudyBuilder::mechanism_composition(request_of(chain()), &context).unwrap_err();
    assert_eq!(refusal.code(), "cancelled_no_claim");
    let prepared = StudyBuilder::mechanism_composition(request_of(chain()), &ctx()).unwrap();
    let refusal = prepared.sample(&[], 100, &context).unwrap_err();
    assert_eq!(refusal.code(), "cancelled_no_claim");
    let refusal = prepared.export(contrast_query(), &context).unwrap_err();
    assert_eq!(refusal.code(), "cancelled_no_claim");
}
