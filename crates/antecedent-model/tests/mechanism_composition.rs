//! Composition of independently fitted mechanisms against closed-form and
//! enumerated truth, with the refusals that individually valid mechanisms owe.
//!
//! Hand-computed oracle (linear-Gaussian chain), with
//!   X = 1 + 2 `e_X`
//!   M = 0.5 + 2 X + 1 `e_M`
//!   Y = -1 + 0.5 X + 3 M + 0.5 `e_Y`
//! so that Y = 0.5 + 6.5 X + 3 `e_M` + `e_Y`:
//!   E[X]=1, E[M]=2.5, E[Y]=7
//!   Var X=4, Var M=17, Var Y=178.25
//!   Cov(X,M)=8, Cov(X,Y)=26, Cov(M,Y)=55
//!   do(X=2): E[M]=4.5, E[Y]=13.5, Var M=1, Var Y=9.25, Cov(M,Y)=3
//!   do(M=1): E[Y]=2.5, Var Y=1.25, Cov(X,Y)=2
//!   E[Y|do(X=1)] - E[Y|do(X=0)] = 6.5

#![allow(
    clippy::needless_range_loop,
    reason = "enumerated finite-SCM oracles index their tables by the level of each variable"
)]

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, CompositionError, ContextCoordinate,
    ContrastArm, CrossCovarianceBlock, Domain, InvarianceDeclaration, LatentDependence,
    MeanContrast, MechanismContext, MechanismDescriptor, MechanismForm, MechanismOrigin,
    MechanismProvenance, MechanismUncertainty, ParameterId, ParameterTerm, ParentBinding,
    PosteriorDependence, SelectionDiagramDeclaration, SelectionNodeDeclaration, TabularRow,
    UncertaintyMeaning, check_composition, uncertainty_findings,
};

const TOL: f64 = 1e-12;

fn provenance(var: &str) -> MechanismProvenance {
    MechanismProvenance {
        origin: MechanismOrigin::Fitted,
        artifact_id: format!("fit:{var}"),
        evidence_id: format!("data:{var}"),
    }
}

fn binding(var: &str) -> ParentBinding {
    ParentBinding {
        variable: var.into(),
        units: "u".into(),
        transform: "identity".into(),
        domain: Domain::Real,
    }
}

fn lg(var: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma,
            coefficients: parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        },
        parents: parents.iter().map(|(p, _)| binding(p)).collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: provenance(var),
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec_of(mechanisms: Vec<MechanismDescriptor>) -> ComposedModelSpec {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.parents {
            edges.push((p.variable.clone(), m.variable.clone()));
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

fn chain() -> Vec<MechanismDescriptor> {
    vec![
        lg("X", 1.0, 2.0, &[]),
        lg("M", 0.5, 1.0, &[("X", 2.0)]),
        lg("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ]
}

fn details(error: &CompositionError) -> Vec<&'static str> {
    error.findings().iter().map(|f| f.detail).collect()
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

#[test]
fn composition_matches_the_hand_computed_gaussian_scm() {
    let model = ComposedLinearGaussian::compile(&spec_of(chain())).unwrap();
    let obs = model.law(&[]).unwrap();
    near(obs.mean_of("X").unwrap(), 1.0);
    near(obs.mean_of("M").unwrap(), 2.5);
    near(obs.mean_of("Y").unwrap(), 7.0);
    near(obs.variance_of("X").unwrap(), 4.0);
    near(obs.variance_of("M").unwrap(), 17.0);
    near(obs.variance_of("Y").unwrap(), 178.25);
    near(obs.covariance_of("X", "M").unwrap(), 8.0);
    near(obs.covariance_of("X", "Y").unwrap(), 26.0);
    near(obs.covariance_of("M", "Y").unwrap(), 55.0);

    let do_x = model.law(&[("X", 2.0)]).unwrap();
    near(do_x.mean_of("M").unwrap(), 4.5);
    near(do_x.mean_of("Y").unwrap(), 13.5);
    near(do_x.variance_of("X").unwrap(), 0.0);
    near(do_x.variance_of("M").unwrap(), 1.0);
    near(do_x.variance_of("Y").unwrap(), 9.25);
    near(do_x.covariance_of("M", "Y").unwrap(), 3.0);

    let do_m = model.law(&[("M", 1.0)]).unwrap();
    near(do_m.mean_of("Y").unwrap(), 2.5);
    near(do_m.variance_of("Y").unwrap(), 1.25);
    near(do_m.mean_of("X").unwrap(), 1.0);
    near(do_m.covariance_of("X", "Y").unwrap(), 2.0);

    let ate = MeanContrast {
        outcome: "Y".into(),
        arms: vec![
            ContrastArm { weight: 1.0, interventions: vec![("X".into(), 1.0)] },
            ContrastArm { weight: -1.0, interventions: vec![("X".into(), 0.0)] },
        ],
    };
    near(model.mean_contrast(&ate).unwrap(), 6.5);
}

#[test]
fn structural_sampling_matches_the_closed_form_moments_and_is_deterministic() {
    let model = ComposedLinearGaussian::compile(&spec_of(chain())).unwrap();
    let ctx = ExecutionContext::for_tests(20_260_101);
    let n = 200_000;
    let table = model.sample(&[("X", 2.0)], n, &ctx).unwrap();
    let col = |name: &str| {
        let i = table.variables.iter().position(|v| v == name).unwrap();
        &table.columns[i]
    };
    let y = col("Y");
    let mean = y.iter().sum::<f64>() / n as f64;
    let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    // Standard error of the mean is sqrt(9.25 / 200000) = 0.0068; allow 6 SE.
    assert!((mean - 13.5).abs() < 0.05, "mean {mean}");
    // Relative standard error of the variance is sqrt(2/n) = 0.0032; allow 6x.
    assert!((var / 9.25 - 1.0).abs() < 0.02, "var {var}");
    // do(X=2) fixes the column exactly.
    assert!(col("X").iter().all(|&x| x.to_bits() == 2.0_f64.to_bits()));

    let again = model.sample(&[("X", 2.0)], n, &ExecutionContext::for_tests(20_260_101)).unwrap();
    assert_eq!(table, again);

    // Streams are keyed by name: adding an unrelated root does not move M's noise.
    let mut mechanisms = chain();
    mechanisms.push(lg("Z", 0.0, 1.0, &[]));
    let bigger = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap();
    let wide = bigger.sample(&[], 1000, &ExecutionContext::for_tests(5)).unwrap();
    let narrow = model.sample(&[], 1000, &ExecutionContext::for_tests(5)).unwrap();
    let pick = |t: &antecedent_model::compose::SampleTable, name: &str| -> Vec<f64> {
        t.columns[t.variables.iter().position(|v| v == name).unwrap()].clone()
    };
    assert_eq!(pick(&wide, "M"), pick(&narrow, "M"));
    assert_eq!(pick(&wide, "Y"), pick(&narrow, "Y"));
}

#[test]
fn sampling_observes_cancellation() {
    let model = ComposedLinearGaussian::compile(&spec_of(chain())).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    assert_eq!(model.sample(&[], 100, &ctx).unwrap_err(), CompositionError::Cancelled);
}

#[test]
fn shared_latent_covariance_is_exact_and_a_dropped_latent_is_refused() {
    // A = 0 + e + 2 U ; B = 1 + e' + 3 U  =>  Var A=5, Var B=10, Cov=6.
    let latent = |loading: Option<f64>, with: &[&str]| LatentDependence {
        latent: "U".into(),
        loading,
        shared_with: with.iter().map(|s| (*s).to_owned()).collect(),
    };
    let mut a = lg("A", 0.0, 1.0, &[]);
    a.latents = vec![latent(Some(2.0), &["B"])];
    let mut b = lg("B", 1.0, 1.0, &[]);
    b.latents = vec![latent(Some(3.0), &["A"])];
    let model = ComposedLinearGaussian::compile(&spec_of(vec![a.clone(), b.clone()])).unwrap();
    let law = model.law(&[]).unwrap();
    near(law.variance_of("A").unwrap(), 5.0);
    near(law.variance_of("B").unwrap(), 10.0);
    near(law.covariance_of("A", "B").unwrap(), 6.0);
    near(law.mean_of("B").unwrap(), 1.0);

    // Dropping B: A's latent is shared with a mechanism that is not here. The
    // composition must not silently treat A as independent of B.
    let err = ComposedLinearGaussian::compile(&spec_of(vec![a.clone()])).unwrap_err();
    assert!(
        details(&err).contains(&"mechanism_composition.latent_shared_with_absent_mechanism"),
        "{err:?}"
    );

    // B does not declare the latent A claims to share.
    let plain_b = lg("B", 1.0, 1.0, &[]);
    let err = ComposedLinearGaussian::compile(&spec_of(vec![a.clone(), plain_b])).unwrap_err();
    assert!(
        details(&err).contains(&"mechanism_composition.latent_sharing_not_reciprocal"),
        "{err:?}"
    );

    // The dependence exists but its strength is not declared.
    let mut unloaded = a.clone();
    unloaded.latents = vec![latent(None, &["B"])];
    let err = ComposedLinearGaussian::compile(&spec_of(vec![unloaded, b.clone()])).unwrap_err();
    assert!(details(&err).contains(&"mechanism_composition.latent_loading_undeclared"), "{err:?}");

    // Both depend on U but neither lists the other: undeclared sharing.
    let mut a_silent = a;
    a_silent.latents = vec![latent(Some(2.0), &[])];
    let mut b_silent = b;
    b_silent.latents = vec![latent(Some(3.0), &[])];
    let err = ComposedLinearGaussian::compile(&spec_of(vec![a_silent, b_silent])).unwrap_err();
    assert!(
        details(&err).contains(&"mechanism_composition.latent_sharing_not_declared"),
        "{err:?}"
    );
}

#[test]
fn individually_valid_mechanisms_that_cannot_compose_are_refused() {
    // Parent sets: Y was fitted on {X} only, the declared graph has {X, M}.
    let mut mechanisms = chain();
    mechanisms[2] = lg("Y", -1.0, 0.5, &[("X", 0.5)]);
    let mut spec = spec_of(mechanisms);
    spec.edges.push(("M".into(), "Y".into()));
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.parents_declared_graph_mismatch"]);
    let finding = &err.findings()[0];
    assert_eq!(finding.expected.as_deref(), Some("{M,X}"));
    assert_eq!(finding.supplied.as_deref(), Some("{X}"));

    // Conditioning set wider than the parents: not the mechanism P(y | parents).
    let mut mechanisms = chain();
    mechanisms[2].conditioning_set.push("Z".into());
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.conditioning_extra_conditioner"]);

    // Units: M is produced in mmHg but Y was fitted on M in kPa.
    let mut mechanisms = chain();
    mechanisms[1].output_units = "mmHg".into();
    mechanisms[2].parents[1].units = "kPa".into();
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.units_mismatch"]);
    assert_eq!(err.findings()[0].subject, "M->Y");

    // Transform: Y was fitted on log M, M is produced on the identity scale.
    let mut mechanisms = chain();
    mechanisms[2].parents[1].transform = "log".into();
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.transform_mismatch"]);

    // Support gap: Y was fitted on X in [0, 10]; X is declared on the whole line.
    let mut mechanisms = chain();
    mechanisms[2].parents[0].domain = Domain::Interval { lo: 0.0, hi: 10.0 };
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.support_not_contained"]);
    // Declaring the producer's own bounded support repairs it.
    let mut mechanisms = chain();
    mechanisms[2].parents[0].domain = Domain::Interval { lo: 0.0, hi: 10.0 };
    mechanisms[0].output_support = Domain::Interval { lo: 1.0, hi: 5.0 };
    // M was fitted on X as Real, which still contains [1,5].
    assert!(ComposedLinearGaussian::compile(&spec_of(mechanisms)).is_ok());
    // No overlap at all.
    let mut mechanisms = chain();
    mechanisms[2].parents[0].domain = Domain::Interval { lo: 0.0, hi: 10.0 };
    mechanisms[0].output_support = Domain::Interval { lo: 20.0, hi: 30.0 };
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.support_disjoint"]);

    // Population: M was fitted elsewhere with no declaration.
    let mut mechanisms = chain();
    mechanisms[1].context.population = "source".into();
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms.clone())).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.population_undeclared_transport"]);

    // A selection diagram that does not point into M admits it ...
    let mut spec = spec_of(mechanisms.clone());
    spec.selection_diagrams.push(SelectionDiagramDeclaration {
        source_population: "source".into(),
        target_population: "pop".into(),
        selection_nodes: vec![SelectionNodeDeclaration {
            name: "S".into(),
            points_into: vec!["X".into()],
        }],
    });
    assert!(ComposedLinearGaussian::compile(&spec).is_ok());
    // ... and one that points into M does not.
    spec.selection_diagrams[0].selection_nodes[0].points_into = vec!["M".into()];
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(
        details(&err),
        vec!["mechanism_composition.population_selection_arrow_into_mechanism"]
    );
    // An invariance declaration contradicts the arrow and cannot override it:
    // the diagram dominates, as in the transport entry point and the finite
    // counterfactual obstruction.
    spec.invariances.push(InvarianceDeclaration {
        variable: "M".into(),
        coordinate: ContextCoordinate::Population,
        source: "source".into(),
        target: "pop".into(),
        assumption_id: "assumption:m-invariant".into(),
    });
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(
        details(&err),
        vec!["mechanism_composition.population_selection_arrow_into_mechanism"]
    );
    // The same declaration admits it where the diagram is silent about M.
    spec.selection_diagrams[0].selection_nodes[0].points_into = vec!["X".into()];
    assert!(ComposedLinearGaussian::compile(&spec).is_ok());

    // Regime: fitted under do(Z=1), composed in the observational regime.
    let mut mechanisms = chain();
    mechanisms[0].context.regime = "do(Z=1)".into();
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.regime_undeclared_invariance"]);

    // Cycle: refused before anything executes.
    let cyc = spec_of(vec![lg("A", 0.0, 1.0, &[("B", 0.5)]), lg("B", 0.0, 1.0, &[("A", 0.5)])]);
    let err = ComposedLinearGaussian::compile(&cyc).unwrap_err();
    assert!(details(&err).contains(&"mechanism_composition.cycle_directed_cycle"), "{err:?}");

    // Duplicate mechanism for one variable.
    let mut mechanisms = chain();
    mechanisms.push(lg("M", 9.0, 1.0, &[("X", 1.0)]));
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert!(details(&err).contains(&"mechanism_composition.identity_duplicate_mechanism"));

    // A parent with no mechanism of its own.
    let err = ComposedLinearGaussian::compile(&spec_of(vec![lg("Y", 0.0, 1.0, &[("W", 1.0)])]))
        .unwrap_err();
    assert!(details(&err).contains(&"mechanism_composition.identity_parent_without_mechanism"));

    // Provenance: an unbound mechanism, and an analyst-declared one.
    let mut mechanisms = chain();
    mechanisms[1].provenance.artifact_id = " ".into();
    let err = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.provenance_missing_artifact"]);
    let mut mechanisms = chain();
    mechanisms[1].provenance.origin = MechanismOrigin::Declared;
    let mut spec = spec_of(mechanisms);
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(
        details(&err),
        vec!["mechanism_composition.provenance_declared_mechanism_unacknowledged"]
    );
    spec.admit_declared_mechanisms = true;
    assert!(ComposedLinearGaussian::compile(&spec).is_ok());
}

#[test]
fn an_intervention_outside_a_consumers_fitted_domain_is_refused() {
    let mut mechanisms = chain();
    mechanisms[0].output_support = Domain::Interval { lo: 0.0, hi: 10.0 };
    mechanisms[1].parents[0].domain = Domain::Interval { lo: 0.0, hi: 10.0 };
    mechanisms[2].parents[0].domain = Domain::Interval { lo: 0.0, hi: 10.0 };
    let model = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap();
    assert!(model.law(&[("X", 3.0)]).is_ok());
    let err = model.law(&[("X", 11.0)]).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.support_intervention_outside_domain"]);
}

fn uncertain_chain(dependence: PosteriorDependence) -> ComposedModelSpec {
    let mut mechanisms = chain();
    let unc = |terms: Vec<ParameterTerm>, cov: Vec<f64>| MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: terms,
        covariance: cov,
    };
    // M: (a_M, b_MX) ; X: (a_X) ; Y: (a_Y, b_YX, b_YM) -- deliberately listed
    // out of canonical order for Y.
    mechanisms[1].uncertainty = Some(unc(
        vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())],
        vec![0.04, -0.01, -0.01, 0.01],
    ));
    mechanisms[0].uncertainty = Some(unc(vec![ParameterTerm::Intercept], vec![0.25]));
    mechanisms[2].uncertainty = Some(unc(
        vec![
            ParameterTerm::Intercept,
            ParameterTerm::Coefficient("X".into()),
            ParameterTerm::Coefficient("M".into()),
        ],
        vec![0.09, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0025],
    ));
    let mut spec = spec_of(mechanisms);
    spec.dependence = dependence;
    spec
}

fn ate_at(x: f64) -> MeanContrast {
    MeanContrast {
        outcome: "Y".into(),
        arms: vec![ContrastArm { weight: 1.0, interventions: vec![("X".into(), x)] }],
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one hand-derived gradient and two checked variances asserted side by side"
)]
fn independent_assumption_and_declared_covariance_give_different_checked_variances() {
    // E[Y | do(X=2)] = a_Y + b_YX*2 + b_YM*(a_M + b_MX*2).
    // Gradient: a_M: b_YM=3, b_MX: b_YM*2=6, a_X: 0, a_Y: 1, b_YX: 2,
    //           b_YM: a_M + 2 b_MX = 4.5.
    // Independent: M block 0.04*9 + 0.01*36 + 2(-0.01)(3)(6) = 0.36
    //              Y block 0.09*1 + 0.01*4 + 0.0025*20.25 = 0.180625
    //              => 0.540625
    let independent = uncertain_chain(PosteriorDependence::IndependentByDesign {
        justification: "disjoint trials, independently generated".into(),
    });
    let model = ComposedLinearGaussian::compile(&independent).unwrap();
    let u = model.mean_contrast_uncertainty(&ate_at(2.0)).unwrap();
    near(u.point, 13.5);
    assert!((u.variance - 0.540_625).abs() < 1e-12, "{}", u.variance);
    assert_eq!(u.method, "first_order_delta_method");
    assert_eq!(u.dependence, "declared_independent_by_design");
    let grad = |id: &str, term: ParameterTerm| {
        u.gradient
            .iter()
            .find(|(p, _)| p.mechanism == id && p.term == term)
            .map(|(_, g)| *g)
            .unwrap()
    };
    near(grad("M", ParameterTerm::Intercept), 3.0);
    near(grad("M", ParameterTerm::Coefficient("X".into())), 6.0);
    near(grad("X", ParameterTerm::Intercept), 0.0);
    near(grad("Y", ParameterTerm::Intercept), 1.0);
    near(grad("Y", ParameterTerm::Coefficient("X".into())), 2.0);
    near(grad("Y", ParameterTerm::Coefficient("M".into())), 4.5);

    // cov(b_MX, b_YM) = 0.002 adds 2 * 6 * 4.5 * 0.002 = 0.108.
    // Every other pair is an explicit zero block (a declared independence).
    let cross =
        |first: &str, second: &str, fp: Vec<ParameterTerm>, sp: Vec<ParameterTerm>, v: Vec<f64>| {
            CrossCovarianceBlock {
                first: first.into(),
                second: second.into(),
                first_parameters: fp,
                second_parameters: sp,
                values: v,
            }
        };
    let m_terms = vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())];
    let x_terms = vec![ParameterTerm::Intercept];
    let y_terms = vec![
        ParameterTerm::Intercept,
        ParameterTerm::Coefficient("M".into()),
        ParameterTerm::Coefficient("X".into()),
    ];
    let blocks = vec![
        // rows (a_M, b_MX), cols (a_Y, b_YM, b_YX)
        cross("M", "Y", m_terms.clone(), y_terms.clone(), vec![0.0, 0.0, 0.0, 0.0, 0.002, 0.0]),
        cross("M", "X", m_terms, x_terms.clone(), vec![0.0, 0.0]),
        cross("X", "Y", x_terms, y_terms, vec![0.0, 0.0, 0.0]),
    ];
    let declared = uncertain_chain(PosteriorDependence::CrossCovariance { blocks });
    let model = ComposedLinearGaussian::compile(&declared).unwrap();
    let v = model.mean_contrast_uncertainty(&ate_at(2.0)).unwrap();
    assert!((v.variance - 0.648_625).abs() < 1e-12, "{}", v.variance);
    assert_eq!(v.dependence, "declared_cross_covariance");
    assert!((v.variance - u.variance - 0.108).abs() < 1e-12);
    assert!((v.standard_error - 0.648_625_f64.sqrt()).abs() < 1e-12);

    // The same dependence supplied as one aligned joint law, listed in an
    // arbitrary parameter order, gives the same number.
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
    let k = ids.len();
    let mut covariance = Vec::with_capacity(k * k);
    for a in &ids {
        for b in &ids {
            covariance.push(value(a, b));
        }
    }
    let joint = uncertain_chain(PosteriorDependence::Joint {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: ids,
        covariance,
    });
    let model = ComposedLinearGaussian::compile(&joint).unwrap();
    let w = model.mean_contrast_uncertainty(&ate_at(2.0)).unwrap();
    assert!((w.variance - 0.648_625).abs() < 1e-12, "{}", w.variance);
    assert_eq!(w.dependence, "aligned_joint_law");
}

#[test]
fn unknown_posterior_dependence_stays_unknown_and_refuses_the_uncertainty_claim() {
    let spec = uncertain_chain(PosteriorDependence::Unknown);
    // Point execution is unaffected ...
    let model = ComposedLinearGaussian::compile(&spec).unwrap();
    near(model.mean_contrast(&ate_at(2.0)).unwrap(), 13.5);
    // ... but the uncertainty claim is a joint-law refusal, not independence.
    let err = model.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.posterior_dependence_unknown"]);
    assert_eq!(uncertainty_findings(&spec).len(), 1);

    // A missing pair in a cross-covariance declaration is unknown for that pair.
    let m_terms = vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())];
    let y_terms = vec![
        ParameterTerm::Intercept,
        ParameterTerm::Coefficient("M".into()),
        ParameterTerm::Coefficient("X".into()),
    ];
    let partial = uncertain_chain(PosteriorDependence::CrossCovariance {
        blocks: vec![CrossCovarianceBlock {
            first: "M".into(),
            second: "Y".into(),
            first_parameters: m_terms,
            second_parameters: y_terms,
            values: vec![0.0; 6],
        }],
    });
    let model = ComposedLinearGaussian::compile(&partial).unwrap();
    let err = model.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    let d = details(&err);
    assert_eq!(d.len(), 2);
    assert!(
        d.iter()
            .all(|x| *x == "mechanism_composition.posterior_dependence_cross_covariance_missing")
    );

    // A mechanism that reports no marginal law has no uncertainty to propagate.
    let mut no_marginal = uncertain_chain(PosteriorDependence::IndependentByDesign {
        justification: "disjoint trials".into(),
    });
    no_marginal.mechanisms[1].uncertainty = None;
    let model = ComposedLinearGaussian::compile(&no_marginal).unwrap();
    let err = model.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.uncertainty_marginal_missing"]);
}

#[test]
fn misaligned_or_contradicted_uncertainty_is_refused_at_composition() {
    // Independence declared while two mechanisms share their evidence.
    let mut spec = uncertain_chain(PosteriorDependence::IndependentByDesign {
        justification: "disjoint trials".into(),
    });
    spec.mechanisms[1].provenance.evidence_id = "data:X".into();
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(
        details(&err),
        vec!["mechanism_composition.posterior_dependence_shared_evidence_declared_independent"]
    );

    // A posterior covariance and a sampling covariance are not one law.
    let mut spec = uncertain_chain(PosteriorDependence::IndependentByDesign {
        justification: "disjoint trials".into(),
    });
    spec.mechanisms[0].uncertainty.as_mut().unwrap().meaning = UncertaintyMeaning::Posterior;
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.uncertainty_meaning_mismatch"]);

    // A covariance over the wrong parameters.
    let mut spec = uncertain_chain(PosteriorDependence::Unknown);
    spec.mechanisms[0].uncertainty.as_mut().unwrap().parameters =
        vec![ParameterTerm::Coefficient("Q".into())];
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.uncertainty_parameter_set_mismatch"]);

    // Not positive semidefinite.
    let mut spec = uncertain_chain(PosteriorDependence::Unknown);
    spec.mechanisms[1].uncertainty.as_mut().unwrap().covariance = vec![1.0, 2.0, 2.0, 1.0];
    let err = ComposedLinearGaussian::compile(&spec).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.uncertainty_not_psd"]);

    // Cross covariance that makes the joint law invalid is caught on assembly.
    let m_terms = vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("X".into())];
    let x_terms = vec![ParameterTerm::Intercept];
    let y_terms = vec![
        ParameterTerm::Intercept,
        ParameterTerm::Coefficient("M".into()),
        ParameterTerm::Coefficient("X".into()),
    ];
    let spec = uncertain_chain(PosteriorDependence::CrossCovariance {
        blocks: vec![
            CrossCovarianceBlock {
                first: "M".into(),
                second: "Y".into(),
                first_parameters: m_terms.clone(),
                second_parameters: y_terms.clone(),
                // |cov| larger than sqrt(var*var) for (b_MX, b_YM).
                values: vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            },
            CrossCovarianceBlock {
                first: "M".into(),
                second: "X".into(),
                first_parameters: m_terms,
                second_parameters: x_terms.clone(),
                values: vec![0.0, 0.0],
            },
            CrossCovarianceBlock {
                first: "X".into(),
                second: "Y".into(),
                first_parameters: x_terms,
                second_parameters: y_terms,
                values: vec![0.0, 0.0, 0.0],
            },
        ],
    });
    let model = ComposedLinearGaussian::compile(&spec).unwrap();
    let err = model.mean_contrast_uncertainty(&ate_at(2.0)).unwrap_err();
    assert_eq!(details(&err), vec!["mechanism_composition.uncertainty_assembled_not_psd"]);
}

#[test]
fn composition_is_invariant_to_listing_order_and_to_consistent_relabeling() {
    let forward = ComposedLinearGaussian::compile(&spec_of(chain())).unwrap();

    // Reverse mechanisms, parents, coefficients and edges.
    let mut mechanisms = chain();
    mechanisms.reverse();
    for m in &mut mechanisms {
        m.parents.reverse();
        m.conditioning_set.reverse();
        if let MechanismForm::LinearGaussian { coefficients, .. } = &mut m.form {
            coefficients.reverse();
        }
    }
    let mut shuffled = spec_of(mechanisms);
    shuffled.edges.reverse();
    assert_eq!(shuffled.canonical(), spec_of(chain()).canonical());
    let backward = ComposedLinearGaussian::compile(&shuffled).unwrap();
    assert_eq!(forward.law(&[]).unwrap(), backward.law(&[]).unwrap());
    assert_eq!(forward.law(&[("X", 2.0)]).unwrap(), backward.law(&[("X", 2.0)]).unwrap());
    let ctx = ExecutionContext::for_tests(9);
    assert_eq!(
        forward.sample(&[], 64, &ctx).unwrap(),
        backward.sample(&[], 64, &ExecutionContext::for_tests(9)).unwrap()
    );

    // Consistent relabeling preserves every number under the relabeling map.
    let rename = |v: &str| match v {
        "X" => "q_treatment".to_owned(),
        "M" => "a_mediator".to_owned(),
        "Y" => "z_outcome".to_owned(),
        other => other.to_owned(),
    };
    let mut mechanisms = chain();
    for m in &mut mechanisms {
        m.variable = rename(&m.variable);
        for p in &mut m.parents {
            p.variable = rename(&p.variable);
        }
        for c in &mut m.conditioning_set {
            *c = rename(c);
        }
        if let MechanismForm::LinearGaussian { coefficients, .. } = &mut m.form {
            for (name, _) in coefficients.iter_mut() {
                *name = rename(name);
            }
        }
    }
    let renamed = ComposedLinearGaussian::compile(&spec_of(mechanisms)).unwrap();
    let a = forward.law(&[("X", 2.0)]).unwrap();
    let b = renamed.law(&[("q_treatment", 2.0)]).unwrap();
    for (orig, new) in [("X", "q_treatment"), ("M", "a_mediator"), ("Y", "z_outcome")] {
        near(a.mean_of(orig).unwrap(), b.mean_of(new).unwrap());
        near(a.variance_of(orig).unwrap(), b.variance_of(new).unwrap());
    }
    near(a.covariance_of("M", "Y").unwrap(), b.covariance_of("a_mediator", "z_outcome").unwrap());
}

fn table_mechanism(
    var: &str,
    parents: &[(&str, &[&str])],
    levels: &[&str],
    rows: &[(&[&str], &[f64])],
) -> MechanismDescriptor {
    let level_names: Vec<String> = levels.iter().map(|s| (*s).to_owned()).collect();
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::Tabular {
            levels: level_names.clone(),
            rows: rows
                .iter()
                .map(|(config, probs)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|((p, _), l)| ((*p).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: level_names.iter().cloned().zip(probs.iter().copied()).collect(),
                })
                .collect(),
        },
        parents: parents
            .iter()
            .map(|(p, lv)| ParentBinding {
                variable: (*p).to_owned(),
                units: "category".into(),
                transform: "identity".into(),
                domain: Domain::Levels(lv.iter().map(|s| (*s).to_owned()).collect()),
            })
            .collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "category".into(),
        output_transform: "identity".into(),
        output_support: Domain::Levels(level_names),
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: provenance(var),
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn binary_chain() -> Vec<MechanismDescriptor> {
    let b: &[&str] = &["0", "1"];
    vec![
        table_mechanism("X", &[], &["0", "1"], &[(&[], &[0.6, 0.4])]),
        table_mechanism(
            "M",
            &[("X", b)],
            &["0", "1"],
            &[(&["0"], &[0.8, 0.2]), (&["1"], &[0.3, 0.7])],
        ),
        // parents in declared order (M, X); rows keyed (M, X)
        table_mechanism(
            "Y",
            &[("M", b), ("X", b)],
            &["0", "1"],
            &[
                (&["0", "0"], &[0.9, 0.1]),
                (&["1", "0"], &[0.5, 0.5]),
                (&["0", "1"], &[0.7, 0.3]),
                (&["1", "1"], &[0.1, 0.9]),
            ],
        ),
    ]
}

/// Independent brute-force enumeration of the same finite SCM: a nested loop
/// over (x, m, y) written from the structural equations, not from the
/// factorization code under test.
fn enumerate(do_x: Option<usize>, do_m: Option<usize>) -> [f64; 8] {
    let px = [0.6, 0.4];
    let pm = [[0.8, 0.2], [0.3, 0.7]];
    let py1 = |x: usize, m: usize| -> f64 {
        match (x, m) {
            (0, 0) => 0.1,
            (0, 1) => 0.5,
            (1, 0) => 0.3,
            _ => 0.9,
        }
    };
    let mut out = [0.0; 8];
    for x in 0..2 {
        for m in 0..2 {
            for y in 0..2 {
                let fx = match do_x {
                    Some(v) => f64::from(u8::from(v == x)),
                    None => px[x],
                };
                let fm = match do_m {
                    Some(v) => f64::from(u8::from(v == m)),
                    None => pm[x][m],
                };
                let fy = if y == 1 { py1(x, m) } else { 1.0 - py1(x, m) };
                out[x * 4 + m * 2 + y] = fx * fm * fy;
            }
        }
    }
    out
}

#[test]
fn composition_matches_the_enumerated_finite_scm() {
    let model = ComposedFinite::compile(&spec_of(binary_chain())).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let obs = model.law(&[], &ctx).unwrap();
    assert_eq!(obs.variables(), ["M", "X", "Y"]);
    // Hand values.
    // P(M=1, X=1, Y=1) = 0.4 * 0.7 * 0.9
    near(obs.probability(&[("X", "1"), ("M", "1"), ("Y", "1")]).unwrap(), 0.252);
    // P(Y=1) = 0.6*(0.8*0.1+0.2*0.5) + 0.4*(0.3*0.3+0.7*0.9) = 0.108 + 0.288
    near(obs.probability(&[("Y", "1")]).unwrap(), 0.396);

    // do(X=1): 0.3*0.3 + 0.7*0.9 = 0.72 ; do(X=0): 0.8*0.1 + 0.2*0.5 = 0.18.
    let d1 = model.law(&[("X", "1")], &ctx).unwrap();
    near(d1.probability(&[("Y", "1")]).unwrap(), 0.72);
    near(d1.probability(&[("M", "1")]).unwrap(), 0.7);
    let d0 = model.law(&[("X", "0")], &ctx).unwrap();
    near(d0.probability(&[("Y", "1")]).unwrap(), 0.18);
    // do(M=1): 0.6*0.5 + 0.4*0.9 = 0.66, and X keeps its observational law.
    let dm = model.law(&[("M", "1")], &ctx).unwrap();
    near(dm.probability(&[("Y", "1")]).unwrap(), 0.66);
    near(dm.probability(&[("X", "1")]).unwrap(), 0.4);

    // Agreement with the independent enumeration on every joint state. The
    // enumeration is indexed (x, m, y); the law is indexed (M, X, Y).
    for (dx, dm_, intervention) in
        [(None, None, vec![]), (Some(1), None, vec![("X", "1")]), (None, Some(1), vec![("M", "1")])]
    {
        let truth = enumerate(dx, dm_);
        let law = model.law(&intervention, &ctx).unwrap();
        for x in 0..2 {
            for m in 0..2 {
                for y in 0..2 {
                    let event = [("X", ["0", "1"][x]), ("M", ["0", "1"][m]), ("Y", ["0", "1"][y])];
                    near(law.probability(&event).unwrap(), truth[x * 4 + m * 2 + y]);
                }
            }
        }
        near(law.probabilities().iter().sum::<f64>(), 1.0);
    }

    // Ancestral sampling approaches the exact law; deterministic per seed.
    let n = 200_000;
    let rows = model.sample(&[("X", "1")], n, &ExecutionContext::for_tests(3)).unwrap();
    let y_pos = model.variables().iter().position(|v| v == "Y").unwrap();
    let freq = rows.iter().filter(|r| r[y_pos] == 1).count() as f64 / n as f64;
    // SE = sqrt(0.72*0.28/200000) = 0.001; allow 6 SE.
    assert!((freq - 0.72).abs() < 0.006, "{freq}");
    assert_eq!(rows, model.sample(&[("X", "1")], n, &ExecutionContext::for_tests(3)).unwrap());
}

#[test]
fn finite_rows_with_rounding_slack_compile_to_a_normalized_law() {
    for slack in [-5.0e-10, 5.0e-10] {
        let mut mechanisms = binary_chain();
        let MechanismForm::Tabular { rows, .. } = &mut mechanisms[0].form else {
            unreachable!();
        };
        rows[0].probabilities[1].1 += slack;
        let model = ComposedFinite::compile(&spec_of(mechanisms)).unwrap();
        let law = model.law(&[], &ExecutionContext::for_tests(1)).unwrap();
        assert!(
            (law.probabilities().iter().sum::<f64>() - 1.0).abs() < 1.0e-15,
            "slack {slack} left a non-normalized compiled law"
        );
    }
}

#[test]
fn finite_tables_that_cannot_compose_are_refused() {
    // A level the producer can emit that the consumer never saw.
    let mut mechanisms = binary_chain();
    if let Domain::Levels(l) = &mut mechanisms[0].output_support {
        l.push("2".into());
    }
    if let MechanismForm::Tabular { levels, rows } = &mut mechanisms[0].form {
        levels.push("2".into());
        rows[0].probabilities = vec![("0".into(), 0.5), ("1".into(), 0.3), ("2".into(), 0.2)];
    }
    let err = ComposedFinite::compile(&spec_of(mechanisms)).unwrap_err();
    assert!(details(&err).contains(&"mechanism_composition.support_not_contained"), "{err:?}");

    // A table with a missing parent configuration.
    let mut mechanisms = binary_chain();
    if let MechanismForm::Tabular { rows, .. } = &mut mechanisms[1].form {
        rows.pop();
    }
    let err = ComposedFinite::compile(&spec_of(mechanisms)).unwrap_err();
    assert!(
        details(&err).contains(&"mechanism_composition.support_missing_parent_configuration"),
        "{err:?}"
    );

    // A row that is not a distribution.
    let mut mechanisms = binary_chain();
    if let MechanismForm::Tabular { rows, .. } = &mut mechanisms[0].form {
        rows[0].probabilities = vec![("0".into(), 0.6), ("1".into(), 0.6)];
    }
    let err = ComposedFinite::compile(&spec_of(mechanisms)).unwrap_err();
    assert!(details(&err).contains(&"mechanism_composition.shape_invalid_table"), "{err:?}");

    // Families are never mixed.
    let mut mechanisms = binary_chain();
    mechanisms.push(lg("Z", 0.0, 1.0, &[]));
    let findings = check_composition(&spec_of(mechanisms));
    assert!(findings.iter().any(|f| f.detail == "mechanism_composition.family_mixed"));
}
