//! Discretization of a composed linear-Gaussian model into a finite model,
//! against the closed-form Gaussian value, with the refusals the bridge owes
//! and the paradigm parity table.
//!
//! Hand-computed oracles.
//!
//! One edge, `X = e_X`, `Y = X + e_Y`, two levels each:
//!   X cells are `(-inf, 0)` and `(0, inf)` with values `+-2 phi(0) = +-0.79788456`;
//!   Var Y = 2 so Y cells split at 0 with values `+-sqrt(2) * 0.79788456 = +-1.12837917`;
//!   `Y | X = +0.79788456 ~ N(0.79788456, 1)` puts mass `Phi(0.79788456) = 0.78753`
//!   in the upper cell, so the discretized table mean is
//!   `(2 * 0.78753 - 1) * 1.12837917 = erf(1/sqrt(pi)) * 2/sqrt(pi) = 0.64889`
//!   against the closed-form `0.79788456`: error `0.14899`, and since the only
//!   mechanism with an error is Y's the computed bound equals it.
//!
//! Chain (the composition oracle of the mechanism tests):
//!   X = 1 + 2 `e_X`, M = 0.5 + 2 X + `e_M`, Y = -1 + 0.5 X + 3 M + 0.5 `e_Y`
//!   so `E[Y | do(X = x)] = 0.5 + 6.5 x` and the contrast of two intervention
//!   values is `6.5 (x1 - x0)`.

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedLinearGaussian, ComposedModelSpec, Domain, LatentDependence, MechanismContext,
    MechanismDescriptor, MechanismForm, MechanismOrigin, MechanismProvenance, MechanismUncertainty,
    ParameterTerm, ParentBinding, PosteriorDependence, TabularRow, UncertaintyMeaning,
};
use antecedent_model::discretized_scm::{
    BRIDGE_CALIBRATION, BRIDGE_INFERENCE, DETAIL_CANCELLED, DETAIL_COMPOSITION_INCOHERENT,
    DETAIL_DEGENERATE_MARGINAL, DETAIL_INTERVENTION_INVALID, DETAIL_LATENT_CONFOUNDING,
    DETAIL_LEVELS_OUT_OF_RANGE, DETAIL_LEVELS_UNDECLARED, DETAIL_NON_GAUSSIAN,
    DETAIL_PARADIGM_CLOSED, DETAIL_POINT_ONLY, DETAIL_STATE_SPACE_TOO_LARGE,
    DETAIL_SUPPORT_NOT_REAL, DETAIL_UNKNOWN_VARIABLE, DiscreteArm, DiscreteContrast,
    DiscretizationMethod, DiscretizationPlan, DiscretizedLinearGaussian, InferenceParadigm,
    MAX_BRIDGE_LEVELS, PARITY_TABLE, ParadigmCell, ParityRoute, parity_row, require_paradigm,
};

const METHOD: DiscretizationMethod = DiscretizationMethod::EqualProbabilityQuantile;

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

fn edge() -> Vec<MechanismDescriptor> {
    vec![lg("X", 0.0, 1.0, &[]), lg("Y", 0.0, 1.0, &[("X", 1.0)])]
}

fn plan_for(mechanisms: &[MechanismDescriptor], levels: usize) -> DiscretizationPlan {
    DiscretizationPlan {
        method: METHOD,
        levels: mechanisms.iter().map(|m| (m.variable.clone(), levels)).collect(),
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn build(mechanisms: Vec<MechanismDescriptor>, levels: usize) -> DiscretizedLinearGaussian {
    let plan = plan_for(&mechanisms, levels);
    DiscretizedLinearGaussian::from_spec(&spec_of(mechanisms), &plan, &ctx()).unwrap()
}

fn contrast(outcome: &str, variable: &str, hi: usize, lo: usize) -> DiscreteContrast {
    DiscreteContrast {
        outcome: outcome.into(),
        arms: vec![
            DiscreteArm { weight: 1.0, interventions: vec![(variable.into(), hi)] },
            DiscreteArm { weight: -1.0, interventions: vec![(variable.into(), lo)] },
        ],
    }
}

fn near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
}

#[test]
fn one_edge_two_levels_matches_the_hand_computed_table_mean() {
    let model = build(edge(), 2);
    let x = model.level_values("X").unwrap();
    let y = model.level_values("Y").unwrap();
    near(x[0], -0.797_884_560_8, 1e-9);
    near(x[1], 0.797_884_560_8, 1e-9);
    near(y[0], -std::f64::consts::FRAC_2_SQRT_PI, 1e-9);
    near(y[1], std::f64::consts::FRAC_2_SQRT_PI, 1e-9);
    near(model.cell_edges("X").unwrap()[0], 0.0, 1e-12);
    near(model.cell_edges("Y").unwrap()[0], 0.0, 1e-12);

    let result = model
        .mean_contrast(
            &DiscreteContrast {
                outcome: "Y".into(),
                arms: vec![DiscreteArm { weight: 1.0, interventions: vec![("X".into(), 1)] }],
            },
            &ctx(),
        )
        .unwrap();
    near(result.closed_form, 0.797_884_560_8, 1e-9);
    // erf(1/sqrt(pi)) * 2/sqrt(pi) and its gap to 2 phi(0), to the digits of the closed form.
    near(result.value, 0.648_888_563_189, 1e-8);
    near(result.absolute_error, 0.148_995_997_614, 1e-8);
    // One mechanism carries the error and the path to the outcome is one
    // edge, so the computed bound is the error itself.
    near(result.error_bound, result.absolute_error, 1e-9);
    assert_eq!(result.inference, BRIDGE_INFERENCE);
    assert_eq!(result.calibration, BRIDGE_CALIBRATION);
    assert!(!result.parameter_uncertainty_not_propagated);
}

#[test]
fn closed_form_oracle_agrees_and_the_computed_bound_holds_and_shrinks() {
    let mut errors = Vec::new();
    let mut bounds = Vec::new();
    for n in [4_usize, 16, 64] {
        let model = build(chain(), n);
        let lo = n / 4;
        let hi = 3 * n / 4 - 1;
        let x = model.level_values("X").unwrap().to_vec();
        let result = model.mean_contrast(&contrast("Y", "X", hi, lo), &ctx()).unwrap();
        // Independent closed form: E[Y | do(X = x)] = 0.5 + 6.5 x.
        near(result.closed_form, 6.5 * (x[hi] - x[lo]), 1e-9);
        near(result.arms[0].closed_form_mean, 0.5 + 6.5 * x[hi], 1e-9);
        // The bound is a guarantee, not a trend.
        assert!(
            result.absolute_error <= result.error_bound + 1e-9,
            "n={n}: error {} above bound {}",
            result.absolute_error,
            result.error_bound
        );
        errors.push(result.absolute_error);
        bounds.push(result.error_bound);
    }
    assert!(bounds[2] < bounds[0], "bound did not shrink: {bounds:?}");
    assert!(errors[2] < 0.02, "64 levels left error {} (bounds {bounds:?})", errors[2]);
    assert!(errors[2] < errors[0], "error did not shrink: {errors:?}");
}

#[test]
fn intervening_on_the_outcome_is_exact_and_has_zero_bound() {
    let model = build(chain(), 8);
    let result = model
        .mean_contrast(
            &DiscreteContrast {
                outcome: "Y".into(),
                arms: vec![DiscreteArm { weight: 1.0, interventions: vec![("Y".into(), 3)] }],
            },
            &ctx(),
        )
        .unwrap();
    let y = model.level_values("Y").unwrap();
    near(result.value, y[3], 1e-12);
    near(result.closed_form, y[3], 1e-12);
    near(result.error_bound, 0.0, 0.0);
}

#[test]
fn discretized_roots_are_uniform_and_reproduce_the_gaussian_mean() {
    let model = build(chain(), 8);
    let finite = model.finite_model();
    let law = finite.law(&[], &ctx()).unwrap();
    for (_, p) in law.marginal("X").unwrap() {
        near(p, 1.0 / 8.0, 1e-9);
    }
    let x = model.level_values("X").unwrap();
    near(x.iter().sum::<f64>() / 8.0, 1.0, 1e-9);
    assert!(x.windows(2).all(|w| w[0] < w[1]));
    // Mean error of a root is zero up to rounding.
    near(model.mechanism_mean_error("X").unwrap(), 0.0, 1e-9);
    assert!(model.mechanism_mean_error("Y").unwrap() > 0.0);
    // Probabilities of every joint state sum to one.
    near(law.probabilities().iter().sum::<f64>(), 1.0, 1e-9);
}

#[test]
fn discretization_is_deterministic_and_independent_of_listing_order() {
    let a = build(chain(), 8);
    let mut reversed = chain();
    reversed.reverse();
    let b = build(reversed, 8);
    assert_eq!(a.identity(), b.identity());
    assert_eq!(a.level_values("Y"), b.level_values("Y"));
    let c = contrast("Y", "X", 5, 2);
    assert_eq!(a.mean_contrast(&c, &ctx()).unwrap(), b.mean_contrast(&c, &ctx()).unwrap());
}

#[test]
fn identity_changes_with_a_number_at_unchanged_shape_and_with_levels() {
    let base = build(chain(), 8);
    let mut changed = chain();
    changed[1] = lg("M", 0.6, 1.0, &[("X", 2.0)]);
    assert_ne!(base.identity(), build(changed, 8).identity());
    assert_ne!(base.identity(), build(chain(), 9).identity());
    let mut other_evidence = chain();
    other_evidence[0].provenance.evidence_id = "data:other".into();
    assert_ne!(base.identity(), build(other_evidence, 8).identity());
}

#[test]
fn declared_parameter_uncertainty_is_flagged_not_propagated() {
    let mut mechanisms = chain();
    mechanisms[0].uncertainty = Some(MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: vec![ParameterTerm::Intercept],
        covariance: vec![0.01],
    });
    let model = build(mechanisms, 8);
    let result = model.mean_contrast(&contrast("Y", "X", 5, 2), &ctx()).unwrap();
    assert!(result.parameter_uncertainty_not_propagated);
    assert_eq!(result.inference, "point_only");
}

#[test]
fn a_finite_model_from_the_bridge_composes_with_the_source_provenance() {
    let model = build(chain(), 4);
    let spec = model.finite_spec();
    assert_eq!(spec.mechanisms.len(), 3);
    for d in &spec.mechanisms {
        assert!(d.provenance.artifact_id.starts_with("discretized:"));
        assert!(matches!(d.provenance.origin, MechanismOrigin::Fitted));
        assert!(d.provenance.evidence_id.starts_with("data:"));
        assert!(d.latents.is_empty());
        assert!(d.uncertainty.is_none());
    }
    // Y has 4 x 4 parent configurations, each a distribution over 4 levels.
    let y = spec.mechanisms.iter().find(|d| d.variable == "Y").unwrap();
    let MechanismForm::Tabular { levels, rows } = &y.form else { unreachable!() };
    assert_eq!(levels.len(), 4);
    assert_eq!(rows.len(), 16);
    for row in rows {
        near(row.probabilities.iter().map(|(_, p)| p).sum::<f64>(), 1.0, 1e-12);
    }
}

#[test]
fn undeclared_out_of_range_and_unknown_levels_are_refused() {
    let spec = spec_of(chain());
    let mut plan = plan_for(&chain(), 8);
    plan.levels.remove("M");
    let err = DiscretizedLinearGaussian::from_spec(&spec, &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_LEVELS_UNDECLARED);
    assert_eq!(err.subject, "M");

    for bad in [1_usize, MAX_BRIDGE_LEVELS + 1] {
        let plan = plan_for(&chain(), bad);
        let err = DiscretizedLinearGaussian::from_spec(&spec, &plan, &ctx()).unwrap_err();
        assert_eq!(err.detail, DETAIL_LEVELS_OUT_OF_RANGE, "{bad}");
        assert_eq!(err.reason_code, "invalid_argument");
    }

    let mut plan = plan_for(&chain(), 8);
    plan.levels.insert("Z".into(), 8);
    let err = DiscretizedLinearGaussian::from_spec(&spec, &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_UNKNOWN_VARIABLE);
}

#[test]
fn a_joint_space_above_the_enumeration_bound_is_refused() {
    let mechanisms = vec![
        lg("A", 0.0, 1.0, &[]),
        lg("B", 0.0, 1.0, &[("A", 1.0)]),
        lg("C", 0.0, 1.0, &[("B", 1.0)]),
        lg("D", 0.0, 1.0, &[("C", 1.0)]),
    ];
    // 64^4 = 16,777,216 joint states.
    let plan = plan_for(&mechanisms, 64);
    let err =
        DiscretizedLinearGaussian::from_spec(&spec_of(mechanisms), &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_STATE_SPACE_TOO_LARGE);
}

#[test]
fn non_gaussian_latent_bounded_and_degenerate_compositions_are_refused() {
    // A finite table is not a Gaussian mechanism.
    let table = MechanismDescriptor {
        variable: "B".into(),
        form: MechanismForm::Tabular {
            levels: vec!["0".into(), "1".into()],
            rows: vec![TabularRow {
                parent_levels: Vec::new(),
                probabilities: vec![("0".into(), 0.5), ("1".into(), 0.5)],
            }],
        },
        parents: Vec::new(),
        conditioning_set: Vec::new(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Levels(vec!["0".into(), "1".into()]),
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: provenance("B"),
        latents: Vec::new(),
        uncertainty: None,
    };
    let plan = plan_for(std::slice::from_ref(&table), 4);
    let err =
        DiscretizedLinearGaussian::from_spec(&spec_of(vec![table]), &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_NON_GAUSSIAN);

    // A shared latent cause makes the composition non-Markovian.
    let latent = |loading: f64, with: &str| LatentDependence {
        latent: "U".into(),
        loading: Some(loading),
        shared_with: vec![with.into()],
    };
    let mut a = lg("A", 0.0, 1.0, &[]);
    a.latents = vec![latent(2.0, "B")];
    let mut b = lg("B", 0.0, 1.0, &[]);
    b.latents = vec![latent(3.0, "A")];
    let mechanisms = vec![a, b];
    let plan = plan_for(&mechanisms, 4);
    let err =
        DiscretizedLinearGaussian::from_spec(&spec_of(mechanisms), &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_LATENT_CONFOUNDING);

    // A declared bounded support contradicts a Gaussian law.
    let mut bounded = lg("X", 0.0, 1.0, &[]);
    bounded.output_support = Domain::Interval { lo: 0.0, hi: 10.0 };
    let plan = plan_for(std::slice::from_ref(&bounded), 4);
    let err =
        DiscretizedLinearGaussian::from_spec(&spec_of(vec![bounded]), &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_SUPPORT_NOT_REAL);

    // A variable with no spread has no grid.
    let flat = lg("X", 1.0, 0.0, &[]);
    let plan = plan_for(std::slice::from_ref(&flat), 4);
    let err =
        DiscretizedLinearGaussian::from_spec(&spec_of(vec![flat]), &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_DEGENERATE_MARGINAL);
}

#[test]
fn an_incoherent_composition_is_refused_with_its_findings() {
    let mut mechanisms = chain();
    mechanisms[1].parents[0].units = "other".into();
    let plan = plan_for(&mechanisms, 4);
    let err =
        DiscretizedLinearGaussian::from_spec(&spec_of(mechanisms), &plan, &ctx()).unwrap_err();
    assert_eq!(err.detail, DETAIL_COMPOSITION_INCOHERENT);
    assert_eq!(err.reason_code, "mechanism_composition_incoherent");
    assert!(!err.findings.is_empty());
    // The source compiles on its own once repaired: the bridge adds no
    // acceptance the composition did not give.
    assert!(ComposedLinearGaussian::compile(&spec_of(chain())).is_ok());
}

#[test]
fn invalid_interventions_and_cancellation_are_refused() {
    let model = build(chain(), 4);
    let bad = [
        DiscreteArm { weight: 1.0, interventions: vec![("Z".into(), 0)] },
        DiscreteArm { weight: 1.0, interventions: vec![("X".into(), 4)] },
        DiscreteArm { weight: 1.0, interventions: vec![("X".into(), 0), ("X".into(), 1)] },
    ];
    for arm in bad {
        let err = model
            .mean_contrast(&DiscreteContrast { outcome: "Y".into(), arms: vec![arm] }, &ctx())
            .unwrap_err();
        assert_eq!(err.detail, DETAIL_INTERVENTION_INVALID);
    }
    let err = model
        .mean_contrast(&DiscreteContrast { outcome: "Z".into(), arms: Vec::new() }, &ctx())
        .unwrap_err();
    assert_eq!(err.detail, DETAIL_INTERVENTION_INVALID);

    let cancelled = ExecutionContext::for_tests(2);
    cancelled.cancellation.cancel();
    let spec = spec_of(chain());
    let err = DiscretizedLinearGaussian::from_spec(&spec, &plan_for(&chain(), 4), &cancelled)
        .unwrap_err();
    assert_eq!(err.detail, DETAIL_CANCELLED);
    let err = model.mean_contrast(&contrast("Y", "X", 1, 0), &cancelled).unwrap_err();
    assert_eq!(err.detail, DETAIL_CANCELLED);
}

#[test]
fn an_interval_or_posterior_request_of_a_discretized_value_is_refused_by_both_paradigms() {
    let model = build(chain(), 4);
    for paradigm in [InferenceParadigm::Frequentist, InferenceParadigm::Bayesian] {
        let refusal = model.uncertainty_refusal(paradigm);
        assert_eq!(refusal.detail, DETAIL_POINT_ONLY);
        assert_eq!(refusal.reason_code, "cell_not_licensed");
        assert!(!refusal.remedy.is_empty());
    }
}

#[test]
fn every_route_has_a_row_and_every_missing_paradigm_is_a_typed_closed_row() {
    assert_eq!(PARITY_TABLE.len(), ParityRoute::ALL.len());
    let mut neither = Vec::new();
    for route in ParityRoute::ALL {
        let row = parity_row(route);
        assert_eq!(row.route, route);
        let mut closed = 0;
        for (paradigm, cell) in [
            (InferenceParadigm::Frequentist, row.frequentist),
            (InferenceParadigm::Bayesian, row.bayesian),
        ] {
            match cell {
                ParadigmCell::Supported(what) => {
                    assert!(!what.is_empty());
                    assert_eq!(require_paradigm(route, paradigm), Ok(what));
                }
                ParadigmCell::Closed(reason) => {
                    closed += 1;
                    assert!(!reason.is_empty());
                    let refusal = require_paradigm(route, paradigm).unwrap_err();
                    assert_eq!(refusal.reason_code, "cell_not_licensed");
                    assert_eq!(refusal.detail, DETAIL_PARADIGM_CLOSED);
                    assert_eq!(refusal.remedy, reason);
                    assert_eq!(
                        refusal.subject,
                        format!("{}:{}", route.as_str(), paradigm.as_str())
                    );
                }
            }
        }
        if closed == 2 {
            neither.push(route);
        }
    }
    // Only the structural envelope has no inferential paradigm.
    assert_eq!(neither, vec![ParityRoute::DiscoveredStructureTransport]);
    // The bridge is Frequentist point-only; its Bayesian request is closed.
    assert!(
        require_paradigm(ParityRoute::DiscretizationBridge, InferenceParadigm::Frequentist).is_ok()
    );
    assert_eq!(
        require_paradigm(ParityRoute::DiscretizationBridge, InferenceParadigm::Bayesian)
            .unwrap_err()
            .detail,
        DETAIL_PARADIGM_CLOSED
    );
}
