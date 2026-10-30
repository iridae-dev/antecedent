//! The discretized linear-Gaussian bridge against a 50-digit reference.
//!
//! `oracle_data/discretized_scm_reference.py` (frozen in the JSON beside it) recomputes the
//! quantile grid, cell values, conditional tables, interventional discretized means (by
//! enumerating the joint space), closed-form means and the computed mean-error bound from the
//! definitions, for chains, a fork with a collider and negative coefficients, tiny and large
//! noise, 64 levels, and independent variables.  The bound is also required to dominate the
//! actual discretized-minus-closed-form error in every case.

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence,
};
use antecedent_model::discretized_scm::{
    DiscreteArm, DiscreteContrast, DiscretizationMethod, DiscretizationPlan,
    DiscretizedLinearGaussian,
};
use serde_json::Value;

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

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn close(got: f64, want: f64, tol: f64, ctx: &str) {
    assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{ctx}: {got} vs {want}");
}

#[test]
fn grid_tables_contrasts_and_bound_match_the_high_precision_reference() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/discretized_scm_reference.json"))
            .expect("frozen reference parses");
    let mut checked_bounds = 0;
    for case in cases.as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let mechanisms: Vec<MechanismDescriptor> = case["mechanisms"]
            .as_array()
            .expect("mechanisms")
            .iter()
            .map(|m| {
                let parents: Vec<(&str, f64)> = m["parents"]
                    .as_array()
                    .expect("parents")
                    .iter()
                    .map(|p| (p[0].as_str().expect("parent"), f(&p[1])))
                    .collect();
                lg(
                    m["variable"].as_str().expect("variable"),
                    f(&m["intercept"]),
                    f(&m["sigma"]),
                    &parents,
                )
            })
            .collect();
        let levels = case["levels"]
            .as_object()
            .expect("levels")
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().expect("count") as usize))
            .collect();
        let plan =
            DiscretizationPlan { method: DiscretizationMethod::EqualProbabilityQuantile, levels };
        let model = DiscretizedLinearGaussian::from_spec(&spec_of(mechanisms), &plan, &ctx())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for (variable, grid) in case["grid"].as_object().expect("grid") {
            let edges = model.cell_edges(variable).expect("edges");
            let values = model.level_values(variable).expect("values");
            let (we, wv) =
                (grid["edges"].as_array().expect("e"), grid["values"].as_array().expect("v"));
            assert_eq!((edges.len(), values.len()), (we.len(), wv.len()), "{name}/{variable}");
            for (g, w) in edges.iter().zip(we) {
                close(*g, f(w), 1e-12, &format!("{name}/{variable} edge"));
            }
            for (g, w) in values.iter().zip(wv) {
                close(*g, f(w), 1e-12, &format!("{name}/{variable} value"));
            }
            close(
                model.mechanism_mean_error(variable).expect("error"),
                f(&grid["mechanism_mean_error"]),
                1e-9,
                &format!("{name}/{variable} mechanism error"),
            );
        }
        for contrast in case["contrasts"].as_array().expect("contrasts") {
            let arms = contrast["arms"]
                .as_array()
                .expect("arms")
                .iter()
                .map(|a| DiscreteArm {
                    weight: f(&a["weight"]),
                    interventions: a["interventions"]
                        .as_array()
                        .expect("interventions")
                        .iter()
                        .map(|p| {
                            (
                                p[0].as_str().expect("var").to_owned(),
                                p[1].as_u64().expect("l") as usize,
                            )
                        })
                        .collect(),
                })
                .collect();
            let outcome = contrast["outcome"].as_str().expect("outcome");
            let got = model
                .mean_contrast(&DiscreteContrast { outcome: outcome.into(), arms }, &ctx())
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let ctxt = format!("{name}/{outcome}");
            close(got.value, f(&contrast["value"]), 1e-9, &format!("{ctxt} value"));
            close(
                got.closed_form,
                f(&contrast["closed_form"]),
                1e-9,
                &format!("{ctxt} closed form"),
            );
            close(got.error_bound, f(&contrast["bound"]), 1e-8, &format!("{ctxt} bound"));
            for (g, w) in got.arms.iter().zip(contrast["arms"].as_array().expect("arms")) {
                close(g.discretized_mean, f(&w["discretized"]), 1e-9, &format!("{ctxt} arm mean"));
                close(g.mean_error_bound, f(&w["bound"]), 1e-8, &format!("{ctxt} arm bound"));
            }
            assert!(
                got.absolute_error <= got.error_bound + 1e-9 * (1.0 + got.error_bound),
                "{ctxt}: error {} above bound {}",
                got.absolute_error,
                got.error_bound
            );
            checked_bounds += 1;
        }
    }
    assert!(checked_bounds == 20, "{checked_bounds}");
}

#[test]
fn three_digit_levels_keep_numeric_mean_and_parent_error_bound() {
    let spec = spec_of(vec![lg("X", 0.0, 1.0, &[]), lg("Y", 0.3, 0.3, &[("X", 2.0)])]);
    let plan = |x, y| DiscretizationPlan {
        method: DiscretizationMethod::EqualProbabilityQuantile,
        levels: [("X".to_owned(), x), ("Y".to_owned(), y)].into_iter().collect(),
    };
    let contrast = |level| DiscreteContrast {
        outcome: "Y".into(),
        arms: vec![DiscreteArm { weight: 1.0, interventions: vec![("X".into(), level)] }],
    };
    let model = DiscretizedLinearGaussian::from_spec(&spec, &plan(2, 101), &ctx()).unwrap();
    let result = model.mean_contrast(&contrast(1), &ctx()).unwrap();
    let x_name = &model.level_names("X").unwrap()[1];
    let law = model.finite_model().law(&[("X", x_name.as_str())], &ctx()).unwrap();
    let names = model.level_names("Y").unwrap();
    let values = model.level_values("Y").unwrap();
    let expected: f64 = law
        .marginal("Y")
        .unwrap()
        .iter()
        .map(|(name, p)| p * values[names.iter().position(|candidate| candidate == name).unwrap()])
        .sum();
    assert!((result.value - expected).abs() < 1e-12, "{} vs {expected}", result.value);

    // With X fixed, Y's mechanism error is the entire bound. This checks the
    // parent-configuration index independently of the label ordering.
    let model = DiscretizedLinearGaussian::from_spec(&spec, &plan(101, 2), &ctx()).unwrap();
    let result = model.mean_contrast(&contrast(100), &ctx()).unwrap();
    let arm = &result.arms[0];
    let error = (arm.discretized_mean - arm.closed_form_mean).abs();
    assert!(
        (arm.mean_error_bound - error).abs() < 1e-10,
        "bound {} vs {error}",
        arm.mean_error_bound
    );
}
