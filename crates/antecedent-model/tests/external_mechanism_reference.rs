//! The equilibrium likelihood, its scores, the contraction bound and the solve of a feedback
//! system with registered external mechanisms against a 50-digit reference.
//!
//! `oracle_data/external_mechanism_reference.py` (frozen in the JSON beside it) evaluates, from
//! the stated equations, the Gaussian data term `-0.5 (r' S^-1 r + log det S + k log 2 pi)` on the
//! free coordinates, the change-of-variables term `log |det(I - J)|`, the central-difference
//! scores of both terms in every fitted scalar, the largest block row sum of the declared
//! Lipschitz bounds and the equilibrium of `x = f(x) + u`. The system has two external mechanisms
//! (`p0 tanh(p1 x1)` with two parameters and `q sin(x0)`) and one built-in affine equation.
//! Cases cover clamps that cut the cycle, a correlated, a nearly singular and a 1e-6-variance
//! disturbance covariance, saturated and offset rows, and a finite-difference declaration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_model::external_mechanism::{
    DerivativeDeclaration, DifferenceScheme, ExternalFeedbackSystem, ExternalFitParameter,
    ExternalMechanismImplementation, ExternalMechanismSpec, MechanismRegistry, SystemEquation,
};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumError, RegimeRow, SolvePolicy,
};
use serde_json::Value;

struct Tanh {
    spec: ExternalMechanismSpec,
    content: Vec<u8>,
}
struct Sine {
    spec: ExternalMechanismSpec,
    content: Vec<u8>,
}

fn tanh_impl(derivative: DerivativeDeclaration) -> Arc<Tanh> {
    let content = b"reference p0 tanh(p1 x)".to_vec();
    let mut spec = ExternalMechanismSpec::for_content(
        "lab/tanh",
        "1",
        &content,
        vec![1],
        vec!["p0".into(), "p1".into()],
        derivative,
    );
    spec.input_lipschitz = Some(vec![0.45]);
    Arc::new(Tanh { spec, content })
}
fn sine_impl(derivative: DerivativeDeclaration) -> Arc<Sine> {
    let content = b"reference q sin(x)".to_vec();
    let mut spec = ExternalMechanismSpec::for_content(
        "lab/sine",
        "1",
        &content,
        vec![0],
        vec!["q".into()],
        derivative,
    );
    spec.input_lipschitz = Some(vec![0.5]);
    Arc::new(Sine { spec, content })
}

impl ExternalMechanismImplementation for Tanh {
    fn spec(&self) -> ExternalMechanismSpec {
        self.spec.clone()
    }
    fn content(&self) -> &[u8] {
        &self.content
    }
    fn value(&self, x: &[f64], p: &[f64]) -> Result<f64, EquilibriumError> {
        Ok(p[0] * (p[1] * x[0]).tanh())
    }
    fn input_gradient(&self, x: &[f64], p: &[f64]) -> Result<Option<Vec<f64>>, EquilibriumError> {
        let t = (p[1] * x[0]).tanh();
        Ok(Some(vec![p[0] * p[1] * (1.0 - t * t)]))
    }
    fn input_gradient_parameter_derivative(
        &self,
        x: &[f64],
        p: &[f64],
    ) -> Result<Option<Vec<Vec<f64>>>, EquilibriumError> {
        let t = (p[1] * x[0]).tanh();
        let s2 = 1.0 - t * t;
        Ok(Some(vec![vec![p[1] * s2], vec![p[0] * s2 - 2.0 * p[0] * p[1] * x[0] * s2 * t]]))
    }
    fn parameter_gradient(
        &self,
        x: &[f64],
        p: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        let t = (p[1] * x[0]).tanh();
        Ok(Some(vec![t, p[0] * x[0] * (1.0 - t * t)]))
    }
}

impl ExternalMechanismImplementation for Sine {
    fn spec(&self) -> ExternalMechanismSpec {
        self.spec.clone()
    }
    fn content(&self) -> &[u8] {
        &self.content
    }
    fn value(&self, x: &[f64], p: &[f64]) -> Result<f64, EquilibriumError> {
        Ok(p[0] * x[0].sin())
    }
    fn input_gradient(&self, x: &[f64], p: &[f64]) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(Some(vec![p[0] * x[0].cos()]))
    }
    fn input_gradient_parameter_derivative(
        &self,
        x: &[f64],
        _p: &[f64],
    ) -> Result<Option<Vec<Vec<f64>>>, EquilibriumError> {
        Ok(Some(vec![vec![x[0].cos()]]))
    }
    fn parameter_gradient(
        &self,
        x: &[f64],
        _p: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(Some(vec![x[0].sin()]))
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/external_mechanism_reference.json")).unwrap()
}

fn theta(r: &Value, name: &str) -> f64 {
    r["theta"][name].as_f64().unwrap()
}

fn system(
    r: &Value,
    cov: &str,
    tanh: &ExternalMechanismSpec,
    sine: &ExternalMechanismSpec,
) -> ExternalFeedbackSystem {
    let covariance: Vec<f64> =
        r["covs"][cov].as_array().unwrap().iter().flat_map(|row| floats(row)).collect();
    ExternalFeedbackSystem {
        equations: vec![
            SystemEquation::External {
                id: "m0".into(),
                intercept: theta(r, "i0"),
                parameters: vec![theta(r, "p0"), theta(r, "p1")],
                mechanism: tanh.clone(),
            },
            SystemEquation::External {
                id: "m1".into(),
                intercept: theta(r, "i1"),
                parameters: vec![theta(r, "q")],
                mechanism: sine.clone(),
            },
            SystemEquation::Builtin {
                equation: Equation {
                    id: "m2".into(),
                    intercept: theta(r, "i2"),
                    weights: vec![0.3, 0.2, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            },
        ],
        noise_covariance: covariance,
    }
}

fn registry(tanh: Arc<Tanh>, sine: Arc<Sine>) -> MechanismRegistry {
    let mut registry = MechanismRegistry::new();
    registry.register(tanh).unwrap();
    registry.register(sine).unwrap();
    registry
}

fn rows(r: &Value, name: &str) -> Vec<RegimeRow> {
    r["rowsets"][name]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let values = floats(&row["x"]);
            let clamps = row["clamps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    let variable = v.as_u64().unwrap() as usize;
                    Clamp { variable, value: values[variable] }
                })
                .collect();
            RegimeRow { values, clamps }
        })
        .collect()
}

fn targets() -> Vec<ExternalFitParameter> {
    vec![
        ExternalFitParameter::Intercept { equation: 0 },
        ExternalFitParameter::Parameter { equation: 0, parameter: 0 },
        ExternalFitParameter::Parameter { equation: 0, parameter: 1 },
        ExternalFitParameter::Intercept { equation: 1 },
        ExternalFitParameter::Parameter { equation: 1, parameter: 0 },
        ExternalFitParameter::Intercept { equation: 2 },
    ]
}

fn close(got: f64, want: f64, rel: f64, scale: f64, what: &str) {
    assert!((got - want).abs() <= rel * scale.max(1.0), "{what}: {got} vs {want}");
}

#[test]
fn likelihood_terms_and_scores_match_the_fifty_digit_reference() {
    let r = reference();
    let (tanh, sine) =
        (tanh_impl(DerivativeDeclaration::Analytic), sine_impl(DerivativeDeclaration::Analytic));
    let registry = registry(tanh.clone(), sine.clone());
    let ctx = ExecutionContext::for_tests(1);
    for entry in r["likelihood"].as_array().unwrap() {
        let (cov, set) = (entry["cov"].as_str().unwrap(), entry["rows"].as_str().unwrap());
        let name = format!("{cov}/{set}");
        let resolved = system(&r, cov, &tanh.spec, &sine.spec).resolve(Some(&registry)).unwrap();
        let rows = rows(&r, set);
        let terms = resolved.likelihood_terms(&rows, &ctx).unwrap();
        let scale = entry["data_scale"].as_f64().unwrap();
        close(terms.data, entry["data"].as_f64().unwrap(), 1e-10, scale, &format!("{name} data"));
        close(
            terms.log_determinant,
            entry["logdet"].as_f64().unwrap(),
            1e-10,
            scale.min(10.0),
            &format!("{name} logdet"),
        );
        if let Some(want) = entry.get("data_score") {
            let want = floats(want);
            let got = resolved.data_term_score(&rows, &targets(), &ctx).unwrap().unwrap();
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                close(*g, *w, 1e-9, w.abs(), &format!("{name} data score {k}"));
            }
            let want = floats(&entry["det_score"]);
            let got = resolved.determinant_term_score(&rows, &targets(), &ctx).unwrap().unwrap();
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                close(*g, *w, 1e-9, w.abs(), &format!("{name} det score {k}"));
            }
        }
    }
}

#[test]
fn a_declared_finite_difference_gives_the_same_likelihood() {
    let r = reference();
    let fd =
        DerivativeDeclaration::FiniteDifference { step: 1e-6, scheme: DifferenceScheme::Central };
    let (tanh, sine) = (tanh_impl(fd), sine_impl(fd));
    let registry = registry(tanh.clone(), sine.clone());
    let ctx = ExecutionContext::for_tests(1);
    for entry in r["likelihood"].as_array().unwrap() {
        let (cov, set) = (entry["cov"].as_str().unwrap(), entry["rows"].as_str().unwrap());
        let resolved = system(&r, cov, &tanh.spec, &sine.spec).resolve(Some(&registry)).unwrap();
        let terms = resolved.likelihood_terms(&rows(&r, set), &ctx).unwrap();
        let scale = entry["data_scale"].as_f64().unwrap();
        close(terms.data, entry["data"].as_f64().unwrap(), 1e-9, scale, "fd data");
        close(terms.log_determinant, entry["logdet"].as_f64().unwrap(), 1e-8, 1.0, "fd logdet");
    }
    // an undeclared analytic derivative is a refusal, not a silent difference: the score of the
    // determinant term declines (None) and the caller falls back to the declared difference.
    let resolved =
        system(&r, "correlated", &tanh.spec, &sine.spec).resolve(Some(&registry)).unwrap();
    let rows = rows(&r, "no_clamps");
    assert!(resolved.determinant_term_score(&rows, &targets(), &ctx).unwrap().is_none());
}

#[test]
fn the_contraction_bound_is_the_largest_block_row_sum_of_the_declared_bounds() {
    let r = reference();
    let (tanh, sine) =
        (tanh_impl(DerivativeDeclaration::Analytic), sine_impl(DerivativeDeclaration::Analytic));
    let registry = registry(tanh.clone(), sine.clone());
    let resolved =
        system(&r, "correlated", &tanh.spec, &sine.spec).resolve(Some(&registry)).unwrap();
    for entry in r["bounds"].as_array().unwrap() {
        let clamps: Vec<Clamp> = entry["clamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| Clamp { variable: v.as_u64().unwrap() as usize, value: 0.0 })
            .collect();
        let got = resolved.contraction_bound(&clamps).unwrap();
        assert!(
            (got - entry["bound"].as_f64().unwrap()).abs() < 1e-15,
            "clamps {:?}: {got}",
            entry["clamps"]
        );
    }
}

#[test]
fn the_solve_matches_the_fifty_digit_equilibrium_and_certifies_uniqueness() {
    let r = reference();
    let (tanh, sine) =
        (tanh_impl(DerivativeDeclaration::Analytic), sine_impl(DerivativeDeclaration::Analytic));
    let registry = registry(tanh.clone(), sine.clone());
    let resolved =
        system(&r, "correlated", &tanh.spec, &sine.spec).resolve(Some(&registry)).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    for entry in r["solves"].as_array().unwrap() {
        let u = floats(&entry["u"]);
        let clamps: Vec<Clamp> = entry["clamps"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| Clamp { variable: k.parse().unwrap(), value: v.as_f64().unwrap() })
            .collect();
        let want = floats(&entry["x"]);
        let out = resolved.solve(&u, &clamps, &[0.0; 3], SolvePolicy::default(), &ctx).unwrap();
        for (i, (g, w)) in out.values.iter().zip(&want).enumerate() {
            close(*g, *w, 1e-8, w.abs(), &format!("solve {u:?} {clamps:?} x{i}"));
        }
        assert!(out.globally_unique && out.existence_certified, "{u:?} {clamps:?}");
    }
}

#[test]
fn the_forward_scheme_is_first_order_and_the_central_scheme_second_order() {
    // At a relative step of 1e-4 the central difference of a smooth mechanism agrees with the
    // analytic gradient to about 1e-8 and the forward difference only to about 1e-4, so a scheme
    // that was declared but silently replaced by the other shows in the log-determinant.
    let r = reference();
    let ctx = ExecutionContext::for_tests(1);
    let logdet = |derivative| {
        let (tanh, sine) = (tanh_impl(derivative), sine_impl(derivative));
        let registry = registry(tanh.clone(), sine.clone());
        let resolved =
            system(&r, "correlated", &tanh.spec, &sine.spec).resolve(Some(&registry)).unwrap();
        resolved.likelihood_terms(&rows(&r, "no_clamps"), &ctx).unwrap().log_determinant
    };
    let exact = logdet(DerivativeDeclaration::Analytic);
    let forward = (logdet(DerivativeDeclaration::FiniteDifference {
        step: 1e-4,
        scheme: DifferenceScheme::Forward,
    }) - exact)
        .abs();
    let central = (logdet(DerivativeDeclaration::FiniteDifference {
        step: 1e-4,
        scheme: DifferenceScheme::Central,
    }) - exact)
        .abs();
    assert!(forward > 1e-6 && forward < 1e-2, "forward error {forward}");
    assert!(central < 1e-6 && central < forward / 100.0, "central error {central}");
}
