use super::*;
use crate::continuous_cyclic::OdePolicy;
use crate::nonlinear_equilibrium::{Activation, RegimeRow};

/// `a * tanh(b * x)`-free affine mechanism: `p0 * x0 (+ p1)` with exact gradients.
struct Affine {
    spec: ExternalMechanismSpec,
    content: Vec<u8>,
}
impl Affine {
    fn new(version: &str, content: &str, derivative: DerivativeDeclaration) -> Arc<Self> {
        let content = content.as_bytes().to_vec();
        let mut spec = ExternalMechanismSpec::for_content(
            "lab/affine",
            version,
            &content,
            vec![1],
            vec!["slope".into()],
            derivative,
        );
        spec.input_lipschitz = Some(vec![0.9]);
        Arc::new(Self { spec, content })
    }
}
impl ExternalMechanismImplementation for Affine {
    fn spec(&self) -> ExternalMechanismSpec {
        self.spec.clone()
    }
    fn content(&self) -> &[u8] {
        &self.content
    }
    fn value(&self, inputs: &[f64], parameters: &[f64]) -> Result<f64, EquilibriumError> {
        Ok(parameters[0] * inputs[0])
    }
    fn input_gradient(
        &self,
        _inputs: &[f64],
        parameters: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(Some(vec![parameters[0]]))
    }
    fn parameter_gradient(
        &self,
        inputs: &[f64],
        _parameters: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Ok(Some(vec![inputs[0]]))
    }
}

fn system(spec: &ExternalMechanismSpec, slope: f64) -> ExternalFeedbackSystem {
    ExternalFeedbackSystem {
        equations: vec![
            SystemEquation::External {
                id: "a".into(),
                intercept: 0.0,
                parameters: vec![slope],
                mechanism: spec.clone(),
            },
            SystemEquation::Builtin {
                equation: Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![0.5, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            },
        ],
        noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
    }
}

#[test]
fn registered_mechanism_solves_the_closed_form_and_stays_verified_external() {
    let implementation = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
    let mut registry = MechanismRegistry::new();
    registry.register(implementation.clone()).unwrap();
    let system = system(&implementation.spec, 0.6);
    let resolved = system.resolve(Some(&registry)).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let (u0, u1) = (0.4, -0.3);
    let result = resolved.solve(&[u0, u1], &[], &[0.0, 0.0], SolvePolicy::default(), &ctx).unwrap();
    // x0 = 0.6 x1 + u0 and x1 = 0.5 x0 + u1.
    let x0 = (u0 + 0.6 * u1) / (1.0 - 0.3);
    assert!((result.values[0] - x0).abs() < 1e-9);
    assert!(result.globally_unique && result.existence_certified);
    assert_eq!(
        result.evidence.contraction_bound_provenance.as_deref(),
        Some("declared_external_lipschitz")
    );
    let wire = serde_json::to_value(&result).unwrap();
    assert_eq!(wire["evidence"]["contraction_bound_provenance"], "declared_external_lipschitz");
    let mut old = wire;
    old["evidence"].as_object_mut().unwrap().remove("contraction_bound_provenance");
    let restored: EquilibriumResult = serde_json::from_value(old).unwrap();
    assert_eq!(restored.evidence.contraction_bound_provenance, None);
    assert_eq!(resolved.trust(), MechanismTrust::VerifiedExternal);
    assert_eq!(resolved.evidence().len(), 1);
}

#[test]
fn replay_is_refused_without_a_resolver_an_implementation_or_with_a_changed_one() {
    let original = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
    let system = system(&original.spec, 0.6);
    let code = |r: Result<ResolvedFeedbackSystem, EquilibriumError>| match r {
        Err(EquilibriumError::Invalid(m)) => m,
        Err(other) => panic!("unexpected {other:?}"),
        Ok(_) => panic!("replay must be refused"),
    };
    assert!(code(system.resolve(None)).contains("mechanism_resolver_missing"));
    let empty = MechanismRegistry::new();
    assert!(code(system.resolve(Some(&empty))).contains("mechanism_implementation_unavailable"));
    let mut other_content = MechanismRegistry::new();
    other_content
        .register(Affine::new("1", "affine v1 edited", DerivativeDeclaration::Analytic))
        .unwrap();
    assert!(
        code(system.resolve(Some(&other_content))).contains("mechanism_implementation_changed")
    );
    let mut other_version = MechanismRegistry::new();
    other_version.register(Affine::new("2", "affine v1", DerivativeDeclaration::Analytic)).unwrap();
    assert!(
        code(system.resolve(Some(&other_version))).contains("mechanism_implementation_unavailable")
    );
}

#[test]
fn finite_difference_gradient_matches_the_analytic_gradient() {
    let analytic = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
    let difference = Affine::new(
        "1",
        "affine v1",
        DerivativeDeclaration::FiniteDifference { step: 1e-6, scheme: DifferenceScheme::Central },
    );
    let ctx = ExecutionContext::for_tests(1);
    let rows: Vec<RegimeRow> = [-1.0_f64, 0.5, 1.5]
        .into_iter()
        .map(|x| RegimeRow {
            values: vec![0.6 * x + 0.1, x],
            clamps: vec![Clamp { variable: 1, value: x }],
        })
        .collect();
    let mut terms = Vec::new();
    for implementation in [analytic, difference] {
        let mut registry = MechanismRegistry::new();
        registry.register(implementation.clone()).unwrap();
        let resolved = system(&implementation.spec, 0.6).resolve(Some(&registry)).unwrap();
        terms.push(resolved.likelihood_terms(&rows, &ctx).unwrap().total());
    }
    assert!((terms[0] - terms[1]).abs() < 1e-8);
}

#[test]
fn spec_identity_ignores_dependency_order_and_sees_every_declaration() {
    let base = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic).spec.clone();
    let mut a = base.clone();
    a.dependencies = vec![
        ExternalDependency { name: "x".into(), version: "1".into(), digest: "d".into() },
        ExternalDependency { name: "y".into(), version: "2".into(), digest: "e".into() },
    ];
    let mut b = a.clone();
    b.dependencies.reverse();
    assert_eq!(a.identity(), b.identity());
    let mut c = a.clone();
    c.input_lipschitz = Some(vec![0.8]);
    assert_ne!(a.identity(), c.identity());
}

#[test]
fn external_ode_matches_the_closed_form_for_a_linear_drift() {
    let mut implementation = Affine::new("1", "affine v1", DerivativeDeclaration::Analytic);
    Arc::get_mut(&mut implementation).unwrap().spec.input_lipschitz = Some(vec![0.5]);
    let mut registry = MechanismRegistry::new();
    registry.register(implementation.clone()).unwrap();
    // dx0/dt = 0.5 x1 - x0 with x1 constant: x0(t) = 1 + (x0(0) - 1) e^{-t}.
    let ode = ExternalOdeSystem {
        equations: vec![
            SystemEquation::External {
                id: "a".into(),
                intercept: 0.0,
                parameters: vec![0.5],
                mechanism: implementation.spec.clone(),
            },
            SystemEquation::Builtin {
                equation: Equation {
                    id: "b".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            },
        ],
        decay: vec![1.0, 0.0],
    };
    let ctx = ExecutionContext::for_tests(1);
    let out = ode
        .resolve(Some(&registry))
        .unwrap()
        .trajectory(&[3.0, 2.0], &[0.0, 1.0, 2.0], &[], OdePolicy::default(), &ctx)
        .unwrap();
    for (k, t) in [0.0_f64, 1.0, 2.0].into_iter().enumerate() {
        assert!((out.trajectory.states[k][0] - (1.0 + 2.0 * (-t).exp())).abs() < 1e-6);
    }
    assert_eq!(out.trust, MechanismTrust::VerifiedExternal);
    assert!((out.drift_lipschitz.unwrap() - 1.5).abs() < 1e-12);
}
