//! The direct process-local feedback solve (`LocalCyclicModel::solve` / `abduce`) and the
//! compiled `compile_local_cyclic` give the same equilibria.
//!
//! The compiled program binds each mechanism as a declared `BatchCallback` that provides its
//! derivatives and runs the same damped Newton solve and additive abduction; here both sides
//! evaluate the same functions and derivatives, so every value, residual and iteration count is
//! compared exactly. A callback that provides no derivatives is refused, as the declaration
//! protocol says.
//!
//! System: `x0 = 0.5 tanh(x1) + u0`, `x1 = 0.3 x0 + 0.2 sin(x0) + u1`, a two-node feedback loop
//! whose Jacobian has spectral radius below one.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    reason = "the test calls the direct solve the compiled program is compared with"
)]
#![allow(clippy::float_cmp, reason = "bit-identical equality is the property under test")]

use std::sync::Arc;

use antecedent::analysis::compiled_local_cyclic::{
    DeclaredCyclicMechanism, LocalCyclicCompileError, LocalCyclicRun, compile_local_cyclic,
};
use antecedent::gcm::local_cyclic::{LocalCyclicMechanism, LocalCyclicModel, LocalCyclicPolicy};
use antecedent::gcm::nonlinear_equilibrium::{Clamp, EquilibriumError};
use antecedent_core::{
    BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec,
    ExecutionContext,
};

type Mechanism = fn(&[f64]) -> (f64, Vec<f64>);

fn first(inputs: &[f64]) -> (f64, Vec<f64>) {
    let t = inputs[0].tanh();
    (0.5 * t, vec![0.5 * (1.0 - t * t)])
}

fn second(inputs: &[f64]) -> (f64, Vec<f64>) {
    let x = inputs[0];
    (0.3 * x + 0.2 * x.sin(), vec![0.3 + 0.2 * x.cos()])
}

struct Direct {
    id: &'static str,
    reads: Vec<usize>,
    f: Mechanism,
}

impl LocalCyclicMechanism for Direct {
    fn id(&self) -> &str {
        self.id
    }
    fn reads(&self) -> &[usize] {
        &self.reads
    }
    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
        Ok((self.f)(inputs))
    }
}

#[derive(Debug)]
struct Declared {
    spec: CallbackSpec,
    f: Option<Mechanism>,
}

impl Declared {
    fn new(id: &str, f: Option<Mechanism>) -> Arc<dyn BatchCallback> {
        Arc::new(Self {
            spec: CallbackSpec {
                name: id.to_owned(),
                inputs: Arc::from(vec!["x".to_owned()]),
                output: CallbackOutput::Real,
                dependencies: Arc::from([]),
                implementation: CallbackImplementation::Runtime,
            },
            f,
        })
    }
}

impl BatchCallback for Declared {
    fn spec(&self) -> &CallbackSpec {
        &self.spec
    }
    fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = self.value_and_derivatives(&[inputs[0][row]])?.0;
        }
        Ok(())
    }
    fn value_and_derivatives(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), CallbackError> {
        match self.f {
            Some(f) => Ok(f(inputs)),
            None => Err(CallbackError::DerivativesNotProvided { name: self.spec.name.clone() }),
        }
    }
}

fn direct_model() -> LocalCyclicModel {
    LocalCyclicModel {
        mechanisms: vec![
            Box::new(Direct { id: "x0", reads: vec![1], f: first }),
            Box::new(Direct { id: "x1", reads: vec![0], f: second }),
        ],
    }
}

fn declared(with_derivatives: bool) -> Vec<DeclaredCyclicMechanism> {
    vec![
        DeclaredCyclicMechanism {
            id: "x0".into(),
            reads: vec![1],
            callback: Declared::new("x0", with_derivatives.then_some(first as Mechanism)),
        },
        DeclaredCyclicMechanism {
            id: "x1".into(),
            reads: vec![0],
            callback: Declared::new("x1", Some(second as Mechanism)),
        },
    ]
}

fn run(clamps: Vec<Clamp>, factual: Option<Vec<f64>>) -> LocalCyclicRun {
    LocalCyclicRun {
        noise: vec![0.4, -0.2],
        clamps,
        initial: vec![0.0, 0.0],
        factual,
        policy: LocalCyclicPolicy::default(),
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_direct_solve_and_abduction_equal_the_compiled_local_cyclic_program() {
    let ctx = ExecutionContext::for_tests(1);
    let model = direct_model();
    let factual = vec![0.7, 0.1];
    let request = run(Vec::new(), Some(factual.clone()));

    let direct = model.solve(&request.noise, &[], &request.initial, request.policy, &ctx).unwrap();
    let shared = model.abduce(&factual, request.policy, &ctx).unwrap();
    let direct_counterfactual = model.solve(&shared, &[], &factual, request.policy, &ctx).unwrap();

    let compiled = compile_local_cyclic(&declared(true), &request, &ctx).unwrap();
    assert_eq!(compiled.lowered().route(), "LocalCyclicCallback");
    assert_eq!(bits(&compiled.result().values), bits(&direct.values));
    assert_eq!(compiled.result().residual.to_bits(), direct.residual.to_bits());
    assert_eq!(compiled.result().iterations, direct.iterations);
    let counterfactual = compiled.counterfactual().expect("a factual state gives a counterfactual");
    assert_eq!(bits(&counterfactual.values), bits(&direct_counterfactual.values));
    // With no clamp the abduced noise replays the factual state.
    for (value, factual) in counterfactual.values.iter().zip(&factual) {
        assert!((value - factual).abs() < 1e-8);
    }
}

#[test]
fn a_clamped_solve_equals_the_direct_clamped_solve() {
    let ctx = ExecutionContext::for_tests(1);
    let model = direct_model();
    let clamps = vec![Clamp { variable: 1, value: 0.5 }];
    let request = run(clamps.clone(), None);
    let direct =
        model.solve(&request.noise, &clamps, &request.initial, request.policy, &ctx).unwrap();
    let compiled = compile_local_cyclic(&declared(true), &request, &ctx).unwrap();
    assert_eq!(bits(&compiled.result().values), bits(&direct.values));
    assert!(compiled.counterfactual().is_none());
}

#[test]
fn a_callback_without_derivatives_is_refused() {
    let ctx = ExecutionContext::for_tests(1);
    let error = compile_local_cyclic(&declared(false), &run(Vec::new(), None), &ctx).unwrap_err();
    assert!(matches!(error, LocalCyclicCompileError::Model(_)), "{error}");
}
